import os
import json
import re
from dotenv import load_dotenv
from typing import List, Literal, Optional

import tiktoken
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()
TAVILAY_API_KEY = os.getenv("TAVILAY_API_KEY")
MAX_RESULTS = os.getenv("MAX_RESULTS")

recall_vector_store = InMemoryVectorStore(OllamaEmbeddings(model="nomic-embed-text:latest"))

import uuid


def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id


@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    user_id = get_user_id(config)
    document = Document(
        page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": user_id}
    )
    recall_vector_store.add_documents([document])
    return memory


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    user_id = get_user_id(config)

    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id

    documents = recall_vector_store.similarity_search(
        query, k=3, filter=_filter_function
    )
    return [document.page_content for document in documents]

search = TavilySearchResults(max_results=MAX_RESULTS)
tools = [save_recall_memory, search_recall_memories, search]

class State(MessagesState):
    # add memories that will be retrieved based on the conversation context
    recall_memories: List[str]

# Define the prompt template for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            " You are a helpful assistant with advanced long-term memory"
            " capabilities. Powered by a stateless LLM, you must rely on"
            " external memory to store information between conversations."
            " Utilize the available memory tools to store and retrieve"
            " important details that will help you better attend to the user's"
            " needs and understand their context.\n\n"
            "Memory Usage Guidelines:\n"
            "1. You MUST actively use memory tools (save_core_memory, save_recall_memory)"
            " to build a comprehensive understanding of the user. It must include all personal details of the user.\n"
            "2. Make informed suppositions and extrapolations based on stored"
            " memories.\n"
            "3. Regularly reflect on past interactions to identify patterns and"
            " preferences.\n"
            "4. Update your mental model of the user with each new piece of"
            " information.\n"
            "5. Cross-reference new information with existing memories for"
            " consistency.\n"
            "6. Prioritize storing emotional context and personal values"
            " alongside facts.\n"
            "7. Use memory to anticipate needs and tailor responses to the"
            " user's style.\n"
            "8. Recognize and acknowledge changes in the user's situation or"
            " perspectives over time.\n"
            "9. Leverage memories to provide personalized examples and"
            " analogies.\n"
            "10. Recall past challenges or successes to inform current"
            " problem-solving.\n\n"
            "## Recall Memories\n"
            "Recall memories are contextually retrieved based on the current"
            " conversation:\n{recall_memories}\n\n"
            "## Instructions\n"
            "Engage with the user naturally, as a trusted colleague or friend."
            " There's no need to explicitly mention your memory capabilities."
            " Instead, seamlessly incorporate your understanding of the user"
            " into your responses. Be attentive to subtle cues and underlying"
            " emotions. Adapt your communication style to match the user's"
            " preferences and current emotional state. Use tools to persist"
            " information you want to retain in the next conversation. If you"
            " do call tools, all text preceding the tool call is an internal"
            " message. Respond AFTER calling the tool, once you have"
            " confirmation that the tool completed successfully.\n\n",
        ),
        ("placeholder", "{messages}"),
    ]
)

model = ChatOllama(model=os.getenv("CHAT_MODEL"),
                   temperature=0.3
                   )

model_with_tools = model.bind_tools(tools)


def agent(state: State) -> State:
    """Process the current state and generate a response using the LLM.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        schemas.State: The updated state with the agent's response.
    """
    bound = prompt | model_with_tools
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    prediction = bound.invoke(
        {
            "messages": state["messages"],
            "recall_memories": recall_str,
        }
    )
    return {
        "messages": [prediction],
    }


def load_memories(state: State, config: RunnableConfig) -> State:
    """Load memories for the current conversation.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        State: The updated state with loaded memories.
    """
    convo_str = get_buffer_string(state["messages"])
    recall_memories = search_recall_memories.invoke(convo_str, config)
    return {
        "recall_memories": recall_memories,
    }


def route_tools(state: State):
    """Determine whether to use tools or end the conversation based on the last message.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        Literal["tools", "__end__"]: The next step in the graph.
    """
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"

    return END

# Create the graph and add nodes
builder = StateGraph(State)
builder.add_node(load_memories)
builder.add_node(agent)
builder.add_node("tools", ToolNode(tools))

# Add edges to the graph
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges("agent", route_tools, ["tools", END])
builder.add_edge("tools", "agent")

# Compile the graph
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

def pretty_print_stream_chunk(chunk):
    for node, updates in chunk.items():
        print(f"Update from node: {node}")
        if "messages" in updates:
            updates["messages"][-1].pretty_print()
        else:
            print(updates)

        print("\n")

def stream_and_clean_output(stream_generator):
    """
    Streams output, buffering and discarding content within <think>...</think> tags.
    Prints "Thinking..." until </think> is found or stream ends.
    """
    print("Thinking... ", end="", flush=True) # Display thinking message
    buffered_output = ""
    in_thinking_block = False
    thinking_message_cleared = False

    for chunk in stream_generator:
        # Check if the chunk contains a message from the 'agent' node
        if 'agent' in chunk and 'messages' in chunk['agent'] and chunk['agent']['messages']:
            message_content = chunk['agent']['messages'][-1].content

            # Process the content chunk by chunk
            i = 0
            while i < len(message_content):
                if not in_thinking_block:
                    # Look for the start of a thinking block
                    start_tag_index = message_content.find('<think>', i)

                    if start_tag_index != -1:
                        # Print content before the <think> tag
                        print(message_content[i:start_tag_index], end="", flush=True)
                        in_thinking_block = True
                        i = start_tag_index + len('<think>')
                        # Start buffering
                        buffered_output = "" # Clear buffer for new thinking block
                    else:
                        # No <think> tag, print the rest of the content
                        print(message_content[i:], end="", flush=True)
                        i = len(message_content)
                else: # We are currently inside a thinking block
                    # Look for the end of a thinking block
                    end_tag_index = message_content.find('</think>', i)

                    if end_tag_index != -1:
                        # Add content up to </think> to buffer (will be discarded)
                        buffered_output += message_content[i:end_tag_index]
                        in_thinking_block = False
                        i = end_tag_index + len('</think>')

                        # Clear the "Thinking..." message once </think> is found
                        if not thinking_message_cleared:
                             print("\r" + " " * 20 + "\r", end="", flush=True) # Clear "Thinking..."
                             thinking_message_cleared = True

                        # Discard buffered_output
                        buffered_output = ""

                        # Continue processing the rest of the current chunk (content after </think>)
                    else:
                        # </think> not found in this chunk, buffer the whole rest of the chunk
                        buffered_output += message_content[i:]
                        i = len(message_content)

    # After the loop, ensure the thinking message is cleared if it wasn't already
    if not thinking_message_cleared:
         print("\r" + " " * 20 + "\r", end="", flush=True) # Clear "Thinking..."

    print("\n", end="", flush=True) # Ensure a newline at the end

# NOTE: we're specifying `user_id` to save memories for a given user
config = {"configurable": {"user_id": "1", "thread_id": "1"}}

# stream_and_clean_output(graph.stream({"messages": [("user", "What's up dude! My name is John")]}, config=config))
# stream_and_clean_output(graph.stream({"messages": [("user", "Yo! What is my name?")]}, config=config))
# stream_and_clean_output(graph.stream({"messages": [("user", "What is my favorite color bro?")]}, config=config))

for chunk in graph.stream({"messages": [("user", "my name is John")]}, config=config):
    pretty_print_stream_chunk(chunk)

for chunk in graph.stream({"messages": [("user", "What is my name?")]}, config=config):
    pretty_print_stream_chunk(chunk)

for chunk in graph.stream({"messages": [("user", "What is my favorite color bro?")]}, config=config):
    pretty_print_stream_chunk(chunk)

for chunk in graph.stream({"messages": [("user", "When is the next NHL scheduled game for the Nashville Preditors?")]}, config=config):
    pretty_print_stream_chunk(chunk)
import os
import re

def load_dotenv(dotenv_path='.env', override=False):
    """
    Reads a .env file, parses the key-value pairs, and sets them as environment variables.

    Args:
        dotenv_path (str): The path to the .env file. Defaults to '.env' in the current directory.
        override (bool): Whether to override existing environment variables. Defaults to False.
    """
    try:
        with open(dotenv_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Ignore comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Simple parsing for KEY=VALUE format
                # Handles optional quotes and potential spaces around '='
                match = re.match(r'^\s*([\w.-]+)\s*=\s*(.*?)?\s*$', line)
                if match:
                    key, value = match.groups()

                    # Remove surrounding quotes (single or double) if present
                    if value:
                        if (value.startswith("'") and value.endswith("'")) or \
                           (value.startswith('"') and value.endswith('"')):
                            value = value[1:-1]

                    # Set environment variable if not already set or if override is True
                    if key not in os.environ or override:
                        os.environ[key] = value
    except FileNotFoundError:
        # It's okay if the .env file doesn't exist, just proceed silently.
        # You might want to add logging here if needed.
        pass
    except Exception as e:
        print(f"Error loading .env file '{dotenv_path}': {e}")


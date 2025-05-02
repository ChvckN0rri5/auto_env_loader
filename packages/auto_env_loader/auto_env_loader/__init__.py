import os
from .loader import load_dotenv

# Determine the path to the .env file (e.g., in the project root)
# This assumes the script importing the package is run from the project root
# or that the .env file is placed relative to where the script is run.
# More sophisticated path finding might be needed for complex scenarios.
dotenv_path = os.path.join(os.getcwd(), '.env')

# Load the .env file automatically upon import
load_dotenv(dotenv_path=dotenv_path)

# Optionally expose the function for manual loading
__all__ = ['load_dotenv']

print(f"AutoEnvLoader: Loaded environment variables from {dotenv_path}")


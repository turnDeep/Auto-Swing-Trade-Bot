import os
from dotenv import load_dotenv

load_dotenv()

# Data Fetching
FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY:
    raise ValueError("FMP_API_KEY is not set in the environment or .env file.")

# Webull Live Execution
WEBULL_APP_KEY = os.getenv("WEBULL_APP_KEY")
WEBULL_APP_SECRET = os.getenv("WEBULL_APP_SECRET")
WEBULL_ACCOUNT_ID = os.getenv("WEBULL_ACCOUNT_ID")

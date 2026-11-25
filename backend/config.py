
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.getenv(
    "LANGFUSE_BASE_URL",
    "https://cloud.langfuse.com"
)

def validate_config():
    required = {
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "LANGFUSE_PUBLIC_KEY": LANGFUSE_PUBLIC_KEY,
        "LANGFUSE_SECRET_KEY": LANGFUSE_SECRET_KEY
    }

    missing = [
        key for key, value in required.items()
        if not value
    ]

    if missing:
        raise ValueError(
            f"Missing environment variables: {', '.join(missing)}"
        )

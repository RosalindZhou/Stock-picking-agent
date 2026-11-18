"""Read provider credentials from the process environment or a local .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"), override=False)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "").strip()

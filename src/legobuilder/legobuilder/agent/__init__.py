from pathlib import Path

from dotenv import load_dotenv

# Load .env from workspace root before anything else
load_dotenv(Path(__file__).resolve().parents[4] / ".env")

from .agent import agent

__all__ = ["agent"]

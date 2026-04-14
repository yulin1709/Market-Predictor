"""
utils/gemini.py — Gemini API helper.

Model: gemini-2.5-flash (10 RPM free tier)
Usage budget:
  - 1 call per dashboard load (market summary, cached 15 min)
  - 1 call per manual prediction click
  Total: well within 10 RPM
"""
import json
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

_client = None
MODEL = "gemini-2.5-flash"


def _get_client():
    global _client
    if _client is None:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in .env")
        _client = genai.Client(api_key=api_key)
    return _client


def is_configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""))


def generate_json(prompt: str) -> dict:
    """Call Gemini and parse response as JSON. ~1 RPM usage."""
    client = _get_client()
    response = client.models.generate_content(model=MODEL, contents=prompt)
    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def generate_text(prompt: str, max_tokens: int = 150) -> str:
    """Call Gemini and return plain text. ~1 RPM usage."""
    client = _get_client()
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text.strip()

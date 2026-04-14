"""
inference/explain.py — Generate plain-English explanation via Gemini (or Claude fallback).

The ML model predicts. The LLM only explains.
"""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

PROMPT = """{label} impact on energy prices. Headline: {headline}. Event: {event_type}, {region}, supply_impact={supply_impact}. Similar: {history}. Explain in 2 sentences why prices may move. Be specific."""


def generate_explanation(
    headline: str,
    label: str,
    score: float,
    entities: dict,
    similar: list[dict],
) -> str:
    history_lines = []
    for i, s in enumerate(similar[:2], 1):  # only top 2 similar events
        history_lines.append(f"[{s['label']}] {s['headline'][:60]} ({s['price_change']:+.2f}%)")
    history = "; ".join(history_lines) if history_lines else "none"

    prompt = PROMPT.format(
        headline=headline[:120],  # truncate long headlines
        label=label,
        score=score * 100,
        event_type=entities.get("event_type", "other"),
        region=entities.get("region", "global"),
        sentiment=float(entities.get("sentiment", 0)),
        supply_impact=bool(entities.get("supply_impact", False)),
        history=history,
    )

    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    # Only call Gemini for manual predictions (not batch article scoring)
    # This keeps usage within 10 RPM free tier
    if provider == "gemini":
        result = _explain_gemini(prompt)
        if result:
            return result
    elif provider == "claude":
        result = _explain_claude(prompt)
        if result:
            return result

    return _fallback_explanation(headline, label, score, entities)


def _explain_gemini(prompt: str) -> str:
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from utils.gemini import generate_text
        return generate_text(prompt, max_tokens=100)
    except Exception as e:
        # Quota or any error — return a rule-based fallback explanation
        return None


def _fallback_explanation(headline: str, label: str, score: float, entities: dict) -> str:
    event = entities.get("event_type", "other").replace("_", " ")
    region = entities.get("region", "global")
    direction = "bullish" if entities.get("sentiment", 0) >= 0 else "bearish"
    supply = "supply disruption" if entities.get("supply_impact") else "market development"
    return (
        f"This article describes a {event} event in {region}, "
        f"which historically triggers {direction} price reactions in energy markets. "
        f"The ML model assigned {label} impact ({score*100:.0f}% confidence) "
        f"based on similar {supply} patterns in the training data."
    )


def _explain_claude(prompt: str) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"Explanation unavailable: {e}"

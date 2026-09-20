"""
agent/llm_client.py

Thin wrapper around a Groq LLM call that explains OCR candidates to a
human verifier. Same principle as everywhere else in this pipeline: the
agent never picks a value, it only comments on plausibility -- see the
system prompt below, which is written specifically to prevent it from
drifting into making the decision itself.

Groq's API is OpenAI-compatible, so this uses the `openai` SDK pointed
at Groq's base_url rather than a Groq-specific client -- if the provider
ever changes, only GROQ_BASE_URL/GROQ_MODEL need updating, not the
calling code.
"""

import json
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

SYSTEM_PROMPT = """You are assisting a human who is verifying an analog gauge reading against OCR candidates for its min and max scale labels.

For each position, briefly note which candidate looks most plausible and why. You are giving an opinion to help a human decide -- you never make the decision. Never say a value "is correct" or tell the human what to pick -- describe plausibility only (e.g. "looks more likely", "may be less consistent with the others").

Rules:
- This is an analog gauge scale. The min value is typically 0. The max value is typically a round number (10, 60, 100, 300, etc.).
- Don't assume the top-ranked candidate is right just because OCR ranked it first -- compare the actual values against each other and against what's typical for a gauge scale. If the candidates look inconsistent with each other, or none of them seem plausible, say so plainly rather than finding something positive to say about each one.
- A candidate marked (assumed, not OCR'd) was not actually read by OCR -- it's a default the pipeline inserts when it can't find a real "0" reading. Don't claim it was confidently detected, but you may note it's a reasonable default since gauges usually start at 0.
- Do not guess the gauge's brand, model, or manufacturer -- you don't have that information.
- If a position has only one or two candidates, comment only on what's given -- don't imply more candidates exist.
- Return ONLY a JSON object, no markdown code fences, no preamble, in exactly this shape:
{"position_a": "...", "position_b": "..."}
Each string must be under 30 words."""


class CandidateExplainer:
    """
    Loaded once per process -- same reasoning as KeypointModel/OcrRunner:
    the API server shouldn't reconstruct an HTTP client on every request.
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not configured")
        self.client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    def explain(self, pos_a_candidates: list[dict], pos_b_candidates: list[dict]) -> dict:
        """
        Returns {"position_a": str | None, "position_b": str | None}.
        None means the LLM's response couldn't be parsed -- the caller
        (the /explain endpoint) decides what to show in that case, this
        function just doesn't fabricate an explanation to fill the gap.
        """
        user_prompt = _format_candidates(pos_a_candidates, pos_b_candidates)

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )

        raw = response.choices[0].message.content
        return _parse_response(raw)


def _format_candidates(pos_a: list[dict], pos_b: list[dict]) -> str:
    def format_position(label: str, candidates: list[dict]) -> str:
        lines = [f"Position {label} OCR candidates, ranked by confidence:"]
        if not candidates:
            lines.append("(none found)")
        for c in candidates:
            note = " (assumed, not OCR'd)" if c.get("is_forced_zero") else ""
            lines.append(f'{c["rank"]}. "{c["text"]}" (parsed as {c["parsed_value"]}){note}')
        return "\n".join(lines)

    return (
        format_position("A (min-labeled)", pos_a)
        + "\n\n"
        + format_position("B (max-labeled)", pos_b)
    )


def _parse_response(raw: str) -> dict:
    """
    Best-effort JSON parse -- strips a markdown code fence if the model
    added one despite being told not to. Smaller/free models follow the
    "no code fences" instruction inconsistently often enough that it's
    worth guarding against here rather than trusting it blindly.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("LLM response wasn't valid JSON: %r", raw)
        return {"position_a": None, "position_b": None}

    return {
        "position_a": parsed.get("position_a"),
        "position_b": parsed.get("position_b"),
    }
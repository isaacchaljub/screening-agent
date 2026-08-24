"""M2 acceptance: one structured extraction and one text completion through Gemini.

python -m screening_agent.llm.smoke
"""

from __future__ import annotations

from pydantic import BaseModel

from screening_agent.llm.base import Message
from screening_agent.llm.client import LLMClient


class _SmokeExtraction(BaseModel):
    full_name: str | None = None
    city: str | None = None


def main() -> None:
    client = LLMClient()

    structured = client.complete_structured(
        "extract",
        system="Extract the candidate's full name and city from their message. Leave a field "
        "null if it isn't clearly stated — never guess.",
        messages=[Message(role="user", content="Hola, soy Ana García y vivo en Sevilla.")],
        schema=_SmokeExtraction,
    )
    print(f"[structured] model={structured.model} -> {structured.data!r}")

    text = client.complete_text(
        "compose",
        system="Reply in under 25 words, one question, plain and warm.",
        messages=[Message(role="user", content="Hola, ¿qué necesitas saber de mí?")],
    )
    print(f"[text] model={text.model} -> {text.text!r}")


if __name__ == "__main__":
    main()

"""One structured extraction and one text completion, for live verification. An optional
`--model` override exercises whichever vendor's adapter needs checking (`LLMClient(model=...)`
forces both the extract and compose calls below onto that one model, per `client.py`'s
`_resolve_with_backup`).

python -m screening_agent.llm.smoke
python -m screening_agent.llm.smoke --model anthropic:claude-haiku-4-5
python -m screening_agent.llm.smoke --model openai:gpt-5.6-terra
"""

from __future__ import annotations

import argparse

from pydantic import BaseModel

from screening_agent.llm.base import Message
from screening_agent.llm.client import LLMClient


class _SmokeExtraction(BaseModel):
    full_name: str | None = None
    city: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser(prog="screening_agent.llm.smoke")
    parser.add_argument(
        "--model", default=None, help="vendor:model-id — forces both calls onto this model"
    )
    args = parser.parse_args()
    client = LLMClient(model=args.model)

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

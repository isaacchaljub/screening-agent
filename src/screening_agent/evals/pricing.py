"""Per-model $/1M-token pricing for the eval sweep's cost-per-conversation column.

Current published rates as of 2026-08-24 — vendor pricing changes too often to trust from memory:

- `anthropic:claude-haiku-4-5` — $1.00 input / $5.00 output per 1M tokens.
- `anthropic:claude-sonnet-5` — **introductory** $2.00 / $10.00 per 1M tokens through 2026-08-31,
  rising to $3.00 / $15.00 after. A cost figure measured after that date would differ by 50%
  through no code change. Revisit this constant then.
- `groq:openai/gpt-oss-120b` — $0.15 input / $0.60 output per 1M tokens.

Not a general-purpose pricing module — just what the eval sweep needs prices for. A model not in
this table has no cost computed (`cost_usd=None` in the report) rather than a guessed number.
"""

from __future__ import annotations

# vendor:model-id -> (input $ / 1M tokens, output $ / 1M tokens)
PRICING_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "anthropic:claude-haiku-4-5": (1.00, 5.00),
    "anthropic:claude-sonnet-5": (2.00, 10.00),  # introductory, through 2026-08-31
    "groq:openai/gpt-oss-120b": (0.15, 0.60),
}


def cost_usd(model: str, *, input_tokens: int, output_tokens: int) -> float | None:
    """`None` if `model` has no known price — never guess a cost."""
    prices = PRICING_PER_MILLION_TOKENS.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000

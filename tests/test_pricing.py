from screening_agent.evals.pricing import cost_usd


def test_known_model_computes_cost():
    # anthropic:claude-haiku-4-5 is $1.00 input / $5.00 output per 1M tokens
    got = cost_usd("anthropic:claude-haiku-4-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert got == 6.00


def test_zero_tokens_is_zero_cost():
    assert cost_usd("anthropic:claude-haiku-4-5", input_tokens=0, output_tokens=0) == 0.0


def test_unpriced_model_returns_none_never_guesses():
    assert cost_usd("google:gemini-3.5-flash-lite", input_tokens=1000, output_tokens=1000) is None

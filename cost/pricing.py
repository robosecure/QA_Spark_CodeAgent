"""
Model pricing table (USD per 1,000 tokens).
Defaults are approximate — override via COST_INPUT_PER_1K / COST_OUTPUT_PER_1K env vars
for internal/proxy pricing that differs from public rates.
"""

# (provider, model_name_lower_prefix): (input_per_1k, output_per_1k)
PRICE_TABLE = {
    # Azure / OpenAI GPT-5 series (estimated — update when pricing published)
    ("azure",  "bdf-glb-gpt-5"):    (0.015, 0.060),
    ("azure",  "gpt-5"):            (0.015, 0.060),
    # GPT-4o
    ("azure",  "gpt-4o"):           (0.005, 0.015),
    ("openai", "gpt-4o"):           (0.005, 0.015),
    # GPT-4 turbo
    ("azure",  "gpt-4-turbo"):      (0.010, 0.030),
    ("openai", "gpt-4-turbo"):      (0.010, 0.030),
    # GPT-3.5
    ("azure",  "gpt-3.5-turbo"):    (0.001, 0.002),
    ("openai", "gpt-3.5-turbo"):    (0.001, 0.002),
    # Bedrock Claude (future)
    ("bedrock", "claude-3-5-sonnet"): (0.003, 0.015),
    ("bedrock", "claude-3-opus"):     (0.015, 0.075),
}

DEFAULT_INPUT_PER_1K  = 0.015
DEFAULT_OUTPUT_PER_1K = 0.060


def get_price(provider: str, model_name: str) -> tuple:
    """Return (input_per_1k, output_per_1k) for the given provider/model."""
    import os
    # Env var overrides take priority
    env_in  = os.environ.get("COST_INPUT_PER_1K")
    env_out = os.environ.get("COST_OUTPUT_PER_1K")
    if env_in and env_out:
        return float(env_in), float(env_out)

    p = provider.lower()
    m = model_name.lower()
    for (prov, model_prefix), prices in PRICE_TABLE.items():
        if prov == p and m.startswith(model_prefix):
            return prices

    return DEFAULT_INPUT_PER_1K, DEFAULT_OUTPUT_PER_1K


def compute_cost(provider: str, model_name: str,
                 prompt_tokens: int, completion_tokens: int) -> float:
    in_price, out_price = get_price(provider, model_name)
    return round(
        (prompt_tokens / 1000) * in_price +
        (completion_tokens / 1000) * out_price,
        6,
    )

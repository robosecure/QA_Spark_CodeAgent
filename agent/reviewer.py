import json
import sys
from pathlib import Path

from openai import OpenAI

# Allow running from project root or agent/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

PROMPTS_DIR = Path(__file__).parent / "prompts"

LANGUAGE_MAP = {
    "impala": "impala.json",
    "pyspark": "pyspark.json",
    "sparksql": "sparksql.json",
    "scala": "scala.json",
    "python": "python.json",
}

SUPPORTED_LANGUAGES = list(LANGUAGE_MAP.keys())


def load_prompt(language: str) -> dict:
    lang = language.lower()
    filename = LANGUAGE_MAP.get(lang)
    if not filename:
        raise ValueError(
            f"Unsupported language: '{language}'. "
            f"Supported languages: {SUPPORTED_LANGUAGES}"
        )
    path = PROMPTS_DIR / filename
    with open(path) as f:
        return json.load(f)


def review_code(code: str, language: str) -> dict:
    """
    Submit code for AI review using the language-specific prompt contract.
    Returns a dict with the full review text, language, model, and token usage.
    """
    cfg = Config()
    prompt_contract = load_prompt(language)

    client = OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.api_base,
    )

    spark_context = (
        f"Apache Spark version: {cfg.spark_version} (default 3.4 unless overridden via SPARK_VERSION)."
        if language in ("pyspark", "sparksql", "scala")
        else ""
    )

    system_prompt = f"""You are QA Spark CodeAgent — an expert code reviewer for the Cloudera CDP platform.

Your job is to review code for:
- Correctness and adherence to platform best practices
- Resource efficiency (CPU, memory, I/O, network)
- Performance optimization (runtime speed, shuffle reduction, predicate pushdown, etc.)
- Code readability and maintainability

{spark_context}

You follow this review contract exactly:
{json.dumps(prompt_contract, indent=2)}

SCORING RULES (mandatory):
- You MUST include a score formatted exactly as: SCORE: XX/100
- Place the score prominently at the top of Section 1
- Be strict and objective. The goal is to minimize resource cost and maximize speed on Cloudera CDP.
- A score of 95+ means code is certified ready. Below 95 means it needs fixes.
- Do not inflate scores. If there are real issues, reflect them in the score."""

    user_prompt = f"""Review the following {language.upper()} code for Cloudera CDP execution:

```{language}
{code}
```

Follow all sections in the contract. Include SCORE: XX/100 at the top of Section 1."""

    response = client.chat.completions.create(
        model=cfg.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
    )

    content = response.choices[0].message.content

    return {
        "review": content,
        "language": language,
        "model": cfg.model_name,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }

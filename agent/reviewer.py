import json
import sys
from pathlib import Path

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
            f"Supported: {SUPPORTED_LANGUAGES}"
        )
    with open(PROMPTS_DIR / filename) as f:
        return json.load(f)


def _get_llm_client(cfg: Config):
    """
    Return a callable that accepts (system_prompt, user_prompt)
    and returns the LLM response string.
    Supports: azure, openai.
    """
    if cfg.provider == "azure":
        from azure.identity import ClientSecretCredential
        from langchain_openai import AzureChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        credential = ClientSecretCredential(
            tenant_id=cfg.azure_tenant_id,
            client_id=cfg.azure_service_principal,
            client_secret=cfg.azure_service_principal_secret,
        )

        def get_token():
            return credential.get_token(cfg.azure_token_audience).token

        llm = AzureChatOpenAI(
            azure_deployment=cfg.azure_deployed_model,
            api_version=cfg.azure_api_version,
            azure_endpoint=cfg.azure_endpoint,
            azure_ad_token_provider=get_token,
            max_tokens=4096,
        )

        def call(system_prompt, user_prompt):
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = llm.invoke(messages)
            return response.content, None  # LangChain doesn't expose usage the same way

        return call

    elif cfg.provider == "openai":
        from openai import OpenAI

        client = OpenAI(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_api_base,
        )

        def call(system_prompt, user_prompt):
            response = client.chat.completions.create(
                model=cfg.openai_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            return response.choices[0].message.content, usage

        return call

    else:
        raise NotImplementedError(f"Provider '{cfg.provider}' not yet implemented.")


def review_code(code: str, language: str) -> dict:
    """
    Submit code for AI review using the language-specific prompt contract.
    Returns a dict with review text, language, model, provider, and token usage.
    """
    cfg = Config()
    prompt_contract = load_prompt(language)

    spark_context = (
        f"Apache Spark version: {cfg.spark_version} "
        f"(default 3.4 — override via SPARK_VERSION env var)."
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

    call = _get_llm_client(cfg)
    content, usage = call(system_prompt, user_prompt)

    return {
        "review": content,
        "language": language,
        "model": cfg.model_name,
        "provider": cfg.provider,
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Provider-agnostic config. Set PROVIDER to switch LLM backends:
      PROVIDER=azure    — Azure OpenAI via Service Principal (current)
      PROVIDER=openai   — Plain OpenAI or OpenAI-compatible proxy
      PROVIDER=bedrock  — AWS Bedrock (future)
    """

    def __init__(self):
        self.provider = os.environ.get("PROVIDER", "azure").lower()
        self.pass_threshold = int(os.environ.get("PASS_THRESHOLD", "95"))
        self.spark_version = os.environ.get("SPARK_VERSION", "3.4")

        # ── Azure OpenAI settings ─────────────────────────────────────────
        self.azure_tenant_id = os.environ.get("AZURE_TENANT_ID", "")
        self.azure_service_principal = os.environ.get("AZURE_SERVICE_PRINCIPAL", "")
        self.azure_service_principal_secret = os.environ.get("AZURE_SERVICE_PRINCIPAL_SECRET", "")
        self.azure_account_name = os.environ.get("AZURE_ACCOUNT_NAME", "")
        self.azure_deployed_model = os.environ.get("AZURE_DEPLOYED_MODEL", "BDF-GLB-GPT-5")
        self.azure_api_version = os.environ.get("AZURE_API_VERSION", "2024-06-01")
        self.azure_endpoint_base = os.environ.get("AZURE_ENDPOINT_BASE", "https://openai.work.iqvia.com/cse/prod/proxy/azure")
        self.azure_token_audience = os.environ.get("AZURE_TOKEN_AUDIENCE", "api://825a47b7-8e55-49b5-99c5-d7ecf65bd64d/.default")

        # ── Plain OpenAI / proxy settings ─────────────────────────────────
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.openai_api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.openai_model_name = os.environ.get("MODEL_NAME", "gpt-4o")

        # Friendly model name for display
        self.model_name = (
            self.azure_deployed_model if self.provider == "azure"
            else self.openai_model_name
        )

        self._validate()

    def _validate(self):
        if self.provider == "azure":
            missing = [
                k for k, v in {
                    "AZURE_TENANT_ID": self.azure_tenant_id,
                    "AZURE_SERVICE_PRINCIPAL": self.azure_service_principal,
                    "AZURE_SERVICE_PRINCIPAL_SECRET": self.azure_service_principal_secret,
                    "AZURE_ACCOUNT_NAME": self.azure_account_name,
                }.items() if not v
            ]
            if missing:
                raise ValueError(f"Azure provider requires: {missing}")

        elif self.provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for provider=openai")

        elif self.provider == "bedrock":
            raise NotImplementedError("Bedrock provider is planned for a future release.")

        else:
            raise ValueError(f"Unknown PROVIDER '{self.provider}'. Use: azure, openai, bedrock")

    @property
    def azure_endpoint(self):
        return f"{self.azure_endpoint_base}/{self.azure_account_name}"

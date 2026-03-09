"""
Abstract base class for all specialist review agents.

Uses the raw openai.AzureOpenAI / openai.OpenAI client (NOT LangChain)
so that we can read actual token usage from response.usage.
"""
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import openai
from azure.identity import ClientSecretCredential

from config import Config
from cost.tracker import CostTracker

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


class BaseAgent(ABC):
    """
    Subclass and implement:
      - name:    str class attribute (e.g. "security_agent")
      - _build_system_prompt(language, spark_version) -> str
      - _build_user_message(code, context_hint)       -> str
      - parse_score(raw_text)                         -> int (0-100)
    """

    name: str = "base_agent"

    def __init__(self, config: Config, tracker: CostTracker):
        self.config = config
        self.tracker = tracker
        self._client: Optional[openai.AzureOpenAI | openai.OpenAI] = None

    # ── LLM client ────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client:
            return self._client

        if self.config.provider == "azure":
            credential = ClientSecretCredential(
                tenant_id=self.config.azure_tenant_id,
                client_id=self.config.azure_service_principal,
                client_secret=self.config.azure_service_principal_secret,
            )
            token_obj = credential.get_token(self.config.azure_token_audience)

            self._client = openai.AzureOpenAI(
                api_key=token_obj.token,
                azure_endpoint=self.config.azure_endpoint,
                api_version=self.config.azure_api_version,
            )

        elif self.config.provider == "openai":
            self._client = openai.OpenAI(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_api_base,
            )

        else:
            raise NotImplementedError(f"Provider '{self.config.provider}' not supported in BaseAgent")

        return self._client

    def _model_name(self) -> str:
        return (
            self.config.azure_deployed_model
            if self.config.provider == "azure"
            else self.config.openai_model_name
        )

    # ── Core LLM call ─────────────────────────────────────────────────────────

    def _call_llm(self, system_prompt: str, user_message: str) -> tuple[str, int, int]:
        """
        Call the LLM and return (response_text, prompt_tokens, completion_tokens).
        Records usage in the shared CostTracker.
        """
        client = self._get_client()
        model = self._model_name()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]

        # o-series models (GPT-5, o1, o3) do not support temperature
        kwargs: dict = {
            "model":    model,
            "messages": messages,
        }

        try:
            response = client.chat.completions.create(**kwargs)
        except openai.AuthenticationError:
            # Token may have expired — refresh and retry once
            logger.warning("Azure token expired, refreshing…")
            self._client = None
            client = self._get_client()
            response = client.chat.completions.create(**kwargs)

        text = response.choices[0].message.content or ""
        prompt_tokens     = response.usage.prompt_tokens     if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0

        self.tracker.record(self.name, prompt_tokens, completion_tokens)
        return text, prompt_tokens, completion_tokens

    # ── Prompt helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_prompt_json(filename: str) -> dict:
        path = PROMPTS_DIR / filename
        if path.exists():
            return json.loads(path.read_text())
        return {}

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def _build_system_prompt(self, language: str, spark_version: str) -> str:
        ...

    @abstractmethod
    def _build_user_message(self, code: str, context_hint: str) -> str:
        ...

    @abstractmethod
    def parse_score(self, raw_text: str) -> int:
        """Extract integer score (0-100) from LLM output."""
        ...

    # ── Public run method ──────────────────────────────────────────────────────

    def run(
        self,
        code: str,
        language: str,
        spark_version: str = "3.4",
        context_hint: str = "",
    ) -> dict:
        """
        Run this agent on the given code.

        Returns:
            {
              "agent":   str,
              "score":   int,
              "raw":     str,     # full LLM response
              "prompt_tokens":     int,
              "completion_tokens": int,
            }
        """
        system_prompt = self._build_system_prompt(language, spark_version)
        user_message  = self._build_user_message(code, context_hint)
        raw, pt, ct   = self._call_llm(system_prompt, user_message)
        score         = self.parse_score(raw)

        return {
            "agent":             self.name,
            "score":             score,
            "raw":               raw,
            "prompt_tokens":     pt,
            "completion_tokens": ct,
        }

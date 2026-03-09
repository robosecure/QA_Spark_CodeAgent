"""
Token budget enforcement.
Estimates token usage before API calls and enforces per-review limits.
Graceful degradation: always returns a usable result, never hard-crashes.
"""
CHARS_PER_TOKEN = 4
SYSTEM_PROMPT_OVERHEAD = 600   # tokens per agent for system prompt
COMPLETION_OVERHEAD = 800      # expected completion per agent call
AGENTS = 3


class TokenBudgetExceeded(Exception):
    """Raised when a review would exceed the configured token budget."""
    def __init__(self, estimated: int, limit: int):
        self.estimated = estimated
        self.limit = limit
        super().__init__(
            f"Estimated {estimated:,} tokens exceeds budget of {limit:,}. "
            f"The file will be automatically chunked into smaller sections."
        )


class TokenBudget:
    def __init__(self, cfg):
        self.max_per_review = getattr(cfg, "max_tokens_per_review", 20_000)
        self.max_per_chunk = getattr(cfg, "max_chunk_tokens", 2_500)

    def estimate_code_tokens(self, code: str) -> int:
        return max(1, len(code) // CHARS_PER_TOKEN)

    def estimate_full_review(self, code: str) -> int:
        code_tokens = self.estimate_code_tokens(code)
        return AGENTS * (SYSTEM_PROMPT_OVERHEAD + code_tokens + COMPLETION_OVERHEAD)

    def needs_chunking(self, code: str) -> bool:
        return self.estimate_code_tokens(code) > self.max_per_chunk

    def check(self, code: str) -> dict:
        """
        Returns a status dict. Never raises — caller decides how to handle.
        """
        code_tokens = self.estimate_code_tokens(code)
        full_estimate = self.estimate_full_review(code)
        will_chunk = self.needs_chunking(code)

        return {
            "code_tokens": code_tokens,
            "estimated_total": full_estimate,
            "within_budget": full_estimate <= self.max_per_review,
            "needs_chunking": will_chunk,
            "max_per_review": self.max_per_review,
            "max_per_chunk": self.max_per_chunk,
        }

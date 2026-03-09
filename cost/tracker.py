"""
Per-review, per-agent cost tracker.
Accumulates token usage across agent calls and computes USD cost.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
from cost.pricing import compute_cost


@dataclass
class AgentUsage:
    agent_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class CostTracker:
    """
    Tracks token usage and USD cost per agent within a single review session.
    Usage:
        tracker = CostTracker(provider="azure", model_name="BDF-GLB-GPT-5")
        tracker.record("security_agent", prompt_tokens=800, completion_tokens=300)
        tracker.record("performance_agent", prompt_tokens=1200, completion_tokens=500)
        summary = tracker.summary()
    """

    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        self._agents: Dict[str, AgentUsage] = {}

    def record(self, agent_name: str, prompt_tokens: int, completion_tokens: int):
        """Accumulate token usage for an agent (can be called multiple times for chunked reviews)."""
        if agent_name not in self._agents:
            self._agents[agent_name] = AgentUsage(agent_name=agent_name)
        entry = self._agents[agent_name]
        entry.prompt_tokens += prompt_tokens
        entry.completion_tokens += completion_tokens
        entry.cost_usd = compute_cost(
            self.provider, self.model_name,
            entry.prompt_tokens, entry.completion_tokens,
        )

    def summary(self) -> dict:
        """Return a structured summary of all agent costs and totals."""
        agents_out = {}
        total_prompt = 0
        total_completion = 0
        total_cost = 0.0

        for name, usage in self._agents.items():
            agents_out[name] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cost_usd": usage.cost_usd,
            }
            total_prompt += usage.prompt_tokens
            total_completion += usage.completion_tokens
            total_cost += usage.cost_usd

        return {
            "provider": self.provider,
            "model": self.model_name,
            "agents": agents_out,
            "totals": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "cost_usd": round(total_cost, 6),
            },
        }

    def total_cost_usd(self) -> float:
        return round(sum(a.cost_usd for a in self._agents.values()), 6)

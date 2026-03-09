"""
Orchestrator — coordinates SecurityAgent, PerformanceAgent, and PracticesAgent.

Flow:
  1. Sanitize code (mask secrets before any LLM call)
  2. Check embedding cache for exact or similar matches → skip/reduce LLM calls
  3. Chunk large files if needed
  4. Run all three agents (sequentially to conserve memory; parallel opt is future)
  5. Compute weighted composite score (language-dependent weights)
  6. Store result in embedding cache
  7. Log cost to ROI log

Composite score weights by language:
  PySpark / SparkSQL / Scala:
    security=0.30, performance=0.45, practices=0.25
  Impala:
    security=0.25, performance=0.50, practices=0.25
  Python (non-Spark):
    security=0.40, performance=0.25, practices=0.35
"""
import logging
import re
from typing import Optional

from config import Config
from cost.tracker import CostTracker
from cost.roi_logger import log_review
from agent.sanitizer import Sanitizer as _Sanitizer
from agent.chunker import CodeChunker, CodeChunk
from agent.token_budget import TokenBudget
from agent.embedding_cache import get_cache
from agent.security_agent import SecurityAgent
from agent.performance_agent import PerformanceAgent
from agent.practices_agent import PracticesAgent
from mcp_server.cloudera_mcp import get_mcp_server
from audit.audit_logger import log_review_session

_sanitizer = _Sanitizer()

logger = logging.getLogger(__name__)

# ── Weight tables (must sum to 1.0 per language) ──────────────────────────────
WEIGHTS: dict[str, dict[str, float]] = {
    "pyspark":  {"security": 0.30, "performance": 0.45, "practices": 0.25},
    "sparksql": {"security": 0.30, "performance": 0.45, "practices": 0.25},
    "scala":    {"security": 0.30, "performance": 0.45, "practices": 0.25},
    "impala":   {"security": 0.25, "performance": 0.50, "practices": 0.25},
    "python":   {"security": 0.40, "performance": 0.25, "practices": 0.35},
}
DEFAULT_WEIGHTS = {"security": 0.33, "performance": 0.34, "practices": 0.33}


def _extract_key_findings(raw_text: str) -> list[str]:
    """Pull bullet lines from 'Key * Findings' section."""
    m = re.search(r'Key \w+ Findings.*?\n((?:[-*•].+\n?)+)', raw_text, re.IGNORECASE)
    if not m:
        return []
    bullets = re.findall(r'[-*•]\s*(.+)', m.group(1))
    return [b.strip() for b in bullets[:5]]


def _extract_corrected_code(raw_text: str) -> str:
    """Pull the corrected code block from PracticesAgent output."""
    m = re.search(r'### Corrected.*?Code\n```[^\n]*\n(.*?)```', raw_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


class Orchestrator:
    def __init__(self, config: Config, file_name: str = "<pasted>"):
        self.config = config
        self.file_name = file_name

    def review(
        self,
        code: str,
        language: str,
        spark_version: Optional[str] = None,
        project_context: str = "",
        user_context: str = "",
        extra_tables: Optional[list] = None,
    ) -> dict:
        """
        Full multi-agent review pipeline.

        Returns:
            {
              "composite_score": int,
              "certified": bool,
              "language": str,
              "agents": {
                  "security_agent":    { "score": int, "raw": str, ... },
                  "performance_agent": { "score": int, "raw": str, ... },
                  "practices_agent":   { "score": int, "raw": str, ... },
              },
              "weights": dict,
              "corrected_code": str,
              "key_findings": [str],
              "cost": dict,      # from CostTracker.summary()
              "chunks": int,
              "cache_hit": bool,
              "cache_exact": bool,
            }
        """
        spark_ver = spark_version or self.config.spark_version
        lang = language.lower()
        weights = WEIGHTS.get(lang, DEFAULT_WEIGHTS)
        tracker = CostTracker(self.config.provider, self.config.model_name)
        cache = get_cache()

        # ── 1. Sanitize ────────────────────────────────────────────────────────
        safe_code, redactions = _sanitizer.run(code)
        if redactions:
            logger.warning("Sanitizer masked %d secret(s) before LLM call", len(redactions))

        # ── 2. Exact cache hit ─────────────────────────────────────────────────
        cached = cache.exact_hit(safe_code, lang)
        if cached:
            logger.info("Exact cache hit — skipping LLM calls")
            return self._from_cache(cached, tracker, lang, cached_exact=True)

        # ── 3. Similar context hint ────────────────────────────────────────────
        cache_hint = cache.similar_context(safe_code, lang)

        # ── 3b. MCP metadata context ───────────────────────────────────────────
        mcp = get_mcp_server()
        mcp_context = mcp.get_context(safe_code, lang, extra_tables=extra_tables or [])

        # ── 3c. Assemble full context header injected into every agent ─────────
        context_parts = []
        if user_context.strip():
            context_parts.append(
                "## Business Context (provided by developer)\n"
                f"{user_context.strip()}\n"
            )
        if project_context.strip():
            context_parts.append(project_context.strip())
        if mcp_context.strip():
            context_parts.append(mcp_context.strip())
        if cache_hint.strip():
            context_parts.append(cache_hint.strip())
        context_hint = "\n\n".join(context_parts) + ("\n\n" if context_parts else "")

        # ── 4. Chunking ────────────────────────────────────────────────────────
        budget = TokenBudget(self.config)
        chunker = CodeChunker(max_tokens=self.config.max_tokens_per_chunk)
        if budget.needs_chunking(safe_code):
            chunks = chunker.split(safe_code, lang)
        else:
            chunks = [CodeChunk(safe_code, 0, 1)]

        # ── 5. Run agents across all chunks ───────────────────────────────────
        sec_agent  = SecurityAgent(self.config, tracker)
        perf_agent = PerformanceAgent(self.config, tracker)
        prac_agent = PracticesAgent(self.config, tracker)

        agent_results: dict = {
            "security_agent":    {"score": 0, "raw": "", "chunks_raw": []},
            "performance_agent": {"score": 0, "raw": "", "chunks_raw": []},
            "practices_agent":   {"score": 0, "raw": "", "chunks_raw": []},
        }

        for chunk in chunks:
            chunk_hint = context_hint
            if chunk.total > 1:
                chunk_hint = f"[Chunk {chunk.index + 1}/{chunk.total}]\n{context_hint}"

            for agent, key in [
                (sec_agent,  "security_agent"),
                (perf_agent, "performance_agent"),
                (prac_agent, "practices_agent"),
            ]:
                res = agent.run(chunk.code, lang, spark_ver, chunk_hint)
                agent_results[key]["chunks_raw"].append(res["raw"])
                agent_results[key]["score"] = max(
                    agent_results[key]["score"], res["score"]
                ) if chunk.index == 0 else min(
                    agent_results[key]["score"], res["score"]
                )
                # For multi-chunk: take worst score (conservative)
                if chunk.total > 1 and chunk.index > 0:
                    agent_results[key]["score"] = min(
                        agent_results[key]["score"], res["score"]
                    )
                else:
                    agent_results[key]["score"] = res["score"]

            # Accumulate raw text (last chunk's output is most representative)
        for key in agent_results:
            agent_results[key]["raw"] = "\n\n---\n\n".join(agent_results[key]["chunks_raw"])

        # ── 6. Composite score ─────────────────────────────────────────────────
        sec_score  = agent_results["security_agent"]["score"]
        perf_score = agent_results["performance_agent"]["score"]
        prac_score = agent_results["practices_agent"]["score"]

        composite = round(
            sec_score  * weights["security"]    +
            perf_score * weights["performance"] +
            prac_score * weights["practices"]
        )

        # Hard rule: if security score < 40, composite cannot exceed 60
        if sec_score < 40 and composite > 60:
            composite = 60

        certified = composite >= self.config.pass_threshold

        # ── 7. Extract outputs ─────────────────────────────────────────────────
        corrected_code = _extract_corrected_code(agent_results["practices_agent"]["raw"])
        all_findings: list[str] = []
        for key in agent_results:
            all_findings.extend(_extract_key_findings(agent_results[key]["raw"]))

        # ── 8. Cache store ─────────────────────────────────────────────────────
        cache.store(safe_code, lang, composite, all_findings[:6])

        # ── 9. ROI log + Audit log ─────────────────────────────────────────────
        cost_summary = tracker.summary()
        log_review(
            language=lang,
            file_name=self.file_name,
            score=composite,
            certified=certified,
            cost_summary=cost_summary,
            review_mode="multi",
            chunks_processed=len(chunks),
            cache_hit=False,
            reviewer_id=None,
        )
        log_review_session(
            original_code=code,
            sanitized_code=safe_code,
            corrected_code=corrected_code,
            language=lang,
            file_name=self.file_name,
            spark_version=spark_ver,
            composite_score=composite,
            certified=certified,
            agent_scores={k: v["score"] for k, v in agent_results.items()},
            weights=weights,
            agent_raw_outputs={k: v["raw"] for k, v in agent_results.items()},
            user_context=user_context,
            project_context=project_context,
            mcp_context_used=bool(mcp_context),
            cache_hit=False,
            cache_exact=False,
            cost_summary=cost_summary,
            chunks_processed=len(chunks),
            review_mode="multi",
        )

        return {
            "composite_score": composite,
            "certified": certified,
            "language": lang,
            "agents": {
                k: {"score": v["score"], "raw": v["raw"]}
                for k, v in agent_results.items()
            },
            "weights": weights,
            "corrected_code": corrected_code,
            "key_findings": all_findings,
            "cost": cost_summary,
            "chunks": len(chunks),
            "cache_hit": False,
            "cache_exact": False,
            "mcp_used": bool(mcp_context),
            "project_context_used": bool(project_context),
            "user_context": user_context,
        }

    def _from_cache(self, cached: dict, tracker: CostTracker, lang: str, cached_exact: bool) -> dict:
        score = cached.get("score", 0)
        certified = score >= self.config.pass_threshold
        cost_summary = tracker.summary()
        log_review(
            language=lang,
            file_name=self.file_name,
            score=score,
            certified=certified,
            cost_summary=cost_summary,
            review_mode="multi",
            chunks_processed=0,
            cache_hit=True,
        )
        return {
            "composite_score": score,
            "certified": certified,
            "language": lang,
            "agents": {},
            "weights": WEIGHTS.get(lang, DEFAULT_WEIGHTS),
            "corrected_code": "",
            "key_findings": cached.get("key_findings", []),
            "cost": cost_summary,
            "chunks": 0,
            "cache_hit": True,
            "cache_exact": cached_exact,
        }

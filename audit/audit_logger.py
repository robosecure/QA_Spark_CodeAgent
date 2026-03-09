"""
Audit Logger — comprehensive before/after capture for every review session.

Every review writes one JSONL record containing:
  - Original code (before)
  - Sanitized code (after masking)
  - Corrected code (agent output)
  - All agent raw outputs
  - Scores (per-agent and composite)
  - Full context injected (user, project, MCP)
  - Cost and token breakdown
  - Session metadata

This data is the source of truth for:
  1. Compliance auditing ("who reviewed what, when")
  2. Model fine-tuning (before/after pairs with quality scores)
  3. Prompt improvement (low-scoring reviews for human review)
  4. ROI reporting

Log location: data/audit_log.jsonl
"""
import json
import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

AUDIT_LOG = Path(__file__).parent.parent / "data" / "audit_log.jsonl"
MAX_CODE_CHARS_IN_LOG = 50_000   # cap very large files in audit record


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _truncate(text: str, limit: int = MAX_CODE_CHARS_IN_LOG) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated — {len(text):,} chars total]"


def log_review_session(
    *,
    # Code state
    original_code: str,
    sanitized_code: str,
    corrected_code: str = "",
    language: str,
    file_name: str = "<pasted>",
    spark_version: str = "3.4",

    # Scores
    composite_score: int,
    certified: bool,
    agent_scores: dict,           # {"security_agent": 72, ...}
    weights: dict,                # {"security": 0.30, ...}

    # Agent outputs
    agent_raw_outputs: dict,      # {"security_agent": "### Security Findings...", ...}

    # Context injected
    user_context: str = "",
    project_context: str = "",
    mcp_context_used: bool = False,
    cache_hit: bool = False,
    cache_exact: bool = False,

    # Cost
    cost_summary: dict,           # from CostTracker.summary()
    chunks_processed: int = 1,

    # Session
    reviewer_id: Optional[str] = None,
    session_id: Optional[str] = None,
    review_mode: str = "multi",
):
    """
    Append one complete review session to the audit log.
    Never raises — audit failure must not block a review.
    """
    try:
        AUDIT_LOG.parent.mkdir(exist_ok=True)

        # Detect what changed between original and corrected
        code_was_modified = bool(corrected_code.strip()) and corrected_code.strip() != original_code.strip()

        record = {
            "schema_version": "1.1",
            "timestamp":       _now(),
            "session_id":      session_id,
            "reviewer_id":     reviewer_id,

            # ── Identity ──────────────────────────────────────────────────
            "file_name":       file_name,
            "language":        language,
            "spark_version":   spark_version,
            "review_mode":     review_mode,

            # ── Code state ────────────────────────────────────────────────
            "code": {
                "original":            _truncate(original_code),
                "original_hash":       _hash(original_code),
                "original_length":     len(original_code),
                "sanitized_hash":      _hash(sanitized_code),
                "secrets_redacted":    original_code != sanitized_code,
                "corrected":           _truncate(corrected_code) if corrected_code else "",
                "code_was_modified":   code_was_modified,
            },

            # ── Scores ───────────────────────────────────────────────────
            "scores": {
                "composite":    composite_score,
                "certified":    certified,
                "per_agent":    agent_scores,
                "weights":      weights,
                "pass_threshold": int(os.environ.get("PASS_THRESHOLD", "95")),
            },

            # ── Agent outputs (full, for training) ───────────────────────
            "agent_outputs": {
                agent: _truncate(output, 20_000)
                for agent, output in (agent_raw_outputs or {}).items()
            },

            # ── Context ──────────────────────────────────────────────────
            "context": {
                "user_context_provided":    bool(user_context.strip()),
                "user_context":             user_context,
                "project_context_provided": bool(project_context.strip()),
                "project_context_summary":  _truncate(project_context, 2000),
                "mcp_metadata_used":        mcp_context_used,
                "cache_hit":                cache_hit,
                "cache_exact":              cache_exact,
                "chunks_processed":         chunks_processed,
            },

            # ── Cost ─────────────────────────────────────────────────────
            "cost": cost_summary,

            # ── Training labels (for fine-tuning) ────────────────────────
            "training": {
                "quality_label":      _quality_label(composite_score),
                "has_corrected_code": code_was_modified,
                "usable_for_training": composite_score > 0 and bool(agent_raw_outputs),
                "training_pairs": _build_training_pairs(
                    original_code, agent_raw_outputs, agent_scores
                ),
            },
        }

        with AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    except Exception as e:
        logger.warning("Audit log write failed (non-fatal): %s", e)


def _quality_label(score: int) -> str:
    if score >= 95:   return "excellent"
    if score >= 80:   return "good"
    if score >= 60:   return "needs_improvement"
    if score >= 40:   return "poor"
    return "critical"


def _build_training_pairs(
    original_code: str,
    agent_raw_outputs: dict,
    agent_scores: dict,
) -> list[dict]:
    """
    Build structured training pairs: (input, expected_output) for each agent.
    Used for future fine-tuning of reviewer models.
    """
    pairs = []
    for agent_name, raw_output in (agent_raw_outputs or {}).items():
        if not raw_output.strip():
            continue
        pairs.append({
            "agent":        agent_name,
            "input_code":   _truncate(original_code, 8000),
            "output":       _truncate(raw_output, 8000),
            "score":        agent_scores.get(agent_name, -1),
            "pair_hash":    _hash(original_code + agent_name),
        })
    return pairs


# ── Query helpers ──────────────────────────────────────────────────────────────

def load_audit_records(
    last_n: Optional[int] = None,
    language: Optional[str] = None,
    certified_only: bool = False,
    min_score: int = 0,
    has_corrections: bool = False,
) -> list[dict]:
    """Load and optionally filter audit records."""
    if not AUDIT_LOG.exists():
        return []
    records = []
    with AUDIT_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                # Filters
                if language and r.get("language") != language:
                    continue
                if certified_only and not r.get("scores", {}).get("certified"):
                    continue
                score = r.get("scores", {}).get("composite", 0)
                if score < min_score:
                    continue
                if has_corrections and not r.get("code", {}).get("code_was_modified"):
                    continue
                records.append(r)
            except json.JSONDecodeError:
                continue
    if last_n:
        records = records[-last_n:]
    return records


def export_training_data(output_path: Optional[str] = None) -> str:
    """
    Export all usable training pairs as a JSONL file.
    Returns the output file path.
    """
    records = load_audit_records()
    pairs = []
    for r in records:
        if not r.get("training", {}).get("usable_for_training"):
            continue
        for pair in r.get("training", {}).get("training_pairs", []):
            pairs.append({
                **pair,
                "language":  r.get("language"),
                "timestamp": r.get("timestamp"),
                "quality":   r.get("training", {}).get("quality_label"),
            })

    out_path = output_path or str(
        Path(__file__).parent.parent / "data" / "training_export.jsonl"
    )
    Path(out_path).parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    logger.info("Exported %d training pairs to %s", len(pairs), out_path)
    return out_path


def audit_summary() -> dict:
    """Aggregate stats from the audit log for the admin dashboard."""
    records = load_audit_records()
    if not records:
        return {"total_sessions": 0}

    total     = len(records)
    certified = sum(1 for r in records if r.get("scores", {}).get("certified"))
    with_ctx  = sum(1 for r in records if r.get("context", {}).get("user_context_provided"))
    with_corrections = sum(1 for r in records if r.get("code", {}).get("code_was_modified"))
    secrets_found    = sum(1 for r in records if r.get("code", {}).get("secrets_redacted"))
    total_cost       = sum(r.get("cost", {}).get("totals", {}).get("cost_usd", 0) for r in records)
    avg_score        = sum(r.get("scores", {}).get("composite", 0) for r in records) / total

    by_lang: dict = {}
    by_quality: dict = {}
    for r in records:
        lang = r.get("language", "unknown")
        by_lang[lang] = by_lang.get(lang, 0) + 1
        q = r.get("training", {}).get("quality_label", "unknown")
        by_quality[q] = by_quality.get(q, 0) + 1

    return {
        "total_sessions":      total,
        "certified":           certified,
        "pass_rate_pct":       round(certified / total * 100, 1),
        "avg_score":           round(avg_score, 1),
        "with_user_context":   with_ctx,
        "with_corrections":    with_corrections,
        "secrets_detected":    secrets_found,
        "total_cost_usd":      round(total_cost, 6),
        "by_language":         by_lang,
        "by_quality":          by_quality,
        "training_pairs_available": sum(
            len(r.get("training", {}).get("training_pairs", []))
            for r in records
            if r.get("training", {}).get("usable_for_training")
        ),
    }

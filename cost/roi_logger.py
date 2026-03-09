"""
Append-only JSONL log for ROI / cost-per-review tracking.
Each line is one review event with timestamp, language, score, and cost breakdown.

Log location: data/roi_log.jsonl  (create with: mkdir -p data)
Use roi_summary() to aggregate stats for dashboards.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG_FILE = Path(__file__).parent.parent / "data" / "roi_log.jsonl"


def log_review(
    *,
    language: str,
    file_name: str,
    score: int,
    certified: bool,
    cost_summary: dict,
    review_mode: str = "single",          # "single" | "multi"
    chunks_processed: int = 1,
    cache_hit: bool = False,
    reviewer_id: Optional[str] = None,
):
    """
    Append one review event to the JSONL log.

    Args:
        language: Code language (pyspark, impala, python, …)
        file_name: Source file or "<pasted>"
        score: Final composite score (0-100)
        certified: Whether score >= PASS_THRESHOLD
        cost_summary: dict from CostTracker.summary()
        review_mode: "single" or "multi" (multi-agent)
        chunks_processed: Number of code chunks reviewed
        cache_hit: True if result came from exact cache hit
        reviewer_id: Optional user/job identifier for attribution
    """
    LOG_FILE.parent.mkdir(exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "language": language,
        "file_name": file_name,
        "score": score,
        "certified": certified,
        "review_mode": review_mode,
        "chunks_processed": chunks_processed,
        "cache_hit": cache_hit,
        "reviewer_id": reviewer_id,
        "cost": cost_summary.get("totals", {}),
        "agent_breakdown": cost_summary.get("agents", {}),
        "model": cost_summary.get("model", ""),
        "provider": cost_summary.get("provider", ""),
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def roi_summary(last_n: Optional[int] = None) -> dict:
    """
    Aggregate stats from the JSONL log.

    Returns:
        {
          "total_reviews": int,
          "certified": int,
          "pass_rate_pct": float,
          "total_cost_usd": float,
          "avg_cost_usd": float,
          "avg_score": float,
          "cache_hits": int,
          "by_language": { lang: {"reviews": int, "cost_usd": float, "avg_score": float} }
        }
    """
    if not LOG_FILE.exists():
        return {"total_reviews": 0}

    records = []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if last_n:
        records = records[-last_n:]

    if not records:
        return {"total_reviews": 0}

    total = len(records)
    certified = sum(1 for r in records if r.get("certified"))
    total_cost = sum(r.get("cost", {}).get("cost_usd", 0.0) for r in records)
    total_score = sum(r.get("score", 0) for r in records)
    cache_hits = sum(1 for r in records if r.get("cache_hit"))

    by_lang: dict = {}
    for r in records:
        lang = r.get("language", "unknown")
        if lang not in by_lang:
            by_lang[lang] = {"reviews": 0, "cost_usd": 0.0, "total_score": 0}
        by_lang[lang]["reviews"] += 1
        by_lang[lang]["cost_usd"] += r.get("cost", {}).get("cost_usd", 0.0)
        by_lang[lang]["total_score"] += r.get("score", 0)

    lang_summary = {
        lang: {
            "reviews": v["reviews"],
            "cost_usd": round(v["cost_usd"], 6),
            "avg_score": round(v["total_score"] / v["reviews"], 1),
        }
        for lang, v in by_lang.items()
    }

    return {
        "total_reviews": total,
        "certified": certified,
        "pass_rate_pct": round(certified / total * 100, 1),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_usd": round(total_cost / total, 6),
        "avg_score": round(total_score / total, 1),
        "cache_hits": cache_hits,
        "by_language": lang_summary,
    }

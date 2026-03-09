"""
Language auto-detection from file extension and content analysis.

Priority order:
  1. Content-based signals (most reliable — handles misnamed files)
  2. File extension mapping
  3. Fallback: python

Used by: Streamlit UI (GitLab fetch + file upload), CLI
"""
import re
from pathlib import Path
from typing import Optional

# ── Extension → candidate language ────────────────────────────────────────────
EXT_MAP = {
    ".sql":   ["sparksql", "impala"],   # disambiguate by content
    ".py":    ["pyspark", "python"],    # disambiguate by content
    ".scala": ["scala"],
    ".sc":    ["scala"],
    ".txt":   ["python"],               # assume python for plain text
    ".hql":   ["impala"],               # Hive QL → impala dialect
}

# ── Content signals (pattern, language, weight) ────────────────────────────────
# Higher weight = stronger signal. First language to reach threshold wins.
CONTENT_SIGNALS = [
    # PySpark
    (re.compile(r'\bSparkSession\b'),                    "pyspark",  10),
    (re.compile(r'\bfrom\s+pyspark\b'),                  "pyspark",  10),
    (re.compile(r'\bimport\s+pyspark\b'),                "pyspark",  10),
    (re.compile(r'\bsc\s*=\s*SparkContext\b'),           "pyspark",   8),
    (re.compile(r'\.createDataFrame\b'),                 "pyspark",   6),
    (re.compile(r'\.withColumn\b'),                      "pyspark",   5),
    (re.compile(r'\.groupBy\b'),                         "pyspark",   4),
    (re.compile(r'\bspark\.read\b'),                     "pyspark",   6),
    (re.compile(r'\bDataFrame\b'),                       "pyspark",   3),

    # Scala Spark
    (re.compile(r'\bimport\s+org\.apache\.spark\b'),     "scala",    10),
    (re.compile(r'\bobject\s+\w+\s+extends\b'),          "scala",     8),
    (re.compile(r'\bval\s+\w+\s*[:=]'),                  "scala",     5),
    (re.compile(r'\bcase\s+class\b'),                    "scala",     7),
    (re.compile(r'\bimplicit\b'),                        "scala",     6),
    (re.compile(r'\.toDS\(\)'),                          "scala",     7),
    (re.compile(r'\bDataset\['),                         "scala",     7),

    # SparkSQL (SQL with Spark-specific features)
    (re.compile(r'/\*\+\s*(BROADCAST|MERGE|SHUFFLE_HASH|COALESCE)\b', re.I), "sparksql", 10),
    (re.compile(r'\bUSING\s+DELTA\b', re.I),             "sparksql", 10),
    (re.compile(r'\bSPARK_CATALOG\b', re.I),             "sparksql",  8),
    (re.compile(r'\bCREATE\s+TABLE.*USING\s+\w+', re.I),"sparksql",  7),
    (re.compile(r'\bINSERT\s+INTO\s+\w+\s+SELECT\b', re.I), "sparksql", 5),
    (re.compile(r'\bWITH\b.*\bAS\b.*\bSELECT\b', re.I | re.DOTALL), "sparksql", 4),
    (re.compile(r'\bFROM\s+\w+\.\w+\.\w+\b', re.I),     "sparksql",  5),  # 3-part catalog name

    # Impala SQL
    (re.compile(r'\bSTRAIGHT_JOIN\b', re.I),             "impala",   10),
    (re.compile(r'/\*\+\s*BROADCAST\b', re.I),           "impala",    7),
    (re.compile(r'\bCOMPUTE\s+STATS\b', re.I),          "impala",   10),
    (re.compile(r'\bINVALIDATE\s+METADATA\b', re.I),    "impala",   10),
    (re.compile(r'\bREFRESH\s+\w+', re.I),              "impala",    7),
    (re.compile(r'\bKUDU\b', re.I),                     "impala",    8),
    (re.compile(r'\bPARQUET\b', re.I),                  "impala",    4),
    (re.compile(r'\bSTORED\s+AS\b', re.I),              "impala",    6),

    # Plain Python (low weight — only wins if no Spark signals)
    (re.compile(r'\bdef\s+\w+\s*\('),                   "python",    2),
    (re.compile(r'\bimport\s+(os|sys|json|re|datetime)\b'), "python", 2),
    (re.compile(r'\bclass\s+\w+[\(:]'),                  "python",   2),
    (re.compile(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]'), "python", 3),
]

DETECTION_THRESHOLD = 8   # minimum weight to commit to a language


def detect_language(code: str, filename: Optional[str] = None) -> dict:
    """
    Detect code language. Returns:
        {
          "language":   str,      # detected language key
          "confidence": str,      # "high" | "medium" | "low"
          "method":     str,      # "content" | "extension" | "fallback"
          "signals":    [str],    # human-readable signal list
        }
    """
    scores: dict[str, int] = {}
    signals: list[str] = []

    # ── 1. Content analysis ────────────────────────────────────────────────────
    for pattern, lang, weight in CONTENT_SIGNALS:
        if pattern.search(code):
            scores[lang] = scores.get(lang, 0) + weight
            signals.append(f"{lang}: {pattern.pattern[:40]} (+{weight})")

    if scores:
        best_lang = max(scores, key=lambda k: scores[k])
        best_score = scores[best_lang]
        if best_score >= DETECTION_THRESHOLD:
            confidence = "high" if best_score >= 20 else "medium"
            return {
                "language":   best_lang,
                "confidence": confidence,
                "method":     "content",
                "signals":    signals[:6],
                "scores":     scores,
            }

    # ── 2. Extension fallback ──────────────────────────────────────────────────
    if filename:
        ext = Path(filename).suffix.lower()
        candidates = EXT_MAP.get(ext, [])
        if len(candidates) == 1:
            return {
                "language":   candidates[0],
                "confidence": "medium",
                "method":     "extension",
                "signals":    [f"File extension: {ext}"],
                "scores":     scores,
            }
        elif candidates:
            # Multiple candidates but weak content signals — pick highest content score
            for candidate in candidates:
                if candidate in scores:
                    return {
                        "language":   candidate,
                        "confidence": "low",
                        "method":     "extension+content",
                        "signals":    [f"Extension {ext}, weak content signals"],
                        "scores":     scores,
                    }
            # No content signals match candidates — pick first
            return {
                "language":   candidates[0],
                "confidence": "low",
                "method":     "extension",
                "signals":    [f"File extension: {ext}, no content signals"],
                "scores":     scores,
            }

    # ── 3. Last resort ─────────────────────────────────────────────────────────
    return {
        "language":   "python",
        "confidence": "low",
        "method":     "fallback",
        "signals":    ["No extension or content signals detected"],
        "scores":     scores,
    }


def detect_from_file_list(files: list[dict]) -> str:
    """
    Given a list of {name, content} dicts from a repo scan,
    return the dominant language across the project.
    """
    lang_votes: dict[str, int] = {}
    for f in files:
        result = detect_language(f.get("content", ""), f.get("name", ""))
        lang = result["language"]
        weight = {"high": 3, "medium": 2, "low": 1}[result["confidence"]]
        lang_votes[lang] = lang_votes.get(lang, 0) + weight

    if not lang_votes:
        return "python"
    return max(lang_votes, key=lambda k: lang_votes[k])

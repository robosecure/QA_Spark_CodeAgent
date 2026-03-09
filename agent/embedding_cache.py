"""
Lightweight code review cache using TF-IDF cosine similarity.
Finds previously reviewed code with similar patterns and injects
key findings as context — reducing token waste on repeated patterns.

Phase 2 upgrade path: replace _vectorize() with Azure OpenAI
text-embedding-ada-002 calls for semantic (not just lexical) similarity.
Interface is unchanged.
"""
import re
import math
import json
import hashlib
from pathlib import Path
from typing import List, Optional

CACHE_FILE = Path(__file__).parent.parent / "data" / "embedding_cache.json"
MAX_ENTRIES = 200
SIMILARITY_THRESHOLD = 0.35
TOP_K = 3


def _vectorize(text: str) -> dict:
    """TF-IDF token frequency vector over code identifiers."""
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', text.lower())
    freq: dict = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = max(len(tokens), 1)
    return {k: v / total for k, v in freq.items()}


def _cosine(a: dict, b: dict) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[k] * b[k] for k in shared)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (mag_a * mag_b) if mag_a * mag_b else 0.0


def _fingerprint(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]


class EmbeddingCache:
    def __init__(self):
        self._entries: list = []
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        if CACHE_FILE.exists():
            try:
                self._entries = json.loads(CACHE_FILE.read_text())
            except Exception:
                self._entries = []

    def _save(self):
        CACHE_FILE.parent.mkdir(exist_ok=True)
        # Keep only the newest MAX_ENTRIES
        to_save = self._entries[-MAX_ENTRIES:]
        CACHE_FILE.write_text(json.dumps(to_save, indent=2))

    def exact_hit(self, code: str, language: str) -> Optional[dict]:
        """Return cached result for identical code (SHA-256 match)."""
        self._load()
        fp = _fingerprint(code)
        for e in self._entries:
            if e.get("fingerprint") == fp and e.get("language") == language:
                return e
        return None

    def similar_context(self, code: str, language: str) -> str:
        """
        Returns a hint string injected into agent prompts.
        Reduces token waste by telling the LLM what similar code scored.
        """
        self._load()
        vec = _vectorize(code)
        scored = []
        for e in self._entries:
            if e.get("language") != language:
                continue
            sim = _cosine(vec, e.get("vector", {}))
            if sim >= SIMILARITY_THRESHOLD:
                scored.append((sim, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:TOP_K]
        if not top:
            return ""
        lines = ["## Context from similar previously reviewed code:"]
        for _, e in top:
            findings = "; ".join(e.get("key_findings", []))
            lines.append(f"- Score {e.get('score', '?')}/100: {findings}")
        lines.append("")
        return "\n".join(lines)

    def store(self, code: str, language: str, final_score: int, key_findings: List[str]):
        self._load()
        self._entries.append({
            "fingerprint": _fingerprint(code),
            "language": language,
            "vector": _vectorize(code),
            "score": final_score,
            "key_findings": key_findings[:6],
        })
        self._save()


# Module-level singleton
_cache = EmbeddingCache()


def get_cache() -> EmbeddingCache:
    return _cache

"""
Splits large code files into token-safe chunks at logical boundaries.
Each chunk is independently reviewable with full context.
"""
import re
from typing import List

CHARS_PER_TOKEN = 4   # conservative estimate

# Regex to split at natural code boundaries per language
SPLIT_PATTERNS = {
    "python":  r'(?=\n(?:def |class |async def ))',
    "pyspark": r'(?=\n(?:def |class |async def ))',
    "scala":   r'(?=\n(?:def |object |class |trait |val |var ))',
    "sparksql": r'(?<=;)\s*\n',
    "impala":   r'(?<=;)\s*\n',
}
DEFAULT_SPLIT = r'\n{3,}'

# Any chunk with a CRITICAL finding caps composite at this score
CRITICAL_SCORE_CAP = 49


class CodeChunk:
    def __init__(self, code: str, index: int, total: int):
        self.code = code
        self.index = index
        self.total = total
        self.lines = len(code.splitlines())

    def header(self) -> str:
        if self.total == 1:
            return ""
        return f"[Chunk {self.index + 1} of {self.total}]\n"


class CodeChunker:
    def __init__(self, max_tokens: int = 2500):
        self.max_tokens = max_tokens
        self.max_chars = max_tokens * CHARS_PER_TOKEN

    def split(self, code: str, language: str) -> List[CodeChunk]:
        if len(code) <= self.max_chars:
            return [CodeChunk(code, 0, 1)]

        pattern = SPLIT_PATTERNS.get(language, DEFAULT_SPLIT)
        parts = [p.strip() for p in re.split(pattern, code) if p.strip()]

        chunks: List[str] = []
        current = ""
        for part in parts:
            if len(current) + len(part) + 2 <= self.max_chars:
                current += part + "\n"
            else:
                if current.strip():
                    chunks.append(current.strip())
                # If single part is too large, hard split it
                if len(part) > self.max_chars:
                    for i in range(0, len(part), self.max_chars):
                        chunks.append(part[i:i + self.max_chars])
                    current = ""
                else:
                    current = part + "\n"

        if current.strip():
            chunks.append(current.strip())

        if not chunks:
            chunks = [code[:self.max_chars]]

        total = len(chunks)
        return [CodeChunk(c, i, total) for i, c in enumerate(chunks)]

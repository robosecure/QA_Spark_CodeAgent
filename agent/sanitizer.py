"""
Input sanitizer — runs BEFORE any code reaches the LLM API.
Masks secrets, validates encoding, enforces size limits.
The masked placeholders are flagged to the SecurityAgent as pre-detected findings.
"""
import re
from typing import Tuple, List

MAX_CODE_CHARS = 80_000   # ~20K tokens hard limit

# (pattern, replacement_label, severity)
MASK_RULES = [
    (r'(?i)(password|passwd|pwd)\s*=\s*["\']([^"\']{3,})["\']',        r'\1="[REDACTED:password]"',         "CRITICAL"),
    (r'(?i)(api_key|apikey|secret_key|access_key)\s*=\s*["\']([^"\']{8,})["\']',
                                                                         r'\1="[REDACTED:api_key]"',          "CRITICAL"),
    (r'(?i)(token|bearer|auth)\s*=\s*["\']([^"\']{8,})["\']',           r'\1="[REDACTED:token]"',            "CRITICAL"),
    (r'(?i)(client_secret)\s*=\s*["\']([^"\']{8,})["\']',               r'\1="[REDACTED:client_secret]"',    "CRITICAL"),
    (r'\b(AKIA[0-9A-Z]{16})\b',                                          '[REDACTED:aws_access_key]',         "CRITICAL"),
    (r'(?i)AccountKey=[A-Za-z0-9+/=]{20,}',                             'AccountKey=[REDACTED:azure_key]',   "CRITICAL"),
    (r'(?i)(jdbc:[a-z:]+://[^\s"\']*password=[^\s"\'&]*)',               r'[REDACTED:jdbc_url_with_password]',"HIGH"),
    (r'(?i)(conn_str|connection_string)\s*=\s*["\']([^"\']{10,})["\']', r'\1="[REDACTED:conn_string]"',      "HIGH"),
]


class Sanitizer:
    def run(self, code: str) -> Tuple[str, List[dict]]:
        """
        Returns (sanitized_code, findings).
        findings is a list of dicts each with: rule, severity, count.
        Never raises — always returns something safe to send.
        """
        warnings: List[dict] = []

        # Enforce encoding safety
        try:
            code = code.encode("utf-8", errors="replace").decode("utf-8")
        except Exception:
            code = repr(code)

        # Hard size limit — chunker handles splitting; this is a backstop
        if len(code) > MAX_CODE_CHARS:
            code = code[:MAX_CODE_CHARS]
            warnings.append({
                "rule": "SIZE_LIMIT",
                "severity": "INFO",
                "count": 1,
                "message": f"Code truncated to {MAX_CODE_CHARS:,} chars before review. Large files are chunked automatically.",
            })

        # Mask secrets
        for pattern, replacement, severity in MASK_RULES:
            new_code, n = re.subn(pattern, replacement, code)
            if n > 0:
                rule_name = re.sub(r'\[REDACTED:(\w+)\]', r'\1', replacement).upper()
                warnings.append({
                    "rule": rule_name,
                    "severity": severity,
                    "count": n,
                    "message": f"{n} potential secret(s) masked before API submission. SecurityAgent will flag this.",
                })
                code = new_code

        return code, warnings

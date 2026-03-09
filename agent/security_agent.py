"""
SecurityAgent — specialist for security vulnerabilities.

Responsibilities:
  - Hardcoded credentials, tokens, secrets
  - SQL injection / command injection patterns
  - Insecure deserialization (pickle, eval, exec)
  - Overly permissive file/table access
  - OWASP Top 10 patterns relevant to data pipelines

Weight in composite score: 25-40% (language-dependent, set by Orchestrator)

Pre-scan: performs fast regex checks before calling LLM.
If pre-scan finds CRITICAL issues, the score is capped at 50 regardless of LLM output.
"""
import re
import logging
from typing import Optional

from agent.base_agent import BaseAgent
from config import Config
from cost.tracker import CostTracker

logger = logging.getLogger(__name__)

# ── Fast regex pre-scan rules ──────────────────────────────────────────────────
# Each entry: (pattern, description, severity)
PRE_SCAN_RULES = [
    (re.compile(r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']{3,}["\']'),
     "Hardcoded password", "CRITICAL"),
    (re.compile(r'(?i)(api_key|apikey|secret_key|secret)\s*=\s*["\'][^"\']{6,}["\']'),
     "Hardcoded API key / secret", "CRITICAL"),
    (re.compile(r'\b(AKIA[0-9A-Z]{16})\b'),
     "Hardcoded AWS access key", "CRITICAL"),
    (re.compile(r'(?i)DefaultEndpointsProtocol=https;AccountName='),
     "Hardcoded Azure storage connection string", "CRITICAL"),
    (re.compile(r'(?i)jdbc:[a-z]+://[^\s;,\'\"]+password=[^\s;,\'\"]+'),
     "JDBC URL with embedded password", "CRITICAL"),
    (re.compile(r'\beval\s*\('),
     "Use of eval()", "HIGH"),
    (re.compile(r'\bexec\s*\('),
     "Use of exec()", "HIGH"),
    (re.compile(r'pickle\.loads?\s*\('),
     "Insecure pickle deserialization", "HIGH"),
    (re.compile(r'subprocess\.\w+\([^)]*shell\s*=\s*True'),
     "shell=True in subprocess call", "HIGH"),
    (re.compile(r'(?i)select\s+.+\s+from\s+.+where\s+.+["\']?\s*\+'),
     "Potential SQL injection (string concatenation in query)", "HIGH"),
]

SYSTEM_PROMPT_TEMPLATE = """\
You are a **Security Review Agent** specializing in data engineering pipelines.

Your ONLY job is to identify security vulnerabilities in the code provided.
Do NOT comment on performance, style, or correctness — only security.

Language: {language}

## Security Categories to Review
1. **Credential / Secret exposure** — hardcoded passwords, API keys, tokens, connection strings
2. **Injection vulnerabilities** — SQL injection, command injection, LDAP injection
3. **Insecure code execution** — eval(), exec(), pickle.loads(), shell=True
4. **Data exposure** — logging sensitive columns, overly broad SELECT *, wildcard permissions
5. **Insecure deserialization** — pickle, yaml.load() without Loader
6. **Supply chain** — importing from untrusted sources at runtime

## Scoring Rubric (Security only, 0-100)
- 100: No security issues found
- 85-99: Minor informational findings only
- 70-84: Low severity issues (easily mitigated)
- 50-69: Medium severity issues (should fix before production)
- 30-49: High severity issues (hardcoded creds, injection risks)
- 0-29: Critical issues (secrets in code, active injection vulnerability)

## Output Format (STRICTLY follow this)
### Security Findings
[List each finding with: severity, line reference if visible, description, remediation]

### Pre-Scan Flags
[List any patterns already flagged by static analysis, or "None"]

### Security Score
SECURITY_SCORE: XX/100

### Key Security Findings (for cache)
[Bullet list of ≤5 most important findings, one line each]
"""

USER_MESSAGE_TEMPLATE = """\
{pre_scan_section}
{context_hint}
## Code to Review
```
{code}
```

Review this code for security issues only. Follow the output format exactly.
"""


class SecurityAgent(BaseAgent):
    name = "security_agent"

    def _build_system_prompt(self, language: str, spark_version: str) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(language=language)

    def _build_user_message(self, code: str, context_hint: str) -> str:
        pre_scan_findings = self._pre_scan(code)
        pre_section = ""
        if pre_scan_findings:
            lines = ["### Pre-Scan Static Findings (before LLM review)"]
            for desc, sev in pre_scan_findings:
                lines.append(f"- [{sev}] {desc}")
            pre_section = "\n".join(lines) + "\n\n"

        return USER_MESSAGE_TEMPLATE.format(
            pre_scan_section=pre_section,
            context_hint=context_hint,
            code=code,
        )

    def _pre_scan(self, code: str) -> list[tuple[str, str]]:
        """Run fast regex checks. Returns [(description, severity)]."""
        findings = []
        for pattern, desc, severity in PRE_SCAN_RULES:
            if pattern.search(code):
                findings.append((desc, severity))
        return findings

    def _has_critical_prescan(self, code: str) -> bool:
        for pattern, _, severity in PRE_SCAN_RULES:
            if severity == "CRITICAL" and pattern.search(code):
                return True
        return False

    def parse_score(self, raw_text: str) -> int:
        m = re.search(r'SECURITY_SCORE:\s*(\d{1,3})/100', raw_text)
        if m:
            return min(100, max(0, int(m.group(1))))
        # Fallback: look for any XX/100 pattern near "security"
        m2 = re.search(r'(\d{1,3})/100', raw_text)
        if m2:
            return min(100, max(0, int(m2.group(1))))
        return 50  # conservative default if parsing fails

    def run(self, code: str, language: str, spark_version: str = "3.4", context_hint: str = "") -> dict:
        result = super().run(code, language, spark_version, context_hint)

        # Cap score if CRITICAL pre-scan findings exist
        if self._has_critical_prescan(code) and result["score"] > 50:
            logger.warning("SecurityAgent: CRITICAL pre-scan finding — capping score at 50")
            result["score"] = 50
            result["score_capped"] = True
        else:
            result["score_capped"] = False

        return result

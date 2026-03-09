"""
ProjectContextAgent — reads an entire codebase and builds a compressed
workflow summary that is injected into all specialist agent prompts.

This prevents agents from reviewing a single file in isolation.
They understand:
  - What the pipeline does end-to-end
  - Which tables are read and written
  - Data flow between files
  - External dependencies
  - Business purpose (if inferable)

Output: A compressed context block (~500-800 tokens) prepended to each
specialist agent's user message.

Token efficiency:
  - Summarizes rather than passing raw file contents
  - Caps at MAX_CONTEXT_CHARS before LLM summarization call
  - Returns "" gracefully if repo is empty or LLM call fails
"""
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 24_000   # ~6K tokens of raw code fed to summarizer
MAX_FILES = 30               # cap files scanned per repo
SUMMARY_MAX_TOKENS = 800     # target compressed output size

# File extensions to include in codebase scan
CODE_EXTENSIONS = {".py", ".scala", ".sql", ".hql", ".sc", ".yaml", ".yml", ".json", ".txt", ".md"}

# Files/dirs to skip
SKIP_PATTERNS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv", ".env",
    "*.pyc", "test_*", "*_test.py", "*.lock", "requirements.txt",
}

SYSTEM_PROMPT = """\
You are a **Code Pipeline Analyst**. Your job is to read a set of source files
from a data engineering project and produce a concise structured summary.

This summary will be used as context for a code quality review.
Be precise, factual, and token-efficient. Do not pad or repeat yourself.

Output exactly this structure — no more, no less:

## Project Workflow Summary

**Purpose:** [1-2 sentence description of what this pipeline does]

**Data Flow:**
[Bullet list: source → transformation → destination. Include table/file names where visible]

**Tables Read:** [comma-separated list, or "unknown"]
**Tables Written:** [comma-separated list, or "unknown"]
**Key Files:** [comma-separated list of the most important files]

**Technologies:** [e.g. PySpark 3.4, Impala, Delta Lake, Iceberg]

**Potential Concerns:** [1-3 architectural notes the reviewer should be aware of — data volume hints, known complexity, unusual patterns]
"""


class ProjectContextAgent:
    """
    Scans a set of code files and produces a compressed workflow summary.
    Uses the same LLM backend as other agents but is called once per review session.
    """

    def __init__(self, config):
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client:
            return self._client
        import openai
        from azure.identity import ClientSecretCredential

        if self.config.provider == "azure":
            cred = ClientSecretCredential(
                tenant_id=self.config.azure_tenant_id,
                client_id=self.config.azure_service_principal,
                client_secret=self.config.azure_service_principal_secret,
            )
            token = cred.get_token(self.config.azure_token_audience)
            self._client = openai.AzureOpenAI(
                api_key=token.token,
                azure_endpoint=self.config.azure_endpoint,
                api_version=self.config.azure_api_version,
            )
        else:
            self._client = openai.OpenAI(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_api_base,
            )
        return self._client

    def build_from_files(self, files: list[dict], primary_file: str = "") -> str:
        """
        files: list of {"name": str, "content": str}
        primary_file: the file currently under review (highlighted in summary)

        Returns a formatted context string to inject into agent prompts.
        Returns "" if summarization fails.
        """
        if not files:
            return ""

        # ── Build combined code block ─────────────────────────────────────────
        combined = []
        total_chars = 0
        for f in files[:MAX_FILES]:
            name = f.get("name", "unknown")
            content = f.get("content", "")[:4000]  # cap per file
            if not content.strip():
                continue
            block = f"### File: {name}\n```\n{content}\n```\n"
            if total_chars + len(block) > MAX_CONTEXT_CHARS:
                combined.append(f"### File: {name}\n[truncated — too large]\n")
                break
            combined.append(block)
            total_chars += len(block)

        if not combined:
            return ""

        user_message = ""
        if primary_file:
            user_message += f"**Primary file under review:** `{primary_file}`\n\n"
        user_message += "**All project files:**\n\n" + "\n".join(combined)

        # ── Call LLM ─────────────────────────────────────────────────────────
        try:
            client = self._get_client()
            model = (self.config.azure_deployed_model
                     if self.config.provider == "azure"
                     else self.config.openai_model_name)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
            )
            summary = response.choices[0].message.content or ""
            return self._format_for_injection(summary)
        except Exception as e:
            logger.warning("ProjectContextAgent failed: %s — continuing without codebase context", e)
            return self._lightweight_summary(files, primary_file)

    def _format_for_injection(self, summary: str) -> str:
        return (
            "---\n"
            "## Codebase Context (read before reviewing)\n"
            f"{summary.strip()}\n"
            "---\n\n"
        )

    def _lightweight_summary(self, files: list[dict], primary_file: str) -> str:
        """
        Fallback: extract tables and imports with regex — no LLM call.
        Used when LLM summarization fails.
        """
        tables_read: set[str] = set()
        tables_written: set[str] = set()
        file_names = [f.get("name", "") for f in files]

        for f in files:
            content = f.get("content", "")
            # SQL read patterns
            for m in re.finditer(r'\bFROM\s+([\w.]+)', content, re.I):
                tables_read.add(m.group(1))
            # SQL write patterns
            for m in re.finditer(r'\bINSERT\s+(?:INTO|OVERWRITE)\s+([\w.]+)', content, re.I):
                tables_written.add(m.group(1))
            # PySpark read
            for m in re.finditer(r'\.table\(["\']([^"\']+)["\']', content):
                tables_read.add(m.group(1))
            # PySpark write
            for m in re.finditer(r'\.saveAsTable\(["\']([^"\']+)["\']', content):
                tables_written.add(m.group(1))

        lines = ["---", "## Codebase Context (lightweight — LLM summary unavailable)"]
        lines.append(f"**Files in project:** {', '.join(file_names[:10])}")
        if primary_file:
            lines.append(f"**File under review:** {primary_file}")
        if tables_read:
            lines.append(f"**Tables Read (detected):** {', '.join(sorted(tables_read)[:10])}")
        if tables_written:
            lines.append(f"**Tables Written (detected):** {', '.join(sorted(tables_written)[:10])}")
        lines.append("---\n")
        return "\n".join(lines)

    def build_from_gitlab(
        self,
        gitlab_url: str,
        token: str,
        ref: str,
        path: str,
        primary_file: str = "",
    ) -> str:
        """
        Fetch all code files from a GitLab repo path and build context.
        Returns "" gracefully on any error.
        """
        try:
            files = _fetch_gitlab_tree(gitlab_url, token, ref, path)
            return self.build_from_files(files, primary_file=primary_file)
        except Exception as e:
            logger.warning("GitLab codebase fetch failed: %s", e)
            return ""


def _fetch_gitlab_tree(project_url: str, token: str, ref: str, path: str) -> list[dict]:
    """Fetch all code files from a GitLab repo directory."""
    import requests
    import urllib.parse

    project_url = project_url.rstrip("/")
    parts = project_url.split("/")
    gitlab_base = "/".join(parts[:3])
    namespace    = "/".join(parts[3:])
    encoded_ns   = urllib.parse.quote_plus(namespace)

    headers = {"PRIVATE-TOKEN": token}

    # List tree
    tree_url = (
        f"{gitlab_base}/api/v4/projects/{encoded_ns}/repository/tree"
        f"?ref={ref}&path={urllib.parse.quote(path)}&recursive=true&per_page=100"
    )
    resp = requests.get(tree_url, headers=headers, timeout=15)
    if resp.status_code != 200:
        return []

    items = resp.json()
    files: list[dict] = []
    for item in items:
        if item.get("type") != "blob":
            continue
        item_path = item.get("path", "")
        ext = Path(item_path).suffix.lower()
        if ext not in CODE_EXTENSIONS:
            continue
        # Skip test files and obvious non-code
        name = Path(item_path).name
        if any(name.startswith(skip.rstrip("*")) or name.endswith(skip.lstrip("*"))
               for skip in SKIP_PATTERNS if "*" in skip):
            continue

        # Fetch file content
        encoded_file = urllib.parse.quote_plus(item_path)
        file_url = (
            f"{gitlab_base}/api/v4/projects/{encoded_ns}"
            f"/repository/files/{encoded_file}/raw?ref={ref}"
        )
        try:
            fr = requests.get(file_url, headers=headers, timeout=10)
            if fr.status_code == 200:
                files.append({"name": item_path, "content": fr.text})
        except Exception:
            continue

        if len(files) >= MAX_FILES:
            break

    return files

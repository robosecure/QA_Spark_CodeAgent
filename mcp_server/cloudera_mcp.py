"""
Cloudera MCP (Metadata Context Provider) Server.

Provides structured Cloudera/Hive/Impala metadata to the agent system.
This is NOT a network server — it is an in-process tool registry that
agents call to enrich their prompts with real cluster metadata.

Architecture:
  - MCPServer holds a HiveClient and a set of registered tools
  - Each tool is a callable that returns a formatted string
  - Orchestrator calls mcp.get_context(tables, code) before running agents
  - Result is injected into all agent prompts as "## Live Metadata Context"

Graceful degradation:
  - If HIVE_HOST is not set → returns ""
  - If any tool fails → logs warning, skips that tool
  - Never blocks a review

Future: Replace in-process calls with actual MCP protocol over stdio/HTTP
once the MCP Python SDK stabilizes for production use.
"""
import re
import logging
from typing import Optional

from mcp_server.hive_client import HiveClient

logger = logging.getLogger(__name__)


class ClouderaMCPServer:
    """
    In-process MCP server providing Cloudera metadata context to agents.

    Usage:
        mcp = ClouderaMCPServer()
        context = mcp.get_context(code=code, language=language)
        # inject context into agent prompts
    """

    def __init__(self):
        self.client = HiveClient()

    @property
    def is_available(self) -> bool:
        return self.client.is_configured

    def get_context(self, code: str, language: str, extra_tables: Optional[list] = None) -> str:
        """
        Main entry point. Extracts table names from code, fetches metadata,
        returns a formatted context string.

        Returns "" if not configured or no tables found.
        """
        if not self.is_available:
            return ""

        tables = self._extract_tables(code, language)
        if extra_tables:
            tables = list(dict.fromkeys(tables + extra_tables))   # dedup, preserve order

        if not tables:
            return ""

        logger.info("MCPServer: fetching metadata for tables: %s", tables)
        return self.client.format_context_for_agent(tables)

    def get_table_schema(self, table_name: str) -> dict:
        return self.client.get_table_schema(table_name)

    def get_partition_info(self, table_name: str) -> dict:
        return self.client.get_partition_info(table_name)

    def get_column_stats(self, table_name: str, column_name: str) -> dict:
        return self.client.get_column_stats(table_name, column_name)

    def list_tables(self, database: Optional[str] = None) -> list[str]:
        return self.client.list_tables(database)

    def list_databases(self) -> list[str]:
        return self.client.list_databases()

    def health_check(self) -> dict:
        """
        Returns connection status. Used by Streamlit UI sidebar.
        """
        if not self.client.is_configured:
            return {
                "connected": False,
                "reason": "HIVE_HOST not configured",
                "host": None,
            }
        try:
            dbs = self.client.list_databases()
            return {
                "connected": True,
                "host": self.client.host,
                "port": self.client.port,
                "databases": dbs[:10],
                "reason": "OK",
            }
        except Exception as e:
            return {
                "connected": False,
                "host": self.client.host,
                "reason": str(e),
            }

    # ── Table extraction from code ─────────────────────────────────────────────

    def _extract_tables(self, code: str, language: str) -> list[str]:
        """
        Extract table names referenced in code using language-appropriate patterns.
        Returns deduped list preserving order of first appearance.
        """
        tables = []
        seen = set()

        patterns = _TABLE_PATTERNS.get(language.lower(), _TABLE_PATTERNS["sql"])
        for pattern in patterns:
            for m in pattern.finditer(code):
                name = m.group(1).strip().strip("`\"'")
                if name and name.lower() not in _SQL_KEYWORDS and name not in seen:
                    seen.add(name)
                    tables.append(name)

        return tables[:20]   # cap


# ── Regex patterns per language ────────────────────────────────────────────────
_TABLE_PATTERNS = {
    "sql": [
        re.compile(r'\bFROM\s+([\w.`]+)', re.I),
        re.compile(r'\bJOIN\s+([\w.`]+)', re.I),
        re.compile(r'\bINSERT\s+(?:INTO|OVERWRITE)\s+([\w.`]+)', re.I),
        re.compile(r'\bUPDATE\s+([\w.`]+)', re.I),
        re.compile(r'\bMERGE\s+INTO\s+([\w.`]+)', re.I),
    ],
    "impala": [
        re.compile(r'\bFROM\s+([\w.`]+)', re.I),
        re.compile(r'\bJOIN\s+([\w.`]+)', re.I),
        re.compile(r'\bINSERT\s+(?:INTO|OVERWRITE)\s+(?:TABLE\s+)?([\w.`]+)', re.I),
        re.compile(r'\bCOMPUTE\s+STATS\s+([\w.`]+)', re.I),
        re.compile(r'\bREFRESH\s+([\w.`]+)', re.I),
    ],
    "sparksql": [
        re.compile(r'\bFROM\s+([\w.`]+)', re.I),
        re.compile(r'\bJOIN\s+([\w.`]+)', re.I),
        re.compile(r'\bINSERT\s+(?:INTO|OVERWRITE)\s+(?:TABLE\s+)?([\w.`]+)', re.I),
        re.compile(r'\bMERGE\s+INTO\s+([\w.`]+)', re.I),
    ],
    "pyspark": [
        re.compile(r'\.table\(["\']([^"\']+)["\']'),
        re.compile(r'spark\.sql\(["\'].*?FROM\s+([\w.`]+)', re.I),
        re.compile(r'\.saveAsTable\(["\']([^"\']+)["\']'),
        re.compile(r'\.insertInto\(["\']([^"\']+)["\']'),
        re.compile(r'spark\.read\.\w+\(["\']([^"\']+)["\']'),
        re.compile(r'DeltaTable\.forName\([^,]+,\s*["\']([^"\']+)["\']'),
    ],
    "scala": [
        re.compile(r'\.table\(["\']([^"\']+)["\']'),
        re.compile(r'spark\.sql\(""".*?FROM\s+([\w.`]+)', re.I | re.DOTALL),
        re.compile(r'\.saveAsTable\(["\']([^"\']+)["\']'),
        re.compile(r'\.insertInto\(["\']([^"\']+)["\']'),
    ],
    "python": [
        re.compile(r'FROM\s+([\w.`]+)', re.I),
        re.compile(r'INTO\s+([\w.`]+)', re.I),
    ],
}

_SQL_KEYWORDS = {
    "select", "where", "from", "join", "on", "and", "or", "not", "null",
    "true", "false", "as", "by", "group", "order", "having", "limit",
    "with", "case", "when", "then", "else", "end", "in", "is", "like",
    "between", "exists", "union", "all", "distinct", "top", "set",
    "values", "into", "table", "view", "database", "schema",
}


# ── Module-level singleton ─────────────────────────────────────────────────────
_mcp_server: Optional[ClouderaMCPServer] = None


def get_mcp_server() -> ClouderaMCPServer:
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = ClouderaMCPServer()
    return _mcp_server

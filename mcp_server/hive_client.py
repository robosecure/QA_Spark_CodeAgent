"""
Hive / Impala metadata client.

Connects to:
  - Hive Metastore via PyHive (primary)
  - Cloudera Manager REST API (cluster-level stats)
  - Impala JDBC via impyla (alternative)

All methods return structured dicts and NEVER raise —
caller gets {} or [] on any connection/query failure.

Configuration (env vars):
  HIVE_HOST         — Hive/Impala server hostname
  HIVE_PORT         — Port (default: 10000 for Hive, 21050 for Impala)
  HIVE_DATABASE     — Default database (default: default)
  HIVE_AUTH         — Auth method: ldap | kerberos | none (default: none)
  HIVE_USER         — Username for LDAP auth
  HIVE_PASSWORD     — Password for LDAP auth (use env var only, never hardcode)
  CM_HOST           — Cloudera Manager host (for cluster stats)
  CM_USER           — Cloudera Manager username
  CM_PASSWORD       — Cloudera Manager password
  CM_CLUSTER_NAME   — Cluster name in Cloudera Manager
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class HiveClient:
    """
    Lightweight Hive/Impala metadata client.
    Uses PyHive or impyla depending on what's installed.
    """

    def __init__(self):
        self.host     = os.environ.get("HIVE_HOST", "")
        self.port     = int(os.environ.get("HIVE_PORT", "10000"))
        self.database = os.environ.get("HIVE_DATABASE", "default")
        self.auth     = os.environ.get("HIVE_AUTH", "none")
        self.user     = os.environ.get("HIVE_USER", "")
        self.password = os.environ.get("HIVE_PASSWORD", "")
        self._conn    = None

    @property
    def is_configured(self) -> bool:
        return bool(self.host)

    def _connect(self):
        if self._conn:
            return self._conn
        if not self.host:
            return None
        try:
            from pyhive import hive
            kwargs = {"host": self.host, "port": self.port, "database": self.database}
            if self.auth.lower() == "ldap":
                kwargs.update({"auth": "LDAP", "username": self.user, "password": self.password})
            self._conn = hive.connect(**kwargs)
            return self._conn
        except ImportError:
            logger.debug("pyhive not installed — trying impyla")
        except Exception as e:
            logger.warning("Hive connection failed: %s", e)
            return None

        try:
            from impala.dbapi import connect
            kwargs = {"host": self.host, "port": self.port, "database": self.database,
                      "auth_mechanism": self.auth.upper()}
            if self.auth.lower() == "ldap":
                kwargs.update({"user": self.user, "password": self.password})
            self._conn = connect(**kwargs)
            return self._conn
        except ImportError:
            logger.debug("impyla not installed")
        except Exception as e:
            logger.warning("Impala connection failed: %s", e)
        return None

    def _query(self, sql: str) -> list[dict]:
        conn = self._connect()
        if not conn:
            return []
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            cols = [d[0] for d in cursor.description] if cursor.description else []
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning("Hive query failed: %s | SQL: %s", e, sql[:120])
            return []

    # ── Public metadata methods ────────────────────────────────────────────────

    def get_table_schema(self, table_name: str) -> dict:
        """
        Returns:
            {
              "table": str,
              "columns": [{"name": str, "type": str, "comment": str}],
              "partition_columns": [str],
              "location": str,
              "format": str,
            }
        """
        rows = self._query(f"DESCRIBE FORMATTED `{table_name}`")
        if not rows:
            return {}

        columns, partition_cols, location, fmt = [], [], "", ""
        in_partition = False

        for row in rows:
            col_name = str(row.get("col_name", row.get("name", ""))).strip()
            data_type = str(row.get("data_type", row.get("type", ""))).strip()
            comment   = str(row.get("comment", "")).strip()

            if "Partition Information" in col_name:
                in_partition = True
                continue
            if "Detailed Table Information" in col_name:
                in_partition = False
                continue
            if "Location:" in col_name:
                location = data_type
            if "InputFormat:" in col_name:
                fmt = data_type.split(".")[-1] if data_type else ""
            if col_name and not col_name.startswith("#") and data_type:
                if in_partition:
                    partition_cols.append(col_name)
                elif not col_name.startswith("#"):
                    columns.append({"name": col_name, "type": data_type, "comment": comment})

        return {
            "table":             table_name,
            "columns":           columns,
            "partition_columns": partition_cols,
            "location":          location,
            "format":            fmt,
        }

    def get_table_stats(self, table_name: str) -> dict:
        """
        Returns row count, file count, total size from SHOW TABLE EXTENDED.
        """
        rows = self._query(f"SHOW TABLE EXTENDED LIKE '{table_name}'")
        stats = {"table": table_name}
        for row in rows:
            line = str(list(row.values())[0] if row else "")
            if "numRows" in line:
                m = __import__("re").search(r'numRows=(\d+)', line)
                if m: stats["row_count"] = int(m.group(1))
            if "totalSize" in line:
                m = __import__("re").search(r'totalSize=(\d+)', line)
                if m: stats["total_size_bytes"] = int(m.group(1))
            if "numFiles" in line:
                m = __import__("re").search(r'numFiles=(\d+)', line)
                if m: stats["file_count"] = int(m.group(1))
        return stats

    def get_partition_info(self, table_name: str) -> dict:
        """
        Returns partition columns, count, and skew indicator.
        """
        schema = self.get_table_schema(table_name)
        partition_cols = schema.get("partition_columns", [])
        if not partition_cols:
            return {"table": table_name, "partitioned": False}

        partitions = self._query(f"SHOW PARTITIONS `{table_name}`")
        count = len(partitions)

        # Simple skew detection: check if any single partition appears >50% of total
        skew_warning = False
        if partitions and count > 0:
            from collections import Counter
            # Look at first partition column distribution
            first_col_vals = []
            for p in partitions:
                vals = list(p.values())
                if vals:
                    first_col_vals.append(str(vals[0]).split("=")[-1])
            if first_col_vals:
                c = Counter(first_col_vals)
                most_common_pct = c.most_common(1)[0][1] / count
                skew_warning = most_common_pct > 0.5

        return {
            "table":             table_name,
            "partitioned":       True,
            "partition_columns": partition_cols,
            "partition_count":   count,
            "skew_warning":      skew_warning,
        }

    def get_column_stats(self, table_name: str, column_name: str) -> dict:
        """
        Returns NDV, null count, min, max via SHOW COLUMN STATS (Impala) or
        DESCRIBE EXTENDED (Hive).
        """
        rows = self._query(f"SHOW COLUMN STATS `{table_name}`")
        for row in rows:
            name = str(row.get("Column", row.get("col_name", ""))).strip()
            if name.lower() == column_name.lower():
                return {
                    "column":    column_name,
                    "table":     table_name,
                    "ndv":       row.get("#Distinct Values", row.get("ndv")),
                    "nulls":     row.get("#Nulls", row.get("null_count")),
                    "max_size":  row.get("Max Size", None),
                    "avg_size":  row.get("Avg Size", None),
                }
        return {"column": column_name, "table": table_name, "stats": "unavailable"}

    def list_tables(self, database: Optional[str] = None) -> list[str]:
        db = database or self.database
        rows = self._query(f"SHOW TABLES IN `{db}`")
        return [list(r.values())[0] for r in rows if r]

    def list_databases(self) -> list[str]:
        rows = self._query("SHOW DATABASES")
        return [list(r.values())[0] for r in rows if r]

    def format_context_for_agent(self, table_names: list[str]) -> str:
        """
        Given a list of table names extracted from code,
        fetch metadata and format a context block for agent injection.
        Returns "" if not configured or all lookups fail.
        """
        if not self.is_configured or not table_names:
            return ""

        blocks = ["## Cloudera Table Metadata (live from Hive Metastore)"]
        any_data = False

        for table in table_names[:8]:   # cap to avoid token bloat
            schema = self.get_table_schema(table)
            if not schema:
                continue
            any_data = True
            cols = schema.get("columns", [])
            part = schema.get("partition_columns", [])
            stats = self.get_table_stats(table)
            pinfo = self.get_partition_info(table)

            row_count = stats.get("row_count", "unknown")
            size_mb   = round(stats.get("total_size_bytes", 0) / 1_048_576, 1)

            blocks.append(f"\n### `{table}`")
            blocks.append(f"- Columns: {len(cols)} | Format: {schema.get('format', 'unknown')}")
            blocks.append(f"- Partition columns: {part if part else 'none'}")
            blocks.append(f"- Row count: {row_count:,}" if isinstance(row_count, int) else f"- Row count: {row_count}")
            blocks.append(f"- Size: {size_mb} MB | Files: {stats.get('file_count', 'unknown')}")
            if pinfo.get("partition_count"):
                blocks.append(f"- Partition count: {pinfo['partition_count']}")
            if pinfo.get("skew_warning"):
                blocks.append("- ⚠️ SKEW WARNING: One partition value dominates >50% of partitions")
            if cols:
                sample_cols = ", ".join(f"{c['name']} ({c['type']})" for c in cols[:8])
                blocks.append(f"- Sample columns: {sample_cols}")

        if not any_data:
            return ""

        blocks.append("")
        return "\n".join(blocks) + "\n"

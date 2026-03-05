"""
Mock test — validates full code pipeline without a real API call.
Simulates an LLM review response and runs it through reviewer → scorer → CLI output.
"""
import sys
import types
from unittest.mock import MagicMock, patch

# ── Fake config so we don't need a real API key ────────────────────────────────
fake_config = MagicMock()
fake_config.api_key = "mock-key"
fake_config.api_base = "https://mock.openai.com/v1"
fake_config.model_name = "gpt-5.2"
fake_config.pass_threshold = 95

# ── Simulated LLM review response ─────────────────────────────────────────────
MOCK_REVIEW = """
## 1. Summary of Findings

SCORE: 42/100

This query has several critical issues that will cause significant resource waste and poor
performance on Cloudera Impala. It uses implicit JOIN syntax, selects all columns (SELECT *),
applies a non-SARGable date filter, and lacks partition pruning. It will trigger a full table
scan on both tables.

---

## 2. Optimized Query

```sql
SELECT
  o.order_id,
  o.order_date,
  o.amount,
  c.id       AS customer_id,
  c.name     AS customer_name
FROM orders o
JOIN customers c ON o.cust_id = c.id
WHERE o.order_date >= '2024-01-01'
  AND o.order_date <  '2025-01-01'
```

---

## 3. Detailed Findings

**1. Implicit JOIN syntax (comma in FROM clause)**
- **Issue:** `FROM orders o, customers c` is an implicit CROSS JOIN filtered by a WHERE predicate.
- **Why it matters:** Impala's optimizer cannot reliably detect join type from implicit syntax. Risk of cartesian product if the WHERE predicate is accidentally removed. Also unreadable.
- **Fix:** Use explicit `JOIN ... ON` syntax.
- **Corrected:** `FROM orders o JOIN customers c ON o.cust_id = c.id`

**2. SELECT * — projecting all columns**
- **Issue:** Selecting all columns pulls every byte from both tables across the network.
- **Why it matters:** Parquet/ORC files are columnar — unused columns have zero read cost only if not selected. SELECT * eliminates that benefit entirely.
- **Fix:** Name only the columns your downstream consumer actually needs.
- **Corrected:** `SELECT o.order_id, o.order_date, o.amount, c.id, c.name`

**3. Non-SARGable date filter: year(o.order_date) = 2024**
- **Issue:** Wrapping a column in a function (`year()`) prevents Impala from using partition pruning or min/max statistics on `order_date`.
- **Why it matters:** Causes a full scan of all partitions regardless of the partition scheme. This is the most expensive anti-pattern for date-partitioned tables.
- **Fix:** Use a range predicate directly on the column.
- **Corrected:** `WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'`

**4. No partition filter specified**
- **Issue:** If `orders` is partitioned (e.g., by year or month), no partition filter is applied.
- **Why it matters:** Without partition pruning, Impala scans every partition — even ones outside 2024.
- **Fix:** Add an explicit partition column filter matching your table's partition scheme.

---

## 4. Advisory Notes

- Run `COMPUTE STATS orders` and `COMPUTE STATS customers` if statistics are stale — the optimizer needs accurate row counts to choose the correct join strategy.
- If `customers` is a small lookup table (< 10M rows), add `/*+ BROADCAST(c) */` after the JOIN keyword to eliminate the shuffle entirely.
- Review column types on `o.cust_id` and `c.id` — they must match exactly to avoid implicit casts that block predicate pushdown.

---

## 5. Expected Performance Impact

| Metric | Before | After (estimated) |
|---|---|---|
| Partitions scanned | ALL | 2024 only |
| Columns read | All | 5 |
| Join type | Implicit / risky | Explicit INNER JOIN |
| Partition pruning | ❌ Disabled | ✅ Enabled |
| Estimated speedup | baseline | 5x–20x depending on data volume |

---

## 6. Certification Decision

❌ **NOT CERTIFIED** — Score: 42/100 (threshold: 95/100)

This query must not run on the Cloudera platform in its current form.
Fix all 4 issues identified above, re-submit, and re-score before execution.
"""

# ── Wire the mock into the OpenAI client ───────────────────────────────────────
mock_choice = MagicMock()
mock_choice.message.content = MOCK_REVIEW

mock_response = MagicMock()
mock_response.choices = [mock_choice]
mock_response.usage.prompt_tokens = 812
mock_response.usage.completion_tokens = 634
mock_response.usage.total_tokens = 1446

mock_client = MagicMock()
mock_client.chat.completions.create.return_value = mock_response

# ── Run the test ───────────────────────────────────────────────────────────────
print("=" * 64)
print("QA SPARK CODEAGENT — MOCK TEST")
print("(No real API call — simulating LLM response)")
print("=" * 64)

with patch("agent.reviewer.OpenAI", return_value=mock_client), \
     patch("agent.reviewer.Config", return_value=fake_config):

    from agent.reviewer import review_code
    from agent.scorer import extract_score, get_certification

    code = open("test_sample.sql").read()
    result = review_code(code, "impala")
    score = extract_score(result["review"])
    cert = get_certification(score)

    print(result["review"])
    print("=" * 64)
    print(f"PIPELINE RESULT: {cert['badge']}  ({cert['score']}/100)")
    print(f"Exit code would be: {'0 (PASS)' if cert['certified'] else '1 (FAIL — MR blocked)'}")
    print("=" * 64)
    print(f"\nToken usage (simulated): {result['usage']['total_tokens']:,} tokens")
    print("\n✅ Mock test passed — all modules wired correctly.")

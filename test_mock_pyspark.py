"""
PySpark mock test — simulates a real-world bad PySpark job being reviewed.
No API call. Demonstrates what a developer would see when they submit code.
"""
import sys
from unittest.mock import MagicMock, patch

fake_config = MagicMock()
fake_config.api_key = "mock-key"
fake_config.api_base = "https://mock.openai.com/v1"
fake_config.model_name = "gpt-5.2"
fake_config.pass_threshold = 95
fake_config.spark_version = "3.4"

# ── Intentionally bad PySpark code submitted for review ───────────────────────
BAD_PYSPARK_CODE = """
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SalesReport").getOrCreate()

# Load full tables
orders = spark.read.csv("/data/orders/")
customers = spark.read.csv("/data/customers/")

# Join and filter
result = orders.join(customers, orders.cust_id == customers.id)
result = result.filter("year(order_date) = 2024")
result = result.select("*")

# UDF for formatting
from pyspark.sql.functions import udf
@udf
def format_name(name):
    return name.upper() + " - VIP"

result = result.withColumn("formatted", format_name(result.name))

# Collect everything to driver for processing
all_rows = result.collect()
for row in all_rows:
    print(row)
"""

# ── Simulated LLM review ──────────────────────────────────────────────────────
MOCK_REVIEW = """
## 1. Summary of Findings

SCORE: 28/100

This PySpark job has 5 critical issues that will cause severe performance degradation and
likely crash the Spark driver on any non-trivial dataset. The most dangerous issue is
.collect() pulling all data to the driver. Combined with a Python UDF in the hot path,
no schema definition, a non-SARGable date filter, and SELECT *, this code will exhaust
cluster resources and produce extremely slow runtimes.

---

## 2. Optimized Code

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, upper, concat, lit, broadcast

spark = SparkSession.builder.appName("SalesReport").getOrCreate()

try:
    # Explicit schemas — eliminates extra scan pass on read
    orders_schema = "order_id STRING, cust_id STRING, order_date DATE, amount DOUBLE"
    customers_schema = "id STRING, name STRING, tier STRING"

    orders = (
        spark.read
        .schema(orders_schema)
        .csv("/data/orders/")
        .filter((col("order_date") >= "2024-01-01") & (col("order_date") < "2025-01-01"))
        .select("order_id", "cust_id", "order_date", "amount")
    )

    customers = (
        spark.read
        .schema(customers_schema)
        .csv("/data/customers/")
        .select("id", "name")
    )

    # Broadcast customers (small lookup table) — eliminates shuffle on orders side
    result = (
        orders.join(broadcast(customers), orders.cust_id == customers.id)
        .withColumn("formatted", concat(upper(col("name")), lit(" - VIP")))
    )

    # Write output — never collect large data to driver
    result.write.mode("overwrite").parquet("/data/output/sales_report_2024/")

finally:
    spark.stop()
```

---

## 3. Detailed Findings

**1. .collect() on unbounded DataFrame — CRITICAL**
- **Issue:** `result.collect()` pulls every row from every executor to the driver JVM.
- **Why it matters:** Driver memory is typically 4-8GB. Even a modest 10M row result set at 500 bytes/row = 5GB. This will OOM the driver and kill the entire application.
- **Fix:** Write results to storage with `.write.parquet()`. Use `.show(20)` only for development inspection.
- **Corrected:** `result.write.mode("overwrite").parquet("/data/output/sales_report_2024/")`

**2. Python UDF on hot path — HIGH**
- **Issue:** `@udf` decorator creates a Python UDF that processes rows one at a time via Py4J serialization.
- **Why it matters:** Each row is serialized from JVM → Python → JVM. On 10M rows this can be 10-50x slower than the equivalent built-in function. Also prevents Catalyst optimization of the entire query stage.
- **Fix:** Use `pyspark.sql.functions.upper()` and `concat()` — both are native JVM functions.
- **Corrected:** `.withColumn("formatted", concat(upper(col("name")), lit(" - VIP")))`

**3. Non-SARGable date filter disables partition pruning — HIGH**
- **Issue:** `filter("year(order_date) = 2024")` wraps the partition column in a function.
- **Why it matters:** Spark cannot push this predicate down to the file system. All partitions are scanned — even 2020, 2021, 2022, 2023. Spark 3.4 AQE cannot fix this at runtime.
- **Fix:** Use a range predicate directly on the column.
- **Corrected:** `.filter((col("order_date") >= "2024-01-01") & (col("order_date") < "2025-01-01"))`

**4. No schema on CSV read — triggers full extra scan — MEDIUM**
- **Issue:** `spark.read.csv()` without a schema infers types by reading the entire file first.
- **Why it matters:** Schema inference doubles the I/O cost of every read. In production with large CSVs this is significant. It also risks incorrect type inference (e.g., IDs read as integers).
- **Fix:** Always define an explicit schema string or StructType.
- **Corrected:** `spark.read.schema("order_id STRING, ...").csv("/data/orders/")`

**5. SELECT * — defeats columnar format optimization — HIGH**
- **Issue:** `.select("*")` reads all columns from CSV/Parquet/ORC.
- **Why it matters:** Columnar formats only skip columns you don't select. SELECT * forces all columns to be deserialized, increasing memory and I/O cost proportionally.
- **Fix:** Name only the columns needed for the output.
- **Corrected:** `.select("order_id", "cust_id", "order_date", "amount")`

---

## 4. Advisory Notes

- **SparkSession**: Wrap execution in `try/finally` with `spark.stop()` to release YARN resources when done.
- **Broadcast join**: `customers` is a lookup table — wrap with `broadcast()` to eliminate shuffle on the orders side. Threshold: `spark.sql.autoBroadcastJoinThreshold = 10MB` (Spark 3.4 default).
- **AQE**: Spark 3.4 AQE is enabled by default (`spark.sql.adaptive.enabled=true`). It will handle shuffle partition coalescing automatically — do not set `spark.sql.shuffle.partitions` to a low static value.
- **Output format**: Switch from print() to Parquet output. CSV output for large results has no compression or column pruning benefits.

---

## 5. Expected Performance Impact

| Issue Fixed | Estimated Impact |
|---|---|
| Remove .collect() | Prevents driver OOM crash — job actually completes |
| Replace Python UDF | 10x–50x faster on that transformation stage |
| Fix date filter | Scans only 2024 partitions — 75%+ I/O reduction on 4-year dataset |
| Add explicit schema | Eliminates double-read on CSV — 50% faster file load |
| Remove SELECT * | Reduces shuffle and memory by # of unused columns |
| Broadcast join | Eliminates full shuffle on orders side for join |

---

## 6. Certification Decision

❌ **NOT CERTIFIED** — Score: 28/100 (threshold: 95/100)

This job will crash in production due to .collect() and will be extremely slow even if
it doesn't crash. Fix all 5 issues above, re-submit, and re-score before scheduling
on the Cloudera CDP cluster.
"""

# ── Run mock ──────────────────────────────────────────────────────────────────
mock_choice = MagicMock()
mock_choice.message.content = MOCK_REVIEW
mock_response = MagicMock()
mock_response.choices = [mock_choice]
mock_response.usage.prompt_tokens = 1024
mock_response.usage.completion_tokens = 891
mock_response.usage.total_tokens = 1915
mock_client = MagicMock()
mock_client.chat.completions.create.return_value = mock_response

print("=" * 64)
print("QA SPARK CODEAGENT — PYSPARK MOCK")
print("Spark 3.4  |  gpt-5.2  |  No real API call")
print("=" * 64)
print("\n--- CODE SUBMITTED FOR REVIEW ---")
print(BAD_PYSPARK_CODE)
print("--- END CODE ---\n")

with patch("agent.reviewer.OpenAI", return_value=mock_client), \
     patch("agent.reviewer.Config", return_value=fake_config):

    from agent.reviewer import review_code
    from agent.scorer import extract_score, get_certification

    result = review_code(BAD_PYSPARK_CODE, "pyspark")
    score = extract_score(result["review"])
    cert = get_certification(score)

    print(result["review"])
    print("=" * 64)
    print(f"PIPELINE RESULT: {cert['badge']}  ({cert['score']}/100)")
    print(f"GitLab MR status: {'✅ CLEAR TO MERGE' if cert['certified'] else '🚫 BLOCKED — fix required before merge'}")
    print(f"Token usage (simulated): {result['usage']['total_tokens']:,} tokens")
    print("=" * 64)

#!/usr/bin/env python3
"""
QA Spark CodeAgent — CLI entry point

Used by:
  - CAI Jobs (triggered manually or via API)
  - GitLab CI pipelines (exits with code 1 if score < threshold)

Usage:
  python cli.py --file path/to/code.sql --language impala
  python cli.py --file path/to/query.sql --language sparksql
  python cli.py --file path/to/job.py --language pyspark --output json
  python cli.py --file path/to/job.scala --language scala
  python cli.py --file path/to/script.py --language python --gitlab-comment

  # Multi-agent mode (default):
  python cli.py --file job.py --language pyspark --agent multi

  # Single-agent mode (legacy):
  python cli.py --file job.py --language pyspark --agent single

Override Spark version (default 3.4):
  SPARK_VERSION=3.5 python cli.py --file job.py --language pyspark
"""
import argparse
import json
import sys
from pathlib import Path

from agent.reviewer import review_code, SUPPORTED_LANGUAGES
from agent.scorer import extract_score, get_certification
from config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="QA Spark CodeAgent — AI-powered code review for Cloudera CDP"
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to the code file to review",
    )
    parser.add_argument(
        "--language", "-l",
        required=True,
        choices=SUPPORTED_LANGUAGES,
        help=f"Language/dialect: {SUPPORTED_LANGUAGES}",
    )
    parser.add_argument(
        "--output", "-o",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--gitlab-comment",
        action="store_true",
        help="Format output as a GitLab MR comment (markdown)",
    )
    parser.add_argument(
        "--agent",
        choices=["single", "multi"],
        default=None,
        help="Review mode: 'multi' (3 specialist agents, default) or 'single' (legacy)",
    )
    return parser.parse_args()


def format_gitlab_comment_single(filename: str, language: str, review: str, cert: dict, model: str) -> str:
    score = cert["score"]
    badge = cert["badge"]
    threshold = cert["threshold"]
    return (
        f"## {badge} — QA Spark CodeAgent\n\n"
        f"**File:** `{filename}` &nbsp;|&nbsp; "
        f"**Language:** {language.upper()} &nbsp;|&nbsp; "
        f"**Model:** {model} &nbsp;|&nbsp; "
        f"**Score:** {score}/100 &nbsp;|&nbsp; "
        f"**Required:** {threshold}/100\n\n"
        f"---\n\n"
        f"{review}\n\n"
        f"---\n"
        f"*Reviewed by [QA Spark CodeAgent](https://github.com/robosecure/QA_Spark_CodeAgent)*"
    )


def format_gitlab_comment_multi(filename: str, language: str, result: dict, cfg: Config) -> str:
    score = result["composite_score"]
    certified = result["certified"]
    badge = "✅ CERTIFIED" if certified else "❌ NOT CERTIFIED"
    threshold = cfg.pass_threshold
    agents = result.get("agents", {})
    weights = result.get("weights", {})
    cost = result.get("cost", {}).get("totals", {})

    agent_rows = ""
    for agent_key, agent_data in agents.items():
        label = agent_key.replace("_agent", "").title()
        w = weights.get(agent_key.replace("_agent", ""), 0)
        agent_rows += f"| {label} | {agent_data['score']}/100 | {int(w*100)}% |\n"

    findings_md = ""
    for f in result.get("key_findings", [])[:6]:
        findings_md += f"- {f}\n"

    cost_usd = cost.get("cost_usd", 0)
    total_tokens = cost.get("total_tokens", 0)

    return (
        f"## {badge} — QA Spark CodeAgent (Multi-Agent)\n\n"
        f"**File:** `{filename}` &nbsp;|&nbsp; "
        f"**Language:** {language.upper()} &nbsp;|&nbsp; "
        f"**Model:** {cfg.model_name}\n\n"
        f"### Composite Score: {score}/100 (Required: {threshold}/100)\n\n"
        f"| Agent | Score | Weight |\n"
        f"|-------|-------|--------|\n"
        f"{agent_rows}"
        f"\n### Key Findings\n{findings_md}\n"
        f"**Review cost:** ${cost_usd:.4f} USD | {total_tokens:,} tokens\n\n"
        f"---\n"
        f"*Reviewed by [QA Spark CodeAgent](https://github.com/robosecure/QA_Spark_CodeAgent)*"
    )


def run_multi_agent(code: str, language: str, args, cfg: Config) -> int:
    """Run the multi-agent orchestrator. Returns exit code."""
    from agent.orchestrator import Orchestrator

    orchestrator = Orchestrator(cfg, file_name=args.file)
    result = orchestrator.review(code, language)

    score = result["composite_score"]
    certified = result["certified"]
    badge = "✅ CERTIFIED" if certified else "❌ NOT CERTIFIED"

    if args.output == "json":
        print(json.dumps(result, indent=2))

    elif args.gitlab_comment:
        print(format_gitlab_comment_multi(args.file, language, result, cfg))

    else:
        divider = "=" * 64
        print(f"\n{divider}")
        print("QA SPARK CODEREVIEW AGENT — MULTI-AGENT MODE")
        print(divider)
        print(f"\nComposite Score: {score}/100")

        agents = result.get("agents", {})
        weights = result.get("weights", {})
        for agent_key, agent_data in agents.items():
            label = agent_key.replace("_agent", "").title()
            w = weights.get(agent_key.replace("_agent", ""), 0)
            print(f"  {label}: {agent_data['score']}/100 (weight {int(w*100)}%)")

        if result.get("key_findings"):
            print("\nKey Findings:")
            for f in result["key_findings"][:6]:
                print(f"  • {f}")

        if result.get("corrected_code"):
            print("\nCorrected Code:")
            print(result["corrected_code"])

        cost = result.get("cost", {}).get("totals", {})
        print(f"\nCost: ${cost.get('cost_usd', 0):.4f} USD | {cost.get('total_tokens', 0):,} tokens")

        if result.get("cache_hit"):
            print("[Cache hit — no LLM calls made]")

        print(f"\n{divider}")
        print(f"RESULT: {badge}  ({score}/100 — threshold {cfg.pass_threshold}/100)")
        print(divider)

    return 0 if certified else 1


def run_single_agent(code: str, language: str, args, cfg: Config) -> int:
    """Run the legacy single-agent reviewer. Returns exit code."""
    print(
        f"[QA Spark CodeAgent] Single-agent mode. Reviewing {args.file} "
        f"({language.upper()}) with model {cfg.model_name} ...",
        file=sys.stderr,
    )

    result = review_code(code, language)
    score = extract_score(result["review"])
    cert = get_certification(score, cfg.pass_threshold)

    if args.output == "json":
        print(json.dumps({**result, "score": score, "certification": cert}, indent=2))

    elif args.gitlab_comment:
        print(format_gitlab_comment_single(
            args.file, language, result["review"], cert, result["model"]
        ))

    else:
        divider = "=" * 64
        print(f"\n{divider}")
        print("QA SPARK CODEREVIEW AGENT")
        print(divider)
        print(result["review"])
        print(f"\n{divider}")
        print(f"RESULT: {cert['badge']}  ({cert['score']}/100 — threshold {cert['threshold']}/100)")
        print(divider)

    return 0 if cert["certified"] else 1


def main():
    args = parse_args()

    try:
        code = Path(args.file).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    if not code.strip():
        print(f"ERROR: File is empty: {args.file}", file=sys.stderr)
        sys.exit(2)

    cfg = Config()
    language = args.language.lower()

    # Determine mode: CLI flag > env var > default (multi)
    mode = args.agent or cfg.agent_mode

    print(
        f"[QA Spark CodeAgent] {mode.upper()} mode | "
        f"{args.file} ({language.upper()}) | model: {cfg.model_name}",
        file=sys.stderr,
    )

    if mode == "multi":
        exit_code = run_multi_agent(code, language, args, cfg)
    else:
        exit_code = run_single_agent(code, language, args, cfg)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

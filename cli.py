#!/usr/bin/env python3
"""
QA Spark CodeAgent — CLI entry point

Used by:
  - CAI Jobs (triggered manually or via API)
  - GitLab CI pipelines (exits with code 1 if score < threshold)

Usage:
  python cli.py --file path/to/code.sql --language impala
  python cli.py --file path/to/job.py --language pyspark --output json
  python cli.py --file path/to/script.py --language python --gitlab-comment
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
    return parser.parse_args()


def format_gitlab_comment(filename: str, language: str, review: str, cert: dict, model: str) -> str:
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


def main():
    args = parse_args()

    # Read source file
    try:
        code = Path(args.file).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    if not code.strip():
        print(f"ERROR: File is empty: {args.file}", file=sys.stderr)
        sys.exit(2)

    cfg = Config()
    print(
        f"[QA Spark CodeAgent] Reviewing {args.file} "
        f"({args.language.upper()}) with model {cfg.model_name} ...",
        file=sys.stderr,
    )

    result = review_code(code, args.language)
    score = extract_score(result["review"])
    cert = get_certification(score, cfg.pass_threshold)

    # ── Output ────────────────────────────────────────────────────
    if args.output == "json":
        print(json.dumps({**result, "score": score, "certification": cert}, indent=2))

    elif args.gitlab_comment:
        print(format_gitlab_comment(
            args.file, args.language, result["review"], cert, result["model"]
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

    # ── Exit code for CI gates ────────────────────────────────────
    # Exit 1 = not certified → GitLab CI job fails → MR is blocked
    sys.exit(0 if cert["certified"] else 1)


if __name__ == "__main__":
    main()

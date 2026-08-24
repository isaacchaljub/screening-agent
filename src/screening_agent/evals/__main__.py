"""M8 acceptance:

    python -m screening_agent.evals run --model groq:openai/gpt-oss-120b

Groq, not Gemini, for a full sweep — 12 scenarios × 2 model calls/turn × several turns is roughly
a hundred-plus calls, which would eat a meaningful chunk of Gemini's free daily quota for one run
(§5); Groq is the other free-tier dev provider and exists specifically for this.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from screening_agent.evals.report import render_report
from screening_agent.evals.runner import SCENARIOS_DIR, load_scenarios, run_all

DEFAULT_REPORT_PATH = Path("_internal") / "eval_report.md"


def main() -> None:
    parser = argparse.ArgumentParser(prog="screening_agent.evals")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run every scenario against one or more models")
    run_parser.add_argument(
        "--model",
        action="append",
        required=True,
        dest="models",
        help="vendor:model-id (repeatable)",
    )
    run_parser.add_argument("--scenarios-dir", type=Path, default=SCENARIOS_DIR)
    run_parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)

    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios_dir)
    if not scenarios:
        parser.error(f"no scenario files found in {args.scenarios_dir}")

    print(
        f"running {len(scenarios)} scenarios against {len(args.models)} model(s)...",
        file=sys.stderr,
    )
    results = run_all(scenarios, models=args.models)

    for result in results:
        status = "PASS" if result.outcome_match else "FAIL"
        detail = (
            result.error or f"outcome={result.actual_outcome} (expected {result.expected_outcome})"
        )
        print(f"  [{status}] {result.scenario} @ {result.model} — {detail}", file=sys.stderr)

    report = render_report(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report written to {args.out}]", file=sys.stderr)

    if not all(r.outcome_match for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

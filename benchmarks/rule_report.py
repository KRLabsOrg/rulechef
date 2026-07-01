#!/usr/bin/env python3
"""TAB-convenience wrapper around ``rulechef.report`` (the installed
``rulechef-report`` command). Adds the --dataset tab loader used in the paper;
for your own data use ``rulechef-report --rules ... --data gold.jsonl``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rulechef.report import generate, load_jsonl  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Rule report (TAB convenience wrapper)")
    p.add_argument("--rules", required=True)
    p.add_argument("--data", default=None, help="gold JSONL; alternative to --dataset")
    p.add_argument("--dataset", default=None, choices=["tab"])
    p.add_argument("--train", type=int, default=1000)
    p.add_argument("--test", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="rule_report.html")
    args = p.parse_args()

    if args.data:
        rows = load_jsonl(args.data)
    elif args.dataset == "tab":
        from benchmark_extract import TAB_FORMAT, TAB_SEMANTIC, load_tab_ds

        _, rows = load_tab_ds(args.train, args.test, args.seed, TAB_FORMAT + TAB_SEMANTIC)
    else:
        raise SystemExit("Provide --data <gold.jsonl> or --dataset tab")
    generate(args.rules, rows, args.out)


if __name__ == "__main__":
    main()

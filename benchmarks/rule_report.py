#!/usr/bin/env python3
"""Generate a browsable HTML report of a learned ruleset against gold data.

Loads any ruleset (checkpoint / comparison result / dataset JSON) with
RuleChef.load_rules, runs each rule over labeled data, classifies every match as
a true or false positive against the gold entities, and writes a self-contained
HTML file. Per rule you get its pattern, precision, TP/FP counts, and collapsible
lists of every true positive and false positive with the matched span
highlighted in its surrounding context (nothing is truncated). Open the file in
any browser; a search box filters rules by name.

Dataset-agnostic. Provide gold data either as a JSONL file (one
{"text": ..., "entities": [{"text","start","end","type"}]} per line) or use the
built-in --dataset tab convenience loader.

Usage:
    python benchmarks/rule_report.py \
        --rules benchmarks/results/results_extract_tab.ckpt_rulechef.json \
        --dataset tab --test 600 --out rules_tab.html

    python benchmarks/rule_report.py --rules my_rules.json --data my_gold.jsonl
"""

import argparse
import html
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))

CONTEXT = 90  # chars of context shown on each side of a matched span


def load_gold(args):
    """Return (list of {text, entities}, list of entity types)."""
    if args.data:
        rows = []
        for line in Path(args.data).read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        types = sorted({e["type"] for r in rows for e in r.get("entities", [])})
        return rows, types
    if args.dataset == "tab":
        from benchmark_extract import TAB_FORMAT, TAB_SEMANTIC, load_tab_ds

        types = TAB_FORMAT + TAB_SEMANTIC
        _, test = load_tab_ds(args.train, args.test, args.seed, types)
        return test, types
    raise SystemExit("Provide --data <jsonl> or --dataset tab")


def is_tp(pred, gold_entities):
    """True positive: a gold entity of the same type whose span overlaps."""
    ps, pe = pred.get("start"), pred.get("end")
    for g in gold_entities:
        if g.get("type") != pred.get("type"):
            continue
        gs, ge = g.get("start"), g.get("end")
        if ps is not None and gs is not None:
            if max(ps, gs) < min(pe, ge):  # overlap
                return True
        elif pred.get("text") and pred["text"].strip() == (g.get("text") or "").strip():
            return True
    return False


def context_html(text, pred):
    """Matched span highlighted in its surrounding context (span never trimmed)."""
    s, e = pred.get("start"), pred.get("end")
    span = pred.get("text", "")
    if s is None or e is None:
        idx = text.find(span)
        if idx < 0:
            return html.escape(span)
        s, e = idx, idx + len(span)
    left = text[max(0, s - CONTEXT) : s]
    right = text[e : e + CONTEXT]
    pre = "…" if s - CONTEXT > 0 else ""
    post = "…" if e + CONTEXT < len(text) else ""
    return (
        pre
        + html.escape(left)
        + "<mark>"
        + html.escape(text[s:e])
        + "</mark>"
        + html.escape(right)
        + post
    )


def main():
    from rulechef import RuleChef
    from rulechef.core import Task, TaskType

    p = argparse.ArgumentParser(description="Browsable HTML report of rules vs gold data")
    p.add_argument("--rules", required=True, help="rules file (checkpoint / result / dataset JSON)")
    p.add_argument("--data", default=None, help="gold data as JSONL: {text, entities:[...]}")
    p.add_argument("--dataset", default=None, choices=["tab"], help="built-in loader")
    p.add_argument("--train", type=int, default=1000)
    p.add_argument("--test", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="rule_report.html")
    args = p.parse_args()

    gold, types = load_gold(args)

    task = Task(
        name="inspect",
        description="Extract entity spans.",
        input_schema={"text": "str"},
        output_schema={"entities": "List[{text,start,end,type}]"},
        type=TaskType.NER,
        text_field="text",
    )
    chef = RuleChef(
        task=task, client=MagicMock(), dataset_name="inspect", storage_path=tempfile.mkdtemp()
    )
    rules = chef.load_rules(args.rules)
    print(f"Loaded {len(rules)} rules from {args.rules}; scoring on {len(gold)} docs")

    # Per rule, collect every match classified as TP or FP.
    per_rule = {r.id: {"rule": r, "tp": [], "fp": []} for r in rules}
    for doc in gold:
        text = doc["text"]
        gold_entities = doc.get("entities", [])
        for r in rules:
            out = chef.learner._apply_rules([r], {"text": text}, TaskType.NER, "text")
            for pred in out.get("entities", []) or []:
                bucket = "tp" if is_tp(pred, gold_entities) else "fp"
                per_rule[r.id][bucket].append(context_html(text, pred))

    # Sort rules: worst precision first (most to inspect/fix).
    def prec(info):
        tp, fp = len(info["tp"]), len(info["fp"])
        return tp / (tp + fp) if (tp + fp) else 1.0

    cards = sorted(per_rule.values(), key=lambda i: (prec(i), -len(i["fp"])))

    parts = []
    for info in cards:
        r = info["rule"]
        tp, fp = info["tp"], info["fp"]
        n = len(tp) + len(fp)
        pr = f"{len(tp) / n:.2f}" if n else "--"
        dev = f"{r.validated_precision:.2f}" if r.validated_precision is not None else "--"
        out_type = (r.output_template or {}).get("type", "?")

        def items(matches):
            return "".join(f"<li>{m}</li>" for m in matches) or "<li><em>none</em></li>"

        parts.append(
            f"""
<div class="rule" data-name="{html.escape(r.name.lower())}">
  <h2>{html.escape(r.name)} <span class="type">{html.escape(out_type)}</span></h2>
  <div class="meta">precision {pr} &middot; dev {dev} &middot;
       <span class="tpc">{len(tp)} TP</span> / <span class="fpc">{len(fp)} FP</span></div>
  <details><summary>{html.escape(r.pattern)}</summary></details>
  <details open><summary>False positives ({len(fp)})</summary><ul class="fp">{items(fp)}</ul></details>
  <details><summary>True positives ({len(tp)})</summary><ul class="tp">{items(tp)}</ul></details>
</div>"""
        )

    css = """
body{font:14px/1.5 -apple-system,system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#222}
h1{font-size:20px} #q{width:100%;padding:.5rem;font-size:14px;margin-bottom:1rem;box-sizing:border-box}
.rule{border:1px solid #ddd;border-radius:8px;padding:.5rem 1rem;margin:1rem 0}
.rule h2{font-size:15px;margin:.3rem 0} .type{font-size:11px;color:#fff;background:#789;padding:1px 6px;border-radius:4px;vertical-align:middle}
.meta{color:#666;font-size:12px;margin-bottom:.4rem} .tpc{color:#197;font-weight:600} .fpc{color:#b33;font-weight:600}
summary{cursor:pointer;color:#456;font-family:ui-monospace,monospace;font-size:12px}
ul{margin:.3rem 0;padding-left:1.2rem} li{margin:.25rem 0;font-family:ui-monospace,monospace;font-size:12px;white-space:pre-wrap}
mark{background:#fe9;padding:0 1px} .fp mark{background:#fbb} .tp mark{background:#bf9}
"""
    js = """
const q=document.getElementById('q');
q.addEventListener('input',()=>{const v=q.value.toLowerCase();
document.querySelectorAll('.rule').forEach(r=>r.style.display=r.dataset.name.includes(v)?'':'none');});
"""
    title = f"Rule report — {len(rules)} rules, {len(gold)} docs"
    doc = f"""<!doctype html><meta charset=utf-8><title>{title}</title><style>{css}</style>
<h1>{title}</h1><input id=q placeholder="filter rules by name…">{''.join(parts)}<script>{js}</script>"""
    Path(args.out).write_text(doc)
    print(f"Wrote {args.out} ({len(doc) // 1024} KB) — open it in a browser")


if __name__ == "__main__":
    main()

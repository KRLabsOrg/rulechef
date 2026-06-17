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

    def stats(info):
        tp, fp = len(info["tp"]), len(info["fp"])
        n = tp + fp
        return tp, fp, n, (tp / n if n else None)

    # Sort by precision (accuracy), highest first; rules that never fired go last.
    def sort_key(info):
        tp, fp, n, p = stats(info)
        return (0 if n else 1, -(p if p is not None else 0), -n)

    cards = sorted(per_rule.values(), key=sort_key)

    def items(matches):
        return "".join(f"<li>{m}</li>" for m in matches) or '<li class="none">none</li>'

    def badge(p):
        if p is None:
            return "na"
        return "hi" if p >= 0.8 else "mid" if p >= 0.5 else "lo"

    parts = []
    for info in cards:
        r = info["rule"]
        tp, fp, n, p = stats(info)
        pr = f"{p:.0%}" if p is not None else "n/a"
        out_type = (r.output_template or {}).get("type", "?")
        parts.append(
            f"""
<div class="rule" data-name="{html.escape(r.name.lower())}">
  <div class="head">
    <span class="rname">{html.escape(r.name)}</span>
    <span class="type">{html.escape(str(out_type))}</span>
    <span class="prec {badge(p)}">{pr}</span>
    <span class="pills"><span class="tp">{tp} TP</span><span class="fp">{fp} FP</span></span>
  </div>
  <details class="pat"><summary>pattern</summary><code>{html.escape(r.pattern)}</code></details>
  <details open><summary>False positives <b>({fp})</b></summary><ul class="fp">{items(info["fp"])}</ul></details>
  <details><summary>True positives <b>({tp})</b></summary><ul class="tp">{items(info["tp"])}</ul></details>
</div>"""
        )

    css = """
:root{--bg:#f6f7f9;--card:#fff;--bd:#e3e6ea;--fg:#1f2328;--mut:#6b7280}
*{box-sizing:border-box}
body{font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  background:var(--bg);color:var(--fg);max-width:920px;margin:0 auto;padding:1.5rem 1rem 4rem}
h1{font-size:18px;font-weight:650;margin:.2rem 0 1rem}
.bar{position:sticky;top:0;background:var(--bg);padding:.6rem 0;z-index:5}
#q{width:100%;padding:.6rem .8rem;font-size:14px;border:1px solid var(--bd);border-radius:8px;background:var(--card)}
.rule{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:.7rem .9rem;margin:.7rem 0;
  box-shadow:0 1px 2px rgba(0,0,0,.03)}
.head{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}
.rname{font-weight:650;font-size:14px}
.type{font-size:10.5px;letter-spacing:.03em;color:#475569;background:#eef2f6;border:1px solid #e0e6ec;
  padding:1px 7px;border-radius:999px;text-transform:uppercase}
.prec{font-weight:650;font-size:12px;padding:1px 8px;border-radius:999px;color:#fff}
.prec.hi{background:#16a34a}.prec.mid{background:#d97706}.prec.lo{background:#dc2626}.prec.na{background:#9aa3ad}
.pills{margin-left:auto;display:flex;gap:.4rem;font-size:11.5px}
.pills span{padding:1px 7px;border-radius:999px;border:1px solid var(--bd);color:var(--mut)}
.pills .tp{color:#15803d;border-color:#bbf7d0;background:#f0fdf4}
.pills .fp{color:#b91c1c;border-color:#fecaca;background:#fef2f2}
details{margin-top:.5rem}
summary{cursor:pointer;color:#475569;font-size:12px;user-select:none}
summary:hover{color:#1f2328}
.pat code{display:block;margin-top:.4rem;padding:.5rem .6rem;background:#0f172a;color:#e2e8f0;border-radius:6px;
  font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word}
ul{margin:.4rem 0 .2rem;padding:0;list-style:none}
li{padding:.3rem .5rem;border-left:2px solid #eef0f3;margin:.2rem 0;font-size:12.5px;
  white-space:pre-wrap;overflow-wrap:anywhere;color:#374151}
li.none{color:var(--mut);font-style:italic;border-left-color:transparent}
.fp li{border-left-color:#fca5a5}.tp li{border-left-color:#86efac}
mark{padding:0 2px;border-radius:3px;font-weight:600}
.fp mark{background:#fecaca}.tp mark{background:#bbf7d0}
"""
    js = """
const q=document.getElementById('q');
q.addEventListener('input',()=>{const v=q.value.toLowerCase();
document.querySelectorAll('.rule').forEach(r=>r.style.display=r.dataset.name.includes(v)?'':'none');});
"""
    title = f"Rule report &mdash; {len(rules)} rules, {len(gold)} docs (sorted by precision)"
    doc = f"""<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Rule report</title><style>{css}</style>
<h1>{title}</h1>
<div class="bar"><input id=q placeholder="filter rules by name…"></div>
{''.join(parts)}<script>{js}</script>"""
    Path(args.out).write_text(doc)
    print(f"Wrote {args.out} ({len(doc) // 1024} KB) — open it in a browser")


if __name__ == "__main__":
    main()

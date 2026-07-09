#!/usr/bin/env python3
"""Generate a KR Labs-branded LLM-savings report from observed traffic.

Answers the deployment question: "how much of my LLM traffic could learned
rules take over, and at what accuracy cost?" Takes a ruleset and a log of
observed LLM calls, replays the rules over the traffic, and writes a
self-contained, print-ready HTML report:

  - headline: % of calls replaceable, fidelity to the LLM on those calls,
    estimated cost savings (--cost-per-call)
  - top rules ranked by calls taken over, each with agreement and examples

Traffic format (JSONL, one call per line):
    {"text": "...", "llm_label": "...", "gold_label": "..."(optional)}

Installed as the ``rulechef-savings`` command:
    rulechef-savings --rules rules.json --traffic traffic.jsonl \
        --cost-per-call 0.002 --out savings.html

Print to PDF: open in a browser and print, or
    chrome --headless --print-to-pdf=savings.pdf savings.html
"""

import argparse
import html
import json
import tempfile
from collections import defaultdict
from pathlib import Path

KR = "#FF3D5A"  # KR Labs Signal


def main():
    from rulechef import RuleChef
    from rulechef.core import Task, TaskType

    p = argparse.ArgumentParser(description="Branded LLM-savings report from observed traffic")
    p.add_argument("--rules", required=True)
    p.add_argument("--traffic", required=True, help="JSONL: {text, llm_label[, gold_label]}")
    p.add_argument("--cost-per-call", type=float, default=None, help="e.g. 0.002 (USD)")
    p.add_argument("--calls-per-month", type=int, default=None, help="volume to project savings on")
    p.add_argument("--title", default="LLM savings report")
    p.add_argument("--out", default="savings_report.html")
    args = p.parse_args()

    rows = [json.loads(line) for line in Path(args.traffic).read_text().splitlines() if line.strip()]
    task = Task(
        name="observed traffic",
        description="Classify observed inputs.",
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
        text_field="text",
    )
    chef = RuleChef(task=task, client=object(), dataset_name="savings", storage_path=tempfile.mkdtemp())
    chef.load_rules(args.rules)

    n = len(rows)
    answered = 0
    agree = 0
    gold_right = 0
    gold_seen = 0
    per_rule = defaultdict(lambda: {"taken": 0, "agree": 0, "examples": []})

    for r in rows:
        out = chef.extract({"text": r["text"]}, validate=False)
        label = (out or {}).get("label") or ""
        if not label:
            continue
        answered += 1
        rule_name = (out or {}).get("rule_name", "?")
        ok = str(label).strip().lower() == str(r.get("llm_label", "")).strip().lower()
        agree += ok
        if "gold_label" in r:
            gold_seen += 1
            gold_right += str(label).strip().lower() == str(r["gold_label"]).strip().lower()
        s = per_rule[rule_name]
        s["taken"] += 1
        s["agree"] += ok
        if len(s["examples"]) < 3:
            s["examples"].append((r["text"], label, ok))

    coverage = answered / n if n else 0.0
    fidelity = agree / answered if answered else 0.0
    gold_acc = (gold_right / gold_seen) if gold_seen else None

    monthly = args.calls_per_month or n
    savings = (
        f"${coverage * monthly * args.cost_per_call:,.0f}" if args.cost_per_call else None
    )

    top = sorted(per_rule.items(), key=lambda kv: -kv[1]["taken"])

    def pct(x):
        return f"{x:.0%}"

    rule_rows = []
    for name, s in top:
        fid = s["agree"] / s["taken"] if s["taken"] else 0
        ex = "".join(
            f'<div class="ex"><span class="mono">{html.escape(t)}</span>'
            f' <span class="arrow">&rarr;</span> <span class="mono lbl">{html.escape(str(lbl))}</span>'
            f'{"" if ok else " <span class=miss>&ne; LLM</span>"}</div>'
            for t, lbl, ok in s["examples"]
        )
        rule_rows.append(
            f"""<tr><td class="mono rn">{html.escape(name)}</td>
<td class="num">{s['taken']}</td><td class="num">{pct(s['taken'] / n)}</td>
<td class="num">{pct(fid)}</td></tr>
<tr class="exrow"><td colspan="4">{ex}</td></tr>"""
        )

    stat_cards = [
        ("calls observed", f"{n:,}"),
        ("replaceable by rules", pct(coverage)),
        ("fidelity to your LLM", pct(fidelity)),
    ]
    if gold_acc is not None:
        stat_cards.append(("accuracy vs. gold", pct(gold_acc)))
    if savings:
        stat_cards.append((f"est. savings / {monthly:,} calls", savings))
    cards = "".join(
        f'<div class="card"><div class="k">{v}</div><div class="l">{k}</div></div>'
        for k, v in stat_cards
    )

    doc = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(args.title)} — KR Labs</title>
<style>
:root{{--kr:{KR};--ink:#111;--mut:#666;--bd:#e6e6e6}}
*{{box-sizing:border-box}}
body{{margin:0;color:var(--ink);background:#fff;
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}}
.page{{max-width:860px;margin:0 auto;padding:2.2rem 1.4rem 0}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:2.6rem}}
.logo{{font-weight:800;font-size:22px;letter-spacing:.01em}}
.logo .br{{color:var(--kr);font-family:ui-monospace,monospace;font-weight:700}}
.tag{{font-family:ui-monospace,monospace;font-size:11px;letter-spacing:.22em;color:var(--mut)}}
.label{{font-family:ui-monospace,monospace;font-size:11.5px;letter-spacing:.2em;color:var(--kr);
  text-transform:uppercase;margin:2.4rem 0 .7rem}}
h1{{font-size:40px;line-height:1.12;font-weight:800;letter-spacing:-.015em;margin:.2rem 0 1rem}}
h1 .kr{{color:var(--kr)}}
.sub{{color:#333;max-width:640px;margin:0 0 1.8rem}}
.cards{{display:flex;flex-wrap:wrap;gap:.8rem;margin:1.2rem 0 .4rem}}
.card{{flex:1 1 140px;border:1px solid var(--bd);padding:.9rem 1rem}}
.card .k{{font-size:26px;font-weight:800;letter-spacing:-.01em}}
.card .l{{font-family:ui-monospace,monospace;font-size:10.5px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--mut);margin-top:.2rem}}
table{{width:100%;border-collapse:collapse;margin:.6rem 0 1.6rem}}
th{{font-family:ui-monospace,monospace;font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--mut);text-align:left;padding:.45rem .5rem;border-bottom:2px solid var(--ink)}}
td{{padding:.5rem;border-bottom:1px solid var(--bd);vertical-align:top}}
td.num,th.num{{text-align:right}}
.rn{{color:var(--kr);font-weight:600;font-size:13px}}
.exrow td{{border-bottom:1px solid var(--bd);padding:.15rem .5rem .6rem;background:#fafafa}}
.ex{{font-size:12px;color:#444;margin:.15rem 0}}
.ex .lbl{{color:var(--kr)}}
.ex .arrow{{color:#999}}
.miss{{color:#b45309;font-size:11px;font-family:ui-monospace,monospace}}
.note{{font-size:12.5px;color:var(--mut);max-width:640px}}
footer{{background:#111;color:#bbb;margin-top:2.6rem}}
footer .page{{padding:1.1rem 1.4rem;display:flex;justify-content:space-between;flex-wrap:wrap;gap:.5rem}}
footer .mono{{font-size:11px;letter-spacing:.08em}}
footer a{{color:#eee;text-decoration:none}}
@media print{{.page{{padding-top:1.2rem}} footer{{position:fixed;bottom:0;left:0;right:0}}
  body{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}
</style>
<div class="page">
<header>
  <div class="logo"><span class="br">[&hairsp;KR&hairsp;]</span> Labs</div>
  <div class="tag">krlabs.eu</div>
</header>

<div class="label">[rulechef] &middot; {html.escape(args.title)}</div>
<h1>{pct(coverage)} of your LLM calls<br>are replaceable by <span class="kr">{answered and len([1 for _, s in top if s['taken']])} rules</span>.</h1>
<p class="sub">We replayed {n:,} observed LLM calls against the rules RuleChef learned from your traffic.
Rules answered {pct(coverage)} of them, agreeing with your LLM on {pct(fidelity)} of the calls they took over.
Every rule is inspectable: what it matches, what it answered, and where it disagrees.</p>

<div class="cards">{cards}</div>

<div class="label">Top rules by calls taken over</div>
<table>
<tr><th>rule</th><th class="num">calls</th><th class="num">share</th><th class="num">agreement</th></tr>
{''.join(rule_rows)}
</table>

<p class="note">Fidelity is agreement with the observed LLM outputs on calls the rules answered; the rules
inherit the LLM's own error rate on those calls. Rules abstain on everything else, which continues to
flow to the LLM. Generated by <span class="mono">rulechef</span> &middot; github.com/KRLabsOrg/rulechef.</p>
</div>
<footer><div class="page">
  <span class="mono">krlabs.eu &middot; contact@krlabs.eu</span>
  <span class="mono"><a href="https://github.com/KRLabsOrg">github.com/KRLabsOrg</a> &middot; KR Labs GmbH, Vienna</span>
</div></footer></html>"""
    Path(args.out).write_text(doc)
    print(
        f"Wrote {args.out}: {n:,} calls, {pct(coverage)} replaceable at {pct(fidelity)} fidelity"
        + (f", est. savings {savings}/{monthly:,} calls" if savings else "")
    )


if __name__ == "__main__":
    main()

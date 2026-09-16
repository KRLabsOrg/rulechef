#!/usr/bin/env python3
"""Structured extraction: RuleChef rules vs GLiNER vs LLM — the "shine" benchmark.

The thesis: for FORMAT-DEFINED entities (dates, money, percentages, codes,
emails, IPs...) interpretable rules match or beat zero-shot GLiNER and
approach the LLM at zero inference cost, while LogReg/SetFit can't extract
spans at all. The SAME datasets contain SEMANTIC entities (person, org,
location, demographics) where rules can't help — giving the "works vs not"
boundary as a per-type result on REAL text.

Datasets (--dataset), all real text:
  ontonotes  — OntoNotes 5 newswire/web NER (tner/ontonotes5)
  tab        — Text Anonymization Benchmark, real ECHR court rulings (PII)
  ai4privacy — synthetic PII (large; scale-test only, flagged as synthetic)

Reports per-type precision/recall/F1 for each method, aggregated into
FORMAT vs SEMANTIC groups, plus the share of gold spans that are format-typed.

Usage:
    OPENAI_API_KEY=... python benchmarks/benchmark_extract.py --dataset ontonotes \
        --base-url https://inference.baseten.co/v1 --model moonshotai/Kimi-K2.6 \
        --train 1500 --test 800 --gliner urchade/gliner_medium-v2.1
"""

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# ── Per-dataset config: format types, semantic types, GLiNER/LLM labels ──

ONTONOTES_FORMAT = ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"]
ONTONOTES_SEMANTIC = [
    "PERSON",
    "ORG",
    "GPE",
    "NORP",
    "FAC",
    "LOC",
    "EVENT",
    "LAW",
    "WORK_OF_ART",
    "PRODUCT",
    "LANGUAGE",
]
ONTONOTES_NL = {
    "DATE": "date",
    "TIME": "time",
    "PERCENT": "percentage",
    "MONEY": "money amount",
    "QUANTITY": "quantity",
    "CARDINAL": "number",
    "ORDINAL": "ordinal number",
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "country, city or state",
    "NORP": "nationality, religious or political group",
    "FAC": "facility",
    "LOC": "location",
    "EVENT": "event",
    "LAW": "law",
    "WORK_OF_ART": "work of art",
    "PRODUCT": "product",
    "LANGUAGE": "language",
}

TAB_FORMAT = ["CODE", "DATETIME", "QUANTITY"]
TAB_SEMANTIC = ["PERSON", "ORG", "LOC", "DEM", "MISC"]
TAB_NL = {
    "CODE": "code or reference number",
    "DATETIME": "date or time",
    "QUANTITY": "quantity",
    "PERSON": "person",
    "ORG": "organization",
    "LOC": "location",
    "DEM": "demographic attribute",
    "MISC": "identifier",
}

AI4P_FORMAT = [
    "EMAIL",
    "URL",
    "IPV4",
    "IPV6",
    "CREDITCARDNUMBER",
    "PHONENUMBER",
    "BITCOINADDRESS",
    "DATE",
    "TIME",
    "MAC",
]
AI4P_SEMANTIC = ["FIRSTNAME", "LASTNAME", "CITY", "JOBTITLE", "COMPANYNAME", "STATE"]
AI4P_NL = {
    "EMAIL": "email",
    "URL": "url",
    "IPV4": "ip address",
    "IPV6": "ipv6 address",
    "CREDITCARDNUMBER": "credit card number",
    "PHONENUMBER": "phone number",
    "BITCOINADDRESS": "bitcoin address",
    "DATE": "date",
    "TIME": "time",
    "MAC": "mac address",
    "FIRSTNAME": "first name",
    "LASTNAME": "last name",
    "CITY": "city",
    "JOBTITLE": "job title",
    "COMPANYNAME": "company name",
    "STATE": "state",
}


# ── Loaders → (train_records, test_records) of {text, entities[{text,start,end,type}]} ──


def load_ontonotes_ds(n_train, n_test, seed, types):
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from benchmark_ontonotes import load_ontonotes

    train, test, _types, _tags = load_ontonotes(filter_types=set(types))
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(test)
    return train[:n_train], test[:n_test]


def _chunk_doc(text, ents, max_len=600):
    """Split a long doc into sentence-ish chunks; localize entity offsets."""
    bounds = [m.end() for m in re.finditer(r"[.\n]\s+", text)] + [len(text)]
    chunks, start = [], 0
    for b in bounds:
        if b - start >= max_len or b == len(text):
            chunks.append((start, b))
            start = b
    out = []
    for cs, ce in chunks:
        local = [
            {"text": e["text"], "start": e["start"] - cs, "end": e["end"] - cs, "type": e["type"]}
            for e in ents
            if cs <= e["start"] < ce
        ]
        if local:
            out.append({"text": text[cs:ce], "entities": local})
    return out


def load_tab_ds(n_train, n_test, seed, types):
    from datasets import load_dataset

    typeset = set(types)
    ds = load_dataset("mattmdjaga/text-anonymization-benchmark-train")["train"]
    records = []
    for r in ds:
        ann = r["annotations"]
        if isinstance(ann, str):
            ann = json.loads(ann)
        # pick the first annotator with mentions
        mentions = None
        for v in (ann or {}).values():
            if isinstance(v, dict) and v.get("entity_mentions"):
                mentions = v["entity_mentions"]
                break
        if not mentions:
            continue
        ents, seen = [], set()
        for m in mentions:
            t = m.get("entity_type")
            if t not in typeset:
                continue
            span = (m["start_offset"], m["end_offset"], t)
            if span in seen:
                continue
            seen.add(span)
            ents.append(
                {
                    "text": m["span_text"],
                    "start": m["start_offset"],
                    "end": m["end_offset"],
                    "type": t,
                }
            )
        records.extend(_chunk_doc(r["text"], ents))
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:n_train], records[n_train : n_train + n_test]


def load_ai4privacy_ds(n_train, n_test, seed, types):
    from datasets import load_dataset

    typeset = set(types)
    ds = load_dataset("ai4privacy/pii-masking-200k")["train"]
    records = []
    for r in ds:
        if r["language"] != "en":
            continue
        mask = r["privacy_mask"]
        if isinstance(mask, str):
            mask = json.loads(mask)
        ents = [
            {"text": m["value"], "start": m["start"], "end": m["end"], "type": m["label"]}
            for m in mask
            if m["label"] in typeset
        ]
        if ents:
            records.append({"text": r["source_text"], "entities": ents})
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:n_train], records[n_train : n_train + n_test]


DATASETS = {
    "ontonotes": (load_ontonotes_ds, ONTONOTES_FORMAT, ONTONOTES_SEMANTIC, ONTONOTES_NL, False),
    "tab": (load_tab_ds, TAB_FORMAT, TAB_SEMANTIC, TAB_NL, False),
    "ai4privacy": (load_ai4privacy_ds, AI4P_FORMAT, AI4P_SEMANTIC, AI4P_NL, True),
}


# ── Unified scorer (text+type match) ────────────────────────


def score(pred_docs, gold_docs):
    per = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for preds, golds in zip(pred_docs, gold_docs):
        gold_keys = defaultdict(list)
        for g in golds:
            gold_keys[(g["text"].strip(), g["type"])].append(g)
        matched = defaultdict(int)
        for p in preds:
            key = (str(p["text"]).strip(), p["type"])
            if matched[key] < len(gold_keys.get(key, [])):
                per[p["type"]]["tp"] += 1
                matched[key] += 1
            else:
                per[p["type"]]["fp"] += 1
        for key, pool in gold_keys.items():
            per[key[1]]["fn"] += max(0, len(pool) - matched.get(key, 0))
    return per


def prf(c):
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def group_f1(per, types):
    tp = sum(per[t]["tp"] for t in types)
    fp = sum(per[t]["fp"] for t in types)
    fn = sum(per[t]["fn"] for t in types)
    return prf({"tp": tp, "fp": fp, "fn": fn})


# ── Methods ─────────────────────────────────────────────────


def run_rulechef(train, test, types, client, model, args):
    from rulechef import RuleChef
    from rulechef.core import Task, TaskType

    task = Task(
        name=f"{args.dataset} extraction",
        description=(
            "Extract entity spans. Entity types: " + ", ".join(types) + ". "
            "Use precise patterns; only match unambiguous spans."
        ),
        input_schema={"text": "str"},
        output_schema={"entities": "List[{text,start,end,type}]"},
        type=TaskType.NER,
        text_field="text",
    )
    storage = tempfile.mkdtemp(prefix="rulechef_ext_")
    from rulechef.coordinator import AgenticCoordinator

    coordinator = AgenticCoordinator(
        client,
        model=model,
        prune_after_learn=True,
        enable_critic=True,
        critic_interval=2,
        audit_interval=3,
    )
    chef = RuleChef(
        task=task,
        client=client,
        dataset_name="ext",
        storage_path=storage,
        allowed_formats=["regex", "code"],
        model=model,
        max_rules=args.max_rules,
        max_samples=args.max_samples,
        synthesis_strategy="per_class",
        coordinator=coordinator,
        synthesis_output_tokens=6000,
        patch_output_tokens=4096,
    )
    for r in train:
        chef.add_example({"text": r["text"]}, {"entities": r["entities"]})
    t0 = time.time()
    result = chef.learn_rules(
        run_evaluation=True,
        max_refinement_iterations=args.max_iterations,
        holdout_fraction=args.holdout_fraction,
        split_seed=args.seed,
    )
    learn_s = time.time() - t0
    if result is None:
        return None
    rules, _ = result
    t0 = time.time()
    preds = [
        chef.learner._apply_rules(rules, {"text": r["text"]}, TaskType.NER, "text").get(
            "entities", []
        )
        or []
        for r in test
    ]
    infer_ms = (time.time() - t0) / max(1, len(test)) * 1000
    import shutil

    shutil.rmtree(storage, ignore_errors=True)
    return {
        "preds": preds,
        "num_rules": len(rules),
        "learn_s": round(learn_s, 1),
        "infer_ms": round(infer_ms, 3),
        "rules": [
            {
                "name": ru.name,
                "format": ru.format.value,
                "content": ru.content,
                "priority": ru.priority,
                "validated_precision": ru.validated_precision,
                "validated_support": ru.validated_support,
                "output_template": ru.output_template,
                "output_key": ru.output_key,
            }
            for ru in rules
        ],
    }


def _ckpt_fingerprint(args) -> dict:
    return {
        "dataset": args.dataset,
        "seed": args.seed,
        "train": args.train,
        "test": args.test,
        "model": args.model,
    }


def with_checkpoint(name, out_path, args, compute):
    """Run a method stage with a disk checkpoint.

    A late-stage crash no longer throws away earlier stages: each stage's
    result is persisted as <output>.ckpt_<name>.json and reloaded on rerun
    when the run fingerprint (dataset/seed/sizes/model) matches.
    """
    ckpt = out_path.with_suffix(f".ckpt_{name}.json")
    fp = _ckpt_fingerprint(args)
    if ckpt.exists():
        try:
            saved = json.loads(ckpt.read_text())
            if saved.get("fingerprint") == fp:
                print(f"    ↻ resuming {name} from checkpoint ({ckpt.name})")
                return saved["result"]
        except (json.JSONDecodeError, KeyError):
            pass
    result = compute()
    if result is not None:
        ckpt.write_text(json.dumps({"fingerprint": fp, "result": result}))
    return result


def run_gliner2(test, types, nl, model_name):
    from gliner2 import GLiNER2

    m = GLiNER2.from_pretrained(model_name)
    labels = [nl[t] for t in types]
    rev = {nl[t]: t for t in types}
    t0 = time.time()
    preds = []
    for r in test:
        out = m.extract_entities(r["text"][:2000], labels, threshold=0.5, include_spans=True)
        ents = []
        # GLiNER2 returns {"entities": {label: [{text,start,end}]}}
        for label, spans in (out.get("entities", {}) or {}).items():
            for e in spans:
                ents.append(
                    {
                        "text": e["text"],
                        "start": e.get("start"),
                        "end": e.get("end"),
                        "type": rev.get(label, label),
                    }
                )
        preds.append(ents)
    return {"preds": preds, "infer_ms": round((time.time() - t0) / max(1, len(test)) * 1000, 3)}


def run_llm(test, types, client, model, cache_path, workers=4):
    """LLM span extraction with a bounded parallel pool.

    Calls are independent, so a small worker pool cuts wall-clock ~4x while
    staying far below the burst rate that tripped Baseten limits before.
    Each call retries with backoff; failures are returned but never cached.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            try:
                e = json.loads(line)
                cache[e["key"]] = e["ents"]
            except (json.JSONDecodeError, KeyError):
                pass
    # NB: built with replace(), not str.format() — the literal JSON braces in
    # the instruction would be parsed as format placeholders (KeyError '"text"').
    tmpl = (
        "Extract all entity spans from the text. Types: " + ", ".join(types) + ".\n"
        'Return ONLY a JSON list of {"text": <exact substring>, "type": <TYPE>}.\n\nTEXT:\n'
    )
    lock = threading.Lock()
    stats = {"calls": 0, "errors": 0}

    def one(r):
        key = hashlib.sha256((model + "|" + r["text"]).encode()).hexdigest()[:24]
        with lock:
            if key in cache:
                return cache[key]
        ents, ok, last_err = [], False, None
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    max_completion_tokens=1024,
                    temperature=0,
                    messages=[{"role": "user", "content": tmpl + r["text"][:3000]}],
                )
                txt = (resp.choices[0].message.content or "").strip()
                if txt.startswith("```"):
                    txt = re.sub(r"^```(?:json)?\s*", "", txt)
                    txt = re.sub(r"\s*```$", "", txt)
                arr = json.loads(txt[txt.index("[") : txt.rindex("]") + 1])
                ents = [
                    {"text": str(e.get("text", "")), "type": str(e.get("type", "")).upper().strip()}
                    for e in arr
                    if isinstance(e, dict)
                ]
                ok = True
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 * (attempt + 1))
        with lock:
            stats["calls"] += 1
            if not ok:
                stats["errors"] += 1
                if stats["errors"] <= 3:
                    print(f"  LLM call failed after retries: {last_err}")
                return ents  # not cached — retried next run
            cache[key] = ents
            with open(cache_path, "a") as f:
                f.write(json.dumps({"key": key, "ents": ents}) + "\n")
        return ents

    with ThreadPoolExecutor(max_workers=workers) as pool:
        preds = list(pool.map(one, test))
    if stats["errors"]:
        print(f"  ⚠ {stats['errors']}/{stats['calls']} LLM calls failed after retries (not cached)")
    return {"preds": preds, "llm_calls": stats["calls"], "errors": stats["errors"]}


def main():
    from openai import OpenAI

    p = argparse.ArgumentParser(description="Structured extraction: RuleChef vs GLiNER vs LLM")
    p.add_argument("--dataset", choices=list(DATASETS), default="ontonotes")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base-url", default=None)
    p.add_argument("--gliner2", default="fastino/gliner2-base")
    p.add_argument("--llm-workers", type=int, default=4)
    p.add_argument("--holdout-fraction", type=float, default=0.2)
    p.add_argument("--train", type=int, default=1500)
    p.add_argument("--test", type=int, default=800)
    p.add_argument("--max-rules", type=int, default=60)
    p.add_argument("--max-samples", type=int, default=80)
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    loader, fmt_types, sem_types, nl, synthetic = DATASETS[args.dataset]
    types = fmt_types + sem_types
    train, test = loader(args.train, args.test, args.seed, types)
    gold = [r["entities"] for r in test]
    tag = " [SYNTHETIC]" if synthetic else " [real text]"
    print(
        f"{args.dataset}{tag}: train {len(train)} test {len(test)} | "
        f"{len(fmt_types)} format + {len(sem_types)} semantic types"
    )

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=args.base_url,
        timeout=240.0,
        max_retries=1,
    )
    out_path = Path(args.output or f"benchmarks/results/results_extract_{args.dataset}.json")
    methods = {}

    print("\n[1] RuleChef...")
    rc = with_checkpoint(
        "rulechef",
        out_path,
        args,
        lambda: run_rulechef(train, test, types, client, args.model, args),
    )
    if rc:
        methods["rulechef"] = rc
        print(f"    {rc['num_rules']} rules, learn {rc['learn_s']}s, infer {rc['infer_ms']}ms/doc")
    print(f"\n[2] GLiNER2 ({args.gliner2})...")
    try:
        methods["gliner2"] = with_checkpoint(
            "gliner2", out_path, args, lambda: run_gliner2(test, types, nl, args.gliner2)
        )
    except Exception as e:
        print(f"    failed: {e}")

    print("\n[3] LLM...")
    methods["llm"] = run_llm(
        test,
        types,
        client,
        args.model,
        out_path.with_suffix(".llm_cache.jsonl"),
        workers=args.llm_workers,
    )
    print(f"    {methods['llm']['llm_calls']} new calls")

    scored = {n: score(m["preds"], gold) for n, m in methods.items()}
    names = [n for n in ["rulechef", "gliner2", "llm"] if n in scored]
    gold_total = sum(len(d) for d in gold)
    gold_fmt = sum(sum(1 for e in d if e["type"] in fmt_types) for d in gold)

    print(f"\n{'=' * 92}\nEXTRACTION — {args.dataset} per-type F1 (text+type match)\n{'=' * 92}")
    print(f"{'type':<16}{'group':<9}" + "".join(f"{n:>12}" for n in names))
    print("-" * 92)
    for t in types:
        grp = "format" if t in fmt_types else "semantic"
        print(f"{t:<16}{grp:<9}" + "".join(f"{prf(scored[n][t])[2]:>12.2f}" for n in names))
    print("-" * 92)
    for grp, ts in [("FORMAT", fmt_types), ("SEMANTIC", sem_types)]:
        print(
            f"{grp + ' micro-F1':<25}"
            + "".join(f"{group_f1(scored[n], ts)[2]:>12.2f}" for n in names)
        )
    print("=" * 92)
    print(f"Gold spans: {gold_total} ({gold_fmt} format = {gold_fmt / max(1, gold_total):.0%})")
    if "rulechef" in scored:
        fp, fr, ff = group_f1(scored["rulechef"], fmt_types)
        print(f"RuleChef FORMAT: P={fp:.2f} R={fr:.2f} F1={ff:.2f}")

    out_path.write_text(
        json.dumps(
            {
                "config": vars(args),
                "synthetic": synthetic,
                "format_types": fmt_types,
                "semantic_types": sem_types,
                "gold_total": gold_total,
                "gold_format": gold_fmt,
                "per_type": {
                    n: {
                        t: {
                            "p": prf(scored[n][t])[0],
                            "r": prf(scored[n][t])[1],
                            "f1": prf(scored[n][t])[2],
                            **scored[n][t],
                        }
                        for t in types
                    }
                    for n in names
                },
                "group_f1": {
                    n: {
                        "format": group_f1(scored[n], fmt_types),
                        "semantic": group_f1(scored[n], sem_types),
                    }
                    for n in names
                },
                "meta": {n: {k: v for k, v in methods[n].items() if k != "preds"} for n in names},
            },
            indent=2,
        )
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

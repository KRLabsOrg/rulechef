#!/usr/bin/env python3
"""Evaluate the learned TAB ruleset on the OFFICIAL TAB test split.

Runs the rules from the benchmark checkpoint over the 127 full documents of
echr_test.json and scores the predicted spans with the benchmark's own
evaluation.py (Pilan et al. 2022): mention-level recall (all / direct / quasi),
entity-level recall, and token-level precision (uniform and BERT-weighted).
This gives a direct comparison against the published Longformer / RoBERTa
baselines (P .836 / R .919 and P .441 / R .906), which use the same protocol.

No LLM involved: the rules execute deterministically.

Usage:
    python benchmarks/eval_tab_official.py \
        --test-json /tmp/tab_official/echr_test.json \
        --eval-module /tmp/tab_official/evaluation.py
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def rebuild_rules(rule_dicts):
    from rulechef.core import Rule

    return [Rule.from_dict(d) for d in rule_dicts]


def main():
    p = argparse.ArgumentParser(description="Official TAB test-split evaluation (rules only)")
    p.add_argument("--ckpt", default="benchmarks/results/results_extract_tab.ckpt_rulechef.json")
    p.add_argument("--test-json", default="/tmp/tab_official/echr_test.json")
    p.add_argument("--eval-module", default="/tmp/tab_official/evaluation.py")
    p.add_argument("--skip-bert", action="store_true", help="skip BERT-weighted precision")
    p.add_argument("--output", default="benchmarks/results/results_tab_official.json")
    args = p.parse_args()

    # --- load official evaluator ---
    spec = importlib.util.spec_from_file_location("tab_eval", args.eval_module)
    tab_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tab_eval)

    # --- load rules ---
    ckpt = json.loads(Path(args.ckpt).read_text())
    rules = rebuild_rules(ckpt["result"]["rules"])
    print(f"Loaded {len(rules)} learned rules from {args.ckpt}")

    # --- predict spans on full documents ---
    from rulechef.core import TaskType
    from rulechef.learner import RuleLearner

    learner = RuleLearner(llm=None)

    docs = json.loads(Path(args.test_json).read_text())
    print(f"Official test split: {len(docs)} full documents")
    masked_docs = []
    t0 = time.time()
    total_spans = 0
    for doc in docs:
        out = learner._apply_rules(rules, {"text": doc["text"]}, TaskType.NER, "text")
        ents = out.get("entities", []) or []
        spans = [(e["start"], e["end"]) for e in ents if e.get("start") is not None]
        total_spans += len(spans)
        masked_docs.append(tab_eval.MaskedDocument(doc_id=doc["doc_id"], masked_spans=spans))
    elapsed = time.time() - t0
    print(
        f"Predicted {total_spans} spans over {len(docs)} docs "
        f"in {elapsed:.1f}s ({1000 * elapsed / len(docs):.1f} ms/doc)"
    )

    # --- official metrics ---
    print("Loading gold corpus (spacy tokenization)...")
    gold = tab_eval.GoldCorpus(args.test_json)

    results = {"num_rules": len(rules), "spans": total_spans, "ms_per_doc": 1000 * elapsed / len(docs)}
    for include_direct, include_quasi, label in (
        (True, True, "all"),
        (True, False, "direct"),
        (False, True, "quasi"),
    ):
        r = gold.get_recall(masked_docs, include_direct, include_quasi)
        er = gold.get_entity_recall(masked_docs, include_direct, include_quasi)
        results[f"mention_recall_{label}"] = r
        results[f"entity_recall_{label}"] = er
        print(f"recall ({label}): mention={r:.3f} entity={er:.3f}")

    per_type = gold.get_recall_per_entity_type(masked_docs)
    results["mention_recall_per_type"] = per_type
    for t, r in sorted(per_type.items()):
        print(f"  {t:<10} mention recall={r:.3f}")

    uniform = tab_eval.UniformTokenWeighting()
    p_uni = gold.get_precision(masked_docs, uniform, token_level=True)
    results["token_precision_uniform"] = p_uni
    print(f"token precision (uniform): {p_uni:.3f}")

    if not args.skip_bert:
        print("Computing BERT-weighted precision (the official metric)...")
        bert = tab_eval.BertTokenWeighting()
        p_bert = gold.get_precision(masked_docs, bert, token_level=True)
        results["token_precision_bert_weighted"] = p_bert
        print(f"token precision (BERT-weighted): {p_bert:.3f}")

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

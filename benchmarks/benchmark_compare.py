#!/usr/bin/env python3
"""Head-to-head comparison: RuleChef vs neural cheap tiers vs the LLM.

For a classification dataset, learns RuleChef rules on a train split and
evaluates every method on a held-out TEST split, reporting the axes that
matter for the "interpretable cheap tier" thesis:

  method                  | test acc | coverage | LLM calls | interp | editable | GPU
  LLM zero-shot           |   x.xx   |   100%   |   100%    |   no   |   no     | no
  RuleChef rules-only     |   x.xx   |   yy%    |    0%     |  YES   |  YES     | no
  RuleChef hybrid         |   x.xx   |   100%   |    zz%    |  part  |  part    | no
  LogReg(MiniLM emb)      |   x.xx   |   100%   |    0%     |   no   |   no     | (cpu ok)
  Zero-shot NLI (DeBERTa) |   x.xx   |   100%   |    0%     |   no   |   no     | (cpu ok)

"coverage" = fraction of test items the method gives a (non-abstain) answer.
"hybrid" routes rule-abstentions to the LLM. Accuracy is vs gold labels.

Usage:
    OPENAI_API_KEY=... python benchmarks/benchmark_compare.py \
        --dataset dbpedia --base-url https://inference.baseten.co/v1 \
        --model moonshotai/Kimi-K2.6 --train 1000 --test 1000
"""

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
import uuid
from pathlib import Path

# ── Dataset loaders → (train_records, test_records, class_names) ────


def _load_dbpedia():
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/dbpedia_14")
    names = ds["train"].features["label"].names

    def recs(split):
        return [
            {"text": f"{r['title']}. {r['content']}".strip(), "label": names[r["label"]]}
            for r in split
        ]

    return recs(ds["train"]), recs(ds["test"]), names


def _load_agnews():
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news")
    names = ds["train"].features["label"].names

    def recs(split):
        return [{"text": r["text"], "label": names[r["label"]]} for r in split]

    return recs(ds["train"]), recs(ds["test"]), names


def _load_banking77():
    from datasets import load_dataset

    ds = load_dataset("legacy-datasets/banking77", trust_remote_code=True)
    names = ds["train"].features["label"].names

    def recs(split):
        return [{"text": r["text"], "label": names[r["label"]]} for r in split]

    return recs(ds["train"]), recs(ds["test"]), names


DATASETS = {"dbpedia": _load_dbpedia, "agnews": _load_agnews, "banking77": _load_banking77}


def stratified(records, n, seed):
    """Sample ~n records stratified by label."""
    from collections import defaultdict

    rng = random.Random(seed)
    by = defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    per = max(1, n // len(by))
    out = []
    for lbl in sorted(by):
        items = list(by[lbl])
        rng.shuffle(items)
        out.extend(items[:per])
    rng.shuffle(out)
    return out


def acc(pairs):
    """(gold, pred) → accuracy over non-empty preds counted against all."""
    if not pairs:
        return 0.0, 0.0
    answered = [(g, p) for g, p in pairs if p]
    correct = sum(1 for g, p in answered if g == p)
    return (
        correct / len(pairs),  # accuracy over ALL (abstain = wrong)
        len(answered) / len(pairs),  # coverage
    )


# ── LLM (cached) ────────────────────────────────────────────


def build_prompt(class_names):
    return (
        "Classify the text into exactly one of these categories:\n"
        f"{', '.join(class_names)}\n\nTEXT:\n{{text}}\n\n"
        "Respond with only the category name, exactly as written above."
    )


class CachedLLM:
    def __init__(self, client, model, cache_path, class_names, prompt):
        self.client, self.model = client, model
        self.cache_path = cache_path
        self.class_names = class_names
        self.prompt = prompt
        self.lower = {c.lower(): c for c in class_names}
        self.cache, self.calls = {}, 0
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                try:
                    e = json.loads(line)
                    self.cache[e["key"]] = e["label"]
                except (json.JSONDecodeError, KeyError):
                    pass

    def _coerce(self, raw):
        r = raw.strip().lower()
        if r in self.lower:
            return self.lower[r]
        for cl, c in self.lower.items():
            if cl in r:
                return c
        return self.class_names[0]

    def label(self, text):
        key = hashlib.sha256(text.encode()).hexdigest()[:24]
        if key in self.cache:
            return self.cache[key]
        out = self.class_names[0]
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self.prompt.format(text=text[:3000])}],
                max_completion_tokens=16,
                temperature=0,
            )
            out = self._coerce(resp.choices[0].message.content or "")
        except Exception as e:
            print(f"  LLM error: {e}")
        self.calls += 1
        self.cache[key] = out
        with open(self.cache_path, "a") as f:
            f.write(json.dumps({"key": key, "label": out}) + "\n")
        return out


# ── Baselines ───────────────────────────────────────────────


def run_logreg(train, test, seed):
    """Logistic regression on MiniLM sentence embeddings (SetFit-style)."""
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    t0 = time.time()
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    xtr = enc.encode([r["text"][:1000] for r in train], show_progress_bar=False)
    xte = enc.encode([r["text"][:1000] for r in test], show_progress_bar=False)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(xtr, [r["label"] for r in train])
    preds = clf.predict(xte)
    a, cov = acc([(r["label"], p) for r, p in zip(test, preds)])
    return {"accuracy": a, "coverage": cov, "train_s": round(time.time() - t0, 1)}


def run_zeroshot_nli(test, class_names):
    """Zero-shot classification via an MNLI model (no training)."""
    from transformers import pipeline

    t0 = time.time()
    clf = pipeline(
        "zero-shot-classification", model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
    )
    preds = []
    for r in test:
        out = clf(r["text"][:1000], candidate_labels=class_names, multi_label=False)
        preds.append(out["labels"][0])
    a, cov = acc([(r["label"], p) for r, p in zip(test, preds)])
    return {"accuracy": a, "coverage": cov, "infer_s": round(time.time() - t0, 1)}


def run_rulechef(train, test, class_names, client, model, args):
    from rulechef import RuleChef
    from rulechef.core import Task, TaskType
    from rulechef.ranking import rank_rules, wilson_lower_bound

    task = Task(
        name=f"{args.dataset} classification",
        description=(
            f"Classify the input text into one of: {', '.join(class_names)}. "
            "Prefer high-precision rules; only label when the signal is unambiguous."
        ),
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
        text_field="text",
    )
    storage = tempfile.mkdtemp(prefix="rulechef_cmp_")
    coordinator = None
    if args.agentic:
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
        dataset_name="cmp",
        storage_path=storage,
        allowed_formats=["regex", "code"],
        model=model,
        max_rules=args.max_rules,
        max_samples=args.max_samples,
        synthesis_strategy="per_class",
        coordinator=coordinator,
        synthesis_output_tokens=16384,
        patch_output_tokens=16384,
    )
    for r in train:
        chef.add_observation({"text": r["text"]}, {"label": r["label"]})

    t0 = time.time()
    result = chef.learn_rules(
        run_evaluation=True,
        max_refinement_iterations=args.max_iterations,
        holdout_fraction=0.2,
        split_seed=args.seed,
    )
    learn_s = time.time() - t0
    if result is None:
        return None
    rules, _ = result

    # Calibrate trust on a held-out dev slice of train
    from rulechef.core import Dataset, Example

    k = max(10, int(0.2 * len(train)))
    dev = Dataset(name="cmp_dev", task=task)
    for r in train[-k:]:
        dev.examples.append(
            Example(
                id=str(uuid.uuid4())[:8],
                input={"text": r["text"]},
                expected_output={"label": r["label"]},
                source="dev",
            )
        )
    rank_rules(rules, dev, chef.learner._apply_rules, compute_marginal=False)
    trust = {
        ru.id: wilson_lower_bound(ru.validated_precision or 0.0, ru.validated_support)
        for ru in rules
    }

    # Rule predictions on test
    rule_pred, rule_trust = [], []
    for r in test:
        out = chef.learner._apply_rules(rules, {"text": r["text"]}, task.type, "text")
        p = str(out.get("label", "")).strip()
        rule_pred.append(p)
        rule_trust.append(trust.get(out.get("rule_id"), 0.0) if p else 0.0)

    import shutil

    shutil.rmtree(storage, ignore_errors=True)
    return rules, learn_s, rule_pred, rule_trust


def main():
    from openai import OpenAI

    p = argparse.ArgumentParser(description="RuleChef vs baselines comparison")
    p.add_argument("--dataset", choices=list(DATASETS), default="dbpedia")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base-url", default=None)
    p.add_argument("--train", type=int, default=1000)
    p.add_argument("--test", type=int, default=1000)
    p.add_argument("--max-rules", type=int, default=80)
    p.add_argument("--max-samples", type=int, default=120)
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument(
        "--hybrid-trust", type=float, default=0.3, help="Min trust for a rule to answer in hybrid"
    )
    p.add_argument("--agentic", action="store_true")
    p.add_argument(
        "--skip-nli", action="store_true", help="Skip the zero-shot NLI baseline (large download)"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    train_all, test_all, class_names = DATASETS[args.dataset]()
    train = stratified(train_all, args.train, args.seed)
    test = stratified(test_all, args.test, args.seed + 1)
    print(f"{args.dataset}: {len(class_names)} classes | train {len(train)} test {len(test)}")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=args.base_url)
    out_path = Path(args.output or f"benchmarks/results/results_compare_{args.dataset}.json")
    cache = CachedLLM(
        client,
        args.model,
        out_path.with_suffix(".llm_cache.jsonl"),
        class_names,
        build_prompt(class_names),
    )

    results = {}

    # 1. LLM zero-shot
    print("\n[1] LLM zero-shot...")
    llm_pred = [cache.label(r["text"]) for r in test]
    a, cov = acc([(r["label"], p) for r, p in zip(test, llm_pred)])
    results["llm_zeroshot"] = {"accuracy": a, "coverage": cov, "llm_calls_pct": 1.0}
    print(f"    acc={a:.3f}")

    # 2. RuleChef
    print("\n[2] RuleChef (learning rules)...")
    rc = run_rulechef(train, test, class_names, client, args.model, args)
    if rc is not None:
        rules, learn_s, rule_pred, rule_trust = rc
        a_rules, cov_rules = acc([(r["label"], p) for r, p in zip(test, rule_pred)])
        # Hybrid: rule answers if trust >= hybrid_trust, else LLM
        hyb, llm_used = [], 0
        for r, rp, rt, lp in zip(test, rule_pred, rule_trust, llm_pred):
            if rp and rt >= args.hybrid_trust:
                hyb.append((r["label"], rp))
            else:
                hyb.append((r["label"], lp))
                llm_used += 1
        a_hyb, _ = acc(hyb)
        # rule coverage at the hybrid trust bar
        rule_cov_at_bar = sum(
            1 for rp, rt in zip(rule_pred, rule_trust) if rp and rt >= args.hybrid_trust
        ) / len(test)
        results["rulechef_rules_only"] = {
            "accuracy": a_rules,
            "coverage": cov_rules,
            "num_rules": len(rules),
            "learn_s": round(learn_s, 1),
            "llm_calls_pct": 0.0,
        }
        results["rulechef_hybrid"] = {
            "accuracy": a_hyb,
            "coverage": 1.0,
            "llm_calls_pct": round(llm_used / len(test), 4),
            "rule_coverage_at_bar": round(rule_cov_at_bar, 4),
            "hybrid_trust": args.hybrid_trust,
        }
        print(f"    rules-only acc={a_rules:.3f} cov={cov_rules:.0%} ({len(rules)} rules)")
        print(
            f"    hybrid     acc={a_hyb:.3f} | LLM used {llm_used / len(test):.0%}, "
            f"rules handled {rule_cov_at_bar:.0%}"
        )

    # 3. LogReg on embeddings
    print("\n[3] LogReg on MiniLM embeddings...")
    try:
        results["logreg_embeddings"] = run_logreg(train, test, args.seed)
        print(f"    acc={results['logreg_embeddings']['accuracy']:.3f}")
    except Exception as e:
        print(f"    skipped: {e}")

    # 4. Zero-shot NLI
    if not args.skip_nli:
        print("\n[4] Zero-shot NLI (DeBERTa)...")
        try:
            results["zeroshot_nli"] = run_zeroshot_nli(test, class_names)
            print(f"    acc={results['zeroshot_nli']['accuracy']:.3f}")
        except Exception as e:
            print(f"    skipped: {e}")

    # Report
    print(
        f"\n{'=' * 84}\nCOMPARISON — {args.dataset} (test={len(test)}, {len(class_names)} classes)\n{'=' * 84}"
    )
    print(f"{'method':<26}{'acc':>7}{'coverage':>10}{'LLM%':>7}  interp/editable")
    print("-" * 84)
    meta = {
        "llm_zeroshot": "no / no",
        "rulechef_rules_only": "YES / YES",
        "rulechef_hybrid": "partial / partial",
        "logreg_embeddings": "no / no",
        "zeroshot_nli": "no / no",
    }
    for name in [
        "llm_zeroshot",
        "rulechef_rules_only",
        "rulechef_hybrid",
        "logreg_embeddings",
        "zeroshot_nli",
    ]:
        r = results.get(name)
        if not r:
            continue
        llm_pct = r.get("llm_calls_pct")
        llm_s = f"{llm_pct:.0%}" if llm_pct is not None else "-"
        print(
            f"{name:<26}{r['accuracy']:>7.3f}{r.get('coverage', 1.0):>10.0%}"
            f"{llm_s:>7}  {meta.get(name, '')}"
        )
    print("=" * 84)

    out_path.write_text(
        json.dumps(
            {
                "config": vars(args),
                "classes": class_names,
                "train": len(train),
                "test": len(test),
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

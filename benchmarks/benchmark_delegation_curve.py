#!/usr/bin/env python3
"""Dataset-agnostic observation-mode delegation curve for RuleChef.

The product question: as RuleChef observes a stream of production LLM calls,
what fraction of FUTURE calls can it serve from learned rules instead of the
model, and at what fidelity? This is the metric RuleChef should win on for
*rule-friendly* tasks (classes with lexical signatures), in contrast to
semantic tasks like safety where rules cover little.

Setup (faithful to Observe mode):
  - A production LLM labels a stream of inputs. RuleChef observes
    (input, llm_label) pairs — the LLM's own output is the target, so rules
    learn to *reproduce the model*.
  - At checkpoints N (e.g. 150/300/600/1200) it learns rules on the first N
    observations (held-out dev for trust calibration), then on a fixed FUTURE
    stream reports, per trust threshold:
        delegation rate = % of future calls a rule answers at trust >= T
        fidelity        = of delegated calls, % where rule == the LLM
        gold accuracy   = of delegated calls, % correct vs the dataset label
  - Trust = Wilson lower bound of the rule's held-out dev precision.

Datasets (--dataset): dbpedia (14-class ontology), agnews (4-class topic),
aegis (binary safety). dbpedia/agnews are rule-friendly; aegis is the
semantic contrast case.

Usage:
    OPENAI_API_KEY=... python benchmarks/benchmark_delegation_curve.py \
        --dataset dbpedia --base-url https://inference.baseten.co/v1 \
        --model moonshotai/Kimi-K2.6 --stream 1200 --future 400
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

# ── Dataset loaders: return (records, class_names) ──────────
# Each record is {"text": str, "label": str}.


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


def _load_aegis():
    from benchmark_aegis import load_aegis

    train, test = load_aegis()
    return train, test, ["safe", "unsafe"]


DATASETS = {"dbpedia": _load_dbpedia, "agnews": _load_agnews, "aegis": _load_aegis}


# ── Generic LLM classifier with cache ───────────────────────


def build_prompt(class_names: list[str]) -> str:
    classes = ", ".join(class_names)
    return (
        "Classify the text below into exactly one of these categories:\n"
        f"{classes}\n\n"
        "TEXT:\n{text}\n\n"
        "Respond with only the category name, exactly as written above."
    )


class CachedLLM:
    """Cached single-label classifier; maps free-text output to a class name."""

    def __init__(self, client, model, cache_path: Path, class_names: list[str], prompt: str):
        self.client = client
        self.model = model
        self.cache_path = cache_path
        self.class_names = class_names
        self.prompt = prompt
        self.lower = {c.lower(): c for c in class_names}
        self.cache = {}
        self.calls = 0
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                try:
                    e = json.loads(line)
                    self.cache[e["key"]] = e["label"]
                except (json.JSONDecodeError, KeyError):
                    continue

    def _coerce(self, raw: str) -> str:
        r = raw.strip().lower()
        if r in self.lower:
            return self.lower[r]
        # substring match (handles "Category: Album" or extra words)
        for c_lower, c in self.lower.items():
            if c_lower in r:
                return c
        return self.class_names[0]  # fallback to first class

    def label(self, text: str) -> str:
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
            print(f"  LLM error (default {out}): {e}")
        self.calls += 1
        self.cache[key] = out
        with open(self.cache_path, "a") as f:
            f.write(json.dumps({"key": key, "label": out}) + "\n")
        return out


def run(args):
    from openai import OpenAI

    from rulechef import RuleChef
    from rulechef.core import Dataset, Example, RuleFormat, Task, TaskType
    from rulechef.ranking import rank_rules, wilson_lower_bound

    train_all, _test_all, class_names = DATASETS[args.dataset]()
    rng = random.Random(args.seed)
    rng.shuffle(train_all)
    stream = train_all[: args.stream]
    future = train_all[args.stream : args.stream + args.future]
    print(f"Dataset {args.dataset}: {len(class_names)} classes")
    print(f"Stream: {len(stream)} observed calls, future test: {len(future)}")

    prompt = build_prompt(class_names)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=args.base_url)

    # Target labels the rules learn to reproduce. Default: the dataset's gold
    # labels (the rules learn the task directly — no extra LLM calls, and no
    # teacher noise). --teacher-llm switches to true Observe-mode distillation,
    # where the target is a black-box LLM's output and fidelity is measured
    # against the model rather than gold.
    if args.teacher_llm:
        cache = CachedLLM(
            client,
            args.model,
            Path(args.output).with_suffix(".llm_cache.jsonl"),
            class_names,
            prompt,
        )
        print("Distillation mode: labeling stream + future with the teacher LLM (cached)...")
        for r in stream + future:
            r["target"] = cache.label(r["text"])
        print(f"  teacher LLM calls this run: {cache.calls}")
        target_name = "teacher-LLM"
    else:
        for r in stream + future:
            r["target"] = r["label"]
        target_name = "gold"
    print(f"  Target labels: {target_name} (rules also use the LLM for synthesis/patch)")

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

    trust_levels = args.trust_levels
    rows = []
    checkpoints = [n for n in args.checkpoints if n <= len(stream)]
    for n in checkpoints:
        observed = stream[:n]
        k = max(5, int(0.2 * len(observed)))
        train_obs, dev_obs = observed[:-k], observed[-k:]
        print(
            f"\n{'=' * 60}\nCheckpoint: {n} observed "
            f"({len(train_obs)} train / {len(dev_obs)} dev)\n{'=' * 60}"
        )

        storage_dir = tempfile.mkdtemp(prefix="rulechef_deleg_")
        chef = RuleChef(
            task=task,
            client=client,
            dataset_name="deleg_curve",
            storage_path=storage_dir,
            allowed_formats=[RuleFormat.REGEX, RuleFormat.CODE],
            model=args.model,
            max_rules=args.max_rules,
            max_samples=args.max_samples,
            synthesis_strategy="per_class",
            synthesis_output_tokens=16384,
            patch_output_tokens=16384,
        )
        for r in train_obs:
            chef.add_observation({"text": r["text"]}, {"label": r["target"]})

        t0 = time.time()
        result = chef.learn_rules(
            run_evaluation=True,
            max_refinement_iterations=args.max_iterations,
            holdout_fraction=0.2,
            split_seed=args.seed,
        )
        learn_s = time.time() - t0
        if result is None:
            print("  learning failed")
            continue
        rules, _ = result

        dev_ds = Dataset(name="deleg_dev", task=task)
        for r in dev_obs:
            dev_ds.examples.append(
                Example(
                    id=str(uuid.uuid4())[:8],
                    input={"text": r["text"]},
                    expected_output={"label": r["target"]},
                    source="dev",
                )
            )
        rank_rules(rules, dev_ds, chef.learner._apply_rules, compute_marginal=False)
        trust = {
            ru.id: wilson_lower_bound(ru.validated_precision or 0.0, ru.validated_support)
            for ru in rules
        }

        # Per future call: (rule pred, trust, target label, gold label)
        scored = []
        for r in future:
            out = chef.learner._apply_rules(rules, {"text": r["text"]}, task.type, "text")
            pred = str(out.get("label", "")).strip()
            scored.append((pred, trust.get(out.get("rule_id"), 0.0), r["target"], r["label"]))
        n_future = len(future)

        by_trust = {}
        for t in trust_levels:
            deleg = [s for s in scored if s[0] and s[1] >= t]
            d = len(deleg)
            fid = sum(1 for p, _, tgt, _ in deleg if p == tgt) / d if d else 0.0
            gold = sum(1 for p, _, _, g in deleg if p == g) / d if d else 0.0
            by_trust[t] = {
                "delegation_rate": round(d / n_future, 4) if n_future else 0.0,
                "fidelity_to_target": round(fid, 4),
                "gold_acc_on_delegated": round(gold, 4),
            }

        summary = "  ".join(
            f"t{t}: {by_trust[t]['delegation_rate']:.0%}/{by_trust[t]['fidelity_to_target']:.0%}"
            for t in trust_levels
        )
        print(f"  rules={len(rules)} learn={learn_s:.0f}s | {summary}")
        rows.append(
            {
                "observed_calls": n,
                "train": len(train_obs),
                "num_rules": len(rules),
                "by_trust": by_trust,
                "learn_s": round(learn_s, 1),
            }
        )
        import shutil

        shutil.rmtree(storage_dir, ignore_errors=True)

    print(f"\n{'=' * 78}")
    print(f"DELEGATION CURVE — {args.dataset} ({target_name} target; delegation% / fidelity%)")
    print(f"{'=' * 78}")
    header = f"{'observed':>9} {'rules':>6}  " + "  ".join(f"t>={t:<6.2f}" for t in trust_levels)
    print(header)
    print("  " + "-" * len(header))
    for row in rows:
        cells = "  ".join(
            f"{row['by_trust'][t]['delegation_rate']:>3.0%}/"
            f"{row['by_trust'][t]['fidelity_to_target']:>3.0%} "
            for t in trust_levels
        )
        print(f"{row['observed_calls']:>9} {row['num_rules']:>6}  {cells}")

    if rows:
        last = rows[-1]
        best_t = max(
            (t for t in trust_levels if last["by_trust"][t]["delegation_rate"] > 0.02),
            default=trust_levels[0],
        )
        bt = last["by_trust"][best_t]
        print(
            f"\nReading: after {last['observed_calls']} observed calls, at trust>={best_t} "
            f"RuleChef serves {bt['delegation_rate']:.0%} of future {args.dataset} calls from "
            f"rules at {bt['fidelity_to_target']:.0%} fidelity to the {target_name} label "
            f"(gold acc on those: {bt['gold_acc_on_delegated']:.0%}) — that fraction of "
            "LLM calls saved."
        )

    Path(args.output).write_text(
        json.dumps(
            {
                "config": {
                    "dataset": args.dataset,
                    "classes": class_names,
                    "model": args.model,
                    "stream": len(stream),
                    "future": len(future),
                    "trust_levels": trust_levels,
                    "checkpoints": checkpoints,
                    "seed": args.seed,
                },
                "curve": rows,
            },
            indent=2,
        )
    )
    print(f"\nResults saved to {args.output}")


def main():
    p = argparse.ArgumentParser(description="RuleChef dataset-agnostic delegation curve")
    p.add_argument("--dataset", choices=list(DATASETS), default="dbpedia")
    p.add_argument(
        "--teacher-llm",
        action="store_true",
        help="Distillation mode: learn to reproduce a teacher LLM's labels "
        "instead of the dataset's gold labels (adds 1 LLM call per stream item)",
    )
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base-url", default=None)
    p.add_argument("--stream", type=int, default=1200)
    p.add_argument("--future", type=int, default=400)
    p.add_argument("--checkpoints", type=int, nargs="+", default=[150, 300, 600, 1200])
    p.add_argument("--trust-levels", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 0.9])
    p.add_argument("--max-rules", type=int, default=80)
    p.add_argument("--max-samples", type=int, default=120)
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="benchmarks/results/results_delegation_dbpedia.json")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""When2Call Tool-Calling Benchmark for RuleChef (v2 — 3-class with full tool context)

3-class classification: "cannot_answer" vs "request_for_info" vs "tool_call".
RuleChef learns patterns from tool descriptions and parameter specs to decide:
  - cannot_answer: no available tool matches the question's intent
  - request_for_info: a relevant tool exists but required parameters are missing
  - tool_call: a matching tool exists and all required info is present

Training uses SFT data (natural cannot_answer + request_for_info examples).
Evaluation on full MCQ test set (gold 3-class labels).
Combined system: RuleChef handles confident cannot_answer/request_for_info,
passes uncertain queries to LLM.

Published baselines (from When2Call paper, NAACL 2025, Table 3 MCQ):
  Llama 3.1 8B:  67% tool hallucination, 16.6 Macro F1
  Llama 3.1 70B: 57% tool hallucination, 37.8 Macro F1
  GPT-4o-Mini:   41% tool hallucination, 52.9 Macro F1
  MNM 8B + RPO:  1.2% tool hallucination, 52.4 Macro F1

Usage:
    export OPENAI_API_KEY=gsk_...
    python benchmarks/benchmark_when2call.py \
        --base-url https://api.groq.com/openai/v1 \
        --model llama-3.1-8b-instant \
        --format both --agentic --shots 500 --max-iterations 10

Requirements:
    pip install datasets rulechef openai
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Published baselines from When2Call paper (NAACL 2025, Table 3 — MCQ evaluation)
PUBLISHED_BASELINES = {
    "Llama 3.1 8B Instruct": {"tool_hall": 67, "macro_f1": 16.6},
    "Llama 3.1 70B Instruct": {"tool_hall": 57, "macro_f1": 37.8},
    "Llama 3.2 3B Instruct": {"tool_hall": 52, "macro_f1": 17.9},
    "GPT-4o-Mini": {"tool_hall": 41, "macro_f1": 52.9},
    "GPT-4o": {"tool_hall": 26, "macro_f1": 61.3},
    "Qwen 2.5 7B Instruct": {"tool_hall": 21, "macro_f1": 32.0},
    "MNM 8B Baseline": {"tool_hall": 19, "macro_f1": 31.9},
    "MNM 8B + SFT": {"tool_hall": 7, "macro_f1": 49.4},
    "MNM 8B + RPO": {"tool_hall": 1.2, "macro_f1": 52.4},
}

CLASSES = ["tool_call", "request_for_info", "cannot_answer"]

# ── Tool Formatting ────────────────────────────────────────


def _parse_tool(t):
    """Parse a tool entry (may be JSON string or dict)."""
    if isinstance(t, str):
        return json.loads(t)
    return t


def _format_tool_names(tools):
    """Format just tool names as a comma-separated list."""
    if not tools:
        return "none"
    names = [_parse_tool(t).get("name", "unknown") for t in tools]
    return ", ".join(names)


def _format_tools_rich(tools):
    """Format tools with full descriptions and required params for RuleChef input."""
    if not tools:
        return "none"
    parts = []
    for t in tools:
        parsed = _parse_tool(t)
        name = parsed.get("name", "unknown")
        desc = parsed.get("description", "")
        params = parsed.get("parameters", {})
        required = params.get("required", [])
        properties = params.get("properties", {})

        # Build param info
        req_parts = []
        for p in required:
            p_info = properties.get(p, {})
            p_desc = p_info.get("description", "")
            if p_desc:
                p_desc = p_desc[:60]
                req_parts.append(f"{p} ({p_desc})")
            else:
                req_parts.append(p)

        tool_str = f"- {name}: {desc}"
        if req_parts:
            tool_str += f"\n  Required: {', '.join(req_parts)}"

        parts.append(tool_str)
    return "\n".join(parts)


def format_rulechef_input(question, tools):
    """Format input text for RuleChef with full tool context."""
    tools_text = _format_tools_rich(tools)
    return f"Question: {question}\nTools:\n{tools_text}"


# ── Dataset Loading ─────────────────────────────────────────


def _classify_sft_response(assistant_msg):
    """Classify SFT response into cannot_answer or request_for_info."""
    lower = assistant_msg.lower()

    # Cannot answer keywords — model says it can't help
    cannot_kw = [
        "unable",
        "cannot",
        "apolog",
        "can't",
        "not able",
        "don't have",
        "i'm sorry",
        "i am sorry",
        "not possible",
        "beyond my",
        "not capable",
        "not equipped",
        "not aware",
        "lack the ability",
        "i'm a text-based",
        "no tool",
        "not designed to",
    ]
    if any(kw in lower for kw in cannot_kw):
        return "cannot_answer"

    # Everything else in SFT is request_for_info (model asks for more details)
    return "request_for_info"


def load_when2call_train(max_examples=None, seed=42):
    """Load When2Call SFT training data with 2-class labels.

    SFT data naturally contains cannot_answer and request_for_info examples.
    No tool_call examples exist in SFT — those are handled by the LLM at inference.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print("Loading When2Call SFT training data...")
    sft = load_dataset("nvidia/When2Call", "train_sft", split="train")

    records = []
    for row in sft:
        msgs = row["messages"]
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        label = _classify_sft_response(assistant_msg)
        tools = row["tools"]

        records.append(
            {
                "question": user_msg,
                "tools": tools,
                "label": label,
            }
        )

    labels = Counter(r["label"] for r in records)
    print(f"  Loaded {len(records)} SFT examples: {dict(labels)}")

    if max_examples and max_examples < len(records):
        rng = random.Random(seed)
        by_label = {}
        for r in records:
            by_label.setdefault(r["label"], []).append(r)
        per_class = max_examples // len(by_label)
        sampled = []
        for _label_name, examples in by_label.items():
            rng.shuffle(examples)
            sampled.extend(examples[:per_class])
        rng.shuffle(sampled)
        records = sampled
        labels = Counter(r["label"] for r in records)
        print(f"  Sampled {len(records)} examples: {dict(labels)}")

    return records


def load_when2call_test():
    """Load When2Call MCQ test data with gold 3-class labels."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print("Loading When2Call MCQ test data...")
    mcq = load_dataset("nvidia/When2Call", "test", split="mcq")

    records = []
    for row in mcq:
        gold = row["correct_answer"]
        tools = row["tools"]

        records.append(
            {
                "question": row["question"],
                "tools": tools,
                "gold_label": gold,
                "has_tools": len(tools) > 0,
            }
        )

    labels = Counter(r["gold_label"] for r in records)
    print(f"  Loaded {len(records)} test examples: {dict(labels)}")

    return records


# ── LLM Baseline ────────────────────────────────────────────


def _format_tools_for_llm(tools):
    """Format tools for LLM system prompt."""
    if not tools:
        return "No tools are available."
    tool_strings = []
    for t in tools:
        parsed = _parse_tool(t)
        tool_strings.append(f"<tool>{json.dumps(parsed)}</tool>")
    return "\n\n".join(tool_strings)


SYSTEM_PROMPT = """You are a helpful AI assistant.
You have access to the following tools described in <tool></tool> which you can use to answer the user's questions.
Only use a tool if it directly answers the user's question.

To use a tool, return JSON in the following format:
{"name": "tool_name", "arguments": {"argument1": "value1", "argument2": "value2", ...}}

If you cannot answer the question with the available tools, say so clearly.
If you need more information from the user before using a tool, ask a follow-up question."""


def classify_llm_response(response_text):
    """Classify an LLM response into one of 3 categories."""
    text = response_text.strip()
    lower = text.lower()

    cannot_keywords = [
        "cannot",
        "unable",
        "apolog",
        "can't",
        "don't have",
        "not able",
        "no tool",
        "not possible",
        "beyond my",
        "i'm sorry, but",
        "unfortunately",
        "not capable",
        "not aware",
        "not equipped",
        "lack the ability",
        "i'm a text-based",
    ]
    lead = lower[:200]
    if any(kw in lead for kw in cannot_keywords):
        return "cannot_answer"

    # Check for tool call (JSON with name + arguments)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "name" in parsed:
            return "tool_call"
    except (json.JSONDecodeError, ValueError):
        pass

    json_match = re.search(r'\{[^{}]*"name"\s*:', text)
    if json_match:
        return "tool_call"

    if any(kw in lower for kw in cannot_keywords):
        return "cannot_answer"

    if text.endswith("?") or any(
        kw in lower
        for kw in [
            "could you provide",
            "could you please",
            "can you provide",
            "i need",
            "please provide",
            "what is the",
            "which",
            "please specify",
            "could you clarify",
        ]
    ):
        return "request_for_info"

    return "tool_call"


def run_llm_baseline(test_data, client, model, cache_path=None):
    """Run LLM on all test examples and classify responses."""
    cache = {}
    legacy_cache = {}
    if cache_path and Path(cache_path).exists():
        print(f"  Loading LLM cache from {cache_path}")
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                cache[entry["question_hash"]] = entry
                # Older cache files used Python's process-randomized hash().
                # Fall back to the stored truncated question/tool names when possible.
                if "question" in entry and "tool_names" in entry:
                    legacy_cache[(entry["question"], entry["tool_names"])] = entry
        print(f"  Cached: {len(cache)} responses")

    results = []
    cache_hits = 0
    cache_file = Path(cache_path).open("a") if cache_path else None  # noqa: SIM115

    for i, record in enumerate(test_data):
        tool_names = _format_tool_names(record["tools"])
        q_hash = hashlib.sha256(
            json.dumps(
                {"question": record["question"], "tool_names": tool_names},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        entry = cache.get(q_hash)
        if entry is None:
            entry = legacy_cache.get((record["question"][:200], tool_names))

        if entry is not None:
            predicted = classify_llm_response(entry["response"])
            results.append(
                {
                    "response": entry["response"],
                    "predicted": predicted,
                    "gold": record["gold_label"],
                }
            )
            cache_hits += 1
            continue

        tools_text = _format_tools_for_llm(record["tools"])
        system = SYSTEM_PROMPT + "\n\n" + tools_text

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": record["question"]},
                ],
                temperature=0,
                max_tokens=512,
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  LLM error on example {i}: {e}")
            response_text = ""

        predicted = classify_llm_response(response_text)

        results.append(
            {
                "response": response_text,
                "predicted": predicted,
                "gold": record["gold_label"],
            }
        )

        if cache_file:
            cache_file.write(
                json.dumps(
                    {
                        "question_hash": q_hash,
                        "question": record["question"][:200],
                        "tool_names": tool_names,
                        "response": response_text,
                        "predicted": predicted,
                        "gold": record["gold_label"],
                    }
                )
                + "\n"
            )
            cache_file.flush()

        if (i + 1) % 50 == 0:
            print(f"  LLM baseline: {i + 1}/{len(test_data)}")

    if cache_file:
        cache_file.close()

    print(f"  LLM baseline complete: {len(results)} results ({cache_hits} cached)")
    return results


# ── Metrics ─────────────────────────────────────────────────


def _per_class_metrics(results, pred_key="predicted", gold_key="gold"):
    """Compute per-class precision/recall/F1."""
    per_class = {}
    for cls in CLASSES:
        tp = sum(1 for r in results if r[pred_key] == cls and r[gold_key] == cls)
        fp = sum(1 for r in results if r[pred_key] == cls and r[gold_key] != cls)
        fn = sum(1 for r in results if r[pred_key] != cls and r[gold_key] == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return per_class


def compute_llm_metrics(llm_results, test_data):
    """Compute LLM baseline metrics."""
    correct = sum(1 for r in llm_results if r["predicted"] == r["gold"])
    accuracy = correct / len(llm_results) if llm_results else 0

    per_class = _per_class_metrics(llm_results)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)

    # Tool hallucination (paper def): tool calls on zero-tool cannot_answer examples
    no_tool_ca = [
        (r, t)
        for r, t in zip(llm_results, test_data)
        if t["gold_label"] == "cannot_answer" and not t["has_tools"]
    ]
    if no_tool_ca:
        hall = sum(1 for r, _ in no_tool_ca if r["predicted"] == "tool_call")
        tool_hall_rate = hall / len(no_tool_ca)
    else:
        tool_hall_rate = 0

    # Tool misuse: tool calls on ALL cannot_answer examples (including irrelevant tools)
    all_ca = [r for r in llm_results if r["gold"] == "cannot_answer"]
    if all_ca:
        misuse = sum(1 for r in all_ca if r["predicted"] == "tool_call")
        tool_misuse_rate = misuse / len(all_ca)
    else:
        tool_misuse_rate = 0

    pred_dist = Counter(r["predicted"] for r in llm_results)
    confusion = {g: Counter() for g in CLASSES}
    for r in llm_results:
        confusion[r["gold"]][r["predicted"]] += 1

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in per_class.items()},
        "tool_hallucination_rate": tool_hall_rate,
        "tool_misuse_rate": tool_misuse_rate,
        "num_no_tool_ca": len(no_tool_ca),
        "prediction_distribution": dict(pred_dist),
        "confusion": {g: dict(confusion[g]) for g in CLASSES},
    }


def compute_rulechef_metrics(rc_results):
    """Compute RuleChef standalone metrics with per-class breakdown."""
    total = len(rc_results)
    covered = sum(1 for r in rc_results if r["predicted"] is not None)
    # For uncovered examples, predicted is None → doesn't match gold
    correct = sum(1 for r in rc_results if r["predicted"] == r["gold"])

    per_class = _per_class_metrics(rc_results)
    # Macro F1 only over classes RuleChef was trained on (cannot_answer, request_for_info)
    # tool_call has 0 training data so F1=0 is expected
    macro_f1_all = sum(v["f1"] for v in per_class.values()) / len(per_class)
    macro_f1_trained = sum(per_class[c]["f1"] for c in ["cannot_answer", "request_for_info"]) / 2

    return {
        "total": total,
        "covered": covered,
        "coverage": covered / total if total else 0,
        "accuracy": correct / total if total else 0,
        "macro_f1_all": macro_f1_all,
        "macro_f1_trained_classes": macro_f1_trained,
        "per_class": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in per_class.items()},
    }


def compute_combined_metrics(rc_results, llm_results, test_data):
    """Combined system: RuleChef handles confident predictions, LLM handles the rest.

    Logic:
    - If RuleChef predicts cannot_answer → use that (skip LLM)
    - If RuleChef predicts request_for_info → use that (skip LLM)
    - Otherwise → use LLM prediction
    """
    combined = []
    rc_handled = 0

    for rc, llm, record in zip(rc_results, llm_results, test_data):
        if rc["predicted"] in ("cannot_answer", "request_for_info"):
            combined_pred = rc["predicted"]
            rc_handled += 1
        else:
            combined_pred = llm["predicted"]

        combined.append(
            {
                "predicted": combined_pred,
                "gold": record["gold_label"],
                "handled_by": "rulechef"
                if rc["predicted"] in ("cannot_answer", "request_for_info")
                else "llm",
            }
        )

    correct = sum(1 for c in combined if c["predicted"] == c["gold"])
    accuracy = correct / len(combined) if combined else 0

    per_class = _per_class_metrics(combined)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)

    # Tool hallucination on combined
    no_tool_ca = [
        (c, t)
        for c, t in zip(combined, test_data)
        if t["gold_label"] == "cannot_answer" and not t["has_tools"]
    ]
    if no_tool_ca:
        hall = sum(1 for c, _ in no_tool_ca if c["predicted"] == "tool_call")
        tool_hall_rate = hall / len(no_tool_ca)
    else:
        tool_hall_rate = 0

    all_ca = [c for c in combined if c["gold"] == "cannot_answer"]
    if all_ca:
        misuse = sum(1 for c in all_ca if c["predicted"] == "tool_call")
        tool_misuse_rate = misuse / len(all_ca)
    else:
        tool_misuse_rate = 0

    llm_calls_saved = rc_handled
    llm_calls_saved_pct = rc_handled / len(combined) if combined else 0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in per_class.items()},
        "tool_hallucination_rate": tool_hall_rate,
        "tool_misuse_rate": tool_misuse_rate,
        "llm_calls_saved": llm_calls_saved,
        "llm_calls_saved_pct": llm_calls_saved_pct,
    }


# ── RuleChef Training & Evaluation ─────────────────────────


def train_rulechef(train_data, args, client):
    """Train RuleChef on 2-class task (cannot_answer + request_for_info)."""
    from rulechef import RuleChef
    from rulechef.core import RuleFormat, Task, TaskType

    task = Task(
        name="Tool Call Routing",
        description=(
            "Given a user question and available tool descriptions (with required parameters), "
            "classify the query into one of:\n"
            "- 'cannot_answer': no available tool can help with this question "
            "(tools are semantically irrelevant to the question's intent)\n"
            "- 'request_for_info': a relevant tool exists but the user hasn't provided "
            "a required parameter value (e.g., tool needs 'city' but user didn't specify one)\n\n"
            "Key patterns:\n"
            "- If 'Tools: none' → cannot_answer\n"
            "- If question topic doesn't match ANY tool description → cannot_answer\n"
            "- If question matches a tool but a required parameter isn't in the question → request_for_info\n"
            "- The input includes tool descriptions and required parameters to help detect these patterns."
        ),
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
        text_field="text",
    )

    format_map = {
        "code": [RuleFormat.CODE],
        "regex": [RuleFormat.REGEX],
        "both": [RuleFormat.REGEX, RuleFormat.CODE],
    }
    allowed_formats = format_map.get(args.format, [RuleFormat.REGEX])

    import tempfile

    storage_dir = tempfile.mkdtemp(prefix="rulechef_when2call_")

    coordinator = None
    if args.agentic:
        from rulechef.coordinator import AgenticCoordinator

        coordinator = AgenticCoordinator(client, model=args.synthesis_model)

    chef = RuleChef(
        task=task,
        client=client,
        dataset_name="when2call_v2",
        storage_path=storage_dir,
        allowed_formats=allowed_formats,
        model=args.synthesis_model,
        use_grex=not args.no_grex,
        max_rules=args.max_rules,
        max_samples=args.max_samples,
        coordinator=coordinator,
        temperature=0,
    )

    print(f"\nAdding {len(train_data)} training examples to RuleChef...")
    for ex in train_data:
        text = format_rulechef_input(ex["question"], ex["tools"])
        chef.add_example({"text": text}, {"label": ex["label"]})

    print("\nLearning rules...")
    print(f"  synthesis model={args.synthesis_model}, format={args.format}")
    print(
        f"  max_rules={args.max_rules}, max_samples={args.max_samples}, max_iterations={args.max_iterations}"
    )
    t0 = time.time()
    result = chef.learn_rules(max_refinement_iterations=args.max_iterations)
    t_learn = time.time() - t0

    if result is None:
        print("ERROR: Learning failed!")
        return None, None, None, storage_dir

    rules, metrics = result
    print(f"  Learned {len(rules)} rules in {t_learn:.1f}s")

    return chef, rules, t_learn, storage_dir


def evaluate_rulechef(chef, rules, test_data):
    """Evaluate RuleChef standalone on test data."""
    from rulechef.core import TaskType

    results = []
    t0 = time.time()

    for record in test_data:
        text = format_rulechef_input(record["question"], record["tools"])
        input_data = {"text": text}
        output = chef.learner._apply_rules(rules, input_data, TaskType.CLASSIFICATION, "text")
        predicted = output.get("label", None)
        results.append(
            {
                "predicted": predicted,
                "gold": record["gold_label"],
                "has_tools": record["has_tools"],
            }
        )

    t_eval = time.time() - t0
    return results, t_eval


# ── Main Benchmark Runner ──────────────────────────────────


def run_benchmark(args):
    from openai import OpenAI

    test_client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=args.base_url,
    )

    synthesis_base_url = args.synthesis_base_url or args.base_url
    synthesis_client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=synthesis_base_url,
    )

    # 1. Load data
    train_data = load_when2call_train(max_examples=args.shots, seed=args.seed)
    test_data = load_when2call_test()

    if args.test_limit:
        rng = random.Random(args.seed)
        rng.shuffle(test_data)
        test_data = test_data[: args.test_limit]
        labels = Counter(r["gold_label"] for r in test_data)
        print(f"  Test subset: {len(test_data)} ({dict(labels)})")

    # Show sample input format
    sample = train_data[0]
    sample_text = format_rulechef_input(sample["question"], sample["tools"])
    print(f"\n  Sample RuleChef input (label={sample['label']}):")
    for line in sample_text.split("\n")[:8]:
        print(f"    {line}")
    if sample_text.count("\n") > 7:
        print("    ...")

    # 2. Run LLM baseline
    print(f"\n{'=' * 70}")
    print(f"STEP 1: LLM BASELINE ({args.model})")
    print(f"{'=' * 70}")
    cache_path = args.cache or f"benchmarks/cache_when2call_{args.model.replace('/', '_')}.jsonl"
    t0 = time.time()
    llm_results = run_llm_baseline(test_data, test_client, args.model, cache_path)
    t_llm = time.time() - t0
    llm_metrics = compute_llm_metrics(llm_results, test_data)

    print(f"\n  LLM Baseline Results ({args.model}):")
    print(f"    Accuracy:              {llm_metrics['accuracy']:.1%}")
    print(f"    Macro F1:              {llm_metrics['macro_f1'] * 100:.1f}")
    print(
        f"    Tool hallucination:    {llm_metrics['tool_hallucination_rate']:.1%} (zero-tool: {llm_metrics['num_no_tool_ca']} examples)"
    )
    print(f"    Tool misuse:           {llm_metrics['tool_misuse_rate']:.1%}")
    print(f"    Prediction dist:       {llm_metrics['prediction_distribution']}")
    print(f"    Time: {t_llm:.1f}s")
    print("\n    Per-class F1:")
    for cls in CLASSES:
        m = llm_metrics["per_class"][cls]
        print(f"      {cls:<20} P={m['precision']:.1%} R={m['recall']:.1%} F1={m['f1']:.1%}")

    # 3. Train RuleChef
    print(f"\n{'=' * 70}")
    print("STEP 2: TRAIN RULECHEF")
    print(f"{'=' * 70}")
    chef, rules, t_learn, storage_dir = train_rulechef(train_data, args, synthesis_client)

    if chef is None:
        print("RuleChef training failed!")
        return

    # 4. Evaluate RuleChef standalone
    print(f"\n{'=' * 70}")
    print("STEP 3: EVALUATE RULECHEF")
    print(f"{'=' * 70}")
    rc_results, t_rc_eval = evaluate_rulechef(chef, rules, test_data)
    rc_metrics = compute_rulechef_metrics(rc_results)

    print(f"\n  RuleChef Standalone ({len(rules)} rules):")
    print(f"    Coverage:              {rc_metrics['coverage']:.1%}")
    print(f"    Accuracy (covered):    {rc_metrics['accuracy']:.1%}")
    print(f"    Macro F1 (all 3):      {rc_metrics['macro_f1_all'] * 100:.1f}")
    print(f"    Macro F1 (trained):    {rc_metrics['macro_f1_trained_classes'] * 100:.1f}")
    print(f"    Time: {t_rc_eval:.3f}s ({t_rc_eval / len(test_data) * 1000:.2f}ms/query)")
    print("\n    Per-class F1:")
    for cls in CLASSES:
        m = rc_metrics["per_class"][cls]
        print(
            f"      {cls:<20} P={m['precision']:.1%} R={m['recall']:.1%} F1={m['f1']:.1%} (TP={m['tp']} FP={m['fp']} FN={m['fn']})"
        )

    # 5. Combined system
    print(f"\n{'=' * 70}")
    print("STEP 4: COMBINED SYSTEM (RuleChef + LLM)")
    print(f"{'=' * 70}")
    combined_metrics = compute_combined_metrics(rc_results, llm_results, test_data)

    print("\n  Combined System:")
    print(f"    Accuracy:              {combined_metrics['accuracy']:.1%}")
    print(f"    Macro F1:              {combined_metrics['macro_f1'] * 100:.1f}")
    print(f"    Tool hallucination:    {combined_metrics['tool_hallucination_rate']:.1%}")
    print(f"    Tool misuse:           {combined_metrics['tool_misuse_rate']:.1%}")
    print(
        f"    LLM calls saved:       {combined_metrics['llm_calls_saved']} ({combined_metrics['llm_calls_saved_pct']:.1%})"
    )
    print("\n    Per-class F1:")
    for cls in CLASSES:
        m = combined_metrics["per_class"][cls]
        print(f"      {cls:<20} P={m['precision']:.1%} R={m['recall']:.1%} F1={m['f1']:.1%}")

    # 6. Summary comparison
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")

    def fmt(v):
        return f"{v:.1%}"

    def fmt_f1(v):
        return f"{v * 100:.1f}"

    print(f"\n  {'Metric':<30} {'LLM':>10} {'RuleChef':>10} {'Combined':>10}")
    print(f"  {'─' * 62}")
    print(
        f"  {'Accuracy':<30} {fmt(llm_metrics['accuracy']):>10} {fmt(rc_metrics['accuracy']):>10} {fmt(combined_metrics['accuracy']):>10}"
    )
    print(
        f"  {'Macro F1':<30} {fmt_f1(llm_metrics['macro_f1']):>10} {fmt_f1(rc_metrics['macro_f1_all']):>10} {fmt_f1(combined_metrics['macro_f1']):>10}"
    )
    print(
        f"  {'Tool hallucination':<30} {fmt(llm_metrics['tool_hallucination_rate']):>10} {'—':>10} {fmt(combined_metrics['tool_hallucination_rate']):>10}"
    )
    print(
        f"  {'Tool misuse':<30} {fmt(llm_metrics['tool_misuse_rate']):>10} {'—':>10} {fmt(combined_metrics['tool_misuse_rate']):>10}"
    )
    print(
        f"  {'LLM calls saved':<30} {'0':>10} {'—':>10} {fmt(combined_metrics['llm_calls_saved_pct']):>10}"
    )
    print(
        f"  {'Latency / query':<30} {t_llm / len(test_data) * 1000:>8.0f}ms {'—':>10} {t_rc_eval / len(test_data) * 1000:>8.2f}ms"
    )

    # Per-class detail
    print("\n  Per-class F1:")
    print(f"  {'Class':<20} {'LLM':>10} {'RuleChef':>10} {'Combined':>10}")
    print(f"  {'─' * 52}")
    for cls in CLASSES:
        lf = llm_metrics["per_class"][cls]["f1"]
        rf = rc_metrics["per_class"][cls]["f1"]
        cf = combined_metrics["per_class"][cls]["f1"]
        print(f"  {cls:<20} {fmt_f1(lf):>10} {fmt_f1(rf):>10} {fmt_f1(cf):>10}")

    # Published baselines
    print(f"\n  {'─' * 62}")
    print("  PUBLISHED BASELINES (When2Call paper, MCQ evaluation)")
    print(f"  {'─' * 62}")
    print(f"  {'Model':<35} {'Hall%':>8} {'Macro F1':>10}")
    print(f"  {'─' * 62}")
    for name, info in PUBLISHED_BASELINES.items():
        print(f"  {name:<35} {info['tool_hall']:>7.1f}% {info['macro_f1']:>9.1f}")
    print(f"  {'─' * 62}")
    print(
        f"  {'RuleChef+' + args.model:<35} {combined_metrics['tool_hallucination_rate'] * 100:>7.1f}% {combined_metrics['macro_f1'] * 100:>9.1f}"
    )
    print(f"  {'─' * 62}")
    print("  Note: Published use MCQ log-prob eval; ours uses free-form generation + classifier")
    print(
        f"  Tool misuse (irrelevant tools): LLM={llm_metrics['tool_misuse_rate']:.1%} → Combined={combined_metrics['tool_misuse_rate']:.1%}"
    )

    # Rules
    print(f"\n  Rules learned ({len(rules)}):")
    for r in sorted(rules, key=lambda r: -r.priority):
        content_preview = r.content.replace("\n", " ")[:100]
        print(f"    [{r.format.value} p={r.priority}] {r.name}: {content_preview}")

    # 7. Save results
    output = {
        "config": {
            "model": args.model,
            "synthesis_model": args.synthesis_model,
            "format": args.format,
            "shots": args.shots,
            "max_rules": args.max_rules,
            "max_samples": args.max_samples,
            "max_iterations": args.max_iterations,
            "seed": args.seed,
            "train_size": len(train_data),
            "test_size": len(test_data),
            "use_grex": not args.no_grex,
            "agentic": args.agentic,
            "task_framing": "3-class with full tool descriptions",
        },
        "llm_baseline": llm_metrics,
        "rulechef_standalone": rc_metrics,
        "combined_system": combined_metrics,
        "rulechef_training_time_s": round(t_learn, 1),
        "rulechef_eval_time_s": round(t_rc_eval, 3),
        "rulechef_per_query_ms": round(t_rc_eval / len(test_data) * 1000, 2),
        "llm_time_s": round(t_llm, 1),
        "published_baselines": PUBLISHED_BASELINES,
        "rules": [
            {
                "name": r.name,
                "format": r.format.value,
                "content": r.content[:500],
                "priority": r.priority,
            }
            for r in rules
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {output_path}")

    import shutil

    shutil.rmtree(storage_dir, ignore_errors=True)


# ── CLI ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="When2Call 3-Class Tool Routing Benchmark for RuleChef"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=500,
        help="Total training examples from SFT (balanced, default: 500)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-8b-instant",
        help="LLM model to test (default: llama-3.1-8b-instant)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.groq.com/openai/v1",
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--synthesis-model",
        type=str,
        default="moonshotai/kimi-k2-instruct-0905",
        help="LLM for RuleChef rule synthesis (default: kimi-k2)",
    )
    parser.add_argument(
        "--synthesis-base-url",
        type=str,
        default=None,
        help="Base URL for synthesis LLM (defaults to --base-url)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["regex", "code", "both"],
        help="Rule format (default: both)",
    )
    parser.add_argument(
        "--max-rules",
        type=int,
        default=50,
        help="Max rules to generate (default: 50)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Max training examples in LLM prompt (default: 100)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Max refinement iterations (default: 10)",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Limit test set size for quick runs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/results_when2call_v2.json",
        help="Save results to JSON",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Path to LLM response cache JSONL",
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Use AgenticCoordinator",
    )
    parser.add_argument(
        "--no-grex",
        action="store_true",
        help="Disable grex regex hints",
    )

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()

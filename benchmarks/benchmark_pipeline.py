import json
import os
import random
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, List

from openai import OpenAI
from pydantic import BaseModel, Field

from benchmarks.data import BenchmarkRun, DataSplit, load_human_feedback
from benchmarks.io import save_checkpoint
from benchmarks.reporting import eval_metrics, evaluate_test, make_oniteration_callback
from ner_datasets.conversion import make_dataset
from rulechef.coordinator import AgenticCoordinator
from rulechef.core import Dataset, RuleFormat, Task, TaskType
from rulechef.engine import RuleChef

CHECKPOINT_FILE = "checkpoint.json"


class Entity(BaseModel):
    text: str = Field(description="The matched text span")
    start: int = Field(description="Start character offset")
    end: int = Field(description="End character offset")
    type: str = Field(description="Entity label")


class NEROutput(BaseModel):
    entities: List[Entity]


def _patch_regex_timeout(chef: RuleChef, timeout_secs: int = 5) -> None:
    """Skip regex rules that hang instead of letting them stall the run."""
    original = chef.learner.executor._execute_regex_rule
    timed_out_rules: set[str] = set()

    def _execute_with_timeout(rule, input_data, text_field=None):
        if rule.id in timed_out_rules:
            return []

        def _handler(signum, frame):
            raise TimeoutError()

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_secs)
        try:
            return original(rule, input_data, text_field)
        except TimeoutError:
            timed_out_rules.add(rule.id)
            print(f"   ⚠ Regex timeout ({timeout_secs}s): {rule.name} — skipping")
            return []
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    chef.learner.executor._execute_regex_rule = _execute_with_timeout


def build_chef(args, split: DataSplit, storage_dir: str, logger=None) -> RuleChef:
    selected_classes = sorted(split.selected_classes)
    task = Task(
        name="German Legal Named Entity Recognition",
        description=(
            f"Recognize named entities in German legal text. "
            f"Entities to look for: {', '.join(selected_classes)}."
        ),
        input_schema={"text": "str"},
        output_schema=NEROutput,
        type=TaskType.NER,
        text_field="text",
    )

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
        base_url=args.base_url,
    )

    coordinator = None
    if args.agentic:
        coordinator = AgenticCoordinator(
            client,
            model=args.model,
            prune_after_learn=args.enable_prune,
            audit_interval=args.audit_interval,
            enable_critic=args.enable_critic,
            critic_interval=args.critic_interval,
            verbose=True,
        )

    chef = RuleChef(
        task=task,
        client=client,
        dataset_name=split.name,
        model=args.model,
        allowed_formats=[RuleFormat.REGEX],
        use_grex=not args.no_grex,
        max_rules=args.max_rules,
        max_samples=args.max_samples,
        max_counter_examples=args.max_counter_examples,
        coordinator=coordinator,
        training_logger=logger,
        storage_path=storage_dir,
        sampling_strategy=args.sampling_strategy,
        synthesis_strategy=args.synthesis_strategy,
    )
    _patch_regex_timeout(chef)
    return chef


def fit_batched(
    chef: RuleChef,
    train_data,
    eval_dataset,
    batch_size=40,
    refine_per_batch=1,
    refine_every=1,
    iteration_callback=None,
    batch_callback=None,
    audit_interval=0,
    seed_rules=None,
    start_batch=0,
):
    """Learn rules incrementally over batches, refining against eval_dataset.

    Returns (rules, t_learn) on success, or None if no batch produced rules.
    """
    if seed_rules is not None:
        chef.dataset.rules = seed_rules
    batches = [train_data[i : i + batch_size] for i in range(0, len(train_data), batch_size)]
    t0 = time.time()
    result = (list(seed_rules), None) if (seed_rules and start_batch > 0) else None
    for batch_idx, batch in enumerate(batches):
        if batch_idx < start_batch:
            continue
        for ex in batch:
            chef.add_example({"text": ex["text"]}, {"entities": ex["entities"]})
        batch_result = chef.learn_rules(
            run_evaluation=False,
            incremental_only=(batch_idx > 0 or seed_rules is not None),
        )
        # current batch should see its own examples but also some previous ones for the full picture
        if chef.dataset.rules:
            MAX_EXAMPLES = 200
            if len(chef.dataset.examples) > MAX_EXAMPLES:
                chef.dataset.examples = chef.dataset.examples[-MAX_EXAMPLES:]
        else:
            # no rules yet — full synthesis path, must stay within token budget
            chef.dataset.examples.clear()

        if batch_result:
            result = batch_result
            rules_so_far, _ = result
            print(f"  Batch {batch_idx + 1}/{len(batches)}: {len(rules_so_far)} rules synthesized")

            if refine_per_batch > 0 and batch_idx % refine_every == 0:
                rules_so_far, refine_eval = chef.learner.evaluate_and_refine(
                    rules_so_far,
                    eval_dataset,
                    max_iterations=refine_per_batch,
                    coordinator=chef.coordinator,
                    iteration_callback=iteration_callback,
                    audit_interval=audit_interval,
                )
                chef.dataset.rules = rules_so_far
                result = (rules_so_far, refine_eval)
                print(f"  After refine: {len(rules_so_far)} rules, F1={refine_eval.micro_f1:.1%}")
            if batch_callback is not None:
                batch_callback(batch_idx, rules_so_far)
    t_learn = time.time() - t0
    if result is None:
        print("ERROR: Learning failed!")
        return

    rules, _ = result
    print(f"\nSynthesis complete ({t_learn:.1f}s)")
    print(f"  Rules generated: {len(rules)}")
    return rules, t_learn


@dataclass
class StepContext:
    rules: list
    split: DataSplit
    chef: RuleChef
    eval_dataset: Dataset
    dev_dataset: Dataset
    batch_metrics: list = field(default_factory=list)
    iteration_metrics: list = field(default_factory=list)
    best_rules: list = field(default_factory=list)
    best_f1: float = 0.0
    best_batch_idx: int = -1
    t_learn: float = 0.0
    t_eval: float = 0.0
    eval_results: Any = None
    history: list = field(default_factory=list)
    checkpoint_path: Path | None = None


class Step(ABC):
    @abstractmethod
    def run(self, context: StepContext) -> StepContext:
        pass


class BenchmarkPipeline:
    def __init__(self, steps: list[Step]):
        self.steps = steps

    def run(self, ctx: StepContext) -> StepContext:
        for step in self.steps:
            ctx = step.run(ctx)
        return ctx


class SynthesisStep(Step):
    def __init__(
        self,
        *,
        batch_size,
        refine_per_batch,
        refine_every,
        audit_interval,
        start_batch,
        phase,
        synthesis_strategy,
        seed=42,
    ):
        self.batch_size = batch_size
        self.refine_per_batch = refine_per_batch
        self.refine_every = refine_every
        self.audit_interval = audit_interval
        self.start_batch = start_batch
        self.phase = phase
        self.seed = seed
        self.synthesis_strategy = synthesis_strategy

    def run(self, ctx: StepContext) -> StepContext:
        if self.synthesis_strategy != "bulk":
            train_for_chef = ctx.split.train + ctx.split.counter_examples
            random.Random(self.seed).shuffle(train_for_chef)
        else:
            train_for_chef = ctx.split.train
        batch_metrics = list(ctx.batch_metrics)
        iteration_metrics = list(ctx.iteration_metrics)
        rules_snapshot = []
        on_iteration = make_oniteration_callback(iteration_metrics)
        best = {
            "f1": ctx.best_f1,
            "rules": list(ctx.best_rules) if ctx.best_rules else [],
            "batch_idx": ctx.best_batch_idx,
        }

        def on_batch(batch_idx, rules):
            rules_snapshot.append(
                {"batch": batch_idx, "rules": [r.to_dict() for r in best["rules"]]}
            )
            result, _ = evaluate_test(ctx.dev_dataset, rules, ctx.chef)
            batch_metrics.append(
                {"batch": batch_idx, "num_rules": len(rules), **eval_metrics(result)}
            )
            if result.micro_f1 > best["f1"]:
                best["f1"] = result.micro_f1
                best["rules"] = list(rules)
                best["batch_idx"] = batch_idx
            print(
                f"  [batch {batch_idx}] dev micro_f1={result.micro_f1:.3f}"
                f"  P={result.micro_precision:.3f}  R={result.micro_recall:.3f}"
            )
            if ctx.checkpoint_path:
                save_checkpoint(
                    ctx.checkpoint_path,
                    {
                        "phase": self.phase,
                        "completed_batches": batch_idx + 1,
                        "rules": [r.to_dict() for r in rules],
                        "best_rules": [r.to_dict() for r in best["rules"]] if best["rules"] else [],
                        "best_f1": best["f1"],
                        "best_batch_idx": best["batch_idx"],
                        "batch_metrics": batch_metrics,
                        "iteration_metrics": iteration_metrics,
                        "rules_snapshot": rules_snapshot,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

        fit_result = fit_batched(
            ctx.chef,
            train_for_chef,
            eval_dataset=ctx.eval_dataset if self.refine_per_batch > 0 else None,
            batch_size=self.batch_size,
            refine_per_batch=self.refine_per_batch,
            refine_every=self.refine_every,
            iteration_callback=on_iteration,
            batch_callback=on_batch,
            audit_interval=self.audit_interval,
            seed_rules=ctx.rules or None,
            start_batch=self.start_batch,
        )
        if fit_result is None:
            return replace(
                ctx,
                batch_metrics=batch_metrics,
                iteration_metrics=iteration_metrics,
                best_f1=best["f1"],
                best_rules=best["rules"],
                best_batch_idx=best["batch_idx"],
            )

        rules, new_t_learn = fit_result
        history_entry = {
            "phase": "train",
            "dataset": ctx.split.name,
            "num_rules": len(rules),
            "t_learn": round(ctx.t_learn + new_t_learn, 1),
            "timestamp": datetime.now().isoformat(),
            "num_train_docs": len(ctx.split.train),
            "num_eval_docs": len(ctx.split.eval),
            "num_train_annotations": sum(len(s["entities"]) for s in ctx.split.train),
            "num_eval_annotations": sum(len(s["entities"]) for s in ctx.split.eval),
            "batch_metrics": batch_metrics,
            "iteration_metrics": iteration_metrics,
            "rules_snapshot": rules_snapshot,
        }

        return replace(
            ctx,
            rules=rules,
            batch_metrics=batch_metrics,
            iteration_metrics=iteration_metrics,
            best_f1=best["f1"],
            best_rules=best["rules"],
            best_batch_idx=best["batch_idx"],
            t_learn=ctx.t_learn + new_t_learn,
            history=ctx.history + [history_entry],
        )


class RefinementStep(Step):
    def __init__(self, *, max_iterations, use_feedback=False):
        self.max_iterations = max_iterations
        self.use_feedback = use_feedback

    def run(self, ctx: StepContext) -> StepContext:
        rules_to_refine = ctx.best_rules or ctx.rules
        print(
            f"\nRefining best {len(rules_to_refine)} rules"
            f" ({len(ctx.split.eval)} eval examples,"
            f" max {self.max_iterations} iterations)..."
        )
        iteration_metrics = []
        on_iteration = make_oniteration_callback(iteration_metrics)
        dataset = ctx.dev_dataset if self.use_feedback else ctx.eval_dataset
        rules, refine_eval = ctx.chef.learner.evaluate_and_refine(
            rules_to_refine,
            dataset,
            max_iterations=self.max_iterations,
            coordinator=ctx.chef.coordinator,
            iteration_callback=on_iteration,
        )
        if refine_eval:
            print(f"  Eval micro F1: {refine_eval.micro_f1:.1%}")

        history_entry = {
            "phase": "refine",
            "dataset": ctx.split.name,
            "num_rules": len(rules),
            "micro_f1": refine_eval.micro_f1 if refine_eval else None,
            "micro_precision": refine_eval.micro_precision if refine_eval else None,
            "micro_recall": refine_eval.micro_recall if refine_eval else None,
            "iteration_metrics": iteration_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        return replace(ctx, rules=rules, history=ctx.history + [history_entry])


class FeedbackStep(Step):
    def __init__(self, *, feedback_path: str):
        self.feedback_path = feedback_path

    def run(self, ctx: StepContext) -> StepContext:
        items = json.loads(Path(self.feedback_path).read_text())
        load_human_feedback(
            self.feedback_path,
            eval_dataset=ctx.dev_dataset,
            learner=ctx.chef,
            rules=ctx.rules,
        )
        learner = ctx.chef.learner
        eval_result = learner._evaluate_rules(ctx.rules, ctx.dev_dataset)
        guidance = "\n".join(ctx.dev_dataset.feedback)
        patch, deleted_names = learner.synthesize_patch_ruleset(
            ctx.rules,
            eval_result.failures,
            dataset=ctx.dev_dataset,
            guidance=guidance,
            class_metrics=eval_result.per_class,
            fp_examples=eval_result.fp_examples,
        )
        rules = learner._merge_patch(ctx.rules, patch, deleted_names)

        history_entry = {
            "phase": "feedback",
            "dataset": ctx.split.name,
            "feedback_path": str(self.feedback_path),
            "num_items": len(items),
            "task_level": sum(1 for f in items if f.get("level", "task") == "task"),
            "rule_level": sum(1 for f in items if f.get("level") == "rule"),
            "timestamp": datetime.now().isoformat(),
        }
        return replace(ctx, rules=rules, best_rules=rules, history=ctx.history + [history_entry])


class EvaluationStep(Step):
    def run(self, ctx: StepContext) -> StepContext:
        eval_results, t_eval = evaluate_test(ctx.dev_dataset, ctx.rules, ctx.chef)
        return replace(ctx, eval_results=eval_results, t_eval=t_eval)


def build_context(
    args,
    split: DataSplit,
    storage_dir: str,
    checkpoint_path: Path | None = None,
    logger=None,
    rules: list | None = None,
) -> StepContext:
    chef = build_chef(args, split, storage_dir, logger=logger)
    eval_dataset = make_dataset(f"{split.name}_eval", split.eval, chef.task)
    dev_dataset = make_dataset(f"{split.name}_dev", split.dev, chef.task)
    return StepContext(
        rules=rules or [],
        split=split,
        chef=chef,
        eval_dataset=eval_dataset,
        dev_dataset=dev_dataset,
        checkpoint_path=checkpoint_path,
    )

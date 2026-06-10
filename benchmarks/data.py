import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ner_datasets import load_ner_dataset_from_conll
from ner_datasets.sampling import sample_few_shot
from ner_datasets.utils import (
    label_distribution_sent,
    print_distribution,
)
from rulechef.core import Correction, Dataset, Example, Feedback, Rule, RuleFormat


@dataclass
class DataSplit:
    """Sampled, train/eval/dev-split data ror eady for a learning phase."""

    name: str
    train: list
    eval: list
    dev: list
    counter_examples: list
    selected_classes: set
    n_train_docs: int
    n_eval_docs: int
    n_test_docs: int


@dataclass
class BenchmarkRun:
    args: Any
    train_data: list
    eval_data: list
    test_data: list
    train_size: int
    eval_size: int
    test_size: int
    train_annotations: int
    eval_annotations: int
    test_annotations: int
    iteration_metrics: list
    batch_test_metrics: list
    eval_results: Any
    t_learn: float
    t_eval: float
    rules: list
    selected_classes: set[str] | list[str]
    metadata: dict = field(default_factory=dict)


def prepare_split(
    args,
    *,
    name: str,
    train_dir: str,
    test_dir: str | None = None,
    classes: str | None = None,
    fallback_dev: list | None = None,
) -> DataSplit:
    """Load and sample one dataset into a DataSplit ready for training."""
    print(f"\nLoading {name} dataset...")
    train_all = load_ner_dataset_from_conll(train_dir)

    dev_all = load_ner_dataset_from_conll(test_dir) if test_dir else None

    class_list = [c.strip() for c in classes.split(",")] if classes else None

    (
        train,
        eval_,
        counter_examples,
        selected_classes,
        n_train_docs,
        n_eval_docs,
    ) = sample_few_shot(
        train_data=train_all.samples,
        seed=args.seed,
        num_classes=args.num_classes,
        classes=class_list,
        pool_size=args.pool_size,
        train_ratio=args.train_ratio,
    )
    if dev_all:
        dev = [
            {
                "doc_id": s.doc_id,
                "sent_id": sent.sent_id,
                "text": sent.text,
                "entities": sent.labels,
            }
            for s in dev_all.samples
            for sent in s.sentences
        ]
        n_test_docs = len(dev_all.samples)
        dev_label = f"{n_test_docs} source docs"
    elif fallback_dev is not None:
        dev = fallback_dev
        n_test_docs = len(dev)
        dev_label = "reusing phase 1 dev"
    else:
        raise ValueError(f"prepare_split({name!r}): provide test_dir or fallback_dev")

    print(f"{'─' * 70}")
    print(f"Source docs — train: {len(train_all.samples)}, dev: {dev_label}")
    print(f"Sampled     — train docs: {n_train_docs}, eval docs: {n_eval_docs}")
    print(f"Sentences   — train: {len(train)}, eval: {len(eval_)}, dev: {len(dev)}")
    print_distribution(train, "TRAIN", fn=label_distribution_sent)
    print_distribution(eval_, "EVAL", fn=label_distribution_sent)
    print(f"Classes     — {len(selected_classes)}: {', '.join(sorted(selected_classes))}")
    print(f"{'─' * 70}")

    return DataSplit(
        name=name,
        train=train,
        eval=eval_,
        dev=dev,
        counter_examples=counter_examples,
        selected_classes=selected_classes,
        n_train_docs=n_train_docs,
        n_eval_docs=n_eval_docs,
        n_test_docs=n_test_docs,
    )


def load_human_feedback(feedback_path, eval_dataset, learner, rules=None):
    def _norm(s):
        return s.replace("\u2011", "-").replace("\u2010", "-")

    feedback_items = json.loads(Path(feedback_path).read_text())
    print(f"  Loading {len(feedback_items)} human feedback items")
    for fb in feedback_items:
        level = fb.get("level", "task")
        if level == "correction":
            correction = Correction(
                id=str(uuid.uuid4())[:8],
                input=fb["input"],
                model_output=fb.get("model_output", {}),
                expected_output=fb["expected_output"],
                feedback=fb.get("text"),
            )
            eval_dataset.corrections.append(correction)
            print(f"✓ Added correction to eval dataset")
            continue
        text = fb["text"]
        target_id = ""
        if level == "rule" and rules is not None:
            rule_name = fb.get("rule_name", "")
            rule_id = fb.get("rule_id", "")
            matched = next((r for r in rules if r.id == rule_id), None)
            if matched:
                target_id = matched.id
            else:
                print(f"Rule not found: {rule_name} — treating as task-level")
                level = "task"
        eval_dataset.structured_feedback.append(
            Feedback(
                id=str(uuid.uuid4())[:8],
                text=text,
                level=level,
                target_id=target_id,
            )
        )
        learner.add_feedback(text, level, target_id)

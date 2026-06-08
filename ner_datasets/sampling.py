import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rulechef.core import Correction, Dataset, Example, Feedback, Rule, RuleFormat
from rulechef.training_logger import TrainingDataLogger

from ner_datasets import load_ner_dataset_from_conll

# ============================================================================
# Sampling
# ============================================================================


def _doc_sentences(doc, classes):
    """Yield {"text", "entities"} for each sentence, entities filtered to `classes`."""
    for sent in doc.sentences:
        entities = sorted(
            [l for l in sent.labels if l["type"] in classes],
            key=lambda x: x["start"],
        )
        yield {"text": sent.text, "entities": entities}


def _resolve_classes(train_data, classes, num_classes, rng):
    """Pick the target label set: caller-supplied, or a random sample of all labels found."""
    if classes:
        return set(classes)
    all_labels = sorted(
        {l["type"] for doc in train_data for sent in doc.sentences for l in sent.labels}
    )
    if num_classes and num_classes < len(all_labels):
        rng.shuffle(all_labels)
        return set(all_labels[:num_classes])
    return set(all_labels)


def _partition_by_entities(train_data, classes):
    """Split docs into those with >=1 selected-class entity, and those with none."""
    pos_docs, neg_docs = [], []
    for doc in train_data:
        sents = list(_doc_sentences(doc, classes))
        (pos_docs if any(s["entities"] for s in sents) else neg_docs).append(sents)
    return pos_docs, neg_docs


def sample_few_shot(
    train_data,
    seed: int = 42,
    num_classes: int | None = None,
    classes: list[str] | set[str] | None = None,
    pool_size: int | None = None,
    train_ratio: float = 0.7,
    shuffle: bool = True,
):
    """Sample a few-shot train/eval split from a list of documents.

    Shuffling and the train/eval split happen at document level, so sentences
    from the same document are never split across sets.

    Args:
        train_data: List of NERSample (document level).
        seed: Random seed for shuffling and class selection.
        num_classes: If `classes` isn't given, sample this many labels at random.
        classes: Explicit set of entity classes to sample for.
        pool_size: Cap the positive-document pool to this many docs (post-shuffle).
        train_ratio: Fraction of the pooled positive docs assigned to train.
        shuffle: Whether to shuffle documents before pooling/splitting.

    Returns:
        (train_examples, eval_examples, counter_examples, selected_classes,
        n_train_docs, n_eval_docs)
    """

    rng = random.Random(seed)
    classes = _resolve_classes(train_data, classes, num_classes, rng)
    pos_docs, neg_docs = _partition_by_entities(train_data, classes)

    if shuffle:
        rng.shuffle(pos_docs)
        rng.shuffle(neg_docs)

    pool = pos_docs[:pool_size] if pool_size else pos_docs
    split_idx = int(len(pool) * train_ratio)

    train_examples = [s for doc in pool[:split_idx] for s in doc if s["entities"]]
    eval_examples = [s for doc in pool[split_idx:] for s in doc if s["entities"]]
    counter_examples = [s for doc in neg_docs + pool for s in doc if not s["entities"]]

    return (
        train_examples,
        eval_examples,
        counter_examples,
        classes,
        split_idx,
        len(pool) - split_idx,
    )


# ============================================================================
# Split assembly
# ============================================================================


@dataclass
class DataSplit:
    """Sampled, train/eval/dev-split data ready for a learning phase."""

    name: str
    train: list
    eval: list
    dev: list
    counter_examples: list
    selected_classes: set
    n_train_docs: int
    n_eval_docs: int
    n_test_docs: int


def build_data_split(
    *,
    name: str,
    train_dir: str,
    test_dir: str,
    classes: str | None = None,
    fallback_dev: list | None = None,
) -> DataSplit:
    """Load and sample one dataset into a DataSplit ready for training."""
    print(f"\nLoading {name} dataset...")
    train_all = load_ner_dataset_from_conll(train_dir)
    dev_all = load_ner_dataset_from_conll(test_dir)
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
        elif fallback_dev is not None:
            dev = fallback_dev
            n_test_docs = len(dev)
        else:
          raise ValueError(f"build_data_split({name!r}): provide test_dir or fallback_dev")
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
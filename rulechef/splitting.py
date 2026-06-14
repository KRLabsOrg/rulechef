"""Train/dev splitting for the refinement loop.

The refinement loop historically evaluated and accepted patches on the same
data it learned from, so accepted rules could memorize the training set.
split_dataset() carves out a stratified held-out dev set: patches are
synthesized from train failures but accepted or rejected on dev metrics.
"""

import random
from collections import defaultdict

from rulechef.core import Dataset, Example, TaskType


def _class_signature(example: Example, task_type: TaskType) -> str:
    """Bucket key used for stratification.

    CLASSIFICATION stratifies by label, NER by the set of entity types in
    the example. Other task types fall into a single bucket (random split).
    """
    output = example.expected_output or {}
    if task_type == TaskType.CLASSIFICATION:
        return str(output.get("label", ""))
    if task_type == TaskType.NER:
        entities = output.get("entities") or []
        types = sorted({e.get("type", "") for e in entities if isinstance(e, dict)})
        return "|".join(types) if types else "_no_entities"
    return "_all"


def split_dataset(
    dataset: Dataset,
    holdout_fraction: float = 0.2,
    seed: int = 42,
    min_dev_size: int = 5,
) -> tuple[Dataset, Dataset | None]:
    """Split a dataset into train and held-out dev portions.

    Examples are split stratified by class signature. Corrections always
    stay in train: they are explicit user fixes that must drive patching,
    and holding them out would hide the highest-value signal from the
    learner. Feedback and existing rules are shared with the train split.

    Args:
        dataset: Source dataset (not mutated).
        holdout_fraction: Fraction of examples to hold out (0 < f < 1).
        seed: Random seed for the shuffle within each stratum.
        min_dev_size: If the resulting dev set would be smaller than this,
            no split is performed and (dataset, None) is returned.

    Returns:
        Tuple of (train_dataset, dev_dataset). dev_dataset is None when the
        dataset is too small to split safely.
    """
    if not 0 < holdout_fraction < 1:
        return dataset, None

    by_signature: dict[str, list[Example]] = defaultdict(list)
    for ex in dataset.examples:
        by_signature[_class_signature(ex, dataset.task.type)].append(ex)

    rng = random.Random(seed)
    train_examples: list[Example] = []
    dev_examples: list[Example] = []

    for signature in sorted(by_signature):
        examples = list(by_signature[signature])
        rng.shuffle(examples)
        # Hold out a proportional share, but never the entire stratum
        n_dev = min(round(len(examples) * holdout_fraction), len(examples) - 1)
        dev_examples.extend(examples[: max(n_dev, 0)])
        train_examples.extend(examples[max(n_dev, 0) :])

    if len(dev_examples) < min_dev_size:
        return dataset, None

    train = Dataset(
        name=f"{dataset.name}_train",
        task=dataset.task,
        description=dataset.description,
        examples=train_examples,
        corrections=dataset.corrections,
        feedback=dataset.feedback,
        structured_feedback=dataset.structured_feedback,
        rules=dataset.rules,
    )
    dev = Dataset(
        name=f"{dataset.name}_dev",
        task=dataset.task,
        description=dataset.description,
        examples=dev_examples,
    )
    return train, dev

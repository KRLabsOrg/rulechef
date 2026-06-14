"""Tests for rulechef.splitting — stratified train/dev splits."""

from rulechef.core import Correction, Dataset, Example, Task, TaskType
from rulechef.splitting import split_dataset


def _classification_dataset(n_per_class=20, classes=("a", "b", "c")):
    task = Task(
        name="clf",
        description="classify",
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
    )
    dataset = Dataset(name="clf-data", task=task)
    for label in classes:
        for i in range(n_per_class):
            dataset.examples.append(
                Example(
                    id=f"{label}{i}",
                    input={"text": f"example {label} {i}"},
                    expected_output={"label": label},
                    source="test",
                )
            )
    return dataset


class TestSplitDataset:
    def test_zero_fraction_returns_original(self):
        dataset = _classification_dataset()
        train, dev = split_dataset(dataset, holdout_fraction=0.0)
        assert train is dataset
        assert dev is None

    def test_split_sizes(self):
        dataset = _classification_dataset(n_per_class=20)
        train, dev = split_dataset(dataset, holdout_fraction=0.2)
        assert dev is not None
        assert len(dev.examples) == 12  # 4 per class
        assert len(train.examples) == 48
        assert len(train.examples) + len(dev.examples) == 60

    def test_stratified_by_label(self):
        dataset = _classification_dataset(n_per_class=20)
        _, dev = split_dataset(dataset, holdout_fraction=0.2)
        labels = [e.expected_output["label"] for e in dev.examples]
        assert sorted(set(labels)) == ["a", "b", "c"]
        assert labels.count("a") == labels.count("b") == labels.count("c")

    def test_no_overlap(self):
        dataset = _classification_dataset()
        train, dev = split_dataset(dataset, holdout_fraction=0.2)
        train_ids = {e.id for e in train.examples}
        dev_ids = {e.id for e in dev.examples}
        assert not train_ids & dev_ids

    def test_corrections_stay_in_train(self):
        dataset = _classification_dataset()
        dataset.corrections.append(
            Correction(
                id="c1",
                input={"text": "fix me"},
                model_output={"label": "a"},
                expected_output={"label": "b"},
            )
        )
        train, dev = split_dataset(dataset, holdout_fraction=0.2)
        assert len(train.corrections) == 1
        assert len(dev.corrections) == 0

    def test_too_small_dataset_skips_split(self):
        dataset = _classification_dataset(n_per_class=2, classes=("a", "b"))
        train, dev = split_dataset(dataset, holdout_fraction=0.2, min_dev_size=5)
        assert train is dataset
        assert dev is None

    def test_never_empties_a_stratum(self):
        dataset = _classification_dataset(n_per_class=2, classes=("a", "b", "c", "d", "e"))
        train, dev = split_dataset(dataset, holdout_fraction=0.9, min_dev_size=1)
        if dev is not None:
            train_labels = {e.expected_output["label"] for e in train.examples}
            assert train_labels == {"a", "b", "c", "d", "e"}

    def test_deterministic_with_seed(self):
        dataset = _classification_dataset()
        _, dev1 = split_dataset(dataset, holdout_fraction=0.2, seed=7)
        _, dev2 = split_dataset(dataset, holdout_fraction=0.2, seed=7)
        assert [e.id for e in dev1.examples] == [e.id for e in dev2.examples]

    def test_ner_stratifies_by_entity_types(self):
        task = Task(
            name="ner",
            description="ner",
            input_schema={"text": "str"},
            output_schema={"entities": "list"},
            type=TaskType.NER,
        )
        dataset = Dataset(name="ner-data", task=task)
        for i in range(20):
            etype = "DATE" if i % 2 == 0 else "MONEY"
            dataset.examples.append(
                Example(
                    id=f"e{i}",
                    input={"text": f"text {i}"},
                    expected_output={"entities": [{"text": "x", "type": etype}]},
                    source="test",
                )
            )
        _, dev = split_dataset(dataset, holdout_fraction=0.3, min_dev_size=2)
        assert dev is not None
        dev_types = {e.expected_output["entities"][0]["type"] for e in dev.examples}
        assert dev_types == {"DATE", "MONEY"}

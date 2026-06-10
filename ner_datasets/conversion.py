import uuid

from rulechef.core import Dataset, Example


def make_dataset(dataset_name, data, task):
    dataset = Dataset(name=dataset_name, task=task)
    for ex in data:
        dataset.examples.append(
            Example(
                id=str(uuid.uuid4())[:8],
                input={
                    "doc_id": ex.get("doc_id", ""),
                    "sent_id": ex.get("sent_id", ""),
                    "text": ex["text"],
                    "sentences": ex.get("sentences", []),
                },
                expected_output={"entities": ex["entities"]},
                source=dataset_name,
            )
        )
    return dataset

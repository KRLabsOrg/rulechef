import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from clear_anonymization.ner_datasets.ner_dataset import NERData, NERDataset, NERSample
from scripts.download_dataset import download_dataset
from torch.utils.data import Dataset


def download_dataset(repository_id: str) -> None:

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)
    dataset_dict = load_dataset(repository_id)

    samples = []
    for split_name, split_dataset in dataset_dict.items():
        for sample in split_dataset:
            samples.append(
                {
                    "tokens": sample["tokens"],
                    "text": " ".join(str(tokens) for tokens in sample["tokens"]),
                    "ner_labels": sample["ner"],
                    "coarse_ner_labels": sample["coarse-ner"],
                    "split": split_name,
                }
            )

    return samples


def tokens_to_substrs(tokens, labels):
    substrs = {}
    current_tokens = []
    current_label = None
    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_tokens:
                substr = " ".join(current_tokens)
                substrs[substr] = current_label
            current_tokens = [token]
            current_label = label[2:]
        elif label.startswith("I-") and current_label == label[2:]:
            current_tokens.append(token)
        else:
            if current_tokens:
                substr = " ".join(current_tokens)
                substrs[substr] = current_label
                current_tokens = []
                current_label = None

    if current_tokens:
        substr = " ".join(current_tokens)
        substrs[substr] = current_label

    return substrs


def create_labels(substrs: dict, sentence: str):
    labels = []
    for sub, label in substrs.items():
        if not sub:
            continue
        match = re.search(re.escape(sub), sentence)
        if match:
            labels.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": sub,
                    "type": label,
                }
            )
    return labels


def create_sample(sample) -> NERSample:
    text = sample["text"]
    split = sample["split"]
    substrs = tokens_to_substrs(sample["tokens"], sample["coarse_ner_labels"])
    labels = create_labels(substrs, sample.get("text", ""))
    return NERSample(text, split, labels)


def main():
    parser = argparse.ArgumentParser(description="Preprocess LER dataset")
    parser.add_argument(
        "--repository_id",
        type=str,
        required=True,
        help="Repository ID on Hugging Face (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path where to save the JSON file",
    )

    args = parser.parse_args()

    data = download_dataset(
        repository_id=args.repository_id,
    )

    ner_data = NERData(samples=[])

    for d in data:
        sample = create_sample(d)

        ner_data.samples.append(sample)

    output_path = Path(args.output_dir)
    output_path.write_text(json.dumps(ner_data.to_json(), indent=4))


if __name__ == "__main__":
    main()

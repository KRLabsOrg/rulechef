import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from clear_anonymization.ner_datasets import NERData, NERSample, NERSentence
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


def create_sample(sample, idx: int) -> NERSample:
    text = sample["text"]
    split = sample["split"]
    labels_bio = sample["coarse_ner_labels"]
    token_dicts = []
    pos = 0
    for i, (tok, label) in enumerate(zip(sample["tokens"], labels_bio)):
        start = text.find(tok, pos)
        end = start + len(tok)
        misc = f"NER={label}|SentStart={start}|SentEnd={end}"
        token_dicts.append({"id": str(i + 1), "text": tok, "misc": misc})
        pos = end
    substrs = tokens_to_substrs(sample["tokens"], labels_bio)
    labels = create_labels(substrs, text)
    sentence = NERSentence(sent_id=str(idx), text=text, tokens=token_dicts)
    return NERSample(doc_id=str(idx), split=split, text=text, sentences=[sentence], labels=labels)

    return data


def recreate_sent_labels_from_tokens(tokens, sent):

    tagged = []
    for tok in tokens:
        misc_parts = {
            p.split("=")[0]: p.split("=")[1] for p in tok.get("misc", "").split("|") if "=" in p
        }
        ner = misc_parts.get("NER", "O")
        sent_start = misc_parts.get("SentStart", "_")
        sent_end = misc_parts.get("SentEnd", "_")
        tagged.append((tok["text"], ner, sent_start, sent_end))
    labels = []
    current_span = None
    for _, tag, sent_start, sent_end in tagged:
        if tag.startswith("B-"):
            if current_span:
                labels.append(current_span)
            current_span = {
                "text": sent[int(sent_start) : int(sent_end)],
                "type": tag[2:],
                "start": int(sent_start),
                "end": int(sent_end),
            }
        elif tag.startswith("I-") and current_span:
            current_span["end"] = int(sent_end)
            current_span["text"] = sent[current_span["start"] : int(sent_end)]
        else:
            if current_span:
                labels.append(current_span)
            current_span = None
    if current_span:
        labels.append(current_span)
    return labels


def main():
    parser = argparse.ArgumentParser(description="Preprocess LER dataset")
    parser.add_argument(
        "--repository-id",
        type=str,
        required=True,
        help="Repository ID on Hugging Face (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="germanler",
        help="Name of the dataset to preprocess",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path where to save the JSON file",
    )

    args = parser.parse_args()

    data = download_dataset(
        repository_id=args.repository_id,
    )

    split_data = defaultdict(lambda: NERData(samples=[]))

    for idx, d in enumerate(data):
        split_data[d["split"]].samples.append(create_sample(d, idx))
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for split_name, ner_data in split_data.items():
        (output_path / f"{split_name}.conllu").write_text(ner_data.to_conll())


if __name__ == "__main__":
    main()

import re
from dataclasses import dataclass

from torch.utils.data import Dataset


@dataclass
class NERSample:
    text: str
    split: str
    labels: list

    @classmethod
    def from_json(cls, json_dict: dict) -> "NERSample":
        return cls(
            text=json_dict.get("text"),
            split=json_dict.get("split", "unknown"),
            labels=json_dict.get("labels", []),
        )

    def to_json(self) -> dict:
        return {
            "text": self.text,
            "split": self.split,
            "labels": self.labels,
        }


@dataclass
class NERData:
    samples: list[NERSample]

    @classmethod
    def from_json(cls, json_dict: list[dict]) -> "NERData":
        return cls(
            samples=[NERSample.from_json(sample) for sample in json_dict],
        )

    def to_json(self) -> list[dict]:
        return [sample.to_json() for sample in self.samples]


class NERDataset(Dataset):
    def __init__(self, samples: list[NERSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

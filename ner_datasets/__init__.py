import json
from pathlib import Path

from ner_datasets.dataset import NERData, NERSample, NERSentence

DATASET_CLASS_DEFINITIONS = {
    "germanler": {
        "PERS": "Personen (Familien-, Vor-, Beinamen und Pseudonyme)",
        "LOC": "Ortsnamen und geographische Bezeichnungen (Land, Stadt, Region)",
        "ORG": "Organisationsnamen (Parteien, Vereine, Institutionen, Unternehmen)",
        "NRM": "Rechtsnormen (europäische Normen, Gesetze, Rechtsverordnungen)",
        "RS": "Rechtsprechung (Zitate von Gerichtsentscheidungen, keine Personennamen)",
        "LIT": "Rechtsliteratur (Fachliteratur, Gesetzgebungsmaterialien)",
        "REG": "Einzelfallregelungen (Vorschriften, Verträge)",
    }
}


def get_dataset_class_definitions(dataset: str) -> dict[str, str]:
    if dataset not in DATASET_CLASS_DEFINITIONS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DATASET_CLASS_DEFINITIONS[dataset]


def load_ner_dataset_from_json(
    data_path: Path,
) -> NERData:
    return NERData.from_json(json.loads(Path(data_path).read_text()))


def load_ner_dataset_from_conll(
    data_path: Path,
) -> NERData:
    with open(data_path) as f:
        data = NERData.from_conll(f.read())
    return data


__all__ = [
    "DATASET_CLASS_DEFINITIONS",
    "get_dataset_class_definitions",
    "load_ner_dataset_from_json",
    "load_ner_dataset_from_conll",
]

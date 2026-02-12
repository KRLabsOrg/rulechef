import os
import json
import random
import subprocess
from typing import List
import sys
from pathlib import Path
from annotated_text import annotated_text

import io
import streamlit as st
from pydantic import BaseModel, Field

from clear_anonymization.ner_datasets.ner_dataset import NERData
from openai import OpenAI
from rulechef import RuleChef, Task, TaskType
from rulechef.executor import RuleExecutor
from rulechef.core import RuleFormat, Rule
from contextlib import contextmanager


# -----------------------------
# Helpers
# -----------------------------


class Entity(BaseModel):
    text: str = Field(description="The matched text span")
    start: int = Field(description="Start character offset")
    end: int = Field(description="End character offset")
    type: str = Field(description="Entity label")


class NEROutput(BaseModel):
    entities: List[Entity]


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv(
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1",
        # "http://localhost:11434/v1",
        # , "http://localhost:8000/v1"
    )

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    return OpenAI(api_key=api_key, base_url=base_url)


def add_data(chef, samples, negative_samples):
    for sample in samples:
        chef.add_example(
            {"text": sample["text"]},
            {"entities": sample["entities"]},
        )

    for negative_sample in negative_samples:
        chef.add_negative_example(
            {"text": negative_sample["text"]},
            {"entities": negative_sample["entities"]},
        )


def sample_data(samples, allowed_classes, k=10, seed=12):
    random.seed(seed)

    positive_samples = []
    negative_samples = []

    for sample in samples:
        pos, neg = [], []

        for label in sample.labels:

            if label["type"] in allowed_classes:
                pos.append(label)
            else:
                neg.append(label)

        if pos:
            positive_samples.append({"text": sample.text, "entities": pos})
        if neg:
            negative_samples.append({"text": sample.text, "entities": neg})

    negative_samples = random.sample(negative_samples, min(k, len(negative_samples)))
    return positive_samples, negative_samples


def build_task(name, description):
    return Task(
        name=name,
        description=description,
        input_schema={"text": "str"},
        output_schema=NEROutput,
        type=TaskType.NER,
    )


# -----------------------------
# Terminal Output Streaming
# -----------------------------


@contextmanager
def stream_to_streamlit(output_box, title="Execution Log"):
    if "terminal_output" not in st.session_state:
        st.session_state.terminal_output = ""

    class StreamlitWriter(io.TextIOBase):
        def write(self, s):
            if s.strip():
                st.session_state.terminal_output += s
                output_box.text_area(
                    title, value=st.session_state.terminal_output, height=300
                )

        def flush(self):
            pass

    old_stdout = sys.stdout
    sys.stdout = StreamlitWriter()
    try:
        yield
    finally:
        sys.stdout = old_stdout


# -----------------------------
# Highlight Output Text
# -----------------------------

"""
def highlight_entities(text, entities):
    entities = sorted(entities, key=lambda e: e["start"], reverse=True)
    for e in entities:
        span = text[e["start"] : e["end"]]
        text = text[: e["start"]] + f"**[{span}]({e['type']})**" + text[e["end"] :]
    return text
"""


def highlight_entities(text, entities):
    entities = sorted(entities, key=lambda e: e["start"], reverse=True)
    chunks = []
    last_idx = 0
    for e in entities:
        if e["start"] > last_idx:
            chunks.append(text[last_idx : e["start"]])

        entity_text = text[e["start"] : e["end"]]
        entity_label = e["type"]
        chunks.append((entity_text, entity_label))
        last_idx = e["end"]

    if last_idx < len(text):
        chunks.append(text[last_idx:])

    annotated_text(*chunks)

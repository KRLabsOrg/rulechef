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
    api_key = "EMPTY"  # os.getenv("OPENAI_API_KEY") #"EMPTY"
    base_url = "http://a-a100-o-1:8000/v1"  # "http://localhost:8000/v1" # "https://api.openai.com/v1" #http://localhost:8000/v1

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    return OpenAI(api_key=api_key, base_url=base_url)


def add_data(chef, samples):
    for sample in samples:
        chef.add_example(
            {"text": sample["text"]},
            {"entities": sample["entities"]},
        )
    """
    for negative_sample in negative_samples:
        chef.add_negative_example(
            {"text": negative_sample["text"]},
            {"entities": negative_sample["entities"]},
        )
    """


def sample_data(samples, allowed_classes, k=50, seed=123, window_size=100):
    random.seed(seed)

    positive_samples = []
    for sample in samples:
        text = sample.text
        entities = sorted(
            [l for l in sample.labels if l["type"] in allowed_classes],
            key=lambda x: x["start"],
        )

        if not entities:
            continue
        merged_windows = []
        for ent in entities:
            start = max(0, ent["start"] - window_size)
            end = min(len(text), ent["end"] + window_size)
            if merged_windows and start <= merged_windows[-1][1]:
                merged_windows[-1][1] = max(end, merged_windows[-1][1])
                merged_windows[-1][2].append(ent)
            else:
                merged_windows.append([start, end, [ent]])

        for start, end, window_entities in merged_windows:
            snippet = text[start:end]
            adjusted_entities = []
            for e in window_entities:
                adjusted_entities.append(
                    {
                        "text": e["text"],
                        "start": e["start"] - start,
                        "end": e["end"] - start,
                        "type": e["type"],
                    }
                )
            positive_samples.append({"text": snippet, "entities": adjusted_entities})

    return positive_samples


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

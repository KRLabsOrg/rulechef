import os
import json
import random
import subprocess
from typing import List
import sys
from pathlib import Path

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
# Models
# -----------------------------


class Entity(BaseModel):
    text: str = Field(description="The matched text span")
    start: int = Field(description="Start character offset")
    end: int = Field(description="End character offset")
    type: str = Field(description="Entity label")


class NEROutput(BaseModel):
    entities: List[Entity]


# -----------------------------
# Helpers
# -----------------------------


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv(
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1",
        # "http://localhost:8000/v1"
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
            span = {
                "text": label["text"],
                "start": label["start"],
                "end": label["end"],
                "type": label["class"],
            }

            if label["class"] in allowed_classes:
                pos.append(span)
            else:
                neg.append(span)

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
    """
    Redirect stdout to a Streamlit text area.
    Usage:
        with stream_to_streamlit(my_box, "Learning Rules"):
            print("Hello World")  # or any function that prints
    """
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
def highlight_entities(text, entities):
    entities = sorted(entities, key=lambda e: e["start"], reverse=True)
    for e in entities:
        span = text[e["start"] : e["end"]]
        text = text[: e["start"]] + f"**[{span}]({e['type']})**" + text[e["end"] :]
    return text


# -----------------------------
# Streamlit App
# -----------------------------


def main():
    st.set_page_config(page_title="RuleChef", layout="wide")

    # ---- Session state ----
    for key in [
        "task",
        "entity_types",
        "language",
        "data",
        "positive",
        "negative",
        "chef",
        "samples",
        "rules",
        "executor",
        "rules_learned",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None

        st.session_state.terminal_output = ""

    # ---- Header ----
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("RuleChef 👨‍🍳")
        st.markdown(
            "Learn rule-based models from examples, corrections, and LLM interactions."
        )

    with col2:
        st.image(
            "https://github.com/KRLabsOrg/rulechef/blob/main/assets/mascot.png?raw=true",
            width=200,
        )

    st.subheader("Rules Source")

    rules_mode = st.radio(
        "How do you want to get rules?",
        [
            "Learn rules from data",
            "Load existing rules",
        ],
    )

    if "rules_learned" not in st.session_state:
        st.session_state.rules_learned = False

    if rules_mode == "Load existing rules":
        uploaded_rules = st.file_uploader(
            "Upload rules file",
            type=["json"],
        )
        if uploaded_rules:
            rules_data = json.loads(uploaded_rules.read().decode("utf-8"))
            rules = [Rule.from_dict(r) for r in rules_data.get("rules")]

            st.session_state.rules = rules
            st.success(f"{len(rules)} rules loaded")

    if rules_mode == "Learn rules from data":
        st.subheader("Define Task")

        with st.form("task_form"):
            task_name = st.text_input("Task name", "Named Entity Recognition")
            task_description = st.text_area(
                "Task description", "Extract entities from text"
            )
            entity_types = st.multiselect(
                "Entity types", ["ORG", "PER", "LOC"], default=["ORG"]
            )
            language = st.selectbox("Language", ["de", "en"])

            submitted = st.form_submit_button("Set task")

            if submitted:
                st.session_state.task = build_task(task_name, task_description)
                st.session_state.entity_types = entity_types
                st.session_state.language = language
                st.success("Task saved")

    if rules_mode == "Learn rules from data" and st.session_state.task:
        uploaded_file = st.file_uploader("Upload JSON file", type="json")

        if uploaded_file:
            content = uploaded_file.read().decode("utf-8")
            st.session_state.data = NERData.from_json(json.loads(content))
            st.session_state.samples = [
                s for s in st.session_state.data.samples if s.split == "train"
            ]

            (
                st.session_state.positive,
                st.session_state.negative,
            ) = sample_data(
                st.session_state.samples,
                st.session_state.entity_types,
            )

            st.success("JSON file uploaded")

        # ---- Create RuleChef ----
        if (
            st.session_state.task
            and st.session_state.data
            and st.session_state.chef is None
        ):
            st.session_state.chef = RuleChef(
                st.session_state.task,
                get_openai_client(),
                dataset_name="myrules",
                allowed_formats=[RuleFormat.REGEX],
                model="gpt-5-mini-2025-08-07",
                use_spacy_ner=False,
                lang=st.session_state.language,
            )

            output_box = st.empty()

            with stream_to_streamlit(output_box, "Adding data"):
                add_data(
                    st.session_state.chef,
                    st.session_state.positive,
                    st.session_state.negative,
                )

            st.success("RuleChef initialized")

        # ---- Learn rules ----

        if rules_mode == "Learn rules from data" and st.session_state.chef:
            if (
                "rules_learned" not in st.session_state
                or not st.session_state.rules_learned
            ):
                output_box = st.empty()

                with stream_to_streamlit(output_box, "Learning Rules Output"):
                    st.session_state.chef.learn_rules()

            st.session_state.rules_learned = True

    if st.session_state.rules or st.session_state.rules_learned:
        st.subheader("Learned Rules")
        if not st.session_state.rules_learned:
            st.json(st.session_state.rules)
        if st.session_state.rules_learned:
            st.json(st.session_state.chef.get_rules_summary())

        st.subheader("Test Rules on New Text")
        test_text = st.text_area(
            "Input text",
            height=150,
            placeholder="Paste or type text here…",
        )

        if not st.session_state.executor:
            st.session_state.executor = RuleExecutor()

        apply_clicked = st.button("Apply Rules")
        if apply_clicked and test_text.strip():
            if not st.session_state.rules_learned:
                with st.spinner("Applying rules..."):
                    result = st.session_state.executor.apply_rules(
                        rules=st.session_state.rules,
                        input_data={"text": test_text},
                        task_type=TaskType.NER,
                    )
            if st.session_state.rules_learned:
                with st.spinner("Applying rules..."):
                    result = st.session_state.chef.extract(
                        input_data={"text": test_text},
                    )

            entities = result.get("entities", [])
            if entities:
                st.subheader("Rule Output")
                st.markdown(highlight_entities(test_text, entities))
                st.json(result)
            else:
                st.info("No entities found.")


if __name__ == "__main__":
    main()

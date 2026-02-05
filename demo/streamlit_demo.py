import os
import json
import random

import streamlit as st
from clear_anonymization.ner_datasets.ner_dataset import NERData
from pydantic import BaseModel, Field
from typing import List

from openai import OpenAI
from rulechef import RuleChef, Task, TaskType
from rulechef.core import RuleFormat



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
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")

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
# Streamlit App
# -----------------------------


def main():
    st.set_page_config(page_title="RuleChef")

    for key in [
        "task",
        "entity_types",
        "language",
        "data",
        "positive",
        "negative",
        "chef",
        "samples"

    ]:
        if key not in st.session_state:
            st.session_state[key] = None

    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("Rulechef 👨‍🍳")
        st.markdown(
            "Learn rule-based models from examples, corrections, and LLM interactions."
        )

    with col2:
        st.image(
            "https://github.com/KRLabsOrg/rulechef/blob/main/assets/mascot.png?raw=true",
            width=200,
        )

    # ---- Task definition ----
    st.subheader("Define Task")

    with st.form("task_form"):
        task_name = st.text_input(
            "Task name",
            value="Named Entity Recognition",
        )

        task_description = st.text_area(
            "Task description",
            value="Extract entities from text",
        )

        entity_types = st.multiselect(
            "Entity types to extract",
            options=["ORG", "PER", "LOC"],
            default=["ORG"],
        )

        language = st.selectbox("Language", ["de", "en"])

        submitted = st.form_submit_button("Set task")

        if submitted:
            st.session_state.task = build_task(task_name, task_description)
            st.session_state.entity_types = entity_types
            st.session_state.language = language
            st.success("Task saved")

    # ---- Upload data ----
    if st.session_state.task:
        st.subheader("Upload JSON file")
        uploaded_file = st.file_uploader("", type="json")

        if uploaded_file:
            content = uploaded_file.read().decode("utf-8")
            st.session_state.data = NERData.from_json(json.loads(content))
            st.session_state.samples = [s for s in st.session_state.data.samples if s.split == "train"]

            st.session_state.positive, st.session_state.negative = sample_data(
                st.session_state.samples,
                st.session_state.entity_types,
            )

            st.success("JSON file uploaded")

    # ---- Create chef + add data ----
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
            model="google/gemma-3-27b-it",
            use_spacy_ner=False,
            lang=st.session_state.language,
        )

        add_data(
            st.session_state.chef,
            st.session_state.positive,
            st.session_state.negative,
        )

        st.success("RuleChef initialized")

    # ---- Learn rules ----
    if st.session_state.chef and st.button("Learn Rules"):
        rules = st.session_state.chef.learn_rules()
        st.success(f"{len(rules)} rules learned")
        st.write(rules)


if __name__ == "__main__":
    main()

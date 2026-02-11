import streamlit as st
import json
from clear_anonymization.ner_datasets.ner_dataset import NERData
from utils import sample_data, build_task


col1, col2 = st.columns([2, 1])

with col1:
    st.title("RuleChef 👨‍🍳")
    st.markdown(
        "Learn rule-based models from examples, corrections, and LLM interactions."
    )

with col2:
    st.image(
        "https://github.com/KRLabsOrg/rulechef/blob/main/assets/mascot.png?raw=true",
        width=100,
    )

st.title("⛁ Add Data")
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
# ---- Entity Labels ----
with st.container(border=True):
    st.subheader("Entity Labels")

    with st.form("task_form"):
        entity_types = st.multiselect(
            "Entity Labels",
            ["ORG", "PER", "LOC"],
            default=st.session_state.entity_types or ["ORG"],
        )

        submitted = st.form_submit_button("Save")

        if submitted:
            st.session_state.entity_types = entity_types
            st.session_state.language = "de"
            st.session_state.task = build_task(
                "Named Entity Recognition", "Extract entities from text"
            )
            st.success("Saved!")

# ---- Upload Data ----
with st.container(border=True):
    st.subheader("Upload JSON Data")

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

        st.success("Training data loaded.")

with st.container(border=True):
    st.subheader("Training Data")
    if st.session_state.samples:
        for sample in st.session_state.samples:
            st.write(sample.text)
            for label in sample.labels:
                st.write(label)
    else:
        st.write("No examples yet!")

if st.session_state.samples:
    with st.container(border=True):
        st.write("You have training examples.")
        st.page_link("pages/🎓_Learn_Rules.py", label="Next: 🎓 Learn Rules")


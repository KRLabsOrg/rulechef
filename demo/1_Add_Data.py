import streamlit as st
import json
from clear_anonymization.ner_datasets.ner_dataset import NERData
from utils import sample_data, build_task, highlight_entities
from annotated_text import annotated_text


st.set_page_config(page_title="RuleChef", layout="wide")

st.markdown(
    """
    <div style="display: flex; align-items: flex-start; gap: 20px;">
        <img src="https://github.com/KRLabsOrg/rulechef/blob/main/assets/mascot.png?raw=true" width="75" style="flex-shrink:0;">
        <div style="display: flex; flex-direction: column;">
            <h1 style="margin: 0;">RuleChef</h1>
            <p style="margin: 0; color: grey; font-size:16px;">
                Learn rule-based models from examples, corrections, and LLM interactions.
            </p>
            <h2 style="margin-top: 10px; font-weight: normal; color: #555;">⛁ Add Data</h2>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---- Session state ----
for key in [
    "task",
    "entity_types",
    "language",
    "data",
    "examples",
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
            ["person", "location", "organisation"],
            default=st.session_state.entity_types or ["organisation"],
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
    
        st.session_state.examples = sample_data(
            st.session_state.data.samples,
            st.session_state.entity_types,
        )
        print(st.session_state.examples[:10]  )

        st.success("Training data loaded.")


with st.container(border=True):
    st.subheader("Training Data")
    
    if st.session_state.data:
        print(st.session_state.examples)
        for sample in st.session_state.examples[:3]:
            print(sample)
            highlight_entities(sample["text"], sample["entities"])
            # for label in sample.labels:
            # st.write(label)
    else:
        st.write("No examples yet!")

if st.session_state.examples:
    with st.container(border=True):
        st.write("You have training examples.")
        with st.container(border=True, width="content", height="content", gap="small"):
            st.page_link("pages/2_Learn_Rules.py", label="Next: 🎓 Learn Rules")

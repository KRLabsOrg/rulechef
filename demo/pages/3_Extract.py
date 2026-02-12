import streamlit as st
import json
from rulechef import RuleChef, TaskType
from rulechef.core import RuleFormat, Rule
from rulechef.executor import RuleExecutor
from utils import get_openai_client, add_data, stream_to_streamlit, highlight_entities

st.set_page_config(page_title="RuleChef", layout="wide")
from annotated_text import annotated_text


st.markdown(
    """
    <div style="display: flex; align-items: flex-start; gap: 20px;">
        <img src="https://github.com/KRLabsOrg/rulechef/blob/main/assets/mascot.png?raw=true" width="75" style="flex-shrink:0;">
        <div style="display: flex; flex-direction: column;">
            <h1 style="margin: 0;">RuleChef</h1>
            <p style="margin: 0; color: grey; font-size:16px;">
                Learn rule-based models from examples, corrections, and LLM interactions.
            </p>
            <h2 style="margin-top: 10px; font-weight: normal; color: #555;">🔍︎ Extract</h2>
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
    "positive",
    "negative",
    "chef",
    "samples",
    "rules",
    "executor",
    "rules_learned",
    "result",
]:
    if key not in st.session_state:
        st.session_state[key] = None

    st.session_state.terminal_output = ""


if st.session_state.chef:
    if st.session_state.chef.dataset.rules:
        rules_learned = True
    else:
        rules_learned = False
else:
    rules_learned = False


if st.session_state.rules or rules_learned:
    with st.container(border=True):
        st.subheader("Input Text")
        test_text = st.text_area(
            "Input text",
            height=150,
            placeholder="Paste or type text here…",
        )
        apply_clicked = st.button("🔍︎  Extract")

        if apply_clicked and test_text.strip():
            if not rules_learned:
                st.session_state.executor = RuleExecutor()
                st.session_state.result = st.session_state.executor.apply_rules(
                    rules=st.session_state.rules,
                    input_data={"text": test_text},
                    task_type=TaskType.NER,
                )

            if rules_learned:
                st.session_state.result = st.session_state.chef.extract(
                    input_data={"text": test_text},
                )

    if st.session_state.result:
        with st.container(border=True):
            entities = st.session_state.result.get("entities", [])
            st.write(entities)
            if entities:
                st.subheader("Rule Output")
                highlight_entities(test_text, entities)
    else:
        st.info("No entities found.")


else:
    with st.container(border=True):
        st.markdown(
            "<h4 style='text-align: left; font-size:5px, font-family: Roboto Mono; color: black;'>No rules learned yet. Extraction works best after you train rules from examples</h4>",
            unsafe_allow_html=True,
        )
        with st.container(border=True, width="content", height="content", gap="small"):
            st.page_link("pages/2_Learn_Rules.py", label="Go to Step 2: Learn Rules!")

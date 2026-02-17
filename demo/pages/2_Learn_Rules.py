import streamlit as st
import json
from rulechef import RuleChef, TaskType
from rulechef.core import RuleFormat, Rule
from rulechef.executor import RuleExecutor
from utils import get_openai_client, add_data, stream_to_streamlit
import pandas as pd

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
            <h2 style="margin-top: 10px; font-weight: normal; color: #555;">🎓 Learn Rules</h2>
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
]:
    if key not in st.session_state:
        st.session_state[key] = None

    st.session_state.terminal_output = ""





has_data = st.session_state.task and st.session_state.data
has_rules = bool(st.session_state.rules)

if not has_data:
    with st.container(border=True):
        st.markdown(
        "<h5 style='text-align: left; font-size:2px, font-family: Roboto Mono; color: black;'>Add data or upload existing rule file.</h5>",
        unsafe_allow_html=True,
    )
        with st.container(border=True, width="content", height="content", gap="small"):
            st.page_link("./1_Add_Data.py", label="Go to Step 1: Add Data!")
        uploaded_rules = st.file_uploader("Upload rules file (JSON)", type=["json"])
        if uploaded_rules:
            rules_data = json.loads(uploaded_rules.read().decode("utf-8"))
            rules = [Rule.from_dict(r) for r in rules_data.get("rules", [])]
            st.session_state.rules = rules
            st.success(f"{len(rules)} rules loaded")



if has_data:
    with st.container(border=True):
        if st.session_state.chef is None:
            st.session_state.chef = RuleChef(
                st.session_state.task,
                get_openai_client(),
                dataset_name="myrules",
                allowed_formats=[RuleFormat.REGEX],
                model= "openai/gpt-oss-120b", #"gpt-5-mini-2025-08-07", #
                use_spacy_ner=False,
                lang=st.session_state.language,
                use_grex=True,
            )

            output_box = st.empty()
            add_data(
                st.session_state.chef,
                st.session_state.examples,
               # st.session_state.negative,
            )
            st.success("RuleChef initialized.")

        if not st.session_state.rules_learned:
            if st.button("▶️ Start Learning"):
                output_box = st.empty()
                with stream_to_streamlit(output_box, "Learning Rules"):
                    st.session_state.chef.learn_rules(incremental_only=True)
                st.session_state.rules_learned = True
                st.success("Rules learned!")


with st.container(border=True):
    rules_to_show = st.session_state.rules

    if not rules_to_show and has_data and st.session_state.rules_learned:
        rules_to_show = getattr(st.session_state.chef.dataset, "rules", [])
    
    if rules_to_show:
        rules_table = []
        st.subheader("Learned Rules")
    
        for i, rule in enumerate(rules_to_show, 1):
            with st.expander(f"Rule #{i} {rule.description}"):
                st.code(f"Format: {rule.format}")
                st.code(f"Confidence: {rule.confidence}")
                st.code(f"Pattern: {rule.content}")
        
    else:
        st.info("No rules available yet.")



if rules_to_show:
    with st.container(border=True):
        with st.container(border=True):
            st.write("Now that you have rules, you can test them on new examples!")
            with st.container(
                border=True, width="content", height="content", gap="small"
            ):
                st.page_link("pages/3_Extract.py", label="Next: 🔍︎ Extract")

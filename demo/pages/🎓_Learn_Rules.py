import streamlit as st
import json
from rulechef import RuleChef, TaskType
from rulechef.core import RuleFormat, Rule
from rulechef.executor import RuleExecutor
from utils import get_openai_client, add_data, stream_to_streamlit

st.title("🎓 Learn Rules")

st.markdown("Add data or upload existing rule file.")
uploaded_rules = st.file_uploader("Upload rules file (JSON)", type=["json"])
if uploaded_rules:
    rules_data = json.loads(uploaded_rules.read().decode("utf-8"))
    rules = [Rule.from_dict(r) for r in rules_data.get("rules", [])]
    st.session_state.rules = rules
    st.success(f"{len(rules)} rules loaded")

has_data = st.session_state.task and st.session_state.data
has_rules = bool(st.session_state.rules)

if has_data:
    if st.session_state.chef is None:
        st.session_state.chef = RuleChef(
            st.session_state.task,
            get_openai_client(),
            dataset_name="myrules",
            allowed_formats=[RuleFormat.REGEX],
            model="mistral",
            use_spacy_ner=False,
            lang=st.session_state.language,
        )

        output_box = st.empty()
        add_data(
            st.session_state.chef,
            st.session_state.positive,
            st.session_state.negative,
        )
        st.success("RuleChef initialized.")

    if not st.session_state.rules_learned:
        if st.button("▶️ Start Learning"):
            output_box = st.empty()
            with stream_to_streamlit(output_box, "Learning Rules"):
                st.session_state.chef.learn_rules()
            st.session_state.rules_learned = True
            st.success("Rules learned!")

with st.container(border=True):
    rules_to_show = st.session_state.rules

    if not rules_to_show and has_data and st.session_state.rules_learned:
        rules_to_show = getattr(st.session_state.chef.dataset, "rules", [])

    if rules_to_show:
        st.subheader("Learned Rules")
        for i, rule in enumerate(rules_to_show, 1):
            st.markdown(f"**Rule {i}**")
            st.write(rule.description)
            st.write(rule.content)
    else:
        st.info(
            "No rules available. Add data to train or upload a rules file to view rules."
        )

if rules_to_show:
    with st.container(border=True):
        with st.container(border=True):
            st.write("Now that you have rules, you can test them on new examples!")
            st.page_link("pages/🔍︎_Extract.py", label="Next: 🔍︎ Extract")
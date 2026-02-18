import streamlit as st
import json
from rulechef import RuleChef, TaskType
from rulechef.core import RuleFormat, Rule
from rulechef.executor import RuleExecutor
from utils import get_openai_client, add_data, stream_to_streamlit
import pandas as pd
from rulechef.matching import evaluate_rules_individually

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
    "examples",
    "negative",
    "chef",
    "samples",
    "rules",
    "executor",
    "rules_learned",
    "active_rules",
]:
    if key not in st.session_state:
        st.session_state[key] = None
st.session_state.terminal_output = ""

# --- Active rules state ---
if "active_rules" not in st.session_state:
    st.session_state.active_rules = (
        st.session_state.rules.copy() if st.session_state.rules else []
    )


has_data = st.session_state.task and st.session_state.data
has_rules = bool(st.session_state.rules)

if not has_data:
    with st.container(border=True):
        st.markdown(
            "<h5 style='text-align: left; font-size:2px, font-family: Roboto Mono; color: black;'>Add data</h5>",
            unsafe_allow_html=True,
        )
        with st.container(border=True, width="content", height="content", gap="small"):
            st.page_link("./1_Add_Data.py", label="Go to Step 1: Add Data!")


if has_data:
    with st.container(border=True):
        st.markdown(
            "<h5 style='text-align: left; font-size:2px, font-family: Roboto Mono; color: black;'>Upload existing rules</h5>",
            unsafe_allow_html=True,
        )
        uploaded_rules = st.file_uploader("Upload rules file (JSON)", type=["json"])
        if uploaded_rules:
            rules_data = json.loads(uploaded_rules.read().decode("utf-8"))
            rules = [Rule.from_dict(r) for r in rules_data.get("rules", [])]
            st.session_state.rules = rules
            st.session_state.active_rules = rules.copy()
            st.success(f"{len(rules)} rules loaded")

    with st.container(border=True):
        st.markdown(
            "<h5 style='text-align: left; font-size:2px, font-family: Roboto Mono; color: black;'>Learn rules from data</h5>",
            unsafe_allow_html=True,
        )
        if st.session_state.chef is None:
            st.session_state.chef = RuleChef(
                st.session_state.task,
                get_openai_client(),
                dataset_name="myrules",
                allowed_formats=[RuleFormat.REGEX],
                model="openai/gpt-oss-120b",  # "gpt-5-mini-2025-08-07", #
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
                st.session_state.active_rules = (
                    st.session_state.chef.dataset.rules.copy()
                )
                st.success("Rules learned!")


def color_metric(value: float) -> str:
    if value >= 0.9:
        return "#2ecc71"
    elif value >= 0.6:
        return "#f39c12"
    else:
        return "#e74c3c"


def metric_badge(label: str, value, color_fn=True) -> str:
    """
    value: can be float (percentage) or int (count)
    color_fn: if True, color by value; if False, use black
    """
    if color_fn:
        color = color_metric(value)
        background_color = color
        display_value = f"{value * 100:.0f}%"
    else:
        color = "black"
        background_color = "black"
        display_value = str(value)

    return (
        f"<span style='"
        f"background-color:{background_color};"
        f"color:{'white'};"
        f"border:1px solid {color};"
        f"border-radius:4px;"
        f"padding:2px 8px;"
        f"font-weight:600;"
        f"font-size:1.2em;"
        f"margin-left:6px;"
        f"'>{label}: {display_value}</span>"
    )


with st.container(border=True):
    rules_to_show = st.session_state.rules

    if not rules_to_show and has_data and st.session_state.rules_learned:
        rules_to_show = getattr(st.session_state.chef.dataset, "rules", [])

    if not rules_to_show:
        st.info("No rules available yet.")
    else:
        metrics = evaluate_rules_individually(
            st.session_state.examples,
            rules_to_show,
            st.session_state.chef,
            rules_learned=False,
            threshold=0.7,
        )

        st.subheader("Learned Rules")
        for i, rule in enumerate(rules_to_show, 1):
            overall = metrics[rule.name]["overall"]
            p = overall["precision"]
            r = overall["recall"]
            f1 = overall["f1"]
            tp = overall["TP"]
            fp = overall["FP"]

            state_key = f"rule_expanded_{i}"
            if state_key not in st.session_state:
                st.session_state[state_key] = False

            is_active = rule in st.session_state.active_rules

            header_html = (
                f"<div style='"
                f"display:flex; align-items:center; flex-wrap:wrap; gap:6px;"
                f"padding:10px 12px;"
                f"background:#f8f9fa; border-radius:6px;"
                f"border:1px solid #dee2e6; cursor:pointer;"
                f"'>"
                f"<span style='font-weight:600; margin-right:4px;'>Rule #{i}</span>"
                f"<span style='color:#666; font-size:1.5em; flex:1'>{rule.name}</span>"
                f"{metric_badge('TP/FP', f'{tp}/{fp}', False)}"
                f"{metric_badge('F1-Score', f1)}"
                f"{metric_badge('Precision', p)}"
                f"{metric_badge('Recall', r)}"
                f"<span style='color:white; background:{'#2ecc71' if is_active else '#e74c3c'}; border-radius:4px; padding:2px 6px; font-weight:600; margin-left:6px;'>{'Active' if is_active else 'Inactive'}</span>"
                f"</div>"
            )
            st.markdown(header_html, unsafe_allow_html=True)

            toggle_label = "▲ Collapse" if st.session_state[state_key] else "▼ Expand"
            if st.button(toggle_label, key=f"toggle_{i}", use_container_width=True):
                st.session_state[state_key] = not st.session_state[state_key]
                st.rerun()

            if st.button("Set to Active/Inactive", key=f"active_{i}"):
                if is_active:
                    st.session_state.active_rules.remove(rule)
                else:
                    st.session_state.active_rules.append(rule)
                st.rerun()

            if st.session_state[state_key]:
                with st.container(border=True):
                    st.code(f"Description:     {rule.description}")
                    st.code(f"Format:     {rule.format}")
                    st.code(f"Confidence: {rule.confidence}")
                    st.code(f"Pattern:    {rule.content}")

            st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)


if rules_to_show:
    with st.container(border=True):
        with st.container(border=True):
            st.write("Now that you have rules, you can test them on new examples!")
            with st.container(
                border=True, width="content", height="content", gap="small"
            ):
                st.page_link("pages/3_Extract.py", label="Next: 🔍︎ Extract")

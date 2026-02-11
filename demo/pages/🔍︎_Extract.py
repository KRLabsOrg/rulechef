import streamlit as st
import json
from rulechef import RuleChef, TaskType
from rulechef.core import RuleFormat, Rule
from rulechef.executor import RuleExecutor
from utils import get_openai_client, add_data, stream_to_streamlit

st.title("🔍︎ Extract")


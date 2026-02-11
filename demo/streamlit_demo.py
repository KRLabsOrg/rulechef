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
from utils import *


# -----------------------------
# Streamlit App
# -----------------------------


def main():
    pass


if __name__ == "__main__":
    main()

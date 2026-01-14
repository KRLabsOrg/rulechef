# RuleChef 🧑‍🍳

<p align="center">
  <img src="https://github.com/KRLabsOrg/rulechef/blob/main/assets/mascot.png?raw=true" alt="RuleChef Logo" width="400"/>
</p>

Learn rule-based models from examples, corrections, and LLM interactions.

## Installation
```bash
pip install -e .
```

## Quick Start

### CLI
```bash
export OPENAI_API_KEY=your_key_here
rulechef
```

### Python
```python
from openai import OpenAI
from rulechef import RuleChef, Task

# Setup
client = OpenAI(api_key="...")
task = Task(
    name="Q&A",
    description="Extract answer spans from text",
    input_schema={"question": "str", "context": "str"},
    output_schema={"spans": "List[Span]"}
)

chef = RuleChef(task, client)

# Add examples
chef.add_example(
    {"question": "When?", "context": "Built in 1991"},
    {"spans": [{"text": "1991", "start": 9, "end": 13}]}
)

# Learn rules
chef.learn_rules()

# Use
result = chef.extract({"question": "When?", "context": "Released in 1995"})
spans = result.get("spans", [])
print([s["text"] for s in spans])  # ["1995"]
```
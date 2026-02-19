# Installation

## Basic Install

```bash
pip install rulechef
```

This installs the core library with `openai` and `pydantic` as dependencies.

## Extras

RuleChef provides optional extras for additional features:

```bash
# Regex pattern suggestions from examples (recommended)
pip install rulechef[grex]

# spaCy token/dependency matcher patterns
pip install rulechef[spacy]

# LLM-powered coordinator for adaptive learning loops
pip install rulechef[agentic]

# Everything
pip install rulechef[all]
```

### Extra Details

| Extra | What It Adds | When You Need It |
|-------|-------------|-----------------|
| `grex` | [grex](https://github.com/pemistahl/grex) regex inference | Improves regex rule quality by suggesting patterns from examples |
| `spacy` | [spaCy](https://spacy.io/) NLP | Linguistic patterns using POS, dependency, lemma attributes |
| `agentic` | pydantic-ai | LLM-driven coordinator that guides refinement loops |
| `all` | All of the above | Full feature set |

### Development Extras

```bash
# Run tests
pip install rulechef[dev]

# Build documentation
pip install rulechef[docs]

# Run benchmarks
pip install rulechef[benchmark]
```

## Requirements

- Python 3.8+
- An OpenAI-compatible API key (OpenAI, Groq, Together, etc.)

## Verify Installation

```python
from rulechef import RuleChef, Task, TaskType, RuleFormat
print("RuleChef installed successfully")
```

## Using with Other LLM Providers

RuleChef uses the OpenAI client, which works with any OpenAI-compatible API:

```python
from openai import OpenAI

# Groq
client = OpenAI(
    api_key="your-groq-key",
    base_url="https://api.groq.com/openai/v1"
)

# Together AI
client = OpenAI(
    api_key="your-together-key",
    base_url="https://api.together.xyz/v1"
)

chef = RuleChef(task, client, model="your-model-name")
```

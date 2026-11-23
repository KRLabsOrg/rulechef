# Changelog

## Unreleased
- Added opt-in span overlap resolution: `overlap_strategy="longest"` keeps only the longest of overlapping spans (default `None` keeps the existing behaviour). 
- `cross_type_overlaps=True` also resolves overlaps across types

## 0.1.1

- Added training data logger for LLM distillation
- Added benchmark suite (Banking77)
- Improved observer mode

## 0.1.0

- Initial release
- Core: RuleChef, Task, Dataset, Example, Rule
- Task types: NER, Extraction, Classification, Transformation
- Rule formats: Regex, Code, spaCy
- Agentic coordinator with rule pruning
- LLM observation mode with auto-discovery
- CLI interactive shell
- grex integration for regex hints

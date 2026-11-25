# Changelog

## Unreleased
- Added optional `seed` on `RuleChef`, sent with all LLM calls for more reproducible runs
- Rule IDs are derived from rule's information now (name, content, format, output_template, output_key). A rule learned again with the same fields keeps the same ID, and an ID can carry a suffix in case of duplicates.
- Added opt-in span overlap resolution: `overlap_strategy="longest"` keeps only the longest of overlapping spans (default `None` keeps the existing behaviour). 
- `cross_type_overlaps=True` also resolves overlaps across types
- Added `spacy_model` option (default `en_core_web_sm`) so spaCy rules can now use other models.

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

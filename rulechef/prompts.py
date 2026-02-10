"""Prompt templates and builders for rule learning"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Any, Set

from rulechef.core import Rule, RuleFormat, Dataset, Correction, TaskType

try:
    from grex import RegExpBuilder

    _HAS_GREX = True
except ImportError:
    _HAS_GREX = False

_GREX_LOG = os.environ.get("RULECHEF_GREX_LOG") == "1"


# ============================================================================
# FORMAT EXAMPLES - CODE
# ============================================================================

CODE_EXAMPLE_EXTRACTION = """
For CODE format, provide a function that takes input dict and returns list of dicts:
```python
def extract(input_data):
    # input_data is dict with keys from input_schema
    # return list of dicts: [{"text": "...", "start": 0, "end": 10}, ...]
    import re
    spans = []
    # your logic here
    return spans
```
"""

CODE_EXAMPLE_CLASSIFICATION = """
For CODE format, provide a function that takes input dict and returns a string label:
```python
def extract(input_data):
    # input_data is dict with keys from input_schema
    # return string label (e.g. "POSITIVE", "SPAM")
    if "bad" in input_data["text"]:
        return "SPAM"
    return "HAM"
```
"""

CODE_EXAMPLE_TRANSFORMATION = """
For CODE format, provide a function that takes input dict and returns the transformed output:
```python
def extract(input_data):
    # input_data is dict with keys from input_schema
    # return transformed data (dict, string, etc)
    return input_data["text"].upper()
```
"""

CODE_EXAMPLE_NER = """
For CODE format, provide a function that extracts entities with their types:
```python
def extract(input_data):
    # input_data is dict with keys from input_schema
    # return list of entities with text, position, AND type
    import re
    entities = []
    text = input_data.get("text", "")

    # Example: Find email addresses (type: LABEL_A)
    for match in re.finditer(r'\\S+@\\S+\\.\\S+', text):
        entities.append({
            "text": match.group(),
            "start": match.start(),
            "end": match.end(),
            "type": "LABEL_A"  # IMPORTANT: Include entity type
        })
    return entities
```
"""

CODE_EXAMPLES = {
    TaskType.EXTRACTION: CODE_EXAMPLE_EXTRACTION,
    TaskType.CLASSIFICATION: CODE_EXAMPLE_CLASSIFICATION,
    TaskType.TRANSFORMATION: CODE_EXAMPLE_TRANSFORMATION,
    TaskType.NER: CODE_EXAMPLE_NER,
}


# ============================================================================
# FORMAT EXAMPLES - SPACY
# ============================================================================

SPACY_INTRO = """
For SPACY format, provide a JSON array of token patterns (spaCy Matcher format).

Available token attributes:
- TEXT, LOWER: Exact or lowercase text match
- POS: Part-of-speech (NOUN, VERB, ADJ, PROPN, NUM, etc.)
- ENT_TYPE: Entity type (PERSON, ORG, GPE, DATE, MONEY, etc.) - requires spaCy NER
- DEP: Dependency label (nsubj, dobj, pobj, etc.)
- LEMMA: Base form of the token
- SHAPE: Word shape (dddd=4 digits, Xxxxx=capitalized)
- LIKE_NUM, LIKE_EMAIL, LIKE_URL: Boolean patterns
- IS_PUNCT, IS_DIGIT, IS_ALPHA: Character type checks
- OP: Quantifiers ("?" optional, "+" one or more, "*" zero or more)
- IN: Match any in list, e.g. {"LOWER": {"IN": ["yes", "yeah", "yep"]}}

Dependency-based patterns do NOT require spaCy NER and work when use_spacy_ner=False.

Dependency matcher patterns are also supported. Use a list of dicts with:
- RIGHT_ID: Node id
- RIGHT_ATTRS: Token attrs (POS, DEP, LEMMA, etc.)
- LEFT_ID: Parent node id
- REL_OP: Relation operator ("<", ">", "<<", ">>", ".")
"""

SPACY_EXAMPLE_EXTRACTION = """
SPACY extraction examples:

Example 1 - Extract 4-digit years:
{
  "name": "year_pattern",
  "format": "spacy",
  "content": "[{\\"SHAPE\\": \\"dddd\\"}]",
  "description": "Match 4-digit years like 1995, 2023"
}

Example 2 - Extract proper-name phrases:
{
  "name": "proper_noun_phrase",
  "format": "spacy",
  "content": "[{\\"POS\\": \\"PROPN\\", \\"OP\\": \\"+\\"}]",
  "description": "Match consecutive proper nouns"
}

Example 3 - Dependency matcher (verb with subject):
{
  "name": "verb_with_subject",
  "format": "spacy",
  "content": "[{\\"RIGHT_ID\\": \\"verb\\", \\"RIGHT_ATTRS\\": {\\"POS\\": \\"VERB\\"}}, {\\"LEFT_ID\\": \\"verb\\", \\"REL_OP\\": \\">\\", \\"RIGHT_ID\\": \\"subj\\", \\"RIGHT_ATTRS\\": {\\"DEP\\": \\"nsubj\\"}}]",
  "description": "Match a verb that has a nominal subject"
}

Example 4 - Extract money amounts:
{
  "name": "money_pattern",
  "format": "spacy",
  "content": "[{\\"LIKE_NUM\\": true}, {\\"LOWER\\": {\\"IN\\": [\\"dollar\\", \\"dollars\\", \\"usd\\", \\"euro\\", \\"euros\\"]}}]",
  "description": "Match amounts like '50 dollars'"
}
"""

SPACY_EXAMPLE_CLASSIFICATION = """
SPACY classification examples (use patterns to detect class indicators):

Example 1 - Detect urgency keywords:
{
  "name": "urgent_language",
  "format": "spacy",
  "content": "[{\\"LOWER\\": {\\"IN\\": [\\"urgent\\", \\"asap\\", \\"immediately\\", \\"emergency\\"]}}]",
  "description": "Detect urgent language -> classify as HIGH_PRIORITY"
}

Example 2 - Detect question patterns:
{
  "name": "question_pattern",
  "format": "spacy",
  "content": "[{\\"LOWER\\": {\\"IN\\": [\\"what\\", \\"where\\", \\"when\\", \\"who\\", \\"how\\", \\"why\\"]}}, {\\"OP\\": \\"*\\"}, {\\"IS_PUNCT\\": true, \\"TEXT\\": \\"?\\"}]",
  "description": "Detect questions -> classify as QUESTION"
}

Example 3 - Detect organization mentions:
{
  "name": "org_mention",
  "format": "spacy",
  "content": "[{\\"ENT_TYPE\\": \\"ORG\\"}]",
  "description": "Text mentions organization -> classify as BUSINESS"
}
"""

SPACY_EXAMPLE_OTHER = """
SPACY pattern examples:

Example 1 - Match noun phrases:
{
  "name": "noun_phrase",
  "format": "spacy",
  "content": "[{\\"POS\\": \\"DET\\", \\"OP\\": \\"?\\"}, {\\"POS\\": \\"ADJ\\", \\"OP\\": \\"*\\"}, {\\"POS\\": \\"NOUN\\"}]",
  "description": "Match noun phrases like 'the big house'"
}

Example 2 - Match email addresses:
{
  "name": "email_pattern",
  "format": "spacy",
  "content": "[{\\"LIKE_EMAIL\\": true}]",
  "description": "Match email addresses"
}
"""

SPACY_EXAMPLE_NER = """
SPACY NER examples (include output_template with entity type):

Example 1 - Person names with type:
{
  "name": "person_entities",
  "format": "spacy",
  "content": "[{\\"ENT_TYPE\\": \\"PERSON\\"}]",
  "output_template": {"text": "$0", "start": "$start", "end": "$end", "type": "LABEL_A"},
  "output_key": "entities",
  "description": "Extract person names as LABEL_A entities"
}

Example 2 - Organizations with type:
{
  "name": "org_entities",
  "format": "spacy",
  "content": "[{\\"ENT_TYPE\\": \\"ORG\\"}]",
  "output_template": {"text": "$0", "start": "$start", "end": "$end", "type": "LABEL_B"},
  "output_key": "entities",
  "description": "Extract organizations as LABEL_B entities"
}
"""

SPACY_EXAMPLES = {
    TaskType.EXTRACTION: SPACY_EXAMPLE_EXTRACTION,
    TaskType.CLASSIFICATION: SPACY_EXAMPLE_CLASSIFICATION,
    TaskType.TRANSFORMATION: SPACY_EXAMPLE_OTHER,
    TaskType.NER: SPACY_EXAMPLE_NER,
}


# ============================================================================
# PROMPT GUIDANCE
# ============================================================================

RULE_QUALITY_GUIDE = """WHAT MAKES A GOOD RULE:
- Prefer precision over recall: a narrow rule that matches exactly what it should is better than a broad rule that matches wrong things.
- Do not overfit to exact training strings: generalize just enough to cover unseen examples of the same pattern.
- Use context clues when possible: words around an entity often disambiguate better than the entity text alone.
- Write multiple focused rules instead of one giant pattern.
- If two entity types look similar as strings, use surrounding context to disambiguate (not just capitalization).
- It is OK to miss rare edge cases; avoid rules that match garbage.
"""

REGEX_TECHNIQUE_GUIDE = """REGEX TECHNIQUES:
- Use \\b word boundaries to avoid partial word matches.
- Use (?:...) for non-capturing groups.
- Use alternation: (word1|word2|word3).
- Use character classes: [A-Z], [a-z], \\d, \\s.
- Use quantifiers: +, *, ?, {n,m}.
- Use lookahead (?=...) / lookbehind (?<=...) for context without consuming.
- Do NOT assume entities are capitalized; check the training data.
- Prefer matching context (e.g. "at <ORG>") and capturing the entity span.
"""

SPACY_TECHNIQUE_GUIDE = """SPACY TECHNIQUES:
- Use POS tags (PROPN, NOUN, VERB) for linguistic patterns.
- Use DEP labels (nsubj, dobj, pobj) for syntactic relationships.
- Use SHAPE for structural patterns (e.g. "dddd" for years, "Xxxxx" for capitalized).
- Use IN for matching any of several values.
- Use OP for quantifiers ("?" optional, "+" one or more).
- Dependency matcher is powerful for verb-argument patterns.
- Do NOT use ENT_TYPE unless spaCy NER is explicitly enabled.
"""


# ============================================================================
# JSON RESPONSE SCHEMAS
# ============================================================================


def get_standard_response_schema(format_options: List[str]) -> str:
    """Get JSON schema for standard (non-schema-aware) tasks"""
    format_str = '" or "'.join(format_options)
    return f'''
Return JSON:
{{
  "analysis": "What patterns did you find?",
  "strategy": "Overall approach",
  "rules": [
    {{
      "name": "Short rule name",
      "description": "What this rule does",
      "format": "{format_str}",
      "content": "regex pattern OR python code",
      "priority": 1-10
    }}
  ]
}}
'''


def get_schema_aware_response_schema(
    format_options: List[str], primary_key: str
) -> str:
    """Get JSON schema for schema-aware tasks (NER, TRANSFORMATION)"""
    format_str = '" or "'.join(format_options)
    if len(format_options) == 1 and format_options[0] == "regex":
        pattern_desc = "regex pattern"
    elif len(format_options) == 1 and format_options[0] == "spacy":
        pattern_desc = "spaCy JSON pattern"
    else:
        pattern_desc = "regex pattern OR spaCy JSON pattern"
    return f'''
Return JSON:
{{
  "analysis": "What patterns did you find? What went wrong in corrections?",
  "strategy": "Overall approach",
  "note": "Every rule must include both output_template and output_key",
  "rules": [
    {{
      "name": "Short rule name",
      "description": "What this rule does",
      "format": "{format_str}",
      "pattern": "{pattern_desc}",
      "output_template": {{"text": "$0", "start": "$start", "end": "$end", "type": "LITERAL_OR_$VAR"}},
      "output_key": "{primary_key}",
      "priority": 1-10 (higher = more important),
      "reasoning": "Why this rule is needed"
    }}
  ]
}}
'''


# ============================================================================
# PROMPT BUILDER
# ============================================================================


class PromptBuilder:
    """Builds prompts for rule learning from composable parts"""

    def __init__(
        self,
        allowed_formats: List[RuleFormat],
        use_spacy_ner: bool = False,
        use_grex: bool = True,
    ):
        self.allowed_formats = allowed_formats
        self.use_spacy_ner = use_spacy_ner
        self.use_grex = use_grex

    # ========================================
    # Main Prompt Builders
    # ========================================

    def build_synthesis_prompt(self, dataset: Dataset, max_rules: int) -> str:
        """Build the complete synthesis prompt"""
        parts = [
            self._build_task_header(dataset),
            self._build_training_data_section(dataset),
            self._build_data_evidence(dataset),
            self._build_feedback_section(dataset),
            self._build_existing_rules_section(dataset),
            self._build_task_instructions(dataset, max_rules),
            self._build_format_instructions(dataset.task.type),
            self._build_response_schema(dataset),
            self._build_format_examples(dataset.task.type),
            self._build_closing_instructions(),
        ]
        return "\n".join(parts)

    def build_refinement_prompt(
        self,
        current_rules: List[Rule],
        failures: List[Dict],
        dataset: Dataset,
    ) -> str:
        """Build prompt for refining rules based on failures (schema-aware)"""
        rules_formatted = self._format_rules(current_rules)
        format_instructions = self._build_format_instructions(dataset.task.type)
        response_schema = self._build_response_schema(dataset)

        return f"""Refine the ruleset for this task while fixing the failures shown.

{self._build_task_header(dataset)}
{self._build_data_evidence(dataset)}

CURRENT RULES:
{rules_formatted or "None"}

FAILURES TO FIX (include these patterns in your updated rules):
{json.dumps(failures, indent=2)}

CRITICAL: Pay special attention to correction failures (is_correction: true) - these are user-verified mistakes.

{format_instructions}

Return refined ruleset in the same JSON format (every rule must include both output_template and output_key):
{response_schema}
"""

    def build_generation_prompt(self, task: Any, seed: int = 0) -> str:
        """Build prompt for generating synthetic training examples"""
        return f"""Generate a realistic training example for this task:

Task: {task.name}
Description: {task.description}
Input schema: {task.input_schema}

Return JSON with input fields only.
Example #{seed + 1}:"""

    # ========================================
    # Section Builders
    # ========================================

    def _build_task_header(self, dataset: Dataset) -> str:
        """Build the task description header with schema and labels"""
        task = dataset.task

        # Get schema representation (uses Pydantic formatting if available)
        output_schema_str = task.get_schema_for_prompt()

        header = f"""Task: {task.name}
Description: {task.description}

Input schema: {task.input_schema}
Output schema:
{output_schema_str}
        """

        # Add label descriptions for NER/classification tasks
        labels = task.get_labels()
        if not labels:
            labels = self._derive_labels_from_data(dataset)
        if labels:
            header += f"\nAVAILABLE ENTITY TYPES: {labels}\n"
            header += "Rules MUST use one of these types in output_template.\n"

        return header

    def _build_training_data_section(self, dataset: Dataset) -> str:
        """Build section with training data (corrections and examples)"""
        parts = []

        # This will be filled by the learner with sampled data
        # We provide the format here
        parts.append("{{TRAINING_DATA}}")

        return "\n".join(parts)

    def _build_corrections_section(self, corrections: List[Correction]) -> str:
        """Build section showing corrections (failures to learn from)"""
        if not corrections:
            return ""

        lines = [f"CORRECTIONS (Learn from failures - {len(corrections)} shown):"]
        for corr in corrections:
            lines.append(f"\nInput: {json.dumps(corr.input)}")
            lines.append(f"Got (WRONG): {json.dumps(corr.model_output)}")
            lines.append(f"Expected (CORRECT): {json.dumps(corr.expected_output)}")
            if corr.feedback:
                lines.append(f"Feedback: {corr.feedback}")

        return "\n".join(lines)

    def _build_examples_section(self, examples: List[Any]) -> str:
        """Build section showing training examples"""
        if not examples:
            return ""

        lines = [f"\nTRAINING EXAMPLES ({len(examples)} shown):"]
        for ex in examples:
            lines.append(f"\nInput: {json.dumps(ex.input)}")
            lines.append(f"Output: {json.dumps(ex.expected_output)}")

        return "\n".join(lines)

    def _build_feedback_section(self, dataset: Dataset) -> str:
        """Build section with user feedback"""
        if not dataset.feedback:
            return ""

        lines = ["\n\nUSER FEEDBACK:"]
        for fb in dataset.feedback:
            lines.append(f"- {fb}")

        return "\n".join(lines)

    def _build_existing_rules_section(self, dataset: Dataset) -> str:
        """Build section showing existing rules with stats"""
        if not dataset.rules:
            return ""

        lines = [f"\n\nEXISTING RULES ({len(dataset.rules)} current):"]

        for rule in dataset.rules:
            success_rate = (
                f"{rule.successes / rule.times_applied * 100:.1f}%"
                if rule.times_applied > 0
                else "untested"
            )
            lines.append(
                f"\n- {rule.name} (priority {rule.priority}, success: {success_rate})"
            )
            lines.append(f"  Format: {rule.format.value}")
            lines.append(
                f"  Pattern: {rule.content[:100]}{'...' if len(rule.content) > 100 else ''}"
            )
            lines.append(f"  Confidence: {rule.confidence:.2f}")

        lines.append("\nCONSIDER:")
        lines.append("- Refine existing high-performing rules")
        lines.append("- Fix or replace low-performing rules")
        lines.append("- Keep rules that work well")
        lines.append("- Add new rules for uncovered patterns")

        return "\n".join(lines)

    def _build_task_instructions(self, dataset: Dataset, max_rules: int) -> str:
        """Build the main task instructions"""
        action = "Update and refine" if dataset.rules else "Synthesize a complete"

        return f"""

YOUR TASK:
{action} ruleset (max {max_rules} rules) that:
1. Handles all corrections correctly (CRITICAL - these show failure modes)
2. Works on all examples
3. Respects user feedback
4. Is general and minimal (avoid redundant rules)

{RULE_QUALITY_GUIDE}

RULES CAN BE:"""

    def _build_format_instructions(self, task_type: TaskType) -> str:
        """Build instructions about allowed formats"""
        lines = []

        if RuleFormat.REGEX in self.allowed_formats:
            lines.append("- Regex patterns (for structured extraction)")
        if RuleFormat.CODE in self.allowed_formats:
            lines.append("- Python code (for complex logic)")
        if RuleFormat.SPACY in self.allowed_formats:
            lines.append("- spaCy token matcher patterns (for linguistic/NLP patterns)")

        lines.append("")
        lines.append(
            "IMPORTANT: You must ONLY use the allowed formats listed above. Do NOT generate rules in other formats."
        )
        if RuleFormat.CODE not in self.allowed_formats:
            lines.append(
                "IMPORTANT: Do NOT include Python/code rules. Only return the listed formats."
            )
        else:
            lines.append(
                "IMPORTANT: For CODE rules, write standard multi-line Python functions with proper indentation. Do NOT write one-liners."
            )
        if RuleFormat.SPACY not in self.allowed_formats:
            lines.append(
                "IMPORTANT: Do NOT include spaCy rules. Only return the listed formats."
            )
        if RuleFormat.SPACY in self.allowed_formats:
            lines.append(
                "IMPORTANT: spaCy rules must be valid JSON arrays of token dicts (spaCy Matcher patterns). Do NOT include Python/spacy code; only JSON."
            )
            if not self.use_spacy_ner:
                lines.append(
                    "IMPORTANT: spaCy NER is disabled. Do NOT use ENT_TYPE or ENT_ID in spaCy patterns."
                )

        # Technique guides are format-specific; only include those that are allowed.
        if RuleFormat.REGEX in self.allowed_formats:
            lines.append("")
            lines.append(REGEX_TECHNIQUE_GUIDE)
        if RuleFormat.SPACY in self.allowed_formats:
            lines.append("")
            lines.append(SPACY_TECHNIQUE_GUIDE)

        return "\n".join(lines)

    def _build_response_schema(self, dataset: Dataset) -> str:
        """Build the expected JSON response schema"""
        format_options = self._get_format_options()
        is_schema_aware = dataset.task.type in (TaskType.NER, TaskType.TRANSFORMATION)

        if is_schema_aware:
            return self._build_schema_aware_section(dataset, format_options)
        else:
            return get_standard_response_schema(format_options)

    def _build_schema_aware_section(
        self, dataset: Dataset, format_options: List[str]
    ) -> str:
        """Build prompt section for schema-aware tasks (NER, TRANSFORMATION)"""
        from rulechef.core import is_pydantic_schema

        # Get output keys - handle both dict and Pydantic schemas
        if is_pydantic_schema(dataset.task.output_schema):
            output_keys = list(dataset.task.output_schema.model_fields.keys())
        else:
            output_keys = list(dataset.task.output_schema.keys())
        primary_key = output_keys[0] if output_keys else "entities"

        pattern_options = []
        if RuleFormat.REGEX in self.allowed_formats:
            pattern_options.append("regex pattern")
        if RuleFormat.SPACY in self.allowed_formats:
            pattern_options.append("spaCy token matcher JSON pattern")

        template_vars = [
            "  - $0: Full match text",
            "  - $start: Start character offset",
            "  - $end: End character offset",
        ]
        if RuleFormat.REGEX in self.allowed_formats:
            template_vars.append("  - $1, $2, ...: Capture groups (regex only)")
        if RuleFormat.SPACY in self.allowed_formats:
            template_vars.append(
                "  - $ent_type: Entity type from spaCy NER (spaCy only)"
            )
            template_vars.append(
                "  - $1.start/$1.end/$1.text: Token offsets/text within spaCy match"
            )

        pattern_desc = " or ".join(pattern_options) if pattern_options else "pattern"

        # Get readable schema representation
        schema_str = dataset.task.get_schema_for_prompt()

        section = f"""

SCHEMA-AWARE RULES:
For this task, rules must include an output_template that maps matches to the output schema.

Output schema:
{schema_str}

Each rule needs:
- pattern: The {pattern_desc} to match
- output_template: JSON template for each match, using variables:
{os.linesep.join(template_vars)}
- output_key: Which output array to populate (e.g., "{primary_key}")
"""
        section += get_schema_aware_response_schema(format_options, primary_key)

        return section

    def _derive_labels_from_data(self, dataset: Dataset) -> List[str]:
        """Derive label strings from expected_output in examples/corrections.

        This is a fallback when Task.get_labels() returns [] (common with dict schemas).
        """
        labels: Set[str] = set()
        for item in list(dataset.examples) + list(dataset.corrections):
            output = getattr(item, "expected_output", None) or {}
            for key in ("entities", "spans"):
                entities = output.get(key, [])
                if not isinstance(entities, list):
                    continue
                for ent in entities:
                    if not isinstance(ent, dict):
                        continue
                    # Support a few common field names.
                    for field in ("type", "label", "tag"):
                        val = ent.get(field)
                        if isinstance(val, str) and val.strip():
                            labels.add(val.strip())
                            break
        return sorted(labels)

    # ========================================
    # grex helpers
    # ========================================

    def _grex_patterns(self, strings: List[str], context: str = "") -> List[str]:
        """Generate regex pattern hints from example strings using grex.

        Returns lines to append to evidence sections. Always emits the exact
        pattern (alternation) and additionally emits a generalized structural
        pattern when the ratio heuristic detects real structure (< 0.7).
        """
        if not self.use_grex:
            return []
        if not _HAS_GREX:
            return []
        if len(strings) < 2:
            return []
        # grex can produce huge patterns that overfit or bloat the prompt; keep it bounded.
        unique: List[str] = []
        for s in strings:
            if not isinstance(s, str):
                continue
            normalized = " ".join(s.split())
            if not normalized:
                continue
            if normalized not in unique:
                unique.append(normalized)

        if len(unique) < 2:
            return []
        if len(unique) > 30:
            return []
        if any(len(s) > 80 for s in unique):
            return []
        if sum(len(s) for s in unique) > 1200:
            return []

        # Make grex deterministic: order-sensitive builders can vary with input order.
        unique = sorted(unique)
        try:
            exact = RegExpBuilder.from_test_cases(unique).without_anchors().build()
            generalized = (
                RegExpBuilder.from_test_cases(unique)
                .without_anchors()
                .with_conversion_of_digits()
                .with_conversion_of_repetitions()
                .build()
            )
            lines = [f"  Exact pattern: {exact}"]
            if generalized != exact and len(exact) > 0:
                ratio = len(generalized) / len(exact)
                if ratio < 0.7:
                    lines.append(f"  Structural pattern: {generalized}")
            if _GREX_LOG:
                label = f" {context}" if context else ""
                print(f"[rulechef][grex] used{label}")
            return lines
        except Exception:
            return []

    # ========================================
    # Data evidence (task-type-aware)
    # ========================================

    def _build_data_evidence(self, dataset: Dataset) -> str:
        """Build data evidence section, dispatching by task type."""
        task_type = dataset.task.type
        if task_type == TaskType.NER:
            return self._build_ner_evidence(dataset)
        elif task_type == TaskType.EXTRACTION:
            return self._build_extraction_evidence(dataset)
        elif task_type == TaskType.CLASSIFICATION:
            return self._build_classification_evidence(dataset)
        elif task_type == TaskType.TRANSFORMATION:
            return self._build_transformation_evidence(dataset)
        return ""

    def _build_ner_evidence(self, dataset: Dataset) -> str:
        """Summarize entity strings seen in training data for NER tasks."""
        max_labels = 25
        max_strings_per_label = 15
        max_total_chars = 3000

        label_to_texts: Dict[str, List[str]] = defaultdict(list)
        saw_lowercase = False
        saw_multiword = False

        def _add(label: str, text: str):
            nonlocal saw_lowercase, saw_multiword
            if not label or not text:
                return
            normalized = " ".join(str(text).split())
            if not normalized:
                return
            if (
                any(c.isalpha() for c in normalized)
                and normalized.lower() == normalized
            ):
                saw_lowercase = True
            if " " in normalized.strip():
                saw_multiword = True
            existing = label_to_texts[label]
            if normalized in existing:
                return
            if len(existing) < max_strings_per_label:
                existing.append(normalized)

        for item in list(dataset.examples) + list(dataset.corrections):
            output = getattr(item, "expected_output", None) or {}
            entities = output.get("entities", [])
            if isinstance(entities, list):
                for ent in entities:
                    if not isinstance(ent, dict):
                        continue
                    label = ent.get("type") or ent.get("label") or ent.get("tag")
                    text = ent.get("text")
                    if isinstance(label, str) and isinstance(text, str):
                        _add(label.strip(), text)

        if not label_to_texts:
            return ""

        labels = sorted(label_to_texts.keys())[:max_labels]
        lines = ["", "DATA EVIDENCE FROM TRAINING:"]
        for label in labels:
            vals = label_to_texts[label]
            preview = ", ".join(json.dumps(v) for v in vals)
            lines.append(f"- {label} ({len(vals)} unique): {preview}")
            lines.extend(self._grex_patterns(vals, context=f"NER:{label}"))

        notes: List[str] = []
        if saw_lowercase:
            notes.append("Some entities are lowercase; do NOT assume capitalization.")
        if saw_multiword:
            notes.append(
                "Some entities contain multiple words; do NOT assume single-token matches."
            )
        if notes:
            lines.append("")
            lines.append("Note: " + " ".join(notes[:2]))

        lines.append("")
        lines.append(
            "Computed patterns match training strings only; generalize carefully."
        )

        out = "\n".join(lines)
        if len(out) > max_total_chars:
            out = out[: max_total_chars - 3] + "..."
        return out

    def _build_extraction_evidence(self, dataset: Dataset) -> str:
        """Summarize extracted span texts for EXTRACTION tasks."""
        max_strings = 30
        max_total_chars = 2000

        texts: List[str] = []
        for item in list(dataset.examples) + list(dataset.corrections):
            output = getattr(item, "expected_output", None) or {}
            spans = output.get("spans", [])
            if not isinstance(spans, list):
                continue
            for span in spans:
                if not isinstance(span, dict):
                    continue
                text = span.get("text")
                if isinstance(text, str) and text.strip():
                    normalized = " ".join(text.split())
                    if normalized not in texts:
                        texts.append(normalized)
                        if len(texts) >= max_strings:
                            break

        if not texts:
            return ""

        lines = ["", "DATA EVIDENCE FROM TRAINING:"]
        preview = ", ".join(json.dumps(t) for t in texts)
        lines.append(f"- Extracted spans ({len(texts)} unique): {preview}")
        lines.extend(self._grex_patterns(texts, context="EXTRACTION:spans"))
        lines.append("")
        lines.append(
            "Computed patterns match training strings only; generalize carefully."
        )

        out = "\n".join(lines)
        if len(out) > max_total_chars:
            out = out[: max_total_chars - 3] + "..."
        return out

    def _build_classification_evidence(self, dataset: Dataset) -> str:
        """Summarize input texts grouped by label for CLASSIFICATION tasks."""
        max_labels = 20
        max_inputs_per_label = 10
        max_total_chars = 3000

        label_to_inputs: Dict[str, List[str]] = defaultdict(list)

        text_field = dataset.task.text_field

        def _get_text(input_data: Dict) -> str:
            if text_field and text_field in input_data:
                return str(input_data[text_field])
            for v in input_data.values():
                if isinstance(v, str):
                    return v
            return ""

        for item in list(dataset.examples) + list(dataset.corrections):
            output = getattr(item, "expected_output", None) or {}
            label = output.get("label") or output.get("class") or output.get("category")
            if not isinstance(label, str) or not label.strip():
                continue
            input_data = getattr(item, "input", None) or {}
            text = _get_text(input_data).strip()
            if not text:
                continue
            existing = label_to_inputs[label.strip()]
            if text not in existing and len(existing) < max_inputs_per_label:
                existing.append(text)

        if not label_to_inputs:
            return ""

        labels = sorted(label_to_inputs.keys())[:max_labels]
        lines = ["", "DATA EVIDENCE FROM TRAINING:"]
        for label in labels:
            inputs = label_to_inputs[label]
            preview = ", ".join(json.dumps(t[:80]) for t in inputs)
            lines.append(f"- {label} ({len(inputs)} examples): {preview}")
            lines.extend(
                self._grex_patterns(
                    [t[:80] for t in inputs], context=f"CLASSIFICATION:{label}"
                )
            )

        lines.append("")
        lines.append(
            "Computed patterns match training strings only; generalize carefully."
        )

        out = "\n".join(lines)
        if len(out) > max_total_chars:
            out = out[: max_total_chars - 3] + "..."
        return out

    def _build_transformation_evidence(self, dataset: Dataset) -> str:
        """Summarize output values per key for TRANSFORMATION tasks."""
        max_keys = 10
        max_values_per_key = 15
        max_total_chars = 3000

        key_to_values: Dict[str, List[str]] = defaultdict(list)

        for item in list(dataset.examples) + list(dataset.corrections):
            output = getattr(item, "expected_output", None) or {}
            for key, value in output.items():
                if not isinstance(value, str) or not value.strip():
                    continue
                existing = key_to_values[key]
                normalized = " ".join(value.split())
                if normalized not in existing and len(existing) < max_values_per_key:
                    existing.append(normalized)

        if not key_to_values:
            return ""

        keys = sorted(key_to_values.keys())[:max_keys]
        lines = ["", "DATA EVIDENCE FROM TRAINING:"]
        for key in keys:
            vals = key_to_values[key]
            preview = ", ".join(json.dumps(v) for v in vals)
            lines.append(f"- {key} ({len(vals)} unique): {preview}")
            lines.extend(self._grex_patterns(vals, context=f"TRANSFORMATION:{key}"))

        lines.append("")
        lines.append(
            "Computed patterns match training strings only; generalize carefully."
        )

        out = "\n".join(lines)
        if len(out) > max_total_chars:
            out = out[: max_total_chars - 3] + "..."
        return out

    def _build_format_examples(self, task_type: TaskType) -> str:
        """Build examples for each allowed format"""
        parts = []

        if RuleFormat.CODE in self.allowed_formats:
            example = CODE_EXAMPLES.get(task_type, CODE_EXAMPLE_EXTRACTION)
            parts.append(example)

        if RuleFormat.SPACY in self.allowed_formats:
            parts.append(SPACY_INTRO)
            example = SPACY_EXAMPLES.get(task_type, SPACY_EXAMPLE_OTHER)
            parts.append(example)

        return "\n".join(parts)

    def _build_closing_instructions(self) -> str:
        """Build closing instructions for the prompt"""
        return """
Focus on learning from CORRECTIONS - they show exactly what went wrong!

IMPORTANT: Return ONLY valid JSON. Ensure:
- All strings use double quotes and are properly escaped
- All braces and brackets are balanced
- No trailing commas
- Response is complete (not truncated)
"""

    # ========================================
    # Utilities
    # ========================================

    def _get_format_options(self) -> List[str]:
        """Get list of format option strings"""
        options = ["regex"]
        if RuleFormat.CODE in self.allowed_formats:
            options.append("code")
        if RuleFormat.SPACY in self.allowed_formats:
            options.append("spacy")
        return options

    def _format_rules(self, rules: List[Rule]) -> str:
        """Format rules for display in prompts"""
        lines = []
        for i, rule in enumerate(rules, 1):
            lines.append(f"{i}. {rule.name}")
            lines.append(f"   Format: {rule.format.value}")
            lines.append(f"   Priority: {rule.priority}")
            content_preview = (
                rule.content[:100] + "..." if len(rule.content) > 100 else rule.content
            )
            lines.append(f"   Content: {content_preview}")
        return "\n".join(lines)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def format_output(output: Dict) -> str:
    """Format output for prompt display"""
    spans = output.get("spans", [])
    return json.dumps(
        [{"text": s["text"], "start": s["start"], "end": s["end"]} for s in spans]
    )

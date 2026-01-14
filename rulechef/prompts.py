"""Prompt templates and builders for rule learning"""

import json
import os
from typing import Dict, List, Any

from rulechef.core import Rule, RuleFormat, Dataset, Correction, TaskType


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

    # Example: Find email addresses (type: EMAIL)
    for match in re.finditer(r'\\S+@\\S+\\.\\S+', text):
        entities.append({
            "text": match.group(),
            "start": match.start(),
            "end": match.end(),
            "type": "EMAIL"  # IMPORTANT: Include entity type
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
- ENT_TYPE: Entity type (PERSON, ORG, GPE, DATE, MONEY, etc.)
- SHAPE: Word shape (dddd=4 digits, Xxxxx=capitalized)
- LIKE_NUM, LIKE_EMAIL, LIKE_URL: Boolean patterns
- IS_PUNCT, IS_DIGIT, IS_ALPHA: Character type checks
- OP: Quantifiers ("?" optional, "+" one or more, "*" zero or more)
- IN: Match any in list, e.g. {"LOWER": {"IN": ["yes", "yeah", "yep"]}}
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

Example 2 - Extract person names:
{
  "name": "person_names",
  "format": "spacy",
  "content": "[{\\"ENT_TYPE\\": \\"PERSON\\"}]",
  "description": "Match named person entities"
}

Example 3 - Extract dates with context:
{
  "name": "date_phrases",
  "format": "spacy",
  "content": "[{\\"LOWER\\": {\\"IN\\": [\\"in\\", \\"on\\", \\"during\\"]}}, {\\"ENT_TYPE\\": \\"DATE\\"}]",
  "description": "Match 'in/on/during [DATE]' patterns"
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
  "output_template": {"text": "$0", "start": "$start", "end": "$end", "type": "PER"},
  "output_key": "entities",
  "description": "Extract person names as PER entities"
}

Example 2 - Organizations with type:
{
  "name": "org_entities",
  "format": "spacy",
  "content": "[{\\"ENT_TYPE\\": \\"ORG\\"}]",
  "output_template": {"text": "$0", "start": "$start", "end": "$end", "type": "ORG"},
  "output_key": "entities",
  "description": "Extract organizations as ORG entities"
}
"""

SPACY_EXAMPLES = {
    TaskType.EXTRACTION: SPACY_EXAMPLE_EXTRACTION,
    TaskType.CLASSIFICATION: SPACY_EXAMPLE_CLASSIFICATION,
    TaskType.TRANSFORMATION: SPACY_EXAMPLE_OTHER,
    TaskType.NER: SPACY_EXAMPLE_NER,
}


# ============================================================================
# SCHEMA-AWARE EXAMPLES (NER, TRANSFORMATION)
# ============================================================================

NER_RULE_EXAMPLES_REGEX = """
NER RULE EXAMPLES (regex):

Example 1 - Organizations with corporate suffixes:
{
  "name": "corporate_suffixes",
  "format": "regex",
  "pattern": "\\b([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)\\s+(Inc\\.|LLC|Corp\\.|Corporation)\\b",
  "output_template": {"text": "$0", "start": "$start", "end": "$end", "type": "ORG"},
  "output_key": "entities",
  "priority": 9
}

Example 2 - Person names (two capitalized words):
{
  "name": "person_names",
  "format": "regex",
  "pattern": "\\b([A-Z][a-z]+)\\s+([A-Z][a-z]+)\\b",
  "output_template": {"text": "$0", "start": "$start", "end": "$end", "type": "PER"},
  "output_key": "entities",
  "priority": 7
}
"""

NER_RULE_EXAMPLES_SPACY = """
NER RULE EXAMPLES (spaCy):

Example 1 - Using spaCy's built-in NER for persons:
{
  "name": "spacy_persons",
  "format": "spacy",
  "pattern": "[{\\"ENT_TYPE\\": \\"PERSON\\"}]",
  "output_template": {"text": "$0", "start": "$start", "end": "$end", "type": "PER"},
  "output_key": "entities",
  "priority": 8
}

Example 2 - Locations with spaCy:
{
  "name": "spacy_locations",
  "format": "spacy",
  "pattern": "[{\\"ENT_TYPE\\": {\\"IN\\": [\\"GPE\\", \\"LOC\\"]}}]",
  "output_template": {"text": "$0", "start": "$start", "end": "$end", "type": "LOC"},
  "output_key": "entities",
  "priority": 8
}
"""

TRANSFORMATION_RULE_EXAMPLES = """
TRANSFORMATION RULE EXAMPLES:

Example 1 - Extract dates:
{
  "name": "date_pattern",
  "format": "regex",
  "pattern": "\\b(\\d{4})-(\\d{2})-(\\d{2})\\b",
  "output_template": {"date": "$0", "year": "$1", "month": "$2", "day": "$3"},
  "output_key": "dates",
  "priority": 8
}

Example 2 - Extract prices:
{
  "name": "price_pattern",
  "format": "regex",
  "pattern": "\\$([\\d,]+(?:\\.\\d{2})?)",
  "output_template": {"amount": "$1", "currency": "USD", "raw": "$0"},
  "output_key": "prices",
  "priority": 8
}
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

    def __init__(self, allowed_formats: List[RuleFormat]):
        self.allowed_formats = allowed_formats

    # ========================================
    # Main Prompt Builders
    # ========================================

    def build_synthesis_prompt(self, dataset: Dataset, max_rules: int) -> str:
        """Build the complete synthesis prompt"""
        parts = [
            self._build_task_header(dataset),
            self._build_training_data_section(dataset),
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

        # Add task-specific examples
        if dataset.task.type == TaskType.NER:
            example_sections = []
            if RuleFormat.REGEX in self.allowed_formats:
                example_sections.append(NER_RULE_EXAMPLES_REGEX)
            if RuleFormat.SPACY in self.allowed_formats:
                example_sections.append(NER_RULE_EXAMPLES_SPACY)
            if example_sections:
                section += "\n\n".join(example_sections)
        elif dataset.task.type == TaskType.TRANSFORMATION:
            section += TRANSFORMATION_RULE_EXAMPLES

        return section

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

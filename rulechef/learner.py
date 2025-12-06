"""LLM-based rule learning"""

import json
import re
from typing import Dict, List, Optional, Callable, Any
from openai import OpenAI

from rulechef.core import Rule, RuleFormat, Span, Dataset, Correction, TaskType


class RuleLearner:
    """Learns extraction rules from examples using LLM"""

    def __init__(
        self,
        llm: OpenAI,
        allowed_formats: Optional[List[RuleFormat]] = None,
        sampling_strategy: str = "balanced",
        model: str = "gpt-4o-mini",
    ):
        self.llm = llm
        self.allowed_formats = allowed_formats or [RuleFormat.REGEX, RuleFormat.CODE]
        self.sampling_strategy = sampling_strategy
        self.model = model

    # ========================================
    # Smart Sampling (for large datasets)
    # ========================================

    def _sample_training_data(
        self, dataset: Dataset, max_samples: int = 100, strategy: str = "balanced"
    ):
        """
        Intelligently sample training data for prompt inclusion.

        Strategies:
        - 'balanced': Mix of examples from across dataset
        - 'corrections_first': Prioritize recent examples (after user corrections)
        - 'recent': Most recently added examples
        - 'diversity': Spread samples across dataset range
        - 'uncertain': Examples with lowest confidence (active learning)
        - 'varied': Mix strategies - corrections + recent + diverse
        """
        samples = []

        # Priority 1: ALL corrections (they're high-value, usually few)
        samples.extend(dataset.corrections)

        if len(samples) >= max_samples:
            return samples[:max_samples]

        # Priority 2: Examples (up to max_samples - corrections)
        remaining_budget = max_samples - len(samples)
        examples = dataset.examples

        if not examples:
            return samples[:max_samples]

        if strategy == "balanced":
            # Simple: take first N examples (original diverse ones)
            samples.extend(examples[:remaining_budget])

        elif strategy == "corrections_first":
            # After user correction, prioritize recent examples
            samples.extend(
                sorted(examples, key=lambda e: e.timestamp, reverse=True)[
                    :remaining_budget
                ]
            )

        elif strategy == "recent":
            # Most recent examples only
            samples.extend(
                sorted(examples, key=lambda e: e.timestamp, reverse=True)[
                    :remaining_budget
                ]
            )

        elif strategy == "diversity":
            # Spread samples across dataset range (every Nth example)
            if len(examples) <= remaining_budget:
                samples.extend(examples)
            else:
                step = len(examples) // remaining_budget
                samples.extend([examples[i * step] for i in range(remaining_budget)])

        elif strategy == "uncertain":
            # Active learning: prioritize low-confidence examples
            # Examples with lower confidence indicate uncertain/edge cases
            sorted_by_confidence = sorted(
                examples, key=lambda e: e.confidence, reverse=False
            )
            samples.extend(sorted_by_confidence[:remaining_budget])

        elif strategy == "varied":
            # Mixed strategy: 40% recent + 40% diverse + 20% low-confidence
            thirds = remaining_budget // 3
            recent = sorted(examples, key=lambda e: e.timestamp, reverse=True)[:thirds]
            diverse = [
                examples[i * (len(examples) // thirds)]
                for i in range(1, thirds + 1)
                if i * (len(examples) // thirds) < len(examples)
            ]
            uncertain = sorted(examples, key=lambda e: e.confidence, reverse=False)[
                : remaining_budget - len(recent) - len(diverse)
            ]
            samples.extend(recent + diverse + uncertain)

        return samples[:max_samples]

    def _sample_failures(self, failures: List[Dict], max_samples: int = 20):
        """
        Intelligently sample failures for refinement.

        Prioritizes:
        1. ALL correction failures (user-verified mistakes - highest value)
        2. Other failures up to remaining budget
        """
        correction_failures = [f for f in failures if f.get("is_correction", False)]
        other_failures = [f for f in failures if not f.get("is_correction", False)]

        # Always include all correction failures
        sampled = correction_failures

        # Add other failures up to budget
        remaining_budget = max_samples - len(sampled)
        if remaining_budget > 0:
            sampled.extend(other_failures[:remaining_budget])

        return sampled[:max_samples]

    # ========================================
    # Rule Synthesis
    # ========================================

    def synthesize_ruleset(self, dataset: Dataset, max_rules: int = 10) -> List[Rule]:
        """Generate initial ruleset from dataset"""
        import time

        prompt = self._build_synthesis_prompt(dataset, max_rules)

        print("📚 Synthesizing rules from dataset...")

        start = time.time()
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                max_completion_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )

            result = self._parse_json(response.choices[0].message.content)

            # Extract rules from response
            rules = []
            for i, rule_data in enumerate(result.get("rules", [])[:max_rules]):
                rule_format = RuleFormat(rule_data.get("format", "regex"))

                # Skip rules not in allowed formats
                if rule_format not in self.allowed_formats:
                    print(
                        f"   ⚠ Skipped {rule_format.value} rule (not allowed): {rule_data.get('name', f'Rule {i + 1}')}"
                    )
                    continue

                rule = Rule(
                    id=self._generate_id(),
                    name=rule_data.get("name", f"Rule {i + 1}"),
                    description=rule_data.get("description", ""),
                    format=rule_format,
                    content=rule_data.get("content", ""),
                    priority=rule_data.get("priority", 5),
                )
                # Validate rule before adding
                if self._validate_rule(rule):
                    rules.append(rule)
                else:
                    print(f"   ⚠ Skipped invalid rule: {rule.name}")

            elapsed = time.time() - start
            print(f"✓ Synthesized {len(rules)} rules ({elapsed:.1f}s)")
            return rules

        except Exception as e:
            print(f"Error synthesizing rules: {e}")
            return []

    def _build_synthesis_prompt(self, dataset: Dataset, max_rules: int) -> str:
        """Build prompt for rule synthesis"""

        prompt = f"""Task: {dataset.task.name}
Description: {dataset.task.description}

Input schema: {dataset.task.input_schema}
Output schema: {dataset.task.output_schema}

"""

        # Smart sample data for prompt (instead of hard [:5] limits)
        sampled_data = self._sample_training_data(
            dataset, max_samples=50, strategy=self.sampling_strategy
        )

        # Add corrections (highest priority)
        corrections_in_sample = [d for d in sampled_data if isinstance(d, Correction)]
        if corrections_in_sample:
            prompt += f"CORRECTIONS (Learn from failures - {len(corrections_in_sample)} shown):\n"
            for corr in corrections_in_sample:
                prompt += f"\nInput: {json.dumps(corr.input)}\n"
                prompt += f"Got (WRONG): {json.dumps(corr.model_output)}\n"
                prompt += f"Expected (CORRECT): {json.dumps(corr.expected_output)}\n"
                if corr.feedback:
                    prompt += f"Feedback: {corr.feedback}\n"

        # Add examples
        examples_in_sample = [d for d in sampled_data if not isinstance(d, Correction)]
        if examples_in_sample:
            prompt += f"\nTRAINING EXAMPLES ({len(examples_in_sample)} shown):\n"
            for ex in examples_in_sample:
                prompt += f"\nInput: {json.dumps(ex.input)}\n"
                prompt += f"Output: {json.dumps(ex.expected_output)}\n"

        # Add feedback
        if dataset.feedback:
            prompt += "\n\nUSER FEEDBACK:\n"
            for fb in dataset.feedback:
                prompt += f"- {fb}\n"

        # Add existing rules for incremental learning
        if dataset.rules:
            prompt += f"\n\nEXISTING RULES ({len(dataset.rules)} current):\n"
            for rule in dataset.rules:
                success_rate = (
                    f"{rule.successes / rule.times_applied * 100:.1f}%"
                    if rule.times_applied > 0
                    else "untested"
                )
                prompt += f"\n- {rule.name} (priority {rule.priority}, success: {success_rate})\n"
                prompt += f"  Format: {rule.format.value}\n"
                prompt += f"  Pattern: {rule.content[:100]}{'...' if len(rule.content) > 100 else ''}\n"
                prompt += f"  Confidence: {rule.confidence:.2f}\n"

            prompt += "\nCONSIDER:\n"
            prompt += "- Refine existing high-performing rules\n"
            prompt += "- Fix or replace low-performing rules\n"
            prompt += "- Keep rules that work well\n"
            prompt += "- Add new rules for uncovered patterns\n"

        prompt += f"""

YOUR TASK:
{"Update and refine" if dataset.rules else "Synthesize a complete"} ruleset (max {max_rules} rules) that:
1. Handles all corrections correctly (CRITICAL - these show failure modes)
2. Works on all examples
3. Respects user feedback
4. Is general and minimal (avoid redundant rules)

RULES CAN BE:"""

        # Show allowed formats
        if RuleFormat.REGEX in self.allowed_formats:
            prompt += "\n- Regex patterns (for structured extraction)"
        if RuleFormat.CODE in self.allowed_formats:
            prompt += "\n- Python code (for complex logic)"
        if RuleFormat.SPACY in self.allowed_formats:
            prompt += "\n- spaCy token matcher patterns (for linguistic/NLP patterns)"

        prompt += "\n\nIMPORTANT: You must ONLY use the allowed formats listed above. Do NOT generate rules in other formats."
        prompt += "\nIMPORTANT: For CODE rules, write standard multi-line Python functions with proper indentation. Do NOT write one-liners."

        # Build format options for JSON schema
        format_options = ["regex"]
        content_options = ["regex pattern"]
        if RuleFormat.CODE in self.allowed_formats:
            format_options.append("code")
            content_options.append("python code")
        if RuleFormat.SPACY in self.allowed_formats:
            format_options.append("spacy")
            content_options.append("spaCy JSON pattern")

        prompt += f"""

Return JSON:
{{
  "analysis": "What patterns did you find? What went wrong in corrections?",
  "strategy": "Overall approach",
  "rules": [
    {{
      "name": "Short rule name",
      "description": "What this rule does",
      "format": "{'" or "'.join(format_options)}",
      "content": "{' OR '.join(content_options)}",
      "priority": 1-10 (higher = more important),
      "reasoning": "Why this rule is needed"
    }}
  ]
}}
"""

        if RuleFormat.CODE in self.allowed_formats:
            if dataset.task.type == TaskType.EXTRACTION:
                prompt += """
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
            elif dataset.task.type == TaskType.CLASSIFICATION:
                prompt += """
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
            else:
                prompt += """
For CODE format, provide a function that takes input dict and returns the transformed output:
```python
def extract(input_data):
    # input_data is dict with keys from input_schema
    # return transformed data (dict, string, etc)
    return input_data["text"].upper()
```
"""

        if RuleFormat.SPACY in self.allowed_formats:
            prompt += """
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
            if dataset.task.type == TaskType.EXTRACTION:
                prompt += """
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
            elif dataset.task.type == TaskType.CLASSIFICATION:
                prompt += """
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
            else:
                prompt += """
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

        prompt += """
Focus on learning from CORRECTIONS - they show exactly what went wrong!

IMPORTANT: Return ONLY valid JSON. Ensure:
- All strings use double quotes and are properly escaped
- All braces and brackets are balanced
- No trailing commas
- Response is complete (not truncated)
"""

        return prompt

    # ========================================
    # Rule Evaluation & Refinement
    # ========================================

    def evaluate_and_refine(
        self, rules: List[Rule], dataset: Dataset, max_iterations: int = 3
    ) -> tuple:
        """Evaluate rules and refine through agentic loop"""
        import time

        print(f"\n🔄 Refinement loop (max {max_iterations} iterations)")

        best_rules = rules
        best_accuracy = 0.0
        results = None

        for iteration in range(max_iterations):
            iter_num = iteration + 1
            print(f"[{iter_num}/{max_iterations}] Evaluating rules...")

            # Evaluate
            results = self._evaluate_rules(rules, dataset)
            accuracy = results["accuracy"]

            print(
                f"[{iter_num}/{max_iterations}] Accuracy: {accuracy:.1%} ({results['correct']}/{results['total']})"
            )

            if accuracy > best_accuracy:
                best_rules = rules
                best_accuracy = accuracy

            # Stop if good enough
            if accuracy >= 0.90:
                print("✓ Achieved 90%+ accuracy!")
                break

            # Refine based on failures
            if results["failures"]:
                print(
                    f"[{iter_num}/{max_iterations}] Refining based on {len(results['failures'])} failures..."
                )
                start = time.time()
                rules = self._refine_rules(rules, results["failures"], dataset)
                elapsed = time.time() - start
                if not rules:
                    print("⚠ Refinement failed, keeping best rules")
                    rules = best_rules
                else:
                    print(
                        f"[{iter_num}/{max_iterations}] Refined {len(rules)} rules ({elapsed:.1f}s)"
                    )
            else:
                print("✓ No failures to fix!")
                break

        return best_rules, {
            "accuracy": best_accuracy,
            "total": results["total"],
            "correct": int(best_accuracy * results["total"]),
        }

    def _evaluate_rules(self, rules: List[Rule], dataset: Dataset) -> Dict:
        """Test rules on all training data"""

        all_data = dataset.get_all_training_data()
        total = len(all_data)
        correct = 0
        failures = []

        for item in all_data:
            # Apply rules
            extracted = self._apply_rules(rules, item.input)
            expected = item.expected_output

            # Check correctness
            if self._outputs_match(extracted, expected, dataset.task.type):
                correct += 1
                if hasattr(item, "update_stats"):
                    # Update rule stats
                    for rule in rules:
                        rule.update_stats(True)
            else:
                failures.append(
                    {
                        "input": item.input,
                        "expected": expected,
                        "got": extracted,
                        "is_correction": isinstance(item, Correction),
                    }
                )
                if hasattr(item, "update_stats"):
                    for rule in rules:
                        rule.update_stats(False)

        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "failures": failures,
        }

    def _refine_rules(
        self, current_rules: List[Rule], failures: List[Dict], dataset: Dataset
    ) -> Optional[List[Rule]]:
        """Refine rules based on failures"""

        # Sample failures intelligently: ALL correction failures + sample of other failures
        sampled_failures = self._sample_failures(failures, max_samples=20)

        prompt = f"""You previously generated these rules:

{self._format_rules(current_rules)}

But they failed on these cases:
{json.dumps(sampled_failures, indent=2)}

Refine the ruleset to fix these failures while maintaining performance on other examples.

CRITICAL: Pay special attention to correction failures (is_correction: true) - these are user-verified mistakes.

Allowed rule formats: {", ".join(fmt.value for fmt in self.allowed_formats)}
"""

        if RuleFormat.CODE in self.allowed_formats:
            prompt += """
For CODE rules, return list of dicts: [{"text": "...", "start": 0, "end": 10}, ...]
"""

        prompt += """
Return refined ruleset in same JSON format:
{
  "reasoning": "Why these changes fix the failures",
  "rules": [...]
}
"""

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )

            result = self._parse_json(response.choices[0].message.content)

            # Extract refined rules
            rules = []
            for rule_data in result.get("rules", []):
                rule_format = RuleFormat(rule_data.get("format", "regex"))

                # Skip rules not in allowed formats
                if rule_format not in self.allowed_formats:
                    print(
                        f"   ⚠ Skipped {rule_format.value} rule (not allowed): {rule_data.get('name', 'Refined rule')}"
                    )
                    continue

                rule = Rule(
                    id=self._generate_id(),
                    name=rule_data.get("name", "Refined rule"),
                    description=rule_data.get("description", ""),
                    format=rule_format,
                    content=rule_data.get("content", ""),
                    priority=rule_data.get("priority", 5),
                )
                # Validate rule before adding
                if self._validate_rule(rule):
                    rules.append(rule)
                else:
                    print(f"   ⚠ Skipped invalid refined rule: {rule.name}")

            return rules if rules else None

        except Exception as e:
            print(f"Error refining rules: {e}")
            return None

    def _apply_rules(self, rules: List[Rule], input_data: Dict) -> Dict:
        """Apply rules to input and return output"""

        # Sort by priority
        sorted_rules = sorted(rules, key=lambda r: r.priority, reverse=True)

        # For Extraction: Aggregate spans from all rules
        # (This assumes rules are additive)
        if (
            not rules or rules[0].format == RuleFormat.REGEX
        ):  # Heuristic check or pass task_type?
            # Ideally we should pass task_type here, but for now let's infer from return type
            # Or better, just handle the list return type
            pass

        # We need to know the task type here, but _apply_rules signature doesn't have it.
        # However, we can infer from the result of the first successful rule.

        all_results = []
        for rule in sorted_rules:
            try:
                result = self._execute_rule(rule, input_data)
                if result is not None:
                    all_results.append(result)
            except Exception:
                pass

        if not all_results:
            return {}

        # Check type of first result to decide aggregation strategy
        first_result = all_results[0]

        if isinstance(first_result, list) and (
            not first_result or isinstance(first_result[0], Span)
        ):
            # It's a list of Spans (Extraction)
            all_spans = []
            for res in all_results:
                if isinstance(res, list):
                    all_spans.extend(res)

            unique_spans = self._deduplicate_spans(all_spans)
            return {"spans": [s.to_dict() for s in unique_spans]}

        elif isinstance(first_result, str):
            # Classification (Label) - Return highest priority match
            return {"label": first_result}

        else:
            # Transformation / Generic JSON - Return highest priority match
            return first_result

    def _execute_rule(self, rule: Rule, input_data: Dict) -> List[Span]:
        """Execute a single rule"""

        if rule.format == RuleFormat.REGEX:
            return self._execute_regex_rule(rule, input_data)
        elif rule.format == RuleFormat.CODE:
            return self._execute_code_rule(rule, input_data)
        elif rule.format == RuleFormat.SPACY:
            return self._execute_spacy_rule(rule, input_data)

        return []

    def _execute_regex_rule(self, rule: Rule, input_data: Dict) -> List[Span]:
        """Execute regex rule"""

        pattern = re.compile(rule.content)
        context = input_data.get("context", "")
        spans = []

        for match in pattern.finditer(context):
            spans.append(
                Span(
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    score=rule.confidence,
                )
            )

        return spans

    def _execute_code_rule(self, rule: Rule, input_data: Dict) -> Any:
        """Execute code rule"""

        try:
            namespace = {"Span": Span, "re": re}
            exec(rule.content, namespace)
            extract_func = namespace.get("extract")

            if extract_func:
                results = extract_func(input_data)
                return results
        except Exception:
            # Rules are validated during learning, so execution errors are rare
            # but silently continue to next rule
            pass

        return None

    def _execute_spacy_rule(self, rule: Rule, input_data: Dict) -> List[Span]:
        """Execute spaCy token matcher rule"""
        try:
            import spacy
            from spacy.matcher import Matcher

            # Lazy load spaCy model (cache on instance)
            if not hasattr(self, "_nlp"):
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # Model not installed, try to download
                    print("   ⚠ spaCy model not found, downloading en_core_web_sm...")
                    from spacy.cli import download
                    download("en_core_web_sm")
                    self._nlp = spacy.load("en_core_web_sm")

            # Parse the pattern (JSON array or list of patterns)
            pattern_data = json.loads(rule.content)
            
            # Handle both single pattern and list of patterns
            if isinstance(pattern_data, list) and pattern_data:
                if isinstance(pattern_data[0], dict):
                    # Single pattern: [{"LOWER": "hello"}, {"POS": "NOUN"}]
                    patterns = [pattern_data]
                else:
                    # Multiple patterns: [[pat1], [pat2]]
                    patterns = pattern_data
            else:
                return []

            # Create matcher and add patterns
            matcher = Matcher(self._nlp.vocab)
            matcher.add(rule.name, patterns)

            # Get text to match against
            text = input_data.get("context", input_data.get("text", ""))
            doc = self._nlp(text)

            # Find matches
            spans = []
            matches = matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                spans.append(
                    Span(
                        text=span.text,
                        start=span.start_char,
                        end=span.end_char,
                        score=rule.confidence,
                    )
                )

            return spans

        except ImportError:
            print("   ⚠ spaCy not installed. Run: pip install spacy")
            return []
        except Exception as e:
            # Silently continue to next rule on error
            return []

    def _deduplicate_spans(self, spans: List[Span]) -> List[Span]:
        """Remove overlapping spans"""

        if not spans:
            return []

        # Convert dicts to Span objects if needed
        span_objects = []
        for s in spans:
            if isinstance(s, dict):
                span_objects.append(
                    Span(
                        text=s["text"],
                        start=s["start"],
                        end=s["end"],
                        score=s.get("score", 0.5),
                    )
                )
            else:
                span_objects.append(s)

        sorted_spans = sorted(span_objects, key=lambda s: s.score, reverse=True)
        unique = []

        for span in sorted_spans:
            is_dup = False
            for existing in unique:
                if span.overlap_ratio(existing) > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(span)

        return unique[:5]

    def _outputs_match(
        self, output1: Dict, output2: Dict, task_type: TaskType = TaskType.EXTRACTION
    ) -> bool:
        """Check if two outputs match based on task type"""

        if task_type == TaskType.EXTRACTION:
            spans1 = output1.get("spans", [])
            spans2 = output2.get("spans", [])

            if len(spans1) != len(spans2):
                return False

            # Check text match (order independent)
            texts1 = sorted(
                [s["text"] if isinstance(s, dict) else s.text for s in spans1]
            )
            texts2 = sorted(
                [s["text"] if isinstance(s, dict) else s.text for s in spans2]
            )

            return texts1 == texts2

        elif task_type == TaskType.CLASSIFICATION:
            # Compare labels (case insensitive)
            label1 = str(output1.get("label", "")).lower().strip()
            label2 = str(output2.get("label", "")).lower().strip()
            return label1 == label2

        else:
            # Transformation / Other: Exact match
            return output1 == output2

    def _format_output(self, output: Dict) -> str:
        """Format output for prompt"""
        spans = output.get("spans", [])
        return json.dumps(
            [{"text": s["text"], "start": s["start"], "end": s["end"]} for s in spans]
        )

    def _format_rules(self, rules: List[Rule]) -> str:
        """Format rules for display"""
        formatted = []
        for i, rule in enumerate(rules, 1):
            formatted.append(f"{i}. {rule.name}")
            formatted.append(f"   Format: {rule.format.value}")
            formatted.append(f"   Priority: {rule.priority}")
            formatted.append(f"   Content: {rule.content[:100]}...")
        return "\n".join(formatted)

    def _parse_json(self, text: str) -> Dict:
        """Parse JSON from LLM response with error handling"""

        # Extract JSON from markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Log error with context
            preview = text[:300] + "..." if len(text) > 300 else text
            print(f"\n⚠️ JSON parsing error: {e}")
            print(f"Failed to parse: {preview}")
            raise

    def _generate_input(self, task, dataset, seed: int = 0) -> Dict:
        """Generate a synthetic input example"""

        prompt = f"""Generate a realistic training example for this task:

Task: {task.name}
Description: {task.description}
Input schema: {task.input_schema}
Output schema: {task.output_schema}

Return JSON:
{{
  "question": "...",
  "context": "..."
}}

Example #{seed + 1}:"""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.choices[0].message.content
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            return json.loads(text.strip())
        except:
            return {"question": "When?", "context": "In 1995"}

    def _validate_rule(self, rule: Rule) -> bool:
        """Validate that a rule's syntax is correct before saving"""
        try:
            if rule.format == RuleFormat.REGEX:
                # Test compile regex pattern
                re.compile(rule.content)
            elif rule.format == RuleFormat.CODE:
                # Test syntax of Python code
                compile(rule.content, "<string>", "exec")
                # Also check that extract function is defined
                if "def extract(" not in rule.content:
                    print("      Code rule must define extract() function")
                    return False
            elif rule.format == RuleFormat.SPACY:
                # Validate spaCy pattern JSON
                pattern_data = json.loads(rule.content)
                if not isinstance(pattern_data, list):
                    print("      spaCy pattern must be a JSON array")
                    return False
                if not pattern_data:
                    print("      spaCy pattern cannot be empty")
                    return False
                # Check it's a list of token dicts
                first = pattern_data[0]
                if not isinstance(first, dict):
                    # Could be list of patterns
                    if isinstance(first, list):
                        for pat in pattern_data:
                            if not all(isinstance(t, dict) for t in pat):
                                print("      spaCy pattern tokens must be dicts")
                                return False
                    else:
                        print("      spaCy pattern must be list of token dicts")
                        return False
            return True
        except re.error as e:
            print(f"      Regex error: {e}")
            return False
        except SyntaxError as e:
            print(f"      Python syntax error: {e}")
            print(f"      Content: {rule.content[:200]}...")
            return False
        except json.JSONDecodeError as e:
            print(f"      spaCy pattern JSON error: {e}")
            return False
        except Exception as e:
            print(f"      Validation error: {e}")
            return False

    def _generate_id(self) -> str:
        """Generate unique ID"""
        import uuid

        return str(uuid.uuid4())[:8]

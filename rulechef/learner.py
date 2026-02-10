"""LLM-based rule learning"""

import json
import re
import time
import uuid
from typing import Dict, List, Optional

from openai import OpenAI

from rulechef.core import Rule, RuleFormat, Dataset, Correction, TaskType
from rulechef.executor import RuleExecutor
from rulechef.matching import outputs_match
from rulechef.prompts import PromptBuilder


class RuleLearner:
    """Learns extraction rules from examples using LLM"""

    def __init__(
        self,
        llm: OpenAI,
        allowed_formats: Optional[List[RuleFormat]] = None,
        sampling_strategy: str = "balanced",
        model: str = "gpt-4o-mini",
        use_spacy_ner: bool = False,
        use_grex: bool = True,
    ):
        self.llm = llm
        self.allowed_formats = allowed_formats or [RuleFormat.REGEX, RuleFormat.CODE]
        self.sampling_strategy = sampling_strategy
        self.model = model
        self.use_spacy_ner = use_spacy_ner
        self.use_grex = use_grex
        self.executor = RuleExecutor(use_spacy_ner=use_spacy_ner)
        self.prompt_builder = PromptBuilder(
            self.allowed_formats,
            use_spacy_ner=use_spacy_ner,
            use_grex=use_grex,
        )

    # ========================================
    # Rule Execution (delegates to executor)
    # ========================================

    def _apply_rules(
        self,
        rules: List[Rule],
        input_data: Dict,
        task_type: Optional[TaskType] = None,
        text_field: Optional[str] = None,
    ) -> Dict:
        """Apply rules to input data. Delegates to executor."""
        return self.executor.apply_rules(rules, input_data, task_type, text_field)

    # ========================================
    # Rule Synthesis
    # ========================================

    def synthesize_ruleset(
        self,
        dataset: Dataset,
        max_rules: int = 10,
    ) -> List[Rule]:
        """Generate initial ruleset from dataset"""
        prompt = self._build_synthesis_prompt(dataset, max_rules)

        print("📚 Synthesizing rules from dataset...")
        start = time.time()

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                max_completion_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            result = self._parse_json(response.choices[0].message.content)
            rules = self._parse_rules_from_response(result, max_rules, dataset)

            elapsed = time.time() - start
            print(f"✓ Synthesized {len(rules)} rules ({elapsed:.1f}s)")
            return rules

        except Exception as e:
            print(f"Error synthesizing rules: {e}")
            return []

    def _parse_rules_from_response(
        self, result: Dict, max_rules: int, dataset: Optional[Dataset] = None
    ) -> List[Rule]:
        """Parse rules from LLM response"""
        from rulechef.core import is_pydantic_schema

        primary_key = None
        if dataset and dataset.task.output_schema:
            if is_pydantic_schema(dataset.task.output_schema):
                # For Pydantic, get first field name from model
                fields = list(dataset.task.output_schema.model_fields.keys())
                primary_key = fields[0] if fields else None
            else:
                primary_key = list(dataset.task.output_schema.keys())[0]

        rules = []
        for i, rule_data in enumerate(result.get("rules", [])[:max_rules]):
            raw_format = rule_data.get("format", "regex")
            # Accept common aliases
            if raw_format == "python":
                raw_format = "code"

            try:
                rule_format = RuleFormat(raw_format)
            except ValueError:
                print(
                    f"   ⚠ Skipped rule with unsupported format '{raw_format}': {rule_data.get('name', f'Rule {i + 1}')}"
                )
                continue

            if rule_format not in self.allowed_formats:
                print(
                    f"   ⚠ Skipped {rule_format.value} rule (not allowed): {rule_data.get('name', f'Rule {i + 1}')}"
                )
                continue

            pattern_content = rule_data.get("pattern") or rule_data.get("content", "")
            # Normalize structured content to JSON string
            if isinstance(pattern_content, (list, dict)):
                try:
                    pattern_content = json.dumps(pattern_content)
                except Exception:
                    pattern_content = str(pattern_content)

            output_template = rule_data.get("output_template")
            output_key = rule_data.get("output_key")

            # Ensure output_template is a dict (LLM might return string/list)
            if output_template and not isinstance(output_template, dict):
                print(
                    f"   ⚠ Skipped rule with invalid output_template (not a dict): {rule_data.get('name', f'Rule {i + 1}')}"
                )
                continue

            if dataset and dataset.task.type == TaskType.NER:
                if not output_template:
                    print(
                        f"   ⚠ Skipped rule without output_template: {rule_data.get('name', f'Rule {i + 1}')}"
                    )
                    continue
                if not output_key:
                    output_key = primary_key or "entities"

                # Validate label against schema if Pydantic
                valid_labels = dataset.task.get_labels()
                if valid_labels and output_template:
                    template_type = output_template.get("type")
                    if (
                        template_type
                        and template_type not in valid_labels
                        and not template_type.startswith("$")
                    ):
                        print(
                            f"   ⚠ Warning: Rule '{rule_data.get('name')}' uses unknown label '{template_type}'. Valid: {valid_labels}"
                        )
            elif output_template and not output_key and primary_key:
                output_key = primary_key
            elif dataset and dataset.task.type == TaskType.TRANSFORMATION:
                if rule_format != RuleFormat.CODE:
                    if not output_key and primary_key:
                        output_key = primary_key
                    if not output_key:
                        print(
                            f"   ⚠ Skipped transformation rule without output_key: {rule_data.get('name', f'Rule {i + 1}')}"
                        )
                        continue

            rule = Rule(
                id=self._generate_id(),
                name=rule_data.get("name", f"Rule {i + 1}"),
                description=rule_data.get("description", ""),
                format=rule_format,
                content=pattern_content,
                priority=rule_data.get("priority", 5),
                output_template=output_template,
                output_key=output_key,
            )

            if self._validate_rule(rule):
                # Ensure spaCy content is stored as JSON string for consistency
                if rule.format == RuleFormat.SPACY and not isinstance(
                    rule.content, str
                ):
                    rule.content = json.dumps(rule.content)
                rules.append(rule)
            else:
                print(f"   ⚠ Skipped invalid rule: {rule.name}")

        return rules

    # ========================================
    # Rule Evaluation & Refinement
    # ========================================

    def evaluate_and_refine(
        self, rules: List[Rule], dataset: Dataset, max_iterations: int = 3
    ) -> tuple:
        """Evaluate rules and refine through patch-based loop.

        Each iteration generates patch rules for failures and merges them
        into the existing set, keeping working rules intact.
        """
        print(f"\n🔄 Refinement loop (max {max_iterations} iterations)")

        best_rules = rules
        best_accuracy = 0.0
        results = None

        for iteration in range(max_iterations):
            iter_num = iteration + 1
            print(f"[{iter_num}/{max_iterations}] Evaluating rules...")

            results = self._evaluate_rules(rules, dataset)
            accuracy = results["accuracy"]

            print(
                f"[{iter_num}/{max_iterations}] Accuracy: {accuracy:.1%} ({results['correct']}/{results['total']})"
            )

            if accuracy > best_accuracy:
                best_rules = rules
                best_accuracy = accuracy

            if accuracy >= 0.90:
                print("✓ Achieved 90%+ accuracy!")
                break

            if results["failures"]:
                print(
                    f"[{iter_num}/{max_iterations}] Patching {len(results['failures'])} failures..."
                )
                start = time.time()
                patch = self.synthesize_patch_ruleset(
                    rules, results["failures"], max_rules=10, dataset=dataset
                )
                elapsed = time.time() - start
                if not patch:
                    print("⚠ Patch synthesis returned nothing, keeping best rules")
                else:
                    candidate = self._merge_patch(rules, patch)
                    candidate_results = self._evaluate_rules(candidate, dataset)
                    candidate_acc = candidate_results["accuracy"]
                    if candidate_acc >= accuracy:
                        rules = candidate
                        print(
                            f"[{iter_num}/{max_iterations}] Patched → {len(rules)} rules, "
                            f"accuracy {accuracy:.1%} → {candidate_acc:.1%} ({elapsed:.1f}s)"
                        )
                        if candidate_acc > best_accuracy:
                            best_rules = rules
                            best_accuracy = candidate_acc
                    else:
                        print(
                            f"[{iter_num}/{max_iterations}] Patch made it worse "
                            f"({accuracy:.1%} → {candidate_acc:.1%}), keeping previous"
                        )
            else:
                print("✓ No failures to fix!")
                break

        return best_rules, {
            "accuracy": best_accuracy,
            "total": results["total"],
            "correct": int(best_accuracy * results["total"]),
        }

    @staticmethod
    def _merge_patch(existing: List[Rule], patches: List[Rule]) -> List[Rule]:
        """Merge patch rules into existing set by name."""
        by_name = {r.name: r for r in existing}
        for pr in patches:
            if pr.name in by_name:
                current = by_name[pr.name]
                pr.times_applied = current.times_applied
                pr.successes = current.successes
                pr.failures = current.failures
                pr.confidence = current.confidence
            by_name[pr.name] = pr
        return list(by_name.values())

    def _evaluate_rules(self, rules: List[Rule], dataset: Dataset) -> Dict:
        """Test rules on all training data"""
        all_data = dataset.get_all_training_data()
        total = len(all_data)
        correct = 0
        failures = []

        for item in all_data:
            extracted = self._apply_rules(
                rules, item.input, dataset.task.type, dataset.task.text_field
            )
            expected = item.expected_output

            if outputs_match(
                expected,
                extracted,
                dataset.task.type,
                dataset.task.output_matcher,
                matching_mode=dataset.task.matching_mode,
            ):
                correct += 1
            else:
                failures.append(
                    {
                        "input": item.input,
                        "expected": expected,
                        "got": extracted,
                        "is_correction": isinstance(item, Correction),
                    }
                )

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
        sampled_failures = self._sample_failures(failures, max_samples=20)
        prompt = self.prompt_builder.build_refinement_prompt(
            current_rules, sampled_failures, dataset
        )

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            result = self._parse_json(response.choices[0].message.content)
            rules = self._parse_rules_from_response(
                result, max_rules=20, dataset=dataset
            )
            return rules if rules else None

        except Exception as e:
            print(f"Error refining rules: {e}")
            return None

    def _coerce_spacy_content(self, content) -> Optional[List]:
        """Coerce spaCy content into a list of patterns."""
        if isinstance(content, list):
            return content
        if isinstance(content, dict):
            return [content]
        if isinstance(content, str):
            text = content.strip()
            try:
                return json.loads(text)
            except Exception:
                # Try to extract the first JSON array substring
                if "[" in text and "]" in text:
                    try:
                        candidate = text[text.index("[") : text.rindex("]") + 1]
                        return json.loads(candidate)
                    except Exception:
                        return None
        return None

    def synthesize_patch_ruleset(
        self,
        current_rules: List[Rule],
        failures: List[Dict],
        max_rules: int = 10,
        dataset: Optional[Dataset] = None,
    ) -> List[Rule]:
        """
        Generate incremental rules targeted at specific failures.
        Returns only new/updated rules; caller merges with existing set.
        """
        sampled_failures = self._sample_failures(failures, max_samples=20)
        prompt = self._build_patch_prompt(current_rules, sampled_failures, max_rules)

        print("🩹 Synthesizing patch rules...")
        start = time.time()

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            result = self._parse_json(response.choices[0].message.content)
            rules = self._parse_rules_from_response(result, max_rules, dataset=dataset)

            elapsed = time.time() - start
            print(f"✓ Patch synthesis returned {len(rules)} rules ({elapsed:.1f}s)")
            return rules
        except Exception as e:
            print(f"Error synthesizing patch rules: {e}")
            return []

    # ========================================
    # Prompt Building
    # ========================================

    def _build_synthesis_prompt(
        self,
        dataset: Dataset,
        max_rules: int,
    ) -> str:
        """Build prompt for rule synthesis using PromptBuilder"""
        # Sample training data
        sampled_data = self._sample_training_data(
            dataset,
            max_samples=50,
            strategy=self.sampling_strategy,
        )

        # Separate corrections and examples
        corrections = [d for d in sampled_data if isinstance(d, Correction)]
        examples = [d for d in sampled_data if not isinstance(d, Correction)]

        # Build base prompt from builder
        prompt = self.prompt_builder._build_task_header(dataset)

        # Add training data sections
        if corrections:
            prompt += self.prompt_builder._build_corrections_section(corrections)
        if examples:
            prompt += self.prompt_builder._build_examples_section(examples)

        # Entity evidence helps the model infer labels/patterns when schemas don't encode them.
        prompt += self.prompt_builder._build_data_evidence(dataset)

        # Add other sections
        prompt += self.prompt_builder._build_feedback_section(dataset)
        prompt += self.prompt_builder._build_existing_rules_section(dataset)
        prompt += self.prompt_builder._build_task_instructions(dataset, max_rules)
        prompt += self.prompt_builder._build_format_instructions(dataset.task.type)
        prompt += self.prompt_builder._build_response_schema(dataset)
        prompt += self.prompt_builder._build_format_examples(dataset.task.type)
        prompt += self.prompt_builder._build_closing_instructions()

        return prompt

    def _build_patch_prompt(
        self,
        current_rules: List[Rule],
        failures: List[Dict],
        max_rules: int,
    ) -> str:
        """Build prompt for targeted patch rules."""
        rules_summary = [
            {
                "name": r.name,
                "description": r.description,
                "format": r.format.value,
                "priority": r.priority,
            }
            for r in current_rules[:10]
        ]

        failure_snippets = []
        for f in failures[:20]:
            failure_snippets.append(
                {
                    "input": f.get("input"),
                    "expected": f.get("expected"),
                    "got": f.get("got"),
                    "is_correction": f.get("is_correction", False),
                }
            )

        prompt = f"""You are updating an existing rule-based extractor. Do NOT rewrite good rules; add or adjust only what is needed.

CURRENT RULES (summary, top 10):
{json.dumps(rules_summary, indent=2)}

FAILURES TO FIX (sampled, corrections are high priority):
{json.dumps(failure_snippets, indent=2)}

Instructions:
- Add or tweak rules to fix the shown failures.
- Prefer minimal, local changes.
- Keep total new/updated rules <= {max_rules}.
- Use formats: {", ".join([f.value for f in self.allowed_formats])}
- Avoid touching unrelated behaviors.

Return JSON:
{{
  "analysis": "short reasoning",
  "rules": [
    {{
      "name": "rule name",
      "description": "what this rule fixes",
      "format": "regex|code|spacy",
      "content": "pattern or code",
      "priority": 1-10
    }}
  ]
}}
"""
        return prompt

    # ========================================
    # Smart Sampling
    # ========================================

    def _sample_training_data(
        self,
        dataset: Dataset,
        max_samples: int = 100,
        strategy: str = "balanced",
    ):
        """Intelligently sample training data for prompt inclusion."""
        samples = []

        # Priority 1: ALL corrections
        samples.extend(dataset.corrections)

        if len(samples) >= max_samples:
            return samples[:max_samples]

        remaining_budget = max_samples - len(samples)
        examples = dataset.examples

        if not examples:
            return samples[:max_samples]

        if strategy == "balanced":
            samples.extend(examples[:remaining_budget])
        elif strategy == "corrections_first" or strategy == "recent":
            samples.extend(
                sorted(examples, key=lambda e: e.timestamp, reverse=True)[
                    :remaining_budget
                ]
            )
        elif strategy == "diversity":
            if len(examples) <= remaining_budget:
                samples.extend(examples)
            else:
                step = len(examples) // remaining_budget
                samples.extend([examples[i * step] for i in range(remaining_budget)])
        elif strategy == "uncertain":
            sorted_by_confidence = sorted(
                examples, key=lambda e: e.confidence, reverse=False
            )
            samples.extend(sorted_by_confidence[:remaining_budget])
        elif strategy == "varied":
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

    def _sample_failures(
        self,
        failures: List[Dict],
        max_samples: int = 20,
    ):
        """Sample failures for refinement/patch, prioritizing corrections."""
        correction_failures = [f for f in failures if f.get("is_correction", False)]
        other_failures = [f for f in failures if not f.get("is_correction", False)]

        # Prioritize corrections
        sampled = list(correction_failures)
        remaining = max_samples - len(sampled)
        if remaining > 0:
            sampled.extend(other_failures[:remaining])
        return sampled[:max_samples]

    # ========================================
    # Utilities
    # ========================================

    def _parse_json(self, text: str) -> Dict:
        """Parse JSON from LLM response"""
        if isinstance(text, dict):
            return text

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            preview = text[:300] + "..." if len(text) > 300 else text
            print(f"\n⚠️ JSON parsing error: {e}")
            print(f"Failed to parse: {preview}")
            # Try to salvage by taking substring between first { and last }
            if "{" in text and "}" in text:
                try:
                    candidate = text[text.index("{") : text.rindex("}") + 1]
                    return json.loads(candidate)
                except Exception:
                    pass
            raise

    def _validate_rule(self, rule: Rule) -> bool:
        """Validate rule syntax"""
        try:
            if rule.format == RuleFormat.REGEX:
                re.compile(rule.content)
            elif rule.format == RuleFormat.CODE:
                compile(rule.content, "<string>", "exec")
                if "def extract(" not in rule.content:
                    print("      Code rule must define extract() function")
                    return False
            elif rule.format == RuleFormat.SPACY:
                pattern_data = self._coerce_spacy_content(rule.content)
                if not isinstance(pattern_data, list) or not pattern_data:
                    return False
                if not self.use_spacy_ner and self._pattern_uses_ent_type(pattern_data):
                    print(
                        "      spaCy NER is disabled; ENT_TYPE/ENT_ID patterns are not allowed"
                    )
                    return False
                rule.content = json.dumps(pattern_data)
            return True
        except Exception as e:
            print(f"      Validation error: {e}")
            return False

    def _generate_id(self) -> str:
        """Generate unique ID"""
        return str(uuid.uuid4())[:8]

    def _pattern_uses_ent_type(self, pattern_data: List) -> bool:
        """Detect spaCy patterns that rely on NER entity types."""

        def _walk(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    if k in ("ENT_TYPE", "ENT_ID"):
                        return True
                    if _walk(v):
                        return True
            elif isinstance(value, list):
                return any(_walk(item) for item in value)
            return False

        return _walk(pattern_data)

    def generate_synthetic_input(self, task, seed: int = 0) -> Dict:
        """Generate a synthetic input example"""
        prompt = self.prompt_builder.build_generation_prompt(task, seed)

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

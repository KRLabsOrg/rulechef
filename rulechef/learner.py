"""LLM-based rule learning"""

import json
import re
import time
import uuid
from typing import Dict, List, Optional

from openai import OpenAI

from rulechef.core import Rule, RuleFormat, Dataset, Correction
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
    ):
        self.llm = llm
        self.allowed_formats = allowed_formats or [RuleFormat.REGEX, RuleFormat.CODE]
        self.sampling_strategy = sampling_strategy
        self.model = model
        self.executor = RuleExecutor()
        self.prompt_builder = PromptBuilder(self.allowed_formats)

    # ========================================
    # Rule Execution (delegates to executor)
    # ========================================

    def _apply_rules(self, rules: List[Rule], input_data: Dict) -> Dict:
        """Apply rules to input data. Delegates to executor."""
        return self.executor.apply_rules(rules, input_data)

    # ========================================
    # Rule Synthesis
    # ========================================

    def synthesize_ruleset(self, dataset: Dataset, max_rules: int = 10) -> List[Rule]:
        """Generate initial ruleset from dataset"""
        prompt = self._build_synthesis_prompt(dataset, max_rules)

        print("📚 Synthesizing rules from dataset...")
        start = time.time()

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                max_completion_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )

            result = self._parse_json(response.choices[0].message.content)
            rules = self._parse_rules_from_response(result, max_rules)

            elapsed = time.time() - start
            print(f"✓ Synthesized {len(rules)} rules ({elapsed:.1f}s)")
            return rules

        except Exception as e:
            print(f"Error synthesizing rules: {e}")
            return []

    def _parse_rules_from_response(self, result: Dict, max_rules: int) -> List[Rule]:
        """Parse rules from LLM response"""
        rules = []
        for i, rule_data in enumerate(result.get("rules", [])[:max_rules]):
            rule_format = RuleFormat(rule_data.get("format", "regex"))

            if rule_format not in self.allowed_formats:
                print(
                    f"   ⚠ Skipped {rule_format.value} rule (not allowed): {rule_data.get('name', f'Rule {i + 1}')}"
                )
                continue

            pattern_content = rule_data.get("pattern") or rule_data.get("content", "")

            rule = Rule(
                id=self._generate_id(),
                name=rule_data.get("name", f"Rule {i + 1}"),
                description=rule_data.get("description", ""),
                format=rule_format,
                content=pattern_content,
                priority=rule_data.get("priority", 5),
                output_template=rule_data.get("output_template"),
                output_key=rule_data.get("output_key"),
            )

            if self._validate_rule(rule):
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
        """Evaluate rules and refine through agentic loop"""
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
                    f"[{iter_num}/{max_iterations}] Refining based on {len(results['failures'])} failures..."
                )
                start = time.time()
                rules = self._refine_rules(rules, results["failures"])
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
            extracted = self._apply_rules(rules, item.input)
            expected = item.expected_output

            if outputs_match(
                expected, extracted, dataset.task.type, dataset.task.output_matcher
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
        self, current_rules: List[Rule], failures: List[Dict]
    ) -> Optional[List[Rule]]:
        """Refine rules based on failures"""
        sampled_failures = self._sample_failures(failures, max_samples=20)
        prompt = self.prompt_builder.build_refinement_prompt(
            current_rules, sampled_failures
        )

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )

            result = self._parse_json(response.choices[0].message.content)
            rules = self._parse_rules_from_response(result, max_rules=20)
            return rules if rules else None

        except Exception as e:
            print(f"Error refining rules: {e}")
            return None

    # ========================================
    # Prompt Building
    # ========================================

    def _build_synthesis_prompt(self, dataset: Dataset, max_rules: int) -> str:
        """Build prompt for rule synthesis using PromptBuilder"""
        # Sample training data
        sampled_data = self._sample_training_data(
            dataset, max_samples=50, strategy=self.sampling_strategy
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

        # Add other sections
        prompt += self.prompt_builder._build_feedback_section(dataset)
        prompt += self.prompt_builder._build_existing_rules_section(dataset)
        prompt += self.prompt_builder._build_task_instructions(dataset, max_rules)
        prompt += self.prompt_builder._build_format_instructions(dataset.task.type)
        prompt += self.prompt_builder._build_response_schema(dataset)
        prompt += self.prompt_builder._build_format_examples(dataset.task.type)
        prompt += self.prompt_builder._build_closing_instructions()

        return prompt

    # ========================================
    # Smart Sampling
    # ========================================

    def _sample_training_data(
        self, dataset: Dataset, max_samples: int = 100, strategy: str = "balanced"
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

    def _sample_failures(self, failures: List[Dict], max_samples: int = 20):
        """Sample failures for refinement, prioritizing corrections."""
        correction_failures = [f for f in failures if f.get("is_correction", False)]
        other_failures = [f for f in failures if not f.get("is_correction", False)]
        sampled = correction_failures
        remaining = max_samples - len(sampled)
        if remaining > 0:
            sampled.extend(other_failures[:remaining])
        return sampled[:max_samples]

    # ========================================
    # Utilities
    # ========================================

    def _parse_json(self, text: str) -> Dict:
        """Parse JSON from LLM response"""
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
                pattern_data = json.loads(rule.content)
                if not isinstance(pattern_data, list) or not pattern_data:
                    return False
            return True
        except Exception as e:
            print(f"      Validation error: {e}")
            return False

    def _generate_id(self) -> str:
        """Generate unique ID"""
        return str(uuid.uuid4())[:8]

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

"""Rule execution - applies rules to input data"""

import json
import re
from typing import Dict, List, Any, Optional

from rulechef.core import Rule, RuleFormat, Span


def substitute_template(
    template: Dict[str, Any],
    match_text: str,
    start: int,
    end: int,
    groups: tuple = (),
    ent_type: Optional[str] = None,
    ent_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Substitute template variables with actual values from a match.

    Variables:
    - $0: Full match text
    - $1, $2, ...: Capture groups
    - $start: Start character offset
    - $end: End character offset
    - $ent_type: Entity type (spaCy only)
    - $ent_label: Entity label (spaCy only)
    """
    result = {}
    for key, value in template.items():
        if isinstance(value, str):
            # Substitute variables
            if value == "$0":
                result[key] = match_text
            elif value == "$start":
                result[key] = start
            elif value == "$end":
                result[key] = end
            elif value == "$ent_type" and ent_type:
                result[key] = ent_type
            elif value == "$ent_label" and ent_label:
                result[key] = ent_label
            elif value.startswith("$") and value[1:].isdigit():
                # Capture group reference
                group_idx = int(value[1:])
                if group_idx == 0:
                    result[key] = match_text
                elif groups and 0 < group_idx <= len(groups):
                    result[key] = groups[group_idx - 1] or ""
                else:
                    result[key] = ""
            else:
                # Literal string (may contain inline substitutions)
                substituted = value
                substituted = substituted.replace("$0", match_text)
                substituted = substituted.replace("$start", str(start))
                substituted = substituted.replace("$end", str(end))
                if ent_type:
                    substituted = substituted.replace("$ent_type", ent_type)
                if ent_label:
                    substituted = substituted.replace("$ent_label", ent_label)
                for i, g in enumerate(groups, 1):
                    substituted = substituted.replace(f"${i}", g or "")
                result[key] = substituted
        elif isinstance(value, dict):
            # Nested dict - recurse
            result[key] = substitute_template(
                value, match_text, start, end, groups, ent_type, ent_label
            )
        else:
            # Literal value (int, bool, etc.)
            result[key] = value
    return result


class RuleExecutor:
    """Executes rules against input data"""

    def __init__(self):
        self._nlp = None  # Lazy-loaded spaCy model

    def apply_rules(self, rules: List[Rule], input_data: Dict) -> Dict:
        """
        Apply rules to input and return output.

        For schema-aware rules (with output_key), aggregates results by key.
        For legacy rules (without output_key), uses type-based inference.
        """
        # Sort by priority
        sorted_rules = sorted(rules, key=lambda r: r.priority, reverse=True)

        # Check if any rules have output_key (schema-aware mode)
        has_schema_aware_rules = any(r.output_key for r in rules)

        if has_schema_aware_rules:
            return self._apply_rules_schema_aware(sorted_rules, input_data)
        else:
            return self._apply_rules_legacy(sorted_rules, input_data)

    def _apply_rules_schema_aware(self, rules: List[Rule], input_data: Dict) -> Dict:
        """Apply rules with schema-aware aggregation."""
        output = {}

        for rule in rules:
            try:
                results = self.execute_rule(rule, input_data)
                if not results:
                    continue

                if rule.output_key:
                    if rule.output_key not in output:
                        output[rule.output_key] = []

                    if isinstance(results, list):
                        output[rule.output_key].extend(results)
                    else:
                        output[rule.output_key].append(results)
                else:
                    # Legacy handling for rules without output_key
                    if (
                        isinstance(results, list)
                        and results
                        and isinstance(results[0], Span)
                    ):
                        if "spans" not in output:
                            output["spans"] = []
                        output["spans"].extend([s.to_dict() for s in results])
                    elif isinstance(results, str):
                        if "label" not in output:
                            output["label"] = results
                    elif isinstance(results, dict):
                        for k, v in results.items():
                            if k not in output:
                                output[k] = v
                            elif isinstance(output[k], list) and isinstance(v, list):
                                output[k].extend(v)
            except Exception:
                pass

        # Deduplicate array fields
        for key, value in output.items():
            if isinstance(value, list) and value:
                output[key] = self._deduplicate_dicts(value)

        return output

    def _apply_rules_legacy(self, rules: List[Rule], input_data: Dict) -> Dict:
        """Apply rules with legacy type-based aggregation."""
        all_results = []
        for rule in rules:
            try:
                result = self.execute_rule(rule, input_data)
                if result is not None:
                    all_results.append(result)
            except Exception:
                pass

        if not all_results:
            return {}

        first_result = all_results[0]

        if isinstance(first_result, list) and (
            not first_result or isinstance(first_result[0], Span)
        ):
            all_spans = []
            for res in all_results:
                if isinstance(res, list):
                    all_spans.extend(res)
            unique_spans = self._deduplicate_spans(all_spans)
            return {"spans": [s.to_dict() for s in unique_spans]}

        elif isinstance(first_result, str):
            return {"label": first_result}

        else:
            return first_result

    def execute_rule(self, rule: Rule, input_data: Dict) -> Any:
        """Execute a single rule"""
        if rule.format == RuleFormat.REGEX:
            return self._execute_regex_rule(rule, input_data)
        elif rule.format == RuleFormat.CODE:
            return self._execute_code_rule(rule, input_data)
        elif rule.format == RuleFormat.SPACY:
            return self._execute_spacy_rule(rule, input_data)
        return []

    def _execute_regex_rule(self, rule: Rule, input_data: Dict) -> Any:
        """Execute regex rule."""
        pattern = re.compile(rule.content)
        # Find the text field from input (use first string value)
        text = ""
        for v in input_data.values():
            if isinstance(v, str):
                text = v
                break

        results = []
        for match in pattern.finditer(text):
            match_text = match.group()
            start = match.start()
            end = match.end()
            groups = match.groups()

            if rule.output_template:
                templated = substitute_template(
                    rule.output_template,
                    match_text,
                    start,
                    end,
                    groups,
                )
                results.append(templated)
            else:
                results.append(
                    Span(
                        text=match_text,
                        start=start,
                        end=end,
                        score=rule.confidence,
                    )
                )

        return results

    def _execute_code_rule(self, rule: Rule, input_data: Dict) -> Any:
        """Execute code rule"""
        try:
            namespace = {"Span": Span, "re": re}
            exec(rule.content, namespace)
            extract_func = namespace.get("extract")

            if extract_func:
                return extract_func(input_data)
        except Exception:
            pass
        return None

    def _execute_spacy_rule(self, rule: Rule, input_data: Dict) -> Any:
        """Execute spaCy token matcher rule."""
        try:
            import spacy
            from spacy.matcher import Matcher

            # Lazy load spaCy model
            if self._nlp is None:
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    print("   ⚠ spaCy model not found, downloading en_core_web_sm...")
                    from spacy.cli import download

                    download("en_core_web_sm")
                    self._nlp = spacy.load("en_core_web_sm")

            pattern_data = json.loads(rule.content)

            if isinstance(pattern_data, list) and pattern_data:
                if isinstance(pattern_data[0], dict):
                    patterns = [pattern_data]
                else:
                    patterns = pattern_data
            else:
                return []

            matcher = Matcher(self._nlp.vocab)
            matcher.add(rule.name, patterns)

            # Get text (use first string value)
            text = ""
            for v in input_data.values():
                if isinstance(v, str):
                    text = v
                    break
            doc = self._nlp(text)

            # Build entity lookup
            char_to_ent = {}
            for ent in doc.ents:
                for i in range(ent.start_char, ent.end_char):
                    char_to_ent[i] = (ent.label_, ent.text)

            results = []
            matches = matcher(doc)
            for match_id, start_tok, end_tok in matches:
                span = doc[start_tok:end_tok]
                match_text = span.text
                start_char = span.start_char
                end_char = span.end_char

                ent_type = None
                ent_label = None
                if start_char in char_to_ent:
                    ent_type, ent_label = char_to_ent[start_char]

                if rule.output_template:
                    templated = substitute_template(
                        rule.output_template,
                        match_text,
                        start_char,
                        end_char,
                        groups=(),
                        ent_type=ent_type,
                        ent_label=ent_label,
                    )
                    results.append(templated)
                else:
                    results.append(
                        Span(
                            text=match_text,
                            start=start_char,
                            end=end_char,
                            score=rule.confidence,
                        )
                    )

            return results

        except ImportError:
            print("   ⚠ spaCy not installed. Run: pip install spacy")
            return []
        except Exception:
            return []

    def _deduplicate_dicts(
        self, items: List[Dict], overlap_threshold: float = 0.7
    ) -> List[Dict]:
        """Deduplicate a list of dicts based on text span overlap."""
        if not items:
            return []

        has_spans = all(
            isinstance(item, dict) and "start" in item and "end" in item
            for item in items
        )

        if not has_spans:
            seen = []
            for item in items:
                if item not in seen:
                    seen.append(item)
            return seen

        unique = []
        for item in items:
            is_dup = False
            item_type = item.get("type")
            item_start = item.get("start", 0)
            item_end = item.get("end", 0)

            for existing in unique:
                existing_type = existing.get("type")
                existing_start = existing.get("start", 0)
                existing_end = existing.get("end", 0)

                inter_start = max(item_start, existing_start)
                inter_end = min(item_end, existing_end)
                intersection = max(0, inter_end - inter_start)
                union = (
                    (item_end - item_start)
                    + (existing_end - existing_start)
                    - intersection
                )
                iou = intersection / union if union > 0 else 0

                if iou > overlap_threshold and item_type == existing_type:
                    is_dup = True
                    break

            if not is_dup:
                unique.append(item)

        return unique

    def _deduplicate_spans(self, spans: List[Span]) -> List[Span]:
        """Remove overlapping spans"""
        if not spans:
            return []

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

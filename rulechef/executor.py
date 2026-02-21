"""Rule execution - applies rules to input data"""

import json
import re
from typing import Any

from rulechef.core import DEFAULT_OUTPUT_KEYS, Rule, RuleFormat, Span, TaskType


def substitute_template(
    template: dict[str, Any],
    match_text: str,
    start: int,
    end: int,
    groups: tuple = (),
    ent_type: str | None = None,
    ent_label: str | None = None,
    token_spans: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Substitute template variables with actual values from a match.

    Args:
        template: Output template dict with variable placeholders.
        match_text: The full matched text ($0).
        start: Start character offset ($start).
        end: End character offset ($end).
        groups: Regex capture groups ($1, $2, ...).
        ent_type: spaCy entity type string ($ent_type).
        ent_label: spaCy entity label string ($ent_label).
        token_spans: Per-token span dicts for spaCy dependency matches
            ($1.text, $1.start, $1.end, etc.).

    Returns:
        Dict with all template variables replaced by their actual values.

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
                token_match = re.match(r"^\$(\d+)\.(start|end|text)$", value)
                if token_match:
                    token_idx = int(token_match.group(1))
                    token_attr = token_match.group(2)
                    if token_idx == 0:
                        if token_attr == "start":
                            result[key] = start
                        elif token_attr == "end":
                            result[key] = end
                        else:
                            result[key] = match_text
                    elif token_spans and 0 < token_idx <= len(token_spans):
                        token_value = token_spans[token_idx - 1].get(token_attr)
                        result[key] = token_value if token_value is not None else ""
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
                    if token_spans:
                        for i, token in enumerate(token_spans, 1):
                            substituted = substituted.replace(
                                f"${i}.start", str(token.get("start", ""))
                            )
                            substituted = substituted.replace(
                                f"${i}.end", str(token.get("end", ""))
                            )
                            substituted = substituted.replace(
                                f"${i}.text", str(token.get("text", ""))
                            )
                    result[key] = substituted
        elif isinstance(value, dict):
            # Nested dict - recurse
            result[key] = substitute_template(
                value,
                match_text,
                start,
                end,
                groups,
                ent_type,
                ent_label,
                token_spans,
            )
        else:
            # Literal value (int, bool, etc.)
            result[key] = value
    return result


class RuleExecutor:
    """Executes rules against input data"""

    def __init__(self, use_spacy_ner: bool = False):
        """Initialize the rule executor.

        Args:
            use_spacy_ner: If True, run spaCy's NER pipeline during rule
                execution so that ENT_TYPE/ENT_ID patterns are available.
        """
        self._nlp = None  # Lazy-loaded spaCy model
        self.use_spacy_ner = use_spacy_ner

    def apply_rules(
        self,
        rules: list[Rule],
        input_data: dict,
        task_type: TaskType | None = None,
        text_field: str | None = None,
    ) -> dict:
        """Apply rules to input and return aggregated output.

        Rules are sorted by priority and applied sequentially. Results are
        deduplicated by span position. If a rule has no output_key, it is
        inferred from task_type using DEFAULT_OUTPUT_KEYS.

        Args:
            rules: List of rules to apply.
            input_data: Input data dict.
            task_type: Task type for inferring the default output_key.
            text_field: Input key to use for regex/spaCy matching.

        Returns:
            Output dict with results keyed by output_key (e.g. 'entities',
            'spans', 'label'). Empty dict if no rules matched.
        """
        # Sort by priority
        sorted_rules = sorted(rules, key=self._rule_sort_key)

        # Get default output key for this task type
        default_key = (
            DEFAULT_OUTPUT_KEYS.get(task_type, "spans") if task_type else "spans"
        )

        output = {}

        for rule in sorted_rules:
            try:
                results = self.execute_rule(rule, input_data, text_field)
                if not results:
                    continue

                # Determine output key: use rule's key or default from task type
                output_key = rule.output_key or default_key

                # Handle classification specially (single label, not list)
                if task_type == TaskType.CLASSIFICATION:
                    label = self._normalize_label(results)
                    if label is not None and "label" not in output:
                        output["label"] = label
                    continue

                # For list results, aggregate into output_key
                if output_key:
                    if output_key not in output:
                        output[output_key] = []

                    if isinstance(results, list):
                        normalized = []
                        for res in results:
                            if isinstance(res, Span):
                                normalized.append(res.to_dict())
                            else:
                                normalized.append(res)
                        # Stamp rule attribution on each dict result
                        for res in normalized:
                            if isinstance(res, dict):
                                res["rule_id"] = rule.id
                                res["rule_name"] = rule.name
                        # Respect rule priority: skip duplicates by position
                        for res in normalized:
                            if (
                                isinstance(res, dict)
                                and "start" in res
                                and "end" in res
                                and any(
                                    isinstance(existing, dict)
                                    and existing.get("start") == res.get("start")
                                    and existing.get("end") == res.get("end")
                                    for existing in output[output_key]
                                )
                            ):
                                continue
                            output[output_key].append(res)
                    else:
                        normalized = (
                            results.to_dict() if isinstance(results, Span) else results
                        )
                        # Stamp rule attribution
                        if isinstance(normalized, dict):
                            normalized["rule_id"] = rule.id
                            normalized["rule_name"] = rule.name
                        if not (
                            isinstance(normalized, dict)
                            and "start" in normalized
                            and "end" in normalized
                            and any(
                                isinstance(existing, dict)
                                and existing.get("start") == normalized.get("start")
                                and existing.get("end") == normalized.get("end")
                                for existing in output[output_key]
                            )
                        ):
                            output[output_key].append(normalized)
                else:
                    # TRANSFORMATION: merge dict results directly
                    if isinstance(results, dict):
                        for k, v in results.items():
                            if k not in output:
                                output[k] = v
                            elif isinstance(output[k], list) and isinstance(v, list):
                                output[k].extend(v)
                    elif task_type == TaskType.TRANSFORMATION:
                        print(
                            f"   ⚠ Transformation rule '{rule.name}' returned list without output_key; skipping."
                        )
            except Exception:
                pass

        # Deduplicate array fields
        for key, value in output.items():
            if isinstance(value, list) and value:
                output[key] = self._deduplicate_dicts(value)

        return output

    def execute_rule(
        self, rule: Rule, input_data: dict, text_field: str | None = None
    ) -> Any:
        """Execute a single rule against input data.

        Args:
            rule: The rule to execute.
            input_data: Input data dict.
            text_field: Input key to use for regex/spaCy matching.

        Returns:
            List of Span or dict results for regex/spaCy rules, or
            arbitrary return value from code rules. Empty list on no match.
        """
        if rule.format == RuleFormat.REGEX:
            return self._execute_regex_rule(rule, input_data, text_field)
        elif rule.format == RuleFormat.CODE:
            return self._execute_code_rule(rule, input_data)
        elif rule.format == RuleFormat.SPACY:
            return self._execute_spacy_rule(rule, input_data, text_field)
        return []

    def _execute_regex_rule(
        self, rule: Rule, input_data: dict, text_field: str | None = None
    ) -> Any:
        """Execute regex rule."""
        pattern = re.compile(rule.content)
        text = self._select_text(input_data, text_field)

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

    _SAFE_BUILTINS: dict[str, Any] = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "filter": filter, "float": float,
        "frozenset": frozenset, "int": int, "isinstance": isinstance,
        "len": len, "list": list, "map": map, "max": max, "min": min,
        "print": print, "range": range, "reversed": reversed, "round": round,
        "set": set, "slice": slice, "sorted": sorted, "str": str, "sum": sum,
        "tuple": tuple, "type": type, "zip": zip,
        "True": True, "False": False, "None": None,
    }

    def _execute_code_rule(self, rule: Rule, input_data: dict) -> Any:
        """Execute code rule"""
        try:
            namespace = {"__builtins__": self._SAFE_BUILTINS, "Span": Span, "re": re}
            exec(rule.content, namespace)
            extract_func = namespace.get("extract")

            if extract_func:
                return extract_func(input_data)
        except Exception:
            pass
        return None

    def _execute_spacy_rule(
        self, rule: Rule, input_data: dict, text_field: str | None = None
    ) -> Any:
        """Execute spaCy token matcher rule."""
        try:
            import spacy
            from spacy.matcher import DependencyMatcher, Matcher

            # Lazy load spaCy model
            if self._nlp is None:
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    print("   ⚠ spaCy model not found, downloading en_core_web_sm...")
                    from spacy.cli import download

                    download("en_core_web_sm")
                    self._nlp = spacy.load("en_core_web_sm")

            # content may be stored as string or list
            if isinstance(rule.content, str):
                try:
                    pattern_data = json.loads(rule.content)
                except Exception:
                    # Attempt to extract first JSON array
                    if "[" in rule.content and "]" in rule.content:
                        candidate = rule.content[
                            rule.content.index("[") : rule.content.rindex("]") + 1
                        ]
                        pattern_data = json.loads(candidate)
                    else:
                        return []
            else:
                pattern_data = rule.content

            if isinstance(pattern_data, list) and pattern_data:
                if isinstance(pattern_data[0], dict):
                    patterns = [pattern_data]
                else:
                    patterns = pattern_data
            else:
                return []

            use_dependency = self._is_dependency_pattern(pattern_data)
            if use_dependency:
                matcher = DependencyMatcher(self._nlp.vocab)
                matcher.add(rule.name, patterns)
            else:
                matcher = Matcher(self._nlp.vocab)
                matcher.add(rule.name, patterns)

            text = self._select_text(input_data, text_field)
            if self.use_spacy_ner:
                doc = self._nlp(text)
            else:
                with self._nlp.disable_pipes("ner"):
                    doc = self._nlp(text)

            # Build entity lookup (optional, uses spaCy NER)
            char_to_ent = {}
            if self.use_spacy_ner:
                for ent in doc.ents:
                    for i in range(ent.start_char, ent.end_char):
                        char_to_ent[i] = (ent.label_, ent.text)

            results = []
            matches = matcher(doc)
            for match in matches:
                if use_dependency:
                    _, token_ids = match
                    if not token_ids:
                        continue
                    start_tok = min(token_ids)
                    end_tok = max(token_ids) + 1
                else:
                    _, start_tok, end_tok = match

                span = doc[start_tok:end_tok]
                match_text = span.text
                start_char = span.start_char
                end_char = span.end_char

                ent_type = None
                ent_label = None
                if start_char in char_to_ent:
                    ent_type, ent_label = char_to_ent[start_char]

                token_spans = None
                if use_dependency:
                    token_spans = [
                        {
                            "text": doc[i].text,
                            "start": doc[i].idx,
                            "end": doc[i].idx + len(doc[i].text),
                        }
                        for i in sorted(token_ids)
                    ]
                else:
                    token_spans = [
                        {
                            "text": token.text,
                            "start": token.idx,
                            "end": token.idx + len(token.text),
                        }
                        for token in doc[start_tok:end_tok]
                    ]

                if rule.output_template:
                    templated = substitute_template(
                        rule.output_template,
                        match_text,
                        start_char,
                        end_char,
                        groups=(),
                        ent_type=ent_type,
                        ent_label=ent_label,
                        token_spans=token_spans,
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

    def _select_text(self, input_data: dict, text_field: str | None) -> str:
        """Pick a string field for regex/spaCy matching."""
        if text_field:
            value = input_data.get(text_field)
            if isinstance(value, str):
                return value

        text_fields = [v for v in input_data.values() if isinstance(v, str)]
        return max(text_fields, key=len) if text_fields else ""

    def _rule_sort_key(self, rule: Rule) -> tuple:
        """Deterministic ordering: priority desc, then name, then id."""
        name_key = rule.name.casefold() if rule.name else ""
        id_key = rule.id or ""
        return (-rule.priority, name_key, id_key)

    def _is_dependency_pattern(self, pattern_data: list) -> bool:
        """Detect spaCy DependencyMatcher patterns."""
        for item in pattern_data:
            if isinstance(item, dict) and (
                "RIGHT_ID" in item or "LEFT_ID" in item or "REL_OP" in item
            ):
                return True
        return False

    def _deduplicate_dicts(
        self, items: list[dict], overlap_threshold: float = 0.7
    ) -> list[dict]:
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

        def _to_int(value, default=0):
            try:
                if isinstance(value, (int, float)):
                    return int(value)
                if isinstance(value, str):
                    return int(float(value))
            except Exception:
                return default
            return default

        unique = []
        for item in items:
            is_dup = False
            item_type = item.get("type") or item.get("label")
            item_start = _to_int(item.get("start", 0), default=0)
            item_end = _to_int(item.get("end", 0), default=0)

            for existing in unique:
                existing_type = existing.get("type") or existing.get("label")
                existing_start = _to_int(existing.get("start", 0), default=0)
                existing_end = _to_int(existing.get("end", 0), default=0)

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

    def _normalize_label(self, results: Any) -> str | None:
        """Normalize classification output to a single label string."""
        if isinstance(results, str):
            return results.strip() or None
        if isinstance(results, Span):
            return results.text.strip() or None
        if isinstance(results, dict):
            for key in ("label", "type", "text"):
                value = results.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None
        if isinstance(results, list) and results:
            return self._normalize_label(results[0])
        return None

"""Interactive CLI for RuleChef"""

import os

from openai import OpenAI

from rulechef.core import RuleFormat, Task, TaskType
from rulechef.engine import RuleChef

TASK_TYPES = {
    "extraction": TaskType.EXTRACTION,
    "ner": TaskType.NER,
    "classification": TaskType.CLASSIFICATION,
    "transformation": TaskType.TRANSFORMATION,
}

COMMANDS = {
    "add": "Add a training example",
    "correct": "Add a correction",
    "extract": "Run extraction on input",
    "learn": "Learn rules (--iterations N, --incremental, --agentic, --prune)",
    "evaluate": "Evaluate rules against dataset",
    "rules": "List learned rules (rules <id> for detail)",
    "delete": "Delete a rule by ID",
    "feedback": "Add feedback (task/rule level)",
    "generate": "Generate synthetic examples with LLM",
    "stats": "Show dataset statistics",
    "help": "Show commands",
    "quit": "Exit",
}


def _input(prompt: str, default: str = "") -> str:
    """Prompt for input with optional default."""
    if default:
        val = input(f"{prompt} [{default}]: ").strip()
        return val or default
    return input(f"{prompt}: ").strip()


def _build_output_schema(task_type: TaskType, labels: list[str]) -> dict:
    """Build output_schema dict for the given task type and labels."""
    if task_type == TaskType.NER:
        return {"entities": "List[{text, start, end, type}]"}
    if task_type == TaskType.CLASSIFICATION:
        return {"label": "str"}
    if task_type == TaskType.EXTRACTION:
        return {"spans": "List[{text, start, end}]"}
    # TRANSFORMATION — ask user
    return {}


def _setup() -> RuleChef:
    """Interactive task configuration wizard."""
    print("\nRuleChef Interactive CLI\n")

    name = _input("Task name")
    description = _input("Task description")

    # Task type
    while True:
        type_str = _input("Task type (extraction/ner/classification/transformation)").lower()
        if type_str in TASK_TYPES:
            task_type = TASK_TYPES[type_str]
            break
        print("  Invalid type. Choose: extraction, ner, classification, transformation")

    # Input fields
    input_fields_str = _input("Input fields (comma-separated, e.g. text)", "text")
    input_fields = [f.strip() for f in input_fields_str.split(",") if f.strip()]
    input_schema = {f: "str" for f in input_fields}

    # Labels (for NER and CLASSIFICATION)
    labels = []
    if task_type in (TaskType.NER, TaskType.CLASSIFICATION):
        labels_str = _input("Labels (comma-separated)")
        labels = [label.strip() for label in labels_str.split(",") if label.strip()]

    # Output schema
    if task_type == TaskType.TRANSFORMATION:
        output_fields_str = _input("Output fields (comma-separated, e.g. company,amount)")
        output_fields = [f.strip() for f in output_fields_str.split(",") if f.strip()]
        output_schema = {f: "str" for f in output_fields}
    else:
        output_schema = _build_output_schema(task_type, labels)

    # Model configuration
    model = _input("Model", "gpt-4o-mini")
    base_url = input("Base URL (blank for OpenAI): ").strip() or None

    # API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = _input("API key")

    # Dataset name
    dataset_name = name.lower().replace(" ", "_")

    # Build objects
    task = Task(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        type=task_type,
    )

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    chef = RuleChef(
        task,
        client,
        dataset_name=dataset_name,
        model=model,
        allowed_formats=[RuleFormat.REGEX],
    )

    print(f"\n✓ Configured: {name} ({type_str})")
    if labels:
        print(f"  Labels: {', '.join(labels)}")
    print(f"  Input: {', '.join(input_fields)}")
    print(f"  Model: {model}")

    # Store labels and input_fields on chef for CLI use
    chef._cli_labels = labels
    chef._cli_input_fields = input_fields

    return chef


# --- Input helpers by task type ---


def _read_input_fields(chef: RuleChef) -> dict:
    """Read input field values from user."""
    fields = getattr(chef, "_cli_input_fields", list(chef.task.input_schema.keys()))
    if len(fields) == 1:
        return {fields[0]: _input(fields[0].capitalize())}
    data = {}
    for f in fields:
        data[f] = _input(f.capitalize())
    return data


def _read_classification_output(chef: RuleChef) -> dict:
    """Read classification label."""
    labels = getattr(chef, "_cli_labels", [])
    if labels:
        print(f"  Labels: {', '.join(labels)}")
    label = _input("Label")
    return {"label": label}


def _read_ner_output(chef: RuleChef, text: str) -> dict:
    """Read NER entities interactively."""
    labels = getattr(chef, "_cli_labels", [])
    entities = []
    print("Enter entities (blank text to finish):")
    while True:
        ent_text = input("  Entity text: ").strip()
        if not ent_text:
            break
        start = text.find(ent_text)
        if start == -1:
            print(f"    '{ent_text}' not found in text, enter positions manually.")
            start = int(_input("    Start"))
            end = int(_input("    End"))
        else:
            end = start + len(ent_text)
            print(f"    Found at [{start}:{end}]")

        if labels:
            label = _input(f"  Label ({'/'.join(labels)})")
        else:
            label = _input("  Label")
        entities.append({"text": ent_text, "start": start, "end": end, "type": label})
    return {"entities": entities}


def _read_extraction_output(chef: RuleChef, text: str) -> dict:
    """Read extraction spans interactively."""
    spans = []
    print("Enter spans (blank text to finish):")
    while True:
        span_text = input("  Span text: ").strip()
        if not span_text:
            break
        start = text.find(span_text)
        if start == -1:
            print(f"    '{span_text}' not found in text, enter positions manually.")
            start = int(_input("    Start"))
            end = int(_input("    End"))
        else:
            end = start + len(span_text)
            print(f"    Found at [{start}:{end}]")
        spans.append({"text": span_text, "start": start, "end": end})
    return {"spans": spans}


def _read_transformation_output(chef: RuleChef) -> dict:
    """Read transformation output fields."""
    output = {}
    for key in chef.task.output_schema:
        if isinstance(chef.task.output_schema, dict):
            output[key] = _input(f"  {key}")
    return output


def _read_output(chef: RuleChef, input_data: dict) -> dict:
    """Read expected output based on task type."""
    task_type = chef.task.type
    # Get the primary text field for span-based tasks
    text = ""
    if task_type in (TaskType.NER, TaskType.EXTRACTION):
        text_field = chef.task.text_field
        if not text_field:
            fields = getattr(chef, "_cli_input_fields", list(chef.task.input_schema.keys()))
            text_field = fields[0]
        text = input_data.get(text_field, "")

    if task_type == TaskType.CLASSIFICATION:
        return _read_classification_output(chef)
    elif task_type == TaskType.NER:
        return _read_ner_output(chef, text)
    elif task_type == TaskType.EXTRACTION:
        return _read_extraction_output(chef, text)
    elif task_type == TaskType.TRANSFORMATION:
        return _read_transformation_output(chef)
    return {}


# --- Command handlers ---


def _cmd_add(chef: RuleChef):
    """Add a training example."""
    print("\n--- Add Example ---")
    input_data = _read_input_fields(chef)
    print("\nExpected output:")
    output_data = _read_output(chef, input_data)
    chef.add_example(input_data, output_data)
    print("✓ Example added")


def _cmd_correct(chef: RuleChef):
    """Add a correction by running extract first."""
    print("\n--- Add Correction ---")
    input_data = _read_input_fields(chef)

    print("\nRunning extraction...")
    result = chef.extract(input_data)
    print(f"Model output: {result}")

    correct = input("\nIs this correct? (y/n): ").strip().lower()
    if correct == "y":
        print("No correction needed.")
        return

    print("\nEnter expected output:")
    expected = _read_output(chef, input_data)
    feedback_text = input("Feedback (optional): ").strip() or None
    chef.add_correction(input_data, result, expected, feedback_text)
    print("✓ Correction added")


def _cmd_extract(chef: RuleChef):
    """Run extraction on input."""
    print("\n--- Extract ---")
    input_data = _read_input_fields(chef)
    result = chef.extract(input_data)

    print(f"\nResult: {_format_output(chef, result)}")

    # Offer to correct
    correct = input("\nCorrect? (y/n, blank to skip): ").strip().lower()
    if correct == "n":
        print("\nEnter expected output:")
        expected = _read_output(chef, input_data)
        feedback_text = input("Feedback (optional): ").strip() or None
        chef.add_correction(input_data, result, expected, feedback_text)
        print("✓ Correction added")


def _format_output(chef: RuleChef, output: dict) -> str:
    """Format output for display."""
    task_type = chef.task.type
    if task_type == TaskType.CLASSIFICATION:
        return f"label: {output.get('label', '?')}"
    elif task_type == TaskType.NER:
        entities = output.get("entities", [])
        if not entities:
            return "(no entities)"
        lines = []
        for e in entities:
            lines.append(
                f"  [{e.get('start', '?')}:{e.get('end', '?')}] "
                f"{e.get('text', '?')} → {e.get('type', '?')}"
            )
        return "\n".join(lines)
    elif task_type == TaskType.EXTRACTION:
        spans = output.get("spans", [])
        if not spans:
            return "(no spans)"
        lines = []
        for s in spans:
            lines.append(f"  [{s.get('start', '?')}:{s.get('end', '?')}] {s.get('text', '?')}")
        return "\n".join(lines)
    else:
        return str(output)


def _cmd_learn(chef: RuleChef, args: list[str]):
    """Learn rules with optional flags."""
    kwargs = {}
    _use_agentic = False
    _use_prune = False

    # Parse flags
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--iterations" and i + 1 < len(args):
            kwargs["max_refinement_iterations"] = int(args[i + 1])
            i += 2
        elif arg == "--incremental":
            kwargs["incremental_only"] = True
            i += 1
        elif arg == "--agentic":
            _use_agentic = True
            i += 1
        elif arg == "--prune":
            _use_prune = True
            i += 1
        elif arg == "--no-eval":
            kwargs["run_evaluation"] = False
            i += 1
        else:
            i += 1

    # Set up agentic coordinator if requested
    if _use_agentic:
        from rulechef.coordinator import AgenticCoordinator

        chef.coordinator = AgenticCoordinator(
            llm_client=chef.llm,
            model=chef.model,
            prune_after_learn=_use_prune,
        )
        print(f"Using agentic coordinator{' with pruning' if _use_prune else ''}")
    elif _use_prune:
        # Enable pruning on existing agentic coordinator
        if hasattr(chef.coordinator, "prune_after_learn"):
            chef.coordinator.prune_after_learn = True
            print("Pruning enabled")
        else:
            from rulechef.coordinator import AgenticCoordinator

            chef.coordinator = AgenticCoordinator(
                llm_client=chef.llm,
                model=chef.model,
                prune_after_learn=True,
            )
            print("Using agentic coordinator with pruning")

    # Default: enable evaluation
    if "run_evaluation" not in kwargs:
        kwargs["run_evaluation"] = True

    stats = chef.get_stats()
    total = stats["corrections"] + stats["examples"]
    buffer_stats = chef.get_buffer_stats()
    buffered = buffer_stats["new_examples"] + buffer_stats["new_corrections"]

    print(f"\nDataset: {total} items, Buffer: {buffered} new items")
    result = chef.learn_rules(**kwargs)

    if result is None:
        print("Not enough data to learn.")
        return

    rules, eval_result = result
    print(f"\n✓ Learned {len(rules)} rule(s)")
    if eval_result:
        print(f"  Precision: {eval_result.micro_precision:.2f}")
        print(f"  Recall:    {eval_result.micro_recall:.2f}")
        print(f"  F1:        {eval_result.micro_f1:.2f}")


def _cmd_evaluate(chef: RuleChef):
    """Evaluate rules against dataset."""
    if not chef.dataset.rules:
        print("\nNo rules to evaluate.")
        return
    result = chef.evaluate(verbose=True)
    print("\n--- Evaluation ---")
    print(f"  Docs:      {result.total_docs}")
    print(f"  Precision: {result.micro_precision:.2f}")
    print(f"  Recall:    {result.micro_recall:.2f}")
    print(f"  F1:        {result.micro_f1:.2f}")
    print(f"  Exact:     {result.exact_match:.2f}")
    if result.per_class:
        print("  Per class:")
        for c in result.per_class:
            print(f"    {c.label}: P={c.precision:.2f} R={c.recall:.2f} F1={c.f1:.2f}")


def _cmd_rules(chef: RuleChef, args: list[str]):
    """List rules or show detail for a specific rule."""
    if not chef.dataset.rules:
        print("\nNo rules learned yet.")
        return

    # Show specific rule by ID or index
    if args:
        target = args[0]
        for rule in chef.dataset.rules:
            if rule.id == target or rule.name == target:
                _print_rule_detail(rule)
                return
        # Try by index
        try:
            idx = int(target) - 1
            if 0 <= idx < len(chef.dataset.rules):
                _print_rule_detail(chef.dataset.rules[idx])
                return
        except ValueError:
            pass
        print(f"Rule '{target}' not found.")
        return

    # List all
    print("\n--- Rules ---\n")
    for i, summary in enumerate(chef.get_rules_summary(), 1):
        print(f"  {i}. {summary['name']}")
        print(f"     {summary['description']}")
        print(
            f"     Format: {summary['format']}, Priority: {summary['priority']}, "
            f"Confidence: {summary['confidence']}, Success: {summary['success_rate']}"
        )


def _print_rule_detail(rule):
    """Print full detail for a single rule."""
    print(f"\n--- Rule: {rule.name} ---")
    print(f"  ID:          {rule.id}")
    print(f"  Format:      {rule.format.value}")
    print(f"  Priority:    {rule.priority}")
    print(f"  Confidence:  {rule.confidence:.2f}")
    print(f"  Applied:     {rule.times_applied} ({rule.successes} ok, {rule.failures} fail)")
    print(f"  Description: {rule.description}")
    if rule.output_template:
        print(f"  Template:    {rule.output_template}")
    if rule.output_key:
        print(f"  Output key:  {rule.output_key}")
    print(f"  Content:\n{rule.content}")


def _cmd_delete(chef: RuleChef, args: list[str]):
    """Delete a rule by ID or index."""
    if not args:
        print("Usage: delete <rule_id or index>")
        return

    target = args[0]

    # Try by index first
    try:
        idx = int(target) - 1
        if 0 <= idx < len(chef.dataset.rules):
            rule = chef.dataset.rules[idx]
            if chef.delete_rule(rule.id):
                print(f"✓ Deleted rule: {rule.name}")
                return
    except ValueError:
        pass

    # Try by ID
    if chef.delete_rule(target):
        print(f"✓ Deleted rule: {target}")
    else:
        print(f"Rule '{target}' not found.")


def _cmd_feedback(chef: RuleChef):
    """Add feedback at task or rule level."""
    print("\n--- Add Feedback ---")
    print("  Levels: task, rule")
    level = _input("Level", "task")

    target_id = ""
    if level == "rule":
        # Show rules for reference
        if chef.dataset.rules:
            for i, r in enumerate(chef.dataset.rules, 1):
                print(f"  {i}. [{r.id[:8]}] {r.name}")
        target_id = _input("Rule ID")

    text = _input("Feedback")
    chef.add_feedback(text, level=level, target_id=target_id)
    print("✓ Feedback added")


def _cmd_generate(chef: RuleChef):
    """Generate synthetic examples with LLM."""
    num = int(_input("How many examples", "5"))
    print(f"Generating {num} examples...")
    chef.generate_llm_examples(num)
    print(f"✓ Generated {num} examples")


def _cmd_stats(chef: RuleChef):
    """Show dataset and buffer statistics."""
    stats = chef.get_stats()
    buffer_stats = chef.get_buffer_stats()

    print("\n--- Dataset ---")
    print(f"  Task:        {stats['task']}")
    print(f"  Examples:    {stats['examples']}")
    print(f"  Corrections: {stats['corrections']}")
    print(f"  Feedback:    {stats['feedback']}")
    print(f"  Rules:       {stats['rules']}")

    print("\n--- Buffer ---")
    print(f"  New examples:    {buffer_stats['new_examples']}")
    print(f"  New corrections: {buffer_stats['new_corrections']}")
    print(f"  LLM observations: {buffer_stats['llm_observations']}")
    print(f"  Total:           {buffer_stats['total_examples']}")


def _cmd_help():
    """Print available commands."""
    print("\nCommands:")
    for cmd, desc in COMMANDS.items():
        print(f"  {cmd:<12} {desc}")


# --- Main loop ---


ALIASES = {
    "a": "add",
    "c": "correct",
    "e": "extract",
    "l": "learn",
    "r": "rules",
    "d": "delete",
    "f": "feedback",
    "g": "generate",
    "s": "stats",
    "h": "help",
    "q": "quit",
    "exit": "quit",
}


def main():
    """Main CLI entry point."""
    try:
        chef = _setup()
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
        return

    _cmd_help()

    while True:
        try:
            line = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        # Resolve aliases
        cmd = ALIASES.get(cmd, cmd)

        try:
            if cmd == "add":
                _cmd_add(chef)
            elif cmd == "correct":
                _cmd_correct(chef)
            elif cmd == "extract":
                _cmd_extract(chef)
            elif cmd == "learn":
                _cmd_learn(chef, args)
            elif cmd == "evaluate":
                _cmd_evaluate(chef)
            elif cmd == "rules":
                _cmd_rules(chef, args)
            elif cmd == "delete":
                _cmd_delete(chef, args)
            elif cmd == "feedback":
                _cmd_feedback(chef)
            elif cmd == "generate":
                _cmd_generate(chef)
            elif cmd == "stats":
                _cmd_stats(chef)
            elif cmd == "help":
                _cmd_help()
            elif cmd == "quit":
                print("Bye!")
                break
            else:
                print(f"Unknown command: {cmd}. Type 'help' for commands.")
        except KeyboardInterrupt:
            print("\n(interrupted)")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()

import os
import signal

from openai import OpenAI

from benchmarks.data import DataSplit
from benchmarks.schemas import NEROutput
from rulechef.coordinator import AgenticCoordinator
from rulechef.core import RuleFormat, Task, TaskType
from rulechef.engine import RuleChef


def patch_regex_timeout(executor, timeout_secs: int = 5) -> None:
    """Skip regex rules that hang instead of letting them stall the run."""
    original = executor._execute_regex_rule
    timed_out_rules: set[str] = set()

    def _execute_with_timeout(rule, input_data, text_field=None):
        if rule.id in timed_out_rules:
            return []

        def _handler(signum, frame):
            raise TimeoutError()

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_secs)
        try:
            return original(rule, input_data, text_field)
        except TimeoutError:
            timed_out_rules.add(rule.id)
            print(f"   ⚠ Regex timeout ({timeout_secs}s): {rule.name} — skipping")
            return []
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    executor._execute_regex_rule = _execute_with_timeout


def build_chef(args, split: DataSplit, storage_dir: str, logger=None) -> RuleChef:
    selected_classes = sorted(split.selected_classes)
    task = Task(
        name="German Legal Named Entity Recognition",
        description=(
            f"Recognize named entities in German legal text. "
            f"Entities to look for: {', '.join(selected_classes)}."
        ),
        input_schema={"text": "str"},
        output_schema=NEROutput,
        type=TaskType.NER,
        text_field="text",
    )

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
        or "EMPTY",
    )

    coordinator = None
    if args.agentic:
        coordinator = AgenticCoordinator(
            client,
            model=args.model,
            prune_after_learn=args.enable_prune,
            audit_interval=args.audit_interval,
            enable_critic=args.enable_critic,
            critic_interval=args.critic_interval,
            verbose=True,
        )

    chef = RuleChef(
        task=task,
        client=client,
        dataset_name=split.name,
        model=args.model,
        allowed_formats=[RuleFormat.REGEX],
        use_grex=not args.no_grex,
        max_rules=args.max_rules,
        max_samples=args.max_samples,
        max_counter_examples=args.max_counter_examples,
        coordinator=coordinator,
        training_logger=logger,
        storage_path=storage_dir,
        sampling_strategy=args.sampling_strategy,
        synthesis_strategy=args.synthesis_strategy,
    )
    patch_regex_timeout(chef.learner.executor)
    return chef

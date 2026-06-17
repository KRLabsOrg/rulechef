# Executor

Rule execution engine. Rules run deterministically at inference time with no
LLM calls.

When several rules produce overlapping or conflicting matches, the executor
orders them by **priority, then validated precision** (falling back to
confidence when a rule has no validated estimate), so the higher-trust rule
wins. Each rule also runs under a wall-clock budget (`RULE_TIMEOUT_S`): a
catastrophically-backtracking regex is bounded via the `regex` module's
matching timeout, and a runaway code rule is interrupted with a signal alarm,
so a single bad LLM-written rule cannot freeze evaluation.

## RuleExecutor

::: rulechef.executor.RuleExecutor
    options:
      members:
        - __init__
        - apply_rules
        - execute_rule

## Functions

### substitute_template

::: rulechef.executor.substitute_template

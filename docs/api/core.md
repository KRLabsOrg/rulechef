# Core Types

Data structures used throughout RuleChef.

## TaskType

::: rulechef.core.TaskType

## RuleFormat

::: rulechef.core.RuleFormat

## Task

::: rulechef.core.Task
    options:
      members:
        - __init__
        - get_labels
        - validate_output
        - get_schema_for_prompt
        - to_dict

## Rule

::: rulechef.core.Rule
    options:
      members:
        - __init__
        - pattern
        - update_stats
        - to_dict

## Span

::: rulechef.core.Span
    options:
      members:
        - __init__
        - overlaps
        - overlap_ratio
        - to_dict

## Dataset

::: rulechef.core.Dataset
    options:
      members:
        - __init__
        - get_all_training_data
        - get_feedback_for
        - to_dict

## Example

::: rulechef.core.Example

## Correction

::: rulechef.core.Correction

## Feedback

::: rulechef.core.Feedback

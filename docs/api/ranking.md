# Ranking

Per-rule and ensemble evaluation, validated-trust statistics, and pruning.

After learning, each rule is evaluated in isolation and in the context of the
full ruleset so that overlapping rules can be ordered by measured trust and
unhelpful rules can be removed. See
[Rule Trust and Conflict Resolution](../guide/evaluation.md#rule-trust-and-conflict-resolution)
for the conceptual overview.

## rank_rules

::: rulechef.ranking.rank_rules

## prune_harmful_rules

::: rulechef.ranking.prune_harmful_rules

## wilson_lower_bound

::: rulechef.ranking.wilson_lower_bound

## RankingReport

::: rulechef.ranking.RankingReport
    options:
      members:
        - to_dict

## RuleRanking

::: rulechef.ranking.RuleRanking
    options:
      members:
        - to_dict

## print_ranking_report

::: rulechef.ranking.print_ranking_report

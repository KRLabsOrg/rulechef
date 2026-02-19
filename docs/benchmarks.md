# Benchmarks

## Banking77 Intent Classification

To measure how well RuleChef performs on a real task, we benchmarked on a subset of the [Banking77](https://huggingface.co/datasets/legacy-datasets/banking77) intent classification dataset — 77 banking customer service intent classes with ~13K examples.

### Setup

- **5 classes** pinned: `beneficiary_not_allowed`, `card_arrival`, `disposable_card_limits`, `exchange_rate`, `pending_cash_withdrawal`
- **5-shot per class** (25 training examples total)
- **Dev set**: remaining ~660 unused training examples (for refinement)
- **Test set**: 200 held-out examples from the official test split (never seen during learning)
- **Regex-only** rules (no code, no spaCy)
- **Agentic coordinator** guiding 15 refinement iterations
- **Model**: Kimi K2 via Groq API

### Results on Held-Out Test Set

| Metric | Value |
|--------|-------|
| Accuracy (exact match) | **60.5%** |
| Micro Precision | **100%** |
| Micro Recall | 60.5% |
| Micro F1 | **75.4%** |
| Macro F1 | **71.7%** |
| Coverage | 60.5% (121/200) |
| Rules learned | 108 |
| Learning time | ~144s |
| Per-query latency | **0.19ms** |

### Per-Class Breakdown

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| exchange_rate | 100% | 95% | 97% |
| pending_cash_withdrawal | 100% | 82% | 90% |
| card_arrival | 100% | 62% | 77% |
| disposable_card_limits | 100% | 40% | 57% |
| beneficiary_not_allowed | 100% | 22% | 37% |

### Sample Rules

Here are a few of the 108 regex rules RuleChef learned (full set in [`benchmarks/results_banking77.json`](https://github.com/KRLabsOrg/rulechef/blob/main/benchmarks/results_banking77.json)):

```
exchange_rate_keywords       (?i)\bexchange\s+rates?\b
track_card_delivery          (?i)\b(?:track|delivery|status|arrival|come|received).*\bcard\b
cash_withdrawal_pending      (?i)\b(?:cash|withdrawal|atm).*\b(?:pending|still|waiting)\b
disposable_limit_keywords    (?i)\bdisposable\s+cards?\b(?=.*\b(?:maximum|limit|how many)\b)
beneficiary_ultra_broad      (?i)\bbeneficiar(?:y|ies)\b.*\b(?:not allowed|fail|denied|can't)\b
```

### Key Takeaways

1. **Precision is perfect** — zero false positives across all classes. In production, wrong answers are worse than no answer, and rules never give wrong answers.

2. **Recall scales with complexity**. Simple keyword patterns (`exchange_rate` at 95%) are easy; nuanced paraphrases (`beneficiary_not_allowed` at 22%) need more examples or refinement iterations.

3. **Zero runtime cost**. After learning, every query is a regex match — no API calls, no tokens, no latency. At 0.19ms per query, you can process ~5K queries per second on a single CPU.

4. **The agentic coordinator matters**. Without it (simple heuristic coordinator, 3 iterations), accuracy drops to ~49% and Macro F1 to ~60%. The coordinator's per-class guidance lifts Macro F1 from ~60% to 71.7%.

### Reproduce

```bash
pip install rulechef[benchmark]
python benchmarks/benchmark_banking77.py \
    --classes beneficiary_not_allowed,card_arrival,disposable_card_limits,exchange_rate,pending_cash_withdrawal \
    --shots 5 --max-iterations 15 --agentic \
    --base-url https://api.groq.com/openai/v1 \
    --model moonshotai/kimi-k2-instruct-0905
```

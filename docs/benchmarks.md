# Benchmarks

## Banking77 Intent Classification

To measure how well RuleChef performs on a real task, we benchmarked on a subset of the [Banking77](https://huggingface.co/datasets/legacy-datasets/banking77) intent classification dataset — 77 banking customer service intent classes with ~13K examples.

### Setup

- **5 classes** selected: `beneficiary_not_allowed`, `card_arrival`, `disposable_card_limits`, `exchange_rate`, `pending_cash_withdrawal`
- **5-shot per class** (25 training examples total)
- **Dev set**: remaining ~666 unused training examples (for refinement)
- **Test set**: 200 held-out examples from the official test split (never seen during learning)
- **Regex-only** rules (no code, no spaCy)
- **Agentic coordinator** guiding 15 refinement iterations
- **Model**: Kimi K2 via Groq API

### Results on Held-Out Test Set

| Metric | Value |
|--------|-------|
| Accuracy (exact match) | **67.0%** |
| Micro Precision | **95.0%** |
| Micro Recall | 67.0% |
| Macro F1 | **78.6%** |
| Rules learned | ~25 |
| Learning time | ~90s |
| Per-query latency | **0.05ms** |

### Per-Class Breakdown

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| exchange_rate | 100% | 95% | 97% |
| card_arrival | 100% | 90% | 95% |
| pending_cash_withdrawal | 100% | 75% | 86% |
| disposable_card_limits | 88% | 70% | 78% |
| beneficiary_not_allowed | 83% | 30% | 44% |

### Key Takeaways

1. **Precision is extremely high** — rules almost never produce false positives. This matters in production where wrong answers are worse than no answer.

2. **Recall scales with complexity**. Simple keyword patterns (`exchange_rate`) are easy; nuanced paraphrases (`beneficiary_not_allowed`) need more examples or refinement iterations.

3. **Zero runtime cost**. After learning, every query is a regex match — no API calls, no tokens, no latency. At 0.05ms per query, you can process 20K queries per second on a single CPU.

4. **The agentic coordinator matters**. Without it (simple heuristic coordinator, 3 iterations), accuracy was 47.5% and F1 was 63.3%. The coordinator's per-class guidance pushed F1 from 63.3% to 78.6%.

### Comparison: With vs Without Agentic Coordinator

| | Simple Coordinator | Agentic Coordinator |
|---|---|---|
| Iterations | 3 | 15 |
| Accuracy | 47.5% | 67.0% |
| Macro F1 | 63.3% | 78.6% |
| Micro Precision | 95%+ | 95.0% |

### Reproduce

```bash
pip install rulechef[benchmark]
python benchmarks/benchmark_banking77.py \
    --shots 5 --num-classes 5 \
    --max-iterations 15 --agentic \
    --base-url https://api.groq.com/openai/v1 \
    --model moonshotai/kimi-k2-instruct-0905
```

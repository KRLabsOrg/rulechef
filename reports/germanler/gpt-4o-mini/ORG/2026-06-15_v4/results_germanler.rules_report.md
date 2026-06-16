# Rule Evaluation Report — gpt-4o-mini

Generated on: 2026-06-15T14:17:19.658754

---

<details>
<summary>Configuration</summary>

Results can be reproduced by running this command: 
```
 python benchmark.py --config reports/germanler/gpt-4o-mini/ORG/2026-06-15_v4/config.yaml 
```
| Parameter | Value |
|---|---|
| Pool size | None |
| Train ratio | 0.80 |
| Validation ratio | 0.20 |
| Shots per class | None |
| Training documents | 3950 |
| Validation documents | 988 |
| Test documents | 6666 |
| Train sentences | 3950 |
| Validation sentences | 988 |
| Test sentences | 6666 |
| Model | gpt-4o-mini |
| Max rules | 30 |
| Max samples in prompt | 150 |
| Refinement iterations | 6 |
| Seed | 42 |
| Agentic | True |
| Enable Critic | True |
| Enable Prune | False |
| Critic Interval | 10 |
| Audit Interval | 0 |
| Use GREX | True |
| Format | regex |
| Synthesis strategy | bulk |
| Sampling strategy | balanced |
| Batch size | 100 |
| Refine per batch | 2 |
| Manually annotated examples | 0 |
| First batch with manual data | None |

</details>

---

**Transfer Learning**

| Property | Value |
|---|---|
| Best Batch Idx | -1 |
| Best Batch F1 | 0.0 |
| Best Rules Serialized | [] |

<details>
<summary>Results</summary>

| Metric | Value |
|---|---|
| Accuracy (exact match) | 100.0% |
| True Positives | 0 |
| False Positives | 0 |
| False Negatives | 0 |
| Total Gold Entities | 0 |
| Micro Precision | 0.0% |
| Micro Recall | 0.0% |
| Micro F1 | 0.0% |
| Macro F1 | 0.0% |

</details>

---

<details>
<summary>📊 Summary</summary>

| Rule | F1 | Precision | Recall | Total Predicted | True Positives | False Positives |
|---|---|---|---|---|---|---|

</details>

---

<details>
<summary>🏆 Most Precise Rules</summary>

</details>

---

<details>
<summary>💣 Least Precise Rules</summary>

</details>

---

<details>
<summary>🔇 Inactive Rules</summary>

</details>

---

<details>
<summary>📋 All Rules</summary>

</details>

---


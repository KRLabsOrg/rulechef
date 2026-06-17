# Rule Evaluation Report — Qwen/Qwen3.5-35B-A3B

Generated on: 2026-06-17T10:55:04.527489

---

<details>
<summary>Configuration</summary>

Results can be reproduced by running this command: 
```
 python benchmark.py --config reports/germanler/Qwen_Qwen3.5-35B-A3B/ORG/2026-06-15/config.yaml 
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
| Model | Qwen/Qwen3.5-35B-A3B |
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

<details>
<summary>Results</summary>

| Metric | Value |
|---|---|
| Accuracy (exact match) | 81.5% |
| True Positives | 518 |
| False Positives | 1467 |
| False Negatives | 289 |
| Total Gold Entities | 807 |
| Micro Precision | 26.1% |
| Micro Recall | 64.2% |
| Micro F1 | 37.1% |
| Macro F1 | 37.1% |

</details>

---

<details>
<summary>📊 Summary</summary>

| Rule | F1 | Precision | Recall | Total Predicted | True Positives | False Positives |
|---|---|---|---|---|---|---|
| `Bundesamt Entities` | 0.5% | 100.0% | 0.2% | 2 | 2 | 0 |
| `Specific Ministry Variations` | 1.0% | 100.0% | 0.5% | 4 | 4 | 0 |
| `Specific Court Names with Location` | 1.5% | 100.0% | 0.7% | 6 | 6 | 0 |
| `Complex Senate Names` | 2.2% | 100.0% | 1.1% | 9 | 9 | 0 |
| `BgA X Pattern` | 0.5% | 100.0% | 0.2% | 2 | 2 | 0 |
| `Bundesverwaltungsgericht Genitive` | 5.3% | 100.0% | 2.7% | 22 | 22 | 0 |
| `Ellipsis Company Names` | 0.7% | 100.0% | 0.4% | 3 | 3 | 0 |
| `Specific Ministry Names (Fixed)` | 0.7% | 100.0% | 0.4% | 3 | 3 | 0 |
| `Finanzamt and Staatsanwaltschaft` | 0.2% | 100.0% | 0.1% | 1 | 1 | 0 |
| `Court Names with Anonymized Location` | 1.2% | 100.0% | 0.6% | 5 | 5 | 0 |
| `Patent Office Full Name` | 3.2% | 92.9% | 1.6% | 14 | 13 | 1 |
| `European Court Full Name` | 2.9% | 92.3% | 1.5% | 13 | 12 | 1 |
| `Bundesverfassungsgericht Genitive` | 11.2% | 90.6% | 5.9% | 53 | 48 | 5 |
| `Anonymized Company Patterns` | 2.2% | 90.0% | 1.1% | 10 | 9 | 1 |
| `Bundesgerichtshof Genitive` | 3.6% | 83.3% | 1.9% | 18 | 15 | 3 |
| `Union and Associations (Fixed)` | 1.2% | 83.3% | 0.6% | 6 | 5 | 1 |
| `Bundeswehr` | 1.7% | 77.8% | 0.9% | 9 | 7 | 2 |
| `Specific Organization Names (Additional)` | 0.7% | 75.0% | 0.4% | 4 | 3 | 1 |
| `Anonymized Ellipsis Companies` | 0.7% | 75.0% | 0.4% | 4 | 3 | 1 |
| `Patent Department Full Context (Fixed)` | 6.2% | 74.3% | 3.2% | 35 | 26 | 9 |
| `Missing Specific Organizations` | 2.9% | 70.6% | 1.5% | 17 | 12 | 5 |
| `Specific Organization Names (Extended)` | 4.8% | 64.5% | 2.5% | 31 | 20 | 11 |
| `Government Ministries and Bodies` | 6.1% | 63.4% | 3.2% | 41 | 26 | 15 |
| `European Entities` | 2.4% | 62.5% | 1.2% | 16 | 10 | 6 |
| `Bundesfinanzhof` | 2.4% | 62.5% | 1.2% | 16 | 10 | 6 |
| `Anonymized Single Letter Companies (Fixed)` | 2.2% | 60.0% | 1.1% | 15 | 9 | 6 |
| `Specific Court Genitives with Location` | 1.2% | 55.6% | 0.6% | 9 | 5 | 4 |
| `Hyphenated Company Names` | 5.9% | 54.3% | 3.1% | 46 | 25 | 21 |
| `Specific Organization Names` | 1.9% | 42.1% | 1.0% | 19 | 8 | 11 |
| `Specific Court Genitives with Location (Fixed)` | 4.0% | 39.5% | 2.1% | 43 | 17 | 26 |
| `Bundespatentgericht` | 2.2% | 34.6% | 1.1% | 26 | 9 | 17 |
| `Specific German Organizations` | 4.7% | 23.3% | 2.6% | 90 | 21 | 69 |
| `Prisons and Detention Centers` | 0.2% | 16.7% | 0.1% | 6 | 1 | 5 |
| `Car Brands` | 0.2% | 14.3% | 0.1% | 7 | 1 | 6 |
| `Generic Court Abbreviations (Tightened)` | 14.9% | 13.3% | 16.9% | 1024 | 136 | 888 |
| `European Union` | 0.5% | 11.1% | 0.2% | 18 | 2 | 16 |
| `Specific Court Names with Location (Fixed)` | 2.0% | 9.2% | 1.1% | 98 | 9 | 89 |
| `Senate/Chamber of Courts` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Government Ministries with Suffix` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Organizations with Stiftung` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Organizations with Verband` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Unique Court Names with Location` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Specific Court Genitives` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Specific Court Name with Hyphen and Role` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Patent Office Departments Full` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Multi-Letter Company Codes` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `JPO and Other Abbreviations` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Senate/Chamber of Courts (Strict)` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Single Letter Companies` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Universities and Associations` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Complex Court Departments` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Patent Office Departments` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Single Letter Companies (Strict)` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Single Letter Entities` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `International Court` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Single Letter Companies with Punctuation` | 0.0% | 0.0% | 0.0% | 16 | 0 | 16 |
| `Specific Court Names with Location (Genitive)` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Quoted Brand Names (Strict)` | 0.0% | 0.0% | 0.0% | 224 | 0 | 224 |

</details>

---

<details>
<summary>🏆 Most Precise Rules</summary>

## `Bundesverwaltungsgericht Genitive`

**F1:** 0.053 | **Precision:** 1.000 | **Recall:** 0.027  

**Format:** `regex`  
**Rule ID:** `7f6a80df`  
**Description:**
Matches 'Bundesverwaltungsgericht' and its genitive form 'Bundesverwaltungsgerichts'.

**Content:**
```
\b(Bundesverwaltungsgericht|Bundesverwaltungsgerichts)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.027 | 0.053 | 22 | 22 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 22 | 0 | 765 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53539`) (sent_id: `53539`)


cc ) Stellt ein Bewerber nicht innerhalb eines Monats nach Zugang der Abbruchmitteilung einen Antrag auf Erlass einer einstweiligen Verfügung , darf der Dienstherr nach der Rechtsprechung des Bundesverwaltungsgerichts darauf vertrauen , dass der Bewerber den Abbruch des Auswahlverfahrens nicht angreift , sondern sein Begehren im Rahmen einer neuen Ausschreibung weiterverfolgt ( vgl. BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14 ) .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Missed by this rule (FN):**

- `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14` (RS)

**Example 1** (doc_id: `53619`) (sent_id: `53619`)


( 2 ) Ob die Umwandlung der Todesstrafe in eine lebenslange Freiheitsstrafe bereits zwingend aus dem seit 1991 praktizierten Moratorium folgt , wie es das Bundesverwaltungsgericht angenommen hat , kann dahinstehen .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Example 2** (doc_id: `54169`) (sent_id: `54169`)


Unabhängig von der Frage , ob ein solcher Antrag - entgegen der Auffassung des Antragstellers - gemäß § 67 Abs. 4 VwGO dem Vertretungszwang unterliegt ( vgl. verneinend z.B. : BVerwG , Beschluss vom 6. Juni 1972 - 3 ER 404.71 - Buchholz 310 § 53 VwGO Nr. 6 ; Ziekow , in : Sodan / Ziekow , VwGO , 4. Aufl. 2014 , § 53 Rn. 20 ; Kraft , in : Eyermann , VwGO , 14. Aufl. 2014 , § 53 Rn. 14 ; Unruh , in : HK-Verwaltungsrecht , 3. Aufl. 2013 , § 53 Rn. 15 ; Scheidler , VR 2012 , 113 ; bejahend z.B. : W. - R. Schenke , in : Kopp / Schenke , VwGO , 23. Aufl. 2017 , § 53 Rn. 10 ; Redeker / von Oertzen , VwGO , 15. Aufl. 2010 , § 53 Rn. 5 ; vermittelnd z.B. : Bier , in : Schoch / Schneider / Bier , VwGO , Stand Oktober 2016 , § 53 Rn. 13 ) , ist der Antrag hier jedenfalls deshalb unzulässig , weil die gesetzlichen Voraussetzungen , unter denen das Bundesverwaltungsgericht eine Zuständigkeitsbestimmung überhaupt vornehmen darf , nicht erfüllt sind .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Missed by this rule (FN):**

- `§ 67 Abs. 4 VwGO` (NRM)
- `BVerwG , Beschluss vom 6. Juni 1972 - 3 ER 404.71 - Buchholz 310 § 53 VwGO Nr. 6` (RS)
- `Ziekow , in : Sodan / Ziekow , VwGO , 4. Aufl. 2014 , § 53 Rn. 20` (LIT)
- `Kraft , in : Eyermann , VwGO , 14. Aufl. 2014 , § 53 Rn. 14` (LIT)
- `Unruh , in : HK-Verwaltungsrecht , 3. Aufl. 2013 , § 53 Rn. 15` (LIT)
- `Scheidler , VR 2012 , 113` (LIT)
- `W. - R. Schenke , in : Kopp / Schenke , VwGO , 23. Aufl. 2017 , § 53 Rn. 10` (LIT)
- `Redeker / von Oertzen , VwGO , 15. Aufl. 2010 , § 53 Rn. 5` (LIT)
- `Bier , in : Schoch / Schneider / Bier , VwGO , Stand Oktober 2016 , § 53 Rn. 13` (LIT)

**Example 3** (doc_id: `54506`) (sent_id: `54506`)


Es mag auf sich beruhen , ob mit der Beschwerde davon auszugehen ist , dass den Ausführungen des Bundesverwaltungsgerichts in seinem Urteil vom 26. Januar 2017 ( - 1 C 10.16 - BVerwGE 157 , 208 Rn. 38 )

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Missed by this rule (FN):**

- `Urteil vom 26. Januar 2017 ( - 1 C 10.16 - BVerwGE 157 , 208 Rn. 38 )` (RS)

**Example 4** (doc_id: `55035`) (sent_id: `55035`)


( b ) Es gibt auch keine hinreichend überzeugenden Anhaltspunkte , die vom Bundesverwaltungsgericht dahin hätten gewertet werden müssen , dass von dieser gesetzlich vorgesehenen Möglichkeit der Umwandlung der Strafe und der Strafrestaussetzung de facto kein Gebrauch gemacht werden wird und der Beschwerdeführer damit keine Aussicht auf Entlassung aus der Haft hätte .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Example 5** (doc_id: `55959`) (sent_id: `55959`)


Das Bundesverwaltungsgericht hat unter Anwendung und Auslegung des materiellen Unionsrechts ausführlich erläutert , warum es zu der Überzeugung gelangt ist , dass die Rechtslage in Bezug auf Art. 10 EH-RL eindeutig ist .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Missed by this rule (FN):**

- `Art. 10 EH-RL` (NRM)

**Example 6** (doc_id: `56038`) (sent_id: `56038`)


3. Darüber hinaus hat das Bundesverwaltungsgericht seine Überzeugung von der Verfassungswidrigkeit des § 67 Abs. 2 Satz 3 Halbsatz 1 BbgHG hinreichend dargelegt .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Missed by this rule (FN):**

- `§ 67 Abs. 2 Satz 3 Halbsatz 1 BbgHG` (NRM)

**Example 7** (doc_id: `56455`) (sent_id: `56455`)


Angesichts des Moratoriums , das in Tunesien seit 27 Jahren ohne Ausnahmen eingehalten wird und das im Zuge der Aufklärung durch das Bundesverwaltungsgericht von den tunesischen Behörden auch bezogen auf den konkreten Fall des Beschwerdeführers nochmals bekräftigt wurde , ist die Befürchtung des Beschwerdeführers , dass eine gegen ihn in Tunesien verhängte Todesstrafe vollstreckt werden könnte , nicht begründet .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Missed by this rule (FN):**

- `Tunesien` (LOC)
- `Tunesien` (LOC)

**Example 8** (doc_id: `56642`) (sent_id: `56642`)


Das Bundesverwaltungsgericht hat weder diesen Maßstab verkannt noch die von ihm ermittelte Prognosegrundlage in verfassungsrechtlich relevanter Weise falsch eingeschätzt .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Example 9** (doc_id: `56654`) (sent_id: `56654`)


Die Geschäftsstelle des Bundesverwaltungsgerichts gliedert sich in sechs Arbeitsgruppen , die jeweils von einer Beamtin oder einem Beamten des gehobenen Dienstes geleitet werden .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Example 10** (doc_id: `56677`) (sent_id: `56677`)


Der Gesetzgeber hat insoweit auch für das gerichtliche Asylverfahren an den allgemeinen Grundsätzen des Revisionsrechts festgehalten und für das Bundesverwaltungsgericht keine Befugnis eröffnet , Tatsachen ( würdigungs ) fragen grundsätzlicher Bedeutung in " Länderleitentscheidungen " , wie sie etwa das britische Prozessrecht kennt , zu treffen .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Example 11** (doc_id: `57409`) (sent_id: `57409`)


Dieser Zulassungsgrund ist erfüllt , wenn die Vorinstanz mit einem ihre Entscheidung tragenden abstrakten Rechtssatz in Anwendung derselben Rechtsvorschrift einem ebensolchen Rechtssatz , der in der Rechtsprechung des Bundesverwaltungsgerichts , des Gemeinsamen Senats der obersten Gerichtshöfe des Bundes oder des Bundesverfassungsgerichts aufgestellt worden ist , widersprochen hat .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Missed by this rule (FN):**

- `Gemeinsamen Senats der obersten Gerichtshöfe des Bundes` (ORG)
- `Bundesverfassungsgerichts` (ORG)

**Example 12** (doc_id: `57801`) (sent_id: `57801`)


6. Gegen die Verfügung des Hessischen Ministeriums des Innern und für Sport vom 1. August 2017 erhob der Beschwerdeführer beim Bundesverwaltungsgericht Klage und beantragte die Anordnung der aufschiebenden Wirkung dieser Klage .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Missed by this rule (FN):**

- `Hessischen Ministeriums des Innern und für Sport` (ORG)

**Example 13** (doc_id: `57977`) (sent_id: `57977`)


A. Die Anträge der Kläger auf Bewilligung von Prozesskostenhilfe und Beiordnung ihres Verfahrensbevollmächtigten für das Verfahren vor dem Bundesverwaltungsgericht werden abgelehnt , weil die Rechtsverfolgung - wie sich aus den nachstehenden Gründen ergibt - keine hinreichende Aussicht auf Erfolg bietet ( § 166 VwGO i. V. m. §§ 114 , 121 Abs. 1 ZPO ) .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Missed by this rule (FN):**

- `§ 166 VwGO` (NRM)
- `§§ 114 , 121 Abs. 1 ZPO` (NRM)

**Example 14** (doc_id: `58397`) (sent_id: `58397`)


Eine Divergenz ist nur dann im Sinne des § 133 Abs. 3 Satz 3 VwGO hinreichend bezeichnet , wenn die Beschwerde einen inhaltlich bestimmten , die angefochtene Entscheidung tragenden abstrakten Rechtssatz benennt , mit dem die Vorinstanz einem in der Rechtsprechung des Bundesverwaltungsgerichts oder eines anderen in der Vorschrift ( § 132 Abs. 2 Nr. 2 VwGO ) genannten Gerichts aufgestellten ebensolchen entscheidungstragenden Rechtssatz in Anwendung derselben Rechtsvorschrift widersprochen hat .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Missed by this rule (FN):**

- `§ 133 Abs. 3 Satz 3 VwGO` (NRM)
- `§ 132 Abs. 2 Nr. 2 VwGO` (NRM)

**Example 15** (doc_id: `58435`) (sent_id: `58435`)


Von diesem Rechtssatz des Bundesverwaltungsgerichts sei umfasst , dass Klarstellungen bzw. Konkretisierungen einer Vorschrift , die mit einer Änderungssatzung vorgenommen würden , die Frist nicht nur für die Regelungen , die Gegenstand der Änderungssatzung selbst seien , sondern auch für die Bestandteile der Vorschrift ( neu ) in Lauf setzten , um deren Klarstellung oder Konkretisierung es gehe .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Example 16** (doc_id: `58986`) (sent_id: `58986`)


Für Tatsachenfragen - und damit auch für Unterschiede bei der tatsächlichen Bewertung identischer Tatsachengrundlagen - hat es vorab ausdrücklich bestätigt , dass wegen der Bindung des Revisionsgerichts an die tatsächlichen Feststellungen des Berufungsgerichts ( § 137 Abs. 2 VwGO ) eine weitergehende Vereinheitlichung der Rechtsprechung durch das Bundesverwaltungsgericht ausscheidet .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Missed by this rule (FN):**

- `§ 137 Abs. 2 VwGO` (NRM)

**Example 17** (doc_id: `59775`) (sent_id: `59775`)


Eine Begründung der Verfassungsbeschwerde sei mangels Vorliegen der Beschlussgründe des Bundesverwaltungsgerichts noch nicht möglich .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Example 18** (doc_id: `59893`) (sent_id: `59893`)


2. Die Entscheidung des Bundesverwaltungsgerichts , der Zuteilungsbescheid der Deutschen Emissionshandelsstelle sei rechtmäßig , verletzt die Beschwerdeführerin nicht in ihren Grundrechten .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Missed by this rule (FN):**

- `Deutschen Emissionshandelsstelle` (ORG)

**Example 19** (doc_id: `59915`) (sent_id: `59915`)


● Anonymisierung von Entscheidungen gemäß Anlage 2 der Dienstanweisung über die Erstellung von Schriftgut beim Bundesverwaltungsgericht für den Versand

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Example 20** (doc_id: `59946`) (sent_id: `59946`)


Ein die Zahlungspflicht generell ausschließender Einwand , der letztlich - entgegen der Rechtsprechung des Bundesverwaltungsgerichts ( vgl. Beschluss vom 6. März 1997 - 8 B 246.96 - Buchholz 401.84 Benutzungsgebühren Nr. 86 S. 69 f. ) - auch der Erhebung von Abwassergebühren entgegenstehen würde , kann hieraus aber nicht hergeleitet werden .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Missed by this rule (FN):**

- `Beschluss vom 6. März 1997 - 8 B 246.96 - Buchholz 401.84 Benutzungsgebühren Nr. 86 S. 69 f.` (RS)

**Example 21** (doc_id: `60006`) (sent_id: `60006`)


Der Vertreter des Bundesinteresses beim Bundesverwaltungsgericht unterstützt die Position der Klägerin .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

</details>

---

## `Complex Senate Names`

**F1:** 0.022 | **Precision:** 1.000 | **Recall:** 0.011  

**Format:** `regex`  
**Rule ID:** `90da40d9`  
**Description:**
Matches specific senate/chamber names of the Patent Court including the full context (e.g., '25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts').

**Content:**
```
\b\d+\.\s+Senat\s+\(\s*[A-Za-zäöüß\s-]+\s*\)\s+des\s+Bundespatentgerichts\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.011 | 0.022 | 9 | 9 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 9 | 0 | 722 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `54018`) (sent_id: `54018`)


In der Beschwerdesache betreffend die Marke 30 2010 022 988 hat der 27. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 10. Mai 2017 durch die Vorsitzende Richterin Klante , den Richter Dr. Himmelmann und die Richterin Lachenmayr-Nikolaou beschlossen :

| Predicted | Gold |
|---|---|
| `27. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` | `27. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Klante` (PER)
- `Himmelmann` (PER)
- `Lachenmayr-Nikolaou` (PER)

**Example 1** (doc_id: `54291`) (sent_id: `54291`)


In der Beschwerdesache betreffend die Designanmeldung 40 2016 201 675.4 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` | `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Hacker` (PER)
- `Merzbach` (PER)
- `Meiser` (PER)

**Example 2** (doc_id: `54886`) (sent_id: `54886`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2012 063 820.1 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 5. Dezember 2017 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` | `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Knoll` (PER)
- `Kriener` (PER)
- `Nielsen` (PER)

**Example 3** (doc_id: `55818`) (sent_id: `55818`)


In der Beschwerdesache betreffend die Patentanmeldung 10 2014 003 988.9 hat der 23. Senat ( Technischer Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 8. Mai 2018 unter Mitwirkung des Vorsitzenden Richters Dr. Strößner sowie der Richter Dr. Friedrich , Dr. Zebisch und Dr. Himmelmann beschlossen :

| Predicted | Gold |
|---|---|
| `23. Senat ( Technischer Beschwerdesenat ) des Bundespatentgerichts` | `23. Senat ( Technischer Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Strößner` (PER)
- `Friedrich` (PER)
- `Zebisch` (PER)
- `Himmelmann` (PER)

**Example 4** (doc_id: `56015`) (sent_id: `56015`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2011 068 984.9 hat der 26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 8. Januar 2018 unter Mitwirkung der Vorsitzenden Richterin Kortge sowie der Richter Jacobi und Schödel beschlossen :

| Predicted | Gold |
|---|---|
| `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` | `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Kortge` (PER)
- `Jacobi` (PER)
- `Schödel` (PER)

**Example 5** (doc_id: `59509`) (sent_id: `59509`)


In der Beschwerdesache betreffend die Marke 30 2009 026 804 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 21. September 2017 unter Mitwirkung der Richter Merzbach , Dr. Meiser und Schödel beschlossen :

| Predicted | Gold |
|---|---|
| `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` | `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Merzbach` (PER)
- `Meiser` (PER)
- `Schödel` (PER)

**Example 6** (doc_id: `59628`) (sent_id: `59628`)


In der Beschwerdesache betreffend die Marke 30 2012 041 338 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 15. November 2017 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` | `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Knoll` (PER)
- `Kriener` (PER)
- `Nielsen` (PER)

**Example 7** (doc_id: `59761`) (sent_id: `59761`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2015 031 519.2 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Knoll sowie der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` | `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Knoll` (PER)
- `Kriener` (PER)
- `Nielsen` (PER)

**Example 8** (doc_id: `59948`) (sent_id: `59948`)


In der Beschwerdesache betreffend die international registrierte Marke IR 1 160 635 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 9. November 2017 unter Mitwirkung des Vorsitzenden Richters Professor Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` | `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Hacker` (PER)
- `Merzbach` (PER)
- `Meiser` (PER)

</details>

---

## `Specific Court Names with Location`

**F1:** 0.015 | **Precision:** 1.000 | **Recall:** 0.007  

**Format:** `regex`  
**Rule ID:** `61dc1199`  
**Description:**
Matches court names that include a location, handling genitive forms.

**Content:**
```
\b(Sozialgerichts Nürnberg|Bayerischen Landessozialgerichts)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.007 | 0.015 | 6 | 6 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 6 | 0 | 556 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `55856`) (sent_id: `55856`)


Die Beschwerde der Klägerin gegen die Nichtzulassung der Revision im Urteil des Bayerischen Landessozialgerichts vom 24. Mai 2017 wird zurückgewiesen .

| Predicted | Gold |
|---|---|
| `Bayerischen Landessozialgerichts` | `Bayerischen Landessozialgerichts` |

**Example 1** (doc_id: `56530`) (sent_id: `56530`)


Die Beklagte beantragt , die Urteile des Bayerischen Landessozialgerichts vom 16. 12. 2015 und des Sozialgerichts München vom 16. 5. 2014 aufzuheben und die Klage abzuweisen .

| Predicted | Gold |
|---|---|
| `Bayerischen Landessozialgerichts` | `Bayerischen Landessozialgerichts` |

**Missed by this rule (FN):**

- `Sozialgerichts München` (ORG)

**Example 2** (doc_id: `56616`) (sent_id: `56616`)


Die Beklagte beantragt , das Urteil des Bayerischen Landessozialgerichts vom 21. Oktober 2015 aufzuheben und das Urteil des Sozialgerichts Nürnberg vom 25. Juni 2013 bezüglich der Beigeladenen zu 1. und 3. aufzuheben und die Klage abzuweisen .

| Predicted | Gold |
|---|---|
| `Bayerischen Landessozialgerichts` | `Bayerischen Landessozialgerichts` |

**Missed by this rule (FN):**

- `Sozialgerichts Nürnberg` (ORG)

**Example 3** (doc_id: `57203`) (sent_id: `57203`)


Die Revision der Klägerin gegen das Urteil des Bayerischen Landessozialgerichts vom 14. September 2016 wird zurückgewiesen .

| Predicted | Gold |
|---|---|
| `Bayerischen Landessozialgerichts` | `Bayerischen Landessozialgerichts` |

**Example 4** (doc_id: `58013`) (sent_id: `58013`)


das Urteil des Bayerischen Landessozialgerichts vom 3. Juni 2016 und den Gerichtsbescheid des Sozialgerichts Nürnberg vom 30. August 2013 sowie den Bescheid der Beklagten vom 19. März 2012 und den Widerspruchsbescheid vom 18. Juni 2012 aufzuheben und die Beklagte zu verurteilen , unter Rücknahme des Bescheides vom 4. Februar 2005 und des Widerspruchsbescheides vom 22. April 2005 die Zeit vom 1. September 1973 bis 30. Juni 1990 als Zeit der Zugehörigkeit zum Zusatzversorgungssystem der technischen Intelligenz und die hierin erzielten Arbeitsentgelte festzustellen .

| Predicted | Gold |
|---|---|
| `Bayerischen Landessozialgerichts` | `Bayerischen Landessozialgerichts` |

**Missed by this rule (FN):**

- `Sozialgerichts Nürnberg` (ORG)

**Example 5** (doc_id: `59022`) (sent_id: `59022`)


Die Beschwerde der Klägerin gegen die Nichtzulassung der Revision im Urteil des Bayerischen Landessozialgerichts vom 24. Mai 2017 wird zurückgewiesen .

| Predicted | Gold |
|---|---|
| `Bayerischen Landessozialgerichts` | `Bayerischen Landessozialgerichts` |

</details>

---

## `Patent Office Full Name`

**F1:** 0.032 | **Precision:** 0.929 | **Recall:** 0.016  

**Format:** `regex`  
**Rule ID:** `f72ad7ef`  
**Description:**
Matches 'Deutschen Patent- und Markenamt' and 'Deutsche Patent- und Markenamt' in various cases.

**Content:**
```
\b(?:Deutschen\s+Patent-\s+und\s+Markenamt|Deutsche\s+Patent-\s+und\s+Markenamt)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.929 | 0.016 | 0.032 | 14 | 13 | 1 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 13 | 1 | 789 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53404`) (sent_id: `53404`)


Hiergegen richtet sich die mit Schriftsatz vom 11. Februar 2014 beim Deutschen Patent- und Markenamt eingelegte Beschwerde der Anmelderin , die ihr Patentbegehren mit den mit Schreiben vom 25. Februar 2011 eingereichten Unterlagen weiterverfolgt .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Example 1** (doc_id: `53791`) (sent_id: `53791`)


Zwischen den Vergleichszeichen besteht hinsichtlich der angegriffenen Waren Verwechslungsgefahr gemäß § 9 Abs. 1 Nr. 2 MarkenG , so dass der Widerspruch aus der Marke 30 2011 047 766 vom Deutschen Patent- und Markenamt zu Unrecht zurückgewiesen wurde .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `§ 9 Abs. 1 Nr. 2 MarkenG` (NRM)

**Example 2** (doc_id: `55285`) (sent_id: `55285`)


I. Das am 26. August 2013 angemeldete Zeichen Fireslim ist am 10. Januar 2014 unter der Nr. 30 2013 048 208 in das beim Deutschen Patent- und Markenamt geführte Markenregister für die nachfolgenden Waren und Dienstleistungen der Klassen 9 , 35 und 38 eingetragen worden :

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `Fireslim` (ORG)

**Example 3** (doc_id: `55981`) (sent_id: `55981`)


I. Die am 6. Mai 2010 angemeldete Wort- / Bildmarke 30 2010 028 176 ist am 20. Dezember 2010 für die nachfolgend genannten Waren in das beim Deutschen Patent- und Markenamt ( DPMA ) geführte Markenregister eingetragen worden :

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `DPMA` (ORG)

**Example 4** (doc_id: `56197`) (sent_id: `56197`)


I. Die am 7. November 2013 angemeldete Wortfolge Rap Shot ist am 23. Januar 2014 unter der Nummer 30 2013 058 941 als Wortmarke für die nachfolgend genannten Waren und Dienstleistungen in das beim Deutschen Patent- und Markenamt geführte Markenregister eingetragen worden :

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `Rap Shot` (ORG)

**Example 5** (doc_id: `57795`) (sent_id: `57795`)


I. Auf die am 26. Januar 2012 beim Deutschen Patent- und Markenamt eingereichte Patentanmeldung ist die Erteilung des Patents 10 2012 201 128 mit der Bezeichnung „ Verfahren , Steuergerät und Speichermedium zur Steuerung einer Harnstoffinjektion bei niedrigen Abgastemperaturen unter Berücksichtigung des Harnstoffgehalts “ am 17. Januar 2013 veröffentlicht worden .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Example 6** (doc_id: `57844`) (sent_id: `57844`)


eingetragenen Wort- / Bildmarke 30 2010 022 988 ( Anmeldetag : 18. Mai 2010 ; Tag der Eintragung im beim Deutschen Patent- und Markenamt geführten Markenregister : 30. Juni 2010 ) ist aus der für die Waren

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Example 7** (doc_id: `57851`) (sent_id: `57851`)


Der Antragsteller beantragte am 5. November 2012 beim Deutschen Patent- und Markenamt ( DPMA ) die Eintragung eines Geschmacksmusters als Sammelanmeldung von 16 Mustern für Erzeugnisse der Klasse 19 - 07 „ Lehrmittel “ .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `DPMA` (ORG)

**Example 8** (doc_id: `57902`) (sent_id: `57902`)


Im Verfahren vor dem Deutschen Patent- und Markenamt ( DPMA ) sieht das Patentgesetz eine Zurückweisung verspäteten Vorbringens nicht vor .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `DPMA` (ORG)
- `Patentgesetz` (NRM)

**Example 9** (doc_id: `58140`) (sent_id: `58140`)


in das beim Deutschen Patent- und Markenamt geführte Register eingetragen worden .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Example 10** (doc_id: `59024`) (sent_id: `59024`)


Am 20. Mai 2016 übermittelte die Anmelderin dem Deutschen Patent- und Markenamt ( DPMA ) Unterlagen für die Einleitung der deutschen nationalen Phase der Anmeldung mit einem gegenüber der ursprünglichen internationalen Anmeldung geänderten Anspruchssatz mit 13 Patentansprüchen .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `DPMA` (ORG)

**Example 11** (doc_id: `59025`) (sent_id: `59025`)


a. Der Inhaber der angegriffenen Marke hat mit Schriftsatz vom 13. September 2012 , welcher per Fax am 13. September 2012 und im Original am 15. September 2012 beim Deutschen Patent- und Markenamt eingegangen ist , die Benutzung der ( Unions- ) Widerspruchsmarke 005 137 708 unbeschränkt bestritten , wobei er die Einrede in den Schriftsätzen vom 4. Dezember 2014 und 20. September 2017 nochmals wiederholt hat .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Example 12** (doc_id: `59559`) (sent_id: `59559`)


I. Das Wortzeichen Wohlfühlfarbe ist am 1. März 2016 zur Eintragung als Marke in das vom Deutschen Patent- und Markenamt geführte Register angemeldet worden für folgende Waren :

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `Wohlfühlfarbe` (ORG)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `57135`) (sent_id: `57135`)


unter Zurückweisung der Beschwerde der Einsprechenden den Beschluss der Patentabteilung 43 des Deutschen Patent- und Markenamt vom 4. November 2015 aufzuheben und das Streitpatent vollumfänglich aufrechtzuerhalten ,

**False Positives:**

- `Deutschen Patent- und Markenamt` — partial — pred is substring of gold: `Patentabteilung 43 des Deutschen Patent- und Markenamt`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Patentabteilung 43 des Deutschen Patent- und Markenamt`(ORG)

</details>

---

## `European Court Full Name`

**F1:** 0.029 | **Precision:** 0.923 | **Recall:** 0.015  

**Format:** `regex`  
**Rule ID:** `a166a523`  
**Description:**
Matches the full name of the European Court of Human Rights.

**Content:**
```
\b(Europ\u00e4ische\s+Gerichtshof\s+f\u00fcr\s+Menschenrechte|Europ\u00e4ischen\s+Gerichtshofs\s+f\u00fcr\s+Menschenrechte|Europ\u00e4ischen\s+Gerichtshofes|Europ\u00e4ischen\s+Gerichtshof|Europ\u00e4ischen\s+Gerichtshofes)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.923 | 0.015 | 0.029 | 13 | 12 | 1 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 12 | 1 | 729 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53903`) (sent_id: `53903`)


Sie genügen den Anforderungen des Europäischen Gerichtshofs für Menschenrechte an die Überprüfbarkeit einer lebenslangen Freiheitsstrafe .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofs für Menschenrechte` | `Europäischen Gerichtshofs für Menschenrechte` |

**Example 1** (doc_id: `54815`) (sent_id: `54815`)


Die Entscheidungen des Europäischen Gerichtshofs für Menschenrechte seien aufgrund der Völkerrechtsfreundlichkeit des Grundgesetzes in Deutschland zu berücksichtigen .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofs für Menschenrechte` | `Europäischen Gerichtshofs für Menschenrechte` |

**Missed by this rule (FN):**

- `Grundgesetzes` (NRM)
- `Deutschland` (LOC)

**Example 2** (doc_id: `55710`) (sent_id: `55710`)


Der Europäische Gerichtshof für Menschenrechte habe seit dem Jahr 2008 in mehreren Entscheidungen das Recht auf Kollektivverhandlungen und Streik als Bestandteil von Art. 11 EMRK anerkannt , auch für beamtete Lehrkräfte in der Türkei .

| Predicted | Gold |
|---|---|
| `Europäische Gerichtshof für Menschenrechte` | `Europäische Gerichtshof für Menschenrechte` |

**Missed by this rule (FN):**

- `Art. 11 EMRK` (NRM)
- `Türkei` (LOC)

**Example 3** (doc_id: `55771`) (sent_id: `55771`)


Ungeachtet möglicher Ungenauigkeiten bei der Übersetzung des in der amtlichen Fassung nur in französischer Sprache vorliegenden Urteils ist bei einer Bewertung dieser Aussage mit Blick auf die einzelnen Ausprägungen der Völkerrechtsfreundlichkeit des Grundgesetzes mit einzustellen , dass der Europäische Gerichtshof für Menschenrechte - wie auch die Parenthese comme en l'espèce verdeutlicht - eine Aussage in einem konkret-individuell zu entscheidenden Verfahren getroffen hat .

| Predicted | Gold |
|---|---|
| `Europäische Gerichtshof für Menschenrechte` | `Europäische Gerichtshof für Menschenrechte` |

**Missed by this rule (FN):**

- `Grundgesetzes` (NRM)

**Example 4** (doc_id: `56026`) (sent_id: `56026`)


Die Streikteilnahme eines Beamten lasse sich auch nicht mit Blick auf Art. 11 EMRK und die hierzu ergangene jüngere Rechtsprechung des Europäischen Gerichtshofs für Menschenrechte rechtfertigen .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofs für Menschenrechte` | `Europäischen Gerichtshofs für Menschenrechte` |

**Missed by this rule (FN):**

- `Art. 11 EMRK` (NRM)

**Example 5** (doc_id: `56194`) (sent_id: `56194`)


Eine andere Beurteilung ergebe sich auch nicht aus der Entscheidung des Europäischen Gerichtshofs für Menschenrechte im Verfahren Enerji Yapi-Yol Sen v. Türkei .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofs für Menschenrechte` | `Europäischen Gerichtshofs für Menschenrechte` |

**Missed by this rule (FN):**

- `Enerji Yapi-Yol Sen` (PER)
- `Türkei` (LOC)

**Example 6** (doc_id: `58230`) (sent_id: `58230`)


Der Konventionstext und die Rechtsprechung des Europäischen Gerichtshofs für Menschenrechte dienen nach der ständigen Rechtsprechung des Bundesverfassungsgerichts auf der Ebene des Verfassungsrechts als Auslegungshilfen für die Bestimmung von Inhalt und Reichweite von Grundrechten und rechtsstaatlichen Grundsätzen des Grundgesetzes , sofern dies nicht zu einer Einschränkung oder Minderung des Grundrechtsschutzes nach dem Grundgesetz führt ( vgl. BVerfGE 74 , 358 < 370 > ; 111 , 307 < 317 > ; 120 , 180 < 200 f. > ) .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofs für Menschenrechte` | `Europäischen Gerichtshofs für Menschenrechte` |

**Missed by this rule (FN):**

- `Bundesverfassungsgerichts` (ORG)
- `Grundgesetzes` (NRM)
- `Grundgesetz` (NRM)
- `BVerfGE 74 , 358 < 370 > ; 111 , 307 < 317 > ; 120 , 180 < 200 f. >` (RS)

**Example 7** (doc_id: `58307`) (sent_id: `58307`)


Zwar hebt der Europäische Gerichtshof für Menschenrechte die Verantwortung für die Verhinderung einer gegen Art. 3 EMRK verstoßenden Behandlung in einem Drittstaat , welche die Europäische Menschenrechtskonvention den Konventionsstaaten bei Überstellung in diesen Drittstaat auferlegt , hervor ( vgl. EGMR , Urteil vom 4. September 2014 - 140/10 - , Trabelsi / Belgien , Rn. 119 f. m. w. N. ) .

| Predicted | Gold |
|---|---|
| `Europäische Gerichtshof für Menschenrechte` | `Europäische Gerichtshof für Menschenrechte` |

**Missed by this rule (FN):**

- `Art. 3 EMRK` (NRM)
- `Europäische Menschenrechtskonvention` (NRM)
- `EGMR , Urteil vom 4. September 2014 - 140/10 - , Trabelsi / Belgien , Rn. 119 f.` (RS)

**Example 8** (doc_id: `58611`) (sent_id: `58611`)


Wie der Europäische Gerichtshof für Menschenrechte in seiner Entscheidung im Verfahren Demir und Baykara v. Türkei ausgeführt hat , setzt die Rechtfertigung eines Eingriffs in Art. 11 Abs. 1 EMRK ein dringendes soziales beziehungsweise gesellschaftliches Bedürfnis ( " pressing social need " ) voraus ; zudem muss die Einschränkung verhältnismäßig sein ( vgl. EGMR < GK > , Demir and Baykara v. Turkey , Urteil vom 12. November 2008 , Nr. 34503/97 , § 119 ) .

| Predicted | Gold |
|---|---|
| `Europäische Gerichtshof für Menschenrechte` | `Europäische Gerichtshof für Menschenrechte` |

**Missed by this rule (FN):**

- `Demir` (PER)
- `Baykara` (PER)
- `Türkei` (LOC)
- `Art. 11 Abs. 1 EMRK` (NRM)
- `EGMR < GK > , Demir and Baykara v. Turkey , Urteil vom 12. November 2008 , Nr. 34503/97 , § 119` (RS)

**Example 9** (doc_id: `58917`) (sent_id: `58917`)


Das deckt sich mit der nach Art. 1 Abs. 2 GG gebotenen Berücksichtigung der EMRK bei der Auslegung des Grundgesetzes und der in diesem Zusammenhang ergangenen Rechtsprechung des Europäischen Gerichtshofs für Menschenrechte ( vgl. hierzu BVerfGE 111 , 307 < 329 f. > ; 128 , 326 < 369 > ; 140 , 317 < 359 Rn. 91 > ; BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 8. Mai 2017 - 2 BvR 157/17 - , NVwZ 2017 , S. 1196 ; Beschluss der 2. Kammer des Zweiten Senats vom 18. August 2017 - 2 BvR 424/17 - , juris , Rn. 36 ) .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofs für Menschenrechte` | `Europäischen Gerichtshofs für Menschenrechte` |

**Missed by this rule (FN):**

- `Art. 1 Abs. 2 GG` (NRM)
- `EMRK` (NRM)
- `Grundgesetzes` (NRM)
- `BVerfGE 111 , 307 < 329 f. > ; 128 , 326 < 369 > ; 140 , 317 < 359 Rn. 91 >` (RS)
- `BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 8. Mai 2017 - 2 BvR 157/17 - , NVwZ 2017 , S. 1196` (RS)
- `Beschluss der 2. Kammer des Zweiten Senats vom 18. August 2017 - 2 BvR 424/17 - , juris , Rn. 36` (RS)

**Example 10** (doc_id: `59621`) (sent_id: `59621`)


Das Vorliegen einer Verwechslungsgefahr für das Publikum ist nach ständiger Rechtsprechung sowohl des Europäischen Gerichtshofes als auch des Bundesgerichtshofes unter Berücksichtigung aller relevanten Umstände des Einzelfalls zu beurteilen ( vgl. hierzu z.B. EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER ; GRUR 2010 , 1098 Rn. 44 – Calvin Klein / HABM ; BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY ; GRUR 2012 , 1040 Rn. 25 – pjur / pure ; GRUR 2013 , 833 Rn. 30 – Culinaria / Villa Culinaria ) .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofes` | `Europäischen Gerichtshofes` |

**Missed by this rule (FN):**

- `Bundesgerichtshofes` (ORG)
- `EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER` (RS)
- `GRUR 2010 , 1098 Rn. 44 – Calvin Klein / HABM` (RS)
- `BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY` (RS)
- `GRUR 2012 , 1040 Rn. 25 – pjur / pure` (RS)
- `GRUR 2013 , 833 Rn. 30 – Culinaria / Villa Culinaria` (RS)

**Example 11** (doc_id: `59735`) (sent_id: `59735`)


( 5 ) Es kann offen bleiben , ob bei den Anforderungen an die konkrete gesetzliche Ausgestaltung des Überprüfungsmechanismus durch nationales Recht nach der Rechtsprechung des Europäischen Gerichtshofs für Menschenrechte zwischen der Vollstreckung der lebenslangen Freiheitsstrafe in Signatarstaaten der Europäischen Menschenrechtskonvention und in Drittstaaten zu unterscheiden ist .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofs für Menschenrechte` | `Europäischen Gerichtshofs für Menschenrechte` |

**Missed by this rule (FN):**

- `Europäischen Menschenrechtskonvention` (NRM)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `59790`) (sent_id: `59790`)


Die vom Europäischen Gerichtshof für Menschenrechte formulierten Grundsätze zum Streikrecht seien im Kern auf deutsche Beamte übertragbar .

**False Positives:**

- `Europäischen Gerichtshof` — partial — pred is substring of gold: `Europäischen Gerichtshof für Menschenrechte`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europäischen Gerichtshof für Menschenrechte`(ORG)

</details>

---

## `Bundesverfassungsgericht Genitive`

**F1:** 0.112 | **Precision:** 0.906 | **Recall:** 0.059  

**Format:** `regex`  
**Rule ID:** `fdd983a0`  
**Description:**
Matches 'Bundesverfassungsgericht' and its genitive form 'Bundesverfassungsgerichts'.

**Content:**
```
\b(Bundesverfassungsgericht|Bundesverfassungsgerichts)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.906 | 0.059 | 0.112 | 53 | 48 | 5 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 48 | 5 | 756 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53402`) (sent_id: `53402`)


Bei offenem Ausgang des Verfassungsbeschwerdeverfahrens muss das Bundesverfassungsgericht die Folgen abwägen , die eintreten würden , wenn die einstweilige Anordnung nicht erginge , die Verfassungsbeschwerde aber Erfolg hätte , gegenüber den Nachteilen , die entstünden , wenn die begehrte einstweilige Anordnung erlassen würde , der Verfassungsbeschwerde aber der Erfolg zu versagen wäre ( vgl. BVerfGE 76 , 253 < 255 > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `BVerfGE 76 , 253 < 255 >` (RS)

**Example 1** (doc_id: `53403`) (sent_id: `53403`)


Vor diesem Hintergrund erweist sich schon die Auseinandersetzung des Beschwerdeführers mit den vom Bundesverfassungsgericht - wenn auch zu Art. 19 Abs. 4 GG , den der Beschwerdeführer nicht rügt - entwickelten verfassungsrechtlichen Maßstäben als unzureichend ; umso weniger ist unter diesen Umständen eine mögliche Willkür der angegriffenen Entscheidung plausibel dargelegt .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Art. 19 Abs. 4 GG` (NRM)

**Example 2** (doc_id: `53700`) (sent_id: `53700`)


3. Das Bundesverfassungsgericht überprüft die Vereinbarkeit eines nationalen Gesetzes mit dem Grundgesetz auch , wenn zugleich Zweifel an der Vereinbarkeit des Gesetzes mit Sekundärrecht der Europäischen Union bestehen .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Grundgesetz` (NRM)
- `Europäischen Union` (ORG)

**Example 3** (doc_id: `53855`) (sent_id: `53855`)


aa ) In organisatorischer Hinsicht ist beim Bundesverfassungsgericht , anders als bei den Fachgerichten , eine Kapazitätsausweitung zur Verkürzung der Verfahrensdauer als Reaktion auf gesteigerte Eingangszahlen grundsätzlich nicht möglich , da die Struktur des Gerichts durch seine Funktion bedingt und durch die Verfassung und das Bundesverfassungsgerichtsgesetz vorgegeben ist ( BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21 ; vgl. BTDrucks 17/3802 , S. 26 ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Bundesverfassungsgerichtsgesetz` (NRM)
- `BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21` (RS)
- `BTDrucks 17/3802 , S. 26` (LIT)

**Example 4** (doc_id: `54126`) (sent_id: `54126`)


1. Aus besonderem Grund , namentlich im Interesse einer verlässlichen Finanz- und Haushaltsplanung und eines gleichmäßigen Verwaltungsvollzugs für Zeiträume einer weitgehend schon abgeschlossenen Veranlagung , hat das Bundesverfassungsgericht wiederholt die weitere Anwendbarkeit verfassungswidriger Normen binnen der dem Gesetzgeber bis zu einer Neuregelung gesetzten Frist oder spätestens bis zur Neuregelung für gerechtfertigt erklärt ( vgl. etwa BVerfGE 87 , 153 < 178 > ; 93 , 121 < 148 f. > ; 123 , 1 < 38 > ; 125 , 175 < 258 > ; 138 , 136 < 251 Rn. 287 > ; 139 , 285 < 319 Rn. 89 > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `BVerfGE 87 , 153 < 178 > ; 93 , 121 < 148 f. > ; 123 , 1 < 38 > ; 125 , 175 < 258 > ; 138 , 136 < 251 Rn. 287 > ; 139 , 285 < 319 Rn. 89 >` (RS)

**Example 5** (doc_id: `54368`) (sent_id: `54368`)


Auf dieser Grundlage habe der Beschwerdeführer auch eine praktische Chance auf Wiedererlangung seiner Freiheit , wie sie das Bundesverfassungsgericht fordere .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 6** (doc_id: `54521`) (sent_id: `54521`)


2. Soweit der Beschwerdeführer weiter rügt , das Urteil des Amtsgerichts verletze ihn auch in seinem Grundrecht aus Art. 2 Abs. 1 i. V. m. Art. 20 Abs. 3 GG , gilt nach ständiger Rechtsprechung des Bundesverfassungsgerichts in Bezug auf § 511 Abs. 4 ZPO derselbe Prüfungsmaßstab wie unter III. 1. dargestellt .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `Art. 2 Abs. 1 i. V. m. Art. 20 Abs. 3 GG` (NRM)
- `§ 511 Abs. 4 ZPO` (NRM)

**Example 7** (doc_id: `54541`) (sent_id: `54541`)


III. Für die vom Kläger begehrte Aussetzung und Vorlage an das Bundesverfassungsgericht sieht der Senat keine Veranlassung .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 8** (doc_id: `54629`) (sent_id: `54629`)


Ob das Bundesverfassungsgericht hinsichtlich der Zulässigkeit von Auslieferungen nach Rumänien unter dem Gesichtspunkt der Einhaltung der sich aus dem Grundgesetz ergebenden Grundrechte Anforderungen an die Haftbedingungen stellen werde , die über diejenigen des Europäischen Gerichtshofs und der Europäischen Menschenrechtskonvention hinausgingen , stehe zurzeit nicht fest .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Rumänien` (LOC)
- `Grundgesetz` (NRM)
- `Europäischen Gerichtshofs` (ORG)
- `Europäischen Menschenrechtskonvention` (NRM)

**Example 9** (doc_id: `54837`) (sent_id: `54837`)


Dies entspricht dem vom Bundesverfassungsgericht in seiner Weitergeltungsanordnung vom 4. Mai 2011 aus Art. 2 Abs. 2 Satz 2 in Verbindung mit Art. 20 Abs. 3 GG abgeleiteten Maßstab ( vgl. BVerfGE 128 , 326 < 332 > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Art. 2 Abs. 2 Satz 2 in Verbindung mit Art. 20 Abs. 3 GG` (NRM)
- `BVerfGE 128 , 326 < 332 >` (RS)

**Example 10** (doc_id: `54935`) (sent_id: `54935`)


Diese Entscheidung kann von der Kammer getroffen werden , weil die maßgeblichen verfassungsrechtlichen Fragen durch das Bundesverfassungsgericht bereits entschieden und die Verfassungsbeschwerde hiernach offensichtlich begründet ist , § 93c Abs. 1 Satz 1 BVerfGG .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `§ 93c Abs. 1 Satz 1 BVerfGG` (NRM)

**Example 11** (doc_id: `55031`) (sent_id: `55031`)


a ) Eine Verfassungsbeschwerde ist nach ständiger Rechtsprechung des Bundesverfassungsgerichts ( vgl. grundlegend BVerfGE 90 , 22 < 24 f. > ) wegen grundsätzlicher verfassungsrechtlicher Bedeutung anzunehmen , wenn sie eine verfassungsrechtliche Frage aufwirft , die sich nicht ohne Weiteres aus dem Grundgesetz beantworten lässt und die noch nicht durch die verfassungsgerichtliche Rechtsprechung geklärt oder die durch veränderte Verhältnisse erneut klärungsbedürftig geworden ist .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `BVerfGE 90 , 22 < 24 f. >` (RS)
- `Grundgesetz` (NRM)

**Example 12** (doc_id: `55168`) (sent_id: `55168`)


a ) Zwar gehe das Bundesverfassungsgericht bislang in ständiger Rechtsprechung von einem Streikverbot für Beamte aus .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 13** (doc_id: `55230`) (sent_id: `55230`)


Dieses Ergebnis stehe in Übereinstimmung mit den vom Bundesverfassungsgericht anerkannten Referenzgruppen der kommunalen Wahlbeamten und der politischen Beamten .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 14** (doc_id: `55446`) (sent_id: `55446`)


Die für die Beurteilung der Verfassungsbeschwerde maßgeblichen verfassungsrechtlichen Fragen sind durch das Bundesverfassungsgericht bereits entschieden ( § 93c Abs. 1 Satz 1 BVerfGG ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `§ 93c Abs. 1 Satz 1 BVerfGG` (NRM)

**Example 15** (doc_id: `55466`) (sent_id: `55466`)


Zu berücksichtigen ist hierbei , dass vor dem Bundesverfassungsgericht regelmäßig - so auch hier - eine überschlägige Beurteilung der Sach- und Rechtslage für erledigt erklärter Verfassungsbeschwerden nicht stattfindet ( vgl. BVerfGE 33 , 247 < 264 f. > ; 85 , 109 < 115 f. > ; 87 , 394 < 397 f. > ) und auch keine der Fallgestaltungen vorliegt , in denen die Erfolgsaussichten der Verfassungsbeschwerde im Sinne des Beschwerdeführers vorhergesagt werden könnte ( vgl. BVerfGE 85 , 109 < 115 f. > ; 133 , 37 < 38 f. > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `BVerfGE 33 , 247 < 264 f. > ; 85 , 109 < 115 f. > ; 87 , 394 < 397 f. >` (RS)
- `BVerfGE 85 , 109 < 115 f. > ; 133 , 37 < 38 f. >` (RS)

**Example 16** (doc_id: `55591`) (sent_id: `55591`)


Insbesondere die durch Entscheidungen des Bundesverfassungsgerichts veranlassten Neuregelungen des Bewertungsgesetzes wurden nicht in die Vorschriften über die Einheitsbewertung eingearbeitet , sondern als Neuregelungen in eigenen Abschnitten in das Bewertungsgesetz eingefügt , ohne dass dabei die Bestimmungen über die Einheitsbewertung inhaltlich neu geformt worden wären .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `Bewertungsgesetzes` (NRM)
- `Bewertungsgesetz` (NRM)

**Example 17** (doc_id: `55695`) (sent_id: `55695`)


Der Beschwerdeführer hat daher ein fortbestehendes schutzwürdiges Interesse an einer nachträglichen verfassungsrechtlichen Überprüfung und gegebenenfalls einer hierauf bezogenen Feststellung der Verfassungswidrigkeit dieses Grundrechtseingriffs durch das Bundesverfassungsgericht ( vgl. BVerfGE 9 , 89 < 92 ff. > ; 32 , 87 < 92 > ; 53 , 152 < 157 f. > ; 91 , 125 < 133 > ; 104 , 220 < 234 f. > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `BVerfGE 9 , 89 < 92 ff. > ; 32 , 87 < 92 > ; 53 , 152 < 157 f. > ; 91 , 125 < 133 > ; 104 , 220 < 234 f. >` (RS)

**Example 18** (doc_id: `55740`) (sent_id: `55740`)


Das Verwaltungsgericht sei - im Einklang mit dem Internationalen Gerichtshof und dem Bundesverfassungsgericht - nicht von einer ausnahmslosen Völkerrechtswidrigkeit der Androhung und / oder des Einsatzes von Atomwaffen ausgegangen .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Internationalen Gerichtshof` (ORG)

**Example 19** (doc_id: `55815`) (sent_id: `55815`)


Im Rahmen der Prüfung , ob die Klägerin diese Zweifel ausräumen konnte , hat das Landesarbeitsgericht der Klägerin rechtliches Gehör nach Maßgabe der Rechtsprechung des Bundesverfassungsgerichts und des Bundesarbeitsgerichts gewährt .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `Bundesarbeitsgerichts` (ORG)

**Example 20** (doc_id: `55852`) (sent_id: `55852`)


Dies hindert das Bundesverfassungsgericht nicht , weitere Grundrechte in die Prüfung einzubeziehen , soweit sich die vom Beschwerdeführer geltend gemachte Rechtsverletzung in Blick auf dieselbe Beschwer auch oder vorrangig im Blick auf andere Grundrechte ergeben kann .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 21** (doc_id: `55922`) (sent_id: `55922`)


Es gab keine Anhaltspunkte dafür , dass die Annahme eines Bedürfnisses für eine bundesgesetzliche Regelung der Grundsteuer und der für sie maßgeblichen Bewertungsbestimmungen der danach verbleibenden , eingeschränkten Kontrolle durch das Bundesverfassungsgericht nicht hätte standhalten können ; solche wurden auch sonst von keiner Seite vorgebracht .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 22** (doc_id: `56228`) (sent_id: `56228`)


1. Nach § 32 Abs. 1 BVerfGG kann das Bundesverfassungsgericht im Streitfall einen Zustand durch einstweilige Anordnung vorläufig regeln , wenn dies zur Abwehr schwerer Nachteile oder aus einem anderen wichtigen Grund zum gemeinen Wohl dringend geboten ist .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `§ 32 Abs. 1 BVerfGG` (NRM)

**Example 23** (doc_id: `56313`) (sent_id: `56313`)


7. Das Bundesverfassungsgericht hat am 16. Januar 2018 eine mündliche Verhandlung durchgeführt .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 24** (doc_id: `56356`) (sent_id: `56356`)


Da das Streikverbot kein geschriebenes Verfassungsrecht , sondern Ergebnis einer Auslegung von Art. 33 Abs. 5 GG sei , müsse das Bundesverfassungsgericht seine frühere Auslegung dieser Bestimmung völkerrechtskonform hin zu einem funktionsbezogenen Streikverbot modifizieren .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Art. 33 Abs. 5 GG` (NRM)

**Example 25** (doc_id: `56443`) (sent_id: `56443`)


Nach der neueren Rechtsprechung des Bundesverfassungsgerichts sei auch innerhalb der Vermögensgruppe des Grundvermögens eine realitätsgerechte Bewertung erforderlich und eine Differenzierung bereits auf der Bewertungsebene verfassungsrechtlich nicht zulässig .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Example 26** (doc_id: `56582`) (sent_id: `56582`)


Das Bundesverfassungsgericht hat in diesem Beschluss nicht entschieden , dass in Fällen , in denen Oberverwaltungsgerichte / Verwaltungsgerichtshöfe auf der Grundlage ( weitestgehend ) identischer Tatsachenfeststellungen zu einer im Ergebnis abweichenden rechtlichen Beurteilung kommen , stets und notwendig eine ( klärungsbedürftige ) Rechtsfrage des Bundesrechts vorliegt , welche eine Rechtsmittelzulassung gebietet , um den Zugang zur Rechtsmittelinstanz nicht in einer durch Sachgründe nicht mehr zu rechtfertigenden Weise zu erschweren .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 27** (doc_id: `56796`) (sent_id: `56796`)


a ) Nach ständiger Rechtsprechung des Bundesverfassungsgerichts verpflichtet Art. 103 Abs. 1 GG ein Gericht , die Ausführungen der Prozessbeteiligten zur Kenntnis zu nehmen und in Erwägung zu ziehen .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `Art. 103 Abs. 1 GG` (NRM)

**Example 28** (doc_id: `56921`) (sent_id: `56921`)


Im vorliegenden Fall hat der Antragsteller zwar bislang in der Hauptsache kein Verfahren beim Bundesverfassungsgericht eingeleitet .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 29** (doc_id: `57236`) (sent_id: `57236`)


Zur Begründung wiederholt er im Wesentlichen sein bisheriges Vorbringen und verweist auf die erhebliche Vergleichbarkeit des Falles mit dem beim Bundesverfassungsgericht anhängigen Verfahren 2 BvR 424 / 17 .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Verfahren 2 BvR 424 / 17` (RS)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `55202`) (sent_id: `55202`)


Hierin lag insbesondere keine , im PKH-Verfahren nur in eng begrenztem Umfang zulässige vorweggenommene Beweiswürdigung ( s. dazu Beschlüsse des Bundesverfassungsgerichts - BVerfG - vom 7. Mai 1997 1 BvR 296/94 , Neue Juristische Wochenschrift 1997 , 2745 , und vom 20. Februar 2002 1 BvR 1450/00 , Neue Juristische Wochenschrift-Rechtsprechungs-Report Zivilrecht - NJW-RR - 2002 , 1069 ) .

**False Positives:**

- `Bundesverfassungsgerichts` — partial — pred is substring of gold: `Beschlüsse des Bundesverfassungsgerichts - BVerfG - vom 7. Mai 1997 1 BvR 296/94 , Neue Juristische Wochenschrift 1997 , 2745`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Beschlüsse des Bundesverfassungsgerichts - BVerfG - vom 7. Mai 1997 1 BvR 296/94 , Neue Juristische Wochenschrift 1997 , 2745`(RS)
- `vom 20. Februar 2002 1 BvR 1450/00 , Neue Juristische Wochenschrift-Rechtsprechungs-Report Zivilrecht - NJW-RR - 2002 , 1069`(RS)

**Example 1** (doc_id: `55266`) (sent_id: `55266`)


In diesem Sinn hat auch die Beklagte in ihrem Bescheid die Bürogemeinschaft zwischen dem Kläger und seinem ehemaligen Sozius deshalb missbilligt , weil nach ihrer Auffassung die Tätigkeit des Letzteren bezüglich der gesetzlichen Verschwiegenheitspflicht ( § 203 StGB ) , des Zeugnisverweigerungsrechts ( § 53 StPO ) und des Beschlagnahmeverbots ( § 97 StPO ) nach der damaligen Rechtslage weder mit den sozietätsfähigen Berufen noch mit den in der Entscheidung des Bundesverfassungsgerichts vom 12. Januar 2016 ( BVerfGE 141 , 82 ) behandelten Berufsgruppen vergleichbar ist .

**False Positives:**

- `Bundesverfassungsgerichts` — partial — pred is substring of gold: `Entscheidung des Bundesverfassungsgerichts vom 12. Januar 2016 ( BVerfGE 141 , 82 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 203 StGB`(NRM)
- `§ 53 StPO`(NRM)
- `§ 97 StPO`(NRM)
- `Entscheidung des Bundesverfassungsgerichts vom 12. Januar 2016 ( BVerfGE 141 , 82 )`(RS)

**Example 2** (doc_id: `58343`) (sent_id: `58343`)


Unter den hier vorliegenden Voraussetzungen des Art. 267 Abs. 3 AEUV ( vergleiche dazu Beschlüsse des Bundesverfassungsgerichts - BVerfG - vom 6. September 2016 1 BvR 1305/13 , Neue Zeitschrift für Verwaltungsrecht - NVwZ - 2017 , 53 , Rz 7 , 8 ; vom 6. Oktober 2017 2 BvR 987/16 , Neue Juristische Wochenschrift - NJW - 2018 , 606 , Rz 4 , 5 ; jeweils mit weiteren Nachweisen ) sind die nationalen Gerichte von Amts wegen gehalten , den EuGH anzurufen ( vergleiche BVerfG-Beschlüsse in NVwZ 2017 , 53 , Rz 6 ; in NJW 2018 , 606 , Rz 3 ; ferner EuGH-Urteil CILFIT vom 6. Oktober 1982 C - 283/81 , EU : C : 1982 : 335 , NJW 1983 , 1257 , Rz 21 ; jeweils mit weiteren Nachweisen ) .

**False Positives:**

- `Bundesverfassungsgerichts` — partial — pred is substring of gold: `Beschlüsse des Bundesverfassungsgerichts - BVerfG - vom 6. September 2016 1 BvR 1305/13 , Neue Zeitschrift für Verwaltungsrecht - NVwZ - 2017 , 53 , Rz 7 , 8`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 267 Abs. 3 AEUV`(NRM)
- `Beschlüsse des Bundesverfassungsgerichts - BVerfG - vom 6. September 2016 1 BvR 1305/13 , Neue Zeitschrift für Verwaltungsrecht - NVwZ - 2017 , 53 , Rz 7 , 8`(RS)
- `vom 6. Oktober 2017 2 BvR 987/16 , Neue Juristische Wochenschrift - NJW - 2018 , 606 , Rz 4 , 5`(RS)
- `EuGH`(ORG)
- `BVerfG-Beschlüsse in NVwZ 2017 , 53 , Rz 6`(LIT)
- `NJW 2018 , 606 , Rz 3`(RS)
- `EuGH-Urteil CILFIT vom 6. Oktober 1982 C - 283/81 , EU : C : 1982 : 335 , NJW 1983 , 1257 , Rz 21`(RS)

**Example 3** (doc_id: `58740`) (sent_id: `58740`)


Das Berufungsgericht wird sich bei seiner neuerlichen , durch diesen Beschluss nicht im Ergebnis vorgeprägten Entscheidung auch mit den - wenngleich in anderem Zusammenhang ergangenen - Entscheidungen des Gerichtshofes der Europäischen Union vom 25. Januar 2018 ( - C- 360/16 [ ECLI : EU :C : 2018 : 35 ] , Hasan - Rn. 30 f. , 40 ) und des Bundesverfassungsgerichts vom 21. April 2016 ( - 2 BvR 273/16 - NVwZ 2016 , 1242 Rn. 11 ) zur Frage des maßgeblichen Zeitpunkts für die Beurteilung des Vorliegens systemischer Schwachstellen auseinanderzusetzen haben .

**False Positives:**

- `Bundesverfassungsgerichts` — partial — pred is substring of gold: `Bundesverfassungsgerichts vom 21. April 2016 ( - 2 BvR 273/16 - NVwZ 2016 , 1242 Rn. 11 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Entscheidungen des Gerichtshofes der Europäischen Union vom 25. Januar 2018 ( - C- 360/16 [ ECLI : EU :C : 2018 : 35 ] , Hasan - Rn. 30 f. , 40 )`(RS)
- `Bundesverfassungsgerichts vom 21. April 2016 ( - 2 BvR 273/16 - NVwZ 2016 , 1242 Rn. 11 )`(RS)

**Example 4** (doc_id: `59129`) (sent_id: `59129`)


Beide Absätze des Art. 4 GG enthalten ein umfassend zu verstehendes einheitliches Grundrecht , das auch die Religionsfreiheit der Korporationen umfasst ( vgl. Urteil des Bundesverfassungsgerichts - BVerfG - vom 1. Dezember 2009 1 BvR 2857/07 , 1 BvR 2858/07 , BVerfGE 125 , 39 , Rz 137 , m. w. N. ) .

**False Positives:**

- `Bundesverfassungsgerichts` — partial — pred is substring of gold: `Urteil des Bundesverfassungsgerichts - BVerfG - vom 1. Dezember 2009 1 BvR 2857/07 , 1 BvR 2858/07 , BVerfGE 125 , 39 , Rz 137`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 4 GG`(NRM)
- `Urteil des Bundesverfassungsgerichts - BVerfG - vom 1. Dezember 2009 1 BvR 2857/07 , 1 BvR 2858/07 , BVerfGE 125 , 39 , Rz 137`(RS)

</details>

---

## `Anonymized Company Patterns`

**F1:** 0.022 | **Precision:** 0.900 | **Recall:** 0.011  

**Format:** `regex`  
**Rule ID:** `5cfc261d`  
**Description:**
Matches anonymized company names with single letters, dots, or ellipsis followed by legal forms.

**Content:**
```
\b(?:[A-Z]\s*\.?\s*\.\.\.\s*GmbH|\.\.\.\s+GmbH|\.\.\.\s+Corp\.|[A-Z]\s*\.?\s*\.\.\.\s+GmbH|Lilly\s*\.\.\.\s*LLC|I\s*\.\.\.\s*Corp\.|E\s*K\s*Co\.|[A-Z]\s+[A-Z]\s*Co\.|[A-Z]\s*\.?\s*\.\.\.\s+AG|V\s+AG|X\s+GmbH|Y\s+GmbH|D\s+P\s+T\s+S\s+GmbH|C\s+GmbH|H\u00c4VG|H\u00c4VG-Rechenzentrum\s+GmbH|H\u00c4VG-Rechenzentrum\s+AG|Haus\u00e4rztliche\s+Vertragsgemeinschaft\s+Aktiengesellschaft|Schleswig-Holsteinische\s+Oberlandesgericht|Finanzgericht\s+N\u00fcrnberg|Reichsversicherungsamt|S\u00e4chsischen\s+Bildungsagentur|Amtsgericht\s+[A-Z]\.|Bundessozialgericht|Bundesamts\s+f\u00fcr\s+das\s+Personalmanagement|Oberlandesgericht\s+D\u00fcsseldorf|Deutsche\s+Industrie-\s+und\s+Handelskammertag|InEK\s+GmbH|Sanktionsausschuss\s+des\s+Sicherheitsrats\s+der\s+Vereinten\s+Nationen|Landesamt\s+f\u00fcr\s+Landwirtschaft\s*,\s*Umwelt\s+und\s+l\u00e4ndliche\s+R\u00e4ume|LLUR)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.900 | 0.011 | 0.022 | 10 | 9 | 1 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 9 | 1 | 724 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `54007`) (sent_id: `54007`)


Das Unterrichtungsschreiben vom 14. November 2005 enthält weder Angaben zum Sitz der D P T S GmbH noch zu deren Anschrift , zum zuständigen Registergericht und zur Registernummer .

| Predicted | Gold |
|---|---|
| `D P T S GmbH` | `D P T S GmbH` |

**Example 1** (doc_id: `54284`) (sent_id: `54284`)


Der Stellung des Sammelantrages beim LLUR komme demgegenüber keine besondere Bedeutung zu .

| Predicted | Gold |
|---|---|
| `LLUR` | `LLUR` |

**Example 2** (doc_id: `55657`) (sent_id: `55657`)


Das Unterrichtungsschreiben vom 14. November 2005 enthält weder Angaben zum Sitz der D P T S GmbH noch zu deren Anschrift , zum zuständigen Registergericht und zur Registernummer .

| Predicted | Gold |
|---|---|
| `D P T S GmbH` | `D P T S GmbH` |

**Example 3** (doc_id: `56299`) (sent_id: `56299`)


Der Kläger hat dem Übergang seines Arbeitsverhältnisses von der Beklagten auf die D P T S GmbH mit Schreiben vom 1. September 2015 nicht wirksam widersprochen .

| Predicted | Gold |
|---|---|
| `D P T S GmbH` | `D P T S GmbH` |

**Example 4** (doc_id: `56463`) (sent_id: `56463`)


" Am 18. 08. 2011 erwarb die Firma H ... e. K. , vertreten durch den Zeugen W ... , zwei Kopiergeräte des Herstellers X ... zum Preis von 17.850 € von der Firma B ... GmbH , vertreten durch den Beschuldigten .

| Predicted | Gold |
|---|---|
| `B ... GmbH` | `B ... GmbH` |

**Missed by this rule (FN):**

- `H ... e. K.` (ORG)
- `W ...` (PER)
- `X ...` (ORG)

**Example 5** (doc_id: `57557`) (sent_id: `57557`)


Die Verfassungsbeschwerde betrifft die Feststellung einer Berufskrankheit nach dem Recht der gesetzlichen Unfallversicherung , wobei der Beschwerde-führer namentlich Verletzungen des Rechts auf rechtliches Gehör geltend macht , weil das Landessozialgericht mehreren Beweisanträgen nicht entsprochen und das Bundessozialgericht dies nicht korrigiert habe .

| Predicted | Gold |
|---|---|
| `Bundessozialgericht` | `Bundessozialgericht` |

**Example 6** (doc_id: `57710`) (sent_id: `57710`)


Die Sache ist zur erneuten Entscheidung an das Oberlandesgericht Düsseldorf zurückzuverweisen ( § 95 Abs. 2 BVerfGG ) .

| Predicted | Gold |
|---|---|
| `Oberlandesgericht Düsseldorf` | `Oberlandesgericht Düsseldorf` |

**Missed by this rule (FN):**

- `§ 95 Abs. 2 BVerfGG` (NRM)

**Example 7** (doc_id: `58142`) (sent_id: `58142`)


Tatsächlich ist dagegen nicht ersichtlich , dass den Versorgungsbehörden , dem Landessozialgericht oder dem Bundessozialgericht eine generelle Vernachlässigung von Grundrechten vorgeworfen werden könnte , sie also die Grundrechte nicht nur im konkreten Fall und mit Blick auf die inzwischen überholte Rechtslage nicht hinreichend beachtet haben könnten .

| Predicted | Gold |
|---|---|
| `Bundessozialgericht` | `Bundessozialgericht` |

**Example 8** (doc_id: `59355`) (sent_id: `59355`)


Es kann vorliegend dahinstehen , ob die neu gegründete D P T S GmbH ihre Tätigkeit überhaupt vor dem 1. Januar 2006 aufgenommen hatte ; jedenfalls wirkt sich auch insoweit aus , dass der Kläger - wie unter Rn. 39 ausgeführt - anders als im Fall einer ordnungsgemäßen Unterrichtung nicht gehalten gewesen wäre , innerhalb der in § 613a Abs. 6 Satz 1 BGB kurzen Frist von einem Monat nach Zugang der Unterrichtung zu entscheiden , ob er sein Widerspruchsrecht ausübt oder nicht , sondern hinreichend Zeit hatte , sich ggf. weitergehend zu erkundigen oder entsprechend beraten zu lassen , wem gegenüber er sein Widerspruchsrecht ausüben konnte .

| Predicted | Gold |
|---|---|
| `D P T S GmbH` | `D P T S GmbH` |

**Missed by this rule (FN):**

- `§ 613a Abs. 6 Satz 1 BGB` (NRM)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `58125`) (sent_id: `58125`)


- als Beschäftigter für folgende Aufgaben von begrenzter Dauer : im Rahmen der Fördermaßnahme der Sächsischen Bildungsagentur D für die sozialpädagogische Betreuung im Berufsvorbereitungsjahr am BSZ Technik und Wirtschaft P , Bewilligungsbescheid für das Schuljahr 2013/2014

**False Positives:**

- `Sächsischen Bildungsagentur` — partial — pred is substring of gold: `Sächsischen Bildungsagentur D`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Sächsischen Bildungsagentur D`(ORG)
- `BSZ Technik und Wirtschaft P`(ORG)

</details>

---

## `Bundesgerichtshof Genitive`

**F1:** 0.036 | **Precision:** 0.833 | **Recall:** 0.019  

**Format:** `regex`  
**Rule ID:** `8529045c`  
**Description:**
Matches 'Bundesgerichtshof' and its genitive form 'Bundesgerichtshofs' and 'Bundesgerichtshofes'.

**Content:**
```
\b(Bundesgerichtshof|Bundesgerichtshofs|Bundesgerichtshofes)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.833 | 0.019 | 0.036 | 18 | 15 | 3 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 15 | 3 | 792 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53387`) (sent_id: `53387`)


7. Die hiergegen eingelegte Nichtzulassungsbeschwerde sowie eine Anhörungsrüge der Beschwerdeführerin wies der Bundesgerichtshof zurück .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Example 1** (doc_id: `53957`) (sent_id: `53957`)


Über keine der anhängigen Rechtsbeschwerden hat der Bundesgerichtshof bislang entschieden .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Example 2** (doc_id: `56707`) (sent_id: `56707`)


In einem Schadensersatzprozess gegen die Beigeladene , mit dem er einen Ausgleich auch für die Kürzung der Regelaltersrente infolge des weiter verminderten Zugangsfaktors geltend machte , unterlag der Kläger auch in letzter Instanz vor dem Bundesgerichtshof ( BGH ) .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Missed by this rule (FN):**

- `BGH` (ORG)

**Example 3** (doc_id: `56787`) (sent_id: `56787`)


( 2 ) Einer Auslegung des Schutzgegenstands eines Designs auf Grundlage der Schnittmenge der allen Darstellungen gemeinsamen Merkmale könnten allerdings die nach der vorgenannten Entscheidung des Bundesgerichtshofes einem abgeleiteten Teilschutz entgegenstehenden und nunmehr für das DesignG geltenden Gesichtspunkte der Klarheit des Registers und der damit verbundenen Rechtssicherheit entgegenstehen .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshofes` | `Bundesgerichtshofes` |

**Missed by this rule (FN):**

- `DesignG` (NRM)

**Example 4** (doc_id: `56848`) (sent_id: `56848`)


Zwischen den sich gegenüberstehenden Zeichen sei jedenfalls eine hohe klangliche Ähnlichkeit gegeben , denn nach den vom Bundesgerichtshof entwickelten Grundsätzen zur Prägung von Wort- / Bildzeichen seien vorliegend in klanglicher Hinsicht jedenfalls die prägenden Zeichenbestandteile " GEA " und " KEA " zu vergleichen .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Example 5** (doc_id: `57029`) (sent_id: `57029`)


Da allein das Fehlen jeglicher Unterscheidungskraft ein Eintragungshindernis begründet , ist nach der Rechtsprechung des Bundesgerichtshofes ein großzügiger Maßstab anzulegen , so dass jede auch noch so geringe Unterscheidungskraft genügt , um das Schutzhindernis zu überwinden ( vgl. BGH GRUR 2012 , 1143 , Rn. 7 - Starsat ; GRUR 2012 , 1044 , 1045 , Rn. 9 - Neuschwanstein ; GRUR 2012 , 270 , Rn. 8 - Link economy ) .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshofes` | `Bundesgerichtshofes` |

**Missed by this rule (FN):**

- `BGH GRUR 2012 , 1143 , Rn. 7 - Starsat` (RS)
- `GRUR 2012 , 1044 , 1045 , Rn. 9 - Neuschwanstein` (RS)
- `GRUR 2012 , 270 , Rn. 8 - Link economy` (RS)

**Example 6** (doc_id: `57625`) (sent_id: `57625`)


8. Mit ihrer Verfassungsbeschwerde wendet sich die Beschwerdeführerin gegen die jüngste Entscheidung des Oberlandesgerichts und die beiden darauffolgenden Entscheidungen des Bundesgerichtshofs .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshofs` | `Bundesgerichtshofs` |

**Example 7** (doc_id: `57715`) (sent_id: `57715`)


Hierin durfte der Bundesgerichtshof einen sachlichen Grund sehen , der das Stadionverbot zu tragen vermag .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Example 8** (doc_id: `58279`) (sent_id: `58279`)


4. Daraufhin bestätigte der Bundesgerichtshof mit Urteil vom 21. April 2016 das Urteil des Oberlandesgerichts ( BGHZ 210 , 77 ) .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Missed by this rule (FN):**

- `BGHZ 210 , 77` (RS)

**Example 9** (doc_id: `58414`) (sent_id: `58414`)


5. Mit - nicht angegriffenem - Urteil vom 18. November 2014 hat der Bundesgerichtshof diese Entscheidung aufgehoben und den Rechtsstreit an das Oberlandesgericht zurückverwiesen .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Example 10** (doc_id: `58703`) (sent_id: `58703`)


Der Bundesgerichtshof enthebt die Veranstalter , wie er ausdrücklich ausführt , nicht von einer Plausibilitätskontrolle , um Fälle auszuschließen , in denen ein Verfahren offensichtlich willkürlich oder aufgrund falscher Tatsachenannahmen eingeleitet wurde .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Example 11** (doc_id: `59207`) (sent_id: `59207`)


Nach der Rechtsprechung des Bundesgerichtshofs wird bezogen auf die subjektive Tatseite in § 266a StGB wie folgt differenziert : Der Vorsatz muss sich auf die Eigenschaft als Arbeitgeber und Arbeitnehmer - dabei allerdings nur auf die statusbegründenden tatsächlichen Voraussetzungen , nicht auf die rechtliche Einordnung als solche und die eigene Verpflichtung zur Beitragsabführung - und alle darüber hinausreichenden , die sozialversicherungsrechtlichen Pflichten begründenden tatsächlichen Umstände erstrecken .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshofs` | `Bundesgerichtshofs` |

**Missed by this rule (FN):**

- `§ 266a StGB` (NRM)

**Example 12** (doc_id: `59253`) (sent_id: `59253`)


Die Annahme des Bundesgerichtshofs , wonach die Dienstaufsicht berechtigt sei , einem Richter ein in Zahlen gemessenes unzureichendes Erledigungspensum vorzuhalten , verstoße gegen Art. 97 Abs. 1 GG .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshofs` | `Bundesgerichtshofs` |

**Missed by this rule (FN):**

- `Art. 97 Abs. 1 GG` (NRM)

**Example 13** (doc_id: `59621`) (sent_id: `59621`)


Das Vorliegen einer Verwechslungsgefahr für das Publikum ist nach ständiger Rechtsprechung sowohl des Europäischen Gerichtshofes als auch des Bundesgerichtshofes unter Berücksichtigung aller relevanten Umstände des Einzelfalls zu beurteilen ( vgl. hierzu z.B. EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER ; GRUR 2010 , 1098 Rn. 44 – Calvin Klein / HABM ; BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY ; GRUR 2012 , 1040 Rn. 25 – pjur / pure ; GRUR 2013 , 833 Rn. 30 – Culinaria / Villa Culinaria ) .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshofes` | `Bundesgerichtshofes` |

**Missed by this rule (FN):**

- `Europäischen Gerichtshofes` (ORG)
- `EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER` (RS)
- `GRUR 2010 , 1098 Rn. 44 – Calvin Klein / HABM` (RS)
- `BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY` (RS)
- `GRUR 2012 , 1040 Rn. 25 – pjur / pure` (RS)
- `GRUR 2013 , 833 Rn. 30 – Culinaria / Villa Culinaria` (RS)

**Example 14** (doc_id: `59703`) (sent_id: `59703`)


9. Die Antragstellerin verweist zur Begründung ihrer Beschwerde zudem auf die Entscheidung „ Dipeptidyl-Peptidase-Inhibitoren “ , in welcher der deutsche Bundesgerichtshof festgestellt hat , dass im Hinblick auf das berechtigte Interesse , eine Erfindung in vollem Umfang zu schützen , die Umschreibung einer Gruppe von Stoffen durch eine funktionelle Definition patentrechtlich grundsätzlich selbst dann zulässig sein kann , wenn eine solche Fassung des Patentanspruchs auch die Verwendung noch unbekannter Möglichkeiten umfasse , die möglicherweise erst zukünftig bereitgestellt oder erfunden werden müssten ( BGH , GRUR 2013 , 1210 , Rnd. 19 ff. ) .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Missed by this rule (FN):**

- `BGH , GRUR 2013 , 1210 , Rnd. 19 ff.` (RS)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `54693`) (sent_id: `54693`)


c ) Die Klägerin wird beiläufig darauf hingewiesen , dass ein eventueller Innenausgleich zwischen N als Organträger und ihr als Organgesellschaft nach bürgerlichem Recht entsprechend § 426 des Bürgerlichen Gesetzbuchs vorgenommen wird und derjenige Beteiligte am Organkreis , aus dessen Umsätzen die an das FA zu zahlende Umsatzsteuer herrührt , im Innenverhältnis der Organschaft die Steuerlast zu tragen hat ( vgl. Urteile des Bundesgerichtshofs vom 29. Januar 2013 II ZR 91/11 , Deutsches Steuerrecht - DStR - 2013 , 478 , Rz 10 f. ; vom 19. Januar 2012 IX ZR 2/11 , DStR 2012 , 527 , Rz 28 und 36 ; BFH-Urteil vom 23. September 2009 VII R 43/08 , BFHE 226 , 391 , BStBl II 2010 , 215 , Rz 30 ) .

**False Positives:**

- `Bundesgerichtshofs` — similar text (different position): `N`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `N`(ORG)
- `§ 426 des Bürgerlichen Gesetzbuchs`(NRM)
- `Urteile des Bundesgerichtshofs vom 29. Januar 2013 II ZR 91/11 , Deutsches Steuerrecht - DStR - 2013 , 478 , Rz 10 f.`(RS)
- `vom 19. Januar 2012 IX ZR 2/11 , DStR 2012 , 527 , Rz 28 und 36`(RS)
- `BFH-Urteil vom 23. September 2009 VII R 43/08 , BFHE 226 , 391 , BStBl II 2010 , 215 , Rz 30`(RS)

**Example 1** (doc_id: `54748`) (sent_id: `54748`)


Dem entsprechend hat der für das Bankrecht allein zuständige XI. Zivilsenat des Bundesgerichtshofs mit Urteil vom 12. 7. 2016 - XI ZR 564/15 - ( NJW 2016 , 3512 , Tz. 34 , 40 ) entschieden , dass das Widerrufsrecht nach § 495 Abs. 1 BGB a. F. ungeachtet des Fehlers der erteilten Widerrufsbelehrung verwirkt werden kann .

**False Positives:**

- `Bundesgerichtshofs` — partial — pred is substring of gold: `XI. Zivilsenat des Bundesgerichtshofs`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `XI. Zivilsenat des Bundesgerichtshofs`(ORG)
- `Urteil vom 12. 7. 2016 - XI ZR 564/15 - ( NJW 2016 , 3512 , Tz. 34 , 40 )`(RS)
- `§ 495 Abs. 1 BGB a. F.`(NRM)

**Example 2** (doc_id: `57181`) (sent_id: `57181`)


Eine vergleichbar verschärfte Verpflichtung zur konkreten , eingehenderen Beschreibung der tatsächlichen Umstände des Beweisantrags ist als gesteigerte Substantiierungslast dann anzunehmen , wenn eine Person aufgrund Zugriffs zu den hierfür benötigten Informationen konkretere Informationen geben kann ( vgl. Urteil des Bundesgerichtshofs vom 25. April 2002 IX ZR 313/99 , BGHZ 150 , 353 , unter III. 2. – zur Substantiierungslast des Insolvenzverwalters für tatsächliche Vorgänge aus der Sphäre des Insolvenzschuldners ) .

**False Positives:**

- `Bundesgerichtshofs` — partial — pred is substring of gold: `Urteil des Bundesgerichtshofs vom 25. April 2002 IX ZR 313/99 , BGHZ 150 , 353 , unter III. 2.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Bundesgerichtshofs vom 25. April 2002 IX ZR 313/99 , BGHZ 150 , 353 , unter III. 2.`(RS)

</details>

---

## `Union and Associations (Fixed)`

**F1:** 0.012 | **Precision:** 0.833 | **Recall:** 0.006  

**Format:** `regex`  
**Rule ID:** `c7b455dc`  
**Description:**
Matches specific unions and associations like ver.di, IG Metall, etc.

**Content:**
```
\b(Vereinten\s+Dienstleistungsgewerkschaft|ver\.di|Gewerkschaft\s+ver\.di|IG\s*Metall|Deutsche\s*Rentenversicherung\s*Bund|Deutsche\s*Rentenversicherung\s*Rheinland|Deutschen\s*Rentenversicherung|Kommunalen\s*Arbeitgeberverband\s*Sachsen\s*e\.\s*V\.|Bayerischen\s*Rechtsanwalts-\s+und\s+Steuerberaterversorgung|Bundessteuerberaterkammer|Deutsche\s*Gesellschaft\s*f\u00fcr\s+Technische\s+Zusammenarbeit\s*\(\s*GTZ\s*\)\s*GmbH|Deutsche\s*Entwicklungsdienst\s*gGmbH|Bund\s*f\u00fcr\s+Lebensmittelrecht\s*und\s+Lebenskunde\s*e\.\s*V\.|Verbraucherzentrale\s*Bundesverband\s*e\.\s*V\.|foodwatch\s*e\.\s*V\.|Deutsche\s*Verband\s*Tiernahrung\s*e\.\s*V\.|Bundesvereinigung\s*der\s*Deutschen\s*Ern\u00e4hrungsindustrie\s*e\.\s*V\.|ZDS|GTS\s*GmbH\s*&\s+Co\.\s*KG|TGAOK|T\u00fcm\s*Bel\s*Sen|Europ\u00e4ischen\s*Kommission|Europ\u00e4ischen\s*Gerichtshofes|Europ\u00e4ischen\s*Gerichtshof|Europ\u00e4ischen\s*Gerichtshofes|Gewerkschaft\s+Erziehung\s+und\s+Wissenschaft|GEW|X-EWIV|VCS|Centralen\s+Marketing-Gesellschaft\s+der\s+deutschen\s+Agrarwirtschaft\s+mbH|CMA|Schleswig-Holsteinische\s+Oberverwaltungsgericht|Ausw\u00e4rtigen\s+Amtes|Deutschen\s+Stiftung\s+f\u00fcr\s+Internationale|Landgericht\s+Potsdam|Sozialgerichts\s+<\s*SG\s*>\s+Hildesheim|Landessozialgerichts\s+<\s*LSG\s*>\s+Niedersachsen-Bremen|11\.\s+Senats\s+des\s+LSG\s+Mecklenburg-Vorpommern|S\u00e4chsischen\s+LSG|Europ\u00e4ischen\s+Parlaments|Bundesministerium\s+des\s+Innern\s*,\s*f\u00fcr\s+Bau\s+und\s+Heimat|Ministerium\s+der\s+Justiz\s+des\s+Landes\s+Nordrhein-Westfalen|Bundesamt\s+f\u00fcr\s+Migration\s+und\s+Fl\u00fcchtlinge|W\u00a1\s+R\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.833 | 0.006 | 0.012 | 6 | 5 | 1 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 5 | 1 | 767 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53658`) (sent_id: `53658`)


E1a BRADLER , Christian : Check the GTS MAPLA system – additional information , Seiten 1 - 9 , © GTS GmbH & Co. KG , 11/2007 ;

| Predicted | Gold |
|---|---|
| `GTS GmbH & Co. KG` | `GTS GmbH & Co. KG` |

**Missed by this rule (FN):**

- `BRADLER , Christian` (PER)
- `GTS` (ORG)

**Example 1** (doc_id: `53858`) (sent_id: `53858`)


E1b BRADLER , Christian : Check the GTS MAPLA system – additional information , Seiten 1 - 9 , © GTS GmbH & Co. KG , 11/2007

| Predicted | Gold |
|---|---|
| `GTS GmbH & Co. KG` | `GTS GmbH & Co. KG` |

**Missed by this rule (FN):**

- `BRADLER , Christian` (PER)
- `GTS` (ORG)

**Example 2** (doc_id: `57124`) (sent_id: `57124`)


1. Der Deutsche Gewerkschaftsbund und ver.di meinen , die einschränkende Auslegung des § 14 Abs. 2 Satz 2 TzBfG durch das Bundesarbeitsgericht überschreite die Grenze zulässiger Rechtsfortbildung .

| Predicted | Gold |
|---|---|
| `ver.di` | `ver.di` |

**Missed by this rule (FN):**

- `Deutsche Gewerkschaftsbund` (ORG)
- `§ 14 Abs. 2 Satz 2 TzBfG` (NRM)
- `Bundesarbeitsgericht` (ORG)

**Example 3** (doc_id: `57679`) (sent_id: `57679`)


IV. Zu Vorlage und Verfassungsbeschwerde haben der Deutsche Gewerkschaftsbund ( DGB ) , die Vereinte Dienstleistungsgewerkschaft ( ver.di ) , die Bundesvereinigung der Deutschen Arbeitgeberverbände e. V. ( BDA ) , das Bundesarbeitsgericht und die Beklagten der Ausgangsverfahren Stellung genommen .

| Predicted | Gold |
|---|---|
| `ver.di` | `ver.di` |

**Missed by this rule (FN):**

- `Deutsche Gewerkschaftsbund` (ORG)
- `DGB` (ORG)
- `Vereinte Dienstleistungsgewerkschaft` (ORG)
- `Bundesvereinigung der Deutschen Arbeitgeberverbände e. V.` (ORG)
- `BDA` (ORG)
- `Bundesarbeitsgericht` (ORG)

**Example 4** (doc_id: `58927`) (sent_id: `58927`)


Der in § 1 KAT erwähnte Kirchliche Tarifvertrag Diakonie ( KTD , GVOBl. NEK S. 317 ) wurde bereits am 15. August 2002 zwischen dem Verband kirchlicher und diakonischer Anstellungsträger Nordelbien einerseits und der Gewerkschaft Kirche und Diakonie , der IG Bauen-Agrar-Umwelt , Bundesvorstand , sowie von ver.di andererseits geschlossen .

| Predicted | Gold |
|---|---|
| `ver.di` | `ver.di` |

**Missed by this rule (FN):**

- `§ 1 KAT` (REG)
- `Kirchliche Tarifvertrag Diakonie ( KTD , GVOBl. NEK S. 317 )` (REG)
- `Verband kirchlicher und diakonischer Anstellungsträger Nordelbien` (ORG)
- `Gewerkschaft Kirche und Diakonie` (ORG)
- `IG Bauen-Agrar-Umwelt , Bundesvorstand` (ORG)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `57554`) (sent_id: `57554`)


festzustellen , dass auf das Arbeitsverhältnis der Parteien der Kirchliche Arbeitnehmerinnen Tarifvertrag , abgeschlossen zwischen dem Verband kirchlicher und diakonischer Anstellungsträger Nordelbien sowie der Gewerkschaft Kirche und Diakonie und ver.di , Landesbezirke Hamburg und Nord , andererseits vom 1. Dezember 2006 Anwendung finde .

**False Positives:**

- `ver.di` — partial — pred is substring of gold: `ver.di , Landesbezirke Hamburg und Nord`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Kirchliche Arbeitnehmerinnen Tarifvertrag`(REG)
- `Verband kirchlicher und diakonischer Anstellungsträger Nordelbien`(ORG)
- `Gewerkschaft Kirche und Diakonie`(ORG)
- `ver.di , Landesbezirke Hamburg und Nord`(ORG)

</details>

---

## `Bundeswehr`

**F1:** 0.017 | **Precision:** 0.778 | **Recall:** 0.009  

**Format:** `regex`  
**Rule ID:** `ed9daf2d`  
**Description:**
Matches 'Bundeswehr' and its variations.

**Content:**
```
\b(Bundeswehr|Bundeswehrkommando\s+[A-Za-z\s]+|Ver\u00e4nderungsmanagement\s*Luftwaffe|Kommando\s*Luftwaffe|Luftwaffe)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.778 | 0.009 | 0.017 | 9 | 7 | 2 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 7 | 2 | 658 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `54964`) (sent_id: `54964`)


Diese besteht darin , dazu beizutragen , einen ordnungsgemäßen Dienstbetrieb wiederherzustellen und / oder aufrechtzuerhalten ( " Wiederherstellung und Sicherung der Integrität , des Ansehens und der Disziplin in der Bundeswehr " , vgl. dazu BVerwG , Urteil vom 11. Juni 2008 - 2 WD 11.07 - Buchholz 450.2 § 38 WDO 2002 Nr. 26 Rn. 23 m. w. N. ) .

| Predicted | Gold |
|---|---|
| `Bundeswehr` | `Bundeswehr` |

**Missed by this rule (FN):**

- `BVerwG , Urteil vom 11. Juni 2008 - 2 WD 11.07 - Buchholz 450.2 § 38 WDO 2002 Nr. 26 Rn. 23` (RS)

**Example 1** (doc_id: `56366`) (sent_id: `56366`)


Einen diesbezüglichen Rechtsanwendungserlass hat das Ministerium in Gestalt der für das Arbeitszeitrecht der Soldatinnen und Soldaten zuständigen Stelle ( BMVg FüSK III 1 ) für die im ENJJPT eingesetzten Fluglehrer der Luftwaffe nicht verfügt .

| Predicted | Gold |
|---|---|
| `Luftwaffe` | `Luftwaffe` |

**Missed by this rule (FN):**

- `BMVg FüSK III 1` (REG)

**Example 2** (doc_id: `57150`) (sent_id: `57150`)


" Die Fluglehrer der Luftwaffe , die als Instructor Pilots ( IP ) zur Durchführung der fliegerischen Ausbildung im ENJJPT abgestellt werden , werden gemäß diesem Memorandum of Understanding ( MoU ) und Program Plan of Operation ( PO ) innerhalb der international gemischten Organisation ( organizational / management structure ) aus insgesamt dreizehn Teilnehmerstaaten eingesetzt .

| Predicted | Gold |
|---|---|
| `Luftwaffe` | `Luftwaffe` |

**Example 3** (doc_id: `57970`) (sent_id: `57970`)


Auch befinde sich der für den Antragsteller vorgesehene Dienstposten nicht im Organisationsbereich Luftwaffe , sondern im Organisationsbereich Heer ; die Besetzungszuständigkeit des Dienstpostens ( Luftwaffe ) ändere daran nichts .

| Predicted | Gold |
|---|---|
| `Luftwaffe` | `Luftwaffe` |
| `Luftwaffe` | `Luftwaffe` |

**Example 4** (doc_id: `59139`) (sent_id: `59139`)


Der Dienstposten ... beim ... in C. , auf den der Antragsteller versetzt wurde , unterliegt - was unstrittig ist - der Besetzungszuständigkeit der Luftwaffe und war ausweislich der gegenständlichen Versetzungsverfügung vom 21. März 2017 jedenfalls in dem maßgeblichen Zeitpunkt der Versetzung dem Organisationsbereich Luftwaffe zugeordnet .

| Predicted | Gold |
|---|---|
| `Luftwaffe` | `Luftwaffe` |
| `Luftwaffe` | `Luftwaffe` |

**Missed by this rule (FN):**

- `...` (ORG)
- `C.` (LOC)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `54769`) (sent_id: `54769`)


Die Dauer der Auslandsverwendung des Antragstellers beim Bundeswehrkommando USA und Kanada in A vom 1. Juli 2014 bis zum 30. Juni 2017 entspricht exakt der regulären Dauer einer Tour of Duty .

**False Positives:**

- `Bundeswehrkommando USA und Kanada in A vom ` — partial — gold is substring of pred: `Bundeswehrkommando USA und Kanada`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundeswehrkommando USA und Kanada`(ORG)
- `A`(LOC)

**Example 1** (doc_id: `57722`) (sent_id: `57722`)


Das dienstliche Bedürfnis für die Wegversetzung des Antragstellers ist darüber hinaus unter dem Gesichtspunkt gegeben , dass seine befristete Auslandsverwendung beim Bundeswehrkommando USA und Kanada zum 30. Juni 2017 geendet hat .

**False Positives:**

- `Bundeswehrkommando USA und Kanada zum ` — partial — gold is substring of pred: `Bundeswehrkommando USA und Kanada`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundeswehrkommando USA und Kanada`(ORG)

</details>

---

</details>

---

<details>
<summary>💣 Least Precise Rules</summary>

## `Generic Court Abbreviations (Tightened)`

**F1:** 0.149 | **Precision:** 0.133 | **Recall:** 0.169  

**Format:** `regex`  
**Rule ID:** `6bc88f26`  
**Description:**
Matches high-priority court abbreviations, ensuring they are standalone and not part of a larger word.

**Content:**
```
\b(BGH|BVerfG|BFH|BSG|EuGH|EGMR|DPMA|BaFin|ZDS|DED|ZIV|EUIPO|STIKO|NEK|KCD-E|KON-KURD|CDK|BAG|ArbG|LAG|GBA|TdL|BgA|MDK|LSG|BA|H\u00c4VG|PKK|KCK|HPG|EED|JOOP|KAGURA|Schott|Urban\s+&\s+Schwarzenberg|National\s+Union\s+of\s+Rail|Maritime\s+and\s+Transport\s+Workers|Europ\u00e4ische\s+Gerichtshof\s+f\u00fcr\s+Menschenrechte|Deutscher\s+Apotheker\s+Verlag|B-KG|VEB\s*Elektronik\s*G\.|EZB|Bundestag|Bundesrat|Bundesregierung|Bundesverwaltungsgericht|Bundesfinanzhof|Bundespatentgericht|BVerwG|BVerfG|BGH|BSG|BAG|BVerfG|EuGH|EGMR|DPMA|BaFin|SG|FG|LAG|LSG|BA|H\u00c4VG|PKK|KCK|HPG|EED|JOOP|KAGURA|Schott|Urban\s+&\s+Schwarzenberg|National\s+Union\s+of\s+Rail|Maritime\s+and\s+Transport\s+Workers|Europ\u00e4ische\s+Gerichtshof\s+f\u00fcr\s+Menschenrechte|Deutscher\s+Apotheker\s+Verlag|B-KG|VEB\s*Elektronik\s*G\.)\b(?!\s+(?:Gericht|Amt|Beh\u00f6rde|Verfahren|Sache|Rechtsprechung|in|auf|von|mit|bei|nach|f\u00fcr|zum|zur|des|der|die|den|Beschluss|Urteil|Senat))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.133 | 0.169 | 0.149 | 1024 | 136 | 888 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 136 | 888 | 670 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53400`) (sent_id: `53400`)


Die Vorgabe des Bundesfinanzhofs ( BFH ) , dass die Einsteller ( Leistungsempfänger ) keine Land- und Forstwirte sein müssten , die von ihnen empfangenen Leistungen ( Zuchtleistungen als einheitliche Leistung ) aber gleichwohl zu landwirtschaftlichen Zwecken genutzt werden müssten , erscheine nicht eindeutig und auf den ersten Blick widersprüchlich .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Missed by this rule (FN):**

- `Bundesfinanzhofs` (ORG)

**Example 1** (doc_id: `53408`) (sent_id: `53408`)


Ein solcher enger sachlicher Zusammenhang mit der gesetzlichen Tätigkeit als Träger der Grundsicherung für Arbeitsuchende ist bei der BA immer dann anzunehmen , wenn sie - objektiv betrachtet - in dem zugrunde liegenden Verfahren eine Aufgabe im Rahmen des Vollzugs des SGB II wahrnimmt .

| Predicted | Gold |
|---|---|
| `BA` | `BA` |

**Missed by this rule (FN):**

- `SGB II` (NRM)

**Example 2** (doc_id: `53430`) (sent_id: `53430`)


II. Die zulässige Beschwerde der Löschungsantragsgegnerin hat sich durch die zwischenzeitlich mit Beschluss des DPMA vom 27. September 2017 angeordnete Verfallslöschung der angegriffenen Marke in der Hauptsache erledigt .

| Predicted | Gold |
|---|---|
| `DPMA` | `DPMA` |

**Example 3** (doc_id: `53524`) (sent_id: `53524`)


Der BFH führte aus , die handelsrechtliche Zurechnung von Vermögensgegenständen entspreche im Wesentlichen der Regelung des § 39 Abs. 2 Nr. 1 Satz 1 AO ( z.B. BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1. ; vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a ) .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Missed by this rule (FN):**

- `§ 39 Abs. 2 Nr. 1 Satz 1 AO` (NRM)
- `BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1.` (RS)
- `vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a` (RS)

**Example 4** (doc_id: `53549`) (sent_id: `53549`)


aa ) Der GBA hat eine IVIG-Therapie zur Behandlung des sekundären Immunglobulinmangels mit MGUS und rezidivierenden Pneumonien nicht empfohlen .

| Predicted | Gold |
|---|---|
| `GBA` | `GBA` |

**Example 5** (doc_id: `53594`) (sent_id: `53594`)


Die von der Krankenkasse nicht getragenen Kosten der stationären Behandlung zahlte der Kläger zu 1. Die Therapie der E in der K-Klinik war im Vorfeld durch den Medizinischen Dienst der Krankenversicherungen ( MDK ) auf ihre medizinische Notwendigkeit geprüft und befürwortet worden .

| Predicted | Gold |
|---|---|
| `MDK` | `MDK` |

**Missed by this rule (FN):**

- `E` (PER)
- `K-Klinik` (ORG)
- `Medizinischen Dienst der Krankenversicherungen` (ORG)

**Example 6** (doc_id: `53656`) (sent_id: `53656`)


Ebenso liegt es , wenn der GBA wegen des Potenzials der Methode bei nicht hinreichend belegtem Nutzen eine Erprobungs-RL beschließt ( § 137c Abs 1 S 3 SGB V ) und die Überprüfung unter Hinzuziehung der durch die Erprobung gewonnenen Erkenntnisse ergibt , dass die Methode nicht den Kriterien nach § 137c Abs 1 S 1 SGB V entspricht ( § 137c Abs 1 S 4 SGB V ) oder wenn eine Erprobungs-RL nicht zustande kommt , weil es an einer nach § 137e Abs 6 SGB V erforderlichen Vereinbarung fehlt ( § 137c Abs 1 S 5 SGB V ) .

| Predicted | Gold |
|---|---|
| `GBA` | `GBA` |

**Missed by this rule (FN):**

- `§ 137c Abs 1 S 3 SGB V` (NRM)
- `§ 137c Abs 1 S 1 SGB V` (NRM)
- `§ 137c Abs 1 S 4 SGB V` (NRM)
- `§ 137e Abs 6 SGB V` (NRM)
- `§ 137c Abs 1 S 5 SGB V` (NRM)

**Example 7** (doc_id: `53690`) (sent_id: `53690`)


An dieser Regelung hielt die Bundesregierung fest , obwohl der Bundesrat die Einbeziehung der einmaligen Einnahmen vorschlug ( BT-Drucks 16/2454 S 11 ) .

| Predicted | Gold |
|---|---|
| `Bundesregierung` | `Bundesregierung` |

**Missed by this rule (FN):**

- `Bundesrat` (ORG)
- `BT-Drucks 16/2454 S 11` (LIT)

**Example 8** (doc_id: `53746`) (sent_id: `53746`)


Das hat zur Folge , dass beim BSG über Kostenerinnerungen nach § 189 Abs 2 S 2 SGG der Senat in der Besetzung mit drei Berufsrichtern zu befinden hat .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Missed by this rule (FN):**

- `§ 189 Abs 2 S 2 SGG` (NRM)

**Example 9** (doc_id: `53867`) (sent_id: `53867`)


Zwar kann das LSG von einer Entscheidung ua des BSG auch dann abweichen , wenn es einen der höchstrichterlichen Rechtsprechung widersprechenden Rechtssatz nur sinngemäß und in scheinbar fallbezogene Ausführungen gekleidet entwickelt .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 10** (doc_id: `53939`) (sent_id: `53939`)


Dort hat der BFH lediglich ausgeführt , die Vergütung für die Hingabe eines partiarischen Darlehens könne auch umsatzabhängig ausgestaltet werden .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Example 11** (doc_id: `54059`) (sent_id: `54059`)


Wie der Kläger selbst vorträgt , wollte sich das LSG vielmehr nach seinem eigenen Verständnis im Rahmen der bereits vorliegenden Rechtsprechung des BSG halten , ist diesen aber in dem von ihm entschiedenen Fall lediglich deshalb nicht gefolgt , weil es - vom Kläger als unzutreffend gerügte - wesentliche Sachverhaltsunterschiede zu den bereits entschiedenen Fällen angenommen hat .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 12** (doc_id: `54064`) (sent_id: `54064`)


Der BFH prüft insofern nur , ob sie gegen Denkgesetze und Erfahrungssätze oder die anerkannten Auslegungsregeln verstößt .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Example 13** (doc_id: `54130`) (sent_id: `54130`)


b ) In Leasingfällen ( auch beim " Lease " im Rahmen eines " Sale-and-lease-back " ) geht der BFH - wie bereits in dem Urteil in BFHE 255 , 386 , BStBl II 2018 , 81 dargestellt - bei Anwendung des § 39 Abs. 2 Nr. 1 Satz 1 AO insbesondere von folgenden Grundsätzen aus :

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Missed by this rule (FN):**

- `Urteil in BFHE 255 , 386 , BStBl II 2018 , 81` (RS)
- `§ 39 Abs. 2 Nr. 1 Satz 1 AO` (NRM)

**Example 14** (doc_id: `54150`) (sent_id: `54150`)


Die Prüfung der Markenanmeldung muss daher nach der maßgeblichen Rechtsprechung des EuGH grundsätzlich streng und vollständig sein , um ungerechtfertigte Eintragungen zu vermeiden ( vgl. EuGH , GRUR 2003 , 604 Rn. 57 , 60 – Libertel ; BGH , GRUR 2014 , 565 Rn. 17 – smartbook ; Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 158 , 159 ) .

| Predicted | Gold |
|---|---|
| `EuGH` | `EuGH` |

**Missed by this rule (FN):**

- `EuGH , GRUR 2003 , 604 Rn. 57 , 60 – Libertel` (RS)
- `BGH , GRUR 2014 , 565 Rn. 17 – smartbook` (RS)
- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 158 , 159` (LIT)

**Example 15** (doc_id: `54257`) (sent_id: `54257`)


Indem die Absender die Postsendungen auf den Weg gebracht , zumindest in einem Teil der Fälle unzureichende Angaben gemacht oder Waren versandt haben , die gegen Verbote und Beschränkungen verstoßen könnten , haben sie zwar eine Bedingung für die vorübergehende Verwahrung bei der Zollstelle in Gestalt einer sog. conditio sine qua non gesetzt , welche allerdings allein für die Annahme eines " willentlichen Herbeiführens einer Amtshandlung " im Sinne vorgenannter Rechtsprechung des BVerwG ( Urteile in BVerwGE 91 , 109 , und in BVerwGE 153 , 321 ) nicht als ausreichend angesehen werden kann .

| Predicted | Gold |
|---|---|
| `BVerwG` | `BVerwG` |

**Missed by this rule (FN):**

- `Urteile in BVerwGE 91 , 109 , und in BVerwGE 153 , 321` (RS)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53392`) (sent_id: `53392`)


Bei Unklarheiten der Anmeldung ist daher der Wille des Anmelders durch Auslegung zu ermitteln ( vgl. BGH GRUR 2012 , 1139 , Nr. 23 , 30 - Weinkaraffe ; Eichmann / v. Falckenstein / Kühne , a. a. O. , § 37 Rn. 11 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH GRUR 2012 , 1139 , Nr. 23 , 30 - Weinkaraffe`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH GRUR 2012 , 1139 , Nr. 23 , 30 - Weinkaraffe`(RS)
- `Eichmann / v. Falckenstein / Kühne , a. a. O. , § 37 Rn. 11`(LIT)

**Example 1** (doc_id: `53395`) (sent_id: `53395`)


Darüber hinaus ist die Darlegung erforderlich , dass und warum die Entscheidung des LSG - ausgehend von dessen materieller Rechtsansicht - auf dem Mangel beruhen kann , dass also die Möglichkeit einer Beeinflussung der Entscheidung besteht .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `53422`) (sent_id: `53422`)


Bei der vom FG zu beurteilenden gesetzlichen Prozessführungsbefugnis ( Beteiligtenstellung ) handelt es sich um eine Sachurteilsvoraussetzung , deren fehlerhafte Beurteilung einen Verfahrensmangel darstellt ( BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943 ; vom 23. November 2010 V B 133/09 , BFH / NV 2011 , 612 , Rz 5 ; jeweils m. w. N. ) .

**False Positives:**

- `FG` — no gold match — likely missing annotation
- `BFH` — partial — pred is substring of gold: `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`
- `BFH` — partial — pred is substring of gold: `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`
- `BFH` — similar text (different position): `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`

> overlaps gold: 3  |  likely missing annotation: 1

**Gold Entities:**

- `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`(RS)
- `vom 23. November 2010 V B 133/09 , BFH / NV 2011 , 612 , Rz 5`(RS)

**Example 3** (doc_id: `53446`) (sent_id: `53446`)


( a ) Die Rechtslage im Nichtzulassungsbeschwerdeverfahren beruht allein darauf , dass ein Verfahrensmangel wie die verspätete Absetzung des Urteils kein Grund für die Zulassung der Revision war und ist ( vgl. zur Rechtslage vor Inkrafttreten des Anhörungsrügengesetzes : BVerfG 26. März 2001 - 1 BvR 383/00 - zu B I 2 c dd der Gründe ; BAG 1. Oktober 2003 - 1 ABN 62/01 - zu II 3 c der Gründe , BAGE 108 , 55 ; zur aktuellen Rechtslage nach § 72b Abs. 1 Satz 2 ArbGG BAG 24. Februar 2015 - 5 AZN 1007/14 - Rn. 3 ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG 26. März 2001 - 1 BvR 383/00 - zu B I 2 c dd der Gründe`
- `BAG` — partial — pred is substring of gold: `BAG 1. Oktober 2003 - 1 ABN 62/01 - zu II 3 c der Gründe , BAGE 108 , 55`
- `BAG` — similar text (different position): `BAG 1. Oktober 2003 - 1 ABN 62/01 - zu II 3 c der Gründe , BAGE 108 , 55`

> overlaps gold: 3  |  likely missing annotation: 0

**Gold Entities:**

- `Anhörungsrügengesetzes`(NRM)
- `BVerfG 26. März 2001 - 1 BvR 383/00 - zu B I 2 c dd der Gründe`(RS)
- `BAG 1. Oktober 2003 - 1 ABN 62/01 - zu II 3 c der Gründe , BAGE 108 , 55`(RS)
- `§ 72b Abs. 1 Satz 2 ArbGG`(NRM)
- `BAG 24. Februar 2015 - 5 AZN 1007/14 - Rn. 3`(RS)

**Example 4** (doc_id: `53451`) (sent_id: `53451`)


2. Ob dem FG im Zusammenhang mit der Vermögenszuwachsrechnung und der Geldverkehrsrechnung als selbständige Schätzungsgrundlagen für sich genommen ebenfalls Verfahrensfehler unterlaufen sind , ist vor diesem Hintergrund nicht mehr zu entscheiden .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `53453`) (sent_id: `53453`)


Zwar hat sich das FG im Urteil nicht ausdrücklich dazu geäußert , es hat die umstrittene Zahlung aber ohne weiteres als " Abfindungszahlung " bezeichnet und nicht infrage gestellt , dass es sich ( zumindest auch ) um eine Ersatzleistung für entgehende Einnahmen handeln sollte .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 6** (doc_id: `53468`) (sent_id: `53468`)


Wie das vorliegende Verfahren zeigt , bietet die Anhörungsrüge die Möglichkeit , durch Zuordnungsschwierigkeiten entstehende Probleme zu bewältigen ( vgl. BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15 ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15`(RS)

**Example 7** (doc_id: `53474`) (sent_id: `53474`)


Beim Leasing ist - wie dargestellt - darauf abzustellen , ob der Herausgabeanspruch des Leasinggebers ( zivilrechtlichen Eigentümers ) noch eine wirtschaftliche Bedeutung hat ( grundlegend BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1. ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1.`(RS)

**Example 8** (doc_id: `53475`) (sent_id: `53475`)


Gegenstand des BgA war die Verpachtung der städtischen Schwimmbäder an die ... GmbH ( S-GmbH ) , eine 100 % -ige Tochtergesellschaft der Klägerin .

**False Positives:**

- `BgA` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `... GmbH`(ORG)
- `S-GmbH`(ORG)

**Example 9** (doc_id: `53481`) (sent_id: `53481`)


Diese können insbesondere dann vorliegen , wenn die Gesamtstrafe sich nicht innerhalb des gesetzlichen Strafrahmens bewegt , die gebotene Begründung für die Gesamtstrafe fehlt , oder wenn die Besorgnis besteht , der Tatrichter habe sich von der Summe der Einzelstrafen leiten lassen ( BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris`(RS)

**Example 10** (doc_id: `53485`) (sent_id: `53485`)


Soweit die Elemente eines Bildzeichens nur die typischen Merkmale der in Rede stehenden Waren und Dienstleistungen darstellen oder sich in einfachen dekorativen Gestaltungsmitteln erschöpfen , an die sich der Verkehr etwa durch häufige Verwendung gewöhnt hat , wird diesem Zeichen im Allgemeinen wegen seines bloß beschreibenden Inhalts ebenfalls die konkrete Eignung fehlen , die mit ihm gekennzeichneten Waren oder Dienstleistungen von denjenigen anderer Herkunft zu unterscheiden ( vgl. BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband ; GRUR 2005 , 257 , 258 – Bürogebäude ; GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel ; GRUR 2001 , 239f. – Zahnpastastrang ; GRUR 2001 , 734 , 735 – Westie-Kopf ; BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband`(RS)
- `GRUR 2005 , 257 , 258 – Bürogebäude`(RS)
- `GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel`(RS)
- `GRUR 2001 , 239f. – Zahnpastastrang`(RS)
- `GRUR 2001 , 734 , 735 – Westie-Kopf`(RS)
- `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`(RS)

**Example 11** (doc_id: `53487`) (sent_id: `53487`)


Das SG verstand das Begehren des Klägers ( im Hauptantrag ) ebenfalls in dem Sinne , dass er mit einer Anfechtungsklage allein die Aufhebung des Widerspruchsbescheides vom 23. 7. 2015 verfolgte .

**False Positives:**

- `SG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 12** (doc_id: `53490`) (sent_id: `53490`)


Es müssen letztlich besondere Verhaltensweisen sowohl des Berechtigten als auch des Verpflichteten vorliegen , die es rechtfertigen , die späte Geltendmachung des Rechts als mit Treu und Glauben unvereinbar und für den Verpflichteten als unzumutbar anzusehen ( vgl. BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27 mwN ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27`(RS)

**Example 13** (doc_id: `53499`) (sent_id: `53499`)


f ) Zwar hätte die Klägerin aus eigenem Antrieb zur mündlichen Verhandlung erscheinen können , nachdem das FG bereits im Vorfeld mitgeteilt hatte , dass die Frage der ladungsfähigen Anschrift problematisch ist .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 14** (doc_id: `53524`) (sent_id: `53524`)


Der BFH führte aus , die handelsrechtliche Zurechnung von Vermögensgegenständen entspreche im Wesentlichen der Regelung des § 39 Abs. 2 Nr. 1 Satz 1 AO ( z.B. BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1. ; vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a ) .

**False Positives:**

- `BFH` — similar text (different position): `BFH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH`(ORG)
- `§ 39 Abs. 2 Nr. 1 Satz 1 AO`(NRM)
- `BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1.`(RS)
- `vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a`(RS)

**Example 15** (doc_id: `53539`) (sent_id: `53539`)


cc ) Stellt ein Bewerber nicht innerhalb eines Monats nach Zugang der Abbruchmitteilung einen Antrag auf Erlass einer einstweiligen Verfügung , darf der Dienstherr nach der Rechtsprechung des Bundesverwaltungsgerichts darauf vertrauen , dass der Bewerber den Abbruch des Auswahlverfahrens nicht angreift , sondern sein Begehren im Rahmen einer neuen Ausschreibung weiterverfolgt ( vgl. BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14 ) .

**False Positives:**

- `BVerwG` — partial — pred is substring of gold: `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesverwaltungsgerichts`(ORG)
- `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14`(RS)

**Example 16** (doc_id: `53551`) (sent_id: `53551`)


Die Beklagte wird das der Klägerin zustehende Honorar unter Zugrundelegung eines festen RLV für die gesamte BAG neu zu ermitteln haben .

**False Positives:**

- `BAG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 17** (doc_id: `53558`) (sent_id: `53558`)


Einen solchen qualifizierten Rechtsanwendungsfehler des FG hat die Klägerin indes nicht dargelegt .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 18** (doc_id: `53562`) (sent_id: `53562`)


Daher geht der Senat von einem Hauptbegehren des Klägers vor dem FG aus , das auf eine Änderung des Verlustfeststellungsbescheids zum 31. Dezember des Streitjahres vom 9. Oktober 2006 gerichtet war .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `53583`) (sent_id: `53583`)


Denn die Hauptfunktion einer Marke liegt darin , die Ursprungsidentität der gekennzeichneten Waren und Dienstleistungen zu gewährleisten ( vgl. u. a. EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel ; BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006 ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel`
- `BGH` — partial — pred is substring of gold: `BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel`(RS)
- `BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006`(RS)

**Example 20** (doc_id: `53598`) (sent_id: `53598`)


Dem FG seien für die Vorsteuerbeträge auch die freiwillig geführten Bestandskonten sowie sämtliche Datenerfassungsprotokolle und Buchungsjournale übermittelt worden .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 21** (doc_id: `53599`) (sent_id: `53599`)


Auch Angaben , die sich auf Umstände beziehen , die die Ware oder die Dienstleistung selbst nicht unmittelbar betreffen , fehlt die Unterscheidungskraft , wenn durch die Angabe ein enger beschreibender Bezug zu den angemeldeten Waren oder Dienstleistungen hergestellt wird und deshalb die Annahme gerechtfertigt ist , dass der Verkehr den beschreibenden Begriffsinhalt als solchen ohne Weiteres und ohne Unklarheiten erfasst und in der Bezeichnung nicht ein Unterscheidungsmittel für die Herkunft der angemeldeten Waren oder Dienstleistungen sieht ( BGH , GRUR 2014 , 569 , Rn. 10 – HOT ; BGH , GRUR 2012 , 1143 , Rn. 9 – Starsat ; BGH , GRUR 2009 , 952 , Rn. 10 – DeutschlandCard ; BGH , GRUR 2006 , 850 , Rn. 19 – FUSSBALL WM 2006 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , GRUR 2014 , 569 , Rn. 10 – HOT`
- `BGH` — similar text (different position): `BGH , GRUR 2014 , 569 , Rn. 10 – HOT`
- `BGH` — similar text (different position): `BGH , GRUR 2014 , 569 , Rn. 10 – HOT`
- `BGH` — similar text (different position): `BGH , GRUR 2014 , 569 , Rn. 10 – HOT`

> overlaps gold: 4  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , GRUR 2014 , 569 , Rn. 10 – HOT`(RS)
- `BGH , GRUR 2012 , 1143 , Rn. 9 – Starsat`(RS)
- `BGH , GRUR 2009 , 952 , Rn. 10 – DeutschlandCard`(RS)
- `BGH , GRUR 2006 , 850 , Rn. 19 – FUSSBALL WM 2006`(RS)

**Example 22** (doc_id: `53600`) (sent_id: `53600`)


Auch danach setzt die Bejahung der Unterscheidungskraft unverändert voraus , dass das Zeichen geeignet sein muss , die beanspruchten Produkte als von einem bestimmten Unternehmen stammend zu kennzeichnen ( vgl. EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK`(RS)

**Example 23** (doc_id: `53601`) (sent_id: `53601`)


a ) Das FG hielt - bei einer Grundmietzeit von vier Jahren - eine betriebsgewöhnliche Nutzungsdauer der Leasingobjekte von drei bis fünf Jahren für möglich .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 24** (doc_id: `53614`) (sent_id: `53614`)


Soweit Personen dezentral untergebracht sind , ist es für die Bejahung einer Einrichtung erforderlich , dass die dezentrale Unterkunft zu den Räumlichkeiten der Einrichtung gehört , der Hilfebedürftige also in die Räumlichkeiten des Einrichtungsträgers eingegliedert ist ( vgl BSG SozR 4 - 3500 § 98 Nr 3 ) .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG SozR 4 - 3500 § 98 Nr 3`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG SozR 4 - 3500 § 98 Nr 3`(RS)

**Example 25** (doc_id: `53618`) (sent_id: `53618`)


Denn die Hauptfunktion einer Marke besteht darin , die Ursprungsidentität der gekennzeichneten Waren oder Dienstleistungen zu gewährleisten ( vgl. etwa EuGH GRUR 2010 , 1008 , 1009 ( Nr. 38 ) - Lego ; GRUR 2008 , 608 , 611 ( Nr. 66 ) - EUROHYPO ; GRUR 2006 , 233 , 235 , Nr. 45 - Standbeutel ; BGH GRUR 2015 , 173 , 174 ( Nr. 15 ) - for you ; GRUR 2009 , 949 ( Nr. 10 ) - My World ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH GRUR 2010 , 1008 , 1009 ( Nr. 38 ) - Lego`
- `BGH` — partial — pred is substring of gold: `BGH GRUR 2015 , 173 , 174 ( Nr. 15 ) - for you`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2010 , 1008 , 1009 ( Nr. 38 ) - Lego`(RS)
- `GRUR 2008 , 608 , 611 ( Nr. 66 ) - EUROHYPO`(RS)
- `GRUR 2006 , 233 , 235 , Nr. 45 - Standbeutel`(RS)
- `BGH GRUR 2015 , 173 , 174 ( Nr. 15 ) - for you`(RS)
- `GRUR 2009 , 949 ( Nr. 10 ) - My World`(RS)

**Example 26** (doc_id: `53623`) (sent_id: `53623`)


Nach der von der Einsprechenden zitierten BGH-Entscheidung „ Kommunikationskanal “ , muss für den Fachmann die im Anspruch bezeichnete Lehre „ unmittelbar und eindeutig “ als mögliche Ausführungsform der Erfindung den Ursprungsunterlagen entnehmbar sein .

**False Positives:**

- `BGH` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 27** (doc_id: `53654`) (sent_id: `53654`)


Entsprechendes gilt , wenn ihre Rüge als Aufklärungsrüge verstanden werden sollte ; auch dazu hätte sie darlegen müssen , warum sich das LSG über die geschilderte Wahrnehmung der Klägerin hinaus zu weiterer Aufklärung hätte gedrängt sehen müssen ( vgl Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160a RdNr 16f ) .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160a RdNr 16f`(LIT)

**Example 28** (doc_id: `53660`) (sent_id: `53660`)


Das LSG hat nach mündlicher Verhandlung in Abwesenheit des Klägers die Berufung zurückgewiesen .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 29** (doc_id: `53661`) (sent_id: `53661`)


Die Klage hatte keinen Erfolg ; das Finanzgericht ( FG ) München hat sie mit Urteil vom 6. Juli 2017 11 K 411/13 als unbegründet abgewiesen .

**False Positives:**

- `FG` — partial — pred is substring of gold: `Finanzgericht ( FG ) München`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Finanzgericht ( FG ) München`(ORG)
- `Urteil vom 6. Juli 2017 11 K 411/13`(RS)

</details>

---

## `Quoted Brand Names (Strict)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `63bcacd5`  
**Description:**
Matches brand names enclosed in German quotation marks, excluding common non-brand terms like 'Entscheidungen', 'Urteil', 'Beschluss', etc.

**Content:**
```
(?:\u201e|\")\s*([A-Za-z\u00e4\u00f6\u00fc\u00df\s]+)\s*(?:\u201e|\")
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 224 | 0 | 224 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 0 | 224 | 798 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53417`) (sent_id: `53417`)


aa ) Das Amt des " Kanzlers " hat seine Wurzeln im Mittelalter und spiegelt die Entwicklung der modernen Universität bis in die heutigen Tage .

**False Positives:**

- `Kanzlers ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `53453`) (sent_id: `53453`)


Zwar hat sich das FG im Urteil nicht ausdrücklich dazu geäußert , es hat die umstrittene Zahlung aber ohne weiteres als " Abfindungszahlung " bezeichnet und nicht infrage gestellt , dass es sich ( zumindest auch ) um eine Ersatzleistung für entgehende Einnahmen handeln sollte .

**False Positives:**

- `Abfindungszahlung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `53491`) (sent_id: `53491`)


die unter 2. bis 5. begehrten Informationen jeweils nur für den " Hintergrund " , also vertraulich und nicht zur Verwendung für eine öffentliche Berichterstattung mit Quellenangabe zu erteilen .

**False Positives:**

- `Hintergrund ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 3** (doc_id: `53588`) (sent_id: `53588`)


Diesbezüglich fehlt es insbesondere an Vorbringen dazu , dass die Entscheidung des LSG auf " diesem Mangel " beruhen kann , dh es hätte Darlegungen zur mangelnden oder zumindest eingeschränkten Verwertbarkeit des Gutachtens bedurft .

**False Positives:**

- `diesem Mangel ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `53592`) (sent_id: `53592`)


DDO / DtA ENJJPT stellt den Dienstbetrieb im " Euro Nato Joint Jet Pilot Training " ( ENJJPT ) sicher .

**False Positives:**

- `Euro Nato Joint Jet Pilot Training ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `53615`) (sent_id: `53615`)


Weil Impfstoffe Arzneimittel ( auch ) iS des § 31 SGB V sind , wird ihre Verordnung auch von § 10 Abs 2 der Prüfvereinbarung erfasst , soweit diese Vorschrift das " Verordnungsverhalten " der Vertragsärzte betrifft .

**False Positives:**

- `Verordnungsverhalten ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 31 SGB V`(NRM)
- `§ 10 Abs 2 der Prüfvereinbarung`(REG)

**Example 6** (doc_id: `53630`) (sent_id: `53630`)


Die Verträge seien derart angelegt gewesen , dass die KG am Ende der Laufzeit ihr " Andienungsrecht " ausübe und die P GmbH die Leasingobjekte zurückerwerben müsse .

**False Positives:**

- `Andienungsrecht ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `P GmbH`(ORG)

**Example 7** (doc_id: `53648`) (sent_id: `53648`)


Das FA vertritt indes die Auffassung , der für das erste Betriebsjahr vereinbarte Abschlag auf den voraussichtlichen Jahresüberschuss in Höhe von 40 % des Nettokaufpreises sei als Vereinbarung der Rückzahlung der " Darlehenssumme " anzusehen .

**False Positives:**

- `Darlehenssumme ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `53720`) (sent_id: `53720`)


cc ) Ohne dass es hierauf nach dem Vorstehenden rechtlich noch ankäme , weist der Senat darauf hin , dass er auch der in der mündlichen Verhandlung vorgebrachten Auffassung des FA nicht folgen kann , dem Kläger habe allenfalls eine " abschnittsweise Verlustbeteiligung " , aber keine " endgültige Verlustbeteiligung " gedroht .

**False Positives:**

- `abschnittsweise Verlustbeteiligung ` — no gold match — likely missing annotation
- `endgültige Verlustbeteiligung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 9** (doc_id: `53729`) (sent_id: `53729`)


Nach dem während der ambulanten Behandlung der Tochter erstellten Bericht habe sich das Mädchen auf einer Gesamtskala im Störungsbereich " Depression " auffällig gezeigt .

**False Positives:**

- `Depression ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 10** (doc_id: `53767`) (sent_id: `53767`)


Ob der Erwerb eigener Anteile auf der Gesellschaftsebene entsprechend der durch das BilMoG geänderten handelsrechtlichen Vorschriften ( Einfügung des § 272 Abs. 1a und 1b HGB ) steuerrechtlich nicht mehr als Erwerbsvorgang anzusehen , sondern nunmehr als " Teilliquidation " und daher " wie " eine Kapitalherabsetzung zu behandeln ist ( so BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f. ; zustimmend z.B. Blumenberg / Lechner , DB 2014 , 141 , 147 ; kritisch z.B. Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55 ) , kann im Streitfall mangels Erheblichkeit für die Entscheidung offenbleiben .

**False Positives:**

- `Teilliquidation ` — no gold match — likely missing annotation
- `wie ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `BilMoG`(NRM)
- `§ 272 Abs. 1a und 1b HGB`(NRM)
- `BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f.`(REG)
- `Blumenberg / Lechner , DB 2014 , 141 , 147`(LIT)
- `Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55`(LIT)

**Example 11** (doc_id: `53836`) (sent_id: `53836`)


Stattgebender Kammerbeschluss : Anforderungen der Rechtsschutzgarantie ( Art 19 Abs 4 S 1 GG , hier iVm Art 2 Abs 2 S 1 GG ) an die Begründung der Abweisung einer Klage auf Zuerkennung internationalen Schutzes sowie auf Feststellung von nationalen Abschiebungsverboten als offensichtlich unbegründet - Pflicht zur " tagesaktuellen " Beurteilung der Sicherheitslage in Afghanistan steht Bildung einer insofern gefestigten obergerichtlichen Rspr entgegen , mithin keine Grundlage für Versagung subsidiären Schutzes wegen offensichtlicher Unbegründetheit - Gegenstandswertfestsetzung

**False Positives:**

- `tagesaktuellen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Art 19 Abs 4 S 1 GG`(NRM)
- `Art 2 Abs 2 S 1 GG`(NRM)
- `Afghanistan`(LOC)

**Example 12** (doc_id: `53863`) (sent_id: `53863`)


Eine Trennung nach " Streikbeamten " und sonstigen Beamten widerspreche insbesondere dem hergebrachten Grundsatz der Einheit des Berufsbeamtentums .

**False Positives:**

- `Streikbeamten ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 13** (doc_id: `53916`) (sent_id: `53916`)


Unter Bezugnahme auf die entsprechenden Feststellungen des Landgerichts stützt sich die Entscheidung darauf , dass der Beschwerdeführer einer aus rund 80 Personen bestehenden Gruppe namens " Schickeria " aus der gewaltbereiten " Ultra " -Szene angehört und sich nach dem fraglichen Spiel in einer Gruppe befunden habe , aus welcher heraus es tatsächlich in erheblichem Umfang zu Provokationen und Körperverletzungsdelikten gekommen sei .

**False Positives:**

- `Schickeria ` — no gold match — likely missing annotation
- `Ultra ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 14** (doc_id: `54096`) (sent_id: `54096`)


Die Folgen des " fehlenden " Kopfteils für die anderen Mitglieder des Haushalts aufgrund einer Versagung gegenüber einem dritten Mitglied , weil dieses die ua in §§ 60 ff SGB I iVm § 9 und §§ 11 ff SGB II zum Ausdruck kommenden Verhaltenserwartungen nicht erfüllt , sind nicht durch höhere Einzelansprüche der anderen Haushaltsmitglieder auszugleichen ( dazu im Einzelnen 4. und 5. ) .

**False Positives:**

- `fehlenden ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§§ 60 ff SGB I`(NRM)
- `§ 9 und §§ 11 ff SGB II`(NRM)

**Example 15** (doc_id: `54106`) (sent_id: `54106`)


Diese Entscheidung hat die Beklagte während des Klageverfahrens " abgeändert " und festgestellt , dass der Kläger in seinen für die Beigeladene zu 1. ausgeübten Tätigkeiten in allen Zweigen der Sozialversicherung wegen Beschäftigung versicherungspflichtig gewesen sei ; die Versicherungspflicht habe schon mit der Aufnahme seiner Beschäftigung begonnen , weil er ausreichenden Versicherungsschutz iS von § 7a Abs 6 S 1 Nr 2 SGB IV nicht nachgewiesen habe ( Bescheid vom 24. 1. 2011 ) .

**False Positives:**

- `abgeändert ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 7a Abs 6 S 1 Nr 2 SGB IV`(NRM)

**Example 16** (doc_id: `54130`) (sent_id: `54130`)


b ) In Leasingfällen ( auch beim " Lease " im Rahmen eines " Sale-and-lease-back " ) geht der BFH - wie bereits in dem Urteil in BFHE 255 , 386 , BStBl II 2018 , 81 dargestellt - bei Anwendung des § 39 Abs. 2 Nr. 1 Satz 1 AO insbesondere von folgenden Grundsätzen aus :

**False Positives:**

- `Lease ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BFH`(ORG)
- `Urteil in BFHE 255 , 386 , BStBl II 2018 , 81`(RS)
- `§ 39 Abs. 2 Nr. 1 Satz 1 AO`(NRM)

**Example 17** (doc_id: `54134`) (sent_id: `54134`)


bb ) Die Bedeutung der Konjunktion " soweit " ist jedoch nicht auf solche Fälle beschränkt .

**False Positives:**

- `soweit ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 18** (doc_id: `54142`) (sent_id: `54142`)


- einem Betrag in Höhe des als Kaufoptionspreis im Anhang 2 aufgeführten Betrags , der um den Betrag einer etwa bereits gezahlten Abschlusszahlung zu mindern ist , - einem nach näheren Maßgaben zu ermittelnden sog. " höheren Marktwert " und- der variablen Gebühr .

**False Positives:**

- `höheren Marktwert ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `54229`) (sent_id: `54229`)


Der Sache nach hat der Beklagte einen Regress wegen unwirtschaftlicher Verordnung von Impfstoffen und keinen verschuldensabhängigen Ersatz wegen der Verursachung eines " sonstigen Schadens " festgesetzt ( c ) .

**False Positives:**

- `sonstigen Schadens ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 20** (doc_id: `54257`) (sent_id: `54257`)


Indem die Absender die Postsendungen auf den Weg gebracht , zumindest in einem Teil der Fälle unzureichende Angaben gemacht oder Waren versandt haben , die gegen Verbote und Beschränkungen verstoßen könnten , haben sie zwar eine Bedingung für die vorübergehende Verwahrung bei der Zollstelle in Gestalt einer sog. conditio sine qua non gesetzt , welche allerdings allein für die Annahme eines " willentlichen Herbeiführens einer Amtshandlung " im Sinne vorgenannter Rechtsprechung des BVerwG ( Urteile in BVerwGE 91 , 109 , und in BVerwGE 153 , 321 ) nicht als ausreichend angesehen werden kann .

**False Positives:**

- `willentlichen Herbeiführens einer Amtshandlung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BVerwG`(ORG)
- `Urteile in BVerwGE 91 , 109 , und in BVerwGE 153 , 321`(RS)

**Example 21** (doc_id: `54293`) (sent_id: `54293`)


Auch kann dies nicht - wie von der Beklagten vorgetragen - als eine weitere Voraussetzung , unter der erst ein Zusammenhang zwischen Berufstätigkeit und Pflichtmitgliedschaft zu bejahen ist , aus der Verwendung der Präposition " wegen " entnommen werden .

**False Positives:**

- `wegen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 22** (doc_id: `54299`) (sent_id: `54299`)


Auch vor dem Hintergrund der Einschätzung der Sachverständigen , dass beim Beschwerdeführer eine deutlich höhere Gefahr von " Hands-off " -Übergriffen im Vergleich zu " Hands-on " -Delikten bestehe , hätte es der konkreten Darlegung der vom Beschwerdeführer drohenden Straftaten bedurft , um die Gefahr " erheblicher Straftaten " im Sinne von § 67d Abs. 2 StGB feststellen und die Verhältnismäßigkeit einer weiteren Unterbringung des Beschwerdeführers bewerten zu können .

**False Positives:**

- `erheblicher Straftaten ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 67d Abs. 2 StGB`(NRM)

**Example 23** (doc_id: `54371`) (sent_id: `54371`)


Hierdurch wurden dem etablierten Arzt der BAG typischerweise deutlich weniger RLV-relevante Fälle zugeordnet , als er real behandelte , während der " Wachstumsarzt " bei der von der Beklagten praktizierten Verfahrensweise nur die Zahl der von ihm im Abrechnungsquartal tatsächlich betreuten Fälle RLV-relevant vergütet bekam .

**False Positives:**

- `Wachstumsarzt ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 24** (doc_id: `54373`) (sent_id: `54373`)


Das Landgericht Düsseldorf begründete diese Anordnung damit , dass die Disposition zu " solchen Taten " tief im Beschwerdeführer verwurzelt sei .

**False Positives:**

- `solchen Taten ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Landgericht Düsseldorf`(ORG)

**Example 25** (doc_id: `54396`) (sent_id: `54396`)


In dem Merkblatt wird zur " Erschöpfung des Rechtswegs " erläutert , dass die Möglichkeit genutzt werden muss , den Grundrechtsverstoß " im Verfahren vor den Fachgerichten abzuwenden " .

**False Positives:**

- `Erschöpfung des Rechtswegs ` — no gold match — likely missing annotation
- `im Verfahren vor den Fachgerichten abzuwenden ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 26** (doc_id: `54428`) (sent_id: `54428`)


Der dort vorgesehene Versetzungsschutz bei Versetzungen in zeitlicher Nähe zum Dienstzeitende wird nicht , wie das Bundesministerium der Verteidigung einwendet , durch die Vorgaben der Zentralen Dienstvorschrift A- 1350/66 über die " Letzte Verwendung vor Zurruhesetzung " ausgeschlossen .

**False Positives:**

- `Letzte Verwendung vor Zurruhesetzung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Bundesministerium der Verteidigung`(ORG)
- `Zentralen Dienstvorschrift A- 1350/66`(REG)

**Example 27** (doc_id: `54445`) (sent_id: `54445`)


bbb ) Das FG hat feststellt , der Kläger habe für die Streitjahre Umsatzsteuer-Voranmeldungen und -Erklärungen abgegeben und zur Prüfung der geltend gemachten Vorsteuerbeträge und Betriebsausgaben mit Vorsteuerabzug alle Belege und Summenziehungen ( die " Bestände der Vorsteuerbeträge " ) im Klageverfahren vorgelegt .

**False Positives:**

- `Bestände der Vorsteuerbeträge ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 28** (doc_id: `54483`) (sent_id: `54483`)


Nachdem die Schüler im Musikunterricht die theoretischen Grundlagen zum Thema " Musik und Werbung " bzw " Wirkung von Musik " erarbeitet hatten , sollten sie in Kleingruppen einen Werbeclip zu einem bestimmten Produkt filmen , schneiden , bearbeiten und mit passender Musik unterlegen .

**False Positives:**

- `Musik und Werbung ` — no gold match — likely missing annotation
- `Wirkung von Musik ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 29** (doc_id: `54508`) (sent_id: `54508`)


2. Die Versetzung des Antragstellers nach C. verstößt jedoch gegen die nach dem Grundsatz der Gleichbehandlung ( Art. 3 Abs. 1 GG ) zu beachtenden Maßgaben der Bereichsvorschrift C1 - 1310/0 - 2001 über die " Organisatorische und personelle Umsetzung von Strukturentscheidungen in der Luftwaffe " ( siehe bereits Beschluss vom 13. Dezember 2017 - 1 WDS-VR 9.17 - Rn. 33 bis 35 ) .

**False Positives:**

- `Organisatorische und personelle Umsetzung von Strukturentscheidungen in der Luftwaffe ` — partial — gold is substring of pred: `Luftwaffe`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C.`(LOC)
- `Art. 3 Abs. 1 GG`(NRM)
- `Bereichsvorschrift C1 - 1310/0 - 2001`(REG)
- `Luftwaffe`(ORG)
- `Beschluss vom 13. Dezember 2017 - 1 WDS-VR 9.17 - Rn. 33 bis 35`(RS)

</details>

---

## `Specific Court Names with Location (Fixed)`

**F1:** 0.020 | **Precision:** 0.092 | **Recall:** 0.011  

**Format:** `regex`  
**Rule ID:** `42e5b2a1`  
**Description:**
Matches court names with location, ensuring the court type is present and not just the location, handling multi-word locations like 'Frankfurt am Main' and hyphenated locations like 'Niedersachsen-Bremen'.

**Content:**
```
\b(?:Amtsgericht|Landgericht|Verwaltungsgericht|Finanzgericht|Sozialgericht|Arbeitsgericht|Oberlandesgericht|Bundesverwaltungsgericht|Bundesfinanzhof|Bundesgerichtshof|Bundessozialgericht|Bundesarbeitsgericht|Bundespatentgericht|BVerwG|BGH|BSG|BAG|BVerfG|EuGH|EGMR|DPMA|BaFin|SG|FG|LAG|LSG|ArbG|Landessozialgericht|Landesarbeitsgericht|Landesverwaltungsgericht|Oberverwaltungsgericht|Verwaltungsgerichtshof|Schleswig-Holsteinische\s+Oberverwaltungsgericht|Truppendienstgericht|Anwaltsgerichtshof)\s+(?:[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)*|[A-Z]\.|[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+-[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+|<\s*[A-Z]{2,3}\s*>\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)*|Frankfurt\s+am\s+Main|Berlin-Brandenburg|Niedersachsen-Bremen|Baden-W\u00fcrttemberg|Nordrhein-Westfalen|Rheinland-Pfalz|Schleswig-Holstein)\b(?!\s+(?:Prozesskostenhilfe|Beschwerde|Verfahren|Urteil|Beschluss|Sache|Rechtsprechung|in|auf|von|mit|bei|nach|f\u00fcr|zum|zur|des|der|die|den|Senat|Nr\.|\.)|\s+(?:Prozesskostenhilfe|Beschwerde|Verfahren|Urteil|Beschluss|Sache|Rechtsprechung|in|auf|von|mit|bei|nach|f\u00fcr|zum|zur|des|der|die|den|Senat|Nr\.))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.092 | 0.011 | 0.020 | 98 | 9 | 89 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 9 | 89 | 780 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53602`) (sent_id: `53602`)


Am 21. September 2017 beschloss das Landgericht Memmingen , der weiteren Beschwerde vom 15. September 2017 nicht abzuhelfen .

| Predicted | Gold |
|---|---|
| `Landgericht Memmingen` | `Landgericht Memmingen` |

**Example 1** (doc_id: `54373`) (sent_id: `54373`)


Das Landgericht Düsseldorf begründete diese Anordnung damit , dass die Disposition zu " solchen Taten " tief im Beschwerdeführer verwurzelt sei .

| Predicted | Gold |
|---|---|
| `Landgericht Düsseldorf` | `Landgericht Düsseldorf` |

**Example 2** (doc_id: `54861`) (sent_id: `54861`)


Das Arbeitsgericht Zwickau verurteilte die Beklagte am 22. April 2015 ( - 9 Ca 146/15 - ) , das abgebrochene Stellenbesetzungsverfahren 01/2014 fortzuführen und über die Bewerbung des Klägers erneut zu entscheiden .

| Predicted | Gold |
|---|---|
| `Arbeitsgericht Zwickau` | `Arbeitsgericht Zwickau` |

**Missed by this rule (FN):**

- `22. April 2015 ( - 9 Ca 146/15 - )` (RS)

**Example 3** (doc_id: `55283`) (sent_id: `55283`)


Die Klägerin beantragt , das Urteil des Sächsischen LSG vom 19. Mai 2016 aufzuheben und die Berufung der Beklagten gegen das Urteil des SG Chemnitz vom 9. Oktober 2014 zurückzuweisen .

| Predicted | Gold |
|---|---|
| `SG Chemnitz` | `SG Chemnitz` |

**Missed by this rule (FN):**

- `Sächsischen LSG` (ORG)

**Example 4** (doc_id: `56408`) (sent_id: `56408`)


Verwaltungs- , Widerspruchs- und erstinstanzliches Verfahren waren erfolglos ( Bescheid vom 2. 4. 2014 , Widerspruchsbescheid vom 18. 7. 2014 , Gerichtsbescheid des SG Karlsruhe vom 12. 8. 2015 ) .

| Predicted | Gold |
|---|---|
| `SG Karlsruhe` | `SG Karlsruhe` |

**Example 5** (doc_id: `56519`) (sent_id: `56519`)


Die Klägerin beantragt , das Urteil des Schleswig-Holsteinischen LSG vom 15. 11. 2016 aufzuheben und die Berufung der Beklagten gegen das Urteil des SG Kiel vom 29. 1. 2014 mit der Maßgabe zurückzuweisen , dass die Beklagte bei der erneuten Bescheidung die Rechtsauffassung des Senats zu beachten hat .

| Predicted | Gold |
|---|---|
| `SG Kiel` | `SG Kiel` |

**Missed by this rule (FN):**

- `Schleswig-Holsteinischen LSG` (ORG)

**Example 6** (doc_id: `57841`) (sent_id: `57841`)


2. Das Amtsgericht Dieburg gab der Klage mit Urteil vom 7. Dezember 2012 statt , erklärte die Zwangsvollstreckung aus dem Vollstreckungsbescheid insgesamt für unzulässig und verurteilte den Beklagten , die vollstreckbare Ausfertigung an den Beschwerdeführer herauszugeben ; alle Forderungen des Beklagten gegen den Beschwerdeführer seien getilgt .

| Predicted | Gold |
|---|---|
| `Amtsgericht Dieburg` | `Amtsgericht Dieburg` |

**Example 7** (doc_id: `59802`) (sent_id: `59802`)


Das SG Karlsruhe hat mit Urteil vom 25. 3. 2015 die Klage mit der Begründung abgewiesen , der Kläger übe im Wesentlichen die Tätigkeit eines Pharmareferenten und damit keine für die Berufsgruppe der Tierärzte spezifische Tätigkeit aus .

| Predicted | Gold |
|---|---|
| `SG Karlsruhe` | `SG Karlsruhe` |

**Example 8** (doc_id: `59867`) (sent_id: `59867`)


Das Landgericht Memmingen verwarf die Beschwerde vom 30. August 2017 mit Beschluss vom 11. September 2017 als unbegründet .

| Predicted | Gold |
|---|---|
| `Landgericht Memmingen` | `Landgericht Memmingen` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53508`) (sent_id: `53508`)


Beruhen vielmehr die Differenzen zwischen den Auffassungen von Sachverständigen darauf , dass diese von verschiedenen tatsächlichen Annahmen ausgehen , dann muss der Tatrichter , ggf nach weiterer Aufklärung , die für seine Überzeugungsbildung maßgebenden Tatsachen feststellen oder begründen , weshalb und zu wessen Lasten sie beweislos geblieben sind ( vgl BGH Urteil vom 23. 9. 1986 - VI ZR 261/85 ) .

**False Positives:**

- `BGH Urteil` — partial — pred is substring of gold: `BGH Urteil vom 23. 9. 1986 - VI ZR 261/85`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH Urteil vom 23. 9. 1986 - VI ZR 261/85`(RS)

**Example 1** (doc_id: `53570`) (sent_id: `53570`)


Sie beruft sich für ihr Zulassungsbegehren ausschließlich auf einen Verfahrensmangel ( § 160 Abs 2 Nr 3 SGG ) , weil das LSG Beweisanträgen ohne hinreichenden Grund nicht gefolgt sei .

**False Positives:**

- `LSG Beweisanträgen` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 160 Abs 2 Nr 3 SGG`(NRM)

**Example 2** (doc_id: `53582`) (sent_id: `53582`)


3. Wird einem Beteiligten das rechtliche Gehör dadurch versagt , dass es ihm nicht ermöglicht wird , an der mündlichen Verhandlung teilzunehmen , so ist davon auszugehen , dass dies für eine aufgrund dieser Verhandlung ergangenen Entscheidung ursächlich geworden ist ( vgl BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7 ) .

**False Positives:**

- `BSG Beschluss` — partial — pred is substring of gold: `BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7`(RS)

**Example 3** (doc_id: `53822`) (sent_id: `53822`)


Dies folgt aus § 73b Abs 5 S 4 SGB V , der Abweichungen von den Vorschriften des Vierten Kapitels und damit auch von dem in § 71 Abs 1 S 1 SGB V verankerten Grundsatz der Beitragssatzstabilität zulässt ( BSG Urteil vom 25. 3. 2015 - B 6 KA 9/14 R - BSGE 118 , 164 = SozR 4 - 2500 § 73b Nr 1 , RdNr 72 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 25. 3. 2015 - B 6 KA 9/14 R - BSGE 118 , 164 = SozR 4 - 2500 § 73b Nr 1 , RdNr 72`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 73b Abs 5 S 4 SGB V`(NRM)
- `§ 71 Abs 1 S 1 SGB V`(NRM)
- `BSG Urteil vom 25. 3. 2015 - B 6 KA 9/14 R - BSGE 118 , 164 = SozR 4 - 2500 § 73b Nr 1 , RdNr 72`(RS)

**Example 4** (doc_id: `53848`) (sent_id: `53848`)


Vielmehr setzt der Sinn und Zweck der Vorschrift voraus , dass auch das konkrete Verfahren von dem Sozialleistungsträger gerade in dieser Eigenschaft geführt wird ; das Verfahren muss also einen engen sachlichen Zusammenhang zu der gesetzlichen Tätigkeit als Träger der in der Vorschrift genannten Sozialleistungen haben ( BGH Beschluss vom 10. 11. 2005 - IX ZR 189/02 - Juris RdNr 6 f ; BGH Beschluss vom 28. 9. 2016 - XII ZB 251/16 - Juris RdNr 30 ) .

**False Positives:**

- `BGH Beschluss` — partial — pred is substring of gold: `BGH Beschluss vom 10. 11. 2005 - IX ZR 189/02 - Juris RdNr 6 f`
- `BGH Beschluss` — similar text (different position): `BGH Beschluss vom 10. 11. 2005 - IX ZR 189/02 - Juris RdNr 6 f`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH Beschluss vom 10. 11. 2005 - IX ZR 189/02 - Juris RdNr 6 f`(RS)
- `BGH Beschluss vom 28. 9. 2016 - XII ZB 251/16 - Juris RdNr 30`(RS)

**Example 5** (doc_id: `53879`) (sent_id: `53879`)


Bemüht sich jemand , der ein Statusfeststellungsverfahren einleitet , zeitnah um private Eigenvorsorge , so kann er diese für den Fall , dass das Statusfeststellungsverfahren entgegen seinen Vorstellungen zu einer Feststellung von Versicherungspflicht führt , möglicherweise gar nicht mehr oder nur mit erheblichem Aufwand rückabwickeln ( zu diesen Konsequenzen siehe LSG Berlin-Brandenburg Urteil vom 18. 9. 2013 - L 9 KR 384/11 - Juris RdNr 38 ) .

**False Positives:**

- `LSG Berlin` — partial — pred is substring of gold: `LSG Berlin-Brandenburg Urteil vom 18. 9. 2013 - L 9 KR 384/11 - Juris RdNr 38`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `LSG Berlin-Brandenburg Urteil vom 18. 9. 2013 - L 9 KR 384/11 - Juris RdNr 38`(RS)

**Example 6** (doc_id: `53991`) (sent_id: `53991`)


Nicht das tatsächliche Verhalten des Arbeitgebers im Lohnsteuerabzugsverfahren bindet dessen Beteiligte ( vgl BSG Urteil vom 23. 5. 2017 - B 12 KR 6/16 R - SozR 4 - 5376 § 1 Nr 1 RdNr 23 ; BSG Urteil vom 26. 3. 2014 - B 10 EG 14/13 R - BSGE 115 , 198 = SozR 4 - 7837 § 2 Nr 25 , RdNr 26 f ) , wohl aber die Rechtsfolgen , die AO und EStG daran knüpfen .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 23. 5. 2017 - B 12 KR 6/16 R - SozR 4 - 5376 § 1 Nr 1 RdNr 23`
- `BSG Urteil` — similar text (different position): `BSG Urteil vom 23. 5. 2017 - B 12 KR 6/16 R - SozR 4 - 5376 § 1 Nr 1 RdNr 23`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Urteil vom 23. 5. 2017 - B 12 KR 6/16 R - SozR 4 - 5376 § 1 Nr 1 RdNr 23`(RS)
- `BSG Urteil vom 26. 3. 2014 - B 10 EG 14/13 R - BSGE 115 , 198 = SozR 4 - 7837 § 2 Nr 25 , RdNr 26 f`(RS)
- `AO`(NRM)
- `EStG`(NRM)

**Example 7** (doc_id: `54082`) (sent_id: `54082`)


Den Beteiligten war nämlich bewusst , denn dies ist sogar protokolliert , dass gegen das besprochene Urteil des FG Berlin-Brandenburg noch eine Revision anhängig war .

**False Positives:**

- `FG Berlin` — partial — pred is substring of gold: `FG Berlin-Brandenburg`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `FG Berlin-Brandenburg`(ORG)

**Example 8** (doc_id: `54171`) (sent_id: `54171`)


Ein anderer als der vom LSG herangezogene Prüfungsmaßstab unter Anwendung weiterer Vorschriften des Bundesrechts folgt entgegen der Rechtsauffassung der Beklagten auch nicht aus einem Beschluss des Senats , in dem die Revision gegen ein Urteil des LSG Baden-Württemberg vom 23. 1. 2013 ( L 5 R 4971/10 ) als unzulässig verworfen wurde ( Senatsbeschluss vom 6. 2. 2014 - B 5 RE 10/14 R ) .

**False Positives:**

- `LSG Baden` — partial — pred is substring of gold: `Urteil des LSG Baden-Württemberg vom 23. 1. 2013 ( L 5 R 4971/10 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des LSG Baden-Württemberg vom 23. 1. 2013 ( L 5 R 4971/10 )`(RS)
- `Senatsbeschluss vom 6. 2. 2014 - B 5 RE 10/14 R`(RS)

**Example 9** (doc_id: `54270`) (sent_id: `54270`)


Sie beantragt , das Urteil des FG Berlin-Brandenburg vom 18. Januar 2017 3 K 3219/16 aufzuheben und die Klage abzuweisen .

**False Positives:**

- `FG Berlin` — partial — pred is substring of gold: `Urteil des FG Berlin-Brandenburg vom 18. Januar 2017 3 K 3219/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des FG Berlin-Brandenburg vom 18. Januar 2017 3 K 3219/16`(RS)

**Example 10** (doc_id: `54285`) (sent_id: `54285`)


Den vom Kläger aus den Rechnungen über den Ankauf der BHKW beanspruchten , vorliegend nicht verfahrensgegenständlichen Vorsteuerabzug erkannte der Beklagte und Revisionskläger ( das Finanzamt - FA - ) nicht an ( bestätigt durch FG Münster , Urteil vom 16. Oktober 2014 5 K 3875/12 U , Entscheidungen der Finanzgerichte - EFG - 2015 , 84 , rechtskräftig ) .

**False Positives:**

- `FG Münster` — partial — pred is substring of gold: `FG Münster , Urteil vom 16. Oktober 2014 5 K 3875/12 U , Entscheidungen der Finanzgerichte - EFG - 2015 , 84`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `FG Münster , Urteil vom 16. Oktober 2014 5 K 3875/12 U , Entscheidungen der Finanzgerichte - EFG - 2015 , 84`(RS)

**Example 11** (doc_id: `54500`) (sent_id: `54500`)


Die Sache wird an das Finanzgericht Rheinland-Pfalz zurückverwiesen .

**False Positives:**

- `Finanzgericht Rheinland` — partial — pred is substring of gold: `Finanzgericht Rheinland-Pfalz`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Finanzgericht Rheinland-Pfalz`(ORG)

**Example 12** (doc_id: `54544`) (sent_id: `54544`)


Dagegen verhielten sich Eltern widersprüchlich , wollten sie einerseits von den Steuervorteilen einer ( unrichtigen ) Besteuerung von Entgeltbestandteilen als sonstige Bezüge profitieren , um diese dann andererseits im nachfolgenden Elterngeldverfahren mit dem Ziel höheren Elterngelds wieder infrage zu stellen ( zur Maßgeblichkeit in Anspruch genommener steuerlicher Vergünstigungen bei der Berechnung des Elterngelds aus selbstständiger Erwerbstätigkeit BSG Urteil vom 15. 12. 2015 - B 10 EG 6/14 R - SozR 4 - 7837 § 2 Nr 30 RdNr 19 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 15. 12. 2015 - B 10 EG 6/14 R - SozR 4 - 7837 § 2 Nr 30 RdNr 19`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Urteil vom 15. 12. 2015 - B 10 EG 6/14 R - SozR 4 - 7837 § 2 Nr 30 RdNr 19`(RS)

**Example 13** (doc_id: `54609`) (sent_id: `54609`)


Sie ist weder vom Kläger noch der Beigeladenen zu 1. - unter Hinweis auf eine Verletzung des § 7 SGB IV - im Revisionsverfahren mit Rechtsmitteln angegriffen worden ( vgl zur Teilbarkeit eines Statusfeststellungsbescheids insoweit schon BSG Urteil vom 24. 3. 2016 - B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 RdNr 18 , 11 ; BSG Urteil vom 24. 3. 2016 - B 12 R 12/14 R - SozR 4 - 2400 § 7a Nr 6 RdNr 11 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 24. 3. 2016 - B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 RdNr 18 , 11`
- `BSG Urteil` — similar text (different position): `BSG Urteil vom 24. 3. 2016 - B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 RdNr 18 , 11`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 7 SGB IV`(NRM)
- `BSG Urteil vom 24. 3. 2016 - B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 RdNr 18 , 11`(RS)
- `BSG Urteil vom 24. 3. 2016 - B 12 R 12/14 R - SozR 4 - 2400 § 7a Nr 6 RdNr 11`(RS)

**Example 14** (doc_id: `54785`) (sent_id: `54785`)


Im Fall des FG Berlin-Brandenburg hatte der dortige Kläger , ein heilkundlicher Verkehrstherapeut , mit den Klienten eine Therapievereinbarung getroffen , wonach u. a. der MPU-Erfolg nicht das Ziel der Therapie sei .

**False Positives:**

- `FG Berlin` — partial — pred is substring of gold: `FG Berlin-Brandenburg`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `FG Berlin-Brandenburg`(ORG)

**Example 15** (doc_id: `54873`) (sent_id: `54873`)


Im Gegenteil nimmt die arbeitsgerichtliche Rechtsprechung an , dass Lehrer an Musikschulen nur dann als Arbeitnehmer anzusehen sind , wenn die Parteien dies vereinbart haben oder im Einzelfall festzustellende Umstände hinzutreten , aus denen sich ergibt , dass der für das Bestehen eines Arbeitsverhältnisses erforderliche Grad der persönlichen Abhängigkeit gegeben ist ( vgl aktuell zu Musikschullehrern BAG Urteil vom 21. 11. 2017 - 9 AZR 117/17 - Juris ; BAG Urteil vom 17. 10. 2017 - 9 AZR 792/16 - Juris ; BAG Urteil vom 27. 6. 2017 - 9 AZR 851/16 - Juris ; BAG Urteil vom 27. 6. 2017 - 9 AZR 852/16 - Juris mwN ) .

**False Positives:**

- `BAG Urteil` — partial — pred is substring of gold: `BAG Urteil vom 21. 11. 2017 - 9 AZR 117/17 - Juris`
- `BAG Urteil` — similar text (different position): `BAG Urteil vom 21. 11. 2017 - 9 AZR 117/17 - Juris`
- `BAG Urteil` — similar text (different position): `BAG Urteil vom 21. 11. 2017 - 9 AZR 117/17 - Juris`
- `BAG Urteil` — similar text (different position): `BAG Urteil vom 21. 11. 2017 - 9 AZR 117/17 - Juris`

> overlaps gold: 4  |  likely missing annotation: 0

**Gold Entities:**

- `BAG Urteil vom 21. 11. 2017 - 9 AZR 117/17 - Juris`(RS)
- `BAG Urteil vom 17. 10. 2017 - 9 AZR 792/16 - Juris`(RS)
- `BAG Urteil vom 27. 6. 2017 - 9 AZR 851/16 - Juris`(RS)
- `BAG Urteil vom 27. 6. 2017 - 9 AZR 852/16 - Juris`(RS)

**Example 16** (doc_id: `55021`) (sent_id: `55021`)


Sittliche Gründe zur Übernahme der Beerdigungskosten kommen im Allgemeinen bei einem nahen Angehörigen in Betracht ( BFH-Urteil in BFHE 150 , 351 , BStBl II 1987 , 715 ; FG Münster , Urteil in EFG 2014 , 44 ; HHR / Kanzler , EStG § 33 Rz 142 ) .

**False Positives:**

- `FG Münster` — partial — pred is substring of gold: `FG Münster , Urteil in EFG 2014 , 44`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH-Urteil in BFHE 150 , 351 , BStBl II 1987 , 715`(RS)
- `FG Münster , Urteil in EFG 2014 , 44`(RS)
- `HHR / Kanzler , EStG § 33 Rz 142`(LIT)

**Example 17** (doc_id: `55400`) (sent_id: `55400`)


Die Aufgabe , bundeseinheitliche Vorgaben für die Honorarverteilung zu treffen , welche die regionalen Partner der Honorarverteilungsvereinbarungen zu beachten hatten , war dem BewA - zusätzlich zu seiner originären Kompetenz der Leistungsbewertung nach § 87 Abs 2 SGB V - übertragen ( BSG Urteil vom 19. 8. 2015 - B 6 KA 34/14 R - BSGE 119 , 231 = SozR 4 - 2500 § 87b Nr 7 , RdNr 25 mwN ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 19. 8. 2015 - B 6 KA 34/14 R - BSGE 119 , 231 = SozR 4 - 2500 § 87b Nr 7 , RdNr 25`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 87 Abs 2 SGB V`(NRM)
- `BSG Urteil vom 19. 8. 2015 - B 6 KA 34/14 R - BSGE 119 , 231 = SozR 4 - 2500 § 87b Nr 7 , RdNr 25`(RS)

**Example 18** (doc_id: `55428`) (sent_id: `55428`)


Zur Frage der entsprechenden Anwendbarkeit wettbewerbsrechtlicher Bestimmungen ( ua zum Schadensersatz ) auf die in § 69 Abs 1 S 1 SGB V geregelten Rechtsverhältnisse hat der Senat bereits entschieden , dass diese zur Kompensation einer unterlassenen oder im Ergebnis erfolglosen Inanspruchnahme gerichtlichen Primärrechtsschutzes , insbesondere von einstweiligem Rechtsschutz nach § 86b SGG , von vornherein nicht zur Verfügung stehen ( BSG Urteil vom 15. 3. 2017 - B 6 KA 35/16 R - zur Veröffentlichung für BSGE und SozR 4 - 5540 Anl 9.1 Nr 12 vorgesehen , RdNr 28 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 15. 3. 2017 - B 6 KA 35/16 R - zur Veröffentlichung für BSGE und SozR 4 - 5540 Anl 9.1 Nr 12 vorgesehen , RdNr 28`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 69 Abs 1 S 1 SGB V`(NRM)
- `§ 86b SGG`(NRM)
- `BSG Urteil vom 15. 3. 2017 - B 6 KA 35/16 R - zur Veröffentlichung für BSGE und SozR 4 - 5540 Anl 9.1 Nr 12 vorgesehen , RdNr 28`(RS)

**Example 19** (doc_id: `55528`) (sent_id: `55528`)


Zwar kann sich derjenige auf einen Anspruch auf rechtliches Gehör stützen , der nach der maßgeblichen Verfahrensordnung an einem gerichtlichen Verfahren als Partei oder in parteiähnlicher Stellung beteiligt oder unmittelbar rechtlich von dem Verfahren betroffen ist ( stRspr ; vgl etwa BVerfG Beschluss vom 14. 4. 1987 - 1 BvR 332/86 - BVerfGE 75 , 201 < 215 > - Juris RdNr 49 ) .

**False Positives:**

- `BVerfG Beschluss` — partial — pred is substring of gold: `BVerfG Beschluss vom 14. 4. 1987 - 1 BvR 332/86 - BVerfGE 75 , 201 < 215 > - Juris RdNr 49`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG Beschluss vom 14. 4. 1987 - 1 BvR 332/86 - BVerfGE 75 , 201 < 215 > - Juris RdNr 49`(RS)

**Example 20** (doc_id: `55560`) (sent_id: `55560`)


Mit der Beitragsforderung wurde durch die Beklagte zumindest in die allgemeine Handlungsfreiheit und damit in das Grundrecht des Klägers aus Art 2 Abs 1 GG eingegriffen , wodurch ein anhörungspflichtiger " Eingriff " iS des § 24 Abs 1 SGB X vorlag ( vgl BSG Urteil vom 25. 1. 1979 - 3 RK 35/77 - SozR 1200 § 34 Nr 7 ; Mutschler in Kasseler Komm , Stand September 2015 , § 24 SGB X RdNr 7 ; Siefert in von Wulffen / Schütze , SGB X , 8. Aufl 2014 , § 24 RdNr 8 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 25. 1. 1979 - 3 RK 35/77 - SozR 1200 § 34 Nr 7`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art 2 Abs 1 GG`(NRM)
- `§ 24 Abs 1 SGB X`(NRM)
- `BSG Urteil vom 25. 1. 1979 - 3 RK 35/77 - SozR 1200 § 34 Nr 7`(RS)
- `Mutschler in Kasseler Komm , Stand September 2015 , § 24 SGB X RdNr 7`(LIT)
- `Siefert in von Wulffen / Schütze , SGB X , 8. Aufl 2014 , § 24 RdNr 8`(LIT)

**Example 21** (doc_id: `55636`) (sent_id: `55636`)


Einstweilige Rechtsschutzanträge des Beklagten zur Fortführung des Vertrages blieben erfolglos ( SG München Beschluss vom 19. 1. 2011 - S 39 KA 1248/10 ER ; Bayerisches LSG Beschluss vom 22. 2. 2011 - L 12 KA 2/11 B ER - NZS 2011 , 386 ) .

**False Positives:**

- `SG München Beschluss` — partial — pred is substring of gold: `SG München Beschluss vom 19. 1. 2011 - S 39 KA 1248/10 ER`
- `LSG Beschluss` — partial — pred is substring of gold: `Bayerisches LSG Beschluss vom 22. 2. 2011 - L 12 KA 2/11 B ER - NZS 2011 , 386`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `SG München Beschluss vom 19. 1. 2011 - S 39 KA 1248/10 ER`(RS)
- `Bayerisches LSG Beschluss vom 22. 2. 2011 - L 12 KA 2/11 B ER - NZS 2011 , 386`(RS)

**Example 22** (doc_id: `55829`) (sent_id: `55829`)


Auch über eine - nach § 69 Abs 1 S 3 SGB V nicht vollständig ausgeschlossene - entsprechende Heranziehung von Vorschriften des BGB können Schadensersatzansprüche einer Krankenkasse gegenüber einem Hausärzteverband oder den an der HzV teilnehmenden Ärzten unter diesen Umständen nicht begründet werden ( zu Schadensersatzansprüchen zwischen Leistungserbringern vgl BSG Urteil vom 15. 3. 2017 - B 6 KA 35/16 R - zur Veröffentlichung für BSGE und SozR 4 - 5540 Anl 9.1 Nr 12 vorgesehen , RdNr 31 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 15. 3. 2017 - B 6 KA 35/16 R - zur Veröffentlichung für BSGE und SozR 4 - 5540 Anl 9.1 Nr 12 vorgesehen , RdNr 31`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 69 Abs 1 S 3 SGB V`(NRM)
- `BGB`(NRM)
- `BSG Urteil vom 15. 3. 2017 - B 6 KA 35/16 R - zur Veröffentlichung für BSGE und SozR 4 - 5540 Anl 9.1 Nr 12 vorgesehen , RdNr 31`(RS)

**Example 23** (doc_id: `56005`) (sent_id: `56005`)


Der Kläger wäre daher , um seine Zuständigkeit nach § 14 Abs 2 Satz 1 SGB IX zu vermeiden , berechtigt gewesen , vor der ersten anstehenden Verlängerung der ( konkludenten ) Leistungsbewilligung nach dem 1. 7. 2001 ( BSG Urteil vom 13. 7. 2017 - B 8 SO 1/16 R - RdNr 22 ) , spätestens aber mit der Prüfung und Entscheidung über den Rehabilitationsantrag vom 28. 2. 2005 seine Zuständigkeit für den Leistungsfall zu prüfen und den Leistungsfall vor einer anstehenden Leistungsbewilligung bzw den Antrag der K. auf Eingliederungshilfe in der Außenwohngruppe ggf an den nach seiner Auffassung originär zuständigen Beklagten weiterzuleiten ; ein Fall des § 103 SGB X liegt ebenso wenig vor wie eine zielgerichtete Zuständigkeitsanmaßung , die eine Erstattung nach § 104 SGB X ausschließen würde ( vgl dazu BSGE 98 , 267 = SozR 4 - 3250 § 14 Nr 4 ; BSG SozR 4 - 3100 § 18c Nr 2 RdNr 30 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 13. 7. 2017 - B 8 SO 1/16 R - RdNr 22 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 14 Abs 2 Satz 1 SGB IX`(NRM)
- `BSG Urteil vom 13. 7. 2017 - B 8 SO 1/16 R - RdNr 22 )`(RS)
- `K.`(PER)
- `§ 103 SGB X`(NRM)
- `§ 104 SGB X`(NRM)
- `BSGE 98 , 267 = SozR 4 - 3250 § 14 Nr 4`(RS)
- `BSG SozR 4 - 3100 § 18c Nr 2 RdNr 30`(RS)

**Example 24** (doc_id: `56125`) (sent_id: `56125`)


Diesen Anspruch hat das LSG Mecklenburg-Vorpommern mit Urteil vom 22. 2. 2017 verneint und für das PKH-Vergütungsfestsetzungsverfahren eine überlange Verfahrensdauer von zwei Monaten festgestellt .

**False Positives:**

- `LSG Mecklenburg` — partial — pred is substring of gold: `LSG Mecklenburg-Vorpommern`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `LSG Mecklenburg-Vorpommern`(ORG)

**Example 25** (doc_id: `56363`) (sent_id: `56363`)


Bei der Ordnung von Massenerscheinungen können typisierende und generalisierende Regelungen notwendig sein ( vgl BSG Urteil vom 23. 7. 2014 - B 12 KR 28/12 R - BSGE 116 , 241 = SozR 4 - 2500 § 229 Nr 18 , RdNr 23 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 23. 7. 2014 - B 12 KR 28/12 R - BSGE 116 , 241 = SozR 4 - 2500 § 229 Nr 18 , RdNr 23`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Urteil vom 23. 7. 2014 - B 12 KR 28/12 R - BSGE 116 , 241 = SozR 4 - 2500 § 229 Nr 18 , RdNr 23`(RS)

**Example 26** (doc_id: `56512`) (sent_id: `56512`)


Denn Darlehen , die gewährt werden , um nach Antragstellung bzw Kenntnis des Sozialhilfeträgers angefallene existenzielle Bedarfe zu decken , sind wegen der von Anfang an bestehenden Rückzahlungsverpflichtung eine nur vorübergehend zur Verfügung gestellte Leistung , die bei der Hilfe zum Lebensunterhalt nicht als Einkommen zu berücksichtigen ist ( BSGE 112 , 67 = SozR 4 - 3500 § 92 Nr 1 , RdNr 26 ; BSG Urteil vom 23. 8. 2013 - B 8 SO 24/11 R - juris , RdNr 25 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 23. 8. 2013 - B 8 SO 24/11 R - juris , RdNr 25`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSGE 112 , 67 = SozR 4 - 3500 § 92 Nr 1 , RdNr 26`(RS)
- `BSG Urteil vom 23. 8. 2013 - B 8 SO 24/11 R - juris , RdNr 25`(RS)

**Example 27** (doc_id: `56591`) (sent_id: `56591`)


Während ansonsten in kostenrechtlichen Verfahren der Erinnerung nach dem GKG bzw dem RVG nunmehr auch in dritter Instanz grundsätzlich eine Entscheidung durch den Einzelrichter vorgesehen ist ( vgl § 66 Abs 6 S 1 GKG bzw § 33 Abs 8 S 1 RVG - s dazu BGH Beschluss vom 23. 4. 2015 - I ZB 73/14 - NJW 2015 , 2194 ) , lässt das SGG bislang auch bei Erinnerungen ( §§ 178 , 189 Abs 2 S 2 SGG ) ein Tätigwerden des Einzelrichters lediglich im Rahmen des § 155 Abs 2 S 1 Nr 5 , Abs 3 und 4 SGG zu ( Reichel in Zeihe / Hauck <Hrsg> , SGG , Stand August 2017 , § 189 RdNr 10a ) .

**False Positives:**

- `BGH Beschluss` — partial — pred is substring of gold: `BGH Beschluss vom 23. 4. 2015 - I ZB 73/14 - NJW 2015 , 2194`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `GKG`(NRM)
- `RVG`(NRM)
- `§ 66 Abs 6 S 1 GKG`(NRM)
- `§ 33 Abs 8 S 1 RVG`(NRM)
- `BGH Beschluss vom 23. 4. 2015 - I ZB 73/14 - NJW 2015 , 2194`(RS)
- `SGG`(NRM)
- `§§ 178 , 189 Abs 2 S 2 SGG`(NRM)
- `§ 155 Abs 2 S 1 Nr 5 , Abs 3 und 4 SGG`(NRM)
- `Reichel in Zeihe / Hauck <Hrsg> , SGG , Stand August 2017 , § 189 RdNr 10a`(LIT)

**Example 28** (doc_id: `56636`) (sent_id: `56636`)


Zudem hätte es einer Darlegung der Beweisanforderungen bedurft ( vgl hierzu insgesamt BSG , aaO ; BSG Urteil vom 19. 3. 1986 - 9a RVi 2/84 - BSGE 60 , 58 = SozR 3850 § 51 Nr 9 ) , wie diese in der angefochtenen Entscheidung des LSG ( S 21 des Urteils ) bereits ausgeführt worden sind , um eine Sachaufklärungsrüge nach § 103 SGG im Rahmen einer grundsätzlichen Bedeutung als Rechtsfrage zu formulieren .

**False Positives:**

- `BSG Urteil` — similar text (different position): `BSG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG`(ORG)
- `BSG Urteil vom 19. 3. 1986 - 9a RVi 2/84 - BSGE 60 , 58 = SozR 3850 § 51 Nr 9`(RS)
- `§ 103 SGG`(NRM)

**Example 29** (doc_id: `56762`) (sent_id: `56762`)


Das hat der erkennende Senat für Arzneimittel - vom BVerfG bestätigt - entschieden und der Gesetzgeber ist dem ebenfalls gefolgt ( vgl zu § 2 Abs 1a SGB V GKV-VStG , BR-Drucks 456/11 S 74 ; BVerfG Beschluss vom 30. 6. 2008 - 1 BvR 1665/07 - SozR 4 - 2500 § 31 Nr 17 im Anschluss an BSG USK 2007 - 25 ; vgl zum Ganzen auch BSG SozR 4 - 2500 § 18 Nr 8 RdNr 20 f mwN ) .

**False Positives:**

- `BVerfG Beschluss` — similar text (different position): `BVerfG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG`(ORG)
- `§ 2 Abs 1a SGB V GKV-VStG`(NRM)
- `BR-Drucks 456/11 S 74`(LIT)
- `BVerfG Beschluss vom 30. 6. 2008 - 1 BvR 1665/07 - SozR 4 - 2500 § 31 Nr 17 im Anschluss an BSG USK 2007 - 25`(RS)
- `BSG SozR 4 - 2500 § 18 Nr 8 RdNr 20 f`(RS)

</details>

---

## `Specific German Organizations`

**F1:** 0.047 | **Precision:** 0.233 | **Recall:** 0.026  

**Format:** `regex`  
**Rule ID:** `21d25153`  
**Description:**
Matches specific German organizations and associations found in training data, including genitive forms, specific names, and abbreviations that were previously missed.

**Content:**
```
\b(?:S\u00e4chsische\s+LSG|Bundeswehr|NATO|Bundeszentrale\s+f\u00fcr\s+politische\s+Bildung|Bundeswehrflugplatz\s+B\u00fcchel|Bundesnachrichtendienst|BND|Arbeitgebervereinigung\s+energiewirtschaftlicher\s+Unternehmen\s+e\.\s*V\.|IG\s+Metall|Ausw\u00e4rtige\s+Amt|Amtsgericht\s+P\.|Bundesarbeitsgericht|Bundessozialgericht|Bundesverwaltungsgericht|Bundesfinanzhof|Bundespatentgericht|BVerwG|BGH|BSG|BAG|BVerfG|EuGH|EGMR|DPMA|BaFin|SG|FG|LAG|LSG|BA|H\u00c4VG|PKK|KCK|HPG|EED|JOOP|KAGURA|Schott|Urban\s+&\s+Schwarzenberg|National\s+Union\s+of\s+Rail|Maritime\s+and\s+Transport\s+Workers|Europ\u00e4ische\s+Gerichtshof\s+f\u00fcr\s+Menschenrechte|Deutscher\s+Apotheker\s+Verlag|B-KG|VEB\s*Elektronik\s*G\.|EZB|Bundestag|Bundesrat|Bundesregierung|Bundesverwaltungsgericht|Bundesfinanzhof|Bundespatentgericht|BVerwG|BGH|BSG|BAG|BVerfG|EuGH|EGMR|DPMA|BaFin|SG|FG|LAG|LSG|BA|H\u00c4VG|PKK|KCK|HPG|EED|JOOP|KAGURA|Schott|Urban\s+&\s+Schwarzenberg|National\s+Union\s+of\s+Rail|Maritime\s+and\s+Transport\s+Workers|Europ\u00e4ische\s+Gerichtshof\s+f\u00fcr\s+Menschenrechte|Deutscher\s+Apotheker\s+Verlag|B-KG|VEB\s*Elektronik\s*G\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.233 | 0.026 | 0.047 | 90 | 21 | 69 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 21 | 69 | 766 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53773`) (sent_id: `53773`)


Vielmehr hat das LSG nach erfolgter Zurückverweisung durch das BSG nach umfangreicher weiterer Sachaufklärung den Anspruch des Klägers erneut geprüft und eine Entscheidung gefällt .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 1** (doc_id: `54446`) (sent_id: `54446`)


b3 ) Soweit der BGH in seinem Beschluss vom 1. April 1965 ( Ia ZB 20/64 – „ Patentanwaltskosten “ , GRUR 1965 , 621 ) Doppelvertretungskosten im Gebrauchsmuster-Löschungsverfahren als regelmäßig nicht berücksichtigungsfähig erachtet hat , geht der Senat davon aus , dass diese Entscheidung überholt ist .

| Predicted | Gold |
|---|---|
| `BGH` | `BGH` |

**Missed by this rule (FN):**

- `Beschluss vom 1. April 1965 ( Ia ZB 20/64 – „ Patentanwaltskosten “ , GRUR 1965 , 621 )` (RS)

**Example 2** (doc_id: `54482`) (sent_id: `54482`)


e ) Eine andere Wertung folgt schließlich nicht aus der Rechtsprechung des BSG zur Rechtslage vor dem 1. 1. 2011 , nach der höhere tatsächliche Kosten bei zentraler Warmwassererzeugung nur dann anstelle des schätzungsweise ermittelten pauschalen Anteils der Warmwassererzeugungskosten in der Regelleistung von den Aufwendungen für Heizung in Abzug zu bringen waren ( vgl oben 4. a ) , wenn diese Kosten über die Einrichtung getrennter Zähler oder sonstiger Vorrichtungen konkret zu erfassen waren ( vgl nur BSG vom 27. 2. 2008 - B 14/11 b AS 15/07 R - BSGE 100 , 94 = SozR 4 - 4200 § 22 Nr 5 , RdNr 27 ) .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Missed by this rule (FN):**

- `BSG vom 27. 2. 2008 - B 14/11 b AS 15/07 R - BSGE 100 , 94 = SozR 4 - 4200 § 22 Nr 5 , RdNr 27` (RS)

**Example 3** (doc_id: `54707`) (sent_id: `54707`)


Im Übrigen fehlt es an Darlegungen dazu , an welcher Stelle seiner " Entscheidung " der BGH die vom Beklagten behauptete Aussage überhaupt getroffen haben soll .

| Predicted | Gold |
|---|---|
| `BGH` | `BGH` |

**Example 4** (doc_id: `55127`) (sent_id: `55127`)


Die einzige Auseinandersetzung des BGH mit landesrechtlichen Bestattungspflichten erfolgt in RdNr 12 seines Hinweisbeschlusses ; eine Aussage dergestalt , wie sie vom Beklagten in der Frage 1 formuliert worden ist , hat der BGH an dieser Stelle nicht getroffen .

| Predicted | Gold |
|---|---|
| `BGH` | `BGH` |

**Example 5** (doc_id: `55426`) (sent_id: `55426`)


Das in diesem Zusammenhang genannte Schreiben vom 27. März 2008 habe das Bundesarbeitsgericht bei seiner damaligen Entscheidung bereits berücksichtigt .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgericht` | `Bundesarbeitsgericht` |

**Example 6** (doc_id: `55822`) (sent_id: `55822`)


Auf die Beschwerde der Klägerin hat das BSG die Sache zur erneuten Verhandlung und Entscheidung an das LSG zurückverwiesen ( Beschluss vom 12. 8. 2013 ) .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 7** (doc_id: `56680`) (sent_id: `56680`)


Zu der ähnlichen Regelung in § 18 Abs. 5 des Tarifvertrags über das Sozialkassenverfahren im Baugewerbe vom 20. Dezember 1999 , wonach Erstattungsforderungen des Arbeitgebers gegen die Urlaubs- und Lohnausgleichskasse mit der Maßgabe zweckgebunden sind , dass der Arbeitgeber über sie nur verfügen kann , wenn das bei der Einzugsstelle bestehende Beitragskonto keinen Debetsaldo ausweist und er seinen Meldepflichten entsprochen hat , hat das Bundesarbeitsgericht entschieden , die Erfüllung der Beitragspflicht sei keine Voraussetzung für das Entstehen des Erstattungsanspruchs des Arbeitgebers ; § 18 Abs. 5 des Tarifvertrags begründe aber bei nicht vollständiger Erfüllung der Beitragspflicht ein Hindernis für die Durchsetzung des bereits mit der Auszahlung der Urlaubsvergütung entstandenen Anspruchs ( BAG , Urteil vom 14. Dezember 2011 - 10 AZR 517/10 , AP Nr. 338 zu TVG § 1 Tarifverträge : Bau , Rn. 27 mwN ) .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgericht` | `Bundesarbeitsgericht` |

**Missed by this rule (FN):**

- `§ 18 Abs. 5 des Tarifvertrags über das Sozialkassenverfahren im Baugewerbe` (REG)
- `§ 18 Abs. 5 des Tarifvertrags` (REG)
- `BAG , Urteil vom 14. Dezember 2011 - 10 AZR 517/10 , AP Nr. 338 zu TVG § 1 Tarifverträge : Bau , Rn. 27` (RS)

**Example 8** (doc_id: `57124`) (sent_id: `57124`)


1. Der Deutsche Gewerkschaftsbund und ver.di meinen , die einschränkende Auslegung des § 14 Abs. 2 Satz 2 TzBfG durch das Bundesarbeitsgericht überschreite die Grenze zulässiger Rechtsfortbildung .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgericht` | `Bundesarbeitsgericht` |

**Missed by this rule (FN):**

- `Deutsche Gewerkschaftsbund` (ORG)
- `ver.di` (ORG)
- `§ 14 Abs. 2 Satz 2 TzBfG` (NRM)

**Example 9** (doc_id: `57221`) (sent_id: `57221`)


Ausdrücklich von einer Stellungnahme abgesehen haben die Bundesregierung , der Bundesrat , das Ministerium für Migration , Justiz und Verbraucherschutz des Freistaats Thüringen , das Justizministerium Mecklenburg-Vorpommern sowie das Bundesarbeitsgericht .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgericht` | `Bundesarbeitsgericht` |

**Missed by this rule (FN):**

- `Bundesregierung` (ORG)
- `Bundesrat` (ORG)
- `Ministerium für Migration , Justiz und Verbraucherschutz des Freistaats Thüringen` (ORG)
- `Justizministerium Mecklenburg-Vorpommern` (ORG)

**Example 10** (doc_id: `57335`) (sent_id: `57335`)


Gemäß § 160a Abs 5 SGG kann das BSG in dem Beschluss über die Nichtzulassungsbeschwerde das angefochtene Urteil aufheben und die Sache zur erneuten Verhandlung und Entscheidung an das LSG zurückverweisen , wenn die Voraussetzungen des § 160 Abs 2 Nr 3 SGG vorliegen .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Missed by this rule (FN):**

- `§ 160a Abs 5 SGG` (NRM)
- `§ 160 Abs 2 Nr 3 SGG` (NRM)

**Example 11** (doc_id: `57679`) (sent_id: `57679`)


IV. Zu Vorlage und Verfassungsbeschwerde haben der Deutsche Gewerkschaftsbund ( DGB ) , die Vereinte Dienstleistungsgewerkschaft ( ver.di ) , die Bundesvereinigung der Deutschen Arbeitgeberverbände e. V. ( BDA ) , das Bundesarbeitsgericht und die Beklagten der Ausgangsverfahren Stellung genommen .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgericht` | `Bundesarbeitsgericht` |

**Missed by this rule (FN):**

- `Deutsche Gewerkschaftsbund` (ORG)
- `DGB` (ORG)
- `Vereinte Dienstleistungsgewerkschaft` (ORG)
- `ver.di` (ORG)
- `Bundesvereinigung der Deutschen Arbeitgeberverbände e. V.` (ORG)
- `BDA` (ORG)

**Example 12** (doc_id: `57730`) (sent_id: `57730`)


Das Bundesarbeitsgericht orientiert sich bei der Auslegung von § 14 Abs. 2 TzBfG zwar maßgebend am Grundrecht der Berufsfreiheit in Art. 12 Abs. 1 GG .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgericht` | `Bundesarbeitsgericht` |

**Missed by this rule (FN):**

- `§ 14 Abs. 2 TzBfG` (NRM)
- `Art. 12 Abs. 1 GG` (NRM)

**Example 13** (doc_id: `57771`) (sent_id: `57771`)


2. Das Bundesarbeitsgericht hatte § 14 Abs. 2 Satz 2 TzBfG zunächst dahin ausgelegt , dass dieselben Arbeitsvertragsparteien nur bei der erstmaligen Einstellung eine sachgrundlose Befristung vereinbaren können .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgericht` | `Bundesarbeitsgericht` |

**Missed by this rule (FN):**

- `§ 14 Abs. 2 Satz 2 TzBfG` (NRM)

**Example 14** (doc_id: `57921`) (sent_id: `57921`)


c ) Eine Zweitbeurteilung war gemäß der Bestimmungen Ziff. 7.5 i. V. m. Nr. 7. 4. Halbs. 2 BB BND nicht zu erstellen .

| Predicted | Gold |
|---|---|
| `BND` | `BND` |

**Example 15** (doc_id: `58153`) (sent_id: `58153`)


Auf die Beschwerde des Klägers hat das BSG mit Beschluss vom 14. 11. 2013 ( B 9 V 33/13 B ) das Urteil des LSG aufgehoben und die Sache zurückverwiesen , weil das LSG das Paul-Ehrlich-Institut ( Prof. Dr. C. ) nicht zu Einwänden des Klägers ergänzend befragt hat .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Missed by this rule (FN):**

- `Beschluss vom 14. 11. 2013 ( B 9 V 33/13 B )` (RS)
- `Paul-Ehrlich-Institut` (ORG)
- `C.` (PER)

**Example 16** (doc_id: `58209`) (sent_id: `58209`)


Das Bundesarbeitsgericht hat das Urteil des Landesarbeitsgerichts aufgehoben und die Sache zur neuen Verhandlung und Entscheidung an das Berufungsgericht zurückverwiesen .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgericht` | `Bundesarbeitsgericht` |

**Example 17** (doc_id: `58450`) (sent_id: `58450`)


In einer ersten Regelbeurteilung vom 23. April 2013 zum Stichtag 1. April 2013 vergab der seinerzeitige Leiter der Abteilung X des BND ( Herr Dr. A. ) das Gesamturteil 7. Auf Einwendungen des Klägers hob der BND diese dienstliche Beurteilung wegen formeller Fehler auf .

| Predicted | Gold |
|---|---|
| `BND` | `BND` |

**Missed by this rule (FN):**

- `Abteilung X des BND` (ORG)
- `A.` (PER)

**Example 18** (doc_id: `58501`) (sent_id: `58501`)


Das Oberverwaltungsgericht hat angenommen , dass eine Prüfung des § 4 Abs. 3 und 4 PassG auf die Vereinbarkeit mit dem Grundgesetz wegen der unionsrechtlichen Determinierung nicht stattfinden kann und die Vereinbarkeit der zwingenden unionsrechtlichen Vorgaben des Art. 1 Abs. 1 und 2 der Verordnung ( EG ) 2252/2004 mit der Charta der Grundrechte der EU und anderem höherrangigen Unionsrecht durch die Rechtsprechung des EuGH mit bindender Wirkung geklärt ist .

| Predicted | Gold |
|---|---|
| `EuGH` | `EuGH` |

**Missed by this rule (FN):**

- `§ 4 Abs. 3 und 4 PassG` (NRM)
- `Grundgesetz` (NRM)
- `Art. 1 Abs. 1 und 2 der Verordnung ( EG ) 2252/2004` (NRM)
- `Charta der Grundrechte der EU` (NRM)

**Example 19** (doc_id: `58892`) (sent_id: `58892`)


Soweit nach der Rspr des BGH der entschädigungspflichtige Erwerbsschaden im zivilen Schadensersatzrecht ( §§ 842 , 843 , 252 BGB , § 11 StVG ) auch den Arbeitgeberanteil am Gesamtsozialversicherungsbeitrag umfasst ( BGHZ 173 , 169 , 174 ; BGHZ 139 , 167 , 172 ; BGHZ 43 , 378 , 382 f ) , beruht dies auf Besonderheiten des normativen Schadensbegriffs ( vgl BGHZ 173 , 169 , 174 ; BGHZ 43 , 378 , 382 ff ) und hat für die Beurteilung des Vergütungsbegriffs in § 35a Abs 6a SGB IV keine Bedeutung .

| Predicted | Gold |
|---|---|
| `BGH` | `BGH` |

**Missed by this rule (FN):**

- `§§ 842 , 843 , 252 BGB` (NRM)
- `§ 11 StVG` (NRM)
- `BGHZ 173 , 169 , 174` (RS)
- `BGHZ 139 , 167 , 172` (RS)
- `BGHZ 43 , 378 , 382 f` (RS)
- `BGHZ 173 , 169 , 174` (RS)
- `BGHZ 43 , 378 , 382 ff` (RS)
- `§ 35a Abs 6a SGB IV` (NRM)

**Example 20** (doc_id: `59848`) (sent_id: `59848`)


Mit Wirkung vom 1. März 2011 wurde der Kläger wiederum zum BND versetzt .

| Predicted | Gold |
|---|---|
| `BND` | `BND` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53527`) (sent_id: `53527`)


Sollte es hierauf ankommen , wird das FG die erforderlichen tatsächlichen Feststellungen zu treffen haben .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `53588`) (sent_id: `53588`)


Diesbezüglich fehlt es insbesondere an Vorbringen dazu , dass die Entscheidung des LSG auf " diesem Mangel " beruhen kann , dh es hätte Darlegungen zur mangelnden oder zumindest eingeschränkten Verwertbarkeit des Gutachtens bedurft .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `53715`) (sent_id: `53715`)


aa ) Obwohl die Parteien in Klausel 33 des Vertriebsvertrags die Geltung des Rechts der USA und des Staates Kalifornien vereinbart haben , ist das FG den deutschen Grundsätzen über die Auslegung von Willenserklärungen und Verträgen gefolgt .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `USA`(LOC)
- `Kalifornien`(LOC)

**Example 3** (doc_id: `53763`) (sent_id: `53763`)


Wird als Verfahrensmangel gerügt , das FG habe einen gestellten Beweisantrag übergangen ( Rüge mangelnder Sachaufklärung gemäß § 76 Abs. 1 Satz 1 FGO ) , so ist darzulegen , welche Tatfrage aufklärungsbedürftig ist , welche Beweismittel das FG zu welchem Beweisthema nicht erhoben hat , die genauen Fundstellen ( Schriftsatz mit Datum und Seitenzahl , Terminprotokoll ) , in denen die Beweismittel und Beweisthemen angeführt worden sind , das voraussichtliche Ergebnis der Beweisaufnahme , inwiefern das Urteil des FG aufgrund dessen sachlich-rechtlicher Auffassung auf der unterbliebenen Beweisaufnahme beruhen kann und dass die Nichterhebung der Beweise vor dem FG rechtzeitig gerügt worden ist oder aufgrund des Verhaltens des FG nicht mehr vor diesem gerügt werden konnte ( ständige BFH-Rechtsprechung , z.B. Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85 , und vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125 , m. w. N. ) .

**False Positives:**

- `FG` — similar text (different position): `§ 76 Abs. 1 Satz 1 FGO`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 76 Abs. 1 Satz 1 FGO`(NRM)
- `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`(RS)
- `vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125`(RS)

**Example 4** (doc_id: `53773`) (sent_id: `53773`)


Vielmehr hat das LSG nach erfolgter Zurückverweisung durch das BSG nach umfangreicher weiterer Sachaufklärung den Anspruch des Klägers erneut geprüft und eine Entscheidung gefällt .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG`(ORG)

**Example 5** (doc_id: `53867`) (sent_id: `53867`)


Zwar kann das LSG von einer Entscheidung ua des BSG auch dann abweichen , wenn es einen der höchstrichterlichen Rechtsprechung widersprechenden Rechtssatz nur sinngemäß und in scheinbar fallbezogene Ausführungen gekleidet entwickelt .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG`(ORG)

**Example 6** (doc_id: `53890`) (sent_id: `53890`)


Im Berufungsverfahren hat der zuständige Berichterstatter des LSG den Sachverständigen Prof. Dr. T. im Termin zur mündlichen Verhandlung ergänzend angehört .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `T.`(PER)

**Example 7** (doc_id: `53927`) (sent_id: `53927`)


Die Klägerin beantragt sinngemäß , das angefochtene Urteil und die Einspruchsentscheidung vom 19. Mai 2014 aufzuheben und den Einkommensteuerbescheid für 2011 vom 17. September 2013 dahingehend zu ändern , dass die Einkommensteuer auf 33.553 € festgesetzt wird , hilfsweise , das angefochtene Urteil aufzuheben und die Sache zur anderweitigen Verhandlung und Entscheidung an das FG zurückzuverweisen .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `53954`) (sent_id: `53954`)


Davon kann schließlich ungeachtet der fehlenden statistischen Erhebungen im Allgemeinen auch im Fall hier nicht hinreichend sicher ausgegangen werden , nachdem zwar der Energieverbrauch des Klägers im streitbefangenen Zeitraum nach Einschätzung des LSG für einen Haushalt mit dezentraler Warmwassererzeugung als durchschnittlich anzusehen ist , die Ausgaben für Haushaltsstrom von 50,27 Euro bzw 44,58 Euro monatlich ( 603,18 Euro bzw 534,93 Euro ÷ 12 ) mit den darauf entfallenden Leistungen zur Sicherung des Lebensunterhalts in Höhe von 36,29 Euro bzw 37,66 Euro monatlich ( 2011 : 28,29 Euro Regelbedarfsanteil Strom + 8 Euro Mehrbedarfspauschale ; 2012 : 29,06 Euro Regelbedarfsanteil Strom + 8,60 Euro Mehrbedarfspauschale ) jedoch nicht vollständig zu bestreiten waren .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 9** (doc_id: `54108`) (sent_id: `54108`)


Mit Beschluss vom 23. September 2015 hat die Patentabteilung des DPMA den Antrag zurückgewiesen .

**False Positives:**

- `DPMA` — partial — pred is substring of gold: `Patentabteilung des DPMA`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Patentabteilung des DPMA`(ORG)

**Example 10** (doc_id: `54175`) (sent_id: `54175`)


Zutreffend führt das LSG in diesem Zusammenhang zudem aus , dass es nicht darauf ankommt , ob Arbeitseinsätze im Rahmen eines Dauerarbeitsverhältnisses von vorneherein feststehen oder von Mal zu Mal vereinbart werden .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 11** (doc_id: `54267`) (sent_id: `54267`)


bb ) Ebenso zutreffend hat das FG auf den Normzweck des § 7 g Abs. 3 EStG hingewiesen .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 7 g Abs. 3 EStG`(NRM)

**Example 12** (doc_id: `54360`) (sent_id: `54360`)


In dem hier streitgegenständlichen Wiederaufnahmeverfahren 2 K 154/15 vernahm das FG in der mündlichen Verhandlung vom 19. Mai 2017 den Prozessbevollmächtigten , dessen früheren Mitarbeiter sowie den Sachbearbeiter des FA als Zeugen zu den Inhalten und Umständen des Gesprächs vom 26. Mai 2015 und wies die Klage ab .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Wiederaufnahmeverfahren 2 K 154/15`(RS)

**Example 13** (doc_id: `54448`) (sent_id: `54448`)


Welchen Einfluss die aufrechterhaltene Stationierung von Atomwaffen in Büchel für das Verhalten von Terroristen ( und im Konflikt mit NATO-Staaten stehenden Drittstaaten ) habe , entziehe sich einer gerichtlichen Feststellung .

**False Positives:**

- `NATO` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Büchel`(LOC)

**Example 14** (doc_id: `54607`) (sent_id: `54607`)


aa ) Zu Recht ist das FG der Auffassung des Klägers nicht gefolgt , der Wortlaut des § 7 g Abs. 3 EStG stehe im Streitfall einer Rückgängigmachung des Abzugs entgegen .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 7 g Abs. 3 EStG`(NRM)

**Example 15** (doc_id: `54708`) (sent_id: `54708`)


Der weiteren Aufklärung habe für das SG die Weigerung der Klägerin zur Abgabe entsprechender Schweigepflichtentbindungserklärungen entgegengestanden .

**False Positives:**

- `SG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 16** (doc_id: `55088`) (sent_id: `55088`)


Zwar hat der Senat durchaus zur Kenntnis genommen , dass die Klägerin in dem von ihr benannten Schriftsatz vom 11. 2. 2016 - auf die Anhörung des LSG zur Entscheidung durch Beschluss - durch die Bezugnahme auf den Schriftsatz vom 7. 3. 2016 ihren Antrag bei Prof. Dr. T. nachzufragen , in welcher Form und in welchem Umfang er an der Erstellung des Sachverständigengutachtens beteiligt gewesen sei , wiederholt hat .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `T.`(PER)

**Example 17** (doc_id: `55103`) (sent_id: `55103`)


Festgestellt hat das LSG insoweit nur , dass K. bis zu ihrer Aufnahme in das Diakonissenhaus An. bei ihrer Mutter in B. , im heutigen Kreisgebiet des Beklagten , " gemeldet " war ; auf die einwohnerrechtliche Meldung kommt es jedoch für die Bestimmung des gewöhnlichen Aufenthalts nicht an ( BSG SozR 5870 § 1 Nr 4 ; SozR 3 - 5870 § 2 Nr 36 ) .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `K.`(PER)
- `Diakonissenhaus An.`(ORG)
- `B.`(LOC)
- `BSG SozR 5870 § 1 Nr 4`(RS)
- `SozR 3 - 5870 § 2 Nr 36`(RS)

**Example 18** (doc_id: `55113`) (sent_id: `55113`)


Der Kläger subsumiert die Sachverhaltskonstellation seines Falls vielmehr selbst anhand der vorgenannten Rechtsprechung des BSG und zieht daraus im Kern den Schluss , dass das LSG auf der Basis dieser höchstrichterlichen Rechtsprechung seinen Fall " falsch entschieden " habe .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG`(ORG)

**Example 19** (doc_id: `55217`) (sent_id: `55217`)


Mit Fax vom 16. 5. 2017 hat das LSG der Klägerin mitgeteilt , dass der Termin vom 17. 5. 2017 nicht aufgehoben werde .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 20** (doc_id: `55312`) (sent_id: `55312`)


Gemäß § 118 Abs. 2 FGO hat das FG für den Senat auch bindend festgestellt , dass keine willkürliche Schätzung bewusst zu Lasten des Klägers erfolgte und keine Umstände vorlagen , die die Schätzung als Verstoß gegen eine ordnungsgemäße Verwaltung erscheinen lassen .

**False Positives:**

- `FG` — similar text (different position): `§ 118 Abs. 2 FGO`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 118 Abs. 2 FGO`(NRM)

**Example 21** (doc_id: `55328`) (sent_id: `55328`)


Sie lässt unberücksichtigt , dass das LSG in seinen entscheidungstragend herangezogenen Obersätzen der BSG-Rechtsprechung nicht " im Grundsätzlichen " ausdrücklich widersprochen und mit dieser Rechtsprechung nicht zu vereinbarende eigene Rechtssätze aufgestellt hat .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 22** (doc_id: `55403`) (sent_id: `55403`)


Auf die Revision des Klägers hob der erkennende Senat mit Urteil in BFHE 248 , 462 , BStBl II 2015 , 730 das Urteil des FG auf und verwies die Sache an das FG zurück .

**False Positives:**

- `FG` — no gold match — likely missing annotation
- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `Urteil in BFHE 248 , 462 , BStBl II 2015 , 730`(RS)

**Example 23** (doc_id: `55668`) (sent_id: `55668`)


Die zulässige Revision des Klägers ist im Sinne der Aufhebung des SG-Urteils und der Zurückverweisung der Sache an das SG zur anderweitigen Verhandlung und Entscheidung begründet ( § 170 Abs 2 Satz 2 Sozialgerichtsgesetz < SGG > ) .

**False Positives:**

- `SG` — similar text (different position): `§ 170 Abs 2 Satz 2 Sozialgerichtsgesetz`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 170 Abs 2 Satz 2 Sozialgerichtsgesetz`(NRM)
- `SGG`(NRM)

**Example 24** (doc_id: `55722`) (sent_id: `55722`)


Vielmehr hat sich das LSG die Feststellungen aus dem Strafurteil nach eigener Prüfung zu eigen gemacht und dabei neben den Feststellungen aus dem Strafurteil nicht nur den Inhalt des Erstattungsbescheides aus dem Jahr 2013 zur Honorarrückforderung in Höhe von 216 492,33 Euro berücksichtigt , sondern auch den Inhalt der Strafakten ausgewertet , aus denen hervorgeht , dass sich das Urteil des Amtsgerichts Fürth nicht allein auf das Geständnis des Klägers stützt , sondern auch auf das Ergebnis einer umfangreichen Beweisaufnahme .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Amtsgerichts Fürth`(ORG)

**Example 25** (doc_id: `55822`) (sent_id: `55822`)


Auf die Beschwerde der Klägerin hat das BSG die Sache zur erneuten Verhandlung und Entscheidung an das LSG zurückverwiesen ( Beschluss vom 12. 8. 2013 ) .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG`(ORG)

**Example 26** (doc_id: `55919`) (sent_id: `55919`)


3. Die vorgenannten Voraussetzungen sind im Streitfall nicht erfüllt , weil nach den Feststellungen des FG die gesamte Dauer des Straßentransports 30 1/2 bis 31 Stunden betrug und damit die grundsätzlich vorgeschriebene Höchstdauer von 29 Stunden überschritten wurde .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 27** (doc_id: `55987`) (sent_id: `55987`)


Soweit in den Streitjahren die von A erbrachten Reisevorleistungen unbelastet von der Steuer bleiben , weil sie in Österreich entgegen den unionsrechtlichen Bestimmungen der Art. 306 ff. MwStSystRL nicht der Margenbesteuerung unterworfen werden , ist dies - wie das FG in Bezug auf Steuerausfälle zu Recht erkannt hat - notwendige Folge des Gebots , das Unionsrecht in jedem Falle gegenüber dem entgegenstehenden nationalen Recht durchzusetzen .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `A`(PER)
- `Österreich`(LOC)
- `Art. 306 ff. MwStSystRL`(NRM)

**Example 28** (doc_id: `56139`) (sent_id: `56139`)


B. Das LSG hat im Ergebnis zu Recht das Urteil des SG aufgehoben .

**False Positives:**

- `SG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 29** (doc_id: `56146`) (sent_id: `56146`)


Aufgrund dessen ist das angefochtene Urteil gemäß § 160a Abs 5 iVm § 160 Abs 2 Nr 3 SGG aufzuheben und die Sache an das LSG zurückzuverweisen .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 160a Abs 5 iVm § 160 Abs 2 Nr 3 SGG`(NRM)

</details>

---

## `Specific Court Genitives with Location (Fixed)`

**F1:** 0.040 | **Precision:** 0.395 | **Recall:** 0.021  

**Format:** `regex`  
**Rule ID:** `0ff0f4df`  
**Description:**
Matches court names in genitive form followed by location, ensuring the court type is present.

**Content:**
```
\b(?:Amtsgerichts|Landgerichts|Verwaltungsgerichts|Oberlandesgerichts|Landesarbeitsgerichts|Landessozialgerichts|Sozialgerichts|Finanzgerichts|Arbeitsgerichts|Bundesverwaltungsgerichts|Bundesfinanzhofs|Bundesgerichtshofs|Bundessozialgerichts|Bundesarbeitsgerichts|Bundespatentgerichts|Oberverwaltungsgerichts|Verwaltungsgerichtshofs|Hamburgischen Oberverwaltungsgerichts|Schleswig-Holsteinische Verwaltungsgericht|Schleswig-Holsteinische Oberlandesgericht|Bayerischen Landeszentrale|Bayerischen Landessozialgerichts|Truppendienstgerichts|Anwaltsgerichtshofs)\s+(?:[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)*|[A-Z]\.|[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+-[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+|<\s*[A-Z]{2,3}\s*>\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)*|Frankfurt\s+am\s+Main|Berlin-Brandenburg|Niedersachsen-Bremen|Baden-W\u00fcrttemberg|Nordrhein-Westfalen|Rheinland-Pfalz|Schleswig-Holstein)\b(?!\s+(?:Senat|Nr\.|\.)|\s+(?:Senat|Nr\.))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.395 | 0.021 | 0.040 | 43 | 17 | 26 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 17 | 26 | 746 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53725`) (sent_id: `53725`)


4. Auf die sofortige Beschwerde der Staatsanwaltschaft Düsseldorf hob das Oberlandesgericht Hamm mit angegriffenem Beschluss vom 28. Juli 2015 den Beschluss des Landgerichts Paderborn auf und ordnete die weitere Vollstreckung der Unterbringung des Beschwerdeführers in der Sicherungsverwahrung an .

| Predicted | Gold |
|---|---|
| `Landgerichts Paderborn` | `Landgerichts Paderborn` |

**Missed by this rule (FN):**

- `Staatsanwaltschaft Düsseldorf` (ORG)
- `Oberlandesgericht Hamm` (ORG)

**Example 1** (doc_id: `53793`) (sent_id: `53793`)


Die Revision des Angeklagten gegen das Urteil des Landgerichts Landshut vom 28. Juni 2017 wird verworfen .

| Predicted | Gold |
|---|---|
| `Landgerichts Landshut` | `Landgerichts Landshut` |

**Example 2** (doc_id: `53949`) (sent_id: `53949`)


1. Auf die Revision des Angeklagten gegen das Urteil des Landgerichts Halle vom 4. Mai 2017 wird das vorbenannte Urteil

| Predicted | Gold |
|---|---|
| `Landgerichts Halle` | `Landgerichts Halle` |

**Example 3** (doc_id: `54930`) (sent_id: `54930`)


3. Mit Schriftsatz seines Bevollmächtigten vom 30. Oktober 2014 legte der Beschwerdeführer " gegen den Durchsuchungsbeschluss des Amtsgerichts München vom 2. Mai 2014 sowie die bereits erfolgte Beschlagnahme der Geschäftsunterlagen " Beschwerde ein mit dem Antrag , den genannten Beschluss aufzuheben .

| Predicted | Gold |
|---|---|
| `Amtsgerichts München` | `Amtsgerichts München` |

**Example 4** (doc_id: `55722`) (sent_id: `55722`)


Vielmehr hat sich das LSG die Feststellungen aus dem Strafurteil nach eigener Prüfung zu eigen gemacht und dabei neben den Feststellungen aus dem Strafurteil nicht nur den Inhalt des Erstattungsbescheides aus dem Jahr 2013 zur Honorarrückforderung in Höhe von 216 492,33 Euro berücksichtigt , sondern auch den Inhalt der Strafakten ausgewertet , aus denen hervorgeht , dass sich das Urteil des Amtsgerichts Fürth nicht allein auf das Geständnis des Klägers stützt , sondern auch auf das Ergebnis einer umfangreichen Beweisaufnahme .

| Predicted | Gold |
|---|---|
| `Amtsgerichts Fürth` | `Amtsgerichts Fürth` |

**Example 5** (doc_id: `56530`) (sent_id: `56530`)


Die Beklagte beantragt , die Urteile des Bayerischen Landessozialgerichts vom 16. 12. 2015 und des Sozialgerichts München vom 16. 5. 2014 aufzuheben und die Klage abzuweisen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts München` | `Sozialgerichts München` |

**Missed by this rule (FN):**

- `Bayerischen Landessozialgerichts` (ORG)

**Example 6** (doc_id: `56596`) (sent_id: `56596`)


Hierfür wären die Urteile des Landgerichts Cottbus vom ... und des Brandenburgischen Oberlandesgerichts vom ... zu beachten , wonach der Kläger durch Kündigung vom ... 2005 wirksam aus der A-GbR ausgeschlossen wurde .

| Predicted | Gold |
|---|---|
| `Landgerichts Cottbus` | `Landgerichts Cottbus` |

**Missed by this rule (FN):**

- `Brandenburgischen Oberlandesgerichts` (ORG)
- `A-GbR` (ORG)

**Example 7** (doc_id: `56616`) (sent_id: `56616`)


Die Beklagte beantragt , das Urteil des Bayerischen Landessozialgerichts vom 21. Oktober 2015 aufzuheben und das Urteil des Sozialgerichts Nürnberg vom 25. Juni 2013 bezüglich der Beigeladenen zu 1. und 3. aufzuheben und die Klage abzuweisen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts Nürnberg` | `Sozialgerichts Nürnberg` |

**Missed by this rule (FN):**

- `Bayerischen Landessozialgerichts` (ORG)

**Example 8** (doc_id: `56951`) (sent_id: `56951`)


Auf die Revision des Angeklagten wird das Urteil des Landgerichts Ravensburg vom 1. August 2017 , soweit es ihn betrifft , mit den Feststellungen aufgehoben .

| Predicted | Gold |
|---|---|
| `Landgerichts Ravensburg` | `Landgerichts Ravensburg` |

**Example 9** (doc_id: `57130`) (sent_id: `57130`)


1. Auf die Revision des Angeklagten gegen das Urteil des Landgerichts Aachen vom 7. Juli 2016 wird

| Predicted | Gold |
|---|---|
| `Landgerichts Aachen` | `Landgerichts Aachen` |

**Example 10** (doc_id: `57570`) (sent_id: `57570`)


Die Klägerin beantragt , den Beschluss des Thüringer Landessozialgerichts vom 21. Juli 2016 und das Urteil des Sozialgerichts Meiningen vom 7. Januar 2015 aufzuheben sowie den Bescheid der Beklagten vom 15. April 2013 in der Gestalt des Widerspruchsbescheids vom 17. Mai 2013 abzuändern und die Beklagte zu verurteilen , ihr für die Zeit vom 1. Januar bis 28. März 2012 höheres Insolvenzgeld zu zahlen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts Meiningen` | `Sozialgerichts Meiningen` |

**Missed by this rule (FN):**

- `Thüringer Landessozialgerichts` (ORG)

**Example 11** (doc_id: `58013`) (sent_id: `58013`)


das Urteil des Bayerischen Landessozialgerichts vom 3. Juni 2016 und den Gerichtsbescheid des Sozialgerichts Nürnberg vom 30. August 2013 sowie den Bescheid der Beklagten vom 19. März 2012 und den Widerspruchsbescheid vom 18. Juni 2012 aufzuheben und die Beklagte zu verurteilen , unter Rücknahme des Bescheides vom 4. Februar 2005 und des Widerspruchsbescheides vom 22. April 2005 die Zeit vom 1. September 1973 bis 30. Juni 1990 als Zeit der Zugehörigkeit zum Zusatzversorgungssystem der technischen Intelligenz und die hierin erzielten Arbeitsentgelte festzustellen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts Nürnberg` | `Sozialgerichts Nürnberg` |

**Missed by this rule (FN):**

- `Bayerischen Landessozialgerichts` (ORG)

**Example 12** (doc_id: `58301`) (sent_id: `58301`)


Der Beklagte beantragt , das Urteil des Sächsischen Landessozialgerichts vom 9. Februar 2017 aufzuheben und die Berufungen der Kläger gegen das Urteil des Sozialgerichts Dresden vom 10. Februar 2014 zurückzuweisen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts Dresden` | `Sozialgerichts Dresden` |

**Missed by this rule (FN):**

- `Sächsischen Landessozialgerichts` (ORG)

**Example 13** (doc_id: `58405`) (sent_id: `58405`)


3. Mit Schriftsatz vom 8. März 2018 beantragt die Beschwerdeführerin durch ihren Bevollmächtigten , " die Vollstreckbarkeit " der Beschlüsse des Landgerichts Potsdam vom 11. März 2014 und vom " 20. Juli 2017 " ( gemeint wohl 17. Juli 2017 ) vorläufig auszusetzen .

| Predicted | Gold |
|---|---|
| `Landgerichts Potsdam` | `Landgerichts Potsdam` |

**Example 14** (doc_id: `59568`) (sent_id: `59568`)


Die Revision des Angeklagten gegen das Urteil des Landgerichts Essen vom 10. April 2017 wird als unbegründet verworfen , da die Nachprüfung des Urteils auf Grund der Revisionsrechtfertigung keinen Rechtsfehler zum Nachteil des Angeklagten ergeben hat ( § 349 Abs. 2 StPO ) .

| Predicted | Gold |
|---|---|
| `Landgerichts Essen` | `Landgerichts Essen` |

**Missed by this rule (FN):**

- `§ 349 Abs. 2 StPO` (NRM)

**Example 15** (doc_id: `59658`) (sent_id: `59658`)


A. I. 1. Der Beschwerdeführer ist auf Grundlage des Urteils des Landgerichts Lübeck vom 7. Oktober 2014 gemäß § 63 StGB wegen Mordes in einem psychiatrischen Krankenhaus der AMEOS Krankenhausgesellschaft Holstein mbH untergebracht , nachdem er im schuldunfähigen Zustand auf Grund einer wahnhaften Störung im Januar 2014 seine vierjährige Tochter und seinen sechs Jahre alten Sohn getötet hatte .

| Predicted | Gold |
|---|---|
| `Landgerichts Lübeck` | `Landgerichts Lübeck` |

**Missed by this rule (FN):**

- `§ 63 StGB` (NRM)
- `AMEOS Krankenhausgesellschaft Holstein mbH` (ORG)

**Example 16** (doc_id: `59742`) (sent_id: `59742`)


1. Dem Angeklagten A. wird auf seinen Antrag Wiedereinsetzung in den vorigen Stand wegen Versäumung der Frist zur Begründung der Revision gegen das Urteil des Landgerichts Göttingen vom 30. Mai 2017 gewährt .

| Predicted | Gold |
|---|---|
| `Landgerichts Göttingen` | `Landgerichts Göttingen` |

**Missed by this rule (FN):**

- `A.` (PER)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `54196`) (sent_id: `54196`)


Die Revision des Klägers gegen das Urteil des Landesarbeitsgerichts Hamburg vom 28. September 2016 - 1 Sa 18/16 - wird zurückgewiesen .

**False Positives:**

- `Landesarbeitsgerichts Hamburg` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Hamburg vom 28. September 2016 - 1 Sa 18/16 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Hamburg vom 28. September 2016 - 1 Sa 18/16 -`(RS)

**Example 1** (doc_id: `54385`) (sent_id: `54385`)


die Urteile des Landessozialgerichts Sachsen-Anhalt vom 9. März 2017 und des Sozialgerichts Dessau-Roßlau vom 2. Dezember 2013 sowie den Bescheid des Beklagten vom 16. Februar 2010 in der Gestalt des Widerspruchsbescheids vom 31. Mai 2010 aufzuheben .

**False Positives:**

- `Landessozialgerichts Sachsen` — partial — pred is substring of gold: `Landessozialgerichts Sachsen-Anhalt`
- `Sozialgerichts Dessau` — partial — pred is substring of gold: `Sozialgerichts Dessau-Roßlau`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Landessozialgerichts Sachsen-Anhalt`(ORG)
- `Sozialgerichts Dessau-Roßlau`(ORG)

**Example 2** (doc_id: `54438`) (sent_id: `54438`)


Der Haftbefehl des Amtsgerichts Neu-Ulm vom 31. Juli 2017 könne unter Berücksichtigung der Bedeutung und Tragweite des Grundrechts auf Freiheit der Person des Beschwerdeführers keinen Bestand haben .

**False Positives:**

- `Amtsgerichts Neu` — partial — pred is substring of gold: `Amtsgerichts Neu-Ulm`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Amtsgerichts Neu-Ulm`(ORG)

**Example 3** (doc_id: `54663`) (sent_id: `54663`)


Die Revision der Klägerin gegen das Urteil des Landesarbeitsgerichts Baden-Württemberg vom 23. Juni 2015 - 9 Sa 79/14 - wird auf ihre Kosten zurückgewiesen .

**False Positives:**

- `Landesarbeitsgerichts Baden` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Baden-Württemberg vom 23. Juni 2015 - 9 Sa 79/14 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Baden-Württemberg vom 23. Juni 2015 - 9 Sa 79/14 -`(RS)

**Example 4** (doc_id: `55158`) (sent_id: `55158`)


1. Die Revision des Beklagten gegen das Urteil des Landesarbeitsgerichts Rheinland-Pfalz vom 7. Juli 2016 - 6 Sa 23/16 - wird zurückgewiesen .

**False Positives:**

- `Landesarbeitsgerichts Rheinland` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Rheinland-Pfalz vom 7. Juli 2016 - 6 Sa 23/16 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Rheinland-Pfalz vom 7. Juli 2016 - 6 Sa 23/16 -`(RS)

**Example 5** (doc_id: `55511`) (sent_id: `55511`)


Die Beschwerde der Antragstellerin gegen den Beschluss des Finanzgerichts Hamburg vom 15. August 2016 1 V 41/16 wird als unbegründet zurückgewiesen .

**False Positives:**

- `Finanzgerichts Hamburg` — partial — pred is substring of gold: `Beschluss des Finanzgerichts Hamburg vom 15. August 2016 1 V 41/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Beschluss des Finanzgerichts Hamburg vom 15. August 2016 1 V 41/16`(RS)

**Example 6** (doc_id: `55622`) (sent_id: `55622`)


Die Revision des Klägers gegen das Urteil des Landesarbeitsgerichts Köln vom 8. Dezember 2016 - 8 Sa 540/16 - wird zurückgewiesen .

**False Positives:**

- `Landesarbeitsgerichts Köln` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Köln vom 8. Dezember 2016 - 8 Sa 540/16 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Köln vom 8. Dezember 2016 - 8 Sa 540/16 -`(RS)

**Example 7** (doc_id: `55659`) (sent_id: `55659`)


Die Beschwerde des Klägers wegen Nichtzulassung der Revision gegen das Urteil des Finanzgerichts Münster vom 12. September 2017 15 K 3562/14 U wird als unbegründet zurückgewiesen .

**False Positives:**

- `Finanzgerichts Münster` — partial — pred is substring of gold: `Urteil des Finanzgerichts Münster vom 12. September 2017 15 K 3562/14 U`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Finanzgerichts Münster vom 12. September 2017 15 K 3562/14 U`(RS)

**Example 8** (doc_id: `56170`) (sent_id: `56170`)


2. Die Berufung des Beklagten gegen das Urteil des Arbeitsgerichts Oberhausen vom 15. Dezember 2016 - 4 Ca 1318/16 - wird zurückgewiesen .

**False Positives:**

- `Arbeitsgerichts Oberhausen` — partial — pred is substring of gold: `Urteil des Arbeitsgerichts Oberhausen vom 15. Dezember 2016 - 4 Ca 1318/16 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Arbeitsgerichts Oberhausen vom 15. Dezember 2016 - 4 Ca 1318/16 -`(RS)

**Example 9** (doc_id: `56230`) (sent_id: `56230`)


Die Revision der Klägerin gegen das Urteil des Landesarbeitsgerichts Baden-Württemberg vom 23. Juni 2015 - 22 Sa 64/14 - wird auf ihre Kosten zurückgewiesen .

**False Positives:**

- `Landesarbeitsgerichts Baden` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Baden-Württemberg vom 23. Juni 2015 - 22 Sa 64/14 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Baden-Württemberg vom 23. Juni 2015 - 22 Sa 64/14 -`(RS)

**Example 10** (doc_id: `56331`) (sent_id: `56331`)


Auf die Berufung der Beklagten wird - unter Zurückweisung der Anschlussberufung des Klägers - das Urteil des Arbeitsgerichts Bonn vom 11. November 2015 - 4 Ca 1615/15 - abgeändert und die Klage abgewiesen .

**False Positives:**

- `Arbeitsgerichts Bonn` — partial — pred is substring of gold: `Urteil des Arbeitsgerichts Bonn vom 11. November 2015 - 4 Ca 1615/15 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Arbeitsgerichts Bonn vom 11. November 2015 - 4 Ca 1615/15 -`(RS)

**Example 11** (doc_id: `56355`) (sent_id: `56355`)


Auf die Revision der Klägerin wird das Urteil des Landesarbeitsgerichts Hamburg vom 5. November 2015 - 1 Sa 11/15 - aufgehoben .

**False Positives:**

- `Landesarbeitsgerichts Hamburg` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Hamburg vom 5. November 2015 - 1 Sa 11/15 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Hamburg vom 5. November 2015 - 1 Sa 11/15 -`(RS)

**Example 12** (doc_id: `56480`) (sent_id: `56480`)


Die Revision des Klägers gegen das Urteil des Landesarbeitsgerichts Rheinland-Pfalz vom 17. Februar 2016 - 4 Sa 202/15 - wird zurückgewiesen .

**False Positives:**

- `Landesarbeitsgerichts Rheinland` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Rheinland-Pfalz vom 17. Februar 2016 - 4 Sa 202/15 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Rheinland-Pfalz vom 17. Februar 2016 - 4 Sa 202/15 -`(RS)

**Example 13** (doc_id: `56544`) (sent_id: `56544`)


1. Auf die Revision der Klägerin wird das Urteil des Landesarbeitsgerichts Düsseldorf vom 14. Juni 2017 - 7 Sa 81/17 - aufgehoben .

**False Positives:**

- `Landesarbeitsgerichts Düsseldorf` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Düsseldorf vom 14. Juni 2017 - 7 Sa 81/17 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Düsseldorf vom 14. Juni 2017 - 7 Sa 81/17 -`(RS)

**Example 14** (doc_id: `57953`) (sent_id: `57953`)


2. Die Berufung des Klägers gegen das Urteil des Arbeitsgerichts Dortmund vom 27. März 2014 - 6 Ca 3695/11 - wird zurückgewiesen .

**False Positives:**

- `Arbeitsgerichts Dortmund` — partial — pred is substring of gold: `Urteil des Arbeitsgerichts Dortmund vom 27. März 2014 - 6 Ca 3695/11 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Arbeitsgerichts Dortmund vom 27. März 2014 - 6 Ca 3695/11 -`(RS)

**Example 15** (doc_id: `58297`) (sent_id: `58297`)


In Bezug auf den gerügten Haftbefehl des Amtsgerichts Neu-Ulm vom 31. Juli 2017 sei die Verfassungsbeschwerde wegen des Grundsatzes der Subsidiarität der Verfassungsbeschwerde hingegen unzulässig , da eine abschließende Sachprüfung durch das Oberlandesgericht München noch nicht stattgefunden habe .

**False Positives:**

- `Amtsgerichts Neu` — partial — pred is substring of gold: `Amtsgerichts Neu-Ulm`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Amtsgerichts Neu-Ulm`(ORG)
- `Oberlandesgericht München`(ORG)

**Example 16** (doc_id: `58399`) (sent_id: `58399`)


Die Revision der Klägerin gegen das Urteil des Finanzgerichts Hamburg vom 21. April 2017 4 K 186/16 wird als unbegründet zurückgewiesen .

**False Positives:**

- `Finanzgerichts Hamburg` — partial — pred is substring of gold: `Urteil des Finanzgerichts Hamburg vom 21. April 2017 4 K 186/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Finanzgerichts Hamburg vom 21. April 2017 4 K 186/16`(RS)

**Example 17** (doc_id: `58546`) (sent_id: `58546`)


Auf die Revision des Beklagten wird das Urteil des Finanzgerichts München vom 29. März 2017 3 K 2565/16 aufgehoben .

**False Positives:**

- `Finanzgerichts München` — partial — pred is substring of gold: `Urteil des Finanzgerichts München vom 29. März 2017 3 K 2565/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Finanzgerichts München vom 29. März 2017 3 K 2565/16`(RS)

**Example 18** (doc_id: `58604`) (sent_id: `58604`)


Die Rechtsbeschwerde des Betriebsrats gegen den Beschluss des Landesarbeitsgerichts Sachsen-Anhalt vom 22. März 2016 - 6 TaBV 39/14 - wird zurückgewiesen .

**False Positives:**

- `Landesarbeitsgerichts Sachsen` — partial — pred is substring of gold: `Beschluss des Landesarbeitsgerichts Sachsen-Anhalt vom 22. März 2016 - 6 TaBV 39/14 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Beschluss des Landesarbeitsgerichts Sachsen-Anhalt vom 22. März 2016 - 6 TaBV 39/14 -`(RS)

**Example 19** (doc_id: `58915`) (sent_id: `58915`)


Die Beschwerde gegen die Nichtzulassung der Revision in dem Urteil des 7. Zivilsenats des Oberlandesgerichts Frankfurt am Main vom 8. Juni 2017 wird auf Kosten des Klägers als unzulässig verworfen .

**False Positives:**

- `Oberlandesgerichts Frankfurt` — partial — pred is substring of gold: `Urteil des 7. Zivilsenats des Oberlandesgerichts Frankfurt am Main`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des 7. Zivilsenats des Oberlandesgerichts Frankfurt am Main`(RS)

**Example 20** (doc_id: `58942`) (sent_id: `58942`)


Eine - über die behauptete Verletzung von Art. 19 Abs. 4 Satz 1 GG hinausgehende - verfassungsgerichtliche Sachprüfung widerspräche dem Grundsatz der Subsidiarität der Verfassungsbeschwerde , weil eine abschließende fachgerichtliche Prüfung des angegriffenen Haftbefehls des Amtsgerichts Neu-Ulm vom 31. Juli 2017 bislang - entgegen den Vorgaben von Art. 19 Abs. 4 Satz 1 GG - nicht erfolgt ist .

**False Positives:**

- `Amtsgerichts Neu` — partial — pred is substring of gold: `Amtsgerichts Neu-Ulm`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 19 Abs. 4 Satz 1 GG`(NRM)
- `Amtsgerichts Neu-Ulm`(ORG)
- `Art. 19 Abs. 4 Satz 1 GG`(NRM)

**Example 21** (doc_id: `59195`) (sent_id: `59195`)


1. Auf die Revision des beklagten Landes wird das Urteil des Landesarbeitsgerichts Hamm vom 4. Juli 2016 - 11 Sa 1330/14 - aufgehoben .

**False Positives:**

- `Landesarbeitsgerichts Hamm` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Hamm vom 4. Juli 2016 - 11 Sa 1330/14 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Hamm vom 4. Juli 2016 - 11 Sa 1330/14 -`(RS)

**Example 22** (doc_id: `59356`) (sent_id: `59356`)


Über die sofortige Beschwerde der Nebenklägerin K. T. gegen die im Urteil des Landgerichts Frankfurt am Main vom 15. August 2017 getroffene Kostenentscheidung hat das Oberlandesgericht Frankfurt am Main zu entscheiden .

**False Positives:**

- `Landgerichts Frankfurt` — partial — pred is substring of gold: `Landgerichts Frankfurt am Main`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K. T.`(PER)
- `Landgerichts Frankfurt am Main`(ORG)
- `Oberlandesgericht Frankfurt am Main`(ORG)

**Example 23** (doc_id: `59434`) (sent_id: `59434`)


1. Das Urteil des Pfälzischen Oberlandesgerichts Zweibrücken vom 29. Januar 2015 - 4 U 81/14 - verletzt die Beschwerdeführerin in ihrem Grundrecht aus Artikel 5 Absatz 1 Satz 2 des Grundgesetzes und wird aufgehoben .

**False Positives:**

- `Oberlandesgerichts Zweibrücken` — partial — pred is substring of gold: `Urteil des Pfälzischen Oberlandesgerichts Zweibrücken vom 29. Januar 2015 - 4 U 81/14 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Pfälzischen Oberlandesgerichts Zweibrücken vom 29. Januar 2015 - 4 U 81/14 -`(RS)
- `Artikel 5 Absatz 1 Satz 2 des Grundgesetzes`(NRM)

**Example 24** (doc_id: `59490`) (sent_id: `59490`)


2. a ) Das Urteil des Landesarbeitsgerichts Nürnberg vom 30. Januar 2014 - 5 Sa 1/13 - und das Endurteil des Arbeitsgerichts Bamberg vom 10. Oktober 2012 - 2 Ca 1097/11 - verletzen den Beschwerdeführer in seinem Grundrecht aus Artikel 2 Absatz 1 Satz 1 in Verbindung mit Artikel 20 Absatz 3 des Grundgesetzes .

**False Positives:**

- `Arbeitsgerichts Bamberg` — partial — pred is substring of gold: `Endurteil des Arbeitsgerichts Bamberg vom 10. Oktober 2012 - 2 Ca 1097/11 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Nürnberg vom 30. Januar 2014 - 5 Sa 1/13 -`(RS)
- `Endurteil des Arbeitsgerichts Bamberg vom 10. Oktober 2012 - 2 Ca 1097/11 -`(RS)
- `Artikel 2 Absatz 1 Satz 1 in Verbindung mit Artikel 20 Absatz 3 des Grundgesetzes`(NRM)

</details>

---

## `Hyphenated Company Names`

**F1:** 0.059 | **Precision:** 0.543 | **Recall:** 0.031  

**Format:** `regex`  
**Rule ID:** `c676eb05`  
**Description:**
Matches company names with hyphens or specific patterns like 'E K-Konzerns' or 'A-GbR' that were missed by generic patterns.

**Content:**
```
\b([A-Z][\-]?[A-Z]?\s*(?:GmbH|AG|KG|GbR|Fonds|V\.|B\.\s*V\.|Klinik|Schulzentrum|Finanzamt|Landratsamt|Berufsschulzentrum|Jobcenter|Botschaft|Kammer|Senat|Stelle|Amt|Verband|Zweckverband|Firma|Bank|Verlag|GmbH\s*&\s*Co\.\s*KG|Konzerns?|AG\s*&\s*Co\.\s*KG|S\.r\.l\.))\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.543 | 0.031 | 0.059 | 46 | 25 | 21 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 25 | 21 | 767 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53475`) (sent_id: `53475`)


Gegenstand des BgA war die Verpachtung der städtischen Schwimmbäder an die ... GmbH ( S-GmbH ) , eine 100 % -ige Tochtergesellschaft der Klägerin .

| Predicted | Gold |
|---|---|
| `S-GmbH` | `S-GmbH` |

**Missed by this rule (FN):**

- `... GmbH` (ORG)

**Example 1** (doc_id: `53517`) (sent_id: `53517`)


An der Klägerin sind als Komplementärin die A-GmbH , als Kommanditistin seit dem Jahr 2000 die Holding-KG beteiligt .

| Predicted | Gold |
|---|---|
| `A-GmbH` | `A-GmbH` |

**Example 2** (doc_id: `53594`) (sent_id: `53594`)


Die von der Krankenkasse nicht getragenen Kosten der stationären Behandlung zahlte der Kläger zu 1. Die Therapie der E in der K-Klinik war im Vorfeld durch den Medizinischen Dienst der Krankenversicherungen ( MDK ) auf ihre medizinische Notwendigkeit geprüft und befürwortet worden .

| Predicted | Gold |
|---|---|
| `K-Klinik` | `K-Klinik` |

**Missed by this rule (FN):**

- `E` (PER)
- `Medizinischen Dienst der Krankenversicherungen` (ORG)
- `MDK` (ORG)

**Example 3** (doc_id: `53777`) (sent_id: `53777`)


Am 14. November 2000 wurden den Aktionären der A-AG vereinbarungsgemäß ... Stück vinkulierte Namensaktien an der B-AG übertragen .

| Predicted | Gold |
|---|---|
| `A-AG` | `A-AG` |
| `B-AG` | `B-AG` |

**Example 4** (doc_id: `53825`) (sent_id: `53825`)


Die Beteiligung an der S-GmbH gehörte zum Betriebsvermögen des BgA .

| Predicted | Gold |
|---|---|
| `S-GmbH` | `S-GmbH` |

**Example 5** (doc_id: `54234`) (sent_id: `54234`)


Der Beklagte und Beschwerdeführer ( das Finanzamt - FA - ) erließ gemäß § 27 Abs. 19 Satz 1 UStG einen Änderungsbescheid gegen den Kläger , nach dem er die an Q-KG erbrachte Bauleistung als Steuerschuldner zu versteuern habe .

| Predicted | Gold |
|---|---|
| `Q-KG` | `Q-KG` |

**Missed by this rule (FN):**

- `§ 27 Abs. 19 Satz 1 UStG` (NRM)

**Example 6** (doc_id: `55059`) (sent_id: `55059`)


Der Feststellungsbescheid benennt - neben dem Anleger ( zum Anleger als Inhaltsadressaten vgl. Lübbehüsen in Berger / Steck / Lübbehüsen , a. a. O. , § 15 Rz 59 , 65 ) - mit dem A-Fonds ein Spezial-Sondervermögen als ( weiteren ) möglichen Feststellungsbeteiligten .

| Predicted | Gold |
|---|---|
| `A-Fonds` | `A-Fonds` |

**Missed by this rule (FN):**

- `Lübbehüsen in Berger / Steck / Lübbehüsen , a. a. O. , § 15 Rz 59 , 65` (LIT)

**Example 7** (doc_id: `55305`) (sent_id: `55305`)


Die ehemalige A-AG verpflichtete sich durch Vermögensübertragungsvertrag vom 17. Juni 1999 , ihr Vermögen als Ganzes mit allen Rechten und Pflichten unter Auflösung ohne Abwicklung nach § 174 Abs. 1 des Umwandlungsgesetzes ( UmwG ) im Wege der Vermögensübertragung mit Wirkung zum 1. Januar 2000 , 0:00 Uhr ( handelsrechtlicher Übertragungsstichtag ) , auf die Klägerin zu übertragen .

| Predicted | Gold |
|---|---|
| `A-AG` | `A-AG` |

**Missed by this rule (FN):**

- `§ 174 Abs. 1 des Umwandlungsgesetzes` (NRM)
- `UmwG` (NRM)

**Example 8** (doc_id: `55316`) (sent_id: `55316`)


Da nach § 6 des Vertrags der Jahresabschluss für die M-GmbH innerhalb von sechs Monaten nach Ablauf eines jeden Geschäftsjahres zu erstellen und dem stillen Gesellschafter zu übermitteln war , kann daraus nur der Rückschluss gezogen werden , dass zum hier maßgeblichen Bilanzstichtag von einer Aufstellung des Jahresabschlusses durch den Kläger auszugehen war .

| Predicted | Gold |
|---|---|
| `M-GmbH` | `M-GmbH` |

**Missed by this rule (FN):**

- `§ 6 des Vertrags` (REG)

**Example 9** (doc_id: `55934`) (sent_id: `55934`)


Die Auflösung der A-AG sei erst im Jahr 2012 gemäß § 262 Abs. 1 des Aktiengesetzes ( AktG ) eingetreten .

| Predicted | Gold |
|---|---|
| `A-AG` | `A-AG` |

**Missed by this rule (FN):**

- `§ 262 Abs. 1 des Aktiengesetzes` (NRM)
- `AktG` (NRM)

**Example 10** (doc_id: `55949`) (sent_id: `55949`)


Aus den vorgelegten Steuerakten ergibt sich indes , dass erst mit Schreiben des steuerlichen Beraters der A-GbR vom 29. Mai 2008 eine als Eröffnungsbilanz bezeichnete Aufstellung vorgelegt wurde .

| Predicted | Gold |
|---|---|
| `A-GbR` | `A-GbR` |

**Example 11** (doc_id: `56134`) (sent_id: `56134`)


Da die A-GbR aufgrund ihrer Geschäftsbeziehung der C- B. V. näher stehe als das FA , hätte es dem Kläger oblegen , die Angaben über die wirtschaftlichen Verhältnisse der C- B. V. weiter zu konkretisieren .

| Predicted | Gold |
|---|---|
| `A-GbR` | `A-GbR` |

**Missed by this rule (FN):**

- `C- B. V.` (ORG)
- `C- B. V.` (ORG)

**Example 12** (doc_id: `56397`) (sent_id: `56397`)


Vertretungsberechtigt für die Holding-KG ist die C-GmbH , vertreten durch ihren Geschäftsführer D.

| Predicted | Gold |
|---|---|
| `C-GmbH` | `C-GmbH` |

**Missed by this rule (FN):**

- `D.` (PER)

**Example 13** (doc_id: `56596`) (sent_id: `56596`)


Hierfür wären die Urteile des Landgerichts Cottbus vom ... und des Brandenburgischen Oberlandesgerichts vom ... zu beachten , wonach der Kläger durch Kündigung vom ... 2005 wirksam aus der A-GbR ausgeschlossen wurde .

| Predicted | Gold |
|---|---|
| `A-GbR` | `A-GbR` |

**Missed by this rule (FN):**

- `Landgerichts Cottbus` (ORG)
- `Brandenburgischen Oberlandesgerichts` (ORG)

**Example 14** (doc_id: `56721`) (sent_id: `56721`)


a ) Zu Recht geht das FG allerdings davon aus , dass der Kläger an der M-GmbH & atypisch Still , einer Personengesellschaft i. S. des § 15 Abs. 1 Satz 1 Nr. 2 EStG , über die zwischengeschaltete B-GbR mittelbar i. S. des § 15 Abs. 1 Satz 1 Nr. 2 Satz 2 EStG beteiligt und er daher hinsichtlich der für seine im Dienst der M-GmbH & atypisch Still erbrachten Tätigkeiten wie ein unmittelbar beteiligter Gesellschafter anzusehen war .

| Predicted | Gold |
|---|---|
| `B-GbR` | `B-GbR` |

**Missed by this rule (FN):**

- `M-GmbH & atypisch Still` (ORG)
- `§ 15 Abs. 1 Satz 1 Nr. 2 EStG` (NRM)
- `§ 15 Abs. 1 Satz 1 Nr. 2 Satz 2 EStG` (NRM)
- `M-GmbH & atypisch Still` (ORG)

**Example 15** (doc_id: `56800`) (sent_id: `56800`)


Der Kläger war zu 50 % an der X-GmbH ( GmbH ) beteiligt , über deren Vermögen am 1. Dezember 2004 das Insolvenzverfahren eröffnet wurde .

| Predicted | Gold |
|---|---|
| `X-GmbH` | `X-GmbH` |

**Example 16** (doc_id: `57272`) (sent_id: `57272`)


Dem Vermögen des A-Fonds waren bei der Depotbank überwiegend Aktien und im Übrigen festverzinsliche Wertpapiere , Genussscheine , Geldmarktpapiere sowie in geringem Umfang auch Derivate zugeordnet .

| Predicted | Gold |
|---|---|
| `A-Fonds` | `A-Fonds` |

**Example 17** (doc_id: `57686`) (sent_id: `57686`)


Das FA erließ hiernach am 22. November 2010 einen " Bescheid über die gesonderte - und einheitliche - Feststellung nach § 15 Abs. 1 InvStG für : A-Fonds " , der auch den Anleger ausdrücklich benennt .

| Predicted | Gold |
|---|---|
| `A-Fonds` | `A-Fonds` |

**Missed by this rule (FN):**

- `§ 15 Abs. 1 InvStG` (NRM)

**Example 18** (doc_id: `58089`) (sent_id: `58089`)


Sie war mehrheitlich an der ehemaligen ... AG , seit dem 18. Oktober 1999 B-AG , beteiligt .

| Predicted | Gold |
|---|---|
| `B-AG` | `B-AG` |

**Missed by this rule (FN):**

- `... AG` (ORG)

**Example 19** (doc_id: `58288`) (sent_id: `58288`)


Gemäß § 4 Abs. 2 des Vertrags war u. a. vereinbart , dass der Prinzipal ( M-GmbH ) , dem die alleinige Geschäftsführung oblag , den Wechsel des steuerlichen Beraters nur mit Einwilligung des stillen Gesellschafters vornehmen dürfe .

| Predicted | Gold |
|---|---|
| `M-GmbH` | `M-GmbH` |

**Missed by this rule (FN):**

- `§ 4 Abs. 2 des Vertrags` (REG)

**Example 20** (doc_id: `58937`) (sent_id: `58937`)


d ) Der Kläger hat der X-GmbH nicht etwa ein partiarisches Darlehen gewährt .

| Predicted | Gold |
|---|---|
| `X-GmbH` | `X-GmbH` |

**Example 21** (doc_id: `59187`) (sent_id: `59187`)


aa ) Die M-GmbH & atypisch Still war durch die Eröffnung des Insolvenzverfahrens über das Vermögen der M-GmbH kraft Gesetzes ( § 728 des Bürgerlichen Gesetzbuchs ) aufgelöst ( vgl. Baumbach / Hopt / Roth , HGB , 37. Aufl. , § 234 Rz 5 ) .

| Predicted | Gold |
|---|---|
| `M-GmbH` | `M-GmbH` |

**Missed by this rule (FN):**

- `M-GmbH & atypisch Still` (ORG)
- `§ 728 des Bürgerlichen Gesetzbuchs` (NRM)
- `Baumbach / Hopt / Roth , HGB , 37. Aufl. , § 234 Rz 5` (LIT)

**Example 22** (doc_id: `59488`) (sent_id: `59488`)


Die A-GbR habe ihre steuerlichen Mitwirkungspflichten verletzt .

| Predicted | Gold |
|---|---|
| `A-GbR` | `A-GbR` |

**Example 23** (doc_id: `59897`) (sent_id: `59897`)


So legt der Kläger - soweit erkennbar unwidersprochen durch das FA - dar , dass das mit der A-GbR vereinbarte Projekt eines Ferienresorts die einzige geschäftliche Tätigkeit der C- B. V. dargestellt habe .

| Predicted | Gold |
|---|---|
| `A-GbR` | `A-GbR` |

**Missed by this rule (FN):**

- `C- B. V.` (ORG)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53973`) (sent_id: `53973`)


Die Festsetzung des Streitwerts folgt aus § 47 Abs. 1 Satz 1 und Abs. 3 , § 52 Abs. 1 GKG .

**False Positives:**

- `GKG` — partial — pred is substring of gold: `§ 47 Abs. 1 Satz 1 und Abs. 3 , § 52 Abs. 1 GKG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 47 Abs. 1 Satz 1 und Abs. 3 , § 52 Abs. 1 GKG`(NRM)

**Example 1** (doc_id: `54330`) (sent_id: `54330`)


Mit Kostenrechnung vom 25. Januar 2018 KostL 1905/17 ( IV B 58/17 ) setzte die Kostenstelle des BFH für das Verfahren auf der Grundlage eines Streitwerts von 1.906.096 € Gerichtskosten nach Nr. 6500 des Kostenverzeichnisses zu § 3 Abs. 2 des Gerichtskostengesetzes ( GKG ) in Höhe von 17.512 € fest .

**False Positives:**

- `GKG` — type mismatch — same span as gold: `GKG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `vom 25. Januar 2018 KostL 1905/17 ( IV B 58/17 )`(RS)
- `BFH`(ORG)
- `Nr. 6500 des Kostenverzeichnisses zu § 3 Abs. 2 des Gerichtskostengesetzes`(NRM)
- `GKG`(NRM)

**Example 2** (doc_id: `54878`) (sent_id: `54878`)


Das entspricht in der Sache der Regelung in § 66 Abs 8 GKG , die bei Erinnerungen gegen den Ansatz von Gerichtskosten nach dem GKG zur Anwendung gelangt .

**False Positives:**

- `GKG` — partial — pred is substring of gold: `§ 66 Abs 8 GKG`
- `GKG` — similar text (different position): `§ 66 Abs 8 GKG`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 66 Abs 8 GKG`(NRM)
- `GKG`(NRM)

**Example 3** (doc_id: `55356`) (sent_id: `55356`)


Dies wird beispielhaft durch Regelungen wie in Art. 13 Abs. 1 Nr. 5 Buchst. b Doppelbuchst. dd des Kommunalabgabengesetzes Bayern ( KAG BY ) bestätigt .

**False Positives:**

- `KAG` — partial — pred is substring of gold: `KAG BY`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 13 Abs. 1 Nr. 5 Buchst. b Doppelbuchst. dd des Kommunalabgabengesetzes Bayern`(NRM)
- `KAG BY`(NRM)

**Example 4** (doc_id: `55464`) (sent_id: `55464`)


Mit Beschluss vom 30. November 2017 X E 12/17 hat der Senat durch die Einzelrichterin nach § 66 Abs. 6 Satz 1 GKG die Erinnerung zurückgewiesen , da die Kostenrechnung , insbesondere der zweifache Ansatz der Gebühr von 60 € , inhaltlich zutreffend sei .

**False Positives:**

- `GKG` — partial — pred is substring of gold: `§ 66 Abs. 6 Satz 1 GKG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Beschluss vom 30. November 2017 X E 12/17`(RS)
- `§ 66 Abs. 6 Satz 1 GKG`(NRM)

**Example 5** (doc_id: `55581`) (sent_id: `55581`)


Rechtsfragen , die sich der Vorinstanz nicht gestellt haben oder auf die sie nicht entscheidend abgehoben hat , können aber nicht zur Zulassung der Revision führen , weil ihre Klärung in einem Revisionsverfahren nicht zu erwarten ist ( BVerwG , Beschlüsse vom 29. Juni 1992 - 3 B 102.91 - Buchholz 418.04 Heilpraktiker Nr. 17 S. 6 , vom 5. Oktober 2009 - 6 B 17.09 - Buchholz 442.066 § 24 TKG Nr. 4 Rn. 7 und vom 21. März 2014 - 6 B 55.13 - Buchholz 442.09 § 23 AEG Nr. 3 Rn. 7 ) .

**False Positives:**

- `TKG` — partial — pred is substring of gold: `vom 5. Oktober 2009 - 6 B 17.09 - Buchholz 442.066 § 24 TKG Nr. 4 Rn. 7`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerwG , Beschlüsse vom 29. Juni 1992 - 3 B 102.91 - Buchholz 418.04 Heilpraktiker Nr. 17 S. 6`(RS)
- `vom 5. Oktober 2009 - 6 B 17.09 - Buchholz 442.066 § 24 TKG Nr. 4 Rn. 7`(RS)
- `vom 21. März 2014 - 6 B 55.13 - Buchholz 442.09 § 23 AEG Nr. 3 Rn. 7`(RS)

**Example 6** (doc_id: `56266`) (sent_id: `56266`)


Gegen den nach seiner Angabe am 5. Januar 2018 zugegangenen Beschluss hat der Rügeführer am 18. Januar 2018 Anhörungsrüge erhoben , die einstweilige AdV der angefochtenen Entscheidung nach § 66 Abs. 7 Satz 2 GKG sowie Akteneinsicht beantragt .

**False Positives:**

- `GKG` — partial — pred is substring of gold: `§ 66 Abs. 7 Satz 2 GKG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 66 Abs. 7 Satz 2 GKG`(NRM)

**Example 7** (doc_id: `56591`) (sent_id: `56591`)


Während ansonsten in kostenrechtlichen Verfahren der Erinnerung nach dem GKG bzw dem RVG nunmehr auch in dritter Instanz grundsätzlich eine Entscheidung durch den Einzelrichter vorgesehen ist ( vgl § 66 Abs 6 S 1 GKG bzw § 33 Abs 8 S 1 RVG - s dazu BGH Beschluss vom 23. 4. 2015 - I ZB 73/14 - NJW 2015 , 2194 ) , lässt das SGG bislang auch bei Erinnerungen ( §§ 178 , 189 Abs 2 S 2 SGG ) ein Tätigwerden des Einzelrichters lediglich im Rahmen des § 155 Abs 2 S 1 Nr 5 , Abs 3 und 4 SGG zu ( Reichel in Zeihe / Hauck <Hrsg> , SGG , Stand August 2017 , § 189 RdNr 10a ) .

**False Positives:**

- `GKG` — type mismatch — same span as gold: `GKG`
- `GKG` — similar text (different position): `GKG`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `GKG`(NRM)
- `RVG`(NRM)
- `§ 66 Abs 6 S 1 GKG`(NRM)
- `§ 33 Abs 8 S 1 RVG`(NRM)
- `BGH Beschluss vom 23. 4. 2015 - I ZB 73/14 - NJW 2015 , 2194`(RS)
- `SGG`(NRM)
- `§§ 178 , 189 Abs 2 S 2 SGG`(NRM)
- `§ 155 Abs 2 S 1 Nr 5 , Abs 3 und 4 SGG`(NRM)
- `Reichel in Zeihe / Hauck <Hrsg> , SGG , Stand August 2017 , § 189 RdNr 10a`(LIT)

**Example 8** (doc_id: `56721`) (sent_id: `56721`)


a ) Zu Recht geht das FG allerdings davon aus , dass der Kläger an der M-GmbH & atypisch Still , einer Personengesellschaft i. S. des § 15 Abs. 1 Satz 1 Nr. 2 EStG , über die zwischengeschaltete B-GbR mittelbar i. S. des § 15 Abs. 1 Satz 1 Nr. 2 Satz 2 EStG beteiligt und er daher hinsichtlich der für seine im Dienst der M-GmbH & atypisch Still erbrachten Tätigkeiten wie ein unmittelbar beteiligter Gesellschafter anzusehen war .

**False Positives:**

- `M-GmbH` — partial — pred is substring of gold: `M-GmbH & atypisch Still`
- `M-GmbH` — similar text (different position): `M-GmbH & atypisch Still`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `M-GmbH & atypisch Still`(ORG)
- `§ 15 Abs. 1 Satz 1 Nr. 2 EStG`(NRM)
- `B-GbR`(ORG)
- `§ 15 Abs. 1 Satz 1 Nr. 2 Satz 2 EStG`(NRM)
- `M-GmbH & atypisch Still`(ORG)

**Example 9** (doc_id: `57957`) (sent_id: `57957`)


Der Gewinnfeststellungsbescheid wurde dem Kläger als Empfangsbevollmächtigtem der M-GmbH & atypisch Still bekanntgegeben .

**False Positives:**

- `M-GmbH` — partial — pred is substring of gold: `M-GmbH & atypisch Still`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `M-GmbH & atypisch Still`(ORG)

**Example 10** (doc_id: `58128`) (sent_id: `58128`)


Für diese Auslegung spricht bereits die gesetzliche Vorgabe in § 101 Abs 1 S 1 Nr 4 SGB V , nach der sich " die Partner der BAG " verpflichten müssen , den " bisherigen Praxisumfang " nicht wesentlich zu überschreiten , sowie die ergänzende Vorgabe in § 23a Nr 4 BedarfsplRL aF , nach der die Erklärungen bei der Aufnahme eines Arztes in eine bereits gebildete BAG von allen Vertragsärzten abzugeben sind .

**False Positives:**

- `BAG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 101 Abs 1 S 1 Nr 4 SGB V`(NRM)
- `§ 23a Nr 4 BedarfsplRL aF`(REG)

**Example 11** (doc_id: `58206`) (sent_id: `58206`)


Die Rüge ist innerhalb bestimmter Frist bei dem Gericht zu erheben , dessen Entscheidung angegriffen wird ( § 69a Abs. 2 Satz 1 bis 4 GKG ) .

**False Positives:**

- `GKG` — partial — pred is substring of gold: `§ 69a Abs. 2 Satz 1 bis 4 GKG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 69a Abs. 2 Satz 1 bis 4 GKG`(NRM)

**Example 12** (doc_id: `58850`) (sent_id: `58850`)


Durch das Gesetz zur weiteren Reform der gesetzlichen Rentenversicherungen und über die Fünfzehnte Anpassung der Renten aus den gesetzlichen Rentenversicherungen sowie über die Anpassung der Geldleistungen aus der gesetzlichen Unfallversicherung ( Rentenreformgesetz - RRG ) vom 16. Oktober 1972 ( BGBl. I S. 1965 ) wurde § 48 Abs. 1 Nr. 1 RKG dahin geändert , dass Knappschaftsruhegeld auf Antrag ua. bereits ab der Vollendung des 63. Lebensjahres gewährt wird , wenn die Wartezeit nach § 49 Abs. 3 RKG erfüllt war .

**False Positives:**

- `RKG` — partial — pred is substring of gold: `§ 48 Abs. 1 Nr. 1 RKG`
- `RKG` — similar text (different position): `§ 48 Abs. 1 Nr. 1 RKG`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Gesetz zur weiteren Reform der gesetzlichen Rentenversicherungen und über die Fünfzehnte Anpassung der Renten aus den gesetzlichen Rentenversicherungen sowie über die Anpassung der Geldleistungen aus der gesetzlichen Unfallversicherung ( Rentenreformgesetz - RRG ) vom 16. Oktober 1972 ( BGBl. I S. 1965 )`(NRM)
- `§ 48 Abs. 1 Nr. 1 RKG`(NRM)
- `§ 49 Abs. 3 RKG`(NRM)

**Example 13** (doc_id: `59187`) (sent_id: `59187`)


aa ) Die M-GmbH & atypisch Still war durch die Eröffnung des Insolvenzverfahrens über das Vermögen der M-GmbH kraft Gesetzes ( § 728 des Bürgerlichen Gesetzbuchs ) aufgelöst ( vgl. Baumbach / Hopt / Roth , HGB , 37. Aufl. , § 234 Rz 5 ) .

**False Positives:**

- `M-GmbH` — similar text (different position): `M-GmbH & atypisch Still`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `M-GmbH & atypisch Still`(ORG)
- `M-GmbH`(ORG)
- `§ 728 des Bürgerlichen Gesetzbuchs`(NRM)
- `Baumbach / Hopt / Roth , HGB , 37. Aufl. , § 234 Rz 5`(LIT)

**Example 14** (doc_id: `59220`) (sent_id: `59220`)


Die Honorarnachteile , die für eine BAG mit einem Arzt in der Aufbauphase aufgrund der Zuweisung nur einer fallzahlabhängigen Obergrenze in typischen Konstellationen ( etwa der Übergabe einer Praxis im Wege einer vorübergehenden gemeinsamen Ausübung der Praxistätigkeit ) entstehen , sind nicht die zwangsläufige Folge dessen , dass Ärzten in der Aufbauphase überhaupt das bundesrechtlich geforderte sofortige Wachstum bis zum Fachgruppendurchschnitt eröffnet wird .

**False Positives:**

- `BAG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 15** (doc_id: `59298`) (sent_id: `59298`)


Die Festsetzung des Streitwerts beruht auf § 197a Abs 1 S 1 Teils 1 SGG iVm § 63 Abs 2 S 1 , § 52 Abs 1 und 3 sowie § 47 Abs 1 GKG .

**False Positives:**

- `GKG` — partial — pred is substring of gold: `§ 63 Abs 2 S 1 , § 52 Abs 1 und 3 sowie § 47 Abs 1 GKG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 197a Abs 1 S 1 Teils 1 SGG`(NRM)
- `§ 63 Abs 2 S 1 , § 52 Abs 1 und 3 sowie § 47 Abs 1 GKG`(NRM)

**Example 16** (doc_id: `59563`) (sent_id: `59563`)


Ausgelöst durch veränderte Marktbedingungen und verstärkt durch die Finanzkrise 2008 befindet sich der E K-Konzern seit Jahren in wirtschaftlichen Schwierigkeiten .

**False Positives:**

- `K-Konzern` — partial — pred is substring of gold: `E K-Konzern`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `E K-Konzern`(ORG)

</details>

---

## `Bundespatentgericht`

**F1:** 0.022 | **Precision:** 0.346 | **Recall:** 0.011  

**Format:** `regex`  
**Rule ID:** `ad145470`  
**Description:**
Matches 'Bundespatentgericht' and its genitive form.

**Content:**
```
\b(Bundespatentgericht|Bundespatentgerichts|BPatG)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.346 | 0.011 | 0.022 | 26 | 9 | 17 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 9 | 17 | 780 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53862`) (sent_id: `53862`)


Patentansprüche 1 bis 13 vom 24. November 2017 , beim BPatG als 6. Hilfsantrag per Fax eingegangen am 27. November 2017

| Predicted | Gold |
|---|---|
| `BPatG` | `BPatG` |

**Example 1** (doc_id: `54129`) (sent_id: `54129`)


Für die ausstehende Entscheidung bleibt auch die Zuständigkeit des Bundespatentgerichts bestehen .

| Predicted | Gold |
|---|---|
| `Bundespatentgerichts` | `Bundespatentgerichts` |

**Example 2** (doc_id: `54754`) (sent_id: `54754`)


Darin unterscheidet sich der vorliegende Fall grundlegend von demjenigen , der dem Beschluss des Bundespatentgerichts vom 13. November 2014 zu Grunde lag .

| Predicted | Gold |
|---|---|
| `Bundespatentgerichts` | `Bundespatentgerichts` |

**Example 3** (doc_id: `55948`) (sent_id: `55948`)


Insofern gibt es auch im Rahmen von unbestimmten Rechtbegriffen keine Selbstbindung der Markenstellen des Deutschen Patent- und Markenamts und erst recht keine irgendwie geartete Bindung für das Bundespatentgericht .

| Predicted | Gold |
|---|---|
| `Bundespatentgericht` | `Bundespatentgericht` |

**Missed by this rule (FN):**

- `Deutschen Patent- und Markenamts` (ORG)

**Example 4** (doc_id: `56133`) (sent_id: `56133`)


Insoweit besteht in der Rechtsprechung des Bundespatentgerichts zwar weitgehend Übereinstimmung dahingehend , dass die Anmeldung eines Schutzrechts nicht schon allein deswegen mutwillig erscheint , weil der Anmelder – auch unter Inanspruchnahme von Verfahrenskostenhilfe – zahlreiche andere Anmeldungen ohne wirtschaftlichen Erfolg getätigt hat ( vgl. BPatGE 45 , 49 , 51 - Massenanmeldung ; BPatGE 42 , 178 , 179 f. ; BPatGE , 224 , 226 , jeweils m. w. N. ) .

| Predicted | Gold |
|---|---|
| `Bundespatentgerichts` | `Bundespatentgerichts` |

**Missed by this rule (FN):**

- `BPatGE 45 , 49 , 51 - Massenanmeldung` (RS)
- `BPatGE 42 , 178 , 179 f.` (RS)
- `BPatGE , 224 , 226` (RS)

**Example 5** (doc_id: `56648`) (sent_id: `56648`)


Im Übrigen könne ein Anspruch auf Zahlung von Lizenzentgelt , selbst wenn er bestünde , ebenso wie ein Anspruch auf Rechnungslegung , nicht im Verfahren zur Erteilung einer Zwangslizenz vor dem Bundespatentgericht geltend gemacht werden .

| Predicted | Gold |
|---|---|
| `Bundespatentgericht` | `Bundespatentgericht` |

**Example 6** (doc_id: `57241`) (sent_id: `57241`)


Ausgehend von diesen Faktoren drängt sich bei der angemeldeten Bezeichnung das vorstehend dargestellte rein sachbeschreibende Verständnis auf , was demzufolge einem betriebskennzeichnenden Verständnis entgegensteht ( vgl. dazu auch BPatG , Beschluss vom 3. Juni 2016 , Aktenzeichen 27 W ( pat ) 1/16 – Lebe Balance ; die Entscheidung ist über die Homepage des Bundespatentgerichts öffentlich zugänglich ) .

| Predicted | Gold |
|---|---|
| `Bundespatentgerichts` | `Bundespatentgerichts` |

**Missed by this rule (FN):**

- `BPatG , Beschluss vom 3. Juni 2016 , Aktenzeichen 27 W ( pat ) 1/16 – Lebe Balance` (RS)

**Example 7** (doc_id: `59087`) (sent_id: `59087`)


Nach § 73 Abs. 3 Satz 2 PatG hat das Deutsche Patent- und Markenamt dann , wenn es der Beschwerde nicht abhilft , sie dem Bundespatentgericht vorzulegen .

| Predicted | Gold |
|---|---|
| `Bundespatentgericht` | `Bundespatentgericht` |

**Missed by this rule (FN):**

- `§ 73 Abs. 3 Satz 2 PatG` (NRM)
- `Deutsche Patent- und Markenamt` (ORG)

**Example 8** (doc_id: `59981`) (sent_id: `59981`)


Der Anmelder verweist des Weiteren auf zahlreiche Entscheidungen des BGH und des Bundespatentgerichts , in denen vergleichbare Buchstabenkürzel für schutzfähig erachtet worden seien ( z.B. ISET / ISETsolar ( BGH I ZB 2/14 ) , „ ume “ ( 27 W ( pat ) 539/14 ) oder EHD , RSV , bb-nrw , CTL , CJD , RDB , UPW , TCP ) .

| Predicted | Gold |
|---|---|
| `Bundespatentgerichts` | `Bundespatentgerichts` |

**Missed by this rule (FN):**

- `BGH` (ORG)
- `ISET / ISETsolar ( BGH I ZB 2/14 )` (RS)
- `„ ume “ ( 27 W ( pat ) 539/14 )` (RS)
- `EHD` (ORG)
- `RSV` (ORG)
- `bb-nrw` (ORG)
- `CTL` (ORG)
- `CJD` (ORG)
- `RDB` (ORG)
- `UPW` (ORG)
- `TCP` (ORG)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53485`) (sent_id: `53485`)


Soweit die Elemente eines Bildzeichens nur die typischen Merkmale der in Rede stehenden Waren und Dienstleistungen darstellen oder sich in einfachen dekorativen Gestaltungsmitteln erschöpfen , an die sich der Verkehr etwa durch häufige Verwendung gewöhnt hat , wird diesem Zeichen im Allgemeinen wegen seines bloß beschreibenden Inhalts ebenfalls die konkrete Eignung fehlen , die mit ihm gekennzeichneten Waren oder Dienstleistungen von denjenigen anderer Herkunft zu unterscheiden ( vgl. BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband ; GRUR 2005 , 257 , 258 – Bürogebäude ; GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel ; GRUR 2001 , 239f. – Zahnpastastrang ; GRUR 2001 , 734 , 735 – Westie-Kopf ; BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband`(RS)
- `GRUR 2005 , 257 , 258 – Bürogebäude`(RS)
- `GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel`(RS)
- `GRUR 2001 , 239f. – Zahnpastastrang`(RS)
- `GRUR 2001 , 734 , 735 – Westie-Kopf`(RS)
- `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`(RS)

**Example 1** (doc_id: `53854`) (sent_id: `53854`)


Die insoweit verfrüht erhobene Einrede entfaltet auch mit dem Ablauf der maßgeblichen Frist ( am 12. Juni 2014 ) nicht die Rechtswirkung einer zulässigen Einrede ( BPatG GRUR 2005 , 773 , 775 - Blue Bull / RED BULL ; Ströbele / Hacker , MarkenG , 11. Aufl. , § 43 Rn. 29 m. w. N. ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG GRUR 2005 , 773 , 775 - Blue Bull / RED BULL`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG GRUR 2005 , 773 , 775 - Blue Bull / RED BULL`(RS)
- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 43 Rn. 29`(LIT)

**Example 2** (doc_id: `53881`) (sent_id: `53881`)


Ob dies der Fall ist , richtet sich nach den Umständen des Einzelfalls , bei denen darauf abzustellen ist , wie das Hoheitszeichen im Rahmen der Designgestaltung konkret verwendet ist ( vgl. BPatG GRUR 2002 , 337 - Schlüsselanhänger ; Eichmann / von Falckenstein / Kühne , a. a. O. , § 3 Rdn. 26 ff. ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG GRUR 2002 , 337 - Schlüsselanhänger`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG GRUR 2002 , 337 - Schlüsselanhänger`(RS)
- `Eichmann / von Falckenstein / Kühne , a. a. O. , § 3 Rdn. 26 ff.`(LIT)

**Example 3** (doc_id: `53963`) (sent_id: `53963`)


Der Gesamteindruck aber kann auch bei Übernahme der geschützten „ Schnittmenge “ durch Hinzufügung weiterer Merkmale im Einzelfall erheblich verändert werden ( vgl. OLG Köln , 6 U 128/09 v. 8. Januar 2010 ; veröffentlicht in juris ; Klawitter in : Festschrift 50 Jahre Bundespatentgericht , S. 1071 , 1079 ) .

**False Positives:**

- `Bundespatentgericht` — partial — pred is substring of gold: `Klawitter in : Festschrift 50 Jahre Bundespatentgericht , S. 1071 , 1079`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `OLG Köln , 6 U 128/09 v. 8. Januar 2010 ; veröffentlicht in juris`(RS)
- `Klawitter in : Festschrift 50 Jahre Bundespatentgericht , S. 1071 , 1079`(LIT)

**Example 4** (doc_id: `55119`) (sent_id: `55119`)


Besteht dieses nur aus der Darstellung des Gegenstands , auf den sich die Dienstleistungen unmittelbar beziehen , stellt es nur typische Merkmale der in Rede stehenden Dienstleistungen dar oder erschöpft sich die bildliche Darstellung in einfachen dekorativen Gestaltungsmitteln , an die der Verkehr sich etwa durch häufige Verwendung gewöhnt hat , wird diesem Zeichen im allgemeinen wegen seines bloß beschreibenden Inhalts die konkrete Eignung fehlen , die mit ihm gekennzeichneten Waren oder Dienstleistungen von denjenigen anderer Herkunft zu unterscheiden ( vgl. BGH , GRUR 2005 , 257 , 258 – Bürogebäude ; GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel ; GRUR 2001 , 239 f. – Zahnpastastrang ; GRUR 2001 , 734 , 735 – Westie-Kopf ; BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , GRUR 2005 , 257 , 258 – Bürogebäude`(RS)
- `GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel`(RS)
- `GRUR 2001 , 239 f. – Zahnpastastrang`(RS)
- `GRUR 2001 , 734 , 735 – Westie-Kopf`(RS)
- `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`(RS)

**Example 5** (doc_id: `55186`) (sent_id: `55186`)


Eine Ähnlichkeit zwischen Handelsdienstleistungen , insbesondere hiervon erfassten Einzelhandelsdienstleistungen , und den auf sie bezogenen Waren ist anzunehmen , wenn die Dienstleistungen sich auf die entsprechenden Waren beziehen und die angesprochenen Verkehrskreise aufgrund dieses Verhältnisses annehmen , die Waren und Dienstleistungen stammten aus denselben Unternehmen ( vgl. BGH GRUR 2014 , 378 , Rdnr. 39 – Otto CAP ; BPatG GRUR-RR 2013 , 430 , 432 – Konzume / Konsum ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG GRUR-RR 2013 , 430 , 432 – Konzume / Konsum`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH GRUR 2014 , 378 , Rdnr. 39 – Otto CAP`(RS)
- `BPatG GRUR-RR 2013 , 430 , 432 – Konzume / Konsum`(RS)

**Example 6** (doc_id: `56236`) (sent_id: `56236`)


Zwischen der technischen Dienstleistung und der Contentvermittlung besteht ein so enger Bezug , dass das entsprechende Verkehrsverständnis zwischen Technik und Inhalt insoweit nicht mehr trennt ( vgl. BGH GRUR 2014 , 1204 Rn. 22 – TOOOR ! ; BPatG , Beschluss vom 11. 05. 2015 , 26 W ( pat ) 72/14 – Shopping Compass ; Beschluss vom 22. 01. 2015 , 29 W ( pat ) 525/13 – The European ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG , Beschluss vom 11. 05. 2015 , 26 W ( pat ) 72/14 – Shopping Compass`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH GRUR 2014 , 1204 Rn. 22 – TOOOR !`(RS)
- `BPatG , Beschluss vom 11. 05. 2015 , 26 W ( pat ) 72/14 – Shopping Compass`(RS)
- `Beschluss vom 22. 01. 2015 , 29 W ( pat ) 525/13 – The European`(RS)

**Example 7** (doc_id: `56412`) (sent_id: `56412`)


Soweit die Anmelderin in Klasse 2 ferner „ Naturharze im Rohzustand “ beansprucht , stellen diese im Bereich von Lacken und ( Öl- ) Farben einen üblichen Inhaltsstoff dar , der auch als Zusatz im Malereibedarf in Betracht kommt ( vgl. BPatG PAVIS PROMA , Beschluss vom 23. April 2013 , 25 W ( pat ) 62/12 - Macchiato ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG PAVIS PROMA , Beschluss vom 23. April 2013 , 25 W ( pat ) 62/12 - Macchiato`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG PAVIS PROMA , Beschluss vom 23. April 2013 , 25 W ( pat ) 62/12 - Macchiato`(RS)

**Example 8** (doc_id: `57241`) (sent_id: `57241`)


Ausgehend von diesen Faktoren drängt sich bei der angemeldeten Bezeichnung das vorstehend dargestellte rein sachbeschreibende Verständnis auf , was demzufolge einem betriebskennzeichnenden Verständnis entgegensteht ( vgl. dazu auch BPatG , Beschluss vom 3. Juni 2016 , Aktenzeichen 27 W ( pat ) 1/16 – Lebe Balance ; die Entscheidung ist über die Homepage des Bundespatentgerichts öffentlich zugänglich ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG , Beschluss vom 3. Juni 2016 , Aktenzeichen 27 W ( pat ) 1/16 – Lebe Balance`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG , Beschluss vom 3. Juni 2016 , Aktenzeichen 27 W ( pat ) 1/16 – Lebe Balance`(RS)
- `Bundespatentgerichts`(ORG)

**Example 9** (doc_id: `57632`) (sent_id: `57632`)


Von daher sind unter den „ Angaben zum Verwendungszweck , der die Kosten umfasst “ diejenigen Informationen zu verstehen , die das Patentamt in die Lage versetzen , die Höhe der zu zahlenden Gebühr festzustellen , diese einem konkreten Verfahren zuzuordnen und auf Basis eines entsprechenden SEPA-Basislastschriftmandats einzuziehen ( vgl. BPatG Mitt. 2016 , 192 , 195 , juris Tz. 21 - babygro ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG Mitt. 2016 , 192 , 195 , juris Tz. 21 - babygro`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG Mitt. 2016 , 192 , 195 , juris Tz. 21 - babygro`(RS)

**Example 10** (doc_id: `57712`) (sent_id: `57712`)


Die Markenbestandteile werden in Übereinstimmung mit ihrem Sinngehalt verwendet und bilden auch in der Gesamtheit keinen neuen , über die bloße Kombination hinausgehenden Begriff ( BPatG , a. a. O. , 30 W ( pat ) 530/13 - WohlFühlFarben ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG , a. a. O. , 30 W ( pat ) 530/13 - WohlFühlFarben`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG , a. a. O. , 30 W ( pat ) 530/13 - WohlFühlFarben`(RS)

**Example 11** (doc_id: `57805`) (sent_id: `57805`)


Einem Zeichen , das den Namen einer berühmten Persönlichkeit aufnimmt , fehlt nur dann die erforderliche Unterscheidungskraft , wenn die angesprochenen Verkehrskreise in dem Namen lediglich eine sachbezogene oder werbewirksame Aussage sehen ( vgl. BPatG GRUR 2008 , 518 , 521 - Karl May ; GRUR 2014 , 79 , 80 ff. - Mark Twain ; Beschluss vom 1. Juni 2017 , 25 W ( pat ) 4/17 - einstein concept / Einstein ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG GRUR 2008 , 518 , 521 - Karl May`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG GRUR 2008 , 518 , 521 - Karl May`(RS)
- `GRUR 2014 , 79 , 80 ff. - Mark Twain`(RS)
- `Beschluss vom 1. Juni 2017 , 25 W ( pat ) 4/17 - einstein concept / Einstein`(RS)

**Example 12** (doc_id: `58268`) (sent_id: `58268`)


Ferner ist es als bloßes Gestaltungsmittel , z.B. als sog. „ Eyecatcher “ werbeüblich ( vgl. BPatG , Beschluss vom 27. 01. 2009 , 27 W ( pat ) 46/09 – LIVE ! SPEAKER ; Beschluss vom 13. 09. 2006 , 29 W ( pat ) 68/04 – REZEPTE pur EINFACH ! KOCHEN ) wie auch als Ersetzung des Buchstaben „ I / i “ ( vgl. z.B. Werbeaussage „ W ! R S ! ND DABE ! “ unter www.bw-stiftung.de) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG , Beschluss vom 27. 01. 2009 , 27 W ( pat ) 46/09 – LIVE ! SPEAKER`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG , Beschluss vom 27. 01. 2009 , 27 W ( pat ) 46/09 – LIVE ! SPEAKER`(RS)
- `Beschluss vom 13. 09. 2006 , 29 W ( pat ) 68/04 – REZEPTE pur EINFACH ! KOCHEN`(RS)

**Example 13** (doc_id: `58693`) (sent_id: `58693`)


1.7.3 Im Hinblick auf die Nacharbeitung D21A hat die Patentinhaberin geltend gemacht , diese würde die nach dem BGH-Urteil „ Metazachlor “ ( BGH Urteil vom 15. März 2011 – X ZR 58/08 – , juris ) und dem Urteil 3 Ni 5/15 des BPatG ( BPatG Urteil vom 11. Oktober 2016 – 3 Ni 5/15 ( EP ) – , juris ) aufgestellten strengen Maßstäbe nicht erfüllen , die bei der Feststellung einer implizierten Offenbarung einer Druckschrift des Standes der Technik durch Nacharbeitung angelegt werden müssten .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `Urteil 3 Ni 5/15 des BPatG ( BPatG Urteil vom 11. Oktober 2016 – 3 Ni 5/15 ( EP ) – , juris )`
- `BPatG` — partial — pred is substring of gold: `Urteil 3 Ni 5/15 des BPatG ( BPatG Urteil vom 11. Oktober 2016 – 3 Ni 5/15 ( EP ) – , juris )`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH-Urteil „ Metazachlor “ ( BGH Urteil vom 15. März 2011 – X ZR 58/08 – , juris )`(RS)
- `Urteil 3 Ni 5/15 des BPatG ( BPatG Urteil vom 11. Oktober 2016 – 3 Ni 5/15 ( EP ) – , juris )`(RS)

**Example 14** (doc_id: `59283`) (sent_id: `59283`)


Hergeleitet aus dem Bereich der Farbtherapie / Farbpsychologie wird Farben / Farbtönen eine Wirkung auf die menschliche Psyche und den Organismus zugeschrieben ( vgl. BPatG , a. a. O. , 30 W ( pat ) 530/13 – WohlFühlFarben ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG , a. a. O. , 30 W ( pat ) 530/13 – WohlFühlFarben`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG , a. a. O. , 30 W ( pat ) 530/13 – WohlFühlFarben`(RS)

**Example 15** (doc_id: `59751`) (sent_id: `59751`)


Im Hinblick auf die massenhaft beim Patentamt eingehenden und zu bearbeitenden Zahlungen sowie aus Gründen der Rechtssicherheit ist zu beachten , dass jede Gebührenentrichtung beim Patentamt so klar und vollständig sein muss , dass die verfahrens- und betragsmäßige Erfassung und Zuordnung ohne verzögernde Ermittlungen gewährleistet und der Geldbetrag zu dem in § 2 PatKostZV bestimmten Zahlungstag zu einem konkreten Vorgang sicher vereinnahmt werden kann ( vgl. BPatGE 48 , 163 , 167 - Unbezifferter Abbuchungsauftrag ; BPatG Mitt. 2016 , 192 , 195 - babygro ) .

**False Positives:**

- `BPatG` — similar text (different position): `BPatGE 48 , 163 , 167 - Unbezifferter Abbuchungsauftrag`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 2 PatKostZV`(NRM)
- `BPatGE 48 , 163 , 167 - Unbezifferter Abbuchungsauftrag`(RS)
- `BPatG Mitt. 2016 , 192 , 195 - babygro`(RS)

</details>

---

## `Single Letter Companies with Punctuation`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `c75aa959`  
**Description:**
Matches single letter or hyphenated codes with punctuation followed by GmbH/AG/KB.

**Content:**
```
\b([A-Z]\s*\.\s*[A-Z]\s*\.\s*[A-Z]?|C-\s*B\.\s*V\.|S\s*AG)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 16 | 0 | 16 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 0 | 16 | 542 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `56041`) (sent_id: `56041`)


Das FA hat unter Berufung auf § 123 i. V. m. § 60 Abs. 3 der Finanzgerichtsordnung ( FGO ) , § 174 Abs. 4 , 5 der Abgabenordnung ( AO ) die Beiladung der Gesellschafter A. X. und B. X. für die GbR angeregt .

**False Positives:**

- `A. X. ` — partial — gold is substring of pred: `A. X.`
- `B. X. ` — partial — gold is substring of pred: `B. X.`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 123 i. V. m. § 60 Abs. 3 der Finanzgerichtsordnung`(NRM)
- `FGO`(NRM)
- `§ 174 Abs. 4 , 5 der Abgabenordnung`(NRM)
- `AO`(NRM)
- `A. X.`(ORG)
- `B. X.`(ORG)

**Example 1** (doc_id: `56134`) (sent_id: `56134`)


Da die A-GbR aufgrund ihrer Geschäftsbeziehung der C- B. V. näher stehe als das FA , hätte es dem Kläger oblegen , die Angaben über die wirtschaftlichen Verhältnisse der C- B. V. weiter zu konkretisieren .

**False Positives:**

- `B. V. ` — positional overlap with gold: `C- B. V.`
- `B. V. ` — positional overlap with gold: `C- B. V.`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `A-GbR`(ORG)
- `C- B. V.`(ORG)
- `C- B. V.`(ORG)

**Example 2** (doc_id: `56444`) (sent_id: `56444`)


So wird ausdrücklich darauf verwiesen , dass auf der Anlage K 1. Ausdruck einer Mail von D. an D. B. vom 30. Januar 2011 ) handschriftlich " Rechnungsanschrift Help Food z. o. o. D. B. ( es folgt die postalische Anschrift der Help Food ) " vermerkt ist , auf Seite 2 der Anlage K 3 ( Präsentationsunterlage mit dem Copyright von D. und W. ) unter " Unsere Kontraktbedingungen " ein " Exklusiver Kontrakt für 2 Jahre mit Help Food " und eine " Haushaltsverfügung durch Help Food ... bis zum Ende 2011 Startphase " erwähnt werden , auf Seite 2 der Anlage K 5 ( mit dem Logo der Klägerin versehenes Protokoll eines Treffens der Beteiligten am 26. August 2011 ) von einem " Vorschlag zum Vertrag zwischen Help Food , M. D. und P. W. " die Rede ist , die Anlage K 9 ( von S. unterzeichnetes Schreiben vom 29. Dezember 2011 ) als Absender die Help Food ausweist und die Anlage K 50 ( Ausdruck einer Mail der Zeugin F. an D. und W. vom 14. September 2011 ) die Absenderadresse " m. @helpfood . eu " trägt .

**False Positives:**

- `D. B. ` — similar text (different position): `D.`
- `M. D. ` — similar text (different position): `D.`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `D.`(PER)
- `D. B.`(PER)
- `Help Food`(ORG)
- `Help Food`(ORG)
- `D.`(PER)
- `W.`(PER)
- `Help Food`(ORG)
- `Help Food`(ORG)
- `Help Food`(ORG)
- `M. D.`(PER)
- `P. W.`(PER)
- `S.`(PER)
- `Help Food`(ORG)
- `F.`(PER)
- `D.`(PER)
- `W.`(PER)

**Example 3** (doc_id: `56479`) (sent_id: `56479`)


I. Nach den Feststellungen des Landgerichts kamen der Angeklagte und sein in der Türkei lebender gesondert verfolgter Bruder E. K. spätestens zu Beginn des Jahres 2015 überein , in arbeitsteiligem Zusammenwirken in der Türkei hergestellte bzw. erworbene Kleidungsstücke , die mit Schriftzügen und Labels verschiedener Markenhersteller versehen waren , unter Verletzung geschützter Gemeinschafts- bzw. Unionsmarken in Deutschland zu verkaufen , obwohl ihnen bewusst war , dass sie nicht über die für deren Verwendung erforderliche Zustimmung der Markenrechtsinhaber verfügten .

**False Positives:**

- `E. K. ` — partial — gold is substring of pred: `E. K.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Türkei`(LOC)
- `E. K.`(PER)
- `Türkei`(LOC)
- `Deutschland`(LOC)

**Example 4** (doc_id: `56849`) (sent_id: `56849`)


Diese Tatsache liegt darin , dass die C- B. V. zwischen dem Jahr 2003 bis zu ihrer Auflösung im Jahr 2006 durchgängig , und damit auch zum maßgeblichen Bilanzstichtag am 31. Dezember 2004 , über kein hinreichendes Vermögen verfügte , um das Schuldanerkenntnis über 2 Mio. € ganz oder in Teilbeträgen zu tilgen .

**False Positives:**

- `B. V. ` — positional overlap with gold: `C- B. V.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C- B. V.`(ORG)

**Example 5** (doc_id: `57157`) (sent_id: `57157`)


Die C- B. V. wurde bereits im Jahr 2006 aufgelöst .

**False Positives:**

- `B. V. ` — positional overlap with gold: `C- B. V.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C- B. V.`(ORG)

**Example 6** (doc_id: `57273`) (sent_id: `57273`)


Der Kläger betrieb vormals eine Anwaltssozietät mit Rechtsanwalt C. B. in F. .

**False Positives:**

- `C. B. ` — partial — gold is substring of pred: `C. B.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C. B.`(PER)
- `F.`(LOC)

**Example 7** (doc_id: `58560`) (sent_id: `58560`)


H. N. hatte sich , wie zuvor verabredet , teilweise maskiert und ging in die Halle 4a der Spielhalle .

**False Positives:**

- `H. N. ` — partial — gold is substring of pred: `H. N.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `H. N.`(PER)

**Example 8** (doc_id: `59356`) (sent_id: `59356`)


Über die sofortige Beschwerde der Nebenklägerin K. T. gegen die im Urteil des Landgerichts Frankfurt am Main vom 15. August 2017 getroffene Kostenentscheidung hat das Oberlandesgericht Frankfurt am Main zu entscheiden .

**False Positives:**

- `K. T. ` — partial — gold is substring of pred: `K. T.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K. T.`(PER)
- `Landgerichts Frankfurt am Main`(ORG)
- `Oberlandesgericht Frankfurt am Main`(ORG)

**Example 9** (doc_id: `59650`) (sent_id: `59650`)


D20 I. C. Madsen et al. , “ Description and survey of methodologies for the determination of amorphous content via X-ray powder diffraction ” , Z. Kristallgr . 2011 , 226 , Seiten 944 bis 955

**False Positives:**

- `I. C. ` — partial — pred is substring of gold: `I. C. Madsen`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `I. C. Madsen`(PER)

**Example 10** (doc_id: `59658`) (sent_id: `59658`)


A. I. 1. Der Beschwerdeführer ist auf Grundlage des Urteils des Landgerichts Lübeck vom 7. Oktober 2014 gemäß § 63 StGB wegen Mordes in einem psychiatrischen Krankenhaus der AMEOS Krankenhausgesellschaft Holstein mbH untergebracht , nachdem er im schuldunfähigen Zustand auf Grund einer wahnhaften Störung im Januar 2014 seine vierjährige Tochter und seinen sechs Jahre alten Sohn getötet hatte .

**False Positives:**

- `A. I. ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Landgerichts Lübeck`(ORG)
- `§ 63 StGB`(NRM)
- `AMEOS Krankenhausgesellschaft Holstein mbH`(ORG)

**Example 11** (doc_id: `59786`) (sent_id: `59786`)


Es sei jedoch nicht erkennbar , dass der Kläger sich bemüht habe , aus der Sphäre der C- B. V. substantiierte Angaben über ihre wirtschaftlichen Verhältnisse zu erhalten .

**False Positives:**

- `B. V. ` — positional overlap with gold: `C- B. V.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C- B. V.`(ORG)

**Example 12** (doc_id: `59897`) (sent_id: `59897`)


So legt der Kläger - soweit erkennbar unwidersprochen durch das FA - dar , dass das mit der A-GbR vereinbarte Projekt eines Ferienresorts die einzige geschäftliche Tätigkeit der C- B. V. dargestellt habe .

**False Positives:**

- `B. V. ` — positional overlap with gold: `C- B. V.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `A-GbR`(ORG)
- `C- B. V.`(ORG)

</details>

---

## `European Union`

**F1:** 0.005 | **Precision:** 0.111 | **Recall:** 0.002  

**Format:** `regex`  
**Rule ID:** `5d8c53d3`  
**Description:**
Matches 'Europäischen Union' in various cases.

**Content:**
```
\b(Europ\u00e4ischen\s*Union|Europ\u00e4ische\s*Union|EU|Gerichtshof\s+der\s+Europ\u00e4ischen\s+Union|Amt\s+der\s+Europ\u00e4ischen\s+Union\s+f\u00fcr\s+geistiges\s+Eigentum|EUGH)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.111 | 0.002 | 0.005 | 18 | 2 | 16 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 2 | 16 | 736 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `55904`) (sent_id: `55904`)


bb ) Der Gerichtshof der Europäischen Union hat entschieden , § 4 Nr. 2 der Rahmenvereinbarung über Teilzeitarbeit sei dahin auszulegen , dass er einer nationalen Bestimmung entgegensteht , nach der bei einer Änderung des Beschäftigungsausmaßes eines Arbeitnehmers das Ausmaß des noch nicht verbrauchten Erholungsurlaubs in der Weise angepasst wird , dass der von einem Arbeitnehmer , der von einer Vollzeit- zu einer Teilzeitbeschäftigung übergeht , in der Zeit der Vollzeitbeschäftigung erworbene Anspruch auf bezahlten Jahresurlaub , dessen Ausübung dem Arbeitnehmer während dieser Zeit nicht möglich war , reduziert wird oder der Arbeitnehmer diesen Urlaub nur mehr mit einem geringeren Urlaubsentgelt verbrauchen kann ( EuGH 22. April 2010 - C- 486/08 - [ Zentralbetriebsrat der Landeskrankenhäuser Tirols ] Rn. 35 ) .

| Predicted | Gold |
|---|---|
| `Gerichtshof der Europäischen Union` | `Gerichtshof der Europäischen Union` |

**Missed by this rule (FN):**

- `§ 4 Nr. 2 der Rahmenvereinbarung über Teilzeitarbeit` (NRM)
- `EuGH 22. April 2010 - C- 486/08 - [ Zentralbetriebsrat der Landeskrankenhäuser Tirols ] Rn. 35` (RS)

**Example 1** (doc_id: `59798`) (sent_id: `59798`)


Insoweit konnte nicht ausgeschlossen werden , dass die Einfuhr der Textilien in die EU in dem betreffenden Jahr jeweils durch eine einheitliche Handlung erfolgte .

| Predicted | Gold |
|---|---|
| `EU` | `EU` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53938`) (sent_id: `53938`)


In dem besonderen Fall der Sanktionierung von Verstößen gegen die Verordnung [ … ] wurden jedoch die straf- oder bußgeldbewehrten Vorschriften der Verordnung [ … ] durch das Inkrafttreten der Sanktionsvorschriften vor dem Anwendungszeitpunkt der bewehrten EU-Verordnung bereits ab dem 2. Juli 2016 in Deutschland für anwendbar erklärt .

**False Positives:**

- `EU` — similar text (different position): `Deutschland`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Deutschland`(LOC)

**Example 1** (doc_id: `54192`) (sent_id: `54192`)


- 100 mg Granulat zur Zubereitung oral einzunehmender Suspensionen unter der Nummer EU / 1 / 07 / 436 / 005 .

**False Positives:**

- `EU` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `54233`) (sent_id: `54233`)


Weder im deutschen AsylG noch in einem anderen deutschen Regelungswerk gebe es eine Norm , in der stehe oder aus der abgeleitet werden könne , die Gewährung subsidiären Schutzes in einem anderen EU-Mitgliedstaat stünde der Auslieferung der betroffenen Person durch die Bundesrepublik Deutschland entgegen .

**False Positives:**

- `EU` — similar text (different position): `Bundesrepublik Deutschland`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `AsylG`(NRM)
- `Bundesrepublik Deutschland`(LOC)

**Example 3** (doc_id: `54786`) (sent_id: `54786`)


Nichtannahmebeschluss : Gewährung subsidiären Schutzes in anderem EU-Mitgliedsstaat stellt gewichtiges Indiz für Vorliegen eines Auslieferungshindernisses dar - hier : Auslieferung eines in Belgien als schutzberechtigt anerkannten Weißrussen an Weißrussland zur Strafverfolgung - mangelnde Substantiierung wegen Nichtvorlage entscheidungserheblicher Unterlagen - zudem Entkräftung der Gefahr einer Verhängung der Todesstrafe bzw menschenunwürdiger Haftbedingungen durch verbindliche Zusicherungen Weißrusslands

**False Positives:**

- `EU` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Belgien`(LOC)
- `Weißrussland`(LOC)
- `Weißrusslands`(LOC)

**Example 4** (doc_id: `55373`) (sent_id: `55373`)


Nach den dargestellten Grundsätzen fehlt der Europäischen Union die Rechtsmacht , einer Regelung des nationalen Rechts die Wirksamkeit für Sachverhalte zu nehmen , welche keinen hinreichenden Bezug zu anderen EU-Mitgliedstaaten aufweisen und deshalb außerhalb der Regelungskompetenz der Europäischen Union liegen .

**False Positives:**

- `EU` — similar text (different position): `Europäischen Union`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europäischen Union`(ORG)
- `Europäischen Union`(ORG)

**Example 5** (doc_id: `56084`) (sent_id: `56084`)


- 25 mg Kautabletten , zugelassen unter der Nummer EU / 1 / 07 / 436/003 ,

**False Positives:**

- `EU` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 6** (doc_id: `57187`) (sent_id: `57187`)


Anlass für eine erneute Vorlage nach Art. 267 AEUV besteht nicht , zumal der EuGH seine Rechtsauffassung in dem Urteil vom 16. April 2015 - C- 446/12 [ ECLI : EU :C : 2015 : 238 ] , Willems - Rn. 46 bestätigt hat .

**False Positives:**

- `EU` — similar text (different position): `Art. 267 AEUV`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 267 AEUV`(NRM)
- `EuGH`(ORG)
- `Urteil vom 16. April 2015 - C- 446/12 [ ECLI : EU :C : 2015 : 238 ] , Willems - Rn. 46`(RS)

**Example 7** (doc_id: `57389`) (sent_id: `57389`)


Es bestünden erhebliche Zweifel , ob Angehörige dieser Gruppe entsprechend den Anforderungen des Art. 28 Abs. 2 der Richtlinie 2013 / 32 / EU des Europäischen Parlaments und des Rates vom 26. Juni 2013 zu gemeinsamen Verfahren für die Zuerkennung und Aberkennung des internationalen Schutzes ( ABl. L 180 S. 60 ) behandelt würden .

**False Positives:**

- `EU` — partial — pred is substring of gold: `Art. 28 Abs. 2 der Richtlinie 2013 / 32 / EU des Europäischen Parlaments und des Rates vom 26. Juni 2013 zu gemeinsamen Verfahren für die Zuerkennung und Aberkennung des internationalen Schutzes ( ABl. L 180 S. 60 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 28 Abs. 2 der Richtlinie 2013 / 32 / EU des Europäischen Parlaments und des Rates vom 26. Juni 2013 zu gemeinsamen Verfahren für die Zuerkennung und Aberkennung des internationalen Schutzes ( ABl. L 180 S. 60 )`(NRM)

**Example 8** (doc_id: `57647`) (sent_id: `57647`)


„ festzustellen , dass für die Durchführung eines elektronischen Abgleiches der Mitarbeiterdaten mit den sog. Sanktionslisten aus den EU-Verordnungen VO ( EG ) 2580/2001 und VO ( EG ) 881/2002 durch die Arbeitgeberin das Mitbestimmungsrecht a ) des Betriebsrats und b ) des Gesamtbetriebsrats besteht “ ,

**False Positives:**

- `EU` — partial — pred is substring of gold: `EU-Verordnungen VO ( EG ) 2580/2001`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EU-Verordnungen VO ( EG ) 2580/2001`(NRM)
- `VO ( EG ) 881/2002`(NRM)

**Example 9** (doc_id: `57786`) (sent_id: `57786`)


Das FG hat aber zu Recht darauf hingewiesen , dass nach der Rechtsprechung des Gerichtshofs der Europäischen Union ( EuGH ) bei der Auslegung des Art. 10 der VO Nr. 574/72 die " allgemeinen Vorschriften " des in Titel I der VO Nr. 1408/71 und damit Art. 12 der VO Nr. 1408/71 zu beachten sind ( EuGH-Urteil Wiering vom 8. Mai 2014 C - 347/12 , EU : C : 2014 : 300 , Rz 54 ff. , Zeitschrift für europäisches Sozial- und Arbeitsrecht - ZESAR - 2015 , 113 ) .

**False Positives:**

- `EU` — similar text (different position): `Gerichtshofs der Europäischen Union`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Gerichtshofs der Europäischen Union`(ORG)
- `EuGH`(ORG)
- `Art. 10 der VO Nr. 574/72`(NRM)
- `Titel I der VO Nr. 1408/71`(NRM)
- `Art. 12 der VO Nr. 1408/71`(NRM)
- `EuGH-Urteil Wiering vom 8. Mai 2014 C - 347/12 , EU : C : 2014 : 300 , Rz 54 ff.`(RS)
- `Zeitschrift für europäisches Sozial- und Arbeitsrecht - ZESAR - 2015 , 113`(LIT)

**Example 10** (doc_id: `58343`) (sent_id: `58343`)


Unter den hier vorliegenden Voraussetzungen des Art. 267 Abs. 3 AEUV ( vergleiche dazu Beschlüsse des Bundesverfassungsgerichts - BVerfG - vom 6. September 2016 1 BvR 1305/13 , Neue Zeitschrift für Verwaltungsrecht - NVwZ - 2017 , 53 , Rz 7 , 8 ; vom 6. Oktober 2017 2 BvR 987/16 , Neue Juristische Wochenschrift - NJW - 2018 , 606 , Rz 4 , 5 ; jeweils mit weiteren Nachweisen ) sind die nationalen Gerichte von Amts wegen gehalten , den EuGH anzurufen ( vergleiche BVerfG-Beschlüsse in NVwZ 2017 , 53 , Rz 6 ; in NJW 2018 , 606 , Rz 3 ; ferner EuGH-Urteil CILFIT vom 6. Oktober 1982 C - 283/81 , EU : C : 1982 : 335 , NJW 1983 , 1257 , Rz 21 ; jeweils mit weiteren Nachweisen ) .

**False Positives:**

- `EU` — similar text (different position): `Art. 267 Abs. 3 AEUV`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 267 Abs. 3 AEUV`(NRM)
- `Beschlüsse des Bundesverfassungsgerichts - BVerfG - vom 6. September 2016 1 BvR 1305/13 , Neue Zeitschrift für Verwaltungsrecht - NVwZ - 2017 , 53 , Rz 7 , 8`(RS)
- `vom 6. Oktober 2017 2 BvR 987/16 , Neue Juristische Wochenschrift - NJW - 2018 , 606 , Rz 4 , 5`(RS)
- `EuGH`(ORG)
- `BVerfG-Beschlüsse in NVwZ 2017 , 53 , Rz 6`(LIT)
- `NJW 2018 , 606 , Rz 3`(RS)
- `EuGH-Urteil CILFIT vom 6. Oktober 1982 C - 283/81 , EU : C : 1982 : 335 , NJW 1983 , 1257 , Rz 21`(RS)

**Example 11** (doc_id: `58501`) (sent_id: `58501`)


Das Oberverwaltungsgericht hat angenommen , dass eine Prüfung des § 4 Abs. 3 und 4 PassG auf die Vereinbarkeit mit dem Grundgesetz wegen der unionsrechtlichen Determinierung nicht stattfinden kann und die Vereinbarkeit der zwingenden unionsrechtlichen Vorgaben des Art. 1 Abs. 1 und 2 der Verordnung ( EG ) 2252/2004 mit der Charta der Grundrechte der EU und anderem höherrangigen Unionsrecht durch die Rechtsprechung des EuGH mit bindender Wirkung geklärt ist .

**False Positives:**

- `EU` — partial — pred is substring of gold: `Charta der Grundrechte der EU`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 4 Abs. 3 und 4 PassG`(NRM)
- `Grundgesetz`(NRM)
- `Art. 1 Abs. 1 und 2 der Verordnung ( EG ) 2252/2004`(NRM)
- `Charta der Grundrechte der EU`(NRM)
- `EuGH`(ORG)

**Example 12** (doc_id: `58740`) (sent_id: `58740`)


Das Berufungsgericht wird sich bei seiner neuerlichen , durch diesen Beschluss nicht im Ergebnis vorgeprägten Entscheidung auch mit den - wenngleich in anderem Zusammenhang ergangenen - Entscheidungen des Gerichtshofes der Europäischen Union vom 25. Januar 2018 ( - C- 360/16 [ ECLI : EU :C : 2018 : 35 ] , Hasan - Rn. 30 f. , 40 ) und des Bundesverfassungsgerichts vom 21. April 2016 ( - 2 BvR 273/16 - NVwZ 2016 , 1242 Rn. 11 ) zur Frage des maßgeblichen Zeitpunkts für die Beurteilung des Vorliegens systemischer Schwachstellen auseinanderzusetzen haben .

**False Positives:**

- `EU` — partial — pred is substring of gold: `Entscheidungen des Gerichtshofes der Europäischen Union vom 25. Januar 2018 ( - C- 360/16 [ ECLI : EU :C : 2018 : 35 ] , Hasan - Rn. 30 f. , 40 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Entscheidungen des Gerichtshofes der Europäischen Union vom 25. Januar 2018 ( - C- 360/16 [ ECLI : EU :C : 2018 : 35 ] , Hasan - Rn. 30 f. , 40 )`(RS)
- `Bundesverfassungsgerichts vom 21. April 2016 ( - 2 BvR 273/16 - NVwZ 2016 , 1242 Rn. 11 )`(RS)

**Example 13** (doc_id: `58845`) (sent_id: `58845`)


Die Klägerin hat weder ihre Ausbildung im EU-Ausland absolviert noch war sie in einem dieser Länder beruflich tätig .

**False Positives:**

- `EU` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 14** (doc_id: `58902`) (sent_id: `58902`)


Insbesondere darf das nationale Gericht trotz einer abweichenden Entscheidung der Vorinstanz davon absehen , dem EuGH eine vor ihm aufgeworfene Frage nach der Auslegung des Unionsrechts vorzulegen ( vgl. EuGH-Urteil Ferreira da Silva e Brito u. a. , EU : C : 2015 : 565 , EuZW 2016 , 111 , Rz 40 bis 42 , m. w. N. ) .

**False Positives:**

- `EU` — similar text (different position): `EuGH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH`(ORG)
- `EuGH-Urteil Ferreira da Silva e Brito u. a. , EU : C : 2015 : 565 , EuZW 2016 , 111 , Rz 40 bis 42`(RS)

**Example 15** (doc_id: `59082`) (sent_id: `59082`)


Der Gesetzgeber hat § 2 BetrAVG aF durch Art. 1 Nr. 2 des Gesetzes zur Umsetzung der EU-Mobilitäts-Richtlinie vom 21. Dezember 2015 ( BGBl. I S. 2553 ) teilweise neu gefasst , ohne dass sich insoweit Änderungen zu der vorher geltenden Rechtslage ergeben sollten ( vgl. BT-Drs. 18/6283 S. 13 ) .

**False Positives:**

- `EU` — partial — pred is substring of gold: `Art. 1 Nr. 2 des Gesetzes zur Umsetzung der EU-Mobilitäts-Richtlinie vom 21. Dezember 2015 ( BGBl. I S. 2553 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 2 BetrAVG aF`(NRM)
- `Art. 1 Nr. 2 des Gesetzes zur Umsetzung der EU-Mobilitäts-Richtlinie vom 21. Dezember 2015 ( BGBl. I S. 2553 )`(NRM)
- `BT-Drs. 18/6283 S. 13`(LIT)

</details>

---

## `Government Ministries and Bodies`

**F1:** 0.061 | **Precision:** 0.634 | **Recall:** 0.032  

**Format:** `regex`  
**Rule ID:** `80b9799f`  
**Description:**
Matches government bodies including optional department codes.

**Content:**
```
\b(Bundesministeriums?\s+der\s+Verteidigung(?:\s+-\s+[A-Z0-9\s-]+)?|Bundesministeriums?\s+der\s+Justiz|Landesregierung\s+von\s*Nordrhein-Westfalen|saarl\u00e4ndischen\s*Landesregierung|Ausw\u00e4rtigen\s*Amtes|Bundesrat|Staatskasse|Justizministerium\s*des\s*Landes\s*Nordrhein-Westfalen|Bayerische\s*Staatsministerium|Bundesministeriums?\s*der\s*Verteidigung|Bundesministerium\s*f\u00fcr\s*Verbraucherschutz\s*,\s*Ern\u00e4hrung\s*und\s*Landwirtschaft|Bundesministerium\s*f\u00fcr\s*Ern\u00e4hrung\s*und\s*Landwirtschaft|Bayerischen\s*Staatsministeriums\s*f\u00fcr\s+Umwelt\s*und\s*Gesundheit|Hessische\s*Ministerium\s*des\s*Innern\s*und\s*f\u00fcr\s*Sport|Deutsche\s*Rentenversicherung\s*Bund|Deutsche\s*Rentenversicherung\s*Rheinland|Deutschen\s*Rentenversicherung|Justizministerium\s*des\s*Landes\s*Niedersachsen|Ministerium\s*f\u00fcr\s*Justiz\s*,\s*Europa\s*,\s*Verbraucherschutz\s*und\s*Gleichstellung\s*des\s*Landes\s*Schleswig-Holstein|Bundesministerium\s+der\s+Finanzen|BMF|Bundesagentur\s+f\u00fcr\s+Arbeit|Bundesrat|Bundestag|Bundesregierung|Staatsministeriums\s+der\s+Justiz\s+sowie\s+des\s+Staatsministeriums\s+f\u00fcr\s+Kultus\s+des\s+Freistaates\s+Sachsen|Wehrbeauftragten\s+des\s+Deutschen\s+Bundestages|Bundesbeauftragte\s+f\u00fcr\s+den\s+Datenschutz\s+und\s+die\s+Informationsfreiheit|Fliegerhorst\s+B\u00fcchel|Dienststellen\s+R\s+I\s+\(\s*R\s+I\s*\)\s*,\s*R\s+II\s+\(\s*R\s+II\s*\)\s*,\s*D\s+\(\s*D\s*\)\s+K\s*,\s+Sp\s*,\s+D\s+L\s+\(\s+DL\s*\)\s+K\s*,\s+D\s+G\s+und\s+DL\s+G|Generalinspekteur\s+der\s+Bundeswehr|Deutschen\s+Fu\u00dfball-Bund)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.634 | 0.032 | 0.061 | 41 | 26 | 15 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 26 | 15 | 743 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53664`) (sent_id: `53664`)


Soweit das Bundesministerium der Verteidigung darauf verweist , dass Personalgespräche dazu dienten , Soldaten über Verwendungsplanungen der Personalführung zu informieren und in diesem Zusammenhang deren persönliche Motivationslage zu ergründen , um das dienstliche Bedürfnis und die privaten Lebensumstände des Soldaten möglichst gut miteinander zu vereinbaren , trifft dies zweifellos auf Personalgespräche im Vorfeld von Personalmaßnahmen zu .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Example 1** (doc_id: `53666`) (sent_id: `53666`)


Der Senat hat mit Verfügungen vom 25. Januar 2018 , 12. Februar 2018 und 14. März 2018 ergänzende Auskünfte des Auswärtigen Amtes eingeholt .

| Predicted | Gold |
|---|---|
| `Auswärtigen Amtes` | `Auswärtigen Amtes` |

**Example 2** (doc_id: `53690`) (sent_id: `53690`)


An dieser Regelung hielt die Bundesregierung fest , obwohl der Bundesrat die Einbeziehung der einmaligen Einnahmen vorschlug ( BT-Drucks 16/2454 S 11 ) .

| Predicted | Gold |
|---|---|
| `Bundesrat` | `Bundesrat` |

**Missed by this rule (FN):**

- `Bundesregierung` (ORG)
- `BT-Drucks 16/2454 S 11` (LIT)

**Example 3** (doc_id: `54428`) (sent_id: `54428`)


Der dort vorgesehene Versetzungsschutz bei Versetzungen in zeitlicher Nähe zum Dienstzeitende wird nicht , wie das Bundesministerium der Verteidigung einwendet , durch die Vorgaben der Zentralen Dienstvorschrift A- 1350/66 über die " Letzte Verwendung vor Zurruhesetzung " ausgeschlossen .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Missed by this rule (FN):**

- `Zentralen Dienstvorschrift A- 1350/66` (REG)

**Example 4** (doc_id: `54509`) (sent_id: `54509`)


Die Vertreterin der Staatskasse ( Erinnerungsgegnerin ) hält die Ermittlung des Streitwerts für zutreffend und beantragt , die Erinnerung als unbegründet zurückzuweisen .

| Predicted | Gold |
|---|---|
| `Staatskasse` | `Staatskasse` |

**Example 5** (doc_id: `55068`) (sent_id: `55068`)


An dieser Regelung hielt die Bundesregierung fest , obwohl der Bundesrat die Einbeziehung der einmaligen Einnahmen vorschlug ( BT-Drucks 16/2454 S 11 ) .

| Predicted | Gold |
|---|---|
| `Bundesrat` | `Bundesrat` |

**Missed by this rule (FN):**

- `Bundesregierung` (ORG)
- `BT-Drucks 16/2454 S 11` (LIT)

**Example 6** (doc_id: `55481`) (sent_id: `55481`)


Das Bundesministerium der Verteidigung beantragt ,

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Example 7** (doc_id: `55893`) (sent_id: `55893`)


Nach dem Gesetzentwurf der Bundesregierung zum Entwurf des Teilzeit- und Befristungsgesetzes vom 24. Oktober 2000 sollte es weiterhin zulässig sein , einen Arbeitsvertrag ohne Vorliegen eines sachlichen Grundes bis zur Dauer von zwei Jahren zu befristen und einen zunächst kürzer befristeten Arbeitsvertrag innerhalb der zweijährigen Höchstbefristungsdauer höchstens dreimal zu verlängern .

| Predicted | Gold |
|---|---|
| `Bundesregierung` | `Bundesregierung` |

**Missed by this rule (FN):**

- `Teilzeit- und Befristungsgesetzes` (NRM)

**Example 8** (doc_id: `56091`) (sent_id: `56091`)


Das Bundesministerium der Verteidigung beantragt ,

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Example 9** (doc_id: `56550`) (sent_id: `56550`)


Zwar verfügen beide Bewerber nicht über die nach dem Planungsbogen ursprünglich als dienstpostenunabhängiges Kriterium geforderte Stabsoffizierverwendung im Bundesministerium der Verteidigung ( oder eine vergleichbare Verwendung ) .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Example 10** (doc_id: `56577`) (sent_id: `56577`)


Dies ist im Beschwerdebescheid des Bundesministeriums der Verteidigung ohne Rechtsfehler mitentschieden worden .

| Predicted | Gold |
|---|---|
| `Bundesministeriums der Verteidigung` | `Bundesministeriums der Verteidigung` |

**Example 11** (doc_id: `56891`) (sent_id: `56891`)


Nach der speziellen Zuständigkeitsvorschrift in § 3 Satz 1 SAZV ist für " Maßnahmen nach der Soldatenarbeitszeitverordnung " das Bundesministerium der Verteidigung zuständig , soweit nichts Abweichendes bestimmt ist .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Missed by this rule (FN):**

- `§ 3 Satz 1 SAZV` (NRM)
- `Soldatenarbeitszeitverordnung` (NRM)

**Example 12** (doc_id: `57265`) (sent_id: `57265`)


2. Die angefochtene Versetzung ist rechtswidrig , weil das Bundesamt für das Personalmanagement und das Bundesministerium der Verteidigung bei der Ausübung des ihnen zustehenden Ermessens die von dem Antragsteller geltend gemachte Betreuung seiner Großmutter nicht berücksichtigt haben ( § 23a Abs. 2 Satz 1 WBO i. V. m. § 114 Satz 1 VwGO ) .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Missed by this rule (FN):**

- `Bundesamt für das Personalmanagement` (ORG)
- `§ 23a Abs. 2 Satz 1 WBO` (NRM)
- `§ 114 Satz 1 VwGO` (NRM)

**Example 13** (doc_id: `57699`) (sent_id: `57699`)


3. Soweit das Verfahren eingestellt wurde , trägt die Staatskasse die Kosten des Verfahrens und die notwendigen Auslagen des Angeklagten ; die verbleibenden Kosten seines Rechtsmittels trägt der Angeklagte selbst .

| Predicted | Gold |
|---|---|
| `Staatskasse` | `Staatskasse` |

**Example 14** (doc_id: `58056`) (sent_id: `58056`)


Das tunesische Justizministerium hat in seinem Schreiben vom 1. März 2018 ( S. 1 , Anlage 1 der Auskunft des Auswärtigen Amtes vom 7. März 2018 ) über Gespräche unter anderem mit Vertretern des deutschen Bundesministeriums der Justiz und für Verbraucherschutz ausgeführt , dass im Jahr 2012 insgesamt 122 Todesurteile in lebenslange Freiheitsstrafen umgewandelt worden sind .

| Predicted | Gold |
|---|---|
| `Auswärtigen Amtes` | `Auswärtigen Amtes` |

**Missed by this rule (FN):**

- `tunesische Justizministerium` (ORG)
- `Bundesministeriums der Justiz und für Verbraucherschutz` (ORG)

**Example 15** (doc_id: `58109`) (sent_id: `58109`)


Aus dem Lagebericht des Auswärtigen Amtes vom 16. Januar 2017 ( S. 17 ) und der Verbalnote des tunesischen Außenministeriums vom 11. Juli 2017 folgt , dass in Tunesien die Todesstrafe aufgrund eines Moratoriums seit 1991 nicht mehr vollstreckt wird .

| Predicted | Gold |
|---|---|
| `Auswärtigen Amtes` | `Auswärtigen Amtes` |

**Missed by this rule (FN):**

- `tunesischen Außenministeriums` (ORG)
- `Tunesien` (LOC)

**Example 16** (doc_id: `58428`) (sent_id: `58428`)


Zur Begründung seines Rechtsschutzbegehrens wiederholt und vertieft der Antragsteller sein Beschwerdevorbringen und betont , dass die vom Bundesministerium der Verteidigung herangezogene Rechtsgrundlage lediglich eine Verwaltungsvorschrift darstelle und deshalb nicht für die Ablehnung ausreiche .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Example 17** (doc_id: `58663`) (sent_id: `58663`)


bb ) Zwischenzeitlich hatte der Bundestag in Art. 11 Nr. 2 des Gesetzes zur Fortführung des Solidarpaktes , zur Neuordnung des bundesstaatlichen Finanzausgleichs und zur Abwicklung des Fonds " Deutsche Einheit " vom 20. Dezember 2001 ( Solidarpaktfortführungsgesetz - SFG ) ebenfalls eine Neufassung des § 7 Satz 2 GewStG beschlossen .

| Predicted | Gold |
|---|---|
| `Bundestag` | `Bundestag` |

**Missed by this rule (FN):**

- `Art. 11 Nr. 2 des Gesetzes zur Fortführung des Solidarpaktes , zur Neuordnung des bundesstaatlichen Finanzausgleichs und zur Abwicklung des Fonds " Deutsche Einheit " vom 20. Dezember 2001` (NRM)
- `Solidarpaktfortführungsgesetz` (NRM)
- `SFG` (NRM)
- `§ 7 Satz 2 GewStG` (NRM)

**Example 18** (doc_id: `58976`) (sent_id: `58976`)


Ebensowenig ist die Nichtanwendung einzelner Bestimmungen einer umsetzungsbedürftigen Verwaltungsvorschrift des Bundesministeriums der Verteidigung eine dienstliche Maßnahme im Sinne des § 17 Abs. 3 Satz 1 WBO ( BVerwG , Beschluss vom 19. Dezember 2017 - 1 WDS-VR 10.17 - Rn. 19 ) .

| Predicted | Gold |
|---|---|
| `Bundesministeriums der Verteidigung` | `Bundesministeriums der Verteidigung` |

**Missed by this rule (FN):**

- `§ 17 Abs. 3 Satz 1 WBO` (NRM)
- `BVerwG , Beschluss vom 19. Dezember 2017 - 1 WDS-VR 10.17 - Rn. 19` (RS)

**Example 19** (doc_id: `59179`) (sent_id: `59179`)


Dazu hat das Bundesministerium der Verteidigung im Beschwerdebescheid vom 3. November 2017 ( Seite 6 ) nachvollziehbar dargelegt , dass in der Hierarchie der genannten Soldaten im Regelfall mehrere weitere höhere Vorgesetzte zwischengeschaltet sind , die ihrerseits für die Anwendung eines sachgerechten Beurteilungsmaßstabs und damit für die Einhaltung der Richtwerte verantwortlich sind .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Example 20** (doc_id: `59291`) (sent_id: `59291`)


V. 1. Von Seiten des Bundes und der Länder haben das Bundesministerium der Finanzen für die Bundesregierung sowie die Bayerische Staatskanzlei für die Landesregierung Bayern und das Hessische Ministerium der Finanzen für die Landesregierung Hessen Stellung genommen .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Finanzen` | `Bundesministerium der Finanzen` |

**Missed by this rule (FN):**

- `Bundesregierung` (ORG)
- `Bayerische Staatskanzlei` (ORG)
- `Landesregierung Bayern` (ORG)
- `Hessische Ministerium der Finanzen` (ORG)
- `Landesregierung Hessen` (ORG)

**Example 21** (doc_id: `59491`) (sent_id: `59491`)


Es geht somit um die Frage , ob die gesetzlichen Regelungen zu Erstattungsansprüchen nach § 68 AufenthG durch Zusagen einer öffentlichen Stelle zu einem Leistungssystem ( hier : Ausländerbehörde zum Asylbewerberleistungsgesetz auf der Grundlage des Landesaufnahmeprogramms Nordrhein-Westfalen ) zulasten einer anderen öffentlichen Stelle in einem anderen Leistungssystem ( hier : Bundesagentur für Arbeit - BA - in der Grundsicherung für Arbeitsuchende ) abänderbar sind " .

| Predicted | Gold |
|---|---|
| `Bundesagentur für Arbeit` | `Bundesagentur für Arbeit` |

**Missed by this rule (FN):**

- `§ 68 AufenthG` (NRM)
- `Asylbewerberleistungsgesetz` (NRM)
- `Nordrhein-Westfalen` (LOC)
- `BA` (ORG)

**Example 22** (doc_id: `59571`) (sent_id: `59571`)


Die gerichtliche Überprüfung richtet sich auch darauf , ob die vom Bundesministerium der Verteidigung im Wege der Selbstbindung in Erlassen und Richtlinien festgelegten Maßgaben und Verfahrensvorschriften eingehalten sind ( vgl. BVerwG , Beschluss vom 27. Februar 2003 - 1 WB 57.02 - BVerwGE 118 , 25 < 27 > ) , wie sie sich hier insbesondere aus dem Zentralerlass ( ZE ) B- 1300/46 ( " Versetzung , Dienstpostenwechsel , Kommandierung " ) sowie aus den Verwaltungsvorschriften zur Auslandsverwendung von Soldaten ( Erlass des BMVg " Verwendung von Soldaten im Ausland und bei integrierten Stäben im Inland " vom 25. November 1999 , VMBl 2000 , S. 7 ; Zentralerlass B- 1340/9 " Verwendung von Soldaten im Ausland und bei integrierten Stäben " vom 11. Juni 2014 und ZDv A- 1340/9 " Verwendung von Soldatinnen und Soldaten im Ausland " vom 7. Dezember 2016 ) ergeben .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Missed by this rule (FN):**

- `BVerwG , Beschluss vom 27. Februar 2003 - 1 WB 57.02 - BVerwGE 118 , 25 < 27 >` (RS)
- `Zentralerlass ( ZE ) B- 1300/46 ( " Versetzung , Dienstpostenwechsel , Kommandierung " )` (REG)
- `Erlass des BMVg " Verwendung von Soldaten im Ausland und bei integrierten Stäben im Inland " vom 25. November 1999 , VMBl 2000 , S. 7` (REG)
- `Zentralerlass B- 1340/9 " Verwendung von Soldaten im Ausland und bei integrierten Stäben " vom 11. Juni 2014 und ZDv A- 1340/9 " Verwendung von Soldatinnen und Soldaten im Ausland " vom 7. Dezember 2016` (REG)

**Example 23** (doc_id: `59633`) (sent_id: `59633`)


Nach Auskunft des Auswärtigen Amtes und des Bundesinnenministeriums ist die Begnadigung die in der Praxis übliche Vorgehensweise für die Strafrestaussetzung .

| Predicted | Gold |
|---|---|
| `Auswärtigen Amtes` | `Auswärtigen Amtes` |

**Missed by this rule (FN):**

- `Bundesinnenministeriums` (ORG)

**Example 24** (doc_id: `59717`) (sent_id: `59717`)


Es kommt parallel dazu eine weitere Begnadigung nach Art. 372 CPP in Betracht ( vgl. Auskunft des Auswärtigen Amtes vom 7. März 2018 , S. 2 ) .

| Predicted | Gold |
|---|---|
| `Auswärtigen Amtes` | `Auswärtigen Amtes` |

**Missed by this rule (FN):**

- `Art. 372 CPP` (NRM)

**Example 25** (doc_id: `59744`) (sent_id: `59744`)


Die Erstattung entspricht hierbei auch der Billigkeit , da der genannte Mangel so schwerwiegend ist , dass nach einer Abwägung zwischen dem fiskalischen Interesse der Staatskasse und den Belangen der Antragstellerin , den zuletzt genannten der Vorrang gebührt ( vgl. Schulte / Püschel , PatG , 10. Aufl. , § 73 Rn. 138 ) .

| Predicted | Gold |
|---|---|
| `Staatskasse` | `Staatskasse` |

**Missed by this rule (FN):**

- `Schulte / Püschel , PatG , 10. Aufl. , § 73 Rn. 138` (LIT)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53767`) (sent_id: `53767`)


Ob der Erwerb eigener Anteile auf der Gesellschaftsebene entsprechend der durch das BilMoG geänderten handelsrechtlichen Vorschriften ( Einfügung des § 272 Abs. 1a und 1b HGB ) steuerrechtlich nicht mehr als Erwerbsvorgang anzusehen , sondern nunmehr als " Teilliquidation " und daher " wie " eine Kapitalherabsetzung zu behandeln ist ( so BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f. ; zustimmend z.B. Blumenberg / Lechner , DB 2014 , 141 , 147 ; kritisch z.B. Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55 ) , kann im Streitfall mangels Erheblichkeit für die Entscheidung offenbleiben .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BilMoG`(NRM)
- `§ 272 Abs. 1a und 1b HGB`(NRM)
- `BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f.`(REG)
- `Blumenberg / Lechner , DB 2014 , 141 , 147`(LIT)
- `Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55`(LIT)

**Example 1** (doc_id: `54854`) (sent_id: `54854`)


Daran ändert auch die Wiederholung der Verwaltungsauffassung durch das zu § 227 der Abgabenordnung ( AO ) ergangene und erst nach Einreichung der Beschwerdebegründung veröffentlichte BMF-Schreiben vom 29. März 2018 ( BStBl I 2018 , 588 ) nichts .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `BMF-Schreiben vom 29. März 2018 ( BStBl I 2018 , 588 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 227 der Abgabenordnung`(NRM)
- `AO`(NRM)
- `BMF-Schreiben vom 29. März 2018 ( BStBl I 2018 , 588 )`(REG)

**Example 2** (doc_id: `55584`) (sent_id: `55584`)


e ) Soweit das FA - wie bereits das BMF im Verfahren V R 61/03 - einwendet , dass Nr. 34 der Anlage 2 zum UStG auf die Unterpos . 2201 9000 des Zolltarifs verweise ( vgl. dazu auch Schrader , Mehrwertsteuerrecht 2013 , 115 ) , hat sich der BFH bereits mit diesem Argument auseinandergesetzt und ausgeführt , daraus könne ein gesetzlicher Ausschluss des Legens eines Hausanschlusses von der Steuerermäßigung nicht hergeleitet werden ( vgl. BFH-Urteil in BFHE 222 , 176 , BStBl II 2009 , 321 , unter II. 3. d ee , Rz 60 und 62 ) .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `BMF im Verfahren V R 61/03`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BMF im Verfahren V R 61/03`(RS)
- `Nr. 34 der Anlage 2 zum UStG`(NRM)
- `Schrader , Mehrwertsteuerrecht 2013 , 115`(LIT)
- `BFH`(ORG)
- `BFH-Urteil in BFHE 222 , 176 , BStBl II 2009 , 321 , unter II. 3. d ee , Rz 60 und 62`(RS)

**Example 3** (doc_id: `55736`) (sent_id: `55736`)


Die Beschwerdeakte des Bundesministeriums der Verteidigung - R II 2 - Az. : ... - und die Personalgrundakte des Antragstellers haben dem Senat bei der Beratung vorgelegen .

**False Positives:**

- `Bundesministeriums der Verteidigung - R II 2 - ` — partial — gold is substring of pred: `Bundesministeriums der Verteidigung - R II 2 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesministeriums der Verteidigung - R II 2 -`(ORG)

**Example 4** (doc_id: `56059`) (sent_id: `56059`)


Nach dem Schreiben des Bundesministeriums der Finanzen ( BMF ) vom 7. April 2009 ( BStBl I 2009 , 531 ) und Abschn. 12.1 Abs. 1 Satz 3 des Umsatzsteuer-Anwendungserlasses ( UStAE ) seien die Grundsätze der o. g. Rechtsprechung auf das Legen des Hausanschlusses durch ein Wasserversorgungsunternehmen beschränkt .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `Schreiben des Bundesministeriums der Finanzen ( BMF ) vom 7. April 2009 ( BStBl I 2009 , 531 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Schreiben des Bundesministeriums der Finanzen ( BMF ) vom 7. April 2009 ( BStBl I 2009 , 531 )`(REG)
- `Abschn. 12.1 Abs. 1 Satz 3 des Umsatzsteuer-Anwendungserlasses`(REG)
- `UStAE`(REG)

**Example 5** (doc_id: `56323`) (sent_id: `56323`)


aa ) Mit Urteil vom 24. 3. 2016 ( B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 ) hat der Senat bereits entschieden , dass es für den Beginn der aufgeschobenen Versicherungspflicht nach § 7a Abs 6 S 1 SGB IV - mit Wirkung für alle Zweige der Sozialversicherung - auf die Bekanntgabe einer ( ersten ) Entscheidung der Deutschen Rentenversicherung Bund über das Bestehen von " Beschäftigung " ankommt und nicht auf eine ( spätere ) - diese unzulässige Elementenfeststellung korrigierende - Entscheidung über " Versicherungspflicht wegen Beschäftigung " .

**False Positives:**

- `Deutschen Rentenversicherung` — partial — pred is substring of gold: `Deutschen Rentenversicherung Bund`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil vom 24. 3. 2016 ( B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 )`(RS)
- `§ 7a Abs 6 S 1 SGB IV`(NRM)
- `Deutschen Rentenversicherung Bund`(ORG)

**Example 6** (doc_id: `56391`) (sent_id: `56391`)


Bei Regiebetrieben setze die Anerkennung von Rücklagen i. S. des § 20 Abs. 1 Nr. 10 Buchst. b Satz 1 EStG voraus , dass die Zwecke des Betriebs gewerblicher Art ohne die Rücklagenbildung nachhaltig nicht erfüllt werden könnten ( vgl. BMF-Schreiben in BStBl I 2005 , 831 , Rz 23 ) .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `BMF-Schreiben in BStBl I 2005 , 831 , Rz 23`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 20 Abs. 1 Nr. 10 Buchst. b Satz 1 EStG`(NRM)
- `BMF-Schreiben in BStBl I 2005 , 831 , Rz 23`(REG)

**Example 7** (doc_id: `56643`) (sent_id: `56643`)


Mit seiner Revision macht das FA geltend , bei Regiebetrieben setze die Anerkennung von Rücklagen i. S. des § 20 Abs. 1 Nr. 10 Buchst. b EStG voraus , dass die Zwecke des Betriebs gewerblicher Art ohne die Rücklagenbildung nachhaltig nicht erfüllt werden könnten ( vgl. Schreiben des Bundesministeriums der Finanzen - BMF - vom 8. August 2005 IV B 7 -S 2706a - 4/05 , BStBl I 2005 , 831 , Rz 23 ; vom 9. Januar 2015 IV C 2 -S 2706- a / 13/10001 , BStBl I 2015 , 111 , Rz 35 ) .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `Schreiben des Bundesministeriums der Finanzen - BMF - vom 8. August 2005 IV B 7 -S 2706a - 4/05 , BStBl I 2005 , 831 , Rz 23`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 20 Abs. 1 Nr. 10 Buchst. b EStG`(NRM)
- `Schreiben des Bundesministeriums der Finanzen - BMF - vom 8. August 2005 IV B 7 -S 2706a - 4/05 , BStBl I 2005 , 831 , Rz 23`(REG)
- `vom 9. Januar 2015 IV C 2 -S 2706- a / 13/10001 , BStBl I 2015 , 111 , Rz 35`(REG)

**Example 8** (doc_id: `56742`) (sent_id: `56742`)


2. Dies gilt auch für Gutschriften auf dem Wertguthabenkonto eines Fremd-Geschäftsführers einer GmbH ( entgegen BMF-Schreiben vom 17. Juni 2009 , BStBl I 2009 , 1286 , A. IV. 2. b. ) .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `BMF-Schreiben vom 17. Juni 2009 , BStBl I 2009 , 1286 , A. IV. 2. b.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BMF-Schreiben vom 17. Juni 2009 , BStBl I 2009 , 1286 , A. IV. 2. b.`(REG)

**Example 9** (doc_id: `57250`) (sent_id: `57250`)


Damit hat der BFH für Eigenbetriebe bestätigt , dass grundsätzlich jedes " Stehenlassen " der handelsrechtlichen Gewinne als Eigenkapital für Zwecke des Betriebs gewerblicher Art ausreicht , unabhängig davon , ob dies in der Form der Zuführung zu den Gewinnrücklagen , als Gewinnvortrag oder unter einer anderen Position des Eigenkapitals geschieht ( vgl. auch BMF-Schreiben in BStBl I 2015 , 111 , Rz 34 ; Verfügung der Oberfinanzdirektion Karlsruhe vom 7. Oktober 2015 S 270. 6/43 -St 212 , unter V. 2.1.1 ; Bott in Ernst & Young , a. a. O. , § 4 Rz 452.5 ; Gastl in Hidien / Jürgens , a. a. O. , § 6 Rz 101 ) .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `BMF-Schreiben in BStBl I 2015 , 111 , Rz 34`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH`(ORG)
- `BMF-Schreiben in BStBl I 2015 , 111 , Rz 34`(REG)
- `Verfügung der Oberfinanzdirektion Karlsruhe vom 7. Oktober 2015 S 270. 6/43 -St 212 , unter V. 2.1.1`(RS)
- `Bott in Ernst & Young , a. a. O. , § 4 Rz 452.5`(LIT)
- `Gastl in Hidien / Jürgens , a. a. O. , § 6 Rz 101`(LIT)

**Example 10** (doc_id: `57748`) (sent_id: `57748`)


Sie verfügten bei Aufnahme der Tätigkeit jeweils über eine Befreiungsentscheidung der Bundesversicherungsanstalt für Angestellte als Rechtsvorgängerin der beklagten Deutschen Rentenversicherung Bund .

**False Positives:**

- `Deutschen Rentenversicherung` — partial — pred is substring of gold: `Deutschen Rentenversicherung Bund`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Deutschen Rentenversicherung Bund`(ORG)

**Example 11** (doc_id: `58056`) (sent_id: `58056`)


Das tunesische Justizministerium hat in seinem Schreiben vom 1. März 2018 ( S. 1 , Anlage 1 der Auskunft des Auswärtigen Amtes vom 7. März 2018 ) über Gespräche unter anderem mit Vertretern des deutschen Bundesministeriums der Justiz und für Verbraucherschutz ausgeführt , dass im Jahr 2012 insgesamt 122 Todesurteile in lebenslange Freiheitsstrafen umgewandelt worden sind .

**False Positives:**

- `Bundesministeriums der Justiz` — partial — pred is substring of gold: `Bundesministeriums der Justiz und für Verbraucherschutz`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `tunesische Justizministerium`(ORG)
- `Auswärtigen Amtes`(ORG)
- `Bundesministeriums der Justiz und für Verbraucherschutz`(ORG)

**Example 12** (doc_id: `58497`) (sent_id: `58497`)


Die Beschwerdeakten des Bundesministeriums der Verteidigung - R II 2 - 1136/17 und 1377/17 - sowie die Gerichtsakten zu den Verfahren BVerwG 1 WB 4.16 , BVerwG 1 WB 33.16 und BVerwG 1 WDS-VR 10.17 haben dem Senat bei der Beratung vorgelegen .

**False Positives:**

- `Bundesministeriums der Verteidigung - R II 2 - 1136` — partial — pred is substring of gold: `Bundesministeriums der Verteidigung - R II 2 - 1136/17 und 1377/17 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesministeriums der Verteidigung - R II 2 - 1136/17 und 1377/17 -`(RS)
- `Verfahren BVerwG 1 WB 4.16 , BVerwG 1 WB 33.16`(RS)
- `BVerwG 1 WDS-VR 10.17`(RS)

**Example 13** (doc_id: `58671`) (sent_id: `58671`)


Zu diesem Zweck regelt Art § 12 Abs 1 S 1 SpTrUG allein die Frage , ob neben der - vorliegend durch § 75 GmbHG bereits geklärten Entstehung der neuen Kapitalgesellschaften - auch die beabsichtigten Vermögensübergänge im Wege der Einzelrechtsnachfolge wirksam geworden sind ( vgl die Gegenäußerung der Bundesregierung zur Stellungnahme des Bundesrats - BR-Drucks 71 / 1/91 - in BT-Drucks 12/214 S 1 ) und ordnet insofern die Heilung an .

**False Positives:**

- `Bundesregierung` — partial — pred is substring of gold: `Gegenäußerung der Bundesregierung zur Stellungnahme des Bundesrats - BR-Drucks 71 / 1/91 - in BT-Drucks 12/214 S 1`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 12 Abs 1 S 1 SpTrUG`(NRM)
- `§ 75 GmbHG`(NRM)
- `Gegenäußerung der Bundesregierung zur Stellungnahme des Bundesrats - BR-Drucks 71 / 1/91 - in BT-Drucks 12/214 S 1`(LIT)

**Example 14** (doc_id: `59047`) (sent_id: `59047`)


Die Beschwerdeakte des Bundesministeriums der Verteidigung - R II 2 - ... - und die Personalgrundakte des Antragstellers , Hauptteile A bis D , haben dem Senat bei der Beratung vorgelegen .

**False Positives:**

- `Bundesministeriums der Verteidigung - R II 2` — partial — pred is substring of gold: `Bundesministeriums der Verteidigung - R II 2 - ... -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesministeriums der Verteidigung - R II 2 - ... -`(ORG)

</details>

---

</details>

---

<details>
<summary>🔇 Inactive Rules</summary>

## `Senate/Chamber of Courts`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `12ca8a62`  
**Description:**
Matches specific senate/chamber names of courts with location.

**Content:**
```
\b(?:\d+\.\s+(?:Senat|Kammer)\s+f\u00fcr\s+[A-Za-z\s]+\s+des\s+(?:Oberlandesgericht|Landgericht|Amtsgericht|Verwaltungsgericht|Arbeitsgericht|Sozialgericht|Finanzgericht))\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Government Ministries with Suffix`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `d54229fb`  
**Description:**
Matches government ministries including optional department codes (e.g., - R II 2 -).

**Content:**
```
\b(Bundesministeriums der Verteidigung - R II 2 -|Bundesministerium der Verteidigung - R II 2 -)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Organizations with Stiftung`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `6998f447`  
**Description:**
Matches organizations ending in 'Stiftung' (Foundation) with full name capture.

**Content:**
```
\b([A-Za-z\u00e4\u00f6\u00fc\u00df\s]+(?:\s+[A-Za-z\u00e4\u00f6\u00fc\u00df\s]+)*\s+Stiftung)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Organizations with Verband`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `0c476cd0`  
**Description:**
Matches organizations ending in 'Verband' (Association) with full name capture.

**Content:**
```
\b([A-Za-z\u00e4\u00f6\u00fc\u00df\s]+(?:\s+[A-Za-z\u00e4\u00f6\u00fc\u00df\s]+)*\s+Verband)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Unique Court Names with Location`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `c649f0f1`  
**Description:**
Matches specific court names that include a location as part of the name (e.g., Landesarbeitsgerichts Rheinland-Pfalz, Verwaltungsgericht München).

**Content:**
```
\b(Land(?:esarbeitsgerichts Rheinland\-Pfalz|gericht Darmstadt)|Verwaltungsgericht (?:M\u00fcnchen|Berlin)|Oberlandesgerichts Celle|Generalstaatsanwaltschaft Celle|19\.\s*Zivilsenats\s*des\s*Oberlandesgerichts\s*K\u00f6ln)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Specific Court Genitives`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `4fd6fd76`  
**Description:**
Matches specific court names with genitive endings followed by a location or context, with strict boundaries.

**Content:**
```
\b(?:Bundesverfassungsgerichts|Bundesverwaltungsgerichts|Bundesfinanzhofs|Bundesgerichtshofs|Landessozialgerichts|Landgerichts|Amtsgerichts|Verwaltungsgerichts|Oberlandesgerichts|Finanzgerichts|Sozialgerichts|Arbeitsgerichts)(?![a-zäöüß\s])
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Specific Court Name with Hyphen and Role`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `debbda68`  
**Description:**
Matches the specific pattern 'Amtsgericht Neu-Ulm - Jugendrichter -' and similar structures with hyphens and roles.

**Content:**
```
\bAmtsgericht\s+Neu-Ulm\s*-\s*Jugendrichter\s*-\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Patent Office Departments Full`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `82768c3f`  
**Description:**
Matches 'Markenstelle' and 'Prüfungsstelle' with class numbers and the full parent organization name.

**Content:**
```
\b(Markenstelle für Klasse \d+(?:\s+-\s+[A-Za-z\s-]+)?\s+des\s+Deutschen\s+Patent-\s+und\s+Markenamts|Prüfungsstelle für Klasse [A-Z0-9]+ des Deutschen Patent- und Markenamts)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Multi-Letter Company Codes`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `bb2f7a54`  
**Description:**
Matches specific multi-letter company codes like 'PTE' followed by name and GmbH/AG.

**Content:**
```
\b([A-Z]{2,}\s+[A-Za-z]+\s+GmbH|[A-Z]{2,}\s+AG)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `JPO and Other Abbreviations`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `d7491371`  
**Description:**
Matches specific abbreviations like JPO that were missing.

**Content:**
```
\b(JPO|G AG)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

</details>

---

<details>
<summary>📋 All Rules</summary>

## `Generic Court Abbreviations (Tightened)`

**F1:** 0.149 | **Precision:** 0.133 | **Recall:** 0.169  

**Format:** `regex`  
**Rule ID:** `6bc88f26`  
**Description:**
Matches high-priority court abbreviations, ensuring they are standalone and not part of a larger word.

**Content:**
```
\b(BGH|BVerfG|BFH|BSG|EuGH|EGMR|DPMA|BaFin|ZDS|DED|ZIV|EUIPO|STIKO|NEK|KCD-E|KON-KURD|CDK|BAG|ArbG|LAG|GBA|TdL|BgA|MDK|LSG|BA|H\u00c4VG|PKK|KCK|HPG|EED|JOOP|KAGURA|Schott|Urban\s+&\s+Schwarzenberg|National\s+Union\s+of\s+Rail|Maritime\s+and\s+Transport\s+Workers|Europ\u00e4ische\s+Gerichtshof\s+f\u00fcr\s+Menschenrechte|Deutscher\s+Apotheker\s+Verlag|B-KG|VEB\s*Elektronik\s*G\.|EZB|Bundestag|Bundesrat|Bundesregierung|Bundesverwaltungsgericht|Bundesfinanzhof|Bundespatentgericht|BVerwG|BVerfG|BGH|BSG|BAG|BVerfG|EuGH|EGMR|DPMA|BaFin|SG|FG|LAG|LSG|BA|H\u00c4VG|PKK|KCK|HPG|EED|JOOP|KAGURA|Schott|Urban\s+&\s+Schwarzenberg|National\s+Union\s+of\s+Rail|Maritime\s+and\s+Transport\s+Workers|Europ\u00e4ische\s+Gerichtshof\s+f\u00fcr\s+Menschenrechte|Deutscher\s+Apotheker\s+Verlag|B-KG|VEB\s*Elektronik\s*G\.)\b(?!\s+(?:Gericht|Amt|Beh\u00f6rde|Verfahren|Sache|Rechtsprechung|in|auf|von|mit|bei|nach|f\u00fcr|zum|zur|des|der|die|den|Beschluss|Urteil|Senat))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.133 | 0.169 | 0.149 | 1024 | 136 | 888 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 136 | 888 | 670 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53400`) (sent_id: `53400`)


Die Vorgabe des Bundesfinanzhofs ( BFH ) , dass die Einsteller ( Leistungsempfänger ) keine Land- und Forstwirte sein müssten , die von ihnen empfangenen Leistungen ( Zuchtleistungen als einheitliche Leistung ) aber gleichwohl zu landwirtschaftlichen Zwecken genutzt werden müssten , erscheine nicht eindeutig und auf den ersten Blick widersprüchlich .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Missed by this rule (FN):**

- `Bundesfinanzhofs` (ORG)

**Example 1** (doc_id: `53408`) (sent_id: `53408`)


Ein solcher enger sachlicher Zusammenhang mit der gesetzlichen Tätigkeit als Träger der Grundsicherung für Arbeitsuchende ist bei der BA immer dann anzunehmen , wenn sie - objektiv betrachtet - in dem zugrunde liegenden Verfahren eine Aufgabe im Rahmen des Vollzugs des SGB II wahrnimmt .

| Predicted | Gold |
|---|---|
| `BA` | `BA` |

**Missed by this rule (FN):**

- `SGB II` (NRM)

**Example 2** (doc_id: `53430`) (sent_id: `53430`)


II. Die zulässige Beschwerde der Löschungsantragsgegnerin hat sich durch die zwischenzeitlich mit Beschluss des DPMA vom 27. September 2017 angeordnete Verfallslöschung der angegriffenen Marke in der Hauptsache erledigt .

| Predicted | Gold |
|---|---|
| `DPMA` | `DPMA` |

**Example 3** (doc_id: `53524`) (sent_id: `53524`)


Der BFH führte aus , die handelsrechtliche Zurechnung von Vermögensgegenständen entspreche im Wesentlichen der Regelung des § 39 Abs. 2 Nr. 1 Satz 1 AO ( z.B. BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1. ; vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a ) .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Missed by this rule (FN):**

- `§ 39 Abs. 2 Nr. 1 Satz 1 AO` (NRM)
- `BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1.` (RS)
- `vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a` (RS)

**Example 4** (doc_id: `53549`) (sent_id: `53549`)


aa ) Der GBA hat eine IVIG-Therapie zur Behandlung des sekundären Immunglobulinmangels mit MGUS und rezidivierenden Pneumonien nicht empfohlen .

| Predicted | Gold |
|---|---|
| `GBA` | `GBA` |

**Example 5** (doc_id: `53594`) (sent_id: `53594`)


Die von der Krankenkasse nicht getragenen Kosten der stationären Behandlung zahlte der Kläger zu 1. Die Therapie der E in der K-Klinik war im Vorfeld durch den Medizinischen Dienst der Krankenversicherungen ( MDK ) auf ihre medizinische Notwendigkeit geprüft und befürwortet worden .

| Predicted | Gold |
|---|---|
| `MDK` | `MDK` |

**Missed by this rule (FN):**

- `E` (PER)
- `K-Klinik` (ORG)
- `Medizinischen Dienst der Krankenversicherungen` (ORG)

**Example 6** (doc_id: `53656`) (sent_id: `53656`)


Ebenso liegt es , wenn der GBA wegen des Potenzials der Methode bei nicht hinreichend belegtem Nutzen eine Erprobungs-RL beschließt ( § 137c Abs 1 S 3 SGB V ) und die Überprüfung unter Hinzuziehung der durch die Erprobung gewonnenen Erkenntnisse ergibt , dass die Methode nicht den Kriterien nach § 137c Abs 1 S 1 SGB V entspricht ( § 137c Abs 1 S 4 SGB V ) oder wenn eine Erprobungs-RL nicht zustande kommt , weil es an einer nach § 137e Abs 6 SGB V erforderlichen Vereinbarung fehlt ( § 137c Abs 1 S 5 SGB V ) .

| Predicted | Gold |
|---|---|
| `GBA` | `GBA` |

**Missed by this rule (FN):**

- `§ 137c Abs 1 S 3 SGB V` (NRM)
- `§ 137c Abs 1 S 1 SGB V` (NRM)
- `§ 137c Abs 1 S 4 SGB V` (NRM)
- `§ 137e Abs 6 SGB V` (NRM)
- `§ 137c Abs 1 S 5 SGB V` (NRM)

**Example 7** (doc_id: `53690`) (sent_id: `53690`)


An dieser Regelung hielt die Bundesregierung fest , obwohl der Bundesrat die Einbeziehung der einmaligen Einnahmen vorschlug ( BT-Drucks 16/2454 S 11 ) .

| Predicted | Gold |
|---|---|
| `Bundesregierung` | `Bundesregierung` |

**Missed by this rule (FN):**

- `Bundesrat` (ORG)
- `BT-Drucks 16/2454 S 11` (LIT)

**Example 8** (doc_id: `53746`) (sent_id: `53746`)


Das hat zur Folge , dass beim BSG über Kostenerinnerungen nach § 189 Abs 2 S 2 SGG der Senat in der Besetzung mit drei Berufsrichtern zu befinden hat .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Missed by this rule (FN):**

- `§ 189 Abs 2 S 2 SGG` (NRM)

**Example 9** (doc_id: `53867`) (sent_id: `53867`)


Zwar kann das LSG von einer Entscheidung ua des BSG auch dann abweichen , wenn es einen der höchstrichterlichen Rechtsprechung widersprechenden Rechtssatz nur sinngemäß und in scheinbar fallbezogene Ausführungen gekleidet entwickelt .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 10** (doc_id: `53939`) (sent_id: `53939`)


Dort hat der BFH lediglich ausgeführt , die Vergütung für die Hingabe eines partiarischen Darlehens könne auch umsatzabhängig ausgestaltet werden .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Example 11** (doc_id: `54059`) (sent_id: `54059`)


Wie der Kläger selbst vorträgt , wollte sich das LSG vielmehr nach seinem eigenen Verständnis im Rahmen der bereits vorliegenden Rechtsprechung des BSG halten , ist diesen aber in dem von ihm entschiedenen Fall lediglich deshalb nicht gefolgt , weil es - vom Kläger als unzutreffend gerügte - wesentliche Sachverhaltsunterschiede zu den bereits entschiedenen Fällen angenommen hat .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 12** (doc_id: `54064`) (sent_id: `54064`)


Der BFH prüft insofern nur , ob sie gegen Denkgesetze und Erfahrungssätze oder die anerkannten Auslegungsregeln verstößt .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Example 13** (doc_id: `54130`) (sent_id: `54130`)


b ) In Leasingfällen ( auch beim " Lease " im Rahmen eines " Sale-and-lease-back " ) geht der BFH - wie bereits in dem Urteil in BFHE 255 , 386 , BStBl II 2018 , 81 dargestellt - bei Anwendung des § 39 Abs. 2 Nr. 1 Satz 1 AO insbesondere von folgenden Grundsätzen aus :

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Missed by this rule (FN):**

- `Urteil in BFHE 255 , 386 , BStBl II 2018 , 81` (RS)
- `§ 39 Abs. 2 Nr. 1 Satz 1 AO` (NRM)

**Example 14** (doc_id: `54150`) (sent_id: `54150`)


Die Prüfung der Markenanmeldung muss daher nach der maßgeblichen Rechtsprechung des EuGH grundsätzlich streng und vollständig sein , um ungerechtfertigte Eintragungen zu vermeiden ( vgl. EuGH , GRUR 2003 , 604 Rn. 57 , 60 – Libertel ; BGH , GRUR 2014 , 565 Rn. 17 – smartbook ; Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 158 , 159 ) .

| Predicted | Gold |
|---|---|
| `EuGH` | `EuGH` |

**Missed by this rule (FN):**

- `EuGH , GRUR 2003 , 604 Rn. 57 , 60 – Libertel` (RS)
- `BGH , GRUR 2014 , 565 Rn. 17 – smartbook` (RS)
- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 158 , 159` (LIT)

**Example 15** (doc_id: `54257`) (sent_id: `54257`)


Indem die Absender die Postsendungen auf den Weg gebracht , zumindest in einem Teil der Fälle unzureichende Angaben gemacht oder Waren versandt haben , die gegen Verbote und Beschränkungen verstoßen könnten , haben sie zwar eine Bedingung für die vorübergehende Verwahrung bei der Zollstelle in Gestalt einer sog. conditio sine qua non gesetzt , welche allerdings allein für die Annahme eines " willentlichen Herbeiführens einer Amtshandlung " im Sinne vorgenannter Rechtsprechung des BVerwG ( Urteile in BVerwGE 91 , 109 , und in BVerwGE 153 , 321 ) nicht als ausreichend angesehen werden kann .

| Predicted | Gold |
|---|---|
| `BVerwG` | `BVerwG` |

**Missed by this rule (FN):**

- `Urteile in BVerwGE 91 , 109 , und in BVerwGE 153 , 321` (RS)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53392`) (sent_id: `53392`)


Bei Unklarheiten der Anmeldung ist daher der Wille des Anmelders durch Auslegung zu ermitteln ( vgl. BGH GRUR 2012 , 1139 , Nr. 23 , 30 - Weinkaraffe ; Eichmann / v. Falckenstein / Kühne , a. a. O. , § 37 Rn. 11 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH GRUR 2012 , 1139 , Nr. 23 , 30 - Weinkaraffe`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH GRUR 2012 , 1139 , Nr. 23 , 30 - Weinkaraffe`(RS)
- `Eichmann / v. Falckenstein / Kühne , a. a. O. , § 37 Rn. 11`(LIT)

**Example 1** (doc_id: `53395`) (sent_id: `53395`)


Darüber hinaus ist die Darlegung erforderlich , dass und warum die Entscheidung des LSG - ausgehend von dessen materieller Rechtsansicht - auf dem Mangel beruhen kann , dass also die Möglichkeit einer Beeinflussung der Entscheidung besteht .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `53422`) (sent_id: `53422`)


Bei der vom FG zu beurteilenden gesetzlichen Prozessführungsbefugnis ( Beteiligtenstellung ) handelt es sich um eine Sachurteilsvoraussetzung , deren fehlerhafte Beurteilung einen Verfahrensmangel darstellt ( BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943 ; vom 23. November 2010 V B 133/09 , BFH / NV 2011 , 612 , Rz 5 ; jeweils m. w. N. ) .

**False Positives:**

- `FG` — no gold match — likely missing annotation
- `BFH` — partial — pred is substring of gold: `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`
- `BFH` — partial — pred is substring of gold: `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`
- `BFH` — similar text (different position): `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`

> overlaps gold: 3  |  likely missing annotation: 1

**Gold Entities:**

- `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`(RS)
- `vom 23. November 2010 V B 133/09 , BFH / NV 2011 , 612 , Rz 5`(RS)

**Example 3** (doc_id: `53446`) (sent_id: `53446`)


( a ) Die Rechtslage im Nichtzulassungsbeschwerdeverfahren beruht allein darauf , dass ein Verfahrensmangel wie die verspätete Absetzung des Urteils kein Grund für die Zulassung der Revision war und ist ( vgl. zur Rechtslage vor Inkrafttreten des Anhörungsrügengesetzes : BVerfG 26. März 2001 - 1 BvR 383/00 - zu B I 2 c dd der Gründe ; BAG 1. Oktober 2003 - 1 ABN 62/01 - zu II 3 c der Gründe , BAGE 108 , 55 ; zur aktuellen Rechtslage nach § 72b Abs. 1 Satz 2 ArbGG BAG 24. Februar 2015 - 5 AZN 1007/14 - Rn. 3 ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG 26. März 2001 - 1 BvR 383/00 - zu B I 2 c dd der Gründe`
- `BAG` — partial — pred is substring of gold: `BAG 1. Oktober 2003 - 1 ABN 62/01 - zu II 3 c der Gründe , BAGE 108 , 55`
- `BAG` — similar text (different position): `BAG 1. Oktober 2003 - 1 ABN 62/01 - zu II 3 c der Gründe , BAGE 108 , 55`

> overlaps gold: 3  |  likely missing annotation: 0

**Gold Entities:**

- `Anhörungsrügengesetzes`(NRM)
- `BVerfG 26. März 2001 - 1 BvR 383/00 - zu B I 2 c dd der Gründe`(RS)
- `BAG 1. Oktober 2003 - 1 ABN 62/01 - zu II 3 c der Gründe , BAGE 108 , 55`(RS)
- `§ 72b Abs. 1 Satz 2 ArbGG`(NRM)
- `BAG 24. Februar 2015 - 5 AZN 1007/14 - Rn. 3`(RS)

**Example 4** (doc_id: `53451`) (sent_id: `53451`)


2. Ob dem FG im Zusammenhang mit der Vermögenszuwachsrechnung und der Geldverkehrsrechnung als selbständige Schätzungsgrundlagen für sich genommen ebenfalls Verfahrensfehler unterlaufen sind , ist vor diesem Hintergrund nicht mehr zu entscheiden .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `53453`) (sent_id: `53453`)


Zwar hat sich das FG im Urteil nicht ausdrücklich dazu geäußert , es hat die umstrittene Zahlung aber ohne weiteres als " Abfindungszahlung " bezeichnet und nicht infrage gestellt , dass es sich ( zumindest auch ) um eine Ersatzleistung für entgehende Einnahmen handeln sollte .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 6** (doc_id: `53468`) (sent_id: `53468`)


Wie das vorliegende Verfahren zeigt , bietet die Anhörungsrüge die Möglichkeit , durch Zuordnungsschwierigkeiten entstehende Probleme zu bewältigen ( vgl. BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15 ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15`(RS)

**Example 7** (doc_id: `53474`) (sent_id: `53474`)


Beim Leasing ist - wie dargestellt - darauf abzustellen , ob der Herausgabeanspruch des Leasinggebers ( zivilrechtlichen Eigentümers ) noch eine wirtschaftliche Bedeutung hat ( grundlegend BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1. ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1.`(RS)

**Example 8** (doc_id: `53475`) (sent_id: `53475`)


Gegenstand des BgA war die Verpachtung der städtischen Schwimmbäder an die ... GmbH ( S-GmbH ) , eine 100 % -ige Tochtergesellschaft der Klägerin .

**False Positives:**

- `BgA` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `... GmbH`(ORG)
- `S-GmbH`(ORG)

**Example 9** (doc_id: `53481`) (sent_id: `53481`)


Diese können insbesondere dann vorliegen , wenn die Gesamtstrafe sich nicht innerhalb des gesetzlichen Strafrahmens bewegt , die gebotene Begründung für die Gesamtstrafe fehlt , oder wenn die Besorgnis besteht , der Tatrichter habe sich von der Summe der Einzelstrafen leiten lassen ( BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris`(RS)

**Example 10** (doc_id: `53485`) (sent_id: `53485`)


Soweit die Elemente eines Bildzeichens nur die typischen Merkmale der in Rede stehenden Waren und Dienstleistungen darstellen oder sich in einfachen dekorativen Gestaltungsmitteln erschöpfen , an die sich der Verkehr etwa durch häufige Verwendung gewöhnt hat , wird diesem Zeichen im Allgemeinen wegen seines bloß beschreibenden Inhalts ebenfalls die konkrete Eignung fehlen , die mit ihm gekennzeichneten Waren oder Dienstleistungen von denjenigen anderer Herkunft zu unterscheiden ( vgl. BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband ; GRUR 2005 , 257 , 258 – Bürogebäude ; GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel ; GRUR 2001 , 239f. – Zahnpastastrang ; GRUR 2001 , 734 , 735 – Westie-Kopf ; BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband`(RS)
- `GRUR 2005 , 257 , 258 – Bürogebäude`(RS)
- `GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel`(RS)
- `GRUR 2001 , 239f. – Zahnpastastrang`(RS)
- `GRUR 2001 , 734 , 735 – Westie-Kopf`(RS)
- `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`(RS)

**Example 11** (doc_id: `53487`) (sent_id: `53487`)


Das SG verstand das Begehren des Klägers ( im Hauptantrag ) ebenfalls in dem Sinne , dass er mit einer Anfechtungsklage allein die Aufhebung des Widerspruchsbescheides vom 23. 7. 2015 verfolgte .

**False Positives:**

- `SG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 12** (doc_id: `53490`) (sent_id: `53490`)


Es müssen letztlich besondere Verhaltensweisen sowohl des Berechtigten als auch des Verpflichteten vorliegen , die es rechtfertigen , die späte Geltendmachung des Rechts als mit Treu und Glauben unvereinbar und für den Verpflichteten als unzumutbar anzusehen ( vgl. BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27 mwN ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27`(RS)

**Example 13** (doc_id: `53499`) (sent_id: `53499`)


f ) Zwar hätte die Klägerin aus eigenem Antrieb zur mündlichen Verhandlung erscheinen können , nachdem das FG bereits im Vorfeld mitgeteilt hatte , dass die Frage der ladungsfähigen Anschrift problematisch ist .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 14** (doc_id: `53524`) (sent_id: `53524`)


Der BFH führte aus , die handelsrechtliche Zurechnung von Vermögensgegenständen entspreche im Wesentlichen der Regelung des § 39 Abs. 2 Nr. 1 Satz 1 AO ( z.B. BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1. ; vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a ) .

**False Positives:**

- `BFH` — similar text (different position): `BFH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH`(ORG)
- `§ 39 Abs. 2 Nr. 1 Satz 1 AO`(NRM)
- `BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1.`(RS)
- `vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a`(RS)

**Example 15** (doc_id: `53539`) (sent_id: `53539`)


cc ) Stellt ein Bewerber nicht innerhalb eines Monats nach Zugang der Abbruchmitteilung einen Antrag auf Erlass einer einstweiligen Verfügung , darf der Dienstherr nach der Rechtsprechung des Bundesverwaltungsgerichts darauf vertrauen , dass der Bewerber den Abbruch des Auswahlverfahrens nicht angreift , sondern sein Begehren im Rahmen einer neuen Ausschreibung weiterverfolgt ( vgl. BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14 ) .

**False Positives:**

- `BVerwG` — partial — pred is substring of gold: `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesverwaltungsgerichts`(ORG)
- `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14`(RS)

**Example 16** (doc_id: `53551`) (sent_id: `53551`)


Die Beklagte wird das der Klägerin zustehende Honorar unter Zugrundelegung eines festen RLV für die gesamte BAG neu zu ermitteln haben .

**False Positives:**

- `BAG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 17** (doc_id: `53558`) (sent_id: `53558`)


Einen solchen qualifizierten Rechtsanwendungsfehler des FG hat die Klägerin indes nicht dargelegt .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 18** (doc_id: `53562`) (sent_id: `53562`)


Daher geht der Senat von einem Hauptbegehren des Klägers vor dem FG aus , das auf eine Änderung des Verlustfeststellungsbescheids zum 31. Dezember des Streitjahres vom 9. Oktober 2006 gerichtet war .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `53583`) (sent_id: `53583`)


Denn die Hauptfunktion einer Marke liegt darin , die Ursprungsidentität der gekennzeichneten Waren und Dienstleistungen zu gewährleisten ( vgl. u. a. EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel ; BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006 ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel`
- `BGH` — partial — pred is substring of gold: `BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel`(RS)
- `BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006`(RS)

**Example 20** (doc_id: `53598`) (sent_id: `53598`)


Dem FG seien für die Vorsteuerbeträge auch die freiwillig geführten Bestandskonten sowie sämtliche Datenerfassungsprotokolle und Buchungsjournale übermittelt worden .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 21** (doc_id: `53599`) (sent_id: `53599`)


Auch Angaben , die sich auf Umstände beziehen , die die Ware oder die Dienstleistung selbst nicht unmittelbar betreffen , fehlt die Unterscheidungskraft , wenn durch die Angabe ein enger beschreibender Bezug zu den angemeldeten Waren oder Dienstleistungen hergestellt wird und deshalb die Annahme gerechtfertigt ist , dass der Verkehr den beschreibenden Begriffsinhalt als solchen ohne Weiteres und ohne Unklarheiten erfasst und in der Bezeichnung nicht ein Unterscheidungsmittel für die Herkunft der angemeldeten Waren oder Dienstleistungen sieht ( BGH , GRUR 2014 , 569 , Rn. 10 – HOT ; BGH , GRUR 2012 , 1143 , Rn. 9 – Starsat ; BGH , GRUR 2009 , 952 , Rn. 10 – DeutschlandCard ; BGH , GRUR 2006 , 850 , Rn. 19 – FUSSBALL WM 2006 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , GRUR 2014 , 569 , Rn. 10 – HOT`
- `BGH` — similar text (different position): `BGH , GRUR 2014 , 569 , Rn. 10 – HOT`
- `BGH` — similar text (different position): `BGH , GRUR 2014 , 569 , Rn. 10 – HOT`
- `BGH` — similar text (different position): `BGH , GRUR 2014 , 569 , Rn. 10 – HOT`

> overlaps gold: 4  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , GRUR 2014 , 569 , Rn. 10 – HOT`(RS)
- `BGH , GRUR 2012 , 1143 , Rn. 9 – Starsat`(RS)
- `BGH , GRUR 2009 , 952 , Rn. 10 – DeutschlandCard`(RS)
- `BGH , GRUR 2006 , 850 , Rn. 19 – FUSSBALL WM 2006`(RS)

**Example 22** (doc_id: `53600`) (sent_id: `53600`)


Auch danach setzt die Bejahung der Unterscheidungskraft unverändert voraus , dass das Zeichen geeignet sein muss , die beanspruchten Produkte als von einem bestimmten Unternehmen stammend zu kennzeichnen ( vgl. EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK`(RS)

**Example 23** (doc_id: `53601`) (sent_id: `53601`)


a ) Das FG hielt - bei einer Grundmietzeit von vier Jahren - eine betriebsgewöhnliche Nutzungsdauer der Leasingobjekte von drei bis fünf Jahren für möglich .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 24** (doc_id: `53614`) (sent_id: `53614`)


Soweit Personen dezentral untergebracht sind , ist es für die Bejahung einer Einrichtung erforderlich , dass die dezentrale Unterkunft zu den Räumlichkeiten der Einrichtung gehört , der Hilfebedürftige also in die Räumlichkeiten des Einrichtungsträgers eingegliedert ist ( vgl BSG SozR 4 - 3500 § 98 Nr 3 ) .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG SozR 4 - 3500 § 98 Nr 3`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG SozR 4 - 3500 § 98 Nr 3`(RS)

**Example 25** (doc_id: `53618`) (sent_id: `53618`)


Denn die Hauptfunktion einer Marke besteht darin , die Ursprungsidentität der gekennzeichneten Waren oder Dienstleistungen zu gewährleisten ( vgl. etwa EuGH GRUR 2010 , 1008 , 1009 ( Nr. 38 ) - Lego ; GRUR 2008 , 608 , 611 ( Nr. 66 ) - EUROHYPO ; GRUR 2006 , 233 , 235 , Nr. 45 - Standbeutel ; BGH GRUR 2015 , 173 , 174 ( Nr. 15 ) - for you ; GRUR 2009 , 949 ( Nr. 10 ) - My World ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH GRUR 2010 , 1008 , 1009 ( Nr. 38 ) - Lego`
- `BGH` — partial — pred is substring of gold: `BGH GRUR 2015 , 173 , 174 ( Nr. 15 ) - for you`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2010 , 1008 , 1009 ( Nr. 38 ) - Lego`(RS)
- `GRUR 2008 , 608 , 611 ( Nr. 66 ) - EUROHYPO`(RS)
- `GRUR 2006 , 233 , 235 , Nr. 45 - Standbeutel`(RS)
- `BGH GRUR 2015 , 173 , 174 ( Nr. 15 ) - for you`(RS)
- `GRUR 2009 , 949 ( Nr. 10 ) - My World`(RS)

**Example 26** (doc_id: `53623`) (sent_id: `53623`)


Nach der von der Einsprechenden zitierten BGH-Entscheidung „ Kommunikationskanal “ , muss für den Fachmann die im Anspruch bezeichnete Lehre „ unmittelbar und eindeutig “ als mögliche Ausführungsform der Erfindung den Ursprungsunterlagen entnehmbar sein .

**False Positives:**

- `BGH` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 27** (doc_id: `53654`) (sent_id: `53654`)


Entsprechendes gilt , wenn ihre Rüge als Aufklärungsrüge verstanden werden sollte ; auch dazu hätte sie darlegen müssen , warum sich das LSG über die geschilderte Wahrnehmung der Klägerin hinaus zu weiterer Aufklärung hätte gedrängt sehen müssen ( vgl Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160a RdNr 16f ) .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160a RdNr 16f`(LIT)

**Example 28** (doc_id: `53660`) (sent_id: `53660`)


Das LSG hat nach mündlicher Verhandlung in Abwesenheit des Klägers die Berufung zurückgewiesen .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 29** (doc_id: `53661`) (sent_id: `53661`)


Die Klage hatte keinen Erfolg ; das Finanzgericht ( FG ) München hat sie mit Urteil vom 6. Juli 2017 11 K 411/13 als unbegründet abgewiesen .

**False Positives:**

- `FG` — partial — pred is substring of gold: `Finanzgericht ( FG ) München`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Finanzgericht ( FG ) München`(ORG)
- `Urteil vom 6. Juli 2017 11 K 411/13`(RS)

**Example 30** (doc_id: `53680`) (sent_id: `53680`)


Bei den Tatbestandsmerkmalen des Anfalls , der Notwendigkeit ( vgl. dazu z.B. BGH , Beschluss vom 18. Juli 2003 - IXa ZB 146/03 - , juris , Rn. 11 ; Smid , in : Wieczorek / Schütze , ZPO , 4. Aufl. 2016 , § 788 Rn. 21 ; Geimer , in : Zöller , ZPO , 32. Aufl. 2018 , § 788 Rn. 9a m. w. N. ) und der Höhe der Zwangsvollstreckungskosten ( gem. § 788 Abs. 1 ZPO ) handelt es sich um vom Gläubiger darzulegende und erforderlichenfalls zu beweisende anspruchsbegründende Tatsachen ( vgl. OLG Zweibrücken , Beschluss vom 21. Juli 1994 - 3 W 93/94 - , Rpfleger 1995 , S. 172 ; LG Düsseldorf , Beschluss vom 25. September 1990 - 25 T 740/90 - , JurBüro 1991 , S. 130 ) , zu deren Beweis der Schuldner den Gläubiger im Rahmen einer Vollstreckungsgegenklage - sofern ein Rechtsschutzbedürfnis besteht - zwingen kann .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , Beschluss vom 18. Juli 2003 - IXa ZB 146/03 - , juris , Rn. 11`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 18. Juli 2003 - IXa ZB 146/03 - , juris , Rn. 11`(RS)
- `Smid , in : Wieczorek / Schütze , ZPO , 4. Aufl. 2016 , § 788 Rn. 21`(LIT)
- `Geimer , in : Zöller , ZPO , 32. Aufl. 2018 , § 788 Rn. 9a`(LIT)
- `§ 788 Abs. 1 ZPO`(NRM)
- `OLG Zweibrücken , Beschluss vom 21. Juli 1994 - 3 W 93/94 - , Rpfleger 1995 , S. 172`(RS)
- `LG Düsseldorf , Beschluss vom 25. September 1990 - 25 T 740/90 - , JurBüro 1991 , S. 130`(RS)

**Example 31** (doc_id: `53688`) (sent_id: `53688`)


Allerdings beschränkt sich die Geltung des Grundsatzes der Bestenauslese im Bereich der Verwendungsentscheidungen auf Entscheidungen über - wie hier - höherwertige , die Beförderung in einen höheren Dienstgrad oder die Einweisung in die Planstelle einer höheren Besoldungsgruppe vorprägende Verwendungen ( vgl. klarstellend BVerwG , Beschluss vom 30. Januar 2014 - 1 WB 1.13 - juris Rn. 32 ) .

**False Positives:**

- `BVerwG` — partial — pred is substring of gold: `BVerwG , Beschluss vom 30. Januar 2014 - 1 WB 1.13 - juris Rn. 32`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerwG , Beschluss vom 30. Januar 2014 - 1 WB 1.13 - juris Rn. 32`(RS)

**Example 32** (doc_id: `53692`) (sent_id: `53692`)


I. Das LSG hat mit Urteil vom 11. 5. 2017 einen Zahlungsanspruch der Klägerin ( eine aus zwei Personen bestehende , im Partnerschaftsregister eingetragene Physiotherapie-Partnerschaft ) in Höhe von 7249,01 Euro für physiotherapeutische Leistungen verneint , nachdem die beklagte Krankenkasse die erbrachten Leistungen zunächst bezahlt , die Zahlungen aber wieder zurückgefordert und die Rückforderung schließlich im Wege der Aufrechnung durchgesetzt hatte .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 33** (doc_id: `53701`) (sent_id: `53701`)


Für eine solche Prognose des Arbeitgebers bedarf es ausreichend konkreter Anhaltspunkte ( BAG 27. Juli 2016 - 7 AZR 545/14 - Rn. 19 ; 24. September 2014 - 7 AZR 987/12 - Rn. 18 ; 7. Mai 2008 - 7 AZR 146/07 - Rn. 15 ; 7. April 2004 - 7 AZR 441/03 - zu II 2 a aa der Gründe ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 27. Juli 2016 - 7 AZR 545/14 - Rn. 19`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 27. Juli 2016 - 7 AZR 545/14 - Rn. 19`(RS)
- `24. September 2014 - 7 AZR 987/12 - Rn. 18`(RS)
- `7. Mai 2008 - 7 AZR 146/07 - Rn. 15`(RS)
- `7. April 2004 - 7 AZR 441/03 - zu II 2 a aa der Gründe`(RS)

**Example 34** (doc_id: `53703`) (sent_id: `53703`)


Hiervon ist auch das FG ausgegangen .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 35** (doc_id: `53707`) (sent_id: `53707`)


Da eine so weitgehende Selbstentäußerung des ausländischen Staates im Zweifel nicht zu vermuten ist ( BGH 30. Januar 2013 - III ZB 40/12 - Rn. 19 ) , dürfen die Umstände des Falles hinsichtlich des Vorliegens und der Reichweite eines Verzichts keinen Zweifel lassen ( BAG 29. Juni 2017 - 2 AZR 759/16 - Rn. 20 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH 30. Januar 2013 - III ZB 40/12 - Rn. 19`
- `BAG` — partial — pred is substring of gold: `BAG 29. Juni 2017 - 2 AZR 759/16 - Rn. 20`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH 30. Januar 2013 - III ZB 40/12 - Rn. 19`(RS)
- `BAG 29. Juni 2017 - 2 AZR 759/16 - Rn. 20`(RS)

**Example 36** (doc_id: `53717`) (sent_id: `53717`)


Bei Anerkennungsbeträgen handelt es sich um eine jener Massenerscheinungen , die ein typisierendes und pauschalierendes Vorgehen auch der Verwaltung rechtfertigen ( vgl. BVerfG , Beschluss vom 31. Mai 1988 - 1 BvR 520/83 - BVerfGE 78 , 214 < 227 > ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Beschluss vom 31. Mai 1988 - 1 BvR 520/83 - BVerfGE 78 , 214 < 227 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG , Beschluss vom 31. Mai 1988 - 1 BvR 520/83 - BVerfGE 78 , 214 < 227 >`(RS)

**Example 37** (doc_id: `53718`) (sent_id: `53718`)


Das FG hat auf der Grundlage der von ihm getroffenen Feststellungen zu Unrecht entschieden , dass dem Kläger für den Zeitraum , in dem sich S in Untersuchungshaft befand , weiterhin Kindergeld für S zustand .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `S`(PER)
- `S`(PER)

**Example 38** (doc_id: `53721`) (sent_id: `53721`)


Die Regelung ist daher geeignet , die Befristung von Arbeitsverträgen mit programmgestaltenden Mitarbeitern bei Rundfunkanstalten zu rechtfertigen ( BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101 ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101`(RS)

**Example 39** (doc_id: `53745`) (sent_id: `53745`)


Der Senat neigt aber zu der bereits vom Berichterstatter des FG im Erörterungstermin geäußerten Auffassung , dass dieser Mangel punktuell auf die Trinkgelder begrenzt ist und allein hieraus keine Schätzungsbefugnis für die Hauptkasse folgen dürfte .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 40** (doc_id: `53748`) (sent_id: `53748`)


Das FG hat zutreffend auf der Grundlage seiner Feststellungen entschieden , dass der in Rede stehende Auflösungsverlust i. S. von § 17 Abs. 4 EStG im Veranlagungszeitraum 2011 noch nicht entstanden und daher ein Verlustrücktrag in das Streitjahr zu versagen war .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 17 Abs. 4 EStG`(NRM)

**Example 41** (doc_id: `53760`) (sent_id: `53760`)


Etwas anderes gilt insbesondere dann , wenn der Arbeitgeber seine Tarifgebundenheit in einer dem Arbeitnehmer hinreichend erkennbaren Weise zur auflösenden Bedingung der Bezugnahme gemacht hat ( BAG 18. April 2007 - 4 AZR 652/05 - Rn. 32 , 40 , BAGE 122 , 74 ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 18. April 2007 - 4 AZR 652/05 - Rn. 32 , 40 , BAGE 122 , 74`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 18. April 2007 - 4 AZR 652/05 - Rn. 32 , 40 , BAGE 122 , 74`(RS)

**Example 42** (doc_id: `53763`) (sent_id: `53763`)


Wird als Verfahrensmangel gerügt , das FG habe einen gestellten Beweisantrag übergangen ( Rüge mangelnder Sachaufklärung gemäß § 76 Abs. 1 Satz 1 FGO ) , so ist darzulegen , welche Tatfrage aufklärungsbedürftig ist , welche Beweismittel das FG zu welchem Beweisthema nicht erhoben hat , die genauen Fundstellen ( Schriftsatz mit Datum und Seitenzahl , Terminprotokoll ) , in denen die Beweismittel und Beweisthemen angeführt worden sind , das voraussichtliche Ergebnis der Beweisaufnahme , inwiefern das Urteil des FG aufgrund dessen sachlich-rechtlicher Auffassung auf der unterbliebenen Beweisaufnahme beruhen kann und dass die Nichterhebung der Beweise vor dem FG rechtzeitig gerügt worden ist oder aufgrund des Verhaltens des FG nicht mehr vor diesem gerügt werden konnte ( ständige BFH-Rechtsprechung , z.B. Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85 , und vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125 , m. w. N. ) .

**False Positives:**

- `FG` — similar text (different position): `§ 76 Abs. 1 Satz 1 FGO`
- `FG` — similar text (different position): `§ 76 Abs. 1 Satz 1 FGO`
- `FG` — similar text (different position): `§ 76 Abs. 1 Satz 1 FGO`
- `FG` — similar text (different position): `§ 76 Abs. 1 Satz 1 FGO`
- `BFH` — similar text (different position): `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`
- `BFH` — similar text (different position): `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`

> overlaps gold: 6  |  likely missing annotation: 0

**Gold Entities:**

- `§ 76 Abs. 1 Satz 1 FGO`(NRM)
- `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`(RS)
- `vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125`(RS)

**Example 43** (doc_id: `53775`) (sent_id: `53775`)


nicht nur das erstmalige Legen eines Hausanschlusses , sondern auch Arbeiten zur Erneuerung oder zur Reduzierung von Wasseranschlüssen unter die Steuerermäßigung fallen ( vgl. BGH-Urteil in HFR 2012 , 1110 , Rz 20 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH-Urteil in HFR 2012 , 1110 , Rz 20`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH-Urteil in HFR 2012 , 1110 , Rz 20`(RS)

**Example 44** (doc_id: `53783`) (sent_id: `53783`)


3. Die Übertragung der Kostenentscheidung auf das FG beruht auf § 143 Abs. 2 FGO .

**False Positives:**

- `FG` — similar text (different position): `§ 143 Abs. 2 FGO`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 143 Abs. 2 FGO`(NRM)

**Example 45** (doc_id: `53792`) (sent_id: `53792`)


Dem vom FG festgestellten Sachverhalt lassen sich weder hinreichende Anhaltspunkte für eine überobligationsmäßige Leistung des Klägers zu 1. noch dafür entnehmen , dass der Kläger zu 1. bei Hingabe der Leistungen an E erkennbar die Absicht hatte , von E Ersatz zu verlangen .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `E`(PER)
- `E`(PER)

**Example 46** (doc_id: `53807`) (sent_id: `53807`)


Das angefochtene LSG-Urteil ist aufzuheben , weil es auf der Verletzung materiellen Rechts beruht und sich nicht aus anderen Gründen als richtig erweist .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 47** (doc_id: `53811`) (sent_id: `53811`)


Diese Grundsätze gelten ebenso für die Anwendung der hergebrachten Grundsätze des Berufsbeamtentums im Sinne des Art. 33 Abs. 5 GG ( BVerfG , Kammerbeschluss vom 7. Oktober 2015 - 2 BvR 413/15 - NVwZ 2016 , 56 Rn. 24 f. m. w. N. ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Kammerbeschluss vom 7. Oktober 2015 - 2 BvR 413/15 - NVwZ 2016 , 56 Rn. 24 f.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 33 Abs. 5 GG`(NRM)
- `BVerfG , Kammerbeschluss vom 7. Oktober 2015 - 2 BvR 413/15 - NVwZ 2016 , 56 Rn. 24 f.`(RS)

**Example 48** (doc_id: `53820`) (sent_id: `53820`)


Wenn es - wie vorliegend - an einer ausdrücklichen Sonderzuweisung für den zuständigen Rechtsweg fehlt , bestimmt sich die gerichtliche Zuständigkeit nach der Natur des Rechtsverhältnisses , aus dem der Klageanspruch hergeleitet wird ( stRspr ; Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2 ; GmSOGB vom 10. 4. 1986 - GmS-OGB 1/85 - BGHZ 97 , 312 = SozR 1500 § 51 Nr 39 ; GmSOGB vom 29. 10. 1987 - GmS-OGB 1/86 - BGHZ 102 , 280 = SozR 1500 § 51 Nr 47 ; GmSOGB vom 10. 7. 1989 - GmS-OGB 1/88 - BGHZ 108 , 284 = SozR 1500 § 51 Nr 53 ; zum Rechtsverhältnis zwischen den Beteiligten als entscheidendes Kriterium zur Beurteilung des Rechtswegs vgl letztens etwa BSG vom 21. 7. 2014 - B 14 SF 1/14 R - SozR 4 - 1500 § 51 Nr 12 RdNr 8 ; BSG vom 25. 10. 2017 - B 7 SF 1/16 R - juris , RdNr 6 ) .

**False Positives:**

- `BSG` — similar text (different position): `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`
- `BSG` — similar text (different position): `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`(RS)
- `GmSOGB vom 10. 4. 1986 - GmS-OGB 1/85 - BGHZ 97 , 312 = SozR 1500 § 51 Nr 39`(RS)
- `GmSOGB vom 29. 10. 1987 - GmS-OGB 1/86 - BGHZ 102 , 280 = SozR 1500 § 51 Nr 47`(RS)
- `GmSOGB vom 10. 7. 1989 - GmS-OGB 1/88 - BGHZ 108 , 284 = SozR 1500 § 51 Nr 53`(RS)
- `BSG vom 21. 7. 2014 - B 14 SF 1/14 R - SozR 4 - 1500 § 51 Nr 12 RdNr 8`(RS)
- `BSG vom 25. 10. 2017 - B 7 SF 1/16 R - juris , RdNr 6`(RS)

**Example 49** (doc_id: `53825`) (sent_id: `53825`)


Die Beteiligung an der S-GmbH gehörte zum Betriebsvermögen des BgA .

**False Positives:**

- `BgA` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `S-GmbH`(ORG)

**Example 50** (doc_id: `53843`) (sent_id: `53843`)


Da die Begriffsdefinitionen des Reisekostenrechts auf die dienstliche Tätigkeit von Beamten zugeschnitten sind und nicht ohne Weiteres davon ausgegangen werden kann , dass sie in jeder Hinsicht mit Normen und Grundsätzen des Personalvertretungsrechts im Einklang stehen , gebietet § 45 Abs. 1 Satz 2 SächsPersVG die entsprechende Anwendung des § 1 Abs. 2 SächsRKG und der dort in Bezug genommenen Bestimmungen ( vgl. BVerwG , Beschlüsse vom 25. November 2004 - 6 P 6.04 - Buchholz 251.7 § 40 NWPersVG Nr. 3 S. 5 ; vom 21. Mai 2007 - 6 P 5.06 - Buchholz 251.5 § 42 HePersVG Nr. 1 Rn. 24 und vom 28. November 2012 - 6 P 3.12 - Buchholz 262 § 9 TGV Nr. 1 Rn. 15 m. w. N. , vgl. auch SächsLT- Drs. 5/4071 ) .

**False Positives:**

- `BVerwG` — partial — pred is substring of gold: `BVerwG , Beschlüsse vom 25. November 2004 - 6 P 6.04 - Buchholz 251.7 § 40 NWPersVG Nr. 3 S. 5`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 45 Abs. 1 Satz 2 SächsPersVG`(NRM)
- `§ 1 Abs. 2 SächsRKG`(NRM)
- `BVerwG , Beschlüsse vom 25. November 2004 - 6 P 6.04 - Buchholz 251.7 § 40 NWPersVG Nr. 3 S. 5`(RS)
- `vom 21. Mai 2007 - 6 P 5.06 - Buchholz 251.5 § 42 HePersVG Nr. 1 Rn. 24`(RS)
- `vom 28. November 2012 - 6 P 3.12 - Buchholz 262 § 9 TGV Nr. 1 Rn. 15`(RS)
- `SächsLT- Drs. 5/4071`(LIT)

**Example 51** (doc_id: `53855`) (sent_id: `53855`)


aa ) In organisatorischer Hinsicht ist beim Bundesverfassungsgericht , anders als bei den Fachgerichten , eine Kapazitätsausweitung zur Verkürzung der Verfahrensdauer als Reaktion auf gesteigerte Eingangszahlen grundsätzlich nicht möglich , da die Struktur des Gerichts durch seine Funktion bedingt und durch die Verfassung und das Bundesverfassungsgerichtsgesetz vorgegeben ist ( BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21 ; vgl. BTDrucks 17/3802 , S. 26 ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesverfassungsgericht`(ORG)
- `Bundesverfassungsgerichtsgesetz`(NRM)
- `BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21`(RS)
- `BTDrucks 17/3802 , S. 26`(LIT)

**Example 52** (doc_id: `53874`) (sent_id: `53874`)


Bei einem in den Jahren 2008 bis 2010 maßgebenden , auf der Mindestbeitragsbemessungsgrundlage für freiwillig Versicherte errechneten ( einheitlichen ) Rentenversicherungsbeitrag in Höhe von 79,80 Euro monatlich erfüllten die vom LSG festgestellten monatlichen Prämienzahlungen des Klägers für seine zum 60. bzw 65. Lebensjahr ablaufenden Lebensversicherungen die og Kriterien .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 53** (doc_id: `53883`) (sent_id: `53883`)


Denn ungeachtet von Differenzen im Einzelnen verlangt die in beiden Vorschriften enthaltene Verantwortbarkeitsklausel eine Wahrscheinlichkeitsprognose für eine Legalbewährung in Freiheit , wobei die Anforderungen an die Aussicht auf künftige Straffreiheit umso höher anzusetzen sind , je schwerer die in Betracht kommenden Taten wiegen ( zu den rechtlichen Maßstäben des § 88 Abs. 1 JGG vgl. OLG Karlsruhe , Beschluss vom 24. Juli 2006 - 3 Ws 213/06 , StV 2007 , 12 , 13 ; Brunner / Dölling , JGG , 13. Aufl. , § 88 Rn. 5 ; HK-JGG / Kern , 2. Aufl. , § 88 Rn. 26 mwN ; zu den rechtlichen Maßstäben des § 57 Abs. 1 Satz 1 Nr. 2 , Satz 2 StGB vgl. BGH , Beschlüsse vom 25. April 2003 - StB 4/03 , BGHR StGB § 57 Abs. 1 Erprobung 2 ; vom 4. Oktober 2011 - StB 14/11 , NStZ-RR 2012 , 8 ; vom 10. April 2014 - StB 4/14 , juris Rn. 3 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , Beschlüsse vom 25. April 2003 - StB 4/03 , BGHR StGB § 57 Abs. 1 Erprobung 2`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 88 Abs. 1 JGG`(NRM)
- `OLG Karlsruhe , Beschluss vom 24. Juli 2006 - 3 Ws 213/06 , StV 2007 , 12 , 13`(RS)
- `Brunner / Dölling , JGG , 13. Aufl. , § 88 Rn. 5`(LIT)
- `HK-JGG / Kern , 2. Aufl. , § 88 Rn. 26`(LIT)
- `§ 57 Abs. 1 Satz 1 Nr. 2 , Satz 2 StGB`(NRM)
- `BGH , Beschlüsse vom 25. April 2003 - StB 4/03 , BGHR StGB § 57 Abs. 1 Erprobung 2`(RS)
- `vom 4. Oktober 2011 - StB 14/11 , NStZ-RR 2012 , 8`(RS)
- `10. April 2014 - StB 4/14 , juris Rn. 3`(RS)

**Example 54** (doc_id: `53888`) (sent_id: `53888`)


Das LSG hat die Berufung des Klägers gegen das Urteil des SG zu Unrecht zurückgewiesen .

**False Positives:**

- `LSG` — no gold match — likely missing annotation
- `SG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 55** (doc_id: `53891`) (sent_id: `53891`)


Sollte der Kläger mit seinen Fragestellungen die Schlussfolgerungen des LSG aus der zitierten Senatsrechtsprechung bezogen auf seinen Einzelfall in Frage stellen wollen , wendet er sich gegen die Unrichtigkeit der Rechtsanwendung in seinem Einzelfall .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 56** (doc_id: `53898`) (sent_id: `53898`)


Das gilt jedenfalls uneingeschränkt für das Elterngeld als fürsorgerische Leistung der Familienförderung , die über die bloße Sicherung des Existenzminimums hinausgeht ( zum Elterngeld vgl BVerfG < Kammer > Beschluss vom 9. 11. 2011 - 1 BvR 1853/11 - BVerfGK 19 , 186 ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG < Kammer > Beschluss vom 9. 11. 2011 - 1 BvR 1853/11 - BVerfGK 19 , 186`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG < Kammer > Beschluss vom 9. 11. 2011 - 1 BvR 1853/11 - BVerfGK 19 , 186`(RS)

**Example 57** (doc_id: `53910`) (sent_id: `53910`)


Zur Vermeidung einer mittelbaren Diskriminierung wegen Behinderung sei er nach der Entscheidung des Gerichtshofs der Europäischen Union ( EuGH ) vom 6. Dezember 2012 ( - C- 152/11 - [ Odar ] ) wie ein nicht schwerbehinderter Arbeitnehmer zu behandeln .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `Entscheidung des Gerichtshofs der Europäischen Union ( EuGH ) vom 6. Dezember 2012 ( - C- 152/11 - [ Odar ] )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Entscheidung des Gerichtshofs der Europäischen Union ( EuGH ) vom 6. Dezember 2012 ( - C- 152/11 - [ Odar ] )`(RS)

**Example 58** (doc_id: `53933`) (sent_id: `53933`)


Weder die Studie der Holy Fashion Group vom 24. Februar 2016 ( Bl. 49/50 d. A. ) noch die im Amtsverfahren vorgelegten Unterlagen zu Marktforschungsergebnissen einer „ Brigitte “ -Studie , Internetausdrucken zu Showrooms und Verkaufsstätten von Waren der Marke „ JOOP “ oder die beigefügten Urteile sind geeignet , einen entsprechenden Benutzungsnachweis für mit der Marke gekennzeichnete Dienstleistungen zu erbringen .

**False Positives:**

- `JOOP` — partial — pred is substring of gold: `„ JOOP “`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Holy Fashion Group`(ORG)
- `„ JOOP “`(ORG)

**Example 59** (doc_id: `53937`) (sent_id: `53937`)


Da allein das Fehlen jeglicher Unterscheidungskraft ein Eintragungshindernis begründet , ist ein großzügiger Maßstab anzulegen , so dass jede auch noch so geringe Unterscheidungskraft genügt , um das Schutzhindernis zu überwinden ( BGH a. a. O. – OUI ; a. a. O. – for you ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH a. a. O. – OUI`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH a. a. O. – OUI`(RS)
- `a. a. O. – for you`(RS)

**Example 60** (doc_id: `53981`) (sent_id: `53981`)


Dies ergibt sich z.B. aus der Begründung zum Entwurf der Bundesregierung eines Gesetzes zur Einführung einer Musterwiderrufsinformation für Verbraucherdarlehensverträge , zur Änderung der Vorschriften über das Widerrufsrecht bei Verbraucherdarlehensverträgen und zur Änderung des Darlehensvermittlungsrechts vom 29. 4. 2010 ( BT-Drs. 17/1394 , S. 15 ) .

**False Positives:**

- `Bundesregierung` — partial — pred is substring of gold: `Begründung zum Entwurf der Bundesregierung eines Gesetzes zur Einführung einer Musterwiderrufsinformation für Verbraucherdarlehensverträge , zur Änderung der Vorschriften über das Widerrufsrecht bei Verbraucherdarlehensverträgen und zur Änderung des Darlehensvermittlungsrechts vom 29. 4. 2010 ( BT-Drs. 17/1394 , S. 15 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Begründung zum Entwurf der Bundesregierung eines Gesetzes zur Einführung einer Musterwiderrufsinformation für Verbraucherdarlehensverträge , zur Änderung der Vorschriften über das Widerrufsrecht bei Verbraucherdarlehensverträgen und zur Änderung des Darlehensvermittlungsrechts vom 29. 4. 2010 ( BT-Drs. 17/1394 , S. 15 )`(LIT)

**Example 61** (doc_id: `53982`) (sent_id: `53982`)


Der Rundfunkbeitrag wird erhoben , um den individuellen Nutzungsvorteil abzugelten ( BVerwG , Urteil vom 18. März 2016 - 6 C 6.15 - BVerwGE 154 , 275 Rn. 25 ff. ) .

**False Positives:**

- `BVerwG` — partial — pred is substring of gold: `BVerwG , Urteil vom 18. März 2016 - 6 C 6.15 - BVerwGE 154 , 275 Rn. 25 ff.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerwG , Urteil vom 18. März 2016 - 6 C 6.15 - BVerwGE 154 , 275 Rn. 25 ff.`(RS)

**Example 62** (doc_id: `53995`) (sent_id: `53995`)


Sind beide Anträge - wie hier - Gegenstand desselben Rechtsstreits , kann über sie gleichzeitig verhandelt und entschieden werden ( vgl. BAG 27. April 2006 - 2 AZR 360/05 - Rn. 19 , aaO ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 27. April 2006 - 2 AZR 360/05 - Rn. 19 , aaO`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 27. April 2006 - 2 AZR 360/05 - Rn. 19 , aaO`(RS)

**Example 63** (doc_id: `54027`) (sent_id: `54027`)


Dies gilt auch in den Fällen einer Erkrankung mit einer nur noch begrenzten Lebenserwartung , da die Regelung des § 64 Abs. 1 Satz 1 Nr. 1 EStDV i. d. F. des StVereinfG 2011 keine Differenzierung zwischen verschiedenen Krankheitskosten enthält ( BFH-Urteil vom 25. April 2017 VIII R 52/13 , BFHE 258 , 53 , BStBl II 2017 , 949 ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH-Urteil vom 25. April 2017 VIII R 52/13 , BFHE 258 , 53 , BStBl II 2017 , 949`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 64 Abs. 1 Satz 1 Nr. 1 EStDV`(NRM)
- `StVereinfG 2011`(NRM)
- `BFH-Urteil vom 25. April 2017 VIII R 52/13 , BFHE 258 , 53 , BStBl II 2017 , 949`(RS)

**Example 64** (doc_id: `54030`) (sent_id: `54030`)


Die vermeintliche Unrichtigkeit einer Entscheidung des Berufungsgerichts eröffnet aber nicht die Revisionsinstanz ( vgl BSG SozR 1500 § 160a Nr 7 ) .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG SozR 1500 § 160a Nr 7`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG SozR 1500 § 160a Nr 7`(RS)

**Example 65** (doc_id: `54033`) (sent_id: `54033`)


Insbesondere lässt sich den ungerügt gebliebenen Feststellungen des LSG nicht entnehmen , dass der gemeinsamen Nutzung der Wohnung durch die Kläger und ihren Sohn bindende vertragliche Regelungen zwischen den Familienmitgliedern zugrunde lagen , die für eine Abweichung vom Kopfteilprinzip streiten könnten .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 66** (doc_id: `54041`) (sent_id: `54041`)


3.1 Zunächst hat der Gerichtshof im Hinblick auf die Auslegungskriterien zu Art. 3 ( a ) AMVO festgestellt , dass es unzulässig ist , ein ergänzendes Schutzzertifikat für solche Wirkstoffe zu erteilen , die in den Ansprüchen des Grundpatents nicht genannt sind ( EuGH , GRUR 2012 , 257 , Rnd. 28 – Medeva ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH , GRUR 2012 , 257 , Rnd. 28 – Medeva`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 3 ( a ) AMVO`(NRM)
- `EuGH , GRUR 2012 , 257 , Rnd. 28 – Medeva`(RS)

**Example 67** (doc_id: `54059`) (sent_id: `54059`)


Wie der Kläger selbst vorträgt , wollte sich das LSG vielmehr nach seinem eigenen Verständnis im Rahmen der bereits vorliegenden Rechtsprechung des BSG halten , ist diesen aber in dem von ihm entschiedenen Fall lediglich deshalb nicht gefolgt , weil es - vom Kläger als unzutreffend gerügte - wesentliche Sachverhaltsunterschiede zu den bereits entschiedenen Fällen angenommen hat .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG`(ORG)

**Example 68** (doc_id: `54066`) (sent_id: `54066`)


Dies folgt bei vorausschauender Betrachtung bereits daraus , dass die Arbeitsvertragsparteien nach den Feststellungen des LSG jeweils lediglich nur den einen streitigen Arbeitseinsatz vereinbart hatten und den Beigeladenen zu 1. und 3. keine weiteren Arbeitseinsätze in Aussicht gestellt worden waren .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 69** (doc_id: `54100`) (sent_id: `54100`)


Eine mögliche Straßenglätte war nach den Feststellungen des LSG nicht unvorhersehbar , ua weil am Vortag eine entsprechende Meldung mit einer Warnung vor Glätte für den folgenden Tag erfolgt war , sodass es bereits deshalb an einem unerwarteten Ereignis fehlte .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 70** (doc_id: `54121`) (sent_id: `54121`)


Die Grundrechtsbindung aus Art. 12 Abs. 1 GG besteht jedoch dann , wenn Normen , die zwar selbst die Berufstätigkeit nicht unmittelbar berühren , aber Rahmenbedingungen der Berufsausübung verändern , in ihrer Zielsetzung und ihren mittelbar-faktischen Wirkungen einem Eingriff als funktionales Äquivalent gleichkommen ( vgl. BVerfGE 105 , 252 < 273 > ; 105 , 279 < 303 > ; 110 , 177 < 191 > ; 113 , 63 < 76 > ; 116 , 135 < 153 > ; 116 , 202 < 222 > ; 118 , 1 < 20 > ; s. auch BVerfG , Beschluss der 2. Kammer des Ersten Senats vom 25. Juli 2007 - 1 BvR 1031/07 - , juris , Rn. 32 ) , die mittelbaren Folgen also kein bloßer Reflex einer nicht entsprechend ausgerichteten gesetzlichen Regelung sind ( vgl. BVerfGE 106 , 275 < 299 > ; BVerfGE 116 , 202 < 222 > m. w. N. ) .

**False Positives:**

- `BVerfG` — similar text (different position): `BVerfGE 105 , 252 < 273 > ; 105 , 279 < 303 > ; 110 , 177 < 191 > ; 113 , 63 < 76 > ; 116 , 135 < 153 > ; 116 , 202 < 222 > ; 118 , 1 < 20 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 12 Abs. 1 GG`(NRM)
- `BVerfGE 105 , 252 < 273 > ; 105 , 279 < 303 > ; 110 , 177 < 191 > ; 113 , 63 < 76 > ; 116 , 135 < 153 > ; 116 , 202 < 222 > ; 118 , 1 < 20 >`(RS)
- `BVerfG , Beschluss der 2. Kammer des Ersten Senats vom 25. Juli 2007 - 1 BvR 1031/07 - , juris , Rn. 32`(RS)
- `BVerfGE 106 , 275 < 299 >`(RS)
- `BVerfGE 116 , 202 < 222 >`(RS)

**Example 71** (doc_id: `54133`) (sent_id: `54133`)


aa ) Ergibt sich aus dem Vortrag der Parteien im Rechtsstreit , dass die normative Wirkung eines Tarifvertrags nach § 4 Abs. 1 , § 5 Abs. 4 TVG in Betracht kommt , muss das Gericht diese Normen nach § 293 ZPO von Amts wegen ermitteln ( BAG 25. Januar 2017 - 4 AZR 520/15 - Rn. 29 mwN ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 25. Januar 2017 - 4 AZR 520/15 - Rn. 29`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 4 Abs. 1 , § 5 Abs. 4 TVG`(NRM)
- `§ 293 ZPO`(NRM)
- `BAG 25. Januar 2017 - 4 AZR 520/15 - Rn. 29`(RS)

**Example 72** (doc_id: `54143`) (sent_id: `54143`)


Dieser während des Revisionsverfahrens eingetretene Zuständigkeitswechsel führt zu einem gesetzlichen Beteiligtenwechsel ( vgl. z.B. Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109 , m. w. N. ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109`(RS)

**Example 73** (doc_id: `54150`) (sent_id: `54150`)


Die Prüfung der Markenanmeldung muss daher nach der maßgeblichen Rechtsprechung des EuGH grundsätzlich streng und vollständig sein , um ungerechtfertigte Eintragungen zu vermeiden ( vgl. EuGH , GRUR 2003 , 604 Rn. 57 , 60 – Libertel ; BGH , GRUR 2014 , 565 Rn. 17 – smartbook ; Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 158 , 159 ) .

**False Positives:**

- `EuGH` — similar text (different position): `EuGH`
- `BGH` — partial — pred is substring of gold: `BGH , GRUR 2014 , 565 Rn. 17 – smartbook`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH`(ORG)
- `EuGH , GRUR 2003 , 604 Rn. 57 , 60 – Libertel`(RS)
- `BGH , GRUR 2014 , 565 Rn. 17 – smartbook`(RS)
- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 158 , 159`(LIT)

**Example 74** (doc_id: `54155`) (sent_id: `54155`)


Gemäß § 1 des am 5. November 1979 geschlossenen Tarifvertrags zur Regelung der Grundlagen einer kirchengemäßen Tarifpartnerschaft ( GVOBl. NEK 1980 S. 12 ) besteht zwischen den Tarifvertragsparteien für die Dauer des Tarifvertrags eine absolute Friedenspflicht .

**False Positives:**

- `NEK` — partial — pred is substring of gold: `§ 1 des am 5. November 1979 geschlossenen Tarifvertrags zur Regelung der Grundlagen einer kirchengemäßen Tarifpartnerschaft ( GVOBl. NEK 1980 S. 12 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 1 des am 5. November 1979 geschlossenen Tarifvertrags zur Regelung der Grundlagen einer kirchengemäßen Tarifpartnerschaft ( GVOBl. NEK 1980 S. 12 )`(REG)

**Example 75** (doc_id: `54160`) (sent_id: `54160`)


aa ) Die Jahressonderzuwendung hat - ähnlich wie die Jahressonderzahlung nach § 20 TV-L / TVöD ( vgl. dazu BAG 12. Dezember 2012 - 10 AZR 922/11 - Rn. 20 , BAGE 144 , 117 ) - mehrere erkennbare Zwecke .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 12. Dezember 2012 - 10 AZR 922/11 - Rn. 20 , BAGE 144 , 117`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 20 TV-L / TVöD`(REG)
- `BAG 12. Dezember 2012 - 10 AZR 922/11 - Rn. 20 , BAGE 144 , 117`(RS)

**Example 76** (doc_id: `54169`) (sent_id: `54169`)


Unabhängig von der Frage , ob ein solcher Antrag - entgegen der Auffassung des Antragstellers - gemäß § 67 Abs. 4 VwGO dem Vertretungszwang unterliegt ( vgl. verneinend z.B. : BVerwG , Beschluss vom 6. Juni 1972 - 3 ER 404.71 - Buchholz 310 § 53 VwGO Nr. 6 ; Ziekow , in : Sodan / Ziekow , VwGO , 4. Aufl. 2014 , § 53 Rn. 20 ; Kraft , in : Eyermann , VwGO , 14. Aufl. 2014 , § 53 Rn. 14 ; Unruh , in : HK-Verwaltungsrecht , 3. Aufl. 2013 , § 53 Rn. 15 ; Scheidler , VR 2012 , 113 ; bejahend z.B. : W. - R. Schenke , in : Kopp / Schenke , VwGO , 23. Aufl. 2017 , § 53 Rn. 10 ; Redeker / von Oertzen , VwGO , 15. Aufl. 2010 , § 53 Rn. 5 ; vermittelnd z.B. : Bier , in : Schoch / Schneider / Bier , VwGO , Stand Oktober 2016 , § 53 Rn. 13 ) , ist der Antrag hier jedenfalls deshalb unzulässig , weil die gesetzlichen Voraussetzungen , unter denen das Bundesverwaltungsgericht eine Zuständigkeitsbestimmung überhaupt vornehmen darf , nicht erfüllt sind .

**False Positives:**

- `BVerwG` — partial — pred is substring of gold: `BVerwG , Beschluss vom 6. Juni 1972 - 3 ER 404.71 - Buchholz 310 § 53 VwGO Nr. 6`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 67 Abs. 4 VwGO`(NRM)
- `BVerwG , Beschluss vom 6. Juni 1972 - 3 ER 404.71 - Buchholz 310 § 53 VwGO Nr. 6`(RS)
- `Ziekow , in : Sodan / Ziekow , VwGO , 4. Aufl. 2014 , § 53 Rn. 20`(LIT)
- `Kraft , in : Eyermann , VwGO , 14. Aufl. 2014 , § 53 Rn. 14`(LIT)
- `Unruh , in : HK-Verwaltungsrecht , 3. Aufl. 2013 , § 53 Rn. 15`(LIT)
- `Scheidler , VR 2012 , 113`(LIT)
- `W. - R. Schenke , in : Kopp / Schenke , VwGO , 23. Aufl. 2017 , § 53 Rn. 10`(LIT)
- `Redeker / von Oertzen , VwGO , 15. Aufl. 2010 , § 53 Rn. 5`(LIT)
- `Bier , in : Schoch / Schneider / Bier , VwGO , Stand Oktober 2016 , § 53 Rn. 13`(LIT)
- `Bundesverwaltungsgericht`(ORG)

**Example 77** (doc_id: `54171`) (sent_id: `54171`)


Ein anderer als der vom LSG herangezogene Prüfungsmaßstab unter Anwendung weiterer Vorschriften des Bundesrechts folgt entgegen der Rechtsauffassung der Beklagten auch nicht aus einem Beschluss des Senats , in dem die Revision gegen ein Urteil des LSG Baden-Württemberg vom 23. 1. 2013 ( L 5 R 4971/10 ) als unzulässig verworfen wurde ( Senatsbeschluss vom 6. 2. 2014 - B 5 RE 10/14 R ) .

**False Positives:**

- `LSG` — similar text (different position): `Urteil des LSG Baden-Württemberg vom 23. 1. 2013 ( L 5 R 4971/10 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des LSG Baden-Württemberg vom 23. 1. 2013 ( L 5 R 4971/10 )`(RS)
- `Senatsbeschluss vom 6. 2. 2014 - B 5 RE 10/14 R`(RS)

**Example 78** (doc_id: `54186`) (sent_id: `54186`)


dd ) Auf dieser Grundlage hat das FG im Ergebnis zu Recht entschieden , dass eine Ablaufhemmung für die Streitjahre gemäß § 171 Abs. 4 Satz 1 AO eingetreten ist .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 171 Abs. 4 Satz 1 AO`(NRM)

**Example 79** (doc_id: `54202`) (sent_id: `54202`)


Eine Vertragslücke , die einer Schließung durch den Rückgriff auf dispositives Gesetzesrecht oder einer ergänzenden Vertragsauslegung bedurft hätte ( BAG 15. Dezember 2016 - 6 AZR 478/15 - Rn. 31 , BAGE 157 , 284 ) , bestand nicht .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 15. Dezember 2016 - 6 AZR 478/15 - Rn. 31 , BAGE 157 , 284`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 15. Dezember 2016 - 6 AZR 478/15 - Rn. 31 , BAGE 157 , 284`(RS)

**Example 80** (doc_id: `54208`) (sent_id: `54208`)


Hierdurch ist das Verfahren über das Ablehnungsgesuch abgeschlossen worden , denn erst zu diesem Zeitpunkt war das Gericht an seine Entscheidung gebunden ( vgl. auch Beschluss des Bundesfinanzhofs - BFH - vom 17. Juli 2008 I B 22/08 , juris , unter II. 2. zum Abschluss des Ablehnungsverfahrens im Zeitpunkt der Absendung der Entscheidung durch die Geschäftsstelle ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `Beschluss des Bundesfinanzhofs - BFH - vom 17. Juli 2008 I B 22/08 , juris , unter II. 2.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Beschluss des Bundesfinanzhofs - BFH - vom 17. Juli 2008 I B 22/08 , juris , unter II. 2.`(RS)

**Example 81** (doc_id: `54212`) (sent_id: `54212`)


Ist es dem Kläger im Rahmen seiner deshalb nötigen Ermittlungen aufgrund des Verhaltens des FG-Präsidenten nicht möglich , diesen Verfahrensmangel zu substantiieren , so hat dies allein zur Folge , dass der BFH insoweit einen geringeren Maßstab der Darlegung des Verfahrensmangels genügen lassen muss .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BFH`(ORG)

**Example 82** (doc_id: `54213`) (sent_id: `54213`)


( c ) Zu den Wertverhältnissen gehören nach der im Verfahren der Normenkontrolle grundsätzlich bindenden Auffassung der Fachgerichte schließlich auch Miet- und Belegungsbindungen aufgrund einer öffentlichen Förderung des Wohnungsbaus ( Vorlagebeschluss vom 17. Dezember 2014 - II R 14/13 - , juris , Rn. 15 in dem Verfahren 1 BvL 1/15 unter Bezugnahme auf die BFH-Urteile vom 26. Juli 1989 - II R 65/86 - , BFHE 158 , 87 , und vom 5. Mai 1993 - II R 71/90 - ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH-Urteile vom 26. Juli 1989 - II R 65/86 - , BFHE 158 , 87`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Vorlagebeschluss vom 17. Dezember 2014 - II R 14/13 - , juris , Rn. 15 in dem Verfahren 1 BvL 1/15`(RS)
- `BFH-Urteile vom 26. Juli 1989 - II R 65/86 - , BFHE 158 , 87`(RS)
- `vom 5. Mai 1993 - II R 71/90 -`(RS)

**Example 83** (doc_id: `54217`) (sent_id: `54217`)


Denn die Hauptfunktion einer Marke liegt darin , die Ursprungsidentität der gekennzeichneten Waren und Dienstleistungen zu gewährleisten ( vgl. BGH , GRUR 2014 , 569 Rn. 10 – HOT ; GRUR 2013 , 731 Rn. 11 – Kaleido ; GRUR 2012 , 1143 Rn. 7 – Starsat ; GRUR 2012 , 270 Rn. 8 – Link economy ; GRUR 2010 , 1100 Rn. 10 – TOOOR ! ; GRUR 2010 , 825 Rn. 13 – Marlene-Dietrich-Bildnis II ; GRUR 2006 , 850 , 854 Rn. 18 – FUSSBALL WM 2006 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , GRUR 2014 , 569 Rn. 10 – HOT`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , GRUR 2014 , 569 Rn. 10 – HOT`(RS)
- `GRUR 2013 , 731 Rn. 11 – Kaleido`(RS)
- `GRUR 2012 , 1143 Rn. 7 – Starsat`(RS)
- `GRUR 2012 , 270 Rn. 8 – Link economy`(RS)
- `GRUR 2010 , 1100 Rn. 10 – TOOOR !`(RS)
- `GRUR 2010 , 825 Rn. 13 – Marlene-Dietrich-Bildnis II`(RS)
- `GRUR 2006 , 850 , 854 Rn. 18 – FUSSBALL WM 2006`(RS)

**Example 84** (doc_id: `54228`) (sent_id: `54228`)


Da allein das Fehlen jeglicher Unterscheidungskraft ein Eintragungshindernis begründet , ist ein großzügiger Maßstab anzulegen , so dass jede auch noch so geringe Unterscheidungskraft genügt , um das Schutzhindernis zu überwinden ( BGH a. a. O. – OUI ; a. a. O. – for you ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH a. a. O. – OUI`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH a. a. O. – OUI`(RS)
- `a. a. O. – for you`(RS)

**Example 85** (doc_id: `54237`) (sent_id: `54237`)


Es genügt , dass die schuldrechtliche Grundlage des Anspruchs vor Eröffnung entstanden ist ( BGH 22. September 2011 - IX ZB 121/11 - Rn. 3 ) , also ihr Rechtsgrund bei Eröffnung gelegt war ( BVerwG 26. Februar 2015 - 3 C 8.14 - Rn. 14 , BVerwGE 151 , 302 ; MüKoInsO / Ehricke 3. Aufl. § 38 Rn. 16 ; Uhlenbruck / Sinz 14. Aufl. § 38 InsO Rn. 30 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH 22. September 2011 - IX ZB 121/11 - Rn. 3`
- `BVerwG` — partial — pred is substring of gold: `BVerwG 26. Februar 2015 - 3 C 8.14 - Rn. 14 , BVerwGE 151 , 302`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH 22. September 2011 - IX ZB 121/11 - Rn. 3`(RS)
- `BVerwG 26. Februar 2015 - 3 C 8.14 - Rn. 14 , BVerwGE 151 , 302`(RS)
- `MüKoInsO / Ehricke 3. Aufl. § 38 Rn. 16`(LIT)
- `Uhlenbruck / Sinz 14. Aufl. § 38 InsO Rn. 30`(LIT)

**Example 86** (doc_id: `54255`) (sent_id: `54255`)


Die bundesrechtlich geforderte Zuweisung eines einheitlichen RLV an eine von mehreren Ärzten gebildete Arztpraxis ( BAG , MVZ ) hat zur Folge , dass innerhalb dieser Arztpraxis bei Beachtung der Fachgebietsgrenzen sowie qualifikationsgebundener Genehmigungen zur Leistungserbringung weitgehende Flexibilität herrscht .

**False Positives:**

- `BAG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Bundesverfassungsgericht Genitive`

**F1:** 0.112 | **Precision:** 0.906 | **Recall:** 0.059  

**Format:** `regex`  
**Rule ID:** `fdd983a0`  
**Description:**
Matches 'Bundesverfassungsgericht' and its genitive form 'Bundesverfassungsgerichts'.

**Content:**
```
\b(Bundesverfassungsgericht|Bundesverfassungsgerichts)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.906 | 0.059 | 0.112 | 53 | 48 | 5 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 48 | 5 | 756 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53402`) (sent_id: `53402`)


Bei offenem Ausgang des Verfassungsbeschwerdeverfahrens muss das Bundesverfassungsgericht die Folgen abwägen , die eintreten würden , wenn die einstweilige Anordnung nicht erginge , die Verfassungsbeschwerde aber Erfolg hätte , gegenüber den Nachteilen , die entstünden , wenn die begehrte einstweilige Anordnung erlassen würde , der Verfassungsbeschwerde aber der Erfolg zu versagen wäre ( vgl. BVerfGE 76 , 253 < 255 > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `BVerfGE 76 , 253 < 255 >` (RS)

**Example 1** (doc_id: `53403`) (sent_id: `53403`)


Vor diesem Hintergrund erweist sich schon die Auseinandersetzung des Beschwerdeführers mit den vom Bundesverfassungsgericht - wenn auch zu Art. 19 Abs. 4 GG , den der Beschwerdeführer nicht rügt - entwickelten verfassungsrechtlichen Maßstäben als unzureichend ; umso weniger ist unter diesen Umständen eine mögliche Willkür der angegriffenen Entscheidung plausibel dargelegt .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Art. 19 Abs. 4 GG` (NRM)

**Example 2** (doc_id: `53700`) (sent_id: `53700`)


3. Das Bundesverfassungsgericht überprüft die Vereinbarkeit eines nationalen Gesetzes mit dem Grundgesetz auch , wenn zugleich Zweifel an der Vereinbarkeit des Gesetzes mit Sekundärrecht der Europäischen Union bestehen .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Grundgesetz` (NRM)
- `Europäischen Union` (ORG)

**Example 3** (doc_id: `53855`) (sent_id: `53855`)


aa ) In organisatorischer Hinsicht ist beim Bundesverfassungsgericht , anders als bei den Fachgerichten , eine Kapazitätsausweitung zur Verkürzung der Verfahrensdauer als Reaktion auf gesteigerte Eingangszahlen grundsätzlich nicht möglich , da die Struktur des Gerichts durch seine Funktion bedingt und durch die Verfassung und das Bundesverfassungsgerichtsgesetz vorgegeben ist ( BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21 ; vgl. BTDrucks 17/3802 , S. 26 ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Bundesverfassungsgerichtsgesetz` (NRM)
- `BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21` (RS)
- `BTDrucks 17/3802 , S. 26` (LIT)

**Example 4** (doc_id: `54126`) (sent_id: `54126`)


1. Aus besonderem Grund , namentlich im Interesse einer verlässlichen Finanz- und Haushaltsplanung und eines gleichmäßigen Verwaltungsvollzugs für Zeiträume einer weitgehend schon abgeschlossenen Veranlagung , hat das Bundesverfassungsgericht wiederholt die weitere Anwendbarkeit verfassungswidriger Normen binnen der dem Gesetzgeber bis zu einer Neuregelung gesetzten Frist oder spätestens bis zur Neuregelung für gerechtfertigt erklärt ( vgl. etwa BVerfGE 87 , 153 < 178 > ; 93 , 121 < 148 f. > ; 123 , 1 < 38 > ; 125 , 175 < 258 > ; 138 , 136 < 251 Rn. 287 > ; 139 , 285 < 319 Rn. 89 > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `BVerfGE 87 , 153 < 178 > ; 93 , 121 < 148 f. > ; 123 , 1 < 38 > ; 125 , 175 < 258 > ; 138 , 136 < 251 Rn. 287 > ; 139 , 285 < 319 Rn. 89 >` (RS)

**Example 5** (doc_id: `54368`) (sent_id: `54368`)


Auf dieser Grundlage habe der Beschwerdeführer auch eine praktische Chance auf Wiedererlangung seiner Freiheit , wie sie das Bundesverfassungsgericht fordere .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 6** (doc_id: `54521`) (sent_id: `54521`)


2. Soweit der Beschwerdeführer weiter rügt , das Urteil des Amtsgerichts verletze ihn auch in seinem Grundrecht aus Art. 2 Abs. 1 i. V. m. Art. 20 Abs. 3 GG , gilt nach ständiger Rechtsprechung des Bundesverfassungsgerichts in Bezug auf § 511 Abs. 4 ZPO derselbe Prüfungsmaßstab wie unter III. 1. dargestellt .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `Art. 2 Abs. 1 i. V. m. Art. 20 Abs. 3 GG` (NRM)
- `§ 511 Abs. 4 ZPO` (NRM)

**Example 7** (doc_id: `54541`) (sent_id: `54541`)


III. Für die vom Kläger begehrte Aussetzung und Vorlage an das Bundesverfassungsgericht sieht der Senat keine Veranlassung .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 8** (doc_id: `54629`) (sent_id: `54629`)


Ob das Bundesverfassungsgericht hinsichtlich der Zulässigkeit von Auslieferungen nach Rumänien unter dem Gesichtspunkt der Einhaltung der sich aus dem Grundgesetz ergebenden Grundrechte Anforderungen an die Haftbedingungen stellen werde , die über diejenigen des Europäischen Gerichtshofs und der Europäischen Menschenrechtskonvention hinausgingen , stehe zurzeit nicht fest .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Rumänien` (LOC)
- `Grundgesetz` (NRM)
- `Europäischen Gerichtshofs` (ORG)
- `Europäischen Menschenrechtskonvention` (NRM)

**Example 9** (doc_id: `54837`) (sent_id: `54837`)


Dies entspricht dem vom Bundesverfassungsgericht in seiner Weitergeltungsanordnung vom 4. Mai 2011 aus Art. 2 Abs. 2 Satz 2 in Verbindung mit Art. 20 Abs. 3 GG abgeleiteten Maßstab ( vgl. BVerfGE 128 , 326 < 332 > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Art. 2 Abs. 2 Satz 2 in Verbindung mit Art. 20 Abs. 3 GG` (NRM)
- `BVerfGE 128 , 326 < 332 >` (RS)

**Example 10** (doc_id: `54935`) (sent_id: `54935`)


Diese Entscheidung kann von der Kammer getroffen werden , weil die maßgeblichen verfassungsrechtlichen Fragen durch das Bundesverfassungsgericht bereits entschieden und die Verfassungsbeschwerde hiernach offensichtlich begründet ist , § 93c Abs. 1 Satz 1 BVerfGG .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `§ 93c Abs. 1 Satz 1 BVerfGG` (NRM)

**Example 11** (doc_id: `55031`) (sent_id: `55031`)


a ) Eine Verfassungsbeschwerde ist nach ständiger Rechtsprechung des Bundesverfassungsgerichts ( vgl. grundlegend BVerfGE 90 , 22 < 24 f. > ) wegen grundsätzlicher verfassungsrechtlicher Bedeutung anzunehmen , wenn sie eine verfassungsrechtliche Frage aufwirft , die sich nicht ohne Weiteres aus dem Grundgesetz beantworten lässt und die noch nicht durch die verfassungsgerichtliche Rechtsprechung geklärt oder die durch veränderte Verhältnisse erneut klärungsbedürftig geworden ist .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `BVerfGE 90 , 22 < 24 f. >` (RS)
- `Grundgesetz` (NRM)

**Example 12** (doc_id: `55168`) (sent_id: `55168`)


a ) Zwar gehe das Bundesverfassungsgericht bislang in ständiger Rechtsprechung von einem Streikverbot für Beamte aus .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 13** (doc_id: `55230`) (sent_id: `55230`)


Dieses Ergebnis stehe in Übereinstimmung mit den vom Bundesverfassungsgericht anerkannten Referenzgruppen der kommunalen Wahlbeamten und der politischen Beamten .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 14** (doc_id: `55446`) (sent_id: `55446`)


Die für die Beurteilung der Verfassungsbeschwerde maßgeblichen verfassungsrechtlichen Fragen sind durch das Bundesverfassungsgericht bereits entschieden ( § 93c Abs. 1 Satz 1 BVerfGG ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `§ 93c Abs. 1 Satz 1 BVerfGG` (NRM)

**Example 15** (doc_id: `55466`) (sent_id: `55466`)


Zu berücksichtigen ist hierbei , dass vor dem Bundesverfassungsgericht regelmäßig - so auch hier - eine überschlägige Beurteilung der Sach- und Rechtslage für erledigt erklärter Verfassungsbeschwerden nicht stattfindet ( vgl. BVerfGE 33 , 247 < 264 f. > ; 85 , 109 < 115 f. > ; 87 , 394 < 397 f. > ) und auch keine der Fallgestaltungen vorliegt , in denen die Erfolgsaussichten der Verfassungsbeschwerde im Sinne des Beschwerdeführers vorhergesagt werden könnte ( vgl. BVerfGE 85 , 109 < 115 f. > ; 133 , 37 < 38 f. > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `BVerfGE 33 , 247 < 264 f. > ; 85 , 109 < 115 f. > ; 87 , 394 < 397 f. >` (RS)
- `BVerfGE 85 , 109 < 115 f. > ; 133 , 37 < 38 f. >` (RS)

**Example 16** (doc_id: `55591`) (sent_id: `55591`)


Insbesondere die durch Entscheidungen des Bundesverfassungsgerichts veranlassten Neuregelungen des Bewertungsgesetzes wurden nicht in die Vorschriften über die Einheitsbewertung eingearbeitet , sondern als Neuregelungen in eigenen Abschnitten in das Bewertungsgesetz eingefügt , ohne dass dabei die Bestimmungen über die Einheitsbewertung inhaltlich neu geformt worden wären .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `Bewertungsgesetzes` (NRM)
- `Bewertungsgesetz` (NRM)

**Example 17** (doc_id: `55695`) (sent_id: `55695`)


Der Beschwerdeführer hat daher ein fortbestehendes schutzwürdiges Interesse an einer nachträglichen verfassungsrechtlichen Überprüfung und gegebenenfalls einer hierauf bezogenen Feststellung der Verfassungswidrigkeit dieses Grundrechtseingriffs durch das Bundesverfassungsgericht ( vgl. BVerfGE 9 , 89 < 92 ff. > ; 32 , 87 < 92 > ; 53 , 152 < 157 f. > ; 91 , 125 < 133 > ; 104 , 220 < 234 f. > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `BVerfGE 9 , 89 < 92 ff. > ; 32 , 87 < 92 > ; 53 , 152 < 157 f. > ; 91 , 125 < 133 > ; 104 , 220 < 234 f. >` (RS)

**Example 18** (doc_id: `55740`) (sent_id: `55740`)


Das Verwaltungsgericht sei - im Einklang mit dem Internationalen Gerichtshof und dem Bundesverfassungsgericht - nicht von einer ausnahmslosen Völkerrechtswidrigkeit der Androhung und / oder des Einsatzes von Atomwaffen ausgegangen .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Internationalen Gerichtshof` (ORG)

**Example 19** (doc_id: `55815`) (sent_id: `55815`)


Im Rahmen der Prüfung , ob die Klägerin diese Zweifel ausräumen konnte , hat das Landesarbeitsgericht der Klägerin rechtliches Gehör nach Maßgabe der Rechtsprechung des Bundesverfassungsgerichts und des Bundesarbeitsgerichts gewährt .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `Bundesarbeitsgerichts` (ORG)

**Example 20** (doc_id: `55852`) (sent_id: `55852`)


Dies hindert das Bundesverfassungsgericht nicht , weitere Grundrechte in die Prüfung einzubeziehen , soweit sich die vom Beschwerdeführer geltend gemachte Rechtsverletzung in Blick auf dieselbe Beschwer auch oder vorrangig im Blick auf andere Grundrechte ergeben kann .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 21** (doc_id: `55922`) (sent_id: `55922`)


Es gab keine Anhaltspunkte dafür , dass die Annahme eines Bedürfnisses für eine bundesgesetzliche Regelung der Grundsteuer und der für sie maßgeblichen Bewertungsbestimmungen der danach verbleibenden , eingeschränkten Kontrolle durch das Bundesverfassungsgericht nicht hätte standhalten können ; solche wurden auch sonst von keiner Seite vorgebracht .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 22** (doc_id: `56228`) (sent_id: `56228`)


1. Nach § 32 Abs. 1 BVerfGG kann das Bundesverfassungsgericht im Streitfall einen Zustand durch einstweilige Anordnung vorläufig regeln , wenn dies zur Abwehr schwerer Nachteile oder aus einem anderen wichtigen Grund zum gemeinen Wohl dringend geboten ist .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `§ 32 Abs. 1 BVerfGG` (NRM)

**Example 23** (doc_id: `56313`) (sent_id: `56313`)


7. Das Bundesverfassungsgericht hat am 16. Januar 2018 eine mündliche Verhandlung durchgeführt .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 24** (doc_id: `56356`) (sent_id: `56356`)


Da das Streikverbot kein geschriebenes Verfassungsrecht , sondern Ergebnis einer Auslegung von Art. 33 Abs. 5 GG sei , müsse das Bundesverfassungsgericht seine frühere Auslegung dieser Bestimmung völkerrechtskonform hin zu einem funktionsbezogenen Streikverbot modifizieren .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Art. 33 Abs. 5 GG` (NRM)

**Example 25** (doc_id: `56443`) (sent_id: `56443`)


Nach der neueren Rechtsprechung des Bundesverfassungsgerichts sei auch innerhalb der Vermögensgruppe des Grundvermögens eine realitätsgerechte Bewertung erforderlich und eine Differenzierung bereits auf der Bewertungsebene verfassungsrechtlich nicht zulässig .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Example 26** (doc_id: `56582`) (sent_id: `56582`)


Das Bundesverfassungsgericht hat in diesem Beschluss nicht entschieden , dass in Fällen , in denen Oberverwaltungsgerichte / Verwaltungsgerichtshöfe auf der Grundlage ( weitestgehend ) identischer Tatsachenfeststellungen zu einer im Ergebnis abweichenden rechtlichen Beurteilung kommen , stets und notwendig eine ( klärungsbedürftige ) Rechtsfrage des Bundesrechts vorliegt , welche eine Rechtsmittelzulassung gebietet , um den Zugang zur Rechtsmittelinstanz nicht in einer durch Sachgründe nicht mehr zu rechtfertigenden Weise zu erschweren .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 27** (doc_id: `56796`) (sent_id: `56796`)


a ) Nach ständiger Rechtsprechung des Bundesverfassungsgerichts verpflichtet Art. 103 Abs. 1 GG ein Gericht , die Ausführungen der Prozessbeteiligten zur Kenntnis zu nehmen und in Erwägung zu ziehen .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `Art. 103 Abs. 1 GG` (NRM)

**Example 28** (doc_id: `56921`) (sent_id: `56921`)


Im vorliegenden Fall hat der Antragsteller zwar bislang in der Hauptsache kein Verfahren beim Bundesverfassungsgericht eingeleitet .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 29** (doc_id: `57236`) (sent_id: `57236`)


Zur Begründung wiederholt er im Wesentlichen sein bisheriges Vorbringen und verweist auf die erhebliche Vergleichbarkeit des Falles mit dem beim Bundesverfassungsgericht anhängigen Verfahren 2 BvR 424 / 17 .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Verfahren 2 BvR 424 / 17` (RS)

**Example 30** (doc_id: `57409`) (sent_id: `57409`)


Dieser Zulassungsgrund ist erfüllt , wenn die Vorinstanz mit einem ihre Entscheidung tragenden abstrakten Rechtssatz in Anwendung derselben Rechtsvorschrift einem ebensolchen Rechtssatz , der in der Rechtsprechung des Bundesverwaltungsgerichts , des Gemeinsamen Senats der obersten Gerichtshöfe des Bundes oder des Bundesverfassungsgerichts aufgestellt worden ist , widersprochen hat .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `Bundesverwaltungsgerichts` (ORG)
- `Gemeinsamen Senats der obersten Gerichtshöfe des Bundes` (ORG)

**Example 31** (doc_id: `57552`) (sent_id: `57552`)


Die Schwelle eines Verstoßes gegen Verfassungsrecht , den das Bundesverfassungsgericht zu korrigieren hat , ist erst dann erreicht , wenn die Auslegung der Zivilgerichte Fehler erkennen lässt , die auf einer grundsätzlich unrichtigen Anschauung von der Bedeutung der Grundrechte beruhen , insbesondere vom Umfang ihres Schutzbereichs , und auch in ihrer materiellen Bedeutung für den konkreten Rechtsfall von einigem Gewicht sind , insbesondere weil darunter die Abwägung der beiderseitigen Rechtspositionen im Rahmen der privatrechtlichen Regelung leidet ( vgl. BVerfGE 142 , 74 < 101 Rn. 83 > m. w. N. ; stRspr ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `BVerfGE 142 , 74 < 101 Rn. 83 >` (RS)

**Example 32** (doc_id: `57644`) (sent_id: `57644`)


Stellt das Bundesverfassungsgericht die Unvereinbarkeit einer Norm mit Art. 3 Abs. 1 GG fest , folgt daraus in der Regel die Verpflichtung des Gesetzgebers , rückwirkend , bezogen auf den in der gerichtlichen Feststellung genannten Zeitpunkt , die Rechtslage verfassungsgemäß umzugestalten .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Art. 3 Abs. 1 GG` (NRM)

**Example 33** (doc_id: `57671`) (sent_id: `57671`)


Urheberrechtliche Schrankenregelungen seien nach der Rechtsprechung des Bundesverfassungsgerichts grundsätzlich nur dann mit Art. 14 Abs. 1 GG vereinbar , wenn sie von einem Anspruch auf angemessene Vergütung begleitet würden .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `Art. 14 Abs. 1 GG` (NRM)

**Example 34** (doc_id: `57949`) (sent_id: `57949`)


Der Senat folgt insoweit der Rechtsprechung des Bundesverfassungsgerichts für das verfassungsrechtliche Verfahren ( vgl BVerfG Nichtannahmebeschluss vom 23. 12. 2016 - 1 BvR 3511/13 - Juris RdNr 3 mwN ; ebenso BFH Beschluss vom 8. 10. 2015 - VII B 147/14 - Juris RdNr 4 ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `BVerfG Nichtannahmebeschluss vom 23. 12. 2016 - 1 BvR 3511/13 - Juris RdNr 3` (RS)
- `BFH Beschluss vom 8. 10. 2015 - VII B 147/14 - Juris RdNr 4` (RS)

**Example 35** (doc_id: `57998`) (sent_id: `57998`)


Darüber hinaus hat das Bundesverfassungsgericht festgestellt , dass § 67d Abs. 3 Satz 1 in Verbindung mit § 2 Abs. 6 StGB - soweit er zur Anordnung der Fortdauer der Sicherungsverwahrung über zehn Jahre hinaus auch bei Verurteilten ermächtigt , deren Anlasstaten vor Inkrafttreten von Art. 1 des Gesetzes zur Bekämpfung von Sexualdelikten und anderen gefährlichen Straftaten vom 26. Januar 1998 ( BGBl I S. 160 ) begangen wurden - mit Art. 2 Abs. 2 Satz 2 in Verbindung mit Art. 20 Abs. 3 GG unvereinbar ist ( BVerfGE 128 , 326 < 331 , 332 > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `§ 67d Abs. 3 Satz 1 in Verbindung mit § 2 Abs. 6 StGB` (NRM)
- `Art. 1 des Gesetzes zur Bekämpfung von Sexualdelikten und anderen gefährlichen Straftaten vom 26. Januar 1998 ( BGBl I S. 160 )` (NRM)
- `Art. 2 Abs. 2 Satz 2 in Verbindung mit Art. 20 Abs. 3 GG` (NRM)
- `BVerfGE 128 , 326 < 331 , 332 >` (RS)

**Example 36** (doc_id: `58028`) (sent_id: `58028`)


Bei dieser Sachlage ist dem Bundesverfassungsgericht eine inhaltliche Prüfung der bereits unzulässigen Verfassungsbeschwerde versagt .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 37** (doc_id: `58156`) (sent_id: `58156`)


Die beiden erstgenannten Entscheidungen des Bundesverfassungsgerichts befassen sich indes schon nicht mit Art. 9 Abs. 3 GG ; die Entscheidung zur antragslosen Teilzeitbeschäftigung ( vgl. BVerfGE 119 , 247 < 264 > ) erwähnt die Koalitionsfreiheit zwar am Rande , trifft aber keine Aussage über das Verhältnis zu den hergebrachten Grundsätzen des Berufsbeamtentums .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `Art. 9 Abs. 3 GG` (NRM)
- `BVerfGE 119 , 247 < 264 >` (RS)

**Example 38** (doc_id: `58230`) (sent_id: `58230`)


Der Konventionstext und die Rechtsprechung des Europäischen Gerichtshofs für Menschenrechte dienen nach der ständigen Rechtsprechung des Bundesverfassungsgerichts auf der Ebene des Verfassungsrechts als Auslegungshilfen für die Bestimmung von Inhalt und Reichweite von Grundrechten und rechtsstaatlichen Grundsätzen des Grundgesetzes , sofern dies nicht zu einer Einschränkung oder Minderung des Grundrechtsschutzes nach dem Grundgesetz führt ( vgl. BVerfGE 74 , 358 < 370 > ; 111 , 307 < 317 > ; 120 , 180 < 200 f. > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `Europäischen Gerichtshofs für Menschenrechte` (ORG)
- `Grundgesetzes` (NRM)
- `Grundgesetz` (NRM)
- `BVerfGE 74 , 358 < 370 > ; 111 , 307 < 317 > ; 120 , 180 < 200 f. >` (RS)

**Example 39** (doc_id: `58575`) (sent_id: `58575`)


Die Aufgabe des Bundesverfassungsgerichts beschränkt sich hier grundsätzlich darauf , zu prüfen , ob die Fachgerichte eine auf das Wohl des Kindes ausgerichtete Entscheidung getroffen und dabei die Tragweite der Grundrechte aller Beteiligten nicht grundlegend verkannt haben ( vgl. BVerfGE 55 , 171 < 180 f. > ; 72 , 122 < 138 > sowie zuletzt BVerfG , Beschluss der 2. Kammer des Ersten Senats vom 7. Dezember 2017 - 1 BvR 1914/17 - , juris , Rn. 29 m. w. N. ; anderes bei der Überprüfung von Entscheidungen , die das Sorgerecht zum Zweck der Trennung des Kindes von den Eltern entziehen < Art. 6 Abs. 3 GG > ; vgl. BVerfGE 72 , 122 < 138 f. > ; 136 , 382 < 391 Rn. 28 f. > m. w. N. ; stRspr ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `BVerfGE 55 , 171 < 180 f. > ; 72 , 122 < 138 >` (RS)
- `BVerfG , Beschluss der 2. Kammer des Ersten Senats vom 7. Dezember 2017 - 1 BvR 1914/17 - , juris , Rn. 29` (RS)
- `Art. 6 Abs. 3 GG` (NRM)
- `BVerfGE 72 , 122 < 138 f. > ; 136 , 382 < 391 Rn. 28 f. >` (RS)

**Example 40** (doc_id: `58724`) (sent_id: `58724`)


Die Auffassung des Bundesfinanzhofs , dass er wegen der zwischenzeitlich am 5. August 2010 erfolgten telefonischen Bekanntgabe der Urteilsformel und der dadurch eingetretenen Selbstbindung die erst am 19. August 2010 veröffentlichten Entscheidungen des Bundesverfassungsgerichts vom 7. Juli 2010 in seinem Urteil nicht mehr hätte berücksichtigen können ( BFH , Beschluss vom 8. März 2011 - IV S 14/10 - , juris , Rn. 8 ff. ) , betrifft die fachgerichtliche Auslegung einfachen Prozessrechts .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `Bundesfinanzhofs` (ORG)
- `BFH , Beschluss vom 8. März 2011 - IV S 14/10 - , juris , Rn. 8 ff.` (RS)

**Example 41** (doc_id: `58759`) (sent_id: `58759`)


Er trägt vor , die Entscheidung des Berufungsgerichts weiche von dem Urteil des BSG vom 26. 7. 2007 ( B 13 R 4/06 R ) und den darin zitierten Entscheidungen der Obersten Gerichtshöfe des Bundes und des Bundesverfassungsgerichts ab .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `Urteil des BSG vom 26. 7. 2007 ( B 13 R 4/06 R )` (RS)
- `Obersten Gerichtshöfe des Bundes` (ORG)

**Example 42** (doc_id: `58991`) (sent_id: `58991`)


Vor allem stelle sich die Frage , ob es sich auch bei der vorliegenden Konstellation des Aufenthalts der Betroffenen in einem Pflegeheim um eine " stationäre Behandlung " im Sinne der Entscheidung des Bundesverfassungsgerichts handle .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Example 43** (doc_id: `59090`) (sent_id: `59090`)


In der Entscheidung zur Alimentation kinderreicher Beamter hat das Bundesverfassungsgericht ausgeführt , verfassungsrechtlich garantiert sei der hergebrachte allgemeine Grundsatz des Berufsbeamtentums , dass die angemessene Alimentierung summenmäßig nicht erstritten oder vereinbart , sondern durch Gesetz festgelegt werde , und dass innerhalb des Beamtenrechts die Zulassung eines Streiks ausgeschlossen sei ( vgl. BVerfGE 44 , 249 < 264 > m. w. N. ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `BVerfGE 44 , 249 < 264 >` (RS)

**Example 44** (doc_id: `59134`) (sent_id: `59134`)


Die für die Beurteilung der Verfassungsbeschwerde maßgeblichen verfassungsrechtlichen Fragen hat das Bundesverfassungsgericht bereits geklärt .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Example 45** (doc_id: `59161`) (sent_id: `59161`)


( 1 ) Das Bundesverfassungsgericht hat in jüngerer Zeit bereits mehrfach entschieden , dass die Einbringung eines Gesetzentwurfs in den Deutschen Bundestag das Vertrauen der Betroffenen auf den Fortbestand der bisherigen Rechtslage zerstören kann und deshalb eine darin vorgesehene Neuregelung ohne Verstoß gegen den verfassungsrechtlichen Grundsatz des Vertrauensschutzes unechte Rückwirkung entfalten darf ( vgl. dazu BVerfGE 127 , 31 < 50 > ; 143 , 246 < 385 Rn. 377 > ; BVerfG , Beschluss vom 7. März 2017 - 1 BvR 1314/12 u. a. - juris , Rn. 199 ; in BVerfGE 132 , 302 < 326 Rn. 60 > offen gelassen , weil in jenem Fall jedenfalls der Vorschlag des Vermittlungsausschusses das Vertrauen zerstört hatte ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Deutschen Bundestag` (ORG)
- `BVerfGE 127 , 31 < 50 > ; 143 , 246 < 385 Rn. 377 >` (RS)
- `BVerfG , Beschluss vom 7. März 2017 - 1 BvR 1314/12 u. a. - juris , Rn. 199` (RS)
- `BVerfGE 132 , 302 < 326 Rn. 60 >` (RS)

**Example 46** (doc_id: `59659`) (sent_id: `59659`)


1. Die Besorgnis der Befangenheit eines Richters des Bundesverfassungsgerichts nach § 19 BVerfGG setzt einen Grund voraus , der geeignet ist , Zweifel an seiner Unparteilichkeit zu rechtfertigen ( vgl. BVerfGE 82 , 30 < 37 > ; 101 , 46 < 50 f. > ; 108 , 122 < 126 > ; 142 , 18 < 21 Rn. 11 > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `§ 19 BVerfGG` (NRM)
- `BVerfGE 82 , 30 < 37 > ; 101 , 46 < 50 f. > ; 108 , 122 < 126 > ; 142 , 18 < 21 Rn. 11 >` (RS)

**Example 47** (doc_id: `59780`) (sent_id: `59780`)


Die Kundgabe politischer Meinungen , die ein Richter zu einer Zeit geäußert hat , als er noch nicht Mitglied des Bundesverfassungsgerichts war und daher den besonderen Anforderungen dieses Richteramtes in seinem Verhalten noch nicht Rechnung zu tragen hatte , rechtfertigt eine Ablehnung des Richters wegen Besorgnis der Befangenheit grundsätzlich nicht ( vgl. BVerfGE 99 , 51 < 56 f. > ; 142 , 18 < 21 f. Rn. 14 > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgerichts` | `Bundesverfassungsgerichts` |

**Missed by this rule (FN):**

- `BVerfGE 99 , 51 < 56 f. > ; 142 , 18 < 21 f. Rn. 14 >` (RS)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `55202`) (sent_id: `55202`)


Hierin lag insbesondere keine , im PKH-Verfahren nur in eng begrenztem Umfang zulässige vorweggenommene Beweiswürdigung ( s. dazu Beschlüsse des Bundesverfassungsgerichts - BVerfG - vom 7. Mai 1997 1 BvR 296/94 , Neue Juristische Wochenschrift 1997 , 2745 , und vom 20. Februar 2002 1 BvR 1450/00 , Neue Juristische Wochenschrift-Rechtsprechungs-Report Zivilrecht - NJW-RR - 2002 , 1069 ) .

**False Positives:**

- `Bundesverfassungsgerichts` — partial — pred is substring of gold: `Beschlüsse des Bundesverfassungsgerichts - BVerfG - vom 7. Mai 1997 1 BvR 296/94 , Neue Juristische Wochenschrift 1997 , 2745`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Beschlüsse des Bundesverfassungsgerichts - BVerfG - vom 7. Mai 1997 1 BvR 296/94 , Neue Juristische Wochenschrift 1997 , 2745`(RS)
- `vom 20. Februar 2002 1 BvR 1450/00 , Neue Juristische Wochenschrift-Rechtsprechungs-Report Zivilrecht - NJW-RR - 2002 , 1069`(RS)

**Example 1** (doc_id: `55266`) (sent_id: `55266`)


In diesem Sinn hat auch die Beklagte in ihrem Bescheid die Bürogemeinschaft zwischen dem Kläger und seinem ehemaligen Sozius deshalb missbilligt , weil nach ihrer Auffassung die Tätigkeit des Letzteren bezüglich der gesetzlichen Verschwiegenheitspflicht ( § 203 StGB ) , des Zeugnisverweigerungsrechts ( § 53 StPO ) und des Beschlagnahmeverbots ( § 97 StPO ) nach der damaligen Rechtslage weder mit den sozietätsfähigen Berufen noch mit den in der Entscheidung des Bundesverfassungsgerichts vom 12. Januar 2016 ( BVerfGE 141 , 82 ) behandelten Berufsgruppen vergleichbar ist .

**False Positives:**

- `Bundesverfassungsgerichts` — partial — pred is substring of gold: `Entscheidung des Bundesverfassungsgerichts vom 12. Januar 2016 ( BVerfGE 141 , 82 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 203 StGB`(NRM)
- `§ 53 StPO`(NRM)
- `§ 97 StPO`(NRM)
- `Entscheidung des Bundesverfassungsgerichts vom 12. Januar 2016 ( BVerfGE 141 , 82 )`(RS)

**Example 2** (doc_id: `58343`) (sent_id: `58343`)


Unter den hier vorliegenden Voraussetzungen des Art. 267 Abs. 3 AEUV ( vergleiche dazu Beschlüsse des Bundesverfassungsgerichts - BVerfG - vom 6. September 2016 1 BvR 1305/13 , Neue Zeitschrift für Verwaltungsrecht - NVwZ - 2017 , 53 , Rz 7 , 8 ; vom 6. Oktober 2017 2 BvR 987/16 , Neue Juristische Wochenschrift - NJW - 2018 , 606 , Rz 4 , 5 ; jeweils mit weiteren Nachweisen ) sind die nationalen Gerichte von Amts wegen gehalten , den EuGH anzurufen ( vergleiche BVerfG-Beschlüsse in NVwZ 2017 , 53 , Rz 6 ; in NJW 2018 , 606 , Rz 3 ; ferner EuGH-Urteil CILFIT vom 6. Oktober 1982 C - 283/81 , EU : C : 1982 : 335 , NJW 1983 , 1257 , Rz 21 ; jeweils mit weiteren Nachweisen ) .

**False Positives:**

- `Bundesverfassungsgerichts` — partial — pred is substring of gold: `Beschlüsse des Bundesverfassungsgerichts - BVerfG - vom 6. September 2016 1 BvR 1305/13 , Neue Zeitschrift für Verwaltungsrecht - NVwZ - 2017 , 53 , Rz 7 , 8`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 267 Abs. 3 AEUV`(NRM)
- `Beschlüsse des Bundesverfassungsgerichts - BVerfG - vom 6. September 2016 1 BvR 1305/13 , Neue Zeitschrift für Verwaltungsrecht - NVwZ - 2017 , 53 , Rz 7 , 8`(RS)
- `vom 6. Oktober 2017 2 BvR 987/16 , Neue Juristische Wochenschrift - NJW - 2018 , 606 , Rz 4 , 5`(RS)
- `EuGH`(ORG)
- `BVerfG-Beschlüsse in NVwZ 2017 , 53 , Rz 6`(LIT)
- `NJW 2018 , 606 , Rz 3`(RS)
- `EuGH-Urteil CILFIT vom 6. Oktober 1982 C - 283/81 , EU : C : 1982 : 335 , NJW 1983 , 1257 , Rz 21`(RS)

**Example 3** (doc_id: `58740`) (sent_id: `58740`)


Das Berufungsgericht wird sich bei seiner neuerlichen , durch diesen Beschluss nicht im Ergebnis vorgeprägten Entscheidung auch mit den - wenngleich in anderem Zusammenhang ergangenen - Entscheidungen des Gerichtshofes der Europäischen Union vom 25. Januar 2018 ( - C- 360/16 [ ECLI : EU :C : 2018 : 35 ] , Hasan - Rn. 30 f. , 40 ) und des Bundesverfassungsgerichts vom 21. April 2016 ( - 2 BvR 273/16 - NVwZ 2016 , 1242 Rn. 11 ) zur Frage des maßgeblichen Zeitpunkts für die Beurteilung des Vorliegens systemischer Schwachstellen auseinanderzusetzen haben .

**False Positives:**

- `Bundesverfassungsgerichts` — partial — pred is substring of gold: `Bundesverfassungsgerichts vom 21. April 2016 ( - 2 BvR 273/16 - NVwZ 2016 , 1242 Rn. 11 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Entscheidungen des Gerichtshofes der Europäischen Union vom 25. Januar 2018 ( - C- 360/16 [ ECLI : EU :C : 2018 : 35 ] , Hasan - Rn. 30 f. , 40 )`(RS)
- `Bundesverfassungsgerichts vom 21. April 2016 ( - 2 BvR 273/16 - NVwZ 2016 , 1242 Rn. 11 )`(RS)

**Example 4** (doc_id: `59129`) (sent_id: `59129`)


Beide Absätze des Art. 4 GG enthalten ein umfassend zu verstehendes einheitliches Grundrecht , das auch die Religionsfreiheit der Korporationen umfasst ( vgl. Urteil des Bundesverfassungsgerichts - BVerfG - vom 1. Dezember 2009 1 BvR 2857/07 , 1 BvR 2858/07 , BVerfGE 125 , 39 , Rz 137 , m. w. N. ) .

**False Positives:**

- `Bundesverfassungsgerichts` — partial — pred is substring of gold: `Urteil des Bundesverfassungsgerichts - BVerfG - vom 1. Dezember 2009 1 BvR 2857/07 , 1 BvR 2858/07 , BVerfGE 125 , 39 , Rz 137`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 4 GG`(NRM)
- `Urteil des Bundesverfassungsgerichts - BVerfG - vom 1. Dezember 2009 1 BvR 2857/07 , 1 BvR 2858/07 , BVerfGE 125 , 39 , Rz 137`(RS)

</details>

---

## `Patent Department Full Context (Fixed)`

**F1:** 0.062 | **Precision:** 0.743 | **Recall:** 0.032  

**Format:** `regex`  
**Rule ID:** `6423d995`  
**Description:**
Matches specific patent department names with class numbers and the full parent organization name.

**Content:**
```
\b(?:Markenstelle\s+f\u00fcr\s+Klasse\s+\d+(?:\s+-\s+[A-Za-z\s-]+)?\s+des\s+Deutschen\s+Patent-\s+und\s+Markenamts|Pr\u00fcfungsstelle\s+f\u00fcr\s+Klasse\s+[A-Za-z0-9\s]+\s+des\s+Deutschen\s+Patent-\s+und\s+Markenamts|Markenabteilung\s+\d+\.?\d*\s+des\s+Deutschen\s+Patent-\s+und\s+Markenamts|Patentabteilung\s+\d+\.?\d*\s+des\s+Deutschen\s+Patent-\s+und\s+Markenamts|Designabteilung\s+\d+\.?\d*\s+des\s+Deutschen\s+Patent-\s+und\s+Markenamts|Markenstelle\s+f\u00fcr\s+Klasse\s+\d+\s+des\s+DPMA|Deutschen\s+Patent-\s+und\s+Markenamts|Deutsche\s+Patent-\s+und\s+Markenamt|Markenstelle\s+f\u00fcr\s+Klasse\s+\d+|Patentabteilung\s+\d+|Designstelle\s+des\s+Deutschen\s+Patent-\s+und\s+Markenamts|Pr\u00fcfungsstelle\s+f\u00fcr\s+die\s+Klasse\s+A62B\s+des\s+Deutschen\s+Patent-\s+und\s+Markenamts)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.743 | 0.032 | 0.062 | 35 | 26 | 9 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 26 | 9 | 772 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53419`) (sent_id: `53419`)


Hiergegen richtet sich die Beschwerde der Anmelderin vom 29. November 2016 , mit der sie sinngemäß beantragt , den Beschluss der Markenstelle für Klasse 35 des Deutschen Patent- und Markenamts vom 11. November 2016 aufzuheben .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 35 des Deutschen Patent- und Markenamts` | `Markenstelle für Klasse 35 des Deutschen Patent- und Markenamts` |

**Example 1** (doc_id: `53574`) (sent_id: `53574`)


Die Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts hat diese unter der Nummer 30 2014 034 614.1 geführte Anmeldung mit Beschluss vom 25. November 2014 wegen fehlender Unterscheidungskraft zurückgewiesen .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts` | `Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts` |

**Example 2** (doc_id: `54179`) (sent_id: `54179`)


I. Mit dem angefochtenen Beschluss vom 4. November 2015 hat die Patentabteilung 43 des Deutschen Patent- und Markenamts das Patent 103 36 913 mit der Bezeichnung

| Predicted | Gold |
|---|---|
| `Patentabteilung 43 des Deutschen Patent- und Markenamts` | `Patentabteilung 43 des Deutschen Patent- und Markenamts` |

**Example 3** (doc_id: `54537`) (sent_id: `54537`)


den Beschluss der Markenstelle für Klasse 5 des Deutschen Patent- und Markenamts vom 17. August 2015 in der Hauptsache aufzuheben und auf ihren Widerspruch hin , die Löschung der angegriffenen Marke 30 2013 058 941 in Bezug auf sämtliche Waren der Klassen 3 und 5 anzuordnen .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 5 des Deutschen Patent- und Markenamts` | `Markenstelle für Klasse 5 des Deutschen Patent- und Markenamts` |

**Example 4** (doc_id: `54862`) (sent_id: `54862`)


den angefochtenen Beschluss der Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts vom 18. Februar 2014 in der Hauptsache aufzuheben und auf ihren Widerspruch aus der Marke 305 02 291 hin die Löschung der angegriffenen Marke 30 2012 030 505 anzuordnen .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts` | `Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts` |

**Example 5** (doc_id: `55132`) (sent_id: `55132`)


a ) Wie das Deutsche Patent- und Markenamt zutreffend ausgeführt hat , ist das Anmeldezeichen zum Zeitpunkt seiner Anmeldung am 20. Januar 2016 von den überwiegend angesprochenen Fachverkehrskreisen , aber - insbesondere in Bezug auf die Dienstleistungen der Klassen 40 und 44 - auch von Auftraggebern oder Empfängern zahntechnischer oder -medizinischer Dienstleistungen im Sinne von „ CAD Labor “ oder „ Labor , das CAD einsetzt “ verstanden worden .

| Predicted | Gold |
|---|---|
| `Deutsche Patent- und Markenamt` | `Deutsche Patent- und Markenamt` |

**Example 6** (doc_id: `55576`) (sent_id: `55576`)


Wegen der weiteren Einzelheiten wird auf die angefochtenen Beschlüsse der Markenstelle für Klasse 42 vom 3. April 2014 und vom 2. Juli 2015 sowie auf die Schriftsätze der Beteiligten , das Protokoll der mündlichen Verhandlung vom 14. September 2017 und den weiteren Akteninhalt Bezug genommen .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 42` | `Markenstelle für Klasse 42` |

**Example 7** (doc_id: `55948`) (sent_id: `55948`)


Insofern gibt es auch im Rahmen von unbestimmten Rechtbegriffen keine Selbstbindung der Markenstellen des Deutschen Patent- und Markenamts und erst recht keine irgendwie geartete Bindung für das Bundespatentgericht .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamts` | `Deutschen Patent- und Markenamts` |

**Missed by this rule (FN):**

- `Bundespatentgericht` (ORG)

**Example 8** (doc_id: `56090`) (sent_id: `56090`)


den Beschluss der Patentabteilung 1.35 des Deutschen Patent- und Markenamts vom 14. Juli 2015 aufzuheben und das Patent 10 2007 056 516 zu widerrufen .

| Predicted | Gold |
|---|---|
| `Patentabteilung 1.35 des Deutschen Patent- und Markenamts` | `Patentabteilung 1.35 des Deutschen Patent- und Markenamts` |

**Example 9** (doc_id: `56251`) (sent_id: `56251`)


Die Anmelderin beantragt sinngemäß den Beschluss Markenstelle für Klasse 41 des DPMA vom 10. Juni 2016 aufzuheben .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 41 des DPMA` | `Markenstelle für Klasse 41 des DPMA` |

**Example 10** (doc_id: `56441`) (sent_id: `56441`)


den Beschluss der Patentabteilung 1.33 des Deutschen Patent- und Markenamts vom 20. Dezember 2016 aufzuheben und das Patent 10 2008 026 411 zu widerrufen ,

| Predicted | Gold |
|---|---|
| `Patentabteilung 1.33 des Deutschen Patent- und Markenamts` | `Patentabteilung 1.33 des Deutschen Patent- und Markenamts` |

**Example 11** (doc_id: `56713`) (sent_id: `56713`)


Gegen das Patent ist Einspruch erhoben worden , worauf die Patentabteilung 13 des Deutschen Patent- und Markenamts das Patent durch Beschluss vom 30. September 2014 aufrechterhalten hat .

| Predicted | Gold |
|---|---|
| `Patentabteilung 13 des Deutschen Patent- und Markenamts` | `Patentabteilung 13 des Deutschen Patent- und Markenamts` |

**Example 12** (doc_id: `57270`) (sent_id: `57270`)


Mit Beschluss vom 6. August 2015 hat die mit einer Beamtin des gehobenen Dienstes besetzte Markenstelle für Klasse 9 des Deutschen Patent- und Markenamts die Anmeldung zurückgewiesen , weil es der angemeldeten Bezeichnung in Bezug auf die beanspruchten Waren und Dienstleistungen an der erforderlichen Unterscheidungskraft fehle ( § 8 Abs. 2 Nr. 1 MarkenG ) .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 9 des Deutschen Patent- und Markenamts` | `Markenstelle für Klasse 9 des Deutschen Patent- und Markenamts` |

**Missed by this rule (FN):**

- `§ 8 Abs. 2 Nr. 1 MarkenG` (NRM)

**Example 13** (doc_id: `57367`) (sent_id: `57367`)


Dass der Begriff des „ Kontors “ für verschiedenste Dienstleistungen - wie etwa Spirituosenherstellung , Versicherungswesen , Werbung oder Immobilienwesen - aktuell Verwendung findet , hat das Deutsche Patent- und Markenamt überzeugend dargetan .

| Predicted | Gold |
|---|---|
| `Deutsche Patent- und Markenamt` | `Deutsche Patent- und Markenamt` |

**Example 14** (doc_id: `58065`) (sent_id: `58065`)


Es kann ihnen aber auch nach weiteren Ermittlungen durch den Senat keine hinreichend deutliche Erklärung dahingehend entnommen werden , dass der Beschluss der Markenstelle für Klasse 35 des DPMA vom 9. Dezember 2016 angegriffen werden sollte .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 35 des DPMA` | `Markenstelle für Klasse 35 des DPMA` |

**Example 15** (doc_id: `58102`) (sent_id: `58102`)


Die Billigkeit der Rückzahlung kann sich unter anderem aus der Sachbehandlung durch das Deutsche Patent- und Markenamt ( z.B. sachliche Fehlbeurteilung , Verfahrensfehler , Verstoß gegen Verfahrensökonomie ) oder aus sonstigen Umständen ergeben , die eine Einbehaltung der Gebühr als unbillig erscheinen lässt ( Schulte / Püschel , PatG , 10. Aufl. 2017 , § 80 Rn. 113 f. ) .

| Predicted | Gold |
|---|---|
| `Deutsche Patent- und Markenamt` | `Deutsche Patent- und Markenamt` |

**Missed by this rule (FN):**

- `Schulte / Püschel , PatG , 10. Aufl. 2017 , § 80 Rn. 113 f.` (LIT)

**Example 16** (doc_id: `58120`) (sent_id: `58120`)


Auf die Beschwerde der Einsprechenden wird der Beschluss der Patentabteilung 21 des Deutschen Patent- und Markenamts vom 4. März 2015 aufgehoben und das Patent 10 2009 029 037 beschränkt aufrechterhalten mit folgenden Unterlagen :

| Predicted | Gold |
|---|---|
| `Patentabteilung 21 des Deutschen Patent- und Markenamts` | `Patentabteilung 21 des Deutschen Patent- und Markenamts` |

**Example 17** (doc_id: `58314`) (sent_id: `58314`)


den Beschluss der Patentabteilung 1.23 des Deutschen Patent- und Markenamts vom 23. Juli 2015 aufzuheben und das Patent 198 31 774 mit folgenden Unterlagen beschränkt aufrecht zu erhalten :

| Predicted | Gold |
|---|---|
| `Patentabteilung 1.23 des Deutschen Patent- und Markenamts` | `Patentabteilung 1.23 des Deutschen Patent- und Markenamts` |

**Example 18** (doc_id: `58320`) (sent_id: `58320`)


Auf Nachfrage des Deutschen Patent- und Markenamts ( DPMA ) erklärten die anwaltlichen Vertreter der Designinhaberin mit Schriftsatz vom 2. Juni 2008 , dass die Wiedergaben „ tatsächlich ein einzelnes Muster “ zeigten .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamts` | `Deutschen Patent- und Markenamts` |

**Missed by this rule (FN):**

- `DPMA` (ORG)

**Example 19** (doc_id: `58404`) (sent_id: `58404`)


den Beschluss der Markenstelle für Klasse 30 des Deutschen Patent- und Markenamts vom 19. Februar 2014 in der Hauptsache aufzuheben , soweit die Widersprüche aus der Unionsmarke 94 921 58 und der deutschen Marke 30 2010 097 454 zurückgewiesen worden sind und wegen dieser Widersprüche die Löschung der Marke 30 2012 041 338 anzuordnen .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 30 des Deutschen Patent- und Markenamts` | `Markenstelle für Klasse 30 des Deutschen Patent- und Markenamts` |

**Example 20** (doc_id: `58786`) (sent_id: `58786`)


Der Beschluss der Markenstelle für Klasse 41 des Deutschen Patent- und Markenamts vom 29. April 2015 ist wirkungslos , soweit der Widerspruch aus der Unionsmarke EM 005 729 819 – LIMBIC gegen die deutsche Wortmarke 30 2012 035 412 – Limbic touch zurückgewiesen worden ist .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 41 des Deutschen Patent- und Markenamts` | `Markenstelle für Klasse 41 des Deutschen Patent- und Markenamts` |

**Missed by this rule (FN):**

- `LIMBIC` (ORG)
- `Limbic touch` (ORG)

**Example 21** (doc_id: `58891`) (sent_id: `58891`)


1. Der Beschluss der Patentabteilung 24 des Deutschen Patent- und Markenamts vom 14. Oktober 2014 wird aufgehoben .

| Predicted | Gold |
|---|---|
| `Patentabteilung 24 des Deutschen Patent- und Markenamts` | `Patentabteilung 24 des Deutschen Patent- und Markenamts` |

**Example 22** (doc_id: `58900`) (sent_id: `58900`)


1. Der angefochtene Beschluss der Patentabteilung 44 des Deutschen Patent- und Markenamts vom 23. September 2015 wird aufgehoben .

| Predicted | Gold |
|---|---|
| `Patentabteilung 44 des Deutschen Patent- und Markenamts` | `Patentabteilung 44 des Deutschen Patent- und Markenamts` |

**Example 23** (doc_id: `59087`) (sent_id: `59087`)


Nach § 73 Abs. 3 Satz 2 PatG hat das Deutsche Patent- und Markenamt dann , wenn es der Beschwerde nicht abhilft , sie dem Bundespatentgericht vorzulegen .

| Predicted | Gold |
|---|---|
| `Deutsche Patent- und Markenamt` | `Deutsche Patent- und Markenamt` |

**Missed by this rule (FN):**

- `§ 73 Abs. 3 Satz 2 PatG` (NRM)
- `Bundespatentgericht` (ORG)

**Example 24** (doc_id: `59444`) (sent_id: `59444`)


Mit am Ende der Anhörung vom 9. Juli 2014 verkündeten Beschluss hat die Patentabteilung 1.35 des Deutschen Patent- und Markenamts das Patent im Umfang der Patentansprüche 1 bis 13 gemäß Hilfsantrag 1 beschränkt aufrechterhalten .

| Predicted | Gold |
|---|---|
| `Patentabteilung 1.35 des Deutschen Patent- und Markenamts` | `Patentabteilung 1.35 des Deutschen Patent- und Markenamts` |

**Example 25** (doc_id: `59938`) (sent_id: `59938`)


Mit Beschlüssen vom 16. April 2012 und 23. Mai 2013 , von denen letzterer im Erinnerungsverfahren ergangen ist , hat die Markenstelle für Klasse 32 des DPMA die Anmeldung wegen fehlender Unterscheidungskraft gemäß §§ 37 Abs. 1 , 8 Abs. 2 Nr. 1 MarkenG zurückgewiesen .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 32 des DPMA` | `Markenstelle für Klasse 32 des DPMA` |

**Missed by this rule (FN):**

- `§§ 37 Abs. 1 , 8 Abs. 2 Nr. 1 MarkenG` (NRM)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53950`) (sent_id: `53950`)


Mit Beschluss vom 30. Mai 2016 hat die Markenstelle für Klasse 35 die Anmeldung zurückgewiesen .

**False Positives:**

- `Markenstelle für Klasse 35` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `54010`) (sent_id: `54010`)


Auf die Beschwerde der Anmelderin werden die Beschlüsse des Deutschen Patent- und Markenamts , Markenstelle für Klasse 41 , vom 3. Juli 2014 und vom 3. Dezember 2015 aufgehoben , soweit die Anmeldung in Bezug auf die nachfolgend genannten Dienstleistungen zurückgewiesen worden ist :

**False Positives:**

- `Deutschen Patent- und Markenamts` — partial — pred is substring of gold: `Deutschen Patent- und Markenamts , Markenstelle für Klasse 41`
- `Markenstelle für Klasse 41` — partial — pred is substring of gold: `Deutschen Patent- und Markenamts , Markenstelle für Klasse 41`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Deutschen Patent- und Markenamts , Markenstelle für Klasse 41`(ORG)

**Example 2** (doc_id: `56103`) (sent_id: `56103`)


den Beschluss der Gebrauchsmusterabteilung des Deutschen Patent- und Markenamts vom 6. November 2014 aufzuheben und die ihr von der Antragsgegnerin zu erstattenden Kosten in Höhe von 7.208,08 € neu festzusetzen .

**False Positives:**

- `Deutschen Patent- und Markenamts` — partial — pred is substring of gold: `Gebrauchsmusterabteilung des Deutschen Patent- und Markenamts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Gebrauchsmusterabteilung des Deutschen Patent- und Markenamts`(ORG)

**Example 3** (doc_id: `57135`) (sent_id: `57135`)


unter Zurückweisung der Beschwerde der Einsprechenden den Beschluss der Patentabteilung 43 des Deutschen Patent- und Markenamt vom 4. November 2015 aufzuheben und das Streitpatent vollumfänglich aufrechtzuerhalten ,

**False Positives:**

- `Patentabteilung 43` — partial — pred is substring of gold: `Patentabteilung 43 des Deutschen Patent- und Markenamt`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Patentabteilung 43 des Deutschen Patent- und Markenamt`(ORG)

**Example 4** (doc_id: `57609`) (sent_id: `57609`)


1. Auf die Beschwerde der Antragstellerin wird der Beschluss der Gebrauchsmusterabteilung des Deutschen Patent- und Markenamts vom 6. November 2014 aufgehoben und die von der Antragsgegnerin der Antragstellerin zu erstattenden Kosten des patentamtlichen Löschungsverfahrens werden auf 7.208,08 € ( in Worten : siebentausendzweihundertacht 8/100 Euro ) festgesetzt .

**False Positives:**

- `Deutschen Patent- und Markenamts` — partial — pred is substring of gold: `Gebrauchsmusterabteilung des Deutschen Patent- und Markenamts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Gebrauchsmusterabteilung des Deutschen Patent- und Markenamts`(ORG)

**Example 5** (doc_id: `58052`) (sent_id: `58052`)


Foto des Kontaktsockels „ Waffle Kelvin “ im Bild 1 des angegriffenen Beschlusses mit Ergänzungen von Bezugszeichen durch die Patentabteilung 1.35

**False Positives:**

- `Patentabteilung 1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Kelvin`(PER)

**Example 6** (doc_id: `58282`) (sent_id: `58282`)


Foto des Kontaktsockels „ Waffle Kelvin “ im Bild 1 des angegriffenen Beschlusses mit Ergänzungen von Bezugszeichen durch die Patentabteilung 1.35

**False Positives:**

- `Patentabteilung 1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Kelvin`(PER)

**Example 7** (doc_id: `58776`) (sent_id: `58776`)


auf die Erinnerung der Inhaberin der eingetragenen Marke 30 2010 022 988 den Beschluss der Markenstelle für Klasse 25 vom 27. März 2012 aufgehoben

**False Positives:**

- `Markenstelle für Klasse 25` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Government Ministries and Bodies`

**F1:** 0.061 | **Precision:** 0.634 | **Recall:** 0.032  

**Format:** `regex`  
**Rule ID:** `80b9799f`  
**Description:**
Matches government bodies including optional department codes.

**Content:**
```
\b(Bundesministeriums?\s+der\s+Verteidigung(?:\s+-\s+[A-Z0-9\s-]+)?|Bundesministeriums?\s+der\s+Justiz|Landesregierung\s+von\s*Nordrhein-Westfalen|saarl\u00e4ndischen\s*Landesregierung|Ausw\u00e4rtigen\s*Amtes|Bundesrat|Staatskasse|Justizministerium\s*des\s*Landes\s*Nordrhein-Westfalen|Bayerische\s*Staatsministerium|Bundesministeriums?\s*der\s*Verteidigung|Bundesministerium\s*f\u00fcr\s*Verbraucherschutz\s*,\s*Ern\u00e4hrung\s*und\s*Landwirtschaft|Bundesministerium\s*f\u00fcr\s*Ern\u00e4hrung\s*und\s*Landwirtschaft|Bayerischen\s*Staatsministeriums\s*f\u00fcr\s+Umwelt\s*und\s*Gesundheit|Hessische\s*Ministerium\s*des\s*Innern\s*und\s*f\u00fcr\s*Sport|Deutsche\s*Rentenversicherung\s*Bund|Deutsche\s*Rentenversicherung\s*Rheinland|Deutschen\s*Rentenversicherung|Justizministerium\s*des\s*Landes\s*Niedersachsen|Ministerium\s*f\u00fcr\s*Justiz\s*,\s*Europa\s*,\s*Verbraucherschutz\s*und\s*Gleichstellung\s*des\s*Landes\s*Schleswig-Holstein|Bundesministerium\s+der\s+Finanzen|BMF|Bundesagentur\s+f\u00fcr\s+Arbeit|Bundesrat|Bundestag|Bundesregierung|Staatsministeriums\s+der\s+Justiz\s+sowie\s+des\s+Staatsministeriums\s+f\u00fcr\s+Kultus\s+des\s+Freistaates\s+Sachsen|Wehrbeauftragten\s+des\s+Deutschen\s+Bundestages|Bundesbeauftragte\s+f\u00fcr\s+den\s+Datenschutz\s+und\s+die\s+Informationsfreiheit|Fliegerhorst\s+B\u00fcchel|Dienststellen\s+R\s+I\s+\(\s*R\s+I\s*\)\s*,\s*R\s+II\s+\(\s*R\s+II\s*\)\s*,\s*D\s+\(\s*D\s*\)\s+K\s*,\s+Sp\s*,\s+D\s+L\s+\(\s+DL\s*\)\s+K\s*,\s+D\s+G\s+und\s+DL\s+G|Generalinspekteur\s+der\s+Bundeswehr|Deutschen\s+Fu\u00dfball-Bund)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.634 | 0.032 | 0.061 | 41 | 26 | 15 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 26 | 15 | 743 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53664`) (sent_id: `53664`)


Soweit das Bundesministerium der Verteidigung darauf verweist , dass Personalgespräche dazu dienten , Soldaten über Verwendungsplanungen der Personalführung zu informieren und in diesem Zusammenhang deren persönliche Motivationslage zu ergründen , um das dienstliche Bedürfnis und die privaten Lebensumstände des Soldaten möglichst gut miteinander zu vereinbaren , trifft dies zweifellos auf Personalgespräche im Vorfeld von Personalmaßnahmen zu .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Example 1** (doc_id: `53666`) (sent_id: `53666`)


Der Senat hat mit Verfügungen vom 25. Januar 2018 , 12. Februar 2018 und 14. März 2018 ergänzende Auskünfte des Auswärtigen Amtes eingeholt .

| Predicted | Gold |
|---|---|
| `Auswärtigen Amtes` | `Auswärtigen Amtes` |

**Example 2** (doc_id: `53690`) (sent_id: `53690`)


An dieser Regelung hielt die Bundesregierung fest , obwohl der Bundesrat die Einbeziehung der einmaligen Einnahmen vorschlug ( BT-Drucks 16/2454 S 11 ) .

| Predicted | Gold |
|---|---|
| `Bundesrat` | `Bundesrat` |

**Missed by this rule (FN):**

- `Bundesregierung` (ORG)
- `BT-Drucks 16/2454 S 11` (LIT)

**Example 3** (doc_id: `54428`) (sent_id: `54428`)


Der dort vorgesehene Versetzungsschutz bei Versetzungen in zeitlicher Nähe zum Dienstzeitende wird nicht , wie das Bundesministerium der Verteidigung einwendet , durch die Vorgaben der Zentralen Dienstvorschrift A- 1350/66 über die " Letzte Verwendung vor Zurruhesetzung " ausgeschlossen .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Missed by this rule (FN):**

- `Zentralen Dienstvorschrift A- 1350/66` (REG)

**Example 4** (doc_id: `54509`) (sent_id: `54509`)


Die Vertreterin der Staatskasse ( Erinnerungsgegnerin ) hält die Ermittlung des Streitwerts für zutreffend und beantragt , die Erinnerung als unbegründet zurückzuweisen .

| Predicted | Gold |
|---|---|
| `Staatskasse` | `Staatskasse` |

**Example 5** (doc_id: `55068`) (sent_id: `55068`)


An dieser Regelung hielt die Bundesregierung fest , obwohl der Bundesrat die Einbeziehung der einmaligen Einnahmen vorschlug ( BT-Drucks 16/2454 S 11 ) .

| Predicted | Gold |
|---|---|
| `Bundesrat` | `Bundesrat` |

**Missed by this rule (FN):**

- `Bundesregierung` (ORG)
- `BT-Drucks 16/2454 S 11` (LIT)

**Example 6** (doc_id: `55481`) (sent_id: `55481`)


Das Bundesministerium der Verteidigung beantragt ,

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Example 7** (doc_id: `55893`) (sent_id: `55893`)


Nach dem Gesetzentwurf der Bundesregierung zum Entwurf des Teilzeit- und Befristungsgesetzes vom 24. Oktober 2000 sollte es weiterhin zulässig sein , einen Arbeitsvertrag ohne Vorliegen eines sachlichen Grundes bis zur Dauer von zwei Jahren zu befristen und einen zunächst kürzer befristeten Arbeitsvertrag innerhalb der zweijährigen Höchstbefristungsdauer höchstens dreimal zu verlängern .

| Predicted | Gold |
|---|---|
| `Bundesregierung` | `Bundesregierung` |

**Missed by this rule (FN):**

- `Teilzeit- und Befristungsgesetzes` (NRM)

**Example 8** (doc_id: `56091`) (sent_id: `56091`)


Das Bundesministerium der Verteidigung beantragt ,

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Example 9** (doc_id: `56550`) (sent_id: `56550`)


Zwar verfügen beide Bewerber nicht über die nach dem Planungsbogen ursprünglich als dienstpostenunabhängiges Kriterium geforderte Stabsoffizierverwendung im Bundesministerium der Verteidigung ( oder eine vergleichbare Verwendung ) .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Example 10** (doc_id: `56577`) (sent_id: `56577`)


Dies ist im Beschwerdebescheid des Bundesministeriums der Verteidigung ohne Rechtsfehler mitentschieden worden .

| Predicted | Gold |
|---|---|
| `Bundesministeriums der Verteidigung` | `Bundesministeriums der Verteidigung` |

**Example 11** (doc_id: `56891`) (sent_id: `56891`)


Nach der speziellen Zuständigkeitsvorschrift in § 3 Satz 1 SAZV ist für " Maßnahmen nach der Soldatenarbeitszeitverordnung " das Bundesministerium der Verteidigung zuständig , soweit nichts Abweichendes bestimmt ist .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Missed by this rule (FN):**

- `§ 3 Satz 1 SAZV` (NRM)
- `Soldatenarbeitszeitverordnung` (NRM)

**Example 12** (doc_id: `57265`) (sent_id: `57265`)


2. Die angefochtene Versetzung ist rechtswidrig , weil das Bundesamt für das Personalmanagement und das Bundesministerium der Verteidigung bei der Ausübung des ihnen zustehenden Ermessens die von dem Antragsteller geltend gemachte Betreuung seiner Großmutter nicht berücksichtigt haben ( § 23a Abs. 2 Satz 1 WBO i. V. m. § 114 Satz 1 VwGO ) .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Missed by this rule (FN):**

- `Bundesamt für das Personalmanagement` (ORG)
- `§ 23a Abs. 2 Satz 1 WBO` (NRM)
- `§ 114 Satz 1 VwGO` (NRM)

**Example 13** (doc_id: `57699`) (sent_id: `57699`)


3. Soweit das Verfahren eingestellt wurde , trägt die Staatskasse die Kosten des Verfahrens und die notwendigen Auslagen des Angeklagten ; die verbleibenden Kosten seines Rechtsmittels trägt der Angeklagte selbst .

| Predicted | Gold |
|---|---|
| `Staatskasse` | `Staatskasse` |

**Example 14** (doc_id: `58056`) (sent_id: `58056`)


Das tunesische Justizministerium hat in seinem Schreiben vom 1. März 2018 ( S. 1 , Anlage 1 der Auskunft des Auswärtigen Amtes vom 7. März 2018 ) über Gespräche unter anderem mit Vertretern des deutschen Bundesministeriums der Justiz und für Verbraucherschutz ausgeführt , dass im Jahr 2012 insgesamt 122 Todesurteile in lebenslange Freiheitsstrafen umgewandelt worden sind .

| Predicted | Gold |
|---|---|
| `Auswärtigen Amtes` | `Auswärtigen Amtes` |

**Missed by this rule (FN):**

- `tunesische Justizministerium` (ORG)
- `Bundesministeriums der Justiz und für Verbraucherschutz` (ORG)

**Example 15** (doc_id: `58109`) (sent_id: `58109`)


Aus dem Lagebericht des Auswärtigen Amtes vom 16. Januar 2017 ( S. 17 ) und der Verbalnote des tunesischen Außenministeriums vom 11. Juli 2017 folgt , dass in Tunesien die Todesstrafe aufgrund eines Moratoriums seit 1991 nicht mehr vollstreckt wird .

| Predicted | Gold |
|---|---|
| `Auswärtigen Amtes` | `Auswärtigen Amtes` |

**Missed by this rule (FN):**

- `tunesischen Außenministeriums` (ORG)
- `Tunesien` (LOC)

**Example 16** (doc_id: `58428`) (sent_id: `58428`)


Zur Begründung seines Rechtsschutzbegehrens wiederholt und vertieft der Antragsteller sein Beschwerdevorbringen und betont , dass die vom Bundesministerium der Verteidigung herangezogene Rechtsgrundlage lediglich eine Verwaltungsvorschrift darstelle und deshalb nicht für die Ablehnung ausreiche .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Example 17** (doc_id: `58663`) (sent_id: `58663`)


bb ) Zwischenzeitlich hatte der Bundestag in Art. 11 Nr. 2 des Gesetzes zur Fortführung des Solidarpaktes , zur Neuordnung des bundesstaatlichen Finanzausgleichs und zur Abwicklung des Fonds " Deutsche Einheit " vom 20. Dezember 2001 ( Solidarpaktfortführungsgesetz - SFG ) ebenfalls eine Neufassung des § 7 Satz 2 GewStG beschlossen .

| Predicted | Gold |
|---|---|
| `Bundestag` | `Bundestag` |

**Missed by this rule (FN):**

- `Art. 11 Nr. 2 des Gesetzes zur Fortführung des Solidarpaktes , zur Neuordnung des bundesstaatlichen Finanzausgleichs und zur Abwicklung des Fonds " Deutsche Einheit " vom 20. Dezember 2001` (NRM)
- `Solidarpaktfortführungsgesetz` (NRM)
- `SFG` (NRM)
- `§ 7 Satz 2 GewStG` (NRM)

**Example 18** (doc_id: `58976`) (sent_id: `58976`)


Ebensowenig ist die Nichtanwendung einzelner Bestimmungen einer umsetzungsbedürftigen Verwaltungsvorschrift des Bundesministeriums der Verteidigung eine dienstliche Maßnahme im Sinne des § 17 Abs. 3 Satz 1 WBO ( BVerwG , Beschluss vom 19. Dezember 2017 - 1 WDS-VR 10.17 - Rn. 19 ) .

| Predicted | Gold |
|---|---|
| `Bundesministeriums der Verteidigung` | `Bundesministeriums der Verteidigung` |

**Missed by this rule (FN):**

- `§ 17 Abs. 3 Satz 1 WBO` (NRM)
- `BVerwG , Beschluss vom 19. Dezember 2017 - 1 WDS-VR 10.17 - Rn. 19` (RS)

**Example 19** (doc_id: `59179`) (sent_id: `59179`)


Dazu hat das Bundesministerium der Verteidigung im Beschwerdebescheid vom 3. November 2017 ( Seite 6 ) nachvollziehbar dargelegt , dass in der Hierarchie der genannten Soldaten im Regelfall mehrere weitere höhere Vorgesetzte zwischengeschaltet sind , die ihrerseits für die Anwendung eines sachgerechten Beurteilungsmaßstabs und damit für die Einhaltung der Richtwerte verantwortlich sind .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Example 20** (doc_id: `59291`) (sent_id: `59291`)


V. 1. Von Seiten des Bundes und der Länder haben das Bundesministerium der Finanzen für die Bundesregierung sowie die Bayerische Staatskanzlei für die Landesregierung Bayern und das Hessische Ministerium der Finanzen für die Landesregierung Hessen Stellung genommen .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Finanzen` | `Bundesministerium der Finanzen` |

**Missed by this rule (FN):**

- `Bundesregierung` (ORG)
- `Bayerische Staatskanzlei` (ORG)
- `Landesregierung Bayern` (ORG)
- `Hessische Ministerium der Finanzen` (ORG)
- `Landesregierung Hessen` (ORG)

**Example 21** (doc_id: `59491`) (sent_id: `59491`)


Es geht somit um die Frage , ob die gesetzlichen Regelungen zu Erstattungsansprüchen nach § 68 AufenthG durch Zusagen einer öffentlichen Stelle zu einem Leistungssystem ( hier : Ausländerbehörde zum Asylbewerberleistungsgesetz auf der Grundlage des Landesaufnahmeprogramms Nordrhein-Westfalen ) zulasten einer anderen öffentlichen Stelle in einem anderen Leistungssystem ( hier : Bundesagentur für Arbeit - BA - in der Grundsicherung für Arbeitsuchende ) abänderbar sind " .

| Predicted | Gold |
|---|---|
| `Bundesagentur für Arbeit` | `Bundesagentur für Arbeit` |

**Missed by this rule (FN):**

- `§ 68 AufenthG` (NRM)
- `Asylbewerberleistungsgesetz` (NRM)
- `Nordrhein-Westfalen` (LOC)
- `BA` (ORG)

**Example 22** (doc_id: `59571`) (sent_id: `59571`)


Die gerichtliche Überprüfung richtet sich auch darauf , ob die vom Bundesministerium der Verteidigung im Wege der Selbstbindung in Erlassen und Richtlinien festgelegten Maßgaben und Verfahrensvorschriften eingehalten sind ( vgl. BVerwG , Beschluss vom 27. Februar 2003 - 1 WB 57.02 - BVerwGE 118 , 25 < 27 > ) , wie sie sich hier insbesondere aus dem Zentralerlass ( ZE ) B- 1300/46 ( " Versetzung , Dienstpostenwechsel , Kommandierung " ) sowie aus den Verwaltungsvorschriften zur Auslandsverwendung von Soldaten ( Erlass des BMVg " Verwendung von Soldaten im Ausland und bei integrierten Stäben im Inland " vom 25. November 1999 , VMBl 2000 , S. 7 ; Zentralerlass B- 1340/9 " Verwendung von Soldaten im Ausland und bei integrierten Stäben " vom 11. Juni 2014 und ZDv A- 1340/9 " Verwendung von Soldatinnen und Soldaten im Ausland " vom 7. Dezember 2016 ) ergeben .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Missed by this rule (FN):**

- `BVerwG , Beschluss vom 27. Februar 2003 - 1 WB 57.02 - BVerwGE 118 , 25 < 27 >` (RS)
- `Zentralerlass ( ZE ) B- 1300/46 ( " Versetzung , Dienstpostenwechsel , Kommandierung " )` (REG)
- `Erlass des BMVg " Verwendung von Soldaten im Ausland und bei integrierten Stäben im Inland " vom 25. November 1999 , VMBl 2000 , S. 7` (REG)
- `Zentralerlass B- 1340/9 " Verwendung von Soldaten im Ausland und bei integrierten Stäben " vom 11. Juni 2014 und ZDv A- 1340/9 " Verwendung von Soldatinnen und Soldaten im Ausland " vom 7. Dezember 2016` (REG)

**Example 23** (doc_id: `59633`) (sent_id: `59633`)


Nach Auskunft des Auswärtigen Amtes und des Bundesinnenministeriums ist die Begnadigung die in der Praxis übliche Vorgehensweise für die Strafrestaussetzung .

| Predicted | Gold |
|---|---|
| `Auswärtigen Amtes` | `Auswärtigen Amtes` |

**Missed by this rule (FN):**

- `Bundesinnenministeriums` (ORG)

**Example 24** (doc_id: `59717`) (sent_id: `59717`)


Es kommt parallel dazu eine weitere Begnadigung nach Art. 372 CPP in Betracht ( vgl. Auskunft des Auswärtigen Amtes vom 7. März 2018 , S. 2 ) .

| Predicted | Gold |
|---|---|
| `Auswärtigen Amtes` | `Auswärtigen Amtes` |

**Missed by this rule (FN):**

- `Art. 372 CPP` (NRM)

**Example 25** (doc_id: `59744`) (sent_id: `59744`)


Die Erstattung entspricht hierbei auch der Billigkeit , da der genannte Mangel so schwerwiegend ist , dass nach einer Abwägung zwischen dem fiskalischen Interesse der Staatskasse und den Belangen der Antragstellerin , den zuletzt genannten der Vorrang gebührt ( vgl. Schulte / Püschel , PatG , 10. Aufl. , § 73 Rn. 138 ) .

| Predicted | Gold |
|---|---|
| `Staatskasse` | `Staatskasse` |

**Missed by this rule (FN):**

- `Schulte / Püschel , PatG , 10. Aufl. , § 73 Rn. 138` (LIT)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53767`) (sent_id: `53767`)


Ob der Erwerb eigener Anteile auf der Gesellschaftsebene entsprechend der durch das BilMoG geänderten handelsrechtlichen Vorschriften ( Einfügung des § 272 Abs. 1a und 1b HGB ) steuerrechtlich nicht mehr als Erwerbsvorgang anzusehen , sondern nunmehr als " Teilliquidation " und daher " wie " eine Kapitalherabsetzung zu behandeln ist ( so BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f. ; zustimmend z.B. Blumenberg / Lechner , DB 2014 , 141 , 147 ; kritisch z.B. Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55 ) , kann im Streitfall mangels Erheblichkeit für die Entscheidung offenbleiben .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BilMoG`(NRM)
- `§ 272 Abs. 1a und 1b HGB`(NRM)
- `BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f.`(REG)
- `Blumenberg / Lechner , DB 2014 , 141 , 147`(LIT)
- `Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55`(LIT)

**Example 1** (doc_id: `54854`) (sent_id: `54854`)


Daran ändert auch die Wiederholung der Verwaltungsauffassung durch das zu § 227 der Abgabenordnung ( AO ) ergangene und erst nach Einreichung der Beschwerdebegründung veröffentlichte BMF-Schreiben vom 29. März 2018 ( BStBl I 2018 , 588 ) nichts .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `BMF-Schreiben vom 29. März 2018 ( BStBl I 2018 , 588 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 227 der Abgabenordnung`(NRM)
- `AO`(NRM)
- `BMF-Schreiben vom 29. März 2018 ( BStBl I 2018 , 588 )`(REG)

**Example 2** (doc_id: `55584`) (sent_id: `55584`)


e ) Soweit das FA - wie bereits das BMF im Verfahren V R 61/03 - einwendet , dass Nr. 34 der Anlage 2 zum UStG auf die Unterpos . 2201 9000 des Zolltarifs verweise ( vgl. dazu auch Schrader , Mehrwertsteuerrecht 2013 , 115 ) , hat sich der BFH bereits mit diesem Argument auseinandergesetzt und ausgeführt , daraus könne ein gesetzlicher Ausschluss des Legens eines Hausanschlusses von der Steuerermäßigung nicht hergeleitet werden ( vgl. BFH-Urteil in BFHE 222 , 176 , BStBl II 2009 , 321 , unter II. 3. d ee , Rz 60 und 62 ) .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `BMF im Verfahren V R 61/03`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BMF im Verfahren V R 61/03`(RS)
- `Nr. 34 der Anlage 2 zum UStG`(NRM)
- `Schrader , Mehrwertsteuerrecht 2013 , 115`(LIT)
- `BFH`(ORG)
- `BFH-Urteil in BFHE 222 , 176 , BStBl II 2009 , 321 , unter II. 3. d ee , Rz 60 und 62`(RS)

**Example 3** (doc_id: `55736`) (sent_id: `55736`)


Die Beschwerdeakte des Bundesministeriums der Verteidigung - R II 2 - Az. : ... - und die Personalgrundakte des Antragstellers haben dem Senat bei der Beratung vorgelegen .

**False Positives:**

- `Bundesministeriums der Verteidigung - R II 2 - ` — partial — gold is substring of pred: `Bundesministeriums der Verteidigung - R II 2 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesministeriums der Verteidigung - R II 2 -`(ORG)

**Example 4** (doc_id: `56059`) (sent_id: `56059`)


Nach dem Schreiben des Bundesministeriums der Finanzen ( BMF ) vom 7. April 2009 ( BStBl I 2009 , 531 ) und Abschn. 12.1 Abs. 1 Satz 3 des Umsatzsteuer-Anwendungserlasses ( UStAE ) seien die Grundsätze der o. g. Rechtsprechung auf das Legen des Hausanschlusses durch ein Wasserversorgungsunternehmen beschränkt .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `Schreiben des Bundesministeriums der Finanzen ( BMF ) vom 7. April 2009 ( BStBl I 2009 , 531 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Schreiben des Bundesministeriums der Finanzen ( BMF ) vom 7. April 2009 ( BStBl I 2009 , 531 )`(REG)
- `Abschn. 12.1 Abs. 1 Satz 3 des Umsatzsteuer-Anwendungserlasses`(REG)
- `UStAE`(REG)

**Example 5** (doc_id: `56323`) (sent_id: `56323`)


aa ) Mit Urteil vom 24. 3. 2016 ( B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 ) hat der Senat bereits entschieden , dass es für den Beginn der aufgeschobenen Versicherungspflicht nach § 7a Abs 6 S 1 SGB IV - mit Wirkung für alle Zweige der Sozialversicherung - auf die Bekanntgabe einer ( ersten ) Entscheidung der Deutschen Rentenversicherung Bund über das Bestehen von " Beschäftigung " ankommt und nicht auf eine ( spätere ) - diese unzulässige Elementenfeststellung korrigierende - Entscheidung über " Versicherungspflicht wegen Beschäftigung " .

**False Positives:**

- `Deutschen Rentenversicherung` — partial — pred is substring of gold: `Deutschen Rentenversicherung Bund`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil vom 24. 3. 2016 ( B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 )`(RS)
- `§ 7a Abs 6 S 1 SGB IV`(NRM)
- `Deutschen Rentenversicherung Bund`(ORG)

**Example 6** (doc_id: `56391`) (sent_id: `56391`)


Bei Regiebetrieben setze die Anerkennung von Rücklagen i. S. des § 20 Abs. 1 Nr. 10 Buchst. b Satz 1 EStG voraus , dass die Zwecke des Betriebs gewerblicher Art ohne die Rücklagenbildung nachhaltig nicht erfüllt werden könnten ( vgl. BMF-Schreiben in BStBl I 2005 , 831 , Rz 23 ) .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `BMF-Schreiben in BStBl I 2005 , 831 , Rz 23`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 20 Abs. 1 Nr. 10 Buchst. b Satz 1 EStG`(NRM)
- `BMF-Schreiben in BStBl I 2005 , 831 , Rz 23`(REG)

**Example 7** (doc_id: `56643`) (sent_id: `56643`)


Mit seiner Revision macht das FA geltend , bei Regiebetrieben setze die Anerkennung von Rücklagen i. S. des § 20 Abs. 1 Nr. 10 Buchst. b EStG voraus , dass die Zwecke des Betriebs gewerblicher Art ohne die Rücklagenbildung nachhaltig nicht erfüllt werden könnten ( vgl. Schreiben des Bundesministeriums der Finanzen - BMF - vom 8. August 2005 IV B 7 -S 2706a - 4/05 , BStBl I 2005 , 831 , Rz 23 ; vom 9. Januar 2015 IV C 2 -S 2706- a / 13/10001 , BStBl I 2015 , 111 , Rz 35 ) .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `Schreiben des Bundesministeriums der Finanzen - BMF - vom 8. August 2005 IV B 7 -S 2706a - 4/05 , BStBl I 2005 , 831 , Rz 23`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 20 Abs. 1 Nr. 10 Buchst. b EStG`(NRM)
- `Schreiben des Bundesministeriums der Finanzen - BMF - vom 8. August 2005 IV B 7 -S 2706a - 4/05 , BStBl I 2005 , 831 , Rz 23`(REG)
- `vom 9. Januar 2015 IV C 2 -S 2706- a / 13/10001 , BStBl I 2015 , 111 , Rz 35`(REG)

**Example 8** (doc_id: `56742`) (sent_id: `56742`)


2. Dies gilt auch für Gutschriften auf dem Wertguthabenkonto eines Fremd-Geschäftsführers einer GmbH ( entgegen BMF-Schreiben vom 17. Juni 2009 , BStBl I 2009 , 1286 , A. IV. 2. b. ) .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `BMF-Schreiben vom 17. Juni 2009 , BStBl I 2009 , 1286 , A. IV. 2. b.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BMF-Schreiben vom 17. Juni 2009 , BStBl I 2009 , 1286 , A. IV. 2. b.`(REG)

**Example 9** (doc_id: `57250`) (sent_id: `57250`)


Damit hat der BFH für Eigenbetriebe bestätigt , dass grundsätzlich jedes " Stehenlassen " der handelsrechtlichen Gewinne als Eigenkapital für Zwecke des Betriebs gewerblicher Art ausreicht , unabhängig davon , ob dies in der Form der Zuführung zu den Gewinnrücklagen , als Gewinnvortrag oder unter einer anderen Position des Eigenkapitals geschieht ( vgl. auch BMF-Schreiben in BStBl I 2015 , 111 , Rz 34 ; Verfügung der Oberfinanzdirektion Karlsruhe vom 7. Oktober 2015 S 270. 6/43 -St 212 , unter V. 2.1.1 ; Bott in Ernst & Young , a. a. O. , § 4 Rz 452.5 ; Gastl in Hidien / Jürgens , a. a. O. , § 6 Rz 101 ) .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `BMF-Schreiben in BStBl I 2015 , 111 , Rz 34`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH`(ORG)
- `BMF-Schreiben in BStBl I 2015 , 111 , Rz 34`(REG)
- `Verfügung der Oberfinanzdirektion Karlsruhe vom 7. Oktober 2015 S 270. 6/43 -St 212 , unter V. 2.1.1`(RS)
- `Bott in Ernst & Young , a. a. O. , § 4 Rz 452.5`(LIT)
- `Gastl in Hidien / Jürgens , a. a. O. , § 6 Rz 101`(LIT)

**Example 10** (doc_id: `57748`) (sent_id: `57748`)


Sie verfügten bei Aufnahme der Tätigkeit jeweils über eine Befreiungsentscheidung der Bundesversicherungsanstalt für Angestellte als Rechtsvorgängerin der beklagten Deutschen Rentenversicherung Bund .

**False Positives:**

- `Deutschen Rentenversicherung` — partial — pred is substring of gold: `Deutschen Rentenversicherung Bund`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Deutschen Rentenversicherung Bund`(ORG)

**Example 11** (doc_id: `58056`) (sent_id: `58056`)


Das tunesische Justizministerium hat in seinem Schreiben vom 1. März 2018 ( S. 1 , Anlage 1 der Auskunft des Auswärtigen Amtes vom 7. März 2018 ) über Gespräche unter anderem mit Vertretern des deutschen Bundesministeriums der Justiz und für Verbraucherschutz ausgeführt , dass im Jahr 2012 insgesamt 122 Todesurteile in lebenslange Freiheitsstrafen umgewandelt worden sind .

**False Positives:**

- `Bundesministeriums der Justiz` — partial — pred is substring of gold: `Bundesministeriums der Justiz und für Verbraucherschutz`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `tunesische Justizministerium`(ORG)
- `Auswärtigen Amtes`(ORG)
- `Bundesministeriums der Justiz und für Verbraucherschutz`(ORG)

**Example 12** (doc_id: `58497`) (sent_id: `58497`)


Die Beschwerdeakten des Bundesministeriums der Verteidigung - R II 2 - 1136/17 und 1377/17 - sowie die Gerichtsakten zu den Verfahren BVerwG 1 WB 4.16 , BVerwG 1 WB 33.16 und BVerwG 1 WDS-VR 10.17 haben dem Senat bei der Beratung vorgelegen .

**False Positives:**

- `Bundesministeriums der Verteidigung - R II 2 - 1136` — partial — pred is substring of gold: `Bundesministeriums der Verteidigung - R II 2 - 1136/17 und 1377/17 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesministeriums der Verteidigung - R II 2 - 1136/17 und 1377/17 -`(RS)
- `Verfahren BVerwG 1 WB 4.16 , BVerwG 1 WB 33.16`(RS)
- `BVerwG 1 WDS-VR 10.17`(RS)

**Example 13** (doc_id: `58671`) (sent_id: `58671`)


Zu diesem Zweck regelt Art § 12 Abs 1 S 1 SpTrUG allein die Frage , ob neben der - vorliegend durch § 75 GmbHG bereits geklärten Entstehung der neuen Kapitalgesellschaften - auch die beabsichtigten Vermögensübergänge im Wege der Einzelrechtsnachfolge wirksam geworden sind ( vgl die Gegenäußerung der Bundesregierung zur Stellungnahme des Bundesrats - BR-Drucks 71 / 1/91 - in BT-Drucks 12/214 S 1 ) und ordnet insofern die Heilung an .

**False Positives:**

- `Bundesregierung` — partial — pred is substring of gold: `Gegenäußerung der Bundesregierung zur Stellungnahme des Bundesrats - BR-Drucks 71 / 1/91 - in BT-Drucks 12/214 S 1`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 12 Abs 1 S 1 SpTrUG`(NRM)
- `§ 75 GmbHG`(NRM)
- `Gegenäußerung der Bundesregierung zur Stellungnahme des Bundesrats - BR-Drucks 71 / 1/91 - in BT-Drucks 12/214 S 1`(LIT)

**Example 14** (doc_id: `59047`) (sent_id: `59047`)


Die Beschwerdeakte des Bundesministeriums der Verteidigung - R II 2 - ... - und die Personalgrundakte des Antragstellers , Hauptteile A bis D , haben dem Senat bei der Beratung vorgelegen .

**False Positives:**

- `Bundesministeriums der Verteidigung - R II 2` — partial — pred is substring of gold: `Bundesministeriums der Verteidigung - R II 2 - ... -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesministeriums der Verteidigung - R II 2 - ... -`(ORG)

</details>

---

## `Hyphenated Company Names`

**F1:** 0.059 | **Precision:** 0.543 | **Recall:** 0.031  

**Format:** `regex`  
**Rule ID:** `c676eb05`  
**Description:**
Matches company names with hyphens or specific patterns like 'E K-Konzerns' or 'A-GbR' that were missed by generic patterns.

**Content:**
```
\b([A-Z][\-]?[A-Z]?\s*(?:GmbH|AG|KG|GbR|Fonds|V\.|B\.\s*V\.|Klinik|Schulzentrum|Finanzamt|Landratsamt|Berufsschulzentrum|Jobcenter|Botschaft|Kammer|Senat|Stelle|Amt|Verband|Zweckverband|Firma|Bank|Verlag|GmbH\s*&\s*Co\.\s*KG|Konzerns?|AG\s*&\s*Co\.\s*KG|S\.r\.l\.))\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.543 | 0.031 | 0.059 | 46 | 25 | 21 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 25 | 21 | 767 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53475`) (sent_id: `53475`)


Gegenstand des BgA war die Verpachtung der städtischen Schwimmbäder an die ... GmbH ( S-GmbH ) , eine 100 % -ige Tochtergesellschaft der Klägerin .

| Predicted | Gold |
|---|---|
| `S-GmbH` | `S-GmbH` |

**Missed by this rule (FN):**

- `... GmbH` (ORG)

**Example 1** (doc_id: `53517`) (sent_id: `53517`)


An der Klägerin sind als Komplementärin die A-GmbH , als Kommanditistin seit dem Jahr 2000 die Holding-KG beteiligt .

| Predicted | Gold |
|---|---|
| `A-GmbH` | `A-GmbH` |

**Example 2** (doc_id: `53594`) (sent_id: `53594`)


Die von der Krankenkasse nicht getragenen Kosten der stationären Behandlung zahlte der Kläger zu 1. Die Therapie der E in der K-Klinik war im Vorfeld durch den Medizinischen Dienst der Krankenversicherungen ( MDK ) auf ihre medizinische Notwendigkeit geprüft und befürwortet worden .

| Predicted | Gold |
|---|---|
| `K-Klinik` | `K-Klinik` |

**Missed by this rule (FN):**

- `E` (PER)
- `Medizinischen Dienst der Krankenversicherungen` (ORG)
- `MDK` (ORG)

**Example 3** (doc_id: `53777`) (sent_id: `53777`)


Am 14. November 2000 wurden den Aktionären der A-AG vereinbarungsgemäß ... Stück vinkulierte Namensaktien an der B-AG übertragen .

| Predicted | Gold |
|---|---|
| `A-AG` | `A-AG` |
| `B-AG` | `B-AG` |

**Example 4** (doc_id: `53825`) (sent_id: `53825`)


Die Beteiligung an der S-GmbH gehörte zum Betriebsvermögen des BgA .

| Predicted | Gold |
|---|---|
| `S-GmbH` | `S-GmbH` |

**Example 5** (doc_id: `54234`) (sent_id: `54234`)


Der Beklagte und Beschwerdeführer ( das Finanzamt - FA - ) erließ gemäß § 27 Abs. 19 Satz 1 UStG einen Änderungsbescheid gegen den Kläger , nach dem er die an Q-KG erbrachte Bauleistung als Steuerschuldner zu versteuern habe .

| Predicted | Gold |
|---|---|
| `Q-KG` | `Q-KG` |

**Missed by this rule (FN):**

- `§ 27 Abs. 19 Satz 1 UStG` (NRM)

**Example 6** (doc_id: `55059`) (sent_id: `55059`)


Der Feststellungsbescheid benennt - neben dem Anleger ( zum Anleger als Inhaltsadressaten vgl. Lübbehüsen in Berger / Steck / Lübbehüsen , a. a. O. , § 15 Rz 59 , 65 ) - mit dem A-Fonds ein Spezial-Sondervermögen als ( weiteren ) möglichen Feststellungsbeteiligten .

| Predicted | Gold |
|---|---|
| `A-Fonds` | `A-Fonds` |

**Missed by this rule (FN):**

- `Lübbehüsen in Berger / Steck / Lübbehüsen , a. a. O. , § 15 Rz 59 , 65` (LIT)

**Example 7** (doc_id: `55305`) (sent_id: `55305`)


Die ehemalige A-AG verpflichtete sich durch Vermögensübertragungsvertrag vom 17. Juni 1999 , ihr Vermögen als Ganzes mit allen Rechten und Pflichten unter Auflösung ohne Abwicklung nach § 174 Abs. 1 des Umwandlungsgesetzes ( UmwG ) im Wege der Vermögensübertragung mit Wirkung zum 1. Januar 2000 , 0:00 Uhr ( handelsrechtlicher Übertragungsstichtag ) , auf die Klägerin zu übertragen .

| Predicted | Gold |
|---|---|
| `A-AG` | `A-AG` |

**Missed by this rule (FN):**

- `§ 174 Abs. 1 des Umwandlungsgesetzes` (NRM)
- `UmwG` (NRM)

**Example 8** (doc_id: `55316`) (sent_id: `55316`)


Da nach § 6 des Vertrags der Jahresabschluss für die M-GmbH innerhalb von sechs Monaten nach Ablauf eines jeden Geschäftsjahres zu erstellen und dem stillen Gesellschafter zu übermitteln war , kann daraus nur der Rückschluss gezogen werden , dass zum hier maßgeblichen Bilanzstichtag von einer Aufstellung des Jahresabschlusses durch den Kläger auszugehen war .

| Predicted | Gold |
|---|---|
| `M-GmbH` | `M-GmbH` |

**Missed by this rule (FN):**

- `§ 6 des Vertrags` (REG)

**Example 9** (doc_id: `55934`) (sent_id: `55934`)


Die Auflösung der A-AG sei erst im Jahr 2012 gemäß § 262 Abs. 1 des Aktiengesetzes ( AktG ) eingetreten .

| Predicted | Gold |
|---|---|
| `A-AG` | `A-AG` |

**Missed by this rule (FN):**

- `§ 262 Abs. 1 des Aktiengesetzes` (NRM)
- `AktG` (NRM)

**Example 10** (doc_id: `55949`) (sent_id: `55949`)


Aus den vorgelegten Steuerakten ergibt sich indes , dass erst mit Schreiben des steuerlichen Beraters der A-GbR vom 29. Mai 2008 eine als Eröffnungsbilanz bezeichnete Aufstellung vorgelegt wurde .

| Predicted | Gold |
|---|---|
| `A-GbR` | `A-GbR` |

**Example 11** (doc_id: `56134`) (sent_id: `56134`)


Da die A-GbR aufgrund ihrer Geschäftsbeziehung der C- B. V. näher stehe als das FA , hätte es dem Kläger oblegen , die Angaben über die wirtschaftlichen Verhältnisse der C- B. V. weiter zu konkretisieren .

| Predicted | Gold |
|---|---|
| `A-GbR` | `A-GbR` |

**Missed by this rule (FN):**

- `C- B. V.` (ORG)
- `C- B. V.` (ORG)

**Example 12** (doc_id: `56397`) (sent_id: `56397`)


Vertretungsberechtigt für die Holding-KG ist die C-GmbH , vertreten durch ihren Geschäftsführer D.

| Predicted | Gold |
|---|---|
| `C-GmbH` | `C-GmbH` |

**Missed by this rule (FN):**

- `D.` (PER)

**Example 13** (doc_id: `56596`) (sent_id: `56596`)


Hierfür wären die Urteile des Landgerichts Cottbus vom ... und des Brandenburgischen Oberlandesgerichts vom ... zu beachten , wonach der Kläger durch Kündigung vom ... 2005 wirksam aus der A-GbR ausgeschlossen wurde .

| Predicted | Gold |
|---|---|
| `A-GbR` | `A-GbR` |

**Missed by this rule (FN):**

- `Landgerichts Cottbus` (ORG)
- `Brandenburgischen Oberlandesgerichts` (ORG)

**Example 14** (doc_id: `56721`) (sent_id: `56721`)


a ) Zu Recht geht das FG allerdings davon aus , dass der Kläger an der M-GmbH & atypisch Still , einer Personengesellschaft i. S. des § 15 Abs. 1 Satz 1 Nr. 2 EStG , über die zwischengeschaltete B-GbR mittelbar i. S. des § 15 Abs. 1 Satz 1 Nr. 2 Satz 2 EStG beteiligt und er daher hinsichtlich der für seine im Dienst der M-GmbH & atypisch Still erbrachten Tätigkeiten wie ein unmittelbar beteiligter Gesellschafter anzusehen war .

| Predicted | Gold |
|---|---|
| `B-GbR` | `B-GbR` |

**Missed by this rule (FN):**

- `M-GmbH & atypisch Still` (ORG)
- `§ 15 Abs. 1 Satz 1 Nr. 2 EStG` (NRM)
- `§ 15 Abs. 1 Satz 1 Nr. 2 Satz 2 EStG` (NRM)
- `M-GmbH & atypisch Still` (ORG)

**Example 15** (doc_id: `56800`) (sent_id: `56800`)


Der Kläger war zu 50 % an der X-GmbH ( GmbH ) beteiligt , über deren Vermögen am 1. Dezember 2004 das Insolvenzverfahren eröffnet wurde .

| Predicted | Gold |
|---|---|
| `X-GmbH` | `X-GmbH` |

**Example 16** (doc_id: `57272`) (sent_id: `57272`)


Dem Vermögen des A-Fonds waren bei der Depotbank überwiegend Aktien und im Übrigen festverzinsliche Wertpapiere , Genussscheine , Geldmarktpapiere sowie in geringem Umfang auch Derivate zugeordnet .

| Predicted | Gold |
|---|---|
| `A-Fonds` | `A-Fonds` |

**Example 17** (doc_id: `57686`) (sent_id: `57686`)


Das FA erließ hiernach am 22. November 2010 einen " Bescheid über die gesonderte - und einheitliche - Feststellung nach § 15 Abs. 1 InvStG für : A-Fonds " , der auch den Anleger ausdrücklich benennt .

| Predicted | Gold |
|---|---|
| `A-Fonds` | `A-Fonds` |

**Missed by this rule (FN):**

- `§ 15 Abs. 1 InvStG` (NRM)

**Example 18** (doc_id: `58089`) (sent_id: `58089`)


Sie war mehrheitlich an der ehemaligen ... AG , seit dem 18. Oktober 1999 B-AG , beteiligt .

| Predicted | Gold |
|---|---|
| `B-AG` | `B-AG` |

**Missed by this rule (FN):**

- `... AG` (ORG)

**Example 19** (doc_id: `58288`) (sent_id: `58288`)


Gemäß § 4 Abs. 2 des Vertrags war u. a. vereinbart , dass der Prinzipal ( M-GmbH ) , dem die alleinige Geschäftsführung oblag , den Wechsel des steuerlichen Beraters nur mit Einwilligung des stillen Gesellschafters vornehmen dürfe .

| Predicted | Gold |
|---|---|
| `M-GmbH` | `M-GmbH` |

**Missed by this rule (FN):**

- `§ 4 Abs. 2 des Vertrags` (REG)

**Example 20** (doc_id: `58937`) (sent_id: `58937`)


d ) Der Kläger hat der X-GmbH nicht etwa ein partiarisches Darlehen gewährt .

| Predicted | Gold |
|---|---|
| `X-GmbH` | `X-GmbH` |

**Example 21** (doc_id: `59187`) (sent_id: `59187`)


aa ) Die M-GmbH & atypisch Still war durch die Eröffnung des Insolvenzverfahrens über das Vermögen der M-GmbH kraft Gesetzes ( § 728 des Bürgerlichen Gesetzbuchs ) aufgelöst ( vgl. Baumbach / Hopt / Roth , HGB , 37. Aufl. , § 234 Rz 5 ) .

| Predicted | Gold |
|---|---|
| `M-GmbH` | `M-GmbH` |

**Missed by this rule (FN):**

- `M-GmbH & atypisch Still` (ORG)
- `§ 728 des Bürgerlichen Gesetzbuchs` (NRM)
- `Baumbach / Hopt / Roth , HGB , 37. Aufl. , § 234 Rz 5` (LIT)

**Example 22** (doc_id: `59488`) (sent_id: `59488`)


Die A-GbR habe ihre steuerlichen Mitwirkungspflichten verletzt .

| Predicted | Gold |
|---|---|
| `A-GbR` | `A-GbR` |

**Example 23** (doc_id: `59897`) (sent_id: `59897`)


So legt der Kläger - soweit erkennbar unwidersprochen durch das FA - dar , dass das mit der A-GbR vereinbarte Projekt eines Ferienresorts die einzige geschäftliche Tätigkeit der C- B. V. dargestellt habe .

| Predicted | Gold |
|---|---|
| `A-GbR` | `A-GbR` |

**Missed by this rule (FN):**

- `C- B. V.` (ORG)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53973`) (sent_id: `53973`)


Die Festsetzung des Streitwerts folgt aus § 47 Abs. 1 Satz 1 und Abs. 3 , § 52 Abs. 1 GKG .

**False Positives:**

- `GKG` — partial — pred is substring of gold: `§ 47 Abs. 1 Satz 1 und Abs. 3 , § 52 Abs. 1 GKG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 47 Abs. 1 Satz 1 und Abs. 3 , § 52 Abs. 1 GKG`(NRM)

**Example 1** (doc_id: `54330`) (sent_id: `54330`)


Mit Kostenrechnung vom 25. Januar 2018 KostL 1905/17 ( IV B 58/17 ) setzte die Kostenstelle des BFH für das Verfahren auf der Grundlage eines Streitwerts von 1.906.096 € Gerichtskosten nach Nr. 6500 des Kostenverzeichnisses zu § 3 Abs. 2 des Gerichtskostengesetzes ( GKG ) in Höhe von 17.512 € fest .

**False Positives:**

- `GKG` — type mismatch — same span as gold: `GKG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `vom 25. Januar 2018 KostL 1905/17 ( IV B 58/17 )`(RS)
- `BFH`(ORG)
- `Nr. 6500 des Kostenverzeichnisses zu § 3 Abs. 2 des Gerichtskostengesetzes`(NRM)
- `GKG`(NRM)

**Example 2** (doc_id: `54878`) (sent_id: `54878`)


Das entspricht in der Sache der Regelung in § 66 Abs 8 GKG , die bei Erinnerungen gegen den Ansatz von Gerichtskosten nach dem GKG zur Anwendung gelangt .

**False Positives:**

- `GKG` — partial — pred is substring of gold: `§ 66 Abs 8 GKG`
- `GKG` — similar text (different position): `§ 66 Abs 8 GKG`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 66 Abs 8 GKG`(NRM)
- `GKG`(NRM)

**Example 3** (doc_id: `55356`) (sent_id: `55356`)


Dies wird beispielhaft durch Regelungen wie in Art. 13 Abs. 1 Nr. 5 Buchst. b Doppelbuchst. dd des Kommunalabgabengesetzes Bayern ( KAG BY ) bestätigt .

**False Positives:**

- `KAG` — partial — pred is substring of gold: `KAG BY`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 13 Abs. 1 Nr. 5 Buchst. b Doppelbuchst. dd des Kommunalabgabengesetzes Bayern`(NRM)
- `KAG BY`(NRM)

**Example 4** (doc_id: `55464`) (sent_id: `55464`)


Mit Beschluss vom 30. November 2017 X E 12/17 hat der Senat durch die Einzelrichterin nach § 66 Abs. 6 Satz 1 GKG die Erinnerung zurückgewiesen , da die Kostenrechnung , insbesondere der zweifache Ansatz der Gebühr von 60 € , inhaltlich zutreffend sei .

**False Positives:**

- `GKG` — partial — pred is substring of gold: `§ 66 Abs. 6 Satz 1 GKG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Beschluss vom 30. November 2017 X E 12/17`(RS)
- `§ 66 Abs. 6 Satz 1 GKG`(NRM)

**Example 5** (doc_id: `55581`) (sent_id: `55581`)


Rechtsfragen , die sich der Vorinstanz nicht gestellt haben oder auf die sie nicht entscheidend abgehoben hat , können aber nicht zur Zulassung der Revision führen , weil ihre Klärung in einem Revisionsverfahren nicht zu erwarten ist ( BVerwG , Beschlüsse vom 29. Juni 1992 - 3 B 102.91 - Buchholz 418.04 Heilpraktiker Nr. 17 S. 6 , vom 5. Oktober 2009 - 6 B 17.09 - Buchholz 442.066 § 24 TKG Nr. 4 Rn. 7 und vom 21. März 2014 - 6 B 55.13 - Buchholz 442.09 § 23 AEG Nr. 3 Rn. 7 ) .

**False Positives:**

- `TKG` — partial — pred is substring of gold: `vom 5. Oktober 2009 - 6 B 17.09 - Buchholz 442.066 § 24 TKG Nr. 4 Rn. 7`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerwG , Beschlüsse vom 29. Juni 1992 - 3 B 102.91 - Buchholz 418.04 Heilpraktiker Nr. 17 S. 6`(RS)
- `vom 5. Oktober 2009 - 6 B 17.09 - Buchholz 442.066 § 24 TKG Nr. 4 Rn. 7`(RS)
- `vom 21. März 2014 - 6 B 55.13 - Buchholz 442.09 § 23 AEG Nr. 3 Rn. 7`(RS)

**Example 6** (doc_id: `56266`) (sent_id: `56266`)


Gegen den nach seiner Angabe am 5. Januar 2018 zugegangenen Beschluss hat der Rügeführer am 18. Januar 2018 Anhörungsrüge erhoben , die einstweilige AdV der angefochtenen Entscheidung nach § 66 Abs. 7 Satz 2 GKG sowie Akteneinsicht beantragt .

**False Positives:**

- `GKG` — partial — pred is substring of gold: `§ 66 Abs. 7 Satz 2 GKG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 66 Abs. 7 Satz 2 GKG`(NRM)

**Example 7** (doc_id: `56591`) (sent_id: `56591`)


Während ansonsten in kostenrechtlichen Verfahren der Erinnerung nach dem GKG bzw dem RVG nunmehr auch in dritter Instanz grundsätzlich eine Entscheidung durch den Einzelrichter vorgesehen ist ( vgl § 66 Abs 6 S 1 GKG bzw § 33 Abs 8 S 1 RVG - s dazu BGH Beschluss vom 23. 4. 2015 - I ZB 73/14 - NJW 2015 , 2194 ) , lässt das SGG bislang auch bei Erinnerungen ( §§ 178 , 189 Abs 2 S 2 SGG ) ein Tätigwerden des Einzelrichters lediglich im Rahmen des § 155 Abs 2 S 1 Nr 5 , Abs 3 und 4 SGG zu ( Reichel in Zeihe / Hauck <Hrsg> , SGG , Stand August 2017 , § 189 RdNr 10a ) .

**False Positives:**

- `GKG` — type mismatch — same span as gold: `GKG`
- `GKG` — similar text (different position): `GKG`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `GKG`(NRM)
- `RVG`(NRM)
- `§ 66 Abs 6 S 1 GKG`(NRM)
- `§ 33 Abs 8 S 1 RVG`(NRM)
- `BGH Beschluss vom 23. 4. 2015 - I ZB 73/14 - NJW 2015 , 2194`(RS)
- `SGG`(NRM)
- `§§ 178 , 189 Abs 2 S 2 SGG`(NRM)
- `§ 155 Abs 2 S 1 Nr 5 , Abs 3 und 4 SGG`(NRM)
- `Reichel in Zeihe / Hauck <Hrsg> , SGG , Stand August 2017 , § 189 RdNr 10a`(LIT)

**Example 8** (doc_id: `56721`) (sent_id: `56721`)


a ) Zu Recht geht das FG allerdings davon aus , dass der Kläger an der M-GmbH & atypisch Still , einer Personengesellschaft i. S. des § 15 Abs. 1 Satz 1 Nr. 2 EStG , über die zwischengeschaltete B-GbR mittelbar i. S. des § 15 Abs. 1 Satz 1 Nr. 2 Satz 2 EStG beteiligt und er daher hinsichtlich der für seine im Dienst der M-GmbH & atypisch Still erbrachten Tätigkeiten wie ein unmittelbar beteiligter Gesellschafter anzusehen war .

**False Positives:**

- `M-GmbH` — partial — pred is substring of gold: `M-GmbH & atypisch Still`
- `M-GmbH` — similar text (different position): `M-GmbH & atypisch Still`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `M-GmbH & atypisch Still`(ORG)
- `§ 15 Abs. 1 Satz 1 Nr. 2 EStG`(NRM)
- `B-GbR`(ORG)
- `§ 15 Abs. 1 Satz 1 Nr. 2 Satz 2 EStG`(NRM)
- `M-GmbH & atypisch Still`(ORG)

**Example 9** (doc_id: `57957`) (sent_id: `57957`)


Der Gewinnfeststellungsbescheid wurde dem Kläger als Empfangsbevollmächtigtem der M-GmbH & atypisch Still bekanntgegeben .

**False Positives:**

- `M-GmbH` — partial — pred is substring of gold: `M-GmbH & atypisch Still`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `M-GmbH & atypisch Still`(ORG)

**Example 10** (doc_id: `58128`) (sent_id: `58128`)


Für diese Auslegung spricht bereits die gesetzliche Vorgabe in § 101 Abs 1 S 1 Nr 4 SGB V , nach der sich " die Partner der BAG " verpflichten müssen , den " bisherigen Praxisumfang " nicht wesentlich zu überschreiten , sowie die ergänzende Vorgabe in § 23a Nr 4 BedarfsplRL aF , nach der die Erklärungen bei der Aufnahme eines Arztes in eine bereits gebildete BAG von allen Vertragsärzten abzugeben sind .

**False Positives:**

- `BAG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 101 Abs 1 S 1 Nr 4 SGB V`(NRM)
- `§ 23a Nr 4 BedarfsplRL aF`(REG)

**Example 11** (doc_id: `58206`) (sent_id: `58206`)


Die Rüge ist innerhalb bestimmter Frist bei dem Gericht zu erheben , dessen Entscheidung angegriffen wird ( § 69a Abs. 2 Satz 1 bis 4 GKG ) .

**False Positives:**

- `GKG` — partial — pred is substring of gold: `§ 69a Abs. 2 Satz 1 bis 4 GKG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 69a Abs. 2 Satz 1 bis 4 GKG`(NRM)

**Example 12** (doc_id: `58850`) (sent_id: `58850`)


Durch das Gesetz zur weiteren Reform der gesetzlichen Rentenversicherungen und über die Fünfzehnte Anpassung der Renten aus den gesetzlichen Rentenversicherungen sowie über die Anpassung der Geldleistungen aus der gesetzlichen Unfallversicherung ( Rentenreformgesetz - RRG ) vom 16. Oktober 1972 ( BGBl. I S. 1965 ) wurde § 48 Abs. 1 Nr. 1 RKG dahin geändert , dass Knappschaftsruhegeld auf Antrag ua. bereits ab der Vollendung des 63. Lebensjahres gewährt wird , wenn die Wartezeit nach § 49 Abs. 3 RKG erfüllt war .

**False Positives:**

- `RKG` — partial — pred is substring of gold: `§ 48 Abs. 1 Nr. 1 RKG`
- `RKG` — similar text (different position): `§ 48 Abs. 1 Nr. 1 RKG`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Gesetz zur weiteren Reform der gesetzlichen Rentenversicherungen und über die Fünfzehnte Anpassung der Renten aus den gesetzlichen Rentenversicherungen sowie über die Anpassung der Geldleistungen aus der gesetzlichen Unfallversicherung ( Rentenreformgesetz - RRG ) vom 16. Oktober 1972 ( BGBl. I S. 1965 )`(NRM)
- `§ 48 Abs. 1 Nr. 1 RKG`(NRM)
- `§ 49 Abs. 3 RKG`(NRM)

**Example 13** (doc_id: `59187`) (sent_id: `59187`)


aa ) Die M-GmbH & atypisch Still war durch die Eröffnung des Insolvenzverfahrens über das Vermögen der M-GmbH kraft Gesetzes ( § 728 des Bürgerlichen Gesetzbuchs ) aufgelöst ( vgl. Baumbach / Hopt / Roth , HGB , 37. Aufl. , § 234 Rz 5 ) .

**False Positives:**

- `M-GmbH` — similar text (different position): `M-GmbH & atypisch Still`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `M-GmbH & atypisch Still`(ORG)
- `M-GmbH`(ORG)
- `§ 728 des Bürgerlichen Gesetzbuchs`(NRM)
- `Baumbach / Hopt / Roth , HGB , 37. Aufl. , § 234 Rz 5`(LIT)

**Example 14** (doc_id: `59220`) (sent_id: `59220`)


Die Honorarnachteile , die für eine BAG mit einem Arzt in der Aufbauphase aufgrund der Zuweisung nur einer fallzahlabhängigen Obergrenze in typischen Konstellationen ( etwa der Übergabe einer Praxis im Wege einer vorübergehenden gemeinsamen Ausübung der Praxistätigkeit ) entstehen , sind nicht die zwangsläufige Folge dessen , dass Ärzten in der Aufbauphase überhaupt das bundesrechtlich geforderte sofortige Wachstum bis zum Fachgruppendurchschnitt eröffnet wird .

**False Positives:**

- `BAG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 15** (doc_id: `59298`) (sent_id: `59298`)


Die Festsetzung des Streitwerts beruht auf § 197a Abs 1 S 1 Teils 1 SGG iVm § 63 Abs 2 S 1 , § 52 Abs 1 und 3 sowie § 47 Abs 1 GKG .

**False Positives:**

- `GKG` — partial — pred is substring of gold: `§ 63 Abs 2 S 1 , § 52 Abs 1 und 3 sowie § 47 Abs 1 GKG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 197a Abs 1 S 1 Teils 1 SGG`(NRM)
- `§ 63 Abs 2 S 1 , § 52 Abs 1 und 3 sowie § 47 Abs 1 GKG`(NRM)

**Example 16** (doc_id: `59563`) (sent_id: `59563`)


Ausgelöst durch veränderte Marktbedingungen und verstärkt durch die Finanzkrise 2008 befindet sich der E K-Konzern seit Jahren in wirtschaftlichen Schwierigkeiten .

**False Positives:**

- `K-Konzern` — partial — pred is substring of gold: `E K-Konzern`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `E K-Konzern`(ORG)

</details>

---

## `Bundesverwaltungsgericht Genitive`

**F1:** 0.053 | **Precision:** 1.000 | **Recall:** 0.027  

**Format:** `regex`  
**Rule ID:** `7f6a80df`  
**Description:**
Matches 'Bundesverwaltungsgericht' and its genitive form 'Bundesverwaltungsgerichts'.

**Content:**
```
\b(Bundesverwaltungsgericht|Bundesverwaltungsgerichts)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.027 | 0.053 | 22 | 22 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 22 | 0 | 765 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53539`) (sent_id: `53539`)


cc ) Stellt ein Bewerber nicht innerhalb eines Monats nach Zugang der Abbruchmitteilung einen Antrag auf Erlass einer einstweiligen Verfügung , darf der Dienstherr nach der Rechtsprechung des Bundesverwaltungsgerichts darauf vertrauen , dass der Bewerber den Abbruch des Auswahlverfahrens nicht angreift , sondern sein Begehren im Rahmen einer neuen Ausschreibung weiterverfolgt ( vgl. BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14 ) .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Missed by this rule (FN):**

- `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14` (RS)

**Example 1** (doc_id: `53619`) (sent_id: `53619`)


( 2 ) Ob die Umwandlung der Todesstrafe in eine lebenslange Freiheitsstrafe bereits zwingend aus dem seit 1991 praktizierten Moratorium folgt , wie es das Bundesverwaltungsgericht angenommen hat , kann dahinstehen .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Example 2** (doc_id: `54169`) (sent_id: `54169`)


Unabhängig von der Frage , ob ein solcher Antrag - entgegen der Auffassung des Antragstellers - gemäß § 67 Abs. 4 VwGO dem Vertretungszwang unterliegt ( vgl. verneinend z.B. : BVerwG , Beschluss vom 6. Juni 1972 - 3 ER 404.71 - Buchholz 310 § 53 VwGO Nr. 6 ; Ziekow , in : Sodan / Ziekow , VwGO , 4. Aufl. 2014 , § 53 Rn. 20 ; Kraft , in : Eyermann , VwGO , 14. Aufl. 2014 , § 53 Rn. 14 ; Unruh , in : HK-Verwaltungsrecht , 3. Aufl. 2013 , § 53 Rn. 15 ; Scheidler , VR 2012 , 113 ; bejahend z.B. : W. - R. Schenke , in : Kopp / Schenke , VwGO , 23. Aufl. 2017 , § 53 Rn. 10 ; Redeker / von Oertzen , VwGO , 15. Aufl. 2010 , § 53 Rn. 5 ; vermittelnd z.B. : Bier , in : Schoch / Schneider / Bier , VwGO , Stand Oktober 2016 , § 53 Rn. 13 ) , ist der Antrag hier jedenfalls deshalb unzulässig , weil die gesetzlichen Voraussetzungen , unter denen das Bundesverwaltungsgericht eine Zuständigkeitsbestimmung überhaupt vornehmen darf , nicht erfüllt sind .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Missed by this rule (FN):**

- `§ 67 Abs. 4 VwGO` (NRM)
- `BVerwG , Beschluss vom 6. Juni 1972 - 3 ER 404.71 - Buchholz 310 § 53 VwGO Nr. 6` (RS)
- `Ziekow , in : Sodan / Ziekow , VwGO , 4. Aufl. 2014 , § 53 Rn. 20` (LIT)
- `Kraft , in : Eyermann , VwGO , 14. Aufl. 2014 , § 53 Rn. 14` (LIT)
- `Unruh , in : HK-Verwaltungsrecht , 3. Aufl. 2013 , § 53 Rn. 15` (LIT)
- `Scheidler , VR 2012 , 113` (LIT)
- `W. - R. Schenke , in : Kopp / Schenke , VwGO , 23. Aufl. 2017 , § 53 Rn. 10` (LIT)
- `Redeker / von Oertzen , VwGO , 15. Aufl. 2010 , § 53 Rn. 5` (LIT)
- `Bier , in : Schoch / Schneider / Bier , VwGO , Stand Oktober 2016 , § 53 Rn. 13` (LIT)

**Example 3** (doc_id: `54506`) (sent_id: `54506`)


Es mag auf sich beruhen , ob mit der Beschwerde davon auszugehen ist , dass den Ausführungen des Bundesverwaltungsgerichts in seinem Urteil vom 26. Januar 2017 ( - 1 C 10.16 - BVerwGE 157 , 208 Rn. 38 )

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Missed by this rule (FN):**

- `Urteil vom 26. Januar 2017 ( - 1 C 10.16 - BVerwGE 157 , 208 Rn. 38 )` (RS)

**Example 4** (doc_id: `55035`) (sent_id: `55035`)


( b ) Es gibt auch keine hinreichend überzeugenden Anhaltspunkte , die vom Bundesverwaltungsgericht dahin hätten gewertet werden müssen , dass von dieser gesetzlich vorgesehenen Möglichkeit der Umwandlung der Strafe und der Strafrestaussetzung de facto kein Gebrauch gemacht werden wird und der Beschwerdeführer damit keine Aussicht auf Entlassung aus der Haft hätte .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Example 5** (doc_id: `55959`) (sent_id: `55959`)


Das Bundesverwaltungsgericht hat unter Anwendung und Auslegung des materiellen Unionsrechts ausführlich erläutert , warum es zu der Überzeugung gelangt ist , dass die Rechtslage in Bezug auf Art. 10 EH-RL eindeutig ist .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Missed by this rule (FN):**

- `Art. 10 EH-RL` (NRM)

**Example 6** (doc_id: `56038`) (sent_id: `56038`)


3. Darüber hinaus hat das Bundesverwaltungsgericht seine Überzeugung von der Verfassungswidrigkeit des § 67 Abs. 2 Satz 3 Halbsatz 1 BbgHG hinreichend dargelegt .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Missed by this rule (FN):**

- `§ 67 Abs. 2 Satz 3 Halbsatz 1 BbgHG` (NRM)

**Example 7** (doc_id: `56455`) (sent_id: `56455`)


Angesichts des Moratoriums , das in Tunesien seit 27 Jahren ohne Ausnahmen eingehalten wird und das im Zuge der Aufklärung durch das Bundesverwaltungsgericht von den tunesischen Behörden auch bezogen auf den konkreten Fall des Beschwerdeführers nochmals bekräftigt wurde , ist die Befürchtung des Beschwerdeführers , dass eine gegen ihn in Tunesien verhängte Todesstrafe vollstreckt werden könnte , nicht begründet .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Missed by this rule (FN):**

- `Tunesien` (LOC)
- `Tunesien` (LOC)

**Example 8** (doc_id: `56642`) (sent_id: `56642`)


Das Bundesverwaltungsgericht hat weder diesen Maßstab verkannt noch die von ihm ermittelte Prognosegrundlage in verfassungsrechtlich relevanter Weise falsch eingeschätzt .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Example 9** (doc_id: `56654`) (sent_id: `56654`)


Die Geschäftsstelle des Bundesverwaltungsgerichts gliedert sich in sechs Arbeitsgruppen , die jeweils von einer Beamtin oder einem Beamten des gehobenen Dienstes geleitet werden .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Example 10** (doc_id: `56677`) (sent_id: `56677`)


Der Gesetzgeber hat insoweit auch für das gerichtliche Asylverfahren an den allgemeinen Grundsätzen des Revisionsrechts festgehalten und für das Bundesverwaltungsgericht keine Befugnis eröffnet , Tatsachen ( würdigungs ) fragen grundsätzlicher Bedeutung in " Länderleitentscheidungen " , wie sie etwa das britische Prozessrecht kennt , zu treffen .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Example 11** (doc_id: `57409`) (sent_id: `57409`)


Dieser Zulassungsgrund ist erfüllt , wenn die Vorinstanz mit einem ihre Entscheidung tragenden abstrakten Rechtssatz in Anwendung derselben Rechtsvorschrift einem ebensolchen Rechtssatz , der in der Rechtsprechung des Bundesverwaltungsgerichts , des Gemeinsamen Senats der obersten Gerichtshöfe des Bundes oder des Bundesverfassungsgerichts aufgestellt worden ist , widersprochen hat .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Missed by this rule (FN):**

- `Gemeinsamen Senats der obersten Gerichtshöfe des Bundes` (ORG)
- `Bundesverfassungsgerichts` (ORG)

**Example 12** (doc_id: `57801`) (sent_id: `57801`)


6. Gegen die Verfügung des Hessischen Ministeriums des Innern und für Sport vom 1. August 2017 erhob der Beschwerdeführer beim Bundesverwaltungsgericht Klage und beantragte die Anordnung der aufschiebenden Wirkung dieser Klage .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Missed by this rule (FN):**

- `Hessischen Ministeriums des Innern und für Sport` (ORG)

**Example 13** (doc_id: `57977`) (sent_id: `57977`)


A. Die Anträge der Kläger auf Bewilligung von Prozesskostenhilfe und Beiordnung ihres Verfahrensbevollmächtigten für das Verfahren vor dem Bundesverwaltungsgericht werden abgelehnt , weil die Rechtsverfolgung - wie sich aus den nachstehenden Gründen ergibt - keine hinreichende Aussicht auf Erfolg bietet ( § 166 VwGO i. V. m. §§ 114 , 121 Abs. 1 ZPO ) .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Missed by this rule (FN):**

- `§ 166 VwGO` (NRM)
- `§§ 114 , 121 Abs. 1 ZPO` (NRM)

**Example 14** (doc_id: `58397`) (sent_id: `58397`)


Eine Divergenz ist nur dann im Sinne des § 133 Abs. 3 Satz 3 VwGO hinreichend bezeichnet , wenn die Beschwerde einen inhaltlich bestimmten , die angefochtene Entscheidung tragenden abstrakten Rechtssatz benennt , mit dem die Vorinstanz einem in der Rechtsprechung des Bundesverwaltungsgerichts oder eines anderen in der Vorschrift ( § 132 Abs. 2 Nr. 2 VwGO ) genannten Gerichts aufgestellten ebensolchen entscheidungstragenden Rechtssatz in Anwendung derselben Rechtsvorschrift widersprochen hat .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Missed by this rule (FN):**

- `§ 133 Abs. 3 Satz 3 VwGO` (NRM)
- `§ 132 Abs. 2 Nr. 2 VwGO` (NRM)

**Example 15** (doc_id: `58435`) (sent_id: `58435`)


Von diesem Rechtssatz des Bundesverwaltungsgerichts sei umfasst , dass Klarstellungen bzw. Konkretisierungen einer Vorschrift , die mit einer Änderungssatzung vorgenommen würden , die Frist nicht nur für die Regelungen , die Gegenstand der Änderungssatzung selbst seien , sondern auch für die Bestandteile der Vorschrift ( neu ) in Lauf setzten , um deren Klarstellung oder Konkretisierung es gehe .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Example 16** (doc_id: `58986`) (sent_id: `58986`)


Für Tatsachenfragen - und damit auch für Unterschiede bei der tatsächlichen Bewertung identischer Tatsachengrundlagen - hat es vorab ausdrücklich bestätigt , dass wegen der Bindung des Revisionsgerichts an die tatsächlichen Feststellungen des Berufungsgerichts ( § 137 Abs. 2 VwGO ) eine weitergehende Vereinheitlichung der Rechtsprechung durch das Bundesverwaltungsgericht ausscheidet .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Missed by this rule (FN):**

- `§ 137 Abs. 2 VwGO` (NRM)

**Example 17** (doc_id: `59775`) (sent_id: `59775`)


Eine Begründung der Verfassungsbeschwerde sei mangels Vorliegen der Beschlussgründe des Bundesverwaltungsgerichts noch nicht möglich .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Example 18** (doc_id: `59893`) (sent_id: `59893`)


2. Die Entscheidung des Bundesverwaltungsgerichts , der Zuteilungsbescheid der Deutschen Emissionshandelsstelle sei rechtmäßig , verletzt die Beschwerdeführerin nicht in ihren Grundrechten .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Missed by this rule (FN):**

- `Deutschen Emissionshandelsstelle` (ORG)

**Example 19** (doc_id: `59915`) (sent_id: `59915`)


● Anonymisierung von Entscheidungen gemäß Anlage 2 der Dienstanweisung über die Erstellung von Schriftgut beim Bundesverwaltungsgericht für den Versand

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Example 20** (doc_id: `59946`) (sent_id: `59946`)


Ein die Zahlungspflicht generell ausschließender Einwand , der letztlich - entgegen der Rechtsprechung des Bundesverwaltungsgerichts ( vgl. Beschluss vom 6. März 1997 - 8 B 246.96 - Buchholz 401.84 Benutzungsgebühren Nr. 86 S. 69 f. ) - auch der Erhebung von Abwassergebühren entgegenstehen würde , kann hieraus aber nicht hergeleitet werden .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Missed by this rule (FN):**

- `Beschluss vom 6. März 1997 - 8 B 246.96 - Buchholz 401.84 Benutzungsgebühren Nr. 86 S. 69 f.` (RS)

**Example 21** (doc_id: `60006`) (sent_id: `60006`)


Der Vertreter des Bundesinteresses beim Bundesverwaltungsgericht unterstützt die Position der Klägerin .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

</details>

---

## `Specific Organization Names (Extended)`

**F1:** 0.048 | **Precision:** 0.645 | **Recall:** 0.025  

**Format:** `regex`  
**Rule ID:** `d094bdd7`  
**Description:**
Matches specific organization names found in training data that are not covered by general patterns, including 'Actavis', 'VCS', 'ADAC', 'JOOP', 'tunesischen Justizministerium', 'Gewerkschaft Erziehung und Wissenschaft', 'X-EWIV', 'A Neurologischen Klinik B', 'Deutsche Wetterdienst', 'A-Vereins', 'Bayerischen Staatsregierung'.

**Content:**
```
\b(?:Gewerkschaft\s+Erziehung\s+und\s+Wissenschaft|GEW|X-EWIV|VCS|A\s+Neurologischen\s+Klinik\s+B|Markenstelle\s+f\u00fcr\s+Klasse\s+\d+\s+des\s+Deutschen\s+Patent-\s+und\s+Markenamts|25\.\s+Senat\s+\(\s*Marken-Beschwerdesenat\s*\)\s+des\s+Bundespatentgerichts|Landessozialgerichts\s+Rheinland-Pfalz|Bundes(?:ministeriums?\s+der\s+Verteidigung|verfassungsgericht)|BVerwG|EuGH|BFH|BVerfG|BGH|BSG|BAG|Bundesarbeitsgerichts|Deutsche\s+Rentenversicherung\s+Bund|Deutsche\s+Rentenversicherung\s*Rheinland|Deutschen\s+Rentenversicherung|Deutsche\s+Rentenversicherung\s+Braunschweig-Hannover|DRV|DB\s+Netz\s+AG|Duden|Centralen\s+Marketing-Gesellschaft\s+der\s+deutschen\s+Agrarwirtschaft\s+mbH|CMA|Schleswig-Holsteinische\s+Oberverwaltungsgericht|Ausw\u00e4rtigen\s+Amtes|Deutschen\s+Stiftung\s+f\u00fcr\s+Internationale|Bundesarbeitsgerichts|Landgericht\s+Potsdam|Sozialgerichts\s+<\s*SG\s*>\s+Hildesheim|Landessozialgerichts\s+<\s*LSG\s*>\s+Niedersachsen-Bremen|11\.\s+Senats\s+des\s+LSG\s+Mecklenburg-Vorpommern|S\u00e4chsischen\s+LSG|Europ\u00e4ischen\s+Parlaments|Bundesministerium\s+des\s+Innern\s*,\s*f\u00fcr\s+Bau\s+und\s+Heimat|Ministerium\s+der\s+Justiz\s+des\s+Landes\s+Nordrhein-Westfalen|Bundesamt\s+f\u00fcr\s+Migration\s+und\s+Fl\u00fcchtlinge|W\u00a1\s+R\.|Deutsche\s+Wetterdienst|A-Vereins|Bayerischen\s+Staatsregierung)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.645 | 0.025 | 0.048 | 31 | 20 | 11 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 20 | 11 | 777 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53424`) (sent_id: `53424`)


Die X-EWIV wäre dem Kläger in vollem Umfang auskunfts- und rechenschaftspflichtig gewesen .

| Predicted | Gold |
|---|---|
| `X-EWIV` | `X-EWIV` |

**Example 1** (doc_id: `54212`) (sent_id: `54212`)


Ist es dem Kläger im Rahmen seiner deshalb nötigen Ermittlungen aufgrund des Verhaltens des FG-Präsidenten nicht möglich , diesen Verfahrensmangel zu substantiieren , so hat dies allein zur Folge , dass der BFH insoweit einen geringeren Maßstab der Darlegung des Verfahrensmangels genügen lassen muss .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Example 2** (doc_id: `54330`) (sent_id: `54330`)


Mit Kostenrechnung vom 25. Januar 2018 KostL 1905/17 ( IV B 58/17 ) setzte die Kostenstelle des BFH für das Verfahren auf der Grundlage eines Streitwerts von 1.906.096 € Gerichtskosten nach Nr. 6500 des Kostenverzeichnisses zu § 3 Abs. 2 des Gerichtskostengesetzes ( GKG ) in Höhe von 17.512 € fest .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Missed by this rule (FN):**

- `vom 25. Januar 2018 KostL 1905/17 ( IV B 58/17 )` (RS)
- `Nr. 6500 des Kostenverzeichnisses zu § 3 Abs. 2 des Gerichtskostengesetzes` (NRM)
- `GKG` (NRM)

**Example 3** (doc_id: `55566`) (sent_id: `55566`)


ccc ) „ Music ” entspricht dem deutschen Wort „ Musik ” ( Langenscheidts Schulwörterbuch Englisch , 1986 ) und hat die Bedeutung „ Tonkunst “ , „ Komposition “ oder „ Musikstücke “ ( Duden - Die deutsche Rechtschreibung , 26. Aufl. 2013 ) .

| Predicted | Gold |
|---|---|
| `Duden` | `Duden` |

**Example 4** (doc_id: `55815`) (sent_id: `55815`)


Im Rahmen der Prüfung , ob die Klägerin diese Zweifel ausräumen konnte , hat das Landesarbeitsgericht der Klägerin rechtliches Gehör nach Maßgabe der Rechtsprechung des Bundesverfassungsgerichts und des Bundesarbeitsgerichts gewährt .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgerichts` | `Bundesarbeitsgerichts` |

**Missed by this rule (FN):**

- `Bundesverfassungsgerichts` (ORG)

**Example 5** (doc_id: `56042`) (sent_id: `56042`)


In den vorliegenden Verfahren steht die dem Beschwerdeführer zu I. sowie den Beschwerdeführerinnen zu II. bis IV. jeweils vorgeworfene Teilnahme an ( Warn- ) Streikmaßnahmen , zu denen die GEW aufgerufen hatte , im Zusammenhang mit seinerzeitigen Tarifverhandlungen im öffentlichen Dienst und ist daher auch nicht von vornherein zur Förderung der mit dem Hauptarbeitskampf verfolgten Ziele offensichtlich ungeeignet ( vgl. auch BAGE 123 , 134 < 146 Rn. 37 > ) .

| Predicted | Gold |
|---|---|
| `GEW` | `GEW` |

**Missed by this rule (FN):**

- `BAGE 123 , 134 < 146 Rn. 37 >` (RS)

**Example 6** (doc_id: `56177`) (sent_id: `56177`)


Die summarische Aufzählung von Vereinen und Verbänden im Anhang genüge nicht den vom BFH aufgestellten Anforderungen .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Example 7** (doc_id: `56302`) (sent_id: `56302`)


Der Kläger hat dem Übergang seines Arbeitsverhältnisses von der Beklagten auf die VCS mit Schreiben vom 3. Juni 2015 nicht wirksam widersprochen .

| Predicted | Gold |
|---|---|
| `VCS` | `VCS` |

**Example 8** (doc_id: `56645`) (sent_id: `56645`)


Mit Schreiben seines Prozessbevollmächtigten vom 3. Juni 2015 widersprach der Kläger gegenüber der Beklagten dem Übergang seines Arbeitsverhältnisses von dieser auf die VCS , bot seine Arbeitsleistung an und machte Ansprüche auf Annahmeverzugslohn geltend .

| Predicted | Gold |
|---|---|
| `VCS` | `VCS` |

**Example 9** (doc_id: `56974`) (sent_id: `56974`)


Nach Erteilung der Jahresabrechnung hatte der Kläger einen Differenzbetrag an die X-EWIV zu erstatten ; ein etwaiger Überschuss hätte hingegen zusätzlich an ihn ausgekehrt werden müssen ( Nr. 3 Abs. 4 des Verwaltungsvertrags ) .

| Predicted | Gold |
|---|---|
| `X-EWIV` | `X-EWIV` |

**Missed by this rule (FN):**

- `Nr. 3 Abs. 4 des Verwaltungsvertrags` (REG)

**Example 10** (doc_id: `57037`) (sent_id: `57037`)


Hierbei ist zu berücksichtigen , dass es sich bei Möbel um Einrichtungsgegenstände zur Ausstattung eines Raumes , damit er benutzt und bewohnt werden kann , handelt ( vgl. Duden , Deutsches Universalwörterbuch , 3. Auflage , Seite 1027 ) .

| Predicted | Gold |
|---|---|
| `Duden` | `Duden` |

**Example 11** (doc_id: `57162`) (sent_id: `57162`)


Das Substantiv „ Farbe “ bezeichnet in der Umgangssprache farbgebende Stoffe ( Duden , a. a. O. , S. 576 ) .

| Predicted | Gold |
|---|---|
| `Duden` | `Duden` |

**Example 12** (doc_id: `57250`) (sent_id: `57250`)


Damit hat der BFH für Eigenbetriebe bestätigt , dass grundsätzlich jedes " Stehenlassen " der handelsrechtlichen Gewinne als Eigenkapital für Zwecke des Betriebs gewerblicher Art ausreicht , unabhängig davon , ob dies in der Form der Zuführung zu den Gewinnrücklagen , als Gewinnvortrag oder unter einer anderen Position des Eigenkapitals geschieht ( vgl. auch BMF-Schreiben in BStBl I 2015 , 111 , Rz 34 ; Verfügung der Oberfinanzdirektion Karlsruhe vom 7. Oktober 2015 S 270. 6/43 -St 212 , unter V. 2.1.1 ; Bott in Ernst & Young , a. a. O. , § 4 Rz 452.5 ; Gastl in Hidien / Jürgens , a. a. O. , § 6 Rz 101 ) .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Missed by this rule (FN):**

- `BMF-Schreiben in BStBl I 2015 , 111 , Rz 34` (REG)
- `Verfügung der Oberfinanzdirektion Karlsruhe vom 7. Oktober 2015 S 270. 6/43 -St 212 , unter V. 2.1.1` (RS)
- `Bott in Ernst & Young , a. a. O. , § 4 Rz 452.5` (LIT)
- `Gastl in Hidien / Jürgens , a. a. O. , § 6 Rz 101` (LIT)

**Example 13** (doc_id: `57916`) (sent_id: `57916`)


Der Beschwerdeführer hält die Zweifel des Bundesarbeitsgerichts an der Verfassungsmäßigkeit des § 14 Abs. 2 Satz 2 TzBfG für verfehlt .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgerichts` | `Bundesarbeitsgerichts` |

**Missed by this rule (FN):**

- `§ 14 Abs. 2 Satz 2 TzBfG` (NRM)

**Example 14** (doc_id: `58110`) (sent_id: `58110`)


Dies hat der BFH bei § 4 Nr. 18 UStG getan .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Missed by this rule (FN):**

- `§ 4 Nr. 18 UStG` (NRM)

**Example 15** (doc_id: `58655`) (sent_id: `58655`)


1. Nach der ständigen Rechtsprechung des Bundesarbeitsgerichts kann es im Einzelfall gegen das Verbot widersprüchlichen Verhaltens ( „ venire contra factum proprium “ ) als Ausprägung des Grundsatzes von Treu und Glauben ( § 242 BGB ) verstoßen , wenn sich der Arbeitgeber auf eine Fehlerhaftigkeit der bisherigen tariflichen Bewertung beruft .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgerichts` | `Bundesarbeitsgerichts` |

**Missed by this rule (FN):**

- `§ 242 BGB` (NRM)

**Example 16** (doc_id: `58960`) (sent_id: `58960`)


Dies folgt daraus , dass die VCS den Kläger durch Unterrichtungsschreiben vom 26. Juli 2007 im Rahmen einer Unterrichtung nach § 613a Abs. 5 BGB - unstreitig - über den mit dem Betriebsübergang verbundenen Übergang seines Arbeitsverhältnisses unter Mitteilung des geplanten Zeitpunkts sowie des Gegenstands des Betriebsübergangs und des Betriebsübernehmers in Textform in Kenntnis gesetzt und über sein Widerspruchsrecht nach § 613a Abs. 6 BGB belehrt hatte und der Kläger in Kenntnis dieser Umstände ab dem Betriebsübergang am 1. September 2007 mehr als sieben Jahre für die VCS gearbeitet hat , ohne von seinem Widerspruchsrecht Gebrauch zu machen .

| Predicted | Gold |
|---|---|
| `VCS` | `VCS` |
| `VCS` | `VCS` |

**Missed by this rule (FN):**

- `§ 613a Abs. 5 BGB` (NRM)
- `§ 613a Abs. 6 BGB` (NRM)

**Example 17** (doc_id: `59133`) (sent_id: `59133`)


III. Da nach alledem das Widerspruchsrecht des Klägers zum Zeitpunkt seiner Ausübung durch den Kläger bereits verwirkt war , verbleibt es dabei , dass das Arbeitsverhältnis des Klägers infolge des Betriebsübergangs am 1. September 2007 gemäß § 613a Abs. 1 Satz 1 BGB von der Beklagten auf die VCS übergegangen ist und damit nicht mit der Beklagten über den 31. August 2007 hinaus fortbestand .

| Predicted | Gold |
|---|---|
| `VCS` | `VCS` |

**Missed by this rule (FN):**

- `§ 613a Abs. 1 Satz 1 BGB` (NRM)

**Example 18** (doc_id: `59605`) (sent_id: `59605`)


Die umfassenden vertraglichen Aufgaben der X-EWIV hätten der Annahme , die Verfügungsbefugnis hätte beim Kläger gelegen , nicht entgegengestanden .

| Predicted | Gold |
|---|---|
| `X-EWIV` | `X-EWIV` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53769`) (sent_id: `53769`)


Dieses Begriffsverständnis deckt sich mit dem allgemeinen Sprachgebrauch , wonach die Kündigung die Lösung eines Vertrags ist ( Duden Das große Wörterbuch der deutschen Sprache 3. Aufl. Stichwort : Kündigung ) .

**False Positives:**

- `Duden` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `54578`) (sent_id: `54578`)


Mit dem Begriff „ erwarten “ wird ausgedrückt , dass man fest mit dem Eintreten eines Ereignisses rechnet ( vgl. Duden Das Bedeutungswörterbuch 4. Aufl. S. 1075 ) .

**False Positives:**

- `Duden` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `55858`) (sent_id: `55858`)


Nach Auffassung des vorlegenden Gerichts findet Art. 8 der Richtlinie 2008 / 94 / EG des Europäischen Parlaments und des Rates vom 22. Oktober 2008 über den Schutz der Arbeitnehmer bei Zahlungsunfähigkeit des Arbeitgebers auch dann Anwendung , wenn die Pensionskasse - ohne selbst zahlungsunfähig gemäß Art. 2 Abs. 1 der Richtlinie 2008 / 94 / EG zu sein - Leistungskürzungen mit Zustimmung der staatlichen Finanzdienstleistungsaufsicht vornimmt , der Arbeitgeber diese Kürzungen aber nicht ausgleichen kann , weil er selbst zahlungsunfähig ist .

**False Positives:**

- `Europäischen Parlaments` — partial — pred is substring of gold: `Art. 8 der Richtlinie 2008 / 94 / EG des Europäischen Parlaments und des Rates vom 22. Oktober 2008 über den Schutz der Arbeitnehmer bei Zahlungsunfähigkeit des Arbeitgebers`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 8 der Richtlinie 2008 / 94 / EG des Europäischen Parlaments und des Rates vom 22. Oktober 2008 über den Schutz der Arbeitnehmer bei Zahlungsunfähigkeit des Arbeitgebers`(NRM)
- `Art. 2 Abs. 1 der Richtlinie 2008 / 94 / EG`(NRM)

**Example 3** (doc_id: `56819`) (sent_id: `56819`)


Nach § 33 Abs. 3 TV DRV KBS endet das Arbeitsverhältnis nicht , wenn der Beschäftigte seine Weiterbeschäftigung binnen zwei Wochen nach Zugang des Rentenbescheids schriftlich beantragt hat und er nach seinem vom zuständigen Rentenversicherungsträger festgestellten Leistungsvermögen auf seinem bisherigen oder einem anderen geeigneten und freien Arbeitsplatz weiterbeschäftigt werden könnte , soweit dringende dienstliche bzw. betriebliche Gründe nicht entgegenstehen .

**False Positives:**

- `DRV` — partial — pred is substring of gold: `§ 33 Abs. 3 TV DRV KBS`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 33 Abs. 3 TV DRV KBS`(REG)

**Example 4** (doc_id: `57389`) (sent_id: `57389`)


Es bestünden erhebliche Zweifel , ob Angehörige dieser Gruppe entsprechend den Anforderungen des Art. 28 Abs. 2 der Richtlinie 2013 / 32 / EU des Europäischen Parlaments und des Rates vom 26. Juni 2013 zu gemeinsamen Verfahren für die Zuerkennung und Aberkennung des internationalen Schutzes ( ABl. L 180 S. 60 ) behandelt würden .

**False Positives:**

- `Europäischen Parlaments` — partial — pred is substring of gold: `Art. 28 Abs. 2 der Richtlinie 2013 / 32 / EU des Europäischen Parlaments und des Rates vom 26. Juni 2013 zu gemeinsamen Verfahren für die Zuerkennung und Aberkennung des internationalen Schutzes ( ABl. L 180 S. 60 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 28 Abs. 2 der Richtlinie 2013 / 32 / EU des Europäischen Parlaments und des Rates vom 26. Juni 2013 zu gemeinsamen Verfahren für die Zuerkennung und Aberkennung des internationalen Schutzes ( ABl. L 180 S. 60 )`(NRM)

**Example 5** (doc_id: `57669`) (sent_id: `57669`)


Der Lohnzahlungszeitraum kann daher nur dem Arbeitsvertragsverhältnis , dh den arbeitsrechtlichen Vereinbarungen oder einer betrieblichen Übung entnommen werden ( BFH Urteil vom 10. 3. 2004 - VI R 27/99 - BFH / NV 2004 , 1239 ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH Urteil vom 10. 3. 2004 - VI R 27/99 - BFH / NV 2004 , 1239`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH Urteil vom 10. 3. 2004 - VI R 27/99 - BFH / NV 2004 , 1239`(RS)

**Example 6** (doc_id: `57949`) (sent_id: `57949`)


Der Senat folgt insoweit der Rechtsprechung des Bundesverfassungsgerichts für das verfassungsrechtliche Verfahren ( vgl BVerfG Nichtannahmebeschluss vom 23. 12. 2016 - 1 BvR 3511/13 - Juris RdNr 3 mwN ; ebenso BFH Beschluss vom 8. 10. 2015 - VII B 147/14 - Juris RdNr 4 ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH Beschluss vom 8. 10. 2015 - VII B 147/14 - Juris RdNr 4`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesverfassungsgerichts`(ORG)
- `BVerfG Nichtannahmebeschluss vom 23. 12. 2016 - 1 BvR 3511/13 - Juris RdNr 3`(RS)
- `BFH Beschluss vom 8. 10. 2015 - VII B 147/14 - Juris RdNr 4`(RS)

**Example 7** (doc_id: `59135`) (sent_id: `59135`)


Die von der Unterrichtung über den Eintritt der auflösenden Bedingung nach §§ 21 , 15 Abs. 2 TzBfG iVm. § 33 Abs. 2 TV DRV KBS ausgehenden tatsächlichen Wirkungen für die Beendigung des Arbeitsverhältnisses sind denjenigen einer Kündigung ähnlich .

**False Positives:**

- `DRV` — partial — pred is substring of gold: `§ 33 Abs. 2 TV DRV KBS`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§§ 21 , 15 Abs. 2 TzBfG`(NRM)
- `§ 33 Abs. 2 TV DRV KBS`(REG)

**Example 8** (doc_id: `59390`) (sent_id: `59390`)


Nach der gebotenen verfassungskonformen Auslegung des § 33 Abs. 3 TV DRV KBS kommt es aber nicht auf diesen Zeitpunkt , sondern darauf an , ob eine Weiterbeschäftigungsmöglichkeit für den Arbeitnehmer bei Zugang der Mitteilung des Arbeitgebers über den Eintritt der auflösenden Bedingung vorhanden ist ( vgl. BAG 30. August 2017 - 7 AZR 204/16 - Rn. 26 ) .

**False Positives:**

- `DRV` — partial — pred is substring of gold: `§ 33 Abs. 3 TV DRV KBS`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 33 Abs. 3 TV DRV KBS`(REG)
- `BAG 30. August 2017 - 7 AZR 204/16 - Rn. 26`(RS)

**Example 9** (doc_id: `59747`) (sent_id: `59747`)


Um das ihm nach § 33 Abs. 3 TV DRV KBS zustehende Recht effektiv wahrnehmen zu können , muss der Arbeitnehmer wissen , welche Rechtsfolgen von einem Rentenbescheid auf sein Arbeitsverhältnis ausgehen und welche Mitwirkung ihm im Hinblick auf eine Wahrnehmung seiner Bestandsschutzinteressen nach Bewilligung einer Rente wegen teilweiser Erwerbsminderung obliegt ( vgl. BAG 23. Juli 2014 - 7 AZR 771/12 - Rn. 66 , BAGE 148 , 357 ) .

**False Positives:**

- `DRV` — partial — pred is substring of gold: `§ 33 Abs. 3 TV DRV KBS`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 33 Abs. 3 TV DRV KBS`(REG)
- `BAG 23. Juli 2014 - 7 AZR 771/12 - Rn. 66 , BAGE 148 , 357`(RS)

**Example 10** (doc_id: `59837`) (sent_id: `59837`)


Vielmehr soll die Bestimmung des § 3 Abs. 1 Nr. 4 DesignG bzw. § 3 Abs. 1 Nr. 4 GeschmMG , mit welcher die fakultative Vorgabe des Art. 11 Abs. 2 Buchst. c der Richtlinie 98 / 71 / EG des Europäischen Parlaments und des Rates über den rechtlichen Schutz von Mustern und Modellen vom 13. Oktober 1998 ( GeschmM-RL ) in deutsches Recht umgesetzt wurde , es erleichtern , Zeichen von öffentlichem Interesse von einer Monopolisierung durch ein Geschmacksmuster bzw. Design auszuschließen ( vgl. Begründung zum Entwurf eines Gesetzes zur Reform des Geschmacksmusterrechts BlPMZ 2004 , 222 , 229 re. Sp. ) .

**False Positives:**

- `Europäischen Parlaments` — partial — pred is substring of gold: `Art. 11 Abs. 2 Buchst. c der Richtlinie 98 / 71 / EG des Europäischen Parlaments und des Rates über den rechtlichen Schutz von Mustern und Modellen`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 3 Abs. 1 Nr. 4 DesignG`(NRM)
- `§ 3 Abs. 1 Nr. 4 GeschmMG`(NRM)
- `Art. 11 Abs. 2 Buchst. c der Richtlinie 98 / 71 / EG des Europäischen Parlaments und des Rates über den rechtlichen Schutz von Mustern und Modellen`(NRM)
- `GeschmM-RL`(NRM)
- `Begründung zum Entwurf eines Gesetzes zur Reform des Geschmacksmusterrechts BlPMZ 2004 , 222 , 229 re. Sp.`(LIT)

</details>

---

## `Specific German Organizations`

**F1:** 0.047 | **Precision:** 0.233 | **Recall:** 0.026  

**Format:** `regex`  
**Rule ID:** `21d25153`  
**Description:**
Matches specific German organizations and associations found in training data, including genitive forms, specific names, and abbreviations that were previously missed.

**Content:**
```
\b(?:S\u00e4chsische\s+LSG|Bundeswehr|NATO|Bundeszentrale\s+f\u00fcr\s+politische\s+Bildung|Bundeswehrflugplatz\s+B\u00fcchel|Bundesnachrichtendienst|BND|Arbeitgebervereinigung\s+energiewirtschaftlicher\s+Unternehmen\s+e\.\s*V\.|IG\s+Metall|Ausw\u00e4rtige\s+Amt|Amtsgericht\s+P\.|Bundesarbeitsgericht|Bundessozialgericht|Bundesverwaltungsgericht|Bundesfinanzhof|Bundespatentgericht|BVerwG|BGH|BSG|BAG|BVerfG|EuGH|EGMR|DPMA|BaFin|SG|FG|LAG|LSG|BA|H\u00c4VG|PKK|KCK|HPG|EED|JOOP|KAGURA|Schott|Urban\s+&\s+Schwarzenberg|National\s+Union\s+of\s+Rail|Maritime\s+and\s+Transport\s+Workers|Europ\u00e4ische\s+Gerichtshof\s+f\u00fcr\s+Menschenrechte|Deutscher\s+Apotheker\s+Verlag|B-KG|VEB\s*Elektronik\s*G\.|EZB|Bundestag|Bundesrat|Bundesregierung|Bundesverwaltungsgericht|Bundesfinanzhof|Bundespatentgericht|BVerwG|BGH|BSG|BAG|BVerfG|EuGH|EGMR|DPMA|BaFin|SG|FG|LAG|LSG|BA|H\u00c4VG|PKK|KCK|HPG|EED|JOOP|KAGURA|Schott|Urban\s+&\s+Schwarzenberg|National\s+Union\s+of\s+Rail|Maritime\s+and\s+Transport\s+Workers|Europ\u00e4ische\s+Gerichtshof\s+f\u00fcr\s+Menschenrechte|Deutscher\s+Apotheker\s+Verlag|B-KG|VEB\s*Elektronik\s*G\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.233 | 0.026 | 0.047 | 90 | 21 | 69 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 21 | 69 | 766 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53773`) (sent_id: `53773`)


Vielmehr hat das LSG nach erfolgter Zurückverweisung durch das BSG nach umfangreicher weiterer Sachaufklärung den Anspruch des Klägers erneut geprüft und eine Entscheidung gefällt .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 1** (doc_id: `54446`) (sent_id: `54446`)


b3 ) Soweit der BGH in seinem Beschluss vom 1. April 1965 ( Ia ZB 20/64 – „ Patentanwaltskosten “ , GRUR 1965 , 621 ) Doppelvertretungskosten im Gebrauchsmuster-Löschungsverfahren als regelmäßig nicht berücksichtigungsfähig erachtet hat , geht der Senat davon aus , dass diese Entscheidung überholt ist .

| Predicted | Gold |
|---|---|
| `BGH` | `BGH` |

**Missed by this rule (FN):**

- `Beschluss vom 1. April 1965 ( Ia ZB 20/64 – „ Patentanwaltskosten “ , GRUR 1965 , 621 )` (RS)

**Example 2** (doc_id: `54482`) (sent_id: `54482`)


e ) Eine andere Wertung folgt schließlich nicht aus der Rechtsprechung des BSG zur Rechtslage vor dem 1. 1. 2011 , nach der höhere tatsächliche Kosten bei zentraler Warmwassererzeugung nur dann anstelle des schätzungsweise ermittelten pauschalen Anteils der Warmwassererzeugungskosten in der Regelleistung von den Aufwendungen für Heizung in Abzug zu bringen waren ( vgl oben 4. a ) , wenn diese Kosten über die Einrichtung getrennter Zähler oder sonstiger Vorrichtungen konkret zu erfassen waren ( vgl nur BSG vom 27. 2. 2008 - B 14/11 b AS 15/07 R - BSGE 100 , 94 = SozR 4 - 4200 § 22 Nr 5 , RdNr 27 ) .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Missed by this rule (FN):**

- `BSG vom 27. 2. 2008 - B 14/11 b AS 15/07 R - BSGE 100 , 94 = SozR 4 - 4200 § 22 Nr 5 , RdNr 27` (RS)

**Example 3** (doc_id: `54707`) (sent_id: `54707`)


Im Übrigen fehlt es an Darlegungen dazu , an welcher Stelle seiner " Entscheidung " der BGH die vom Beklagten behauptete Aussage überhaupt getroffen haben soll .

| Predicted | Gold |
|---|---|
| `BGH` | `BGH` |

**Example 4** (doc_id: `55127`) (sent_id: `55127`)


Die einzige Auseinandersetzung des BGH mit landesrechtlichen Bestattungspflichten erfolgt in RdNr 12 seines Hinweisbeschlusses ; eine Aussage dergestalt , wie sie vom Beklagten in der Frage 1 formuliert worden ist , hat der BGH an dieser Stelle nicht getroffen .

| Predicted | Gold |
|---|---|
| `BGH` | `BGH` |

**Example 5** (doc_id: `55426`) (sent_id: `55426`)


Das in diesem Zusammenhang genannte Schreiben vom 27. März 2008 habe das Bundesarbeitsgericht bei seiner damaligen Entscheidung bereits berücksichtigt .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgericht` | `Bundesarbeitsgericht` |

**Example 6** (doc_id: `55822`) (sent_id: `55822`)


Auf die Beschwerde der Klägerin hat das BSG die Sache zur erneuten Verhandlung und Entscheidung an das LSG zurückverwiesen ( Beschluss vom 12. 8. 2013 ) .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 7** (doc_id: `56680`) (sent_id: `56680`)


Zu der ähnlichen Regelung in § 18 Abs. 5 des Tarifvertrags über das Sozialkassenverfahren im Baugewerbe vom 20. Dezember 1999 , wonach Erstattungsforderungen des Arbeitgebers gegen die Urlaubs- und Lohnausgleichskasse mit der Maßgabe zweckgebunden sind , dass der Arbeitgeber über sie nur verfügen kann , wenn das bei der Einzugsstelle bestehende Beitragskonto keinen Debetsaldo ausweist und er seinen Meldepflichten entsprochen hat , hat das Bundesarbeitsgericht entschieden , die Erfüllung der Beitragspflicht sei keine Voraussetzung für das Entstehen des Erstattungsanspruchs des Arbeitgebers ; § 18 Abs. 5 des Tarifvertrags begründe aber bei nicht vollständiger Erfüllung der Beitragspflicht ein Hindernis für die Durchsetzung des bereits mit der Auszahlung der Urlaubsvergütung entstandenen Anspruchs ( BAG , Urteil vom 14. Dezember 2011 - 10 AZR 517/10 , AP Nr. 338 zu TVG § 1 Tarifverträge : Bau , Rn. 27 mwN ) .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgericht` | `Bundesarbeitsgericht` |

**Missed by this rule (FN):**

- `§ 18 Abs. 5 des Tarifvertrags über das Sozialkassenverfahren im Baugewerbe` (REG)
- `§ 18 Abs. 5 des Tarifvertrags` (REG)
- `BAG , Urteil vom 14. Dezember 2011 - 10 AZR 517/10 , AP Nr. 338 zu TVG § 1 Tarifverträge : Bau , Rn. 27` (RS)

**Example 8** (doc_id: `57124`) (sent_id: `57124`)


1. Der Deutsche Gewerkschaftsbund und ver.di meinen , die einschränkende Auslegung des § 14 Abs. 2 Satz 2 TzBfG durch das Bundesarbeitsgericht überschreite die Grenze zulässiger Rechtsfortbildung .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgericht` | `Bundesarbeitsgericht` |

**Missed by this rule (FN):**

- `Deutsche Gewerkschaftsbund` (ORG)
- `ver.di` (ORG)
- `§ 14 Abs. 2 Satz 2 TzBfG` (NRM)

**Example 9** (doc_id: `57221`) (sent_id: `57221`)


Ausdrücklich von einer Stellungnahme abgesehen haben die Bundesregierung , der Bundesrat , das Ministerium für Migration , Justiz und Verbraucherschutz des Freistaats Thüringen , das Justizministerium Mecklenburg-Vorpommern sowie das Bundesarbeitsgericht .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgericht` | `Bundesarbeitsgericht` |

**Missed by this rule (FN):**

- `Bundesregierung` (ORG)
- `Bundesrat` (ORG)
- `Ministerium für Migration , Justiz und Verbraucherschutz des Freistaats Thüringen` (ORG)
- `Justizministerium Mecklenburg-Vorpommern` (ORG)

**Example 10** (doc_id: `57335`) (sent_id: `57335`)


Gemäß § 160a Abs 5 SGG kann das BSG in dem Beschluss über die Nichtzulassungsbeschwerde das angefochtene Urteil aufheben und die Sache zur erneuten Verhandlung und Entscheidung an das LSG zurückverweisen , wenn die Voraussetzungen des § 160 Abs 2 Nr 3 SGG vorliegen .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Missed by this rule (FN):**

- `§ 160a Abs 5 SGG` (NRM)
- `§ 160 Abs 2 Nr 3 SGG` (NRM)

**Example 11** (doc_id: `57679`) (sent_id: `57679`)


IV. Zu Vorlage und Verfassungsbeschwerde haben der Deutsche Gewerkschaftsbund ( DGB ) , die Vereinte Dienstleistungsgewerkschaft ( ver.di ) , die Bundesvereinigung der Deutschen Arbeitgeberverbände e. V. ( BDA ) , das Bundesarbeitsgericht und die Beklagten der Ausgangsverfahren Stellung genommen .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgericht` | `Bundesarbeitsgericht` |

**Missed by this rule (FN):**

- `Deutsche Gewerkschaftsbund` (ORG)
- `DGB` (ORG)
- `Vereinte Dienstleistungsgewerkschaft` (ORG)
- `ver.di` (ORG)
- `Bundesvereinigung der Deutschen Arbeitgeberverbände e. V.` (ORG)
- `BDA` (ORG)

**Example 12** (doc_id: `57730`) (sent_id: `57730`)


Das Bundesarbeitsgericht orientiert sich bei der Auslegung von § 14 Abs. 2 TzBfG zwar maßgebend am Grundrecht der Berufsfreiheit in Art. 12 Abs. 1 GG .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgericht` | `Bundesarbeitsgericht` |

**Missed by this rule (FN):**

- `§ 14 Abs. 2 TzBfG` (NRM)
- `Art. 12 Abs. 1 GG` (NRM)

**Example 13** (doc_id: `57771`) (sent_id: `57771`)


2. Das Bundesarbeitsgericht hatte § 14 Abs. 2 Satz 2 TzBfG zunächst dahin ausgelegt , dass dieselben Arbeitsvertragsparteien nur bei der erstmaligen Einstellung eine sachgrundlose Befristung vereinbaren können .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgericht` | `Bundesarbeitsgericht` |

**Missed by this rule (FN):**

- `§ 14 Abs. 2 Satz 2 TzBfG` (NRM)

**Example 14** (doc_id: `57921`) (sent_id: `57921`)


c ) Eine Zweitbeurteilung war gemäß der Bestimmungen Ziff. 7.5 i. V. m. Nr. 7. 4. Halbs. 2 BB BND nicht zu erstellen .

| Predicted | Gold |
|---|---|
| `BND` | `BND` |

**Example 15** (doc_id: `58153`) (sent_id: `58153`)


Auf die Beschwerde des Klägers hat das BSG mit Beschluss vom 14. 11. 2013 ( B 9 V 33/13 B ) das Urteil des LSG aufgehoben und die Sache zurückverwiesen , weil das LSG das Paul-Ehrlich-Institut ( Prof. Dr. C. ) nicht zu Einwänden des Klägers ergänzend befragt hat .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Missed by this rule (FN):**

- `Beschluss vom 14. 11. 2013 ( B 9 V 33/13 B )` (RS)
- `Paul-Ehrlich-Institut` (ORG)
- `C.` (PER)

**Example 16** (doc_id: `58209`) (sent_id: `58209`)


Das Bundesarbeitsgericht hat das Urteil des Landesarbeitsgerichts aufgehoben und die Sache zur neuen Verhandlung und Entscheidung an das Berufungsgericht zurückverwiesen .

| Predicted | Gold |
|---|---|
| `Bundesarbeitsgericht` | `Bundesarbeitsgericht` |

**Example 17** (doc_id: `58450`) (sent_id: `58450`)


In einer ersten Regelbeurteilung vom 23. April 2013 zum Stichtag 1. April 2013 vergab der seinerzeitige Leiter der Abteilung X des BND ( Herr Dr. A. ) das Gesamturteil 7. Auf Einwendungen des Klägers hob der BND diese dienstliche Beurteilung wegen formeller Fehler auf .

| Predicted | Gold |
|---|---|
| `BND` | `BND` |

**Missed by this rule (FN):**

- `Abteilung X des BND` (ORG)
- `A.` (PER)

**Example 18** (doc_id: `58501`) (sent_id: `58501`)


Das Oberverwaltungsgericht hat angenommen , dass eine Prüfung des § 4 Abs. 3 und 4 PassG auf die Vereinbarkeit mit dem Grundgesetz wegen der unionsrechtlichen Determinierung nicht stattfinden kann und die Vereinbarkeit der zwingenden unionsrechtlichen Vorgaben des Art. 1 Abs. 1 und 2 der Verordnung ( EG ) 2252/2004 mit der Charta der Grundrechte der EU und anderem höherrangigen Unionsrecht durch die Rechtsprechung des EuGH mit bindender Wirkung geklärt ist .

| Predicted | Gold |
|---|---|
| `EuGH` | `EuGH` |

**Missed by this rule (FN):**

- `§ 4 Abs. 3 und 4 PassG` (NRM)
- `Grundgesetz` (NRM)
- `Art. 1 Abs. 1 und 2 der Verordnung ( EG ) 2252/2004` (NRM)
- `Charta der Grundrechte der EU` (NRM)

**Example 19** (doc_id: `58892`) (sent_id: `58892`)


Soweit nach der Rspr des BGH der entschädigungspflichtige Erwerbsschaden im zivilen Schadensersatzrecht ( §§ 842 , 843 , 252 BGB , § 11 StVG ) auch den Arbeitgeberanteil am Gesamtsozialversicherungsbeitrag umfasst ( BGHZ 173 , 169 , 174 ; BGHZ 139 , 167 , 172 ; BGHZ 43 , 378 , 382 f ) , beruht dies auf Besonderheiten des normativen Schadensbegriffs ( vgl BGHZ 173 , 169 , 174 ; BGHZ 43 , 378 , 382 ff ) und hat für die Beurteilung des Vergütungsbegriffs in § 35a Abs 6a SGB IV keine Bedeutung .

| Predicted | Gold |
|---|---|
| `BGH` | `BGH` |

**Missed by this rule (FN):**

- `§§ 842 , 843 , 252 BGB` (NRM)
- `§ 11 StVG` (NRM)
- `BGHZ 173 , 169 , 174` (RS)
- `BGHZ 139 , 167 , 172` (RS)
- `BGHZ 43 , 378 , 382 f` (RS)
- `BGHZ 173 , 169 , 174` (RS)
- `BGHZ 43 , 378 , 382 ff` (RS)
- `§ 35a Abs 6a SGB IV` (NRM)

**Example 20** (doc_id: `59848`) (sent_id: `59848`)


Mit Wirkung vom 1. März 2011 wurde der Kläger wiederum zum BND versetzt .

| Predicted | Gold |
|---|---|
| `BND` | `BND` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53527`) (sent_id: `53527`)


Sollte es hierauf ankommen , wird das FG die erforderlichen tatsächlichen Feststellungen zu treffen haben .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `53588`) (sent_id: `53588`)


Diesbezüglich fehlt es insbesondere an Vorbringen dazu , dass die Entscheidung des LSG auf " diesem Mangel " beruhen kann , dh es hätte Darlegungen zur mangelnden oder zumindest eingeschränkten Verwertbarkeit des Gutachtens bedurft .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `53715`) (sent_id: `53715`)


aa ) Obwohl die Parteien in Klausel 33 des Vertriebsvertrags die Geltung des Rechts der USA und des Staates Kalifornien vereinbart haben , ist das FG den deutschen Grundsätzen über die Auslegung von Willenserklärungen und Verträgen gefolgt .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `USA`(LOC)
- `Kalifornien`(LOC)

**Example 3** (doc_id: `53763`) (sent_id: `53763`)


Wird als Verfahrensmangel gerügt , das FG habe einen gestellten Beweisantrag übergangen ( Rüge mangelnder Sachaufklärung gemäß § 76 Abs. 1 Satz 1 FGO ) , so ist darzulegen , welche Tatfrage aufklärungsbedürftig ist , welche Beweismittel das FG zu welchem Beweisthema nicht erhoben hat , die genauen Fundstellen ( Schriftsatz mit Datum und Seitenzahl , Terminprotokoll ) , in denen die Beweismittel und Beweisthemen angeführt worden sind , das voraussichtliche Ergebnis der Beweisaufnahme , inwiefern das Urteil des FG aufgrund dessen sachlich-rechtlicher Auffassung auf der unterbliebenen Beweisaufnahme beruhen kann und dass die Nichterhebung der Beweise vor dem FG rechtzeitig gerügt worden ist oder aufgrund des Verhaltens des FG nicht mehr vor diesem gerügt werden konnte ( ständige BFH-Rechtsprechung , z.B. Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85 , und vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125 , m. w. N. ) .

**False Positives:**

- `FG` — similar text (different position): `§ 76 Abs. 1 Satz 1 FGO`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 76 Abs. 1 Satz 1 FGO`(NRM)
- `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`(RS)
- `vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125`(RS)

**Example 4** (doc_id: `53773`) (sent_id: `53773`)


Vielmehr hat das LSG nach erfolgter Zurückverweisung durch das BSG nach umfangreicher weiterer Sachaufklärung den Anspruch des Klägers erneut geprüft und eine Entscheidung gefällt .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG`(ORG)

**Example 5** (doc_id: `53867`) (sent_id: `53867`)


Zwar kann das LSG von einer Entscheidung ua des BSG auch dann abweichen , wenn es einen der höchstrichterlichen Rechtsprechung widersprechenden Rechtssatz nur sinngemäß und in scheinbar fallbezogene Ausführungen gekleidet entwickelt .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG`(ORG)

**Example 6** (doc_id: `53890`) (sent_id: `53890`)


Im Berufungsverfahren hat der zuständige Berichterstatter des LSG den Sachverständigen Prof. Dr. T. im Termin zur mündlichen Verhandlung ergänzend angehört .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `T.`(PER)

**Example 7** (doc_id: `53927`) (sent_id: `53927`)


Die Klägerin beantragt sinngemäß , das angefochtene Urteil und die Einspruchsentscheidung vom 19. Mai 2014 aufzuheben und den Einkommensteuerbescheid für 2011 vom 17. September 2013 dahingehend zu ändern , dass die Einkommensteuer auf 33.553 € festgesetzt wird , hilfsweise , das angefochtene Urteil aufzuheben und die Sache zur anderweitigen Verhandlung und Entscheidung an das FG zurückzuverweisen .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `53954`) (sent_id: `53954`)


Davon kann schließlich ungeachtet der fehlenden statistischen Erhebungen im Allgemeinen auch im Fall hier nicht hinreichend sicher ausgegangen werden , nachdem zwar der Energieverbrauch des Klägers im streitbefangenen Zeitraum nach Einschätzung des LSG für einen Haushalt mit dezentraler Warmwassererzeugung als durchschnittlich anzusehen ist , die Ausgaben für Haushaltsstrom von 50,27 Euro bzw 44,58 Euro monatlich ( 603,18 Euro bzw 534,93 Euro ÷ 12 ) mit den darauf entfallenden Leistungen zur Sicherung des Lebensunterhalts in Höhe von 36,29 Euro bzw 37,66 Euro monatlich ( 2011 : 28,29 Euro Regelbedarfsanteil Strom + 8 Euro Mehrbedarfspauschale ; 2012 : 29,06 Euro Regelbedarfsanteil Strom + 8,60 Euro Mehrbedarfspauschale ) jedoch nicht vollständig zu bestreiten waren .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 9** (doc_id: `54108`) (sent_id: `54108`)


Mit Beschluss vom 23. September 2015 hat die Patentabteilung des DPMA den Antrag zurückgewiesen .

**False Positives:**

- `DPMA` — partial — pred is substring of gold: `Patentabteilung des DPMA`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Patentabteilung des DPMA`(ORG)

**Example 10** (doc_id: `54175`) (sent_id: `54175`)


Zutreffend führt das LSG in diesem Zusammenhang zudem aus , dass es nicht darauf ankommt , ob Arbeitseinsätze im Rahmen eines Dauerarbeitsverhältnisses von vorneherein feststehen oder von Mal zu Mal vereinbart werden .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 11** (doc_id: `54267`) (sent_id: `54267`)


bb ) Ebenso zutreffend hat das FG auf den Normzweck des § 7 g Abs. 3 EStG hingewiesen .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 7 g Abs. 3 EStG`(NRM)

**Example 12** (doc_id: `54360`) (sent_id: `54360`)


In dem hier streitgegenständlichen Wiederaufnahmeverfahren 2 K 154/15 vernahm das FG in der mündlichen Verhandlung vom 19. Mai 2017 den Prozessbevollmächtigten , dessen früheren Mitarbeiter sowie den Sachbearbeiter des FA als Zeugen zu den Inhalten und Umständen des Gesprächs vom 26. Mai 2015 und wies die Klage ab .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Wiederaufnahmeverfahren 2 K 154/15`(RS)

**Example 13** (doc_id: `54448`) (sent_id: `54448`)


Welchen Einfluss die aufrechterhaltene Stationierung von Atomwaffen in Büchel für das Verhalten von Terroristen ( und im Konflikt mit NATO-Staaten stehenden Drittstaaten ) habe , entziehe sich einer gerichtlichen Feststellung .

**False Positives:**

- `NATO` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Büchel`(LOC)

**Example 14** (doc_id: `54607`) (sent_id: `54607`)


aa ) Zu Recht ist das FG der Auffassung des Klägers nicht gefolgt , der Wortlaut des § 7 g Abs. 3 EStG stehe im Streitfall einer Rückgängigmachung des Abzugs entgegen .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 7 g Abs. 3 EStG`(NRM)

**Example 15** (doc_id: `54708`) (sent_id: `54708`)


Der weiteren Aufklärung habe für das SG die Weigerung der Klägerin zur Abgabe entsprechender Schweigepflichtentbindungserklärungen entgegengestanden .

**False Positives:**

- `SG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 16** (doc_id: `55088`) (sent_id: `55088`)


Zwar hat der Senat durchaus zur Kenntnis genommen , dass die Klägerin in dem von ihr benannten Schriftsatz vom 11. 2. 2016 - auf die Anhörung des LSG zur Entscheidung durch Beschluss - durch die Bezugnahme auf den Schriftsatz vom 7. 3. 2016 ihren Antrag bei Prof. Dr. T. nachzufragen , in welcher Form und in welchem Umfang er an der Erstellung des Sachverständigengutachtens beteiligt gewesen sei , wiederholt hat .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `T.`(PER)

**Example 17** (doc_id: `55103`) (sent_id: `55103`)


Festgestellt hat das LSG insoweit nur , dass K. bis zu ihrer Aufnahme in das Diakonissenhaus An. bei ihrer Mutter in B. , im heutigen Kreisgebiet des Beklagten , " gemeldet " war ; auf die einwohnerrechtliche Meldung kommt es jedoch für die Bestimmung des gewöhnlichen Aufenthalts nicht an ( BSG SozR 5870 § 1 Nr 4 ; SozR 3 - 5870 § 2 Nr 36 ) .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `K.`(PER)
- `Diakonissenhaus An.`(ORG)
- `B.`(LOC)
- `BSG SozR 5870 § 1 Nr 4`(RS)
- `SozR 3 - 5870 § 2 Nr 36`(RS)

**Example 18** (doc_id: `55113`) (sent_id: `55113`)


Der Kläger subsumiert die Sachverhaltskonstellation seines Falls vielmehr selbst anhand der vorgenannten Rechtsprechung des BSG und zieht daraus im Kern den Schluss , dass das LSG auf der Basis dieser höchstrichterlichen Rechtsprechung seinen Fall " falsch entschieden " habe .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG`(ORG)

**Example 19** (doc_id: `55217`) (sent_id: `55217`)


Mit Fax vom 16. 5. 2017 hat das LSG der Klägerin mitgeteilt , dass der Termin vom 17. 5. 2017 nicht aufgehoben werde .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 20** (doc_id: `55312`) (sent_id: `55312`)


Gemäß § 118 Abs. 2 FGO hat das FG für den Senat auch bindend festgestellt , dass keine willkürliche Schätzung bewusst zu Lasten des Klägers erfolgte und keine Umstände vorlagen , die die Schätzung als Verstoß gegen eine ordnungsgemäße Verwaltung erscheinen lassen .

**False Positives:**

- `FG` — similar text (different position): `§ 118 Abs. 2 FGO`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 118 Abs. 2 FGO`(NRM)

**Example 21** (doc_id: `55328`) (sent_id: `55328`)


Sie lässt unberücksichtigt , dass das LSG in seinen entscheidungstragend herangezogenen Obersätzen der BSG-Rechtsprechung nicht " im Grundsätzlichen " ausdrücklich widersprochen und mit dieser Rechtsprechung nicht zu vereinbarende eigene Rechtssätze aufgestellt hat .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 22** (doc_id: `55403`) (sent_id: `55403`)


Auf die Revision des Klägers hob der erkennende Senat mit Urteil in BFHE 248 , 462 , BStBl II 2015 , 730 das Urteil des FG auf und verwies die Sache an das FG zurück .

**False Positives:**

- `FG` — no gold match — likely missing annotation
- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `Urteil in BFHE 248 , 462 , BStBl II 2015 , 730`(RS)

**Example 23** (doc_id: `55668`) (sent_id: `55668`)


Die zulässige Revision des Klägers ist im Sinne der Aufhebung des SG-Urteils und der Zurückverweisung der Sache an das SG zur anderweitigen Verhandlung und Entscheidung begründet ( § 170 Abs 2 Satz 2 Sozialgerichtsgesetz < SGG > ) .

**False Positives:**

- `SG` — similar text (different position): `§ 170 Abs 2 Satz 2 Sozialgerichtsgesetz`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 170 Abs 2 Satz 2 Sozialgerichtsgesetz`(NRM)
- `SGG`(NRM)

**Example 24** (doc_id: `55722`) (sent_id: `55722`)


Vielmehr hat sich das LSG die Feststellungen aus dem Strafurteil nach eigener Prüfung zu eigen gemacht und dabei neben den Feststellungen aus dem Strafurteil nicht nur den Inhalt des Erstattungsbescheides aus dem Jahr 2013 zur Honorarrückforderung in Höhe von 216 492,33 Euro berücksichtigt , sondern auch den Inhalt der Strafakten ausgewertet , aus denen hervorgeht , dass sich das Urteil des Amtsgerichts Fürth nicht allein auf das Geständnis des Klägers stützt , sondern auch auf das Ergebnis einer umfangreichen Beweisaufnahme .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Amtsgerichts Fürth`(ORG)

**Example 25** (doc_id: `55822`) (sent_id: `55822`)


Auf die Beschwerde der Klägerin hat das BSG die Sache zur erneuten Verhandlung und Entscheidung an das LSG zurückverwiesen ( Beschluss vom 12. 8. 2013 ) .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG`(ORG)

**Example 26** (doc_id: `55919`) (sent_id: `55919`)


3. Die vorgenannten Voraussetzungen sind im Streitfall nicht erfüllt , weil nach den Feststellungen des FG die gesamte Dauer des Straßentransports 30 1/2 bis 31 Stunden betrug und damit die grundsätzlich vorgeschriebene Höchstdauer von 29 Stunden überschritten wurde .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 27** (doc_id: `55987`) (sent_id: `55987`)


Soweit in den Streitjahren die von A erbrachten Reisevorleistungen unbelastet von der Steuer bleiben , weil sie in Österreich entgegen den unionsrechtlichen Bestimmungen der Art. 306 ff. MwStSystRL nicht der Margenbesteuerung unterworfen werden , ist dies - wie das FG in Bezug auf Steuerausfälle zu Recht erkannt hat - notwendige Folge des Gebots , das Unionsrecht in jedem Falle gegenüber dem entgegenstehenden nationalen Recht durchzusetzen .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `A`(PER)
- `Österreich`(LOC)
- `Art. 306 ff. MwStSystRL`(NRM)

**Example 28** (doc_id: `56139`) (sent_id: `56139`)


B. Das LSG hat im Ergebnis zu Recht das Urteil des SG aufgehoben .

**False Positives:**

- `SG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 29** (doc_id: `56146`) (sent_id: `56146`)


Aufgrund dessen ist das angefochtene Urteil gemäß § 160a Abs 5 iVm § 160 Abs 2 Nr 3 SGG aufzuheben und die Sache an das LSG zurückzuverweisen .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 160a Abs 5 iVm § 160 Abs 2 Nr 3 SGG`(NRM)

**Example 30** (doc_id: `56245`) (sent_id: `56245`)


In diesem Sinne ist vorliegend jedoch kein Verfahrensfehler des LSG aufzeigbar .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 31** (doc_id: `56378`) (sent_id: `56378`)


Sie führt zur Aufhebung der Vorentscheidung und zur Zurückverweisung der Sache an das FG zur anderweitigen Verhandlung und Entscheidung gemäß § 116 Abs. 6 der Finanzgerichtsordnung ( FGO ) .

**False Positives:**

- `FG` — similar text (different position): `FGO`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 116 Abs. 6 der Finanzgerichtsordnung`(NRM)
- `FGO`(NRM)

**Example 32** (doc_id: `56536`) (sent_id: `56536`)


Aufgrund dessen ist das angefochtene Urteil gemäß § 160a Abs 5 iVm § 160 Abs 2 Nr 3 SGG aufzuheben und die Sache an das LSG zurückzuverweisen .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 160a Abs 5 iVm § 160 Abs 2 Nr 3 SGG`(NRM)

**Example 33** (doc_id: `56750`) (sent_id: `56750`)


Es bedarf hier keiner näheren Prüfung , unter welchen Voraussetzungen das Revisionsgericht die Auslegung eins Antrags durch das LSG überprüfen kann ( näher dazu BSG SozR 5070 § 10a Nr 3 zur fehlenden Bindung des BSG an eine vom LSG aufgestellte Auslegungsregel ) .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG SozR 5070 § 10a Nr 3`(RS)
- `BSG`(ORG)

**Example 34** (doc_id: `56765`) (sent_id: `56765`)


Da der gerügte Verfahrensmangel auch vorliegt , konnte der angefochtene Beschluss gemäß § 160a Abs 5 SGG aufgehoben und die Sache zur erneuten Verhandlung und Entscheidung an das LSG zurückverwiesen werden .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 160a Abs 5 SGG`(NRM)

**Example 35** (doc_id: `57056`) (sent_id: `57056`)


Insbesondere tragen die von ihm gezogenen Parallelen zu bereits ergangener Rechtsprechung des 3. Senats des BSG zur Leistungspflicht der Krankenkassen für " hilfsmittelbezogene Nebenleistungen im Rahmen des bestimmungsgemäßen Gebrauchs " in Bezug auf die Entsorgung von Inkontinenzmaterialien nicht .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `3. Senats des BSG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `3. Senats des BSG`(ORG)

**Example 36** (doc_id: `57178`) (sent_id: `57178`)


c ) Aus Rz 58 des BFH-Urteils in BFHE 255 , 300 , BStBl II 2017 , 590 ergibt sich ebenfalls nichts anderes , weil die hier zu beurteilenden Leistungen der Klägerin nach den tatsächlichen Feststellungen des FG die Wasserleitung von der Grundstücksgrenze bis ins Haus betreffen und damit nicht dem Aufbau und Betrieb einer leistungsfähigen Wasserversorgung für den jeweiligen Zweckverband gedient haben .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Rz 58 des BFH-Urteils in BFHE 255 , 300 , BStBl II 2017 , 590`(RS)

**Example 37** (doc_id: `57230`) (sent_id: `57230`)


2. Das LSG hat zu Recht die Berufung der Beklagten gegen das Urteil des SG zurückgewiesen .

**False Positives:**

- `SG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 38** (doc_id: `57335`) (sent_id: `57335`)


Gemäß § 160a Abs 5 SGG kann das BSG in dem Beschluss über die Nichtzulassungsbeschwerde das angefochtene Urteil aufheben und die Sache zur erneuten Verhandlung und Entscheidung an das LSG zurückverweisen , wenn die Voraussetzungen des § 160 Abs 2 Nr 3 SGG vorliegen .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 160a Abs 5 SGG`(NRM)
- `BSG`(ORG)
- `§ 160 Abs 2 Nr 3 SGG`(NRM)

**Example 39** (doc_id: `57358`) (sent_id: `57358`)


5. Wegen dieses Verfahrensfehlers , der zur Fehlerhaftigkeit des angefochtenen Urteils geführt hat , ist die Vorentscheidung gemäß § 116 Abs. 6 FGO aufzuheben und die Sache an das FG zur anderweitigen Verhandlung und Entscheidung zurückzuverweisen .

**False Positives:**

- `FG` — similar text (different position): `§ 116 Abs. 6 FGO`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 116 Abs. 6 FGO`(NRM)

**Example 40** (doc_id: `57407`) (sent_id: `57407`)


Soweit der Kläger das Fehlen weiterer Ermittlungen und hierauf basierender Feststellungen rügt , ist hierauf schon deshalb nicht weiter einzugehen , weil das LSG nach Zurückverweisung ausgehend von der Rechtsauffassung des Senats ohnehin den Sachverhalt weiter aufzuklären hat .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 41** (doc_id: `57410`) (sent_id: `57410`)


3. Ferner hat das FG den Entnahmewert dieses Grundstücks in nicht zu beanstandender Weise mit 108.225 € errechnet .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 42** (doc_id: `57594`) (sent_id: `57594`)


Er zeigt damit keine grobe Fehleinschätzung des LSG auf .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 43** (doc_id: `57800`) (sent_id: `57800`)


Es kann offenbleiben , welche rechtliche Bedeutung dem Umstand zukommt , dass die Klägerin nach den Feststellungen des LSG die Coils abweichend von den vom Hersteller angegebenen Indikationen bei der Versicherten eingesetzt hat .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 44** (doc_id: `57870`) (sent_id: `57870`)


Soweit - wie vorliegend - Verstöße gegen die tatrichterliche Sachaufklärungspflicht ( § 103 SGG ) gerügt werden , muss die Beschwerdebegründung hierzu jeweils folgende Punkte enthalten :( 1. ) Bezeichnung eines für das Revisionsgericht ohne Weiteres auffindbaren , bis zuletzt aufrechterhaltenen Beweisantrags , dem das LSG nicht gefolgt ist , ( 2. ) Wiedergabe der Rechtsauffassung des LSG , aufgrund derer bestimmte Tatfragen als klärungsbedürftig hätten erscheinen müssen , ( 3. ) Darlegung der von dem betreffenden Beweisantrag berührten Tatumstände , die zu einer weiteren Sachaufklärung Anlass gegeben hätten , ( 4. ) Angabe des voraussichtlichen Ergebnisses der unterbliebenen Beweisaufnahme und ( 5. ) Schilderung , dass und warum die Entscheidung des LSG auf der angeblich fehlerhaft unterlassenen Beweisaufnahme beruhen kann , das LSG mithin bei Kenntnis des behaupteten Ergebnisses der unterlassenen Beweisaufnahme von seinem Rechtsstandpunkt aus zu einem anderen , dem Beschwerdeführer günstigeren Ergebnis hätte gelangen können ( vgl BSG Beschluss vom 21. 12. 2017 - B 9 SB 70/17 B - Juris RdNr 3 mwN ) .

**False Positives:**

- `LSG` — no gold match — likely missing annotation
- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `§ 103 SGG`(NRM)
- `BSG Beschluss vom 21. 12. 2017 - B 9 SB 70/17 B - Juris RdNr 3`(RS)

**Example 45** (doc_id: `57920`) (sent_id: `57920`)


Auch hinsichtlich der AfA gab das FG der Klage teilweise statt , als es für die Jahre 2008 und 2009 bei Gesamtanschaffungskosten von 1.336.797,90 € , einem Gebäudeanteil von 57,25 % und einem AfA-Satz von 2 % zu einer AfA von 15.306,34 € kam .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 46** (doc_id: `57941`) (sent_id: `57941`)


Sie finden sich im FG-Urteil ab Seite 12. So hat das FG zum einen festgestellt , dass die Formulierung in den Steuererklärungen klar und verständlich sei .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 47** (doc_id: `58031`) (sent_id: `58031`)


Eine Abweichung ( Divergenz ) ist nur dann hinreichend dargelegt , wenn aufgezeigt wird , mit welcher genau bestimmten entscheidungserheblichen rechtlichen Aussage die angegriffene Entscheidung des LSG von welcher ebenfalls genau bezeichneten rechtlichen Aussage des BSG , des Gemeinsamen Senats der obersten Gerichtshöfe des Bundes ( GmSOGB ) oder des BVerfG abweicht ( BSG SozR 1500 § 160a Nr 21 , 29 und 54 ) .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG`(ORG)
- `Gemeinsamen Senats der obersten Gerichtshöfe des Bundes`(ORG)
- `GmSOGB`(ORG)
- `BVerfG`(ORG)
- `BSG SozR 1500 § 160a Nr 21 , 29 und 54`(RS)

**Example 48** (doc_id: `58044`) (sent_id: `58044`)


a ) Die Klägerin beruft sich darauf , dass das FG in dem Verfahren 2 K 2039/15 E , G , U rechtliches Gehör verletzt habe .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Verfahren 2 K 2039/15 E , G , U`(RS)

**Example 49** (doc_id: `58153`) (sent_id: `58153`)


Auf die Beschwerde des Klägers hat das BSG mit Beschluss vom 14. 11. 2013 ( B 9 V 33/13 B ) das Urteil des LSG aufgehoben und die Sache zurückverwiesen , weil das LSG das Paul-Ehrlich-Institut ( Prof. Dr. C. ) nicht zu Einwänden des Klägers ergänzend befragt hat .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG`(ORG)
- `Beschluss vom 14. 11. 2013 ( B 9 V 33/13 B )`(RS)
- `Paul-Ehrlich-Institut`(ORG)
- `C.`(PER)

**Example 50** (doc_id: `58262`) (sent_id: `58262`)


Soweit das Vorbringen des FA dahingehend zu verstehen sein sollte , dass es eine Befassung des FG mit der Frage vermisst , ob aus der Gesamtheit der zwischen dem Kläger und den verschiedenen Gesellschaften der X-Gruppe getroffenen Vereinbarungen ein anderes Ergebnis folgt als aus dem Wortlaut der Einzelregelungen , teilt der Senat diese Bedenken nicht .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `X-Gruppe`(ORG)

**Example 51** (doc_id: `58312`) (sent_id: `58312`)


Am 27. März 2017 begann ohne Beteiligung der Atommächte sowie mehrerer NATO-Staaten , darunter Deutschland , eine UN-Atomwaffenverbotskonferenz als erster Schritt zu einer Nuklearwaffenkonvention .

**False Positives:**

- `NATO` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschland`(LOC)

**Example 52** (doc_id: `58323`) (sent_id: `58323`)


Der Koalitionsvertrag für die 17. Legislaturperiode binde die Forderung nach dem Abzug der Atomwaffen in Büchel zudem an die NATO-Strategie .

**False Positives:**

- `NATO` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Koalitionsvertrag`(NRM)
- `Büchel`(LOC)

**Example 53** (doc_id: `58326`) (sent_id: `58326`)


Zu Recht hätte das FG des Weiteren auch darauf abgestellt , dass das Verständnis der Schlusszahlung als Nutzungsentgelt auch der Interessenlage der Vertragsparteien entspreche .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 54** (doc_id: `58349`) (sent_id: `58349`)


Sie hält den Beschluss des LSG für zutreffend .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 55** (doc_id: `58420`) (sent_id: `58420`)


1. Das angefochtene Urteil verletzt den Anspruch auf rechtliches Gehör ( § 119 Nr. 3 FGO , Art. 103 Abs. 1 des Grundgesetzes - GG - ) , so dass ein Verfahrensfehler nach § 115 Abs. 2 Nr. 3 FGO gegeben ist , da das FG zur Sache mündlich verhandelt und entschieden hat , obwohl die prozessbevollmächtigte Sozietät des Klägers am Tag der mündlichen Verhandlung wegen Erkrankung des sachbearbeitenden Rechtsanwalts eine Verlegung des Termins beantragt hatte .

**False Positives:**

- `FG` — similar text (different position): `§ 119 Nr. 3 FGO`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 119 Nr. 3 FGO`(NRM)
- `Art. 103 Abs. 1 des Grundgesetzes`(NRM)
- `GG`(NRM)
- `§ 115 Abs. 2 Nr. 3 FGO`(NRM)

**Example 56** (doc_id: `58450`) (sent_id: `58450`)


In einer ersten Regelbeurteilung vom 23. April 2013 zum Stichtag 1. April 2013 vergab der seinerzeitige Leiter der Abteilung X des BND ( Herr Dr. A. ) das Gesamturteil 7. Auf Einwendungen des Klägers hob der BND diese dienstliche Beurteilung wegen formeller Fehler auf .

**False Positives:**

- `BND` — similar text (different position): `Abteilung X des BND`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Abteilung X des BND`(ORG)
- `A.`(PER)
- `BND`(ORG)

**Example 57** (doc_id: `58571`) (sent_id: `58571`)


Sie rügt eine Verletzung des rechtlichen Gehörs , weil das LSG den Bescheid über die Feststellung der Gleichwertigkeit ihrer Prüfung als Diätköchin mit der Abschlussprüfung als Köchin nicht zur Kenntnis genommen habe .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 58** (doc_id: `58576`) (sent_id: `58576`)


Die Eingangsperforation 30- 07- 10 auf dem Einspruchsschriftsatz stammt unstreitig von der Dienststelle des DPMA in München und beweist nach der gesetzlichen Beweisregel für öffentliche Urkunden , dass der Einspruch am 30. Juli 2010 beim Patentamt in München eingegangen ist .

**False Positives:**

- `DPMA` — partial — pred is substring of gold: `Dienststelle des DPMA in München`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Dienststelle des DPMA in München`(ORG)
- `München`(LOC)

**Example 59** (doc_id: `58697`) (sent_id: `58697`)


Dem Beschluss des BVerfG in BVerfGE 135 , 1 lag auch kein mit dem Streitfall und dem Senatsurteil in BFHE 237 , 156 , BStBl II 2012 , 577 vergleichbarer Sachverhalt zugrunde .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `Beschluss des BVerfG in BVerfGE 135 , 1`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Beschluss des BVerfG in BVerfGE 135 , 1`(RS)
- `Senatsurteil in BFHE 237 , 156 , BStBl II 2012 , 577`(RS)

**Example 60** (doc_id: `58805`) (sent_id: `58805`)


I. Der Kläger begehrt in der Hauptsache die Feststellung eines Grades der Behinderung ( GdB ) von 80 und die Zuerkennung des Merkzeichens G. Dieses Begehren hat das LSG mit Urteil vom 16. 11. 2017 verneint .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 61** (doc_id: `59000`) (sent_id: `59000`)


Vielmehr bilden Richtwertvorgaben eine einzelne ( quantitative ) Komponente des anzulegenden Beurteilungsmaßstabs , der außerdem durch weitere ( qualitative ) Kriterien wie Eignung , Leistung und Befähigung im Sinne des § 3 Abs. 1 SG in den zehn Einzelmerkmalen Zielerreichung , Eigenständigkeit , Belastbarkeit , Fachkenntnis und praktisches Können , Planung und Organisation , Informations- und Kommunikationsverhalten , Zusammenarbeit , wirtschaftliches Verhalten , Ausbildung und Führungsverhalten ( Nr. 609 Buchst. a ZDv A- 1340/50 ) geprägt wird .

**False Positives:**

- `SG` — partial — pred is substring of gold: `§ 3 Abs. 1 SG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 3 Abs. 1 SG`(NRM)
- `Nr. 609 Buchst. a ZDv A- 1340/50`(REG)

**Example 62** (doc_id: `59204`) (sent_id: `59204`)


Insoweit hat der erkennende Senat die Sache zur erneuten Verhandlung und Entscheidung an das LSG zurückverwiesen .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 63** (doc_id: `59329`) (sent_id: `59329`)


Schließlich habe das LSG die in den Akten vorliegenden Tatsachen nicht korrekt gewürdigt , da es die ihm zuzuerkennenden Merkzeichen weiterhin abgelehnt habe .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 64** (doc_id: `59463`) (sent_id: `59463`)


4. Da das FG von anderen Rechtsgrundsätzen ausgegangen ist , war die Vorentscheidung aufzuheben .

**False Positives:**

- `FG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 65** (doc_id: `59486`) (sent_id: `59486`)


Die Klägerin schreibt dem LSG den Rechtssatz zu , dass beim Vorliegen einer schweren spezifischen Leistungsbehinderung keine konkrete Verweisungstätigkeit benannt werden müsse , wenn noch Tätigkeiten des allgemeinen Arbeitsmarktes wie Transportieren , Reinigen , Bedienen von Maschinen , Kleben , Sortieren , Verpacken , oder Zusammensetzungen von Teilen möglich seien .

**False Positives:**

- `LSG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 66** (doc_id: `60031`) (sent_id: `60031`)


1. Zu Recht hat das FG von einer ( zusätzlichen ) Beiladung des Klägers als Klagebevollmächtigter i. S. des § 48 Abs. 1 Nr. 1 Alternative 2 , Abs. 2 FGO abgesehen .

**False Positives:**

- `FG` — similar text (different position): `§ 48 Abs. 1 Nr. 1 Alternative 2 , Abs. 2 FGO`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 48 Abs. 1 Nr. 1 Alternative 2 , Abs. 2 FGO`(NRM)

</details>

---

## `Specific Court Genitives with Location (Fixed)`

**F1:** 0.040 | **Precision:** 0.395 | **Recall:** 0.021  

**Format:** `regex`  
**Rule ID:** `0ff0f4df`  
**Description:**
Matches court names in genitive form followed by location, ensuring the court type is present.

**Content:**
```
\b(?:Amtsgerichts|Landgerichts|Verwaltungsgerichts|Oberlandesgerichts|Landesarbeitsgerichts|Landessozialgerichts|Sozialgerichts|Finanzgerichts|Arbeitsgerichts|Bundesverwaltungsgerichts|Bundesfinanzhofs|Bundesgerichtshofs|Bundessozialgerichts|Bundesarbeitsgerichts|Bundespatentgerichts|Oberverwaltungsgerichts|Verwaltungsgerichtshofs|Hamburgischen Oberverwaltungsgerichts|Schleswig-Holsteinische Verwaltungsgericht|Schleswig-Holsteinische Oberlandesgericht|Bayerischen Landeszentrale|Bayerischen Landessozialgerichts|Truppendienstgerichts|Anwaltsgerichtshofs)\s+(?:[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)*|[A-Z]\.|[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+-[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+|<\s*[A-Z]{2,3}\s*>\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)*|Frankfurt\s+am\s+Main|Berlin-Brandenburg|Niedersachsen-Bremen|Baden-W\u00fcrttemberg|Nordrhein-Westfalen|Rheinland-Pfalz|Schleswig-Holstein)\b(?!\s+(?:Senat|Nr\.|\.)|\s+(?:Senat|Nr\.))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.395 | 0.021 | 0.040 | 43 | 17 | 26 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 17 | 26 | 746 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53725`) (sent_id: `53725`)


4. Auf die sofortige Beschwerde der Staatsanwaltschaft Düsseldorf hob das Oberlandesgericht Hamm mit angegriffenem Beschluss vom 28. Juli 2015 den Beschluss des Landgerichts Paderborn auf und ordnete die weitere Vollstreckung der Unterbringung des Beschwerdeführers in der Sicherungsverwahrung an .

| Predicted | Gold |
|---|---|
| `Landgerichts Paderborn` | `Landgerichts Paderborn` |

**Missed by this rule (FN):**

- `Staatsanwaltschaft Düsseldorf` (ORG)
- `Oberlandesgericht Hamm` (ORG)

**Example 1** (doc_id: `53793`) (sent_id: `53793`)


Die Revision des Angeklagten gegen das Urteil des Landgerichts Landshut vom 28. Juni 2017 wird verworfen .

| Predicted | Gold |
|---|---|
| `Landgerichts Landshut` | `Landgerichts Landshut` |

**Example 2** (doc_id: `53949`) (sent_id: `53949`)


1. Auf die Revision des Angeklagten gegen das Urteil des Landgerichts Halle vom 4. Mai 2017 wird das vorbenannte Urteil

| Predicted | Gold |
|---|---|
| `Landgerichts Halle` | `Landgerichts Halle` |

**Example 3** (doc_id: `54930`) (sent_id: `54930`)


3. Mit Schriftsatz seines Bevollmächtigten vom 30. Oktober 2014 legte der Beschwerdeführer " gegen den Durchsuchungsbeschluss des Amtsgerichts München vom 2. Mai 2014 sowie die bereits erfolgte Beschlagnahme der Geschäftsunterlagen " Beschwerde ein mit dem Antrag , den genannten Beschluss aufzuheben .

| Predicted | Gold |
|---|---|
| `Amtsgerichts München` | `Amtsgerichts München` |

**Example 4** (doc_id: `55722`) (sent_id: `55722`)


Vielmehr hat sich das LSG die Feststellungen aus dem Strafurteil nach eigener Prüfung zu eigen gemacht und dabei neben den Feststellungen aus dem Strafurteil nicht nur den Inhalt des Erstattungsbescheides aus dem Jahr 2013 zur Honorarrückforderung in Höhe von 216 492,33 Euro berücksichtigt , sondern auch den Inhalt der Strafakten ausgewertet , aus denen hervorgeht , dass sich das Urteil des Amtsgerichts Fürth nicht allein auf das Geständnis des Klägers stützt , sondern auch auf das Ergebnis einer umfangreichen Beweisaufnahme .

| Predicted | Gold |
|---|---|
| `Amtsgerichts Fürth` | `Amtsgerichts Fürth` |

**Example 5** (doc_id: `56530`) (sent_id: `56530`)


Die Beklagte beantragt , die Urteile des Bayerischen Landessozialgerichts vom 16. 12. 2015 und des Sozialgerichts München vom 16. 5. 2014 aufzuheben und die Klage abzuweisen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts München` | `Sozialgerichts München` |

**Missed by this rule (FN):**

- `Bayerischen Landessozialgerichts` (ORG)

**Example 6** (doc_id: `56596`) (sent_id: `56596`)


Hierfür wären die Urteile des Landgerichts Cottbus vom ... und des Brandenburgischen Oberlandesgerichts vom ... zu beachten , wonach der Kläger durch Kündigung vom ... 2005 wirksam aus der A-GbR ausgeschlossen wurde .

| Predicted | Gold |
|---|---|
| `Landgerichts Cottbus` | `Landgerichts Cottbus` |

**Missed by this rule (FN):**

- `Brandenburgischen Oberlandesgerichts` (ORG)
- `A-GbR` (ORG)

**Example 7** (doc_id: `56616`) (sent_id: `56616`)


Die Beklagte beantragt , das Urteil des Bayerischen Landessozialgerichts vom 21. Oktober 2015 aufzuheben und das Urteil des Sozialgerichts Nürnberg vom 25. Juni 2013 bezüglich der Beigeladenen zu 1. und 3. aufzuheben und die Klage abzuweisen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts Nürnberg` | `Sozialgerichts Nürnberg` |

**Missed by this rule (FN):**

- `Bayerischen Landessozialgerichts` (ORG)

**Example 8** (doc_id: `56951`) (sent_id: `56951`)


Auf die Revision des Angeklagten wird das Urteil des Landgerichts Ravensburg vom 1. August 2017 , soweit es ihn betrifft , mit den Feststellungen aufgehoben .

| Predicted | Gold |
|---|---|
| `Landgerichts Ravensburg` | `Landgerichts Ravensburg` |

**Example 9** (doc_id: `57130`) (sent_id: `57130`)


1. Auf die Revision des Angeklagten gegen das Urteil des Landgerichts Aachen vom 7. Juli 2016 wird

| Predicted | Gold |
|---|---|
| `Landgerichts Aachen` | `Landgerichts Aachen` |

**Example 10** (doc_id: `57570`) (sent_id: `57570`)


Die Klägerin beantragt , den Beschluss des Thüringer Landessozialgerichts vom 21. Juli 2016 und das Urteil des Sozialgerichts Meiningen vom 7. Januar 2015 aufzuheben sowie den Bescheid der Beklagten vom 15. April 2013 in der Gestalt des Widerspruchsbescheids vom 17. Mai 2013 abzuändern und die Beklagte zu verurteilen , ihr für die Zeit vom 1. Januar bis 28. März 2012 höheres Insolvenzgeld zu zahlen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts Meiningen` | `Sozialgerichts Meiningen` |

**Missed by this rule (FN):**

- `Thüringer Landessozialgerichts` (ORG)

**Example 11** (doc_id: `58013`) (sent_id: `58013`)


das Urteil des Bayerischen Landessozialgerichts vom 3. Juni 2016 und den Gerichtsbescheid des Sozialgerichts Nürnberg vom 30. August 2013 sowie den Bescheid der Beklagten vom 19. März 2012 und den Widerspruchsbescheid vom 18. Juni 2012 aufzuheben und die Beklagte zu verurteilen , unter Rücknahme des Bescheides vom 4. Februar 2005 und des Widerspruchsbescheides vom 22. April 2005 die Zeit vom 1. September 1973 bis 30. Juni 1990 als Zeit der Zugehörigkeit zum Zusatzversorgungssystem der technischen Intelligenz und die hierin erzielten Arbeitsentgelte festzustellen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts Nürnberg` | `Sozialgerichts Nürnberg` |

**Missed by this rule (FN):**

- `Bayerischen Landessozialgerichts` (ORG)

**Example 12** (doc_id: `58301`) (sent_id: `58301`)


Der Beklagte beantragt , das Urteil des Sächsischen Landessozialgerichts vom 9. Februar 2017 aufzuheben und die Berufungen der Kläger gegen das Urteil des Sozialgerichts Dresden vom 10. Februar 2014 zurückzuweisen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts Dresden` | `Sozialgerichts Dresden` |

**Missed by this rule (FN):**

- `Sächsischen Landessozialgerichts` (ORG)

**Example 13** (doc_id: `58405`) (sent_id: `58405`)


3. Mit Schriftsatz vom 8. März 2018 beantragt die Beschwerdeführerin durch ihren Bevollmächtigten , " die Vollstreckbarkeit " der Beschlüsse des Landgerichts Potsdam vom 11. März 2014 und vom " 20. Juli 2017 " ( gemeint wohl 17. Juli 2017 ) vorläufig auszusetzen .

| Predicted | Gold |
|---|---|
| `Landgerichts Potsdam` | `Landgerichts Potsdam` |

**Example 14** (doc_id: `59568`) (sent_id: `59568`)


Die Revision des Angeklagten gegen das Urteil des Landgerichts Essen vom 10. April 2017 wird als unbegründet verworfen , da die Nachprüfung des Urteils auf Grund der Revisionsrechtfertigung keinen Rechtsfehler zum Nachteil des Angeklagten ergeben hat ( § 349 Abs. 2 StPO ) .

| Predicted | Gold |
|---|---|
| `Landgerichts Essen` | `Landgerichts Essen` |

**Missed by this rule (FN):**

- `§ 349 Abs. 2 StPO` (NRM)

**Example 15** (doc_id: `59658`) (sent_id: `59658`)


A. I. 1. Der Beschwerdeführer ist auf Grundlage des Urteils des Landgerichts Lübeck vom 7. Oktober 2014 gemäß § 63 StGB wegen Mordes in einem psychiatrischen Krankenhaus der AMEOS Krankenhausgesellschaft Holstein mbH untergebracht , nachdem er im schuldunfähigen Zustand auf Grund einer wahnhaften Störung im Januar 2014 seine vierjährige Tochter und seinen sechs Jahre alten Sohn getötet hatte .

| Predicted | Gold |
|---|---|
| `Landgerichts Lübeck` | `Landgerichts Lübeck` |

**Missed by this rule (FN):**

- `§ 63 StGB` (NRM)
- `AMEOS Krankenhausgesellschaft Holstein mbH` (ORG)

**Example 16** (doc_id: `59742`) (sent_id: `59742`)


1. Dem Angeklagten A. wird auf seinen Antrag Wiedereinsetzung in den vorigen Stand wegen Versäumung der Frist zur Begründung der Revision gegen das Urteil des Landgerichts Göttingen vom 30. Mai 2017 gewährt .

| Predicted | Gold |
|---|---|
| `Landgerichts Göttingen` | `Landgerichts Göttingen` |

**Missed by this rule (FN):**

- `A.` (PER)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `54196`) (sent_id: `54196`)


Die Revision des Klägers gegen das Urteil des Landesarbeitsgerichts Hamburg vom 28. September 2016 - 1 Sa 18/16 - wird zurückgewiesen .

**False Positives:**

- `Landesarbeitsgerichts Hamburg` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Hamburg vom 28. September 2016 - 1 Sa 18/16 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Hamburg vom 28. September 2016 - 1 Sa 18/16 -`(RS)

**Example 1** (doc_id: `54385`) (sent_id: `54385`)


die Urteile des Landessozialgerichts Sachsen-Anhalt vom 9. März 2017 und des Sozialgerichts Dessau-Roßlau vom 2. Dezember 2013 sowie den Bescheid des Beklagten vom 16. Februar 2010 in der Gestalt des Widerspruchsbescheids vom 31. Mai 2010 aufzuheben .

**False Positives:**

- `Landessozialgerichts Sachsen` — partial — pred is substring of gold: `Landessozialgerichts Sachsen-Anhalt`
- `Sozialgerichts Dessau` — partial — pred is substring of gold: `Sozialgerichts Dessau-Roßlau`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Landessozialgerichts Sachsen-Anhalt`(ORG)
- `Sozialgerichts Dessau-Roßlau`(ORG)

**Example 2** (doc_id: `54438`) (sent_id: `54438`)


Der Haftbefehl des Amtsgerichts Neu-Ulm vom 31. Juli 2017 könne unter Berücksichtigung der Bedeutung und Tragweite des Grundrechts auf Freiheit der Person des Beschwerdeführers keinen Bestand haben .

**False Positives:**

- `Amtsgerichts Neu` — partial — pred is substring of gold: `Amtsgerichts Neu-Ulm`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Amtsgerichts Neu-Ulm`(ORG)

**Example 3** (doc_id: `54663`) (sent_id: `54663`)


Die Revision der Klägerin gegen das Urteil des Landesarbeitsgerichts Baden-Württemberg vom 23. Juni 2015 - 9 Sa 79/14 - wird auf ihre Kosten zurückgewiesen .

**False Positives:**

- `Landesarbeitsgerichts Baden` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Baden-Württemberg vom 23. Juni 2015 - 9 Sa 79/14 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Baden-Württemberg vom 23. Juni 2015 - 9 Sa 79/14 -`(RS)

**Example 4** (doc_id: `55158`) (sent_id: `55158`)


1. Die Revision des Beklagten gegen das Urteil des Landesarbeitsgerichts Rheinland-Pfalz vom 7. Juli 2016 - 6 Sa 23/16 - wird zurückgewiesen .

**False Positives:**

- `Landesarbeitsgerichts Rheinland` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Rheinland-Pfalz vom 7. Juli 2016 - 6 Sa 23/16 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Rheinland-Pfalz vom 7. Juli 2016 - 6 Sa 23/16 -`(RS)

**Example 5** (doc_id: `55511`) (sent_id: `55511`)


Die Beschwerde der Antragstellerin gegen den Beschluss des Finanzgerichts Hamburg vom 15. August 2016 1 V 41/16 wird als unbegründet zurückgewiesen .

**False Positives:**

- `Finanzgerichts Hamburg` — partial — pred is substring of gold: `Beschluss des Finanzgerichts Hamburg vom 15. August 2016 1 V 41/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Beschluss des Finanzgerichts Hamburg vom 15. August 2016 1 V 41/16`(RS)

**Example 6** (doc_id: `55622`) (sent_id: `55622`)


Die Revision des Klägers gegen das Urteil des Landesarbeitsgerichts Köln vom 8. Dezember 2016 - 8 Sa 540/16 - wird zurückgewiesen .

**False Positives:**

- `Landesarbeitsgerichts Köln` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Köln vom 8. Dezember 2016 - 8 Sa 540/16 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Köln vom 8. Dezember 2016 - 8 Sa 540/16 -`(RS)

**Example 7** (doc_id: `55659`) (sent_id: `55659`)


Die Beschwerde des Klägers wegen Nichtzulassung der Revision gegen das Urteil des Finanzgerichts Münster vom 12. September 2017 15 K 3562/14 U wird als unbegründet zurückgewiesen .

**False Positives:**

- `Finanzgerichts Münster` — partial — pred is substring of gold: `Urteil des Finanzgerichts Münster vom 12. September 2017 15 K 3562/14 U`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Finanzgerichts Münster vom 12. September 2017 15 K 3562/14 U`(RS)

**Example 8** (doc_id: `56170`) (sent_id: `56170`)


2. Die Berufung des Beklagten gegen das Urteil des Arbeitsgerichts Oberhausen vom 15. Dezember 2016 - 4 Ca 1318/16 - wird zurückgewiesen .

**False Positives:**

- `Arbeitsgerichts Oberhausen` — partial — pred is substring of gold: `Urteil des Arbeitsgerichts Oberhausen vom 15. Dezember 2016 - 4 Ca 1318/16 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Arbeitsgerichts Oberhausen vom 15. Dezember 2016 - 4 Ca 1318/16 -`(RS)

**Example 9** (doc_id: `56230`) (sent_id: `56230`)


Die Revision der Klägerin gegen das Urteil des Landesarbeitsgerichts Baden-Württemberg vom 23. Juni 2015 - 22 Sa 64/14 - wird auf ihre Kosten zurückgewiesen .

**False Positives:**

- `Landesarbeitsgerichts Baden` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Baden-Württemberg vom 23. Juni 2015 - 22 Sa 64/14 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Baden-Württemberg vom 23. Juni 2015 - 22 Sa 64/14 -`(RS)

**Example 10** (doc_id: `56331`) (sent_id: `56331`)


Auf die Berufung der Beklagten wird - unter Zurückweisung der Anschlussberufung des Klägers - das Urteil des Arbeitsgerichts Bonn vom 11. November 2015 - 4 Ca 1615/15 - abgeändert und die Klage abgewiesen .

**False Positives:**

- `Arbeitsgerichts Bonn` — partial — pred is substring of gold: `Urteil des Arbeitsgerichts Bonn vom 11. November 2015 - 4 Ca 1615/15 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Arbeitsgerichts Bonn vom 11. November 2015 - 4 Ca 1615/15 -`(RS)

**Example 11** (doc_id: `56355`) (sent_id: `56355`)


Auf die Revision der Klägerin wird das Urteil des Landesarbeitsgerichts Hamburg vom 5. November 2015 - 1 Sa 11/15 - aufgehoben .

**False Positives:**

- `Landesarbeitsgerichts Hamburg` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Hamburg vom 5. November 2015 - 1 Sa 11/15 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Hamburg vom 5. November 2015 - 1 Sa 11/15 -`(RS)

**Example 12** (doc_id: `56480`) (sent_id: `56480`)


Die Revision des Klägers gegen das Urteil des Landesarbeitsgerichts Rheinland-Pfalz vom 17. Februar 2016 - 4 Sa 202/15 - wird zurückgewiesen .

**False Positives:**

- `Landesarbeitsgerichts Rheinland` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Rheinland-Pfalz vom 17. Februar 2016 - 4 Sa 202/15 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Rheinland-Pfalz vom 17. Februar 2016 - 4 Sa 202/15 -`(RS)

**Example 13** (doc_id: `56544`) (sent_id: `56544`)


1. Auf die Revision der Klägerin wird das Urteil des Landesarbeitsgerichts Düsseldorf vom 14. Juni 2017 - 7 Sa 81/17 - aufgehoben .

**False Positives:**

- `Landesarbeitsgerichts Düsseldorf` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Düsseldorf vom 14. Juni 2017 - 7 Sa 81/17 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Düsseldorf vom 14. Juni 2017 - 7 Sa 81/17 -`(RS)

**Example 14** (doc_id: `57953`) (sent_id: `57953`)


2. Die Berufung des Klägers gegen das Urteil des Arbeitsgerichts Dortmund vom 27. März 2014 - 6 Ca 3695/11 - wird zurückgewiesen .

**False Positives:**

- `Arbeitsgerichts Dortmund` — partial — pred is substring of gold: `Urteil des Arbeitsgerichts Dortmund vom 27. März 2014 - 6 Ca 3695/11 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Arbeitsgerichts Dortmund vom 27. März 2014 - 6 Ca 3695/11 -`(RS)

**Example 15** (doc_id: `58297`) (sent_id: `58297`)


In Bezug auf den gerügten Haftbefehl des Amtsgerichts Neu-Ulm vom 31. Juli 2017 sei die Verfassungsbeschwerde wegen des Grundsatzes der Subsidiarität der Verfassungsbeschwerde hingegen unzulässig , da eine abschließende Sachprüfung durch das Oberlandesgericht München noch nicht stattgefunden habe .

**False Positives:**

- `Amtsgerichts Neu` — partial — pred is substring of gold: `Amtsgerichts Neu-Ulm`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Amtsgerichts Neu-Ulm`(ORG)
- `Oberlandesgericht München`(ORG)

**Example 16** (doc_id: `58399`) (sent_id: `58399`)


Die Revision der Klägerin gegen das Urteil des Finanzgerichts Hamburg vom 21. April 2017 4 K 186/16 wird als unbegründet zurückgewiesen .

**False Positives:**

- `Finanzgerichts Hamburg` — partial — pred is substring of gold: `Urteil des Finanzgerichts Hamburg vom 21. April 2017 4 K 186/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Finanzgerichts Hamburg vom 21. April 2017 4 K 186/16`(RS)

**Example 17** (doc_id: `58546`) (sent_id: `58546`)


Auf die Revision des Beklagten wird das Urteil des Finanzgerichts München vom 29. März 2017 3 K 2565/16 aufgehoben .

**False Positives:**

- `Finanzgerichts München` — partial — pred is substring of gold: `Urteil des Finanzgerichts München vom 29. März 2017 3 K 2565/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Finanzgerichts München vom 29. März 2017 3 K 2565/16`(RS)

**Example 18** (doc_id: `58604`) (sent_id: `58604`)


Die Rechtsbeschwerde des Betriebsrats gegen den Beschluss des Landesarbeitsgerichts Sachsen-Anhalt vom 22. März 2016 - 6 TaBV 39/14 - wird zurückgewiesen .

**False Positives:**

- `Landesarbeitsgerichts Sachsen` — partial — pred is substring of gold: `Beschluss des Landesarbeitsgerichts Sachsen-Anhalt vom 22. März 2016 - 6 TaBV 39/14 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Beschluss des Landesarbeitsgerichts Sachsen-Anhalt vom 22. März 2016 - 6 TaBV 39/14 -`(RS)

**Example 19** (doc_id: `58915`) (sent_id: `58915`)


Die Beschwerde gegen die Nichtzulassung der Revision in dem Urteil des 7. Zivilsenats des Oberlandesgerichts Frankfurt am Main vom 8. Juni 2017 wird auf Kosten des Klägers als unzulässig verworfen .

**False Positives:**

- `Oberlandesgerichts Frankfurt` — partial — pred is substring of gold: `Urteil des 7. Zivilsenats des Oberlandesgerichts Frankfurt am Main`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des 7. Zivilsenats des Oberlandesgerichts Frankfurt am Main`(RS)

**Example 20** (doc_id: `58942`) (sent_id: `58942`)


Eine - über die behauptete Verletzung von Art. 19 Abs. 4 Satz 1 GG hinausgehende - verfassungsgerichtliche Sachprüfung widerspräche dem Grundsatz der Subsidiarität der Verfassungsbeschwerde , weil eine abschließende fachgerichtliche Prüfung des angegriffenen Haftbefehls des Amtsgerichts Neu-Ulm vom 31. Juli 2017 bislang - entgegen den Vorgaben von Art. 19 Abs. 4 Satz 1 GG - nicht erfolgt ist .

**False Positives:**

- `Amtsgerichts Neu` — partial — pred is substring of gold: `Amtsgerichts Neu-Ulm`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 19 Abs. 4 Satz 1 GG`(NRM)
- `Amtsgerichts Neu-Ulm`(ORG)
- `Art. 19 Abs. 4 Satz 1 GG`(NRM)

**Example 21** (doc_id: `59195`) (sent_id: `59195`)


1. Auf die Revision des beklagten Landes wird das Urteil des Landesarbeitsgerichts Hamm vom 4. Juli 2016 - 11 Sa 1330/14 - aufgehoben .

**False Positives:**

- `Landesarbeitsgerichts Hamm` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Hamm vom 4. Juli 2016 - 11 Sa 1330/14 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Hamm vom 4. Juli 2016 - 11 Sa 1330/14 -`(RS)

**Example 22** (doc_id: `59356`) (sent_id: `59356`)


Über die sofortige Beschwerde der Nebenklägerin K. T. gegen die im Urteil des Landgerichts Frankfurt am Main vom 15. August 2017 getroffene Kostenentscheidung hat das Oberlandesgericht Frankfurt am Main zu entscheiden .

**False Positives:**

- `Landgerichts Frankfurt` — partial — pred is substring of gold: `Landgerichts Frankfurt am Main`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K. T.`(PER)
- `Landgerichts Frankfurt am Main`(ORG)
- `Oberlandesgericht Frankfurt am Main`(ORG)

**Example 23** (doc_id: `59434`) (sent_id: `59434`)


1. Das Urteil des Pfälzischen Oberlandesgerichts Zweibrücken vom 29. Januar 2015 - 4 U 81/14 - verletzt die Beschwerdeführerin in ihrem Grundrecht aus Artikel 5 Absatz 1 Satz 2 des Grundgesetzes und wird aufgehoben .

**False Positives:**

- `Oberlandesgerichts Zweibrücken` — partial — pred is substring of gold: `Urteil des Pfälzischen Oberlandesgerichts Zweibrücken vom 29. Januar 2015 - 4 U 81/14 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Pfälzischen Oberlandesgerichts Zweibrücken vom 29. Januar 2015 - 4 U 81/14 -`(RS)
- `Artikel 5 Absatz 1 Satz 2 des Grundgesetzes`(NRM)

**Example 24** (doc_id: `59490`) (sent_id: `59490`)


2. a ) Das Urteil des Landesarbeitsgerichts Nürnberg vom 30. Januar 2014 - 5 Sa 1/13 - und das Endurteil des Arbeitsgerichts Bamberg vom 10. Oktober 2012 - 2 Ca 1097/11 - verletzen den Beschwerdeführer in seinem Grundrecht aus Artikel 2 Absatz 1 Satz 1 in Verbindung mit Artikel 20 Absatz 3 des Grundgesetzes .

**False Positives:**

- `Arbeitsgerichts Bamberg` — partial — pred is substring of gold: `Endurteil des Arbeitsgerichts Bamberg vom 10. Oktober 2012 - 2 Ca 1097/11 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Nürnberg vom 30. Januar 2014 - 5 Sa 1/13 -`(RS)
- `Endurteil des Arbeitsgerichts Bamberg vom 10. Oktober 2012 - 2 Ca 1097/11 -`(RS)
- `Artikel 2 Absatz 1 Satz 1 in Verbindung mit Artikel 20 Absatz 3 des Grundgesetzes`(NRM)

</details>

---

## `Bundesgerichtshof Genitive`

**F1:** 0.036 | **Precision:** 0.833 | **Recall:** 0.019  

**Format:** `regex`  
**Rule ID:** `8529045c`  
**Description:**
Matches 'Bundesgerichtshof' and its genitive form 'Bundesgerichtshofs' and 'Bundesgerichtshofes'.

**Content:**
```
\b(Bundesgerichtshof|Bundesgerichtshofs|Bundesgerichtshofes)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.833 | 0.019 | 0.036 | 18 | 15 | 3 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 15 | 3 | 792 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53387`) (sent_id: `53387`)


7. Die hiergegen eingelegte Nichtzulassungsbeschwerde sowie eine Anhörungsrüge der Beschwerdeführerin wies der Bundesgerichtshof zurück .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Example 1** (doc_id: `53957`) (sent_id: `53957`)


Über keine der anhängigen Rechtsbeschwerden hat der Bundesgerichtshof bislang entschieden .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Example 2** (doc_id: `56707`) (sent_id: `56707`)


In einem Schadensersatzprozess gegen die Beigeladene , mit dem er einen Ausgleich auch für die Kürzung der Regelaltersrente infolge des weiter verminderten Zugangsfaktors geltend machte , unterlag der Kläger auch in letzter Instanz vor dem Bundesgerichtshof ( BGH ) .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Missed by this rule (FN):**

- `BGH` (ORG)

**Example 3** (doc_id: `56787`) (sent_id: `56787`)


( 2 ) Einer Auslegung des Schutzgegenstands eines Designs auf Grundlage der Schnittmenge der allen Darstellungen gemeinsamen Merkmale könnten allerdings die nach der vorgenannten Entscheidung des Bundesgerichtshofes einem abgeleiteten Teilschutz entgegenstehenden und nunmehr für das DesignG geltenden Gesichtspunkte der Klarheit des Registers und der damit verbundenen Rechtssicherheit entgegenstehen .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshofes` | `Bundesgerichtshofes` |

**Missed by this rule (FN):**

- `DesignG` (NRM)

**Example 4** (doc_id: `56848`) (sent_id: `56848`)


Zwischen den sich gegenüberstehenden Zeichen sei jedenfalls eine hohe klangliche Ähnlichkeit gegeben , denn nach den vom Bundesgerichtshof entwickelten Grundsätzen zur Prägung von Wort- / Bildzeichen seien vorliegend in klanglicher Hinsicht jedenfalls die prägenden Zeichenbestandteile " GEA " und " KEA " zu vergleichen .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Example 5** (doc_id: `57029`) (sent_id: `57029`)


Da allein das Fehlen jeglicher Unterscheidungskraft ein Eintragungshindernis begründet , ist nach der Rechtsprechung des Bundesgerichtshofes ein großzügiger Maßstab anzulegen , so dass jede auch noch so geringe Unterscheidungskraft genügt , um das Schutzhindernis zu überwinden ( vgl. BGH GRUR 2012 , 1143 , Rn. 7 - Starsat ; GRUR 2012 , 1044 , 1045 , Rn. 9 - Neuschwanstein ; GRUR 2012 , 270 , Rn. 8 - Link economy ) .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshofes` | `Bundesgerichtshofes` |

**Missed by this rule (FN):**

- `BGH GRUR 2012 , 1143 , Rn. 7 - Starsat` (RS)
- `GRUR 2012 , 1044 , 1045 , Rn. 9 - Neuschwanstein` (RS)
- `GRUR 2012 , 270 , Rn. 8 - Link economy` (RS)

**Example 6** (doc_id: `57625`) (sent_id: `57625`)


8. Mit ihrer Verfassungsbeschwerde wendet sich die Beschwerdeführerin gegen die jüngste Entscheidung des Oberlandesgerichts und die beiden darauffolgenden Entscheidungen des Bundesgerichtshofs .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshofs` | `Bundesgerichtshofs` |

**Example 7** (doc_id: `57715`) (sent_id: `57715`)


Hierin durfte der Bundesgerichtshof einen sachlichen Grund sehen , der das Stadionverbot zu tragen vermag .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Example 8** (doc_id: `58279`) (sent_id: `58279`)


4. Daraufhin bestätigte der Bundesgerichtshof mit Urteil vom 21. April 2016 das Urteil des Oberlandesgerichts ( BGHZ 210 , 77 ) .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Missed by this rule (FN):**

- `BGHZ 210 , 77` (RS)

**Example 9** (doc_id: `58414`) (sent_id: `58414`)


5. Mit - nicht angegriffenem - Urteil vom 18. November 2014 hat der Bundesgerichtshof diese Entscheidung aufgehoben und den Rechtsstreit an das Oberlandesgericht zurückverwiesen .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Example 10** (doc_id: `58703`) (sent_id: `58703`)


Der Bundesgerichtshof enthebt die Veranstalter , wie er ausdrücklich ausführt , nicht von einer Plausibilitätskontrolle , um Fälle auszuschließen , in denen ein Verfahren offensichtlich willkürlich oder aufgrund falscher Tatsachenannahmen eingeleitet wurde .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Example 11** (doc_id: `59207`) (sent_id: `59207`)


Nach der Rechtsprechung des Bundesgerichtshofs wird bezogen auf die subjektive Tatseite in § 266a StGB wie folgt differenziert : Der Vorsatz muss sich auf die Eigenschaft als Arbeitgeber und Arbeitnehmer - dabei allerdings nur auf die statusbegründenden tatsächlichen Voraussetzungen , nicht auf die rechtliche Einordnung als solche und die eigene Verpflichtung zur Beitragsabführung - und alle darüber hinausreichenden , die sozialversicherungsrechtlichen Pflichten begründenden tatsächlichen Umstände erstrecken .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshofs` | `Bundesgerichtshofs` |

**Missed by this rule (FN):**

- `§ 266a StGB` (NRM)

**Example 12** (doc_id: `59253`) (sent_id: `59253`)


Die Annahme des Bundesgerichtshofs , wonach die Dienstaufsicht berechtigt sei , einem Richter ein in Zahlen gemessenes unzureichendes Erledigungspensum vorzuhalten , verstoße gegen Art. 97 Abs. 1 GG .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshofs` | `Bundesgerichtshofs` |

**Missed by this rule (FN):**

- `Art. 97 Abs. 1 GG` (NRM)

**Example 13** (doc_id: `59621`) (sent_id: `59621`)


Das Vorliegen einer Verwechslungsgefahr für das Publikum ist nach ständiger Rechtsprechung sowohl des Europäischen Gerichtshofes als auch des Bundesgerichtshofes unter Berücksichtigung aller relevanten Umstände des Einzelfalls zu beurteilen ( vgl. hierzu z.B. EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER ; GRUR 2010 , 1098 Rn. 44 – Calvin Klein / HABM ; BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY ; GRUR 2012 , 1040 Rn. 25 – pjur / pure ; GRUR 2013 , 833 Rn. 30 – Culinaria / Villa Culinaria ) .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshofes` | `Bundesgerichtshofes` |

**Missed by this rule (FN):**

- `Europäischen Gerichtshofes` (ORG)
- `EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER` (RS)
- `GRUR 2010 , 1098 Rn. 44 – Calvin Klein / HABM` (RS)
- `BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY` (RS)
- `GRUR 2012 , 1040 Rn. 25 – pjur / pure` (RS)
- `GRUR 2013 , 833 Rn. 30 – Culinaria / Villa Culinaria` (RS)

**Example 14** (doc_id: `59703`) (sent_id: `59703`)


9. Die Antragstellerin verweist zur Begründung ihrer Beschwerde zudem auf die Entscheidung „ Dipeptidyl-Peptidase-Inhibitoren “ , in welcher der deutsche Bundesgerichtshof festgestellt hat , dass im Hinblick auf das berechtigte Interesse , eine Erfindung in vollem Umfang zu schützen , die Umschreibung einer Gruppe von Stoffen durch eine funktionelle Definition patentrechtlich grundsätzlich selbst dann zulässig sein kann , wenn eine solche Fassung des Patentanspruchs auch die Verwendung noch unbekannter Möglichkeiten umfasse , die möglicherweise erst zukünftig bereitgestellt oder erfunden werden müssten ( BGH , GRUR 2013 , 1210 , Rnd. 19 ff. ) .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Missed by this rule (FN):**

- `BGH , GRUR 2013 , 1210 , Rnd. 19 ff.` (RS)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `54693`) (sent_id: `54693`)


c ) Die Klägerin wird beiläufig darauf hingewiesen , dass ein eventueller Innenausgleich zwischen N als Organträger und ihr als Organgesellschaft nach bürgerlichem Recht entsprechend § 426 des Bürgerlichen Gesetzbuchs vorgenommen wird und derjenige Beteiligte am Organkreis , aus dessen Umsätzen die an das FA zu zahlende Umsatzsteuer herrührt , im Innenverhältnis der Organschaft die Steuerlast zu tragen hat ( vgl. Urteile des Bundesgerichtshofs vom 29. Januar 2013 II ZR 91/11 , Deutsches Steuerrecht - DStR - 2013 , 478 , Rz 10 f. ; vom 19. Januar 2012 IX ZR 2/11 , DStR 2012 , 527 , Rz 28 und 36 ; BFH-Urteil vom 23. September 2009 VII R 43/08 , BFHE 226 , 391 , BStBl II 2010 , 215 , Rz 30 ) .

**False Positives:**

- `Bundesgerichtshofs` — similar text (different position): `N`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `N`(ORG)
- `§ 426 des Bürgerlichen Gesetzbuchs`(NRM)
- `Urteile des Bundesgerichtshofs vom 29. Januar 2013 II ZR 91/11 , Deutsches Steuerrecht - DStR - 2013 , 478 , Rz 10 f.`(RS)
- `vom 19. Januar 2012 IX ZR 2/11 , DStR 2012 , 527 , Rz 28 und 36`(RS)
- `BFH-Urteil vom 23. September 2009 VII R 43/08 , BFHE 226 , 391 , BStBl II 2010 , 215 , Rz 30`(RS)

**Example 1** (doc_id: `54748`) (sent_id: `54748`)


Dem entsprechend hat der für das Bankrecht allein zuständige XI. Zivilsenat des Bundesgerichtshofs mit Urteil vom 12. 7. 2016 - XI ZR 564/15 - ( NJW 2016 , 3512 , Tz. 34 , 40 ) entschieden , dass das Widerrufsrecht nach § 495 Abs. 1 BGB a. F. ungeachtet des Fehlers der erteilten Widerrufsbelehrung verwirkt werden kann .

**False Positives:**

- `Bundesgerichtshofs` — partial — pred is substring of gold: `XI. Zivilsenat des Bundesgerichtshofs`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `XI. Zivilsenat des Bundesgerichtshofs`(ORG)
- `Urteil vom 12. 7. 2016 - XI ZR 564/15 - ( NJW 2016 , 3512 , Tz. 34 , 40 )`(RS)
- `§ 495 Abs. 1 BGB a. F.`(NRM)

**Example 2** (doc_id: `57181`) (sent_id: `57181`)


Eine vergleichbar verschärfte Verpflichtung zur konkreten , eingehenderen Beschreibung der tatsächlichen Umstände des Beweisantrags ist als gesteigerte Substantiierungslast dann anzunehmen , wenn eine Person aufgrund Zugriffs zu den hierfür benötigten Informationen konkretere Informationen geben kann ( vgl. Urteil des Bundesgerichtshofs vom 25. April 2002 IX ZR 313/99 , BGHZ 150 , 353 , unter III. 2. – zur Substantiierungslast des Insolvenzverwalters für tatsächliche Vorgänge aus der Sphäre des Insolvenzschuldners ) .

**False Positives:**

- `Bundesgerichtshofs` — partial — pred is substring of gold: `Urteil des Bundesgerichtshofs vom 25. April 2002 IX ZR 313/99 , BGHZ 150 , 353 , unter III. 2.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Bundesgerichtshofs vom 25. April 2002 IX ZR 313/99 , BGHZ 150 , 353 , unter III. 2.`(RS)

</details>

---

## `Patent Office Full Name`

**F1:** 0.032 | **Precision:** 0.929 | **Recall:** 0.016  

**Format:** `regex`  
**Rule ID:** `f72ad7ef`  
**Description:**
Matches 'Deutschen Patent- und Markenamt' and 'Deutsche Patent- und Markenamt' in various cases.

**Content:**
```
\b(?:Deutschen\s+Patent-\s+und\s+Markenamt|Deutsche\s+Patent-\s+und\s+Markenamt)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.929 | 0.016 | 0.032 | 14 | 13 | 1 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 13 | 1 | 789 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53404`) (sent_id: `53404`)


Hiergegen richtet sich die mit Schriftsatz vom 11. Februar 2014 beim Deutschen Patent- und Markenamt eingelegte Beschwerde der Anmelderin , die ihr Patentbegehren mit den mit Schreiben vom 25. Februar 2011 eingereichten Unterlagen weiterverfolgt .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Example 1** (doc_id: `53791`) (sent_id: `53791`)


Zwischen den Vergleichszeichen besteht hinsichtlich der angegriffenen Waren Verwechslungsgefahr gemäß § 9 Abs. 1 Nr. 2 MarkenG , so dass der Widerspruch aus der Marke 30 2011 047 766 vom Deutschen Patent- und Markenamt zu Unrecht zurückgewiesen wurde .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `§ 9 Abs. 1 Nr. 2 MarkenG` (NRM)

**Example 2** (doc_id: `55285`) (sent_id: `55285`)


I. Das am 26. August 2013 angemeldete Zeichen Fireslim ist am 10. Januar 2014 unter der Nr. 30 2013 048 208 in das beim Deutschen Patent- und Markenamt geführte Markenregister für die nachfolgenden Waren und Dienstleistungen der Klassen 9 , 35 und 38 eingetragen worden :

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `Fireslim` (ORG)

**Example 3** (doc_id: `55981`) (sent_id: `55981`)


I. Die am 6. Mai 2010 angemeldete Wort- / Bildmarke 30 2010 028 176 ist am 20. Dezember 2010 für die nachfolgend genannten Waren in das beim Deutschen Patent- und Markenamt ( DPMA ) geführte Markenregister eingetragen worden :

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `DPMA` (ORG)

**Example 4** (doc_id: `56197`) (sent_id: `56197`)


I. Die am 7. November 2013 angemeldete Wortfolge Rap Shot ist am 23. Januar 2014 unter der Nummer 30 2013 058 941 als Wortmarke für die nachfolgend genannten Waren und Dienstleistungen in das beim Deutschen Patent- und Markenamt geführte Markenregister eingetragen worden :

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `Rap Shot` (ORG)

**Example 5** (doc_id: `57795`) (sent_id: `57795`)


I. Auf die am 26. Januar 2012 beim Deutschen Patent- und Markenamt eingereichte Patentanmeldung ist die Erteilung des Patents 10 2012 201 128 mit der Bezeichnung „ Verfahren , Steuergerät und Speichermedium zur Steuerung einer Harnstoffinjektion bei niedrigen Abgastemperaturen unter Berücksichtigung des Harnstoffgehalts “ am 17. Januar 2013 veröffentlicht worden .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Example 6** (doc_id: `57844`) (sent_id: `57844`)


eingetragenen Wort- / Bildmarke 30 2010 022 988 ( Anmeldetag : 18. Mai 2010 ; Tag der Eintragung im beim Deutschen Patent- und Markenamt geführten Markenregister : 30. Juni 2010 ) ist aus der für die Waren

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Example 7** (doc_id: `57851`) (sent_id: `57851`)


Der Antragsteller beantragte am 5. November 2012 beim Deutschen Patent- und Markenamt ( DPMA ) die Eintragung eines Geschmacksmusters als Sammelanmeldung von 16 Mustern für Erzeugnisse der Klasse 19 - 07 „ Lehrmittel “ .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `DPMA` (ORG)

**Example 8** (doc_id: `57902`) (sent_id: `57902`)


Im Verfahren vor dem Deutschen Patent- und Markenamt ( DPMA ) sieht das Patentgesetz eine Zurückweisung verspäteten Vorbringens nicht vor .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `DPMA` (ORG)
- `Patentgesetz` (NRM)

**Example 9** (doc_id: `58140`) (sent_id: `58140`)


in das beim Deutschen Patent- und Markenamt geführte Register eingetragen worden .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Example 10** (doc_id: `59024`) (sent_id: `59024`)


Am 20. Mai 2016 übermittelte die Anmelderin dem Deutschen Patent- und Markenamt ( DPMA ) Unterlagen für die Einleitung der deutschen nationalen Phase der Anmeldung mit einem gegenüber der ursprünglichen internationalen Anmeldung geänderten Anspruchssatz mit 13 Patentansprüchen .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `DPMA` (ORG)

**Example 11** (doc_id: `59025`) (sent_id: `59025`)


a. Der Inhaber der angegriffenen Marke hat mit Schriftsatz vom 13. September 2012 , welcher per Fax am 13. September 2012 und im Original am 15. September 2012 beim Deutschen Patent- und Markenamt eingegangen ist , die Benutzung der ( Unions- ) Widerspruchsmarke 005 137 708 unbeschränkt bestritten , wobei er die Einrede in den Schriftsätzen vom 4. Dezember 2014 und 20. September 2017 nochmals wiederholt hat .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Example 12** (doc_id: `59559`) (sent_id: `59559`)


I. Das Wortzeichen Wohlfühlfarbe ist am 1. März 2016 zur Eintragung als Marke in das vom Deutschen Patent- und Markenamt geführte Register angemeldet worden für folgende Waren :

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `Wohlfühlfarbe` (ORG)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `57135`) (sent_id: `57135`)


unter Zurückweisung der Beschwerde der Einsprechenden den Beschluss der Patentabteilung 43 des Deutschen Patent- und Markenamt vom 4. November 2015 aufzuheben und das Streitpatent vollumfänglich aufrechtzuerhalten ,

**False Positives:**

- `Deutschen Patent- und Markenamt` — partial — pred is substring of gold: `Patentabteilung 43 des Deutschen Patent- und Markenamt`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Patentabteilung 43 des Deutschen Patent- und Markenamt`(ORG)

</details>

---

## `European Court Full Name`

**F1:** 0.029 | **Precision:** 0.923 | **Recall:** 0.015  

**Format:** `regex`  
**Rule ID:** `a166a523`  
**Description:**
Matches the full name of the European Court of Human Rights.

**Content:**
```
\b(Europ\u00e4ische\s+Gerichtshof\s+f\u00fcr\s+Menschenrechte|Europ\u00e4ischen\s+Gerichtshofs\s+f\u00fcr\s+Menschenrechte|Europ\u00e4ischen\s+Gerichtshofes|Europ\u00e4ischen\s+Gerichtshof|Europ\u00e4ischen\s+Gerichtshofes)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.923 | 0.015 | 0.029 | 13 | 12 | 1 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 12 | 1 | 729 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53903`) (sent_id: `53903`)


Sie genügen den Anforderungen des Europäischen Gerichtshofs für Menschenrechte an die Überprüfbarkeit einer lebenslangen Freiheitsstrafe .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofs für Menschenrechte` | `Europäischen Gerichtshofs für Menschenrechte` |

**Example 1** (doc_id: `54815`) (sent_id: `54815`)


Die Entscheidungen des Europäischen Gerichtshofs für Menschenrechte seien aufgrund der Völkerrechtsfreundlichkeit des Grundgesetzes in Deutschland zu berücksichtigen .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofs für Menschenrechte` | `Europäischen Gerichtshofs für Menschenrechte` |

**Missed by this rule (FN):**

- `Grundgesetzes` (NRM)
- `Deutschland` (LOC)

**Example 2** (doc_id: `55710`) (sent_id: `55710`)


Der Europäische Gerichtshof für Menschenrechte habe seit dem Jahr 2008 in mehreren Entscheidungen das Recht auf Kollektivverhandlungen und Streik als Bestandteil von Art. 11 EMRK anerkannt , auch für beamtete Lehrkräfte in der Türkei .

| Predicted | Gold |
|---|---|
| `Europäische Gerichtshof für Menschenrechte` | `Europäische Gerichtshof für Menschenrechte` |

**Missed by this rule (FN):**

- `Art. 11 EMRK` (NRM)
- `Türkei` (LOC)

**Example 3** (doc_id: `55771`) (sent_id: `55771`)


Ungeachtet möglicher Ungenauigkeiten bei der Übersetzung des in der amtlichen Fassung nur in französischer Sprache vorliegenden Urteils ist bei einer Bewertung dieser Aussage mit Blick auf die einzelnen Ausprägungen der Völkerrechtsfreundlichkeit des Grundgesetzes mit einzustellen , dass der Europäische Gerichtshof für Menschenrechte - wie auch die Parenthese comme en l'espèce verdeutlicht - eine Aussage in einem konkret-individuell zu entscheidenden Verfahren getroffen hat .

| Predicted | Gold |
|---|---|
| `Europäische Gerichtshof für Menschenrechte` | `Europäische Gerichtshof für Menschenrechte` |

**Missed by this rule (FN):**

- `Grundgesetzes` (NRM)

**Example 4** (doc_id: `56026`) (sent_id: `56026`)


Die Streikteilnahme eines Beamten lasse sich auch nicht mit Blick auf Art. 11 EMRK und die hierzu ergangene jüngere Rechtsprechung des Europäischen Gerichtshofs für Menschenrechte rechtfertigen .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofs für Menschenrechte` | `Europäischen Gerichtshofs für Menschenrechte` |

**Missed by this rule (FN):**

- `Art. 11 EMRK` (NRM)

**Example 5** (doc_id: `56194`) (sent_id: `56194`)


Eine andere Beurteilung ergebe sich auch nicht aus der Entscheidung des Europäischen Gerichtshofs für Menschenrechte im Verfahren Enerji Yapi-Yol Sen v. Türkei .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofs für Menschenrechte` | `Europäischen Gerichtshofs für Menschenrechte` |

**Missed by this rule (FN):**

- `Enerji Yapi-Yol Sen` (PER)
- `Türkei` (LOC)

**Example 6** (doc_id: `58230`) (sent_id: `58230`)


Der Konventionstext und die Rechtsprechung des Europäischen Gerichtshofs für Menschenrechte dienen nach der ständigen Rechtsprechung des Bundesverfassungsgerichts auf der Ebene des Verfassungsrechts als Auslegungshilfen für die Bestimmung von Inhalt und Reichweite von Grundrechten und rechtsstaatlichen Grundsätzen des Grundgesetzes , sofern dies nicht zu einer Einschränkung oder Minderung des Grundrechtsschutzes nach dem Grundgesetz führt ( vgl. BVerfGE 74 , 358 < 370 > ; 111 , 307 < 317 > ; 120 , 180 < 200 f. > ) .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofs für Menschenrechte` | `Europäischen Gerichtshofs für Menschenrechte` |

**Missed by this rule (FN):**

- `Bundesverfassungsgerichts` (ORG)
- `Grundgesetzes` (NRM)
- `Grundgesetz` (NRM)
- `BVerfGE 74 , 358 < 370 > ; 111 , 307 < 317 > ; 120 , 180 < 200 f. >` (RS)

**Example 7** (doc_id: `58307`) (sent_id: `58307`)


Zwar hebt der Europäische Gerichtshof für Menschenrechte die Verantwortung für die Verhinderung einer gegen Art. 3 EMRK verstoßenden Behandlung in einem Drittstaat , welche die Europäische Menschenrechtskonvention den Konventionsstaaten bei Überstellung in diesen Drittstaat auferlegt , hervor ( vgl. EGMR , Urteil vom 4. September 2014 - 140/10 - , Trabelsi / Belgien , Rn. 119 f. m. w. N. ) .

| Predicted | Gold |
|---|---|
| `Europäische Gerichtshof für Menschenrechte` | `Europäische Gerichtshof für Menschenrechte` |

**Missed by this rule (FN):**

- `Art. 3 EMRK` (NRM)
- `Europäische Menschenrechtskonvention` (NRM)
- `EGMR , Urteil vom 4. September 2014 - 140/10 - , Trabelsi / Belgien , Rn. 119 f.` (RS)

**Example 8** (doc_id: `58611`) (sent_id: `58611`)


Wie der Europäische Gerichtshof für Menschenrechte in seiner Entscheidung im Verfahren Demir und Baykara v. Türkei ausgeführt hat , setzt die Rechtfertigung eines Eingriffs in Art. 11 Abs. 1 EMRK ein dringendes soziales beziehungsweise gesellschaftliches Bedürfnis ( " pressing social need " ) voraus ; zudem muss die Einschränkung verhältnismäßig sein ( vgl. EGMR < GK > , Demir and Baykara v. Turkey , Urteil vom 12. November 2008 , Nr. 34503/97 , § 119 ) .

| Predicted | Gold |
|---|---|
| `Europäische Gerichtshof für Menschenrechte` | `Europäische Gerichtshof für Menschenrechte` |

**Missed by this rule (FN):**

- `Demir` (PER)
- `Baykara` (PER)
- `Türkei` (LOC)
- `Art. 11 Abs. 1 EMRK` (NRM)
- `EGMR < GK > , Demir and Baykara v. Turkey , Urteil vom 12. November 2008 , Nr. 34503/97 , § 119` (RS)

**Example 9** (doc_id: `58917`) (sent_id: `58917`)


Das deckt sich mit der nach Art. 1 Abs. 2 GG gebotenen Berücksichtigung der EMRK bei der Auslegung des Grundgesetzes und der in diesem Zusammenhang ergangenen Rechtsprechung des Europäischen Gerichtshofs für Menschenrechte ( vgl. hierzu BVerfGE 111 , 307 < 329 f. > ; 128 , 326 < 369 > ; 140 , 317 < 359 Rn. 91 > ; BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 8. Mai 2017 - 2 BvR 157/17 - , NVwZ 2017 , S. 1196 ; Beschluss der 2. Kammer des Zweiten Senats vom 18. August 2017 - 2 BvR 424/17 - , juris , Rn. 36 ) .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofs für Menschenrechte` | `Europäischen Gerichtshofs für Menschenrechte` |

**Missed by this rule (FN):**

- `Art. 1 Abs. 2 GG` (NRM)
- `EMRK` (NRM)
- `Grundgesetzes` (NRM)
- `BVerfGE 111 , 307 < 329 f. > ; 128 , 326 < 369 > ; 140 , 317 < 359 Rn. 91 >` (RS)
- `BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 8. Mai 2017 - 2 BvR 157/17 - , NVwZ 2017 , S. 1196` (RS)
- `Beschluss der 2. Kammer des Zweiten Senats vom 18. August 2017 - 2 BvR 424/17 - , juris , Rn. 36` (RS)

**Example 10** (doc_id: `59621`) (sent_id: `59621`)


Das Vorliegen einer Verwechslungsgefahr für das Publikum ist nach ständiger Rechtsprechung sowohl des Europäischen Gerichtshofes als auch des Bundesgerichtshofes unter Berücksichtigung aller relevanten Umstände des Einzelfalls zu beurteilen ( vgl. hierzu z.B. EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER ; GRUR 2010 , 1098 Rn. 44 – Calvin Klein / HABM ; BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY ; GRUR 2012 , 1040 Rn. 25 – pjur / pure ; GRUR 2013 , 833 Rn. 30 – Culinaria / Villa Culinaria ) .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofes` | `Europäischen Gerichtshofes` |

**Missed by this rule (FN):**

- `Bundesgerichtshofes` (ORG)
- `EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER` (RS)
- `GRUR 2010 , 1098 Rn. 44 – Calvin Klein / HABM` (RS)
- `BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY` (RS)
- `GRUR 2012 , 1040 Rn. 25 – pjur / pure` (RS)
- `GRUR 2013 , 833 Rn. 30 – Culinaria / Villa Culinaria` (RS)

**Example 11** (doc_id: `59735`) (sent_id: `59735`)


( 5 ) Es kann offen bleiben , ob bei den Anforderungen an die konkrete gesetzliche Ausgestaltung des Überprüfungsmechanismus durch nationales Recht nach der Rechtsprechung des Europäischen Gerichtshofs für Menschenrechte zwischen der Vollstreckung der lebenslangen Freiheitsstrafe in Signatarstaaten der Europäischen Menschenrechtskonvention und in Drittstaaten zu unterscheiden ist .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofs für Menschenrechte` | `Europäischen Gerichtshofs für Menschenrechte` |

**Missed by this rule (FN):**

- `Europäischen Menschenrechtskonvention` (NRM)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `59790`) (sent_id: `59790`)


Die vom Europäischen Gerichtshof für Menschenrechte formulierten Grundsätze zum Streikrecht seien im Kern auf deutsche Beamte übertragbar .

**False Positives:**

- `Europäischen Gerichtshof` — partial — pred is substring of gold: `Europäischen Gerichtshof für Menschenrechte`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europäischen Gerichtshof für Menschenrechte`(ORG)

</details>

---

## `Missing Specific Organizations`

**F1:** 0.029 | **Precision:** 0.706 | **Recall:** 0.015  

**Format:** `regex`  
**Rule ID:** `0bfd9088`  
**Description:**
Matches specific organizations found in training data that were missed, including Schott AG, Europäisches Patentamt, Fliegerhorst, Monster Abwehr Spray, DB, GBA, etc.

**Content:**
```
\b(?:Schott\s+AG|Europ\u00e4isches\s+Patentamt|Europ\u00e4ischen\s+Patentamt|Fliegerhorst\s+B\u00fcchel|Fliegerhorsts\s+B\u00fcchel|Monster\s+Abwehr\s+Spray|DB\s+Konzern|DB\s+Netz|GBA|Bundesnetzagentur|S\u00e4chsische\s+Bildungsagentur|Evangelischen\s+Kirche\s+Deutschland|Institut\s+f\u00fcr\s+das\s+Entgeltsystem\s+im\s+Krankenhaus|InEK|Nationalen\s+Volksarmee\s+der\s+Deutschen\s+Demokratischen\s+Republik|Bundesministerium\s+des\s+Innern|Verein\s+S\s+e\.\s*V\.|bayerische\s+Fl\u00fcchtlingsrat|Institut\s+f\u00fcr\s+das\s+Entgeltsystem\s+im\s+Krankenhaus|InEK|High\s+Court\s+of\s+Justice|Bundeszentralamt\s+f\u00fcr\s+Steuern|Neue\s+Richtervereinigung|RAP|Kooperationskasse\s+des\s+Vereins\s+Neudeutschland|Vorstand\s+von\s+Neudeutschland|Jobcenters\s+Landkreis|Justizvollzuganstalten\s+\(\s*JVA\s*\)|4\.\s+Kammer\s+des\s+Truppendienstgerichts\s+S\u00fcd|I\.\s+Senats\s+des\s+Anwaltsgerichtshofs\s+in\s+der\s+Freien\s+und\s+Hansestadt\s+Hamburg|29\.\s+Zivilkammer\s+des\s+Landgerichts|2\.\s+Strafsenats|V\.\s+Senats\s+des\s+BFH|25\.\s+Senat\s+\(\s*Marken-Beschwerdesenat\s*\)\s+des\s+Bundespatentgerichts|2\.\s+Senats\s+des\s+LSG|11\.\s+Senats\s+des\s+LSG|S\u00e4chsischen\s+LSG|Landessozialgerichts\s+Rheinland-Pfalz|Landessozialgerichts\s+<\s*LSG\s*>\s+Niedersachsen-Bremen|Landessozialgerichts\s+Mecklenburg-Vorpommern|Landgericht\s+Potsdam|Sozialgerichts\s+<\s*SG\s*>\s+Hildesheim|Amtsgericht\s+Frankfurt\s+am\s+Main\s+-\s+Au\u00dfenstelle\s+H\u00f6chst\s+-|Amtsgericht\s+Frankfurt|Landgericht\s+Karlsruhe|Landgericht\s+Bremen|Landgericht\s+Halle|Landgericht\s+K\u00f6ln|Landgericht\s+M\u00fcnchen|Finanzgericht\s+M\u00fcnchen|Oberlandesgericht\s+D\u00fcsseldorf|Oberlandesgericht\s+M\u00fcnchen|Amtsgericht\s+Frankfurt\s+am\s+Main|Amtsgericht\s+Frankfurt|Landgericht\s+Halle|Landgericht\s+K\u00f6ln|Landgericht\s+M\u00fcnchen|Finanzgericht\s+M\u00fcnchen|Oberlandesgericht\s+D\u00fcsseldorf|Oberlandesgericht\s+M\u00fcnchen|Amtsgericht\s+Frankfurt\s+am\s+Main|Amtsgericht\s+Frankfurt|Landgericht\s+Halle|Landgericht\s+K\u00f6ln|Landgericht\s+M\u00fcnchen|Finanzgericht\s+M\u00fcnchen|Oberlandesgericht\s+D\u00fcsseldorf|Oberlandesgericht\s+M\u00fcnchen)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.706 | 0.015 | 0.029 | 17 | 12 | 5 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 12 | 5 | 659 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `55265`) (sent_id: `55265`)


Die Revision des Klägers gegen das Urteil des Landessozialgerichts Rheinland-Pfalz vom 9. Juni 2016 wird zurückgewiesen .

| Predicted | Gold |
|---|---|
| `Landessozialgerichts Rheinland-Pfalz` | `Landessozialgerichts Rheinland-Pfalz` |

**Example 1** (doc_id: `55283`) (sent_id: `55283`)


Die Klägerin beantragt , das Urteil des Sächsischen LSG vom 19. Mai 2016 aufzuheben und die Berufung der Beklagten gegen das Urteil des SG Chemnitz vom 9. Oktober 2014 zurückzuweisen .

| Predicted | Gold |
|---|---|
| `Sächsischen LSG` | `Sächsischen LSG` |

**Missed by this rule (FN):**

- `SG Chemnitz` (ORG)

**Example 2** (doc_id: `57702`) (sent_id: `57702`)


Die Prüfung der eingesetzten Methoden im zugelassenen Krankenhaus erfolgt vielmehr bis zu einer Entscheidung des GBA nach § 137c SGB V individuell , grundsätzlich also zunächst präventiv im Rahmen einer Binnenkontrolle durch das Krankenhaus selbst , sodann im Wege der nachgelagerten Außenkontrolle lediglich im Einzelfall anlässlich von Beanstandungen ex post durch die KK und anschließender Prüfung durch die Gerichte .

| Predicted | Gold |
|---|---|
| `GBA` | `GBA` |

**Missed by this rule (FN):**

- `§ 137c SGB V` (NRM)

**Example 3** (doc_id: `57706`) (sent_id: `57706`)


Nach § 92 Abs 1 S 1 , S 2 Nr 6 SGB V beschließt der GBA die zur Sicherung der ärztlichen Versorgung erforderlichen Richtlinien über die Gewähr für eine ausreichende , zweckmäßige und wirtschaftliche Versorgung der Versicherten mit Arzneimitteln .

| Predicted | Gold |
|---|---|
| `GBA` | `GBA` |

**Missed by this rule (FN):**

- `§ 92 Abs 1 S 1 , S 2 Nr 6 SGB V` (NRM)

**Example 4** (doc_id: `57780`) (sent_id: `57780`)


Dementsprechend hat der EuGH auch das ihm vom britischen High Court of Justice als Prüfungskriterium für die Anwendung des Art. 3 ( a ) AMVO vorgeschlagene Konzept des sogenannten „ inventive advance “ ( High Court , [ 2012 ] EWHC 2545 ( Pat ) , Rnd. 75 ff. – Actavis / Sanofi , ) bei der Auslegung dieser Vorschrift nicht aufgegriffen , sondern stattdessen im Rahmen der Auslegung des Art. 3 ( c ) AMVO berücksichtigt ( vgl. EuGH , GRUR Int. 2014 , 153 , Rnd. 41 f. – Actavis / Sanofi ) .

| Predicted | Gold |
|---|---|
| `High Court of Justice` | `High Court of Justice` |

**Missed by this rule (FN):**

- `EuGH` (ORG)
- `Art. 3 ( a ) AMVO` (NRM)
- `High Court , [ 2012 ] EWHC 2545 ( Pat ) , Rnd. 75 ff. – Actavis / Sanofi` (RS)
- `Art. 3 ( c ) AMVO` (NRM)
- `EuGH , GRUR Int. 2014 , 153 , Rnd. 41 f. – Actavis / Sanofi` (RS)

**Example 5** (doc_id: `58297`) (sent_id: `58297`)


In Bezug auf den gerügten Haftbefehl des Amtsgerichts Neu-Ulm vom 31. Juli 2017 sei die Verfassungsbeschwerde wegen des Grundsatzes der Subsidiarität der Verfassungsbeschwerde hingegen unzulässig , da eine abschließende Sachprüfung durch das Oberlandesgericht München noch nicht stattgefunden habe .

| Predicted | Gold |
|---|---|
| `Oberlandesgericht München` | `Oberlandesgericht München` |

**Missed by this rule (FN):**

- `Amtsgerichts Neu-Ulm` (ORG)

**Example 6** (doc_id: `58469`) (sent_id: `58469`)


Die vom Senat in seiner Zwischenverfügung eingenommene Position stehe im Widerspruch zur Auslegung der Rechtsprechung des EuGH durch den britischen High Court of Justice .

| Predicted | Gold |
|---|---|
| `High Court of Justice` | `High Court of Justice` |

**Missed by this rule (FN):**

- `EuGH` (ORG)

**Example 7** (doc_id: `58582`) (sent_id: `58582`)


Im Einspruchsverfahren vor dem Europäischen Patentamt wurde das Grundpatent in erster Instanz wegen fehlender Patentfähigkeit widerrufen .

| Predicted | Gold |
|---|---|
| `Europäischen Patentamt` | `Europäischen Patentamt` |

**Example 8** (doc_id: `58597`) (sent_id: `58597`)


Gemäß § 91 Abs 6 SGB V sind die Beschlüsse des GBA mit Ausnahme der Beschlüsse zu Entscheidungen nach § 136d SGB V ( vor dem 1. 1. 2016 : § 137b SGB V ) für die Träger iS des § 91 Abs 1 S 1 SGB V , deren Mitglieder und Mitgliedskassen sowie für die Versicherten und die Leistungserbringer verbindlich .

| Predicted | Gold |
|---|---|
| `GBA` | `GBA` |

**Missed by this rule (FN):**

- `§ 91 Abs 6 SGB V` (NRM)
- `§ 136d SGB V` (NRM)
- `§ 137b SGB V` (NRM)
- `§ 91 Abs 1 S 1 SGB V` (NRM)

**Example 9** (doc_id: `58794`) (sent_id: `58794`)


Den LSG-Feststellungen aufgrund des Gutachtens " Liposuktion bei Lip- und Lymphödemen " der Sozialmedizinischen Expertengruppe 7 des MDK vom 6. 10. 2011 nebst Gutachtensaktualisierung ( 15. 1. 2015 ; abrufbar unter www.mds-ev.de/richtlinien-publikationen/gutachten-nutzenbewertungen.html dort Gutachten Liposuktion bei Lip- und Lymphödemen ) entspricht im Übrigen die Beurteilung des GBA in den " Tragenden Gründen zum Beschluss des Gemeinsamen Bundesausschusses über eine Änderung der Richtlinie Methoden Krankenhausbehandlung : Liposuktion bei Lipödem vom 20. 7. 2017 " ( abrufbar unter www.g-ba.de/informationen/beschluesse/3013/ ; zur Möglichkeit , Erkenntnisse auf Beschlüsse des GBA zu stützen : BSGE 101 , 177 = SozR 4 - 2500 § 109 Nr 6 , RdNr 50 ) .

| Predicted | Gold |
|---|---|
| `GBA` | `GBA` |

**Missed by this rule (FN):**

- `MDK` (ORG)
- `Gemeinsamen Bundesausschusses` (ORG)
- `BSGE 101 , 177 = SozR 4 - 2500 § 109 Nr 6 , RdNr 50` (RS)

**Example 10** (doc_id: `59206`) (sent_id: `59206`)


HLNK13 Änderungsmarkierte Version der im Beschränkungsverfahren vor dem Europäischen Patentamt beschränkten Ansprüche vom 14. Februar 2014

| Predicted | Gold |
|---|---|
| `Europäischen Patentamt` | `Europäischen Patentamt` |

**Example 11** (doc_id: `59909`) (sent_id: `59909`)


Es ist ohne Belang , dass der GBA nach den Selbstbeschaffungen beschlossen hat , das Verfahren zur Bewertung der Liposuktion bei Lipödem auszusetzen und eine entsprechende Erprobungsrichtlinie nach § 137e SGB V zu erlassen ( 20. 7. 2017 ; vgl www.g-ba.de/informationen/beschluesse/3013/ ; zur grundsätzlichen Berücksichtigungsfähigkeit von Rechtsänderungen im Revisionsverfahren vgl Hauck in Zeihe / Hauck , SGG , Stand August 2017 , § 162 Anm 10b und § 163 Anm 4f mwN ) .

| Predicted | Gold |
|---|---|
| `GBA` | `GBA` |

**Missed by this rule (FN):**

- `§ 137e SGB V` (NRM)
- `Hauck in Zeihe / Hauck , SGG , Stand August 2017 , § 162 Anm 10b und § 163 Anm 4f` (LIT)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `54750`) (sent_id: `54750`)


Nach diesen Grundsätzen besteht zwischen der angegriffenen Wortmarke „ Monster Abwehr Spray “ und den älteren Widerspruchsmarken „ Monster “ im maßgeblichen Zusammenhang mit den beschwerdegegenständlichen Waren und Dienstleistungen keine Verwechslungsgefahr .

**False Positives:**

- `Monster Abwehr Spray` — partial — pred is substring of gold: `„ Monster Abwehr Spray “`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `„ Monster Abwehr Spray “`(ORG)
- `„ Monster “`(ORG)

**Example 1** (doc_id: `55225`) (sent_id: `55225`)


2. Hat der Aufhebungs- und Erstattungsbescheid des Jobcenters Landkreis W vor den Sozialgerichten Bestand , fehlt es gleichwohl an einem Schaden .

**False Positives:**

- `Jobcenters Landkreis` — partial — pred is substring of gold: `Jobcenters Landkreis W`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Jobcenters Landkreis W`(ORG)

**Example 2** (doc_id: `55768`) (sent_id: `55768`)


Schließlich ließ sich der „ Vorstand von Neudeutschland “ die Berechtigung einräumen , die Einzahlung für die Verwirklichung eines Projekts zusammenzulegen .

**False Positives:**

- `Vorstand von Neudeutschland` — partial — gold is substring of pred: `Neudeutschland`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Neudeutschland`(ORG)

**Example 3** (doc_id: `56210`) (sent_id: `56210`)


- vorbehaltlich der Fördermittelzusage durch die Sächsische Bildungsagentur D für das Schuljahr 2012/2013

**False Positives:**

- `Sächsische Bildungsagentur` — partial — pred is substring of gold: `Sächsische Bildungsagentur D`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Sächsische Bildungsagentur D`(ORG)

**Example 4** (doc_id: `59773`) (sent_id: `59773`)


4. Durch Beschluss vom 20. November 2014 verwarf das Landgericht München I die Beschwerde des Beschwerdeführers als unbegründet .

**False Positives:**

- `Landgericht München` — partial — pred is substring of gold: `Landgericht München I`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Landgericht München I`(ORG)

</details>

---

## `European Entities`

**F1:** 0.024 | **Precision:** 0.625 | **Recall:** 0.012  

**Format:** `regex`  
**Rule ID:** `a09e7f00`  
**Description:**
Matches specific European entities.

**Content:**
```
\b(Europ\u00e4ischer\s+Gerichtshof\s+f\u00fcr\s+Menschenrechte|Europ\u00e4ische\s+Menschenrechtskonvention|Europ\u00e4ische\s+Kommission|Europ\u00e4ischer\s+Gerichtshof|Europ\u00e4ische\s+Union|Europ\u00e4ischen\s+Union|Europ\u00e4ischen\s+Gerichtshofs\s+f\u00fcr\s+Menschenrechte|Gerichtshofs\s+der\s+Europ\u00e4ischen\s+Union)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.625 | 0.012 | 0.024 | 16 | 10 | 6 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 10 | 6 | 790 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53410`) (sent_id: `53410`)


Ob die Europäische Kommission in Anwendung der Grundsätze der Nr. 89 der Vertikal-Leitlinien eine andere Auffassung vertrete , sei unerheblich , weil es bei der Feststellung des relevanten Marktes im Sinne des § 18 Abs. 1 GWB um eine Frage des nationalen Rechts gehe .

| Predicted | Gold |
|---|---|
| `Europäische Kommission` | `Europäische Kommission` |

**Missed by this rule (FN):**

- `Nr. 89 der Vertikal-Leitlinien` (NRM)
- `§ 18 Abs. 1 GWB` (NRM)

**Example 1** (doc_id: `53700`) (sent_id: `53700`)


3. Das Bundesverfassungsgericht überprüft die Vereinbarkeit eines nationalen Gesetzes mit dem Grundgesetz auch , wenn zugleich Zweifel an der Vereinbarkeit des Gesetzes mit Sekundärrecht der Europäischen Union bestehen .

| Predicted | Gold |
|---|---|
| `Europäischen Union` | `Europäischen Union` |

**Missed by this rule (FN):**

- `Bundesverfassungsgericht` (ORG)
- `Grundgesetz` (NRM)

**Example 2** (doc_id: `53856`) (sent_id: `53856`)


c ) Die nach § 9 Abs. 1 Satz 3 DVO.EKD aF erfolgte Zuordnung der Klägerin zu höchstens Stufe 2 der Entgeltgruppe 14 DVO.EKD verstieß nicht gegen das Recht der Europäischen Union .

| Predicted | Gold |
|---|---|
| `Europäischen Union` | `Europäischen Union` |

**Missed by this rule (FN):**

- `§ 9 Abs. 1 Satz 3 DVO.EKD aF` (REG)
- `DVO.EKD` (REG)

**Example 3** (doc_id: `55109`) (sent_id: `55109`)


III. Zur Anrufung des Gerichtshofs der Europäischen Union ( EuGH )

| Predicted | Gold |
|---|---|
| `Gerichtshofs der Europäischen Union` | `Gerichtshofs der Europäischen Union` |

**Missed by this rule (FN):**

- `EuGH` (ORG)

**Example 4** (doc_id: `55373`) (sent_id: `55373`)


Nach den dargestellten Grundsätzen fehlt der Europäischen Union die Rechtsmacht , einer Regelung des nationalen Rechts die Wirksamkeit für Sachverhalte zu nehmen , welche keinen hinreichenden Bezug zu anderen EU-Mitgliedstaaten aufweisen und deshalb außerhalb der Regelungskompetenz der Europäischen Union liegen .

| Predicted | Gold |
|---|---|
| `Europäischen Union` | `Europäischen Union` |
| `Europäischen Union` | `Europäischen Union` |

**Example 5** (doc_id: `57786`) (sent_id: `57786`)


Das FG hat aber zu Recht darauf hingewiesen , dass nach der Rechtsprechung des Gerichtshofs der Europäischen Union ( EuGH ) bei der Auslegung des Art. 10 der VO Nr. 574/72 die " allgemeinen Vorschriften " des in Titel I der VO Nr. 1408/71 und damit Art. 12 der VO Nr. 1408/71 zu beachten sind ( EuGH-Urteil Wiering vom 8. Mai 2014 C - 347/12 , EU : C : 2014 : 300 , Rz 54 ff. , Zeitschrift für europäisches Sozial- und Arbeitsrecht - ZESAR - 2015 , 113 ) .

| Predicted | Gold |
|---|---|
| `Gerichtshofs der Europäischen Union` | `Gerichtshofs der Europäischen Union` |

**Missed by this rule (FN):**

- `EuGH` (ORG)
- `Art. 10 der VO Nr. 574/72` (NRM)
- `Titel I der VO Nr. 1408/71` (NRM)
- `Art. 12 der VO Nr. 1408/71` (NRM)
- `EuGH-Urteil Wiering vom 8. Mai 2014 C - 347/12 , EU : C : 2014 : 300 , Rz 54 ff.` (RS)
- `Zeitschrift für europäisches Sozial- und Arbeitsrecht - ZESAR - 2015 , 113` (LIT)

**Example 6** (doc_id: `58096`) (sent_id: `58096`)


Führt der Arbeitgeber im Wege der elektronischen Datenverarbeitung einen Abgleich von Vor- und Nachnamen der bei ihm beschäftigten Arbeitnehmer mit den auf Grundlage der sog. Anti-Terror-Verordnungen der Europäischen Union erstellten Namenslisten durch , ist der Betriebsrat nicht nach § 87 Abs. 1 Nr. 6 BetrVG zu beteiligen .

| Predicted | Gold |
|---|---|
| `Europäischen Union` | `Europäischen Union` |

**Missed by this rule (FN):**

- `§ 87 Abs. 1 Nr. 6 BetrVG` (NRM)

**Example 7** (doc_id: `59538`) (sent_id: `59538`)


Die Überwachung der Einhaltung der unmittelbar geltenden Rechtsakte der Europäischen Union ist nach § 38 Abs. 1 , § 39 Abs. 1 LFGB die Aufgabe der nach Landesrecht zuständigen Behörden ; in Niedersachsen ist sie den Landkreisen und kreisfreien Städten übertragen ( vgl. § 2 Nr. 5 Buchst. a ZustVO-SOG aF bzw. § 2 Abs. 1 Nr. 5 ZustVO-SOG nF ) .

| Predicted | Gold |
|---|---|
| `Europäischen Union` | `Europäischen Union` |

**Missed by this rule (FN):**

- `§ 38 Abs. 1 , § 39 Abs. 1 LFGB` (NRM)
- `Niedersachsen` (LOC)
- `§ 2 Nr. 5 Buchst. a ZustVO-SOG aF` (NRM)
- `§ 2 Abs. 1 Nr. 5 ZustVO-SOG nF` (NRM)

**Example 8** (doc_id: `59818`) (sent_id: `59818`)


II. Das Revisionsverfahren wird bis zur Entscheidung des Gerichtshofs der Europäischen Union über das Vorabentscheidungsersuchen ausgesetzt .

| Predicted | Gold |
|---|---|
| `Gerichtshofs der Europäischen Union` | `Gerichtshofs der Europäischen Union` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53910`) (sent_id: `53910`)


Zur Vermeidung einer mittelbaren Diskriminierung wegen Behinderung sei er nach der Entscheidung des Gerichtshofs der Europäischen Union ( EuGH ) vom 6. Dezember 2012 ( - C- 152/11 - [ Odar ] ) wie ein nicht schwerbehinderter Arbeitnehmer zu behandeln .

**False Positives:**

- `Gerichtshofs der Europäischen Union` — partial — pred is substring of gold: `Entscheidung des Gerichtshofs der Europäischen Union ( EuGH ) vom 6. Dezember 2012 ( - C- 152/11 - [ Odar ] )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Entscheidung des Gerichtshofs der Europäischen Union ( EuGH ) vom 6. Dezember 2012 ( - C- 152/11 - [ Odar ] )`(RS)

**Example 1** (doc_id: `55178`) (sent_id: `55178`)


Eine solche Einstellung sei vorliegend wegen Verstoßes gegen die Europäische Menschenrechtskonvention vorzunehmen gewesen .

**False Positives:**

- `Europäische Menschenrechtskonvention` — type mismatch — same span as gold: `Europäische Menschenrechtskonvention`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europäische Menschenrechtskonvention`(NRM)

**Example 2** (doc_id: `57527`) (sent_id: `57527`)


Aus der Stellungnahme der rumänischen Behörden ergäben sich Haftbedingungen , die im Rahmen einer Gesamtbetrachtung eine echte Gefahr unmenschlicher oder erniedrigender Behandlung im Sinne von Art. 4 der Charta der Grundrechte der Europäischen Union ( GRCh ) nicht erkennen ließen .

**False Positives:**

- `Europäischen Union` — partial — pred is substring of gold: `Art. 4 der Charta der Grundrechte der Europäischen Union`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 4 der Charta der Grundrechte der Europäischen Union`(NRM)
- `GRCh`(NRM)

**Example 3** (doc_id: `58307`) (sent_id: `58307`)


Zwar hebt der Europäische Gerichtshof für Menschenrechte die Verantwortung für die Verhinderung einer gegen Art. 3 EMRK verstoßenden Behandlung in einem Drittstaat , welche die Europäische Menschenrechtskonvention den Konventionsstaaten bei Überstellung in diesen Drittstaat auferlegt , hervor ( vgl. EGMR , Urteil vom 4. September 2014 - 140/10 - , Trabelsi / Belgien , Rn. 119 f. m. w. N. ) .

**False Positives:**

- `Europäische Menschenrechtskonvention` — type mismatch — same span as gold: `Europäische Menschenrechtskonvention`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europäische Gerichtshof für Menschenrechte`(ORG)
- `Art. 3 EMRK`(NRM)
- `Europäische Menschenrechtskonvention`(NRM)
- `EGMR , Urteil vom 4. September 2014 - 140/10 - , Trabelsi / Belgien , Rn. 119 f.`(RS)

**Example 4** (doc_id: `58740`) (sent_id: `58740`)


Das Berufungsgericht wird sich bei seiner neuerlichen , durch diesen Beschluss nicht im Ergebnis vorgeprägten Entscheidung auch mit den - wenngleich in anderem Zusammenhang ergangenen - Entscheidungen des Gerichtshofes der Europäischen Union vom 25. Januar 2018 ( - C- 360/16 [ ECLI : EU :C : 2018 : 35 ] , Hasan - Rn. 30 f. , 40 ) und des Bundesverfassungsgerichts vom 21. April 2016 ( - 2 BvR 273/16 - NVwZ 2016 , 1242 Rn. 11 ) zur Frage des maßgeblichen Zeitpunkts für die Beurteilung des Vorliegens systemischer Schwachstellen auseinanderzusetzen haben .

**False Positives:**

- `Europäischen Union` — partial — pred is substring of gold: `Entscheidungen des Gerichtshofes der Europäischen Union vom 25. Januar 2018 ( - C- 360/16 [ ECLI : EU :C : 2018 : 35 ] , Hasan - Rn. 30 f. , 40 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Entscheidungen des Gerichtshofes der Europäischen Union vom 25. Januar 2018 ( - C- 360/16 [ ECLI : EU :C : 2018 : 35 ] , Hasan - Rn. 30 f. , 40 )`(RS)
- `Bundesverfassungsgerichts vom 21. April 2016 ( - 2 BvR 273/16 - NVwZ 2016 , 1242 Rn. 11 )`(RS)

**Example 5** (doc_id: `59093`) (sent_id: `59093`)


4. Rechtsgrundlage für die Anrufung des EuGH ist Art. 267 Abs. 3 des Vertrags über die Arbeitsweise der Europäischen Union ( AEUV ) .

**False Positives:**

- `Europäischen Union` — partial — pred is substring of gold: `Art. 267 Abs. 3 des Vertrags über die Arbeitsweise der Europäischen Union`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH`(ORG)
- `Art. 267 Abs. 3 des Vertrags über die Arbeitsweise der Europäischen Union`(NRM)
- `AEUV`(NRM)

</details>

---

## `Bundesfinanzhof`

**F1:** 0.024 | **Precision:** 0.625 | **Recall:** 0.012  

**Format:** `regex`  
**Rule ID:** `d2e9163d`  
**Description:**
Matches 'Bundesfinanzhof' and its genitive form 'Bundesfinanzhofs'.

**Content:**
```
\b(Bundesfinanzhof|Bundesfinanzhofs)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.625 | 0.012 | 0.024 | 16 | 10 | 6 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 10 | 6 | 796 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53400`) (sent_id: `53400`)


Die Vorgabe des Bundesfinanzhofs ( BFH ) , dass die Einsteller ( Leistungsempfänger ) keine Land- und Forstwirte sein müssten , die von ihnen empfangenen Leistungen ( Zuchtleistungen als einheitliche Leistung ) aber gleichwohl zu landwirtschaftlichen Zwecken genutzt werden müssten , erscheine nicht eindeutig und auf den ersten Blick widersprüchlich .

| Predicted | Gold |
|---|---|
| `Bundesfinanzhofs` | `Bundesfinanzhofs` |

**Missed by this rule (FN):**

- `BFH` (ORG)

**Example 1** (doc_id: `53586`) (sent_id: `53586`)


d ) Mit dem mit der Verfassungsbeschwerde ebenfalls angegriffenen Beschluss vom 8. März 2011 wies der Bundesfinanzhof die von der Beschwerdeführerin erhobene Anhörungsrüge und Gegenvorstellung zurück .

| Predicted | Gold |
|---|---|
| `Bundesfinanzhof` | `Bundesfinanzhof` |

**Example 2** (doc_id: `54626`) (sent_id: `54626`)


III. 1. Der Bundesfinanzhof ging bis zu den hier zu entscheidenden Vorlagebeschlüssen von der Verfassungsmäßigkeit der Einheitswerte als Bemessungsgrundlage der Grundsteuer aus .

| Predicted | Gold |
|---|---|
| `Bundesfinanzhof` | `Bundesfinanzhof` |

**Example 3** (doc_id: `55187`) (sent_id: `55187`)


Vor dem Bundesfinanzhof muss sich - wie auch aus der Rechtsmittelbelehrung in dem vorbezeichneten Beschluss hervorgeht - jeder Beteiligte , sofern es sich nicht um eine juristische Person des öffentlichen Rechts oder um eine Behörde handelt , durch einen Rechtsanwalt , Steuerberater , Steuerbevollmächtigten , Wirtschaftsprüfer oder vereidigten Buchprüfer als Bevollmächtigten vertreten lassen ; zur Vertretung berechtigt sind auch Gesellschaften i. S. des § 3 Nr. 2 und 3 des Steuerberatungsgesetzes ( StBerG ) , die durch solche Personen handeln ( § 62 Abs. 4 i. V. m. Abs. 2 Satz 1 der Finanzgerichtsordnung - FGO - ) .

| Predicted | Gold |
|---|---|
| `Bundesfinanzhof` | `Bundesfinanzhof` |

**Missed by this rule (FN):**

- `§ 3 Nr. 2 und 3 des Steuerberatungsgesetzes` (NRM)
- `StBerG` (NRM)
- `§ 62 Abs. 4 i. V. m. Abs. 2 Satz 1 der Finanzgerichtsordnung` (NRM)
- `FGO` (NRM)

**Example 4** (doc_id: `55678`) (sent_id: `55678`)


Dies widerspreche den Vorgaben des Bundesfinanzhofs ( BFH ) in dessen Beschluss vom 10. September 2013 XI B 114/12 ( BFH / NV 2013 , 1947 ) .

| Predicted | Gold |
|---|---|
| `Bundesfinanzhofs` | `Bundesfinanzhofs` |

**Missed by this rule (FN):**

- `BFH` (ORG)
- `Beschluss vom 10. September 2013 XI B 114/12 ( BFH / NV 2013 , 1947` (RS)

**Example 5** (doc_id: `55801`) (sent_id: `55801`)


Das FG führte ausweislich des Protokolls hierzu aus , es sei ihm nicht möglich , einen sicheren richterlichen Hinweis dazu zu geben , welche Anforderungen nach Auffassung des Bundesfinanzhofs ( BFH ) im Einzelfall an den Gegenbeweis der fehlenden Manipulationsmöglichkeiten zu stellen seien .

| Predicted | Gold |
|---|---|
| `Bundesfinanzhofs` | `Bundesfinanzhofs` |

**Missed by this rule (FN):**

- `BFH` (ORG)

**Example 6** (doc_id: `56233`) (sent_id: `56233`)


4. Der Bundesfinanzhof hat in seinen Vorlagebeschlüssen nicht mehr hinnehmbare Defizite beim Gesetzesvollzug beanstandet .

| Predicted | Gold |
|---|---|
| `Bundesfinanzhof` | `Bundesfinanzhof` |

**Example 7** (doc_id: `58054`) (sent_id: `58054`)


Das Urteil des Bundesfinanzhofs verletzt die Beschwerdeführerin auch nicht in ihren prozessualen grundrechtsgleichen Rechten ( III. .

| Predicted | Gold |
|---|---|
| `Bundesfinanzhofs` | `Bundesfinanzhofs` |

**Example 8** (doc_id: `58724`) (sent_id: `58724`)


Die Auffassung des Bundesfinanzhofs , dass er wegen der zwischenzeitlich am 5. August 2010 erfolgten telefonischen Bekanntgabe der Urteilsformel und der dadurch eingetretenen Selbstbindung die erst am 19. August 2010 veröffentlichten Entscheidungen des Bundesverfassungsgerichts vom 7. Juli 2010 in seinem Urteil nicht mehr hätte berücksichtigen können ( BFH , Beschluss vom 8. März 2011 - IV S 14/10 - , juris , Rn. 8 ff. ) , betrifft die fachgerichtliche Auslegung einfachen Prozessrechts .

| Predicted | Gold |
|---|---|
| `Bundesfinanzhofs` | `Bundesfinanzhofs` |

**Missed by this rule (FN):**

- `Bundesverfassungsgerichts` (ORG)
- `BFH , Beschluss vom 8. März 2011 - IV S 14/10 - , juris , Rn. 8 ff.` (RS)

**Example 9** (doc_id: `58854`) (sent_id: `58854`)


Unschädlich ist , dass der Bundesfinanzhof in seinen Vorlagebeschlüssen keine konkreten Feststellungen dazu getroffen hat , ob die Kläger der Ausgangsverfahren durch die geltend gemachten Wertverzerrungen individuell benachteiligt werden .

| Predicted | Gold |
|---|---|
| `Bundesfinanzhof` | `Bundesfinanzhof` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `54143`) (sent_id: `54143`)


Dieser während des Revisionsverfahrens eingetretene Zuständigkeitswechsel führt zu einem gesetzlichen Beteiligtenwechsel ( vgl. z.B. Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109 , m. w. N. ) .

**False Positives:**

- `Bundesfinanzhofs` — partial — pred is substring of gold: `Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109`(RS)

**Example 1** (doc_id: `54208`) (sent_id: `54208`)


Hierdurch ist das Verfahren über das Ablehnungsgesuch abgeschlossen worden , denn erst zu diesem Zeitpunkt war das Gericht an seine Entscheidung gebunden ( vgl. auch Beschluss des Bundesfinanzhofs - BFH - vom 17. Juli 2008 I B 22/08 , juris , unter II. 2. zum Abschluss des Ablehnungsverfahrens im Zeitpunkt der Absendung der Entscheidung durch die Geschäftsstelle ) .

**False Positives:**

- `Bundesfinanzhofs` — partial — pred is substring of gold: `Beschluss des Bundesfinanzhofs - BFH - vom 17. Juli 2008 I B 22/08 , juris , unter II. 2.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Beschluss des Bundesfinanzhofs - BFH - vom 17. Juli 2008 I B 22/08 , juris , unter II. 2.`(RS)

**Example 2** (doc_id: `54526`) (sent_id: `54526`)


Danach gehören alle Leistungen des Erwerbers zur grunderwerbsteuerrechtlichen Gegenleistung ( Bemessungsgrundlage ) , die dieser nach den vertraglichen Vereinbarungen gewährt , um das Grundstück zu erwerben ( Urteil des Bundesfinanzhofs - BFH - vom 8. März 2017 II R 38/14 , BFHE 257 , 368 , BStBl II 2017 , 1005 , Rz 26 , m. w. N. ) .

**False Positives:**

- `Bundesfinanzhofs` — partial — pred is substring of gold: `Urteil des Bundesfinanzhofs - BFH - vom 8. März 2017 II R 38/14 , BFHE 257 , 368 , BStBl II 2017 , 1005 , Rz 26`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Bundesfinanzhofs - BFH - vom 8. März 2017 II R 38/14 , BFHE 257 , 368 , BStBl II 2017 , 1005 , Rz 26`(RS)

**Example 3** (doc_id: `57079`) (sent_id: `57079`)


Denn Sinn und Zweck der Pauschbetragsregelung ist es gerade , typisierend zu unterstellen , dass bestimmten Gruppen von behinderten Menschen gewisse außergewöhnliche Belastungen erwachsen ( vgl. z.B. Blümich / K. Heger , § 33b EStG Rz 4 , m. w. N. ; Urteil des Bundesfinanzhofs - BFH - vom 28. September 1984 VI R 164/80 , BFHE 142 , 377 , BStBl II 1985 , 129 , unter 1. c , m. w. N. , und BFH-Beschluss vom 13. Juli 2011 VI B 20/11 , BFH / NV 2011 , 1863 , Rz 9 , m. w. N. ) .

**False Positives:**

- `Bundesfinanzhofs` — partial — pred is substring of gold: `Urteil des Bundesfinanzhofs - BFH - vom 28. September 1984 VI R 164/80 , BFHE 142 , 377 , BStBl II 1985 , 129 , unter 1. c`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Blümich / K. Heger , § 33b EStG Rz 4`(LIT)
- `Urteil des Bundesfinanzhofs - BFH - vom 28. September 1984 VI R 164/80 , BFHE 142 , 377 , BStBl II 1985 , 129 , unter 1. c`(RS)
- `BFH-Beschluss vom 13. Juli 2011 VI B 20/11 , BFH / NV 2011 , 1863 , Rz 9`(RS)

**Example 4** (doc_id: `58290`) (sent_id: `58290`)


Das FG weiche von dem Beschluss des Bundesfinanzhofs ( BFH ) vom 17. Januar 2002 VI B 114/01 ( BFHE 198 , 1 , BStBl II 2002 , 306 ) ab , in dem der BFH festgestellt habe , dass der Antrag " Ablehnungsbescheid über den Antrag auf Änderung ... " den Gegenstand des Klagebegehrens hinreichend bezeichne .

**False Positives:**

- `Bundesfinanzhofs` — partial — pred is substring of gold: `Beschluss des Bundesfinanzhofs ( BFH ) vom 17. Januar 2002 VI B 114/01 ( BFHE 198 , 1 , BStBl II 2002 , 306 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Beschluss des Bundesfinanzhofs ( BFH ) vom 17. Januar 2002 VI B 114/01 ( BFHE 198 , 1 , BStBl II 2002 , 306 )`(RS)
- `BFH`(ORG)

**Example 5** (doc_id: `59593`) (sent_id: `59593`)


Nach ständiger höchstrichterlicher Rechtsprechung trägt die Finanzverwaltung zudem die Feststellungslast für den Zeitpunkt der Aufgabe eines Verwaltungsaktes zur Post ; eine Beweiserleichterung durch einen Anscheinsbeweis kann sie dabei nicht in Anspruch nehmen ( vgl. Entscheidungen des Bundesfinanzhofs vom 14. Oktober 2003 IX R 68/98 , BFHE 203 , 26 , BStBl II 2003 , 898 , unter II. 1. b bb , und vom 26. Januar 2010 X B 147/09 , BFH / NV 2010 , 1081 , Rz 3 , jeweils mit zahlreichen weiteren Nachweisen ) .

**False Positives:**

- `Bundesfinanzhofs` — partial — pred is substring of gold: `Entscheidungen des Bundesfinanzhofs vom 14. Oktober 2003 IX R 68/98 , BFHE 203 , 26 , BStBl II 2003 , 898 , unter II. 1. b bb`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Entscheidungen des Bundesfinanzhofs vom 14. Oktober 2003 IX R 68/98 , BFHE 203 , 26 , BStBl II 2003 , 898 , unter II. 1. b bb`(RS)
- `vom 26. Januar 2010 X B 147/09 , BFH / NV 2010 , 1081 , Rz 3`(RS)

</details>

---

## `Complex Senate Names`

**F1:** 0.022 | **Precision:** 1.000 | **Recall:** 0.011  

**Format:** `regex`  
**Rule ID:** `90da40d9`  
**Description:**
Matches specific senate/chamber names of the Patent Court including the full context (e.g., '25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts').

**Content:**
```
\b\d+\.\s+Senat\s+\(\s*[A-Za-zäöüß\s-]+\s*\)\s+des\s+Bundespatentgerichts\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.011 | 0.022 | 9 | 9 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 9 | 0 | 722 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `54018`) (sent_id: `54018`)


In der Beschwerdesache betreffend die Marke 30 2010 022 988 hat der 27. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 10. Mai 2017 durch die Vorsitzende Richterin Klante , den Richter Dr. Himmelmann und die Richterin Lachenmayr-Nikolaou beschlossen :

| Predicted | Gold |
|---|---|
| `27. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` | `27. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Klante` (PER)
- `Himmelmann` (PER)
- `Lachenmayr-Nikolaou` (PER)

**Example 1** (doc_id: `54291`) (sent_id: `54291`)


In der Beschwerdesache betreffend die Designanmeldung 40 2016 201 675.4 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` | `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Hacker` (PER)
- `Merzbach` (PER)
- `Meiser` (PER)

**Example 2** (doc_id: `54886`) (sent_id: `54886`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2012 063 820.1 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 5. Dezember 2017 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` | `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Knoll` (PER)
- `Kriener` (PER)
- `Nielsen` (PER)

**Example 3** (doc_id: `55818`) (sent_id: `55818`)


In der Beschwerdesache betreffend die Patentanmeldung 10 2014 003 988.9 hat der 23. Senat ( Technischer Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 8. Mai 2018 unter Mitwirkung des Vorsitzenden Richters Dr. Strößner sowie der Richter Dr. Friedrich , Dr. Zebisch und Dr. Himmelmann beschlossen :

| Predicted | Gold |
|---|---|
| `23. Senat ( Technischer Beschwerdesenat ) des Bundespatentgerichts` | `23. Senat ( Technischer Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Strößner` (PER)
- `Friedrich` (PER)
- `Zebisch` (PER)
- `Himmelmann` (PER)

**Example 4** (doc_id: `56015`) (sent_id: `56015`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2011 068 984.9 hat der 26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 8. Januar 2018 unter Mitwirkung der Vorsitzenden Richterin Kortge sowie der Richter Jacobi und Schödel beschlossen :

| Predicted | Gold |
|---|---|
| `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` | `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Kortge` (PER)
- `Jacobi` (PER)
- `Schödel` (PER)

**Example 5** (doc_id: `59509`) (sent_id: `59509`)


In der Beschwerdesache betreffend die Marke 30 2009 026 804 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 21. September 2017 unter Mitwirkung der Richter Merzbach , Dr. Meiser und Schödel beschlossen :

| Predicted | Gold |
|---|---|
| `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` | `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Merzbach` (PER)
- `Meiser` (PER)
- `Schödel` (PER)

**Example 6** (doc_id: `59628`) (sent_id: `59628`)


In der Beschwerdesache betreffend die Marke 30 2012 041 338 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 15. November 2017 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` | `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Knoll` (PER)
- `Kriener` (PER)
- `Nielsen` (PER)

**Example 7** (doc_id: `59761`) (sent_id: `59761`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2015 031 519.2 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Knoll sowie der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` | `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Knoll` (PER)
- `Kriener` (PER)
- `Nielsen` (PER)

**Example 8** (doc_id: `59948`) (sent_id: `59948`)


In der Beschwerdesache betreffend die international registrierte Marke IR 1 160 635 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 9. November 2017 unter Mitwirkung des Vorsitzenden Richters Professor Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` | `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` |

**Missed by this rule (FN):**

- `Hacker` (PER)
- `Merzbach` (PER)
- `Meiser` (PER)

</details>

---

## `Anonymized Company Patterns`

**F1:** 0.022 | **Precision:** 0.900 | **Recall:** 0.011  

**Format:** `regex`  
**Rule ID:** `5cfc261d`  
**Description:**
Matches anonymized company names with single letters, dots, or ellipsis followed by legal forms.

**Content:**
```
\b(?:[A-Z]\s*\.?\s*\.\.\.\s*GmbH|\.\.\.\s+GmbH|\.\.\.\s+Corp\.|[A-Z]\s*\.?\s*\.\.\.\s+GmbH|Lilly\s*\.\.\.\s*LLC|I\s*\.\.\.\s*Corp\.|E\s*K\s*Co\.|[A-Z]\s+[A-Z]\s*Co\.|[A-Z]\s*\.?\s*\.\.\.\s+AG|V\s+AG|X\s+GmbH|Y\s+GmbH|D\s+P\s+T\s+S\s+GmbH|C\s+GmbH|H\u00c4VG|H\u00c4VG-Rechenzentrum\s+GmbH|H\u00c4VG-Rechenzentrum\s+AG|Haus\u00e4rztliche\s+Vertragsgemeinschaft\s+Aktiengesellschaft|Schleswig-Holsteinische\s+Oberlandesgericht|Finanzgericht\s+N\u00fcrnberg|Reichsversicherungsamt|S\u00e4chsischen\s+Bildungsagentur|Amtsgericht\s+[A-Z]\.|Bundessozialgericht|Bundesamts\s+f\u00fcr\s+das\s+Personalmanagement|Oberlandesgericht\s+D\u00fcsseldorf|Deutsche\s+Industrie-\s+und\s+Handelskammertag|InEK\s+GmbH|Sanktionsausschuss\s+des\s+Sicherheitsrats\s+der\s+Vereinten\s+Nationen|Landesamt\s+f\u00fcr\s+Landwirtschaft\s*,\s*Umwelt\s+und\s+l\u00e4ndliche\s+R\u00e4ume|LLUR)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.900 | 0.011 | 0.022 | 10 | 9 | 1 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 9 | 1 | 724 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `54007`) (sent_id: `54007`)


Das Unterrichtungsschreiben vom 14. November 2005 enthält weder Angaben zum Sitz der D P T S GmbH noch zu deren Anschrift , zum zuständigen Registergericht und zur Registernummer .

| Predicted | Gold |
|---|---|
| `D P T S GmbH` | `D P T S GmbH` |

**Example 1** (doc_id: `54284`) (sent_id: `54284`)


Der Stellung des Sammelantrages beim LLUR komme demgegenüber keine besondere Bedeutung zu .

| Predicted | Gold |
|---|---|
| `LLUR` | `LLUR` |

**Example 2** (doc_id: `55657`) (sent_id: `55657`)


Das Unterrichtungsschreiben vom 14. November 2005 enthält weder Angaben zum Sitz der D P T S GmbH noch zu deren Anschrift , zum zuständigen Registergericht und zur Registernummer .

| Predicted | Gold |
|---|---|
| `D P T S GmbH` | `D P T S GmbH` |

**Example 3** (doc_id: `56299`) (sent_id: `56299`)


Der Kläger hat dem Übergang seines Arbeitsverhältnisses von der Beklagten auf die D P T S GmbH mit Schreiben vom 1. September 2015 nicht wirksam widersprochen .

| Predicted | Gold |
|---|---|
| `D P T S GmbH` | `D P T S GmbH` |

**Example 4** (doc_id: `56463`) (sent_id: `56463`)


" Am 18. 08. 2011 erwarb die Firma H ... e. K. , vertreten durch den Zeugen W ... , zwei Kopiergeräte des Herstellers X ... zum Preis von 17.850 € von der Firma B ... GmbH , vertreten durch den Beschuldigten .

| Predicted | Gold |
|---|---|
| `B ... GmbH` | `B ... GmbH` |

**Missed by this rule (FN):**

- `H ... e. K.` (ORG)
- `W ...` (PER)
- `X ...` (ORG)

**Example 5** (doc_id: `57557`) (sent_id: `57557`)


Die Verfassungsbeschwerde betrifft die Feststellung einer Berufskrankheit nach dem Recht der gesetzlichen Unfallversicherung , wobei der Beschwerde-führer namentlich Verletzungen des Rechts auf rechtliches Gehör geltend macht , weil das Landessozialgericht mehreren Beweisanträgen nicht entsprochen und das Bundessozialgericht dies nicht korrigiert habe .

| Predicted | Gold |
|---|---|
| `Bundessozialgericht` | `Bundessozialgericht` |

**Example 6** (doc_id: `57710`) (sent_id: `57710`)


Die Sache ist zur erneuten Entscheidung an das Oberlandesgericht Düsseldorf zurückzuverweisen ( § 95 Abs. 2 BVerfGG ) .

| Predicted | Gold |
|---|---|
| `Oberlandesgericht Düsseldorf` | `Oberlandesgericht Düsseldorf` |

**Missed by this rule (FN):**

- `§ 95 Abs. 2 BVerfGG` (NRM)

**Example 7** (doc_id: `58142`) (sent_id: `58142`)


Tatsächlich ist dagegen nicht ersichtlich , dass den Versorgungsbehörden , dem Landessozialgericht oder dem Bundessozialgericht eine generelle Vernachlässigung von Grundrechten vorgeworfen werden könnte , sie also die Grundrechte nicht nur im konkreten Fall und mit Blick auf die inzwischen überholte Rechtslage nicht hinreichend beachtet haben könnten .

| Predicted | Gold |
|---|---|
| `Bundessozialgericht` | `Bundessozialgericht` |

**Example 8** (doc_id: `59355`) (sent_id: `59355`)


Es kann vorliegend dahinstehen , ob die neu gegründete D P T S GmbH ihre Tätigkeit überhaupt vor dem 1. Januar 2006 aufgenommen hatte ; jedenfalls wirkt sich auch insoweit aus , dass der Kläger - wie unter Rn. 39 ausgeführt - anders als im Fall einer ordnungsgemäßen Unterrichtung nicht gehalten gewesen wäre , innerhalb der in § 613a Abs. 6 Satz 1 BGB kurzen Frist von einem Monat nach Zugang der Unterrichtung zu entscheiden , ob er sein Widerspruchsrecht ausübt oder nicht , sondern hinreichend Zeit hatte , sich ggf. weitergehend zu erkundigen oder entsprechend beraten zu lassen , wem gegenüber er sein Widerspruchsrecht ausüben konnte .

| Predicted | Gold |
|---|---|
| `D P T S GmbH` | `D P T S GmbH` |

**Missed by this rule (FN):**

- `§ 613a Abs. 6 Satz 1 BGB` (NRM)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `58125`) (sent_id: `58125`)


- als Beschäftigter für folgende Aufgaben von begrenzter Dauer : im Rahmen der Fördermaßnahme der Sächsischen Bildungsagentur D für die sozialpädagogische Betreuung im Berufsvorbereitungsjahr am BSZ Technik und Wirtschaft P , Bewilligungsbescheid für das Schuljahr 2013/2014

**False Positives:**

- `Sächsischen Bildungsagentur` — partial — pred is substring of gold: `Sächsischen Bildungsagentur D`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Sächsischen Bildungsagentur D`(ORG)
- `BSZ Technik und Wirtschaft P`(ORG)

</details>

---

## `Anonymized Single Letter Companies (Fixed)`

**F1:** 0.022 | **Precision:** 0.600 | **Recall:** 0.011  

**Format:** `regex`  
**Rule ID:** `c4d10f6b`  
**Description:**
Matches anonymized company names like 'w GmbH', 'H. AG', 'A-AG', 'M. GmbH' using single letters or initials, specifically targeting court and organization contexts.

**Content:**
```
\b([A-Z]\s*\.?\s*[A-Z]?\s*\.?\s*[A-Z]?|w|H|A|M|B|F|S|T|K|E|D|C|G|L|N|P|R|U|V|X|Y|Z)\s+(?:GmbH|AG|KG|GbR|Fonds|V\.|B\.\s*V\.|Klinik|Schulzentrum|Finanzamt|Landratsamt|Berufsschulzentrum|Jobcenter|Botschaft|Kammer|Senat|Stelle|Amt|Verband|Zweckverband|Firma|Bank|Verlag|GmbH\s*&\s+Co\.\s*KG|Konzerns?|AG\s*&\s+Co\.\s*KG|S\.r\.l\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.600 | 0.011 | 0.022 | 15 | 9 | 6 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 9 | 6 | 766 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53630`) (sent_id: `53630`)


Die Verträge seien derart angelegt gewesen , dass die KG am Ende der Laufzeit ihr " Andienungsrecht " ausübe und die P GmbH die Leasingobjekte zurückerwerben müsse .

| Predicted | Gold |
|---|---|
| `P GmbH` | `P GmbH` |

**Example 1** (doc_id: `53833`) (sent_id: `53833`)


Zuschuss zum Anpassungsgeld bei der RAG AG - Berücksichtigung der Grubenwehrzulage bei der Bemessung des Zuschusses

| Predicted | Gold |
|---|---|
| `RAG AG` | `RAG AG` |

**Example 2** (doc_id: `53844`) (sent_id: `53844`)


Der im Juli 1950 geborene Kläger war seit dem 1. März 1971 bei einer Rechtsvorgängerin der Beklagten , der H AG ( im Folgenden H AG alt ) als Arbeitnehmer tätig .

| Predicted | Gold |
|---|---|
| `H AG` | `H AG` |

**Missed by this rule (FN):**

- `H AG alt` (ORG)

**Example 3** (doc_id: `54044`) (sent_id: `54044`)


Die S AG schloss mit ihrem Gesamtbetriebsrat am 16. Oktober 2003 die Gesamtbetriebsvereinbarung zur Modernisierung und Neuordnung der betrieblichen Altersversorgung für die Mitarbeiter im ÜT-Kreis ( im Folgenden GBV BSAV ) .

| Predicted | Gold |
|---|---|
| `S AG` | `S AG` |

**Missed by this rule (FN):**

- `Gesamtbetriebsvereinbarung zur Modernisierung und Neuordnung der betrieblichen Altersversorgung für die Mitarbeiter im ÜT-Kreis` (REG)
- `GBV BSAV` (REG)

**Example 4** (doc_id: `54589`) (sent_id: `54589`)


Denn die Leasingobjekte seien nach § 39 Abs. 2 Nr. 1 Satz 1 der Abgabenordnung ( AO ) nicht der KG als der zivilrechtlichen , sondern der P GmbH als der wirtschaftlichen Eigentümerin steuerrechtlich zuzurechnen .

| Predicted | Gold |
|---|---|
| `P GmbH` | `P GmbH` |

**Missed by this rule (FN):**

- `§ 39 Abs. 2 Nr. 1 Satz 1 der Abgabenordnung` (NRM)
- `AO` (NRM)

**Example 5** (doc_id: `56880`) (sent_id: `56880`)


Schließlich wurde am 14. Dezember 2005 zwischen dem Kreis als Käufer einerseits und der I GmbH und der L GmbH als Verkäufer andererseits ein Ankaufsrechtsvertrag über Geschäftsanteile notariell beurkundet .

| Predicted | Gold |
|---|---|
| `I GmbH` | `I GmbH` |
| `L GmbH` | `L GmbH` |

**Example 6** (doc_id: `59073`) (sent_id: `59073`)


Danach hat der Kreis unter bestimmten Voraussetzungen das Recht , die Geschäftsanteile der Verkäufer an der Z GmbH zum 30. November 2017 oder 30. November 2023 zu kaufen .

| Predicted | Gold |
|---|---|
| `Z GmbH` | `Z GmbH` |

**Example 7** (doc_id: `59794`) (sent_id: `59794`)


Der Generalbundesanwalt hat auf die zunächst nicht näher begründeten Sachrügen zu den durch die Taten entstandenen Vermögensnachteilen nach Erwägungen zu Fall 1 der Urteilsgründe ( Fall " A. " ) zu Fall 2 der Urteilsgründe ( Fall " B. “ ) lediglich ausgeführt , dass auch gegen die Bestimmung des Vermögensnachteils im Zusammenhang mit der Beteiligung an der Grundstücksgesellschaft F. GbR aus revisionsrechtlicher Sicht nichts zu erinnern sei .

| Predicted | Gold |
|---|---|
| `F. GbR` | `F. GbR` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `54043`) (sent_id: `54043`)


Nach den zwingenden gesetzlichen Vorschriften des § 613 a Abs. 1 des BGB gehen die Arbeitsverhältnisse der in den Kreiskrankenhäusern M und R Beschäftigten , sofern sie nicht fristgemäß widersprechen , auf die Ekliniken M-R GmbH & Co KG zum Stichtag über , ohne dass es hierfür einer gesonderten Vereinbarung bedarf .

**False Positives:**

- `R GmbH` — partial — pred is substring of gold: `Ekliniken M-R GmbH & Co KG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 613 a Abs. 1 des BGB`(NRM)
- `Kreiskrankenhäusern M und R`(ORG)
- `Ekliniken M-R GmbH & Co KG`(ORG)

**Example 1** (doc_id: `54391`) (sent_id: `54391`)


Mit Wirkung zum 1. Juli 2005 schlossen die H AG neu , die später als V H AG firmierte , und der bei ihr gebildete Betriebsrat die Betriebsvereinbarung Nr. 2005.03 ( im Folgenden BV 2005.03 ) .

**False Positives:**

- `H AG` — partial — pred is substring of gold: `H AG neu`
- `V H AG` — partial — pred is substring of gold: `V H AG firmierte`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `H AG neu`(ORG)
- `V H AG firmierte`(ORG)
- `Betriebsvereinbarung Nr. 2005.03`(REG)
- `BV 2005.03`(REG)

**Example 2** (doc_id: `54591`) (sent_id: `54591`)


( 1 ) Die Ekliniken M-R GmbH & Co KG tritt gemäß § 613 a BGB in die Rechte und Pflichten aus den zum Stichtag

**False Positives:**

- `R GmbH` — partial — pred is substring of gold: `Ekliniken M-R GmbH & Co KG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Ekliniken M-R GmbH & Co KG`(ORG)
- `§ 613 a BGB`(NRM)

**Example 3** (doc_id: `59069`) (sent_id: `59069`)


b ) Aus der Formulierung „ die Arbeitsverhältnisse der Arbeitnehmerinnen und Arbeitnehmer , die die Ekliniken M-R GmbH & Co KG vom Landkreis M übernommen hat “ folgt , dass die Regelung entgegen der von der Klägerin in der Revision vertretenen Ansicht nicht eine Pflicht des Landkreises zur dynamischen Anwendung begründen sollte , die sodann mit dem Betriebsübergang gemäß § 613a Abs. 1 Satz 2 BGB auf die Beklagte hätte übergehen können .

**False Positives:**

- `R GmbH` — partial — pred is substring of gold: `Ekliniken M-R GmbH & Co KG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Ekliniken M-R GmbH & Co KG`(ORG)
- `M`(LOC)
- `§ 613a Abs. 1 Satz 2 BGB`(NRM)

**Example 4** (doc_id: `59160`) (sent_id: `59160`)


Das Arbeitsverhältnis des Klägers ging aufgrund eines Betriebsübergangs zum 1. Oktober 2005 auf die B GmbH & Co. OHG ( im Folgenden Insolvenzschuldnerin ) über und endete mit Ablauf des 30. November 2006 .

**False Positives:**

- `B GmbH` — partial — pred is substring of gold: `B GmbH & Co. OHG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `B GmbH & Co. OHG`(ORG)

</details>

---

## `Bundespatentgericht`

**F1:** 0.022 | **Precision:** 0.346 | **Recall:** 0.011  

**Format:** `regex`  
**Rule ID:** `ad145470`  
**Description:**
Matches 'Bundespatentgericht' and its genitive form.

**Content:**
```
\b(Bundespatentgericht|Bundespatentgerichts|BPatG)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.346 | 0.011 | 0.022 | 26 | 9 | 17 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 9 | 17 | 780 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53862`) (sent_id: `53862`)


Patentansprüche 1 bis 13 vom 24. November 2017 , beim BPatG als 6. Hilfsantrag per Fax eingegangen am 27. November 2017

| Predicted | Gold |
|---|---|
| `BPatG` | `BPatG` |

**Example 1** (doc_id: `54129`) (sent_id: `54129`)


Für die ausstehende Entscheidung bleibt auch die Zuständigkeit des Bundespatentgerichts bestehen .

| Predicted | Gold |
|---|---|
| `Bundespatentgerichts` | `Bundespatentgerichts` |

**Example 2** (doc_id: `54754`) (sent_id: `54754`)


Darin unterscheidet sich der vorliegende Fall grundlegend von demjenigen , der dem Beschluss des Bundespatentgerichts vom 13. November 2014 zu Grunde lag .

| Predicted | Gold |
|---|---|
| `Bundespatentgerichts` | `Bundespatentgerichts` |

**Example 3** (doc_id: `55948`) (sent_id: `55948`)


Insofern gibt es auch im Rahmen von unbestimmten Rechtbegriffen keine Selbstbindung der Markenstellen des Deutschen Patent- und Markenamts und erst recht keine irgendwie geartete Bindung für das Bundespatentgericht .

| Predicted | Gold |
|---|---|
| `Bundespatentgericht` | `Bundespatentgericht` |

**Missed by this rule (FN):**

- `Deutschen Patent- und Markenamts` (ORG)

**Example 4** (doc_id: `56133`) (sent_id: `56133`)


Insoweit besteht in der Rechtsprechung des Bundespatentgerichts zwar weitgehend Übereinstimmung dahingehend , dass die Anmeldung eines Schutzrechts nicht schon allein deswegen mutwillig erscheint , weil der Anmelder – auch unter Inanspruchnahme von Verfahrenskostenhilfe – zahlreiche andere Anmeldungen ohne wirtschaftlichen Erfolg getätigt hat ( vgl. BPatGE 45 , 49 , 51 - Massenanmeldung ; BPatGE 42 , 178 , 179 f. ; BPatGE , 224 , 226 , jeweils m. w. N. ) .

| Predicted | Gold |
|---|---|
| `Bundespatentgerichts` | `Bundespatentgerichts` |

**Missed by this rule (FN):**

- `BPatGE 45 , 49 , 51 - Massenanmeldung` (RS)
- `BPatGE 42 , 178 , 179 f.` (RS)
- `BPatGE , 224 , 226` (RS)

**Example 5** (doc_id: `56648`) (sent_id: `56648`)


Im Übrigen könne ein Anspruch auf Zahlung von Lizenzentgelt , selbst wenn er bestünde , ebenso wie ein Anspruch auf Rechnungslegung , nicht im Verfahren zur Erteilung einer Zwangslizenz vor dem Bundespatentgericht geltend gemacht werden .

| Predicted | Gold |
|---|---|
| `Bundespatentgericht` | `Bundespatentgericht` |

**Example 6** (doc_id: `57241`) (sent_id: `57241`)


Ausgehend von diesen Faktoren drängt sich bei der angemeldeten Bezeichnung das vorstehend dargestellte rein sachbeschreibende Verständnis auf , was demzufolge einem betriebskennzeichnenden Verständnis entgegensteht ( vgl. dazu auch BPatG , Beschluss vom 3. Juni 2016 , Aktenzeichen 27 W ( pat ) 1/16 – Lebe Balance ; die Entscheidung ist über die Homepage des Bundespatentgerichts öffentlich zugänglich ) .

| Predicted | Gold |
|---|---|
| `Bundespatentgerichts` | `Bundespatentgerichts` |

**Missed by this rule (FN):**

- `BPatG , Beschluss vom 3. Juni 2016 , Aktenzeichen 27 W ( pat ) 1/16 – Lebe Balance` (RS)

**Example 7** (doc_id: `59087`) (sent_id: `59087`)


Nach § 73 Abs. 3 Satz 2 PatG hat das Deutsche Patent- und Markenamt dann , wenn es der Beschwerde nicht abhilft , sie dem Bundespatentgericht vorzulegen .

| Predicted | Gold |
|---|---|
| `Bundespatentgericht` | `Bundespatentgericht` |

**Missed by this rule (FN):**

- `§ 73 Abs. 3 Satz 2 PatG` (NRM)
- `Deutsche Patent- und Markenamt` (ORG)

**Example 8** (doc_id: `59981`) (sent_id: `59981`)


Der Anmelder verweist des Weiteren auf zahlreiche Entscheidungen des BGH und des Bundespatentgerichts , in denen vergleichbare Buchstabenkürzel für schutzfähig erachtet worden seien ( z.B. ISET / ISETsolar ( BGH I ZB 2/14 ) , „ ume “ ( 27 W ( pat ) 539/14 ) oder EHD , RSV , bb-nrw , CTL , CJD , RDB , UPW , TCP ) .

| Predicted | Gold |
|---|---|
| `Bundespatentgerichts` | `Bundespatentgerichts` |

**Missed by this rule (FN):**

- `BGH` (ORG)
- `ISET / ISETsolar ( BGH I ZB 2/14 )` (RS)
- `„ ume “ ( 27 W ( pat ) 539/14 )` (RS)
- `EHD` (ORG)
- `RSV` (ORG)
- `bb-nrw` (ORG)
- `CTL` (ORG)
- `CJD` (ORG)
- `RDB` (ORG)
- `UPW` (ORG)
- `TCP` (ORG)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53485`) (sent_id: `53485`)


Soweit die Elemente eines Bildzeichens nur die typischen Merkmale der in Rede stehenden Waren und Dienstleistungen darstellen oder sich in einfachen dekorativen Gestaltungsmitteln erschöpfen , an die sich der Verkehr etwa durch häufige Verwendung gewöhnt hat , wird diesem Zeichen im Allgemeinen wegen seines bloß beschreibenden Inhalts ebenfalls die konkrete Eignung fehlen , die mit ihm gekennzeichneten Waren oder Dienstleistungen von denjenigen anderer Herkunft zu unterscheiden ( vgl. BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband ; GRUR 2005 , 257 , 258 – Bürogebäude ; GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel ; GRUR 2001 , 239f. – Zahnpastastrang ; GRUR 2001 , 734 , 735 – Westie-Kopf ; BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband`(RS)
- `GRUR 2005 , 257 , 258 – Bürogebäude`(RS)
- `GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel`(RS)
- `GRUR 2001 , 239f. – Zahnpastastrang`(RS)
- `GRUR 2001 , 734 , 735 – Westie-Kopf`(RS)
- `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`(RS)

**Example 1** (doc_id: `53854`) (sent_id: `53854`)


Die insoweit verfrüht erhobene Einrede entfaltet auch mit dem Ablauf der maßgeblichen Frist ( am 12. Juni 2014 ) nicht die Rechtswirkung einer zulässigen Einrede ( BPatG GRUR 2005 , 773 , 775 - Blue Bull / RED BULL ; Ströbele / Hacker , MarkenG , 11. Aufl. , § 43 Rn. 29 m. w. N. ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG GRUR 2005 , 773 , 775 - Blue Bull / RED BULL`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG GRUR 2005 , 773 , 775 - Blue Bull / RED BULL`(RS)
- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 43 Rn. 29`(LIT)

**Example 2** (doc_id: `53881`) (sent_id: `53881`)


Ob dies der Fall ist , richtet sich nach den Umständen des Einzelfalls , bei denen darauf abzustellen ist , wie das Hoheitszeichen im Rahmen der Designgestaltung konkret verwendet ist ( vgl. BPatG GRUR 2002 , 337 - Schlüsselanhänger ; Eichmann / von Falckenstein / Kühne , a. a. O. , § 3 Rdn. 26 ff. ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG GRUR 2002 , 337 - Schlüsselanhänger`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG GRUR 2002 , 337 - Schlüsselanhänger`(RS)
- `Eichmann / von Falckenstein / Kühne , a. a. O. , § 3 Rdn. 26 ff.`(LIT)

**Example 3** (doc_id: `53963`) (sent_id: `53963`)


Der Gesamteindruck aber kann auch bei Übernahme der geschützten „ Schnittmenge “ durch Hinzufügung weiterer Merkmale im Einzelfall erheblich verändert werden ( vgl. OLG Köln , 6 U 128/09 v. 8. Januar 2010 ; veröffentlicht in juris ; Klawitter in : Festschrift 50 Jahre Bundespatentgericht , S. 1071 , 1079 ) .

**False Positives:**

- `Bundespatentgericht` — partial — pred is substring of gold: `Klawitter in : Festschrift 50 Jahre Bundespatentgericht , S. 1071 , 1079`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `OLG Köln , 6 U 128/09 v. 8. Januar 2010 ; veröffentlicht in juris`(RS)
- `Klawitter in : Festschrift 50 Jahre Bundespatentgericht , S. 1071 , 1079`(LIT)

**Example 4** (doc_id: `55119`) (sent_id: `55119`)


Besteht dieses nur aus der Darstellung des Gegenstands , auf den sich die Dienstleistungen unmittelbar beziehen , stellt es nur typische Merkmale der in Rede stehenden Dienstleistungen dar oder erschöpft sich die bildliche Darstellung in einfachen dekorativen Gestaltungsmitteln , an die der Verkehr sich etwa durch häufige Verwendung gewöhnt hat , wird diesem Zeichen im allgemeinen wegen seines bloß beschreibenden Inhalts die konkrete Eignung fehlen , die mit ihm gekennzeichneten Waren oder Dienstleistungen von denjenigen anderer Herkunft zu unterscheiden ( vgl. BGH , GRUR 2005 , 257 , 258 – Bürogebäude ; GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel ; GRUR 2001 , 239 f. – Zahnpastastrang ; GRUR 2001 , 734 , 735 – Westie-Kopf ; BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , GRUR 2005 , 257 , 258 – Bürogebäude`(RS)
- `GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel`(RS)
- `GRUR 2001 , 239 f. – Zahnpastastrang`(RS)
- `GRUR 2001 , 734 , 735 – Westie-Kopf`(RS)
- `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`(RS)

**Example 5** (doc_id: `55186`) (sent_id: `55186`)


Eine Ähnlichkeit zwischen Handelsdienstleistungen , insbesondere hiervon erfassten Einzelhandelsdienstleistungen , und den auf sie bezogenen Waren ist anzunehmen , wenn die Dienstleistungen sich auf die entsprechenden Waren beziehen und die angesprochenen Verkehrskreise aufgrund dieses Verhältnisses annehmen , die Waren und Dienstleistungen stammten aus denselben Unternehmen ( vgl. BGH GRUR 2014 , 378 , Rdnr. 39 – Otto CAP ; BPatG GRUR-RR 2013 , 430 , 432 – Konzume / Konsum ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG GRUR-RR 2013 , 430 , 432 – Konzume / Konsum`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH GRUR 2014 , 378 , Rdnr. 39 – Otto CAP`(RS)
- `BPatG GRUR-RR 2013 , 430 , 432 – Konzume / Konsum`(RS)

**Example 6** (doc_id: `56236`) (sent_id: `56236`)


Zwischen der technischen Dienstleistung und der Contentvermittlung besteht ein so enger Bezug , dass das entsprechende Verkehrsverständnis zwischen Technik und Inhalt insoweit nicht mehr trennt ( vgl. BGH GRUR 2014 , 1204 Rn. 22 – TOOOR ! ; BPatG , Beschluss vom 11. 05. 2015 , 26 W ( pat ) 72/14 – Shopping Compass ; Beschluss vom 22. 01. 2015 , 29 W ( pat ) 525/13 – The European ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG , Beschluss vom 11. 05. 2015 , 26 W ( pat ) 72/14 – Shopping Compass`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH GRUR 2014 , 1204 Rn. 22 – TOOOR !`(RS)
- `BPatG , Beschluss vom 11. 05. 2015 , 26 W ( pat ) 72/14 – Shopping Compass`(RS)
- `Beschluss vom 22. 01. 2015 , 29 W ( pat ) 525/13 – The European`(RS)

**Example 7** (doc_id: `56412`) (sent_id: `56412`)


Soweit die Anmelderin in Klasse 2 ferner „ Naturharze im Rohzustand “ beansprucht , stellen diese im Bereich von Lacken und ( Öl- ) Farben einen üblichen Inhaltsstoff dar , der auch als Zusatz im Malereibedarf in Betracht kommt ( vgl. BPatG PAVIS PROMA , Beschluss vom 23. April 2013 , 25 W ( pat ) 62/12 - Macchiato ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG PAVIS PROMA , Beschluss vom 23. April 2013 , 25 W ( pat ) 62/12 - Macchiato`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG PAVIS PROMA , Beschluss vom 23. April 2013 , 25 W ( pat ) 62/12 - Macchiato`(RS)

**Example 8** (doc_id: `57241`) (sent_id: `57241`)


Ausgehend von diesen Faktoren drängt sich bei der angemeldeten Bezeichnung das vorstehend dargestellte rein sachbeschreibende Verständnis auf , was demzufolge einem betriebskennzeichnenden Verständnis entgegensteht ( vgl. dazu auch BPatG , Beschluss vom 3. Juni 2016 , Aktenzeichen 27 W ( pat ) 1/16 – Lebe Balance ; die Entscheidung ist über die Homepage des Bundespatentgerichts öffentlich zugänglich ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG , Beschluss vom 3. Juni 2016 , Aktenzeichen 27 W ( pat ) 1/16 – Lebe Balance`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG , Beschluss vom 3. Juni 2016 , Aktenzeichen 27 W ( pat ) 1/16 – Lebe Balance`(RS)
- `Bundespatentgerichts`(ORG)

**Example 9** (doc_id: `57632`) (sent_id: `57632`)


Von daher sind unter den „ Angaben zum Verwendungszweck , der die Kosten umfasst “ diejenigen Informationen zu verstehen , die das Patentamt in die Lage versetzen , die Höhe der zu zahlenden Gebühr festzustellen , diese einem konkreten Verfahren zuzuordnen und auf Basis eines entsprechenden SEPA-Basislastschriftmandats einzuziehen ( vgl. BPatG Mitt. 2016 , 192 , 195 , juris Tz. 21 - babygro ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG Mitt. 2016 , 192 , 195 , juris Tz. 21 - babygro`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG Mitt. 2016 , 192 , 195 , juris Tz. 21 - babygro`(RS)

**Example 10** (doc_id: `57712`) (sent_id: `57712`)


Die Markenbestandteile werden in Übereinstimmung mit ihrem Sinngehalt verwendet und bilden auch in der Gesamtheit keinen neuen , über die bloße Kombination hinausgehenden Begriff ( BPatG , a. a. O. , 30 W ( pat ) 530/13 - WohlFühlFarben ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG , a. a. O. , 30 W ( pat ) 530/13 - WohlFühlFarben`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG , a. a. O. , 30 W ( pat ) 530/13 - WohlFühlFarben`(RS)

**Example 11** (doc_id: `57805`) (sent_id: `57805`)


Einem Zeichen , das den Namen einer berühmten Persönlichkeit aufnimmt , fehlt nur dann die erforderliche Unterscheidungskraft , wenn die angesprochenen Verkehrskreise in dem Namen lediglich eine sachbezogene oder werbewirksame Aussage sehen ( vgl. BPatG GRUR 2008 , 518 , 521 - Karl May ; GRUR 2014 , 79 , 80 ff. - Mark Twain ; Beschluss vom 1. Juni 2017 , 25 W ( pat ) 4/17 - einstein concept / Einstein ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG GRUR 2008 , 518 , 521 - Karl May`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG GRUR 2008 , 518 , 521 - Karl May`(RS)
- `GRUR 2014 , 79 , 80 ff. - Mark Twain`(RS)
- `Beschluss vom 1. Juni 2017 , 25 W ( pat ) 4/17 - einstein concept / Einstein`(RS)

**Example 12** (doc_id: `58268`) (sent_id: `58268`)


Ferner ist es als bloßes Gestaltungsmittel , z.B. als sog. „ Eyecatcher “ werbeüblich ( vgl. BPatG , Beschluss vom 27. 01. 2009 , 27 W ( pat ) 46/09 – LIVE ! SPEAKER ; Beschluss vom 13. 09. 2006 , 29 W ( pat ) 68/04 – REZEPTE pur EINFACH ! KOCHEN ) wie auch als Ersetzung des Buchstaben „ I / i “ ( vgl. z.B. Werbeaussage „ W ! R S ! ND DABE ! “ unter www.bw-stiftung.de) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG , Beschluss vom 27. 01. 2009 , 27 W ( pat ) 46/09 – LIVE ! SPEAKER`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG , Beschluss vom 27. 01. 2009 , 27 W ( pat ) 46/09 – LIVE ! SPEAKER`(RS)
- `Beschluss vom 13. 09. 2006 , 29 W ( pat ) 68/04 – REZEPTE pur EINFACH ! KOCHEN`(RS)

**Example 13** (doc_id: `58693`) (sent_id: `58693`)


1.7.3 Im Hinblick auf die Nacharbeitung D21A hat die Patentinhaberin geltend gemacht , diese würde die nach dem BGH-Urteil „ Metazachlor “ ( BGH Urteil vom 15. März 2011 – X ZR 58/08 – , juris ) und dem Urteil 3 Ni 5/15 des BPatG ( BPatG Urteil vom 11. Oktober 2016 – 3 Ni 5/15 ( EP ) – , juris ) aufgestellten strengen Maßstäbe nicht erfüllen , die bei der Feststellung einer implizierten Offenbarung einer Druckschrift des Standes der Technik durch Nacharbeitung angelegt werden müssten .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `Urteil 3 Ni 5/15 des BPatG ( BPatG Urteil vom 11. Oktober 2016 – 3 Ni 5/15 ( EP ) – , juris )`
- `BPatG` — partial — pred is substring of gold: `Urteil 3 Ni 5/15 des BPatG ( BPatG Urteil vom 11. Oktober 2016 – 3 Ni 5/15 ( EP ) – , juris )`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH-Urteil „ Metazachlor “ ( BGH Urteil vom 15. März 2011 – X ZR 58/08 – , juris )`(RS)
- `Urteil 3 Ni 5/15 des BPatG ( BPatG Urteil vom 11. Oktober 2016 – 3 Ni 5/15 ( EP ) – , juris )`(RS)

**Example 14** (doc_id: `59283`) (sent_id: `59283`)


Hergeleitet aus dem Bereich der Farbtherapie / Farbpsychologie wird Farben / Farbtönen eine Wirkung auf die menschliche Psyche und den Organismus zugeschrieben ( vgl. BPatG , a. a. O. , 30 W ( pat ) 530/13 – WohlFühlFarben ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG , a. a. O. , 30 W ( pat ) 530/13 – WohlFühlFarben`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG , a. a. O. , 30 W ( pat ) 530/13 – WohlFühlFarben`(RS)

**Example 15** (doc_id: `59751`) (sent_id: `59751`)


Im Hinblick auf die massenhaft beim Patentamt eingehenden und zu bearbeitenden Zahlungen sowie aus Gründen der Rechtssicherheit ist zu beachten , dass jede Gebührenentrichtung beim Patentamt so klar und vollständig sein muss , dass die verfahrens- und betragsmäßige Erfassung und Zuordnung ohne verzögernde Ermittlungen gewährleistet und der Geldbetrag zu dem in § 2 PatKostZV bestimmten Zahlungstag zu einem konkreten Vorgang sicher vereinnahmt werden kann ( vgl. BPatGE 48 , 163 , 167 - Unbezifferter Abbuchungsauftrag ; BPatG Mitt. 2016 , 192 , 195 - babygro ) .

**False Positives:**

- `BPatG` — similar text (different position): `BPatGE 48 , 163 , 167 - Unbezifferter Abbuchungsauftrag`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 2 PatKostZV`(NRM)
- `BPatGE 48 , 163 , 167 - Unbezifferter Abbuchungsauftrag`(RS)
- `BPatG Mitt. 2016 , 192 , 195 - babygro`(RS)

</details>

---

## `Specific Court Names with Location (Fixed)`

**F1:** 0.020 | **Precision:** 0.092 | **Recall:** 0.011  

**Format:** `regex`  
**Rule ID:** `42e5b2a1`  
**Description:**
Matches court names with location, ensuring the court type is present and not just the location, handling multi-word locations like 'Frankfurt am Main' and hyphenated locations like 'Niedersachsen-Bremen'.

**Content:**
```
\b(?:Amtsgericht|Landgericht|Verwaltungsgericht|Finanzgericht|Sozialgericht|Arbeitsgericht|Oberlandesgericht|Bundesverwaltungsgericht|Bundesfinanzhof|Bundesgerichtshof|Bundessozialgericht|Bundesarbeitsgericht|Bundespatentgericht|BVerwG|BGH|BSG|BAG|BVerfG|EuGH|EGMR|DPMA|BaFin|SG|FG|LAG|LSG|ArbG|Landessozialgericht|Landesarbeitsgericht|Landesverwaltungsgericht|Oberverwaltungsgericht|Verwaltungsgerichtshof|Schleswig-Holsteinische\s+Oberverwaltungsgericht|Truppendienstgericht|Anwaltsgerichtshof)\s+(?:[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)*|[A-Z]\.|[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+-[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+|<\s*[A-Z]{2,3}\s*>\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)*|Frankfurt\s+am\s+Main|Berlin-Brandenburg|Niedersachsen-Bremen|Baden-W\u00fcrttemberg|Nordrhein-Westfalen|Rheinland-Pfalz|Schleswig-Holstein)\b(?!\s+(?:Prozesskostenhilfe|Beschwerde|Verfahren|Urteil|Beschluss|Sache|Rechtsprechung|in|auf|von|mit|bei|nach|f\u00fcr|zum|zur|des|der|die|den|Senat|Nr\.|\.)|\s+(?:Prozesskostenhilfe|Beschwerde|Verfahren|Urteil|Beschluss|Sache|Rechtsprechung|in|auf|von|mit|bei|nach|f\u00fcr|zum|zur|des|der|die|den|Senat|Nr\.))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.092 | 0.011 | 0.020 | 98 | 9 | 89 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 9 | 89 | 780 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53602`) (sent_id: `53602`)


Am 21. September 2017 beschloss das Landgericht Memmingen , der weiteren Beschwerde vom 15. September 2017 nicht abzuhelfen .

| Predicted | Gold |
|---|---|
| `Landgericht Memmingen` | `Landgericht Memmingen` |

**Example 1** (doc_id: `54373`) (sent_id: `54373`)


Das Landgericht Düsseldorf begründete diese Anordnung damit , dass die Disposition zu " solchen Taten " tief im Beschwerdeführer verwurzelt sei .

| Predicted | Gold |
|---|---|
| `Landgericht Düsseldorf` | `Landgericht Düsseldorf` |

**Example 2** (doc_id: `54861`) (sent_id: `54861`)


Das Arbeitsgericht Zwickau verurteilte die Beklagte am 22. April 2015 ( - 9 Ca 146/15 - ) , das abgebrochene Stellenbesetzungsverfahren 01/2014 fortzuführen und über die Bewerbung des Klägers erneut zu entscheiden .

| Predicted | Gold |
|---|---|
| `Arbeitsgericht Zwickau` | `Arbeitsgericht Zwickau` |

**Missed by this rule (FN):**

- `22. April 2015 ( - 9 Ca 146/15 - )` (RS)

**Example 3** (doc_id: `55283`) (sent_id: `55283`)


Die Klägerin beantragt , das Urteil des Sächsischen LSG vom 19. Mai 2016 aufzuheben und die Berufung der Beklagten gegen das Urteil des SG Chemnitz vom 9. Oktober 2014 zurückzuweisen .

| Predicted | Gold |
|---|---|
| `SG Chemnitz` | `SG Chemnitz` |

**Missed by this rule (FN):**

- `Sächsischen LSG` (ORG)

**Example 4** (doc_id: `56408`) (sent_id: `56408`)


Verwaltungs- , Widerspruchs- und erstinstanzliches Verfahren waren erfolglos ( Bescheid vom 2. 4. 2014 , Widerspruchsbescheid vom 18. 7. 2014 , Gerichtsbescheid des SG Karlsruhe vom 12. 8. 2015 ) .

| Predicted | Gold |
|---|---|
| `SG Karlsruhe` | `SG Karlsruhe` |

**Example 5** (doc_id: `56519`) (sent_id: `56519`)


Die Klägerin beantragt , das Urteil des Schleswig-Holsteinischen LSG vom 15. 11. 2016 aufzuheben und die Berufung der Beklagten gegen das Urteil des SG Kiel vom 29. 1. 2014 mit der Maßgabe zurückzuweisen , dass die Beklagte bei der erneuten Bescheidung die Rechtsauffassung des Senats zu beachten hat .

| Predicted | Gold |
|---|---|
| `SG Kiel` | `SG Kiel` |

**Missed by this rule (FN):**

- `Schleswig-Holsteinischen LSG` (ORG)

**Example 6** (doc_id: `57841`) (sent_id: `57841`)


2. Das Amtsgericht Dieburg gab der Klage mit Urteil vom 7. Dezember 2012 statt , erklärte die Zwangsvollstreckung aus dem Vollstreckungsbescheid insgesamt für unzulässig und verurteilte den Beklagten , die vollstreckbare Ausfertigung an den Beschwerdeführer herauszugeben ; alle Forderungen des Beklagten gegen den Beschwerdeführer seien getilgt .

| Predicted | Gold |
|---|---|
| `Amtsgericht Dieburg` | `Amtsgericht Dieburg` |

**Example 7** (doc_id: `59802`) (sent_id: `59802`)


Das SG Karlsruhe hat mit Urteil vom 25. 3. 2015 die Klage mit der Begründung abgewiesen , der Kläger übe im Wesentlichen die Tätigkeit eines Pharmareferenten und damit keine für die Berufsgruppe der Tierärzte spezifische Tätigkeit aus .

| Predicted | Gold |
|---|---|
| `SG Karlsruhe` | `SG Karlsruhe` |

**Example 8** (doc_id: `59867`) (sent_id: `59867`)


Das Landgericht Memmingen verwarf die Beschwerde vom 30. August 2017 mit Beschluss vom 11. September 2017 als unbegründet .

| Predicted | Gold |
|---|---|
| `Landgericht Memmingen` | `Landgericht Memmingen` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53508`) (sent_id: `53508`)


Beruhen vielmehr die Differenzen zwischen den Auffassungen von Sachverständigen darauf , dass diese von verschiedenen tatsächlichen Annahmen ausgehen , dann muss der Tatrichter , ggf nach weiterer Aufklärung , die für seine Überzeugungsbildung maßgebenden Tatsachen feststellen oder begründen , weshalb und zu wessen Lasten sie beweislos geblieben sind ( vgl BGH Urteil vom 23. 9. 1986 - VI ZR 261/85 ) .

**False Positives:**

- `BGH Urteil` — partial — pred is substring of gold: `BGH Urteil vom 23. 9. 1986 - VI ZR 261/85`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH Urteil vom 23. 9. 1986 - VI ZR 261/85`(RS)

**Example 1** (doc_id: `53570`) (sent_id: `53570`)


Sie beruft sich für ihr Zulassungsbegehren ausschließlich auf einen Verfahrensmangel ( § 160 Abs 2 Nr 3 SGG ) , weil das LSG Beweisanträgen ohne hinreichenden Grund nicht gefolgt sei .

**False Positives:**

- `LSG Beweisanträgen` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 160 Abs 2 Nr 3 SGG`(NRM)

**Example 2** (doc_id: `53582`) (sent_id: `53582`)


3. Wird einem Beteiligten das rechtliche Gehör dadurch versagt , dass es ihm nicht ermöglicht wird , an der mündlichen Verhandlung teilzunehmen , so ist davon auszugehen , dass dies für eine aufgrund dieser Verhandlung ergangenen Entscheidung ursächlich geworden ist ( vgl BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7 ) .

**False Positives:**

- `BSG Beschluss` — partial — pred is substring of gold: `BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7`(RS)

**Example 3** (doc_id: `53822`) (sent_id: `53822`)


Dies folgt aus § 73b Abs 5 S 4 SGB V , der Abweichungen von den Vorschriften des Vierten Kapitels und damit auch von dem in § 71 Abs 1 S 1 SGB V verankerten Grundsatz der Beitragssatzstabilität zulässt ( BSG Urteil vom 25. 3. 2015 - B 6 KA 9/14 R - BSGE 118 , 164 = SozR 4 - 2500 § 73b Nr 1 , RdNr 72 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 25. 3. 2015 - B 6 KA 9/14 R - BSGE 118 , 164 = SozR 4 - 2500 § 73b Nr 1 , RdNr 72`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 73b Abs 5 S 4 SGB V`(NRM)
- `§ 71 Abs 1 S 1 SGB V`(NRM)
- `BSG Urteil vom 25. 3. 2015 - B 6 KA 9/14 R - BSGE 118 , 164 = SozR 4 - 2500 § 73b Nr 1 , RdNr 72`(RS)

**Example 4** (doc_id: `53848`) (sent_id: `53848`)


Vielmehr setzt der Sinn und Zweck der Vorschrift voraus , dass auch das konkrete Verfahren von dem Sozialleistungsträger gerade in dieser Eigenschaft geführt wird ; das Verfahren muss also einen engen sachlichen Zusammenhang zu der gesetzlichen Tätigkeit als Träger der in der Vorschrift genannten Sozialleistungen haben ( BGH Beschluss vom 10. 11. 2005 - IX ZR 189/02 - Juris RdNr 6 f ; BGH Beschluss vom 28. 9. 2016 - XII ZB 251/16 - Juris RdNr 30 ) .

**False Positives:**

- `BGH Beschluss` — partial — pred is substring of gold: `BGH Beschluss vom 10. 11. 2005 - IX ZR 189/02 - Juris RdNr 6 f`
- `BGH Beschluss` — similar text (different position): `BGH Beschluss vom 10. 11. 2005 - IX ZR 189/02 - Juris RdNr 6 f`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH Beschluss vom 10. 11. 2005 - IX ZR 189/02 - Juris RdNr 6 f`(RS)
- `BGH Beschluss vom 28. 9. 2016 - XII ZB 251/16 - Juris RdNr 30`(RS)

**Example 5** (doc_id: `53879`) (sent_id: `53879`)


Bemüht sich jemand , der ein Statusfeststellungsverfahren einleitet , zeitnah um private Eigenvorsorge , so kann er diese für den Fall , dass das Statusfeststellungsverfahren entgegen seinen Vorstellungen zu einer Feststellung von Versicherungspflicht führt , möglicherweise gar nicht mehr oder nur mit erheblichem Aufwand rückabwickeln ( zu diesen Konsequenzen siehe LSG Berlin-Brandenburg Urteil vom 18. 9. 2013 - L 9 KR 384/11 - Juris RdNr 38 ) .

**False Positives:**

- `LSG Berlin` — partial — pred is substring of gold: `LSG Berlin-Brandenburg Urteil vom 18. 9. 2013 - L 9 KR 384/11 - Juris RdNr 38`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `LSG Berlin-Brandenburg Urteil vom 18. 9. 2013 - L 9 KR 384/11 - Juris RdNr 38`(RS)

**Example 6** (doc_id: `53991`) (sent_id: `53991`)


Nicht das tatsächliche Verhalten des Arbeitgebers im Lohnsteuerabzugsverfahren bindet dessen Beteiligte ( vgl BSG Urteil vom 23. 5. 2017 - B 12 KR 6/16 R - SozR 4 - 5376 § 1 Nr 1 RdNr 23 ; BSG Urteil vom 26. 3. 2014 - B 10 EG 14/13 R - BSGE 115 , 198 = SozR 4 - 7837 § 2 Nr 25 , RdNr 26 f ) , wohl aber die Rechtsfolgen , die AO und EStG daran knüpfen .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 23. 5. 2017 - B 12 KR 6/16 R - SozR 4 - 5376 § 1 Nr 1 RdNr 23`
- `BSG Urteil` — similar text (different position): `BSG Urteil vom 23. 5. 2017 - B 12 KR 6/16 R - SozR 4 - 5376 § 1 Nr 1 RdNr 23`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Urteil vom 23. 5. 2017 - B 12 KR 6/16 R - SozR 4 - 5376 § 1 Nr 1 RdNr 23`(RS)
- `BSG Urteil vom 26. 3. 2014 - B 10 EG 14/13 R - BSGE 115 , 198 = SozR 4 - 7837 § 2 Nr 25 , RdNr 26 f`(RS)
- `AO`(NRM)
- `EStG`(NRM)

**Example 7** (doc_id: `54082`) (sent_id: `54082`)


Den Beteiligten war nämlich bewusst , denn dies ist sogar protokolliert , dass gegen das besprochene Urteil des FG Berlin-Brandenburg noch eine Revision anhängig war .

**False Positives:**

- `FG Berlin` — partial — pred is substring of gold: `FG Berlin-Brandenburg`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `FG Berlin-Brandenburg`(ORG)

**Example 8** (doc_id: `54171`) (sent_id: `54171`)


Ein anderer als der vom LSG herangezogene Prüfungsmaßstab unter Anwendung weiterer Vorschriften des Bundesrechts folgt entgegen der Rechtsauffassung der Beklagten auch nicht aus einem Beschluss des Senats , in dem die Revision gegen ein Urteil des LSG Baden-Württemberg vom 23. 1. 2013 ( L 5 R 4971/10 ) als unzulässig verworfen wurde ( Senatsbeschluss vom 6. 2. 2014 - B 5 RE 10/14 R ) .

**False Positives:**

- `LSG Baden` — partial — pred is substring of gold: `Urteil des LSG Baden-Württemberg vom 23. 1. 2013 ( L 5 R 4971/10 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des LSG Baden-Württemberg vom 23. 1. 2013 ( L 5 R 4971/10 )`(RS)
- `Senatsbeschluss vom 6. 2. 2014 - B 5 RE 10/14 R`(RS)

**Example 9** (doc_id: `54270`) (sent_id: `54270`)


Sie beantragt , das Urteil des FG Berlin-Brandenburg vom 18. Januar 2017 3 K 3219/16 aufzuheben und die Klage abzuweisen .

**False Positives:**

- `FG Berlin` — partial — pred is substring of gold: `Urteil des FG Berlin-Brandenburg vom 18. Januar 2017 3 K 3219/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des FG Berlin-Brandenburg vom 18. Januar 2017 3 K 3219/16`(RS)

**Example 10** (doc_id: `54285`) (sent_id: `54285`)


Den vom Kläger aus den Rechnungen über den Ankauf der BHKW beanspruchten , vorliegend nicht verfahrensgegenständlichen Vorsteuerabzug erkannte der Beklagte und Revisionskläger ( das Finanzamt - FA - ) nicht an ( bestätigt durch FG Münster , Urteil vom 16. Oktober 2014 5 K 3875/12 U , Entscheidungen der Finanzgerichte - EFG - 2015 , 84 , rechtskräftig ) .

**False Positives:**

- `FG Münster` — partial — pred is substring of gold: `FG Münster , Urteil vom 16. Oktober 2014 5 K 3875/12 U , Entscheidungen der Finanzgerichte - EFG - 2015 , 84`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `FG Münster , Urteil vom 16. Oktober 2014 5 K 3875/12 U , Entscheidungen der Finanzgerichte - EFG - 2015 , 84`(RS)

**Example 11** (doc_id: `54500`) (sent_id: `54500`)


Die Sache wird an das Finanzgericht Rheinland-Pfalz zurückverwiesen .

**False Positives:**

- `Finanzgericht Rheinland` — partial — pred is substring of gold: `Finanzgericht Rheinland-Pfalz`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Finanzgericht Rheinland-Pfalz`(ORG)

**Example 12** (doc_id: `54544`) (sent_id: `54544`)


Dagegen verhielten sich Eltern widersprüchlich , wollten sie einerseits von den Steuervorteilen einer ( unrichtigen ) Besteuerung von Entgeltbestandteilen als sonstige Bezüge profitieren , um diese dann andererseits im nachfolgenden Elterngeldverfahren mit dem Ziel höheren Elterngelds wieder infrage zu stellen ( zur Maßgeblichkeit in Anspruch genommener steuerlicher Vergünstigungen bei der Berechnung des Elterngelds aus selbstständiger Erwerbstätigkeit BSG Urteil vom 15. 12. 2015 - B 10 EG 6/14 R - SozR 4 - 7837 § 2 Nr 30 RdNr 19 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 15. 12. 2015 - B 10 EG 6/14 R - SozR 4 - 7837 § 2 Nr 30 RdNr 19`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Urteil vom 15. 12. 2015 - B 10 EG 6/14 R - SozR 4 - 7837 § 2 Nr 30 RdNr 19`(RS)

**Example 13** (doc_id: `54609`) (sent_id: `54609`)


Sie ist weder vom Kläger noch der Beigeladenen zu 1. - unter Hinweis auf eine Verletzung des § 7 SGB IV - im Revisionsverfahren mit Rechtsmitteln angegriffen worden ( vgl zur Teilbarkeit eines Statusfeststellungsbescheids insoweit schon BSG Urteil vom 24. 3. 2016 - B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 RdNr 18 , 11 ; BSG Urteil vom 24. 3. 2016 - B 12 R 12/14 R - SozR 4 - 2400 § 7a Nr 6 RdNr 11 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 24. 3. 2016 - B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 RdNr 18 , 11`
- `BSG Urteil` — similar text (different position): `BSG Urteil vom 24. 3. 2016 - B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 RdNr 18 , 11`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 7 SGB IV`(NRM)
- `BSG Urteil vom 24. 3. 2016 - B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 RdNr 18 , 11`(RS)
- `BSG Urteil vom 24. 3. 2016 - B 12 R 12/14 R - SozR 4 - 2400 § 7a Nr 6 RdNr 11`(RS)

**Example 14** (doc_id: `54785`) (sent_id: `54785`)


Im Fall des FG Berlin-Brandenburg hatte der dortige Kläger , ein heilkundlicher Verkehrstherapeut , mit den Klienten eine Therapievereinbarung getroffen , wonach u. a. der MPU-Erfolg nicht das Ziel der Therapie sei .

**False Positives:**

- `FG Berlin` — partial — pred is substring of gold: `FG Berlin-Brandenburg`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `FG Berlin-Brandenburg`(ORG)

**Example 15** (doc_id: `54873`) (sent_id: `54873`)


Im Gegenteil nimmt die arbeitsgerichtliche Rechtsprechung an , dass Lehrer an Musikschulen nur dann als Arbeitnehmer anzusehen sind , wenn die Parteien dies vereinbart haben oder im Einzelfall festzustellende Umstände hinzutreten , aus denen sich ergibt , dass der für das Bestehen eines Arbeitsverhältnisses erforderliche Grad der persönlichen Abhängigkeit gegeben ist ( vgl aktuell zu Musikschullehrern BAG Urteil vom 21. 11. 2017 - 9 AZR 117/17 - Juris ; BAG Urteil vom 17. 10. 2017 - 9 AZR 792/16 - Juris ; BAG Urteil vom 27. 6. 2017 - 9 AZR 851/16 - Juris ; BAG Urteil vom 27. 6. 2017 - 9 AZR 852/16 - Juris mwN ) .

**False Positives:**

- `BAG Urteil` — partial — pred is substring of gold: `BAG Urteil vom 21. 11. 2017 - 9 AZR 117/17 - Juris`
- `BAG Urteil` — similar text (different position): `BAG Urteil vom 21. 11. 2017 - 9 AZR 117/17 - Juris`
- `BAG Urteil` — similar text (different position): `BAG Urteil vom 21. 11. 2017 - 9 AZR 117/17 - Juris`
- `BAG Urteil` — similar text (different position): `BAG Urteil vom 21. 11. 2017 - 9 AZR 117/17 - Juris`

> overlaps gold: 4  |  likely missing annotation: 0

**Gold Entities:**

- `BAG Urteil vom 21. 11. 2017 - 9 AZR 117/17 - Juris`(RS)
- `BAG Urteil vom 17. 10. 2017 - 9 AZR 792/16 - Juris`(RS)
- `BAG Urteil vom 27. 6. 2017 - 9 AZR 851/16 - Juris`(RS)
- `BAG Urteil vom 27. 6. 2017 - 9 AZR 852/16 - Juris`(RS)

**Example 16** (doc_id: `55021`) (sent_id: `55021`)


Sittliche Gründe zur Übernahme der Beerdigungskosten kommen im Allgemeinen bei einem nahen Angehörigen in Betracht ( BFH-Urteil in BFHE 150 , 351 , BStBl II 1987 , 715 ; FG Münster , Urteil in EFG 2014 , 44 ; HHR / Kanzler , EStG § 33 Rz 142 ) .

**False Positives:**

- `FG Münster` — partial — pred is substring of gold: `FG Münster , Urteil in EFG 2014 , 44`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH-Urteil in BFHE 150 , 351 , BStBl II 1987 , 715`(RS)
- `FG Münster , Urteil in EFG 2014 , 44`(RS)
- `HHR / Kanzler , EStG § 33 Rz 142`(LIT)

**Example 17** (doc_id: `55400`) (sent_id: `55400`)


Die Aufgabe , bundeseinheitliche Vorgaben für die Honorarverteilung zu treffen , welche die regionalen Partner der Honorarverteilungsvereinbarungen zu beachten hatten , war dem BewA - zusätzlich zu seiner originären Kompetenz der Leistungsbewertung nach § 87 Abs 2 SGB V - übertragen ( BSG Urteil vom 19. 8. 2015 - B 6 KA 34/14 R - BSGE 119 , 231 = SozR 4 - 2500 § 87b Nr 7 , RdNr 25 mwN ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 19. 8. 2015 - B 6 KA 34/14 R - BSGE 119 , 231 = SozR 4 - 2500 § 87b Nr 7 , RdNr 25`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 87 Abs 2 SGB V`(NRM)
- `BSG Urteil vom 19. 8. 2015 - B 6 KA 34/14 R - BSGE 119 , 231 = SozR 4 - 2500 § 87b Nr 7 , RdNr 25`(RS)

**Example 18** (doc_id: `55428`) (sent_id: `55428`)


Zur Frage der entsprechenden Anwendbarkeit wettbewerbsrechtlicher Bestimmungen ( ua zum Schadensersatz ) auf die in § 69 Abs 1 S 1 SGB V geregelten Rechtsverhältnisse hat der Senat bereits entschieden , dass diese zur Kompensation einer unterlassenen oder im Ergebnis erfolglosen Inanspruchnahme gerichtlichen Primärrechtsschutzes , insbesondere von einstweiligem Rechtsschutz nach § 86b SGG , von vornherein nicht zur Verfügung stehen ( BSG Urteil vom 15. 3. 2017 - B 6 KA 35/16 R - zur Veröffentlichung für BSGE und SozR 4 - 5540 Anl 9.1 Nr 12 vorgesehen , RdNr 28 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 15. 3. 2017 - B 6 KA 35/16 R - zur Veröffentlichung für BSGE und SozR 4 - 5540 Anl 9.1 Nr 12 vorgesehen , RdNr 28`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 69 Abs 1 S 1 SGB V`(NRM)
- `§ 86b SGG`(NRM)
- `BSG Urteil vom 15. 3. 2017 - B 6 KA 35/16 R - zur Veröffentlichung für BSGE und SozR 4 - 5540 Anl 9.1 Nr 12 vorgesehen , RdNr 28`(RS)

**Example 19** (doc_id: `55528`) (sent_id: `55528`)


Zwar kann sich derjenige auf einen Anspruch auf rechtliches Gehör stützen , der nach der maßgeblichen Verfahrensordnung an einem gerichtlichen Verfahren als Partei oder in parteiähnlicher Stellung beteiligt oder unmittelbar rechtlich von dem Verfahren betroffen ist ( stRspr ; vgl etwa BVerfG Beschluss vom 14. 4. 1987 - 1 BvR 332/86 - BVerfGE 75 , 201 < 215 > - Juris RdNr 49 ) .

**False Positives:**

- `BVerfG Beschluss` — partial — pred is substring of gold: `BVerfG Beschluss vom 14. 4. 1987 - 1 BvR 332/86 - BVerfGE 75 , 201 < 215 > - Juris RdNr 49`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG Beschluss vom 14. 4. 1987 - 1 BvR 332/86 - BVerfGE 75 , 201 < 215 > - Juris RdNr 49`(RS)

**Example 20** (doc_id: `55560`) (sent_id: `55560`)


Mit der Beitragsforderung wurde durch die Beklagte zumindest in die allgemeine Handlungsfreiheit und damit in das Grundrecht des Klägers aus Art 2 Abs 1 GG eingegriffen , wodurch ein anhörungspflichtiger " Eingriff " iS des § 24 Abs 1 SGB X vorlag ( vgl BSG Urteil vom 25. 1. 1979 - 3 RK 35/77 - SozR 1200 § 34 Nr 7 ; Mutschler in Kasseler Komm , Stand September 2015 , § 24 SGB X RdNr 7 ; Siefert in von Wulffen / Schütze , SGB X , 8. Aufl 2014 , § 24 RdNr 8 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 25. 1. 1979 - 3 RK 35/77 - SozR 1200 § 34 Nr 7`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art 2 Abs 1 GG`(NRM)
- `§ 24 Abs 1 SGB X`(NRM)
- `BSG Urteil vom 25. 1. 1979 - 3 RK 35/77 - SozR 1200 § 34 Nr 7`(RS)
- `Mutschler in Kasseler Komm , Stand September 2015 , § 24 SGB X RdNr 7`(LIT)
- `Siefert in von Wulffen / Schütze , SGB X , 8. Aufl 2014 , § 24 RdNr 8`(LIT)

**Example 21** (doc_id: `55636`) (sent_id: `55636`)


Einstweilige Rechtsschutzanträge des Beklagten zur Fortführung des Vertrages blieben erfolglos ( SG München Beschluss vom 19. 1. 2011 - S 39 KA 1248/10 ER ; Bayerisches LSG Beschluss vom 22. 2. 2011 - L 12 KA 2/11 B ER - NZS 2011 , 386 ) .

**False Positives:**

- `SG München Beschluss` — partial — pred is substring of gold: `SG München Beschluss vom 19. 1. 2011 - S 39 KA 1248/10 ER`
- `LSG Beschluss` — partial — pred is substring of gold: `Bayerisches LSG Beschluss vom 22. 2. 2011 - L 12 KA 2/11 B ER - NZS 2011 , 386`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `SG München Beschluss vom 19. 1. 2011 - S 39 KA 1248/10 ER`(RS)
- `Bayerisches LSG Beschluss vom 22. 2. 2011 - L 12 KA 2/11 B ER - NZS 2011 , 386`(RS)

**Example 22** (doc_id: `55829`) (sent_id: `55829`)


Auch über eine - nach § 69 Abs 1 S 3 SGB V nicht vollständig ausgeschlossene - entsprechende Heranziehung von Vorschriften des BGB können Schadensersatzansprüche einer Krankenkasse gegenüber einem Hausärzteverband oder den an der HzV teilnehmenden Ärzten unter diesen Umständen nicht begründet werden ( zu Schadensersatzansprüchen zwischen Leistungserbringern vgl BSG Urteil vom 15. 3. 2017 - B 6 KA 35/16 R - zur Veröffentlichung für BSGE und SozR 4 - 5540 Anl 9.1 Nr 12 vorgesehen , RdNr 31 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 15. 3. 2017 - B 6 KA 35/16 R - zur Veröffentlichung für BSGE und SozR 4 - 5540 Anl 9.1 Nr 12 vorgesehen , RdNr 31`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 69 Abs 1 S 3 SGB V`(NRM)
- `BGB`(NRM)
- `BSG Urteil vom 15. 3. 2017 - B 6 KA 35/16 R - zur Veröffentlichung für BSGE und SozR 4 - 5540 Anl 9.1 Nr 12 vorgesehen , RdNr 31`(RS)

**Example 23** (doc_id: `56005`) (sent_id: `56005`)


Der Kläger wäre daher , um seine Zuständigkeit nach § 14 Abs 2 Satz 1 SGB IX zu vermeiden , berechtigt gewesen , vor der ersten anstehenden Verlängerung der ( konkludenten ) Leistungsbewilligung nach dem 1. 7. 2001 ( BSG Urteil vom 13. 7. 2017 - B 8 SO 1/16 R - RdNr 22 ) , spätestens aber mit der Prüfung und Entscheidung über den Rehabilitationsantrag vom 28. 2. 2005 seine Zuständigkeit für den Leistungsfall zu prüfen und den Leistungsfall vor einer anstehenden Leistungsbewilligung bzw den Antrag der K. auf Eingliederungshilfe in der Außenwohngruppe ggf an den nach seiner Auffassung originär zuständigen Beklagten weiterzuleiten ; ein Fall des § 103 SGB X liegt ebenso wenig vor wie eine zielgerichtete Zuständigkeitsanmaßung , die eine Erstattung nach § 104 SGB X ausschließen würde ( vgl dazu BSGE 98 , 267 = SozR 4 - 3250 § 14 Nr 4 ; BSG SozR 4 - 3100 § 18c Nr 2 RdNr 30 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 13. 7. 2017 - B 8 SO 1/16 R - RdNr 22 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 14 Abs 2 Satz 1 SGB IX`(NRM)
- `BSG Urteil vom 13. 7. 2017 - B 8 SO 1/16 R - RdNr 22 )`(RS)
- `K.`(PER)
- `§ 103 SGB X`(NRM)
- `§ 104 SGB X`(NRM)
- `BSGE 98 , 267 = SozR 4 - 3250 § 14 Nr 4`(RS)
- `BSG SozR 4 - 3100 § 18c Nr 2 RdNr 30`(RS)

**Example 24** (doc_id: `56125`) (sent_id: `56125`)


Diesen Anspruch hat das LSG Mecklenburg-Vorpommern mit Urteil vom 22. 2. 2017 verneint und für das PKH-Vergütungsfestsetzungsverfahren eine überlange Verfahrensdauer von zwei Monaten festgestellt .

**False Positives:**

- `LSG Mecklenburg` — partial — pred is substring of gold: `LSG Mecklenburg-Vorpommern`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `LSG Mecklenburg-Vorpommern`(ORG)

**Example 25** (doc_id: `56363`) (sent_id: `56363`)


Bei der Ordnung von Massenerscheinungen können typisierende und generalisierende Regelungen notwendig sein ( vgl BSG Urteil vom 23. 7. 2014 - B 12 KR 28/12 R - BSGE 116 , 241 = SozR 4 - 2500 § 229 Nr 18 , RdNr 23 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 23. 7. 2014 - B 12 KR 28/12 R - BSGE 116 , 241 = SozR 4 - 2500 § 229 Nr 18 , RdNr 23`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Urteil vom 23. 7. 2014 - B 12 KR 28/12 R - BSGE 116 , 241 = SozR 4 - 2500 § 229 Nr 18 , RdNr 23`(RS)

**Example 26** (doc_id: `56512`) (sent_id: `56512`)


Denn Darlehen , die gewährt werden , um nach Antragstellung bzw Kenntnis des Sozialhilfeträgers angefallene existenzielle Bedarfe zu decken , sind wegen der von Anfang an bestehenden Rückzahlungsverpflichtung eine nur vorübergehend zur Verfügung gestellte Leistung , die bei der Hilfe zum Lebensunterhalt nicht als Einkommen zu berücksichtigen ist ( BSGE 112 , 67 = SozR 4 - 3500 § 92 Nr 1 , RdNr 26 ; BSG Urteil vom 23. 8. 2013 - B 8 SO 24/11 R - juris , RdNr 25 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 23. 8. 2013 - B 8 SO 24/11 R - juris , RdNr 25`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSGE 112 , 67 = SozR 4 - 3500 § 92 Nr 1 , RdNr 26`(RS)
- `BSG Urteil vom 23. 8. 2013 - B 8 SO 24/11 R - juris , RdNr 25`(RS)

**Example 27** (doc_id: `56591`) (sent_id: `56591`)


Während ansonsten in kostenrechtlichen Verfahren der Erinnerung nach dem GKG bzw dem RVG nunmehr auch in dritter Instanz grundsätzlich eine Entscheidung durch den Einzelrichter vorgesehen ist ( vgl § 66 Abs 6 S 1 GKG bzw § 33 Abs 8 S 1 RVG - s dazu BGH Beschluss vom 23. 4. 2015 - I ZB 73/14 - NJW 2015 , 2194 ) , lässt das SGG bislang auch bei Erinnerungen ( §§ 178 , 189 Abs 2 S 2 SGG ) ein Tätigwerden des Einzelrichters lediglich im Rahmen des § 155 Abs 2 S 1 Nr 5 , Abs 3 und 4 SGG zu ( Reichel in Zeihe / Hauck <Hrsg> , SGG , Stand August 2017 , § 189 RdNr 10a ) .

**False Positives:**

- `BGH Beschluss` — partial — pred is substring of gold: `BGH Beschluss vom 23. 4. 2015 - I ZB 73/14 - NJW 2015 , 2194`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `GKG`(NRM)
- `RVG`(NRM)
- `§ 66 Abs 6 S 1 GKG`(NRM)
- `§ 33 Abs 8 S 1 RVG`(NRM)
- `BGH Beschluss vom 23. 4. 2015 - I ZB 73/14 - NJW 2015 , 2194`(RS)
- `SGG`(NRM)
- `§§ 178 , 189 Abs 2 S 2 SGG`(NRM)
- `§ 155 Abs 2 S 1 Nr 5 , Abs 3 und 4 SGG`(NRM)
- `Reichel in Zeihe / Hauck <Hrsg> , SGG , Stand August 2017 , § 189 RdNr 10a`(LIT)

**Example 28** (doc_id: `56636`) (sent_id: `56636`)


Zudem hätte es einer Darlegung der Beweisanforderungen bedurft ( vgl hierzu insgesamt BSG , aaO ; BSG Urteil vom 19. 3. 1986 - 9a RVi 2/84 - BSGE 60 , 58 = SozR 3850 § 51 Nr 9 ) , wie diese in der angefochtenen Entscheidung des LSG ( S 21 des Urteils ) bereits ausgeführt worden sind , um eine Sachaufklärungsrüge nach § 103 SGG im Rahmen einer grundsätzlichen Bedeutung als Rechtsfrage zu formulieren .

**False Positives:**

- `BSG Urteil` — similar text (different position): `BSG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG`(ORG)
- `BSG Urteil vom 19. 3. 1986 - 9a RVi 2/84 - BSGE 60 , 58 = SozR 3850 § 51 Nr 9`(RS)
- `§ 103 SGG`(NRM)

**Example 29** (doc_id: `56762`) (sent_id: `56762`)


Das hat der erkennende Senat für Arzneimittel - vom BVerfG bestätigt - entschieden und der Gesetzgeber ist dem ebenfalls gefolgt ( vgl zu § 2 Abs 1a SGB V GKV-VStG , BR-Drucks 456/11 S 74 ; BVerfG Beschluss vom 30. 6. 2008 - 1 BvR 1665/07 - SozR 4 - 2500 § 31 Nr 17 im Anschluss an BSG USK 2007 - 25 ; vgl zum Ganzen auch BSG SozR 4 - 2500 § 18 Nr 8 RdNr 20 f mwN ) .

**False Positives:**

- `BVerfG Beschluss` — similar text (different position): `BVerfG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG`(ORG)
- `§ 2 Abs 1a SGB V GKV-VStG`(NRM)
- `BR-Drucks 456/11 S 74`(LIT)
- `BVerfG Beschluss vom 30. 6. 2008 - 1 BvR 1665/07 - SozR 4 - 2500 § 31 Nr 17 im Anschluss an BSG USK 2007 - 25`(RS)
- `BSG SozR 4 - 2500 § 18 Nr 8 RdNr 20 f`(RS)

**Example 30** (doc_id: `56841`) (sent_id: `56841`)


Zugleich fehlen Darlegungen , wonach die in der Beschwerdebegründung wiedergegebene Frage nach der Festlegung des Versicherungsfalls unter Berücksichtigung der vorgelegten Zeugnisse ( S 3 f der Beschwerdebegründung vom 21. 10. 2015 ) sachdienlich ( vgl BSG Beschluss vom 27. 11. 2007 - B 5a / 5 R 60/07 B - SozR 4 - 1500 § 116 Nr 1 RdNr 10 ) ist .

**False Positives:**

- `BSG Beschluss` — partial — pred is substring of gold: `BSG Beschluss vom 27. 11. 2007 - B 5a / 5 R 60/07 B - SozR 4 - 1500 § 116 Nr 1 RdNr 10`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Beschluss vom 27. 11. 2007 - B 5a / 5 R 60/07 B - SozR 4 - 1500 § 116 Nr 1 RdNr 10`(RS)

**Example 31** (doc_id: `56997`) (sent_id: `56997`)


Nach ständiger Rechtsprechung wirken Statuserteilungen und -aufhebungen im Vertragsarztrecht nur ex nunc und nicht ex tunc ( BSG Beschluss vom 5. 6. 2013 - B 6 KA 4/13 B - Juris RdNr 10 mwN ; BSG Urteil vom 31. 5. 2006 - B 6 KA 7/05 R - SozR 4 - 5520 § 24 Nr 2 RdNr 13 zur Genehmigung einer Verlegung des Praxissitzes ; BSG Urteil vom 11. 3. 2009 - B 6 KA 15/08 R - SozR 4 - 2500 § 96 Nr 1 RdNr 15 f , 22 ; BSG Urteil vom 29. 1. 1997 - 6 RKa 24/96 - BSGE 80 , 48 , 49/50 = SozR 3 - 2500 § 85 Nr 19 S 119/120 , Juris RdNr 15 bezogen auf eine Großgeräte-Genehmigung ) .

**False Positives:**

- `BSG Beschluss` — partial — pred is substring of gold: `BSG Beschluss vom 5. 6. 2013 - B 6 KA 4/13 B - Juris RdNr 10`
- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 31. 5. 2006 - B 6 KA 7/05 R - SozR 4 - 5520 § 24 Nr 2 RdNr 13 zur Genehmigung einer Verlegung des Praxissitzes`
- `BSG Urteil` — similar text (different position): `BSG Urteil vom 31. 5. 2006 - B 6 KA 7/05 R - SozR 4 - 5520 § 24 Nr 2 RdNr 13 zur Genehmigung einer Verlegung des Praxissitzes`
- `BSG Urteil` — similar text (different position): `BSG Urteil vom 31. 5. 2006 - B 6 KA 7/05 R - SozR 4 - 5520 § 24 Nr 2 RdNr 13 zur Genehmigung einer Verlegung des Praxissitzes`

> overlaps gold: 4  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Beschluss vom 5. 6. 2013 - B 6 KA 4/13 B - Juris RdNr 10`(RS)
- `BSG Urteil vom 31. 5. 2006 - B 6 KA 7/05 R - SozR 4 - 5520 § 24 Nr 2 RdNr 13 zur Genehmigung einer Verlegung des Praxissitzes`(RS)
- `BSG Urteil vom 11. 3. 2009 - B 6 KA 15/08 R - SozR 4 - 2500 § 96 Nr 1 RdNr 15 f , 22`(RS)
- `BSG Urteil vom 29. 1. 1997 - 6 RKa 24/96 - BSGE 80 , 48 , 49/50 = SozR 3 - 2500 § 85 Nr 19 S 119/120 , Juris RdNr 15`(RS)

**Example 32** (doc_id: `57048`) (sent_id: `57048`)


Ein Verfahrensmangel iS von § 160 Abs 2 Nr 3 SGG ist der Verstoß des Gerichts im Rahmen des prozessualen Vorgehens im unmittelbar vorangehenden Rechtszug ( vgl zB BSG Urteil vom 29. 11. 1955 - 1 RA 15/54 - BSGE 2 , 81 , 82 ; BSG Urteil vom 24. 10. 1961 - 6 RKa 19/60 - BSGE 15 , 169 , 172 = SozR Nr 3 zu § 52 SGG ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 29. 11. 1955 - 1 RA 15/54 - BSGE 2 , 81 , 82`
- `BSG Urteil` — similar text (different position): `BSG Urteil vom 29. 11. 1955 - 1 RA 15/54 - BSGE 2 , 81 , 82`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 160 Abs 2 Nr 3 SGG`(NRM)
- `BSG Urteil vom 29. 11. 1955 - 1 RA 15/54 - BSGE 2 , 81 , 82`(RS)
- `BSG Urteil vom 24. 10. 1961 - 6 RKa 19/60 - BSGE 15 , 169 , 172 = SozR Nr 3 zu § 52 SGG`(RS)

**Example 33** (doc_id: `57219`) (sent_id: `57219`)


Differenzierungen bedürfen jedoch stets der Rechtfertigung durch Sachgründe , die dem Ziel und dem Ausmaß der Ungleichbehandlung angemessen sind ( BVerfG Beschluss vom 24. 3. 2015 - 1 BvR 2880/11 - BVerfGE 139 , 1 RdNr 38 mwN ) .

**False Positives:**

- `BVerfG Beschluss` — partial — pred is substring of gold: `BVerfG Beschluss vom 24. 3. 2015 - 1 BvR 2880/11 - BVerfGE 139 , 1 RdNr 38`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG Beschluss vom 24. 3. 2015 - 1 BvR 2880/11 - BVerfGE 139 , 1 RdNr 38`(RS)

**Example 34** (doc_id: `57254`) (sent_id: `57254`)


Das Oberverwaltungsgericht Berlin-Brandenburg wies die Berufung mit Urteil vom 13. November 2014 - 4 B 31.11 - zurück .

**False Positives:**

- `Oberverwaltungsgericht Berlin` — partial — pred is substring of gold: `Oberverwaltungsgericht Berlin-Brandenburg`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Oberverwaltungsgericht Berlin-Brandenburg`(ORG)
- `Urteil vom 13. November 2014 - 4 B 31.11 -`(RS)

**Example 35** (doc_id: `57277`) (sent_id: `57277`)


Für die Geltendmachung der grundsätzlichen Bedeutung einer Rechtssache muss in der Beschwerdebegründung eine konkrete Rechtsfrage in klarer Formulierung bezeichnet ( vgl BVerfG Beschluss vom 14. 6. 1994 - 1 BvR 1022/88 - BVerfGE 91 , 93 , 107 = SozR 3 - 5870 § 10 Nr 5 S 31 ; BSG Beschluss vom 13. 5. 1997 - 13 BJ 271/96 - SozR 3 - 1500 § 160a Nr 21 S 37 f ) und ausgeführt werden , inwiefern diese Rechtsfrage in dem mit der Beschwerde angestrebten Revisionsverfahren entscheidungserheblich ( klärungsfähig ) sowie klärungsbedürftig ist .

**False Positives:**

- `BVerfG Beschluss` — partial — pred is substring of gold: `BVerfG Beschluss vom 14. 6. 1994 - 1 BvR 1022/88 - BVerfGE 91 , 93 , 107 = SozR 3 - 5870 § 10 Nr 5 S 31`
- `BSG Beschluss` — partial — pred is substring of gold: `BSG Beschluss vom 13. 5. 1997 - 13 BJ 271/96 - SozR 3 - 1500 § 160a Nr 21 S 37 f`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG Beschluss vom 14. 6. 1994 - 1 BvR 1022/88 - BVerfGE 91 , 93 , 107 = SozR 3 - 5870 § 10 Nr 5 S 31`(RS)
- `BSG Beschluss vom 13. 5. 1997 - 13 BJ 271/96 - SozR 3 - 1500 § 160a Nr 21 S 37 f`(RS)

**Example 36** (doc_id: `57492`) (sent_id: `57492`)


4. Die Kostenentscheidung beruht auf einer entsprechenden Anwendung von § 193 SGG ( vgl BSG Beschluss vom 29. 9. 2017 - B 13 SF 8/17 S - Juris RdNr 30 ) .

**False Positives:**

- `BSG Beschluss` — partial — pred is substring of gold: `BSG Beschluss vom 29. 9. 2017 - B 13 SF 8/17 S - Juris RdNr 30`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 193 SGG`(NRM)
- `BSG Beschluss vom 29. 9. 2017 - B 13 SF 8/17 S - Juris RdNr 30`(RS)

**Example 37** (doc_id: `57615`) (sent_id: `57615`)


Auf die abstrakte berufliche Qualifikation des Beschäftigten bzw Selbstständigen kommt es nicht an ( BSG Urteil vom 31. 10. 2012 - B 12 R 3/11 R - BSGE 112 , 108 = SozR 4 - 2600 § 6 Nr 9 , RdNr 34 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 31. 10. 2012 - B 12 R 3/11 R - BSGE 112 , 108 = SozR 4 - 2600 § 6 Nr 9 , RdNr 34`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Urteil vom 31. 10. 2012 - B 12 R 3/11 R - BSGE 112 , 108 = SozR 4 - 2600 § 6 Nr 9 , RdNr 34`(RS)

**Example 38** (doc_id: `57825`) (sent_id: `57825`)


Nach ständiger Rechtsprechung des Senats ( vgl. insb. BVerwG , Beschluss vom 28. Mai 2008 - 1 WB 19.07 - Buchholz 449 § 3 SG Nr. 44 Rn. 23 und 26 ) erlangen Verwaltungsvorschriften Außenwirkung gegenüber dem Soldaten mittelbar über den allgemeinen Gleichheitssatz des Art. 3 Abs. 1 GG .

**False Positives:**

- `SG Nr` — partial — pred is substring of gold: `BVerwG , Beschluss vom 28. Mai 2008 - 1 WB 19.07 - Buchholz 449 § 3 SG Nr. 44 Rn. 23 und 26`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerwG , Beschluss vom 28. Mai 2008 - 1 WB 19.07 - Buchholz 449 § 3 SG Nr. 44 Rn. 23 und 26`(RS)
- `Art. 3 Abs. 1 GG`(NRM)

**Example 39** (doc_id: `57870`) (sent_id: `57870`)


Soweit - wie vorliegend - Verstöße gegen die tatrichterliche Sachaufklärungspflicht ( § 103 SGG ) gerügt werden , muss die Beschwerdebegründung hierzu jeweils folgende Punkte enthalten :( 1. ) Bezeichnung eines für das Revisionsgericht ohne Weiteres auffindbaren , bis zuletzt aufrechterhaltenen Beweisantrags , dem das LSG nicht gefolgt ist , ( 2. ) Wiedergabe der Rechtsauffassung des LSG , aufgrund derer bestimmte Tatfragen als klärungsbedürftig hätten erscheinen müssen , ( 3. ) Darlegung der von dem betreffenden Beweisantrag berührten Tatumstände , die zu einer weiteren Sachaufklärung Anlass gegeben hätten , ( 4. ) Angabe des voraussichtlichen Ergebnisses der unterbliebenen Beweisaufnahme und ( 5. ) Schilderung , dass und warum die Entscheidung des LSG auf der angeblich fehlerhaft unterlassenen Beweisaufnahme beruhen kann , das LSG mithin bei Kenntnis des behaupteten Ergebnisses der unterlassenen Beweisaufnahme von seinem Rechtsstandpunkt aus zu einem anderen , dem Beschwerdeführer günstigeren Ergebnis hätte gelangen können ( vgl BSG Beschluss vom 21. 12. 2017 - B 9 SB 70/17 B - Juris RdNr 3 mwN ) .

**False Positives:**

- `BSG Beschluss` — partial — pred is substring of gold: `BSG Beschluss vom 21. 12. 2017 - B 9 SB 70/17 B - Juris RdNr 3`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 103 SGG`(NRM)
- `BSG Beschluss vom 21. 12. 2017 - B 9 SB 70/17 B - Juris RdNr 3`(RS)

**Example 40** (doc_id: `57949`) (sent_id: `57949`)


Der Senat folgt insoweit der Rechtsprechung des Bundesverfassungsgerichts für das verfassungsrechtliche Verfahren ( vgl BVerfG Nichtannahmebeschluss vom 23. 12. 2016 - 1 BvR 3511/13 - Juris RdNr 3 mwN ; ebenso BFH Beschluss vom 8. 10. 2015 - VII B 147/14 - Juris RdNr 4 ) .

**False Positives:**

- `BVerfG Nichtannahmebeschluss` — partial — pred is substring of gold: `BVerfG Nichtannahmebeschluss vom 23. 12. 2016 - 1 BvR 3511/13 - Juris RdNr 3`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesverfassungsgerichts`(ORG)
- `BVerfG Nichtannahmebeschluss vom 23. 12. 2016 - 1 BvR 3511/13 - Juris RdNr 3`(RS)
- `BFH Beschluss vom 8. 10. 2015 - VII B 147/14 - Juris RdNr 4`(RS)

**Example 41** (doc_id: `58116`) (sent_id: `58116`)


Selbst bei groben Zuständigkeitsverstößen ist ein Feststellungsbescheid daher zwar rechtswidrig , aber nicht nichtig ( s zuletzt zur fehlenden Nichtigkeit bei Verstoß gegen europarechtliche Kollisionsnormen Urteil des erkennenden Senats vom 3. 4. 2014 - B 2 U 25/12 R - BSGE 115 , 256 = SozR 4 - 2700 § 136 Nr 6 , RdNr 25 ; BSG Urteil vom 28. 11. 1961 - 2 RU 36/58 - BSGE 15 , 282 , 285 = SozR Nr 1 zu § 666 RVO ; BSG Urteil vom 30. 10. 1974 - 2 RU 42/73 - BSGE 38 , 187 , 192 = SozR 2200 § 664 Nr 1 S 7 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 28. 11. 1961 - 2 RU 36/58 - BSGE 15 , 282 , 285 = SozR Nr 1 zu § 666 RVO`
- `BSG Urteil` — similar text (different position): `BSG Urteil vom 28. 11. 1961 - 2 RU 36/58 - BSGE 15 , 282 , 285 = SozR Nr 1 zu § 666 RVO`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des erkennenden Senats vom 3. 4. 2014 - B 2 U 25/12 R - BSGE 115 , 256 = SozR 4 - 2700 § 136 Nr 6 , RdNr 25`(RS)
- `BSG Urteil vom 28. 11. 1961 - 2 RU 36/58 - BSGE 15 , 282 , 285 = SozR Nr 1 zu § 666 RVO`(RS)
- `BSG Urteil vom 30. 10. 1974 - 2 RU 42/73 - BSGE 38 , 187 , 192 = SozR 2200 § 664 Nr 1 S 7`(RS)

**Example 42** (doc_id: `58247`) (sent_id: `58247`)


Richter am BAG Waskowist an der Beifügung der Unterschrift verhindert Gräfl

**False Positives:**

- `BAG Waskowist` — partial — gold is substring of pred: `BAG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG`(ORG)
- `Waskowist`(PER)
- `Gräfl`(PER)

**Example 43** (doc_id: `58308`) (sent_id: `58308`)


Es besteht vielmehr mit allen Rechten und Pflichten fort , bis es durch - rechtskräftiges - rechtsgestaltendes Urteil - ggf. rückwirkend - aufgelöst wird ( vgl. LAG Köln 12. November 2014 - 5 Sa 419/14 - zu II 2 b aa der Gründe ) .

**False Positives:**

- `LAG Köln` — partial — pred is substring of gold: `LAG Köln 12. November 2014 - 5 Sa 419/14 - zu II 2 b aa der Gründe`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `LAG Köln 12. November 2014 - 5 Sa 419/14 - zu II 2 b aa der Gründe`(RS)

**Example 44** (doc_id: `58408`) (sent_id: `58408`)


Anhaltspunkte dafür , dass der Vertragsschluss und die darin übereinstimmend getroffenen Regelungen allein aufgrund eines erheblichen Ungleichgewichts der Verhandlungspositionen oder unter Ausnutzung besonderer Umstände des Beigeladenen zu 1. ( denkbar wären zB geschäftliche Unerfahrenheit , Ausnutzung einer akuten Zwangslage bzw Notsituation ) zustande gekommen sind ( vgl BSG Urteil vom 18. 11. 2015 - B 12 KR 16/13 R - BSGE 120 , 99 = SozR 4 - 2400 § 7 Nr 25 , RdNr 26 mwN ) , liegen nicht vor .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 18. 11. 2015 - B 12 KR 16/13 R - BSGE 120 , 99 = SozR 4 - 2400 § 7 Nr 25 , RdNr 26`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Urteil vom 18. 11. 2015 - B 12 KR 16/13 R - BSGE 120 , 99 = SozR 4 - 2400 § 7 Nr 25 , RdNr 26`(RS)

**Example 45** (doc_id: `58505`) (sent_id: `58505`)


3. Der Antrag des Klägers auf Gewährung von PKH zur Durchführung des Revisionsverfahrens gegen das Urteil des LSG Rheinland-Pfalz vom 8. 8. 2017 ist abzulehnen .

**False Positives:**

- `LSG Rheinland` — partial — pred is substring of gold: `LSG Rheinland-Pfalz`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `LSG Rheinland-Pfalz`(ORG)

**Example 46** (doc_id: `58549`) (sent_id: `58549`)


Der Senat hält in diesem Zusammenhang nicht mehr an der spezifisch elterngeldrechtlichen Auslegung des § 2c Abs 1 S 2 BEEG ( § 2 Abs 7 S 2 BEEG aF ) fest , der zufolge es - noch unterschieden durch den Anspruchsgrund - in einem Arbeitsverhältnis mehrere laufende , dh regelmäßige Arbeitslöhne in verschiedenen Lohnzahlungszeiträumen nebeneinander geben kann ( anders noch BSG Urteil vom 26. 3. 2014 - B 10 EG 14/13 R - BSGE 115 , 198 = SozR 4 - 7837 § 2 Nr 25 , RdNr 35 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 26. 3. 2014 - B 10 EG 14/13 R - BSGE 115 , 198 = SozR 4 - 7837 § 2 Nr 25 , RdNr 35`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 2c Abs 1 S 2 BEEG`(NRM)
- `§ 2 Abs 7 S 2 BEEG aF`(NRM)
- `BSG Urteil vom 26. 3. 2014 - B 10 EG 14/13 R - BSGE 115 , 198 = SozR 4 - 7837 § 2 Nr 25 , RdNr 35`(RS)

**Example 47** (doc_id: `58693`) (sent_id: `58693`)


1.7.3 Im Hinblick auf die Nacharbeitung D21A hat die Patentinhaberin geltend gemacht , diese würde die nach dem BGH-Urteil „ Metazachlor “ ( BGH Urteil vom 15. März 2011 – X ZR 58/08 – , juris ) und dem Urteil 3 Ni 5/15 des BPatG ( BPatG Urteil vom 11. Oktober 2016 – 3 Ni 5/15 ( EP ) – , juris ) aufgestellten strengen Maßstäbe nicht erfüllen , die bei der Feststellung einer implizierten Offenbarung einer Druckschrift des Standes der Technik durch Nacharbeitung angelegt werden müssten .

**False Positives:**

- `BGH Urteil` — partial — pred is substring of gold: `BGH-Urteil „ Metazachlor “ ( BGH Urteil vom 15. März 2011 – X ZR 58/08 – , juris )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH-Urteil „ Metazachlor “ ( BGH Urteil vom 15. März 2011 – X ZR 58/08 – , juris )`(RS)
- `Urteil 3 Ni 5/15 des BPatG ( BPatG Urteil vom 11. Oktober 2016 – 3 Ni 5/15 ( EP ) – , juris )`(RS)

**Example 48** (doc_id: `58756`) (sent_id: `58756`)


Wenn das FG Zweifel an der Richtigkeit des Erinnerungsprotokolls und seines Inhalts gehabt hätte und sein Urteil darauf hätte stützen wollen , hätte es zuvor den Sachverhalt vollständig aufklären müssen ( z.B. durch Vernehmung des Täters als Zeugen oder subsidiär des Klägers als Partei und des Ohrenzeugen Z als Zeuge ) .

**False Positives:**

- `FG Zweifel` — similar text (different position): `Z`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Z`(PER)

**Example 49** (doc_id: `58923`) (sent_id: `58923`)


Dementsprechend hat die Krankenkasse im Regelfall keine Möglichkeit , den Vertragsarzt unmittelbar und ohne Tätigwerden der vertragsarztrechtlichen Gremien in Regress zu nehmen ( BSG Urteil vom 5. 5. 2010 - B 6 KA 5/09 R - SozR 4 - 2500 § 106 Nr 28 RdNr 44 ; BSG Urteil vom 20. 3. 2013 - B 6 KA 17/12 R - SozR 4 - 5540 § 48 Nr 2 RdNr 24 ; zum zahnärztlichen Bereich vgl BSG Urteil vom 25. 3. 2003 - B 1 KR 29/02 R - SozR 4 - 1500 § 55 Nr 1 RdNr 3 f , Juris RdNr 9 f ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 5. 5. 2010 - B 6 KA 5/09 R - SozR 4 - 2500 § 106 Nr 28 RdNr 44`
- `BSG Urteil` — similar text (different position): `BSG Urteil vom 5. 5. 2010 - B 6 KA 5/09 R - SozR 4 - 2500 § 106 Nr 28 RdNr 44`
- `BSG Urteil` — similar text (different position): `BSG Urteil vom 5. 5. 2010 - B 6 KA 5/09 R - SozR 4 - 2500 § 106 Nr 28 RdNr 44`

> overlaps gold: 3  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Urteil vom 5. 5. 2010 - B 6 KA 5/09 R - SozR 4 - 2500 § 106 Nr 28 RdNr 44`(RS)
- `BSG Urteil vom 20. 3. 2013 - B 6 KA 17/12 R - SozR 4 - 5540 § 48 Nr 2 RdNr 24`(RS)
- `BSG Urteil vom 25. 3. 2003 - B 1 KR 29/02 R - SozR 4 - 1500 § 55 Nr 1 RdNr 3 f , Juris RdNr 9 f`(RS)

**Example 50** (doc_id: `59056`) (sent_id: `59056`)


Dabei muss sich der Erfolg aus wissenschaftlich einwandfrei geführten Statistiken über die Zahl der behandelten Fälle und die Wirksamkeit der neuen Methode ablesen lassen ( stRspr ; vgl BSGE 76 , 194 = SozR 3 - 2500 § 27 Nr 5 = Juris RdNr 22 ff ; BSGE 115 , 95 = SozR 4 - 2500 § 2 Nr 4 , RdNr 21 ; BSG Urteil vom 19. 12. 2017 - B 1 KR 17/17 R - für BSGE und SozR 4 vorgesehen , Juris RdNr 14 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 19. 12. 2017 - B 1 KR 17/17 R - für BSGE und SozR 4 vorgesehen , Juris RdNr 14`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSGE 76 , 194 = SozR 3 - 2500 § 27 Nr 5 = Juris RdNr 22 ff`(RS)
- `BSGE 115 , 95 = SozR 4 - 2500 § 2 Nr 4 , RdNr 21`(RS)
- `BSG Urteil vom 19. 12. 2017 - B 1 KR 17/17 R - für BSGE und SozR 4 vorgesehen , Juris RdNr 14`(RS)

**Example 51** (doc_id: `59115`) (sent_id: `59115`)


In solchen Fällen begründet § 14 Abs 1 Satz 1 iVm Abs 2 Satz 1 und 2 SGB IX für das Erstattungsverhältnis zwischen den Trägern eine nachrangige Zuständigkeit des erstangegangenen Trägers , wenn er außerhalb der durch § 14 SGB IX geschaffenen Zuständigkeitsordnung unzuständig , ein anderer Träger aber eigentlich zuständig gewesen wäre ( vgl dazu nur BSGE 98 , 267 = SozR 4 - 3250 § 14 Nr 4 , RdNr 9 ; BSG Urteil vom 26. 10. 2017 - B 8 SO 12/16 R - RdNr 18 für SozR 4 vorgesehen ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 26. 10. 2017 - B 8 SO 12/16 R - RdNr 18 für SozR 4 vorgesehen`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 14 Abs 1 Satz 1 iVm Abs 2 Satz 1 und 2 SGB IX`(NRM)
- `§ 14 SGB IX`(NRM)
- `BSGE 98 , 267 = SozR 4 - 3250 § 14 Nr 4 , RdNr 9`(RS)
- `BSG Urteil vom 26. 10. 2017 - B 8 SO 12/16 R - RdNr 18 für SozR 4 vorgesehen`(RS)

**Example 52** (doc_id: `59146`) (sent_id: `59146`)


Er beantragt , das Urteil des FG Rheinland-Pfalz vom 14. Mai 2014 2 K 1454/13 aufzuheben und die Einkommensteuerbescheide 2008 und 2009 jeweils vom 19. März 2012 in Gestalt der Einspruchsentscheidung vom 14. März 2013 dahingehend zu ändern , dass die Einkünfte aus Land- und Forstwirtschaft um jeweils 19.723 € auf 34.338 € ( 2008 ) und auf 34.349 € ( 2009 ) gemindert werden und die Einkommensteuer für 2008 auf 5.166 € und für 2009 auf 4.937 € festgesetzt wird .

**False Positives:**

- `FG Rheinland` — partial — pred is substring of gold: `Urteil des FG Rheinland-Pfalz vom 14. Mai 2014 2 K 1454/13`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des FG Rheinland-Pfalz vom 14. Mai 2014 2 K 1454/13`(RS)

**Example 53** (doc_id: `59203`) (sent_id: `59203`)


Eine Abweichung liegt nicht schon dann vor , wenn das LSG eine höchstrichterliche Entscheidung nur unrichtig ausgelegt oder das Recht unrichtig angewandt hat , sondern erst , wenn das LSG Kriterien , die ein in der Norm genanntes Gericht aufgestellt hat , widersprochen , also andere Maßstäbe entwickelt hat .

**False Positives:**

- `LSG Kriterien` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 54** (doc_id: `59230`) (sent_id: `59230`)


I. Der Kläger begehrt eine Entschädigung wegen eines nach seiner Ansicht unangemessen langen Gerichtsverfahrens vor dem SG Gotha ( S 13 AL 118/98 ) und dem Thüringer LSG ( L 3 AL 229/00 ) .

**False Positives:**

- `SG Gotha` — partial — pred is substring of gold: `SG Gotha ( S 13 AL 118/98 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `SG Gotha ( S 13 AL 118/98 )`(RS)
- `Thüringer LSG ( L 3 AL 229/00 )`(RS)

**Example 55** (doc_id: `59341`) (sent_id: `59341`)


Nach Klageerhebung hat der Kläger im einstweiligen Rechtsschutz obsiegt ( LSG Beschluss vom 3. 3. 2010 - L 4 KR 44/10 B ER ) und erhält seither weiterhin die IVIG-Behandlung zu Lasten der Beklagten .

**False Positives:**

- `LSG Beschluss` — partial — pred is substring of gold: `LSG Beschluss vom 3. 3. 2010 - L 4 KR 44/10 B ER`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `LSG Beschluss vom 3. 3. 2010 - L 4 KR 44/10 B ER`(RS)

**Example 56** (doc_id: `59356`) (sent_id: `59356`)


Über die sofortige Beschwerde der Nebenklägerin K. T. gegen die im Urteil des Landgerichts Frankfurt am Main vom 15. August 2017 getroffene Kostenentscheidung hat das Oberlandesgericht Frankfurt am Main zu entscheiden .

**False Positives:**

- `Oberlandesgericht Frankfurt` — partial — pred is substring of gold: `Oberlandesgericht Frankfurt am Main`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K. T.`(PER)
- `Landgerichts Frankfurt am Main`(ORG)
- `Oberlandesgericht Frankfurt am Main`(ORG)

**Example 57** (doc_id: `59517`) (sent_id: `59517`)


Bleibt dieses Verfahren erfolglos , kann der Beschäftigte sodann in einem Rechtsstreit vor den Sozialgerichten die Verpflichtung der Einzugsstelle zu einem entsprechenden Beitragseinzug gerichtlich klären lassen ( vgl BSG Urteil vom 12. 9. 1995 - 12 RK 63/94 - SozR 3 - 2400 § 28 h Nr 5 ; BSG Urteil vom 26. 9. 1996 - 12 RK 37/95 - SozR 3 - 2400 § 28 h Nr 7 ) .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 12. 9. 1995 - 12 RK 63/94 - SozR 3 - 2400 § 28 h Nr 5`
- `BSG Urteil` — similar text (different position): `BSG Urteil vom 12. 9. 1995 - 12 RK 63/94 - SozR 3 - 2400 § 28 h Nr 5`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Urteil vom 12. 9. 1995 - 12 RK 63/94 - SozR 3 - 2400 § 28 h Nr 5`(RS)
- `BSG Urteil vom 26. 9. 1996 - 12 RK 37/95 - SozR 3 - 2400 § 28 h Nr 7`(RS)

**Example 58** (doc_id: `59525`) (sent_id: `59525`)


Auf die dagegen erhobene Klage hat das Finanzgericht ( FG ) die Einkommensteuer jeweils auf ... € herabgesetzt ( FG Mecklenburg-Vorpommern , Urteil vom 13. Januar 2016 1 K 453/13 , abgedruckt in Entscheidungen der Finanzgerichte - EFG - 2016 , 576 ) .

**False Positives:**

- `FG Mecklenburg` — partial — pred is substring of gold: `FG Mecklenburg-Vorpommern , Urteil vom 13. Januar 2016 1 K 453/13 , abgedruckt in Entscheidungen der Finanzgerichte - EFG - 2016 , 576`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `FG Mecklenburg-Vorpommern , Urteil vom 13. Januar 2016 1 K 453/13 , abgedruckt in Entscheidungen der Finanzgerichte - EFG - 2016 , 576`(RS)

**Example 59** (doc_id: `59574`) (sent_id: `59574`)


Nach ständiger Rechtsprechung des BSG sind Fragen zu einer Rechtsnorm , bei der es sich um ausgelaufenes Recht handelt , regelmäßig nicht von grundsätzlicher Bedeutung , weil die grundsätzliche Bedeutung einer Rechtsfrage daraus erwächst , dass ihre Klärung nicht nur für den Einzelfall , sondern im Interesse der Fortbildung des Rechts oder seiner einheitlichen Auslegung erforderlich ist ( vgl BSG Beschluss vom 29. 11. 2017 - B 6 KA 51/17 B - Juris RdNr 15 ; BSG Beschluss vom 28. 6. 2017 - B 6 KA 84/16 B - Juris RdNr 6 ; BSG Beschluss vom 19. 7. 2012 - B 1 KR 65/11 B - SozR 4 - 1500 § 160a Nr 32 RdNr 10 mwN ) .

**False Positives:**

- `BSG Beschluss` — similar text (different position): `BSG`
- `BSG Beschluss` — similar text (different position): `BSG`
- `BSG Beschluss` — similar text (different position): `BSG`

> overlaps gold: 3  |  likely missing annotation: 0

**Gold Entities:**

- `BSG`(ORG)
- `BSG Beschluss vom 29. 11. 2017 - B 6 KA 51/17 B - Juris RdNr 15`(RS)
- `BSG Beschluss vom 28. 6. 2017 - B 6 KA 84/16 B - Juris RdNr 6`(RS)
- `BSG Beschluss vom 19. 7. 2012 - B 1 KR 65/11 B - SozR 4 - 1500 § 160a Nr 32 RdNr 10`(RS)

**Example 60** (doc_id: `59626`) (sent_id: `59626`)


In allen Rechtszügen war der Auffangstreitwert festzusetzen ( vgl zB BSG Urteil vom 11. 3. 2009 - B 12 R 11/07 R - BSGE 103 , 17 = SozR 4 - 2400 § 7a Nr 2 RdNr 30 ; BSG Urteil vom 4. 6. 2009 - B 12 R 6/08 R - USK 2009 - 72 ; BSG Urteil vom 30. 10. 2013 - B 12 KR 17/11 R - Die Beiträge Beilage 2014 , 387 , 400 ) , weil Gegenstand des Rechtsstreits nicht ( auch ) eine Beitrags ( nach ) forderung war .

**False Positives:**

- `BSG Urteil` — partial — pred is substring of gold: `BSG Urteil vom 11. 3. 2009 - B 12 R 11/07 R - BSGE 103 , 17 = SozR 4 - 2400 § 7a Nr 2 RdNr 30`
- `BSG Urteil` — similar text (different position): `BSG Urteil vom 11. 3. 2009 - B 12 R 11/07 R - BSGE 103 , 17 = SozR 4 - 2400 § 7a Nr 2 RdNr 30`
- `BSG Urteil` — similar text (different position): `BSG Urteil vom 11. 3. 2009 - B 12 R 11/07 R - BSGE 103 , 17 = SozR 4 - 2400 § 7a Nr 2 RdNr 30`

> overlaps gold: 3  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Urteil vom 11. 3. 2009 - B 12 R 11/07 R - BSGE 103 , 17 = SozR 4 - 2400 § 7a Nr 2 RdNr 30`(RS)
- `BSG Urteil vom 4. 6. 2009 - B 12 R 6/08 R - USK 2009 - 72`(RS)
- `BSG Urteil vom 30. 10. 2013 - B 12 KR 17/11 R - Die Beiträge Beilage 2014 , 387 , 400`(RS)

**Example 61** (doc_id: `59662`) (sent_id: `59662`)


Das BSG hat dementsprechend das Vorliegen einer lebensbedrohlichen oder regelmäßig tödlich verlaufenden oder wertungsmäßig hiermit vergleichbaren Erkrankung ua verneint bei einem Prostatakarzinom im Anfangsstadium ohne Hinweise auf metastatische Absiedlungen ( BSG SozR 4 - 2500 § 27 Nr 8 - Interstitielle Brachytherapie ) , bei einem in schwerwiegender Form bestehenden Restless-Legs-Syndrom mit massiven Schlafstörungen und daraus resultierenden erheblichen körperlichen und seelischen Beeinträchtigungen sowie Suizidandrohung ( BSG SozR 4 - 2500 § 31 Nr 6 RdNr 11 , 18 - Cabaseril ) , bei Friedreich'scher Ataxie - Zunahme der Wanddicke des Herzmuskels , allgemeiner Leistungsminderung und langfristig eingeschränkter Lebenserwartung ( BSG SozR 4 - 2500 § 31 Nr 8 RdNr 17 ff - Mnesis ) und bei Zungenschwellungen mit Erstickungsgefahr im Rahmen von Urtikaria-Episoden , die medikamentös mit Hilfe eines stets mitgeführten Notfallsets zu beherrschen waren ( vgl BSG SozR 4 - 2500 § 31 Nr 28 , zur Veröffentlichung auch in BSGE vorgesehen , RdNr 21 ; zustimmend BVerfG Beschluss vom 11. 4. 2017 - 1 BvR 452/17 - NJW 2017 , 2096 = NZS 2017 , 582 ) .

**False Positives:**

- `BVerfG Beschluss` — partial — pred is substring of gold: `BVerfG Beschluss vom 11. 4. 2017 - 1 BvR 452/17 - NJW 2017 , 2096 = NZS 2017 , 582`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG`(ORG)
- `BSG SozR 4 - 2500 § 27 Nr 8 - Interstitielle Brachytherapie`(RS)
- `BSG SozR 4 - 2500 § 31 Nr 6 RdNr 11 , 18 - Cabaseril`(RS)
- `BSG SozR 4 - 2500 § 31 Nr 8 RdNr 17 ff - Mnesis`(RS)
- `BSG SozR 4 - 2500 § 31 Nr 28 , zur Veröffentlichung auch in BSGE vorgesehen , RdNr 21`(RS)
- `BVerfG Beschluss vom 11. 4. 2017 - 1 BvR 452/17 - NJW 2017 , 2096 = NZS 2017 , 582`(RS)

**Example 62** (doc_id: `59702`) (sent_id: `59702`)


Darüber hinaus ist in der Rechtsprechung bereits geklärt , dass die Sozialgerichte bei ihrer Feststellung , ob ein Arzt ein Delikt begangen und damit seine vertragsärztlichen Pflichten gröblich verletzt und sich als ungeeignet für die vertragsärztliche Tätigkeit erwiesen hat , vorliegende bestandskräftige Entscheidungen anderer Gerichte und auch die Ergebnisse staatsanwaltschaftlicher Ermittlungen verwerten dürfen ( BSG Beschluss vom 2. 4. 2014 - B 6 KA 58/13 B - Juris RdNr 17 ; BSG Beschluss vom 27. 6. 2007 - B 6 KA 20/07 B - Juris RdNr 12 ; BSG Beschluss vom 5. 5. 2010 - B 6 KA 32/09 B - MedR 2011 , 307 RdNr 9 ; BSG Beschluss vom 31. 8. 1990 - 6 BKa 33/90 - Juris RdNr 5 ; BSG Beschluss vom 27. 2. 1992 - 6 BKa 15/91 - Juris RdNr 27 ) .

**False Positives:**

- `BSG Beschluss` — partial — pred is substring of gold: `BSG Beschluss vom 2. 4. 2014 - B 6 KA 58/13 B - Juris RdNr 17`
- `BSG Beschluss` — similar text (different position): `BSG Beschluss vom 2. 4. 2014 - B 6 KA 58/13 B - Juris RdNr 17`
- `BSG Beschluss` — similar text (different position): `BSG Beschluss vom 2. 4. 2014 - B 6 KA 58/13 B - Juris RdNr 17`
- `BSG Beschluss` — similar text (different position): `BSG Beschluss vom 2. 4. 2014 - B 6 KA 58/13 B - Juris RdNr 17`
- `BSG Beschluss` — similar text (different position): `BSG Beschluss vom 2. 4. 2014 - B 6 KA 58/13 B - Juris RdNr 17`

> overlaps gold: 5  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Beschluss vom 2. 4. 2014 - B 6 KA 58/13 B - Juris RdNr 17`(RS)
- `BSG Beschluss vom 27. 6. 2007 - B 6 KA 20/07 B - Juris RdNr 12`(RS)
- `BSG Beschluss vom 5. 5. 2010 - B 6 KA 32/09 B - MedR 2011 , 307 RdNr 9`(RS)
- `BSG Beschluss vom 31. 8. 1990 - 6 BKa 33/90 - Juris RdNr 5`(RS)
- `BSG Beschluss vom 27. 2. 1992 - 6 BKa 15/91 - Juris RdNr 27`(RS)

**Example 63** (doc_id: `59718`) (sent_id: `59718`)


Danach bedarf es für die Entscheidung der Kostenbeamten , ob Pauschgebühren zu erheben sind oder davon wegen Gerichtskostenfreiheit abzusehen ist , keiner Ermittlungen , ob eine Aufgabenübertragung vom Jobcenter auf die BA gemäß § 44b Abs 4 und 5 SGB II im Einzelfall ordnungsgemäß vorgenommen worden ist ( ebenso Hessisches LSG Beschluss vom 27. 5. 2016 - L 2 SF 15/16 - Juris RdNr 14 ; aA Thüringer LSG Beschluss vom 19. 2. 2015 - L 6 SF 70/14 E - Juris RdNr 4 , 8 sowie Beschluss vom 11. 6. 2015 - L 6 SF 502/15 E - Juris RdNr 4 , 9 ff ) .

**False Positives:**

- `LSG Beschluss` — partial — pred is substring of gold: `Hessisches LSG Beschluss vom 27. 5. 2016 - L 2 SF 15/16 - Juris RdNr 14`
- `LSG Beschluss` — similar text (different position): `Hessisches LSG Beschluss vom 27. 5. 2016 - L 2 SF 15/16 - Juris RdNr 14`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BA`(ORG)
- `§ 44b Abs 4 und 5 SGB II`(NRM)
- `Hessisches LSG Beschluss vom 27. 5. 2016 - L 2 SF 15/16 - Juris RdNr 14`(RS)
- `Thüringer LSG Beschluss vom 19. 2. 2015 - L 6 SF 70/14 E - Juris RdNr 4 , 8`(RS)
- `Beschluss vom 11. 6. 2015 - L 6 SF 502/15 E - Juris RdNr 4 , 9 ff`(RS)

</details>

---

## `Specific Organization Names`

**F1:** 0.019 | **Precision:** 0.421 | **Recall:** 0.010  

**Format:** `regex`  
**Rule ID:** `46ac37c4`  
**Description:**
Matches specific organization names found in training data, including quoted brands, abbreviations, and multi-word names.

**Content:**
```
\b(?:D P T S GmbH|Becker Mining|Becker|Europ\u00e4ische Gerichtshof|EuGH|EU|BSG|GmSOGB|BVerfG|ZDS|Oberlandesgerichts D\u00fcsseldorf|FACEYOURMUSIC|Bundesrat|F\. D\.|\u201e JOOP \u201e|\u201e Arrow \u201e|\u201e BEAST \u201e|\u201e KONTRON \u201e|Bund Deutscher Verwaltungsrichter und Verwaltungsrichterinnen|BDVR|ARD|ZDF|Deutschlandfunk|AGH H\.|M \u2026|Bundesvereinigung der Arbeitgeberverb\u00e4nde|Generalstaatsanwalt in M\u00fcnchen|Oberlandesgericht M\u00fcnchen|Beruflichen Schulzentrum f\u00fcr Technik und Wirtschaft P|C- B\. V\.|S AG|C GmbH|VW Tiguan|Bundeskanzleramt|Landgericht F\. \( P\. \)|Oberverwaltungsgerichts f\u00fcr das Land Nordrhein-Westfalen|Gemeinsamen Senats der obersten Gerichtsh\u00f6fe des Bundes)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.421 | 0.010 | 0.019 | 19 | 8 | 11 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 8 | 11 | 768 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `56332`) (sent_id: `56332`)


aa ) Der Europäische Gerichtshof ist gesetzlicher Richter im Sinne von Art. 101 Abs. 1 Satz 2 GG .

| Predicted | Gold |
|---|---|
| `Europäische Gerichtshof` | `Europäische Gerichtshof` |

**Missed by this rule (FN):**

- `Art. 101 Abs. 1 Satz 2 GG` (NRM)

**Example 1** (doc_id: `56647`) (sent_id: `56647`)


Der Senat muss nach § 557 Abs. 3 Satz 1 ZPO dennoch prüfen , ob vernünftige Zweifel an der Tariffähigkeit und der Tarifzuständigkeit des ZDS bei Abschluss des TV AKS 2012 und des TV AKS 2014 bestehen .

| Predicted | Gold |
|---|---|
| `ZDS` | `ZDS` |

**Missed by this rule (FN):**

- `§ 557 Abs. 3 Satz 1 ZPO` (NRM)
- `TV AKS 2012` (REG)
- `TV AKS 2014` (REG)

**Example 2** (doc_id: `57409`) (sent_id: `57409`)


Dieser Zulassungsgrund ist erfüllt , wenn die Vorinstanz mit einem ihre Entscheidung tragenden abstrakten Rechtssatz in Anwendung derselben Rechtsvorschrift einem ebensolchen Rechtssatz , der in der Rechtsprechung des Bundesverwaltungsgerichts , des Gemeinsamen Senats der obersten Gerichtshöfe des Bundes oder des Bundesverfassungsgerichts aufgestellt worden ist , widersprochen hat .

| Predicted | Gold |
|---|---|
| `Gemeinsamen Senats der obersten Gerichtshöfe des Bundes` | `Gemeinsamen Senats der obersten Gerichtshöfe des Bundes` |

**Missed by this rule (FN):**

- `Bundesverwaltungsgerichts` (ORG)
- `Bundesverfassungsgerichts` (ORG)

**Example 3** (doc_id: `57628`) (sent_id: `57628`)


Zur Begründung führte sie ergänzend zu dem Vorbringen im Verwaltungsverfahren Folgendes aus : Die Tätigkeit des Klägers im Bundeskanzleramt sei nicht bei der Beurteilung zu berücksichtigen gewesen , weil dies nur für den Fall der Abordnung vorgesehen sei , der Kläger jedoch an das Bundeskanzleramt versetzt worden sei .

| Predicted | Gold |
|---|---|
| `Bundeskanzleramt` | `Bundeskanzleramt` |
| `Bundeskanzleramt` | `Bundeskanzleramt` |

**Example 4** (doc_id: `58031`) (sent_id: `58031`)


Eine Abweichung ( Divergenz ) ist nur dann hinreichend dargelegt , wenn aufgezeigt wird , mit welcher genau bestimmten entscheidungserheblichen rechtlichen Aussage die angegriffene Entscheidung des LSG von welcher ebenfalls genau bezeichneten rechtlichen Aussage des BSG , des Gemeinsamen Senats der obersten Gerichtshöfe des Bundes ( GmSOGB ) oder des BVerfG abweicht ( BSG SozR 1500 § 160a Nr 21 , 29 und 54 ) .

| Predicted | Gold |
|---|---|
| `Gemeinsamen Senats der obersten Gerichtshöfe des Bundes` | `Gemeinsamen Senats der obersten Gerichtshöfe des Bundes` |
| `GmSOGB` | `GmSOGB` |

**Missed by this rule (FN):**

- `BSG` (ORG)
- `BVerfG` (ORG)
- `BSG SozR 1500 § 160a Nr 21 , 29 und 54` (RS)

**Example 5** (doc_id: `59297`) (sent_id: `59297`)


( 2 ) Bei Wortzeichen wie dem Anmeldezeichen , die gehört und gelesen werden können , hat der Europäische Gerichtshof eine schutzbegründende Abweichung von der jeweiligen Sachangabe sowohl im akustischen wie auch im visuellen Gesamteindruck als erforderlich angesehen ( vgl. EuGH GRUR 2004 , 674 Rdnr. 99 - Postkantoor ; GRUR 2004 , 680 Rdnr. 40 – BIOMILD ) .

| Predicted | Gold |
|---|---|
| `Europäische Gerichtshof` | `Europäische Gerichtshof` |

**Missed by this rule (FN):**

- `EuGH GRUR 2004 , 674 Rdnr. 99 - Postkantoor` (RS)
- `GRUR 2004 , 680 Rdnr. 40 – BIOMILD` (RS)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53613`) (sent_id: `53613`)


Allerdings kann der Zeichenbestandteil „ Becker “ nicht als „ Allerweltsname “ angesehen werden .

**False Positives:**

- `Becker` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `53820`) (sent_id: `53820`)


Wenn es - wie vorliegend - an einer ausdrücklichen Sonderzuweisung für den zuständigen Rechtsweg fehlt , bestimmt sich die gerichtliche Zuständigkeit nach der Natur des Rechtsverhältnisses , aus dem der Klageanspruch hergeleitet wird ( stRspr ; Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2 ; GmSOGB vom 10. 4. 1986 - GmS-OGB 1/85 - BGHZ 97 , 312 = SozR 1500 § 51 Nr 39 ; GmSOGB vom 29. 10. 1987 - GmS-OGB 1/86 - BGHZ 102 , 280 = SozR 1500 § 51 Nr 47 ; GmSOGB vom 10. 7. 1989 - GmS-OGB 1/88 - BGHZ 108 , 284 = SozR 1500 § 51 Nr 53 ; zum Rechtsverhältnis zwischen den Beteiligten als entscheidendes Kriterium zur Beurteilung des Rechtswegs vgl letztens etwa BSG vom 21. 7. 2014 - B 14 SF 1/14 R - SozR 4 - 1500 § 51 Nr 12 RdNr 8 ; BSG vom 25. 10. 2017 - B 7 SF 1/16 R - juris , RdNr 6 ) .

**False Positives:**

- `GmSOGB` — partial — pred is substring of gold: `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`
- `GmSOGB` — similar text (different position): `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`
- `GmSOGB` — similar text (different position): `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`
- `GmSOGB` — similar text (different position): `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`

> overlaps gold: 4  |  likely missing annotation: 0

**Gold Entities:**

- `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`(RS)
- `GmSOGB vom 10. 4. 1986 - GmS-OGB 1/85 - BGHZ 97 , 312 = SozR 1500 § 51 Nr 39`(RS)
- `GmSOGB vom 29. 10. 1987 - GmS-OGB 1/86 - BGHZ 102 , 280 = SozR 1500 § 51 Nr 47`(RS)
- `GmSOGB vom 10. 7. 1989 - GmS-OGB 1/88 - BGHZ 108 , 284 = SozR 1500 § 51 Nr 53`(RS)
- `BSG vom 21. 7. 2014 - B 14 SF 1/14 R - SozR 4 - 1500 § 51 Nr 12 RdNr 8`(RS)
- `BSG vom 25. 10. 2017 - B 7 SF 1/16 R - juris , RdNr 6`(RS)

**Example 2** (doc_id: `55131`) (sent_id: `55131`)


Becker

**False Positives:**

- `Becker` — type mismatch — same span as gold: `Becker`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Becker`(PER)

**Example 3** (doc_id: `55256`) (sent_id: `55256`)


Becker

**False Positives:**

- `Becker` — type mismatch — same span as gold: `Becker`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Becker`(PER)

**Example 4** (doc_id: `57469`) (sent_id: `57469`)


aa ) Im Schrifttum ist zwar anerkannt , dass die Rechtsfolgen bei Verstößen gegen § 63 AO unter Anwendung des rechtsstaatlich fundierten Verhältnismäßigkeitsprinzips am Ausmaß und Gewicht der Pflichtverletzung auszurichten sind ( Seer in Tipke / Kruse , a. a. O. , § 63 AO Rz 12 ; Hüttemann , a. a. O. , Rz 4.163 ; Becker , DStR 2010 , 953 , unter 2.2.1 ; Jäschke , DStR 2009 , 1669 , Rz 2.4 ; Bott in Schauhoff , Handbuch der Gemeinnützigkeit , § 10 Rz 80 ) .

**False Positives:**

- `Becker` — partial — pred is substring of gold: `Becker , DStR 2010 , 953 , unter 2.2.1`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 63 AO`(NRM)
- `Seer in Tipke / Kruse , a. a. O. , § 63 AO Rz 12`(LIT)
- `Hüttemann , a. a. O. , Rz 4.163`(LIT)
- `Becker , DStR 2010 , 953 , unter 2.2.1`(LIT)
- `Jäschke , DStR 2009 , 1669 , Rz 2.4`(LIT)
- `Bott in Schauhoff , Handbuch der Gemeinnützigkeit , § 10 Rz 80`(LIT)

**Example 5** (doc_id: `57613`) (sent_id: `57613`)


Ob der Kläger damit eine Rechtsfrage hinreichend bezeichnet hat , die auf die Auslegung eines gesetzlichen Tatbestandsmerkmals abzielt ( vgl Becker , SGb 2007 , 261 , 265 zu Fn 42 mwN ) , kann hier dahinstehen .

**False Positives:**

- `Becker` — partial — pred is substring of gold: `Becker , SGb 2007 , 261 , 265 zu Fn 42`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Becker , SGb 2007 , 261 , 265 zu Fn 42`(LIT)

**Example 6** (doc_id: `58498`) (sent_id: `58498`)


Becker

**False Positives:**

- `Becker` — type mismatch — same span as gold: `Becker`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Becker`(PER)

**Example 7** (doc_id: `58637`) (sent_id: `58637`)


Diese Zielsetzung würde unterlaufen , wenn ausschließlich an Staaten gerichtete Normen des Völkerrechts , die - wie das Gewaltverbot - nicht bereits von sich aus eine subjektive Schutzwirkung aufweisen , über Art. 25 Satz 2 Halbsatz 2 GG inhaltlich verändert , nämlich individualisiert , in das Bundesrecht übernommen würden und dadurch letztlich eine über die Intention des Völkerrechts hinausgehende innerstaatliche Rechtslage geschaffen würde ( Kunig , in : Graf Vitzthum / Proelß , Völkerrecht , 7. Aufl. 2016 , S. 61 < 118 ff. ; Rn. 150 ff. > ; vgl. auch Hofmann , in : Umbach / Clemens , GG , 2002 , Art. 25 Rn. 25 ; Rojahn , in : von Münch / Kunig , GG , 6. Aufl. 2012 , Art. 25 Rn. 41 , 49 f. ; Cremer , Allgemeine Regeln des Völkerrechts , in : Isensee / Kirchhof , HStR XI , 3. Aufl. 2013 , § 235 Rn. 32 ; Kessler / Salomon , DÖV 2014 , S. 283 < 288 f. > ; Wollenschläger , in : Dreier , GG , Bd. 2 , 3. Aufl. 2015 , Art. 25 Rn. 36 ; Herdegen , in : Maunz / Dürig , GG , Art. 25 Rn. 90 < September 2017 > ; Schorkopf , Staatsrecht der internationalen Beziehungen , 2017 , S. 162 f. , Rn. 40 ; für das Gewaltverbot abweichend Fischer-Lescano / Hanschmann , in : Becker / Braun / Deiseroth , Frieden durch Recht ? , 2010 , S. 181 < 189 ff. > ; offengelassen in BVerwGE 154 , 328 < 347 > ) .

**False Positives:**

- `Becker` — partial — pred is substring of gold: `Fischer-Lescano / Hanschmann , in : Becker / Braun / Deiseroth , Frieden durch Recht ? , 2010 , S. 181 < 189 ff. >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 25 Satz 2 Halbsatz 2 GG`(NRM)
- `Kunig , in : Graf Vitzthum / Proelß , Völkerrecht , 7. Aufl. 2016 , S. 61 < 118 ff. ; Rn. 150 ff. >`(LIT)
- `Hofmann , in : Umbach / Clemens , GG , 2002 , Art. 25 Rn. 25`(LIT)
- `Rojahn , in : von Münch / Kunig , GG , 6. Aufl. 2012 , Art. 25 Rn. 41 , 49 f.`(LIT)
- `Cremer , Allgemeine Regeln des Völkerrechts , in : Isensee / Kirchhof , HStR XI , 3. Aufl. 2013 , § 235 Rn. 32`(LIT)
- `Kessler / Salomon , DÖV 2014 , S. 283 < 288 f. >`(LIT)
- `Wollenschläger , in : Dreier , GG , Bd. 2 , 3. Aufl. 2015 , Art. 25 Rn. 36`(LIT)
- `Herdegen , in : Maunz / Dürig , GG , Art. 25 Rn. 90 < September 2017 >`(LIT)
- `Schorkopf , Staatsrecht der internationalen Beziehungen , 2017 , S. 162 f. , Rn. 40`(LIT)
- `Fischer-Lescano / Hanschmann , in : Becker / Braun / Deiseroth , Frieden durch Recht ? , 2010 , S. 181 < 189 ff. >`(LIT)
- `BVerwGE 154 , 328 < 347 >`(RS)

</details>

---

## `Bundeswehr`

**F1:** 0.017 | **Precision:** 0.778 | **Recall:** 0.009  

**Format:** `regex`  
**Rule ID:** `ed9daf2d`  
**Description:**
Matches 'Bundeswehr' and its variations.

**Content:**
```
\b(Bundeswehr|Bundeswehrkommando\s+[A-Za-z\s]+|Ver\u00e4nderungsmanagement\s*Luftwaffe|Kommando\s*Luftwaffe|Luftwaffe)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.778 | 0.009 | 0.017 | 9 | 7 | 2 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 7 | 2 | 658 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `54964`) (sent_id: `54964`)


Diese besteht darin , dazu beizutragen , einen ordnungsgemäßen Dienstbetrieb wiederherzustellen und / oder aufrechtzuerhalten ( " Wiederherstellung und Sicherung der Integrität , des Ansehens und der Disziplin in der Bundeswehr " , vgl. dazu BVerwG , Urteil vom 11. Juni 2008 - 2 WD 11.07 - Buchholz 450.2 § 38 WDO 2002 Nr. 26 Rn. 23 m. w. N. ) .

| Predicted | Gold |
|---|---|
| `Bundeswehr` | `Bundeswehr` |

**Missed by this rule (FN):**

- `BVerwG , Urteil vom 11. Juni 2008 - 2 WD 11.07 - Buchholz 450.2 § 38 WDO 2002 Nr. 26 Rn. 23` (RS)

**Example 1** (doc_id: `56366`) (sent_id: `56366`)


Einen diesbezüglichen Rechtsanwendungserlass hat das Ministerium in Gestalt der für das Arbeitszeitrecht der Soldatinnen und Soldaten zuständigen Stelle ( BMVg FüSK III 1 ) für die im ENJJPT eingesetzten Fluglehrer der Luftwaffe nicht verfügt .

| Predicted | Gold |
|---|---|
| `Luftwaffe` | `Luftwaffe` |

**Missed by this rule (FN):**

- `BMVg FüSK III 1` (REG)

**Example 2** (doc_id: `57150`) (sent_id: `57150`)


" Die Fluglehrer der Luftwaffe , die als Instructor Pilots ( IP ) zur Durchführung der fliegerischen Ausbildung im ENJJPT abgestellt werden , werden gemäß diesem Memorandum of Understanding ( MoU ) und Program Plan of Operation ( PO ) innerhalb der international gemischten Organisation ( organizational / management structure ) aus insgesamt dreizehn Teilnehmerstaaten eingesetzt .

| Predicted | Gold |
|---|---|
| `Luftwaffe` | `Luftwaffe` |

**Example 3** (doc_id: `57970`) (sent_id: `57970`)


Auch befinde sich der für den Antragsteller vorgesehene Dienstposten nicht im Organisationsbereich Luftwaffe , sondern im Organisationsbereich Heer ; die Besetzungszuständigkeit des Dienstpostens ( Luftwaffe ) ändere daran nichts .

| Predicted | Gold |
|---|---|
| `Luftwaffe` | `Luftwaffe` |
| `Luftwaffe` | `Luftwaffe` |

**Example 4** (doc_id: `59139`) (sent_id: `59139`)


Der Dienstposten ... beim ... in C. , auf den der Antragsteller versetzt wurde , unterliegt - was unstrittig ist - der Besetzungszuständigkeit der Luftwaffe und war ausweislich der gegenständlichen Versetzungsverfügung vom 21. März 2017 jedenfalls in dem maßgeblichen Zeitpunkt der Versetzung dem Organisationsbereich Luftwaffe zugeordnet .

| Predicted | Gold |
|---|---|
| `Luftwaffe` | `Luftwaffe` |
| `Luftwaffe` | `Luftwaffe` |

**Missed by this rule (FN):**

- `...` (ORG)
- `C.` (LOC)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `54769`) (sent_id: `54769`)


Die Dauer der Auslandsverwendung des Antragstellers beim Bundeswehrkommando USA und Kanada in A vom 1. Juli 2014 bis zum 30. Juni 2017 entspricht exakt der regulären Dauer einer Tour of Duty .

**False Positives:**

- `Bundeswehrkommando USA und Kanada in A vom ` — partial — gold is substring of pred: `Bundeswehrkommando USA und Kanada`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundeswehrkommando USA und Kanada`(ORG)
- `A`(LOC)

**Example 1** (doc_id: `57722`) (sent_id: `57722`)


Das dienstliche Bedürfnis für die Wegversetzung des Antragstellers ist darüber hinaus unter dem Gesichtspunkt gegeben , dass seine befristete Auslandsverwendung beim Bundeswehrkommando USA und Kanada zum 30. Juni 2017 geendet hat .

**False Positives:**

- `Bundeswehrkommando USA und Kanada zum ` — partial — gold is substring of pred: `Bundeswehrkommando USA und Kanada`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundeswehrkommando USA und Kanada`(ORG)

</details>

---

## `Specific Court Names with Location`

**F1:** 0.015 | **Precision:** 1.000 | **Recall:** 0.007  

**Format:** `regex`  
**Rule ID:** `61dc1199`  
**Description:**
Matches court names that include a location, handling genitive forms.

**Content:**
```
\b(Sozialgerichts Nürnberg|Bayerischen Landessozialgerichts)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.007 | 0.015 | 6 | 6 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 6 | 0 | 556 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `55856`) (sent_id: `55856`)


Die Beschwerde der Klägerin gegen die Nichtzulassung der Revision im Urteil des Bayerischen Landessozialgerichts vom 24. Mai 2017 wird zurückgewiesen .

| Predicted | Gold |
|---|---|
| `Bayerischen Landessozialgerichts` | `Bayerischen Landessozialgerichts` |

**Example 1** (doc_id: `56530`) (sent_id: `56530`)


Die Beklagte beantragt , die Urteile des Bayerischen Landessozialgerichts vom 16. 12. 2015 und des Sozialgerichts München vom 16. 5. 2014 aufzuheben und die Klage abzuweisen .

| Predicted | Gold |
|---|---|
| `Bayerischen Landessozialgerichts` | `Bayerischen Landessozialgerichts` |

**Missed by this rule (FN):**

- `Sozialgerichts München` (ORG)

**Example 2** (doc_id: `56616`) (sent_id: `56616`)


Die Beklagte beantragt , das Urteil des Bayerischen Landessozialgerichts vom 21. Oktober 2015 aufzuheben und das Urteil des Sozialgerichts Nürnberg vom 25. Juni 2013 bezüglich der Beigeladenen zu 1. und 3. aufzuheben und die Klage abzuweisen .

| Predicted | Gold |
|---|---|
| `Bayerischen Landessozialgerichts` | `Bayerischen Landessozialgerichts` |

**Missed by this rule (FN):**

- `Sozialgerichts Nürnberg` (ORG)

**Example 3** (doc_id: `57203`) (sent_id: `57203`)


Die Revision der Klägerin gegen das Urteil des Bayerischen Landessozialgerichts vom 14. September 2016 wird zurückgewiesen .

| Predicted | Gold |
|---|---|
| `Bayerischen Landessozialgerichts` | `Bayerischen Landessozialgerichts` |

**Example 4** (doc_id: `58013`) (sent_id: `58013`)


das Urteil des Bayerischen Landessozialgerichts vom 3. Juni 2016 und den Gerichtsbescheid des Sozialgerichts Nürnberg vom 30. August 2013 sowie den Bescheid der Beklagten vom 19. März 2012 und den Widerspruchsbescheid vom 18. Juni 2012 aufzuheben und die Beklagte zu verurteilen , unter Rücknahme des Bescheides vom 4. Februar 2005 und des Widerspruchsbescheides vom 22. April 2005 die Zeit vom 1. September 1973 bis 30. Juni 1990 als Zeit der Zugehörigkeit zum Zusatzversorgungssystem der technischen Intelligenz und die hierin erzielten Arbeitsentgelte festzustellen .

| Predicted | Gold |
|---|---|
| `Bayerischen Landessozialgerichts` | `Bayerischen Landessozialgerichts` |

**Missed by this rule (FN):**

- `Sozialgerichts Nürnberg` (ORG)

**Example 5** (doc_id: `59022`) (sent_id: `59022`)


Die Beschwerde der Klägerin gegen die Nichtzulassung der Revision im Urteil des Bayerischen Landessozialgerichts vom 24. Mai 2017 wird zurückgewiesen .

| Predicted | Gold |
|---|---|
| `Bayerischen Landessozialgerichts` | `Bayerischen Landessozialgerichts` |

</details>

---

## `Court Names with Anonymized Location`

**F1:** 0.012 | **Precision:** 1.000 | **Recall:** 0.006  

**Format:** `regex`  
**Rule ID:** `206852a7`  
**Description:**
Matches court names followed by an anonymized single letter location (e.g., 'Amtsgericht P.', 'Landgericht F.').

**Content:**
```
\b(?:Amtsgericht|Landgericht|Verwaltungsgericht|Finanzgericht|Sozialgericht|Arbeitsgericht|Oberlandesgericht|Bundesverwaltungsgericht|Bundesfinanzhof|Bundesgerichtshof|Bundessozialgericht|Bundesarbeitsgericht|Bundespatentgericht|BVerwG|BGH|BSG|BAG|BVerfG|EuGH|EGMR|DPMA|BaFin|SG|FG|LAG|LSG|ArbG|Landessozialgericht|Landesarbeitsgericht|Landesverwaltungsgericht|Oberverwaltungsgericht|Verwaltungsgerichtshof|Truppendienstgericht|Anwaltsgerichtshof)\s+[A-Z]\.
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.006 | 0.012 | 5 | 5 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 5 | 0 | 656 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `54836`) (sent_id: `54836`)


Das Landgericht T. verurteilte den Kläger mit rechtskräftigem Urteil vom 7. August 2015 wegen mehrfachen Verstoßes gegen die vorgenannte Unterlassungserklärung zu einer Vertragsstrafe von 20.000,04 € .

| Predicted | Gold |
|---|---|
| `Landgericht T.` | `Landgericht T.` |

**Example 1** (doc_id: `54916`) (sent_id: `54916`)


Zur Wahrung ihrer eigenen Rechte habe sie damit nur die Möglichkeit gehabt , einen eigenen Antrag beim zuständigen Amtsgericht O. zu stellen .

| Predicted | Gold |
|---|---|
| `Amtsgericht O.` | `Amtsgericht O.` |

**Example 2** (doc_id: `55892`) (sent_id: `55892`)


Das Amtsgericht P. habe ausschließlich über den Zugewinnausgleichsanspruch des geschiedenen Ehemannes entschieden und dabei festgestellt , dass diesem ein solcher Anspruch nicht zustehe .

| Predicted | Gold |
|---|---|
| `Amtsgericht P.` | `Amtsgericht P.` |

**Example 3** (doc_id: `58095`) (sent_id: `58095`)


Durch die angegriffene Beschwerdeentscheidung wurde die vom Amtsgericht O. festgestellte Zahlungsverpflichtung ihres geschiedenen Ehemannes von 23.030,97 € nebst Zinsen auf 6.759,94 € nebst Zinsen herabgesetzt .

| Predicted | Gold |
|---|---|
| `Amtsgericht O.` | `Amtsgericht O.` |

**Example 4** (doc_id: `58592`) (sent_id: `58592`)


Am 10. April 2012 bestellte das Amtsgericht H. eine Betreuung für ihn hinsichtlich der Vertretung gegenüber Behörden , Sozialversicherungsträgern und der Rechtsanwaltskammer sowie der Schuldenregulierung einschließlich der Einleitung eines Insolvenzverfahrens .

| Predicted | Gold |
|---|---|
| `Amtsgericht H.` | `Amtsgericht H.` |

</details>

---

## `Union and Associations (Fixed)`

**F1:** 0.012 | **Precision:** 0.833 | **Recall:** 0.006  

**Format:** `regex`  
**Rule ID:** `c7b455dc`  
**Description:**
Matches specific unions and associations like ver.di, IG Metall, etc.

**Content:**
```
\b(Vereinten\s+Dienstleistungsgewerkschaft|ver\.di|Gewerkschaft\s+ver\.di|IG\s*Metall|Deutsche\s*Rentenversicherung\s*Bund|Deutsche\s*Rentenversicherung\s*Rheinland|Deutschen\s*Rentenversicherung|Kommunalen\s*Arbeitgeberverband\s*Sachsen\s*e\.\s*V\.|Bayerischen\s*Rechtsanwalts-\s+und\s+Steuerberaterversorgung|Bundessteuerberaterkammer|Deutsche\s*Gesellschaft\s*f\u00fcr\s+Technische\s+Zusammenarbeit\s*\(\s*GTZ\s*\)\s*GmbH|Deutsche\s*Entwicklungsdienst\s*gGmbH|Bund\s*f\u00fcr\s+Lebensmittelrecht\s*und\s+Lebenskunde\s*e\.\s*V\.|Verbraucherzentrale\s*Bundesverband\s*e\.\s*V\.|foodwatch\s*e\.\s*V\.|Deutsche\s*Verband\s*Tiernahrung\s*e\.\s*V\.|Bundesvereinigung\s*der\s*Deutschen\s*Ern\u00e4hrungsindustrie\s*e\.\s*V\.|ZDS|GTS\s*GmbH\s*&\s+Co\.\s*KG|TGAOK|T\u00fcm\s*Bel\s*Sen|Europ\u00e4ischen\s*Kommission|Europ\u00e4ischen\s*Gerichtshofes|Europ\u00e4ischen\s*Gerichtshof|Europ\u00e4ischen\s*Gerichtshofes|Gewerkschaft\s+Erziehung\s+und\s+Wissenschaft|GEW|X-EWIV|VCS|Centralen\s+Marketing-Gesellschaft\s+der\s+deutschen\s+Agrarwirtschaft\s+mbH|CMA|Schleswig-Holsteinische\s+Oberverwaltungsgericht|Ausw\u00e4rtigen\s+Amtes|Deutschen\s+Stiftung\s+f\u00fcr\s+Internationale|Landgericht\s+Potsdam|Sozialgerichts\s+<\s*SG\s*>\s+Hildesheim|Landessozialgerichts\s+<\s*LSG\s*>\s+Niedersachsen-Bremen|11\.\s+Senats\s+des\s+LSG\s+Mecklenburg-Vorpommern|S\u00e4chsischen\s+LSG|Europ\u00e4ischen\s+Parlaments|Bundesministerium\s+des\s+Innern\s*,\s*f\u00fcr\s+Bau\s+und\s+Heimat|Ministerium\s+der\s+Justiz\s+des\s+Landes\s+Nordrhein-Westfalen|Bundesamt\s+f\u00fcr\s+Migration\s+und\s+Fl\u00fcchtlinge|W\u00a1\s+R\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.833 | 0.006 | 0.012 | 6 | 5 | 1 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 5 | 1 | 767 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53658`) (sent_id: `53658`)


E1a BRADLER , Christian : Check the GTS MAPLA system – additional information , Seiten 1 - 9 , © GTS GmbH & Co. KG , 11/2007 ;

| Predicted | Gold |
|---|---|
| `GTS GmbH & Co. KG` | `GTS GmbH & Co. KG` |

**Missed by this rule (FN):**

- `BRADLER , Christian` (PER)
- `GTS` (ORG)

**Example 1** (doc_id: `53858`) (sent_id: `53858`)


E1b BRADLER , Christian : Check the GTS MAPLA system – additional information , Seiten 1 - 9 , © GTS GmbH & Co. KG , 11/2007

| Predicted | Gold |
|---|---|
| `GTS GmbH & Co. KG` | `GTS GmbH & Co. KG` |

**Missed by this rule (FN):**

- `BRADLER , Christian` (PER)
- `GTS` (ORG)

**Example 2** (doc_id: `57124`) (sent_id: `57124`)


1. Der Deutsche Gewerkschaftsbund und ver.di meinen , die einschränkende Auslegung des § 14 Abs. 2 Satz 2 TzBfG durch das Bundesarbeitsgericht überschreite die Grenze zulässiger Rechtsfortbildung .

| Predicted | Gold |
|---|---|
| `ver.di` | `ver.di` |

**Missed by this rule (FN):**

- `Deutsche Gewerkschaftsbund` (ORG)
- `§ 14 Abs. 2 Satz 2 TzBfG` (NRM)
- `Bundesarbeitsgericht` (ORG)

**Example 3** (doc_id: `57679`) (sent_id: `57679`)


IV. Zu Vorlage und Verfassungsbeschwerde haben der Deutsche Gewerkschaftsbund ( DGB ) , die Vereinte Dienstleistungsgewerkschaft ( ver.di ) , die Bundesvereinigung der Deutschen Arbeitgeberverbände e. V. ( BDA ) , das Bundesarbeitsgericht und die Beklagten der Ausgangsverfahren Stellung genommen .

| Predicted | Gold |
|---|---|
| `ver.di` | `ver.di` |

**Missed by this rule (FN):**

- `Deutsche Gewerkschaftsbund` (ORG)
- `DGB` (ORG)
- `Vereinte Dienstleistungsgewerkschaft` (ORG)
- `Bundesvereinigung der Deutschen Arbeitgeberverbände e. V.` (ORG)
- `BDA` (ORG)
- `Bundesarbeitsgericht` (ORG)

**Example 4** (doc_id: `58927`) (sent_id: `58927`)


Der in § 1 KAT erwähnte Kirchliche Tarifvertrag Diakonie ( KTD , GVOBl. NEK S. 317 ) wurde bereits am 15. August 2002 zwischen dem Verband kirchlicher und diakonischer Anstellungsträger Nordelbien einerseits und der Gewerkschaft Kirche und Diakonie , der IG Bauen-Agrar-Umwelt , Bundesvorstand , sowie von ver.di andererseits geschlossen .

| Predicted | Gold |
|---|---|
| `ver.di` | `ver.di` |

**Missed by this rule (FN):**

- `§ 1 KAT` (REG)
- `Kirchliche Tarifvertrag Diakonie ( KTD , GVOBl. NEK S. 317 )` (REG)
- `Verband kirchlicher und diakonischer Anstellungsträger Nordelbien` (ORG)
- `Gewerkschaft Kirche und Diakonie` (ORG)
- `IG Bauen-Agrar-Umwelt , Bundesvorstand` (ORG)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `57554`) (sent_id: `57554`)


festzustellen , dass auf das Arbeitsverhältnis der Parteien der Kirchliche Arbeitnehmerinnen Tarifvertrag , abgeschlossen zwischen dem Verband kirchlicher und diakonischer Anstellungsträger Nordelbien sowie der Gewerkschaft Kirche und Diakonie und ver.di , Landesbezirke Hamburg und Nord , andererseits vom 1. Dezember 2006 Anwendung finde .

**False Positives:**

- `ver.di` — partial — pred is substring of gold: `ver.di , Landesbezirke Hamburg und Nord`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Kirchliche Arbeitnehmerinnen Tarifvertrag`(REG)
- `Verband kirchlicher und diakonischer Anstellungsträger Nordelbien`(ORG)
- `Gewerkschaft Kirche und Diakonie`(ORG)
- `ver.di , Landesbezirke Hamburg und Nord`(ORG)

</details>

---

## `Specific Court Genitives with Location`

**F1:** 0.012 | **Precision:** 0.556 | **Recall:** 0.006  

**Format:** `regex`  
**Rule ID:** `ffb8c069`  
**Description:**
Matches specific court names with genitive endings followed by a location, including hyphenated locations like Berlin-Brandenburg and Schleswig-Holstein.

**Content:**
```
\b(?:Landessozialgerichts\s+(?:Baden-W\u00fcrttemberg|Niedersachsen-Bremen|S\u00e4chsischen|Th\u00fcringer|Rheinland-Pfalz|Nordrhein-Westfalen|Schleswig-Holsteinischen|Hessischen|Berlin-Brandenburg)|Landgerichts\s+(?:Hamburg|Ansbach|Darmstadt|Memmingen|D\u00fcsseldorf|Berlin)|Verwaltungsgerichts\s+(?:M\u00fcnchen|Berlin|Schwerin|Greifswald)|Oberlandesgerichts\s+(?:K\u00f6ln|Celle|M\u00fcnchen)|Amtsgerichts\s+(?:N\.|Bamberg|M\u00fchldorf|O\.|D\u00fcsseldorf|P\.)|Sozialgerichts\s+(?:Itzehoe|Stuttgart|Duisburg|Berlin)|Finanzgerichts\s+(?:Berlin-Brandenburg)|Landesarbeitsgerichts\s+(?:N\u00fcrnberg)|Landessozialgerichts\s+(?:Berlin-Brandenburg))\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.556 | 0.006 | 0.012 | 9 | 5 | 4 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 5 | 4 | 597 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `55533`) (sent_id: `55533`)


Auf die Revision des Angeklagten wird das Urteil des Landgerichts Berlin vom 27. Juni 2017 mit den Feststellungen aufgehoben .

| Predicted | Gold |
|---|---|
| `Landgerichts Berlin` | `Landgerichts Berlin` |

**Example 1** (doc_id: `57051`) (sent_id: `57051`)


Die Beschwerde der Klägerin gegen die Nichtzulassung der Revision im Urteil des Landessozialgerichts Nordrhein-Westfalen vom 22. Juni 2017 wird zurückgewiesen .

| Predicted | Gold |
|---|---|
| `Landessozialgerichts Nordrhein-Westfalen` | `Landessozialgerichts Nordrhein-Westfalen` |

**Example 2** (doc_id: `57287`) (sent_id: `57287`)


Die Beschwerde des Klägers gegen die Nichtzulassung der Revision in dem Beschluss des Landessozialgerichts Niedersachsen-Bremen vom 9. November 2017 wird als unzulässig verworfen .

| Predicted | Gold |
|---|---|
| `Landessozialgerichts Niedersachsen-Bremen` | `Landessozialgerichts Niedersachsen-Bremen` |

**Example 3** (doc_id: `57361`) (sent_id: `57361`)


Auf die Revision der Beklagten wird das Urteil des Landessozialgerichts Nordrhein-Westfalen vom 17. Dezember 2014 aufgehoben , soweit das Bestehen von Rentenversicherungspflicht des Klägers wegen Beschäftigung bei der Beigeladenen zu 1. für die Zeit ab 10. Juli 2008 verneint wird .

| Predicted | Gold |
|---|---|
| `Landessozialgerichts Nordrhein-Westfalen` | `Landessozialgerichts Nordrhein-Westfalen` |

**Example 4** (doc_id: `59091`) (sent_id: `59091`)


Auf die Revision des Klägers wird der Beschluss des Landessozialgerichts Baden-Württemberg vom 9. Februar 2015 aufgehoben und die Sache zur erneuten Verhandlung und Entscheidung an dieses Gericht zurückverwiesen .

| Predicted | Gold |
|---|---|
| `Landessozialgerichts Baden-Württemberg` | `Landessozialgerichts Baden-Württemberg` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `56576`) (sent_id: `56576`)


b ) Das Urteil des Landesarbeitsgerichts Nürnberg vom 30. Januar 2014 - 5 Sa 1/13 - wird aufgehoben .

**False Positives:**

- `Landesarbeitsgerichts Nürnberg` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Nürnberg vom 30. Januar 2014 - 5 Sa 1/13 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Nürnberg vom 30. Januar 2014 - 5 Sa 1/13 -`(RS)

**Example 1** (doc_id: `56780`) (sent_id: `56780`)


Die Revision der Beklagten gegen das Urteil des Finanzgerichts Berlin-Brandenburg vom 18. Januar 2017 3 K 3219/16 wird als unbegründet zurückgewiesen .

**False Positives:**

- `Finanzgerichts Berlin-Brandenburg` — partial — pred is substring of gold: `Urteil des Finanzgerichts Berlin-Brandenburg vom 18. Januar 2017 3 K 3219/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Finanzgerichts Berlin-Brandenburg vom 18. Januar 2017 3 K 3219/16`(RS)

**Example 2** (doc_id: `57442`) (sent_id: `57442`)


Die Revision der Klägerin gegen das Urteil des Landesarbeitsgerichts Nürnberg vom 6. Februar 2017 - 7 Sa 319/16 - wird auf ihre Kosten zurückgewiesen .

**False Positives:**

- `Landesarbeitsgerichts Nürnberg` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Nürnberg vom 6. Februar 2017 - 7 Sa 319/16 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Nürnberg vom 6. Februar 2017 - 7 Sa 319/16 -`(RS)

**Example 3** (doc_id: `59490`) (sent_id: `59490`)


2. a ) Das Urteil des Landesarbeitsgerichts Nürnberg vom 30. Januar 2014 - 5 Sa 1/13 - und das Endurteil des Arbeitsgerichts Bamberg vom 10. Oktober 2012 - 2 Ca 1097/11 - verletzen den Beschwerdeführer in seinem Grundrecht aus Artikel 2 Absatz 1 Satz 1 in Verbindung mit Artikel 20 Absatz 3 des Grundgesetzes .

**False Positives:**

- `Landesarbeitsgerichts Nürnberg` — partial — pred is substring of gold: `Urteil des Landesarbeitsgerichts Nürnberg vom 30. Januar 2014 - 5 Sa 1/13 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Landesarbeitsgerichts Nürnberg vom 30. Januar 2014 - 5 Sa 1/13 -`(RS)
- `Endurteil des Arbeitsgerichts Bamberg vom 10. Oktober 2012 - 2 Ca 1097/11 -`(RS)
- `Artikel 2 Absatz 1 Satz 1 in Verbindung mit Artikel 20 Absatz 3 des Grundgesetzes`(NRM)

</details>

---

## `Specific Ministry Variations`

**F1:** 0.010 | **Precision:** 1.000 | **Recall:** 0.005  

**Format:** `regex`  
**Rule ID:** `b15f8f0d`  
**Description:**
Matches specific ministry names including genitive forms and redacted department codes.

**Content:**
```
\b(?:Bundesministeriums?\s+der\s+Verteidigung(?:\s+-\s+\.\.\.\s-)?|Bundesministeriums?\s+der\s+Justiz|tunesischen\s+Justizministeriums|Nieders\u00e4chsische\s+Landesregierung|Bezirksregierung\s+[A-Z]\s*\w*|Landratsamt\s+[A-Z]\s*-\s*kreis|Finanzamts\s+[A-Z]|Berufsschulzentrum\s+[A-Z]|Gemeinsame\s+Bundesausschuss|Luftwaffe|H\s*AG(?:\s+alt)?|\.\.\.\s+GmbH|F\.\s*-\s*S\.\s*-Universit\u00e4t\s+[A-Z]|Bundesrat|Bundesverwaltungsgericht|Bundesfinanzhof|Bundespatentgericht|BVerwG|BGH|BSG|BAG|BVerfG|EuGH|EGMR|DPMA|BaFin|SG|FG|LAG|LSG|BA|H\u00c4VG|PKK|KCK|HPG|EED|JOOP|KAGURA|Schott|Urban\s+&\s+Schwarzenberg|National\s+Union\s+of\s+Rail|Maritime\s+and\s+Transport\s+Workers|Europ\u00e4ische\s+Gerichtshof\s+f\u00fcr\s+Menschenrechte|Deutscher\s+Apotheker\s+Verlag|B-KG|VEB\s+Elektronik\s+G\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.005 | 0.010 | 4 | 4 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 4 | 0 | 747 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53844`) (sent_id: `53844`)


Der im Juli 1950 geborene Kläger war seit dem 1. März 1971 bei einer Rechtsvorgängerin der Beklagten , der H AG ( im Folgenden H AG alt ) als Arbeitnehmer tätig .

| Predicted | Gold |
|---|---|
| `H AG alt` | `H AG alt` |

**Missed by this rule (FN):**

- `H AG` (ORG)

**Example 1** (doc_id: `55554`) (sent_id: `55554`)


Mit seiner allein das Streitjahr betreffenden Revision macht der im Laufe des Revisionsverfahrens anstelle des zunächst beklagten und revisionsführenden Finanzamts X für die Besteuerung der Kläger zuständig gewordene Beklagte und Revisionskläger ( das Finanzamt - FA - ) geltend , das FG sei zu Unrecht davon ausgegangen , dass auf den Abfluss der Beitragszahlung im Jahr 2007 abzustellen sei .

| Predicted | Gold |
|---|---|
| `Finanzamts X` | `Finanzamts X` |

**Example 2** (doc_id: `56872`) (sent_id: `56872`)


Nach dieser Entscheidung kann gemäß Art. 72 des tunesischen Strafgesetzbuchs ( CP ) die Todesstrafe verhängt werden , die jedoch aufgrund eines Moratoriums nicht angewendet wird ( vgl. Ziffer 3 des Schreibens des tunesischen Justizministeriums vom 1. März 2018 ) .

| Predicted | Gold |
|---|---|
| `tunesischen Justizministeriums` | `tunesischen Justizministeriums` |

**Missed by this rule (FN):**

- `Art. 72 des tunesischen Strafgesetzbuchs` (NRM)
- `CP` (NRM)

**Example 3** (doc_id: `57863`) (sent_id: `57863`)


Wegen dieser Taten drohe eine " lebenslange Freiheitsstrafe von maximal 20 Jahren " ( vgl. Ziffer 4 des Schreibens des tunesischen Justizministeriums vom 1. März 2018 ) .

| Predicted | Gold |
|---|---|
| `tunesischen Justizministeriums` | `tunesischen Justizministeriums` |

</details>

---

## `Ellipsis Company Names`

**F1:** 0.007 | **Precision:** 1.000 | **Recall:** 0.004  

**Format:** `regex`  
**Rule ID:** `cf89737c`  
**Description:**
Matches company names containing ellipsis (e.g., 'S …', 'R … GmbH & Co. KG', 'M … Konzerns') with correct suffix handling.

**Content:**
```
\b([A-Z]\s*…(?:\s*[A-Z][a-zA-Zäöüß]*)?\s*(?:GmbH\s*&\s*Co\.\s*KG|GmbH|AG|KG|Konzerns?))\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.004 | 0.007 | 3 | 3 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 3 | 0 | 636 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `55099`) (sent_id: `55099`)


Die Gegenstände der Ansprüche 1 , 9 und 11 haben somit in der Fa. R … GmbH & Co. KG als nicht öffentlich zugänglich im Sinne des § 3 ( 1 ) , 2 PatG zu gelten , so dass insoweit keine offenkundige Vorbenutzung vorliegt .

| Predicted | Gold |
|---|---|
| `R … GmbH & Co. KG` | `R … GmbH & Co. KG` |

**Missed by this rule (FN):**

- `§ 3 ( 1 ) , 2 PatG` (NRM)

**Example 1** (doc_id: `55422`) (sent_id: `55422`)


Anlage A3 Handzeichnung zum Ablauf von Werksbesichtigungen bei der R … GmbH & Co. KG ,

| Predicted | Gold |
|---|---|
| `R … GmbH & Co. KG` | `R … GmbH & Co. KG` |

**Example 2** (doc_id: `58075`) (sent_id: `58075`)


Jedoch trägt sie vor , dass sämtliche Besucher , die die Fa. R … GmbH & Co. KG besuchten , in der Vergangenheit und bis heute stets zur Geheimhaltung verpflichtet worden seien .

| Predicted | Gold |
|---|---|
| `R … GmbH & Co. KG` | `R … GmbH & Co. KG` |

</details>

---

## `Specific Ministry Names (Fixed)`

**F1:** 0.007 | **Precision:** 1.000 | **Recall:** 0.004  

**Format:** `regex`  
**Rule ID:** `cae95c48`  
**Description:**
Matches specific long-form ministry names including genitive endings and department codes.

**Content:**
```
\b(Bundesministeriums?\s*f\u00fcr\s+Verbraucherschutz\s*,\s+Ern\u00e4hrung\s+und\s+Landwirtschaft|Bundesministerium\s+der\s+Verteidigung\s+-\s+R\s+II\s+2\s+-|Bundesministerium\s+f\u00fcr\s+Inneres\s+und\s+Kommunales\s+des\s+Landes\s+Nordrhein-Westfalen|Bundesamt\s+f\u00fcr\s+das\s+Personalmanagement|Bundesnachrichtendienstes|Bundesministerium\s+des\s+Innern|Bundesministerium\s+der\s+Justiz|Bayerische\s+Staatsministerium|Bayerischen\s+Staatsministeriums\s+f\u00fcr\s+Umwelt\s+und\s+Gesundheit|Hessische\s+Ministerium\s+des\s+Innern\s+und\s+f\u00fcr\s+Sport|Deutsche\s+Rentenversicherung\s+Bund|Deutsche\s+Rentenversicherung\s+Rheinland|Deutschen\s+Rentenversicherung|Justizministerium\s+des\s+Landes\s+Niedersachsen|Ministerium\s+f\u00fcr\s+Justiz\s*,\s+Europa\s*,\s+Verbraucherschutz\s+und\s+Gleichstellung\s+des\s+Landes\s+Schleswig-Holstein|Bundesministerium\s+der\s+Finanzen|BMF|Bundesagentur\s+f\u00fcr\s+Arbeit|Bundesrat|Bundestag|Bundesregierung|Staatsministeriums\s+der\s+Justiz\s+sowie\s+des\s+Staatsministeriums\s+f\u00fcr\s+Kultus\s+des\s+Freistaates\s+Sachsen|Wehrbeauftragten\s+des\s+Deutschen\s+Bundestages|Bundesbeauftragte\s+f\u00fcr\s+den\s+Datenschutz\s+und\s+die\s+Informationsfreiheit|Fliegerhorst\s+B\u00fcchel|Dienststellen\s+R\s+I\s+\(\s*R\s+I\s*\)\s*,\s*R\s+II\s+\(\s*R\s+II\s*\)\s*,\s*D\s+\(\s*D\s*\)\s+K\s*,\s+Sp\s*,\s+D\s+L\s+\(\s+DL\s*\)\s+K\s*,\s+D\s+G\s+und\s+DL\s+G|Generalinspekteur\s+der\s+Bundeswehr|Deutschen\s+Fu\u00dfball-Bund|Bayerische\s+Staatsministerium\s+der\s+Justiz|Bundesministerium\s+des\s+Innern\s*,\s*f\u00fcr\s+Bau\s+und\s+Heimat|Ministerium\s+der\s+Justiz\s+des\s+Landes\s+Nordrhein-Westfalen|Bundesamt\s+f\u00fcr\s+Migration\s+und\s+Fl\u00fcchtlinge|W\u00a1\s+R\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.004 | 0.007 | 3 | 3 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 3 | 0 | 721 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `54088`) (sent_id: `54088`)


4. Die Beklagten des Ausgangsverfahrens und das Ministerium der Justiz des Landes Nordrhein-Westfalen hatten Gelegenheit zur Stellungnahme .

| Predicted | Gold |
|---|---|
| `Ministerium der Justiz des Landes Nordrhein-Westfalen` | `Ministerium der Justiz des Landes Nordrhein-Westfalen` |

**Example 1** (doc_id: `57141`) (sent_id: `57141`)


Die Beigeladenen zu 5. und 7. waren bei dem klagenden Bundesamt für Migration und Flüchtlinge ( früher Bundesamt für die Anerkennung ausländischer Flüchtlinge ; im Folgenden : Klägerin ) als Volljuristen in der Funktion eines Einzelentscheiders bei der Bearbeitung von Asylanträgen angestellt .

| Predicted | Gold |
|---|---|
| `Bundesamt für Migration und Flüchtlinge` | `Bundesamt für Migration und Flüchtlinge` |

**Missed by this rule (FN):**

- `Bundesamt für die Anerkennung ausländischer Flüchtlinge` (ORG)

**Example 2** (doc_id: `57153`) (sent_id: `57153`)


Das Bundesamt für Migration und Flüchtlinge hat mit der Entscheidung , die Vollziehbarkeit der Abschiebungsandrohung bis zum rechtskräftigen Abschluss des verwaltungsgerichtlichen Hauptsacheverfahrens auszusetzen , selbst die Erledigung des Antrags auf Erlass einer einstweiligen Anordnung herbeigeführt und insoweit zum Ausdruck gebracht , dass es das Begehren der Beschwerdeführerin , bis zum Abschluss des Hauptsacheverfahrens im Bundesgebiet verbleiben zu dürfen , für berechtigt erachtet .

| Predicted | Gold |
|---|---|
| `Bundesamt für Migration und Flüchtlinge` | `Bundesamt für Migration und Flüchtlinge` |

</details>

---

## `Specific Organization Names (Additional)`

**F1:** 0.007 | **Precision:** 0.750 | **Recall:** 0.004  

**Format:** `regex`  
**Rule ID:** `d8f91077`  
**Description:**
Matches additional specific organizations like 'Bundesanstalt für Finanzdienstleistungsaufsicht', 'VKDA', 'TFS', 'w GmbH', 'm …', 'BSZ F …', 'Ernst-Moritz-Arndt-Universität Greifswald', 'Staatlichen Hochschule für Musik und darstellende Kunst in Ma.', 'Dignitas Deutschland', 'Staatsanwaltschaft München I', 'Flughafen Frankfurt am Main', 'UNHCR', 'BAMF', 'TGAOK', 'VBL', 'Electronicon-GmbH', 'X-GmbH & Co. KG', 'A-AG'.

**Content:**
```
\b(?:Bundesanstalt\s+f\u00fcr\s+Finanzdienstleistungsaufsicht|VKDA|TFS|w\s+GmbH|m\s+\u2026|BSZ\s+F\s+\u2026|Ernst-Moritz-Arndt-Universit\u00e4t\s+Greifswald|Staatlichen\s+Hochschule\s+f\u00fcr\s+Musik\s+und\s+darstellende\s+Kunst\s+in\s+Ma\.|'\s*Dignitas\s+Deutschland\s*'|'\s*\u201e\s*Dignitas\s+Deutschland\s*\u201c\s*'|Staatsanwaltschaft\s+M\u00fcnchen\s+I|Flughafen\s+Frankfurt\s+am\s+Main|UNHCR|BAMF|TGAOK|VBL|Electronicon-GmbH|X-GmbH\s+&\s+Co\.\s+KG|A-AG|TFS|VKDA|w\s+GmbH|m\s+\u2026|BSZ\s+F\s+\u2026|Ernst-Moritz-Arndt-Universit\u00e4t\s+Greifswald|Staatlichen\s+Hochschule\s+f\u00fcr\s+Musik\s+und\s+darstellende\s+Kunst\s+in\s+Ma\.|'\s*Dignitas\s+Deutschland\s*'|'\s*\u201e\s*Dignitas\s+Deutschland\s*\u201c\s*'|Staatsanwaltschaft\s+M\u00fcnchen\s+I|Flughafen\s+Frankfurt\s+am\s+Main|UNHCR|BAMF|TGAOK|VBL|Electronicon-GmbH|X-GmbH\s+&\s+Co\.\s+KG|A-AG)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.750 | 0.004 | 0.007 | 4 | 3 | 1 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 3 | 1 | 611 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `55330`) (sent_id: `55330`)


Zu diesem Zeitpunkt hatte der Kläger , der seit dem 18. Juli 1990 bei der Beklagten beschäftigt war , auch nach der Beendigung der Mitgliedschaft bei der VBL eine versorgungsfähige Dienstzeit von fünf Jahren zurückgelegt und die Beklagte hatte entsprechend lange Versorgungsbeiträge nach § 6 TV IKK-BR erbracht .

| Predicted | Gold |
|---|---|
| `VBL` | `VBL` |

**Missed by this rule (FN):**

- `§ 6 TV IKK-BR` (REG)

**Example 1** (doc_id: `58695`) (sent_id: `58695`)


Dies ergebe sich aus den im Klageverfahren vorgelegten Erkenntnissen sowie aus dem Afghanistan-Bericht des UNHCR vom Dezember 2016 , in dem die Auffassung vertreten werde , dass das gesamte Staatsgebiet Afghanistans von einem innerstaatlichen Konflikt betroffen sei .

| Predicted | Gold |
|---|---|
| `UNHCR` | `UNHCR` |

**Missed by this rule (FN):**

- `Afghanistans` (LOC)

**Example 2** (doc_id: `60012`) (sent_id: `60012`)


Der Beklagte unterliegt nach § 14 Abs. 2 Satz 1 Betriebsrentengesetz der Aufsicht durch die Bundesanstalt für Finanzdienstleistungsaufsicht .

| Predicted | Gold |
|---|---|
| `Bundesanstalt für Finanzdienstleistungsaufsicht` | `Bundesanstalt für Finanzdienstleistungsaufsicht` |

**Missed by this rule (FN):**

- `§ 14 Abs. 2 Satz 1 Betriebsrentengesetz` (NRM)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `58233`) (sent_id: `58233`)


Er erfülle zwar die persönliche und die sachliche Voraussetzung , sei aber am Stichtag 30. 6. 1990 nicht mehr beim VEB Elektronik G. , sondern bei der Electronicon-GmbH G. beschäftigt gewesen , sodass es an der betrieblichen Voraussetzung fehle .

**False Positives:**

- `Electronicon-GmbH` — partial — pred is substring of gold: `Electronicon-GmbH G.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `VEB Elektronik G.`(ORG)
- `Electronicon-GmbH G.`(ORG)

</details>

---

## `Anonymized Ellipsis Companies`

**F1:** 0.007 | **Precision:** 0.750 | **Recall:** 0.004  

**Format:** `regex`  
**Rule ID:** `68975aa0`  
**Description:**
Matches anonymized company names containing ellipsis (e.g., 'S …', 'R … GmbH & Co. KG', 'M … Konzerns') with correct suffix handling.

**Content:**
```
\b([A-Z]\s+\u2026(?:\s+[A-Z])?\s*(?:GmbH|AG|KG|GbR|Fonds|V\.|B\.\s*V\.|Klinik|Schulzentrum|Finanzamt|Landratsamt|Berufsschulzentrum|Jobcenter|Botschaft|Kammer|Senat|Stelle|Amt|Verband|Zweckverband|Firma|Bank|Verlag|GmbH\s*&\s+Co\.\s*KG|Konzerns?|AG\s*&\s+Co\.\s*KG|S\.r\.l\.)|\u2026\s+Inc\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.750 | 0.004 | 0.007 | 4 | 3 | 1 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 3 | 1 | 790 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53472`) (sent_id: `53472`)


Sie ist dann allerdings die einzige Beschwerdeführerin , denn für die Beschwerde der R … GmbH in S … , fehlt es an der Zahlung der erforderlichen weiteren , zweiten Beschwerdegebühr , so dass diese als nicht eingelegt gilt .

| Predicted | Gold |
|---|---|
| `R … GmbH` | `R … GmbH` |

**Missed by this rule (FN):**

- `S …` (LOC)

**Example 1** (doc_id: `56697`) (sent_id: `56697`)


E4c : Rechnung der S … GmbH , Nummer 9090503004 vom 19. 10. 2016 an die P … GmbH , D … . in M … .

| Predicted | Gold |
|---|---|
| `S … GmbH` | `S … GmbH` |

**Missed by this rule (FN):**

- `P … GmbH , D …` (ORG)
- `M …` (LOC)

**Example 2** (doc_id: `58293`) (sent_id: `58293`)


Der Senat hat aufgrund einer Zusammenschau der Rechnung D4 , des zugehörigen Lieferscheins D8 und des Produktprogramms D3 keinen Zweifel daran , dass ein aus dem Produktprogramm Sicherheitstechnik der Firma E … bekannter Sicherheitsschalter am 18. / 20. 02. 2009 an die Firma C … GmbH in E … verkauft und auch geliefert wurde .

| Predicted | Gold |
|---|---|
| `C … GmbH` | `C … GmbH` |

**Missed by this rule (FN):**

- `E …` (ORG)
- `E …` (LOC)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `56697`) (sent_id: `56697`)


E4c : Rechnung der S … GmbH , Nummer 9090503004 vom 19. 10. 2016 an die P … GmbH , D … . in M … .

**False Positives:**

- `P … GmbH` — partial — pred is substring of gold: `P … GmbH , D …`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S … GmbH`(ORG)
- `P … GmbH , D …`(ORG)
- `M …`(LOC)

</details>

---

## `Bundesamt Entities`

**F1:** 0.005 | **Precision:** 1.000 | **Recall:** 0.002  

**Format:** `regex`  
**Rule ID:** `7a559d8a`  
**Description:**
Matches 'Bundesamt' entities including the full name with 'f\u00fcr das Personalmanagement'.

**Content:**
```
\b(Bundesamts\s+f\u00fcr\s+das\s+Personalmanagement\s+der\s+Bundeswehr|Bundesamt\s+f\u00fcr\s+das\s+Personalmanagement\s+der\s+Bundeswehr|Bundesamt\s+f\u00fcr\s+das\s+Personalmanagement)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.002 | 0.005 | 2 | 2 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 2 | 0 | 373 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `57265`) (sent_id: `57265`)


2. Die angefochtene Versetzung ist rechtswidrig , weil das Bundesamt für das Personalmanagement und das Bundesministerium der Verteidigung bei der Ausübung des ihnen zustehenden Ermessens die von dem Antragsteller geltend gemachte Betreuung seiner Großmutter nicht berücksichtigt haben ( § 23a Abs. 2 Satz 1 WBO i. V. m. § 114 Satz 1 VwGO ) .

| Predicted | Gold |
|---|---|
| `Bundesamt für das Personalmanagement` | `Bundesamt für das Personalmanagement` |

**Missed by this rule (FN):**

- `Bundesministerium der Verteidigung` (ORG)
- `§ 23a Abs. 2 Satz 1 WBO` (NRM)
- `§ 114 Satz 1 VwGO` (NRM)

**Example 1** (doc_id: `58115`) (sent_id: `58115`)


Nach Nr. 103 ZDv A- 1340/9 bedürfen Verlängerungen der Verwendungsdauer über sechs Jahre hinaus der vorherigen Zustimmung des oder der für die Personalführung des oder der Betroffenen zuständigen Unterabteilungsleiters / Unterabteilungsleiterin im Bundesamt für das Personalmanagement der Bundeswehr bzw. des Referatsleiters bzw. der Referatsleiterin BMVg - P II 2 - .

| Predicted | Gold |
|---|---|
| `Bundesamt für das Personalmanagement der Bundeswehr` | `Bundesamt für das Personalmanagement der Bundeswehr` |

**Missed by this rule (FN):**

- `Nr. 103 ZDv A- 1340/9` (REG)
- `BMVg - P II 2 -` (ORG)

</details>

---

## `BgA X Pattern`

**F1:** 0.005 | **Precision:** 1.000 | **Recall:** 0.002  

**Format:** `regex`  
**Rule ID:** `fa3e8e18`  
**Description:**
Matches the specific pattern 'BgA X' found in training data.

**Content:**
```
\bBgA\s+X\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.002 | 0.005 | 2 | 2 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 2 | 0 | 522 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `56162`) (sent_id: `56162`)


Denn im Zusammenhang mit den Satzungsregelungen in § 3 Abs. 1 und § 5 Abs. 4 wird aus diesem Vertrag trotzdem deutlich , dass die aus diesen Verträgen zufließenden Einnahmen letztlich beim BgA X zu einer verzinsten Rückführung der dem BgA Projekte vorfinanzierten Ausgaben führen .

| Predicted | Gold |
|---|---|
| `BgA X` | `BgA X` |

**Example 1** (doc_id: `58552`) (sent_id: `58552`)


1. Zwar durfte das FA den Kläger wegen fehlender Abgabe der Kapitalertragsteueranmeldungen i. S. des § 45a EStG grundsätzlich für die Entrichtungsschulden des BgA X im Wege des Nachforderungsbescheids in Anspruch nehmen ( § 167 Abs. 1 Satz 1 Alternative 2 der Abgabenordnung - AO - i. V. m. § 155 Abs. 1 Satz 1 AO , § 20 Abs. 1 Nr. 10 Buchst. b EStG , § 43 Abs. 1 Satz 1 Nr. 7c EStG und § 44 Abs. 6 Sätze 1 und 4 sowie Abs. 1 Sätze 3 bis 5 EStG ) .

| Predicted | Gold |
|---|---|
| `BgA X` | `BgA X` |

**Missed by this rule (FN):**

- `§ 45a EStG` (NRM)
- `§ 167 Abs. 1 Satz 1 Alternative 2 der Abgabenordnung` (NRM)
- `AO` (NRM)
- `§ 155 Abs. 1 Satz 1 AO` (NRM)
- `§ 20 Abs. 1 Nr. 10 Buchst. b EStG` (NRM)
- `§ 43 Abs. 1 Satz 1 Nr. 7c EStG` (NRM)
- `§ 44 Abs. 6 Sätze 1 und 4 sowie Abs. 1 Sätze 3 bis 5 EStG` (NRM)

</details>

---

## `European Union`

**F1:** 0.005 | **Precision:** 0.111 | **Recall:** 0.002  

**Format:** `regex`  
**Rule ID:** `5d8c53d3`  
**Description:**
Matches 'Europäischen Union' in various cases.

**Content:**
```
\b(Europ\u00e4ischen\s*Union|Europ\u00e4ische\s*Union|EU|Gerichtshof\s+der\s+Europ\u00e4ischen\s+Union|Amt\s+der\s+Europ\u00e4ischen\s+Union\s+f\u00fcr\s+geistiges\s+Eigentum|EUGH)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.111 | 0.002 | 0.005 | 18 | 2 | 16 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 2 | 16 | 736 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `55904`) (sent_id: `55904`)


bb ) Der Gerichtshof der Europäischen Union hat entschieden , § 4 Nr. 2 der Rahmenvereinbarung über Teilzeitarbeit sei dahin auszulegen , dass er einer nationalen Bestimmung entgegensteht , nach der bei einer Änderung des Beschäftigungsausmaßes eines Arbeitnehmers das Ausmaß des noch nicht verbrauchten Erholungsurlaubs in der Weise angepasst wird , dass der von einem Arbeitnehmer , der von einer Vollzeit- zu einer Teilzeitbeschäftigung übergeht , in der Zeit der Vollzeitbeschäftigung erworbene Anspruch auf bezahlten Jahresurlaub , dessen Ausübung dem Arbeitnehmer während dieser Zeit nicht möglich war , reduziert wird oder der Arbeitnehmer diesen Urlaub nur mehr mit einem geringeren Urlaubsentgelt verbrauchen kann ( EuGH 22. April 2010 - C- 486/08 - [ Zentralbetriebsrat der Landeskrankenhäuser Tirols ] Rn. 35 ) .

| Predicted | Gold |
|---|---|
| `Gerichtshof der Europäischen Union` | `Gerichtshof der Europäischen Union` |

**Missed by this rule (FN):**

- `§ 4 Nr. 2 der Rahmenvereinbarung über Teilzeitarbeit` (NRM)
- `EuGH 22. April 2010 - C- 486/08 - [ Zentralbetriebsrat der Landeskrankenhäuser Tirols ] Rn. 35` (RS)

**Example 1** (doc_id: `59798`) (sent_id: `59798`)


Insoweit konnte nicht ausgeschlossen werden , dass die Einfuhr der Textilien in die EU in dem betreffenden Jahr jeweils durch eine einheitliche Handlung erfolgte .

| Predicted | Gold |
|---|---|
| `EU` | `EU` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53938`) (sent_id: `53938`)


In dem besonderen Fall der Sanktionierung von Verstößen gegen die Verordnung [ … ] wurden jedoch die straf- oder bußgeldbewehrten Vorschriften der Verordnung [ … ] durch das Inkrafttreten der Sanktionsvorschriften vor dem Anwendungszeitpunkt der bewehrten EU-Verordnung bereits ab dem 2. Juli 2016 in Deutschland für anwendbar erklärt .

**False Positives:**

- `EU` — similar text (different position): `Deutschland`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Deutschland`(LOC)

**Example 1** (doc_id: `54192`) (sent_id: `54192`)


- 100 mg Granulat zur Zubereitung oral einzunehmender Suspensionen unter der Nummer EU / 1 / 07 / 436 / 005 .

**False Positives:**

- `EU` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `54233`) (sent_id: `54233`)


Weder im deutschen AsylG noch in einem anderen deutschen Regelungswerk gebe es eine Norm , in der stehe oder aus der abgeleitet werden könne , die Gewährung subsidiären Schutzes in einem anderen EU-Mitgliedstaat stünde der Auslieferung der betroffenen Person durch die Bundesrepublik Deutschland entgegen .

**False Positives:**

- `EU` — similar text (different position): `Bundesrepublik Deutschland`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `AsylG`(NRM)
- `Bundesrepublik Deutschland`(LOC)

**Example 3** (doc_id: `54786`) (sent_id: `54786`)


Nichtannahmebeschluss : Gewährung subsidiären Schutzes in anderem EU-Mitgliedsstaat stellt gewichtiges Indiz für Vorliegen eines Auslieferungshindernisses dar - hier : Auslieferung eines in Belgien als schutzberechtigt anerkannten Weißrussen an Weißrussland zur Strafverfolgung - mangelnde Substantiierung wegen Nichtvorlage entscheidungserheblicher Unterlagen - zudem Entkräftung der Gefahr einer Verhängung der Todesstrafe bzw menschenunwürdiger Haftbedingungen durch verbindliche Zusicherungen Weißrusslands

**False Positives:**

- `EU` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Belgien`(LOC)
- `Weißrussland`(LOC)
- `Weißrusslands`(LOC)

**Example 4** (doc_id: `55373`) (sent_id: `55373`)


Nach den dargestellten Grundsätzen fehlt der Europäischen Union die Rechtsmacht , einer Regelung des nationalen Rechts die Wirksamkeit für Sachverhalte zu nehmen , welche keinen hinreichenden Bezug zu anderen EU-Mitgliedstaaten aufweisen und deshalb außerhalb der Regelungskompetenz der Europäischen Union liegen .

**False Positives:**

- `EU` — similar text (different position): `Europäischen Union`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europäischen Union`(ORG)
- `Europäischen Union`(ORG)

**Example 5** (doc_id: `56084`) (sent_id: `56084`)


- 25 mg Kautabletten , zugelassen unter der Nummer EU / 1 / 07 / 436/003 ,

**False Positives:**

- `EU` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 6** (doc_id: `57187`) (sent_id: `57187`)


Anlass für eine erneute Vorlage nach Art. 267 AEUV besteht nicht , zumal der EuGH seine Rechtsauffassung in dem Urteil vom 16. April 2015 - C- 446/12 [ ECLI : EU :C : 2015 : 238 ] , Willems - Rn. 46 bestätigt hat .

**False Positives:**

- `EU` — similar text (different position): `Art. 267 AEUV`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 267 AEUV`(NRM)
- `EuGH`(ORG)
- `Urteil vom 16. April 2015 - C- 446/12 [ ECLI : EU :C : 2015 : 238 ] , Willems - Rn. 46`(RS)

**Example 7** (doc_id: `57389`) (sent_id: `57389`)


Es bestünden erhebliche Zweifel , ob Angehörige dieser Gruppe entsprechend den Anforderungen des Art. 28 Abs. 2 der Richtlinie 2013 / 32 / EU des Europäischen Parlaments und des Rates vom 26. Juni 2013 zu gemeinsamen Verfahren für die Zuerkennung und Aberkennung des internationalen Schutzes ( ABl. L 180 S. 60 ) behandelt würden .

**False Positives:**

- `EU` — partial — pred is substring of gold: `Art. 28 Abs. 2 der Richtlinie 2013 / 32 / EU des Europäischen Parlaments und des Rates vom 26. Juni 2013 zu gemeinsamen Verfahren für die Zuerkennung und Aberkennung des internationalen Schutzes ( ABl. L 180 S. 60 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 28 Abs. 2 der Richtlinie 2013 / 32 / EU des Europäischen Parlaments und des Rates vom 26. Juni 2013 zu gemeinsamen Verfahren für die Zuerkennung und Aberkennung des internationalen Schutzes ( ABl. L 180 S. 60 )`(NRM)

**Example 8** (doc_id: `57647`) (sent_id: `57647`)


„ festzustellen , dass für die Durchführung eines elektronischen Abgleiches der Mitarbeiterdaten mit den sog. Sanktionslisten aus den EU-Verordnungen VO ( EG ) 2580/2001 und VO ( EG ) 881/2002 durch die Arbeitgeberin das Mitbestimmungsrecht a ) des Betriebsrats und b ) des Gesamtbetriebsrats besteht “ ,

**False Positives:**

- `EU` — partial — pred is substring of gold: `EU-Verordnungen VO ( EG ) 2580/2001`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EU-Verordnungen VO ( EG ) 2580/2001`(NRM)
- `VO ( EG ) 881/2002`(NRM)

**Example 9** (doc_id: `57786`) (sent_id: `57786`)


Das FG hat aber zu Recht darauf hingewiesen , dass nach der Rechtsprechung des Gerichtshofs der Europäischen Union ( EuGH ) bei der Auslegung des Art. 10 der VO Nr. 574/72 die " allgemeinen Vorschriften " des in Titel I der VO Nr. 1408/71 und damit Art. 12 der VO Nr. 1408/71 zu beachten sind ( EuGH-Urteil Wiering vom 8. Mai 2014 C - 347/12 , EU : C : 2014 : 300 , Rz 54 ff. , Zeitschrift für europäisches Sozial- und Arbeitsrecht - ZESAR - 2015 , 113 ) .

**False Positives:**

- `EU` — similar text (different position): `Gerichtshofs der Europäischen Union`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Gerichtshofs der Europäischen Union`(ORG)
- `EuGH`(ORG)
- `Art. 10 der VO Nr. 574/72`(NRM)
- `Titel I der VO Nr. 1408/71`(NRM)
- `Art. 12 der VO Nr. 1408/71`(NRM)
- `EuGH-Urteil Wiering vom 8. Mai 2014 C - 347/12 , EU : C : 2014 : 300 , Rz 54 ff.`(RS)
- `Zeitschrift für europäisches Sozial- und Arbeitsrecht - ZESAR - 2015 , 113`(LIT)

**Example 10** (doc_id: `58343`) (sent_id: `58343`)


Unter den hier vorliegenden Voraussetzungen des Art. 267 Abs. 3 AEUV ( vergleiche dazu Beschlüsse des Bundesverfassungsgerichts - BVerfG - vom 6. September 2016 1 BvR 1305/13 , Neue Zeitschrift für Verwaltungsrecht - NVwZ - 2017 , 53 , Rz 7 , 8 ; vom 6. Oktober 2017 2 BvR 987/16 , Neue Juristische Wochenschrift - NJW - 2018 , 606 , Rz 4 , 5 ; jeweils mit weiteren Nachweisen ) sind die nationalen Gerichte von Amts wegen gehalten , den EuGH anzurufen ( vergleiche BVerfG-Beschlüsse in NVwZ 2017 , 53 , Rz 6 ; in NJW 2018 , 606 , Rz 3 ; ferner EuGH-Urteil CILFIT vom 6. Oktober 1982 C - 283/81 , EU : C : 1982 : 335 , NJW 1983 , 1257 , Rz 21 ; jeweils mit weiteren Nachweisen ) .

**False Positives:**

- `EU` — similar text (different position): `Art. 267 Abs. 3 AEUV`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 267 Abs. 3 AEUV`(NRM)
- `Beschlüsse des Bundesverfassungsgerichts - BVerfG - vom 6. September 2016 1 BvR 1305/13 , Neue Zeitschrift für Verwaltungsrecht - NVwZ - 2017 , 53 , Rz 7 , 8`(RS)
- `vom 6. Oktober 2017 2 BvR 987/16 , Neue Juristische Wochenschrift - NJW - 2018 , 606 , Rz 4 , 5`(RS)
- `EuGH`(ORG)
- `BVerfG-Beschlüsse in NVwZ 2017 , 53 , Rz 6`(LIT)
- `NJW 2018 , 606 , Rz 3`(RS)
- `EuGH-Urteil CILFIT vom 6. Oktober 1982 C - 283/81 , EU : C : 1982 : 335 , NJW 1983 , 1257 , Rz 21`(RS)

**Example 11** (doc_id: `58501`) (sent_id: `58501`)


Das Oberverwaltungsgericht hat angenommen , dass eine Prüfung des § 4 Abs. 3 und 4 PassG auf die Vereinbarkeit mit dem Grundgesetz wegen der unionsrechtlichen Determinierung nicht stattfinden kann und die Vereinbarkeit der zwingenden unionsrechtlichen Vorgaben des Art. 1 Abs. 1 und 2 der Verordnung ( EG ) 2252/2004 mit der Charta der Grundrechte der EU und anderem höherrangigen Unionsrecht durch die Rechtsprechung des EuGH mit bindender Wirkung geklärt ist .

**False Positives:**

- `EU` — partial — pred is substring of gold: `Charta der Grundrechte der EU`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 4 Abs. 3 und 4 PassG`(NRM)
- `Grundgesetz`(NRM)
- `Art. 1 Abs. 1 und 2 der Verordnung ( EG ) 2252/2004`(NRM)
- `Charta der Grundrechte der EU`(NRM)
- `EuGH`(ORG)

**Example 12** (doc_id: `58740`) (sent_id: `58740`)


Das Berufungsgericht wird sich bei seiner neuerlichen , durch diesen Beschluss nicht im Ergebnis vorgeprägten Entscheidung auch mit den - wenngleich in anderem Zusammenhang ergangenen - Entscheidungen des Gerichtshofes der Europäischen Union vom 25. Januar 2018 ( - C- 360/16 [ ECLI : EU :C : 2018 : 35 ] , Hasan - Rn. 30 f. , 40 ) und des Bundesverfassungsgerichts vom 21. April 2016 ( - 2 BvR 273/16 - NVwZ 2016 , 1242 Rn. 11 ) zur Frage des maßgeblichen Zeitpunkts für die Beurteilung des Vorliegens systemischer Schwachstellen auseinanderzusetzen haben .

**False Positives:**

- `EU` — partial — pred is substring of gold: `Entscheidungen des Gerichtshofes der Europäischen Union vom 25. Januar 2018 ( - C- 360/16 [ ECLI : EU :C : 2018 : 35 ] , Hasan - Rn. 30 f. , 40 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Entscheidungen des Gerichtshofes der Europäischen Union vom 25. Januar 2018 ( - C- 360/16 [ ECLI : EU :C : 2018 : 35 ] , Hasan - Rn. 30 f. , 40 )`(RS)
- `Bundesverfassungsgerichts vom 21. April 2016 ( - 2 BvR 273/16 - NVwZ 2016 , 1242 Rn. 11 )`(RS)

**Example 13** (doc_id: `58845`) (sent_id: `58845`)


Die Klägerin hat weder ihre Ausbildung im EU-Ausland absolviert noch war sie in einem dieser Länder beruflich tätig .

**False Positives:**

- `EU` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 14** (doc_id: `58902`) (sent_id: `58902`)


Insbesondere darf das nationale Gericht trotz einer abweichenden Entscheidung der Vorinstanz davon absehen , dem EuGH eine vor ihm aufgeworfene Frage nach der Auslegung des Unionsrechts vorzulegen ( vgl. EuGH-Urteil Ferreira da Silva e Brito u. a. , EU : C : 2015 : 565 , EuZW 2016 , 111 , Rz 40 bis 42 , m. w. N. ) .

**False Positives:**

- `EU` — similar text (different position): `EuGH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH`(ORG)
- `EuGH-Urteil Ferreira da Silva e Brito u. a. , EU : C : 2015 : 565 , EuZW 2016 , 111 , Rz 40 bis 42`(RS)

**Example 15** (doc_id: `59082`) (sent_id: `59082`)


Der Gesetzgeber hat § 2 BetrAVG aF durch Art. 1 Nr. 2 des Gesetzes zur Umsetzung der EU-Mobilitäts-Richtlinie vom 21. Dezember 2015 ( BGBl. I S. 2553 ) teilweise neu gefasst , ohne dass sich insoweit Änderungen zu der vorher geltenden Rechtslage ergeben sollten ( vgl. BT-Drs. 18/6283 S. 13 ) .

**False Positives:**

- `EU` — partial — pred is substring of gold: `Art. 1 Nr. 2 des Gesetzes zur Umsetzung der EU-Mobilitäts-Richtlinie vom 21. Dezember 2015 ( BGBl. I S. 2553 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 2 BetrAVG aF`(NRM)
- `Art. 1 Nr. 2 des Gesetzes zur Umsetzung der EU-Mobilitäts-Richtlinie vom 21. Dezember 2015 ( BGBl. I S. 2553 )`(NRM)
- `BT-Drs. 18/6283 S. 13`(LIT)

</details>

---

## `Finanzamt and Staatsanwaltschaft`

**F1:** 0.002 | **Precision:** 1.000 | **Recall:** 0.001  

**Format:** `regex`  
**Rule ID:** `7e60475b`  
**Description:**
Matches Finanzamt and Staatsanwaltschaft with location or anonymized names.

**Content:**
```
\b(?:Finanzamt\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)*|Staatsanwaltschaft\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)*|Generalstaatsanwaltschaft\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)*)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.001 | 0.002 | 1 | 1 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 1 | 0 | 762 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53725`) (sent_id: `53725`)


4. Auf die sofortige Beschwerde der Staatsanwaltschaft Düsseldorf hob das Oberlandesgericht Hamm mit angegriffenem Beschluss vom 28. Juli 2015 den Beschluss des Landgerichts Paderborn auf und ordnete die weitere Vollstreckung der Unterbringung des Beschwerdeführers in der Sicherungsverwahrung an .

| Predicted | Gold |
|---|---|
| `Staatsanwaltschaft Düsseldorf` | `Staatsanwaltschaft Düsseldorf` |

**Missed by this rule (FN):**

- `Oberlandesgericht Hamm` (ORG)
- `Landgerichts Paderborn` (ORG)

</details>

---

## `Prisons and Detention Centers`

**F1:** 0.002 | **Precision:** 0.167 | **Recall:** 0.001  

**Format:** `regex`  
**Rule ID:** `a2bb4dd2`  
**Description:**
Matches prisons (JVA) and similar institutions, allowing for optional periods in location identifiers.

**Content:**
```
\b(Justizvollzugsanstalt(?:\s+[A-Za-z\u00e4\u00f6\u00fc\u00df]+)?(?:\s*-\s*[A-Za-z\u00e4\u00f6\u00fc\u00df]+)?|JVA\s+[A-Za-z\u00e4\u00f6\u00fc\u00df]+\.?)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.167 | 0.001 | 0.002 | 6 | 1 | 5 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 1 | 5 | 740 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `59823`) (sent_id: `59823`)


Erstens würden anderen Gefangenen in vergleichbaren Situationen vollzugsöffnende Maßnahmen gewährt , und zweitens habe das Oberlandesgericht in einem anderen Verfahren vertreten , dass auch die Justizvollzugsanstalt Bruchsal Möglichkeiten der Diagnose vorhalten müsse und der Grundsatz der bestmöglichen Sachaufklärung die Einholung gutachterlicher Expertise gebiete .

| Predicted | Gold |
|---|---|
| `Justizvollzugsanstalt Bruchsal` | `Justizvollzugsanstalt Bruchsal` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53899`) (sent_id: `53899`)


Ermessensfehlerfrei habe die Justizvollzugsanstalt ausgeführt , dass sich die Weigerung des Beschwerdeführers , die notwendigen Behandlungsschritte durchzuführen , auf die Mindestverbüßungsdauer auswirke .

**False Positives:**

- `Justizvollzugsanstalt ausgeführt` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `55568`) (sent_id: `55568`)


Es sei bisher nicht einmal absehbar , in welcher Justizvollzugsanstalt der Beschwerdeführer gegebenenfalls inhaftiert werde .

**False Positives:**

- `Justizvollzugsanstalt der` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `59221`) (sent_id: `59221`)


Dabei wird nicht verkannt , dass die Justizvollzugsanstalt insoweit durchaus bezweckte , die Resozialisierung des Beschwerdeführers voranzutreiben und ihn zur weiteren Mitarbeit am Vollzugsziel zu motivieren .

**False Positives:**

- `Justizvollzugsanstalt insoweit` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 3** (doc_id: `59812`) (sent_id: `59812`)


Er habe aus der Justizvollzugsanstalt seine Familie kontaktiert , die sich daraufhin an den bayerischen Flüchtlingsrat gewandt habe .

**False Positives:**

- `Justizvollzugsanstalt seine` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `bayerischen Flüchtlingsrat`(ORG)

**Example 4** (doc_id: `59853`) (sent_id: `59853`)


Die - umfänglichen - Schriftsätze des Beschwerdeführers fügte das Gericht dabei unverändert in den Tatbestand des Beschlusses ein , während es den Vortrag der Justizvollzugsanstalt mit eigenen Worten wiedergab .

**False Positives:**

- `Justizvollzugsanstalt mit` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Car Brands`

**F1:** 0.002 | **Precision:** 0.143 | **Recall:** 0.001  

**Format:** `regex`  
**Rule ID:** `90ee6b6f`  
**Description:**
Matches specific car brand names found in legal texts.

**Content:**
```
\b(Audi|BMW|Mercedes|VW|Opel|Ford|Toyota|Honda|Nissan|Hyundai|Kia|Subaru|Mazda|Suzuki|Lexus|Infiniti|Acura|Porsche|Ferrari|Lamborghini|Maserati|Bentley|Rolls-Royce|Jaguar|Land Rover|Volvo|Saab|Mini|Smart|Dacia|Skoda|Seat|Renault|Peugeot|Citro\u00ebn|Fiat|Alfa Romeo|Lancia|Jeep|Chrysler|Dodge|Ram|Buick|Cadillac|GMC|Lincoln|Tesla|Morgan|Lotus|Aston Martin|McLaren|Bugatti|Koenigsegg|Pagani|Maybach|Bentley|Rolls-Royce)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.143 | 0.001 | 0.002 | 7 | 1 | 6 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 1 | 6 | 773 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `54570`) (sent_id: `54570`)


1. Am 13. Januar 2015 fuhr der Angeklagte auf der Bundesstraße 43 in Frankfurt am Main mit einem Pkw der Marke Audi , Typ A 3 , sehr dicht auf das vorausfahrende Fahrzeug der Geschädigten B. auf , nachdem diese zuvor auf die Abbiegespur in Richtung Nied gewechselt war .

| Predicted | Gold |
|---|---|
| `Audi` | `Audi` |

**Missed by this rule (FN):**

- `Bundesstraße 43` (LOC)
- `Frankfurt am Main` (LOC)
- `B.` (PER)
- `Nied` (LOC)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53637`) (sent_id: `53637`)


aa ) Das Berufungsgericht ist zu der Einschätzung gelangt , die Klägerin habe nicht dargelegt , dass die Zulassung als Jaguar- und Land-Rover-Vertragswerkstatt eine Ressource darstelle , ohne die der Zugang zu dem nachgelagerten Endkundenmarkt nicht oder nicht sinnvoll möglich sei .

**False Positives:**

- `Jaguar` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `54698`) (sent_id: `54698`)


In Bezug auf die in Klasse 12 beanspruchten Waren „ Kraftfahrzeuge “ kann die Angabe „ ATTENTIONGUARD “ auf die Ausstattung des Fahrzeugs mit einem Aufmerksamkeitsassistenten hinweisen ( vgl. Anlage 5 zum Ladungshinweis vom 19. September 2017 : „ Hyundai Neuwagen …

**False Positives:**

- `Hyundai` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `57175`) (sent_id: `57175`)


Denn die Hauptfunktion der Marke besteht darin , die Ursprungsidentität der gekennzeichneten Waren und Dienstleistungen zu gewährleisten ( EuGH a. a. O. – Audi AG / HABM [ Vorsprung durch Technik ] ; BGH a. a. O. – OUI ; a. a. O. – for you ) .

**False Positives:**

- `Audi` — partial — pred is substring of gold: `EuGH a. a. O. – Audi AG / HABM [ Vorsprung durch Technik ]`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH a. a. O. – Audi AG / HABM [ Vorsprung durch Technik ]`(RS)
- `BGH a. a. O. – OUI`(RS)
- `a. a. O. – for you`(RS)

**Example 3** (doc_id: `57627`) (sent_id: `57627`)


Daher ist die Tatsache allein , dass ein Zeichen von den angesprochenen Verkehrskreisen als Werbeslogan wahrgenommen wird , nicht genügend zur Verneinung der für die Schutzfähigkeit erforderlichen Unterscheidungskraft ( vgl. EuGH , GRUR 2010 , 228 , Rn. 44 – Audi / HABM [ Vorsprung durch Technik ] ) .

**False Positives:**

- `Audi` — partial — pred is substring of gold: `EuGH , GRUR 2010 , 228 , Rn. 44 – Audi / HABM [ Vorsprung durch Technik ]`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH , GRUR 2010 , 228 , Rn. 44 – Audi / HABM [ Vorsprung durch Technik ]`(RS)

**Example 4** (doc_id: `58403`) (sent_id: `58403`)


Hätte die Beklagte demgegenüber die Umstellung des Systems ihrer Werkstattverträge zu einer quantitativen Selektion genutzt , könnte das damit verfolgte Interesse im Rahmen der Abwägung mit dem Interesse der Klägerin , auch nach der Systemumstellung dem Netz der Jaguar- und Land-Rover-Vertragswerkstätten anzugehören , im Regelfall nicht berücksichtigt werden ( vgl. BGH , Urteil vom 26. Januar 2016 - KZR 41/14 , NJW 2016 , 2504 Rn. 33 - Jaguar-Vertragswerkstatt ) .

**False Positives:**

- `Jaguar` — similar text (different position): `BGH , Urteil vom 26. Januar 2016 - KZR 41/14 , NJW 2016 , 2504 Rn. 33 - Jaguar-Vertragswerkstatt`
- `Jaguar` — partial — pred is substring of gold: `BGH , Urteil vom 26. Januar 2016 - KZR 41/14 , NJW 2016 , 2504 Rn. 33 - Jaguar-Vertragswerkstatt`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 26. Januar 2016 - KZR 41/14 , NJW 2016 , 2504 Rn. 33 - Jaguar-Vertragswerkstatt`(RS)

</details>

---

## `Senate/Chamber of Courts`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `12ca8a62`  
**Description:**
Matches specific senate/chamber names of courts with location.

**Content:**
```
\b(?:\d+\.\s+(?:Senat|Kammer)\s+f\u00fcr\s+[A-Za-z\s]+\s+des\s+(?:Oberlandesgericht|Landgericht|Amtsgericht|Verwaltungsgericht|Arbeitsgericht|Sozialgericht|Finanzgericht))\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Government Ministries with Suffix`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `d54229fb`  
**Description:**
Matches government ministries including optional department codes (e.g., - R II 2 -).

**Content:**
```
\b(Bundesministeriums der Verteidigung - R II 2 -|Bundesministerium der Verteidigung - R II 2 -)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Organizations with Stiftung`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `6998f447`  
**Description:**
Matches organizations ending in 'Stiftung' (Foundation) with full name capture.

**Content:**
```
\b([A-Za-z\u00e4\u00f6\u00fc\u00df\s]+(?:\s+[A-Za-z\u00e4\u00f6\u00fc\u00df\s]+)*\s+Stiftung)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Organizations with Verband`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `0c476cd0`  
**Description:**
Matches organizations ending in 'Verband' (Association) with full name capture.

**Content:**
```
\b([A-Za-z\u00e4\u00f6\u00fc\u00df\s]+(?:\s+[A-Za-z\u00e4\u00f6\u00fc\u00df\s]+)*\s+Verband)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Unique Court Names with Location`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `c649f0f1`  
**Description:**
Matches specific court names that include a location as part of the name (e.g., Landesarbeitsgerichts Rheinland-Pfalz, Verwaltungsgericht München).

**Content:**
```
\b(Land(?:esarbeitsgerichts Rheinland\-Pfalz|gericht Darmstadt)|Verwaltungsgericht (?:M\u00fcnchen|Berlin)|Oberlandesgerichts Celle|Generalstaatsanwaltschaft Celle|19\.\s*Zivilsenats\s*des\s*Oberlandesgerichts\s*K\u00f6ln)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Specific Court Genitives`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `4fd6fd76`  
**Description:**
Matches specific court names with genitive endings followed by a location or context, with strict boundaries.

**Content:**
```
\b(?:Bundesverfassungsgerichts|Bundesverwaltungsgerichts|Bundesfinanzhofs|Bundesgerichtshofs|Landessozialgerichts|Landgerichts|Amtsgerichts|Verwaltungsgerichts|Oberlandesgerichts|Finanzgerichts|Sozialgerichts|Arbeitsgerichts)(?![a-zäöüß\s])
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Specific Court Name with Hyphen and Role`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `debbda68`  
**Description:**
Matches the specific pattern 'Amtsgericht Neu-Ulm - Jugendrichter -' and similar structures with hyphens and roles.

**Content:**
```
\bAmtsgericht\s+Neu-Ulm\s*-\s*Jugendrichter\s*-\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Patent Office Departments Full`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `82768c3f`  
**Description:**
Matches 'Markenstelle' and 'Prüfungsstelle' with class numbers and the full parent organization name.

**Content:**
```
\b(Markenstelle für Klasse \d+(?:\s+-\s+[A-Za-z\s-]+)?\s+des\s+Deutschen\s+Patent-\s+und\s+Markenamts|Prüfungsstelle für Klasse [A-Z0-9]+ des Deutschen Patent- und Markenamts)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Multi-Letter Company Codes`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `bb2f7a54`  
**Description:**
Matches specific multi-letter company codes like 'PTE' followed by name and GmbH/AG.

**Content:**
```
\b([A-Z]{2,}\s+[A-Za-z]+\s+GmbH|[A-Z]{2,}\s+AG)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `JPO and Other Abbreviations`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `d7491371`  
**Description:**
Matches specific abbreviations like JPO that were missing.

**Content:**
```
\b(JPO|G AG)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Senate/Chamber of Courts (Strict)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `597548a9`  
**Description:**
Matches specific senates or chambers of courts, handling the full context including the parent court (e.g., '25. Senat ... des Bundespatentgerichts').

**Content:**
```
\b(\d+\.\s+Senat(?:\s+\([^)]+\))?\s+des\s+(?:Bundesverfassungsgericht|Bundesgerichtshof|Bundesfinanzhof|Bundespatentgericht|Oberlandesgericht|Landgericht|Verwaltungsgericht|Sozialgericht|Finanzgericht|Arbeitsgericht|Anwaltsgerichtshof|Landessozialgericht|Landesarbeitsgericht|Landesverwaltungsgericht|Oberverwaltungsgericht|Verwaltungsgerichtshof|Europ\u00e4ischen\s+Gerichtshof(?:\s+f\u00fcr\s+Menschenrechte)?))\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Single Letter Companies`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `8f509e92`  
**Description:**
Matches single letter or hyphenated codes followed by GmbH/AG/KB to catch 'X GmbH', 'S AG', 'A-AG', etc., with stricter boundaries.

**Content:**
```
\b([A-Z][\-]?[A-Z]?\s*(?:GmbH|AG|KG|GbR|Fonds|V\.|B\.\s*V\.|Klinik|Schulzentrum|Finanzamt|Landratsamt|Berufsschulzentrum|Jobcenter|Botschaft|Kammer|Senat|Stelle|Amt|Verband|Zweckverband|Firma|Bank|Verlag|GmbH\s*&\s*Co\.\s*KG))\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Universities and Associations`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `8d03de76`  
**Description:**
Matches universities and associations with 'e. V.' or 'Universität' patterns.

**Content:**
```
\b([A-Za-zäöüß]+\s+Technische\s+Universität\s+[A-Za-zäöüß\-]+|[A-Za-zäöüß]+\s*e\.\s*V\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Complex Court Departments`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `e8739c37`  
**Description:**
Matches specific court departments including senate numbers and full context (e.g., '25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts').

**Content:**
```
\b\d+\.\s*Senat(?:\s*\([^)]+\))?\s*des\s+(?:Bundespatentgericht|Bundesverwaltungsgericht|Bundesfinanzhof|Bundesgerichtshof|Bundessozialgericht|Bundesarbeitsgericht|Bundesverfassungsgericht|Landessozialgericht|Landesarbeitsgericht|Landesverwaltungsgericht|Oberlandesgericht|Verwaltungsgerichtshof|Oberverwaltungsgericht|Finanzgericht|Sozialgericht|Arbeitsgericht|Amtsgericht|Landgericht)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Patent Office Departments`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `155bed32`  
**Description:**
Matches specific departments of the German Patent and Trade Mark Office including class numbers and full context.

**Content:**
```
\b(?:Markenstelle\s+f\u00fcr\s+Klasse\s+\d+(?:\s+-\s+[A-Za-z\s-]+)?\s+des\s+Deutschen\s+Patent-\s+und\s+Markenamts|Pr\u00fcfungsstelle\s+f\u00fcr\s+Klasse\s+[A-Za-z0-9\s]+\s+des\s+Deutschen\s+Patent-\s+und\s+Markenamts|Markenabteilung\s+\d+\.?\d*\s+des\s+Deutschen\s+Patent-\s+und\s+Markenamts|Patentabteilung\s+\d+\.?\d*\s+des\s+Deutschen\s+Patent-\s+und\s+Markenamts|Markenstelle\s+f\u00fcr\s+Klasse\s+\d+\s+des\s+DPMA|Deutschen\s+Patent-\s+und\s+Markenamts|Deutsche\s+Patent-\s+und\s+Markenamt|Markenstelle\s+f\u00fcr\s+Klasse\s+\d+|Patentabteilung\s+\d+|Designstelle\s+des\s+Deutschen\s+Patent-\s+und\s+Markenamts)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Single Letter Companies (Strict)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `111bce2c`  
**Description:**
Matches single letter or hyphenated codes followed by GmbH/AG/KB, ensuring it's a distinct entity name and not part of a larger phrase.

**Content:**
```
\b([A-Z]\s*\u2026\s*AG|A-GbR|C-\s*B\.\s*V\.|D\s*P\s*T\s*S\s*GmbH|S\s*\u2026\s*AG|B\s*\u2026\s*AG|K1\s*\u2026\s*AG|HSG\s*Z\s*\u2026|Z\s*\u2026|S\s*\u2026|M\s*\u2026|F\.\s*D\.|X\s*GmbH\s*&\s*Co\.\s*KG|X\s*GmbH|X\s*AG|S\s*AG|C\s*GmbH|C-\s*B\.\s*V\.|A-GbR|D\s*P\s*T\s*S\s*GmbH|S\s*\u2026\s*AG|B\s*\u2026\s*AG|K1\s*\u2026\s*AG|HSG\s*Z\s*\u2026|Z\s*\u2026|S\s*\u2026|M\s*\u2026|F\.\s*D\.|X\s*GmbH\s*&\s*Co\.\s*KG|X\s*GmbH|X\s*AG|S\s*AG|C\s*GmbH|C-\s*B\.\s*V\.|A-GbR|D\s*P\s*T\s*S\s*GmbH|S\s*\u2026\s*AG|B\s*\u2026\s*AG|K1\s*\u2026\s*AG|HSG\s*Z\s*\u2026|Z\s*\u2026|S\s*\u2026|M\s*\u2026|F\.\s*D\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Single Letter Entities`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `ca93f0af`  
**Description:**
Matches single letter entities often used as anonymized names in legal texts, e.g., 'A', 'H', 'Wi.', 'T.', 'K.'.

**Content:**
```
\b([A-Z]\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `International Court`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `d7405280`  
**Description:**
Matches 'Internationale Gerichtshof' specifically.

**Content:**
```
\b(Internationale\s+Gerichtshof)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Single Letter Companies with Punctuation`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `c75aa959`  
**Description:**
Matches single letter or hyphenated codes with punctuation followed by GmbH/AG/KB.

**Content:**
```
\b([A-Z]\s*\.\s*[A-Z]\s*\.\s*[A-Z]?|C-\s*B\.\s*V\.|S\s*AG)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 16 | 0 | 16 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 0 | 16 | 542 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `56041`) (sent_id: `56041`)


Das FA hat unter Berufung auf § 123 i. V. m. § 60 Abs. 3 der Finanzgerichtsordnung ( FGO ) , § 174 Abs. 4 , 5 der Abgabenordnung ( AO ) die Beiladung der Gesellschafter A. X. und B. X. für die GbR angeregt .

**False Positives:**

- `A. X. ` — partial — gold is substring of pred: `A. X.`
- `B. X. ` — partial — gold is substring of pred: `B. X.`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 123 i. V. m. § 60 Abs. 3 der Finanzgerichtsordnung`(NRM)
- `FGO`(NRM)
- `§ 174 Abs. 4 , 5 der Abgabenordnung`(NRM)
- `AO`(NRM)
- `A. X.`(ORG)
- `B. X.`(ORG)

**Example 1** (doc_id: `56134`) (sent_id: `56134`)


Da die A-GbR aufgrund ihrer Geschäftsbeziehung der C- B. V. näher stehe als das FA , hätte es dem Kläger oblegen , die Angaben über die wirtschaftlichen Verhältnisse der C- B. V. weiter zu konkretisieren .

**False Positives:**

- `B. V. ` — positional overlap with gold: `C- B. V.`
- `B. V. ` — positional overlap with gold: `C- B. V.`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `A-GbR`(ORG)
- `C- B. V.`(ORG)
- `C- B. V.`(ORG)

**Example 2** (doc_id: `56444`) (sent_id: `56444`)


So wird ausdrücklich darauf verwiesen , dass auf der Anlage K 1. Ausdruck einer Mail von D. an D. B. vom 30. Januar 2011 ) handschriftlich " Rechnungsanschrift Help Food z. o. o. D. B. ( es folgt die postalische Anschrift der Help Food ) " vermerkt ist , auf Seite 2 der Anlage K 3 ( Präsentationsunterlage mit dem Copyright von D. und W. ) unter " Unsere Kontraktbedingungen " ein " Exklusiver Kontrakt für 2 Jahre mit Help Food " und eine " Haushaltsverfügung durch Help Food ... bis zum Ende 2011 Startphase " erwähnt werden , auf Seite 2 der Anlage K 5 ( mit dem Logo der Klägerin versehenes Protokoll eines Treffens der Beteiligten am 26. August 2011 ) von einem " Vorschlag zum Vertrag zwischen Help Food , M. D. und P. W. " die Rede ist , die Anlage K 9 ( von S. unterzeichnetes Schreiben vom 29. Dezember 2011 ) als Absender die Help Food ausweist und die Anlage K 50 ( Ausdruck einer Mail der Zeugin F. an D. und W. vom 14. September 2011 ) die Absenderadresse " m. @helpfood . eu " trägt .

**False Positives:**

- `D. B. ` — similar text (different position): `D.`
- `M. D. ` — similar text (different position): `D.`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `D.`(PER)
- `D. B.`(PER)
- `Help Food`(ORG)
- `Help Food`(ORG)
- `D.`(PER)
- `W.`(PER)
- `Help Food`(ORG)
- `Help Food`(ORG)
- `Help Food`(ORG)
- `M. D.`(PER)
- `P. W.`(PER)
- `S.`(PER)
- `Help Food`(ORG)
- `F.`(PER)
- `D.`(PER)
- `W.`(PER)

**Example 3** (doc_id: `56479`) (sent_id: `56479`)


I. Nach den Feststellungen des Landgerichts kamen der Angeklagte und sein in der Türkei lebender gesondert verfolgter Bruder E. K. spätestens zu Beginn des Jahres 2015 überein , in arbeitsteiligem Zusammenwirken in der Türkei hergestellte bzw. erworbene Kleidungsstücke , die mit Schriftzügen und Labels verschiedener Markenhersteller versehen waren , unter Verletzung geschützter Gemeinschafts- bzw. Unionsmarken in Deutschland zu verkaufen , obwohl ihnen bewusst war , dass sie nicht über die für deren Verwendung erforderliche Zustimmung der Markenrechtsinhaber verfügten .

**False Positives:**

- `E. K. ` — partial — gold is substring of pred: `E. K.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Türkei`(LOC)
- `E. K.`(PER)
- `Türkei`(LOC)
- `Deutschland`(LOC)

**Example 4** (doc_id: `56849`) (sent_id: `56849`)


Diese Tatsache liegt darin , dass die C- B. V. zwischen dem Jahr 2003 bis zu ihrer Auflösung im Jahr 2006 durchgängig , und damit auch zum maßgeblichen Bilanzstichtag am 31. Dezember 2004 , über kein hinreichendes Vermögen verfügte , um das Schuldanerkenntnis über 2 Mio. € ganz oder in Teilbeträgen zu tilgen .

**False Positives:**

- `B. V. ` — positional overlap with gold: `C- B. V.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C- B. V.`(ORG)

**Example 5** (doc_id: `57157`) (sent_id: `57157`)


Die C- B. V. wurde bereits im Jahr 2006 aufgelöst .

**False Positives:**

- `B. V. ` — positional overlap with gold: `C- B. V.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C- B. V.`(ORG)

**Example 6** (doc_id: `57273`) (sent_id: `57273`)


Der Kläger betrieb vormals eine Anwaltssozietät mit Rechtsanwalt C. B. in F. .

**False Positives:**

- `C. B. ` — partial — gold is substring of pred: `C. B.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C. B.`(PER)
- `F.`(LOC)

**Example 7** (doc_id: `58560`) (sent_id: `58560`)


H. N. hatte sich , wie zuvor verabredet , teilweise maskiert und ging in die Halle 4a der Spielhalle .

**False Positives:**

- `H. N. ` — partial — gold is substring of pred: `H. N.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `H. N.`(PER)

**Example 8** (doc_id: `59356`) (sent_id: `59356`)


Über die sofortige Beschwerde der Nebenklägerin K. T. gegen die im Urteil des Landgerichts Frankfurt am Main vom 15. August 2017 getroffene Kostenentscheidung hat das Oberlandesgericht Frankfurt am Main zu entscheiden .

**False Positives:**

- `K. T. ` — partial — gold is substring of pred: `K. T.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K. T.`(PER)
- `Landgerichts Frankfurt am Main`(ORG)
- `Oberlandesgericht Frankfurt am Main`(ORG)

**Example 9** (doc_id: `59650`) (sent_id: `59650`)


D20 I. C. Madsen et al. , “ Description and survey of methodologies for the determination of amorphous content via X-ray powder diffraction ” , Z. Kristallgr . 2011 , 226 , Seiten 944 bis 955

**False Positives:**

- `I. C. ` — partial — pred is substring of gold: `I. C. Madsen`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `I. C. Madsen`(PER)

**Example 10** (doc_id: `59658`) (sent_id: `59658`)


A. I. 1. Der Beschwerdeführer ist auf Grundlage des Urteils des Landgerichts Lübeck vom 7. Oktober 2014 gemäß § 63 StGB wegen Mordes in einem psychiatrischen Krankenhaus der AMEOS Krankenhausgesellschaft Holstein mbH untergebracht , nachdem er im schuldunfähigen Zustand auf Grund einer wahnhaften Störung im Januar 2014 seine vierjährige Tochter und seinen sechs Jahre alten Sohn getötet hatte .

**False Positives:**

- `A. I. ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Landgerichts Lübeck`(ORG)
- `§ 63 StGB`(NRM)
- `AMEOS Krankenhausgesellschaft Holstein mbH`(ORG)

**Example 11** (doc_id: `59786`) (sent_id: `59786`)


Es sei jedoch nicht erkennbar , dass der Kläger sich bemüht habe , aus der Sphäre der C- B. V. substantiierte Angaben über ihre wirtschaftlichen Verhältnisse zu erhalten .

**False Positives:**

- `B. V. ` — positional overlap with gold: `C- B. V.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C- B. V.`(ORG)

**Example 12** (doc_id: `59897`) (sent_id: `59897`)


So legt der Kläger - soweit erkennbar unwidersprochen durch das FA - dar , dass das mit der A-GbR vereinbarte Projekt eines Ferienresorts die einzige geschäftliche Tätigkeit der C- B. V. dargestellt habe .

**False Positives:**

- `B. V. ` — positional overlap with gold: `C- B. V.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `A-GbR`(ORG)
- `C- B. V.`(ORG)

</details>

---

## `Specific Court Names with Location (Genitive)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `2d503e75`  
**Description:**
Matches court names in genitive form followed by location, e.g., 'Landgerichts Bückeburg', 'Landgerichts Traunstein', and complex genitive forms.

**Content:**
```
\b(?:Amtsgerichts|Landgerichts|Verwaltungsgerichts|Oberlandesgerichts|Landesarbeitsgerichts|Landessozialgerichts|Sozialgerichts|Finanzgerichts|Arbeitsgerichts|Bundesverwaltungsgerichts|Bundesfinanzhofs|Bundesgerichtshofs|Bundessozialgerichts|Bundesarbeitsgerichts|Bundespatentgerichts|Oberverwaltungsgerichts|Verwaltungsgerichtshofs|Hamburgischen Oberverwaltungsgerichts|Schleswig-Holsteinische Verwaltungsgericht|Schleswig-Holsteinische Oberlandesgericht|Bayerischen Landeszentrale|Bayerischen Landessozialgerichts)\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)*\b(?!\s+(?:Senat|Nr\.|\.)|\s+(?:Senat|Nr\.))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Quoted Brand Names (Strict)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `63bcacd5`  
**Description:**
Matches brand names enclosed in German quotation marks, excluding common non-brand terms like 'Entscheidungen', 'Urteil', 'Beschluss', etc.

**Content:**
```
(?:\u201e|\")\s*([A-Za-z\u00e4\u00f6\u00fc\u00df\s]+)\s*(?:\u201e|\")
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 224 | 0 | 224 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 0 | 224 | 798 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53417`) (sent_id: `53417`)


aa ) Das Amt des " Kanzlers " hat seine Wurzeln im Mittelalter und spiegelt die Entwicklung der modernen Universität bis in die heutigen Tage .

**False Positives:**

- `Kanzlers ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `53453`) (sent_id: `53453`)


Zwar hat sich das FG im Urteil nicht ausdrücklich dazu geäußert , es hat die umstrittene Zahlung aber ohne weiteres als " Abfindungszahlung " bezeichnet und nicht infrage gestellt , dass es sich ( zumindest auch ) um eine Ersatzleistung für entgehende Einnahmen handeln sollte .

**False Positives:**

- `Abfindungszahlung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `53491`) (sent_id: `53491`)


die unter 2. bis 5. begehrten Informationen jeweils nur für den " Hintergrund " , also vertraulich und nicht zur Verwendung für eine öffentliche Berichterstattung mit Quellenangabe zu erteilen .

**False Positives:**

- `Hintergrund ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 3** (doc_id: `53588`) (sent_id: `53588`)


Diesbezüglich fehlt es insbesondere an Vorbringen dazu , dass die Entscheidung des LSG auf " diesem Mangel " beruhen kann , dh es hätte Darlegungen zur mangelnden oder zumindest eingeschränkten Verwertbarkeit des Gutachtens bedurft .

**False Positives:**

- `diesem Mangel ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `53592`) (sent_id: `53592`)


DDO / DtA ENJJPT stellt den Dienstbetrieb im " Euro Nato Joint Jet Pilot Training " ( ENJJPT ) sicher .

**False Positives:**

- `Euro Nato Joint Jet Pilot Training ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `53615`) (sent_id: `53615`)


Weil Impfstoffe Arzneimittel ( auch ) iS des § 31 SGB V sind , wird ihre Verordnung auch von § 10 Abs 2 der Prüfvereinbarung erfasst , soweit diese Vorschrift das " Verordnungsverhalten " der Vertragsärzte betrifft .

**False Positives:**

- `Verordnungsverhalten ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 31 SGB V`(NRM)
- `§ 10 Abs 2 der Prüfvereinbarung`(REG)

**Example 6** (doc_id: `53630`) (sent_id: `53630`)


Die Verträge seien derart angelegt gewesen , dass die KG am Ende der Laufzeit ihr " Andienungsrecht " ausübe und die P GmbH die Leasingobjekte zurückerwerben müsse .

**False Positives:**

- `Andienungsrecht ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `P GmbH`(ORG)

**Example 7** (doc_id: `53648`) (sent_id: `53648`)


Das FA vertritt indes die Auffassung , der für das erste Betriebsjahr vereinbarte Abschlag auf den voraussichtlichen Jahresüberschuss in Höhe von 40 % des Nettokaufpreises sei als Vereinbarung der Rückzahlung der " Darlehenssumme " anzusehen .

**False Positives:**

- `Darlehenssumme ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `53720`) (sent_id: `53720`)


cc ) Ohne dass es hierauf nach dem Vorstehenden rechtlich noch ankäme , weist der Senat darauf hin , dass er auch der in der mündlichen Verhandlung vorgebrachten Auffassung des FA nicht folgen kann , dem Kläger habe allenfalls eine " abschnittsweise Verlustbeteiligung " , aber keine " endgültige Verlustbeteiligung " gedroht .

**False Positives:**

- `abschnittsweise Verlustbeteiligung ` — no gold match — likely missing annotation
- `endgültige Verlustbeteiligung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 9** (doc_id: `53729`) (sent_id: `53729`)


Nach dem während der ambulanten Behandlung der Tochter erstellten Bericht habe sich das Mädchen auf einer Gesamtskala im Störungsbereich " Depression " auffällig gezeigt .

**False Positives:**

- `Depression ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 10** (doc_id: `53767`) (sent_id: `53767`)


Ob der Erwerb eigener Anteile auf der Gesellschaftsebene entsprechend der durch das BilMoG geänderten handelsrechtlichen Vorschriften ( Einfügung des § 272 Abs. 1a und 1b HGB ) steuerrechtlich nicht mehr als Erwerbsvorgang anzusehen , sondern nunmehr als " Teilliquidation " und daher " wie " eine Kapitalherabsetzung zu behandeln ist ( so BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f. ; zustimmend z.B. Blumenberg / Lechner , DB 2014 , 141 , 147 ; kritisch z.B. Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55 ) , kann im Streitfall mangels Erheblichkeit für die Entscheidung offenbleiben .

**False Positives:**

- `Teilliquidation ` — no gold match — likely missing annotation
- `wie ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `BilMoG`(NRM)
- `§ 272 Abs. 1a und 1b HGB`(NRM)
- `BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f.`(REG)
- `Blumenberg / Lechner , DB 2014 , 141 , 147`(LIT)
- `Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55`(LIT)

**Example 11** (doc_id: `53836`) (sent_id: `53836`)


Stattgebender Kammerbeschluss : Anforderungen der Rechtsschutzgarantie ( Art 19 Abs 4 S 1 GG , hier iVm Art 2 Abs 2 S 1 GG ) an die Begründung der Abweisung einer Klage auf Zuerkennung internationalen Schutzes sowie auf Feststellung von nationalen Abschiebungsverboten als offensichtlich unbegründet - Pflicht zur " tagesaktuellen " Beurteilung der Sicherheitslage in Afghanistan steht Bildung einer insofern gefestigten obergerichtlichen Rspr entgegen , mithin keine Grundlage für Versagung subsidiären Schutzes wegen offensichtlicher Unbegründetheit - Gegenstandswertfestsetzung

**False Positives:**

- `tagesaktuellen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Art 19 Abs 4 S 1 GG`(NRM)
- `Art 2 Abs 2 S 1 GG`(NRM)
- `Afghanistan`(LOC)

**Example 12** (doc_id: `53863`) (sent_id: `53863`)


Eine Trennung nach " Streikbeamten " und sonstigen Beamten widerspreche insbesondere dem hergebrachten Grundsatz der Einheit des Berufsbeamtentums .

**False Positives:**

- `Streikbeamten ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 13** (doc_id: `53916`) (sent_id: `53916`)


Unter Bezugnahme auf die entsprechenden Feststellungen des Landgerichts stützt sich die Entscheidung darauf , dass der Beschwerdeführer einer aus rund 80 Personen bestehenden Gruppe namens " Schickeria " aus der gewaltbereiten " Ultra " -Szene angehört und sich nach dem fraglichen Spiel in einer Gruppe befunden habe , aus welcher heraus es tatsächlich in erheblichem Umfang zu Provokationen und Körperverletzungsdelikten gekommen sei .

**False Positives:**

- `Schickeria ` — no gold match — likely missing annotation
- `Ultra ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 14** (doc_id: `54096`) (sent_id: `54096`)


Die Folgen des " fehlenden " Kopfteils für die anderen Mitglieder des Haushalts aufgrund einer Versagung gegenüber einem dritten Mitglied , weil dieses die ua in §§ 60 ff SGB I iVm § 9 und §§ 11 ff SGB II zum Ausdruck kommenden Verhaltenserwartungen nicht erfüllt , sind nicht durch höhere Einzelansprüche der anderen Haushaltsmitglieder auszugleichen ( dazu im Einzelnen 4. und 5. ) .

**False Positives:**

- `fehlenden ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§§ 60 ff SGB I`(NRM)
- `§ 9 und §§ 11 ff SGB II`(NRM)

**Example 15** (doc_id: `54106`) (sent_id: `54106`)


Diese Entscheidung hat die Beklagte während des Klageverfahrens " abgeändert " und festgestellt , dass der Kläger in seinen für die Beigeladene zu 1. ausgeübten Tätigkeiten in allen Zweigen der Sozialversicherung wegen Beschäftigung versicherungspflichtig gewesen sei ; die Versicherungspflicht habe schon mit der Aufnahme seiner Beschäftigung begonnen , weil er ausreichenden Versicherungsschutz iS von § 7a Abs 6 S 1 Nr 2 SGB IV nicht nachgewiesen habe ( Bescheid vom 24. 1. 2011 ) .

**False Positives:**

- `abgeändert ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 7a Abs 6 S 1 Nr 2 SGB IV`(NRM)

**Example 16** (doc_id: `54130`) (sent_id: `54130`)


b ) In Leasingfällen ( auch beim " Lease " im Rahmen eines " Sale-and-lease-back " ) geht der BFH - wie bereits in dem Urteil in BFHE 255 , 386 , BStBl II 2018 , 81 dargestellt - bei Anwendung des § 39 Abs. 2 Nr. 1 Satz 1 AO insbesondere von folgenden Grundsätzen aus :

**False Positives:**

- `Lease ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BFH`(ORG)
- `Urteil in BFHE 255 , 386 , BStBl II 2018 , 81`(RS)
- `§ 39 Abs. 2 Nr. 1 Satz 1 AO`(NRM)

**Example 17** (doc_id: `54134`) (sent_id: `54134`)


bb ) Die Bedeutung der Konjunktion " soweit " ist jedoch nicht auf solche Fälle beschränkt .

**False Positives:**

- `soweit ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 18** (doc_id: `54142`) (sent_id: `54142`)


- einem Betrag in Höhe des als Kaufoptionspreis im Anhang 2 aufgeführten Betrags , der um den Betrag einer etwa bereits gezahlten Abschlusszahlung zu mindern ist , - einem nach näheren Maßgaben zu ermittelnden sog. " höheren Marktwert " und- der variablen Gebühr .

**False Positives:**

- `höheren Marktwert ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `54229`) (sent_id: `54229`)


Der Sache nach hat der Beklagte einen Regress wegen unwirtschaftlicher Verordnung von Impfstoffen und keinen verschuldensabhängigen Ersatz wegen der Verursachung eines " sonstigen Schadens " festgesetzt ( c ) .

**False Positives:**

- `sonstigen Schadens ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 20** (doc_id: `54257`) (sent_id: `54257`)


Indem die Absender die Postsendungen auf den Weg gebracht , zumindest in einem Teil der Fälle unzureichende Angaben gemacht oder Waren versandt haben , die gegen Verbote und Beschränkungen verstoßen könnten , haben sie zwar eine Bedingung für die vorübergehende Verwahrung bei der Zollstelle in Gestalt einer sog. conditio sine qua non gesetzt , welche allerdings allein für die Annahme eines " willentlichen Herbeiführens einer Amtshandlung " im Sinne vorgenannter Rechtsprechung des BVerwG ( Urteile in BVerwGE 91 , 109 , und in BVerwGE 153 , 321 ) nicht als ausreichend angesehen werden kann .

**False Positives:**

- `willentlichen Herbeiführens einer Amtshandlung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BVerwG`(ORG)
- `Urteile in BVerwGE 91 , 109 , und in BVerwGE 153 , 321`(RS)

**Example 21** (doc_id: `54293`) (sent_id: `54293`)


Auch kann dies nicht - wie von der Beklagten vorgetragen - als eine weitere Voraussetzung , unter der erst ein Zusammenhang zwischen Berufstätigkeit und Pflichtmitgliedschaft zu bejahen ist , aus der Verwendung der Präposition " wegen " entnommen werden .

**False Positives:**

- `wegen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 22** (doc_id: `54299`) (sent_id: `54299`)


Auch vor dem Hintergrund der Einschätzung der Sachverständigen , dass beim Beschwerdeführer eine deutlich höhere Gefahr von " Hands-off " -Übergriffen im Vergleich zu " Hands-on " -Delikten bestehe , hätte es der konkreten Darlegung der vom Beschwerdeführer drohenden Straftaten bedurft , um die Gefahr " erheblicher Straftaten " im Sinne von § 67d Abs. 2 StGB feststellen und die Verhältnismäßigkeit einer weiteren Unterbringung des Beschwerdeführers bewerten zu können .

**False Positives:**

- `erheblicher Straftaten ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 67d Abs. 2 StGB`(NRM)

**Example 23** (doc_id: `54371`) (sent_id: `54371`)


Hierdurch wurden dem etablierten Arzt der BAG typischerweise deutlich weniger RLV-relevante Fälle zugeordnet , als er real behandelte , während der " Wachstumsarzt " bei der von der Beklagten praktizierten Verfahrensweise nur die Zahl der von ihm im Abrechnungsquartal tatsächlich betreuten Fälle RLV-relevant vergütet bekam .

**False Positives:**

- `Wachstumsarzt ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 24** (doc_id: `54373`) (sent_id: `54373`)


Das Landgericht Düsseldorf begründete diese Anordnung damit , dass die Disposition zu " solchen Taten " tief im Beschwerdeführer verwurzelt sei .

**False Positives:**

- `solchen Taten ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Landgericht Düsseldorf`(ORG)

**Example 25** (doc_id: `54396`) (sent_id: `54396`)


In dem Merkblatt wird zur " Erschöpfung des Rechtswegs " erläutert , dass die Möglichkeit genutzt werden muss , den Grundrechtsverstoß " im Verfahren vor den Fachgerichten abzuwenden " .

**False Positives:**

- `Erschöpfung des Rechtswegs ` — no gold match — likely missing annotation
- `im Verfahren vor den Fachgerichten abzuwenden ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 26** (doc_id: `54428`) (sent_id: `54428`)


Der dort vorgesehene Versetzungsschutz bei Versetzungen in zeitlicher Nähe zum Dienstzeitende wird nicht , wie das Bundesministerium der Verteidigung einwendet , durch die Vorgaben der Zentralen Dienstvorschrift A- 1350/66 über die " Letzte Verwendung vor Zurruhesetzung " ausgeschlossen .

**False Positives:**

- `Letzte Verwendung vor Zurruhesetzung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Bundesministerium der Verteidigung`(ORG)
- `Zentralen Dienstvorschrift A- 1350/66`(REG)

**Example 27** (doc_id: `54445`) (sent_id: `54445`)


bbb ) Das FG hat feststellt , der Kläger habe für die Streitjahre Umsatzsteuer-Voranmeldungen und -Erklärungen abgegeben und zur Prüfung der geltend gemachten Vorsteuerbeträge und Betriebsausgaben mit Vorsteuerabzug alle Belege und Summenziehungen ( die " Bestände der Vorsteuerbeträge " ) im Klageverfahren vorgelegt .

**False Positives:**

- `Bestände der Vorsteuerbeträge ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 28** (doc_id: `54483`) (sent_id: `54483`)


Nachdem die Schüler im Musikunterricht die theoretischen Grundlagen zum Thema " Musik und Werbung " bzw " Wirkung von Musik " erarbeitet hatten , sollten sie in Kleingruppen einen Werbeclip zu einem bestimmten Produkt filmen , schneiden , bearbeiten und mit passender Musik unterlegen .

**False Positives:**

- `Musik und Werbung ` — no gold match — likely missing annotation
- `Wirkung von Musik ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 29** (doc_id: `54508`) (sent_id: `54508`)


2. Die Versetzung des Antragstellers nach C. verstößt jedoch gegen die nach dem Grundsatz der Gleichbehandlung ( Art. 3 Abs. 1 GG ) zu beachtenden Maßgaben der Bereichsvorschrift C1 - 1310/0 - 2001 über die " Organisatorische und personelle Umsetzung von Strukturentscheidungen in der Luftwaffe " ( siehe bereits Beschluss vom 13. Dezember 2017 - 1 WDS-VR 9.17 - Rn. 33 bis 35 ) .

**False Positives:**

- `Organisatorische und personelle Umsetzung von Strukturentscheidungen in der Luftwaffe ` — partial — gold is substring of pred: `Luftwaffe`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C.`(LOC)
- `Art. 3 Abs. 1 GG`(NRM)
- `Bereichsvorschrift C1 - 1310/0 - 2001`(REG)
- `Luftwaffe`(ORG)
- `Beschluss vom 13. Dezember 2017 - 1 WDS-VR 9.17 - Rn. 33 bis 35`(RS)

**Example 30** (doc_id: `54511`) (sent_id: `54511`)


Da hiernach Basis der Vergütung der " Wachstumsärztin " S. nicht die Fallzahl aus dem Vorjahresquartal , sondern die im Abrechnungsquartal tatsächlich erreichte Fallzahl ( maximal bis zum Fachgruppendurchschnitt ) gewesen sei , habe in der Vorab-Mitteilung eine feste Gesamt-Obergrenze nicht angegeben werden können .

**False Positives:**

- `Wachstumsärztin ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `S.`(PER)

**Example 31** (doc_id: `54651`) (sent_id: `54651`)


Die farbliche Gestaltung in Grün unterstütze lediglich die insgesamt positive Aussage der Kennzeichnung , da die Farbe Grün häufig als Synonym für " natürlich " , " ökologisch " oder " naturbelassen " verwendet werde .

**False Positives:**

- `natürlich ` — no gold match — likely missing annotation
- `ökologisch ` — no gold match — likely missing annotation
- `naturbelassen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 3

**Example 32** (doc_id: `54707`) (sent_id: `54707`)


Im Übrigen fehlt es an Darlegungen dazu , an welcher Stelle seiner " Entscheidung " der BGH die vom Beklagten behauptete Aussage überhaupt getroffen haben soll .

**False Positives:**

- `Entscheidung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BGH`(ORG)

**Example 33** (doc_id: `54737`) (sent_id: `54737`)


Eine " Zwangssituation " , auf die es nach der Rechtsprechung für die Annahme einer Entschädigung i. S. von § 24 Nr. 1 EStG ankommt , ist , wenn man an dem Erfordernis festhält ( zweifelnd BFH-Urteil vom 23. November 2016 X R 48/14 , BFHE 256 , 290 , BStBl II 2017 , 383 , Rz 26 ) , beim Arbeitnehmer jedenfalls nicht deshalb zu verneinen , weil er einer gütlichen Einigung zugestimmt hat ( vgl. BFH-Urteil in BFHE 237 , 56 , BStBl II 2012 , 569 ) .

**False Positives:**

- `Zwangssituation ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 24 Nr. 1 EStG`(NRM)
- `BFH-Urteil vom 23. November 2016 X R 48/14 , BFHE 256 , 290 , BStBl II 2017 , 383 , Rz 26`(RS)
- `BFH-Urteil in BFHE 237 , 56 , BStBl II 2012 , 569`(RS)

**Example 34** (doc_id: `54807`) (sent_id: `54807`)


Zwar hat eine Fortschreibung nach § 558d Abs 2 BGB - auch ohne Erhöhung der Werte - zwingend zu erfolgen , während § 22c Abs 2 SGB II einen vom Gesetzgeber nicht näher konkretisierten Spielraum der Fortschreibung durch den Satzungsgeber belässt ( " gegebenenfalls " ) .

**False Positives:**

- `gegebenenfalls ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 558d Abs 2 BGB`(NRM)
- `§ 22c Abs 2 SGB II`(NRM)

**Example 35** (doc_id: `54852`) (sent_id: `54852`)


Eine Dosis von 1 bis 5 mg wird demgegenüber für die " daily " -Verabreichung nicht offenbart .

**False Positives:**

- `daily ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 36** (doc_id: `54890`) (sent_id: `54890`)


Dessen Aufhebungsverwaltungsakt bezeichnet in seinem Verfügungssatz die Bewilligungsentscheidungen , die vom 1. 1. 2005 bis 30. 4. 2010 " ganz zurückgenommen " werden , und der Erstattungsverwaltungsakt beziffert in seinem Verfügungssatz eine zu erstattende " Gesamtforderung " in Höhe von 48 179,87 Euro sowie die Teilbeträge , aus denen sie sich zusammensetzt .

**False Positives:**

- `ganz zurückgenommen ` — no gold match — likely missing annotation
- `Gesamtforderung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 37** (doc_id: `54928`) (sent_id: `54928`)


Die Bedeutung von " wegen " erschöpft sich damit in der Herstellung eines ursächlichen Verhältnisses .

**False Positives:**

- `wegen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 38** (doc_id: `54973`) (sent_id: `54973`)


Das Legen eines Hauswasseranschlusses ist auch dann als " Lieferung von Wasser " i. S. des § 12 Abs. 2 Nr. 1 UStG i. V. m. Nr. 34 der Anlage 2 zum UStG anzusehen , wenn diese Leistung nicht von dem Wasserversorgungsunternehmen erbracht wird , das das Wasser liefert ( Anschluss an das BGH-Urteil vom 18. April 2012 VIII ZR 253/11 , HFR 2012 , 1110 ) .

**False Positives:**

- `Lieferung von Wasser ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 12 Abs. 2 Nr. 1 UStG`(NRM)
- `Nr. 34 der Anlage 2 zum UStG`(NRM)
- `BGH-Urteil vom 18. April 2012 VIII ZR 253/11 , HFR 2012 , 1110`(RS)

**Example 39** (doc_id: `55103`) (sent_id: `55103`)


Festgestellt hat das LSG insoweit nur , dass K. bis zu ihrer Aufnahme in das Diakonissenhaus An. bei ihrer Mutter in B. , im heutigen Kreisgebiet des Beklagten , " gemeldet " war ; auf die einwohnerrechtliche Meldung kommt es jedoch für die Bestimmung des gewöhnlichen Aufenthalts nicht an ( BSG SozR 5870 § 1 Nr 4 ; SozR 3 - 5870 § 2 Nr 36 ) .

**False Positives:**

- `gemeldet ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `K.`(PER)
- `Diakonissenhaus An.`(ORG)
- `B.`(LOC)
- `BSG SozR 5870 § 1 Nr 4`(RS)
- `SozR 3 - 5870 § 2 Nr 36`(RS)

**Example 40** (doc_id: `55113`) (sent_id: `55113`)


Der Kläger subsumiert die Sachverhaltskonstellation seines Falls vielmehr selbst anhand der vorgenannten Rechtsprechung des BSG und zieht daraus im Kern den Schluss , dass das LSG auf der Basis dieser höchstrichterlichen Rechtsprechung seinen Fall " falsch entschieden " habe .

**False Positives:**

- `falsch entschieden ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG`(ORG)

**Example 41** (doc_id: `55140`) (sent_id: `55140`)


Vorliegend werden die angesprochenen Verkehrskreise die Wortfolge " gesundleben " , wenn sie ihnen i. V. m. den vorliegend beanspruchten Waren begegnet , lediglich als einen werbenden Hinweis dahingehend auffassen , dass die so bezeichneten Waren dazu bestimmt sind , zu einem gesunden Leben beizutragen , oder sich mit dieser Thematik befassen .

**False Positives:**

- `gesundleben ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 42** (doc_id: `55259`) (sent_id: `55259`)


Hinsichtlich der M ... Adresse werde dies anhand des Zusatzes , dass sich die Wohnung des Beschwerdeführers " bei Fa. B ... " befinde und somit gerade nicht " in " den zu durchsuchenden Geschäftsräumen , deutlich .

**False Positives:**

- `befinde und somit gerade nicht ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `M ...`(LOC)
- `B ...`(ORG)

**Example 43** (doc_id: `55289`) (sent_id: `55289`)


Die Ausschüttungsfiktion des § 20 Abs. 1 Nr. 10 Buchst. b EStG und die Fiktion des § 44 Abs. 6 Satz 1 EStG , die Trägerkörperschaft als Gläubigerin der Kapitalerträge und den Betrieb gewerblicher Art als Schuldner der Kapitalerträge anzusehen , beruhen auf dem Gedanken , den Betrieb gewerblicher Art zur Schaffung zweier Besteuerungsebenen wie eine " virtuelle Kapitalgesellschaft " ( Märtens in Gosch , KStG , 3. Aufl. , § 4 Rz 22 ) zu behandeln .

**False Positives:**

- `virtuelle Kapitalgesellschaft ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 20 Abs. 1 Nr. 10 Buchst. b EStG`(NRM)
- `§ 44 Abs. 6 Satz 1 EStG`(NRM)
- `Märtens in Gosch , KStG , 3. Aufl. , § 4 Rz 22`(LIT)

**Example 44** (doc_id: `55313`) (sent_id: `55313`)


Nur am Rande sei daher angemerkt , dass – sofern man ein dahingehendes Verständnis mit der Begründung ausschließen wollte , dass derartige Waren unter keinen Umständen zu einem " gesunden Leben " beitragen könnten , – das angemeldete Zeichen ersichtlich geeignet wäre , das Publikum über die Art bzw. die Beschaffenheit dieser Waren zu täuschen i. S. v. § 8 Abs. 2 Nr. 4 MarkenG .

**False Positives:**

- `gesunden Leben ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 8 Abs. 2 Nr. 4 MarkenG`(NRM)

**Example 45** (doc_id: `55328`) (sent_id: `55328`)


Sie lässt unberücksichtigt , dass das LSG in seinen entscheidungstragend herangezogenen Obersätzen der BSG-Rechtsprechung nicht " im Grundsätzlichen " ausdrücklich widersprochen und mit dieser Rechtsprechung nicht zu vereinbarende eigene Rechtssätze aufgestellt hat .

**False Positives:**

- `im Grundsätzlichen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 46** (doc_id: `55395`) (sent_id: `55395`)


Danach steht bei einer Zuordnung der Sozialversicherungsrenten zu den tatbestandlichen Begriffen " Ruhegehälter und ähnliche Vergütungen " ( auch ) Deutschland für die der Klägerin gezahlten Renten das Besteuerungsrecht zu .

**False Positives:**

- `Ruhegehälter und ähnliche Vergütungen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschland`(LOC)

**Example 47** (doc_id: `55485`) (sent_id: `55485`)


Im Störungsbereich " Angst " und " soziale Phobie " sei das Kind ebenfalls sehr auffällig gewesen .

**False Positives:**

- `Angst ` — no gold match — likely missing annotation
- `soziale Phobie ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 48** (doc_id: `55523`) (sent_id: `55523`)


Soweit die Markenstelle pauschal ausführt , " ARANCINI " bezeichne jeweils den Gegenstand der Dienstleistungen , vermag dies in Bezug auf die Dienstleistungen " Werbung ; Geschäftsführung " und " Dienstleistungen zur Beherbergung von Gästen " nicht zu überzeugen .

**False Positives:**

- `ARANCINI ` — no gold match — likely missing annotation
- `und ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 49** (doc_id: `55560`) (sent_id: `55560`)


Mit der Beitragsforderung wurde durch die Beklagte zumindest in die allgemeine Handlungsfreiheit und damit in das Grundrecht des Klägers aus Art 2 Abs 1 GG eingegriffen , wodurch ein anhörungspflichtiger " Eingriff " iS des § 24 Abs 1 SGB X vorlag ( vgl BSG Urteil vom 25. 1. 1979 - 3 RK 35/77 - SozR 1200 § 34 Nr 7 ; Mutschler in Kasseler Komm , Stand September 2015 , § 24 SGB X RdNr 7 ; Siefert in von Wulffen / Schütze , SGB X , 8. Aufl 2014 , § 24 RdNr 8 ) .

**False Positives:**

- `Eingriff ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Art 2 Abs 1 GG`(NRM)
- `§ 24 Abs 1 SGB X`(NRM)
- `BSG Urteil vom 25. 1. 1979 - 3 RK 35/77 - SozR 1200 § 34 Nr 7`(RS)
- `Mutschler in Kasseler Komm , Stand September 2015 , § 24 SGB X RdNr 7`(LIT)
- `Siefert in von Wulffen / Schütze , SGB X , 8. Aufl 2014 , § 24 RdNr 8`(LIT)

**Example 50** (doc_id: `55585`) (sent_id: `55585`)


Der vorzeitige Altersrentenbezug des Klägers sei durch dessen " Finanzierung " seitens der Beigeladenen nicht weggefallen .

**False Positives:**

- `Finanzierung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 51** (doc_id: `55589`) (sent_id: `55589`)


Auch der behauptete , eventuell drohende " Reputationsverlust " durch die unterschiedliche Bewertung von Rechtsfragen würde allenfalls ein allgemeines Interesse am Ausgang des Verfahrens nach § 18 Abs. 2 BVerfGG begründen , das für die Annahme einer Beteiligtenstellung gerade nicht ausreicht .

**False Positives:**

- `Reputationsverlust ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 18 Abs. 2 BVerfGG`(NRM)

**Example 52** (doc_id: `55634`) (sent_id: `55634`)


Die menschenrechtlichen Gehalte des jeweils in Rede stehenden völkerrechtlichen Vertrags müssen im Rahmen eines aktiven ( Rezeptions- ) Vorgangs in den Kontext der aufnehmenden Verfassungsordnung " umgedacht " werden ( vgl. BVerfGE 128 , 326 < 370 > , unter Verweis auf Häberle , Europäische Verfassungslehre , 7. Aufl. 2011 , S. 255 f. ; vgl. auch Dreier , in : Dreier , GG , Bd. 1 , 3. Aufl. 2013 , Art. 1 Abs. 2 Rn. 20 ) .

**False Positives:**

- `umgedacht ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BVerfGE 128 , 326 < 370 >`(RS)
- `Häberle , Europäische Verfassungslehre , 7. Aufl. 2011 , S. 255 f.`(LIT)
- `Dreier , in : Dreier , GG , Bd. 1 , 3. Aufl. 2013 , Art. 1 Abs. 2 Rn. 20`(LIT)

**Example 53** (doc_id: `55712`) (sent_id: `55712`)


Solange auf den in Rede stehenden Flächen Bäume wachsen oder nachwachsen , kann daher von einem " Brachliegenlassen " nicht gesprochen werden , auch wenn über einen langen Zeitraum keine Pflege- oder Erhaltungsmaßnahmen vorgenommen werden .

**False Positives:**

- `Brachliegenlassen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 54** (doc_id: `55742`) (sent_id: `55742`)


Hiermit korrespondiert die abschließende Ankündigung des Brigadegenerals " A " , " weitere Maßnahmen " veranlassen zu wollen , wenn sich nach der Herausgabe der ZDv A- 1420/34 ( " Anwendung der Verordnung über die Arbeitszeit der Soldatinnen und Soldaten " ) und weiterer Folgedokumente zur Soldatenarbeitszeitverordnung eine von seinem Schreiben abweichende Auslegung der Bestimmungen ergeben sollte .

**False Positives:**

- `A ` — partial — pred is substring of gold: `" A "`
- `weitere Maßnahmen ` — no gold match — likely missing annotation
- `Anwendung der Verordnung über die Arbeitszeit der Soldatinnen und Soldaten ` — partial — gold is substring of pred: `Verordnung über die Arbeitszeit der Soldatinnen und Soldaten`

> overlaps gold: 2  |  likely missing annotation: 1

**Gold Entities:**

- `" A "`(PER)
- `ZDv A- 1420/34`(REG)
- `Verordnung über die Arbeitszeit der Soldatinnen und Soldaten`(NRM)
- `Soldatenarbeitszeitverordnung`(NRM)

**Example 55** (doc_id: `55795`) (sent_id: `55795`)


b ) Bereits aus dem Wortlaut " außergewöhnliche Belastungen " folgt - auch ohne einen Klammerverweis auf die §§ 33 bis 33b EStG - , dass § 26a Abs. 2 Satz 1 Halbsatz 1 EStG ( auch ) solche Aufwendungen erfasst , die über den Behinderten-Pauschbetrag i. S. des § 33b Abs. 1 EStG abgedeckt werden ( anders z.B. Blümich / Ettlich , § 26a EStG Rz 25 ; Pflüger in Herrmann / Heuer / Raupach - HHR - , § 26a EStG Rz 60 ; Nacke in Littmann / Bitz / Pust , Das Einkommensteuerrecht , Kommentar , § 33b Rz 51a ) .

**False Positives:**

- `außergewöhnliche Belastungen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§§ 33 bis 33b EStG -`(NRM)
- `§ 26a Abs. 2 Satz 1 Halbsatz 1 EStG`(NRM)
- `§ 33b Abs. 1 EStG`(NRM)
- `Blümich / Ettlich , § 26a EStG Rz 25`(LIT)
- `Pflüger in Herrmann / Heuer / Raupach - HHR - , § 26a EStG Rz 60`(LIT)
- `Nacke in Littmann / Bitz / Pust , Das Einkommensteuerrecht , Kommentar , § 33b Rz 51a`(LIT)

**Example 56** (doc_id: `55796`) (sent_id: `55796`)


Die Klägerin ist aber keine " unechte " Grenzgängerin , weil sie nicht lediglich sporadisch in ihren Wohnmitgliedstaat zurückgekehrt ist , sondern - wie bereits dargelegt ( s unter 2 ) - als Grenzgängerin iS von Art 1 Buchst f VO ( EG ) Nr 883/2004 regelmäßig wenigstens einmal wöchentlich ( zur Abgrenzung s auch BSG vom 3. 7. 2003 - B 7 AL 42/02 R - SozR 4 - 6050 Art 71 Nr 2 RdNr 20 ; BSG vom 13. 6. 1985 - 7 RAr 62/83 - juris RdNr 19 ) .

**False Positives:**

- `unechte ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Art 1 Buchst f VO ( EG ) Nr 883/2004`(NRM)
- `BSG vom 3. 7. 2003 - B 7 AL 42/02 R - SozR 4 - 6050 Art 71 Nr 2 RdNr 20`(RS)
- `BSG vom 13. 6. 1985 - 7 RAr 62/83 - juris RdNr 19`(RS)

**Example 57** (doc_id: `55800`) (sent_id: `55800`)


Den Begriff des " Wohnorts " , dem eine eigenständige unionsrechtliche Bedeutung zukommt ( vgl EuGH vom 25. 2. 1999 - C- 90/97 < Swaddling > - juris RdNr 28 = EuGHE I 1999 , 1075 ) , definiert Art 1 Buchst j VO ( EG ) Nr 883/2004 als den Ort des gewöhnlichen Aufenthalts einer Person , in dem sich der gewöhnliche Mittelpunkt deren Interessen befindet ( vgl EuGH vom 5. 6. 2014 - C- 255/13 < Health Service > - juris RdNr 44 = ZESAR 2014 , 495 ff ; BSG vom 3. 4. 2014 - B 2 U 25/12 R - BSGE 115 , 256 = SozR 4 - 2700 § 136 Nr 6 , RdNr 22 ; s auch Art 11 der VO < EG > Nr 987/2009 ) .

**False Positives:**

- `Wohnorts ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `EuGH vom 25. 2. 1999 - C- 90/97 < Swaddling > - juris RdNr 28 = EuGHE I 1999 , 1075`(RS)
- `Art 1 Buchst j VO ( EG ) Nr 883/2004`(NRM)
- `EuGH vom 5. 6. 2014 - C- 255/13 < Health Service > - juris RdNr 44 = ZESAR 2014 , 495 ff`(RS)
- `BSG vom 3. 4. 2014 - B 2 U 25/12 R - BSGE 115 , 256 = SozR 4 - 2700 § 136 Nr 6 , RdNr 22`(RS)
- `Art 11 der VO < EG > Nr 987/2009`(NRM)

**Example 58** (doc_id: `55806`) (sent_id: `55806`)


Es entstehe der Eindruck einer " Personalleihe " von der Verwaltung an die sie kontrollierenden Gerichte .

**False Positives:**

- `Personalleihe ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 59** (doc_id: `55848`) (sent_id: `55848`)


Aber auch insoweit hat sich die Beschwerde weder mit den tatbestandlichen Voraussetzungen noch mit der hierzu ergangenen Rechtsprechung des BSG auseinandergesetzt , nach der der Nachweis einer Primärschädigung im Vollbeweis geführt werden muss und deshalb Ermittlungen zur Kausalität auf der Grundlage des abgesenkten Beweismaßstabs der Wahrscheinlichkeit für einen Nachweis " nicht erkennbar zutage getretener Primärschädigungen " nicht ausreichen .

**False Positives:**

- `nicht erkennbar zutage getretener Primärschädigungen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG`(ORG)

**Example 60** (doc_id: `55851`) (sent_id: `55851`)


Im Fall 33 der Urteilsgründe beruht die Verurteilung wegen Handeltreibens mit Betäubungsmitteln in nicht geringer Menge darauf , dass der Angeklagte in der " Bunkerwohnung " seines Nachbarn K. über ein Kilogramm Amphetaminzubereitung und etwa 1,5 kg Amphetaminöl sowie gut 350 Gramm Marihuana aufbewahrte , die zum gewinnbringenden Verkauf vorgesehen waren .

**False Positives:**

- `Bunkerwohnung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `K.`(PER)

**Example 61** (doc_id: `55867`) (sent_id: `55867`)


Kann eine bereits nach § 106 Abs 5a SGB V erfolgte Beratung in eine Beratung nach § 106 Abs. 5e SGB V umgewidmet werden , ohne dass diese " individuell " nach Maßgabe des § 106 Abs. 5e Satz 1 SGB V erfolgte ?

**False Positives:**

- `individuell ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 106 Abs 5a SGB V`(NRM)
- `§ 106 Abs. 5e SGB V`(NRM)
- `§ 106 Abs. 5e Satz 1 SGB V`(NRM)

**Example 62** (doc_id: `55908`) (sent_id: `55908`)


Der Wortlaut des § 14 Abs. 2 Satz 2 TzBfG , wonach " bereits zuvor " bestehende Arbeitsverhältnisse entscheidend seien , sei nicht eindeutig , sondern könne etwa " jemals zuvor " , " irgendwann zuvor " oder " unmittelbar zuvor " bedeuten ( vgl. BAG , Urteil vom 6. April 2011 - 7 AZR 716/09 - , BAGE 137 , 275 < 279 Rn. 17 > ; Urteil vom 21. September 2011 - 7 AZR 375/10 - , BAGE 139 , 213 < 220 Rn. 24 > ) .

**False Positives:**

- `bereits zuvor ` — no gold match — likely missing annotation
- `jemals zuvor ` — no gold match — likely missing annotation
- `irgendwann zuvor ` — no gold match — likely missing annotation
- `unmittelbar zuvor ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 4

**Gold Entities:**

- `§ 14 Abs. 2 Satz 2 TzBfG`(NRM)
- `BAG , Urteil vom 6. April 2011 - 7 AZR 716/09 - , BAGE 137 , 275 < 279 Rn. 17 >`(RS)
- `Urteil vom 21. September 2011 - 7 AZR 375/10 - , BAGE 139 , 213 < 220 Rn. 24 >`(RS)

**Example 63** (doc_id: `55942`) (sent_id: `55942`)


Das Anmeldezeichen " Schmucke Sache " sei eine sprachüblich gebildete Wortkombination , die sinngemäß " hübsche Sache " bedeute .

**False Positives:**

- `Schmucke Sache ` — no gold match — likely missing annotation
- `hübsche Sache ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 64** (doc_id: `56017`) (sent_id: `56017`)


Auf dieser Grundlage muss das Fachgericht unter Anwendung und Auslegung des materiellen Unionsrechts ( vgl. BVerfGE 135 , 155 < 233 Rn. 184 > ) die vertretbare Überzeugung bilden , dass die Rechtslage entweder von vornherein eindeutig ( " acte clair " ) oder durch Rechtsprechung in einer Weise geklärt ist , die keinen vernünftigen Zweifel offenlässt ( " acte éclairé " ; vgl. BVerfGE 129 , 78 < 107 > ; 135 , 155 < 233 Rn. 184 > ) .

**False Positives:**

- `acte clair ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BVerfGE 135 , 155 < 233 Rn. 184 >`(RS)
- `BVerfGE 129 , 78 < 107 > ; 135 , 155 < 233 Rn. 184 >`(RS)

**Example 65** (doc_id: `56069`) (sent_id: `56069`)


Aus dem Wortlaut der Vorschrift ergibt sich durch die Verwendung des Merkmals " derselben " die Notwendigkeit eines Vergleichs und als dessen Ergebnis einer Identität zwischen der ursprünglichen Beschäftigung oder selbstständigen Tätigkeit und der aktuellen Beschäftigung oder selbstständigen Tätigkeit .

**False Positives:**

- `derselben ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 66** (doc_id: `56082`) (sent_id: `56082`)


An welchen gewöhnlichen Aufenthalt bei der Beurteilung der Zuständigkeit der Leistungserbringung ab 15. 2. 2005 anzuknüpfen ist , hängt davon ab , ob K. während ihrer Teilnahme am Modellprojekt " Enthospitalisierung " in einer stationären Einrichtung gelebt hat oder eine Form des ambulant-betreuten Wohnens vorlag ( dazu gleich ) .

**False Positives:**

- `Enthospitalisierung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `K.`(PER)

**Example 67** (doc_id: `56121`) (sent_id: `56121`)


Der Gesetzentwurf für die Bemessungsgrundlage der Grundsteuer sah eine Abkehr vom bisherigen Bewertungsziel " gemeiner Wert " hin zum so genannten Kostenwert vor , der typisiert den Investitionsaufwand für die Immobilie abbilden sollte ( BRDrucks 515/16 , S. 36 ) .

**False Positives:**

- `gemeiner Wert ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BRDrucks 515/16 , S. 36`(LIT)

**Example 68** (doc_id: `56149`) (sent_id: `56149`)


Das Rechtsmittelgericht darf ein von der jeweiligen Prozessordnung eröffnetes Rechtsmittel daher nicht ineffektiv machen und für den Beschwerdeführer " leerlaufen " lassen ( vgl. BVerfGE 78 , 88 < 98 f. > ; 96 , 27 < 39 > ; 104 , 220 < 232 > ; BVerfGK 6 , 303 < 308 > ) .

**False Positives:**

- `leerlaufen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BVerfGE 78 , 88 < 98 f. > ; 96 , 27 < 39 > ; 104 , 220 < 232 >`(RS)
- `BVerfGK 6 , 303 < 308 >`(RS)

**Example 69** (doc_id: `56158`) (sent_id: `56158`)


Unter " Bestimmen " ist die Einflussnahme auf den Willen eines anderen zu verstehen , die diesen zu dem im Gesetz beschriebenen Verhalten bringt ; dies setzt einen kommunikativen Akt voraus ( vgl. BGH , Beschluss vom 5. August 2008 - 3 StR 224/08 , NStZ 2009 , 393 f. ) .

**False Positives:**

- `Bestimmen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BGH , Beschluss vom 5. August 2008 - 3 StR 224/08 , NStZ 2009 , 393 f.`(RS)

**Example 70** (doc_id: `56200`) (sent_id: `56200`)


Mangels gesetzlicher Beschränkungen reicht für deren steuerliche Anerkennung jedes " Stehenlassen " der handelsrechtlichen Gewinne als Eigenkapital aus , sofern anhand objektiver Umstände nachvollzogen und überprüft werden kann , dass dem Regiebetrieb die entsprechenden Mittel weiterhin als Eigenkapital zur Verfügung stehen sollen .

**False Positives:**

- `Stehenlassen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 71** (doc_id: `56221`) (sent_id: `56221`)


Das Bestehen eines Mietvertrages zwischen den Parteien ist sehr wohl streitig , weil die Parteien darüber streiten , ob es sich um einen fingierten beziehungsweise erst nach der Beschlagnahme geschlossenen und damit den Klägern gegenüber nicht wirksamen " Vertrag " handelt .

**False Positives:**

- `Vertrag ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 72** (doc_id: `56306`) (sent_id: `56306`)


Das Urteil des Niedersächsischen FG vom 16. November 2011 9 K 316/15 ( Entscheidungen der Finanzgerichte 2017 , 482 ) betrifft eine " ehemalige nichteheliche Lebensgemeinschaft " sowie eine erhebliche betriebliche Nutzung des überlassenen PKW .

**False Positives:**

- `ehemalige nichteheliche Lebensgemeinschaft ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Urteil des Niedersächsischen FG vom 16. November 2011 9 K 316/15 ( Entscheidungen der Finanzgerichte 2017 , 482 )`(RS)

**Example 73** (doc_id: `56323`) (sent_id: `56323`)


aa ) Mit Urteil vom 24. 3. 2016 ( B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 ) hat der Senat bereits entschieden , dass es für den Beginn der aufgeschobenen Versicherungspflicht nach § 7a Abs 6 S 1 SGB IV - mit Wirkung für alle Zweige der Sozialversicherung - auf die Bekanntgabe einer ( ersten ) Entscheidung der Deutschen Rentenversicherung Bund über das Bestehen von " Beschäftigung " ankommt und nicht auf eine ( spätere ) - diese unzulässige Elementenfeststellung korrigierende - Entscheidung über " Versicherungspflicht wegen Beschäftigung " .

**False Positives:**

- `Beschäftigung ` — no gold match — likely missing annotation
- `Versicherungspflicht wegen Beschäftigung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `Urteil vom 24. 3. 2016 ( B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 )`(RS)
- `§ 7a Abs 6 S 1 SGB IV`(NRM)
- `Deutschen Rentenversicherung Bund`(ORG)

**Example 74** (doc_id: `56342`) (sent_id: `56342`)


Denn gemäß § 2 Abs. 4 DGL-VO SH kann sich das für die Erlangung einer Genehmigung erforderliche ( Ersatz- ) Dauergrünland auch auf den Flächen anderer Personen ( im Streitfall der Klägerin ) als der des Umbruchwilligen ( im Streitfall " Käufer " ) befinden .

**False Positives:**

- `Käufer ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 2 Abs. 4 DGL-VO SH`(NRM)

**Example 75** (doc_id: `56346`) (sent_id: `56346`)


Es ist jedoch nicht festzustellen , dass es sich hierbei um eine " gefestigte " Rechtsprechung im Sinne der vorstehenden Maßstäbe handelt .

**False Positives:**

- `gefestigte ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 76** (doc_id: `56424`) (sent_id: `56424`)


In seinem Gutachten führte der Sachverständige ( S ) aus , die vom Kläger verwendete Kassensoftware basiere auf dem relationalen Datenbanksystem " Microsoft Access " .

**False Positives:**

- `Microsoft Access ` — similar text (different position): `S`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S`(PER)

**Example 77** (doc_id: `56444`) (sent_id: `56444`)


So wird ausdrücklich darauf verwiesen , dass auf der Anlage K 1. Ausdruck einer Mail von D. an D. B. vom 30. Januar 2011 ) handschriftlich " Rechnungsanschrift Help Food z. o. o. D. B. ( es folgt die postalische Anschrift der Help Food ) " vermerkt ist , auf Seite 2 der Anlage K 3 ( Präsentationsunterlage mit dem Copyright von D. und W. ) unter " Unsere Kontraktbedingungen " ein " Exklusiver Kontrakt für 2 Jahre mit Help Food " und eine " Haushaltsverfügung durch Help Food ... bis zum Ende 2011 Startphase " erwähnt werden , auf Seite 2 der Anlage K 5 ( mit dem Logo der Klägerin versehenes Protokoll eines Treffens der Beteiligten am 26. August 2011 ) von einem " Vorschlag zum Vertrag zwischen Help Food , M. D. und P. W. " die Rede ist , die Anlage K 9 ( von S. unterzeichnetes Schreiben vom 29. Dezember 2011 ) als Absender die Help Food ausweist und die Anlage K 50 ( Ausdruck einer Mail der Zeugin F. an D. und W. vom 14. September 2011 ) die Absenderadresse " m. @helpfood . eu " trägt .

**False Positives:**

- `Unsere Kontraktbedingungen ` — no gold match — likely missing annotation
- `und eine ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `D.`(PER)
- `D. B.`(PER)
- `Help Food`(ORG)
- `Help Food`(ORG)
- `D.`(PER)
- `W.`(PER)
- `Help Food`(ORG)
- `Help Food`(ORG)
- `Help Food`(ORG)
- `M. D.`(PER)
- `P. W.`(PER)
- `S.`(PER)
- `Help Food`(ORG)
- `F.`(PER)
- `D.`(PER)
- `W.`(PER)

**Example 78** (doc_id: `56459`) (sent_id: `56459`)


Die Marktmissbrauchsverordnung spricht ihrerseits in ihrem Art. 30 Abs. 1 UAbs. 2 von " Verstößen " gegen einzelne Artikel der Marktmissbrauchsverordnung vor dem 3. Juli 2016 .

**False Positives:**

- `Verstößen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Marktmissbrauchsverordnung`(NRM)
- `Marktmissbrauchsverordnung`(NRM)

**Example 79** (doc_id: `56477`) (sent_id: `56477`)


Fünftens sei das Streikverbot als hergebrachter Grundsatz des Berufsbeamtentums " gesetzlich vorgesehen " im Sinne von Art. 11 Abs. 2 Satz 1 EMRK .

**False Positives:**

- `gesetzlich vorgesehen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Art. 11 Abs. 2 Satz 1 EMRK`(NRM)

**Example 80** (doc_id: `56487`) (sent_id: `56487`)


Die Vergleichszeichen seien in einer Art und Weise grafisch ausgestaltet , dass die jeweiligen Bestandteile " KEA " und " GEA " jedenfalls nicht die allein prägenden Elemente seien .

**False Positives:**

- `KEA ` — no gold match — likely missing annotation
- `GEA ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 81** (doc_id: `56501`) (sent_id: `56501`)


General " A " habe mit dem Außerkraftsetzen der Soldatenarbeitszeitverordnung seine Kompetenzen überschritten .

**False Positives:**

- `A ` — partial — pred is substring of gold: `" A "`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `" A "`(PER)
- `Soldatenarbeitszeitverordnung`(NRM)

**Example 82** (doc_id: `56552`) (sent_id: `56552`)


Eine unechte Rückwirkung liegt vor , wenn eine Norm auf gegenwärtige , noch nicht abgeschlossene Sachverhalte und Rechtsbeziehungen für die Zukunft einwirkt und damit zugleich die betroffene Rechtsposition entwertet ( vgl. BVerfGE 101 , 239 < 263 > ; 123 , 186 < 257 > ) , etwa wenn belastende Rechtsfolgen einer Norm erst nach ihrer Verkündung eintreten , tatbestandlich aber von einem bereits ins Werk gesetzten Sachverhalt ausgelöst werden ( vgl. BVerfGE 132 , 302 < 318 Rn. 43 > m. w. N. ; ferner BVerfGE 127 , 1 < 17 > m. w. N. ; " tatbestandliche Rückanknüpfung " ) .

**False Positives:**

- `tatbestandliche Rückanknüpfung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BVerfGE 101 , 239 < 263 > ; 123 , 186 < 257 >`(RS)
- `BVerfGE 132 , 302 < 318 Rn. 43 >`(RS)
- `BVerfGE 127 , 1 < 17 >`(RS)

**Example 83** (doc_id: `56566`) (sent_id: `56566`)


Der Vorinstanz wäre vielmehr darin beizupflichten , dass die " Schlusszahlung " insofern lediglich eine Rechengröße darstellt , die die Obergrenze der Erlösbeteiligung kennzeichnet .

**False Positives:**

- `Schlusszahlung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 84** (doc_id: `56605`) (sent_id: `56605`)


Sie üben ihr Amt auf Zeit aus ; die gesetzlichen Regelungen sehen eine Amtsdauer von zumindest vier Jahren vor ( vgl. Art. 139 Abs. 2 Satz 2 der Landesverfassung der Freien Hansestadt Bremen - Bindung an die Wahlperiode der Bürgerschaft ; vgl. BVerfGE 40 , 356 < 362 ff. > : zweijährige Amtszeit als das " untere Ende der denkbaren Möglichkeiten " ) , überwiegend auch die Möglichkeit einer Wiederwahl .

**False Positives:**

- `untere Ende der denkbaren Möglichkeiten ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Art. 139 Abs. 2 Satz 2 der Landesverfassung der Freien Hansestadt Bremen`(NRM)
- `BVerfGE 40 , 356 < 362 ff. >`(RS)

**Example 85** (doc_id: `56614`) (sent_id: `56614`)


Die Dienstleistung betreffe auch keine Mittel , die ein landwirtschaftlicher Erzeuger " normalerweise " oder " gewöhnlich " zum Betrieb seiner eigenen Landwirtschaft verwende .

**False Positives:**

- `normalerweise ` — no gold match — likely missing annotation
- `gewöhnlich ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 86** (doc_id: `56677`) (sent_id: `56677`)


Der Gesetzgeber hat insoweit auch für das gerichtliche Asylverfahren an den allgemeinen Grundsätzen des Revisionsrechts festgehalten und für das Bundesverwaltungsgericht keine Befugnis eröffnet , Tatsachen ( würdigungs ) fragen grundsätzlicher Bedeutung in " Länderleitentscheidungen " , wie sie etwa das britische Prozessrecht kennt , zu treffen .

**False Positives:**

- `Länderleitentscheidungen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Bundesverwaltungsgericht`(ORG)

**Example 87** (doc_id: `56705`) (sent_id: `56705`)


Es kann dahinstehen , ob sich diese Beurteilung bereits aus dem Wortsinn ergibt ( vgl aber www.duden.de , wonach " Stiefvater " als " Mann , der mit der leiblichen Mutter eines Kindes verheiratet ist und die Stelle des Vaters einnimmt " , bezeichnet wird ) , der ihr jedenfalls nicht entgegensteht .

**False Positives:**

- `Stiefvater ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 88** (doc_id: `56706`) (sent_id: `56706`)


Nichtzulassungsbeschwerde , Divergenz , Verkauf von Speisen an einer " Heißen Theke "

**False Positives:**

- `Heißen Theke ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 89** (doc_id: `56710`) (sent_id: `56710`)


Der Beschwerdeführer werde eindringlich darauf hingewiesen , dass für " die Festsetzung der Mindestverbüßungsdauer auch das vollzugliche Fortkommen bezüglich festgelegter Behandlungsmaßnahmen eine große Rolle spielen " werde und es daher " dringend erforderlich " sei , dass er seine Verlegung nach Offenburg anstrebe .

**False Positives:**

- `die Festsetzung der Mindestverbüßungsdauer auch das vollzugliche Fortkommen bezüglich festgelegter Behandlungsmaßnahmen eine große Rolle spielen ` — no gold match — likely missing annotation
- `dringend erforderlich ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `Offenburg`(LOC)

**Example 90** (doc_id: `56718`) (sent_id: `56718`)


Nach § 3 Abs. 1 Nr. 5 SchwbAWV wird durch die Eintragung des Merkzeichens " RF " nachgewiesen , dass der schwerbehinderte Mensch die landesrechtlich festgelegten Voraussetzungen für die Befreiung von der Rundfunkgebührenpflicht erfüllt .

**False Positives:**

- `RF ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 3 Abs. 1 Nr. 5 SchwbAWV`(NRM)

**Example 91** (doc_id: `56728`) (sent_id: `56728`)


Auf der Zustellungsurkunde , die unter dem 5. Mai 2017 ( unleserlich ) gezeichnet ist , ist der Vermerk angekreuzt " Adressat unter der angegebenen Anschrift nicht zu ermitteln " .

**False Positives:**

- `Adressat unter der angegebenen Anschrift nicht zu ermitteln ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 92** (doc_id: `56817`) (sent_id: `56817`)


Die Klägerin könne gegenüber einer etablierten Praxis schon deshalb nicht benachteiligt worden sein , weil sie keine " Wachstumspraxis " sei .

**False Positives:**

- `Wachstumspraxis ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 93** (doc_id: `56848`) (sent_id: `56848`)


Zwischen den sich gegenüberstehenden Zeichen sei jedenfalls eine hohe klangliche Ähnlichkeit gegeben , denn nach den vom Bundesgerichtshof entwickelten Grundsätzen zur Prägung von Wort- / Bildzeichen seien vorliegend in klanglicher Hinsicht jedenfalls die prägenden Zeichenbestandteile " GEA " und " KEA " zu vergleichen .

**False Positives:**

- `GEA ` — no gold match — likely missing annotation
- `KEA ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `Bundesgerichtshof`(ORG)

**Example 94** (doc_id: `56857`) (sent_id: `56857`)


Soweit das Berufungsgericht zur Begründung seiner Zulassungsentscheidung ausgeführt hat , die Revision sei für die Klägerin " hinsichtlich des Hauptantrags " zuzulassen , hat es damit ersichtlich nur das auf Zulassung als Vertragswerkstatt gerichtete Klagebegehren von den weiteren Klageanträgen abgrenzen wollen , mit denen die Klägerin die Unwirksamkeit der Vertragskündigungen vom 23. Mai 2011 geltend gemacht hat .

**False Positives:**

- `hinsichtlich des Hauptantrags ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 95** (doc_id: `56858`) (sent_id: `56858`)


Nach dem ursprünglichen Gesetzentwurf sollte künftig der Gewinn aus der Veräußerung oder Aufgabe unter anderem ( Nr. 2 ) des Anteils eines Gesellschafters , der als Unternehmer ( Mitunternehmer ) des Betriebs einer Mitunternehmerschaft anzusehen ist , der Gewerbesteuer unterfallen , " soweit er nicht auf eine natürliche Person als Mitunternehmer entfällt " ( Halbsatz 2 ) .

**False Positives:**

- `soweit er nicht auf eine natürliche Person als Mitunternehmer entfällt ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 96** (doc_id: `56891`) (sent_id: `56891`)


Nach der speziellen Zuständigkeitsvorschrift in § 3 Satz 1 SAZV ist für " Maßnahmen nach der Soldatenarbeitszeitverordnung " das Bundesministerium der Verteidigung zuständig , soweit nichts Abweichendes bestimmt ist .

**False Positives:**

- `Maßnahmen nach der Soldatenarbeitszeitverordnung ` — partial — gold is substring of pred: `Soldatenarbeitszeitverordnung`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 3 Satz 1 SAZV`(NRM)
- `Soldatenarbeitszeitverordnung`(NRM)
- `Bundesministerium der Verteidigung`(ORG)

**Example 97** (doc_id: `56909`) (sent_id: `56909`)


Bei den Vorschriften über die Besorgnis der Befangenheit geht es auch darum , bereits den " bösen Schein " einer möglicherweise fehlenden Unvoreingenommenheit zu vermeiden ( vgl. BVerfGE 108 , 122 < 129 > ) .

**False Positives:**

- `bösen Schein ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BVerfGE 108 , 122 < 129 >`(RS)

**Example 98** (doc_id: `56918`) (sent_id: `56918`)


Nachdem er am Morgen des Tattages Heroin , Kokain und Alkohol " in den für ihn üblichen Bereichen " konsumiert hatte , transportierte er am Nachmittag - unter dem Einfluss der Rauschmittel , indes ohne erhebliche Beeinträchtigung seiner Einsichts- und Steuerungsfähigkeit - im Kofferraum seines PKW 1.703,5 g einer MDMA-Zubereitung mit einem Wirkstoffanteil von 369 g MDMA-Base und 1.924,75 g Haschisch mit einem Wirkstoffanteil von ca. 273 g THC aus den Niederlanden über die Grenze in das Bundesgebiet , wo die Drogen gewinnbringend verkauft werden sollten .

**False Positives:**

- `in den für ihn üblichen Bereichen ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Niederlanden`(LOC)

**Example 99** (doc_id: `57025`) (sent_id: `57025`)


Im Hinblick auf die Prüfungskriterien des Art. 3 ( a ) AMVO hat der Gerichtshof in seinen Entscheidungen " Actavis / Sanofi " und " Actavis / Boehringer " trotz ausdrücklicher Fragen der vorlegenden Gerichte keinen über " Medeva " und " Eli Lilly " hinausgehenden Auslegungsbedarf gesehen und die Beantwortung dieser Fragen deshalb dahingestellt sein lassen ( EuGH , GRUR Int. 2014 , 153 , Rnd. 25 , 44 – Actavis / Sanofi ; EuGH , GRUR Int. 2015 , 446 , Rnd. 24 , 41 – Actavis / Boehringer ) .

**False Positives:**

- `und ` — no gold match — likely missing annotation
- `trotz ausdrücklicher Fragen der vorlegenden Gerichte keinen über ` — no gold match — likely missing annotation
- `und ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 3

**Gold Entities:**

- `Art. 3 ( a ) AMVO`(NRM)
- `Actavis`(ORG)
- `Sanofi`(ORG)
- `Actavis`(ORG)
- `Boehringer`(ORG)
- `" Medeva "`(ORG)
- `" Eli Lilly "`(ORG)
- `EuGH , GRUR Int. 2014 , 153 , Rnd. 25 , 44 – Actavis / Sanofi`(RS)
- `EuGH , GRUR Int. 2015 , 446 , Rnd. 24 , 41 – Actavis / Boehringer`(RS)

</details>

---

</details>

---


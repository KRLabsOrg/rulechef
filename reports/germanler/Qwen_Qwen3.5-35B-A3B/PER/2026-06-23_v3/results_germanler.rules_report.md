# Rule Evaluation Report — Qwen/Qwen3.5-35B-A3B

Generated on: 2026-06-23T08:03:59.200777

---

<details>
<summary>Configuration</summary>

Results can be reproduced by running this command: 
```
 python benchmark.py --config reports/germanler/Qwen_Qwen3.5-35B-A3B/PER/2026-06-23_v3/config.yaml 
```
| Parameter | Value |
|---|---|
| Pool size | None |
| Train ratio | 0.80 |
| Validation ratio | 0.20 |
| Shots per class | None |
| Training documents | 2298 |
| Validation documents | 6666 |
| Test documents | 6673 |
| Train sentences | 2298 |
| Validation sentences | 2849 |
| Test sentences | 6673 |
| Model | Qwen/Qwen3.5-35B-A3B |
| Max rules | 30 |
| Max samples in prompt | 150 |
| Refinement iterations | 6 |
| Seed | 42 |
| Agentic | False |
| Enable Critic | False |
| Enable Prune | False |
| Critic Interval | 10 |
| Audit Interval | 0 |
| Use GREX | True |
| Format | regex |
| Synthesis strategy | bulk |
| Sampling strategy | balanced |
| Batch size | 100 |
| Refine per batch | 1 |
| Manually annotated examples | 0 |
| First batch with manual data | None |

</details>

---

**Transfer Learning**

| Property | Value |
|---|---|
| Best Batch Idx | 21 |
| Best Batch F1 | 0.37128712871287134 |
| Best Rules Serialized | [{'id': 'a3f99de3', 'name': 'Initials with dots and spaces (e.g., T. D.)', 'description': 'Captures sequences of initial-dot-space-initial-dot patterns.', 'format': 'regex', 'content': '\\b([A-Z]\\.[ ]+[A-Z]\\.)\\b', 'priority': 8, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:34:24.899224', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '1e1f1dc2', 'name': "Names after 'Rechtsanwalt' or 'Rechtsanwältin'", 'description': 'Captures names following legal profession titles, ensuring no trailing space and handling initials with dots.', 'format': 'regex', 'content': '(?:Rechtsanwalt|Rechtsanwältin)\\s+([A-Z][a-zäöüß]+(?:\\s+[A-Z][a-zäöüß]+)*|\\b[A-Z]\\s*\\.\\b|\\b[A-Z]\\b)(?=\\s*(?:,|\\.|\\)|\\]|\\s|$))', 'priority': 8, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:45:05.458674', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'e697a848', 'name': 'Full names with titles (corrected)', 'description': 'Captures full names preceded by titles like Dr., Prof., ensuring the entire name is captured including middle initials, excluding the title itself.', 'format': 'regex', 'content': '(?:Dr\\.?\\s+|Prof\\.?\\s+|Dipl\\.-Ing\\.\\s+|Dipl\\.-Psych\\.\\s+|Dipl\\.-Ing\\.\\s+Univ\\.\\s+)([A-Z][a-z\\u00e4\\u00f6\\u00fc\\u00df]+(?:\\s+[A-Z]\\s*\\.)?(?:\\s+[A-Z][a-z\\u00e4\\u00f6\\u00fc\\u00df]+)*)', 'priority': 10, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:53:22.795030', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '1b824d7a', 'name': 'Anonymized names with ellipses (space)', 'description': "Captures anonymized names with a letter and ellipsis (e.g., 'K …', 'T …', 'A …', 'R …').", 'format': 'regex', 'content': '(?<![A-Za-zäöüß])([A-Z])\\s+…', 'priority': 8, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:51:52.134554', 'output_template': {'text': '$1 …', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'ae810878', 'name': "Names after 'Herr' or 'Herrn' (corrected)", 'description': "Captures names following 'Herr' or 'Herrn', handling full names and initials.", 'format': 'regex', 'content': '(?:Herr|Herrn)\\s+([A-Z][a-zäöüß]+(?:\\s+[A-Z]\\s*\\.)?(?:\\s+[A-Z][a-zäöüß]+)*)', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:51:52.134733', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'd2bd3fae', 'name': "Names after 'Richter' or 'Richterin' (corrected)", 'description': "Captures names following 'Richter', 'Richterin', 'Vorsitzender Richter', etc., ensuring no trailing space.", 'format': 'regex', 'content': '(?:Richter|Richterin|Vorsitzender\\s+Richter|Vorsitzende\\s+Richterin)\\s+([A-Z][a-z\\u00e4\\u00f6\\u00fc\\u00df]*\\.?\\s*(?:[A-Z][a-z\\u00e4\\u00f6\\u00fc\\u00df]*\\.?\\s*)*|\\b[A-Z]\\s*\\.\\b|\\b[A-Z]\\b)(?=\\s*(?:,|\\.|\\)|\\]|\\s|$))', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:53:22.795039', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '5e2b25bc', 'name': "Names after 'Angeklagte' or 'Angeklagten' (corrected)", 'description': "Captures names following 'Angeklagte' or 'Angeklagten', ensuring no trailing space and handling initials with dots.", 'format': 'regex', 'content': '(?:Angeklagte|Angeklagten)\\s+([A-Z][a-zäöüß]*\\.?\\s*(?:[A-Z][a-zäöüß]*\\.?\\s*)*|\\b[A-Z]\\s*\\.\\b|\\b[A-Z]\\b)(?=\\s*(?:,|\\.|\\)|\\]|\\s|$))', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:51:52.134420', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'd3f9d622', 'name': 'Anonymized initials with dots and spaces (e.g., T. D.)', 'description': 'Captures sequences of initial-dot-space-initial-dot patterns.', 'format': 'regex', 'content': '\\b([A-Z]\\.)\\s+([A-Z]\\.)\\b', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:45:20.828080', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'b7b02b0b', 'name': 'Anonymized names with dots and ellipses or spaces', 'description': "Captures anonymized names with dots and ellipses or spaces (e.g., 'K …', 'H …'), excluding company names.", 'format': 'regex', 'content': '\\b([A-Z]\\d?\\.?)\\s+…\\b', 'priority': 8, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:44:38.594286', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'a6567e07', 'name': 'Initials after legal roles', 'description': "Captures single letter initials (with or without dots) following legal role indicators like 'Angeklagten', 'Kläger', 'Zeuge', 'Zeugin'.", 'format': 'regex', 'content': '(?:Angeklagten|Kläger|Zeuge|Zeugin|Vertrauensmann)\\s+([A-Z](?:\\.)?)(?=\\s*(?:,|\\.|\\)|\\]|\\s|$))', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:51:52.134718', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '7124ef3f', 'name': 'Anonymized initials with dots (e.g., R., A.)', 'description': "Captures anonymized initials with a trailing dot (e.g., 'R.', 'A.', 'B.') to ensure the dot is included in the entity, excluding common non-name words.", 'format': 'regex', 'content': '(?<![A-Za-zäöüß\\.\\s])([A-Z]\\.)(?![A-Za-zäöüß\\.\\s])', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:45:05.458595', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '243eaaf1', 'name': 'Compound surnames', 'description': 'Captures specific known German compound surnames (e.g., Lachenmayr-Nikolaou, Sost-Scheible) while excluding geographical or technical terms.', 'format': 'regex', 'content': '\\b(?:Lachenmayr-Nikolaou|Sost-Scheible|Mittenberger-Huber|Meier-Beck|Fuchs-Wissemann|Kopp-Schenke|Dreier-Gro\\u00df|Sch\\u00e4fer-Weber|Koch-Schmidt|Bender-Brune|Harsdorf-Gebhardt)\\b', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:53:22.795242', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '3b9a194a', 'name': 'Initials with surname (e.g., K. Schmidt)', 'description': "Captures an initial followed by a capitalized surname, ensuring it's a name and not a sentence start or common verb.", 'format': 'regex', 'content': '(?<!^)(?<!\\w)([A-Z]\\s*\\.)\\s+([A-Z][a-zäöüß]+(?:-[A-Z][a-zäöüß]+)?)', 'priority': 8, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:46:14.226670', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '5434427d', 'name': 'Initials with dots (e.g., K., T.)', 'description': 'Captures single letter initials followed by a dot, ensuring the dot is included.', 'format': 'regex', 'content': '\\b([A-Z]\\.)\\b', 'priority': 8, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:47:33.580387', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '1395461f', 'name': 'Initials with surname (corrected)', 'description': "Captures patterns like 'K. Schmidt' or 'M. P. Borom' where an initial (with dot) is followed by a surname.", 'format': 'regex', 'content': '\\b([A-Z]\\.)\\s+([A-Z][a-zäöüß]+(?:-[A-Z][a-zäöüß]+)?)\\b', 'priority': 10, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:51:52.134739', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'e2864f95', 'name': 'Names with specific formatting (BRADLER , Christian)', 'description': "Captures names with specific formatting like 'BRADLER , Christian' or 'Surname , Firstname', excluding non-name patterns like 'BMW, Typ' or 'BGH, Beschluss'.", 'format': 'regex', 'content': '\\b([A-Z][A-Z\\u00e4\\u00f6\\u00fc\\u00df]+\\s*,\\s*[A-Z][a-z\\u00e4\\u00f6\\u00fc\\u00df]+)\\b', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:53:22.795382', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '40a4571f', 'name': 'Initials with surname', 'description': "Captures patterns like 'K. Schmidt' or 'M. P. Borom' where an initial (with dot) is followed by a surname.", 'format': 'regex', 'content': '\\b([A-Z]\\.)\\s+([A-Z][a-z\\u00e4\\u00f6\\u00fc\\u00df]+(?:-[A-Z][a-z\\u00e4\\u00f6\\u00fc\\u00df]+)?)\\b', 'priority': 10, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:53:22.795460', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '64c0e08f', 'name': 'Names after legal roles (Herr, Richter, etc.)', 'description': "Captures names following legal role indicators like 'Herr', 'Herrn', 'Richter', 'Richterin', 'Angeklagte', 'Angeklagten', 'Kl\\u00e4ger', 'Zeuge', ensuring no trailing space and handling titles correctly.", 'format': 'regex', 'content': '(?:Herr\\s+|Herrn\\s+|Richter\\s+|Richterin\\s+|Vorsitzender\\s+Richter\\s+|Angeklagte\\s+|Angeklagten\\s+|Kl\\u00e4ger\\s+|Zeuge\\s+|Zeugin\\s+)([A-Z][a-z\\u00e4\\u00f6\\u00fc\\u00df]+(?:\\s+[A-Z]\\s*\\.)?(?:\\s+[A-Z][a-z\\u00e4\\u00f6\\u00fc\\u00df]+)*)', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:53:22.795543', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'f64bfcbd', 'name': 'Initials with surname (contextual)', 'description': "Captures an initial followed by a capitalized surname (e.g., 'K. Schmidt'), ensuring it's a name and not a sentence start or common verb.", 'format': 'regex', 'content': '(?:^|\\s|[,;])([A-Z]\\.)\\s+([A-Z][a-zäöüß]+)', 'priority': 8, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:51:52.134438', 'output_template': {'text': '$1 $2', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '001ef128', 'name': 'Anonymized names with comma and initial', 'description': "Captures anonymized names in the format 'Surname , Initial.' (e.g., 'Boolell , M.', 'Rosen , R. C.').", 'format': 'regex', 'content': '\\b([A-Z][a-zäöüß]+\\s*,\\s*[A-Z]\\s*\\.)\\b', 'priority': 10, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:52:47.976355', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '23f66d88', 'name': 'Multi-initial names with surname', 'description': "Captures names with multiple initials followed by a surname (e.g., 'P. W. McMillan', 'Tomlinson , J. M.').", 'format': 'regex', 'content': '\\b([A-Z]\\s*\\.\\s*[A-Z]\\s*\\.\\s*[A-Z][a-zäöüß]+|[A-Z][a-zäöüß]+\\s*,\\s*[A-Z]\\s*\\.\\s*[A-Z]\\s*\\.)\\b', 'priority': 10, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:52:47.976501', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'de849019', 'name': 'Known German Surnames', 'description': 'Captures a comprehensive list of common German surnames and other frequently occurring names in legal texts, including compound names.', 'format': 'regex', 'content': '\\b(?:Berger|Sander|Schmaltz|Bender|Fritz|Becker|Feilcke|Kiel|Brunke|Mutzbauer|Zeng|Benrath|Cierniak|Appl|Franke|Brune|Roggenbuck|Niemann|Grube|Volz|Quentin|Spinner|Schlewing|Marx|Fischer|Bredendiek|Stresemann|Kayser|Koch|Volk|Liebert|Limperg|Schlünder|Berg|Matthias|Hohoff|Leitz|Krumbiegel|Paul|Treber|Spaniol|Feddersen|Schultz|Schuh|Lauer|Lipphaus|Gräfl|Schäfer|König|Müller|Dauber|Tiemann|Deichfuß|Ahrendt|Graßnack|Schmidt-Räntschist|Schmidt|Radtke|Pohl|Nielebock|Fischermeier|Bellay|Kirchhof|Busch|Krehl|Hayen|Glock|Redeker|Morawek|Eder|Baumgardt|Hoffmann|Kaya|Seyhan|Çerikci|Hacker|Merzbach|Meiser|Knoll|Kriener|Nielsen|Musiol|Dorn|Albertshofer|Wollny|Bieringer|Hilber|Paetzold|Baumgart|Geier|Höchst|Fritze|Wiegele|Kleinschmidt|Kirschneck|Arnoldi|Haupt|Demir|Baykara|Aranyosi|Căldăraru|Nassauer|van den Berg|Einstein|Shah|Kuemmerle|Tomlinson|Wright|Vogt|Saime|Özcan|Sen|Bar|Refaeli|Josh|Duhamel|Kelvin|Heinkel|Pape|Harsdorf|Gebhardt|Enerji Yapi-Yol Sen|Eschelbach|Bormann|Möhring|Zimmermann|Rose|Kohout|Abdullah Öcalan|Schwabe|Paffrath|Jaehde|Eckstein|Matter|KCK|PKK)\\b', 'priority': 8, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:52:47.976676', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '5ba69691', 'name': 'Anonymized initials with dots (corrected)', 'description': "Captures anonymized initials consisting of a single uppercase letter followed by a dot, ensuring it's not part of a larger multi-initial sequence already captured.", 'format': 'regex', 'content': '(?<![A-Za-zäöüß\\.\\s])([A-Z])\\.(?![A-Za-zäöüß\\.\\s])', 'priority': 7, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:52:47.977840', 'output_template': {'text': '$1.', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'c4f45684', 'name': 'Anonymized initials with dots', 'description': "Captures anonymized initials (e.g., 'J.', 'K.') ONLY when preceded by a role indicator or title, preventing false positives on legal abbreviations like 'S.' or 'V.'.", 'format': 'regex', 'content': '(?:Herr\\s+|Herrn\\s+|Dr\\.\\s+|Prof\\.\\s+|Richter\\s+|Richterin\\s+|Vorsitzender\\s+Richter\\s+|Angeklagte\\s+|Angeklagten\\s+|Kl\\u00e4ger\\s+|Zeuge\\s+|Zeugin\\s+|Gesch\\u00e4ftsf\\u00fchrer\\s+|Rechtsanwalt\\s+|Rechtsanw\\u00e4ltin\\s+|Beteiligte\\s+|Beteiligter\\s+)([A-Z])\\.(?=\\s*(?:,|\\.|\\)|\\]|\\s|$))', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:53:22.794348', 'output_template': {'text': '$1.', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '70843e1c', 'name': 'Anonymized single letters', 'description': "Captures single letter anonymized names (e.g., 'A', 'B') ONLY when preceded by a role indicator or title, preventing false positives on section markers like 'I.', 'II.' or abbreviations.", 'format': 'regex', 'content': '(?:Herr\\s+|Herrn\\s+|Dr\\.\\s+|Prof\\.\\s+|Richter\\s+|Richterin\\s+|Vorsitzender\\s+Richter\\s+|Angeklagte\\s+|Angeklagten\\s+|Kl\\u00e4ger\\s+|Zeuge\\s+|Zeugin\\s+|Gesch\\u00e4ftsf\\u00fchrer\\s+|Rechtsanwalt\\s+|Rechtsanw\\u00e4ltin\\s+|Beteiligte\\s+|Beteiligter\\s+)([A-Z])(?=\\s*(?:,|\\.|\\)|\\]|\\s|$))', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:53:22.794755', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}] |

<details>
<summary>Results</summary>

| Metric | Value |
|---|---|
| Accuracy (exact match) | 93.7% |
| True Positives | 150 |
| False Positives | 334 |
| False Negatives | 174 |
| Total Gold Entities | 324 |
| Micro Precision | 31.0% |
| Micro Recall | 46.3% |
| Micro F1 | 37.1% |
| Macro F1 | 37.1% |

</details>

---

<details>
<summary>📊 Summary</summary>

| Rule | F1 | Precision | Recall | Total Predicted | True Positives | False Positives |
|---|---|---|---|---|---|---|
| `Compound surnames` | 2.4% | 100.0% | 1.2% | 4 | 4 | 0 |
| `Anonymized initials with dots` | 16.9% | 100.0% | 9.3% | 30 | 30 | 0 |
| `Names after 'Richter' or 'Richterin' (corrected)` | 7.7% | 92.9% | 4.0% | 14 | 13 | 1 |
| `Known German Surnames` | 28.9% | 49.6% | 20.4% | 133 | 66 | 67 |
| `Anonymized names with ellipses (space)` | 6.3% | 42.3% | 3.4% | 26 | 11 | 15 |
| `Full names with titles (corrected)` | 4.1% | 38.9% | 2.2% | 18 | 7 | 11 |
| `Anonymized single letters` | 3.0% | 38.5% | 1.5% | 13 | 5 | 8 |
| `Names after 'Angeklagte' or 'Angeklagten' (corrected)` | 1.2% | 33.3% | 0.6% | 6 | 2 | 4 |
| `Initials with surname` | 5.4% | 9.8% | 3.7% | 122 | 12 | 110 |
| `Initials with dots and spaces (e.g., T. D.)` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Names after 'Rechtsanwalt' or 'Rechtsanwältin'` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Names after 'Herr' or 'Herrn' (corrected)` | 0.0% | 0.0% | 0.0% | 2 | 0 | 2 |
| `Anonymized initials with dots and spaces (e.g., T. D.)` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Anonymized names with dots and ellipses or spaces` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Initials after legal roles` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Anonymized initials with dots (e.g., R., A.)` | 0.0% | 0.0% | 0.0% | 1 | 0 | 1 |
| `Initials with surname (e.g., K. Schmidt)` | 0.0% | 0.0% | 0.0% | 6 | 0 | 6 |
| `Initials with dots (e.g., K., T.)` | 0.0% | 0.0% | 0.0% | 5 | 0 | 5 |
| `Initials with surname (corrected)` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Names with specific formatting (BRADLER , Christian)` | 0.0% | 0.0% | 0.0% | 93 | 0 | 93 |
| `Names after legal roles (Herr, Richter, etc.)` | 0.0% | 0.0% | 0.0% | 7 | 0 | 7 |
| `Initials with surname (contextual)` | 0.0% | 0.0% | 0.0% | 4 | 0 | 4 |
| `Anonymized names with comma and initial` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Multi-initial names with surname` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Anonymized initials with dots (corrected)` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |

</details>

---

<details>
<summary>🏆 Most Precise Rules</summary>

## `Anonymized initials with dots`

**F1:** 0.169 | **Precision:** 1.000 | **Recall:** 0.093  

**Format:** `regex`  
**Rule ID:** `c4f45684`  
**Description:**
Captures anonymized initials (e.g., 'J.', 'K.') ONLY when preceded by a role indicator or title, preventing false positives on legal abbreviations like 'S.' or 'V.'.

**Content:**
```
(?:Herr\s+|Herrn\s+|Dr\.\s+|Prof\.\s+|Richter\s+|Richterin\s+|Vorsitzender\s+Richter\s+|Angeklagte\s+|Angeklagten\s+|Kl\u00e4ger\s+|Zeuge\s+|Zeugin\s+|Gesch\u00e4ftsf\u00fchrer\s+|Rechtsanwalt\s+|Rechtsanw\u00e4ltin\s+|Beteiligte\s+|Beteiligter\s+)([A-Z])\.(?=\s*(?:,|\.|\)|\]|\s|$))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.093 | 0.169 | 30 | 30 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 30 | 0 | 291 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60161`) (sent_id: `60161`)


Allenfalls käme ein solches Vorgehen in Betracht , wenn Dr. T. im maßgeblichen Vorquartal noch nicht im MVZ tätig gewesen wäre ( vgl BSG SozR 4 - 2500 § 87b Nr 2 RdNr 30 : " Hinzurechnung der vom Eintretenden zuvor erbrachten Fallzahlen " ) .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |

**Missed by this rule (FN):**

- `BSG SozR 4 - 2500 § 87b Nr 2 RdNr 30` (RS)

**Example 1** (doc_id: `60445`) (sent_id: `60445`)


Der Facharzt für Kinder- und Jugendpsychiatrie und -psychotherapie Dr. K. führte in seinem Gutachten vom 16. Februar 2017 u. a. aus : Der Kläger habe noch zum Aufnahmezeitpunkt im Klinikum konkrete Suizidgedanken benannt , die er eigenen Angaben zufolge bereits längere Zeit und wiederholt gehabt habe ; von Anschlagsgedanken zumindest auf nicht-zivile Ziele habe er sich nicht ausreichend distanzieren können .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Example 2** (doc_id: `60888`) (sent_id: `60888`)


Die sich aus der Aktenlage und dem Gutachten des Sachverständigen Dr. K. ergebende Persönlichkeitsbewertung deutet nicht auf eine Bereitschaft oder Neigung des Klägers , seinem Leben unabhängig von einem Terroranschlag ein Ende zu setzen .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Example 3** (doc_id: `61021`) (sent_id: `61021`)


Dass der Kläger nach dem Gutachten des Dr. K. noch nicht wie ein Erwachsener wirkt und ihm nach Beobachtungen von Pflegern in der B. er Klinik " jegliche Alltagspraxis " fehle , rechtfertigte bei seiner Abschiebung keine andere Prognose .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `B. er Klinik` (ORG)

**Example 4** (doc_id: `61069`) (sent_id: `61069`)


Nach Zurückverweisung hat das LSG Dr. K. , Institut für neurologisch psychiatrische Begutachtung in B. , mit der Erstellung eines neurologisch-psychiatrischen Gutachtens nach ambulanter Untersuchung des Klägers beauftragt .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `Institut für neurologisch psychiatrische Begutachtung in B.` (ORG)

**Example 5** (doc_id: `61306`) (sent_id: `61306`)


Das Landgericht hat den Angeklagten A. wegen Diebstahls mit Waffen in Tateinheit mit „ fahrlässiger “ Gefährdung des Straßenverkehrs , vorsätzlichem Fahren ohne Fahrerlaubnis und fahrlässiger Körperverletzung sowie wegen unerlaubten Entfernens vom Unfallort und vorsätzlicher Körperverletzung zu der Gesamtfreiheitsstrafe von zwei Jahren und vier Monaten verurteilt ; ferner hat es die Verwaltungsbehörde angewiesen , dem Angeklagten vor Ablauf einer Frist von drei Jahren keine Fahrerlaubnis zu erteilen .

| Predicted | Gold |
|---|---|
| `A.` | `A.` |

**Example 6** (doc_id: `61486`) (sent_id: `61486`)


Das Gericht wies am dritten Hauptverhandlungstag im Zusammenhang mit einem Antrag von Rechtsanwalt P. , den dieser unter Bezugnahme auf das zuvor genannte Schreiben begründet hatte , unter anderem darauf hin , dass sich in der Akte ein „ Terminverlegungsantrag vom 12. April 2016 “ befinde .

| Predicted | Gold |
|---|---|
| `P.` | `P.` |

**Example 7** (doc_id: `61586`) (sent_id: `61586`)


Die Klägerin hatte ihren Sitz zum Zeitpunkt des Eintritts der Job-Sharing-Partnerin Dr. E. im Bezirk der KÄV Nordbaden .

| Predicted | Gold |
|---|---|
| `E.` | `E.` |

**Missed by this rule (FN):**

- `KÄV Nordbaden` (ORG)

**Example 8** (doc_id: `61864`) (sent_id: `61864`)


Zu dem Sachverständigengutachten des Prof. Dr. T. sowie dessen ergänzender Stellungnahme hat die Beklagte mitgeteilt , dass sie diesem nach prüfärztlicher Stellungnahme nicht folgen könne ; der Einschätzung des Dr. S. hat sie sich hingegen angeschlossen .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |
| `S.` | `S.` |

**Example 9** (doc_id: `61871`) (sent_id: `61871`)


Als die Geschädigte während dieses Geschehens von der Zeugin K. angerufen wurde , riss M. der Geschädigten das Mobiltelefon aus der Hand und nahm es im Einverständnis mit dem Angeklagten R. an sich , um zu verhindern , dass die Geschädigte um Hilfe rief .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |
| `R.` | `R.` |

**Missed by this rule (FN):**

- `M.` (PER)

**Example 10** (doc_id: `62635`) (sent_id: `62635`)


Schließlich wird das Stellen eines ordnungsgemäßen Beweisantrags mit der Beschwerdebegründung nicht dargelegt , soweit die Klägerin die Sachaufklärungspflicht des LSG dadurch verletzt sieht , dass dieses keine ergänzende gutachterliche Äußerung Dr. R. eingeholt hat .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |

**Example 11** (doc_id: `62752`) (sent_id: `62752`)


Die Implantation der Coils als alleiniger Grund für die stationäre Behandlung der Versicherten sei nach dem überzeugenden MDK-Gutachten ( Dr. S. ) nicht erforderlich gewesen .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Example 12** (doc_id: `63218`) (sent_id: `63218`)


Zur Revision des Angeklagten R. führte er aus , dass eine Ahndungslücke nicht bestanden habe .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |

**Example 13** (doc_id: `63754`) (sent_id: `63754`)


Soweit das LSG Ausführungen von Prof. Dr. S. referiert , dass sich im Zusammenhang mit den Pneumonien auch immer wieder Septitiden entwickelt hätten , macht das LSG nicht deutlich , dass es diese als festgestellt ansieht , und erst recht nicht , von welchem Begriff oder Schweregrad der Sepsis es im Anschluss an Prof. Dr. S. aufgrund welcher Befunde dabei ausgeht ( ältere Klassifizierungen : Systemisches inflammatorisches Response-Syndrom < SIRS > , Sepsis , schwere Sepsis und septischer Schock ; zur darauf aufbauenden DRG-Kodierung vgl Hanser in Zaiß , DRG : Verschlüsseln leicht gemacht , 14. Aufl 2016 , S 101 ff ; neuere Klassifizierungen : Sepsis-related organ failure assessment score < SOFA > , insbesondere für Intensivstationen , und quickSOFA außerhalb von Intensivstationen ) .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |
| `S.` | `S.` |

**Example 14** (doc_id: `63885`) (sent_id: `63885`)


Nach den Ausführungen des im Verfahren von Amts wegen gehörten Sachverständigen Prof. Dr. T. hätten die vom Kläger vorgetragenen Gewalterfahrungen während seiner Heimaufenthalte ab April 1959 nicht die entscheidende Traumatisierung dargestellt .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |

**Example 15** (doc_id: `63999`) (sent_id: `63999`)


Der weitere vom LSG beauftragte Sachverständige Dr. S. ( Neurologe und Psychiater / Psychotherapeut ) hat die quantitative Leistungsfähigkeit der Klägerin mit mindestens 6 Stunden für leichte Arbeiten mit qualitativen Einschränkungen beurteilt .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Example 16** (doc_id: `64271`) (sent_id: `64271`)


Der Antrag des Klägers , ihm für das Verfahren der Beschwerde gegen die Nichtzulassung der Revision im Urteil des Landessozialgerichts Niedersachsen-Bremen vom 16. November 2017 Prozesskostenhilfe zu bewilligen und Rechtsanwältin K. aus H. beizuordnen , wird abgelehnt .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `Landessozialgerichts Niedersachsen-Bremen` (ORG)
- `H.` (LOC)

**Example 17** (doc_id: `64291`) (sent_id: `64291`)


Ferner hat sie beantragt , Prof. Dr. B. , Zentralinstitut für seelische Gesundheit M. , Institut für Psychiatrie und Psychosomatische Psychotherapie , als sachverständigen Zeugen zu vernehmen sowie den Sachverständigen Prof. Dr. S. zu den mit Schriftsatz vom 21. 3. 2017 erhobenen Einwendungen anzuhören .

| Predicted | Gold |
|---|---|
| `B.` | `B.` |
| `S.` | `S.` |

**Missed by this rule (FN):**

- `Zentralinstitut für seelische Gesundheit M. , Institut für Psychiatrie und Psychosomatische Psychotherapie` (ORG)

**Example 18** (doc_id: `64530`) (sent_id: `64530`)


Das LSG hat vielmehr im Anschluss an die Begründung , warum es dessen sachverständige Bewertung für überzeugend hält , ausgeführt : " Hingegen hat Dr. H. lediglich auf einer Seite kurz dargestellt , dass beim Kläger seinerzeit kein KIG Grad 3 oder höher vorliege .

| Predicted | Gold |
|---|---|
| `H.` | `H.` |

**Example 19** (doc_id: `65228`) (sent_id: `65228`)


Statt einer beantragten orthopädischen Begutachtung unter Berücksichtigung der Schmerzsymptomatik sei eine Begutachtung durch Dr. N. angeordnet worden , obwohl er ( der Kläger ) auf neurologischem Gebiet völlig gesund sei .

| Predicted | Gold |
|---|---|
| `N.` | `N.` |

**Example 20** (doc_id: `65282`) (sent_id: `65282`)


Die Taten fanden ein Ende , nachdem die Zeugin R. im Sexualkundeunterricht aufgeklärt worden war .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |

**Example 21** (doc_id: `65674`) (sent_id: `65674`)


b ) Soweit das Landgericht bei der Zumessung der Freiheitsstrafe bezüglich des Angeklagten R. rechtsfehlerhaft zu seinem Nachteil ebenfalls die tateinheitliche Begehung eines Raubes gewürdigt hat , schließt der Senat aus , dass angesichts der verbleibenden gewichtigen Strafschärfungsgründe , insbesondere im Hinblick auf die verwirklichte Vergewaltigung , das Landgericht ohne den aufgezeigten Rechtsfehler eine niedrigere Freiheitsstrafe verhängt hätte .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |

**Example 22** (doc_id: `66269`) (sent_id: `66269`)


Die Beklagte hat sich hierzu nicht geäußert und nach der Übersendung des Sachverständigengutachtens des Dr. B. ohne weitere inhaltliche Einlassung mit einer Entscheidung ohne mündliche Verhandlung einverstanden erklärt .

| Predicted | Gold |
|---|---|
| `B.` | `B.` |

**Example 23** (doc_id: `66350`) (sent_id: `66350`)


Im Berufungsverfahren fand zunächst ein Erörterungstermin am 7. 9. 2016 statt , in dem der Berichterstatter des LSG die Klägerin persönlich anhörte und Herrn G. als Zeugen vernahm .

| Predicted | Gold |
|---|---|
| `G.` | `G.` |

**Example 24** (doc_id: `66540`) (sent_id: `66540`)


„ Ausgehend vom nach Teileinstellungen noch angeklagten Sachverhalt nach Maßgabe des Hinweisbeschlusses vom 6. Hauptverhandlungstag “ wird hinsichtlich der Angeklagten K. „ bei insoweit glaubhaftem Geständnis und kooperativem Verhalten “ eine Verurteilung zu einer Gesamtfreiheitsstrafe von mindestens neun Monaten bis zu einem Jahr und drei Monaten , deren Vollstreckung zur Bewährung ausgesetzt wird , erfolgen .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Example 25** (doc_id: `66608`) (sent_id: `66608`)


Nach den getroffenen Feststellungen ist unzweifelhaft , dass der Zeuge K. , der im Fall II. 1. der Urteilsgründe selbst Cannabis vom Angeklagten erhielt und weiterverkaufte , dabei auch in der Vorstellung , den Betäubungsmittelhandel des Angeklagten zu fördern , tätig wurde .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

</details>

---

## `Names after 'Richter' or 'Richterin' (corrected)`

**F1:** 0.077 | **Precision:** 0.929 | **Recall:** 0.040  

**Format:** `regex`  
**Rule ID:** `d2bd3fae`  
**Description:**
Captures names following 'Richter', 'Richterin', 'Vorsitzender Richter', etc., ensuring no trailing space.

**Content:**
```
(?:Richter|Richterin|Vorsitzender\s+Richter|Vorsitzende\s+Richterin)\s+([A-Z][a-z\u00e4\u00f6\u00fc\u00df]*\.?\s*(?:[A-Z][a-z\u00e4\u00f6\u00fc\u00df]*\.?\s*)*|\b[A-Z]\s*\.\b|\b[A-Z]\b)(?=\s*(?:,|\.|\)|\]|\s|$))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.929 | 0.040 | 0.077 | 14 | 13 | 1 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 13 | 1 | 293 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60487`) (sent_id: `60487`)


Abweichende Meinung der Richterin Hermanns zum Beschluss des Zweiten Senats vom 22. März 2018 - 2 BvR 780/16 -

| Predicted | Gold |
|---|---|
| `Hermanns` | `Hermanns` |

**Missed by this rule (FN):**

- `Beschluss des Zweiten Senats vom 22. März 2018 - 2 BvR 780/16 -` (RS)

**Example 1** (doc_id: `61787`) (sent_id: `61787`)


In der Beschwerdesache betreffend die Marke 30 2009 056 266 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 14. September 2017 unter Mitwirkung des Vorsitzenden Richters Knoll sowie der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Kriener` | `Kriener` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Nielsen` (PER)

**Example 2** (doc_id: `61969`) (sent_id: `61969`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2013 036 234.9 hat der 26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 16. Oktober 2017 unter Mitwirkung der Vorsitzenden Richterin Kortge sowie der Richter Jacobi und Schödel beschlossen :

| Predicted | Gold |
|---|---|
| `Kortge` | `Kortge` |
| `Jacobi` | `Jacobi` |

**Missed by this rule (FN):**

- `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Schödel` (PER)

**Example 3** (doc_id: `62802`) (sent_id: `62802`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2014 004 201.0 hat der 29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 29. November 2017 durch die Vorsitzende Richterin Dr. Mittenberger-Huber , die Richterin Akintche und die Richterin Seyfarth beschlossen :

| Predicted | Gold |
|---|---|
| `Akintche` | `Akintche` |
| `Seyfarth` | `Seyfarth` |

**Missed by this rule (FN):**

- `29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Mittenberger-Huber` (PER)

**Example 4** (doc_id: `62983`) (sent_id: `62983`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2011 068 984.9 hat der 26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 8. Januar 2018 unter Mitwirkung der Vorsitzenden Richterin Kortge sowie der Richter Jacobi und Schödel beschlossen :

| Predicted | Gold |
|---|---|
| `Kortge` | `Kortge` |
| `Jacobi` | `Jacobi` |

**Missed by this rule (FN):**

- `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Schödel` (PER)

**Example 5** (doc_id: `64394`) (sent_id: `64394`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2013 053 470.0 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 9. November 2017 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Kriener` | `Kriener` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Nielsen` (PER)

**Example 6** (doc_id: `64527`) (sent_id: `64527`)


In der Beschwerdesache betreffend die Marke 30 2011 035 856 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Merzbach` | `Merzbach` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Hacker` (PER)
- `Meiser` (PER)

**Example 7** (doc_id: `64925`) (sent_id: `64925`)


In der Beschwerdesache betreffend die Marke 30 2010 003 649 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Merzbach` | `Merzbach` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Hacker` (PER)
- `Meiser` (PER)

**Example 8** (doc_id: `64967`) (sent_id: `64967`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2015 034 000.6 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 11. Januar 2018 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Kriener` | `Kriener` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Nielsen` (PER)

**Example 9** (doc_id: `65500`) (sent_id: `65500`)


In der Beschwerdesache betreffend die Designanmeldung .... ( hier : Antrag auf Verfahrenskostenhilfe ) hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Merzbach` | `Merzbach` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Hacker` (PER)
- `Meiser` (PER)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `62802`) (sent_id: `62802`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2014 004 201.0 hat der 29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 29. November 2017 durch die Vorsitzende Richterin Dr. Mittenberger-Huber , die Richterin Akintche und die Richterin Seyfarth beschlossen :

**False Positives:**

- `Dr.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Mittenberger-Huber`(PER)
- `Akintche`(PER)
- `Seyfarth`(PER)

</details>

---

## `Known German Surnames`

**F1:** 0.289 | **Precision:** 0.496 | **Recall:** 0.204  

**Format:** `regex`  
**Rule ID:** `de849019`  
**Description:**
Captures a comprehensive list of common German surnames and other frequently occurring names in legal texts, including compound names.

**Content:**
```
\b(?:Berger|Sander|Schmaltz|Bender|Fritz|Becker|Feilcke|Kiel|Brunke|Mutzbauer|Zeng|Benrath|Cierniak|Appl|Franke|Brune|Roggenbuck|Niemann|Grube|Volz|Quentin|Spinner|Schlewing|Marx|Fischer|Bredendiek|Stresemann|Kayser|Koch|Volk|Liebert|Limperg|Schlünder|Berg|Matthias|Hohoff|Leitz|Krumbiegel|Paul|Treber|Spaniol|Feddersen|Schultz|Schuh|Lauer|Lipphaus|Gräfl|Schäfer|König|Müller|Dauber|Tiemann|Deichfuß|Ahrendt|Graßnack|Schmidt-Räntschist|Schmidt|Radtke|Pohl|Nielebock|Fischermeier|Bellay|Kirchhof|Busch|Krehl|Hayen|Glock|Redeker|Morawek|Eder|Baumgardt|Hoffmann|Kaya|Seyhan|Çerikci|Hacker|Merzbach|Meiser|Knoll|Kriener|Nielsen|Musiol|Dorn|Albertshofer|Wollny|Bieringer|Hilber|Paetzold|Baumgart|Geier|Höchst|Fritze|Wiegele|Kleinschmidt|Kirschneck|Arnoldi|Haupt|Demir|Baykara|Aranyosi|Căldăraru|Nassauer|van den Berg|Einstein|Shah|Kuemmerle|Tomlinson|Wright|Vogt|Saime|Özcan|Sen|Bar|Refaeli|Josh|Duhamel|Kelvin|Heinkel|Pape|Harsdorf|Gebhardt|Enerji Yapi-Yol Sen|Eschelbach|Bormann|Möhring|Zimmermann|Rose|Kohout|Abdullah Öcalan|Schwabe|Paffrath|Jaehde|Eckstein|Matter|KCK|PKK)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.496 | 0.204 | 0.289 | 133 | 66 | 67 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 66 | 67 | 258 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60115`) (sent_id: `60115`)


Marx

| Predicted | Gold |
|---|---|
| `Marx` | `Marx` |

**Example 1** (doc_id: `60200`) (sent_id: `60200`)


Quentin

| Predicted | Gold |
|---|---|
| `Quentin` | `Quentin` |

**Example 2** (doc_id: `60460`) (sent_id: `60460`)


Koch

| Predicted | Gold |
|---|---|
| `Koch` | `Koch` |

**Example 3** (doc_id: `60485`) (sent_id: `60485`)


Radtke

| Predicted | Gold |
|---|---|
| `Radtke` | `Radtke` |

**Example 4** (doc_id: `60542`) (sent_id: `60542`)


Kohout

| Predicted | Gold |
|---|---|
| `Kohout` | `Kohout` |

**Example 5** (doc_id: `60579`) (sent_id: `60579`)


Spinner

| Predicted | Gold |
|---|---|
| `Spinner` | `Spinner` |

**Example 6** (doc_id: `60726`) (sent_id: `60726`)


Krehl

| Predicted | Gold |
|---|---|
| `Krehl` | `Krehl` |

**Example 7** (doc_id: `60994`) (sent_id: `60994`)


Hayen

| Predicted | Gold |
|---|---|
| `Hayen` | `Hayen` |

**Example 8** (doc_id: `61122`) (sent_id: `61122`)


Selbst wenn mit § 12 Nr. 4 des Arbeitsvertrags der Parteien eine unzulässige Umgehung von § 622 Abs. 6 BGB verbunden wäre , führte dies lediglich zur Nichtigkeit der Optionsklausel ( Rein NZA-RR 2009 , 462 ; Vogt Befristungs- und Optionsvereinbarungen im professionellen Mannschaftssport S. 161 f. ) , nicht aber zur Verlängerung des Vertrags .

| Predicted | Gold |
|---|---|
| `Vogt` | `Vogt` |

**Missed by this rule (FN):**

- `§ 12 Nr. 4 des Arbeitsvertrags` (REG)
- `§ 622 Abs. 6 BGB` (NRM)
- `Rein NZA-RR 2009 , 462` (LIT)

**Example 9** (doc_id: `61174`) (sent_id: `61174`)


Hohoff

| Predicted | Gold |
|---|---|
| `Hohoff` | `Hohoff` |

**Example 10** (doc_id: `61183`) (sent_id: `61183`)


Schmidt

| Predicted | Gold |
|---|---|
| `Schmidt` | `Schmidt` |

**Example 11** (doc_id: `61238`) (sent_id: `61238`)


Becker

| Predicted | Gold |
|---|---|
| `Becker` | `Becker` |

**Example 12** (doc_id: `61487`) (sent_id: `61487`)


Fischermeier

| Predicted | Gold |
|---|---|
| `Fischermeier` | `Fischermeier` |

**Example 13** (doc_id: `61517`) (sent_id: `61517`)


Volz

| Predicted | Gold |
|---|---|
| `Volz` | `Volz` |

**Example 14** (doc_id: `61573`) (sent_id: `61573`)


Becker

| Predicted | Gold |
|---|---|
| `Becker` | `Becker` |

**Example 15** (doc_id: `61618`) (sent_id: `61618`)


Busch

| Predicted | Gold |
|---|---|
| `Busch` | `Busch` |

**Example 16** (doc_id: `61671`) (sent_id: `61671`)


Krumbiegel

| Predicted | Gold |
|---|---|
| `Krumbiegel` | `Krumbiegel` |

**Example 17** (doc_id: `61787`) (sent_id: `61787`)


In der Beschwerdesache betreffend die Marke 30 2009 056 266 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 14. September 2017 unter Mitwirkung des Vorsitzenden Richters Knoll sowie der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Knoll` | `Knoll` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Kriener` (PER)
- `Nielsen` (PER)

**Example 18** (doc_id: `61916`) (sent_id: `61916`)


In dem Rechtsstreit Demir und Baykara v. Türkei seien Fragen des Streikverbots im öffentlichen Dienst nicht Gegenstand des Verfahrens gewesen .

| Predicted | Gold |
|---|---|
| `Demir` | `Demir` |
| `Baykara` | `Baykara` |

**Missed by this rule (FN):**

- `Türkei` (LOC)

**Example 19** (doc_id: `62100`) (sent_id: `62100`)


Feddersen

| Predicted | Gold |
|---|---|
| `Feddersen` | `Feddersen` |

**Example 20** (doc_id: `62189`) (sent_id: `62189`)


Berg

| Predicted | Gold |
|---|---|
| `Berg` | `Berg` |

**Example 21** (doc_id: `62451`) (sent_id: `62451`)


Bormann

| Predicted | Gold |
|---|---|
| `Bormann` | `Bormann` |

**Example 22** (doc_id: `62613`) (sent_id: `62613`)


2. Der vorliegende Fall ist durch solche besonderen , zusätzlichen Umstände gekennzeichnet , die über eine bloße Mitwirkung des Richters Müller in einem Gesetzgebungsverfahren deutlich hinausreichen und die Besorgnis seiner Befangenheit begründen .

| Predicted | Gold |
|---|---|
| `Müller` | `Müller` |

**Example 23** (doc_id: `62675`) (sent_id: `62675`)


Die Einsprechende legt zur geltend gemachten Lieferung des Kontaktsockels „ Waffle Kelvin “ an die A … Inc. die teilgeschwärzte Ablichtung einer Rechnung vor :

| Predicted | Gold |
|---|---|
| `Kelvin` | `Kelvin` |

**Missed by this rule (FN):**

- `A … Inc.` (ORG)

**Example 24** (doc_id: `62927`) (sent_id: `62927`)


Mutzbauer

| Predicted | Gold |
|---|---|
| `Mutzbauer` | `Mutzbauer` |

**Example 25** (doc_id: `63317`) (sent_id: `63317`)


Schäfer

| Predicted | Gold |
|---|---|
| `Schäfer` | `Schäfer` |

**Example 26** (doc_id: `63360`) (sent_id: `63360`)


Fischermeier

| Predicted | Gold |
|---|---|
| `Fischermeier` | `Fischermeier` |

**Example 27** (doc_id: `63382`) (sent_id: `63382`)


Franke

| Predicted | Gold |
|---|---|
| `Franke` | `Franke` |

**Example 28** (doc_id: `63457`) (sent_id: `63457`)


Berger

| Predicted | Gold |
|---|---|
| `Berger` | `Berger` |

**Example 29** (doc_id: `63556`) (sent_id: `63556`)


Quentin

| Predicted | Gold |
|---|---|
| `Quentin` | `Quentin` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60070`) (sent_id: `60070`)


Eine Mindestentfernung zwischen Haupt- und beruflicher Zweitwohnung bestimmt das Einkommensteuergesetz nicht ( Geserich , in : Kirchhof / Söhn / Mellinghoff , EStG , § 9 Rz G 60 ) .

**False Positives:**

- `Haupt` — no gold match — likely missing annotation
- `Kirchhof` — partial — pred is substring of gold: `Geserich , in : Kirchhof / Söhn / Mellinghoff , EStG , § 9 Rz G 60`

> overlaps gold: 1  |  likely missing annotation: 1

**Gold Entities:**

- `Einkommensteuergesetz`(NRM)
- `Geserich , in : Kirchhof / Söhn / Mellinghoff , EStG , § 9 Rz G 60`(LIT)

**Example 1** (doc_id: `60324`) (sent_id: `60324`)


Soweit die Widersprechende sich darauf beruft , dass auch kennzeichnungsschwache Marken zumindest Schutz gegen eine identische Übernahme beanspruchen könnten , führt dieser grundsätzlich zutreffende Einwand gleichfalls nicht zur Bejahung der Verwechslungsgefahr , da sich die hier zu vergleichenden Zeichen – wie nachfolgend unter Ziffer 1. 3. dargelegt – erheblich unterscheiden ( vgl. im Übrigen zum Schutzumfang zu Unrecht eingetragener , materiell schutzunfähiger Marken Ströbele / Hacker , MarkenG , 11. Aufl. , § 9 Rn. 194 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele / Hacker , MarkenG , 11. Aufl. , § 9 Rn. 194`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 9 Rn. 194`(LIT)

**Example 2** (doc_id: `60329`) (sent_id: `60329`)


Die strafschärfende Berücksichtigung der hierin liegenden Schuldsteigerung gerate weder mit dem in § 46 Abs. 3 StGB verankerten Doppelverwertungsverbot von Tatbestandsmerkmalen ( SSW-StGB / Eschelbach , aaO , § 46 Rn. 93 , 185 ; von Heintschel-Heinegg , Streng-FS 2017 S. 229 , 239 ) noch mit dem Gedanken in Konflikt , dass es sich um das Regeltatbild des Totschlags handele ( Fahl , JR 2017 , 391 , 393 ; MüKo / Schneider , aaO , § 212 Rn. 82 ; Tomiak , HRRS 2017 , 225 ff. ) .

**False Positives:**

- `Eschelbach` — partial — pred is substring of gold: `SSW-StGB / Eschelbach , aaO , § 46 Rn. 93 , 185`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 46 Abs. 3 StGB`(NRM)
- `SSW-StGB / Eschelbach , aaO , § 46 Rn. 93 , 185`(LIT)
- `von Heintschel-Heinegg , Streng-FS 2017 S. 229 , 239`(LIT)
- `Fahl , JR 2017 , 391 , 393`(LIT)
- `MüKo / Schneider , aaO , § 212 Rn. 82`(LIT)
- `Tomiak , HRRS 2017 , 225 ff.`(LIT)

**Example 3** (doc_id: `60363`) (sent_id: `60363`)


Übe ein Arbeitnehmer eine vom Arbeitgeber entlohnte Nebentätigkeit aus , so seien die Einnahmen aus der Nebentätigkeit durch das Arbeitsverhältnis veranlasst , wenn Haupt- und Nebentätigkeit gleichartig seien und die Nebentätigkeit unter ähnlichen organisatorischen Bedingungen ausgeübt werde wie die Haupttätigkeit oder wenn der Steuerpflichtige mit der Nebentätigkeit ihm aus seinem Dienstverhältnis - faktisch oder rechtlich - obliegende Nebenpflichten erfülle .

**False Positives:**

- `Haupt` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `60577`) (sent_id: `60577`)


Hingegen ist die Zulassung wegen Divergenz gegen eine Entscheidung eines anderen obersten Gerichtshofes des Bundes oder des EuGH nicht zulässig ( vgl Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160 RdNr 11 mwN ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160 RdNr 11`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH`(ORG)
- `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160 RdNr 11`(LIT)

**Example 5** (doc_id: `60634`) (sent_id: `60634`)


Das Landgericht hätte sich an dieser Stelle daher auch damit auseinandersetzen müssen , dass der Angeklagte am 12. Juli 2015 einen Diebstahl „ im besonders schweren Fall “ beging , wofür er am 6. Juli 2016 vom Amtsgericht Frankfurt am Main - Außenstelle Höchst - verurteilt wurde .

**False Positives:**

- `Höchst` — partial — pred is substring of gold: `Amtsgericht Frankfurt am Main - Außenstelle Höchst -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Amtsgericht Frankfurt am Main - Außenstelle Höchst -`(ORG)

**Example 6** (doc_id: `60870`) (sent_id: `60870`)


Dabei stehen die genannten Faktoren in einem Verhältnis der Wechselwirkung , so dass ein geringerer Grad eines Faktors durch einen höheren Grad eines anderen Faktors ausgeglichen werden kann ( EuGH GRUR 1998 , 387 , 389 Rn. 22 – Sabél / Puma ; GRUR 1998 , 922 , 923 Rn. 17 – Canon ; GRUR Int. 1999 , 734 , 736 Rn. 19 – Lloyd ; GRUR Int. 2000 , 899 , 901 Rn. 40 – Marca / Adidas ; GRUR 2008 , 343 , 345 Rn. 48 – Il Ponte Finanziaria Spa / HABM ; BGH GRUR 2012 , 1040 , 1042 Rn. 25 – pjur / pure ; GRUR 2012 , 930 , 932 Rn. 22 – Bogner B / Barbie B / ; GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY ; GRUR 2011 , 826 Rn. 11 – Enzymax / Enzymix ; GRUR 2011 , 824 Rn. 18 – Kappa ; GRUR 2010 , 235 Rn. 35 – AIDA / AIDU ; GRUR 2009 , 766 , 768 Rn. 26 – Stofffähnchen ; GRUR 2009 , 772 , 776 Rn. 51 – Augsburger Puppenkiste ; GRUR 2009 , 484 , 486 Rn. 23 – Metrobus ; GRUR 2008 , 1002 , 1004 Rn. 23 – Schuhpark ; Hacker , a. a. O. , § 9 Rn. 41 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Hacker , a. a. O. , § 9 Rn. 41`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 1998 , 387 , 389 Rn. 22 – Sabél / Puma`(RS)
- `GRUR 1998 , 922 , 923 Rn. 17 – Canon`(RS)
- `GRUR Int. 1999 , 734 , 736 Rn. 19 – Lloyd`(RS)
- `GRUR Int. 2000 , 899 , 901 Rn. 40 – Marca / Adidas`(RS)
- `GRUR 2008 , 343 , 345 Rn. 48 – Il Ponte Finanziaria Spa / HABM`(RS)
- `BGH GRUR 2012 , 1040 , 1042 Rn. 25 – pjur / pure`(RS)
- `GRUR 2012 , 930 , 932 Rn. 22 – Bogner B / Barbie B /`(RS)
- `GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY`(RS)
- `GRUR 2011 , 826 Rn. 11 – Enzymax / Enzymix`(RS)
- `GRUR 2011 , 824 Rn. 18 – Kappa`(RS)
- `GRUR 2010 , 235 Rn. 35 – AIDA / AIDU`(RS)
- `GRUR 2009 , 766 , 768 Rn. 26 – Stofffähnchen`(RS)
- `GRUR 2009 , 772 , 776 Rn. 51 – Augsburger Puppenkiste`(RS)
- `GRUR 2009 , 484 , 486 Rn. 23 – Metrobus`(RS)
- `GRUR 2008 , 1002 , 1004 Rn. 23 – Schuhpark`(RS)
- `Hacker , a. a. O. , § 9 Rn. 41`(LIT)

**Example 7** (doc_id: `60929`) (sent_id: `60929`)


Darauf zielt der jeweilige Gegenstand des Patentanspruchs 1 nach Haupt- und Hilfsantrag ersichtlich nicht .

**False Positives:**

- `Haupt` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `61007`) (sent_id: `61007`)


Im Allgemeinen lassen unter anderem Angaben über Werbeaufwendungen Schlüsse auf die Verkehrsbekanntheit einer Marke zu ( BGH GRUR 2013 , 833 , 836 Rn. 41 – Culinaria / Villa Culinaria ; Hacker , a. a. O. , § 9 Rn. 160 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Hacker , a. a. O. , § 9 Rn. 160`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH GRUR 2013 , 833 , 836 Rn. 41 – Culinaria / Villa Culinaria`(RS)
- `Hacker , a. a. O. , § 9 Rn. 160`(LIT)

**Example 9** (doc_id: `61028`) (sent_id: `61028`)


2. Der Grundsatz der Gewaltenteilung ( Art. 20 Abs. 2 Satz 2 GG ) verlangt , dass die Rechtsprechung durch " besondere " , das heißt von den Organen der Gesetzgebung und der vollziehenden Gewalt verschiedene Organe des Staates ausgeübt wird ( BVerfGE 18 , 241 < 254 > ) ; dies wird durch das in Art. 92 1. Halbsatz GG begründete Rechtsprechungsmonopol der Richter konkretisiert ( vgl. BVerfGE 22 , 49 < 76 > ; Hopfauf , in : Schmidt-Bleibtreu / Hofmann / Henneke , GG , 14. Aufl. 2018 , Art. 92 Rn. 2 ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Hopfauf , in : Schmidt-Bleibtreu / Hofmann / Henneke , GG , 14. Aufl. 2018 , Art. 92 Rn. 2`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 20 Abs. 2 Satz 2 GG`(NRM)
- `BVerfGE 18 , 241 < 254 >`(RS)
- `Art. 92 1. Halbsatz GG`(NRM)
- `BVerfGE 22 , 49 < 76 >`(RS)
- `Hopfauf , in : Schmidt-Bleibtreu / Hofmann / Henneke , GG , 14. Aufl. 2018 , Art. 92 Rn. 2`(LIT)

**Example 10** (doc_id: `61312`) (sent_id: `61312`)


In der Rechtsprechung des Bundesverfassungsgerichts ist jedoch anerkannt , dass eine Erledigung nicht zur Unzulässigkeit der Verfassungsbeschwerde führt , wenn der gerügte Grundrechtseingriff besonders schwer wiegt und anderenfalls die Klärung einer verfassungsrechtlichen Frage von grundsätzlicher Bedeutung unterbliebe ( vgl. BVerfGE 81 , 138 < 141 f. > ; 91 , 125 < 133 > ; 98 , 169 < 198 > ; 103 , 44 < 58 > ) , die gegenstandslos gewordene Maßnahme den Beschwerdeführer weiterhin beeinträchtigt ( vgl. BVerfGE 99 , 129 < 138 > ) oder ein Rehabilitationsinteresse des Beschwerdeführers besteht ( vgl. auch BVerfG , Urteil des Zweiten Senats vom 7. November 2017 - 2 BvE 2/11 - , juris , Rn. 183 ; Bethge , in : Maunz / Schmidt-Bleibtreu / Klein / Bethge , BVerfGG , § 90 Rn. 269a < Oktober 2013 > ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Bethge , in : Maunz / Schmidt-Bleibtreu / Klein / Bethge , BVerfGG , § 90 Rn. 269a < Oktober 2013 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesverfassungsgerichts`(ORG)
- `BVerfGE 81 , 138 < 141 f. > ; 91 , 125 < 133 > ; 98 , 169 < 198 > ; 103 , 44 < 58 >`(RS)
- `BVerfGE 99 , 129 < 138 >`(RS)
- `BVerfG , Urteil des Zweiten Senats vom 7. November 2017 - 2 BvE 2/11 - , juris , Rn. 183`(RS)
- `Bethge , in : Maunz / Schmidt-Bleibtreu / Klein / Bethge , BVerfGG , § 90 Rn. 269a < Oktober 2013 >`(LIT)

**Example 11** (doc_id: `61434`) (sent_id: `61434`)


Zwar ist anerkannt , dass die Zulassung der Revision nicht notwendig im Tenor erfolgen muss , sondern auch in den Entscheidungsgründen erfolgen kann ( BSG Beschluss vom 30. 6. 2008 - B 2 U 1/08 RH - SozR 4 - 1500 § 160 Nr 17 RdNr 11 mwN ; BSG Urteil vom 29. 6. 1977 - 11 RA 94/76 - SozR 1500 § 161 Nr 16 ) , sofern sie eindeutig ausgesprochen wird ( Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160 RdNr 24a mwN ; Voelzke in juris-PK SGG , § 160 RdNr 58 ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160 RdNr 24a`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Beschluss vom 30. 6. 2008 - B 2 U 1/08 RH - SozR 4 - 1500 § 160 Nr 17 RdNr 11`(RS)
- `BSG Urteil vom 29. 6. 1977 - 11 RA 94/76 - SozR 1500 § 161 Nr 16`(RS)
- `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160 RdNr 24a`(LIT)
- `Voelzke in juris-PK SGG , § 160 RdNr 58`(LIT)

**Example 12** (doc_id: `61445`) (sent_id: `61445`)


Denn der Verkehr ist daran gewöhnt , im Geschäftsleben ständig mit neuen Wortschöpfungen konfrontiert zu werden , durch die sachbezogene Informationen übermittelt werden sollen und die sich häufig nicht an grammatikalischen Regeln orientieren ( vgl. Ströbele in : Ströbele / Hacker , Markengesetz , 12. Aufl. 2018 , § 8 , Rn. 199 ; BPatG 24 W ( pat ) 510/15 – Knetmonster , verfügbar über PAVIS PROMA ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele in : Ströbele / Hacker , Markengesetz , 12. Aufl. 2018 , § 8 , Rn. 199`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Ströbele in : Ströbele / Hacker , Markengesetz , 12. Aufl. 2018 , § 8 , Rn. 199`(LIT)
- `BPatG 24 W ( pat ) 510/15 – Knetmonster , verfügbar über PAVIS PROMA`(RS)

**Example 13** (doc_id: `61635`) (sent_id: `61635`)


Maßgeblich ist bei Wortkombinationen letztlich , ob die Kombination der Bestandteile über die bloße Zusammenfügung beschreibender Elemente hinausgeht oder sich – wie vorliegend – in deren Summenwirkung erschöpft , was der Unterscheidungskraft entgegensteht ( siehe dazu auch Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 182 mit weiteren Rechtsprechungsnachweisen ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 182`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 182`(LIT)

**Example 14** (doc_id: `61642`) (sent_id: `61642`)


M2.2. 2a wobei die Kontaktfedern ( 7a , 7b ) eines Kontaktfederpaares so angeordnet sind , dass sie für eine Kelvin-Kontaktierung beide jeweils gegen denselben Anschlusskontakt ( 5 ) des Bauelements ( 3 ) gedrückt werden , und wobei

**False Positives:**

- `Kelvin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 15** (doc_id: `61690`) (sent_id: `61690`)


Eine statthafte und zulässige Vollstreckungserinnerung setzt eine erinnerungsfähige Vollstreckungsmaßnahme oder ein Unterlassen voraus ( vgl. Spohnheimer , in : Wieczorek / Schütze , ZPO , 4. Aufl. 2016 , § 766 Rn. 63 ; Schmidt / Brinkmann , in : MüKo-ZPO , 5. Aufl. 2016 , § 766 Rn. 48 ) , die Erinnerung nach § 766 Abs. 2 Alternative 3 ZPO in der Regel einen Kostenansatz , eine Zahlungsaufforderung oder die Vorbereitung der Abrechnung durch den Gerichtsvollzieher ( vgl. LG Dortmund , Beschluss vom 19. Oktober 2006 - 9 T 613/06 - , NJOZ 2007 , S. 65 < 66 f. > ; LG Hannover , Beschluss vom 4. Februar 1977 - 11 T 162/76 - , juris , Rn. 2 f. ; vgl. auch Schmidt / Brinkmann , in : MüKo-ZPO , 5. Aufl. 2016 , § 766 Rn. 62 ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Schmidt / Brinkmann , in : MüKo-ZPO , 5. Aufl. 2016 , § 766 Rn. 48`
- `Schmidt` — similar text (different position): `Schmidt / Brinkmann , in : MüKo-ZPO , 5. Aufl. 2016 , § 766 Rn. 48`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Spohnheimer , in : Wieczorek / Schütze , ZPO , 4. Aufl. 2016 , § 766 Rn. 63`(LIT)
- `Schmidt / Brinkmann , in : MüKo-ZPO , 5. Aufl. 2016 , § 766 Rn. 48`(LIT)
- `§ 766 Abs. 2 Alternative 3 ZPO`(NRM)
- `LG Dortmund , Beschluss vom 19. Oktober 2006 - 9 T 613/06 - , NJOZ 2007 , S. 65 < 66 f. >`(RS)
- `LG Hannover , Beschluss vom 4. Februar 1977 - 11 T 162/76 - , juris , Rn. 2 f.`(RS)
- `Schmidt / Brinkmann , in : MüKo-ZPO , 5. Aufl. 2016 , § 766 Rn. 62`(LIT)

**Example 16** (doc_id: `61693`) (sent_id: `61693`)


Insoweit hätte es des klägerseitigen Vortrags bedurft , weshalb nach den dem LSG vorliegenden Beweismitteln Fragen zum tatsächlichen und medizinischen Sachverhalt aus der rechtlichen Sicht des LSG erkennbar offengeblieben sind und damit zu einer weiteren Aufklärung des Sachverhalts zwingende Veranlassung bestanden haben soll ( vgl Becker , Die Nichtzulassungsbeschwerde zum BSG < Teil II > , SGb 2007 , 328 , 332 zu RdNr 188 unter Hinweis auf BSG Beschluss vom 14. 12. 1999 - B 2 U 311/99 B - mwN ) .

**False Positives:**

- `Becker` — partial — pred is substring of gold: `Becker , Die Nichtzulassungsbeschwerde zum BSG < Teil II > , SGb 2007 , 328 , 332 zu RdNr 188`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Becker , Die Nichtzulassungsbeschwerde zum BSG < Teil II > , SGb 2007 , 328 , 332 zu RdNr 188`(LIT)
- `BSG Beschluss vom 14. 12. 1999 - B 2 U 311/99 B -`(RS)

**Example 17** (doc_id: `61703`) (sent_id: `61703`)


Entscheidend ist daher , ob es im Hinblick auf die beanspruchten Waren für sich genommen als beschreibende Angabe verwendet werden kann ( vgl. BGH Mitt. 1995 , 184 - quattro ; GRUR 2000 , 231 , 232 - FÜNFER ; BPatG , Beschluss vom 30. Januar 2008 , 29 W ( pat ) 92/04 - DUO ; Beschluss vom 27. Januar 2010 , 28 W ( pat ) 96/09 - TRIO ; Ströbele / Hacker / Thiering , a. a. O. , § 8 , Rdnr. 508 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele / Hacker / Thiering , a. a. O. , § 8 , Rdnr. 508`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH Mitt. 1995 , 184 - quattro`(RS)
- `GRUR 2000 , 231 , 232 - FÜNFER`(RS)
- `BPatG , Beschluss vom 30. Januar 2008 , 29 W ( pat ) 92/04 - DUO`(RS)
- `Beschluss vom 27. Januar 2010 , 28 W ( pat ) 96/09 - TRIO`(RS)
- `Ströbele / Hacker / Thiering , a. a. O. , § 8 , Rdnr. 508`(LIT)

**Example 18** (doc_id: `61770`) (sent_id: `61770`)


Der Schaft und der Dorn sind , wie in Absatz [ 0040 ] beschrieben und in Fig. 1 gezeigt , aus einem einzigen Stift gebildet und beim Verschrauben axial fluchtend angeordnet .

**False Positives:**

- `Dorn` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `62068`) (sent_id: `62068`)


Wird das Vorliegen eines Verfahrensmangels nach § 160 Abs 2 Nr 3 SGG gerügt , so müssen bei dessen Bezeichnung wie bei einer Verfahrensrüge innerhalb einer zugelassenen Revision zunächst die diesen Verfahrensmangel des LSG ( vermeintlich ) begründenden Tatsachen substantiiert dargelegt werden ( vgl nur BSG SozR 1500 § 160a Nr 14 und 36 ; Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160a RdNr 16 mwN ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160a RdNr 16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 160 Abs 2 Nr 3 SGG`(NRM)
- `BSG SozR 1500 § 160a Nr 14 und 36`(RS)
- `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160a RdNr 16`(LIT)

**Example 20** (doc_id: `62090`) (sent_id: `62090`)


Ob Werbungskosten in diesem Sinne in einem unmittelbaren wirtschaftlichen Zusammenhang mit den Einnahmen stehen , ist unter Rückgriff auf die zu § 3c Abs. 1 EStG entwickelten Grundsätze zu klären ( vgl. Lübbehüsen in Berger / Steck / Lübbehüsen , a. a. O. , § 3 Rz 23 ; Köhler in Berger / Steck / Lübbehüsen , a. a. O. , § 3 Rz 87 , m. w. N. ; Feyerabend / Mielke / Rieger , Recht der Finanzinstrumente 2011 , 191 , 194 ; vgl. auch BMF-Schreiben in BStBl I 2005 , 728 , Rz 45 ) .

**False Positives:**

- `Berger` — partial — pred is substring of gold: `Lübbehüsen in Berger / Steck / Lübbehüsen , a. a. O. , § 3 Rz 23`
- `Berger` — similar text (different position): `Lübbehüsen in Berger / Steck / Lübbehüsen , a. a. O. , § 3 Rz 23`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 3c Abs. 1 EStG`(NRM)
- `Lübbehüsen in Berger / Steck / Lübbehüsen , a. a. O. , § 3 Rz 23`(LIT)
- `Köhler in Berger / Steck / Lübbehüsen , a. a. O. , § 3 Rz 87`(LIT)
- `Feyerabend / Mielke / Rieger , Recht der Finanzinstrumente 2011 , 191 , 194`(LIT)
- `BMF-Schreiben in BStBl I 2005 , 728 , Rz 45`(REG)

**Example 21** (doc_id: `62499`) (sent_id: `62499`)


Kritische Stimmen ( vgl. Jescheck / Weigend Strafrecht AT , 5. Aufl. S. 887 ; SSW-StGB / Eschelbach , 2. Aufl. § 46 Rn. 93 , 185 ; Frisch , in : 50 Jahre Bundesgerichtshof , Festgabe aus der Wissenschaft , 2000 , Band IV , S. 269 , 290 f. ; Hörnle , Tatproportionale Strafzumessung , 1999 , S. 260 , 263 ; Grünewald , Das vorsätzliche Tötungsdelikt , 2010 , S. 148 ff. ; Foth , JR 1985 , 397 , 398 ; Bruns , JR 1981 , 512 , 513 ; Müller , NStZ 1985 , 158 , 161 ) haben darauf hingewiesen , dass die Auffassung , wonach die Vorsatzform als eine eigenständige Strafzumessungstatsache ausscheide , den aus dem besonderen Teil des Strafgesetzbuchs ersichtlichen gesetzgeberischen Wertungen widerspreche ( vgl. Foth , JR 1985 , 398 ; Fahl , Zur Bedeutung des Regeltatbildes bei der Bemessung der Strafe , Diss. 1996 , S. 154 ) .

**False Positives:**

- `Eschelbach` — partial — pred is substring of gold: `SSW-StGB / Eschelbach , 2. Aufl. § 46 Rn. 93 , 185`
- `Müller` — partial — pred is substring of gold: `Müller , NStZ 1985 , 158 , 161`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Jescheck / Weigend Strafrecht AT , 5. Aufl. S. 887`(LIT)
- `SSW-StGB / Eschelbach , 2. Aufl. § 46 Rn. 93 , 185`(LIT)
- `Frisch , in : 50 Jahre Bundesgerichtshof , Festgabe aus der Wissenschaft , 2000 , Band IV , S. 269 , 290 f.`(LIT)
- `Hörnle , Tatproportionale Strafzumessung , 1999 , S. 260 , 263`(LIT)
- `Grünewald , Das vorsätzliche Tötungsdelikt , 2010 , S. 148 ff.`(LIT)
- `Foth , JR 1985 , 397 , 398`(LIT)
- `Bruns , JR 1981 , 512 , 513`(LIT)
- `Müller , NStZ 1985 , 158 , 161`(LIT)
- `Strafgesetzbuchs`(NRM)
- `Foth , JR 1985 , 398`(LIT)
- `Fahl , Zur Bedeutung des Regeltatbildes bei der Bemessung der Strafe , Diss. 1996 , S. 154`(LIT)

**Example 22** (doc_id: `62527`) (sent_id: `62527`)


2. An der Besteuerung wird Deutschland durch das DBA-Kanada 2001 nicht gehindert ; der Senat hält insoweit an der in einem Verfahren des einstweiligen Rechtsschutzes bereits geäußerten Rechtsauffassung ( Senatsbeschluss vom 13. Dezember 2011 I B 159/11 , BFH / NV 2012 , 417 ) auch nach erneuter Prüfung fest ( dieser Rechtslage zustimmend z.B. Gosch in Kirchhof , EStG , 16. Aufl. , § 49 Rz 90 mit Fußn. 6 ; Hick in Schönfeld / Ditz , DBA , Art. 18 Rz 167 f. , 171 , 174 ; Blümich / Wied , § 49 EStG Rz 218 ; Kuhn in Herrmann / Heuer / Raupach , § 49 EStG Rz 1003 ; Kühnen , EFG 2016 , 578 ; Schober , EFG 2016 , 990 ; Ismer in Vogel / Lehner , DBA , 6. Aufl. , Art. 18 Rz 87 ; Holthaus , Internationale Wirtschaftsbriefe 2017 , 796 , 797 ; im Ergebnis ebenso die Verwaltungspraxis , s. Bayerisches Landesamt für Steuern , Verfügung vom 8. Juni 2011 , Internationales Steuerrecht 2011 , 776 ; a. A. W. Wassermeyer in Wassermeyer , DBA , Kanada Art. 18 Rz 70a ; Hagemann / Kahlenberg / Cloer , Betriebs-Berater 2017 , 2775 , 2785 f. ) .

**False Positives:**

- `Kirchhof` — partial — pred is substring of gold: `Gosch in Kirchhof , EStG , 16. Aufl. , § 49 Rz 90 mit Fußn. 6`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Deutschland`(LOC)
- `DBA-Kanada 2001`(NRM)
- `Senatsbeschluss vom 13. Dezember 2011 I B 159/11 , BFH / NV 2012 , 417`(RS)
- `Gosch in Kirchhof , EStG , 16. Aufl. , § 49 Rz 90 mit Fußn. 6`(LIT)
- `Hick in Schönfeld / Ditz , DBA , Art. 18 Rz 167 f. , 171 , 174`(LIT)
- `Blümich / Wied , § 49 EStG Rz 218`(LIT)
- `Kuhn in Herrmann / Heuer / Raupach , § 49 EStG Rz 1003`(LIT)
- `Kühnen , EFG 2016 , 578`(LIT)
- `Schober , EFG 2016 , 990`(LIT)
- `Ismer in Vogel / Lehner , DBA , 6. Aufl. , Art. 18 Rz 87`(LIT)
- `Holthaus , Internationale Wirtschaftsbriefe 2017 , 796 , 797`(LIT)
- `Bayerisches Landesamt für Steuern , Verfügung vom 8. Juni 2011 , Internationales Steuerrecht 2011 , 776`(LIT)
- `W. Wassermeyer in Wassermeyer , DBA , Kanada Art. 18 Rz 70a`(LIT)
- `Hagemann / Kahlenberg / Cloer , Betriebs-Berater 2017 , 2775 , 2785 f.`(LIT)

**Example 23** (doc_id: `62660`) (sent_id: `62660`)


Die Lebenszeiternennung gewährleistet allerdings das Höchstmaß an Unabhängigkeit ( vgl. Schmidt-Räntsch , DRiG , 6. Aufl. 2009 , § 11 Rn. 2 ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Schmidt-Räntsch , DRiG , 6. Aufl. 2009 , § 11 Rn. 2`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Schmidt-Räntsch , DRiG , 6. Aufl. 2009 , § 11 Rn. 2`(LIT)

**Example 24** (doc_id: `62671`) (sent_id: `62671`)


Es kann dahinstehen , ob es fachüblich ist , vor der ( Vierleiter- ) Testung an jedem Anschlusskontakt des Bauelements in einer Schleife über die beiden Kontaktfedern der Kelvin-Kontaktierung den Übergangswiderstand der Kontaktierung zu den Kontaktfedern eines Kontaktfederpaares zu bestimmen und später zur Korrektur der Messergebnisse zu verwenden , vgl. Patentschrift , Absatz 0020 , denn der Senat kann auch im Zusammenhang mit einer derartigen Messung des Übergangswiderstands keine Veranlassung des Fachmanns erkennen , die Kontaktfedern C entsprechend der Anweisungen in den Merkmalen M3 bis M3.2 lamelliert für eine hohe Stromtragfähigkeit auszubilden .

**False Positives:**

- `Kelvin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 25** (doc_id: `62688`) (sent_id: `62688`)


Dabei muss der schutzwürdige Besitzstand durch eine hinreichende Marktpräsenz und daraus folgende ( gewisse ) Bekanntheit der Kennzeichnung im Inland belegt sein ( vgl. BGH GRUR 2014 , 780 – Liquidrom ; Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 877 ; Ingerl / Rohnke , MarkenG , 3. Auflage , § 8 Rn. 308 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 877`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH GRUR 2014 , 780 – Liquidrom`(RS)
- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 877`(LIT)
- `Ingerl / Rohnke , MarkenG , 3. Auflage , § 8 Rn. 308`(LIT)

**Example 26** (doc_id: `62715`) (sent_id: `62715`)


Es ist also zu fragen , ob bei den beteiligten Verkehrskreisen der Eindruck aufkommen kann , Ware und Dienstleistung unterlägen der Kontrolle desselben Unternehmens , sei es , dass das Dienstleistungsunternehmen sich selbständig auch mit der Herstellung bzw. dem Vertrieb der Waren befasst , sei es , dass der Warenhersteller oder -vertreiber sich auch auf dem entsprechenden Dienstleistungsbereich selbständig gewerblich betätigt ( Ströbele / Hacker , MarkenG , 11. Auflage , § 9 Rn. 115 ; BGH , GRUR 2012 , 1145 , 1148 Rn. 35 - Pelikan ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele / Hacker , MarkenG , 11. Auflage , § 9 Rn. 115`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Ströbele / Hacker , MarkenG , 11. Auflage , § 9 Rn. 115`(LIT)
- `BGH , GRUR 2012 , 1145 , 1148 Rn. 35 - Pelikan`(RS)

**Example 27** (doc_id: `63072`) (sent_id: `63072`)


Schmidt-Räntsch

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Schmidt-Räntsch`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Schmidt-Räntsch`(PER)

**Example 28** (doc_id: `63088`) (sent_id: `63088`)


Denn eine Abkürzung ist nur dann nicht schutzfähig , wenn sie im Verkehr als solche gebräuchlich oder aus sich heraus verständlich ist sowie von den beteiligten Verkehrskreisen ohne weiteres der betreffenden Sachangabe gleichgesetzt und insoweit verstanden werden kann ( vgl. Ströbele in Ströbele / Hacker / Thiering , a. a. O. , § 8 Rn. 224 , 226 , 227 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele in Ströbele / Hacker / Thiering , a. a. O. , § 8 Rn. 224 , 226 , 227`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Ströbele in Ströbele / Hacker / Thiering , a. a. O. , § 8 Rn. 224 , 226 , 227`(LIT)

**Example 29** (doc_id: `63162`) (sent_id: `63162`)


Notwendig hierfür ist eine Grundlage im nationalen Recht ( vgl. EGMR , Enerji Yapi-Yol Sen c. Turquie , Urteil vom 21. April 2009 , Nr. 68959/01 , § 26 ) .

**False Positives:**

- `Enerji Yapi-Yol Sen` — partial — pred is substring of gold: `EGMR , Enerji Yapi-Yol Sen c. Turquie , Urteil vom 21. April 2009 , Nr. 68959/01 , § 26`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EGMR , Enerji Yapi-Yol Sen c. Turquie , Urteil vom 21. April 2009 , Nr. 68959/01 , § 26`(RS)

</details>

---

## `Anonymized names with ellipses (space)`

**F1:** 0.063 | **Precision:** 0.423 | **Recall:** 0.034  

**Format:** `regex`  
**Rule ID:** `1b824d7a`  
**Description:**
Captures anonymized names with a letter and ellipsis (e.g., 'K …', 'T …', 'A …', 'R …').

**Content:**
```
(?<![A-Za-zäöüß])([A-Z])\s+…
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.423 | 0.034 | 0.063 | 26 | 11 | 15 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 11 | 15 | 303 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60716`) (sent_id: `60716`)


Daraufhin ist das Anschreiben vom 31. Juli 2012 zusammen mit dem Bescheid nochmals mit Zustellungsurkunde an Patentanwalt K … verschickt und ausweislich der Zustellungsurkunde am 11. August 2012 durch Einlegen des Schriftstücks in den zur Wohnung gehörenden Briefkasten oder in eine ähnliche Vorrichtung zugestellt worden .

| Predicted | Gold |
|---|---|
| `K …` | `K …` |

**Example 1** (doc_id: `61830`) (sent_id: `61830`)


Ferner hat die Einsprechende das „ Gutachten zum Sicherheitsschalter CES der Firma E … “ des Herrn Prof. Dr. - Ing. G … vom 15. September 2014 vorgelegt .

| Predicted | Gold |
|---|---|
| `G …` | `G …` |

**Missed by this rule (FN):**

- `E …` (ORG)

**Example 2** (doc_id: `62848`) (sent_id: `62848`)


Anlage A20 : Besucherausweis der Herren D … und S … vom 09. 09. 2010 ,

| Predicted | Gold |
|---|---|
| `D …` | `D …` |
| `S …` | `S …` |

**Example 3** (doc_id: `62874`) (sent_id: `62874`)


Am Nachmittag des Ablauftages der Einspruchsfrist sei die in der Kanzlei der Vertreter damals noch in der Ausbildung befindliche Patentanwaltsfachangestellte B … beauftragt gewesen , farbig markierte Zeichnungen aus im Einspruch zitierten Schriften zum besseren Verständnis von deren Offenbarung anzufertigen .

| Predicted | Gold |
|---|---|
| `B …` | `B …` |

**Example 4** (doc_id: `63250`) (sent_id: `63250`)


Anlage A19 : Besucherausweis der Herren S … und M2 … vom 28. 04. 2010 ,

| Predicted | Gold |
|---|---|
| `S …` | `S …` |

**Missed by this rule (FN):**

- `M2 …` (PER)

**Example 5** (doc_id: `63808`) (sent_id: `63808`)


Aufgrund der dargelegten Sachlage hätte die Prüfungsstelle die Unwirksamkeit der Zustellungen erkennen können , insbesondere nachdem sie durch die Mitteilung von Patentanwalt B1 … vom 2. Mai 2013 Kenntnis von dem Bescheid der Patentanwaltskammer vom 4. April 2013 und damit von der Tatsache erhalten hat , dass die Kanzlei des beigeordneten Patentanwalts K … jedenfalls zum Zeit- punkt der vermeintlichen Zustellung der Fristverlängerung mit Beschlussankündigung am 14. März 2013 schon seit einiger Zeit verwaist war .

| Predicted | Gold |
|---|---|
| `K …` | `K …` |

**Missed by this rule (FN):**

- `B1 …` (PER)

**Example 6** (doc_id: `64429`) (sent_id: `64429`)


Als Beweis für ihren diesbezüglichen Vortrag zur offenkundigen Vorbenutzung hat die Einsprechende zuletzt nur noch den Zeugen H … angeboten ( vgl. Protokoll der mündlichen Verhandlung vom 04. 12. 2017 , S. 2 ) .

| Predicted | Gold |
|---|---|
| `H …` | `H …` |

**Example 7** (doc_id: `64657`) (sent_id: `64657`)


Sie hat insbesondere ausgesagt : Herr F … nahm das betreffende Kuvert und bestand darauf , das Kuvert selbst abzuliefern .

| Predicted | Gold |
|---|---|
| `F …` | `F …` |

**Example 8** (doc_id: `64795`) (sent_id: `64795`)


Weiterhin macht er geltend , Patentanwalt K … sei wegen einer psychischen Erkrankung zur Zeit der Zustellversuche des DPMA geschäftsunfähig nach § 104 Abs. 2 BGB gewesen , weshalb die Zustellungen unwirksam seien .

| Predicted | Gold |
|---|---|
| `K …` | `K …` |

**Missed by this rule (FN):**

- `DPMA` (ORG)
- `§ 104 Abs. 2 BGB` (NRM)

**Example 9** (doc_id: `65305`) (sent_id: `65305`)


c ) Damit erledigt sich der Antrag auf Gewährung von Prozesskostenhilfe und Beiordnung von Rechtsanwältin H … für das Verfahren auf Erlass einer einstweiligen Anordnung .

| Predicted | Gold |
|---|---|
| `H …` | `H …` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60393`) (sent_id: `60393`)


Anlage A4 Handzeichnung des Werkzeugs zum Zusammendrehen von Schraubentellerfedern zu der bei der R … GmbH & Co. KG geltend gemachten offenkundigen Vorbenutzung ,

**False Positives:**

- `R …` — partial — pred is substring of gold: `R … GmbH & Co. KG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `R … GmbH & Co. KG`(ORG)

**Example 1** (doc_id: `61011`) (sent_id: `61011`)


Das Geständnis des Beschwerdeführers , die inkriminierten Äußerungen stammten von ihm , bezieht sich nur auf den Blogeintrag und ist daher für den ebenfalls vom Anfangsverdacht umfassten Kommentar auf der Webseite " D … " unbeachtlich .

**False Positives:**

- `D …` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `61414`) (sent_id: `61414`)


Mit Schreiben vom 3. Juni 2014 machte die Beklagte gegenüber dem japanischen Tochterunternehmen des M … geltend , dass das Arzneimittel Isen- tress in den Schutzbereich des zur Familie des Streitpatents gehörenden japanischen Patents JP 5 207 392 falle .

**False Positives:**

- `M …` — type mismatch — same span as gold: `M …`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `M …`(ORG)

**Example 3** (doc_id: `61559`) (sent_id: `61559`)


Ferner reicht sie einen Handelsregisterauszug der K … GmbH zu den Akten .

**False Positives:**

- `K …` — partial — pred is substring of gold: `K … GmbH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K … GmbH`(ORG)

**Example 4** (doc_id: `61830`) (sent_id: `61830`)


Ferner hat die Einsprechende das „ Gutachten zum Sicherheitsschalter CES der Firma E … “ des Herrn Prof. Dr. - Ing. G … vom 15. September 2014 vorgelegt .

**False Positives:**

- `E …` — type mismatch — same span as gold: `E …`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `E …`(ORG)
- `G …`(PER)

**Example 5** (doc_id: `62675`) (sent_id: `62675`)


Die Einsprechende legt zur geltend gemachten Lieferung des Kontaktsockels „ Waffle Kelvin “ an die A … Inc. die teilgeschwärzte Ablichtung einer Rechnung vor :

**False Positives:**

- `A …` — partial — pred is substring of gold: `A … Inc.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Kelvin`(PER)
- `A … Inc.`(ORG)

**Example 6** (doc_id: `62911`) (sent_id: `62911`)


E4 : Modular Safety Controller System UE410 FLEXI . User Manual . S … AG - Industrial Safety Systems - Germany - All rightsreserved . 8011509 / ÄND / 06 - 05- 19 .

**False Positives:**

- `S …` — partial — pred is substring of gold: `S … AG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S … AG`(ORG)
- `Germany`(LOC)

**Example 7** (doc_id: `63141`) (sent_id: `63141`)


Nach Auffassung der Klägerin zu 2. ist die Priorität bereits formalrechtlich nicht wirksam in Anspruch genommen , da die Prioritätsanmeldung US 60/132036 bzw. das aus dieser Anmeldung folgende Prioritätsrecht nicht nachweislich innerhalb des Prioritätsjahres von den Erfindern als Voranmelder auf die L … LLC als Nachanmelderin übergegangen sei .

**False Positives:**

- `L …` — partial — pred is substring of gold: `L … LLC`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `L … LLC`(ORG)

**Example 8** (doc_id: `63599`) (sent_id: `63599`)


E4c : Rechnung der S … GmbH , Nummer 9090503004 vom 19. 10. 2016 an die P … GmbH , D … . in M … .

**False Positives:**

- `S …` — partial — pred is substring of gold: `S … GmbH`
- `P …` — partial — pred is substring of gold: `P … GmbH , D …`
- `D …` — partial — pred is substring of gold: `P … GmbH , D …`
- `M …` — type mismatch — same span as gold: `M …`

> overlaps gold: 4  |  likely missing annotation: 0

**Gold Entities:**

- `S … GmbH`(ORG)
- `P … GmbH , D …`(ORG)
- `M …`(LOC)

**Example 9** (doc_id: `63826`) (sent_id: `63826`)


Die Widerspruchsmarke sei namentlich im Blick auf den Vertrieb der „ Arrow “ -Kollektion bekannt , den die Widersprechende zusammen mit dem Lizenznehmer S … betreibe .

**False Positives:**

- `S …` — type mismatch — same span as gold: `S …`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S …`(ORG)

**Example 10** (doc_id: `65549`) (sent_id: `65549`)


Zur geltend gemachten Lieferung des Kontaktsockels „ Waffle Kelvin “ an die Firma A … Inc. hat die Einsprechende ebenfalls Zeugenbeweis ange- boten und verschiedene Dokumente eingereicht :

**False Positives:**

- `A …` — partial — pred is substring of gold: `A … Inc.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Kelvin`(PER)
- `A … Inc.`(ORG)

**Example 11** (doc_id: `66640`) (sent_id: `66640`)


Der Beschwerdeführer stehe im Verdacht , am 22. Oktober 2014 auf der Webseite " D … " den folgenden Kommentar veröffentlicht zu haben :

**False Positives:**

- `D …` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Full names with titles (corrected)`

**F1:** 0.041 | **Precision:** 0.389 | **Recall:** 0.022  

**Format:** `regex`  
**Rule ID:** `e697a848`  
**Description:**
Captures full names preceded by titles like Dr., Prof., ensuring the entire name is captured including middle initials, excluding the title itself.

**Content:**
```
(?:Dr\.?\s+|Prof\.?\s+|Dipl\.-Ing\.\s+|Dipl\.-Psych\.\s+|Dipl\.-Ing\.\s+Univ\.\s+)([A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+[A-Z]\s*\.)?(?:\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)*)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.389 | 0.022 | 0.041 | 18 | 7 | 11 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 7 | 11 | 251 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `61554`) (sent_id: `61554`)


Dr. Achilles

| Predicted | Gold |
|---|---|
| `Achilles` | `Achilles` |

**Example 1** (doc_id: `61787`) (sent_id: `61787`)


In der Beschwerdesache betreffend die Marke 30 2009 056 266 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 14. September 2017 unter Mitwirkung des Vorsitzenden Richters Knoll sowie der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Nielsen` | `Nielsen` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Kriener` (PER)

**Example 2** (doc_id: `64394`) (sent_id: `64394`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2013 053 470.0 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 9. November 2017 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Nielsen` | `Nielsen` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Kriener` (PER)

**Example 3** (doc_id: `64527`) (sent_id: `64527`)


In der Beschwerdesache betreffend die Marke 30 2011 035 856 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Meiser` | `Meiser` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Hacker` (PER)
- `Merzbach` (PER)

**Example 4** (doc_id: `64925`) (sent_id: `64925`)


In der Beschwerdesache betreffend die Marke 30 2010 003 649 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Meiser` | `Meiser` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Hacker` (PER)
- `Merzbach` (PER)

**Example 5** (doc_id: `64967`) (sent_id: `64967`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2015 034 000.6 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 11. Januar 2018 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Nielsen` | `Nielsen` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Kriener` (PER)

**Example 6** (doc_id: `65500`) (sent_id: `65500`)


In der Beschwerdesache betreffend die Designanmeldung .... ( hier : Antrag auf Verfahrenskostenhilfe ) hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Meiser` | `Meiser` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Hacker` (PER)
- `Merzbach` (PER)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `61830`) (sent_id: `61830`)


Ferner hat die Einsprechende das „ Gutachten zum Sicherheitsschalter CES der Firma E … “ des Herrn Prof. Dr. - Ing. G … vom 15. September 2014 vorgelegt .

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `E …`(ORG)
- `G …`(PER)

**Example 1** (doc_id: `61864`) (sent_id: `61864`)


Zu dem Sachverständigengutachten des Prof. Dr. T. sowie dessen ergänzender Stellungnahme hat die Beklagte mitgeteilt , dass sie diesem nach prüfärztlicher Stellungnahme nicht folgen könne ; der Einschätzung des Dr. S. hat sie sich hingegen angeschlossen .

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `T.`(PER)
- `S.`(PER)

**Example 2** (doc_id: `63653`) (sent_id: `63653`)


Als sachkundige Auskunftsperson hat sich in der mündlichen Verhandlung Prof. Dr. Klaus-Dieter Drüen geäußert .

**False Positives:**

- `Dr` — similar text (different position): `Klaus-Dieter Drüen`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Klaus-Dieter Drüen`(PER)

**Example 3** (doc_id: `63754`) (sent_id: `63754`)


Soweit das LSG Ausführungen von Prof. Dr. S. referiert , dass sich im Zusammenhang mit den Pneumonien auch immer wieder Septitiden entwickelt hätten , macht das LSG nicht deutlich , dass es diese als festgestellt ansieht , und erst recht nicht , von welchem Begriff oder Schweregrad der Sepsis es im Anschluss an Prof. Dr. S. aufgrund welcher Befunde dabei ausgeht ( ältere Klassifizierungen : Systemisches inflammatorisches Response-Syndrom < SIRS > , Sepsis , schwere Sepsis und septischer Schock ; zur darauf aufbauenden DRG-Kodierung vgl Hanser in Zaiß , DRG : Verschlüsseln leicht gemacht , 14. Aufl 2016 , S 101 ff ; neuere Klassifizierungen : Sepsis-related organ failure assessment score < SOFA > , insbesondere für Intensivstationen , und quickSOFA außerhalb von Intensivstationen ) .

**False Positives:**

- `Dr` — no gold match — likely missing annotation
- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `S.`(PER)
- `S.`(PER)

**Example 4** (doc_id: `63885`) (sent_id: `63885`)


Nach den Ausführungen des im Verfahren von Amts wegen gehörten Sachverständigen Prof. Dr. T. hätten die vom Kläger vorgetragenen Gewalterfahrungen während seiner Heimaufenthalte ab April 1959 nicht die entscheidende Traumatisierung dargestellt .

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `T.`(PER)

**Example 5** (doc_id: `64291`) (sent_id: `64291`)


Ferner hat sie beantragt , Prof. Dr. B. , Zentralinstitut für seelische Gesundheit M. , Institut für Psychiatrie und Psychosomatische Psychotherapie , als sachverständigen Zeugen zu vernehmen sowie den Sachverständigen Prof. Dr. S. zu den mit Schriftsatz vom 21. 3. 2017 erhobenen Einwendungen anzuhören .

**False Positives:**

- `Dr` — no gold match — likely missing annotation
- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `B.`(PER)
- `Zentralinstitut für seelische Gesundheit M. , Institut für Psychiatrie und Psychosomatische Psychotherapie`(ORG)
- `S.`(PER)

**Example 6** (doc_id: `64527`) (sent_id: `64527`)


In der Beschwerdesache betreffend die Marke 30 2011 035 856 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 7** (doc_id: `64925`) (sent_id: `64925`)


In der Beschwerdesache betreffend die Marke 30 2010 003 649 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 8** (doc_id: `65500`) (sent_id: `65500`)


In der Beschwerdesache betreffend die Designanmeldung .... ( hier : Antrag auf Verfahrenskostenhilfe ) hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

</details>

---

## `Anonymized single letters`

**F1:** 0.030 | **Precision:** 0.385 | **Recall:** 0.015  

**Format:** `regex`  
**Rule ID:** `70843e1c`  
**Description:**
Captures single letter anonymized names (e.g., 'A', 'B') ONLY when preceded by a role indicator or title, preventing false positives on section markers like 'I.', 'II.' or abbreviations.

**Content:**
```
(?:Herr\s+|Herrn\s+|Dr\.\s+|Prof\.\s+|Richter\s+|Richterin\s+|Vorsitzender\s+Richter\s+|Angeklagte\s+|Angeklagten\s+|Kl\u00e4ger\s+|Zeuge\s+|Zeugin\s+|Gesch\u00e4ftsf\u00fchrer\s+|Rechtsanwalt\s+|Rechtsanw\u00e4ltin\s+|Beteiligte\s+|Beteiligter\s+)([A-Z])(?=\s*(?:,|\.|\)|\]|\s|$))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.385 | 0.015 | 0.030 | 13 | 5 | 8 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 5 | 8 | 264 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `62066`) (sent_id: `62066`)


Die Ausgestaltung dieser Rahmenvereinbarung entspricht ausdrücklich dem von Herrn N geäußerten Wunsch , da Herr N aus pers. Gründen nicht öfter zur Verfügung stehen kann .

| Predicted | Gold |
|---|---|
| `N` | `N` |
| `N` | `N` |

**Example 1** (doc_id: `63744`) (sent_id: `63744`)


Über den Betriebs ( teil- ) übergang sowie den Übergang seines Arbeitsverhältnisses auf die D P T S GmbH wurde der Kläger durch ein Unterrichtungsschreiben vom 14. November 2005 informiert , das auf dem Briefkopf der Beklagten erstellt und für diese von deren Abteilungsleiter Personal / Service R und für die D P T S GmbH von deren Geschäftsführer C unterzeichnet war .

| Predicted | Gold |
|---|---|
| `C` | `C` |

**Missed by this rule (FN):**

- `D P T S GmbH` (ORG)
- `R` (PER)
- `D P T S GmbH` (ORG)

**Example 2** (doc_id: `65215`) (sent_id: `65215`)


Auf der Grundlage des § 27 Abs. 2 MTV-DP AG hatte die Beklagte Herrn N zunächst für die Zeit vom 1. Oktober 2013 bis zum 1. Oktober 2014 Sonderurlaub gewährt und diesen bis zum 1. Oktober 2015 verlängert .

| Predicted | Gold |
|---|---|
| `N` | `N` |

**Missed by this rule (FN):**

- `§ 27 Abs. 2 MTV-DP AG` (REG)

**Example 3** (doc_id: `65437`) (sent_id: `65437`)


Dabei hat sich der Senat auf zwei Erwägungen gestützt : Einerseits habe es die Beklagte rechtswidrig unterlassen , einen Beurteilungsbeitrag von dem erkrankten und inzwischen im Ruhestand befindlichen ehemaligen Abteilungsleiter X ( Herr Dr. A ) einzuholen .

| Predicted | Gold |
|---|---|
| `A` | `A` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `61243`) (sent_id: `61243`)


Doch auch wenn die Taten 21 bis 23 und 33 der Urteilsgründe sich damit im Hinblick auf das Marihuana als selbständige Umsatzgeschäfte darstellen , fallen die darauf bezogenen Handlungen des Angeklagten mit denjenigen zusammen , die dem Absatz des zugleich in diesen Fällen gehandelten Amphetamins dienten , hinsichtlich dessen aufgrund des einheitlichen Erwerbs im Fall 27 der Urteilsgründe von einer Bewertungseinheit und damit von einer Tat im Rechtssinne auszugehen ist : Im Fall 21 der Urteilsgründe nahm der Angeklagte die Bestellung beider Betäubungsmittel einheitlich entgegen , in den Fällen 22 und 23 der Urteilsgründe lieferte er sie auch gleichzeitig an die Angeklagte A. A. .

**False Positives:**

- `A` — similar text (different position): `A. A.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `A. A.`(PER)

**Example 1** (doc_id: `61306`) (sent_id: `61306`)


Das Landgericht hat den Angeklagten A. wegen Diebstahls mit Waffen in Tateinheit mit „ fahrlässiger “ Gefährdung des Straßenverkehrs , vorsätzlichem Fahren ohne Fahrerlaubnis und fahrlässiger Körperverletzung sowie wegen unerlaubten Entfernens vom Unfallort und vorsätzlicher Körperverletzung zu der Gesamtfreiheitsstrafe von zwei Jahren und vier Monaten verurteilt ; ferner hat es die Verwaltungsbehörde angewiesen , dem Angeklagten vor Ablauf einer Frist von drei Jahren keine Fahrerlaubnis zu erteilen .

**False Positives:**

- `A` — similar text (different position): `A.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `A.`(PER)

**Example 2** (doc_id: `62293`) (sent_id: `62293`)


Das Landgericht hat den Angeklagten T. D. wegen schweren sexuellen Missbrauchs eines Kindes in Tateinheit mit sexuellem Missbrauch eines Schutzbefohlenen in zwei Fällen sowie schweren sexuellen Missbrauchs einer widerstandsunfähigen Person in fünf Fällen zu einer Gesamtfreiheitsstrafe von vier Jahren und drei Monaten verurteilt .

**False Positives:**

- `T` — similar text (different position): `T. D.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `T. D.`(PER)

**Example 3** (doc_id: `63047`) (sent_id: `63047`)


1. Die Verurteilung des Angeklagten T. D. wegen schweren sexuellen Missbrauchs einer widerstandsunfähigen Person in den Fällen III. 3 bis 7 der Urteilsgründe nach dem zur Tatzeit geltenden § 179 Abs. 1 Nr. 1 , Abs. 5 Nr. 1 StGB in der bis zum 9. November 2016 geltenden Fassung hält revisionsrechtlicher Überprüfung nicht stand , weil die Urteilsgründe eine Widerstandsunfähigkeit des Nebenklägers nicht belegen .

**False Positives:**

- `T` — similar text (different position): `T. D.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `T. D.`(PER)
- `§ 179 Abs. 1 Nr. 1 , Abs. 5 Nr. 1 StGB`(NRM)

**Example 4** (doc_id: `64409`) (sent_id: `64409`)


Da der Zeuge W ... diesen Angaben glaubte , erwarb er für die Firma die beiden Geräte und zahlte den Kaufpreis .

**False Positives:**

- `W` — partial — pred is substring of gold: `W ...`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `W ...`(PER)

**Example 5** (doc_id: `65248`) (sent_id: `65248`)


3. Im Umfang der Aufhebung wird die Sache zu neuer Verhandlung und Entscheidung , auch über die Kosten des Rechtsmittels des Angeklagten A. , an eine andere Strafkammer des Landgerichts zurückverwiesen .

**False Positives:**

- `A` — similar text (different position): `A.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `A.`(PER)

**Example 6** (doc_id: `65305`) (sent_id: `65305`)


c ) Damit erledigt sich der Antrag auf Gewährung von Prozesskostenhilfe und Beiordnung von Rechtsanwältin H … für das Verfahren auf Erlass einer einstweiligen Anordnung .

**False Positives:**

- `H` — similar text (different position): `H …`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `H …`(PER)

**Example 7** (doc_id: `66540`) (sent_id: `66540`)


„ Ausgehend vom nach Teileinstellungen noch angeklagten Sachverhalt nach Maßgabe des Hinweisbeschlusses vom 6. Hauptverhandlungstag “ wird hinsichtlich der Angeklagten K. „ bei insoweit glaubhaftem Geständnis und kooperativem Verhalten “ eine Verurteilung zu einer Gesamtfreiheitsstrafe von mindestens neun Monaten bis zu einem Jahr und drei Monaten , deren Vollstreckung zur Bewährung ausgesetzt wird , erfolgen .

**False Positives:**

- `K` — similar text (different position): `K.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K.`(PER)

</details>

---

## `Initials with surname`

**F1:** 0.054 | **Precision:** 0.098 | **Recall:** 0.037  

**Format:** `regex`  
**Rule ID:** `40a4571f`  
**Description:**
Captures patterns like 'K. Schmidt' or 'M. P. Borom' where an initial (with dot) is followed by a surname.

**Content:**
```
\b([A-Z]\.)\s+([A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:-[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)?)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.098 | 0.037 | 0.054 | 122 | 12 | 110 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 12 | 110 | 312 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60446`) (sent_id: `60446`)


K. Schmidt

| Predicted | Gold |
|---|---|
| `K. Schmidt` | `K. Schmidt` |

**Example 1** (doc_id: `62502`) (sent_id: `62502`)


W. Reinfelder

| Predicted | Gold |
|---|---|
| `W. Reinfelder` | `W. Reinfelder` |

**Example 2** (doc_id: `62684`) (sent_id: `62684`)


M. Rennpferdt

| Predicted | Gold |
|---|---|
| `M. Rennpferdt` | `M. Rennpferdt` |

**Example 3** (doc_id: `63862`) (sent_id: `63862`)


Die durch sie erlaubten Kollektivbestrafungen werden von den Behörden im Nordkaukasus bereits angewendet ( Österreichisches Bundesamt für Fremdenwesen und Asyl , Länderinformationsblatt der Staatendokumentation Russische Föderation , Gesamtaktualisierung am 1. Juni 2016 , S. 34 ; Schweizerische Flüchtlingshilfe / A. Schuster , Russland : Verfolgung von Verwandten dagestanischer Terrorverdächtiger ausserhalb Dagestans , Auskunft vom 25. Juli 2014 , S. 4 f. ) .

| Predicted | Gold |
|---|---|
| `A. Schuster` | `A. Schuster` |

**Missed by this rule (FN):**

- `Nordkaukasus` (LOC)
- `Österreichisches Bundesamt für Fremdenwesen und Asyl` (ORG)
- `Russische Föderation` (LOC)
- `Schweizerische Flüchtlingshilfe` (ORG)
- `Russland` (LOC)
- `Dagestans` (LOC)

**Example 4** (doc_id: `63901`) (sent_id: `63901`)


K. Schmidt

| Predicted | Gold |
|---|---|
| `K. Schmidt` | `K. Schmidt` |

**Example 5** (doc_id: `63927`) (sent_id: `63927`)


M. Trümner

| Predicted | Gold |
|---|---|
| `M. Trümner` | `M. Trümner` |

**Example 6** (doc_id: `64317`) (sent_id: `64317`)


K. Schmidt

| Predicted | Gold |
|---|---|
| `K. Schmidt` | `K. Schmidt` |

**Example 7** (doc_id: `64439`) (sent_id: `64439`)


Vor diesem Hintergrund vermag der Senat auch aus der Auskunft der Schweizerischen Flüchtlingshilfe vom 25. Juli 2014 ( A. Schuster , Russland : Verfolgung von Verwandten dagestanischer Terrorverdächtiger ausserhalb Dagestans , S. 3 f. ) nicht abzuleiten , dass dem Kläger in der Russischen Föderation außerhalb des Nordkaukasus mit beachtlicher Wahrscheinlichkeit eine Art. 3 EMRK zuwiderlaufende Behandlung drohen würde .

| Predicted | Gold |
|---|---|
| `A. Schuster` | `A. Schuster` |

**Missed by this rule (FN):**

- `Schweizerischen Flüchtlingshilfe` (ORG)
- `Russland` (LOC)
- `Dagestans` (LOC)
- `Russischen Föderation` (LOC)
- `Nordkaukasus` (LOC)
- `Art. 3 EMRK` (NRM)

**Example 8** (doc_id: `64693`) (sent_id: `64693`)


K. Schmidt

| Predicted | Gold |
|---|---|
| `K. Schmidt` | `K. Schmidt` |

**Example 9** (doc_id: `64861`) (sent_id: `64861`)


J. Ratayczak

| Predicted | Gold |
|---|---|
| `J. Ratayczak` | `J. Ratayczak` |

**Example 10** (doc_id: `65286`) (sent_id: `65286`)


D14 J. Deubener et al. , " Induction time analysis of nucleation and crystal growth in di- and metasilicate glasses " , Journal of Non-Crystalline Solids 1993 , 163 , Seiten 1 bis 12 ,

| Predicted | Gold |
|---|---|
| `J. Deubener` | `J. Deubener` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60068`) (sent_id: `60068`)


I. Die Klägerin und Revisionsbeklagte ( Klägerin ) , eine GmbH , war in den Jahren 2009 bis 2012 ( Streitjahre ) als Reiseveranstalterin unternehmerisch tätig .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `60117`) (sent_id: `60117`)


I. Die Befristungskontrollklage ist unbegründet .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `60238`) (sent_id: `60238`)


V. Die Klage ist nicht abweisungsreif ( vgl. § 563 Abs. 3 ZPO ) .

**False Positives:**

- `V. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 563 Abs. 3 ZPO`(NRM)

**Example 3** (doc_id: `60477`) (sent_id: `60477`)


I. Die Würdigung des Landesarbeitsgerichts , das beklagte Königreich sei im vorliegenden Rechtsstreit grundsätzlich nicht der deutschen Gerichtsbarkeit unterworfen , sondern genieße - sollte es darauf nicht verzichtet haben - Staatenimmunität , ist revisionsrechtlich nicht zu beanstanden .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `60549`) (sent_id: `60549`)


Die zivilgerichtliche Rechtsprechung wende im Rahmen von § 315 BGB materielle , die Äquivalenz der Leistungen betreffende Kriterien an , die in den Bestimmungen der Richtlinie 2001 / 14 / EG nicht vorgesehen seien ( a. a. O. Rn. 72 ) .

**False Positives:**

- `O. Rn` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 315 BGB`(NRM)
- `Richtlinie 2001 / 14 / EG`(NRM)

**Example 5** (doc_id: `60609`) (sent_id: `60609`)


Daran gemessen war der Vertrag vom 30. März 1989 unabhängig davon , ob man ihn als - unzutreffend beurkundetes - mehrseitiges Rechtsgeschäft zwischen den Beigeladenen , den Eltern des Beigeladenen zu 2 und U. Sch. versteht oder ob man ihn als lediglich zwischen U. Sch. und den Beigeladenen geschlossenen Vertrag ansieht , der Redlichkeitsprüfung zugänglich .

**False Positives:**

- `U. Sch` — partial — pred is substring of gold: `U. Sch.`
- `U. Sch` — similar text (different position): `U. Sch.`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `U. Sch.`(PER)
- `U. Sch.`(PER)

**Example 6** (doc_id: `60693`) (sent_id: `60693`)


I. Die Antragsgegnerin und Beschwerdegegnerin ( im Folgenden : Antragsgegnerin ) war Inhaberin des am 4. Mai 2000 eingetragenen Gebrauchsmusters 298 20 129.1 ( Streitgebrauchsmuster ) mit der Bezeichnung „ … “ , das am 1. Dezember 2008 nach Erreichen der maximalen Schutzdauer erloschen war .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 7** (doc_id: `60767`) (sent_id: `60767`)


I. Mit dem angefochtenen Beschluss vom 15. Juli 2015 hat die Patentabteilung 1.25 des Deutschen Patent- und Markenamts das Patent DE 10 2008 017 350 mit der Bezeichnung „ Steuerung für Fahrmischer “ beschränkt aufrechterhalten .

**False Positives:**

- `I. Mit` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Patentabteilung 1.25 des Deutschen Patent- und Markenamts`(ORG)

**Example 8** (doc_id: `60783`) (sent_id: `60783`)


Dabei ist § 129 AO schon dann nicht anwendbar , wenn auch nur die ernsthafte Möglichkeit besteht , dass die Nichtbeachtung einer feststehenden Tatsache auf einer fehlerhaften Tatsachenwürdigung oder einem sonstigen sachverhaltsbezogenen Denk- oder Überlegungsfehler gründet oder auf mangelnder Sachverhaltsaufklärung beruht ( ständige Rechtsprechung , z.B. Senatsbeschluss vom 28. Mai 2015 VI R 63/13 , BFH / NV 2015 , 1078 , m. w. N. ) .

**False Positives:**

- `B. Senatsbeschluss` — positional overlap with gold: `Senatsbeschluss vom 28. Mai 2015 VI R 63/13 , BFH / NV 2015 , 1078`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 129 AO`(NRM)
- `Senatsbeschluss vom 28. Mai 2015 VI R 63/13 , BFH / NV 2015 , 1078`(RS)

**Example 9** (doc_id: `60926`) (sent_id: `60926`)


I. Die Kläger und Beschwerdeführer ( Kläger ) werden zusammen veranlagt .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 10** (doc_id: `61070`) (sent_id: `61070`)


I. Auf die am 30. Mai 2012 beim Deutschen Patent- und Markenamt eingereichte Patentanmeldung ist die Erteilung des Patents 10 2012 104 673 mit der Bezeichnung „ Werkzeug , System und Verfahren zum Verschrauben von Schraubendruckfedern zu einer Schraubentellerfeder “ am 14. August 2013 veröffentlicht worden .

**False Positives:**

- `I. Auf` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschen Patent- und Markenamt`(ORG)

**Example 11** (doc_id: `61076`) (sent_id: `61076`)


D4 M. P. Borom et al. , „ Strength and Microstructure in Lithium Disilicate Glass-Ceramics “ , Journal of the American Ceramic Society , 1975 , 58 , Seiten 385 bis 391 ,

**False Positives:**

- `P. Borom` — partial — pred is substring of gold: `M. P. Borom`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `M. P. Borom`(PER)

**Example 12** (doc_id: `61141`) (sent_id: `61141`)


In einer Auswerteeinheit würden die von einer externen Beschaltung – Signalgebern , wie z.B. Not-Aus-Tastern , Seilzugschaltern , Magnetschaltern , Positionsschaltern – stammenden Signale nach sicherheitstechnischen Vorschriften erfasst und verarbeitet .

**False Positives:**

- `B. Not-Aus` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 13** (doc_id: `61218`) (sent_id: `61218`)


C. Danach ist § 40 Abs. 1a LFGB insoweit mit Art. 12 Abs. 1 GG unvereinbar , als die Information der Öffentlichkeit nicht gesetzlich befristet ist .

**False Positives:**

- `C. Danach` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 40 Abs. 1a LFGB`(NRM)
- `Art. 12 Abs. 1 GG`(NRM)

**Example 14** (doc_id: `61272`) (sent_id: `61272`)


I. Nach § 72 Abs. 5 ArbGG iVm. § 551 Abs. 1 ZPO muss der Revisionskläger die Revision begründen .

**False Positives:**

- `I. Nach` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 72 Abs. 5 ArbGG`(NRM)
- `§ 551 Abs. 1 ZPO`(NRM)

**Example 15** (doc_id: `61319`) (sent_id: `61319`)


I. Der Feststellungsantrag ist zulässig .

**False Positives:**

- `I. Der` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 16** (doc_id: `61342`) (sent_id: `61342`)


I. Die vorliegende Patentanmeldung wurde am 26. Januar 2012 beim Deutschen Patent- und Markenamt eingereicht .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschen Patent- und Markenamt`(ORG)

**Example 17** (doc_id: `61353`) (sent_id: `61353`)


I. Die Anmelderin hat am 3. Januar 2013 beim Deutschen Patent- und Markenamt beantragt , die Bezeichnung A-ÖFFNER für die nachgenannten Waren und Dienstleistungen als Wortmarke in das Markenregister einzutragen :

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschen Patent- und Markenamt`(ORG)
- `A-ÖFFNER`(ORG)

**Example 18** (doc_id: `61516`) (sent_id: `61516`)


I. Der Kläger und Revisionskläger ( Kläger ) war in den Streitjahren ( 1995 bis 1997 ) u. a. als Steuerberater in einer Einzelkanzlei tätig .

**False Positives:**

- `I. Der` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `61557`) (sent_id: `61557`)


B. Die zulässige Rechtsbeschwerde des Betriebsrats ist unbegründet .

**False Positives:**

- `B. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 20** (doc_id: `61631`) (sent_id: `61631`)


b ) Beschäftigungsort i. S. des § 9 Abs. 1 Satz 3 Nr. 5 Satz 2 EStG ist der Ort der langfristig und dauerhaft angelegten Arbeitsstätte ( z.B. Senatsurteile vom 11. Mai 2005 VI R 7/02 , BFHE 209 , 502 , BStBl II 2005 , 782 , und VI R 34/04 , BFHE 209 , 527 , BStBl II 2005 , 793 , sowie vom 19. September 2012 VI R 78/10 , BFHE 239 , 80 , BStBl II 2013 , 284 ) .

**False Positives:**

- `B. Senatsurteile` — positional overlap with gold: `Senatsurteile vom 11. Mai 2005 VI R 7/02 , BFHE 209 , 502 , BStBl II 2005 , 782`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 9 Abs. 1 Satz 3 Nr. 5 Satz 2 EStG`(NRM)
- `Senatsurteile vom 11. Mai 2005 VI R 7/02 , BFHE 209 , 502 , BStBl II 2005 , 782`(RS)
- `VI R 34/04 , BFHE 209 , 527 , BStBl II 2005 , 793`(RS)
- `vom 19. September 2012 VI R 78/10 , BFHE 239 , 80 , BStBl II 2013 , 284`(RS)

**Example 21** (doc_id: `61784`) (sent_id: `61784`)


Dies ist zunächst dann der Fall , wenn das eingetragene Design Gestaltungen zum Gegenstand hat , bei denen es sich nicht um ein Erzeugnis im Sinne von § 1 Nr. 2 DesignG , d. h. um einen industriellen oder handwerklichen Gegenstand , bzw. um ein komplexes Erzeugnis im Sinne von § 1 Nr. 3 DesignG handelt , wie es z.B. bei anorganischen und organischen Naturprodukten , Menschen und Tieren , Verfahren und anderen Nichterzeugnissen aufgrund unkonkreter Gestalt , fehlender Sichtbarkeit oder auch einer dem Charakter eines ganzen Erzeugnisses widersprechenden Kombination von Gegenständen wie z.B. Backware und Uhr der Fall sein kann ( vgl. Eichmann / v. Falckenstein / Kühne , Designgesetz , 5. Aufl. , § 18 Rn. 2 ) .

**False Positives:**

- `B. Backware` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 1 Nr. 2 DesignG`(NRM)
- `§ 1 Nr. 3 DesignG`(NRM)
- `Eichmann / v. Falckenstein / Kühne , Designgesetz , 5. Aufl. , § 18 Rn. 2`(LIT)

**Example 22** (doc_id: `61798`) (sent_id: `61798`)


Zur Zeit ist die Beigeladene aufgrund des Anstellungsvertrags vom 18. / 27. Oktober 2015 bei der S. Gesellschaft als " Administrative Direktorin " beschäftigt .

**False Positives:**

- `S. Gesellschaft` — type mismatch — same span as gold: `S. Gesellschaft`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S. Gesellschaft`(ORG)

**Example 23** (doc_id: `61825`) (sent_id: `61825`)


D3 M. P. Borom et al. , “ Strength and Microstructure in Lithium Disilicate Glass-Ceramics ” , Journal of the American Ceramic Society , 1975 , 58 , Seiten 385 bis 391

**False Positives:**

- `P. Borom` — partial — pred is substring of gold: `M. P. Borom`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `M. P. Borom`(PER)

**Example 24** (doc_id: `61893`) (sent_id: `61893`)


I. Die Bezeichnung MAM Munich Asset Management ist am 16. März 2015 zur Eintragung als Wortmarke in das beim Deutschen Patent- und Markenamt ( DPMA ) geführte Register für folgende Dienstleistungen der Klassen 35 , 36 und 42 angemeldet worden :

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `MAM Munich Asset Management`(ORG)
- `Deutschen Patent- und Markenamt`(ORG)
- `DPMA`(ORG)

**Example 25** (doc_id: `61932`) (sent_id: `61932`)


V. Die Kostenentscheidung beruht auf § 90 Satz 2 EnWG , die Festsetzung des Gegenstandswerts auf § 50 Abs. 1 Satz 1 Nr. 2 GKG und § 3 ZPO .

**False Positives:**

- `V. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 90 Satz 2 EnWG`(NRM)
- `§ 50 Abs. 1 Satz 1 Nr. 2 GKG`(NRM)
- `§ 3 ZPO`(NRM)

**Example 26** (doc_id: `62040`) (sent_id: `62040`)


I. Die in § 33 Abs. 2 Satz 1 TV DRV KBS geregelte auflösende Bedingung gilt nicht nach §§ 21 , 17 Satz 2 TzBfG iVm. § 7 Halbs. 1 KSchG als wirksam und eingetreten .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 33 Abs. 2 Satz 1 TV DRV KBS`(REG)
- `§§ 21 , 17 Satz 2 TzBfG`(NRM)
- `§ 7 Halbs. 1 KSchG`(NRM)

**Example 27** (doc_id: `62109`) (sent_id: `62109`)


A. Die Richtervorlage betrifft die Frage , ob § 1906 Abs. 3 BGB in der Fassung des Gesetzes zur Regelung der betreuungsrechtlichen Einwilligung in eine ärztliche Zwangsmaßnahme vom 18. Februar 2013 ( BGBl I S. 266 ) mit Art. 3 Abs. 1 GG vereinbar ist , soweit er ärztliche Zwangsmaßnahmen außerhalb eines stationären Aufenthalts in einem Krankenhaus ausschließt .

**False Positives:**

- `A. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 1906 Abs. 3 BGB`(NRM)
- `Gesetzes zur Regelung der betreuungsrechtlichen Einwilligung in eine ärztliche Zwangsmaßnahme vom 18. Februar 2013 ( BGBl I S. 266 )`(NRM)
- `Art. 3 Abs. 1 GG`(NRM)

**Example 28** (doc_id: `62118`) (sent_id: `62118`)


I. Die von der Beschwerdeführerin als gleichheitswidrig beanstandeten Regelungen durch den von ihr mittelbar angegriffenen § 7 Satz 2 Nr. 2 GewStG sind verfassungsgemäß ; der Gesetzgeber bewegt sich mit dieser Neuregelung des Jahres 2002 im Rahmen seiner Gestaltungsbefugnis .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 7 Satz 2 Nr. 2 GewStG`(NRM)

**Example 29** (doc_id: `62176`) (sent_id: `62176`)


I. Der Kläger und Revisionsbeklagte ( Kläger ) war im Jahr 2011 ( Streitjahr ) Eigentümer des Grundstücks in X , Y-Straße ... ( Grundstück ) , das er bis März 2020 steuerpflichtig an die A ( Pächterin ) verpachtet hatte .

**False Positives:**

- `I. Der` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `X`(LOC)
- `Y-Straße ...`(LOC)
- `A`(PER)

</details>

---

</details>

---

<details>
<summary>💣 Least Precise Rules</summary>

## `Names with specific formatting (BRADLER , Christian)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `e2864f95`  
**Description:**
Captures names with specific formatting like 'BRADLER , Christian' or 'Surname , Firstname', excluding non-name patterns like 'BMW, Typ' or 'BGH, Beschluss'.

**Content:**
```
\b([A-Z][A-Z\u00e4\u00f6\u00fc\u00df]+\s*,\s*[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 93 | 0 | 93 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 93 | 324 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60069`) (sent_id: `60069`)


Demgegenüber kommt nach dem Regelungsplan des Gesetzgebers eine Zuständigkeit des Berufungsgerichts in Betracht , wenn es um den vom Berufungsgericht festgestellten oder festzustellenden Sachverhalt geht ( BGH , Urteil vom 13. Juli 1954 aaO ; ferner Beschluss vom 8. Juni 1973 - I ZR 25/72 , BGHZ 61 , 95 , 97 , 100 [ juris Rn. 9 f. ] ; BVerwG vom 7. Dezember 2015 - 6 PKH 10/15 , juris Rn. 12 ; Musielak in Musielak / Voith , ZPO 14. Aufl. , § 584 Rn. 2 ; MünchKomm-ZPO / Braun , 5. Aufl. , § 584 Rn. 1 ) .

**False Positives:**

- `BGH , Urteil` — partial — gold is substring of pred: `BGH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH`(ORG)
- `Beschluss vom 8. Juni 1973 - I ZR 25/72 , BGHZ 61 , 95 , 97 , 100 [ juris Rn. 9 f. ]`(RS)
- `BVerwG vom 7. Dezember 2015 - 6 PKH 10/15 , juris Rn. 12`(RS)
- `Musielak in Musielak / Voith , ZPO 14. Aufl. , § 584 Rn. 2`(LIT)
- `MünchKomm-ZPO / Braun , 5. Aufl. , § 584 Rn. 1`(LIT)

**Example 1** (doc_id: `60126`) (sent_id: `60126`)


Schließlich kann die Jugendkammer aber auch nach § 31 Abs. 3 Satz 1 i. V. m. § 105 Abs. 2 JGG von einer Einbeziehung absehen , wenn dies aus erzieherischen Gründen zweckmäßig ist ( vgl. BGH , Beschluss vom 23. November 1993 - 5 StR 573/93 , BGHSt 40 , 1 , 2 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 23. November 1993 - 5 StR 573/93 , BGHSt 40 , 1 , 2`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 31 Abs. 3 Satz 1 i. V. m. § 105 Abs. 2 JGG`(NRM)
- `BGH , Beschluss vom 23. November 1993 - 5 StR 573/93 , BGHSt 40 , 1 , 2`(RS)

**Example 2** (doc_id: `60279`) (sent_id: `60279`)


Eine Einstellung ohne Sicherheitsleistung kommt dabei nur in Betracht , wenn zusätzlich glaubhaft gemacht wird , dass der Schuldner zu einer Sicherheitsleistung nicht in der Lage ist ( vgl. BGH , Beschluss vom 8. Dezember 2009 - VIII ZR 305/09 , BGHZ 183 , 281 Rn. 6 ff ; Zöller / Herget , ZPO , 32. Aufl. , § 719 Rn. 8 ; MüKoZPO / Götz , 5. Aufl. , § 719 Rn. 15 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 8. Dezember 2009 - VIII ZR 305/09 , BGHZ 183 , 281 Rn. 6 ff`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 8. Dezember 2009 - VIII ZR 305/09 , BGHZ 183 , 281 Rn. 6 ff`(RS)
- `Zöller / Herget , ZPO , 32. Aufl. , § 719 Rn. 8`(LIT)
- `MüKoZPO / Götz , 5. Aufl. , § 719 Rn. 15`(LIT)

**Example 3** (doc_id: `60291`) (sent_id: `60291`)


Anders als in anderen von den Strafsenaten des Bundesgerichtshofs entschiedenen Fallkonstellationen ( vgl. BGH , Beschluss vom 18. Januar 2007 - 4 StR 394/06 , NStZ 2007 , 332 mit Anm. Kudlich JR 2007 , 381 ; Beschluss vom 6. September 2007 - 4 StR 227/07 , StraFo 2008 , 85 ; Beschluss vom 5. Juni 2007 - 4 StR 184/07 , StRR 2007 , 163 ; Beschluss vom 8. Juli 2008 - 3 StR 229/08 , NStZ-RR 2008 , 342 und Urteil vom 15. August 2007 - 5 StR 216/07 , NStZ-RR 2007 , 375 ) steht vorliegend aus Sicht eines objektiven Betrachters fest , dass es sich bei dem vom Angeklagten als Drohmittel verwendeten rund 50 Zentimeter langen Brecheisen aus Metall - ebenso wie bei einem Holzknüppel ( Senat , Beschluss vom 4. September 1998 - 2 StR 390/98 , NStZ-RR 1999 , 15 ) , einem Besenstiel ( BGH , Beschluss vom 20. Mai 1999 - 4 StR 168/99 , NStZ-RR 1999 , 355 ) , einem Schraubendreher ( BGH , Urteil vom 18. Februar 2010 - 3 StR 556/09 , NStZ 2011 , 158 ) oder einem abgesägten Metallstück in Form eines Winkeleisens ( Senat , Beschluss vom 21. November 2001 - 2 StR 400/01 , NStZ-RR 2002 , 108 , 109 ) - um einen objektiv gefährlichen Gegenstand handelt , weil es im Falle seines Einsatzes als Schlag- oder Stichwerkzeug ( vgl. BGH , Beschluss vom 27. März 2014 - 1 StR 24/14 , juris ) geeignet ist , erhebliche Verletzungen herbeizuführen .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 18. Januar 2007 - 4 StR 394/06 , NStZ 2007 , 332 mit Anm. Kudlich JR 2007 , 381`
- `BGH , Beschluss` — similar text (different position): `BGH , Beschluss vom 18. Januar 2007 - 4 StR 394/06 , NStZ 2007 , 332 mit Anm. Kudlich JR 2007 , 381`
- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 18. Februar 2010 - 3 StR 556/09 , NStZ 2011 , 158`
- `BGH , Beschluss` — similar text (different position): `BGH , Beschluss vom 18. Januar 2007 - 4 StR 394/06 , NStZ 2007 , 332 mit Anm. Kudlich JR 2007 , 381`

> overlaps gold: 4  |  likely missing annotation: 0

**Gold Entities:**

- `Strafsenaten`(ORG)
- `Bundesgerichtshofs`(ORG)
- `BGH , Beschluss vom 18. Januar 2007 - 4 StR 394/06 , NStZ 2007 , 332 mit Anm. Kudlich JR 2007 , 381`(RS)
- `Beschluss vom 6. September 2007 - 4 StR 227/07 , StraFo 2008 , 85`(RS)
- `Beschluss vom 5. Juni 2007 - 4 StR 184/07 , StRR 2007 , 163`(RS)
- `Beschluss vom 8. Juli 2008 - 3 StR 229/08 , NStZ-RR 2008 , 342`(RS)
- `Urteil vom 15. August 2007 - 5 StR 216/07 , NStZ-RR 2007 , 375`(RS)
- `Senat , Beschluss vom 4. September 1998 - 2 StR 390/98 , NStZ-RR 1999 , 15`(RS)
- `BGH , Beschluss vom 20. Mai 1999 - 4 StR 168/99 , NStZ-RR 1999 , 355`(RS)
- `BGH , Urteil vom 18. Februar 2010 - 3 StR 556/09 , NStZ 2011 , 158`(RS)
- `Senat , Beschluss vom 21. November 2001 - 2 StR 400/01 , NStZ-RR 2002 , 108 , 109 ) -`(RS)
- `BGH , Beschluss vom 27. März 2014 - 1 StR 24/14 , juris`(RS)

**Example 4** (doc_id: `60567`) (sent_id: `60567`)


Abzurechnen sei die geringer vergütete DRG B76C ( Anfälle , mehr als ein Belegungstag , ohne komplexe Diagnostik u. Therapie , mit schw. CC , Alter < 3 J. od. mit komplexer Diagnose od. m. äußerst schw. CC , Alter > 15 J. od. ohne äußerst schw. od. schw. CC , mit EEG , mit kompl. Diagnose ) .

**False Positives:**

- `CC , Alter` — no gold match — likely missing annotation
- `CC , Alter` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 5** (doc_id: `60622`) (sent_id: `60622`)


In solchen Fällen ist - auch unter Berücksichtigung des Grundsatzes in dubio pro reo - eine nicht auf einer ausreichenden Tatsachengrundlage beruhende und damit letztlich willkürliche Zusammenfassung mehrerer Umsatzgeschäfte zu einer Tat nicht geboten ( st. Rspr. ; vgl. etwa BGH , Beschluss vom 12. Januar 2016 - 3 StR 467/15 , juris Rn. 5 mwN ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 12. Januar 2016 - 3 StR 467/15 , juris Rn. 5`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 12. Januar 2016 - 3 StR 467/15 , juris Rn. 5`(RS)

**Example 6** (doc_id: `60662`) (sent_id: `60662`)


Die Beschwerdeführerin beantragt sinngemäß , den Beschluss des DPMA , Markenstelle für Klasse 41 , vom 26. November 2015 aufzuheben .

**False Positives:**

- `DPMA , Markenstelle` — partial — pred is substring of gold: `DPMA , Markenstelle für Klasse 41`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `DPMA , Markenstelle für Klasse 41`(ORG)

**Example 7** (doc_id: `60734`) (sent_id: `60734`)


Da sich die Klägerin das Verschulden ihres Prozessbevollmächtigten , der eine Berufungs- und Berufungsbegründungsschrift dem Gericht über einen Erklärungsboten zuleitet , nach § 85 Abs. 2 ZPO zurechnen lassen muss ( BGH , Beschluss vom 19. Juni 2007 - VI ZB 81/05 , FamRZ 2007 , 1638 Rn. 6 ) , war der Mangel der Form auch nicht unverschuldet .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 19. Juni 2007 - VI ZB 81/05 , FamRZ 2007 , 1638 Rn. 6`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 85 Abs. 2 ZPO`(NRM)
- `BGH , Beschluss vom 19. Juni 2007 - VI ZB 81/05 , FamRZ 2007 , 1638 Rn. 6`(RS)

**Example 8** (doc_id: `61086`) (sent_id: `61086`)


2. Der Senat kann die Revision durch Beschluss nach § 349 Abs. 2 StPO verwerfen ( vgl. BGH , Beschluss vom 23. Juli 2015 - 1 StR 279/15 , wistra 2015 , 476 mwN ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 23. Juli 2015 - 1 StR 279/15 , wistra 2015 , 476`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 349 Abs. 2 StPO`(NRM)
- `BGH , Beschluss vom 23. Juli 2015 - 1 StR 279/15 , wistra 2015 , 476`(RS)

**Example 9** (doc_id: `61389`) (sent_id: `61389`)


Dies ist in sachlich-rechtlicher Hinsicht der Fall , wenn die Beweiswürdigung widersprüchlich , unklar oder lückenhaft ist oder gegen Denkgesetze oder gesicherte Erfahrungssätze verstößt ( st. Rspr. ; vgl. nur BGH , Urteile vom 1. Februar 2017 - 2 StR 78/16 , NStZ-RR 2017 , 183 , 184 ; vom 13. Juli 2016 - 1 StR 94/16 , juris Rn. 9 und vom 14. September 2017 - 4 StR 45/17 , juris Rn. 7 ) .

**False Positives:**

- `BGH , Urteile` — partial — pred is substring of gold: `BGH , Urteile vom 1. Februar 2017 - 2 StR 78/16 , NStZ-RR 2017 , 183 , 184`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteile vom 1. Februar 2017 - 2 StR 78/16 , NStZ-RR 2017 , 183 , 184`(RS)
- `vom 13. Juli 2016 - 1 StR 94/16 , juris Rn. 9`(RS)
- `vom 14. September 2017 - 4 StR 45/17 , juris Rn. 7`(RS)

**Example 10** (doc_id: `61444`) (sent_id: `61444`)


Die Entscheidung des Beschwerdegerichts , die Rechtsbeschwerde zuzulassen , ist für den Senat nach § 574 Abs. 1 Satz 1 Nr. 2 , Abs. 3 Satz 2 ZPO unabhängig davon bindend , ob es die Voraussetzungen des § 574 Abs. 2 ZPO zutreffend beurteilt hat ( st. Rspr. ; vgl. BGH , Beschlüsse vom 7. Oktober 2008 - XI ZB 24/07 , NJW-RR 2009 , 425 Rn. 9 ; vom 8. Mai 2012 - VIII ZB 91/11 , WuM 2012 , 332 Rn. 3 mwN ) .

**False Positives:**

- `BGH , Beschlüsse` — partial — pred is substring of gold: `BGH , Beschlüsse vom 7. Oktober 2008 - XI ZB 24/07 , NJW-RR 2009 , 425 Rn. 9`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 574 Abs. 1 Satz 1 Nr. 2 , Abs. 3 Satz 2 ZPO`(NRM)
- `§ 574 Abs. 2 ZPO`(NRM)
- `BGH , Beschlüsse vom 7. Oktober 2008 - XI ZB 24/07 , NJW-RR 2009 , 425 Rn. 9`(RS)
- `vom 8. Mai 2012 - VIII ZB 91/11 , WuM 2012 , 332 Rn. 3`(RS)

**Example 11** (doc_id: `61464`) (sent_id: `61464`)


2. Der Senat weist jedoch - den zutreffenden Ausführungen des Generalbundesanwalts in seiner Antragsschrift folgend - darauf hin , dass das Mordmerkmal der Verdeckungsabsicht voraussetzt , dass der Täter die Tötungshandlung vornimmt oder die ihm zur Abwendung des Todeseintritts gebotene Handlung unterlässt , um dadurch eine andere Straftat zu verdecken ( BGH , Urteil vom 12. Dezember 2002 - 4 StR 297/02 , BGHR StGB § 211 Abs. 2 Verdeckung 15 ) .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 12. Dezember 2002 - 4 StR 297/02 , BGHR StGB § 211 Abs. 2 Verdeckung 15`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 12. Dezember 2002 - 4 StR 297/02 , BGHR StGB § 211 Abs. 2 Verdeckung 15`(RS)

**Example 12** (doc_id: `61612`) (sent_id: `61612`)


Bei der Auslegung von Prozesserklärungen ist der Grundsatz zu beachten , dass im Zweifel dasjenige gewollt ist , was nach den Maßstäben der Rechtsordnung vernünftig ist und der wohlverstandenen Interessenlage entspricht ( BGH , Urteil vom 2. Februar 2017 - VII ZR 261/14 , BauR 2017 , 915 Rn. 17 ; Urteil vom 1. August 2013 - VII ZR 268/11 , NJW 2014 , 155 Rn. 30 m. w. N. ) .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 2. Februar 2017 - VII ZR 261/14 , BauR 2017 , 915 Rn. 17`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 2. Februar 2017 - VII ZR 261/14 , BauR 2017 , 915 Rn. 17`(RS)
- `Urteil vom 1. August 2013 - VII ZR 268/11 , NJW 2014 , 155 Rn. 30`(RS)

**Example 13** (doc_id: `61650`) (sent_id: `61650`)


Vielmehr erlegt er Nicht-Konventionsstaaten grundsätzlich keine Standards der Europäischen Menschenrechtskonvention auf ( vgl. EGMR , Urteil vom 4. September 2014 - 140/10 - , Trabelsi / Belgien , Rn. 119 ) .

**False Positives:**

- `EGMR , Urteil` — partial — pred is substring of gold: `EGMR , Urteil vom 4. September 2014 - 140/10 - , Trabelsi / Belgien , Rn. 119`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europäischen Menschenrechtskonvention`(NRM)
- `EGMR , Urteil vom 4. September 2014 - 140/10 - , Trabelsi / Belgien , Rn. 119`(RS)

**Example 14** (doc_id: `61695`) (sent_id: `61695`)


Nicht erforderlich ist , dass die Verwaltung die fragliche Regelung statt durch Vertrag auch durch Verwaltungsakt regeln könnte ; neben derartigen subordinationsrechtlichen Verträgen ( vgl. § 54 Satz 2 VwVfG ) sind auch koordinationsrechtliche öffentlich-rechtliche Verträge denkbar , und nicht nur zwischen mehreren Verwaltungsträgern ( vgl. GmS-OGB , Beschluss vom 10. April 1986 - 1/85 - BVerwGE 74 , 368 ff. Rn. 11 ; BVerwG , Beschluss vom 17. November 2008 - 6 B 41.08 - Buchholz 442.066 § 75 TKG Nr. 1 ) .

**False Positives:**

- `OGB , Beschluss` — partial — pred is substring of gold: `GmS-OGB , Beschluss vom 10. April 1986 - 1/85 - BVerwGE 74 , 368 ff. Rn. 11`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 54 Satz 2 VwVfG`(NRM)
- `GmS-OGB , Beschluss vom 10. April 1986 - 1/85 - BVerwGE 74 , 368 ff. Rn. 11`(RS)
- `BVerwG , Beschluss vom 17. November 2008 - 6 B 41.08 - Buchholz 442.066 § 75 TKG Nr. 1`(RS)

**Example 15** (doc_id: `61737`) (sent_id: `61737`)


Für einen Unterstützungsstreik hat der Europäische Gerichtshof für Menschenrechte entschieden , dass dieser nicht den Kernbereich der Vereinigungsfreiheit betreffe , sondern lediglich einen Nebenaspekt darstelle und daher dem betroffenen Staat bei Einschränkungen ein weiterer Beurteilungsspielraum zuzugestehen sei ( vgl. EGMR , National Union of Rail , Maritime and Transport Workers v. United Kingdom , Urteil vom 8. April 2014 , Nr. 31045/10 , § 88 ) .

**False Positives:**

- `EGMR , National` — partial — pred is substring of gold: `EGMR , National Union of Rail , Maritime and Transport Workers v. United Kingdom , Urteil vom 8. April 2014 , Nr. 31045/10 , § 88`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europäische Gerichtshof für Menschenrechte`(ORG)
- `EGMR , National Union of Rail , Maritime and Transport Workers v. United Kingdom , Urteil vom 8. April 2014 , Nr. 31045/10 , § 88`(RS)

**Example 16** (doc_id: `61912`) (sent_id: `61912`)


Die gerichtliche Fürsorgepflicht greift nicht so weit , dass in Fällen , in denen die Unterschrift unter einem bestimmenden Schriftsatz mit dem Zusatz " i. A. " versehen ist , das Gericht innerhalb einer noch laufenden Frist auf den Mangel der Form hinweisen müsste ( BGH , Beschluss vom 20. Juni 2012 - IV ZB 18/11 , NJW-RR 2012 , 1269 Rn. 12 ff. ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 20. Juni 2012 - IV ZB 18/11 , NJW-RR 2012 , 1269 Rn. 12 ff.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 20. Juni 2012 - IV ZB 18/11 , NJW-RR 2012 , 1269 Rn. 12 ff.`(RS)

**Example 17** (doc_id: `61924`) (sent_id: `61924`)


Die einstweilige Einstellung der Zwangsvollstreckung kommt allerdings nicht in Betracht , wenn das Rechtsmittel der Nichtzulassungsbeschwerde keine Aussicht auf Erfolg hat ( vgl. BGH , Beschluss vom 23. März 2016 - VIII ZR 26/16 , juris Rn. 5 mwN ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 23. März 2016 - VIII ZR 26/16 , juris Rn. 5`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 23. März 2016 - VIII ZR 26/16 , juris Rn. 5`(RS)

**Example 18** (doc_id: `61933`) (sent_id: `61933`)


Die zum beendeten Versuch führende gedankliche Indifferenz des Täters gegenüber den von ihm bis dahin angestrebten oder doch zumindest in Kauf genommenen Konsequenzen ist eine innere Tatsache , die festgestellt werden muss , wozu es in der Regel einer zusammenfassenden Würdigung aller maßgeblichen objektiven Umstände bedarf ( BGH , Urteile vom 2. November 1994 , 2 StR 449/94 , aaO und vom 3. Juni 2008 - 1 StR 59/08 , NStZ 2009 , 264 ; Beschlüsse vom 22. Mai 2013 - 4 StR 170/13 , NStZ 2013 , 703 , 704 , und vom 27. Januar 2014 - 4 StR 565/13 , NStZ-RR 2014 , 202 f. ) .

**False Positives:**

- `BGH , Urteile` — partial — pred is substring of gold: `BGH , Urteile vom 2. November 1994 , 2 StR 449/94 , aaO`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteile vom 2. November 1994 , 2 StR 449/94 , aaO`(RS)
- `vom 3. Juni 2008 - 1 StR 59/08 , NStZ 2009 , 264`(RS)
- `Beschlüsse vom 22. Mai 2013 - 4 StR 170/13 , NStZ 2013 , 703 , 704`(RS)
- `vom 27. Januar 2014 - 4 StR 565/13 , NStZ-RR 2014 , 202 f.`(RS)

**Example 19** (doc_id: `61959`) (sent_id: `61959`)


Soweit die Anmelderin in Klasse 2 ferner „ Naturharze im Rohzustand “ beansprucht , stellen diese im Bereich von Lacken und ( Öl- ) Farben einen üblichen Inhaltsstoff dar , der auch als Zusatz im Malereibedarf in Betracht kommt ( vgl. BPatG PAVIS PROMA , Beschluss vom 23. April 2013 , 25 W ( pat ) 62/12 - Macchiato ) .

**False Positives:**

- `PROMA , Beschluss` — partial — pred is substring of gold: `BPatG PAVIS PROMA , Beschluss vom 23. April 2013 , 25 W ( pat ) 62/12 - Macchiato`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG PAVIS PROMA , Beschluss vom 23. April 2013 , 25 W ( pat ) 62/12 - Macchiato`(RS)

**Example 20** (doc_id: `61975`) (sent_id: `61975`)


Darüber , ob die Dauer eines Verfahrens angemessen ist , muss unter Berücksichtigung der Schwierigkeit des Falles , des Verhaltens des Beschwerdeführers und der zuständigen Behörden und Gerichte sowie der Bedeutung des Rechtsstreits für den Beschwerdeführer entschieden werden ( EGMR , Urteil vom 2. September 2010 , Nr. 46344/06 , Rumpf . / . Deutschland , Z. 41 , NJW 2010 , S. 3355 < 3356 > ; Urteil vom 21. Oktober 2010 , Nr. 43155/08 , Grumann . / . Deutschland , Z. 26 , NJW 2011 , S. 1055 < 1056 > ; BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 19 ) .

**False Positives:**

- `EGMR , Urteil` — partial — pred is substring of gold: `EGMR , Urteil vom 2. September 2010 , Nr. 46344/06 , Rumpf . / . Deutschland , Z. 41 , NJW 2010 , S. 3355 < 3356 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EGMR , Urteil vom 2. September 2010 , Nr. 46344/06 , Rumpf . / . Deutschland , Z. 41 , NJW 2010 , S. 3355 < 3356 >`(RS)
- `Urteil vom 21. Oktober 2010 , Nr. 43155/08 , Grumann . / . Deutschland , Z. 26 , NJW 2011 , S. 1055 < 1056 >`(RS)
- `BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 19`(RS)

**Example 21** (doc_id: `62085`) (sent_id: `62085`)


Innerhalb kürzester Zeit kann das schuldnerische Unternehmen durch den Verlust von Kunden , Lieferanten und Arbeitnehmern auseinander fallen ( vgl. BGH , Beschluss vom 27. Juli 2006 - IX ZB 204/04 - BGHZ 169 , 17 Rn. 12 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 27. Juli 2006 - IX ZB 204/04 - BGHZ 169 , 17 Rn. 12`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 27. Juli 2006 - IX ZB 204/04 - BGHZ 169 , 17 Rn. 12`(RS)

**Example 22** (doc_id: `62148`) (sent_id: `62148`)


a ) Ob der Rechtsmittelführer nur einzelne abtrennbare Teile eines Urteils angreifen will , ist eine Frage , die im Zweifelsfall im Wege der Auslegung seiner Rechtsmittelerklärungen zu beantworten ist ( vgl. BGH , Urteil vom 2. Februar 2017 - 4 StR 481/16 , NStZ-RR 2017 , 105 , 106 ; Beschluss vom 21. Oktober 1980 - 1 StR 262/80 , BGHSt 29 , 359 , 365 [ zu § 318 StPO ] ) .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 2. Februar 2017 - 4 StR 481/16 , NStZ-RR 2017 , 105 , 106`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 2. Februar 2017 - 4 StR 481/16 , NStZ-RR 2017 , 105 , 106`(RS)
- `Beschluss vom 21. Oktober 1980 - 1 StR 262/80 , BGHSt 29 , 359 , 365 [ zu § 318 StPO ]`(RS)

**Example 23** (doc_id: `62522`) (sent_id: `62522`)


Die Übertragung anfallender Arbeiten auf Büropersonal setzt voraus , dass es sich um geschultes , als zuverlässig erprobtes und sorgfältig überwachtes Personal handelt ( vgl. BGH , Beschluss vom 10. März 2011 - VII ZB 37/10 , NJW 2011 , 1597 Rn. 10 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 10. März 2011 - VII ZB 37/10 , NJW 2011 , 1597 Rn. 10`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 10. März 2011 - VII ZB 37/10 , NJW 2011 , 1597 Rn. 10`(RS)

**Example 24** (doc_id: `62527`) (sent_id: `62527`)


2. An der Besteuerung wird Deutschland durch das DBA-Kanada 2001 nicht gehindert ; der Senat hält insoweit an der in einem Verfahren des einstweiligen Rechtsschutzes bereits geäußerten Rechtsauffassung ( Senatsbeschluss vom 13. Dezember 2011 I B 159/11 , BFH / NV 2012 , 417 ) auch nach erneuter Prüfung fest ( dieser Rechtslage zustimmend z.B. Gosch in Kirchhof , EStG , 16. Aufl. , § 49 Rz 90 mit Fußn. 6 ; Hick in Schönfeld / Ditz , DBA , Art. 18 Rz 167 f. , 171 , 174 ; Blümich / Wied , § 49 EStG Rz 218 ; Kuhn in Herrmann / Heuer / Raupach , § 49 EStG Rz 1003 ; Kühnen , EFG 2016 , 578 ; Schober , EFG 2016 , 990 ; Ismer in Vogel / Lehner , DBA , 6. Aufl. , Art. 18 Rz 87 ; Holthaus , Internationale Wirtschaftsbriefe 2017 , 796 , 797 ; im Ergebnis ebenso die Verwaltungspraxis , s. Bayerisches Landesamt für Steuern , Verfügung vom 8. Juni 2011 , Internationales Steuerrecht 2011 , 776 ; a. A. W. Wassermeyer in Wassermeyer , DBA , Kanada Art. 18 Rz 70a ; Hagemann / Kahlenberg / Cloer , Betriebs-Berater 2017 , 2775 , 2785 f. ) .

**False Positives:**

- `DBA , Art` — partial — pred is substring of gold: `Hick in Schönfeld / Ditz , DBA , Art. 18 Rz 167 f. , 171 , 174`
- `DBA , Kanada` — partial — pred is substring of gold: `W. Wassermeyer in Wassermeyer , DBA , Kanada Art. 18 Rz 70a`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Deutschland`(LOC)
- `DBA-Kanada 2001`(NRM)
- `Senatsbeschluss vom 13. Dezember 2011 I B 159/11 , BFH / NV 2012 , 417`(RS)
- `Gosch in Kirchhof , EStG , 16. Aufl. , § 49 Rz 90 mit Fußn. 6`(LIT)
- `Hick in Schönfeld / Ditz , DBA , Art. 18 Rz 167 f. , 171 , 174`(LIT)
- `Blümich / Wied , § 49 EStG Rz 218`(LIT)
- `Kuhn in Herrmann / Heuer / Raupach , § 49 EStG Rz 1003`(LIT)
- `Kühnen , EFG 2016 , 578`(LIT)
- `Schober , EFG 2016 , 990`(LIT)
- `Ismer in Vogel / Lehner , DBA , 6. Aufl. , Art. 18 Rz 87`(LIT)
- `Holthaus , Internationale Wirtschaftsbriefe 2017 , 796 , 797`(LIT)
- `Bayerisches Landesamt für Steuern , Verfügung vom 8. Juni 2011 , Internationales Steuerrecht 2011 , 776`(LIT)
- `W. Wassermeyer in Wassermeyer , DBA , Kanada Art. 18 Rz 70a`(LIT)
- `Hagemann / Kahlenberg / Cloer , Betriebs-Berater 2017 , 2775 , 2785 f.`(LIT)

**Example 25** (doc_id: `62563`) (sent_id: `62563`)


Durch diese wird ein durchgreifender Rechtsfehler nicht aufgezeigt , so dass zu weitergehenden Ausführungen kein Anlass besteht ( st. Rspr. ; vgl. aus neuerer Zeit etwa BVerfG , Beschluss vom 30. Juni 2014 - 2 BvR 792/11 , NJW 2014 , 2563 , 2564 ; BGH , Beschluss vom 4. April 2016 - 1 StR 406/15 , NStZ-RR 2016 , 251 , 252 jeweils mwN ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 4. April 2016 - 1 StR 406/15 , NStZ-RR 2016 , 251 , 252`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG , Beschluss vom 30. Juni 2014 - 2 BvR 792/11 , NJW 2014 , 2563 , 2564`(RS)
- `BGH , Beschluss vom 4. April 2016 - 1 StR 406/15 , NStZ-RR 2016 , 251 , 252`(RS)

**Example 26** (doc_id: `62654`) (sent_id: `62654`)


Denn ohne Kenntnis der bereits teilweise in die Hauptverhandlung eingeführten Aussage kann er insbesondere sein Fragerecht gegenüber weiteren Zeugen grundsätzlich nicht sachgerecht ausüben ( vgl. BGH , Urteil vom 31. März 1992 aaO ; Beschluss vom 6. September 1989 - 3 StR 235/89 , BGHR StPO § 247 Satz 4 Unterrichtung 3 ) .

**False Positives:**

- `BGH , Urteil` — partial — gold is substring of pred: `BGH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH`(ORG)
- `Beschluss vom 6. September 1989 - 3 StR 235/89 , BGHR StPO § 247 Satz 4 Unterrichtung 3`(RS)

**Example 27** (doc_id: `62658`) (sent_id: `62658`)


Die 39. BImSchV dient unter anderem der Umsetzung der Richtlinie 2008 / 50 / EG des Europäischen Parlaments und des Rates vom 21. Mai 2008 über Luftqualität und saubere Luft für Europa ( ABl. L 152 S. 1 ) , in der die ab 1. Januar 2010 einzuhaltenden , vom Verordnungsgeber übernommenen Grenzwerte in Anhang XI , Abschnitt B , festgelegt sind .

**False Positives:**

- `XI , Abschnitt` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `39. BImSchV`(NRM)
- `Richtlinie 2008 / 50 / EG des Europäischen Parlaments und des Rates vom 21. Mai 2008 über Luftqualität und saubere Luft für Europa ( ABl. L 152 S. 1 )`(NRM)

**Example 28** (doc_id: `62668`) (sent_id: `62668`)


Dabei wurden nach dem Vortrag des Beschwerdeführers unter anderem ein PC , Laptops , zwei digitale Videokameras sowie mehrere USB-Datenträger sichergestellt .

**False Positives:**

- `PC , Laptops` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 29** (doc_id: `62804`) (sent_id: `62804`)


Jede spätere sachgrundlose Befristung sei gemäß § 14 Abs. 2 Satz 2 TzBfG unwirksam ( vgl. BAG , Urteil vom 6. November 2003 - 2 AZR 690/02 - , BAGE 108 , 269 < 274 > ; Urteil vom 13. Mai 2004 - 2 AZR 426/03 - , juris , Rn. 28 ; Beschluss vom 29. Juli 2009 - 7 AZN 368/09 - , www.bag.de , Rn. 2 ) .

**False Positives:**

- `BAG , Urteil` — partial — pred is substring of gold: `BAG , Urteil vom 6. November 2003 - 2 AZR 690/02 - , BAGE 108 , 269 < 274 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 14 Abs. 2 Satz 2 TzBfG`(NRM)
- `BAG , Urteil vom 6. November 2003 - 2 AZR 690/02 - , BAGE 108 , 269 < 274 >`(RS)
- `Urteil vom 13. Mai 2004 - 2 AZR 426/03 - , juris , Rn. 28`(RS)
- `Beschluss vom 29. Juli 2009 - 7 AZN 368/09 - , www.bag.de , Rn. 2`(RS)

</details>

---

## `Names after legal roles (Herr, Richter, etc.)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `64c0e08f`  
**Description:**
Captures names following legal role indicators like 'Herr', 'Herrn', 'Richter', 'Richterin', 'Angeklagte', 'Angeklagten', 'Kl\u00e4ger', 'Zeuge', ensuring no trailing space and handling titles correctly.

**Content:**
```
(?:Herr\s+|Herrn\s+|Richter\s+|Richterin\s+|Vorsitzender\s+Richter\s+|Angeklagte\s+|Angeklagten\s+|Kl\u00e4ger\s+|Zeuge\s+|Zeugin\s+)([A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+[A-Z]\s*\.)?(?:\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)*)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 7 | 0 | 7 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 7 | 287 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60944`) (sent_id: `60944`)


Nachdem er vom FA auf den fehlenden Nachweis einer regelmäßigen Summenziehung hingewiesen worden sei , habe der Kläger Erfassungsprotokolle beim FG eingereicht , die eine chronologische Auflistung der Geschäftsvorfälle ohne Angabe von Belegnummern enthalten hätten .

**False Positives:**

- `Erfassungsprotokolle` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `61328`) (sent_id: `61328`)


Am 18. November 2015 hat der Kläger Klage erhoben und zur Begründung geltend gemacht , der Jahresmittelgrenzwert für Stickstoffdioxid ( NO2 ) sei im Jahr 2013 an allen verkehrsnahen Messstationen zum Teil um mehr als das Doppelte überschritten worden und habe auch im Jahr 2014 an bestimmten Messstationen deutlich über den Grenzwerten gelegen .

**False Positives:**

- `Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `62721`) (sent_id: `62721`)


Seit 16. 2. 2013 ist der Kläger Pflichtmitglied der Landestierärztekammer Baden-Württemberg ( im Folgenden : Landestierärztekammer ) und Pflichtmitglied der Baden-Württembergischen Versorgungsanstalt für Ärzte , Zahnärzte und Tierärzte ( Beigeladene zu 1. ) , die ihren Teilnehmern und deren Hinterbliebenen Altersruhegeld , Ruhegeld bei Berufsunfähigkeit sowie eine Hinterbliebenenversorgung gewährt .

**False Positives:**

- `Pflichtmitglied` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Landestierärztekammer Baden-Württemberg`(ORG)
- `Baden-Württembergischen Versorgungsanstalt für Ärzte , Zahnärzte und Tierärzte`(ORG)

**Example 3** (doc_id: `63009`) (sent_id: `63009`)


Hiergegen hat der Kläger Klage zum SG erhoben , das durch Urteil vom 2. 10. 2012 den Bescheid der Beklagten vom 18. 4. 2011 in der Gestalt des Widerspruchsbescheids vom 8. 6. 2011 aufgehoben hat , weil das Grundstück des Klägers aufgrund der anzuwendenden Ausnahmevorschrift des § 123 Abs 2 SGB VII als versicherungsfreier Haus- und Ziergarten einzuordnen sei .

**False Positives:**

- `Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 123 Abs 2 SGB VII`(NRM)

**Example 4** (doc_id: `63492`) (sent_id: `63492`)


Mit der Revision rügen die Kläger Verletzung formellen und materiellen Rechts .

**False Positives:**

- `Verletzung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `66649`) (sent_id: `66649`)


M. hielt der Zeugin Mund und Nase zu , so dass sie nicht mehr schreien konnte .

**False Positives:**

- `Mund` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `M.`(PER)

**Example 6** (doc_id: `66655`) (sent_id: `66655`)


Am 18. November 2015 hat der Kläger Klage erhoben und zur Begründung geltend gemacht , die anhaltende Überschreitung der Grenzwerte sei ein Indiz dafür , dass die bisherigen Maßnahmen nicht geeignet seien , die Überschreitungszeiträume so kurz wie möglich zu halten .

**False Positives:**

- `Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Initials with surname (e.g., K. Schmidt)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `3b9a194a`  
**Description:**
Captures an initial followed by a capitalized surname, ensuring it's a name and not a sentence start or common verb.

**Content:**
```
(?<!^)(?<!\w)([A-Z]\s*\.)\s+([A-Z][a-zäöüß]+(?:-[A-Z][a-zäöüß]+)?)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 6 | 0 | 6 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 6 | 295 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60707`) (sent_id: `60707`)


c ) Werden die von einem Steuerpflichtigen erworbenen Gegenstände oder Dienstleistungen dagegen für die Zwecke steuerbefreiter Umsätze oder solcher Umsätze verwendet , die nicht vom Anwendungsbereich der Mehrwertsteuer erfasst werden , so kann es weder zur Erhebung der Steuer auf der folgenden Stufe noch zum Abzug der Vorsteuer kommen ( ständige Rechtsprechung , vgl. dazu z.B. EuGH-Urteile SKF , EU : C : 2009 : 665 , UR 2010 , 107 , Rz 59 , m. w. N. ; Iberdrola Inmobiliaria Real Estate Investments , EU : C : 2017 : 683 , DStR 2017 , 2044 , Rz 30 ) .

**False Positives:**

- `B. Eu` — positional overlap with gold: `EuGH-Urteile SKF , EU : C : 2009 : 665 , UR 2010 , 107 , Rz 59`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH-Urteile SKF , EU : C : 2009 : 665 , UR 2010 , 107 , Rz 59`(RS)
- `Iberdrola Inmobiliaria Real Estate Investments , EU : C : 2017 : 683 , DStR 2017 , 2044 , Rz 30`(RS)

**Example 1** (doc_id: `60757`) (sent_id: `60757`)


1. Unterscheidungskraft im Sinne von § 8 Abs. 2 Nr. 1 MarkenG ist die einem Zeichen innewohnende ( konkrete ) Eignung , vom Verkehr als Unterscheidungsmittel aufgefasst zu werden , das die von der Anmeldung erfassten Waren oder Dienstleistungen als von einem bestimmten Unternehmen stammend kennzeichnet und diese somit von denjenigen anderer Unternehmen unterscheidet ( vgl. z.B. EuGH GRUR 2012 , 610 ( Nr. 42 ) - Freixenet ; GRUR 2008 , 608 , 611 ( Nr. 66 ) - EUROHYPO ; BGH GRUR 2015 , 173 , 174 ( Nr. 15 ) - for you ; GRUR 2014 , 565 , 567 ( Nr. 12 ) - smartbook ; GRUR 2013 , 731 ( Nr. 11 ) - Kaleido ; GRUR 2012 , 1143 ( Nr. 7 ) - Starsat , jeweils m. w. N. ) .

**False Positives:**

- `B. Eu` — positional overlap with gold: `EuGH GRUR 2012 , 610 ( Nr. 42 ) - Freixenet`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 8 Abs. 2 Nr. 1 MarkenG`(NRM)
- `EuGH GRUR 2012 , 610 ( Nr. 42 ) - Freixenet`(RS)
- `GRUR 2008 , 608 , 611 ( Nr. 66 ) - EUROHYPO`(RS)
- `BGH GRUR 2015 , 173 , 174 ( Nr. 15 ) - for you`(RS)
- `GRUR 2014 , 565 , 567 ( Nr. 12 ) - smartbook`(RS)
- `GRUR 2013 , 731 ( Nr. 11 ) - Kaleido`(RS)
- `GRUR 2012 , 1143 ( Nr. 7 ) - Starsat`(RS)

**Example 2** (doc_id: `61664`) (sent_id: `61664`)


b ) Da das gemeinsame Mehrwertsteuersystem eine völlige Neutralität hinsichtlich der steuerlichen Belastung aller wirtschaftlichen Tätigkeiten unabhängig von ihrem Zweck und ihrem Ergebnis gewährleistet , sofern diese selbst der Mehrwertsteuer unterliegen ( ständige Rechtsprechung , vgl. dazu z.B. EuGH-Urteil Fini H vom 3. März 2005 C - 32/03 , EU : C : 2005 : 128 , UR 2005 , 443 , Rz 25 , m. w. N. ) , darf nicht willkürlich zwischen Ausgaben für die Zwecke eines Unternehmens vor der tatsächlichen Aufnahme seiner Tätigkeit sowie während dieser Tätigkeit und Ausgaben zum Zweck der Beendigung dieser Tätigkeit unterschieden werden ( ständige Rechtsprechung , vgl. dazu z.B. EuGH-Urteile Abbey National vom 22. Februar 2001 C - 408/98 , EU : C : 2001 : 110 , UR 2001 , 164 , Rz 35 ; Faxworld vom 29. April 2004 C - 137/02 , EU : C : 2004 : 267 , UR 2004 , 362 , Rz 39 ; Fini H , EU : C : 2005 : 128 , UR 2005 , 443 , Rz 23 ; ferner Wind Inovation 1 vom 9. November 2017 C - 552/16 , EU : C : 2017 : 849 , Höchstrichterliche Finanzrechtsprechung 2018 , 84 , Rz 45 ; jeweils m. w. N. ) .

**False Positives:**

- `B. Eu` — positional overlap with gold: `EuGH-Urteil Fini H vom 3. März 2005 C - 32/03 , EU : C : 2005 : 128 , UR 2005 , 443 , Rz 25`
- `B. Eu` — positional overlap with gold: `EuGH-Urteile Abbey National vom 22. Februar 2001 C - 408/98 , EU : C : 2001 : 110 , UR 2001 , 164 , Rz 35`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH-Urteil Fini H vom 3. März 2005 C - 32/03 , EU : C : 2005 : 128 , UR 2005 , 443 , Rz 25`(RS)
- `EuGH-Urteile Abbey National vom 22. Februar 2001 C - 408/98 , EU : C : 2001 : 110 , UR 2001 , 164 , Rz 35`(RS)
- `Faxworld vom 29. April 2004 C - 137/02 , EU : C : 2004 : 267 , UR 2004 , 362 , Rz 39`(RS)
- `Fini H , EU : C : 2005 : 128 , UR 2005 , 443 , Rz 23`(RS)
- `Wind Inovation 1 vom 9. November 2017 C - 552/16 , EU : C : 2017 : 849 , Höchstrichterliche Finanzrechtsprechung 2018 , 84 , Rz 45`(RS)

**Example 3** (doc_id: `61963`) (sent_id: `61963`)


Das Vorliegen einer Verwechslungsgefahr für das Publikum ist nach ständiger Rechtsprechung sowohl des Europäischen Gerichtshofes als auch des Bundesgerichtshofes unter Berücksichtigung aller relevanten Umstände des Einzelfalls zu beurteilen ( vgl. hierzu z.B. EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER ; GRUR 2010 , 1098 Rn. 44 – Calvin Klein / HABM ; BGH GRUR 2012 , 64 ; BGH GRUR 2012 , 64 Rn. 9 ; BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY ; GRUR 2012 , 1040 Rn. 25 – pjur / pure ; GRUR 2013 , 833 Rn. 30 – Culinaria / Villa Culinaria ) .

**False Positives:**

- `B. Eu` — positional overlap with gold: `EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europäischen Gerichtshofes`(ORG)
- `Bundesgerichtshofes`(ORG)
- `EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER`(RS)
- `GRUR 2010 , 1098 Rn. 44 – Calvin Klein / HABM`(RS)
- `BGH GRUR 2012 , 64`(RS)
- `BGH GRUR 2012 , 64 Rn. 9`(RS)
- `BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY`(RS)
- `GRUR 2012 , 1040 Rn. 25 – pjur / pure`(RS)
- `GRUR 2013 , 833 Rn. 30 – Culinaria / Villa Culinaria`(RS)

**Example 4** (doc_id: `66392`) (sent_id: `66392`)


1. Das Vorliegen einer Verwechslungsgefahr für das Publikum ist nach ständiger Rechtsprechung sowohl des Europäischen Gerichtshofes als auch des Bundesgerichtshofes unter Berücksichtigung aller relevanten Umstände des Einzelfalls zu beurteilen ( vgl. hierzu z.B. EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER ; GRUR 2010 , 1098 Rn. 44 – Calvin Klein / HABM ; BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY ; GRUR 2012 , 1040 Rn. 25 – pjur / pure ; GRUR 2013 , 833 Rn. 30 – Culinaria / Villa Culinaria ; GRUR 2016 , 382 Rn. 19 – BioGourmet ) .

**False Positives:**

- `B. Eu` — positional overlap with gold: `EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europäischen Gerichtshofes`(ORG)
- `Bundesgerichtshofes`(ORG)
- `EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER`(RS)
- `GRUR 2010 , 1098 Rn. 44 – Calvin Klein / HABM`(RS)
- `BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY`(RS)
- `GRUR 2012 , 1040 Rn. 25 – pjur / pure`(RS)
- `GRUR 2013 , 833 Rn. 30 – Culinaria / Villa Culinaria`(RS)
- `GRUR 2016 , 382 Rn. 19 – BioGourmet`(RS)

</details>

---

## `Names after 'Angeklagte' or 'Angeklagten' (corrected)`

**F1:** 0.012 | **Precision:** 0.333 | **Recall:** 0.006  

**Format:** `regex`  
**Rule ID:** `5e2b25bc`  
**Description:**
Captures names following 'Angeklagte' or 'Angeklagten', ensuring no trailing space and handling initials with dots.

**Content:**
```
(?:Angeklagte|Angeklagten)\s+([A-Z][a-zäöüß]*\.?\s*(?:[A-Z][a-zäöüß]*\.?\s*)*|\b[A-Z]\s*\.\b|\b[A-Z]\b)(?=\s*(?:,|\.|\)|\]|\s|$))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.333 | 0.006 | 0.012 | 6 | 2 | 4 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 2 | 4 | 267 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `62293`) (sent_id: `62293`)


Das Landgericht hat den Angeklagten T. D. wegen schweren sexuellen Missbrauchs eines Kindes in Tateinheit mit sexuellem Missbrauch eines Schutzbefohlenen in zwei Fällen sowie schweren sexuellen Missbrauchs einer widerstandsunfähigen Person in fünf Fällen zu einer Gesamtfreiheitsstrafe von vier Jahren und drei Monaten verurteilt .

| Predicted | Gold |
|---|---|
| `T. D.` | `T. D.` |

**Example 1** (doc_id: `63047`) (sent_id: `63047`)


1. Die Verurteilung des Angeklagten T. D. wegen schweren sexuellen Missbrauchs einer widerstandsunfähigen Person in den Fällen III. 3 bis 7 der Urteilsgründe nach dem zur Tatzeit geltenden § 179 Abs. 1 Nr. 1 , Abs. 5 Nr. 1 StGB in der bis zum 9. November 2016 geltenden Fassung hält revisionsrechtlicher Überprüfung nicht stand , weil die Urteilsgründe eine Widerstandsunfähigkeit des Nebenklägers nicht belegen .

| Predicted | Gold |
|---|---|
| `T. D.` | `T. D.` |

**Missed by this rule (FN):**

- `§ 179 Abs. 1 Nr. 1 , Abs. 5 Nr. 1 StGB` (NRM)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `61243`) (sent_id: `61243`)


Doch auch wenn die Taten 21 bis 23 und 33 der Urteilsgründe sich damit im Hinblick auf das Marihuana als selbständige Umsatzgeschäfte darstellen , fallen die darauf bezogenen Handlungen des Angeklagten mit denjenigen zusammen , die dem Absatz des zugleich in diesen Fällen gehandelten Amphetamins dienten , hinsichtlich dessen aufgrund des einheitlichen Erwerbs im Fall 27 der Urteilsgründe von einer Bewertungseinheit und damit von einer Tat im Rechtssinne auszugehen ist : Im Fall 21 der Urteilsgründe nahm der Angeklagte die Bestellung beider Betäubungsmittel einheitlich entgegen , in den Fällen 22 und 23 der Urteilsgründe lieferte er sie auch gleichzeitig an die Angeklagte A. A. .

**False Positives:**

- `A. A. ` — partial — gold is substring of pred: `A. A.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `A. A.`(PER)

**Example 1** (doc_id: `64047`) (sent_id: `64047`)


In einem weiteren Fall öffnete der Angeklagte Knopf und Reißverschluss seiner Hose und forderte die Zeugin sinngemäß auf , an seinem Glied zu reiben .

**False Positives:**

- `Knopf` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `64798`) (sent_id: `64798`)


2. Im Umfang der Aufhebung wird die Sache zu neuer Verhandlung und Entscheidung , auch über die Kosten des Rechtsmittels des Angeklagten S. , an das Amtsgericht - Schöffengericht - Aachen zurückverwiesen .

**False Positives:**

- `S. ` — partial — gold is substring of pred: `S.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S.`(PER)
- `Amtsgericht - Schöffengericht - Aachen`(ORG)

**Example 3** (doc_id: `65248`) (sent_id: `65248`)


3. Im Umfang der Aufhebung wird die Sache zu neuer Verhandlung und Entscheidung , auch über die Kosten des Rechtsmittels des Angeklagten A. , an eine andere Strafkammer des Landgerichts zurückverwiesen .

**False Positives:**

- `A. ` — partial — gold is substring of pred: `A.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `A.`(PER)

</details>

---

</details>

---

<details>
<summary>🔇 Inactive Rules</summary>

## `Initials with dots and spaces (e.g., T. D.)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `a3f99de3`  
**Description:**
Captures sequences of initial-dot-space-initial-dot patterns.

**Content:**
```
\b([A-Z]\.[ ]+[A-Z]\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Names after 'Rechtsanwalt' or 'Rechtsanwältin'`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `1e1f1dc2`  
**Description:**
Captures names following legal profession titles, ensuring no trailing space and handling initials with dots.

**Content:**
```
(?:Rechtsanwalt|Rechtsanwältin)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*|\b[A-Z]\s*\.\b|\b[A-Z]\b)(?=\s*(?:,|\.|\)|\]|\s|$))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Anonymized initials with dots and spaces (e.g., T. D.)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `d3f9d622`  
**Description:**
Captures sequences of initial-dot-space-initial-dot patterns.

**Content:**
```
\b([A-Z]\.)\s+([A-Z]\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Anonymized names with dots and ellipses or spaces`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `b7b02b0b`  
**Description:**
Captures anonymized names with dots and ellipses or spaces (e.g., 'K …', 'H …'), excluding company names.

**Content:**
```
\b([A-Z]\d?\.?)\s+…\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Initials after legal roles`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `a6567e07`  
**Description:**
Captures single letter initials (with or without dots) following legal role indicators like 'Angeklagten', 'Kläger', 'Zeuge', 'Zeugin'.

**Content:**
```
(?:Angeklagten|Kläger|Zeuge|Zeugin|Vertrauensmann)\s+([A-Z](?:\.)?)(?=\s*(?:,|\.|\)|\]|\s|$))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Initials with surname (corrected)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `1395461f`  
**Description:**
Captures patterns like 'K. Schmidt' or 'M. P. Borom' where an initial (with dot) is followed by a surname.

**Content:**
```
\b([A-Z]\.)\s+([A-Z][a-zäöüß]+(?:-[A-Z][a-zäöüß]+)?)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Anonymized names with comma and initial`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `001ef128`  
**Description:**
Captures anonymized names in the format 'Surname , Initial.' (e.g., 'Boolell , M.', 'Rosen , R. C.').

**Content:**
```
\b([A-Z][a-zäöüß]+\s*,\s*[A-Z]\s*\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Multi-initial names with surname`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `23f66d88`  
**Description:**
Captures names with multiple initials followed by a surname (e.g., 'P. W. McMillan', 'Tomlinson , J. M.').

**Content:**
```
\b([A-Z]\s*\.\s*[A-Z]\s*\.\s*[A-Z][a-zäöüß]+|[A-Z][a-zäöüß]+\s*,\s*[A-Z]\s*\.\s*[A-Z]\s*\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Anonymized initials with dots (corrected)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `5ba69691`  
**Description:**
Captures anonymized initials consisting of a single uppercase letter followed by a dot, ensuring it's not part of a larger multi-initial sequence already captured.

**Content:**
```
(?<![A-Za-zäöüß\.\s])([A-Z])\.(?![A-Za-zäöüß\.\s])
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

## `Known German Surnames`

**F1:** 0.289 | **Precision:** 0.496 | **Recall:** 0.204  

**Format:** `regex`  
**Rule ID:** `de849019`  
**Description:**
Captures a comprehensive list of common German surnames and other frequently occurring names in legal texts, including compound names.

**Content:**
```
\b(?:Berger|Sander|Schmaltz|Bender|Fritz|Becker|Feilcke|Kiel|Brunke|Mutzbauer|Zeng|Benrath|Cierniak|Appl|Franke|Brune|Roggenbuck|Niemann|Grube|Volz|Quentin|Spinner|Schlewing|Marx|Fischer|Bredendiek|Stresemann|Kayser|Koch|Volk|Liebert|Limperg|Schlünder|Berg|Matthias|Hohoff|Leitz|Krumbiegel|Paul|Treber|Spaniol|Feddersen|Schultz|Schuh|Lauer|Lipphaus|Gräfl|Schäfer|König|Müller|Dauber|Tiemann|Deichfuß|Ahrendt|Graßnack|Schmidt-Räntschist|Schmidt|Radtke|Pohl|Nielebock|Fischermeier|Bellay|Kirchhof|Busch|Krehl|Hayen|Glock|Redeker|Morawek|Eder|Baumgardt|Hoffmann|Kaya|Seyhan|Çerikci|Hacker|Merzbach|Meiser|Knoll|Kriener|Nielsen|Musiol|Dorn|Albertshofer|Wollny|Bieringer|Hilber|Paetzold|Baumgart|Geier|Höchst|Fritze|Wiegele|Kleinschmidt|Kirschneck|Arnoldi|Haupt|Demir|Baykara|Aranyosi|Căldăraru|Nassauer|van den Berg|Einstein|Shah|Kuemmerle|Tomlinson|Wright|Vogt|Saime|Özcan|Sen|Bar|Refaeli|Josh|Duhamel|Kelvin|Heinkel|Pape|Harsdorf|Gebhardt|Enerji Yapi-Yol Sen|Eschelbach|Bormann|Möhring|Zimmermann|Rose|Kohout|Abdullah Öcalan|Schwabe|Paffrath|Jaehde|Eckstein|Matter|KCK|PKK)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.496 | 0.204 | 0.289 | 133 | 66 | 67 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 66 | 67 | 258 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60115`) (sent_id: `60115`)


Marx

| Predicted | Gold |
|---|---|
| `Marx` | `Marx` |

**Example 1** (doc_id: `60200`) (sent_id: `60200`)


Quentin

| Predicted | Gold |
|---|---|
| `Quentin` | `Quentin` |

**Example 2** (doc_id: `60460`) (sent_id: `60460`)


Koch

| Predicted | Gold |
|---|---|
| `Koch` | `Koch` |

**Example 3** (doc_id: `60485`) (sent_id: `60485`)


Radtke

| Predicted | Gold |
|---|---|
| `Radtke` | `Radtke` |

**Example 4** (doc_id: `60542`) (sent_id: `60542`)


Kohout

| Predicted | Gold |
|---|---|
| `Kohout` | `Kohout` |

**Example 5** (doc_id: `60579`) (sent_id: `60579`)


Spinner

| Predicted | Gold |
|---|---|
| `Spinner` | `Spinner` |

**Example 6** (doc_id: `60726`) (sent_id: `60726`)


Krehl

| Predicted | Gold |
|---|---|
| `Krehl` | `Krehl` |

**Example 7** (doc_id: `60994`) (sent_id: `60994`)


Hayen

| Predicted | Gold |
|---|---|
| `Hayen` | `Hayen` |

**Example 8** (doc_id: `61122`) (sent_id: `61122`)


Selbst wenn mit § 12 Nr. 4 des Arbeitsvertrags der Parteien eine unzulässige Umgehung von § 622 Abs. 6 BGB verbunden wäre , führte dies lediglich zur Nichtigkeit der Optionsklausel ( Rein NZA-RR 2009 , 462 ; Vogt Befristungs- und Optionsvereinbarungen im professionellen Mannschaftssport S. 161 f. ) , nicht aber zur Verlängerung des Vertrags .

| Predicted | Gold |
|---|---|
| `Vogt` | `Vogt` |

**Missed by this rule (FN):**

- `§ 12 Nr. 4 des Arbeitsvertrags` (REG)
- `§ 622 Abs. 6 BGB` (NRM)
- `Rein NZA-RR 2009 , 462` (LIT)

**Example 9** (doc_id: `61174`) (sent_id: `61174`)


Hohoff

| Predicted | Gold |
|---|---|
| `Hohoff` | `Hohoff` |

**Example 10** (doc_id: `61183`) (sent_id: `61183`)


Schmidt

| Predicted | Gold |
|---|---|
| `Schmidt` | `Schmidt` |

**Example 11** (doc_id: `61238`) (sent_id: `61238`)


Becker

| Predicted | Gold |
|---|---|
| `Becker` | `Becker` |

**Example 12** (doc_id: `61487`) (sent_id: `61487`)


Fischermeier

| Predicted | Gold |
|---|---|
| `Fischermeier` | `Fischermeier` |

**Example 13** (doc_id: `61517`) (sent_id: `61517`)


Volz

| Predicted | Gold |
|---|---|
| `Volz` | `Volz` |

**Example 14** (doc_id: `61573`) (sent_id: `61573`)


Becker

| Predicted | Gold |
|---|---|
| `Becker` | `Becker` |

**Example 15** (doc_id: `61618`) (sent_id: `61618`)


Busch

| Predicted | Gold |
|---|---|
| `Busch` | `Busch` |

**Example 16** (doc_id: `61671`) (sent_id: `61671`)


Krumbiegel

| Predicted | Gold |
|---|---|
| `Krumbiegel` | `Krumbiegel` |

**Example 17** (doc_id: `61787`) (sent_id: `61787`)


In der Beschwerdesache betreffend die Marke 30 2009 056 266 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 14. September 2017 unter Mitwirkung des Vorsitzenden Richters Knoll sowie der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Knoll` | `Knoll` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Kriener` (PER)
- `Nielsen` (PER)

**Example 18** (doc_id: `61916`) (sent_id: `61916`)


In dem Rechtsstreit Demir und Baykara v. Türkei seien Fragen des Streikverbots im öffentlichen Dienst nicht Gegenstand des Verfahrens gewesen .

| Predicted | Gold |
|---|---|
| `Demir` | `Demir` |
| `Baykara` | `Baykara` |

**Missed by this rule (FN):**

- `Türkei` (LOC)

**Example 19** (doc_id: `62100`) (sent_id: `62100`)


Feddersen

| Predicted | Gold |
|---|---|
| `Feddersen` | `Feddersen` |

**Example 20** (doc_id: `62189`) (sent_id: `62189`)


Berg

| Predicted | Gold |
|---|---|
| `Berg` | `Berg` |

**Example 21** (doc_id: `62451`) (sent_id: `62451`)


Bormann

| Predicted | Gold |
|---|---|
| `Bormann` | `Bormann` |

**Example 22** (doc_id: `62613`) (sent_id: `62613`)


2. Der vorliegende Fall ist durch solche besonderen , zusätzlichen Umstände gekennzeichnet , die über eine bloße Mitwirkung des Richters Müller in einem Gesetzgebungsverfahren deutlich hinausreichen und die Besorgnis seiner Befangenheit begründen .

| Predicted | Gold |
|---|---|
| `Müller` | `Müller` |

**Example 23** (doc_id: `62675`) (sent_id: `62675`)


Die Einsprechende legt zur geltend gemachten Lieferung des Kontaktsockels „ Waffle Kelvin “ an die A … Inc. die teilgeschwärzte Ablichtung einer Rechnung vor :

| Predicted | Gold |
|---|---|
| `Kelvin` | `Kelvin` |

**Missed by this rule (FN):**

- `A … Inc.` (ORG)

**Example 24** (doc_id: `62927`) (sent_id: `62927`)


Mutzbauer

| Predicted | Gold |
|---|---|
| `Mutzbauer` | `Mutzbauer` |

**Example 25** (doc_id: `63317`) (sent_id: `63317`)


Schäfer

| Predicted | Gold |
|---|---|
| `Schäfer` | `Schäfer` |

**Example 26** (doc_id: `63360`) (sent_id: `63360`)


Fischermeier

| Predicted | Gold |
|---|---|
| `Fischermeier` | `Fischermeier` |

**Example 27** (doc_id: `63382`) (sent_id: `63382`)


Franke

| Predicted | Gold |
|---|---|
| `Franke` | `Franke` |

**Example 28** (doc_id: `63457`) (sent_id: `63457`)


Berger

| Predicted | Gold |
|---|---|
| `Berger` | `Berger` |

**Example 29** (doc_id: `63556`) (sent_id: `63556`)


Quentin

| Predicted | Gold |
|---|---|
| `Quentin` | `Quentin` |

**Example 30** (doc_id: `63746`) (sent_id: `63746`)


Treber

| Predicted | Gold |
|---|---|
| `Treber` | `Treber` |

**Example 31** (doc_id: `63811`) (sent_id: `63811`)


Mutzbauer

| Predicted | Gold |
|---|---|
| `Mutzbauer` | `Mutzbauer` |

**Example 32** (doc_id: `63922`) (sent_id: `63922`)


Quentin

| Predicted | Gold |
|---|---|
| `Quentin` | `Quentin` |

**Example 33** (doc_id: `63964`) (sent_id: `63964`)


Spinner

| Predicted | Gold |
|---|---|
| `Spinner` | `Spinner` |

**Example 34** (doc_id: `63981`) (sent_id: `63981`)


Grube

| Predicted | Gold |
|---|---|
| `Grube` | `Grube` |

**Example 35** (doc_id: `64075`) (sent_id: `64075`)


Krumbiegel

| Predicted | Gold |
|---|---|
| `Krumbiegel` | `Krumbiegel` |

**Example 36** (doc_id: `64093`) (sent_id: `64093`)


Bredendiek

| Predicted | Gold |
|---|---|
| `Bredendiek` | `Bredendiek` |

**Example 37** (doc_id: `64126`) (sent_id: `64126`)


Heinkel

| Predicted | Gold |
|---|---|
| `Heinkel` | `Heinkel` |

**Example 38** (doc_id: `64133`) (sent_id: `64133`)


Becker

| Predicted | Gold |
|---|---|
| `Becker` | `Becker` |

**Example 39** (doc_id: `64153`) (sent_id: `64153`)


Gräfl

| Predicted | Gold |
|---|---|
| `Gräfl` | `Gräfl` |

**Example 40** (doc_id: `64345`) (sent_id: `64345`)


Heinkel

| Predicted | Gold |
|---|---|
| `Heinkel` | `Heinkel` |

**Example 41** (doc_id: `64370`) (sent_id: `64370`)


Bender

| Predicted | Gold |
|---|---|
| `Bender` | `Bender` |

**Example 42** (doc_id: `64394`) (sent_id: `64394`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2013 053 470.0 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 9. November 2017 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Knoll` | `Knoll` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Kriener` (PER)
- `Nielsen` (PER)

**Example 43** (doc_id: `64485`) (sent_id: `64485`)


Becker

| Predicted | Gold |
|---|---|
| `Becker` | `Becker` |

**Example 44** (doc_id: `64527`) (sent_id: `64527`)


In der Beschwerdesache betreffend die Marke 30 2011 035 856 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Hacker` | `Hacker` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Merzbach` (PER)
- `Meiser` (PER)

**Example 45** (doc_id: `64854`) (sent_id: `64854`)


Quentin

| Predicted | Gold |
|---|---|
| `Quentin` | `Quentin` |

**Example 46** (doc_id: `64925`) (sent_id: `64925`)


In der Beschwerdesache betreffend die Marke 30 2010 003 649 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Hacker` | `Hacker` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Merzbach` (PER)
- `Meiser` (PER)

**Example 47** (doc_id: `64967`) (sent_id: `64967`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2015 034 000.6 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 11. Januar 2018 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Knoll` | `Knoll` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Kriener` (PER)
- `Nielsen` (PER)

**Example 48** (doc_id: `65192`) (sent_id: `65192`)


Krehl

| Predicted | Gold |
|---|---|
| `Krehl` | `Krehl` |

**Example 49** (doc_id: `65226`) (sent_id: `65226`)


6.1 Der Gegenstand des erteilten Anspruchs 1 gilt gegenüber dem Stand der Technik nach dem Kontaktsockel „ Waffle Kelvin “ nicht als neu .

| Predicted | Gold |
|---|---|
| `Kelvin` | `Kelvin` |

**Example 50** (doc_id: `65292`) (sent_id: `65292`)


Hayen

| Predicted | Gold |
|---|---|
| `Hayen` | `Hayen` |

**Example 51** (doc_id: `65500`) (sent_id: `65500`)


In der Beschwerdesache betreffend die Designanmeldung .... ( hier : Antrag auf Verfahrenskostenhilfe ) hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Hacker` | `Hacker` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Merzbach` (PER)
- `Meiser` (PER)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60070`) (sent_id: `60070`)


Eine Mindestentfernung zwischen Haupt- und beruflicher Zweitwohnung bestimmt das Einkommensteuergesetz nicht ( Geserich , in : Kirchhof / Söhn / Mellinghoff , EStG , § 9 Rz G 60 ) .

**False Positives:**

- `Haupt` — no gold match — likely missing annotation
- `Kirchhof` — partial — pred is substring of gold: `Geserich , in : Kirchhof / Söhn / Mellinghoff , EStG , § 9 Rz G 60`

> overlaps gold: 1  |  likely missing annotation: 1

**Gold Entities:**

- `Einkommensteuergesetz`(NRM)
- `Geserich , in : Kirchhof / Söhn / Mellinghoff , EStG , § 9 Rz G 60`(LIT)

**Example 1** (doc_id: `60324`) (sent_id: `60324`)


Soweit die Widersprechende sich darauf beruft , dass auch kennzeichnungsschwache Marken zumindest Schutz gegen eine identische Übernahme beanspruchen könnten , führt dieser grundsätzlich zutreffende Einwand gleichfalls nicht zur Bejahung der Verwechslungsgefahr , da sich die hier zu vergleichenden Zeichen – wie nachfolgend unter Ziffer 1. 3. dargelegt – erheblich unterscheiden ( vgl. im Übrigen zum Schutzumfang zu Unrecht eingetragener , materiell schutzunfähiger Marken Ströbele / Hacker , MarkenG , 11. Aufl. , § 9 Rn. 194 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele / Hacker , MarkenG , 11. Aufl. , § 9 Rn. 194`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 9 Rn. 194`(LIT)

**Example 2** (doc_id: `60329`) (sent_id: `60329`)


Die strafschärfende Berücksichtigung der hierin liegenden Schuldsteigerung gerate weder mit dem in § 46 Abs. 3 StGB verankerten Doppelverwertungsverbot von Tatbestandsmerkmalen ( SSW-StGB / Eschelbach , aaO , § 46 Rn. 93 , 185 ; von Heintschel-Heinegg , Streng-FS 2017 S. 229 , 239 ) noch mit dem Gedanken in Konflikt , dass es sich um das Regeltatbild des Totschlags handele ( Fahl , JR 2017 , 391 , 393 ; MüKo / Schneider , aaO , § 212 Rn. 82 ; Tomiak , HRRS 2017 , 225 ff. ) .

**False Positives:**

- `Eschelbach` — partial — pred is substring of gold: `SSW-StGB / Eschelbach , aaO , § 46 Rn. 93 , 185`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 46 Abs. 3 StGB`(NRM)
- `SSW-StGB / Eschelbach , aaO , § 46 Rn. 93 , 185`(LIT)
- `von Heintschel-Heinegg , Streng-FS 2017 S. 229 , 239`(LIT)
- `Fahl , JR 2017 , 391 , 393`(LIT)
- `MüKo / Schneider , aaO , § 212 Rn. 82`(LIT)
- `Tomiak , HRRS 2017 , 225 ff.`(LIT)

**Example 3** (doc_id: `60363`) (sent_id: `60363`)


Übe ein Arbeitnehmer eine vom Arbeitgeber entlohnte Nebentätigkeit aus , so seien die Einnahmen aus der Nebentätigkeit durch das Arbeitsverhältnis veranlasst , wenn Haupt- und Nebentätigkeit gleichartig seien und die Nebentätigkeit unter ähnlichen organisatorischen Bedingungen ausgeübt werde wie die Haupttätigkeit oder wenn der Steuerpflichtige mit der Nebentätigkeit ihm aus seinem Dienstverhältnis - faktisch oder rechtlich - obliegende Nebenpflichten erfülle .

**False Positives:**

- `Haupt` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `60577`) (sent_id: `60577`)


Hingegen ist die Zulassung wegen Divergenz gegen eine Entscheidung eines anderen obersten Gerichtshofes des Bundes oder des EuGH nicht zulässig ( vgl Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160 RdNr 11 mwN ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160 RdNr 11`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH`(ORG)
- `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160 RdNr 11`(LIT)

**Example 5** (doc_id: `60634`) (sent_id: `60634`)


Das Landgericht hätte sich an dieser Stelle daher auch damit auseinandersetzen müssen , dass der Angeklagte am 12. Juli 2015 einen Diebstahl „ im besonders schweren Fall “ beging , wofür er am 6. Juli 2016 vom Amtsgericht Frankfurt am Main - Außenstelle Höchst - verurteilt wurde .

**False Positives:**

- `Höchst` — partial — pred is substring of gold: `Amtsgericht Frankfurt am Main - Außenstelle Höchst -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Amtsgericht Frankfurt am Main - Außenstelle Höchst -`(ORG)

**Example 6** (doc_id: `60870`) (sent_id: `60870`)


Dabei stehen die genannten Faktoren in einem Verhältnis der Wechselwirkung , so dass ein geringerer Grad eines Faktors durch einen höheren Grad eines anderen Faktors ausgeglichen werden kann ( EuGH GRUR 1998 , 387 , 389 Rn. 22 – Sabél / Puma ; GRUR 1998 , 922 , 923 Rn. 17 – Canon ; GRUR Int. 1999 , 734 , 736 Rn. 19 – Lloyd ; GRUR Int. 2000 , 899 , 901 Rn. 40 – Marca / Adidas ; GRUR 2008 , 343 , 345 Rn. 48 – Il Ponte Finanziaria Spa / HABM ; BGH GRUR 2012 , 1040 , 1042 Rn. 25 – pjur / pure ; GRUR 2012 , 930 , 932 Rn. 22 – Bogner B / Barbie B / ; GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY ; GRUR 2011 , 826 Rn. 11 – Enzymax / Enzymix ; GRUR 2011 , 824 Rn. 18 – Kappa ; GRUR 2010 , 235 Rn. 35 – AIDA / AIDU ; GRUR 2009 , 766 , 768 Rn. 26 – Stofffähnchen ; GRUR 2009 , 772 , 776 Rn. 51 – Augsburger Puppenkiste ; GRUR 2009 , 484 , 486 Rn. 23 – Metrobus ; GRUR 2008 , 1002 , 1004 Rn. 23 – Schuhpark ; Hacker , a. a. O. , § 9 Rn. 41 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Hacker , a. a. O. , § 9 Rn. 41`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 1998 , 387 , 389 Rn. 22 – Sabél / Puma`(RS)
- `GRUR 1998 , 922 , 923 Rn. 17 – Canon`(RS)
- `GRUR Int. 1999 , 734 , 736 Rn. 19 – Lloyd`(RS)
- `GRUR Int. 2000 , 899 , 901 Rn. 40 – Marca / Adidas`(RS)
- `GRUR 2008 , 343 , 345 Rn. 48 – Il Ponte Finanziaria Spa / HABM`(RS)
- `BGH GRUR 2012 , 1040 , 1042 Rn. 25 – pjur / pure`(RS)
- `GRUR 2012 , 930 , 932 Rn. 22 – Bogner B / Barbie B /`(RS)
- `GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY`(RS)
- `GRUR 2011 , 826 Rn. 11 – Enzymax / Enzymix`(RS)
- `GRUR 2011 , 824 Rn. 18 – Kappa`(RS)
- `GRUR 2010 , 235 Rn. 35 – AIDA / AIDU`(RS)
- `GRUR 2009 , 766 , 768 Rn. 26 – Stofffähnchen`(RS)
- `GRUR 2009 , 772 , 776 Rn. 51 – Augsburger Puppenkiste`(RS)
- `GRUR 2009 , 484 , 486 Rn. 23 – Metrobus`(RS)
- `GRUR 2008 , 1002 , 1004 Rn. 23 – Schuhpark`(RS)
- `Hacker , a. a. O. , § 9 Rn. 41`(LIT)

**Example 7** (doc_id: `60929`) (sent_id: `60929`)


Darauf zielt der jeweilige Gegenstand des Patentanspruchs 1 nach Haupt- und Hilfsantrag ersichtlich nicht .

**False Positives:**

- `Haupt` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `61007`) (sent_id: `61007`)


Im Allgemeinen lassen unter anderem Angaben über Werbeaufwendungen Schlüsse auf die Verkehrsbekanntheit einer Marke zu ( BGH GRUR 2013 , 833 , 836 Rn. 41 – Culinaria / Villa Culinaria ; Hacker , a. a. O. , § 9 Rn. 160 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Hacker , a. a. O. , § 9 Rn. 160`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH GRUR 2013 , 833 , 836 Rn. 41 – Culinaria / Villa Culinaria`(RS)
- `Hacker , a. a. O. , § 9 Rn. 160`(LIT)

**Example 9** (doc_id: `61028`) (sent_id: `61028`)


2. Der Grundsatz der Gewaltenteilung ( Art. 20 Abs. 2 Satz 2 GG ) verlangt , dass die Rechtsprechung durch " besondere " , das heißt von den Organen der Gesetzgebung und der vollziehenden Gewalt verschiedene Organe des Staates ausgeübt wird ( BVerfGE 18 , 241 < 254 > ) ; dies wird durch das in Art. 92 1. Halbsatz GG begründete Rechtsprechungsmonopol der Richter konkretisiert ( vgl. BVerfGE 22 , 49 < 76 > ; Hopfauf , in : Schmidt-Bleibtreu / Hofmann / Henneke , GG , 14. Aufl. 2018 , Art. 92 Rn. 2 ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Hopfauf , in : Schmidt-Bleibtreu / Hofmann / Henneke , GG , 14. Aufl. 2018 , Art. 92 Rn. 2`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 20 Abs. 2 Satz 2 GG`(NRM)
- `BVerfGE 18 , 241 < 254 >`(RS)
- `Art. 92 1. Halbsatz GG`(NRM)
- `BVerfGE 22 , 49 < 76 >`(RS)
- `Hopfauf , in : Schmidt-Bleibtreu / Hofmann / Henneke , GG , 14. Aufl. 2018 , Art. 92 Rn. 2`(LIT)

**Example 10** (doc_id: `61312`) (sent_id: `61312`)


In der Rechtsprechung des Bundesverfassungsgerichts ist jedoch anerkannt , dass eine Erledigung nicht zur Unzulässigkeit der Verfassungsbeschwerde führt , wenn der gerügte Grundrechtseingriff besonders schwer wiegt und anderenfalls die Klärung einer verfassungsrechtlichen Frage von grundsätzlicher Bedeutung unterbliebe ( vgl. BVerfGE 81 , 138 < 141 f. > ; 91 , 125 < 133 > ; 98 , 169 < 198 > ; 103 , 44 < 58 > ) , die gegenstandslos gewordene Maßnahme den Beschwerdeführer weiterhin beeinträchtigt ( vgl. BVerfGE 99 , 129 < 138 > ) oder ein Rehabilitationsinteresse des Beschwerdeführers besteht ( vgl. auch BVerfG , Urteil des Zweiten Senats vom 7. November 2017 - 2 BvE 2/11 - , juris , Rn. 183 ; Bethge , in : Maunz / Schmidt-Bleibtreu / Klein / Bethge , BVerfGG , § 90 Rn. 269a < Oktober 2013 > ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Bethge , in : Maunz / Schmidt-Bleibtreu / Klein / Bethge , BVerfGG , § 90 Rn. 269a < Oktober 2013 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesverfassungsgerichts`(ORG)
- `BVerfGE 81 , 138 < 141 f. > ; 91 , 125 < 133 > ; 98 , 169 < 198 > ; 103 , 44 < 58 >`(RS)
- `BVerfGE 99 , 129 < 138 >`(RS)
- `BVerfG , Urteil des Zweiten Senats vom 7. November 2017 - 2 BvE 2/11 - , juris , Rn. 183`(RS)
- `Bethge , in : Maunz / Schmidt-Bleibtreu / Klein / Bethge , BVerfGG , § 90 Rn. 269a < Oktober 2013 >`(LIT)

**Example 11** (doc_id: `61434`) (sent_id: `61434`)


Zwar ist anerkannt , dass die Zulassung der Revision nicht notwendig im Tenor erfolgen muss , sondern auch in den Entscheidungsgründen erfolgen kann ( BSG Beschluss vom 30. 6. 2008 - B 2 U 1/08 RH - SozR 4 - 1500 § 160 Nr 17 RdNr 11 mwN ; BSG Urteil vom 29. 6. 1977 - 11 RA 94/76 - SozR 1500 § 161 Nr 16 ) , sofern sie eindeutig ausgesprochen wird ( Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160 RdNr 24a mwN ; Voelzke in juris-PK SGG , § 160 RdNr 58 ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160 RdNr 24a`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Beschluss vom 30. 6. 2008 - B 2 U 1/08 RH - SozR 4 - 1500 § 160 Nr 17 RdNr 11`(RS)
- `BSG Urteil vom 29. 6. 1977 - 11 RA 94/76 - SozR 1500 § 161 Nr 16`(RS)
- `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160 RdNr 24a`(LIT)
- `Voelzke in juris-PK SGG , § 160 RdNr 58`(LIT)

**Example 12** (doc_id: `61445`) (sent_id: `61445`)


Denn der Verkehr ist daran gewöhnt , im Geschäftsleben ständig mit neuen Wortschöpfungen konfrontiert zu werden , durch die sachbezogene Informationen übermittelt werden sollen und die sich häufig nicht an grammatikalischen Regeln orientieren ( vgl. Ströbele in : Ströbele / Hacker , Markengesetz , 12. Aufl. 2018 , § 8 , Rn. 199 ; BPatG 24 W ( pat ) 510/15 – Knetmonster , verfügbar über PAVIS PROMA ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele in : Ströbele / Hacker , Markengesetz , 12. Aufl. 2018 , § 8 , Rn. 199`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Ströbele in : Ströbele / Hacker , Markengesetz , 12. Aufl. 2018 , § 8 , Rn. 199`(LIT)
- `BPatG 24 W ( pat ) 510/15 – Knetmonster , verfügbar über PAVIS PROMA`(RS)

**Example 13** (doc_id: `61635`) (sent_id: `61635`)


Maßgeblich ist bei Wortkombinationen letztlich , ob die Kombination der Bestandteile über die bloße Zusammenfügung beschreibender Elemente hinausgeht oder sich – wie vorliegend – in deren Summenwirkung erschöpft , was der Unterscheidungskraft entgegensteht ( siehe dazu auch Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 182 mit weiteren Rechtsprechungsnachweisen ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 182`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 182`(LIT)

**Example 14** (doc_id: `61642`) (sent_id: `61642`)


M2.2. 2a wobei die Kontaktfedern ( 7a , 7b ) eines Kontaktfederpaares so angeordnet sind , dass sie für eine Kelvin-Kontaktierung beide jeweils gegen denselben Anschlusskontakt ( 5 ) des Bauelements ( 3 ) gedrückt werden , und wobei

**False Positives:**

- `Kelvin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 15** (doc_id: `61690`) (sent_id: `61690`)


Eine statthafte und zulässige Vollstreckungserinnerung setzt eine erinnerungsfähige Vollstreckungsmaßnahme oder ein Unterlassen voraus ( vgl. Spohnheimer , in : Wieczorek / Schütze , ZPO , 4. Aufl. 2016 , § 766 Rn. 63 ; Schmidt / Brinkmann , in : MüKo-ZPO , 5. Aufl. 2016 , § 766 Rn. 48 ) , die Erinnerung nach § 766 Abs. 2 Alternative 3 ZPO in der Regel einen Kostenansatz , eine Zahlungsaufforderung oder die Vorbereitung der Abrechnung durch den Gerichtsvollzieher ( vgl. LG Dortmund , Beschluss vom 19. Oktober 2006 - 9 T 613/06 - , NJOZ 2007 , S. 65 < 66 f. > ; LG Hannover , Beschluss vom 4. Februar 1977 - 11 T 162/76 - , juris , Rn. 2 f. ; vgl. auch Schmidt / Brinkmann , in : MüKo-ZPO , 5. Aufl. 2016 , § 766 Rn. 62 ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Schmidt / Brinkmann , in : MüKo-ZPO , 5. Aufl. 2016 , § 766 Rn. 48`
- `Schmidt` — similar text (different position): `Schmidt / Brinkmann , in : MüKo-ZPO , 5. Aufl. 2016 , § 766 Rn. 48`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Spohnheimer , in : Wieczorek / Schütze , ZPO , 4. Aufl. 2016 , § 766 Rn. 63`(LIT)
- `Schmidt / Brinkmann , in : MüKo-ZPO , 5. Aufl. 2016 , § 766 Rn. 48`(LIT)
- `§ 766 Abs. 2 Alternative 3 ZPO`(NRM)
- `LG Dortmund , Beschluss vom 19. Oktober 2006 - 9 T 613/06 - , NJOZ 2007 , S. 65 < 66 f. >`(RS)
- `LG Hannover , Beschluss vom 4. Februar 1977 - 11 T 162/76 - , juris , Rn. 2 f.`(RS)
- `Schmidt / Brinkmann , in : MüKo-ZPO , 5. Aufl. 2016 , § 766 Rn. 62`(LIT)

**Example 16** (doc_id: `61693`) (sent_id: `61693`)


Insoweit hätte es des klägerseitigen Vortrags bedurft , weshalb nach den dem LSG vorliegenden Beweismitteln Fragen zum tatsächlichen und medizinischen Sachverhalt aus der rechtlichen Sicht des LSG erkennbar offengeblieben sind und damit zu einer weiteren Aufklärung des Sachverhalts zwingende Veranlassung bestanden haben soll ( vgl Becker , Die Nichtzulassungsbeschwerde zum BSG < Teil II > , SGb 2007 , 328 , 332 zu RdNr 188 unter Hinweis auf BSG Beschluss vom 14. 12. 1999 - B 2 U 311/99 B - mwN ) .

**False Positives:**

- `Becker` — partial — pred is substring of gold: `Becker , Die Nichtzulassungsbeschwerde zum BSG < Teil II > , SGb 2007 , 328 , 332 zu RdNr 188`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Becker , Die Nichtzulassungsbeschwerde zum BSG < Teil II > , SGb 2007 , 328 , 332 zu RdNr 188`(LIT)
- `BSG Beschluss vom 14. 12. 1999 - B 2 U 311/99 B -`(RS)

**Example 17** (doc_id: `61703`) (sent_id: `61703`)


Entscheidend ist daher , ob es im Hinblick auf die beanspruchten Waren für sich genommen als beschreibende Angabe verwendet werden kann ( vgl. BGH Mitt. 1995 , 184 - quattro ; GRUR 2000 , 231 , 232 - FÜNFER ; BPatG , Beschluss vom 30. Januar 2008 , 29 W ( pat ) 92/04 - DUO ; Beschluss vom 27. Januar 2010 , 28 W ( pat ) 96/09 - TRIO ; Ströbele / Hacker / Thiering , a. a. O. , § 8 , Rdnr. 508 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele / Hacker / Thiering , a. a. O. , § 8 , Rdnr. 508`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH Mitt. 1995 , 184 - quattro`(RS)
- `GRUR 2000 , 231 , 232 - FÜNFER`(RS)
- `BPatG , Beschluss vom 30. Januar 2008 , 29 W ( pat ) 92/04 - DUO`(RS)
- `Beschluss vom 27. Januar 2010 , 28 W ( pat ) 96/09 - TRIO`(RS)
- `Ströbele / Hacker / Thiering , a. a. O. , § 8 , Rdnr. 508`(LIT)

**Example 18** (doc_id: `61770`) (sent_id: `61770`)


Der Schaft und der Dorn sind , wie in Absatz [ 0040 ] beschrieben und in Fig. 1 gezeigt , aus einem einzigen Stift gebildet und beim Verschrauben axial fluchtend angeordnet .

**False Positives:**

- `Dorn` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `62068`) (sent_id: `62068`)


Wird das Vorliegen eines Verfahrensmangels nach § 160 Abs 2 Nr 3 SGG gerügt , so müssen bei dessen Bezeichnung wie bei einer Verfahrensrüge innerhalb einer zugelassenen Revision zunächst die diesen Verfahrensmangel des LSG ( vermeintlich ) begründenden Tatsachen substantiiert dargelegt werden ( vgl nur BSG SozR 1500 § 160a Nr 14 und 36 ; Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160a RdNr 16 mwN ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160a RdNr 16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 160 Abs 2 Nr 3 SGG`(NRM)
- `BSG SozR 1500 § 160a Nr 14 und 36`(RS)
- `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160a RdNr 16`(LIT)

**Example 20** (doc_id: `62090`) (sent_id: `62090`)


Ob Werbungskosten in diesem Sinne in einem unmittelbaren wirtschaftlichen Zusammenhang mit den Einnahmen stehen , ist unter Rückgriff auf die zu § 3c Abs. 1 EStG entwickelten Grundsätze zu klären ( vgl. Lübbehüsen in Berger / Steck / Lübbehüsen , a. a. O. , § 3 Rz 23 ; Köhler in Berger / Steck / Lübbehüsen , a. a. O. , § 3 Rz 87 , m. w. N. ; Feyerabend / Mielke / Rieger , Recht der Finanzinstrumente 2011 , 191 , 194 ; vgl. auch BMF-Schreiben in BStBl I 2005 , 728 , Rz 45 ) .

**False Positives:**

- `Berger` — partial — pred is substring of gold: `Lübbehüsen in Berger / Steck / Lübbehüsen , a. a. O. , § 3 Rz 23`
- `Berger` — similar text (different position): `Lübbehüsen in Berger / Steck / Lübbehüsen , a. a. O. , § 3 Rz 23`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 3c Abs. 1 EStG`(NRM)
- `Lübbehüsen in Berger / Steck / Lübbehüsen , a. a. O. , § 3 Rz 23`(LIT)
- `Köhler in Berger / Steck / Lübbehüsen , a. a. O. , § 3 Rz 87`(LIT)
- `Feyerabend / Mielke / Rieger , Recht der Finanzinstrumente 2011 , 191 , 194`(LIT)
- `BMF-Schreiben in BStBl I 2005 , 728 , Rz 45`(REG)

**Example 21** (doc_id: `62499`) (sent_id: `62499`)


Kritische Stimmen ( vgl. Jescheck / Weigend Strafrecht AT , 5. Aufl. S. 887 ; SSW-StGB / Eschelbach , 2. Aufl. § 46 Rn. 93 , 185 ; Frisch , in : 50 Jahre Bundesgerichtshof , Festgabe aus der Wissenschaft , 2000 , Band IV , S. 269 , 290 f. ; Hörnle , Tatproportionale Strafzumessung , 1999 , S. 260 , 263 ; Grünewald , Das vorsätzliche Tötungsdelikt , 2010 , S. 148 ff. ; Foth , JR 1985 , 397 , 398 ; Bruns , JR 1981 , 512 , 513 ; Müller , NStZ 1985 , 158 , 161 ) haben darauf hingewiesen , dass die Auffassung , wonach die Vorsatzform als eine eigenständige Strafzumessungstatsache ausscheide , den aus dem besonderen Teil des Strafgesetzbuchs ersichtlichen gesetzgeberischen Wertungen widerspreche ( vgl. Foth , JR 1985 , 398 ; Fahl , Zur Bedeutung des Regeltatbildes bei der Bemessung der Strafe , Diss. 1996 , S. 154 ) .

**False Positives:**

- `Eschelbach` — partial — pred is substring of gold: `SSW-StGB / Eschelbach , 2. Aufl. § 46 Rn. 93 , 185`
- `Müller` — partial — pred is substring of gold: `Müller , NStZ 1985 , 158 , 161`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Jescheck / Weigend Strafrecht AT , 5. Aufl. S. 887`(LIT)
- `SSW-StGB / Eschelbach , 2. Aufl. § 46 Rn. 93 , 185`(LIT)
- `Frisch , in : 50 Jahre Bundesgerichtshof , Festgabe aus der Wissenschaft , 2000 , Band IV , S. 269 , 290 f.`(LIT)
- `Hörnle , Tatproportionale Strafzumessung , 1999 , S. 260 , 263`(LIT)
- `Grünewald , Das vorsätzliche Tötungsdelikt , 2010 , S. 148 ff.`(LIT)
- `Foth , JR 1985 , 397 , 398`(LIT)
- `Bruns , JR 1981 , 512 , 513`(LIT)
- `Müller , NStZ 1985 , 158 , 161`(LIT)
- `Strafgesetzbuchs`(NRM)
- `Foth , JR 1985 , 398`(LIT)
- `Fahl , Zur Bedeutung des Regeltatbildes bei der Bemessung der Strafe , Diss. 1996 , S. 154`(LIT)

**Example 22** (doc_id: `62527`) (sent_id: `62527`)


2. An der Besteuerung wird Deutschland durch das DBA-Kanada 2001 nicht gehindert ; der Senat hält insoweit an der in einem Verfahren des einstweiligen Rechtsschutzes bereits geäußerten Rechtsauffassung ( Senatsbeschluss vom 13. Dezember 2011 I B 159/11 , BFH / NV 2012 , 417 ) auch nach erneuter Prüfung fest ( dieser Rechtslage zustimmend z.B. Gosch in Kirchhof , EStG , 16. Aufl. , § 49 Rz 90 mit Fußn. 6 ; Hick in Schönfeld / Ditz , DBA , Art. 18 Rz 167 f. , 171 , 174 ; Blümich / Wied , § 49 EStG Rz 218 ; Kuhn in Herrmann / Heuer / Raupach , § 49 EStG Rz 1003 ; Kühnen , EFG 2016 , 578 ; Schober , EFG 2016 , 990 ; Ismer in Vogel / Lehner , DBA , 6. Aufl. , Art. 18 Rz 87 ; Holthaus , Internationale Wirtschaftsbriefe 2017 , 796 , 797 ; im Ergebnis ebenso die Verwaltungspraxis , s. Bayerisches Landesamt für Steuern , Verfügung vom 8. Juni 2011 , Internationales Steuerrecht 2011 , 776 ; a. A. W. Wassermeyer in Wassermeyer , DBA , Kanada Art. 18 Rz 70a ; Hagemann / Kahlenberg / Cloer , Betriebs-Berater 2017 , 2775 , 2785 f. ) .

**False Positives:**

- `Kirchhof` — partial — pred is substring of gold: `Gosch in Kirchhof , EStG , 16. Aufl. , § 49 Rz 90 mit Fußn. 6`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Deutschland`(LOC)
- `DBA-Kanada 2001`(NRM)
- `Senatsbeschluss vom 13. Dezember 2011 I B 159/11 , BFH / NV 2012 , 417`(RS)
- `Gosch in Kirchhof , EStG , 16. Aufl. , § 49 Rz 90 mit Fußn. 6`(LIT)
- `Hick in Schönfeld / Ditz , DBA , Art. 18 Rz 167 f. , 171 , 174`(LIT)
- `Blümich / Wied , § 49 EStG Rz 218`(LIT)
- `Kuhn in Herrmann / Heuer / Raupach , § 49 EStG Rz 1003`(LIT)
- `Kühnen , EFG 2016 , 578`(LIT)
- `Schober , EFG 2016 , 990`(LIT)
- `Ismer in Vogel / Lehner , DBA , 6. Aufl. , Art. 18 Rz 87`(LIT)
- `Holthaus , Internationale Wirtschaftsbriefe 2017 , 796 , 797`(LIT)
- `Bayerisches Landesamt für Steuern , Verfügung vom 8. Juni 2011 , Internationales Steuerrecht 2011 , 776`(LIT)
- `W. Wassermeyer in Wassermeyer , DBA , Kanada Art. 18 Rz 70a`(LIT)
- `Hagemann / Kahlenberg / Cloer , Betriebs-Berater 2017 , 2775 , 2785 f.`(LIT)

**Example 23** (doc_id: `62660`) (sent_id: `62660`)


Die Lebenszeiternennung gewährleistet allerdings das Höchstmaß an Unabhängigkeit ( vgl. Schmidt-Räntsch , DRiG , 6. Aufl. 2009 , § 11 Rn. 2 ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Schmidt-Räntsch , DRiG , 6. Aufl. 2009 , § 11 Rn. 2`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Schmidt-Räntsch , DRiG , 6. Aufl. 2009 , § 11 Rn. 2`(LIT)

**Example 24** (doc_id: `62671`) (sent_id: `62671`)


Es kann dahinstehen , ob es fachüblich ist , vor der ( Vierleiter- ) Testung an jedem Anschlusskontakt des Bauelements in einer Schleife über die beiden Kontaktfedern der Kelvin-Kontaktierung den Übergangswiderstand der Kontaktierung zu den Kontaktfedern eines Kontaktfederpaares zu bestimmen und später zur Korrektur der Messergebnisse zu verwenden , vgl. Patentschrift , Absatz 0020 , denn der Senat kann auch im Zusammenhang mit einer derartigen Messung des Übergangswiderstands keine Veranlassung des Fachmanns erkennen , die Kontaktfedern C entsprechend der Anweisungen in den Merkmalen M3 bis M3.2 lamelliert für eine hohe Stromtragfähigkeit auszubilden .

**False Positives:**

- `Kelvin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 25** (doc_id: `62688`) (sent_id: `62688`)


Dabei muss der schutzwürdige Besitzstand durch eine hinreichende Marktpräsenz und daraus folgende ( gewisse ) Bekanntheit der Kennzeichnung im Inland belegt sein ( vgl. BGH GRUR 2014 , 780 – Liquidrom ; Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 877 ; Ingerl / Rohnke , MarkenG , 3. Auflage , § 8 Rn. 308 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 877`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH GRUR 2014 , 780 – Liquidrom`(RS)
- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 877`(LIT)
- `Ingerl / Rohnke , MarkenG , 3. Auflage , § 8 Rn. 308`(LIT)

**Example 26** (doc_id: `62715`) (sent_id: `62715`)


Es ist also zu fragen , ob bei den beteiligten Verkehrskreisen der Eindruck aufkommen kann , Ware und Dienstleistung unterlägen der Kontrolle desselben Unternehmens , sei es , dass das Dienstleistungsunternehmen sich selbständig auch mit der Herstellung bzw. dem Vertrieb der Waren befasst , sei es , dass der Warenhersteller oder -vertreiber sich auch auf dem entsprechenden Dienstleistungsbereich selbständig gewerblich betätigt ( Ströbele / Hacker , MarkenG , 11. Auflage , § 9 Rn. 115 ; BGH , GRUR 2012 , 1145 , 1148 Rn. 35 - Pelikan ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele / Hacker , MarkenG , 11. Auflage , § 9 Rn. 115`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Ströbele / Hacker , MarkenG , 11. Auflage , § 9 Rn. 115`(LIT)
- `BGH , GRUR 2012 , 1145 , 1148 Rn. 35 - Pelikan`(RS)

**Example 27** (doc_id: `63072`) (sent_id: `63072`)


Schmidt-Räntsch

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Schmidt-Räntsch`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Schmidt-Räntsch`(PER)

**Example 28** (doc_id: `63088`) (sent_id: `63088`)


Denn eine Abkürzung ist nur dann nicht schutzfähig , wenn sie im Verkehr als solche gebräuchlich oder aus sich heraus verständlich ist sowie von den beteiligten Verkehrskreisen ohne weiteres der betreffenden Sachangabe gleichgesetzt und insoweit verstanden werden kann ( vgl. Ströbele in Ströbele / Hacker / Thiering , a. a. O. , § 8 Rn. 224 , 226 , 227 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele in Ströbele / Hacker / Thiering , a. a. O. , § 8 Rn. 224 , 226 , 227`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Ströbele in Ströbele / Hacker / Thiering , a. a. O. , § 8 Rn. 224 , 226 , 227`(LIT)

**Example 29** (doc_id: `63162`) (sent_id: `63162`)


Notwendig hierfür ist eine Grundlage im nationalen Recht ( vgl. EGMR , Enerji Yapi-Yol Sen c. Turquie , Urteil vom 21. April 2009 , Nr. 68959/01 , § 26 ) .

**False Positives:**

- `Enerji Yapi-Yol Sen` — partial — pred is substring of gold: `EGMR , Enerji Yapi-Yol Sen c. Turquie , Urteil vom 21. April 2009 , Nr. 68959/01 , § 26`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EGMR , Enerji Yapi-Yol Sen c. Turquie , Urteil vom 21. April 2009 , Nr. 68959/01 , § 26`(RS)

**Example 30** (doc_id: `63163`) (sent_id: `63163`)


Nachdem das LSG ausschließlich über eine echte Leistungsklage ( § 54 Abs 5 SGG ) entschieden und eine Entscheidung über die isolierte Anfechtungsklage gegen den Widerspruchsbescheid vom 23. 7. 2015 überhaupt nicht getroffen hat , konnte der Senat auch nicht von einer Zurückverweisung absehen und sich auf eine ( teilweise ) Aufhebung des LSG-Urteils beschränken ( vgl dazu BSG SozR 4 - 1500 § 160a Nr 13 ; vgl auch BSG SozR 4 - 1500 § 144 Nr 7 RdNr 13 ; Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , aaO , § 160a RdNr 19e mwN ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , aaO , § 160a RdNr 19e`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 54 Abs 5 SGG`(NRM)
- `BSG SozR 4 - 1500 § 160a Nr 13`(RS)
- `BSG SozR 4 - 1500 § 144 Nr 7 RdNr 13`(RS)
- `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , aaO , § 160a RdNr 19e`(LIT)

**Example 31** (doc_id: `63254`) (sent_id: `63254`)


Die Verbürgung des Art. 97 Abs. 2 GG greift demnach nicht für Richter auf Probe , Richter kraft Auftrags , abgeordnete Richter ( soweit das Abordnungsverhältnis betroffen ist ) , Richter im Nebenamt und ehrenamtliche Richter ( Heusch , in : Schmidt-Bleibtreu / Hofmann / Henneke , GG , 14. Aufl. 2018 , Art. 97 Rn. 42 ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Heusch , in : Schmidt-Bleibtreu / Hofmann / Henneke , GG , 14. Aufl. 2018 , Art. 97 Rn. 42`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 97 Abs. 2 GG`(NRM)
- `Heusch , in : Schmidt-Bleibtreu / Hofmann / Henneke , GG , 14. Aufl. 2018 , Art. 97 Rn. 42`(LIT)

**Example 32** (doc_id: `63546`) (sent_id: `63546`)


Die Organisationseinheiten stellen der PKK Finanzmittel bereit , rekrutieren Nachwuchs für den Guerillakampf und betreiben Propaganda .

**False Positives:**

- `PKK` — type mismatch — same span as gold: `PKK`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `PKK`(ORG)

**Example 33** (doc_id: `64003`) (sent_id: `64003`)


Gestützt wird dieser Befund durch die weiteren in § 1 SGB III genannten Programmsätzen bzw Handlungsrichtlinien ( ausführlich dazu Schmidt-De Caluwe in Mutschler / Schmidt-De Caluwe / Coseriu , SGB III , 6. Aufl 2017 , § 1 RdNr 28 ff ; Deinert in Gagel , SGB II / SGB III , § 1 RdNr 102 ff , Stand Juni 2016 ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Schmidt-De Caluwe in Mutschler / Schmidt-De Caluwe / Coseriu , SGB III , 6. Aufl 2017 , § 1 RdNr 28 ff`
- `Schmidt` — partial — pred is substring of gold: `Schmidt-De Caluwe in Mutschler / Schmidt-De Caluwe / Coseriu , SGB III , 6. Aufl 2017 , § 1 RdNr 28 ff`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 1 SGB III`(NRM)
- `Schmidt-De Caluwe in Mutschler / Schmidt-De Caluwe / Coseriu , SGB III , 6. Aufl 2017 , § 1 RdNr 28 ff`(LIT)
- `Deinert in Gagel , SGB II / SGB III , § 1 RdNr 102 ff , Stand Juni 2016`(LIT)

**Example 34** (doc_id: `64134`) (sent_id: `64134`)


Zu weitgehend und mit den Vorgaben der Europäischen Menschenrechtskonvention nicht mehr zu vereinbaren wäre allerdings ein Verständnis , das alle Beschäftigten des öffentlichen Dienstes eines Staates - gegebenenfalls unter Einschluss von Beschäftigten staatlicher Wirtschafts- oder Industrieunternehmen ( vgl. EGMR , Enerji Yapi-Yol Sen c. Turquie , Urteil vom 21. April 2009 , Nr. 68959/01 , § 32 ) - dem Bereich der Staatsverwaltung zuordnete .

**False Positives:**

- `Enerji Yapi-Yol Sen` — partial — pred is substring of gold: `EGMR , Enerji Yapi-Yol Sen c. Turquie , Urteil vom 21. April 2009 , Nr. 68959/01 , § 32`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europäischen Menschenrechtskonvention`(NRM)
- `EGMR , Enerji Yapi-Yol Sen c. Turquie , Urteil vom 21. April 2009 , Nr. 68959/01 , § 32`(RS)

**Example 35** (doc_id: `64263`) (sent_id: `64263`)


Es hat jedoch nicht über den mit dem Schriftsatz vom 8. Dezember 2017 vorgeschalteten , nach herrschender Auffassung nicht an Fristen oder Antragsteller-Quoren gebundenen ( vgl. für die Wahl zum Gesamtvertrauenspersonenausschuss Höges , SBG , Stand März 2018 , § 47 Rn. 20 ) ( Haupt- ) Antrag , die Nichtigkeit der Wahl festzustellen , entschieden .

**False Positives:**

- `Haupt` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Wahl zum Gesamtvertrauenspersonenausschuss Höges , SBG , Stand März 2018 , § 47 Rn. 20`(LIT)

**Example 36** (doc_id: `64459`) (sent_id: `64459`)


Andererseits nahm er bestimmenden Einfluss auf die Arbeit der ihm in der Parteihierarchie unterstellten PKK-Kader , indem er ihre Arbeit koordinierte , ihnen Anweisungen gab und sich über die Entwicklungen in den von ihnen geleiteten Räumen unterrichten ließ .

**False Positives:**

- `PKK` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 37** (doc_id: `64692`) (sent_id: `64692`)


Diese einzelnen Faktoren sind zwar für sich gesehen voneinander unabhängig , bestimmen aber in ihrer Wechselwirkung den Rechtsbegriff der Verwechslungsgefahr ( vgl. dazu EuGH GRUR 2008 , 343 Rn. 48 – Il Ponte Finanziaria Spa / HABM ; BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY ; GRUR 2012 , 1040 Rn. 25 – pjur / pure ; siehe auch Ströbele / Hacker , Markengesetz , 11. Aufl. , § 9 Rn. 40 ff. m. w. N. ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele / Hacker , Markengesetz , 11. Aufl. , § 9 Rn. 40 ff.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2008 , 343 Rn. 48 – Il Ponte Finanziaria Spa / HABM`(RS)
- `BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY`(RS)
- `GRUR 2012 , 1040 Rn. 25 – pjur / pure`(RS)
- `Ströbele / Hacker , Markengesetz , 11. Aufl. , § 9 Rn. 40 ff.`(LIT)

**Example 38** (doc_id: `64871`) (sent_id: `64871`)


Der Gesamteindruck der Widerspruchsmarke werde auch nicht durch den Zeichenbestandteil „ Becker “ geprägt .

**False Positives:**

- `Becker` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 39** (doc_id: `64923`) (sent_id: `64923`)


Eine solche Abrede steht der Annahme einer Einlage oder anderer unbedingt rückzahlbarer Gelder des Publikums und damit eines Bankgeschäfts im Sinne des § 1 Abs. 1 Satz 2 Nr. 1 KWG entgegen ( vgl. BT- Drucks. 15/3641 , S. 36 ; BGH , Urteil vom 16. Mai 2017 - VI ZR 266/16 , NJW 2017 , 2463 ; Urteil vom 10. Februar 2015 - VI ZR 569/13 , NJW-RR 2015 , 675 , 676 ; Schäfer in Boos / Fischer / Schulter-Mattler , KWG , 5. Aufl. , § 1 Rn 46 ; Gehrlein , WM 2017 , 1385 , 1386 ; vgl. zur Rechtsanwendungspraxis der BaFin deren Merkblatt „ Hinweise zum Tatbestand des Einlagengeschäfts “ , Stand März 2014 , NZG 2014 , 379 , 381 ) .

**False Positives:**

- `Schäfer` — partial — pred is substring of gold: `Schäfer in Boos / Fischer / Schulter-Mattler , KWG , 5. Aufl. , § 1 Rn 46`
- `Fischer` — partial — pred is substring of gold: `Schäfer in Boos / Fischer / Schulter-Mattler , KWG , 5. Aufl. , § 1 Rn 46`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 1 Abs. 1 Satz 2 Nr. 1 KWG`(NRM)
- `BT- Drucks. 15/3641 , S. 36`(LIT)
- `BGH , Urteil vom 16. Mai 2017 - VI ZR 266/16 , NJW 2017 , 2463`(RS)
- `Urteil vom 10. Februar 2015 - VI ZR 569/13 , NJW-RR 2015 , 675 , 676`(RS)
- `Schäfer in Boos / Fischer / Schulter-Mattler , KWG , 5. Aufl. , § 1 Rn 46`(LIT)
- `Gehrlein , WM 2017 , 1385 , 1386`(LIT)
- `BaFin`(ORG)
- `Einlagengeschäfts “ , Stand März 2014 , NZG 2014 , 379 , 381`(LIT)

**Example 40** (doc_id: `64966`) (sent_id: `64966`)


Nach diesen Maßstäben sei Paul Kirchhof Beteiligter an der Sache im Sinne des § 18 Abs. 1 Nr. 1 BVerfGG .

**False Positives:**

- `Paul` — partial — pred is substring of gold: `Paul Kirchhof`
- `Kirchhof` — partial — pred is substring of gold: `Paul Kirchhof`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Paul Kirchhof`(PER)
- `§ 18 Abs. 1 Nr. 1 BVerfGG`(NRM)

**Example 41** (doc_id: `65067`) (sent_id: `65067`)


ee ) Für Gutschriften auf dem Wertguthabenkonto eines Fremd-Geschäftsführers einer GmbH , über die im Streitfall zu entscheiden ist , gilt nichts anderes ( im Ergebnis ebenso FG Düsseldorf in EFG 2012 , 1400 , aus anderen Gründen aufgehoben durch Senatsurteil in BFH / NV 2014 , 1372 ; FG Baden-Württemberg in EFG 2017 , 1585 ; Schmidt / Krüger , a. a. O. , § 19 Rz 100 " Arbeitszeitkonten " ; Blümich / Geserich , § 19 EStG Rz 280 " Zeitwertkonten " ; Breinersdorfer , in : Kirchhof / Söhn / Mellinghoff , EStG , § 19 Rz A 194 ; Pust in Littmann / Bitz / Pust , a. a. O. , § 11 Rz 25 ; Graefe , DStR 2017 , 2199 ; Wellisch / Quiring , BB 2012 , 2029 ; Hilbert / Paul , NWB 2012 , 3391 ; Portner , DStR 2009 , 1838 ; Wellisch / Quast , DB 2006 , 1024 ; a. A. BMF-Schreiben in BStBl I 2009 , 1286 ; Sterzinger , BB 2012 , 2728 ; Harder-Buschner , NWB 2009 , 2132 ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Schmidt / Krüger , a. a. O. , § 19 Rz 100 " Arbeitszeitkonten "`
- `Kirchhof` — partial — pred is substring of gold: `Breinersdorfer , in : Kirchhof / Söhn / Mellinghoff , EStG , § 19 Rz A 194`
- `Paul` — partial — pred is substring of gold: `Hilbert / Paul , NWB 2012 , 3391`

> overlaps gold: 3  |  likely missing annotation: 0

**Gold Entities:**

- `FG Düsseldorf in EFG 2012 , 1400`(RS)
- `Senatsurteil in BFH / NV 2014 , 1372`(RS)
- `FG Baden-Württemberg in EFG 2017 , 1585`(RS)
- `Schmidt / Krüger , a. a. O. , § 19 Rz 100 " Arbeitszeitkonten "`(LIT)
- `Blümich / Geserich , § 19 EStG Rz 280 " Zeitwertkonten "`(LIT)
- `Breinersdorfer , in : Kirchhof / Söhn / Mellinghoff , EStG , § 19 Rz A 194`(LIT)
- `Pust in Littmann / Bitz / Pust , a. a. O. , § 11 Rz 25`(LIT)
- `Graefe , DStR 2017 , 2199`(LIT)
- `Wellisch / Quiring , BB 2012 , 2029`(LIT)
- `Hilbert / Paul , NWB 2012 , 3391`(LIT)
- `Portner , DStR 2009 , 1838`(LIT)
- `Wellisch / Quast , DB 2006 , 1024`(LIT)
- `BMF-Schreiben in BStBl I 2009 , 1286`(REG)
- `Sterzinger , BB 2012 , 2728`(LIT)
- `Harder-Buschner , NWB 2009 , 2132`(LIT)

**Example 42** (doc_id: `65078`) (sent_id: `65078`)


Die mittelbare Verwechslungsgefahr setzt voraus , dass die beteiligten Verkehrskreise zwar die Unterschiede zwischen den Vergleichsmarken erkennen ( und insoweit keinen unmittelbaren Verwechslungen unterliegen ) , gleichwohl einen in beiden Marken übereinstimmend enthaltenen Bestandteil als Stammzeichen des Inhabers der älteren Marke werten , diesem Stammbestandteil also für sich schon die maßgebliche Herkunftsfunktion beimessen und deshalb die übrigen ( abweichenden ) Markenteile nur noch als Kennzeichen für bestimmte Waren / Dienstleistungen aus dem Geschäftsbetrieb des Inhabers der älteren Marke ansehen ( BGH GRUR 2010 , 729 , 732 Rn. 40 – MIXI ; Hacker , a. a. O. , § 9 Rn. 488 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Hacker , a. a. O. , § 9 Rn. 488`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH GRUR 2010 , 729 , 732 Rn. 40 – MIXI`(RS)
- `Hacker , a. a. O. , § 9 Rn. 488`(LIT)

**Example 43** (doc_id: `65220`) (sent_id: `65220`)


Allerdings kann ein Markenbestandteil eine selbständig kollisionsbegründende Bedeutung haben , wenn er den Gesamteindruck der mehrgliedrigen Marke prägt ( Hacker , a. a. O. , § 9 Rn. 361 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Hacker , a. a. O. , § 9 Rn. 361`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Hacker , a. a. O. , § 9 Rn. 361`(LIT)

**Example 44** (doc_id: `65333`) (sent_id: `65333`)


Eine solche aus dem materiellen Recht zu entnehmende Begrenzung , die aufgrund ihrer Wirkung als Rechtswegsperre gegebenenfalls auf ihre Vereinbarkeit mit verfassungsrechtlichen Rechtsschutzgarantien ( vgl. dazu etwa Schmidt-Aßmann , in : Maunz / Dürig , GG , Stand September 2017 , Art. 19 Abs. 4 Rn. 16 ff. m. w. N. ) zu prüfen wäre , liegt hier jedoch nicht vor .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Schmidt-Aßmann , in : Maunz / Dürig , GG , Stand September 2017 , Art. 19 Abs. 4 Rn. 16 ff.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Schmidt-Aßmann , in : Maunz / Dürig , GG , Stand September 2017 , Art. 19 Abs. 4 Rn. 16 ff.`(LIT)

**Example 45** (doc_id: `65350`) (sent_id: `65350`)


Entgegen der Auffassung des Berufungsgerichts handelt es sich bei dem PÜV 2002 nicht um einen zwischen einem Betriebsveräußerer und -erwerber abgeschlossenen , wirksamen berechtigenden Vertrag zugunsten Dritter ( zu Personalüberleitungsvereinbarungen als mögliche Verträge zugunsten Dritter allgemein : Schaub ArbR-HdB / Ahrendt 17. Aufl. § 116 Rn. 52 ) , mit dem sich der Betriebserwerber ua. verpflichtet , die Tarifwerke des öffentlichen Dienstes auch im übergegangenen Arbeitsverhältnis dynamisch - weiter - anzuwenden .

**False Positives:**

- `Ahrendt` — partial — pred is substring of gold: `Schaub ArbR-HdB / Ahrendt 17. Aufl. § 116 Rn. 52`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `PÜV 2002`(REG)
- `Schaub ArbR-HdB / Ahrendt 17. Aufl. § 116 Rn. 52`(LIT)

**Example 46** (doc_id: `65368`) (sent_id: `65368`)


bb ) Dementsprechend haben auch die Finanzgerichte , die Finanzverwaltung sowie die Kommentarliteratur eine Wohnung am Beschäftigungsort bejaht , wenn der Arbeitnehmer von dort üblicherweise täglich zu seiner Arbeitsstätte fahren kann ( z.B. FG des Saarlandes , Urteil vom 25. Juni 1993 1 K 189/92 , EFG 1994 , 201 ; FG Münster , Urteil vom 19. Oktober 1999 13 K 2468/94 E , juris ; FG Hamburg , Urteil vom 26. Februar 2014 1 K 234/12 , EFG 2014 , 1185 ; FG Baden-Württemberg , Urteil vom 16. Juni 2016 1 K 3229/14 , EFG 2016 , 1423 , bestätigt durch Senatsurteil vom 16. November 2017 VI R 31/16 , BFHE 260 , 143 ; Amtliches Lohnsteuer-Handbuch 2013 H 9.11 ( 1 - 4 ) " Zweitwohnung am Beschäftigungsort " ; Schmidt / Loschelder , a. a. O. , § 9 Rz 229 ; Wagner , in : Heuermann / Wagner , LohnSt , F , Rz 313 ; Hartz / Meeßen / Wolf , ABC-Führer Lohnsteuer , " Doppelte Haushaltsführung " Rz 47 , 48 ; Oertel in Kirchhof , a. a. O. , § 9 Rz 109 ; Blümich / Thürmer , § 9 EStG Rz 360 ; Zimmer in Littmann / Bitz / Pust , a. a. O. , § 9 Rz 1040 ; Lochte in Frotscher , EStG , Freiburg 2011 , § 9 Rz 186 ; Geserich , Deutsches Steuerrecht - DStR - 2012 , 1737 , 1740 ; kritisch aber Dürr , Deutsche Steuer-Zeitung 2017 , 323 ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Schmidt / Loschelder , a. a. O. , § 9 Rz 229`
- `Kirchhof` — partial — pred is substring of gold: `Oertel in Kirchhof , a. a. O. , § 9 Rz 109`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `FG des Saarlandes , Urteil vom 25. Juni 1993 1 K 189/92 , EFG 1994 , 201`(RS)
- `FG Münster , Urteil vom 19. Oktober 1999 13 K 2468/94 E , juris`(RS)
- `FG Hamburg , Urteil vom 26. Februar 2014 1 K 234/12 , EFG 2014 , 1185`(RS)
- `FG Baden-Württemberg , Urteil vom 16. Juni 2016 1 K 3229/14 , EFG 2016 , 1423`(RS)
- `Senatsurteil vom 16. November 2017 VI R 31/16 , BFHE 260 , 143`(RS)
- `Amtliches Lohnsteuer-Handbuch 2013 H 9.11 ( 1 - 4 ) " Zweitwohnung am Beschäftigungsort "`(RS)
- `Schmidt / Loschelder , a. a. O. , § 9 Rz 229`(LIT)
- `Wagner , in : Heuermann / Wagner , LohnSt , F , Rz 313`(LIT)
- `Hartz / Meeßen / Wolf , ABC-Führer Lohnsteuer , " Doppelte Haushaltsführung " Rz 47 , 48`(LIT)
- `Oertel in Kirchhof , a. a. O. , § 9 Rz 109`(LIT)
- `Blümich / Thürmer , § 9 EStG Rz 360`(LIT)
- `Zimmer in Littmann / Bitz / Pust , a. a. O. , § 9 Rz 1040`(LIT)
- `Lochte in Frotscher , EStG , Freiburg 2011 , § 9 Rz 186`(LIT)
- `Geserich , Deutsches Steuerrecht - DStR - 2012 , 1737 , 1740`(LIT)
- `Dürr , Deutsche Steuer-Zeitung 2017 , 323`(LIT)

**Example 47** (doc_id: `65454`) (sent_id: `65454`)


Seine nicht weiter begründete Annahme , „ irgendwelche abweichenden Abreden , insbesondere sogenannte Nachrangabreden , stellen für den Anleger offensichtlich überraschende und damit unwirksame Klauseln dar “ , hält aber auch eingedenk der nur eingeschränkten revisionsgerichtlichen Kontrolle der tatrichterlichen Auslegung von Verträgen und der ihnen zugrunde liegenden Erklärungen der Vertragsparteien ( vgl. BGH , Urteil vom 13. Mai 2004 - 5 StR 73/03 , NJW 2004 , 2248 , 2250 mwN [ insoweit in BGHSt 49 , 147 nicht abgedruckt ] ; Sander in : Löwe / Rosenberg , StPO , 26. Aufl. , § 261 Rn. 97 ) einer rechtlichen Überprüfung nicht stand , weil sie über erörterungsbedürftige Feststellungen hinweggeht und deshalb lückenhaft ist .

**False Positives:**

- `Sander` — partial — pred is substring of gold: `Sander in : Löwe / Rosenberg , StPO , 26. Aufl. , § 261 Rn. 97`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 13. Mai 2004 - 5 StR 73/03 , NJW 2004 , 2248 , 2250 mwN [ insoweit in BGHSt 49 , 147 nicht abgedruckt ]`(RS)
- `Sander in : Löwe / Rosenberg , StPO , 26. Aufl. , § 261 Rn. 97`(LIT)

</details>

---

## `Anonymized initials with dots`

**F1:** 0.169 | **Precision:** 1.000 | **Recall:** 0.093  

**Format:** `regex`  
**Rule ID:** `c4f45684`  
**Description:**
Captures anonymized initials (e.g., 'J.', 'K.') ONLY when preceded by a role indicator or title, preventing false positives on legal abbreviations like 'S.' or 'V.'.

**Content:**
```
(?:Herr\s+|Herrn\s+|Dr\.\s+|Prof\.\s+|Richter\s+|Richterin\s+|Vorsitzender\s+Richter\s+|Angeklagte\s+|Angeklagten\s+|Kl\u00e4ger\s+|Zeuge\s+|Zeugin\s+|Gesch\u00e4ftsf\u00fchrer\s+|Rechtsanwalt\s+|Rechtsanw\u00e4ltin\s+|Beteiligte\s+|Beteiligter\s+)([A-Z])\.(?=\s*(?:,|\.|\)|\]|\s|$))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.093 | 0.169 | 30 | 30 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 30 | 0 | 291 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60161`) (sent_id: `60161`)


Allenfalls käme ein solches Vorgehen in Betracht , wenn Dr. T. im maßgeblichen Vorquartal noch nicht im MVZ tätig gewesen wäre ( vgl BSG SozR 4 - 2500 § 87b Nr 2 RdNr 30 : " Hinzurechnung der vom Eintretenden zuvor erbrachten Fallzahlen " ) .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |

**Missed by this rule (FN):**

- `BSG SozR 4 - 2500 § 87b Nr 2 RdNr 30` (RS)

**Example 1** (doc_id: `60445`) (sent_id: `60445`)


Der Facharzt für Kinder- und Jugendpsychiatrie und -psychotherapie Dr. K. führte in seinem Gutachten vom 16. Februar 2017 u. a. aus : Der Kläger habe noch zum Aufnahmezeitpunkt im Klinikum konkrete Suizidgedanken benannt , die er eigenen Angaben zufolge bereits längere Zeit und wiederholt gehabt habe ; von Anschlagsgedanken zumindest auf nicht-zivile Ziele habe er sich nicht ausreichend distanzieren können .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Example 2** (doc_id: `60888`) (sent_id: `60888`)


Die sich aus der Aktenlage und dem Gutachten des Sachverständigen Dr. K. ergebende Persönlichkeitsbewertung deutet nicht auf eine Bereitschaft oder Neigung des Klägers , seinem Leben unabhängig von einem Terroranschlag ein Ende zu setzen .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Example 3** (doc_id: `61021`) (sent_id: `61021`)


Dass der Kläger nach dem Gutachten des Dr. K. noch nicht wie ein Erwachsener wirkt und ihm nach Beobachtungen von Pflegern in der B. er Klinik " jegliche Alltagspraxis " fehle , rechtfertigte bei seiner Abschiebung keine andere Prognose .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `B. er Klinik` (ORG)

**Example 4** (doc_id: `61069`) (sent_id: `61069`)


Nach Zurückverweisung hat das LSG Dr. K. , Institut für neurologisch psychiatrische Begutachtung in B. , mit der Erstellung eines neurologisch-psychiatrischen Gutachtens nach ambulanter Untersuchung des Klägers beauftragt .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `Institut für neurologisch psychiatrische Begutachtung in B.` (ORG)

**Example 5** (doc_id: `61306`) (sent_id: `61306`)


Das Landgericht hat den Angeklagten A. wegen Diebstahls mit Waffen in Tateinheit mit „ fahrlässiger “ Gefährdung des Straßenverkehrs , vorsätzlichem Fahren ohne Fahrerlaubnis und fahrlässiger Körperverletzung sowie wegen unerlaubten Entfernens vom Unfallort und vorsätzlicher Körperverletzung zu der Gesamtfreiheitsstrafe von zwei Jahren und vier Monaten verurteilt ; ferner hat es die Verwaltungsbehörde angewiesen , dem Angeklagten vor Ablauf einer Frist von drei Jahren keine Fahrerlaubnis zu erteilen .

| Predicted | Gold |
|---|---|
| `A.` | `A.` |

**Example 6** (doc_id: `61486`) (sent_id: `61486`)


Das Gericht wies am dritten Hauptverhandlungstag im Zusammenhang mit einem Antrag von Rechtsanwalt P. , den dieser unter Bezugnahme auf das zuvor genannte Schreiben begründet hatte , unter anderem darauf hin , dass sich in der Akte ein „ Terminverlegungsantrag vom 12. April 2016 “ befinde .

| Predicted | Gold |
|---|---|
| `P.` | `P.` |

**Example 7** (doc_id: `61586`) (sent_id: `61586`)


Die Klägerin hatte ihren Sitz zum Zeitpunkt des Eintritts der Job-Sharing-Partnerin Dr. E. im Bezirk der KÄV Nordbaden .

| Predicted | Gold |
|---|---|
| `E.` | `E.` |

**Missed by this rule (FN):**

- `KÄV Nordbaden` (ORG)

**Example 8** (doc_id: `61864`) (sent_id: `61864`)


Zu dem Sachverständigengutachten des Prof. Dr. T. sowie dessen ergänzender Stellungnahme hat die Beklagte mitgeteilt , dass sie diesem nach prüfärztlicher Stellungnahme nicht folgen könne ; der Einschätzung des Dr. S. hat sie sich hingegen angeschlossen .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |
| `S.` | `S.` |

**Example 9** (doc_id: `61871`) (sent_id: `61871`)


Als die Geschädigte während dieses Geschehens von der Zeugin K. angerufen wurde , riss M. der Geschädigten das Mobiltelefon aus der Hand und nahm es im Einverständnis mit dem Angeklagten R. an sich , um zu verhindern , dass die Geschädigte um Hilfe rief .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |
| `R.` | `R.` |

**Missed by this rule (FN):**

- `M.` (PER)

**Example 10** (doc_id: `62635`) (sent_id: `62635`)


Schließlich wird das Stellen eines ordnungsgemäßen Beweisantrags mit der Beschwerdebegründung nicht dargelegt , soweit die Klägerin die Sachaufklärungspflicht des LSG dadurch verletzt sieht , dass dieses keine ergänzende gutachterliche Äußerung Dr. R. eingeholt hat .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |

**Example 11** (doc_id: `62752`) (sent_id: `62752`)


Die Implantation der Coils als alleiniger Grund für die stationäre Behandlung der Versicherten sei nach dem überzeugenden MDK-Gutachten ( Dr. S. ) nicht erforderlich gewesen .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Example 12** (doc_id: `63218`) (sent_id: `63218`)


Zur Revision des Angeklagten R. führte er aus , dass eine Ahndungslücke nicht bestanden habe .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |

**Example 13** (doc_id: `63754`) (sent_id: `63754`)


Soweit das LSG Ausführungen von Prof. Dr. S. referiert , dass sich im Zusammenhang mit den Pneumonien auch immer wieder Septitiden entwickelt hätten , macht das LSG nicht deutlich , dass es diese als festgestellt ansieht , und erst recht nicht , von welchem Begriff oder Schweregrad der Sepsis es im Anschluss an Prof. Dr. S. aufgrund welcher Befunde dabei ausgeht ( ältere Klassifizierungen : Systemisches inflammatorisches Response-Syndrom < SIRS > , Sepsis , schwere Sepsis und septischer Schock ; zur darauf aufbauenden DRG-Kodierung vgl Hanser in Zaiß , DRG : Verschlüsseln leicht gemacht , 14. Aufl 2016 , S 101 ff ; neuere Klassifizierungen : Sepsis-related organ failure assessment score < SOFA > , insbesondere für Intensivstationen , und quickSOFA außerhalb von Intensivstationen ) .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |
| `S.` | `S.` |

**Example 14** (doc_id: `63885`) (sent_id: `63885`)


Nach den Ausführungen des im Verfahren von Amts wegen gehörten Sachverständigen Prof. Dr. T. hätten die vom Kläger vorgetragenen Gewalterfahrungen während seiner Heimaufenthalte ab April 1959 nicht die entscheidende Traumatisierung dargestellt .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |

**Example 15** (doc_id: `63999`) (sent_id: `63999`)


Der weitere vom LSG beauftragte Sachverständige Dr. S. ( Neurologe und Psychiater / Psychotherapeut ) hat die quantitative Leistungsfähigkeit der Klägerin mit mindestens 6 Stunden für leichte Arbeiten mit qualitativen Einschränkungen beurteilt .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Example 16** (doc_id: `64271`) (sent_id: `64271`)


Der Antrag des Klägers , ihm für das Verfahren der Beschwerde gegen die Nichtzulassung der Revision im Urteil des Landessozialgerichts Niedersachsen-Bremen vom 16. November 2017 Prozesskostenhilfe zu bewilligen und Rechtsanwältin K. aus H. beizuordnen , wird abgelehnt .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `Landessozialgerichts Niedersachsen-Bremen` (ORG)
- `H.` (LOC)

**Example 17** (doc_id: `64291`) (sent_id: `64291`)


Ferner hat sie beantragt , Prof. Dr. B. , Zentralinstitut für seelische Gesundheit M. , Institut für Psychiatrie und Psychosomatische Psychotherapie , als sachverständigen Zeugen zu vernehmen sowie den Sachverständigen Prof. Dr. S. zu den mit Schriftsatz vom 21. 3. 2017 erhobenen Einwendungen anzuhören .

| Predicted | Gold |
|---|---|
| `B.` | `B.` |
| `S.` | `S.` |

**Missed by this rule (FN):**

- `Zentralinstitut für seelische Gesundheit M. , Institut für Psychiatrie und Psychosomatische Psychotherapie` (ORG)

**Example 18** (doc_id: `64530`) (sent_id: `64530`)


Das LSG hat vielmehr im Anschluss an die Begründung , warum es dessen sachverständige Bewertung für überzeugend hält , ausgeführt : " Hingegen hat Dr. H. lediglich auf einer Seite kurz dargestellt , dass beim Kläger seinerzeit kein KIG Grad 3 oder höher vorliege .

| Predicted | Gold |
|---|---|
| `H.` | `H.` |

**Example 19** (doc_id: `65228`) (sent_id: `65228`)


Statt einer beantragten orthopädischen Begutachtung unter Berücksichtigung der Schmerzsymptomatik sei eine Begutachtung durch Dr. N. angeordnet worden , obwohl er ( der Kläger ) auf neurologischem Gebiet völlig gesund sei .

| Predicted | Gold |
|---|---|
| `N.` | `N.` |

**Example 20** (doc_id: `65282`) (sent_id: `65282`)


Die Taten fanden ein Ende , nachdem die Zeugin R. im Sexualkundeunterricht aufgeklärt worden war .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |

**Example 21** (doc_id: `65674`) (sent_id: `65674`)


b ) Soweit das Landgericht bei der Zumessung der Freiheitsstrafe bezüglich des Angeklagten R. rechtsfehlerhaft zu seinem Nachteil ebenfalls die tateinheitliche Begehung eines Raubes gewürdigt hat , schließt der Senat aus , dass angesichts der verbleibenden gewichtigen Strafschärfungsgründe , insbesondere im Hinblick auf die verwirklichte Vergewaltigung , das Landgericht ohne den aufgezeigten Rechtsfehler eine niedrigere Freiheitsstrafe verhängt hätte .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |

**Example 22** (doc_id: `66269`) (sent_id: `66269`)


Die Beklagte hat sich hierzu nicht geäußert und nach der Übersendung des Sachverständigengutachtens des Dr. B. ohne weitere inhaltliche Einlassung mit einer Entscheidung ohne mündliche Verhandlung einverstanden erklärt .

| Predicted | Gold |
|---|---|
| `B.` | `B.` |

**Example 23** (doc_id: `66350`) (sent_id: `66350`)


Im Berufungsverfahren fand zunächst ein Erörterungstermin am 7. 9. 2016 statt , in dem der Berichterstatter des LSG die Klägerin persönlich anhörte und Herrn G. als Zeugen vernahm .

| Predicted | Gold |
|---|---|
| `G.` | `G.` |

**Example 24** (doc_id: `66540`) (sent_id: `66540`)


„ Ausgehend vom nach Teileinstellungen noch angeklagten Sachverhalt nach Maßgabe des Hinweisbeschlusses vom 6. Hauptverhandlungstag “ wird hinsichtlich der Angeklagten K. „ bei insoweit glaubhaftem Geständnis und kooperativem Verhalten “ eine Verurteilung zu einer Gesamtfreiheitsstrafe von mindestens neun Monaten bis zu einem Jahr und drei Monaten , deren Vollstreckung zur Bewährung ausgesetzt wird , erfolgen .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Example 25** (doc_id: `66608`) (sent_id: `66608`)


Nach den getroffenen Feststellungen ist unzweifelhaft , dass der Zeuge K. , der im Fall II. 1. der Urteilsgründe selbst Cannabis vom Angeklagten erhielt und weiterverkaufte , dabei auch in der Vorstellung , den Betäubungsmittelhandel des Angeklagten zu fördern , tätig wurde .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

</details>

---

## `Names after 'Richter' or 'Richterin' (corrected)`

**F1:** 0.077 | **Precision:** 0.929 | **Recall:** 0.040  

**Format:** `regex`  
**Rule ID:** `d2bd3fae`  
**Description:**
Captures names following 'Richter', 'Richterin', 'Vorsitzender Richter', etc., ensuring no trailing space.

**Content:**
```
(?:Richter|Richterin|Vorsitzender\s+Richter|Vorsitzende\s+Richterin)\s+([A-Z][a-z\u00e4\u00f6\u00fc\u00df]*\.?\s*(?:[A-Z][a-z\u00e4\u00f6\u00fc\u00df]*\.?\s*)*|\b[A-Z]\s*\.\b|\b[A-Z]\b)(?=\s*(?:,|\.|\)|\]|\s|$))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.929 | 0.040 | 0.077 | 14 | 13 | 1 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 13 | 1 | 293 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60487`) (sent_id: `60487`)


Abweichende Meinung der Richterin Hermanns zum Beschluss des Zweiten Senats vom 22. März 2018 - 2 BvR 780/16 -

| Predicted | Gold |
|---|---|
| `Hermanns` | `Hermanns` |

**Missed by this rule (FN):**

- `Beschluss des Zweiten Senats vom 22. März 2018 - 2 BvR 780/16 -` (RS)

**Example 1** (doc_id: `61787`) (sent_id: `61787`)


In der Beschwerdesache betreffend die Marke 30 2009 056 266 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 14. September 2017 unter Mitwirkung des Vorsitzenden Richters Knoll sowie der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Kriener` | `Kriener` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Nielsen` (PER)

**Example 2** (doc_id: `61969`) (sent_id: `61969`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2013 036 234.9 hat der 26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 16. Oktober 2017 unter Mitwirkung der Vorsitzenden Richterin Kortge sowie der Richter Jacobi und Schödel beschlossen :

| Predicted | Gold |
|---|---|
| `Kortge` | `Kortge` |
| `Jacobi` | `Jacobi` |

**Missed by this rule (FN):**

- `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Schödel` (PER)

**Example 3** (doc_id: `62802`) (sent_id: `62802`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2014 004 201.0 hat der 29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 29. November 2017 durch die Vorsitzende Richterin Dr. Mittenberger-Huber , die Richterin Akintche und die Richterin Seyfarth beschlossen :

| Predicted | Gold |
|---|---|
| `Akintche` | `Akintche` |
| `Seyfarth` | `Seyfarth` |

**Missed by this rule (FN):**

- `29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Mittenberger-Huber` (PER)

**Example 4** (doc_id: `62983`) (sent_id: `62983`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2011 068 984.9 hat der 26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 8. Januar 2018 unter Mitwirkung der Vorsitzenden Richterin Kortge sowie der Richter Jacobi und Schödel beschlossen :

| Predicted | Gold |
|---|---|
| `Kortge` | `Kortge` |
| `Jacobi` | `Jacobi` |

**Missed by this rule (FN):**

- `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Schödel` (PER)

**Example 5** (doc_id: `64394`) (sent_id: `64394`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2013 053 470.0 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 9. November 2017 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Kriener` | `Kriener` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Nielsen` (PER)

**Example 6** (doc_id: `64527`) (sent_id: `64527`)


In der Beschwerdesache betreffend die Marke 30 2011 035 856 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Merzbach` | `Merzbach` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Hacker` (PER)
- `Meiser` (PER)

**Example 7** (doc_id: `64925`) (sent_id: `64925`)


In der Beschwerdesache betreffend die Marke 30 2010 003 649 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Merzbach` | `Merzbach` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Hacker` (PER)
- `Meiser` (PER)

**Example 8** (doc_id: `64967`) (sent_id: `64967`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2015 034 000.6 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 11. Januar 2018 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Kriener` | `Kriener` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Nielsen` (PER)

**Example 9** (doc_id: `65500`) (sent_id: `65500`)


In der Beschwerdesache betreffend die Designanmeldung .... ( hier : Antrag auf Verfahrenskostenhilfe ) hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Merzbach` | `Merzbach` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Hacker` (PER)
- `Meiser` (PER)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `62802`) (sent_id: `62802`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2014 004 201.0 hat der 29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 29. November 2017 durch die Vorsitzende Richterin Dr. Mittenberger-Huber , die Richterin Akintche und die Richterin Seyfarth beschlossen :

**False Positives:**

- `Dr.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Mittenberger-Huber`(PER)
- `Akintche`(PER)
- `Seyfarth`(PER)

</details>

---

## `Anonymized names with ellipses (space)`

**F1:** 0.063 | **Precision:** 0.423 | **Recall:** 0.034  

**Format:** `regex`  
**Rule ID:** `1b824d7a`  
**Description:**
Captures anonymized names with a letter and ellipsis (e.g., 'K …', 'T …', 'A …', 'R …').

**Content:**
```
(?<![A-Za-zäöüß])([A-Z])\s+…
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.423 | 0.034 | 0.063 | 26 | 11 | 15 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 11 | 15 | 303 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60716`) (sent_id: `60716`)


Daraufhin ist das Anschreiben vom 31. Juli 2012 zusammen mit dem Bescheid nochmals mit Zustellungsurkunde an Patentanwalt K … verschickt und ausweislich der Zustellungsurkunde am 11. August 2012 durch Einlegen des Schriftstücks in den zur Wohnung gehörenden Briefkasten oder in eine ähnliche Vorrichtung zugestellt worden .

| Predicted | Gold |
|---|---|
| `K …` | `K …` |

**Example 1** (doc_id: `61830`) (sent_id: `61830`)


Ferner hat die Einsprechende das „ Gutachten zum Sicherheitsschalter CES der Firma E … “ des Herrn Prof. Dr. - Ing. G … vom 15. September 2014 vorgelegt .

| Predicted | Gold |
|---|---|
| `G …` | `G …` |

**Missed by this rule (FN):**

- `E …` (ORG)

**Example 2** (doc_id: `62848`) (sent_id: `62848`)


Anlage A20 : Besucherausweis der Herren D … und S … vom 09. 09. 2010 ,

| Predicted | Gold |
|---|---|
| `D …` | `D …` |
| `S …` | `S …` |

**Example 3** (doc_id: `62874`) (sent_id: `62874`)


Am Nachmittag des Ablauftages der Einspruchsfrist sei die in der Kanzlei der Vertreter damals noch in der Ausbildung befindliche Patentanwaltsfachangestellte B … beauftragt gewesen , farbig markierte Zeichnungen aus im Einspruch zitierten Schriften zum besseren Verständnis von deren Offenbarung anzufertigen .

| Predicted | Gold |
|---|---|
| `B …` | `B …` |

**Example 4** (doc_id: `63250`) (sent_id: `63250`)


Anlage A19 : Besucherausweis der Herren S … und M2 … vom 28. 04. 2010 ,

| Predicted | Gold |
|---|---|
| `S …` | `S …` |

**Missed by this rule (FN):**

- `M2 …` (PER)

**Example 5** (doc_id: `63808`) (sent_id: `63808`)


Aufgrund der dargelegten Sachlage hätte die Prüfungsstelle die Unwirksamkeit der Zustellungen erkennen können , insbesondere nachdem sie durch die Mitteilung von Patentanwalt B1 … vom 2. Mai 2013 Kenntnis von dem Bescheid der Patentanwaltskammer vom 4. April 2013 und damit von der Tatsache erhalten hat , dass die Kanzlei des beigeordneten Patentanwalts K … jedenfalls zum Zeit- punkt der vermeintlichen Zustellung der Fristverlängerung mit Beschlussankündigung am 14. März 2013 schon seit einiger Zeit verwaist war .

| Predicted | Gold |
|---|---|
| `K …` | `K …` |

**Missed by this rule (FN):**

- `B1 …` (PER)

**Example 6** (doc_id: `64429`) (sent_id: `64429`)


Als Beweis für ihren diesbezüglichen Vortrag zur offenkundigen Vorbenutzung hat die Einsprechende zuletzt nur noch den Zeugen H … angeboten ( vgl. Protokoll der mündlichen Verhandlung vom 04. 12. 2017 , S. 2 ) .

| Predicted | Gold |
|---|---|
| `H …` | `H …` |

**Example 7** (doc_id: `64657`) (sent_id: `64657`)


Sie hat insbesondere ausgesagt : Herr F … nahm das betreffende Kuvert und bestand darauf , das Kuvert selbst abzuliefern .

| Predicted | Gold |
|---|---|
| `F …` | `F …` |

**Example 8** (doc_id: `64795`) (sent_id: `64795`)


Weiterhin macht er geltend , Patentanwalt K … sei wegen einer psychischen Erkrankung zur Zeit der Zustellversuche des DPMA geschäftsunfähig nach § 104 Abs. 2 BGB gewesen , weshalb die Zustellungen unwirksam seien .

| Predicted | Gold |
|---|---|
| `K …` | `K …` |

**Missed by this rule (FN):**

- `DPMA` (ORG)
- `§ 104 Abs. 2 BGB` (NRM)

**Example 9** (doc_id: `65305`) (sent_id: `65305`)


c ) Damit erledigt sich der Antrag auf Gewährung von Prozesskostenhilfe und Beiordnung von Rechtsanwältin H … für das Verfahren auf Erlass einer einstweiligen Anordnung .

| Predicted | Gold |
|---|---|
| `H …` | `H …` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60393`) (sent_id: `60393`)


Anlage A4 Handzeichnung des Werkzeugs zum Zusammendrehen von Schraubentellerfedern zu der bei der R … GmbH & Co. KG geltend gemachten offenkundigen Vorbenutzung ,

**False Positives:**

- `R …` — partial — pred is substring of gold: `R … GmbH & Co. KG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `R … GmbH & Co. KG`(ORG)

**Example 1** (doc_id: `61011`) (sent_id: `61011`)


Das Geständnis des Beschwerdeführers , die inkriminierten Äußerungen stammten von ihm , bezieht sich nur auf den Blogeintrag und ist daher für den ebenfalls vom Anfangsverdacht umfassten Kommentar auf der Webseite " D … " unbeachtlich .

**False Positives:**

- `D …` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `61414`) (sent_id: `61414`)


Mit Schreiben vom 3. Juni 2014 machte die Beklagte gegenüber dem japanischen Tochterunternehmen des M … geltend , dass das Arzneimittel Isen- tress in den Schutzbereich des zur Familie des Streitpatents gehörenden japanischen Patents JP 5 207 392 falle .

**False Positives:**

- `M …` — type mismatch — same span as gold: `M …`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `M …`(ORG)

**Example 3** (doc_id: `61559`) (sent_id: `61559`)


Ferner reicht sie einen Handelsregisterauszug der K … GmbH zu den Akten .

**False Positives:**

- `K …` — partial — pred is substring of gold: `K … GmbH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K … GmbH`(ORG)

**Example 4** (doc_id: `61830`) (sent_id: `61830`)


Ferner hat die Einsprechende das „ Gutachten zum Sicherheitsschalter CES der Firma E … “ des Herrn Prof. Dr. - Ing. G … vom 15. September 2014 vorgelegt .

**False Positives:**

- `E …` — type mismatch — same span as gold: `E …`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `E …`(ORG)
- `G …`(PER)

**Example 5** (doc_id: `62675`) (sent_id: `62675`)


Die Einsprechende legt zur geltend gemachten Lieferung des Kontaktsockels „ Waffle Kelvin “ an die A … Inc. die teilgeschwärzte Ablichtung einer Rechnung vor :

**False Positives:**

- `A …` — partial — pred is substring of gold: `A … Inc.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Kelvin`(PER)
- `A … Inc.`(ORG)

**Example 6** (doc_id: `62911`) (sent_id: `62911`)


E4 : Modular Safety Controller System UE410 FLEXI . User Manual . S … AG - Industrial Safety Systems - Germany - All rightsreserved . 8011509 / ÄND / 06 - 05- 19 .

**False Positives:**

- `S …` — partial — pred is substring of gold: `S … AG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S … AG`(ORG)
- `Germany`(LOC)

**Example 7** (doc_id: `63141`) (sent_id: `63141`)


Nach Auffassung der Klägerin zu 2. ist die Priorität bereits formalrechtlich nicht wirksam in Anspruch genommen , da die Prioritätsanmeldung US 60/132036 bzw. das aus dieser Anmeldung folgende Prioritätsrecht nicht nachweislich innerhalb des Prioritätsjahres von den Erfindern als Voranmelder auf die L … LLC als Nachanmelderin übergegangen sei .

**False Positives:**

- `L …` — partial — pred is substring of gold: `L … LLC`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `L … LLC`(ORG)

**Example 8** (doc_id: `63599`) (sent_id: `63599`)


E4c : Rechnung der S … GmbH , Nummer 9090503004 vom 19. 10. 2016 an die P … GmbH , D … . in M … .

**False Positives:**

- `S …` — partial — pred is substring of gold: `S … GmbH`
- `P …` — partial — pred is substring of gold: `P … GmbH , D …`
- `D …` — partial — pred is substring of gold: `P … GmbH , D …`
- `M …` — type mismatch — same span as gold: `M …`

> overlaps gold: 4  |  likely missing annotation: 0

**Gold Entities:**

- `S … GmbH`(ORG)
- `P … GmbH , D …`(ORG)
- `M …`(LOC)

**Example 9** (doc_id: `63826`) (sent_id: `63826`)


Die Widerspruchsmarke sei namentlich im Blick auf den Vertrieb der „ Arrow “ -Kollektion bekannt , den die Widersprechende zusammen mit dem Lizenznehmer S … betreibe .

**False Positives:**

- `S …` — type mismatch — same span as gold: `S …`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S …`(ORG)

**Example 10** (doc_id: `65549`) (sent_id: `65549`)


Zur geltend gemachten Lieferung des Kontaktsockels „ Waffle Kelvin “ an die Firma A … Inc. hat die Einsprechende ebenfalls Zeugenbeweis ange- boten und verschiedene Dokumente eingereicht :

**False Positives:**

- `A …` — partial — pred is substring of gold: `A … Inc.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Kelvin`(PER)
- `A … Inc.`(ORG)

**Example 11** (doc_id: `66640`) (sent_id: `66640`)


Der Beschwerdeführer stehe im Verdacht , am 22. Oktober 2014 auf der Webseite " D … " den folgenden Kommentar veröffentlicht zu haben :

**False Positives:**

- `D …` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Initials with surname`

**F1:** 0.054 | **Precision:** 0.098 | **Recall:** 0.037  

**Format:** `regex`  
**Rule ID:** `40a4571f`  
**Description:**
Captures patterns like 'K. Schmidt' or 'M. P. Borom' where an initial (with dot) is followed by a surname.

**Content:**
```
\b([A-Z]\.)\s+([A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:-[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)?)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.098 | 0.037 | 0.054 | 122 | 12 | 110 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 12 | 110 | 312 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60446`) (sent_id: `60446`)


K. Schmidt

| Predicted | Gold |
|---|---|
| `K. Schmidt` | `K. Schmidt` |

**Example 1** (doc_id: `62502`) (sent_id: `62502`)


W. Reinfelder

| Predicted | Gold |
|---|---|
| `W. Reinfelder` | `W. Reinfelder` |

**Example 2** (doc_id: `62684`) (sent_id: `62684`)


M. Rennpferdt

| Predicted | Gold |
|---|---|
| `M. Rennpferdt` | `M. Rennpferdt` |

**Example 3** (doc_id: `63862`) (sent_id: `63862`)


Die durch sie erlaubten Kollektivbestrafungen werden von den Behörden im Nordkaukasus bereits angewendet ( Österreichisches Bundesamt für Fremdenwesen und Asyl , Länderinformationsblatt der Staatendokumentation Russische Föderation , Gesamtaktualisierung am 1. Juni 2016 , S. 34 ; Schweizerische Flüchtlingshilfe / A. Schuster , Russland : Verfolgung von Verwandten dagestanischer Terrorverdächtiger ausserhalb Dagestans , Auskunft vom 25. Juli 2014 , S. 4 f. ) .

| Predicted | Gold |
|---|---|
| `A. Schuster` | `A. Schuster` |

**Missed by this rule (FN):**

- `Nordkaukasus` (LOC)
- `Österreichisches Bundesamt für Fremdenwesen und Asyl` (ORG)
- `Russische Föderation` (LOC)
- `Schweizerische Flüchtlingshilfe` (ORG)
- `Russland` (LOC)
- `Dagestans` (LOC)

**Example 4** (doc_id: `63901`) (sent_id: `63901`)


K. Schmidt

| Predicted | Gold |
|---|---|
| `K. Schmidt` | `K. Schmidt` |

**Example 5** (doc_id: `63927`) (sent_id: `63927`)


M. Trümner

| Predicted | Gold |
|---|---|
| `M. Trümner` | `M. Trümner` |

**Example 6** (doc_id: `64317`) (sent_id: `64317`)


K. Schmidt

| Predicted | Gold |
|---|---|
| `K. Schmidt` | `K. Schmidt` |

**Example 7** (doc_id: `64439`) (sent_id: `64439`)


Vor diesem Hintergrund vermag der Senat auch aus der Auskunft der Schweizerischen Flüchtlingshilfe vom 25. Juli 2014 ( A. Schuster , Russland : Verfolgung von Verwandten dagestanischer Terrorverdächtiger ausserhalb Dagestans , S. 3 f. ) nicht abzuleiten , dass dem Kläger in der Russischen Föderation außerhalb des Nordkaukasus mit beachtlicher Wahrscheinlichkeit eine Art. 3 EMRK zuwiderlaufende Behandlung drohen würde .

| Predicted | Gold |
|---|---|
| `A. Schuster` | `A. Schuster` |

**Missed by this rule (FN):**

- `Schweizerischen Flüchtlingshilfe` (ORG)
- `Russland` (LOC)
- `Dagestans` (LOC)
- `Russischen Föderation` (LOC)
- `Nordkaukasus` (LOC)
- `Art. 3 EMRK` (NRM)

**Example 8** (doc_id: `64693`) (sent_id: `64693`)


K. Schmidt

| Predicted | Gold |
|---|---|
| `K. Schmidt` | `K. Schmidt` |

**Example 9** (doc_id: `64861`) (sent_id: `64861`)


J. Ratayczak

| Predicted | Gold |
|---|---|
| `J. Ratayczak` | `J. Ratayczak` |

**Example 10** (doc_id: `65286`) (sent_id: `65286`)


D14 J. Deubener et al. , " Induction time analysis of nucleation and crystal growth in di- and metasilicate glasses " , Journal of Non-Crystalline Solids 1993 , 163 , Seiten 1 bis 12 ,

| Predicted | Gold |
|---|---|
| `J. Deubener` | `J. Deubener` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60068`) (sent_id: `60068`)


I. Die Klägerin und Revisionsbeklagte ( Klägerin ) , eine GmbH , war in den Jahren 2009 bis 2012 ( Streitjahre ) als Reiseveranstalterin unternehmerisch tätig .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `60117`) (sent_id: `60117`)


I. Die Befristungskontrollklage ist unbegründet .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `60238`) (sent_id: `60238`)


V. Die Klage ist nicht abweisungsreif ( vgl. § 563 Abs. 3 ZPO ) .

**False Positives:**

- `V. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 563 Abs. 3 ZPO`(NRM)

**Example 3** (doc_id: `60477`) (sent_id: `60477`)


I. Die Würdigung des Landesarbeitsgerichts , das beklagte Königreich sei im vorliegenden Rechtsstreit grundsätzlich nicht der deutschen Gerichtsbarkeit unterworfen , sondern genieße - sollte es darauf nicht verzichtet haben - Staatenimmunität , ist revisionsrechtlich nicht zu beanstanden .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `60549`) (sent_id: `60549`)


Die zivilgerichtliche Rechtsprechung wende im Rahmen von § 315 BGB materielle , die Äquivalenz der Leistungen betreffende Kriterien an , die in den Bestimmungen der Richtlinie 2001 / 14 / EG nicht vorgesehen seien ( a. a. O. Rn. 72 ) .

**False Positives:**

- `O. Rn` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 315 BGB`(NRM)
- `Richtlinie 2001 / 14 / EG`(NRM)

**Example 5** (doc_id: `60609`) (sent_id: `60609`)


Daran gemessen war der Vertrag vom 30. März 1989 unabhängig davon , ob man ihn als - unzutreffend beurkundetes - mehrseitiges Rechtsgeschäft zwischen den Beigeladenen , den Eltern des Beigeladenen zu 2 und U. Sch. versteht oder ob man ihn als lediglich zwischen U. Sch. und den Beigeladenen geschlossenen Vertrag ansieht , der Redlichkeitsprüfung zugänglich .

**False Positives:**

- `U. Sch` — partial — pred is substring of gold: `U. Sch.`
- `U. Sch` — similar text (different position): `U. Sch.`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `U. Sch.`(PER)
- `U. Sch.`(PER)

**Example 6** (doc_id: `60693`) (sent_id: `60693`)


I. Die Antragsgegnerin und Beschwerdegegnerin ( im Folgenden : Antragsgegnerin ) war Inhaberin des am 4. Mai 2000 eingetragenen Gebrauchsmusters 298 20 129.1 ( Streitgebrauchsmuster ) mit der Bezeichnung „ … “ , das am 1. Dezember 2008 nach Erreichen der maximalen Schutzdauer erloschen war .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 7** (doc_id: `60767`) (sent_id: `60767`)


I. Mit dem angefochtenen Beschluss vom 15. Juli 2015 hat die Patentabteilung 1.25 des Deutschen Patent- und Markenamts das Patent DE 10 2008 017 350 mit der Bezeichnung „ Steuerung für Fahrmischer “ beschränkt aufrechterhalten .

**False Positives:**

- `I. Mit` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Patentabteilung 1.25 des Deutschen Patent- und Markenamts`(ORG)

**Example 8** (doc_id: `60783`) (sent_id: `60783`)


Dabei ist § 129 AO schon dann nicht anwendbar , wenn auch nur die ernsthafte Möglichkeit besteht , dass die Nichtbeachtung einer feststehenden Tatsache auf einer fehlerhaften Tatsachenwürdigung oder einem sonstigen sachverhaltsbezogenen Denk- oder Überlegungsfehler gründet oder auf mangelnder Sachverhaltsaufklärung beruht ( ständige Rechtsprechung , z.B. Senatsbeschluss vom 28. Mai 2015 VI R 63/13 , BFH / NV 2015 , 1078 , m. w. N. ) .

**False Positives:**

- `B. Senatsbeschluss` — positional overlap with gold: `Senatsbeschluss vom 28. Mai 2015 VI R 63/13 , BFH / NV 2015 , 1078`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 129 AO`(NRM)
- `Senatsbeschluss vom 28. Mai 2015 VI R 63/13 , BFH / NV 2015 , 1078`(RS)

**Example 9** (doc_id: `60926`) (sent_id: `60926`)


I. Die Kläger und Beschwerdeführer ( Kläger ) werden zusammen veranlagt .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 10** (doc_id: `61070`) (sent_id: `61070`)


I. Auf die am 30. Mai 2012 beim Deutschen Patent- und Markenamt eingereichte Patentanmeldung ist die Erteilung des Patents 10 2012 104 673 mit der Bezeichnung „ Werkzeug , System und Verfahren zum Verschrauben von Schraubendruckfedern zu einer Schraubentellerfeder “ am 14. August 2013 veröffentlicht worden .

**False Positives:**

- `I. Auf` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschen Patent- und Markenamt`(ORG)

**Example 11** (doc_id: `61076`) (sent_id: `61076`)


D4 M. P. Borom et al. , „ Strength and Microstructure in Lithium Disilicate Glass-Ceramics “ , Journal of the American Ceramic Society , 1975 , 58 , Seiten 385 bis 391 ,

**False Positives:**

- `P. Borom` — partial — pred is substring of gold: `M. P. Borom`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `M. P. Borom`(PER)

**Example 12** (doc_id: `61141`) (sent_id: `61141`)


In einer Auswerteeinheit würden die von einer externen Beschaltung – Signalgebern , wie z.B. Not-Aus-Tastern , Seilzugschaltern , Magnetschaltern , Positionsschaltern – stammenden Signale nach sicherheitstechnischen Vorschriften erfasst und verarbeitet .

**False Positives:**

- `B. Not-Aus` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 13** (doc_id: `61218`) (sent_id: `61218`)


C. Danach ist § 40 Abs. 1a LFGB insoweit mit Art. 12 Abs. 1 GG unvereinbar , als die Information der Öffentlichkeit nicht gesetzlich befristet ist .

**False Positives:**

- `C. Danach` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 40 Abs. 1a LFGB`(NRM)
- `Art. 12 Abs. 1 GG`(NRM)

**Example 14** (doc_id: `61272`) (sent_id: `61272`)


I. Nach § 72 Abs. 5 ArbGG iVm. § 551 Abs. 1 ZPO muss der Revisionskläger die Revision begründen .

**False Positives:**

- `I. Nach` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 72 Abs. 5 ArbGG`(NRM)
- `§ 551 Abs. 1 ZPO`(NRM)

**Example 15** (doc_id: `61319`) (sent_id: `61319`)


I. Der Feststellungsantrag ist zulässig .

**False Positives:**

- `I. Der` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 16** (doc_id: `61342`) (sent_id: `61342`)


I. Die vorliegende Patentanmeldung wurde am 26. Januar 2012 beim Deutschen Patent- und Markenamt eingereicht .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschen Patent- und Markenamt`(ORG)

**Example 17** (doc_id: `61353`) (sent_id: `61353`)


I. Die Anmelderin hat am 3. Januar 2013 beim Deutschen Patent- und Markenamt beantragt , die Bezeichnung A-ÖFFNER für die nachgenannten Waren und Dienstleistungen als Wortmarke in das Markenregister einzutragen :

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschen Patent- und Markenamt`(ORG)
- `A-ÖFFNER`(ORG)

**Example 18** (doc_id: `61516`) (sent_id: `61516`)


I. Der Kläger und Revisionskläger ( Kläger ) war in den Streitjahren ( 1995 bis 1997 ) u. a. als Steuerberater in einer Einzelkanzlei tätig .

**False Positives:**

- `I. Der` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `61557`) (sent_id: `61557`)


B. Die zulässige Rechtsbeschwerde des Betriebsrats ist unbegründet .

**False Positives:**

- `B. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 20** (doc_id: `61631`) (sent_id: `61631`)


b ) Beschäftigungsort i. S. des § 9 Abs. 1 Satz 3 Nr. 5 Satz 2 EStG ist der Ort der langfristig und dauerhaft angelegten Arbeitsstätte ( z.B. Senatsurteile vom 11. Mai 2005 VI R 7/02 , BFHE 209 , 502 , BStBl II 2005 , 782 , und VI R 34/04 , BFHE 209 , 527 , BStBl II 2005 , 793 , sowie vom 19. September 2012 VI R 78/10 , BFHE 239 , 80 , BStBl II 2013 , 284 ) .

**False Positives:**

- `B. Senatsurteile` — positional overlap with gold: `Senatsurteile vom 11. Mai 2005 VI R 7/02 , BFHE 209 , 502 , BStBl II 2005 , 782`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 9 Abs. 1 Satz 3 Nr. 5 Satz 2 EStG`(NRM)
- `Senatsurteile vom 11. Mai 2005 VI R 7/02 , BFHE 209 , 502 , BStBl II 2005 , 782`(RS)
- `VI R 34/04 , BFHE 209 , 527 , BStBl II 2005 , 793`(RS)
- `vom 19. September 2012 VI R 78/10 , BFHE 239 , 80 , BStBl II 2013 , 284`(RS)

**Example 21** (doc_id: `61784`) (sent_id: `61784`)


Dies ist zunächst dann der Fall , wenn das eingetragene Design Gestaltungen zum Gegenstand hat , bei denen es sich nicht um ein Erzeugnis im Sinne von § 1 Nr. 2 DesignG , d. h. um einen industriellen oder handwerklichen Gegenstand , bzw. um ein komplexes Erzeugnis im Sinne von § 1 Nr. 3 DesignG handelt , wie es z.B. bei anorganischen und organischen Naturprodukten , Menschen und Tieren , Verfahren und anderen Nichterzeugnissen aufgrund unkonkreter Gestalt , fehlender Sichtbarkeit oder auch einer dem Charakter eines ganzen Erzeugnisses widersprechenden Kombination von Gegenständen wie z.B. Backware und Uhr der Fall sein kann ( vgl. Eichmann / v. Falckenstein / Kühne , Designgesetz , 5. Aufl. , § 18 Rn. 2 ) .

**False Positives:**

- `B. Backware` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 1 Nr. 2 DesignG`(NRM)
- `§ 1 Nr. 3 DesignG`(NRM)
- `Eichmann / v. Falckenstein / Kühne , Designgesetz , 5. Aufl. , § 18 Rn. 2`(LIT)

**Example 22** (doc_id: `61798`) (sent_id: `61798`)


Zur Zeit ist die Beigeladene aufgrund des Anstellungsvertrags vom 18. / 27. Oktober 2015 bei der S. Gesellschaft als " Administrative Direktorin " beschäftigt .

**False Positives:**

- `S. Gesellschaft` — type mismatch — same span as gold: `S. Gesellschaft`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S. Gesellschaft`(ORG)

**Example 23** (doc_id: `61825`) (sent_id: `61825`)


D3 M. P. Borom et al. , “ Strength and Microstructure in Lithium Disilicate Glass-Ceramics ” , Journal of the American Ceramic Society , 1975 , 58 , Seiten 385 bis 391

**False Positives:**

- `P. Borom` — partial — pred is substring of gold: `M. P. Borom`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `M. P. Borom`(PER)

**Example 24** (doc_id: `61893`) (sent_id: `61893`)


I. Die Bezeichnung MAM Munich Asset Management ist am 16. März 2015 zur Eintragung als Wortmarke in das beim Deutschen Patent- und Markenamt ( DPMA ) geführte Register für folgende Dienstleistungen der Klassen 35 , 36 und 42 angemeldet worden :

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `MAM Munich Asset Management`(ORG)
- `Deutschen Patent- und Markenamt`(ORG)
- `DPMA`(ORG)

**Example 25** (doc_id: `61932`) (sent_id: `61932`)


V. Die Kostenentscheidung beruht auf § 90 Satz 2 EnWG , die Festsetzung des Gegenstandswerts auf § 50 Abs. 1 Satz 1 Nr. 2 GKG und § 3 ZPO .

**False Positives:**

- `V. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 90 Satz 2 EnWG`(NRM)
- `§ 50 Abs. 1 Satz 1 Nr. 2 GKG`(NRM)
- `§ 3 ZPO`(NRM)

**Example 26** (doc_id: `62040`) (sent_id: `62040`)


I. Die in § 33 Abs. 2 Satz 1 TV DRV KBS geregelte auflösende Bedingung gilt nicht nach §§ 21 , 17 Satz 2 TzBfG iVm. § 7 Halbs. 1 KSchG als wirksam und eingetreten .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 33 Abs. 2 Satz 1 TV DRV KBS`(REG)
- `§§ 21 , 17 Satz 2 TzBfG`(NRM)
- `§ 7 Halbs. 1 KSchG`(NRM)

**Example 27** (doc_id: `62109`) (sent_id: `62109`)


A. Die Richtervorlage betrifft die Frage , ob § 1906 Abs. 3 BGB in der Fassung des Gesetzes zur Regelung der betreuungsrechtlichen Einwilligung in eine ärztliche Zwangsmaßnahme vom 18. Februar 2013 ( BGBl I S. 266 ) mit Art. 3 Abs. 1 GG vereinbar ist , soweit er ärztliche Zwangsmaßnahmen außerhalb eines stationären Aufenthalts in einem Krankenhaus ausschließt .

**False Positives:**

- `A. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 1906 Abs. 3 BGB`(NRM)
- `Gesetzes zur Regelung der betreuungsrechtlichen Einwilligung in eine ärztliche Zwangsmaßnahme vom 18. Februar 2013 ( BGBl I S. 266 )`(NRM)
- `Art. 3 Abs. 1 GG`(NRM)

**Example 28** (doc_id: `62118`) (sent_id: `62118`)


I. Die von der Beschwerdeführerin als gleichheitswidrig beanstandeten Regelungen durch den von ihr mittelbar angegriffenen § 7 Satz 2 Nr. 2 GewStG sind verfassungsgemäß ; der Gesetzgeber bewegt sich mit dieser Neuregelung des Jahres 2002 im Rahmen seiner Gestaltungsbefugnis .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 7 Satz 2 Nr. 2 GewStG`(NRM)

**Example 29** (doc_id: `62176`) (sent_id: `62176`)


I. Der Kläger und Revisionsbeklagte ( Kläger ) war im Jahr 2011 ( Streitjahr ) Eigentümer des Grundstücks in X , Y-Straße ... ( Grundstück ) , das er bis März 2020 steuerpflichtig an die A ( Pächterin ) verpachtet hatte .

**False Positives:**

- `I. Der` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `X`(LOC)
- `Y-Straße ...`(LOC)
- `A`(PER)

**Example 30** (doc_id: `62232`) (sent_id: `62232`)


D. Der Kläger hat gem. § 97 Abs. 1 ZPO die Kosten seiner erfolglosen Revision zu tragen .

**False Positives:**

- `D. Der` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 97 Abs. 1 ZPO`(NRM)

**Example 31** (doc_id: `62353`) (sent_id: `62353`)


Als Beendigung der Rechtsfähigkeit des Betriebs ist der 3. 7. 1990 , als Rechtsnachfolger sind die Electronicon-GmbH G. und die B. Kondensatoren-GmbH eingetragen .

**False Positives:**

- `B. Kondensatoren` — partial — pred is substring of gold: `B. Kondensatoren-GmbH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Electronicon-GmbH G.`(ORG)
- `B. Kondensatoren-GmbH`(ORG)

**Example 32** (doc_id: `62356`) (sent_id: `62356`)


Der Vorlagebeschluss geht daher davon aus , dass der Präsident dem Kanzler als Dienstvorgesetzter auch Einzelanweisungen erteilen kann ( BVerwG , Beschluss vom 23. Juni 2016 - 2 C 1.15 - , juris , Rn. 84 ; a. A. Sandberger , WissR 44 [ 2011 ] , S. 118 < 148 > ) .

**False Positives:**

- `A. Sandberger` — positional overlap with gold: `Sandberger , WissR 44 [ 2011 ] , S. 118 < 148 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerwG , Beschluss vom 23. Juni 2016 - 2 C 1.15 - , juris , Rn. 84`(RS)
- `Sandberger , WissR 44 [ 2011 ] , S. 118 < 148 >`(LIT)

**Example 33** (doc_id: `62413`) (sent_id: `62413`)


I. Der Kläger und Revisionskläger ( Kläger ) war im Streitjahr ( 2004 ) Steuerberater .

**False Positives:**

- `I. Der` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 34** (doc_id: `62461`) (sent_id: `62461`)


I. Die angegriffene farbige Wort- / Bildmarke ist am 9. Juli 2012 angemeldet und am 15. Oktober 2012 in das beim Deutschen Patent- und Markenamt ( DPMA ) geführte Register für die Dienstleistungen der

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschen Patent- und Markenamt`(ORG)
- `DPMA`(ORG)

**Example 35** (doc_id: `62527`) (sent_id: `62527`)


2. An der Besteuerung wird Deutschland durch das DBA-Kanada 2001 nicht gehindert ; der Senat hält insoweit an der in einem Verfahren des einstweiligen Rechtsschutzes bereits geäußerten Rechtsauffassung ( Senatsbeschluss vom 13. Dezember 2011 I B 159/11 , BFH / NV 2012 , 417 ) auch nach erneuter Prüfung fest ( dieser Rechtslage zustimmend z.B. Gosch in Kirchhof , EStG , 16. Aufl. , § 49 Rz 90 mit Fußn. 6 ; Hick in Schönfeld / Ditz , DBA , Art. 18 Rz 167 f. , 171 , 174 ; Blümich / Wied , § 49 EStG Rz 218 ; Kuhn in Herrmann / Heuer / Raupach , § 49 EStG Rz 1003 ; Kühnen , EFG 2016 , 578 ; Schober , EFG 2016 , 990 ; Ismer in Vogel / Lehner , DBA , 6. Aufl. , Art. 18 Rz 87 ; Holthaus , Internationale Wirtschaftsbriefe 2017 , 796 , 797 ; im Ergebnis ebenso die Verwaltungspraxis , s. Bayerisches Landesamt für Steuern , Verfügung vom 8. Juni 2011 , Internationales Steuerrecht 2011 , 776 ; a. A. W. Wassermeyer in Wassermeyer , DBA , Kanada Art. 18 Rz 70a ; Hagemann / Kahlenberg / Cloer , Betriebs-Berater 2017 , 2775 , 2785 f. ) .

**False Positives:**

- `B. Gosch` — positional overlap with gold: `Gosch in Kirchhof , EStG , 16. Aufl. , § 49 Rz 90 mit Fußn. 6`
- `W. Wassermeyer` — partial — pred is substring of gold: `W. Wassermeyer in Wassermeyer , DBA , Kanada Art. 18 Rz 70a`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Deutschland`(LOC)
- `DBA-Kanada 2001`(NRM)
- `Senatsbeschluss vom 13. Dezember 2011 I B 159/11 , BFH / NV 2012 , 417`(RS)
- `Gosch in Kirchhof , EStG , 16. Aufl. , § 49 Rz 90 mit Fußn. 6`(LIT)
- `Hick in Schönfeld / Ditz , DBA , Art. 18 Rz 167 f. , 171 , 174`(LIT)
- `Blümich / Wied , § 49 EStG Rz 218`(LIT)
- `Kuhn in Herrmann / Heuer / Raupach , § 49 EStG Rz 1003`(LIT)
- `Kühnen , EFG 2016 , 578`(LIT)
- `Schober , EFG 2016 , 990`(LIT)
- `Ismer in Vogel / Lehner , DBA , 6. Aufl. , Art. 18 Rz 87`(LIT)
- `Holthaus , Internationale Wirtschaftsbriefe 2017 , 796 , 797`(LIT)
- `Bayerisches Landesamt für Steuern , Verfügung vom 8. Juni 2011 , Internationales Steuerrecht 2011 , 776`(LIT)
- `W. Wassermeyer in Wassermeyer , DBA , Kanada Art. 18 Rz 70a`(LIT)
- `Hagemann / Kahlenberg / Cloer , Betriebs-Berater 2017 , 2775 , 2785 f.`(LIT)

**Example 36** (doc_id: `62565`) (sent_id: `62565`)


Es bestehe auch kein Bedürfnis für einen Schutz von Teilen oder Elementen eines Geschmacksmusters , da es möglich sei , auch für die Erscheinungsform von Teilen oder Elementen eines Erzeugnisses den Schutz als Geschmacksmuster zu erlangen ( a. a. O. Nr. 39 ) .

**False Positives:**

- `O. Nr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 37** (doc_id: `62674`) (sent_id: `62674`)


1. Zur Begründung seiner Entscheidung hat das FG u. a. ausgeführt , eine tarifbegünstigte Entschädigung könne nur angenommen werden , wenn das dem weggefallenen Anspruch zugrunde liegende Rechtsverhältnis vollständig beendet sei ( z.B. Urteil des Bundesfinanzhofs - BFH - vom 6. März 2002 XI R 36/01 , BFH / NV 2002 , 1144 ) .

**False Positives:**

- `B. Urteil` — positional overlap with gold: `Urteil des Bundesfinanzhofs - BFH - vom 6. März 2002 XI R 36/01 , BFH / NV 2002 , 1144`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Bundesfinanzhofs - BFH - vom 6. März 2002 XI R 36/01 , BFH / NV 2002 , 1144`(RS)

**Example 38** (doc_id: `62682`) (sent_id: `62682`)


I. Streitig ist , ob ein Auflösungsverlust nach § 17 Abs. 4 des Einkommensteuergesetzes ( EStG ) im Veranlagungszeitraum 2011 entstanden ist , der im Wege des Verlustrücktrags im Streitjahr 2010 vom Gesamtbetrag der Einkünfte abgezogen werden soll .

**False Positives:**

- `I. Streitig` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 17 Abs. 4 des Einkommensteuergesetzes`(NRM)
- `EStG`(NRM)

**Example 39** (doc_id: `62775`) (sent_id: `62775`)


Der Charakter einer Sachangabe entfällt bei der Zusammenfügung beschreibender Begriffe jedoch dann , wenn die beschreibenden Angaben durch die Kombination eine ungewöhnliche Änderung erfahren , die hinreichend weit von der Sachangabe wegführt ( EuGH MarkenR 2007 , 204 Rdnr. 77 f. – CELLTECH ; BGH a. a. O. Rdnr. 16 – DüsseldorfCongress ) .

**False Positives:**

- `O. Rdnr` — partial — pred is substring of gold: `BGH a. a. O. Rdnr. 16 – DüsseldorfCongress`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH MarkenR 2007 , 204 Rdnr. 77 f. – CELLTECH`(RS)
- `BGH a. a. O. Rdnr. 16 – DüsseldorfCongress`(RS)

**Example 40** (doc_id: `62805`) (sent_id: `62805`)


I. Die Klage ist zulässig , die Klageanträge bedürfen jedoch der Auslegung .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 41** (doc_id: `62878`) (sent_id: `62878`)


I. Die Verfassungsbeschwerde betrifft eine Entscheidung über den von der Beschwerdeführerin geltend gemachten Anspruch auf Zugewinnausgleich .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 42** (doc_id: `62931`) (sent_id: `62931`)


Deshalb können nur solche Aufwendungen als Werbungskosten i. S. des § 9 Abs. 1 EStG abgezogen werden , welche die persönliche Leistungsfähigkeit des Steuerpflichtigen mindern ( ständige Rechtsprechung , z.B. Beschluss des Großen Senats des BFH in BFHE 189 , 160 , BStBl II 1999 , 782 , unter C. IV. 1. b ; BFH-Urteil vom 15. November 2005 IX R 25/03 , BFHE 211 , 318 , BStBl II 2006 , 623 , m. w. N. ; Senatsurteil vom 13. März 1996 VI R 103/95 , BFHE 180 , 139 , BStBl II 1996 , 375 ) .

**False Positives:**

- `B. Beschluss` — positional overlap with gold: `Beschluss des Großen Senats des BFH in BFHE 189 , 160 , BStBl II 1999 , 782 , unter C. IV. 1. b`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 9 Abs. 1 EStG`(NRM)
- `Beschluss des Großen Senats des BFH in BFHE 189 , 160 , BStBl II 1999 , 782 , unter C. IV. 1. b`(RS)
- `BFH-Urteil vom 15. November 2005 IX R 25/03 , BFHE 211 , 318 , BStBl II 2006 , 623`(RS)
- `Senatsurteil vom 13. März 1996 VI R 103/95 , BFHE 180 , 139 , BStBl II 1996 , 375`(RS)

**Example 43** (doc_id: `63134`) (sent_id: `63134`)


I. Die Ablehnungsgesuche sind unzulässig , weil kein Befangenheitsgrund dargelegt wurde .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 44** (doc_id: `63229`) (sent_id: `63229`)


So sind regelmäßig auch rechtsgeschäftliche Lizenzen kündbar oder können bei Wegfall des Patents und damit der Geschäftsgrundlage angepasst werden ( vgl. z.B. Keukenschrijver , Patentnichtigkeitsverfahren , 6. Aufl. , Rn. 405 ) .

**False Positives:**

- `B. Keukenschrijver` — positional overlap with gold: `Keukenschrijver , Patentnichtigkeitsverfahren , 6. Aufl. , Rn. 405`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Keukenschrijver , Patentnichtigkeitsverfahren , 6. Aufl. , Rn. 405`(LIT)

**Example 45** (doc_id: `63308`) (sent_id: `63308`)


c. Der Betriebsprüfungsbescheid vom 23. 12. 2008 ist auch inhaltlich hinreichend bestimmt iS des § 33 Abs 1 SGB X. Aus dem streitigen Bescheid ergibt sich eindeutig der Adressat des Bescheids - die Klägerin - ebenso wie die an diese gerichtete Aufforderung , insgesamt 251 604,84 Euro an Beiträgen zur GRV an die jeweiligen Einzugsstellen nachzuzahlen ; gleiches gilt für die Änderungsbescheide .

**False Positives:**

- `X. Aus` — positional overlap with gold: `§ 33 Abs 1 SGB X.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 33 Abs 1 SGB X.`(NRM)

**Example 46** (doc_id: `63413`) (sent_id: `63413`)


C. Das Landesarbeitsgericht hat die gegen die Beendigung des Arbeitsverhältnisses der Parteien durch die außerordentliche Kündigung der Beklagten vom 28. Juli 2016 gerichtete Kündigungsschutzklage zu Recht abgewiesen .

**False Positives:**

- `C. Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 47** (doc_id: `63498`) (sent_id: `63498`)


Nach der gemäß § 163 SGG den Senat bindenden Auslegung des Landesrechts stehen damit die folgenden rechtlichen Beschränkungen fest : Das Flurstück des Klägers ist gemäß § 2 Abs 1 LandesVO L. K. Bestandteil dieses Naturschutzgebietes .

**False Positives:**

- `K. Bestandteil` — positional overlap with gold: `§ 2 Abs 1 LandesVO L. K.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 163 SGG`(NRM)
- `§ 2 Abs 1 LandesVO L. K.`(NRM)

**Example 48** (doc_id: `63514`) (sent_id: `63514`)


I. Gegen die Eintragung der für die Waren und Dienstleistungen

**False Positives:**

- `I. Gegen` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 49** (doc_id: `63549`) (sent_id: `63549`)


I. Das Landesarbeitsgericht ist mit einer rechtsfehlerhaften Begründung zu dem Ergebnis gelangt , die Befristung sei nach § 14 Abs. 1 Satz 2 Nr. 4 TzBfG gerechtfertigt .

**False Positives:**

- `I. Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 14 Abs. 1 Satz 2 Nr. 4 TzBfG`(NRM)

**Example 50** (doc_id: `63552`) (sent_id: `63552`)


Sie ergibt sich aber aus der Verwaltungspraxis innerhalb der Abteilung X. Nach der schriftsätzlichen Schilderung der Beklagten ging die weitere Vertretung in der Abteilung X " traditionell " auf denjenigen Referatsleiter der Abteilung über , der die Referatsleiterstellung innerhalb der Abteilung am längsten innehatte .

**False Positives:**

- `X. Nach` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 51** (doc_id: `63569`) (sent_id: `63569`)


Für diese zusätzlichen Voraussetzungen fehlt eine gesetzliche Grundlage ( gl. A. Bott , DStZ 2015 , 112 , 122 ; Bott / Schiffers , DStZ 2013 , 886 , 900 ) .

**False Positives:**

- `A. Bott` — positional overlap with gold: `Bott , DStZ 2015 , 112 , 122`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bott , DStZ 2015 , 112 , 122`(LIT)
- `Bott / Schiffers , DStZ 2013 , 886 , 900`(LIT)

**Example 52** (doc_id: `63597`) (sent_id: `63597`)


Erst recht nicht erkennbar war der Umstand , dass zumindest die beiden an der privatschriftlichen Vereinbarung vom Oktober 1987 beteiligten Vertragspartner , nämlich U. Sch. und der Beigeladene zu 2 , von einer zusätzlichen Verpflichtung zum Ausgleich der Differenz des Taxwertes beider Grundstücke ausgingen .

**False Positives:**

- `U. Sch` — partial — pred is substring of gold: `U. Sch.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `U. Sch.`(PER)

**Example 53** (doc_id: `63626`) (sent_id: `63626`)


Nach dem Tod von E. Sch. wurde sein Enkel U. Sch. 1976 oder 1978 als Eigentümer des Grundstücks R. straße ... im Grundbuch eingetragen .

**False Positives:**

- `E. Sch` — partial — pred is substring of gold: `E. Sch.`
- `U. Sch` — partial — pred is substring of gold: `U. Sch.`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `E. Sch.`(PER)
- `U. Sch.`(PER)
- `R. straße ...`(LOC)

**Example 54** (doc_id: `63691`) (sent_id: `63691`)


I. Die Beschwerde der Antragstellerin wird zurückgewiesen .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 55** (doc_id: `63750`) (sent_id: `63750`)


I. Die Klage ist zulässig , insbesondere hinreichend bestimmt iSv. § 253 Abs. 2 Nr. 2 ZPO .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 253 Abs. 2 Nr. 2 ZPO`(NRM)

**Example 56** (doc_id: `63877`) (sent_id: `63877`)


I. Mit Beschluss vom 25. September 2017 X B 79/17 hatte der Senat zum einen eine Beschwerde des Kostenschuldners , Erinnerungsführers und Rügeführers ( Rügeführer ) gegen die Verwerfung einer Anhörungsrüge durch das Finanzgericht als unzulässig seinerseits als unzulässig verworfen , zum anderen eine Beschwerde gegen die Ablehnung eines Antrags auf Akteneinsicht als unbegründet zurückgewiesen und die Kosten des Beschwerdeverfahrens dem Rügeführer auferlegt .

**False Positives:**

- `I. Mit` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Beschluss vom 25. September 2017 X B 79/17`(RS)

**Example 57** (doc_id: `63920`) (sent_id: `63920`)


I. Auf die am 3. Dezember 2007 eingereichte Anmeldung ist mit Beschluss vom 18. Januar 2010 das Patent 10 2007 058 365 mit der Bezeichnung „ Kontaktierungseinheit zur Kontaktierung von Anschlusskontakten elektronischer Bauelemente “ erteilt worden .

**False Positives:**

- `I. Auf` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 58** (doc_id: `64107`) (sent_id: `64107`)


Die berührungslose Übergabe von Signalen von einem Bedienhebel an eine Steuerung stellt allgemeines Fachwissen dar ( z.B. Joysticks im Computerbereich oder bei der Steuerung von Baugeräten oder Werkzeugmaschinen ) und dient der Vermeidung einer mechanischen Verbindung von Bedienhebel und Steuerung , die sonst häufig einem hohen Verschleiß durch eindringenden Schmutz und Feuchtigkeit unterliegt .

**False Positives:**

- `B. Joysticks` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 59** (doc_id: `64117`) (sent_id: `64117`)


Durch den bewusst unterlassenen Abgleich der der Steuererklärung elektronisch beigestellten Daten mit den vom Steuerpflichtigen erklärten Daten liegt insbesondere kein bloßes Übersehen erklärter Daten vor , das regelmäßig zu einer Berichtigungsmöglichkeit nach § 129 AO führt ( z.B. Senatsurteile vom 29. März 1985 VI R 140/81 , BFHE 144 , 118 , BStBl II 1985 , 569 , und in BFH / NV 1989 , 619 ) .

**False Positives:**

- `B. Senatsurteile` — positional overlap with gold: `Senatsurteile vom 29. März 1985 VI R 140/81 , BFHE 144 , 118 , BStBl II 1985 , 569`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 129 AO`(NRM)
- `Senatsurteile vom 29. März 1985 VI R 140/81 , BFHE 144 , 118 , BStBl II 1985 , 569`(RS)
- `BFH / NV 1989 , 619`(RS)

**Example 60** (doc_id: `64124`) (sent_id: `64124`)


" Anfang 2009 habe ihn ein P. -Mitarbeiter [ Mitarbeiter der P. AG ] gebeten , spätabends zum Seiteneingang der H. -Zentrale in der H. Innenstadt zu kommen , um einen heiklen Spezialauftrag auszuführen .

**False Positives:**

- `H. Innenstadt` — type mismatch — same span as gold: `H. Innenstadt`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `P. AG`(ORG)
- `H. -Zentrale`(ORG)
- `H. Innenstadt`(LOC)

**Example 61** (doc_id: `64328`) (sent_id: `64328`)


Damit geht es hier - entgegen der seitens des Landes Brandenburg vertretenen Auffassung - nicht bloß um ein Begehren des Klägers auf lebenszeitige Übertragung eines konkreten funktionellen Amtes ( a. A. Wolff , ZBR 2017 , S. 239 < 241 > ) , sondern eines Amtes im statusrechtlichen Sinne .

**False Positives:**

- `A. Wolff` — positional overlap with gold: `Wolff , ZBR 2017 , S. 239 < 241 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Brandenburg`(LOC)
- `Wolff , ZBR 2017 , S. 239 < 241 >`(LIT)

**Example 62** (doc_id: `64443`) (sent_id: `64443`)


Mit der Zustimmung des Mieters , die als Annahme eines solchen Änderungsantrags zu werten ist ( MünchKommBGB / Artz , aaO ; Staudinger / V. Emmerich , aaO ; jeweils mwN ) , kommt eine den bisherigen Mietvertrag abändernde Mieterhöhungsvereinbarung zustande ( Senatsurteil vom 10. November 2010 - VIII ZR 300/09 , NJW 2011 , 295 Rn. 14 ) .

**False Positives:**

- `V. Emmerich` — partial — pred is substring of gold: `Staudinger / V. Emmerich , aaO`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `MünchKommBGB / Artz , aaO`(LIT)
- `Staudinger / V. Emmerich , aaO`(LIT)
- `Senatsurteil vom 10. November 2010 - VIII ZR 300/09 , NJW 2011 , 295 Rn. 14`(RS)

**Example 63** (doc_id: `64534`) (sent_id: `64534`)


g ) Im Streitfall kann auf sich beruhen , ob es der Übertragung des Erbbaurechts am Ende des Vertragszeitraums gleichzustellen ist , wenn die vereinbarte Kooperationsdauer in der ÖPP und die Laufzeit des Erbbaurechts übereinstimmen und das Erbbaurecht daher gemäß § 27 Abs. 1 Satz 1 der am 1. Januar 2006 geltenden Verordnung über das Erbbaurecht ( jetzt § 27 Abs. 1 Satz 1 des Erbbaurechtsgesetzes ) am Ende des Vertragszeitraums erlischt ( so zu § 4 Nr. 5 GrEStG Viskorf in Boruttau , Grunderwerbsteuergesetz , 18. Aufl. , § 4 Rz 54 ; Hofmann , Grunderwerbsteuergesetz , Kommentar , 11. Aufl. , § 4 Rz 18 ; Pahlke , Grunderwerbsteuergesetz , Kommentar , 5. Aufl. , § 4 Rz 41 ; a. A. Troll / Eisele , Grundsteuergesetz , Kommentar , 11. Aufl. , § 3 Rz 60a ) .

**False Positives:**

- `A. Troll` — positional overlap with gold: `Troll / Eisele , Grundsteuergesetz , Kommentar , 11. Aufl. , § 3 Rz 60a`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 27 Abs. 1 Satz 1 der am 1. Januar 2006 geltenden Verordnung über das Erbbaurecht`(NRM)
- `§ 27 Abs. 1 Satz 1 des Erbbaurechtsgesetzes`(NRM)
- `§ 4 Nr. 5 GrEStG`(NRM)
- `Viskorf in Boruttau , Grunderwerbsteuergesetz , 18. Aufl. , § 4 Rz 54`(LIT)
- `Hofmann , Grunderwerbsteuergesetz , Kommentar , 11. Aufl. , § 4 Rz 18`(LIT)
- `Pahlke , Grunderwerbsteuergesetz , Kommentar , 5. Aufl. , § 4 Rz 41`(LIT)
- `Troll / Eisele , Grundsteuergesetz , Kommentar , 11. Aufl. , § 3 Rz 60a`(LIT)

**Example 64** (doc_id: `64575`) (sent_id: `64575`)


I. Mit seiner Verfassungsbeschwerde wendet sich der Beschwerdeführer gegen einen Sorgerechtsentzug nach § 1666 BGB für seine beiden minderjährigen Kinder in einem einstweiligen Rechtsschutzverfahren .

**False Positives:**

- `I. Mit` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 1666 BGB`(NRM)

**Example 65** (doc_id: `64614`) (sent_id: `64614`)


Hierbei handelt es sich um einen Antrag ( § 145 BGB ) auf Abschluss eines Änderungsvertrages ( Palandt / Weidenkaff , BGB , 77. Aufl. , § 558b Rn. 3 ; § 558a Rn. 2 ; Staudinger / V. Emmerich , BGB , Neubearb. 2018 , § 558a Rn. 2 ; § 558b Rn. 3 ; MünchKommBGB / Artz , BGB , 7. Aufl. , § 558b Rn. 3 ; vgl. auch BayObLG , NJW-RR 1993 , 202 mwN [ zu § 2 MHG ] ) .

**False Positives:**

- `V. Emmerich` — partial — pred is substring of gold: `Staudinger / V. Emmerich , BGB , Neubearb. 2018 , § 558a Rn. 2 ; § 558b Rn. 3`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 145 BGB`(NRM)
- `Palandt / Weidenkaff , BGB , 77. Aufl. , § 558b Rn. 3 ; § 558a Rn. 2`(LIT)
- `Staudinger / V. Emmerich , BGB , Neubearb. 2018 , § 558a Rn. 2 ; § 558b Rn. 3`(LIT)
- `MünchKommBGB / Artz , BGB , 7. Aufl. , § 558b Rn. 3`(LIT)
- `BayObLG , NJW-RR 1993 , 202 mwN [ zu § 2 MHG ]`(RS)

**Example 66** (doc_id: `64617`) (sent_id: `64617`)


I. Die Klage ist , soweit sie in die Revision gelangt ist , unbegründet .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 67** (doc_id: `64704`) (sent_id: `64704`)


I. Der Zulässigkeit der Verfassungsbeschwerden in den Verfahren 2 BvR 1738/12 und 2 BvR 1068/14 steht nicht entgegen , dass die Beschwerdeführerin zu III. bereits während des fachgerichtlichen Verfahrens und damit vor Erhebung der Verfassungsbeschwerde auf eigenen Wunsch aus dem Beamtenverhältnis ausgeschieden ist und der Beschwerdeführer zu I. während des Verfassungsbeschwerdeverfahrens die Altersgrenze des § 35 Abs. 1 Satz 2 , Abs. 2 des Niedersächsischen Beamtengesetzes erreicht hat und in den Ruhestand getreten ist .

**False Positives:**

- `I. Der` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Verfahren 2 BvR 1738/12 und 2 BvR 1068/14`(RS)
- `§ 35 Abs. 1 Satz 2 , Abs. 2 des Niedersächsischen Beamtengesetzes`(NRM)

**Example 68** (doc_id: `64768`) (sent_id: `64768`)


I. Die Klägerin begehrt die Gewährung einer Rente wegen Erwerbsminderung .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 69** (doc_id: `64780`) (sent_id: `64780`)


I. Der Kläger und Beschwerdeführer ( Kläger ) , ein Heilpraktiker und approbierter Psychotherapeut , führte im Rahmen seiner psychotherapeutischen Leistungen in den Streitjahren ( 2010 bis 2012 ) u. a. auch verkehrspsychologische Behandlungen durch .

**False Positives:**

- `I. Der` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 70** (doc_id: `64801`) (sent_id: `64801`)


Der Pauschbetrag für behinderungsbedingte Aufwendungen i. S. des § 33b Abs. 1 EStG kann grundsätzlich auch nur " anstelle " einer Steuerermäßigung nach § 33 EStG für außergewöhnliche Belastungen geltend gemacht werden ( vgl. z.B. Blümich / K. Heger , § 33b EStG Rz 11 ) .

**False Positives:**

- `B. Blümich` — positional overlap with gold: `Blümich / K. Heger , § 33b EStG Rz 11`
- `K. Heger` — partial — pred is substring of gold: `Blümich / K. Heger , § 33b EStG Rz 11`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 33b Abs. 1 EStG`(NRM)
- `§ 33 EStG`(NRM)
- `Blümich / K. Heger , § 33b EStG Rz 11`(LIT)

**Example 71** (doc_id: `64806`) (sent_id: `64806`)


Das Umstandsmoment ist in der Regel erfüllt , wenn der Schuldner im Hinblick auf die Nichtgeltendmachung des Rechts Vermögensdispositionen getroffen hat ( Palandt / Grüneberg a. a. O. Rn. 95 m. w. N. ) .

**False Positives:**

- `O. Rn` — partial — pred is substring of gold: `Palandt / Grüneberg a. a. O. Rn. 95`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Palandt / Grüneberg a. a. O. Rn. 95`(LIT)

**Example 72** (doc_id: `64969`) (sent_id: `64969`)


I. Die Kläger und Revisionsbeklagten ( Kläger ) sind verheiratet und wurden für das Streitjahr ( 2012 ) zur Einkommensteuer zusammen veranlagt .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 73** (doc_id: `65021`) (sent_id: `65021`)


B. Die Rechtsbeschwerde der Antragsteller ist begründet .

**False Positives:**

- `B. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 74** (doc_id: `65024`) (sent_id: `65024`)


I. Die in § 33 Abs. 2 Satz 1 TV DRV KBS geregelte auflösende Bedingung gilt nicht nach §§ 21 , 17 Satz 2 TzBfG iVm. § 7 Halbs. 1 KSchG als wirksam und eingetreten .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 33 Abs. 2 Satz 1 TV DRV KBS`(REG)
- `§§ 21 , 17 Satz 2 TzBfG`(NRM)
- `§ 7 Halbs. 1 KSchG`(NRM)

**Example 75** (doc_id: `65026`) (sent_id: `65026`)


I. Die Revision ist zulässig .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 76** (doc_id: `65037`) (sent_id: `65037`)


I. Das mit den Farben rot und weiß beanspruchte Bildzeichen ist am 3. Juni 2016 zur Eintragung als Marke in das beim Deutschen Patent- und Markenamt ( DPMA ) geführte Register für Waren und Dienstleistungen der Klassen 9 , 16 , 18 , 21 , 24 , 25 , 30 , 32 , 35 , 38 , 41 und 42 angemeldet worden .

**False Positives:**

- `I. Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschen Patent- und Markenamt`(ORG)
- `DPMA`(ORG)

**Example 77** (doc_id: `65141`) (sent_id: `65141`)


C. Im Hinblick auf die aufgezeigten Bedenken , ob nach der von Rechts wegen gebotenen Aufgabe der sog. „ geschmacksmusterrechtlichen Unterkombination “ überhaupt noch eine Auslegung des Schutzgegenstands eines eingetragenen Designs auf Grundlage der Schnittmenge der allen Darstellungen gemeinsamen Merkmale in Betracht kommt , war nach § 23 Abs. 5 DesignG i. V. m. § 100 Abs. 2 Nr. 1 PatG die Zulassung der Rechtsbeschwerde veranlasst , da es sich insoweit um eine Rechtsfrage von grundsätzlicher Bedeutung handelt .

**False Positives:**

- `C. Im` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 23 Abs. 5 DesignG`(NRM)
- `§ 100 Abs. 2 Nr. 1 PatG`(NRM)

**Example 78** (doc_id: `65193`) (sent_id: `65193`)


E. Auch weitere vom Beklagten und von der Beigeladenen gegen das in Betracht zu ziehende Verkehrsverbot vorgebrachte Einwände greifen nicht durch .

**False Positives:**

- `E. Auch` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 79** (doc_id: `65218`) (sent_id: `65218`)


I. Die Beschwerdeführerin , eine albanische Staatsangehörige , wendet sich gegen die Versagung einstweiligen Rechtsschutzes in ihrem Asylverfahren .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 80** (doc_id: `65287`) (sent_id: `65287`)


b ) Verpachtet ein Unternehmer ein Grundstück an einen Landwirt , der seine Umsätze gemäß § 24 Abs. 1 UStG nach Durchschnittssätzen versteuert , kann der Verpächter nicht auf die Steuerfreiheit seiner Umsätze nach § 9 Abs. 2 Satz 1 UStG verzichten ( zutreffend Nieuwenhuis , a. a. O. , § 9 UStG Rz 78 ; Schüler-Täsch in Sölch / Ringleb , Umsatzsteuer , § 9 Rz 47 , und Mehrwertsteuerrecht 2013 , 540 , 543 ; Stadie in Stadie , UStG , 3. Aufl. , § 9 Rz 28 und § 24 Rz 41 ; a. M. Abschn. 9.2 Abs. 2 des Umsatzsteuer-Anwendungserlasses - UStAE - , sowie Lange in Offerhaus / Söhn / Lange , § 24 UStG Rz 456 , und Widmann in Schwarz / Widmann / Radeisen , UStG , § 9 Rz 171 ) .

**False Positives:**

- `M. Abschn` — positional overlap with gold: `Abschn. 9.2 Abs. 2 des Umsatzsteuer-Anwendungserlasses - UStAE`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 24 Abs. 1 UStG`(NRM)
- `§ 9 Abs. 2 Satz 1 UStG`(NRM)
- `Nieuwenhuis , a. a. O. , § 9 UStG Rz 78`(LIT)
- `Schüler-Täsch in Sölch / Ringleb , Umsatzsteuer , § 9 Rz 47 , und Mehrwertsteuerrecht 2013 , 540 , 543`(LIT)
- `Stadie in Stadie , UStG , 3. Aufl. , § 9 Rz 28 und § 24 Rz 41`(LIT)
- `Abschn. 9.2 Abs. 2 des Umsatzsteuer-Anwendungserlasses - UStAE`(REG)
- `Lange in Offerhaus / Söhn / Lange , § 24 UStG Rz 456`(LIT)
- `Widmann in Schwarz / Widmann / Radeisen , UStG , § 9 Rz 171`(LIT)

**Example 81** (doc_id: `65323`) (sent_id: `65323`)


aa ) Die Klägerin trägt zunächst vor , sie habe die Klage als eingetragene Partnerschaft unter dem Namen " S. und T. R. Physiotherapie-Partnerschaft " erhoben .

**False Positives:**

- `R. Physiotherapie-Partnerschaft` — positional overlap with gold: `T. R.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S.`(PER)
- `T. R.`(PER)

**Example 82** (doc_id: `65328`) (sent_id: `65328`)


I. Die Bezeichnung

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 83** (doc_id: `65344`) (sent_id: `65344`)


A. Gegenstand des Ausgangsverfahrens

**False Positives:**

- `A. Gegenstand` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 84** (doc_id: `65361`) (sent_id: `65361`)


I. Die am 7. November 2013 angemeldete Wortfolge Rap Shot ist am 23. Januar 2014 unter der Nummer 30 2013 058 941 als Wortmarke für die nachfolgend genannten Waren und Dienstleistungen in das beim Deutschen Patent- und Markenamt geführte Markenregister eingetragen worden :

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Rap Shot`(ORG)
- `Deutschen Patent- und Markenamt`(ORG)

**Example 85** (doc_id: `65413`) (sent_id: `65413`)


I. Die Klägerin und Revisionsbeklagte ( Klägerin ) , eine GmbH , führt Tiefbauarbeiten aus .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 86** (doc_id: `65416`) (sent_id: `65416`)


I. Die Klägerin , Revisionsklägerin und Revisionsbeklagte ( Klägerin ) unterhielt bis zum Streitjahr 2002 den Regiebetrieb " X " ( im Folgenden : BgA ) , einen Betrieb gewerblicher Art i. S. des § 4 des Körperschaftsteuergesetzes ( KStG ) , für den der Beklagte , Revisionskläger und Revisionsbeklagte ( das Finanzamt - FA - ) gegenüber der Klägerin im Zusammenhang mit der Einbringung des BgA in eine Kapitalgesellschaft für den Anmeldungszeitraum 2002 Kapitalertragsteuer zuzüglich Solidaritätszuschlag festsetzte .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `" X "`(ORG)
- `§ 4 des Körperschaftsteuergesetzes`(NRM)
- `KStG`(NRM)

**Example 87** (doc_id: `65429`) (sent_id: `65429`)


I. Anders als der Beklagte zu 3. meint , sind die Revisionen zulässig .

**False Positives:**

- `I. Anders` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 88** (doc_id: `65590`) (sent_id: `65590`)


I. In dem der Nichtzulassungsbeschwerde zugrunde liegenden Rechtsstreit streiten die Beteiligten darüber , ob die Beklagte den Widerspruch des Klägers gegen ihr Schreiben vom 14. 4. 2015 als unzulässig zurückweisen durfte .

**False Positives:**

- `I. In` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Full names with titles (corrected)`

**F1:** 0.041 | **Precision:** 0.389 | **Recall:** 0.022  

**Format:** `regex`  
**Rule ID:** `e697a848`  
**Description:**
Captures full names preceded by titles like Dr., Prof., ensuring the entire name is captured including middle initials, excluding the title itself.

**Content:**
```
(?:Dr\.?\s+|Prof\.?\s+|Dipl\.-Ing\.\s+|Dipl\.-Psych\.\s+|Dipl\.-Ing\.\s+Univ\.\s+)([A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+[A-Z]\s*\.)?(?:\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)*)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.389 | 0.022 | 0.041 | 18 | 7 | 11 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 7 | 11 | 251 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `61554`) (sent_id: `61554`)


Dr. Achilles

| Predicted | Gold |
|---|---|
| `Achilles` | `Achilles` |

**Example 1** (doc_id: `61787`) (sent_id: `61787`)


In der Beschwerdesache betreffend die Marke 30 2009 056 266 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 14. September 2017 unter Mitwirkung des Vorsitzenden Richters Knoll sowie der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Nielsen` | `Nielsen` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Kriener` (PER)

**Example 2** (doc_id: `64394`) (sent_id: `64394`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2013 053 470.0 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 9. November 2017 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Nielsen` | `Nielsen` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Kriener` (PER)

**Example 3** (doc_id: `64527`) (sent_id: `64527`)


In der Beschwerdesache betreffend die Marke 30 2011 035 856 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Meiser` | `Meiser` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Hacker` (PER)
- `Merzbach` (PER)

**Example 4** (doc_id: `64925`) (sent_id: `64925`)


In der Beschwerdesache betreffend die Marke 30 2010 003 649 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Meiser` | `Meiser` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Hacker` (PER)
- `Merzbach` (PER)

**Example 5** (doc_id: `64967`) (sent_id: `64967`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2015 034 000.6 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 11. Januar 2018 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Nielsen` | `Nielsen` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Kriener` (PER)

**Example 6** (doc_id: `65500`) (sent_id: `65500`)


In der Beschwerdesache betreffend die Designanmeldung .... ( hier : Antrag auf Verfahrenskostenhilfe ) hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Meiser` | `Meiser` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Hacker` (PER)
- `Merzbach` (PER)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `61830`) (sent_id: `61830`)


Ferner hat die Einsprechende das „ Gutachten zum Sicherheitsschalter CES der Firma E … “ des Herrn Prof. Dr. - Ing. G … vom 15. September 2014 vorgelegt .

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `E …`(ORG)
- `G …`(PER)

**Example 1** (doc_id: `61864`) (sent_id: `61864`)


Zu dem Sachverständigengutachten des Prof. Dr. T. sowie dessen ergänzender Stellungnahme hat die Beklagte mitgeteilt , dass sie diesem nach prüfärztlicher Stellungnahme nicht folgen könne ; der Einschätzung des Dr. S. hat sie sich hingegen angeschlossen .

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `T.`(PER)
- `S.`(PER)

**Example 2** (doc_id: `63653`) (sent_id: `63653`)


Als sachkundige Auskunftsperson hat sich in der mündlichen Verhandlung Prof. Dr. Klaus-Dieter Drüen geäußert .

**False Positives:**

- `Dr` — similar text (different position): `Klaus-Dieter Drüen`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Klaus-Dieter Drüen`(PER)

**Example 3** (doc_id: `63754`) (sent_id: `63754`)


Soweit das LSG Ausführungen von Prof. Dr. S. referiert , dass sich im Zusammenhang mit den Pneumonien auch immer wieder Septitiden entwickelt hätten , macht das LSG nicht deutlich , dass es diese als festgestellt ansieht , und erst recht nicht , von welchem Begriff oder Schweregrad der Sepsis es im Anschluss an Prof. Dr. S. aufgrund welcher Befunde dabei ausgeht ( ältere Klassifizierungen : Systemisches inflammatorisches Response-Syndrom < SIRS > , Sepsis , schwere Sepsis und septischer Schock ; zur darauf aufbauenden DRG-Kodierung vgl Hanser in Zaiß , DRG : Verschlüsseln leicht gemacht , 14. Aufl 2016 , S 101 ff ; neuere Klassifizierungen : Sepsis-related organ failure assessment score < SOFA > , insbesondere für Intensivstationen , und quickSOFA außerhalb von Intensivstationen ) .

**False Positives:**

- `Dr` — no gold match — likely missing annotation
- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `S.`(PER)
- `S.`(PER)

**Example 4** (doc_id: `63885`) (sent_id: `63885`)


Nach den Ausführungen des im Verfahren von Amts wegen gehörten Sachverständigen Prof. Dr. T. hätten die vom Kläger vorgetragenen Gewalterfahrungen während seiner Heimaufenthalte ab April 1959 nicht die entscheidende Traumatisierung dargestellt .

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `T.`(PER)

**Example 5** (doc_id: `64291`) (sent_id: `64291`)


Ferner hat sie beantragt , Prof. Dr. B. , Zentralinstitut für seelische Gesundheit M. , Institut für Psychiatrie und Psychosomatische Psychotherapie , als sachverständigen Zeugen zu vernehmen sowie den Sachverständigen Prof. Dr. S. zu den mit Schriftsatz vom 21. 3. 2017 erhobenen Einwendungen anzuhören .

**False Positives:**

- `Dr` — no gold match — likely missing annotation
- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `B.`(PER)
- `Zentralinstitut für seelische Gesundheit M. , Institut für Psychiatrie und Psychosomatische Psychotherapie`(ORG)
- `S.`(PER)

**Example 6** (doc_id: `64527`) (sent_id: `64527`)


In der Beschwerdesache betreffend die Marke 30 2011 035 856 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 7** (doc_id: `64925`) (sent_id: `64925`)


In der Beschwerdesache betreffend die Marke 30 2010 003 649 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 8** (doc_id: `65500`) (sent_id: `65500`)


In der Beschwerdesache betreffend die Designanmeldung .... ( hier : Antrag auf Verfahrenskostenhilfe ) hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

</details>

---

## `Anonymized single letters`

**F1:** 0.030 | **Precision:** 0.385 | **Recall:** 0.015  

**Format:** `regex`  
**Rule ID:** `70843e1c`  
**Description:**
Captures single letter anonymized names (e.g., 'A', 'B') ONLY when preceded by a role indicator or title, preventing false positives on section markers like 'I.', 'II.' or abbreviations.

**Content:**
```
(?:Herr\s+|Herrn\s+|Dr\.\s+|Prof\.\s+|Richter\s+|Richterin\s+|Vorsitzender\s+Richter\s+|Angeklagte\s+|Angeklagten\s+|Kl\u00e4ger\s+|Zeuge\s+|Zeugin\s+|Gesch\u00e4ftsf\u00fchrer\s+|Rechtsanwalt\s+|Rechtsanw\u00e4ltin\s+|Beteiligte\s+|Beteiligter\s+)([A-Z])(?=\s*(?:,|\.|\)|\]|\s|$))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.385 | 0.015 | 0.030 | 13 | 5 | 8 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 5 | 8 | 264 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `62066`) (sent_id: `62066`)


Die Ausgestaltung dieser Rahmenvereinbarung entspricht ausdrücklich dem von Herrn N geäußerten Wunsch , da Herr N aus pers. Gründen nicht öfter zur Verfügung stehen kann .

| Predicted | Gold |
|---|---|
| `N` | `N` |
| `N` | `N` |

**Example 1** (doc_id: `63744`) (sent_id: `63744`)


Über den Betriebs ( teil- ) übergang sowie den Übergang seines Arbeitsverhältnisses auf die D P T S GmbH wurde der Kläger durch ein Unterrichtungsschreiben vom 14. November 2005 informiert , das auf dem Briefkopf der Beklagten erstellt und für diese von deren Abteilungsleiter Personal / Service R und für die D P T S GmbH von deren Geschäftsführer C unterzeichnet war .

| Predicted | Gold |
|---|---|
| `C` | `C` |

**Missed by this rule (FN):**

- `D P T S GmbH` (ORG)
- `R` (PER)
- `D P T S GmbH` (ORG)

**Example 2** (doc_id: `65215`) (sent_id: `65215`)


Auf der Grundlage des § 27 Abs. 2 MTV-DP AG hatte die Beklagte Herrn N zunächst für die Zeit vom 1. Oktober 2013 bis zum 1. Oktober 2014 Sonderurlaub gewährt und diesen bis zum 1. Oktober 2015 verlängert .

| Predicted | Gold |
|---|---|
| `N` | `N` |

**Missed by this rule (FN):**

- `§ 27 Abs. 2 MTV-DP AG` (REG)

**Example 3** (doc_id: `65437`) (sent_id: `65437`)


Dabei hat sich der Senat auf zwei Erwägungen gestützt : Einerseits habe es die Beklagte rechtswidrig unterlassen , einen Beurteilungsbeitrag von dem erkrankten und inzwischen im Ruhestand befindlichen ehemaligen Abteilungsleiter X ( Herr Dr. A ) einzuholen .

| Predicted | Gold |
|---|---|
| `A` | `A` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `61243`) (sent_id: `61243`)


Doch auch wenn die Taten 21 bis 23 und 33 der Urteilsgründe sich damit im Hinblick auf das Marihuana als selbständige Umsatzgeschäfte darstellen , fallen die darauf bezogenen Handlungen des Angeklagten mit denjenigen zusammen , die dem Absatz des zugleich in diesen Fällen gehandelten Amphetamins dienten , hinsichtlich dessen aufgrund des einheitlichen Erwerbs im Fall 27 der Urteilsgründe von einer Bewertungseinheit und damit von einer Tat im Rechtssinne auszugehen ist : Im Fall 21 der Urteilsgründe nahm der Angeklagte die Bestellung beider Betäubungsmittel einheitlich entgegen , in den Fällen 22 und 23 der Urteilsgründe lieferte er sie auch gleichzeitig an die Angeklagte A. A. .

**False Positives:**

- `A` — similar text (different position): `A. A.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `A. A.`(PER)

**Example 1** (doc_id: `61306`) (sent_id: `61306`)


Das Landgericht hat den Angeklagten A. wegen Diebstahls mit Waffen in Tateinheit mit „ fahrlässiger “ Gefährdung des Straßenverkehrs , vorsätzlichem Fahren ohne Fahrerlaubnis und fahrlässiger Körperverletzung sowie wegen unerlaubten Entfernens vom Unfallort und vorsätzlicher Körperverletzung zu der Gesamtfreiheitsstrafe von zwei Jahren und vier Monaten verurteilt ; ferner hat es die Verwaltungsbehörde angewiesen , dem Angeklagten vor Ablauf einer Frist von drei Jahren keine Fahrerlaubnis zu erteilen .

**False Positives:**

- `A` — similar text (different position): `A.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `A.`(PER)

**Example 2** (doc_id: `62293`) (sent_id: `62293`)


Das Landgericht hat den Angeklagten T. D. wegen schweren sexuellen Missbrauchs eines Kindes in Tateinheit mit sexuellem Missbrauch eines Schutzbefohlenen in zwei Fällen sowie schweren sexuellen Missbrauchs einer widerstandsunfähigen Person in fünf Fällen zu einer Gesamtfreiheitsstrafe von vier Jahren und drei Monaten verurteilt .

**False Positives:**

- `T` — similar text (different position): `T. D.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `T. D.`(PER)

**Example 3** (doc_id: `63047`) (sent_id: `63047`)


1. Die Verurteilung des Angeklagten T. D. wegen schweren sexuellen Missbrauchs einer widerstandsunfähigen Person in den Fällen III. 3 bis 7 der Urteilsgründe nach dem zur Tatzeit geltenden § 179 Abs. 1 Nr. 1 , Abs. 5 Nr. 1 StGB in der bis zum 9. November 2016 geltenden Fassung hält revisionsrechtlicher Überprüfung nicht stand , weil die Urteilsgründe eine Widerstandsunfähigkeit des Nebenklägers nicht belegen .

**False Positives:**

- `T` — similar text (different position): `T. D.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `T. D.`(PER)
- `§ 179 Abs. 1 Nr. 1 , Abs. 5 Nr. 1 StGB`(NRM)

**Example 4** (doc_id: `64409`) (sent_id: `64409`)


Da der Zeuge W ... diesen Angaben glaubte , erwarb er für die Firma die beiden Geräte und zahlte den Kaufpreis .

**False Positives:**

- `W` — partial — pred is substring of gold: `W ...`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `W ...`(PER)

**Example 5** (doc_id: `65248`) (sent_id: `65248`)


3. Im Umfang der Aufhebung wird die Sache zu neuer Verhandlung und Entscheidung , auch über die Kosten des Rechtsmittels des Angeklagten A. , an eine andere Strafkammer des Landgerichts zurückverwiesen .

**False Positives:**

- `A` — similar text (different position): `A.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `A.`(PER)

**Example 6** (doc_id: `65305`) (sent_id: `65305`)


c ) Damit erledigt sich der Antrag auf Gewährung von Prozesskostenhilfe und Beiordnung von Rechtsanwältin H … für das Verfahren auf Erlass einer einstweiligen Anordnung .

**False Positives:**

- `H` — similar text (different position): `H …`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `H …`(PER)

**Example 7** (doc_id: `66540`) (sent_id: `66540`)


„ Ausgehend vom nach Teileinstellungen noch angeklagten Sachverhalt nach Maßgabe des Hinweisbeschlusses vom 6. Hauptverhandlungstag “ wird hinsichtlich der Angeklagten K. „ bei insoweit glaubhaftem Geständnis und kooperativem Verhalten “ eine Verurteilung zu einer Gesamtfreiheitsstrafe von mindestens neun Monaten bis zu einem Jahr und drei Monaten , deren Vollstreckung zur Bewährung ausgesetzt wird , erfolgen .

**False Positives:**

- `K` — similar text (different position): `K.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K.`(PER)

</details>

---

## `Compound surnames`

**F1:** 0.024 | **Precision:** 1.000 | **Recall:** 0.012  

**Format:** `regex`  
**Rule ID:** `243eaaf1`  
**Description:**
Captures specific known German compound surnames (e.g., Lachenmayr-Nikolaou, Sost-Scheible) while excluding geographical or technical terms.

**Content:**
```
\b(?:Lachenmayr-Nikolaou|Sost-Scheible|Mittenberger-Huber|Meier-Beck|Fuchs-Wissemann|Kopp-Schenke|Dreier-Gro\u00df|Sch\u00e4fer-Weber|Koch-Schmidt|Bender-Brune|Harsdorf-Gebhardt)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.012 | 0.024 | 4 | 4 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 4 | 0 | 299 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60546`) (sent_id: `60546`)


Sost-Scheible

| Predicted | Gold |
|---|---|
| `Sost-Scheible` | `Sost-Scheible` |

**Example 1** (doc_id: `62802`) (sent_id: `62802`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2014 004 201.0 hat der 29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 29. November 2017 durch die Vorsitzende Richterin Dr. Mittenberger-Huber , die Richterin Akintche und die Richterin Seyfarth beschlossen :

| Predicted | Gold |
|---|---|
| `Mittenberger-Huber` | `Mittenberger-Huber` |

**Missed by this rule (FN):**

- `29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Akintche` (PER)
- `Seyfarth` (PER)

**Example 2** (doc_id: `63445`) (sent_id: `63445`)


Sost-Scheible

| Predicted | Gold |
|---|---|
| `Sost-Scheible` | `Sost-Scheible` |

**Example 3** (doc_id: `64220`) (sent_id: `64220`)


Harsdorf-Gebhardt

| Predicted | Gold |
|---|---|
| `Harsdorf-Gebhardt` | `Harsdorf-Gebhardt` |

</details>

---

## `Names after 'Angeklagte' or 'Angeklagten' (corrected)`

**F1:** 0.012 | **Precision:** 0.333 | **Recall:** 0.006  

**Format:** `regex`  
**Rule ID:** `5e2b25bc`  
**Description:**
Captures names following 'Angeklagte' or 'Angeklagten', ensuring no trailing space and handling initials with dots.

**Content:**
```
(?:Angeklagte|Angeklagten)\s+([A-Z][a-zäöüß]*\.?\s*(?:[A-Z][a-zäöüß]*\.?\s*)*|\b[A-Z]\s*\.\b|\b[A-Z]\b)(?=\s*(?:,|\.|\)|\]|\s|$))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.333 | 0.006 | 0.012 | 6 | 2 | 4 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 2 | 4 | 267 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `62293`) (sent_id: `62293`)


Das Landgericht hat den Angeklagten T. D. wegen schweren sexuellen Missbrauchs eines Kindes in Tateinheit mit sexuellem Missbrauch eines Schutzbefohlenen in zwei Fällen sowie schweren sexuellen Missbrauchs einer widerstandsunfähigen Person in fünf Fällen zu einer Gesamtfreiheitsstrafe von vier Jahren und drei Monaten verurteilt .

| Predicted | Gold |
|---|---|
| `T. D.` | `T. D.` |

**Example 1** (doc_id: `63047`) (sent_id: `63047`)


1. Die Verurteilung des Angeklagten T. D. wegen schweren sexuellen Missbrauchs einer widerstandsunfähigen Person in den Fällen III. 3 bis 7 der Urteilsgründe nach dem zur Tatzeit geltenden § 179 Abs. 1 Nr. 1 , Abs. 5 Nr. 1 StGB in der bis zum 9. November 2016 geltenden Fassung hält revisionsrechtlicher Überprüfung nicht stand , weil die Urteilsgründe eine Widerstandsunfähigkeit des Nebenklägers nicht belegen .

| Predicted | Gold |
|---|---|
| `T. D.` | `T. D.` |

**Missed by this rule (FN):**

- `§ 179 Abs. 1 Nr. 1 , Abs. 5 Nr. 1 StGB` (NRM)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `61243`) (sent_id: `61243`)


Doch auch wenn die Taten 21 bis 23 und 33 der Urteilsgründe sich damit im Hinblick auf das Marihuana als selbständige Umsatzgeschäfte darstellen , fallen die darauf bezogenen Handlungen des Angeklagten mit denjenigen zusammen , die dem Absatz des zugleich in diesen Fällen gehandelten Amphetamins dienten , hinsichtlich dessen aufgrund des einheitlichen Erwerbs im Fall 27 der Urteilsgründe von einer Bewertungseinheit und damit von einer Tat im Rechtssinne auszugehen ist : Im Fall 21 der Urteilsgründe nahm der Angeklagte die Bestellung beider Betäubungsmittel einheitlich entgegen , in den Fällen 22 und 23 der Urteilsgründe lieferte er sie auch gleichzeitig an die Angeklagte A. A. .

**False Positives:**

- `A. A. ` — partial — gold is substring of pred: `A. A.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `A. A.`(PER)

**Example 1** (doc_id: `64047`) (sent_id: `64047`)


In einem weiteren Fall öffnete der Angeklagte Knopf und Reißverschluss seiner Hose und forderte die Zeugin sinngemäß auf , an seinem Glied zu reiben .

**False Positives:**

- `Knopf` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `64798`) (sent_id: `64798`)


2. Im Umfang der Aufhebung wird die Sache zu neuer Verhandlung und Entscheidung , auch über die Kosten des Rechtsmittels des Angeklagten S. , an das Amtsgericht - Schöffengericht - Aachen zurückverwiesen .

**False Positives:**

- `S. ` — partial — gold is substring of pred: `S.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S.`(PER)
- `Amtsgericht - Schöffengericht - Aachen`(ORG)

**Example 3** (doc_id: `65248`) (sent_id: `65248`)


3. Im Umfang der Aufhebung wird die Sache zu neuer Verhandlung und Entscheidung , auch über die Kosten des Rechtsmittels des Angeklagten A. , an eine andere Strafkammer des Landgerichts zurückverwiesen .

**False Positives:**

- `A. ` — partial — gold is substring of pred: `A.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `A.`(PER)

</details>

---

## `Initials with dots and spaces (e.g., T. D.)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `a3f99de3`  
**Description:**
Captures sequences of initial-dot-space-initial-dot patterns.

**Content:**
```
\b([A-Z]\.[ ]+[A-Z]\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Names after 'Rechtsanwalt' or 'Rechtsanwältin'`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `1e1f1dc2`  
**Description:**
Captures names following legal profession titles, ensuring no trailing space and handling initials with dots.

**Content:**
```
(?:Rechtsanwalt|Rechtsanwältin)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*|\b[A-Z]\s*\.\b|\b[A-Z]\b)(?=\s*(?:,|\.|\)|\]|\s|$))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Names after 'Herr' or 'Herrn' (corrected)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `ae810878`  
**Description:**
Captures names following 'Herr' or 'Herrn', handling full names and initials.

**Content:**
```
(?:Herr|Herrn)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z]\s*\.)?(?:\s+[A-Z][a-zäöüß]+)*)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 2 | 0 | 2 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 2 | 242 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `61830`) (sent_id: `61830`)


Ferner hat die Einsprechende das „ Gutachten zum Sicherheitsschalter CES der Firma E … “ des Herrn Prof. Dr. - Ing. G … vom 15. September 2014 vorgelegt .

**False Positives:**

- `Prof` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `E …`(ORG)
- `G …`(PER)

**Example 1** (doc_id: `65437`) (sent_id: `65437`)


Dabei hat sich der Senat auf zwei Erwägungen gestützt : Einerseits habe es die Beklagte rechtswidrig unterlassen , einen Beurteilungsbeitrag von dem erkrankten und inzwischen im Ruhestand befindlichen ehemaligen Abteilungsleiter X ( Herr Dr. A ) einzuholen .

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `A`(PER)

</details>

---

## `Anonymized initials with dots and spaces (e.g., T. D.)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `d3f9d622`  
**Description:**
Captures sequences of initial-dot-space-initial-dot patterns.

**Content:**
```
\b([A-Z]\.)\s+([A-Z]\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Anonymized names with dots and ellipses or spaces`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `b7b02b0b`  
**Description:**
Captures anonymized names with dots and ellipses or spaces (e.g., 'K …', 'H …'), excluding company names.

**Content:**
```
\b([A-Z]\d?\.?)\s+…\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Initials after legal roles`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `a6567e07`  
**Description:**
Captures single letter initials (with or without dots) following legal role indicators like 'Angeklagten', 'Kläger', 'Zeuge', 'Zeugin'.

**Content:**
```
(?:Angeklagten|Kläger|Zeuge|Zeugin|Vertrauensmann)\s+([A-Z](?:\.)?)(?=\s*(?:,|\.|\)|\]|\s|$))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Anonymized initials with dots (e.g., R., A.)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `7124ef3f`  
**Description:**
Captures anonymized initials with a trailing dot (e.g., 'R.', 'A.', 'B.') to ensure the dot is included in the entity, excluding common non-name words.

**Content:**
```
(?<![A-Za-zäöüß\.\s])([A-Z]\.)(?![A-Za-zäöüß\.\s])
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 1 | 0 | 1 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 1 | 274 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `61101`) (sent_id: `61101`)


X.

**False Positives:**

- `X.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Initials with surname (e.g., K. Schmidt)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `3b9a194a`  
**Description:**
Captures an initial followed by a capitalized surname, ensuring it's a name and not a sentence start or common verb.

**Content:**
```
(?<!^)(?<!\w)([A-Z]\s*\.)\s+([A-Z][a-zäöüß]+(?:-[A-Z][a-zäöüß]+)?)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 6 | 0 | 6 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 6 | 295 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60707`) (sent_id: `60707`)


c ) Werden die von einem Steuerpflichtigen erworbenen Gegenstände oder Dienstleistungen dagegen für die Zwecke steuerbefreiter Umsätze oder solcher Umsätze verwendet , die nicht vom Anwendungsbereich der Mehrwertsteuer erfasst werden , so kann es weder zur Erhebung der Steuer auf der folgenden Stufe noch zum Abzug der Vorsteuer kommen ( ständige Rechtsprechung , vgl. dazu z.B. EuGH-Urteile SKF , EU : C : 2009 : 665 , UR 2010 , 107 , Rz 59 , m. w. N. ; Iberdrola Inmobiliaria Real Estate Investments , EU : C : 2017 : 683 , DStR 2017 , 2044 , Rz 30 ) .

**False Positives:**

- `B. Eu` — positional overlap with gold: `EuGH-Urteile SKF , EU : C : 2009 : 665 , UR 2010 , 107 , Rz 59`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH-Urteile SKF , EU : C : 2009 : 665 , UR 2010 , 107 , Rz 59`(RS)
- `Iberdrola Inmobiliaria Real Estate Investments , EU : C : 2017 : 683 , DStR 2017 , 2044 , Rz 30`(RS)

**Example 1** (doc_id: `60757`) (sent_id: `60757`)


1. Unterscheidungskraft im Sinne von § 8 Abs. 2 Nr. 1 MarkenG ist die einem Zeichen innewohnende ( konkrete ) Eignung , vom Verkehr als Unterscheidungsmittel aufgefasst zu werden , das die von der Anmeldung erfassten Waren oder Dienstleistungen als von einem bestimmten Unternehmen stammend kennzeichnet und diese somit von denjenigen anderer Unternehmen unterscheidet ( vgl. z.B. EuGH GRUR 2012 , 610 ( Nr. 42 ) - Freixenet ; GRUR 2008 , 608 , 611 ( Nr. 66 ) - EUROHYPO ; BGH GRUR 2015 , 173 , 174 ( Nr. 15 ) - for you ; GRUR 2014 , 565 , 567 ( Nr. 12 ) - smartbook ; GRUR 2013 , 731 ( Nr. 11 ) - Kaleido ; GRUR 2012 , 1143 ( Nr. 7 ) - Starsat , jeweils m. w. N. ) .

**False Positives:**

- `B. Eu` — positional overlap with gold: `EuGH GRUR 2012 , 610 ( Nr. 42 ) - Freixenet`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 8 Abs. 2 Nr. 1 MarkenG`(NRM)
- `EuGH GRUR 2012 , 610 ( Nr. 42 ) - Freixenet`(RS)
- `GRUR 2008 , 608 , 611 ( Nr. 66 ) - EUROHYPO`(RS)
- `BGH GRUR 2015 , 173 , 174 ( Nr. 15 ) - for you`(RS)
- `GRUR 2014 , 565 , 567 ( Nr. 12 ) - smartbook`(RS)
- `GRUR 2013 , 731 ( Nr. 11 ) - Kaleido`(RS)
- `GRUR 2012 , 1143 ( Nr. 7 ) - Starsat`(RS)

**Example 2** (doc_id: `61664`) (sent_id: `61664`)


b ) Da das gemeinsame Mehrwertsteuersystem eine völlige Neutralität hinsichtlich der steuerlichen Belastung aller wirtschaftlichen Tätigkeiten unabhängig von ihrem Zweck und ihrem Ergebnis gewährleistet , sofern diese selbst der Mehrwertsteuer unterliegen ( ständige Rechtsprechung , vgl. dazu z.B. EuGH-Urteil Fini H vom 3. März 2005 C - 32/03 , EU : C : 2005 : 128 , UR 2005 , 443 , Rz 25 , m. w. N. ) , darf nicht willkürlich zwischen Ausgaben für die Zwecke eines Unternehmens vor der tatsächlichen Aufnahme seiner Tätigkeit sowie während dieser Tätigkeit und Ausgaben zum Zweck der Beendigung dieser Tätigkeit unterschieden werden ( ständige Rechtsprechung , vgl. dazu z.B. EuGH-Urteile Abbey National vom 22. Februar 2001 C - 408/98 , EU : C : 2001 : 110 , UR 2001 , 164 , Rz 35 ; Faxworld vom 29. April 2004 C - 137/02 , EU : C : 2004 : 267 , UR 2004 , 362 , Rz 39 ; Fini H , EU : C : 2005 : 128 , UR 2005 , 443 , Rz 23 ; ferner Wind Inovation 1 vom 9. November 2017 C - 552/16 , EU : C : 2017 : 849 , Höchstrichterliche Finanzrechtsprechung 2018 , 84 , Rz 45 ; jeweils m. w. N. ) .

**False Positives:**

- `B. Eu` — positional overlap with gold: `EuGH-Urteil Fini H vom 3. März 2005 C - 32/03 , EU : C : 2005 : 128 , UR 2005 , 443 , Rz 25`
- `B. Eu` — positional overlap with gold: `EuGH-Urteile Abbey National vom 22. Februar 2001 C - 408/98 , EU : C : 2001 : 110 , UR 2001 , 164 , Rz 35`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH-Urteil Fini H vom 3. März 2005 C - 32/03 , EU : C : 2005 : 128 , UR 2005 , 443 , Rz 25`(RS)
- `EuGH-Urteile Abbey National vom 22. Februar 2001 C - 408/98 , EU : C : 2001 : 110 , UR 2001 , 164 , Rz 35`(RS)
- `Faxworld vom 29. April 2004 C - 137/02 , EU : C : 2004 : 267 , UR 2004 , 362 , Rz 39`(RS)
- `Fini H , EU : C : 2005 : 128 , UR 2005 , 443 , Rz 23`(RS)
- `Wind Inovation 1 vom 9. November 2017 C - 552/16 , EU : C : 2017 : 849 , Höchstrichterliche Finanzrechtsprechung 2018 , 84 , Rz 45`(RS)

**Example 3** (doc_id: `61963`) (sent_id: `61963`)


Das Vorliegen einer Verwechslungsgefahr für das Publikum ist nach ständiger Rechtsprechung sowohl des Europäischen Gerichtshofes als auch des Bundesgerichtshofes unter Berücksichtigung aller relevanten Umstände des Einzelfalls zu beurteilen ( vgl. hierzu z.B. EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER ; GRUR 2010 , 1098 Rn. 44 – Calvin Klein / HABM ; BGH GRUR 2012 , 64 ; BGH GRUR 2012 , 64 Rn. 9 ; BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY ; GRUR 2012 , 1040 Rn. 25 – pjur / pure ; GRUR 2013 , 833 Rn. 30 – Culinaria / Villa Culinaria ) .

**False Positives:**

- `B. Eu` — positional overlap with gold: `EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europäischen Gerichtshofes`(ORG)
- `Bundesgerichtshofes`(ORG)
- `EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER`(RS)
- `GRUR 2010 , 1098 Rn. 44 – Calvin Klein / HABM`(RS)
- `BGH GRUR 2012 , 64`(RS)
- `BGH GRUR 2012 , 64 Rn. 9`(RS)
- `BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY`(RS)
- `GRUR 2012 , 1040 Rn. 25 – pjur / pure`(RS)
- `GRUR 2013 , 833 Rn. 30 – Culinaria / Villa Culinaria`(RS)

**Example 4** (doc_id: `66392`) (sent_id: `66392`)


1. Das Vorliegen einer Verwechslungsgefahr für das Publikum ist nach ständiger Rechtsprechung sowohl des Europäischen Gerichtshofes als auch des Bundesgerichtshofes unter Berücksichtigung aller relevanten Umstände des Einzelfalls zu beurteilen ( vgl. hierzu z.B. EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER ; GRUR 2010 , 1098 Rn. 44 – Calvin Klein / HABM ; BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY ; GRUR 2012 , 1040 Rn. 25 – pjur / pure ; GRUR 2013 , 833 Rn. 30 – Culinaria / Villa Culinaria ; GRUR 2016 , 382 Rn. 19 – BioGourmet ) .

**False Positives:**

- `B. Eu` — positional overlap with gold: `EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europäischen Gerichtshofes`(ORG)
- `Bundesgerichtshofes`(ORG)
- `EuGH GRUR 2010 , 933 Rn. 32 – BARBARA BECKER`(RS)
- `GRUR 2010 , 1098 Rn. 44 – Calvin Klein / HABM`(RS)
- `BGH GRUR 2012 , 64 Rn. 9 – Maalox / Melox-GRY`(RS)
- `GRUR 2012 , 1040 Rn. 25 – pjur / pure`(RS)
- `GRUR 2013 , 833 Rn. 30 – Culinaria / Villa Culinaria`(RS)
- `GRUR 2016 , 382 Rn. 19 – BioGourmet`(RS)

</details>

---

## `Initials with dots (e.g., K., T.)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `5434427d`  
**Description:**
Captures single letter initials followed by a dot, ensuring the dot is included.

**Content:**
```
\b([A-Z]\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 5 | 0 | 5 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 5 | 165 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `63289`) (sent_id: `63289`)


dd ) Angesichts der Unvollständigkeit der Rechtsprechung des EuGH ( dazu unten b ) sind auch keine Ausnahmen von der unionsrechtlichen Vorlagepflicht des Oberlandesgerichts ersichtlich , etwa weil die entscheidungserhebliche Frage bereits Gegenstand einer Auslegung durch den EuGH war oder die richtige Anwendung des Unionsrechts derart offenkundig ist , dass für einen vernünftigen Zweifel keinerlei Raum bleibt ( vgl. EuGH , Urteil vom 6. Oktober 1982 , C.I.L.F.I.T. , C - 283/81 , Slg. 1982 , S. 3415 ff. Rn. 21 ) .

**False Positives:**

- `C.` — partial — pred is substring of gold: `EuGH , Urteil vom 6. Oktober 1982 , C.I.L.F.I.T. , C - 283/81 , Slg. 1982 , S. 3415 ff. Rn. 21`
- `I.` — partial — pred is substring of gold: `EuGH , Urteil vom 6. Oktober 1982 , C.I.L.F.I.T. , C - 283/81 , Slg. 1982 , S. 3415 ff. Rn. 21`
- `L.` — partial — pred is substring of gold: `EuGH , Urteil vom 6. Oktober 1982 , C.I.L.F.I.T. , C - 283/81 , Slg. 1982 , S. 3415 ff. Rn. 21`
- `F.` — partial — pred is substring of gold: `EuGH , Urteil vom 6. Oktober 1982 , C.I.L.F.I.T. , C - 283/81 , Slg. 1982 , S. 3415 ff. Rn. 21`
- `I.` — partial — pred is substring of gold: `EuGH , Urteil vom 6. Oktober 1982 , C.I.L.F.I.T. , C - 283/81 , Slg. 1982 , S. 3415 ff. Rn. 21`

> overlaps gold: 5  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH`(ORG)
- `EuGH`(ORG)
- `EuGH , Urteil vom 6. Oktober 1982 , C.I.L.F.I.T. , C - 283/81 , Slg. 1982 , S. 3415 ff. Rn. 21`(RS)

</details>

---

## `Initials with surname (corrected)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `1395461f`  
**Description:**
Captures patterns like 'K. Schmidt' or 'M. P. Borom' where an initial (with dot) is followed by a surname.

**Content:**
```
\b([A-Z]\.)\s+([A-Z][a-zäöüß]+(?:-[A-Z][a-zäöüß]+)?)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Names with specific formatting (BRADLER , Christian)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `e2864f95`  
**Description:**
Captures names with specific formatting like 'BRADLER , Christian' or 'Surname , Firstname', excluding non-name patterns like 'BMW, Typ' or 'BGH, Beschluss'.

**Content:**
```
\b([A-Z][A-Z\u00e4\u00f6\u00fc\u00df]+\s*,\s*[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 93 | 0 | 93 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 93 | 324 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60069`) (sent_id: `60069`)


Demgegenüber kommt nach dem Regelungsplan des Gesetzgebers eine Zuständigkeit des Berufungsgerichts in Betracht , wenn es um den vom Berufungsgericht festgestellten oder festzustellenden Sachverhalt geht ( BGH , Urteil vom 13. Juli 1954 aaO ; ferner Beschluss vom 8. Juni 1973 - I ZR 25/72 , BGHZ 61 , 95 , 97 , 100 [ juris Rn. 9 f. ] ; BVerwG vom 7. Dezember 2015 - 6 PKH 10/15 , juris Rn. 12 ; Musielak in Musielak / Voith , ZPO 14. Aufl. , § 584 Rn. 2 ; MünchKomm-ZPO / Braun , 5. Aufl. , § 584 Rn. 1 ) .

**False Positives:**

- `BGH , Urteil` — partial — gold is substring of pred: `BGH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH`(ORG)
- `Beschluss vom 8. Juni 1973 - I ZR 25/72 , BGHZ 61 , 95 , 97 , 100 [ juris Rn. 9 f. ]`(RS)
- `BVerwG vom 7. Dezember 2015 - 6 PKH 10/15 , juris Rn. 12`(RS)
- `Musielak in Musielak / Voith , ZPO 14. Aufl. , § 584 Rn. 2`(LIT)
- `MünchKomm-ZPO / Braun , 5. Aufl. , § 584 Rn. 1`(LIT)

**Example 1** (doc_id: `60126`) (sent_id: `60126`)


Schließlich kann die Jugendkammer aber auch nach § 31 Abs. 3 Satz 1 i. V. m. § 105 Abs. 2 JGG von einer Einbeziehung absehen , wenn dies aus erzieherischen Gründen zweckmäßig ist ( vgl. BGH , Beschluss vom 23. November 1993 - 5 StR 573/93 , BGHSt 40 , 1 , 2 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 23. November 1993 - 5 StR 573/93 , BGHSt 40 , 1 , 2`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 31 Abs. 3 Satz 1 i. V. m. § 105 Abs. 2 JGG`(NRM)
- `BGH , Beschluss vom 23. November 1993 - 5 StR 573/93 , BGHSt 40 , 1 , 2`(RS)

**Example 2** (doc_id: `60279`) (sent_id: `60279`)


Eine Einstellung ohne Sicherheitsleistung kommt dabei nur in Betracht , wenn zusätzlich glaubhaft gemacht wird , dass der Schuldner zu einer Sicherheitsleistung nicht in der Lage ist ( vgl. BGH , Beschluss vom 8. Dezember 2009 - VIII ZR 305/09 , BGHZ 183 , 281 Rn. 6 ff ; Zöller / Herget , ZPO , 32. Aufl. , § 719 Rn. 8 ; MüKoZPO / Götz , 5. Aufl. , § 719 Rn. 15 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 8. Dezember 2009 - VIII ZR 305/09 , BGHZ 183 , 281 Rn. 6 ff`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 8. Dezember 2009 - VIII ZR 305/09 , BGHZ 183 , 281 Rn. 6 ff`(RS)
- `Zöller / Herget , ZPO , 32. Aufl. , § 719 Rn. 8`(LIT)
- `MüKoZPO / Götz , 5. Aufl. , § 719 Rn. 15`(LIT)

**Example 3** (doc_id: `60291`) (sent_id: `60291`)


Anders als in anderen von den Strafsenaten des Bundesgerichtshofs entschiedenen Fallkonstellationen ( vgl. BGH , Beschluss vom 18. Januar 2007 - 4 StR 394/06 , NStZ 2007 , 332 mit Anm. Kudlich JR 2007 , 381 ; Beschluss vom 6. September 2007 - 4 StR 227/07 , StraFo 2008 , 85 ; Beschluss vom 5. Juni 2007 - 4 StR 184/07 , StRR 2007 , 163 ; Beschluss vom 8. Juli 2008 - 3 StR 229/08 , NStZ-RR 2008 , 342 und Urteil vom 15. August 2007 - 5 StR 216/07 , NStZ-RR 2007 , 375 ) steht vorliegend aus Sicht eines objektiven Betrachters fest , dass es sich bei dem vom Angeklagten als Drohmittel verwendeten rund 50 Zentimeter langen Brecheisen aus Metall - ebenso wie bei einem Holzknüppel ( Senat , Beschluss vom 4. September 1998 - 2 StR 390/98 , NStZ-RR 1999 , 15 ) , einem Besenstiel ( BGH , Beschluss vom 20. Mai 1999 - 4 StR 168/99 , NStZ-RR 1999 , 355 ) , einem Schraubendreher ( BGH , Urteil vom 18. Februar 2010 - 3 StR 556/09 , NStZ 2011 , 158 ) oder einem abgesägten Metallstück in Form eines Winkeleisens ( Senat , Beschluss vom 21. November 2001 - 2 StR 400/01 , NStZ-RR 2002 , 108 , 109 ) - um einen objektiv gefährlichen Gegenstand handelt , weil es im Falle seines Einsatzes als Schlag- oder Stichwerkzeug ( vgl. BGH , Beschluss vom 27. März 2014 - 1 StR 24/14 , juris ) geeignet ist , erhebliche Verletzungen herbeizuführen .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 18. Januar 2007 - 4 StR 394/06 , NStZ 2007 , 332 mit Anm. Kudlich JR 2007 , 381`
- `BGH , Beschluss` — similar text (different position): `BGH , Beschluss vom 18. Januar 2007 - 4 StR 394/06 , NStZ 2007 , 332 mit Anm. Kudlich JR 2007 , 381`
- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 18. Februar 2010 - 3 StR 556/09 , NStZ 2011 , 158`
- `BGH , Beschluss` — similar text (different position): `BGH , Beschluss vom 18. Januar 2007 - 4 StR 394/06 , NStZ 2007 , 332 mit Anm. Kudlich JR 2007 , 381`

> overlaps gold: 4  |  likely missing annotation: 0

**Gold Entities:**

- `Strafsenaten`(ORG)
- `Bundesgerichtshofs`(ORG)
- `BGH , Beschluss vom 18. Januar 2007 - 4 StR 394/06 , NStZ 2007 , 332 mit Anm. Kudlich JR 2007 , 381`(RS)
- `Beschluss vom 6. September 2007 - 4 StR 227/07 , StraFo 2008 , 85`(RS)
- `Beschluss vom 5. Juni 2007 - 4 StR 184/07 , StRR 2007 , 163`(RS)
- `Beschluss vom 8. Juli 2008 - 3 StR 229/08 , NStZ-RR 2008 , 342`(RS)
- `Urteil vom 15. August 2007 - 5 StR 216/07 , NStZ-RR 2007 , 375`(RS)
- `Senat , Beschluss vom 4. September 1998 - 2 StR 390/98 , NStZ-RR 1999 , 15`(RS)
- `BGH , Beschluss vom 20. Mai 1999 - 4 StR 168/99 , NStZ-RR 1999 , 355`(RS)
- `BGH , Urteil vom 18. Februar 2010 - 3 StR 556/09 , NStZ 2011 , 158`(RS)
- `Senat , Beschluss vom 21. November 2001 - 2 StR 400/01 , NStZ-RR 2002 , 108 , 109 ) -`(RS)
- `BGH , Beschluss vom 27. März 2014 - 1 StR 24/14 , juris`(RS)

**Example 4** (doc_id: `60567`) (sent_id: `60567`)


Abzurechnen sei die geringer vergütete DRG B76C ( Anfälle , mehr als ein Belegungstag , ohne komplexe Diagnostik u. Therapie , mit schw. CC , Alter < 3 J. od. mit komplexer Diagnose od. m. äußerst schw. CC , Alter > 15 J. od. ohne äußerst schw. od. schw. CC , mit EEG , mit kompl. Diagnose ) .

**False Positives:**

- `CC , Alter` — no gold match — likely missing annotation
- `CC , Alter` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 5** (doc_id: `60622`) (sent_id: `60622`)


In solchen Fällen ist - auch unter Berücksichtigung des Grundsatzes in dubio pro reo - eine nicht auf einer ausreichenden Tatsachengrundlage beruhende und damit letztlich willkürliche Zusammenfassung mehrerer Umsatzgeschäfte zu einer Tat nicht geboten ( st. Rspr. ; vgl. etwa BGH , Beschluss vom 12. Januar 2016 - 3 StR 467/15 , juris Rn. 5 mwN ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 12. Januar 2016 - 3 StR 467/15 , juris Rn. 5`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 12. Januar 2016 - 3 StR 467/15 , juris Rn. 5`(RS)

**Example 6** (doc_id: `60662`) (sent_id: `60662`)


Die Beschwerdeführerin beantragt sinngemäß , den Beschluss des DPMA , Markenstelle für Klasse 41 , vom 26. November 2015 aufzuheben .

**False Positives:**

- `DPMA , Markenstelle` — partial — pred is substring of gold: `DPMA , Markenstelle für Klasse 41`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `DPMA , Markenstelle für Klasse 41`(ORG)

**Example 7** (doc_id: `60734`) (sent_id: `60734`)


Da sich die Klägerin das Verschulden ihres Prozessbevollmächtigten , der eine Berufungs- und Berufungsbegründungsschrift dem Gericht über einen Erklärungsboten zuleitet , nach § 85 Abs. 2 ZPO zurechnen lassen muss ( BGH , Beschluss vom 19. Juni 2007 - VI ZB 81/05 , FamRZ 2007 , 1638 Rn. 6 ) , war der Mangel der Form auch nicht unverschuldet .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 19. Juni 2007 - VI ZB 81/05 , FamRZ 2007 , 1638 Rn. 6`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 85 Abs. 2 ZPO`(NRM)
- `BGH , Beschluss vom 19. Juni 2007 - VI ZB 81/05 , FamRZ 2007 , 1638 Rn. 6`(RS)

**Example 8** (doc_id: `61086`) (sent_id: `61086`)


2. Der Senat kann die Revision durch Beschluss nach § 349 Abs. 2 StPO verwerfen ( vgl. BGH , Beschluss vom 23. Juli 2015 - 1 StR 279/15 , wistra 2015 , 476 mwN ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 23. Juli 2015 - 1 StR 279/15 , wistra 2015 , 476`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 349 Abs. 2 StPO`(NRM)
- `BGH , Beschluss vom 23. Juli 2015 - 1 StR 279/15 , wistra 2015 , 476`(RS)

**Example 9** (doc_id: `61389`) (sent_id: `61389`)


Dies ist in sachlich-rechtlicher Hinsicht der Fall , wenn die Beweiswürdigung widersprüchlich , unklar oder lückenhaft ist oder gegen Denkgesetze oder gesicherte Erfahrungssätze verstößt ( st. Rspr. ; vgl. nur BGH , Urteile vom 1. Februar 2017 - 2 StR 78/16 , NStZ-RR 2017 , 183 , 184 ; vom 13. Juli 2016 - 1 StR 94/16 , juris Rn. 9 und vom 14. September 2017 - 4 StR 45/17 , juris Rn. 7 ) .

**False Positives:**

- `BGH , Urteile` — partial — pred is substring of gold: `BGH , Urteile vom 1. Februar 2017 - 2 StR 78/16 , NStZ-RR 2017 , 183 , 184`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteile vom 1. Februar 2017 - 2 StR 78/16 , NStZ-RR 2017 , 183 , 184`(RS)
- `vom 13. Juli 2016 - 1 StR 94/16 , juris Rn. 9`(RS)
- `vom 14. September 2017 - 4 StR 45/17 , juris Rn. 7`(RS)

**Example 10** (doc_id: `61444`) (sent_id: `61444`)


Die Entscheidung des Beschwerdegerichts , die Rechtsbeschwerde zuzulassen , ist für den Senat nach § 574 Abs. 1 Satz 1 Nr. 2 , Abs. 3 Satz 2 ZPO unabhängig davon bindend , ob es die Voraussetzungen des § 574 Abs. 2 ZPO zutreffend beurteilt hat ( st. Rspr. ; vgl. BGH , Beschlüsse vom 7. Oktober 2008 - XI ZB 24/07 , NJW-RR 2009 , 425 Rn. 9 ; vom 8. Mai 2012 - VIII ZB 91/11 , WuM 2012 , 332 Rn. 3 mwN ) .

**False Positives:**

- `BGH , Beschlüsse` — partial — pred is substring of gold: `BGH , Beschlüsse vom 7. Oktober 2008 - XI ZB 24/07 , NJW-RR 2009 , 425 Rn. 9`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 574 Abs. 1 Satz 1 Nr. 2 , Abs. 3 Satz 2 ZPO`(NRM)
- `§ 574 Abs. 2 ZPO`(NRM)
- `BGH , Beschlüsse vom 7. Oktober 2008 - XI ZB 24/07 , NJW-RR 2009 , 425 Rn. 9`(RS)
- `vom 8. Mai 2012 - VIII ZB 91/11 , WuM 2012 , 332 Rn. 3`(RS)

**Example 11** (doc_id: `61464`) (sent_id: `61464`)


2. Der Senat weist jedoch - den zutreffenden Ausführungen des Generalbundesanwalts in seiner Antragsschrift folgend - darauf hin , dass das Mordmerkmal der Verdeckungsabsicht voraussetzt , dass der Täter die Tötungshandlung vornimmt oder die ihm zur Abwendung des Todeseintritts gebotene Handlung unterlässt , um dadurch eine andere Straftat zu verdecken ( BGH , Urteil vom 12. Dezember 2002 - 4 StR 297/02 , BGHR StGB § 211 Abs. 2 Verdeckung 15 ) .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 12. Dezember 2002 - 4 StR 297/02 , BGHR StGB § 211 Abs. 2 Verdeckung 15`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 12. Dezember 2002 - 4 StR 297/02 , BGHR StGB § 211 Abs. 2 Verdeckung 15`(RS)

**Example 12** (doc_id: `61612`) (sent_id: `61612`)


Bei der Auslegung von Prozesserklärungen ist der Grundsatz zu beachten , dass im Zweifel dasjenige gewollt ist , was nach den Maßstäben der Rechtsordnung vernünftig ist und der wohlverstandenen Interessenlage entspricht ( BGH , Urteil vom 2. Februar 2017 - VII ZR 261/14 , BauR 2017 , 915 Rn. 17 ; Urteil vom 1. August 2013 - VII ZR 268/11 , NJW 2014 , 155 Rn. 30 m. w. N. ) .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 2. Februar 2017 - VII ZR 261/14 , BauR 2017 , 915 Rn. 17`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 2. Februar 2017 - VII ZR 261/14 , BauR 2017 , 915 Rn. 17`(RS)
- `Urteil vom 1. August 2013 - VII ZR 268/11 , NJW 2014 , 155 Rn. 30`(RS)

**Example 13** (doc_id: `61650`) (sent_id: `61650`)


Vielmehr erlegt er Nicht-Konventionsstaaten grundsätzlich keine Standards der Europäischen Menschenrechtskonvention auf ( vgl. EGMR , Urteil vom 4. September 2014 - 140/10 - , Trabelsi / Belgien , Rn. 119 ) .

**False Positives:**

- `EGMR , Urteil` — partial — pred is substring of gold: `EGMR , Urteil vom 4. September 2014 - 140/10 - , Trabelsi / Belgien , Rn. 119`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europäischen Menschenrechtskonvention`(NRM)
- `EGMR , Urteil vom 4. September 2014 - 140/10 - , Trabelsi / Belgien , Rn. 119`(RS)

**Example 14** (doc_id: `61695`) (sent_id: `61695`)


Nicht erforderlich ist , dass die Verwaltung die fragliche Regelung statt durch Vertrag auch durch Verwaltungsakt regeln könnte ; neben derartigen subordinationsrechtlichen Verträgen ( vgl. § 54 Satz 2 VwVfG ) sind auch koordinationsrechtliche öffentlich-rechtliche Verträge denkbar , und nicht nur zwischen mehreren Verwaltungsträgern ( vgl. GmS-OGB , Beschluss vom 10. April 1986 - 1/85 - BVerwGE 74 , 368 ff. Rn. 11 ; BVerwG , Beschluss vom 17. November 2008 - 6 B 41.08 - Buchholz 442.066 § 75 TKG Nr. 1 ) .

**False Positives:**

- `OGB , Beschluss` — partial — pred is substring of gold: `GmS-OGB , Beschluss vom 10. April 1986 - 1/85 - BVerwGE 74 , 368 ff. Rn. 11`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 54 Satz 2 VwVfG`(NRM)
- `GmS-OGB , Beschluss vom 10. April 1986 - 1/85 - BVerwGE 74 , 368 ff. Rn. 11`(RS)
- `BVerwG , Beschluss vom 17. November 2008 - 6 B 41.08 - Buchholz 442.066 § 75 TKG Nr. 1`(RS)

**Example 15** (doc_id: `61737`) (sent_id: `61737`)


Für einen Unterstützungsstreik hat der Europäische Gerichtshof für Menschenrechte entschieden , dass dieser nicht den Kernbereich der Vereinigungsfreiheit betreffe , sondern lediglich einen Nebenaspekt darstelle und daher dem betroffenen Staat bei Einschränkungen ein weiterer Beurteilungsspielraum zuzugestehen sei ( vgl. EGMR , National Union of Rail , Maritime and Transport Workers v. United Kingdom , Urteil vom 8. April 2014 , Nr. 31045/10 , § 88 ) .

**False Positives:**

- `EGMR , National` — partial — pred is substring of gold: `EGMR , National Union of Rail , Maritime and Transport Workers v. United Kingdom , Urteil vom 8. April 2014 , Nr. 31045/10 , § 88`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europäische Gerichtshof für Menschenrechte`(ORG)
- `EGMR , National Union of Rail , Maritime and Transport Workers v. United Kingdom , Urteil vom 8. April 2014 , Nr. 31045/10 , § 88`(RS)

**Example 16** (doc_id: `61912`) (sent_id: `61912`)


Die gerichtliche Fürsorgepflicht greift nicht so weit , dass in Fällen , in denen die Unterschrift unter einem bestimmenden Schriftsatz mit dem Zusatz " i. A. " versehen ist , das Gericht innerhalb einer noch laufenden Frist auf den Mangel der Form hinweisen müsste ( BGH , Beschluss vom 20. Juni 2012 - IV ZB 18/11 , NJW-RR 2012 , 1269 Rn. 12 ff. ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 20. Juni 2012 - IV ZB 18/11 , NJW-RR 2012 , 1269 Rn. 12 ff.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 20. Juni 2012 - IV ZB 18/11 , NJW-RR 2012 , 1269 Rn. 12 ff.`(RS)

**Example 17** (doc_id: `61924`) (sent_id: `61924`)


Die einstweilige Einstellung der Zwangsvollstreckung kommt allerdings nicht in Betracht , wenn das Rechtsmittel der Nichtzulassungsbeschwerde keine Aussicht auf Erfolg hat ( vgl. BGH , Beschluss vom 23. März 2016 - VIII ZR 26/16 , juris Rn. 5 mwN ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 23. März 2016 - VIII ZR 26/16 , juris Rn. 5`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 23. März 2016 - VIII ZR 26/16 , juris Rn. 5`(RS)

**Example 18** (doc_id: `61933`) (sent_id: `61933`)


Die zum beendeten Versuch führende gedankliche Indifferenz des Täters gegenüber den von ihm bis dahin angestrebten oder doch zumindest in Kauf genommenen Konsequenzen ist eine innere Tatsache , die festgestellt werden muss , wozu es in der Regel einer zusammenfassenden Würdigung aller maßgeblichen objektiven Umstände bedarf ( BGH , Urteile vom 2. November 1994 , 2 StR 449/94 , aaO und vom 3. Juni 2008 - 1 StR 59/08 , NStZ 2009 , 264 ; Beschlüsse vom 22. Mai 2013 - 4 StR 170/13 , NStZ 2013 , 703 , 704 , und vom 27. Januar 2014 - 4 StR 565/13 , NStZ-RR 2014 , 202 f. ) .

**False Positives:**

- `BGH , Urteile` — partial — pred is substring of gold: `BGH , Urteile vom 2. November 1994 , 2 StR 449/94 , aaO`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteile vom 2. November 1994 , 2 StR 449/94 , aaO`(RS)
- `vom 3. Juni 2008 - 1 StR 59/08 , NStZ 2009 , 264`(RS)
- `Beschlüsse vom 22. Mai 2013 - 4 StR 170/13 , NStZ 2013 , 703 , 704`(RS)
- `vom 27. Januar 2014 - 4 StR 565/13 , NStZ-RR 2014 , 202 f.`(RS)

**Example 19** (doc_id: `61959`) (sent_id: `61959`)


Soweit die Anmelderin in Klasse 2 ferner „ Naturharze im Rohzustand “ beansprucht , stellen diese im Bereich von Lacken und ( Öl- ) Farben einen üblichen Inhaltsstoff dar , der auch als Zusatz im Malereibedarf in Betracht kommt ( vgl. BPatG PAVIS PROMA , Beschluss vom 23. April 2013 , 25 W ( pat ) 62/12 - Macchiato ) .

**False Positives:**

- `PROMA , Beschluss` — partial — pred is substring of gold: `BPatG PAVIS PROMA , Beschluss vom 23. April 2013 , 25 W ( pat ) 62/12 - Macchiato`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG PAVIS PROMA , Beschluss vom 23. April 2013 , 25 W ( pat ) 62/12 - Macchiato`(RS)

**Example 20** (doc_id: `61975`) (sent_id: `61975`)


Darüber , ob die Dauer eines Verfahrens angemessen ist , muss unter Berücksichtigung der Schwierigkeit des Falles , des Verhaltens des Beschwerdeführers und der zuständigen Behörden und Gerichte sowie der Bedeutung des Rechtsstreits für den Beschwerdeführer entschieden werden ( EGMR , Urteil vom 2. September 2010 , Nr. 46344/06 , Rumpf . / . Deutschland , Z. 41 , NJW 2010 , S. 3355 < 3356 > ; Urteil vom 21. Oktober 2010 , Nr. 43155/08 , Grumann . / . Deutschland , Z. 26 , NJW 2011 , S. 1055 < 1056 > ; BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 19 ) .

**False Positives:**

- `EGMR , Urteil` — partial — pred is substring of gold: `EGMR , Urteil vom 2. September 2010 , Nr. 46344/06 , Rumpf . / . Deutschland , Z. 41 , NJW 2010 , S. 3355 < 3356 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EGMR , Urteil vom 2. September 2010 , Nr. 46344/06 , Rumpf . / . Deutschland , Z. 41 , NJW 2010 , S. 3355 < 3356 >`(RS)
- `Urteil vom 21. Oktober 2010 , Nr. 43155/08 , Grumann . / . Deutschland , Z. 26 , NJW 2011 , S. 1055 < 1056 >`(RS)
- `BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 19`(RS)

**Example 21** (doc_id: `62085`) (sent_id: `62085`)


Innerhalb kürzester Zeit kann das schuldnerische Unternehmen durch den Verlust von Kunden , Lieferanten und Arbeitnehmern auseinander fallen ( vgl. BGH , Beschluss vom 27. Juli 2006 - IX ZB 204/04 - BGHZ 169 , 17 Rn. 12 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 27. Juli 2006 - IX ZB 204/04 - BGHZ 169 , 17 Rn. 12`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 27. Juli 2006 - IX ZB 204/04 - BGHZ 169 , 17 Rn. 12`(RS)

**Example 22** (doc_id: `62148`) (sent_id: `62148`)


a ) Ob der Rechtsmittelführer nur einzelne abtrennbare Teile eines Urteils angreifen will , ist eine Frage , die im Zweifelsfall im Wege der Auslegung seiner Rechtsmittelerklärungen zu beantworten ist ( vgl. BGH , Urteil vom 2. Februar 2017 - 4 StR 481/16 , NStZ-RR 2017 , 105 , 106 ; Beschluss vom 21. Oktober 1980 - 1 StR 262/80 , BGHSt 29 , 359 , 365 [ zu § 318 StPO ] ) .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 2. Februar 2017 - 4 StR 481/16 , NStZ-RR 2017 , 105 , 106`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 2. Februar 2017 - 4 StR 481/16 , NStZ-RR 2017 , 105 , 106`(RS)
- `Beschluss vom 21. Oktober 1980 - 1 StR 262/80 , BGHSt 29 , 359 , 365 [ zu § 318 StPO ]`(RS)

**Example 23** (doc_id: `62522`) (sent_id: `62522`)


Die Übertragung anfallender Arbeiten auf Büropersonal setzt voraus , dass es sich um geschultes , als zuverlässig erprobtes und sorgfältig überwachtes Personal handelt ( vgl. BGH , Beschluss vom 10. März 2011 - VII ZB 37/10 , NJW 2011 , 1597 Rn. 10 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 10. März 2011 - VII ZB 37/10 , NJW 2011 , 1597 Rn. 10`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 10. März 2011 - VII ZB 37/10 , NJW 2011 , 1597 Rn. 10`(RS)

**Example 24** (doc_id: `62527`) (sent_id: `62527`)


2. An der Besteuerung wird Deutschland durch das DBA-Kanada 2001 nicht gehindert ; der Senat hält insoweit an der in einem Verfahren des einstweiligen Rechtsschutzes bereits geäußerten Rechtsauffassung ( Senatsbeschluss vom 13. Dezember 2011 I B 159/11 , BFH / NV 2012 , 417 ) auch nach erneuter Prüfung fest ( dieser Rechtslage zustimmend z.B. Gosch in Kirchhof , EStG , 16. Aufl. , § 49 Rz 90 mit Fußn. 6 ; Hick in Schönfeld / Ditz , DBA , Art. 18 Rz 167 f. , 171 , 174 ; Blümich / Wied , § 49 EStG Rz 218 ; Kuhn in Herrmann / Heuer / Raupach , § 49 EStG Rz 1003 ; Kühnen , EFG 2016 , 578 ; Schober , EFG 2016 , 990 ; Ismer in Vogel / Lehner , DBA , 6. Aufl. , Art. 18 Rz 87 ; Holthaus , Internationale Wirtschaftsbriefe 2017 , 796 , 797 ; im Ergebnis ebenso die Verwaltungspraxis , s. Bayerisches Landesamt für Steuern , Verfügung vom 8. Juni 2011 , Internationales Steuerrecht 2011 , 776 ; a. A. W. Wassermeyer in Wassermeyer , DBA , Kanada Art. 18 Rz 70a ; Hagemann / Kahlenberg / Cloer , Betriebs-Berater 2017 , 2775 , 2785 f. ) .

**False Positives:**

- `DBA , Art` — partial — pred is substring of gold: `Hick in Schönfeld / Ditz , DBA , Art. 18 Rz 167 f. , 171 , 174`
- `DBA , Kanada` — partial — pred is substring of gold: `W. Wassermeyer in Wassermeyer , DBA , Kanada Art. 18 Rz 70a`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Deutschland`(LOC)
- `DBA-Kanada 2001`(NRM)
- `Senatsbeschluss vom 13. Dezember 2011 I B 159/11 , BFH / NV 2012 , 417`(RS)
- `Gosch in Kirchhof , EStG , 16. Aufl. , § 49 Rz 90 mit Fußn. 6`(LIT)
- `Hick in Schönfeld / Ditz , DBA , Art. 18 Rz 167 f. , 171 , 174`(LIT)
- `Blümich / Wied , § 49 EStG Rz 218`(LIT)
- `Kuhn in Herrmann / Heuer / Raupach , § 49 EStG Rz 1003`(LIT)
- `Kühnen , EFG 2016 , 578`(LIT)
- `Schober , EFG 2016 , 990`(LIT)
- `Ismer in Vogel / Lehner , DBA , 6. Aufl. , Art. 18 Rz 87`(LIT)
- `Holthaus , Internationale Wirtschaftsbriefe 2017 , 796 , 797`(LIT)
- `Bayerisches Landesamt für Steuern , Verfügung vom 8. Juni 2011 , Internationales Steuerrecht 2011 , 776`(LIT)
- `W. Wassermeyer in Wassermeyer , DBA , Kanada Art. 18 Rz 70a`(LIT)
- `Hagemann / Kahlenberg / Cloer , Betriebs-Berater 2017 , 2775 , 2785 f.`(LIT)

**Example 25** (doc_id: `62563`) (sent_id: `62563`)


Durch diese wird ein durchgreifender Rechtsfehler nicht aufgezeigt , so dass zu weitergehenden Ausführungen kein Anlass besteht ( st. Rspr. ; vgl. aus neuerer Zeit etwa BVerfG , Beschluss vom 30. Juni 2014 - 2 BvR 792/11 , NJW 2014 , 2563 , 2564 ; BGH , Beschluss vom 4. April 2016 - 1 StR 406/15 , NStZ-RR 2016 , 251 , 252 jeweils mwN ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 4. April 2016 - 1 StR 406/15 , NStZ-RR 2016 , 251 , 252`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG , Beschluss vom 30. Juni 2014 - 2 BvR 792/11 , NJW 2014 , 2563 , 2564`(RS)
- `BGH , Beschluss vom 4. April 2016 - 1 StR 406/15 , NStZ-RR 2016 , 251 , 252`(RS)

**Example 26** (doc_id: `62654`) (sent_id: `62654`)


Denn ohne Kenntnis der bereits teilweise in die Hauptverhandlung eingeführten Aussage kann er insbesondere sein Fragerecht gegenüber weiteren Zeugen grundsätzlich nicht sachgerecht ausüben ( vgl. BGH , Urteil vom 31. März 1992 aaO ; Beschluss vom 6. September 1989 - 3 StR 235/89 , BGHR StPO § 247 Satz 4 Unterrichtung 3 ) .

**False Positives:**

- `BGH , Urteil` — partial — gold is substring of pred: `BGH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH`(ORG)
- `Beschluss vom 6. September 1989 - 3 StR 235/89 , BGHR StPO § 247 Satz 4 Unterrichtung 3`(RS)

**Example 27** (doc_id: `62658`) (sent_id: `62658`)


Die 39. BImSchV dient unter anderem der Umsetzung der Richtlinie 2008 / 50 / EG des Europäischen Parlaments und des Rates vom 21. Mai 2008 über Luftqualität und saubere Luft für Europa ( ABl. L 152 S. 1 ) , in der die ab 1. Januar 2010 einzuhaltenden , vom Verordnungsgeber übernommenen Grenzwerte in Anhang XI , Abschnitt B , festgelegt sind .

**False Positives:**

- `XI , Abschnitt` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `39. BImSchV`(NRM)
- `Richtlinie 2008 / 50 / EG des Europäischen Parlaments und des Rates vom 21. Mai 2008 über Luftqualität und saubere Luft für Europa ( ABl. L 152 S. 1 )`(NRM)

**Example 28** (doc_id: `62668`) (sent_id: `62668`)


Dabei wurden nach dem Vortrag des Beschwerdeführers unter anderem ein PC , Laptops , zwei digitale Videokameras sowie mehrere USB-Datenträger sichergestellt .

**False Positives:**

- `PC , Laptops` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 29** (doc_id: `62804`) (sent_id: `62804`)


Jede spätere sachgrundlose Befristung sei gemäß § 14 Abs. 2 Satz 2 TzBfG unwirksam ( vgl. BAG , Urteil vom 6. November 2003 - 2 AZR 690/02 - , BAGE 108 , 269 < 274 > ; Urteil vom 13. Mai 2004 - 2 AZR 426/03 - , juris , Rn. 28 ; Beschluss vom 29. Juli 2009 - 7 AZN 368/09 - , www.bag.de , Rn. 2 ) .

**False Positives:**

- `BAG , Urteil` — partial — pred is substring of gold: `BAG , Urteil vom 6. November 2003 - 2 AZR 690/02 - , BAGE 108 , 269 < 274 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 14 Abs. 2 Satz 2 TzBfG`(NRM)
- `BAG , Urteil vom 6. November 2003 - 2 AZR 690/02 - , BAGE 108 , 269 < 274 >`(RS)
- `Urteil vom 13. Mai 2004 - 2 AZR 426/03 - , juris , Rn. 28`(RS)
- `Beschluss vom 29. Juli 2009 - 7 AZN 368/09 - , www.bag.de , Rn. 2`(RS)

**Example 30** (doc_id: `62907`) (sent_id: `62907`)


Soweit das Landgericht darüber hinaus schon bei der Abgrenzung des beendeten vom unbeendeten Versuch auf den Grundsatz in dubio pro reo zurückgreift , begegnet auch dies keinen durchgreifenden rechtlichen Bedenken ( vgl. BGH , Beschluss vom 22. Mai 2013 - 4 StR 170/13 , NStZ 2013 , 703 , 704 ; Urteil vom 8. Dezember 2010 - 2 StR 536/10 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 22. Mai 2013 - 4 StR 170/13 , NStZ 2013 , 703 , 704`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 22. Mai 2013 - 4 StR 170/13 , NStZ 2013 , 703 , 704`(RS)
- `Urteil vom 8. Dezember 2010 - 2 StR 536/10`(RS)

**Example 31** (doc_id: `62974`) (sent_id: `62974`)


Da jedoch jegliche Ausführungen dazu fehlen , warum die Patentabteilung eine Aufrechterhaltung des Patents nach Hauptantrag ausschließt , ist der Beschluss insoweit jedenfalls nicht mit Gründen versehen ( vgl. BGH , Beschluss vom 16. Oktober 1973 – X ZB 15/72 , GRUR 1974 , 294 , II. 2. c ) – Richterwechsel II ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 16. Oktober 1973 – X ZB 15/72 , GRUR 1974 , 294 , II. 2. c ) – Richterwechsel II`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 16. Oktober 1973 – X ZB 15/72 , GRUR 1974 , 294 , II. 2. c ) – Richterwechsel II`(RS)

**Example 32** (doc_id: `63041`) (sent_id: `63041`)


Fehlt eine solche , kann sich die erhöhte Beweiskraft mittelbar - unter Beachtung der Anschauung des Rechtsverkehrs - aus den Vorschriften ergeben , die für die Errichtung und den Zweck der Urkunde maßgeblich sind ( vgl. BGH , Urteile vom 12. Oktober 1995 - 4 StR 259/95 , NJW 1996 , 470 ; vom 16. April 1996 - 1 StR 127/96 , BGHSt 42 , 131 ; vom 25. Mai 2001 - 2 StR 88/01 , BGHSt 47 , 39 , 42 ; Beschluss vom 14. Juni 2016 - 3 StR 128/16 , aaO ) .

**False Positives:**

- `BGH , Urteile` — partial — pred is substring of gold: `BGH , Urteile vom 12. Oktober 1995 - 4 StR 259/95 , NJW 1996 , 470`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteile vom 12. Oktober 1995 - 4 StR 259/95 , NJW 1996 , 470`(RS)
- `vom 16. April 1996 - 1 StR 127/96 , BGHSt 42 , 131`(RS)
- `vom 25. Mai 2001 - 2 StR 88/01 , BGHSt 47 , 39 , 42`(RS)
- `Beschluss vom 14. Juni 2016 - 3 StR 128/16 , aaO`(RS)

**Example 33** (doc_id: `63070`) (sent_id: `63070`)


Zu den besonderen Begleitumständen gehören der Gang und der Inhalt der Vertragsverhandlungen sowie der äußere Zuschnitt des Vertrags ( vgl. BGH , Urteil vom 21. Juni 2016 - IX ZR 475/15 , VersR 2016 , 1330 , 1331 ; Urteil vom 20. Februar 2014 - IX ZR 137/13 , NJW-RR 2014 , 937 , 938 ; Urteil vom 18. Mai 1995 - IX ZR 108/94 , BGHZ 130 , 19 , 25 [ zu der gleichlautenden Vorschrift in § 3 AGBG ] ; weitere Nachweise bei Basedow in : Münch. Komm. z. BGB , 7. Aufl. , § 305c Rn. 6 ) .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 21. Juni 2016 - IX ZR 475/15 , VersR 2016 , 1330 , 1331`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 21. Juni 2016 - IX ZR 475/15 , VersR 2016 , 1330 , 1331`(RS)
- `Urteil vom 20. Februar 2014 - IX ZR 137/13 , NJW-RR 2014 , 937 , 938`(RS)
- `Urteil vom 18. Mai 1995 - IX ZR 108/94 , BGHZ 130 , 19 , 25`(RS)
- `§ 3 AGBG`(NRM)
- `Basedow in : Münch. Komm. z. BGB , 7. Aufl. , § 305c Rn. 6`(LIT)

**Example 34** (doc_id: `63246`) (sent_id: `63246`)


Dabei hat die Partei darzulegen , dass die Beendigung des Mandats nicht auf ihr Verschulden zurückzuführen ist ( BGH , Beschluss vom 18. Dezember 2013 aaO mwN ) .

**False Positives:**

- `BGH , Beschluss` — partial — gold is substring of pred: `BGH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH`(ORG)

**Example 35** (doc_id: `63443`) (sent_id: `63443`)


An diesem Tag zählte die kreisfreie Stadt Sch. 127815 Einwohner , der Kreis Sch. Land einschließlich der kreisangehörigen Stadt C. 33997 Einwohner ( Statistisches Landesamt Mecklenburg-Vorpommern , Statistische Berichte , Unterreihe A. IS , Bevölkerungsstand der Kreise und Gemeinden des Landes Mecklenburg-Vorpommern , Sch. 1990 ) .

**False Positives:**

- `IS , Bevölkerungsstand` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Sch.`(LOC)
- `Sch. Land`(LOC)
- `C.`(LOC)
- `Statistisches Landesamt Mecklenburg-Vorpommern`(ORG)
- `Mecklenburg-Vorpommern`(LOC)
- `Sch.`(LOC)

**Example 36** (doc_id: `63487`) (sent_id: `63487`)


Die UN-Behindertenrechtskonvention ist nach dem Gesetz vom 21. Dezember 2008 ( BGBl. II S. 1419 ) seit dem 1. Januar 2009 als innerstaatliches Recht im Rang einfachen Bundesrechts anzuwenden und kann als Auslegungshilfe für die Bestimmung und den Inhalt der Grundrechte ( BVerfG , Beschluss vom 23. März 2011 - 2 BvR 882/09 - BVerfGE 128 , 282 Rn. 52 ; BSG , Urteil vom 6. März 2012 - B 1 KR 10/11 R - BSGE 110 , 194 Rn. 31 ) und des einfachen Gesetzesrechts herangezogen werden .

**False Positives:**

- `BSG , Urteil` — partial — pred is substring of gold: `BSG , Urteil vom 6. März 2012 - B 1 KR 10/11 R - BSGE 110 , 194 Rn. 31`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `UN-Behindertenrechtskonvention`(NRM)
- `Gesetz vom 21. Dezember 2008 ( BGBl. II S. 1419 )`(NRM)
- `BVerfG , Beschluss vom 23. März 2011 - 2 BvR 882/09 - BVerfGE 128 , 282 Rn. 52`(RS)
- `BSG , Urteil vom 6. März 2012 - B 1 KR 10/11 R - BSGE 110 , 194 Rn. 31`(RS)

**Example 37** (doc_id: `63538`) (sent_id: `63538`)


2. Die Verständigung der Polizei und die Kenntnis des Angeklagten davon rechtfertigen für sich genommen weder die Annahme eines fehlgeschlagenen Versuchs , noch stehen sie grundsätzlich einer Freiwilligkeit im Sinne des § 24 Abs. 1 Satz 1 StGB entgegen , da ein Täter in der Zeit bis zum Eintreffen derselben grundsätzlich noch ungehindert weitere Ausführungshandlungen vornehmen kann , ohne dass damit für ihn eine beträchtliche Risikoerhöhung verbunden sein muss ( vgl. auch BGH , Beschluss vom 24. Oktober 2017 - 1 StR 393/17 ; zu einer beträchtlichen Risikoerhöhung BGH , Urteil vom 15. September 2005 - 4 StR 216/05 , NStZ-RR 2006 , 168 , [ 169 ] ; Beschluss vom 19. Dezember 2006 - 4 StR 537/06 , NStZ 2007 , 265 , [ 266 ] ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 24. Oktober 2017 - 1 StR 393/17`
- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 15. September 2005 - 4 StR 216/05 , NStZ-RR 2006 , 168 , [ 169 ]`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 24 Abs. 1 Satz 1 StGB`(NRM)
- `BGH , Beschluss vom 24. Oktober 2017 - 1 StR 393/17`(RS)
- `BGH , Urteil vom 15. September 2005 - 4 StR 216/05 , NStZ-RR 2006 , 168 , [ 169 ]`(RS)
- `Beschluss vom 19. Dezember 2006 - 4 StR 537/06 , NStZ 2007 , 265 , [ 266 ]`(RS)

**Example 38** (doc_id: `63693`) (sent_id: `63693`)


§ 30a Abs. 2 Nr. 1 BtMG in der Variante des Bestimmens eines Minderjährigen zum Fördern einer der dort genannten Handlungen erfordert weiter , dass der angestiftete Minderjährige neben den objektiven auch die subjektiven Voraussetzungen einer Beihilfehandlung im Sinne des § 27 StGB verwirklicht ( BGH , Beschluss vom 7. August 2014 - 3 StR 17/14 , NStZ 2015 , 347 f. mwN ; zum Begriff des Förderns vgl. auch MüKoStGB / Öğlakcioğlu , aaO Rn. 57 ; Körner / Patzak / Volkmer , BtMG , 8. Aufl. , § 30a Rn. 41 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 7. August 2014 - 3 StR 17/14 , NStZ 2015 , 347 f.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 30a Abs. 2 Nr. 1 BtMG`(NRM)
- `§ 27 StGB`(NRM)
- `BGH , Beschluss vom 7. August 2014 - 3 StR 17/14 , NStZ 2015 , 347 f.`(RS)
- `MüKoStGB / Öğlakcioğlu , aaO Rn. 57`(LIT)
- `Körner / Patzak / Volkmer , BtMG , 8. Aufl. , § 30a Rn. 41`(LIT)

**Example 39** (doc_id: `63720`) (sent_id: `63720`)


Seine Aufhebung erfolgt nur klarstellend ( vgl. BGH , Beschluss vom 9. Februar 2005 - XII ZB 225/04 , FamRZ 2005 , 791 , 792 ; Senat , Beschluss vom 9. März 2017 - V ZB 18/16 , NJW 2017 , 3002 Rn. 17 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 9. Februar 2005 - XII ZB 225/04 , FamRZ 2005 , 791 , 792`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 9. Februar 2005 - XII ZB 225/04 , FamRZ 2005 , 791 , 792`(RS)
- `Senat , Beschluss vom 9. März 2017 - V ZB 18/16 , NJW 2017 , 3002 Rn. 17`(RS)

**Example 40** (doc_id: `63932`) (sent_id: `63932`)


Zusätzlich müssen die Gründe für die Umverteilung dargelegt und dokumentiert werden ( vgl. BVerfG , Kammerbeschluss vom 18. März 2009 - 2 BvR 229/09 - a. a. O. ; BGH , Urteil vom 9. April 2009 - 3 StR 376/08 - BGHSt 53 , 268 Rn. 11 und 17 ff. und Beschluss vom 10. Juni 2014 - 3 StR 57/14 - juris Rn. 21 ; Kissel / Mayer , GVG , 8. Aufl. 2015 , § 21e Rn. 99 ) .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 9. April 2009 - 3 StR 376/08 - BGHSt 53 , 268 Rn. 11 und 17 ff.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG , Kammerbeschluss vom 18. März 2009 - 2 BvR 229/09 - a. a. O.`(RS)
- `BGH , Urteil vom 9. April 2009 - 3 StR 376/08 - BGHSt 53 , 268 Rn. 11 und 17 ff.`(RS)
- `Beschluss vom 10. Juni 2014 - 3 StR 57/14 - juris Rn. 21`(RS)
- `Kissel / Mayer , GVG , 8. Aufl. 2015 , § 21e Rn. 99`(LIT)

**Example 41** (doc_id: `63980`) (sent_id: `63980`)


[ 6 ] Verwirkung setzt voraus , dass der Berechtigte ein Recht längere Zeit nicht geltend gemacht hat , obwohl er dazu in der Lage gewesen wäre , der Gegner sich mit Rücksicht auf das gesamte Verhalten des Berechtigten darauf einrichten durfte und eingerichtet hat , dass dieser sein Recht auch in Zukunft nicht geltend machen werde , und die verspätete Geltendmachung daher gegen den Grundsatz von Treu und Glauben verstößt ( BGH , Urteil vom 18. Oktober 2001 - I ZR 91/99 - GRUR 2002 , 280 ; BGH , Urteil vom 14. 06. 2004 - II ZR 392/01 - WM 2004 , 1518 , 1520 , jeweils m. w. N. ) .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 18. Oktober 2001 - I ZR 91/99 - GRUR 2002 , 280`
- `BGH , Urteil` — similar text (different position): `BGH , Urteil vom 18. Oktober 2001 - I ZR 91/99 - GRUR 2002 , 280`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 18. Oktober 2001 - I ZR 91/99 - GRUR 2002 , 280`(RS)
- `BGH , Urteil vom 14. 06. 2004 - II ZR 392/01 - WM 2004 , 1518 , 1520`(RS)

**Example 42** (doc_id: `63989`) (sent_id: `63989`)


Zwar verhält sich diese Begründungsschrift vornehmlich zu der als fehlerhaft gewerteten Anwendung von § 213 Alt. 2 StGB seitens des Landgerichts , was von der Nebenklage nicht isoliert gerügt werden kann ( BGH , Beschluss vom 21. April 1999 - 2 StR 64/99 , NStZ-RR 2000 , 40 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 21. April 1999 - 2 StR 64/99 , NStZ-RR 2000 , 40`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 213 Alt. 2 StGB`(NRM)
- `BGH , Beschluss vom 21. April 1999 - 2 StR 64/99 , NStZ-RR 2000 , 40`(RS)

**Example 43** (doc_id: `64147`) (sent_id: `64147`)


Tatsächlich lauten die vom Bundesgerichtshof in den zitierten Entscheidungen formulierten Maßstäbe , nicht die Höhe der vom Sachverständigen erstellten Rechnung als solche , sondern alleine der vom Geschädigten in Übereinstimmung mit der Rechnung und der ihr zugrunde liegenden getroffenen Preisvereinbarung tatsächlich erbrachte Aufwand bilde einen Anhalt zur Bestimmung des zur Herstellung erforderlichen Betrags im Sinne von § 249 Abs. 2 Satz 1 BGB ( namentlich BGH , Urteil vom 19. Juli 2016 - VI ZR 491/15 - , juris , Rn. 19 ) .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 19. Juli 2016 - VI ZR 491/15 - , juris , Rn. 19`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesgerichtshof`(ORG)
- `§ 249 Abs. 2 Satz 1 BGB`(NRM)
- `BGH , Urteil vom 19. Juli 2016 - VI ZR 491/15 - , juris , Rn. 19`(RS)

**Example 44** (doc_id: `64171`) (sent_id: `64171`)


Dieser Zulassungsgrund setzt voraus , dass ein einzelner tragender Rechtssatz oder eine erhebliche Tatsachenfeststellung mit schlüssigen Argumenten in Frage gestellt wird ( st. Rspr. ; vgl. etwa BGH , Beschluss vom 28. Oktober 2011 - AnwZ ( Brfg ) 30/11 , NJW-RR 2012 , 189 Rn. 5 mwN ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 28. Oktober 2011 - AnwZ ( Brfg ) 30/11 , NJW-RR 2012 , 189 Rn. 5`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 28. Oktober 2011 - AnwZ ( Brfg ) 30/11 , NJW-RR 2012 , 189 Rn. 5`(RS)

**Example 45** (doc_id: `64238`) (sent_id: `64238`)


In Verbindung mit den Erläuterungen der Präsidentin des Oberverwaltungsgerichts vom 2. Februar 2018 und den ergänzenden Unterlagen ermöglicht der Präsidiumsbeschluss die Prüfung seiner Rechtmäßigkeit ( vgl. BGH , Beschluss vom 25. März 2015 - 5 StR 70/15 - NStZ 2015 , 658 Rn. 12 m. w. N. ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 25. März 2015 - 5 StR 70/15 - NStZ 2015 , 658 Rn. 12`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 25. März 2015 - 5 StR 70/15 - NStZ 2015 , 658 Rn. 12`(RS)

**Example 46** (doc_id: `64263`) (sent_id: `64263`)


Es hat jedoch nicht über den mit dem Schriftsatz vom 8. Dezember 2017 vorgeschalteten , nach herrschender Auffassung nicht an Fristen oder Antragsteller-Quoren gebundenen ( vgl. für die Wahl zum Gesamtvertrauenspersonenausschuss Höges , SBG , Stand März 2018 , § 47 Rn. 20 ) ( Haupt- ) Antrag , die Nichtigkeit der Wahl festzustellen , entschieden .

**False Positives:**

- `SBG , Stand` — partial — pred is substring of gold: `Wahl zum Gesamtvertrauenspersonenausschuss Höges , SBG , Stand März 2018 , § 47 Rn. 20`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Wahl zum Gesamtvertrauenspersonenausschuss Höges , SBG , Stand März 2018 , § 47 Rn. 20`(LIT)

**Example 47** (doc_id: `64438`) (sent_id: `64438`)


Der Anmelderin kann zugestimmt werden , dass der von der Prüfungsstelle herangezogene BGH-Beschluss „ Buchungsblatt “ ( vgl. BGH , Beschluss vom 18. März 1975 , X ZB 9/74 , GRUR 1975 , 549 ) hinsichtlich der dort geforderten unmittelbaren Einwirkung auf die Außenwelt überholt und inhaltlich völlig am Anmeldungsgegenstand vorbei geht .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 18. März 1975 , X ZB 9/74 , GRUR 1975 , 549`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 18. März 1975 , X ZB 9/74 , GRUR 1975 , 549`(RS)

**Example 48** (doc_id: `64483`) (sent_id: `64483`)


Dieser Ansicht kann jedenfalls im vorliegenden Fall nicht zugestimmt werden ( Fezer , Markenrecht , 4. Aufl. , § 14 MarkenG Rn. 860 ; Ebert-Weidenfeller in : Achenbach / Ransiek / Rönnau , Handbuch des Wirtschaftsstrafrechts , 4. Aufl. , Kapitel Markenstrafrecht Rn. 69 ; vgl. auch BGH , Urteil vom 10. Juni 1998 - 5 StR 72/98 , StV 1998 , 663 ) .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 10. Juni 1998 - 5 StR 72/98 , StV 1998 , 663`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Fezer , Markenrecht , 4. Aufl. , § 14 MarkenG Rn. 860`(LIT)
- `Ebert-Weidenfeller in : Achenbach / Ransiek / Rönnau , Handbuch des Wirtschaftsstrafrechts , 4. Aufl. , Kapitel Markenstrafrecht Rn. 69`(LIT)
- `BGH , Urteil vom 10. Juni 1998 - 5 StR 72/98 , StV 1998 , 663`(RS)

**Example 49** (doc_id: `64552`) (sent_id: `64552`)


Ab dem 25. Mai 2018 sind die datenschutzrechtlich Verantwortlichen nach Art. 30 DSGVO verpflichtet , ihre Verarbeitungsvorgänge in einem Verzeichnis zu dokumentieren ; nach Absatz 4 der Vorschrift ist das Verzeichnis der Aufsichtsbehörde auf Anfrage zur Verfügung zu stellen ( vgl. auch Raum , in : Auernhammer , DSGVO / BDSG , Kommentar , 5. Aufl. 2017 , § 4d BDSG , Rn. 58 f. ) .

**False Positives:**

- `BDSG , Kommentar` — partial — pred is substring of gold: `Raum , in : Auernhammer , DSGVO / BDSG , Kommentar , 5. Aufl. 2017 , § 4d BDSG , Rn. 58 f.`
- `BDSG , Rn` — partial — pred is substring of gold: `Raum , in : Auernhammer , DSGVO / BDSG , Kommentar , 5. Aufl. 2017 , § 4d BDSG , Rn. 58 f.`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 30 DSGVO`(NRM)
- `Raum , in : Auernhammer , DSGVO / BDSG , Kommentar , 5. Aufl. 2017 , § 4d BDSG , Rn. 58 f.`(LIT)

**Example 50** (doc_id: `64562`) (sent_id: `64562`)


Entgegen der Auffassung von Schulte , a. a. O. , Einleitung Rdn. 236 , erachtet der Senat die Begründung der Nichtzulassung eines von der Einsprechenden nach Ablauf der Einspruchsfrist vorgebrachten neuen Widerrufsgrundes allein mit Verspätung , auch im Hinblick auf die neuere Entscheidung des BGH zur Zulassung neuer Widerrufsgründe durch den beschwerdeführenden Einsprechenden im Einspruchsbeschwerdeverfahren nach Maßgabe des § 263 ZPO ( BGH , Beschluss vom 8. November 2016 , X ZB 1/16 , GRUR 2017 , 54 – Ventileinrichtung ) , für unzureichend .

**False Positives:**

- `BGH , Beschluss` — similar text (different position): `BGH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Schulte , a. a. O. , Einleitung Rdn. 236`(LIT)
- `BGH`(ORG)
- `§ 263 ZPO`(NRM)
- `BGH , Beschluss vom 8. November 2016 , X ZB 1/16 , GRUR 2017 , 54 – Ventileinrichtung`(RS)

**Example 51** (doc_id: `64584`) (sent_id: `64584`)


Dabei hat das Revisionsgericht die tatrichterliche Überzeugungsbildung selbst dann hinzunehmen , wenn eine andere Beurteilung näher gelegen hätte oder überzeugender gewesen wäre ( vgl. BGH , Urteil vom 24. März 2015 - 5 StR 521/14 , NStZ-RR 2015 , 178 , 179 ) .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 24. März 2015 - 5 StR 521/14 , NStZ-RR 2015 , 178 , 179`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 24. März 2015 - 5 StR 521/14 , NStZ-RR 2015 , 178 , 179`(RS)

**Example 52** (doc_id: `64599`) (sent_id: `64599`)


cc ) Schließlich kann die Rolle des Bundesverfassungsgerichts es gebieten , bei der Bearbeitung der Verfahren in stärkerem Maße als in der Fachgerichtsbarkeit andere Umstände zu berücksichtigen als nur die chronologische Reihenfolge der Eintragung in das Gerichtsregister , wenn Verfahren für das Gemeinwesen von besonderer Bedeutung sind oder ihre Entscheidung von dem Ergebnis eines sogenannten Pilotverfahrens abhängig ist ( vgl. BTDrucks 17/3802 , S. 26 ; siehe auch BVerfGK 19 , 110 < 121 > ; 20 , 65 < 73 > ; BVerfG , Beschluss der Beschwerdekammer vom 20. August 2015 - 1 BvR 2781/13 - Vz 11/14 - , NJW 2015 , S. 3361 < 3363 Rn. 31 > ; Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 23 ; EGMR , Urteil vom 25. Februar 2000 , Nr. 29357/95 , Gast und Popp . / . Deutschland , Z. 75 , NJW 2001 , S. 211 < 212 > ; Urteil vom 8. Januar 2004 , Nr. 47169/99 , Voggenreiter . / . Deutschland , Z. 49 , NJW 2005 , S. 41 < 43 > ; Urteil vom 6. November 2008 , Nr. 58911/00 , Leela Förderkreis e. V. u. a. . / . Deutschland , Z. 63 , NVwZ 2010 , S. 177 < 178 > ; Urteil vom 4. September 2014 , Nr. 68919/10 , Peter . / . Deutschland , Z. 40 , NJW 2015 , S. 3359 < 3360 > ) .

**False Positives:**

- `EGMR , Urteil` — partial — pred is substring of gold: `EGMR , Urteil vom 25. Februar 2000 , Nr. 29357/95 , Gast und Popp . / . Deutschland , Z. 75 , NJW 2001 , S. 211 < 212 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesverfassungsgerichts`(ORG)
- `BTDrucks 17/3802 , S. 26`(LIT)
- `BVerfGK 19 , 110 < 121 > ; 20 , 65 < 73 >`(RS)
- `BVerfG , Beschluss der Beschwerdekammer vom 20. August 2015 - 1 BvR 2781/13 - Vz 11/14 - , NJW 2015 , S. 3361 < 3363 Rn. 31 >`(RS)
- `Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 23`(RS)
- `EGMR , Urteil vom 25. Februar 2000 , Nr. 29357/95 , Gast und Popp . / . Deutschland , Z. 75 , NJW 2001 , S. 211 < 212 >`(RS)
- `Urteil vom 8. Januar 2004 , Nr. 47169/99 , Voggenreiter . / . Deutschland , Z. 49 , NJW 2005 , S. 41 < 43 >`(RS)
- `Urteil vom 6. November 2008 , Nr. 58911/00 , Leela Förderkreis e. V. u. a. . / . Deutschland , Z. 63 , NVwZ 2010 , S. 177 < 178 >`(RS)
- `Urteil vom 4. September 2014 , Nr. 68919/10 , Peter . / . Deutschland , Z. 40 , NJW 2015 , S. 3359 < 3360 >`(RS)

**Example 53** (doc_id: `64614`) (sent_id: `64614`)


Hierbei handelt es sich um einen Antrag ( § 145 BGB ) auf Abschluss eines Änderungsvertrages ( Palandt / Weidenkaff , BGB , 77. Aufl. , § 558b Rn. 3 ; § 558a Rn. 2 ; Staudinger / V. Emmerich , BGB , Neubearb. 2018 , § 558a Rn. 2 ; § 558b Rn. 3 ; MünchKommBGB / Artz , BGB , 7. Aufl. , § 558b Rn. 3 ; vgl. auch BayObLG , NJW-RR 1993 , 202 mwN [ zu § 2 MHG ] ) .

**False Positives:**

- `BGB , Neubearb` — partial — pred is substring of gold: `Staudinger / V. Emmerich , BGB , Neubearb. 2018 , § 558a Rn. 2 ; § 558b Rn. 3`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 145 BGB`(NRM)
- `Palandt / Weidenkaff , BGB , 77. Aufl. , § 558b Rn. 3 ; § 558a Rn. 2`(LIT)
- `Staudinger / V. Emmerich , BGB , Neubearb. 2018 , § 558a Rn. 2 ; § 558b Rn. 3`(LIT)
- `MünchKommBGB / Artz , BGB , 7. Aufl. , § 558b Rn. 3`(LIT)
- `BayObLG , NJW-RR 1993 , 202 mwN [ zu § 2 MHG ]`(RS)

**Example 54** (doc_id: `64669`) (sent_id: `64669`)


Erst die Kenntnis des Umstandes , dass ihm neben der zur Bewährung ausgesetzten Freiheitsstrafe weitere Maßnahmen mit Vergeltungscharakter drohen , die - wie hier in Form der Zahlungsauflage nebst kumulativ verhängter Arbeitsauflage - eine erhebliche Belastung darstellen können , versetzt den Angeklagten in die Lage , von seiner Entscheidungsfreiheit , ob er auf das Angebot des Gerichts eingehen möchte , auf einer hinreichenden tatsächlichen Grundlage Gebrauch zu machen ( BGH , Beschlüsse vom 8. September 2016 - 1 StR 346/16 , NStZ-RR 2016 , 379 , 380 ; vom 29. Januar 2014 - 4 StR 254/13 , BGHSt 59 , 172 , 174 f. und vom 11. September 2014 - 4 StR 148/14 , NJW 2014 , 3173 ) .

**False Positives:**

- `BGH , Beschlüsse` — partial — pred is substring of gold: `BGH , Beschlüsse vom 8. September 2016 - 1 StR 346/16 , NStZ-RR 2016 , 379 , 380`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschlüsse vom 8. September 2016 - 1 StR 346/16 , NStZ-RR 2016 , 379 , 380`(RS)
- `vom 29. Januar 2014 - 4 StR 254/13 , BGHSt 59 , 172 , 174 f.`(RS)
- `vom 11. September 2014 - 4 StR 148/14 , NJW 2014 , 3173`(RS)

**Example 55** (doc_id: `64757`) (sent_id: `64757`)


Demgegenüber gehört nach ständiger Rechtsprechung des Bundesgerichtshofs zum Vorsatz der Steuerhinterziehung , dass der Täter den Steueranspruch dem Grunde und der Höhe nach kennt oder zumindest für möglich hält und ihn auch verkürzen will ( vgl. BGH , Urteile vom 13. November 1953 - 5 StR 342/53 , BGHSt 5 , 90 , 91 f. und vom 5. März 1986 - 2 StR 666/85 , wistra 1986 , 174 ; Beschlüsse vom 19. Mai 1989 - 3 StR 590/88 , BGHR AO § 370 Abs. 1 Vorsatz 2 ; vom 24. Oktober 1990 - 3 StR 16/90 , BGHR AO § 370 Abs. 1 Vorsatz 4 und vom 8. September 2011 - 1 StR 38/11 , NStZ 2012 , 160 , 161 Rn. 21 f. ) .

**False Positives:**

- `BGH , Urteile` — partial — pred is substring of gold: `BGH , Urteile vom 13. November 1953 - 5 StR 342/53 , BGHSt 5 , 90 , 91 f.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesgerichtshofs`(ORG)
- `BGH , Urteile vom 13. November 1953 - 5 StR 342/53 , BGHSt 5 , 90 , 91 f.`(RS)
- `vom 5. März 1986 - 2 StR 666/85 , wistra 1986 , 174`(RS)
- `Beschlüsse vom 19. Mai 1989 - 3 StR 590/88 , BGHR AO § 370 Abs. 1 Vorsatz 2`(RS)
- `vom 24. Oktober 1990 - 3 StR 16/90 , BGHR AO § 370 Abs. 1 Vorsatz`(RS)
- `vom 8. September 2011 - 1 StR 38/11 , NStZ 2012 , 160 , 161 Rn. 21 f.`(RS)

**Example 56** (doc_id: `64761`) (sent_id: `64761`)


Zwar kann nach dieser Rechtsprechung ein ständiger Rückstand infolge chronischer Überlastung auch beim Bundesverfassungsgericht eine überlange Verfahrensdauer nicht rechtfertigen ( EGMR , Urteil vom 25. Februar 2000 , Nr. 29357/95 , Gast und Popp . / . Deutschland , Z. 78 , NJW 2001 , S. 211 < 212 > ; Urteil vom 27. Juli 2000 , Nr. 33379/96 , Klein . / . Deutschland , Z. 29 und 43 , NJW 2001 , S. 213 < 213 , 214 > ; BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 27 f. ) .

**False Positives:**

- `EGMR , Urteil` — partial — pred is substring of gold: `EGMR , Urteil vom 25. Februar 2000 , Nr. 29357/95 , Gast und Popp . / . Deutschland , Z. 78 , NJW 2001 , S. 211 < 212 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesverfassungsgericht`(ORG)
- `EGMR , Urteil vom 25. Februar 2000 , Nr. 29357/95 , Gast und Popp . / . Deutschland , Z. 78 , NJW 2001 , S. 211 < 212 >`(RS)
- `Urteil vom 27. Juli 2000 , Nr. 33379/96 , Klein . / . Deutschland , Z. 29 und 43 , NJW 2001 , S. 213 < 213 , 214 >`(RS)
- `BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 27 f.`(RS)

**Example 57** (doc_id: `64772`) (sent_id: `64772`)


Er setzt auch nicht voraus , dass die Rauschmittelgewöhnung auf täglichen oder häufig wiederholten Genuss zurückgeht ; vielmehr kann es genügen , wenn der Täter von Zeit zu Zeit oder bei passender Gelegenheit seiner Neigung zum Rauschmittelkonsum folgt ( BGH , Beschluss vom 7. Januar 2009 - 5 StR 586/08 , NStZ-RR 2009 , 137 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 7. Januar 2009 - 5 StR 586/08 , NStZ-RR 2009 , 137`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 7. Januar 2009 - 5 StR 586/08 , NStZ-RR 2009 , 137`(RS)

**Example 58** (doc_id: `64923`) (sent_id: `64923`)


Eine solche Abrede steht der Annahme einer Einlage oder anderer unbedingt rückzahlbarer Gelder des Publikums und damit eines Bankgeschäfts im Sinne des § 1 Abs. 1 Satz 2 Nr. 1 KWG entgegen ( vgl. BT- Drucks. 15/3641 , S. 36 ; BGH , Urteil vom 16. Mai 2017 - VI ZR 266/16 , NJW 2017 , 2463 ; Urteil vom 10. Februar 2015 - VI ZR 569/13 , NJW-RR 2015 , 675 , 676 ; Schäfer in Boos / Fischer / Schulter-Mattler , KWG , 5. Aufl. , § 1 Rn 46 ; Gehrlein , WM 2017 , 1385 , 1386 ; vgl. zur Rechtsanwendungspraxis der BaFin deren Merkblatt „ Hinweise zum Tatbestand des Einlagengeschäfts “ , Stand März 2014 , NZG 2014 , 379 , 381 ) .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 16. Mai 2017 - VI ZR 266/16 , NJW 2017 , 2463`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 1 Abs. 1 Satz 2 Nr. 1 KWG`(NRM)
- `BT- Drucks. 15/3641 , S. 36`(LIT)
- `BGH , Urteil vom 16. Mai 2017 - VI ZR 266/16 , NJW 2017 , 2463`(RS)
- `Urteil vom 10. Februar 2015 - VI ZR 569/13 , NJW-RR 2015 , 675 , 676`(RS)
- `Schäfer in Boos / Fischer / Schulter-Mattler , KWG , 5. Aufl. , § 1 Rn 46`(LIT)
- `Gehrlein , WM 2017 , 1385 , 1386`(LIT)
- `BaFin`(ORG)
- `Einlagengeschäfts “ , Stand März 2014 , NZG 2014 , 379 , 381`(LIT)

**Example 59** (doc_id: `64931`) (sent_id: `64931`)


Dies entspricht der Rechtsprechung des Bundesarbeitsgerichts , wonach die Antragsbefugnis im arbeitsgerichtlichen Beschlussverfahren - ebenso wie die Prozessführungsbefugnis im Urteilsverfahren - dazu dient , Popularklagen auszuschließen , und gegeben ist , wenn der Antragsteller durch die begehrte Entscheidung in seiner kollektivrechtlichen Rechtsposition betroffen sein kann , was wiederum regelmäßig der Fall ist , wenn er eigene Rechte geltend macht und dies nicht von vornherein als aussichtslos erscheint ( BAG , Beschlüsse vom 18. Januar 2017 - 7 ABR 60/15 - NZA 2017 , 865 Rn. 10 und vom 21. März 2017 - 7 ABR 17/15 - NZA 2017 , 1014 Rn. 9 , jeweils m. w. N. ) .

**False Positives:**

- `BAG , Beschlüsse` — partial — pred is substring of gold: `BAG , Beschlüsse vom 18. Januar 2017 - 7 ABR 60/15 - NZA 2017 , 865 Rn. 10`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesarbeitsgerichts`(ORG)
- `BAG , Beschlüsse vom 18. Januar 2017 - 7 ABR 60/15 - NZA 2017 , 865 Rn. 10`(RS)
- `vom 21. März 2017 - 7 ABR 17/15 - NZA 2017 , 1014 Rn. 9`(RS)

**Example 60** (doc_id: `64970`) (sent_id: `64970`)


Denn nach § 155 Abs. 1 SGG kann der Vorsitzende seine Aufgaben nach den §§ 104 , 106 bis 108 und 120 SGG einem Berufsrichter des Senats übertragen ; dies gilt insbesondere für alle Beweisaufnahmen ( vgl. BSG , Beschluss vom 8. Dezember 1988 , 2 BU 52/88 ) .

**False Positives:**

- `BSG , Beschluss` — partial — pred is substring of gold: `BSG , Beschluss vom 8. Dezember 1988 , 2 BU 52/88`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 155 Abs. 1 SGG`(NRM)
- `§§ 104 , 106 bis 108 und 120 SGG`(NRM)
- `BSG , Beschluss vom 8. Dezember 1988 , 2 BU 52/88`(RS)

**Example 61** (doc_id: `65147`) (sent_id: `65147`)


b ) Der Beschwerdeführer ist auch nicht in seinem Anspruch auf ein faires Verfahren ( Art. 20 Abs. 3 GG , Art. 6 Abs. 1 Satz 1 MRK ) verletzt .

**False Positives:**

- `GG , Art` — positional overlap with gold: `Art. 20 Abs. 3 GG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 20 Abs. 3 GG`(NRM)
- `Art. 6 Abs. 1 Satz 1 MRK`(NRM)

**Example 62** (doc_id: `65219`) (sent_id: `65219`)


b ) Im Hinblick auf die vom Antragsteller ferner gerügte Verletzung seiner Rechte aus Art. 19 Abs. 4 in Verbindung mit Art. 2 Abs. 2 Satz 1 GG , Art. 103 Abs. 1 GG und Art. 20 Abs. 3 GG in Verbindung mit Art. 2 Abs. 1 GG in Verbindung mit Art. 6 EMRK durch die Entscheidung des Verwaltungsgerichts fehlt es an jeder substantiierten Darlegung eines Rechtsverstoßes .

**False Positives:**

- `GG , Art` — positional overlap with gold: `Art. 19 Abs. 4 in Verbindung mit Art. 2 Abs. 2 Satz 1 GG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 19 Abs. 4 in Verbindung mit Art. 2 Abs. 2 Satz 1 GG`(NRM)
- `Art. 103 Abs. 1 GG`(NRM)
- `Art. 20 Abs. 3 GG`(NRM)
- `Art. 2 Abs. 1 GG`(NRM)
- `Art. 6 EMRK`(NRM)

**Example 63** (doc_id: `65302`) (sent_id: `65302`)


c ) An diesen Grundsätzen gemessen ist gegen die Beweiswürdigung der Strafkammer - zumal eingedenk des auch insoweit eingeschränkten revisionsgerichtlichen Prüfungsmaßstabs ( vgl. BGH , Urteil vom 29. September 2016 - 4 StR 320/16 , NStZ-RR 2016 , 380 f. ) - von Rechts wegen nichts zu erinnern .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 29. September 2016 - 4 StR 320/16 , NStZ-RR 2016 , 380 f.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 29. September 2016 - 4 StR 320/16 , NStZ-RR 2016 , 380 f.`(RS)

**Example 64** (doc_id: `65333`) (sent_id: `65333`)


Eine solche aus dem materiellen Recht zu entnehmende Begrenzung , die aufgrund ihrer Wirkung als Rechtswegsperre gegebenenfalls auf ihre Vereinbarkeit mit verfassungsrechtlichen Rechtsschutzgarantien ( vgl. dazu etwa Schmidt-Aßmann , in : Maunz / Dürig , GG , Stand September 2017 , Art. 19 Abs. 4 Rn. 16 ff. m. w. N. ) zu prüfen wäre , liegt hier jedoch nicht vor .

**False Positives:**

- `GG , Stand` — partial — pred is substring of gold: `Schmidt-Aßmann , in : Maunz / Dürig , GG , Stand September 2017 , Art. 19 Abs. 4 Rn. 16 ff.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Schmidt-Aßmann , in : Maunz / Dürig , GG , Stand September 2017 , Art. 19 Abs. 4 Rn. 16 ff.`(LIT)

**Example 65** (doc_id: `65360`) (sent_id: `65360`)


Soweit damit Anschuldigungen gegen eine andere Person verbunden sind , werden die Grenzen eines zulässigen Verteidigungsverhaltens dadurch nicht überschritten ( vgl. BGH , Beschlüsse vom 29. Januar 2013 - 4 StR 532/12 , NStZ-RR 2013 , 170 , 171 ; vom 6. Juli 2010 - 3 StR 219/10 , NStZ 2010 , 692 ; vom 22. März 2007 - 4 StR 60/07 , NStZ 2007 , 463 ) .

**False Positives:**

- `BGH , Beschlüsse` — partial — pred is substring of gold: `BGH , Beschlüsse vom 29. Januar 2013 - 4 StR 532/12 , NStZ-RR 2013 , 170 , 171`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschlüsse vom 29. Januar 2013 - 4 StR 532/12 , NStZ-RR 2013 , 170 , 171`(RS)
- `vom 6. Juli 2010 - 3 StR 219/10 , NStZ 2010 , 692`(RS)
- `vom 22. März 2007 - 4 StR 60/07 , NStZ 2007 , 463`(RS)

**Example 66** (doc_id: `65454`) (sent_id: `65454`)


Seine nicht weiter begründete Annahme , „ irgendwelche abweichenden Abreden , insbesondere sogenannte Nachrangabreden , stellen für den Anleger offensichtlich überraschende und damit unwirksame Klauseln dar “ , hält aber auch eingedenk der nur eingeschränkten revisionsgerichtlichen Kontrolle der tatrichterlichen Auslegung von Verträgen und der ihnen zugrunde liegenden Erklärungen der Vertragsparteien ( vgl. BGH , Urteil vom 13. Mai 2004 - 5 StR 73/03 , NJW 2004 , 2248 , 2250 mwN [ insoweit in BGHSt 49 , 147 nicht abgedruckt ] ; Sander in : Löwe / Rosenberg , StPO , 26. Aufl. , § 261 Rn. 97 ) einer rechtlichen Überprüfung nicht stand , weil sie über erörterungsbedürftige Feststellungen hinweggeht und deshalb lückenhaft ist .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 13. Mai 2004 - 5 StR 73/03 , NJW 2004 , 2248 , 2250 mwN [ insoweit in BGHSt 49 , 147 nicht abgedruckt ]`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 13. Mai 2004 - 5 StR 73/03 , NJW 2004 , 2248 , 2250 mwN [ insoweit in BGHSt 49 , 147 nicht abgedruckt ]`(RS)
- `Sander in : Löwe / Rosenberg , StPO , 26. Aufl. , § 261 Rn. 97`(LIT)

**Example 67** (doc_id: `65540`) (sent_id: `65540`)


Dazu werden die ehemaligen Ansprüche und Anwartschaften nach einem eigenen bundesrechtlichen Maßstab anerkannt , der nur partiell an Gegebenheiten in der DDR anknüpft ( vgl. BSG , Urteil vom 12. Juni 2001 - B 4 RA 117/00 R - SozR 3 - 8570 § 5 AAÜG Nr. 6 und Berchtold , SGb 2018 , 7 ff. ) .

**False Positives:**

- `BSG , Urteil` — partial — pred is substring of gold: `BSG , Urteil vom 12. Juni 2001 - B 4 RA 117/00 R - SozR 3 - 8570 § 5 AAÜG Nr. 6`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `DDR`(LOC)
- `BSG , Urteil vom 12. Juni 2001 - B 4 RA 117/00 R - SozR 3 - 8570 § 5 AAÜG Nr. 6`(RS)
- `Berchtold , SGb 2018 , 7 ff.`(LIT)

**Example 68** (doc_id: `65561`) (sent_id: `65561`)


Denn auch einem Beklagten , der - wie hier - die Streitwertfestsetzungen in den Vorinstanzen weder beanstandet noch sonst glaubhaft gemacht hat , dass bereits in der Vorinstanz für die Festlegung des Streitwerts maßgebliche Umstände , die dort vorgebracht worden sind , nicht ausreichend berücksichtigt worden seien , ist es in aller Regel versagt , sich im Verfahren der Nichtzulassungsbeschwerde noch auf einen höheren , die erforderliche Rechtsmittelbeschwer erstmals erreichenden Wert zu berufen ( BGH , Beschlüsse vom 9. Dezember 2014 - VIII ZR 160/14 , aaO Rn. 7 ; vom 1. Juni 2016 - I ZR 112/15 , aaO ) .

**False Positives:**

- `BGH , Beschlüsse` — partial — pred is substring of gold: `BGH , Beschlüsse vom 9. Dezember 2014 - VIII ZR 160/14 , aaO Rn. 7`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschlüsse vom 9. Dezember 2014 - VIII ZR 160/14 , aaO Rn. 7`(RS)
- `vom 1. Juni 2016 - I ZR 112/15 , aaO`(RS)

**Example 69** (doc_id: `65689`) (sent_id: `65689`)


Die Anordnung des Vorwegvollzugs eines Teils der Freiheitsstrafe vor der Unterbringung in einer Entziehungsanstalt hat allerdings zu unterbleiben , wenn sich - wie hier - der Vorwegvollzug durch die vom Angeklagten seit seiner Festnahme erlittene Polizei- und Untersuchungshaft im Urteilszeitpunkt bereits erledigt hat ( vgl. BGH , Beschluss vom 13. Dezember 2011 - 5 StR 423/11 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 13. Dezember 2011 - 5 StR 423/11`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 13. Dezember 2011 - 5 StR 423/11`(RS)

**Example 70** (doc_id: `65735`) (sent_id: `65735`)


Für die Frage , ob die Ermittlungsbehörden eine richterliche Entscheidung rechtzeitig erreichen können , kommt es auf den Zeitpunkt an , zu dem die Staatsanwaltschaft oder ihre Hilfsbeamten die Durchsuchung für erforderlich hielten ( BGH , Urteil vom 18. April 2007 - 5 StR 546/06 , BGHSt 51 , 285 , 288 f. ) .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 18. April 2007 - 5 StR 546/06 , BGHSt 51 , 285 , 288 f.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 18. April 2007 - 5 StR 546/06 , BGHSt 51 , 285 , 288 f.`(RS)

**Example 71** (doc_id: `65943`) (sent_id: `65943`)


Das technische Problem ist aus dem zu entwickeln , was die Erfindung tatsächlich leistet ( BGH , Urteil vom 04. 02. 2010 – Xa ZR 36/08 – Gelenkanordnung ) .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 04. 02. 2010 – Xa ZR 36/08 – Gelenkanordnung`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 04. 02. 2010 – Xa ZR 36/08 – Gelenkanordnung`(RS)

**Example 72** (doc_id: `66023`) (sent_id: `66023`)


I. 1. Nach bisheriger Rechtsprechung des Bundesgerichtshofs wurde es überwiegend als ein Verstoß gegen das in § 46 Abs. 3 StGB verankerte Verbot der Doppelverwertung von Tatbestandsmerkmalen und damit als rechtsfehlerhaft angesehen , wenn der Tatrichter das subjektive Tatbestandsmerkmal direkten Tötungsvorsatzes strafschärfend berücksichtigt ( vgl. BGH , Beschluss vom 11. März 2015 - 1 StR 3/15 , NStZ-RR 2015 , 171 ( Ls. ) ; Senat , Beschlüsse vom 25. Juni 2015 - 2 StR 83/15 , BGHR StGB § 46 Abs. 3 Tötungsvorsatz 7 , vom 21. Januar 2004 - 2 StR 449/03 , vom 23. Oktober 1992 - 2 StR 483/92 , StV 1993 , 72 und vom 1. Dezember 1989 - 2 StR 555/89 , BGHR StGB § 46 Abs. 3 Tötungsvorsatz 3 ; BGH , Beschlüsse vom 5. Oktober 1977 - 3 StR 369/77 , vom 8. Februar 1978 - 3 StR 425/77 und vom 13. Mai 1981 - 3 StR 126/81 , NJW 1981 , 2204 ; BGH , Urteil vom 28. Juni 1968 - 4 StR 226/68 ; Beschlüsse vom 16. September 1986 - 4 StR 457/86 , BGHR StGB § 46 Abs. 3 Tötungsvorsatz 1 , vom 26. April 1988 - 4 StR 157/88 , NStE Nr. 41 zu § 46 StGB , vom 30. Juli 1998 - 4 StR 346/98 , NStZ 1999 , 23 , vom 3. Februar 2004 - 4 StR 403/03 und vom 14. Oktober 2015 - 5 StR 355/15 , NStZ-RR 2016 , 8 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 11. März 2015 - 1 StR 3/15 , NStZ-RR 2015 , 171 ( Ls. )`
- `BGH , Beschlüsse` — partial — pred is substring of gold: `BGH , Beschlüsse vom 5. Oktober 1977 - 3 StR 369/77 , vom 8. Februar 1978 - 3 StR 425/77 und vom 13. Mai 1981 - 3 StR 126/81 , NJW 1981 , 2204`
- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 28. Juni 1968 - 4 StR 226/68`

> overlaps gold: 3  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesgerichtshofs`(ORG)
- `§ 46 Abs. 3 StGB`(NRM)
- `BGH , Beschluss vom 11. März 2015 - 1 StR 3/15 , NStZ-RR 2015 , 171 ( Ls. )`(RS)
- `Senat , Beschlüsse vom 25. Juni 2015 - 2 StR 83/15 , BGHR StGB § 46 Abs. 3 Tötungsvorsatz 7`(RS)
- `vom 21. Januar 2004 - 2 StR 449/03`(RS)
- `vom 23. Oktober 1992 - 2 StR 483/92 , StV 1993 , 72`(RS)
- `vom 1. Dezember 1989 - 2 StR 555/89 , BGHR StGB § 46 Abs. 3 Tötungsvorsatz 3`(RS)
- `BGH , Beschlüsse vom 5. Oktober 1977 - 3 StR 369/77 , vom 8. Februar 1978 - 3 StR 425/77 und vom 13. Mai 1981 - 3 StR 126/81 , NJW 1981 , 2204`(RS)
- `BGH , Urteil vom 28. Juni 1968 - 4 StR 226/68`(RS)
- `Beschlüsse vom 16. September 1986 - 4 StR 457/86 , BGHR StGB § 46 Abs. 3 Tötungsvorsatz 1`(RS)
- `vom 26. April 1988 - 4 StR 157/88 , NStE Nr. 41 zu § 46 StGB`(RS)
- `vom 30. Juli 1998 - 4 StR 346/98 , NStZ 1999 , 23`(RS)
- `vom 3. Februar 2004 - 4 StR 403/03`(RS)
- `vom 14. Oktober 2015 - 5 StR 355/15 , NStZ-RR 2016 , 8`(RS)

**Example 73** (doc_id: `66092`) (sent_id: `66092`)


Diese Zeitspanne entspreche der gesetzgeberischen Wertung , die in der Dauer der regelmäßigen zivilrechtlichen Verjährungsfrist nach § 195 BGB zum Ausdruck komme ( vgl. BAG , Urteil vom 21. September 2011 - 7 AZR 375/10 - , BAGE 139 , 213 < 225 Rn. 35 > ) .

**False Positives:**

- `BAG , Urteil` — partial — pred is substring of gold: `BAG , Urteil vom 21. September 2011 - 7 AZR 375/10 - , BAGE 139 , 213 < 225 Rn. 35 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 195 BGB`(NRM)
- `BAG , Urteil vom 21. September 2011 - 7 AZR 375/10 - , BAGE 139 , 213 < 225 Rn. 35 >`(RS)

**Example 74** (doc_id: `66099`) (sent_id: `66099`)


Das Landgericht hat die nach Lage der Dinge nicht fern liegende Möglichkeit eines unbeendeten Versuchs ( vgl. BGH , Beschluss vom 20. November 2013 - 3 StR 325/13 , NStZ-RR 2014 , 105 ) zwar hypothetisch erörtert .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 20. November 2013 - 3 StR 325/13 , NStZ-RR 2014 , 105`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 20. November 2013 - 3 StR 325/13 , NStZ-RR 2014 , 105`(RS)

**Example 75** (doc_id: `66188`) (sent_id: `66188`)


Hätte sie sich - wie geboten ( vgl. BGH , Beschluss vom 6. Juli 1994 - VIII ZB 26/94 , NJW 1994 , 2831 mwN ) - die Akten mit einer Vorfrist von etwa einer Woche vor Ablauf der nach ihrer Berechnung am 18. Mai 2017 endenden Berufungsbegründungsfrist und damit am 11. Mai 2017 vorlegen lassen , hätte sie jedenfalls zu diesem Zeitpunkt festgestellt bzw. feststellen müssen , dass ihr eine gerichtliche Verfügung zu der beantragten Fristverlängerung noch nicht zugegangen war .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 6. Juli 1994 - VIII ZB 26/94 , NJW 1994 , 2831`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 6. Juli 1994 - VIII ZB 26/94 , NJW 1994 , 2831`(RS)

**Example 76** (doc_id: `66395`) (sent_id: `66395`)


Allerdings steht es dem Prozessgericht frei , die materiellen Voraussetzungen der Beschränkung mit zu prüfen ( vgl. nur BGH , Urteile vom 9. März 1983 - IVa ZR 211/81 , NJW 1983 , 2378 , 2379 ; vom 13. Juli 1989 - IX ZR 227/87 , NJW-RR 1989 , 1226 , 1230 und vom 2. Februar 2010 - VI ZR 82/09 , NJW-RR 2010 , 664 Rn. 7 f ) und zum Beispiel die Verurteilung auf Leistung aus dem Nachlass zu beschränken ( vgl. nur BayObLGZ 1999 , 323 , 328 f ; siehe auch Zöller / Geimer , ZPO , 32. Aufl. , § 780 Rn. 15 ; MüKoZPO / Schmidt / Brinkmann , 5. Aufl. , § 780 Rn. 10 , 13 ) .

**False Positives:**

- `BGH , Urteile` — partial — pred is substring of gold: `BGH , Urteile vom 9. März 1983 - IVa ZR 211/81 , NJW 1983 , 2378 , 2379`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteile vom 9. März 1983 - IVa ZR 211/81 , NJW 1983 , 2378 , 2379`(RS)
- `vom 13. Juli 1989 - IX ZR 227/87 , NJW-RR 1989 , 1226 , 1230`(RS)
- `vom 2. Februar 2010 - VI ZR 82/09 , NJW-RR 2010 , 664 Rn. 7 f`(RS)
- `BayObLGZ 1999 , 323 , 328 f`(LIT)
- `Zöller / Geimer , ZPO , 32. Aufl. , § 780 Rn. 15`(LIT)
- `MüKoZPO / Schmidt / Brinkmann , 5. Aufl. , § 780 Rn. 10 , 13`(LIT)

**Example 77** (doc_id: `66428`) (sent_id: `66428`)


Er rügt eine Verletzung von Art. 97 Abs. 1 GG in Verbindung mit Art. 33 Abs. 5 GG , Art. 103 Abs. 1 GG und Art. 3 Abs. 1 GG .

**False Positives:**

- `GG , Art` — positional overlap with gold: `Art. 33 Abs. 5 GG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 97 Abs. 1 GG`(NRM)
- `Art. 33 Abs. 5 GG`(NRM)
- `Art. 103 Abs. 1 GG`(NRM)
- `Art. 3 Abs. 1 GG`(NRM)

**Example 78** (doc_id: `66523`) (sent_id: `66523`)


Macht der Tatrichter von der Möglichkeit einer solchen Verweisung , durch welche das Lichtbild selbst Bestandteil der Urteilsgründe wird , keinen Gebrauch , muss das Urteil Ausführungen zur Bildqualität , insbesondere zur Bildschärfe , enthalten und die abgebildete Person oder jedenfalls mehrere Identifizierungsmerkmale so präzise beschreiben , dass anhand der Beschreibung in gleicher Weise wie bei Betrachtung des Fotos die Prüfung der Ergiebigkeit des Fotos ermöglicht wird ( vgl. BGH , Beschluss vom 19. Dezember 1995 - 4 StR 170/95 , BGHSt 41 , 376 , 382 ff. ; Sander in Löwe / Rosenberg , StPO , 26. Aufl. , § 261 Rn. 101 mwN ; Seitz / Bauer in Göhler , OWiG , 17. Aufl. , § 71 Rn. 47a ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 19. Dezember 1995 - 4 StR 170/95 , BGHSt 41 , 376 , 382 ff.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 19. Dezember 1995 - 4 StR 170/95 , BGHSt 41 , 376 , 382 ff.`(RS)
- `Sander in Löwe / Rosenberg , StPO , 26. Aufl. , § 261 Rn. 101`(LIT)
- `Seitz / Bauer in Göhler , OWiG , 17. Aufl. , § 71 Rn. 47a`(LIT)

**Example 79** (doc_id: `66619`) (sent_id: `66619`)


Die Überlastung eines Spruchkörpers i. S. v. § 21e Abs. 3 Satz 1 GVG , die eine Änderung der Geschäftsverteilung nötig macht , liegt vor , wenn über einen längeren Zeitraum ein erheblicher Überhang der Eingänge über die Erledigungen zu verzeichnen ist , sodass mit einer Bearbeitung der Sachen innerhalb eines angemessenen Zeitraums nicht zu rechnen ist und sich die Überlastung als so erheblich darstellt , dass der Ausgleich nicht bis zum Ende des Geschäftsjahres zurückgestellt werden kann ( BGH , Beschluss vom 4. August 2009 - 3 StR 174/09 - juris Rn. 16 ; Kissel / Mayer , GVG , 8. Aufl. 2015 , § 21e Rn. 112 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 4. August 2009 - 3 StR 174/09 - juris Rn. 16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 21e Abs. 3 Satz 1 GVG`(NRM)
- `BGH , Beschluss vom 4. August 2009 - 3 StR 174/09 - juris Rn. 16`(RS)
- `Kissel / Mayer , GVG , 8. Aufl. 2015 , § 21e Rn. 112`(LIT)

**Example 80** (doc_id: `66669`) (sent_id: `66669`)


Die Überprüfung , die erforderlich ist , damit eine lebenslange Freiheitsstrafe reduzierbar ist , soll den innerstaatlichen Behörden erlauben zu erwägen , ob eine Änderung des Gefangenen und ein Fortschritt in Richtung seiner Resozialisierung von solcher Bedeutung sind , dass die weitere Inhaftierung nicht länger durch legitime Strafgründe gerechtfertigt ist ( EGMR , Urteil vom 4. September 2014 - Nr. 140/10 , Trabelsi / Belgien - Rn. 115 ; EGMR < GK > , Urteil vom 26. April 2016 - Nr. 10511/10 , Murray / Niederlande - Rn. 100 ) .

**False Positives:**

- `EGMR , Urteil` — partial — pred is substring of gold: `EGMR , Urteil vom 4. September 2014 - Nr. 140/10 , Trabelsi / Belgien - Rn. 115`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EGMR , Urteil vom 4. September 2014 - Nr. 140/10 , Trabelsi / Belgien - Rn. 115`(RS)
- `EGMR < GK > , Urteil vom 26. April 2016 - Nr. 10511/10 , Murray / Niederlande - Rn. 100`(RS)

**Example 81** (doc_id: `66673`) (sent_id: `66673`)


Danach darf bei durch § 100a StPO gerechtfertigter Aufzeichnung eines Telefongesprächs das gesamte während des Telefonats aufgezeichnete Gespräch einschließlich der Hintergrundgeräusche und -gepräche verwertet werden ( BGH , Beschluss vom 24. April 2008 - 1 StR 169/08 , NStZ 2008 , 473 f. mwN ; siehe auch KK-StPO / Bruns , 7. Aufl. , § 100a Rn. 54 ) .

**False Positives:**

- `BGH , Beschluss` — partial — pred is substring of gold: `BGH , Beschluss vom 24. April 2008 - 1 StR 169/08 , NStZ 2008 , 473 f.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 100a StPO`(NRM)
- `BGH , Beschluss vom 24. April 2008 - 1 StR 169/08 , NStZ 2008 , 473 f.`(RS)
- `KK-StPO / Bruns , 7. Aufl. , § 100a Rn. 54`(LIT)

**Example 82** (doc_id: `66713`) (sent_id: `66713`)


Erforderlich ist , dass er die Tatsachen kennt , die dem normativen Begriff zugrunde liegen , und auf der Grundlage dieses Wissens den sozialen Sinngehalt des Tatbestandsmerkmals richtig begreift ( vgl. BGH , Urteil vom 3. April 2008 - 3 StR 394/07 , BGHR StGB § 17 Vermeidbarkeit 8 ; Urteil vom 24. September 1953 - 5 StR 225/53 , BGHSt 4 , 347 , 352 ; Urteil vom 16. Mai 2017 - VI ZR 266/16 , NJW 2017 , 2463 , 2464 ; Urteil vom 15. Mai 2012 - VI ZR 166/11 , NJW 2012 , 3177 , 3179 f. mwN ; Janssen in : Münch. Komm. z. StGB , 2. Aufl. , § 54 KWG Rn. 83 ; Papathanasiou , jurisPR-StrafR 25/2017 Anm. 4 unter C ) .

**False Positives:**

- `BGH , Urteil` — partial — pred is substring of gold: `BGH , Urteil vom 3. April 2008 - 3 StR 394/07 , BGHR StGB § 17 Vermeidbarkeit 8`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 3. April 2008 - 3 StR 394/07 , BGHR StGB § 17 Vermeidbarkeit 8`(RS)
- `Urteil vom 24. September 1953 - 5 StR 225/53 , BGHSt 4 , 347 , 352`(RS)
- `Urteil vom 16. Mai 2017 - VI ZR 266/16 , NJW 2017 , 2463 , 2464`(RS)
- `Urteil vom 15. Mai 2012 - VI ZR 166/11 , NJW 2012 , 3177 , 3179 f.`(RS)
- `Janssen in : Münch. Komm. z. StGB , 2. Aufl. , § 54 KWG Rn. 83`(LIT)
- `Papathanasiou , jurisPR-StrafR 25/2017 Anm. 4 unter C`(LIT)

</details>

---

## `Names after legal roles (Herr, Richter, etc.)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `64c0e08f`  
**Description:**
Captures names following legal role indicators like 'Herr', 'Herrn', 'Richter', 'Richterin', 'Angeklagte', 'Angeklagten', 'Kl\u00e4ger', 'Zeuge', ensuring no trailing space and handling titles correctly.

**Content:**
```
(?:Herr\s+|Herrn\s+|Richter\s+|Richterin\s+|Vorsitzender\s+Richter\s+|Angeklagte\s+|Angeklagten\s+|Kl\u00e4ger\s+|Zeuge\s+|Zeugin\s+)([A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+[A-Z]\s*\.)?(?:\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)*)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 7 | 0 | 7 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 7 | 287 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60944`) (sent_id: `60944`)


Nachdem er vom FA auf den fehlenden Nachweis einer regelmäßigen Summenziehung hingewiesen worden sei , habe der Kläger Erfassungsprotokolle beim FG eingereicht , die eine chronologische Auflistung der Geschäftsvorfälle ohne Angabe von Belegnummern enthalten hätten .

**False Positives:**

- `Erfassungsprotokolle` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `61328`) (sent_id: `61328`)


Am 18. November 2015 hat der Kläger Klage erhoben und zur Begründung geltend gemacht , der Jahresmittelgrenzwert für Stickstoffdioxid ( NO2 ) sei im Jahr 2013 an allen verkehrsnahen Messstationen zum Teil um mehr als das Doppelte überschritten worden und habe auch im Jahr 2014 an bestimmten Messstationen deutlich über den Grenzwerten gelegen .

**False Positives:**

- `Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `62721`) (sent_id: `62721`)


Seit 16. 2. 2013 ist der Kläger Pflichtmitglied der Landestierärztekammer Baden-Württemberg ( im Folgenden : Landestierärztekammer ) und Pflichtmitglied der Baden-Württembergischen Versorgungsanstalt für Ärzte , Zahnärzte und Tierärzte ( Beigeladene zu 1. ) , die ihren Teilnehmern und deren Hinterbliebenen Altersruhegeld , Ruhegeld bei Berufsunfähigkeit sowie eine Hinterbliebenenversorgung gewährt .

**False Positives:**

- `Pflichtmitglied` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Landestierärztekammer Baden-Württemberg`(ORG)
- `Baden-Württembergischen Versorgungsanstalt für Ärzte , Zahnärzte und Tierärzte`(ORG)

**Example 3** (doc_id: `63009`) (sent_id: `63009`)


Hiergegen hat der Kläger Klage zum SG erhoben , das durch Urteil vom 2. 10. 2012 den Bescheid der Beklagten vom 18. 4. 2011 in der Gestalt des Widerspruchsbescheids vom 8. 6. 2011 aufgehoben hat , weil das Grundstück des Klägers aufgrund der anzuwendenden Ausnahmevorschrift des § 123 Abs 2 SGB VII als versicherungsfreier Haus- und Ziergarten einzuordnen sei .

**False Positives:**

- `Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 123 Abs 2 SGB VII`(NRM)

**Example 4** (doc_id: `63492`) (sent_id: `63492`)


Mit der Revision rügen die Kläger Verletzung formellen und materiellen Rechts .

**False Positives:**

- `Verletzung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `66649`) (sent_id: `66649`)


M. hielt der Zeugin Mund und Nase zu , so dass sie nicht mehr schreien konnte .

**False Positives:**

- `Mund` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `M.`(PER)

**Example 6** (doc_id: `66655`) (sent_id: `66655`)


Am 18. November 2015 hat der Kläger Klage erhoben und zur Begründung geltend gemacht , die anhaltende Überschreitung der Grenzwerte sei ein Indiz dafür , dass die bisherigen Maßnahmen nicht geeignet seien , die Überschreitungszeiträume so kurz wie möglich zu halten .

**False Positives:**

- `Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Initials with surname (contextual)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `f64bfcbd`  
**Description:**
Captures an initial followed by a capitalized surname (e.g., 'K. Schmidt'), ensuring it's a name and not a sentence start or common verb.

**Content:**
```
(?:^|\s|[,;])([A-Z]\.)\s+([A-Z][a-zäöüß]+)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 4 | 0 | 4 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 4 | 253 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `61627`) (sent_id: `61627`)


D15 P. W. McMillan et al. , “ The Structure and Properties of a Lithium Zinc Silicate Glass-Ceramic ” , Journal of Materials Science , 1966 , 1 , Seiten 269 bis 279

**False Positives:**

- `W. Mc` — partial — pred is substring of gold: `P. W. McMillan`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `P. W. McMillan`(PER)

**Example 1** (doc_id: `62472`) (sent_id: `62472`)


Er schloss mit der zu 1. beigeladenen B. GmbH ( jetzt : B. GmbH ) als Arbeitgeberin einen Darstellervertrag für die Fernsehproduktion " Rosenheim-Cops " .

**False Positives:**

- `B. Gmb` — partial — pred is substring of gold: `B. GmbH`
- `B. Gmb` — similar text (different position): `B. GmbH`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `B. GmbH`(ORG)
- `B. GmbH`(ORG)

**Example 2** (doc_id: `66706`) (sent_id: `66706`)


In den unter IV. der Urteilsgründe zusammengefassten zwei Fällen ( Beiseiteschaffen von Fahrzeugen und Gerätschaften aus dem Vermögen der M. GmbH und aus dem Vermögen des nicht revidierenden Mitangeklagten Ma. ) hat das Landgericht den Angeklagten jeweils wegen Beihilfe zum Bankrott verurteilt und den Strafrahmen des § 283 Abs. 1 StGB jeweils gemäß § 27 Abs. 2 , § 49 Abs. 1 StGB gemildert .

**False Positives:**

- `M. Gmb` — partial — pred is substring of gold: `M. GmbH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `M. GmbH`(ORG)
- `Ma.`(PER)
- `§ 283 Abs. 1 StGB`(NRM)
- `§ 27 Abs. 2 , § 49 Abs. 1 StGB`(NRM)

</details>

---

## `Anonymized names with comma and initial`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `001ef128`  
**Description:**
Captures anonymized names in the format 'Surname , Initial.' (e.g., 'Boolell , M.', 'Rosen , R. C.').

**Content:**
```
\b([A-Z][a-zäöüß]+\s*,\s*[A-Z]\s*\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Multi-initial names with surname`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `23f66d88`  
**Description:**
Captures names with multiple initials followed by a surname (e.g., 'P. W. McMillan', 'Tomlinson , J. M.').

**Content:**
```
\b([A-Z]\s*\.\s*[A-Z]\s*\.\s*[A-Z][a-zäöüß]+|[A-Z][a-zäöüß]+\s*,\s*[A-Z]\s*\.\s*[A-Z]\s*\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Anonymized initials with dots (corrected)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `5ba69691`  
**Description:**
Captures anonymized initials consisting of a single uppercase letter followed by a dot, ensuring it's not part of a larger multi-initial sequence already captured.

**Content:**
```
(?<![A-Za-zäöüß\.\s])([A-Z])\.(?![A-Za-zäöüß\.\s])
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


# Rule Evaluation Report — Qwen/Qwen3.5-35B-A3B

Generated on: 2026-06-18T20:14:40.327066

---

<details>
<summary>Configuration</summary>

Results can be reproduced by running this command: 
```
 python benchmark.py --config reports/germanler/Qwen_Qwen3.5-35B-A3B/PER/2026-06-18_v5/config.yaml 
```
| Parameter | Value |
|---|---|
| Pool size | None |
| Train ratio | 0.80 |
| Validation ratio | 0.20 |
| Shots per class | None |
| Training documents | 1838 |
| Validation documents | 460 |
| Test documents | 6666 |
| Train sentences | 1838 |
| Validation sentences | 460 |
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

**Transfer Learning**

| Property | Value |
|---|---|
| Best Batch Idx | 10 |
| Best Batch F1 | 0.18181818181818182 |
| Best Rules Serialized | [{'id': '70340c5f', 'name': 'Anonymized Initials with Dots', 'description': "Captures anonymized names consisting of a capital letter followed by a dot (e.g., 'M.', 'F.', 'K.'), often preceded by role indicators.", 'format': 'regex', 'content': '\\b(?:Angeklagte|Angeklagten|Kläger|Klägerin|Beklagte|Beklagten|Zeuge|Zeugin|Vorsitzender|Vorsitzende|Richter|Richterin|Geschädigter|Geschädigte|Ministerpräsident|Ministerpräsidentin|Herr|Frau|Dr\\.?|Prof\\.?|Patentanwalt|Rechtsanwalt|Sachverständiger|Sachverständige)\\s+([A-Z]\\.)', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-18T19:51:12.675504', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'f8edee3f', 'name': 'Known Surnames', 'description': 'Captures specific verified surnames from training data, excluding single letters and common words.', 'format': 'regex', 'content': '\\b(?:Franke|Haberkamp|Spelge|Schlünder|Hoch|Roloff|Sieberts|Grube|Suckow|Schneider|Roggenbuck|Sibylle|Spoo|Appl|Bredendiek|Bacher|Wemheuer|Mutzbauer|Koch|Pape|Winzenried|Pielenz|Raum|Brühler|Bender|Lohmann|Remmert|Kley|Schultz|Kleinschmidt|Kirschneck|Matter|Kapels|Reinfelder|Brown|Schäfer|Brückner|Volz|Knoll|Kriener|Nielsen|Mayen|Seiters|Busch|Linck|Leitz|Hamdorf|Fiamingo|Spaniol|Kirchhoff|Gericke|Fritz|Vogelsang|Zwanziger|Kelvin|Lauer|Zeng|Tiemann|Sander|Fischermeier|Çerikci|Kaya|Seyhan|Lorsbach|Maksymiw|Schell|Münzberg|Jäger|Peter|Eschelbach|Kortbein|Schmid|Söchtig|Hacker|Merzbach|Meiser|Ts|Arnoldi|Haupt|Niemann|Becker|Waskow|Eylert|Marx|Fischer|Stresemann|Heinkel|Hayen|Volk|Liebert|Matthias|Kayser|Klein|Maekawa|Bar|Refaeli|Josh|Duhamel|Saime|Özcan|Boolell|McMillan|Rennpferdt|Wollny|Albertshofer|Dorn|Musiol|Kuemmerle|Drüen|Klaus-Dieter|Quentin|Spinner|Schlewing|Schmidt|Limperg|Merkel|Bormann|Berg|Demir|Baykara|Shah|St|Bu|W|G|F|E|A|S|N|D|K|R|V|Y|I|X|H|C|L|M|O|Z|B1|G1|J1|S1|K1|N1|P1|T1|V1|Y1|A1|D1|E1|F1|G1|H1|I1|J1|K1|L1|M1|N1|O1|P1|Q1|R1|S1|T1|U1|V1|W1|X1|Y1|Z1)\\b(?!\\s*(?:von|der|die|das|ein|eine|mit|für|auf|in|an|bei|zu|nach|vor|über|unter|durch|ohne|gegen|seit|statt|wegen|um|bis|neben|zwischen|entlang|außer|neben|ab|Gericht|Firma|Behörde|Polizei|Steuern|Umsatz|Kabel|Anlage|Umsatzsteuer|Rechtsanwältin|Rechtsanwalt|Ackerstatusrechten|Höchst-Schmerzgrenze|Marken-Beschwerdesenat|Rechtsanwalt|...|Frau|Herr|Dr\\.?|Prof\\.?|Anwalt|Gericht|Firma|Behörde|Polizei|Steuern|Umsatz|Kabel|Anlage|Umsatzsteuer|Rechtsanwältin|Rechtsanwalt|Ackerstatusrechten|Höchst-Schmerzgrenze|Marken-Beschwerdesenat|Rechtsanwalt|...))', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-18T19:51:12.672853', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '24426646', 'name': 'Names after Legal Roles', 'description': "Captures names following legal role indicators like 'Angeklagten', 'Kläger', 'Zeugen', 'Richter', 'Geschädigten', 'Vorsitzenden', etc., ensuring titles are not included in the capture.", 'format': 'regex', 'content': '\\b(?:Angeklagten|Kläger|Zeugen|Richter|Geschädigten|Vorsitzenden|Ministerpräsidenten|Dr\\.\\s+|Prof\\.\\s+|Herrn|Frau|Dipl\\.-Ing\\.\\s+|Dipl\\.-Psych\\.\\s+|Rechtsanwalt\\s+|Rechtsanwältin\\s+)([A-Z][a-zäöüß]+(?:\\s+[A-Z][a-zäöüß]+)*)', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-18T19:51:12.676945', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'b224d30a', 'name': 'Anonymized Initials with Ellipsis and Numbers', 'description': "Captures anonymized names with an ellipsis (e.g., 'P …', 'B1 …', 'G1 …') following role indicators or in legal contexts.", 'format': 'regex', 'content': '\\b([A-Z]\\d*\\s+…)', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-18T19:48:31.865979', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'd8741160', 'name': 'Full Names with Titles', 'description': 'Captures names preceded by titles like Dr., Prof., Dipl.-Ing., etc., ensuring the title is not captured and the name is a valid surname.', 'format': 'regex', 'content': '(?:Dr\\.?\\s+|Prof\\.?\\s+|Dipl\\.-Ing\\.\\s+|Dipl\\.-Psych\\.\\s+|Richter\\s+|Anwalt\\s+|Rechtsanwältin\\s+|Rechtsanwalt\\s+)([A-Z][a-zäöüß]+(?:\\s+[A-Z][a-zäöüß]+)*)', 'priority': 10, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-18T19:51:12.672822', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '030b825e', 'name': "Names after 'von' (Noble/Preposition)", 'description': "Captures names following the preposition 'von', excluding common non-name words and titles.", 'format': 'regex', 'content': '\\bvon\\s+([A-Z][a-zäöüß]+)(?!\\s+(?:Frau|Herr|Dr\\.?|Prof\\.?|Anwalt|Gericht|Firma|Behörde|Polizei|Steuern|Umsatz|Kabel|Anlage|Umsatzsteuer|Rechtsanwältin|Rechtsanwalt|Ackerstatusrechten|Höchst-Schmerzgrenze|Marken-Beschwerdesenat|Rechtsanwalt|...|von|der|die|das|ein|eine|mit|für|auf|in|an|bei|zu|nach|vor|über|unter|durch|ohne|gegen|seit|statt|wegen|um|bis|neben|zwischen|entlang|außer|neben|ab))', 'priority': 7, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-18T19:51:12.676927', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'acae7cb4', 'name': 'Hyphenated Surnames', 'description': "Captures hyphenated surnames (e.g., 'Sost-Scheible', 'Meier-Beck') only when preceded by a title or legal role.", 'format': 'regex', 'content': '(?:Dr\\.?\\s+|Prof\\.?\\s+|Dipl\\.-Ing\\.\\s+|Dipl\\.-Psych\\.\\s+|Richter\\s+|Anwalt\\s+|Rechtsanwältin\\s+|Rechtsanwalt\\s+|Angeklagten|Kläger|Zeugen|Richter|Geschädigten|Vorsitzenden|Ministerpräsidenten|Herrn|Frau)\\s+([A-Z][a-zäöüß]+-[A-Z][a-zäöüß]+)', 'priority': 8, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-18T19:51:12.676441', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'c2457fe9', 'name': 'Single Letter Anonymized Names', 'description': "Captures single capital letters used as anonymized names in legal contexts (e.g., 'N', 'C', 'V', 'B').", 'format': 'regex', 'content': '\\b([A-Z])\\b(?!\\s*(?:von|der|die|das|ein|eine|mit|für|auf|in|an|bei|zu|nach|vor|über|unter|durch|ohne|gegen|seit|statt|wegen|um|bis|neben|zwischen|entlang|außer|neben|ab|Gericht|Firma|Behörde|Polizei|Steuern|Umsatz|Kabel|Anlage|Umsatzsteuer|Rechtsanwältin|Rechtsanwalt|Ackerstatusrechten|Höchst-Schmerzgrenze|Marken-Beschwerdesenat|Rechtsanwalt|...|Frau|Herr|Dr\\.?|Prof\\.?|Anwalt|Gericht|Firma|Behörde|Polizei|Steuern|Umsatz|Kabel|Anlage|Umsatzsteuer|Rechtsanwältin|Rechtsanwalt|Ackerstatusrechten|Höchst-Schmerzgrenze|Marken-Beschwerdesenat|Rechtsanwalt|...))', 'priority': 8, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-18T19:51:12.675526', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '8f967a4c', 'name': 'Multi-Initial Anonymized Names', 'description': "Captures anonymized names with multiple initials (e.g., 'M. D.', 'A. A.', 'P. W.').", 'format': 'regex', 'content': '\\b([A-Z]\\.[\\s]+[A-Z]\\.)\\b', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-18T19:51:12.676328', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}] |

<details>
<summary>Results</summary>

| Metric | Value |
|---|---|
| Accuracy (exact match) | 87.8% |
| True Positives | 118 |
| False Positives | 859 |
| False Negatives | 230 |
| Total Gold Entities | 348 |
| Micro Precision | 12.1% |
| Micro Recall | 33.9% |
| Micro F1 | 17.8% |
| Macro F1 | 17.8% |

</details>

---

<details>
<summary>📊 Summary</summary>

| Rule | F1 | Precision | Recall | Total Predicted | True Positives | False Positives |
|---|---|---|---|---|---|---|
| `Anonymized Initials with Dots` | 17.3% | 97.1% | 9.5% | 34 | 33 | 1 |
| `Standalone Surname` | 25.9% | 85.5% | 15.2% | 62 | 53 | 9 |
| `Full Names with Titles` | 9.1% | 68.0% | 4.9% | 25 | 17 | 8 |
| `Anonymized Initials with Ellipsis and Numbers` | 3.2% | 23.1% | 1.7% | 26 | 6 | 20 |
| `Initials with Surname` | 3.9% | 8.1% | 2.6% | 111 | 9 | 102 |
| `Names after 'von' (Noble/Preposition)` | 0.0% | 0.0% | 0.0% | 702 | 0 | 702 |
| `Hyphenated Surnames` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Single Letter Anonymized Names` | 0.0% | 0.0% | 0.0% | 17 | 0 | 17 |
| `Multi-Initial Anonymized Names` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |

</details>

---

<details>
<summary>🏆 Most Precise Rules</summary>

## `Anonymized Initials with Dots`

**F1:** 0.173 | **Precision:** 0.971 | **Recall:** 0.095  

**Format:** `regex`  
**Rule ID:** `70340c5f`  
**Description:**
Captures anonymized names consisting of a capital letter followed by a dot (e.g., 'M.', 'F.', 'K.'), often preceded by role indicators.

**Content:**
```
\b(?:Angeklagte|Angeklagten|Kläger|Klägerin|Beklagte|Beklagten|Zeuge|Zeugin|Vorsitzender|Vorsitzende|Richter|Richterin|Geschädigter|Geschädigte|Ministerpräsident|Ministerpräsidentin|Herr|Frau|Dr\.?|Prof\.?|Patentanwalt|Rechtsanwalt|Sachverständiger|Sachverständige)\s+([A-Z]\.)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.971 | 0.095 | 0.173 | 34 | 33 | 1 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 33 | 1 | 310 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53522`) (sent_id: `53522`)


Den Fällen komme gleichwohl eigenständige Bedeutung zu , weil sich für das gleichfalls - jeweils in nicht geringer Menge - gehandelte bzw. zum Handeltreiben vorgesehene Marihuana ein einheitlicher Erwerbsvorgang nicht feststellen lasse , so dass insoweit jeweils eine eigenständige Strafbarkeit des Angeklagten H. nach § 29a Abs. 1 Nr. 2 BtMG begründet werde .

| Predicted | Gold |
|---|---|
| `H.` | `H.` |

**Missed by this rule (FN):**

- `§ 29a Abs. 1 Nr. 2 BtMG` (NRM)

**Example 1** (doc_id: `53890`) (sent_id: `53890`)


Im Berufungsverfahren hat der zuständige Berichterstatter des LSG den Sachverständigen Prof. Dr. T. im Termin zur mündlichen Verhandlung ergänzend angehört .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |

**Example 2** (doc_id: `54671`) (sent_id: `54671`)


Sie hat für Dr. T. und für Dr. L. jeweils RLV berechnet und deren Summe der Klägerin als RLV zugewiesen .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |
| `L.` | `L.` |

**Example 3** (doc_id: `54771`) (sent_id: `54771`)


Der Angeklagte A. entfernte sich - ebenso wie seine Tatgenossen - von der Unfallstelle , ohne zuvor dem Zeugen K. gegenüber Angaben zu seiner Person und der Art der Unfallbeteiligung gemacht zu haben .

| Predicted | Gold |
|---|---|
| `A.` | `A.` |

**Missed by this rule (FN):**

- `K.` (PER)

**Example 4** (doc_id: `54995`) (sent_id: `54995`)


Das LSG hat vielmehr im Anschluss an die Begründung , warum es dessen sachverständige Bewertung für überzeugend hält , ausgeführt : " Hingegen hat Dr. H. lediglich auf einer Seite kurz dargestellt , dass beim Kläger seinerzeit kein KIG Grad 3 oder höher vorliege .

| Predicted | Gold |
|---|---|
| `H.` | `H.` |

**Example 5** (doc_id: `55088`) (sent_id: `55088`)


Zwar hat der Senat durchaus zur Kenntnis genommen , dass die Klägerin in dem von ihr benannten Schriftsatz vom 11. 2. 2016 - auf die Anhörung des LSG zur Entscheidung durch Beschluss - durch die Bezugnahme auf den Schriftsatz vom 7. 3. 2016 ihren Antrag bei Prof. Dr. T. nachzufragen , in welcher Form und in welchem Umfang er an der Erstellung des Sachverständigengutachtens beteiligt gewesen sei , wiederholt hat .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |

**Example 6** (doc_id: `55218`) (sent_id: `55218`)


Mit Schreiben vom 8. 7. 2009 ( eingegangen am 21. 10. 2009 ) stellte Dr. T. für die Quartale ab I / 2009 einen Antrag auf Erhöhung der Fallzahl - Anpassung des RLV wegen Jungpraxis - und beantragte die Übernahme der Fallzahl der von ihm vor seiner Anstellung im MVZ betriebenen Praxis .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |

**Example 7** (doc_id: `56131`) (sent_id: `56131`)


Der Zeuge K. hat diesen Tatentschluss umgesetzt und mit der erfolgreichen Anwerbung des Zeugen S. , der sich zum Verkauf von Cannabis des Angeklagten auf Kommissionsbasis bereiterklärte , den Handel des Angeklagten mit Betäubungsmitteln gefördert .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `S.` (PER)

**Example 8** (doc_id: `56284`) (sent_id: `56284`)


3. Der Senat setzt den Wert des Gegenstands der anwaltlichen Tätigkeit des Antragstellers zur Verteidigung des Angeklagten K. gegen die beantragte Feststellung nach § 111i Abs. 2 StPO aF antragsgemäß auf 2.006.713,43 € fest .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `§ 111i Abs. 2 StPO aF` (NRM)

**Example 9** (doc_id: `56427`) (sent_id: `56427`)


3 ) Die Einwände des Klägers gegen die Feststellung des LSG , dass alle tatsächlichen Voraussetzungen für die Eintragung der Elektronicon-GmbH gegeben waren , insbesondere dass die Eintragung auf einer richterlichen Verfügung beruhte und im Original von der Zeugin S. unterschrieben worden war , greifen nicht durch .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Missed by this rule (FN):**

- `Elektronicon-GmbH` (ORG)

**Example 10** (doc_id: `56444`) (sent_id: `56444`)


So wird ausdrücklich darauf verwiesen , dass auf der Anlage K 1. Ausdruck einer Mail von D. an D. B. vom 30. Januar 2011 ) handschriftlich " Rechnungsanschrift Help Food z. o. o. D. B. ( es folgt die postalische Anschrift der Help Food ) " vermerkt ist , auf Seite 2 der Anlage K 3 ( Präsentationsunterlage mit dem Copyright von D. und W. ) unter " Unsere Kontraktbedingungen " ein " Exklusiver Kontrakt für 2 Jahre mit Help Food " und eine " Haushaltsverfügung durch Help Food ... bis zum Ende 2011 Startphase " erwähnt werden , auf Seite 2 der Anlage K 5 ( mit dem Logo der Klägerin versehenes Protokoll eines Treffens der Beteiligten am 26. August 2011 ) von einem " Vorschlag zum Vertrag zwischen Help Food , M. D. und P. W. " die Rede ist , die Anlage K 9 ( von S. unterzeichnetes Schreiben vom 29. Dezember 2011 ) als Absender die Help Food ausweist und die Anlage K 50 ( Ausdruck einer Mail der Zeugin F. an D. und W. vom 14. September 2011 ) die Absenderadresse " m. @helpfood . eu " trägt .

| Predicted | Gold |
|---|---|
| `F.` | `F.` |

**Missed by this rule (FN):**

- `D.` (PER)
- `D. B.` (PER)
- `Help Food` (ORG)
- `Help Food` (ORG)
- `D.` (PER)
- `W.` (PER)
- `Help Food` (ORG)
- `Help Food` (ORG)
- `Help Food` (ORG)
- `M. D.` (PER)
- `P. W.` (PER)
- `S.` (PER)
- `Help Food` (ORG)
- `D.` (PER)
- `W.` (PER)

**Example 11** (doc_id: `56534`) (sent_id: `56534`)


Den Migrationshintergrund habe Dr. S. in seinem Gutachten nicht berücksichtigt , was sie an einzelnen Passagen des Gutachtens belegt .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Example 12** (doc_id: `56774`) (sent_id: `56774`)


Der Angeklagte O. räumte sodann über eine Erklärung seines Verteidigers die Tat in objektiver und subjektiver Hinsicht ein und bestätigte persönlich die Erklärung seines Verteidigers .

| Predicted | Gold |
|---|---|
| `O.` | `O.` |

**Example 13** (doc_id: `57122`) (sent_id: `57122`)


Einen Verfügungssatz , der die nachfolgend genannten " Vergleichspunktzahlvolumen , bei dessen Überschreitung eine Honorarkürzung zulässig ist " , allein auf das " Job-Sharing-Pärchen " Dr. R. und Dr. E. beziehen würde , enthält der Bescheid nicht .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |
| `E.` | `E.` |

**Example 14** (doc_id: `57131`) (sent_id: `57131`)


Der geringe Erfolg des Rechtsmittels des Angeklagten I. lässt es nicht unbillig erscheinen , ihn mit den gesamten Kosten seines Rechtsmittels zu belasten .

| Predicted | Gold |
|---|---|
| `I.` | `I.` |

**Example 15** (doc_id: `57355`) (sent_id: `57355`)


Die Taten fanden ein Ende , nachdem die Zeugin R. im Sexualkundeunterricht aufgeklärt worden war .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |

**Example 16** (doc_id: `57413`) (sent_id: `57413`)


Vielmehr bedinge das seelische Leiden , das ursächlich für die vom Kläger beschriebenen Konzentrations- und Orientierungsstörungen sei , nach den überzeugenden gutachterlichen Ausführungen des nervenärztlichen Sachverständigen G. vom 24. 7. 2017 lediglich einen Einzel-GdB von 40. Auch der Befundbericht des behandelnden Facharztes für Neurologie und Psychiatrie Dr. H. vom 12. 5. 2016 und der Entlassungsbericht der M. -Klinik vom 29. 8. 2013 rechtfertigten keine andere Beurteilung .

| Predicted | Gold |
|---|---|
| `H.` | `H.` |

**Missed by this rule (FN):**

- `G.` (PER)
- `M. -Klinik` (ORG)

**Example 17** (doc_id: `57610`) (sent_id: `57610`)


Zwar wird angeführt , dass die mitgeteilte Verurteilung vom 15. Mai 2017 erst nach der Verurteilung des Beschwerdeführers erfolgt ist ; beanstandet wird aber nur , dass die Annahme des Landgerichts auf UA 30 , der Zeuge M. sei glaubhaft , weil er sich durch seine Angaben erheblich selbst belastet habe , mit der Verurteilung vom 15. Mai 2017 nicht belegt werden könne , da diese andere Taten betreffe .

| Predicted | Gold |
|---|---|
| `M.` | `M.` |

**Example 18** (doc_id: `57665`) (sent_id: `57665`)


Mit Schreiben ihres vorinstanzlichen Prozessbevollmächtigten vom 6. Mai 2015 , das den Briefkopf " T. Ts. & Partner Rechtsanwälte " trägt , die Rechtsanwälte T. , Ts. , M. und Dr. T. auflistet und von Rechtsanwalt T. unterzeichnet wurde , wiederholte die Klägerin den Widerruf .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |
| `T.` | `T.` |

**Missed by this rule (FN):**

- `" T. Ts. & Partner Rechtsanwälte "` (ORG)
- `Ts.` (PER)
- `M.` (PER)

**Example 19** (doc_id: `57819`) (sent_id: `57819`)


Rechtsanwalt B. war dem Angeklagten , der mit der Bestellung eines Pflichtverteidigers nicht einverstanden war , mit Beschluss des Vorsitzenden vom 21. März 2016 zur Verfahrenssicherung als Pflichtverteidiger neben dem Wahlverteidiger Rechtsanwalt P. beigeordnet worden .

| Predicted | Gold |
|---|---|
| `B.` | `B.` |
| `P.` | `P.` |

**Example 20** (doc_id: `57960`) (sent_id: `57960`)


Zu dem Sachverständigengutachten des Prof. Dr. T. sowie dessen ergänzender Stellungnahme hat die Beklagte mitgeteilt , dass sie diesem nach prüfärztlicher Stellungnahme nicht folgen könne ; der Einschätzung des Dr. S. hat sie sich hingegen angeschlossen .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |
| `S.` | `S.` |

**Example 21** (doc_id: `58153`) (sent_id: `58153`)


Auf die Beschwerde des Klägers hat das BSG mit Beschluss vom 14. 11. 2013 ( B 9 V 33/13 B ) das Urteil des LSG aufgehoben und die Sache zurückverwiesen , weil das LSG das Paul-Ehrlich-Institut ( Prof. Dr. C. ) nicht zu Einwänden des Klägers ergänzend befragt hat .

| Predicted | Gold |
|---|---|
| `C.` | `C.` |

**Missed by this rule (FN):**

- `BSG` (ORG)
- `Beschluss vom 14. 11. 2013 ( B 9 V 33/13 B )` (RS)
- `Paul-Ehrlich-Institut` (ORG)

**Example 22** (doc_id: `58450`) (sent_id: `58450`)


In einer ersten Regelbeurteilung vom 23. April 2013 zum Stichtag 1. April 2013 vergab der seinerzeitige Leiter der Abteilung X des BND ( Herr Dr. A. ) das Gesamturteil 7. Auf Einwendungen des Klägers hob der BND diese dienstliche Beurteilung wegen formeller Fehler auf .

| Predicted | Gold |
|---|---|
| `A.` | `A.` |

**Missed by this rule (FN):**

- `Abteilung X des BND` (ORG)
- `BND` (ORG)

**Example 23** (doc_id: `58530`) (sent_id: `58530`)


Ab dem 1. 7. 2008 stellte die Klägerin Dr. L. ( Internist mit Schwerpunkt Hämatologie / Onkologie ) mit einem Beschäftigungsumfang von 10 Stunden / Woche an .

| Predicted | Gold |
|---|---|
| `L.` | `L.` |

**Example 24** (doc_id: `58718`) (sent_id: `58718`)


Der Geschäftsführer der Beklagten S. ist zugleich Geschäftsführer der - mittlerweile in Liquidation befindlichen - Help Food .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Missed by this rule (FN):**

- `Help Food` (ORG)

**Example 25** (doc_id: `58781`) (sent_id: `58781`)


3. Der Angeklagte R. hat die Kosten seines Rechtsmittels zu tragen .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |

**Example 26** (doc_id: `59176`) (sent_id: `59176`)


Nach Verlassen der Bar gegen 2.30 Uhr begleiteten die Angeklagten die Zeugin L. auf dem Nachhauseweg .

| Predicted | Gold |
|---|---|
| `L.` | `L.` |

**Example 27** (doc_id: `59742`) (sent_id: `59742`)


1. Dem Angeklagten A. wird auf seinen Antrag Wiedereinsetzung in den vorigen Stand wegen Versäumung der Frist zur Begründung der Revision gegen das Urteil des Landgerichts Göttingen vom 30. Mai 2017 gewährt .

| Predicted | Gold |
|---|---|
| `A.` | `A.` |

**Missed by this rule (FN):**

- `Landgerichts Göttingen` (ORG)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `57273`) (sent_id: `57273`)


Der Kläger betrieb vormals eine Anwaltssozietät mit Rechtsanwalt C. B. in F. .

**False Positives:**

- `C.` — partial — pred is substring of gold: `C. B.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C. B.`(PER)
- `F.`(LOC)

</details>

---

## `Standalone Surname`

**F1:** 0.259 | **Precision:** 0.855 | **Recall:** 0.152  

**Format:** `regex`  
**Rule ID:** `b6b7238a`  
**Description:**
Captures verified surnames that appear without titles, excluding single letters and common words.

**Content:**
```
\b(?:Franke|Haberkamp|Spelge|Schlünder|Hoch|Roloff|Sieberts|Grube|Suckow|Schneider|Roggenbuck|Sibylle|Spoo|Appl|Bredendiek|Bacher|Wemheuer|Mutzbauer|Koch|Pape|Winzenried|Pielenz|Raum|Brühler|Bender|Lohmann|Remmert|Kley|Schultz|Kleinschmidt|Kirschneck|Matter|Kapels|Reinfelder|Brown|Schäfer|Brückner|Volz|Knoll|Kriener|Nielsen|Mayen|Seiters|Busch|Linck|Leitz|Hamdorf|Fiamingo|Spaniol|Kirchhoff|Gericke|Fritz|Vogelsang|Zwanziger|Kelvin|Lauer|Zeng|Tiemann|Sander|Fischermeier|Çerikci|Kaya|Seyhan|Lorsbach|Maksymiw|Schell|Münzberg|Jäger|Peter|Eschelbach|Kortbein|Schmid|Söchtig|Hacker|Merzbach|Meiser|Ts|Arnoldi|Haupt|Niemann|Becker|Waskow|Eylert|Marx|Fischer|Stresemann|Heinkel|Hayen|Volk|Liebert|Matthias|Kayser|Klein|Maekawa|Bar|Refaeli|Josh|Duhamel|Saime|Özcan|Boolell|McMillan|Rennpferdt|Wollny|Albertshofer|Dorn|Musiol|Kuemmerle|Drüen|Klaus-Dieter|Quentin|Spinner|Schlewing|Schmidt|Limperg|Merkel|Bormann|Berg|Demir|Baykara|Shah|Bu|W|G|F|E|A|S|N|D|K|R|V|Y|I|X|H|C|L|M|O|Z|B1|G1|J1|S1|K1|N1|P1|T1|V1|Y1|A1|D1|E1|F1|G1|H1|I1|J1|K1|L1|M1|N1|O1|P1|Q1|R1|S1|T1|U1|V1|W1|X1|Y1|Z1)\b(?!\s*(?:von|der|die|das|ein|eine|mit|für|auf|in|an|bei|zu|nach|vor|über|unter|durch|ohne|gegen|seit|statt|wegen|um|bis|neben|zwischen|entlang|außer|neben|ab|Gericht|Firma|Behörde|Polizei|Steuern|Umsatz|Kabel|Anlage|Umsatzsteuer|Rechtsanwältin|Rechtsanwalt|Ackerstatusrechten|Höchst-Schmerzgrenze|Marken-Beschwerdesenat|Rechtsanwalt|...|Frau|Herr|Dr\.?|Prof\.?|Anwalt|Gericht|Firma|Behörde|Polizei|Steuern|Umsatz|Kabel|Anlage|Umsatzsteuer|Rechtsanwältin|Rechtsanwalt|Ackerstatusrechten|Höchst-Schmerzgrenze|Marken-Beschwerdesenat|Rechtsanwalt|...))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.855 | 0.152 | 0.259 | 62 | 53 | 9 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 53 | 9 | 293 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53409`) (sent_id: `53409`)


Bender

| Predicted | Gold |
|---|---|
| `Bender` | `Bender` |

**Example 1** (doc_id: `53565`) (sent_id: `53565`)


Koch

| Predicted | Gold |
|---|---|
| `Koch` | `Koch` |

**Example 2** (doc_id: `53669`) (sent_id: `53669`)


Fritz

| Predicted | Gold |
|---|---|
| `Fritz` | `Fritz` |

**Example 3** (doc_id: `54009`) (sent_id: `54009`)


Schäfer

| Predicted | Gold |
|---|---|
| `Schäfer` | `Schäfer` |

**Example 4** (doc_id: `54032`) (sent_id: `54032`)


Hayen

| Predicted | Gold |
|---|---|
| `Hayen` | `Hayen` |

**Example 5** (doc_id: `54290`) (sent_id: `54290`)


Jäger

| Predicted | Gold |
|---|---|
| `Jäger` | `Jäger` |

**Example 6** (doc_id: `54389`) (sent_id: `54389`)


Merkel

| Predicted | Gold |
|---|---|
| `Merkel` | `Merkel` |

**Example 7** (doc_id: `54416`) (sent_id: `54416`)


Eylert

| Predicted | Gold |
|---|---|
| `Eylert` | `Eylert` |

**Example 8** (doc_id: `54477`) (sent_id: `54477`)


Schäfer

| Predicted | Gold |
|---|---|
| `Schäfer` | `Schäfer` |

**Example 9** (doc_id: `54597`) (sent_id: `54597`)


Kayser

| Predicted | Gold |
|---|---|
| `Kayser` | `Kayser` |

**Example 10** (doc_id: `54825`) (sent_id: `54825`)


Spinner

| Predicted | Gold |
|---|---|
| `Spinner` | `Spinner` |

**Example 11** (doc_id: `55048`) (sent_id: `55048`)


Stresemann

| Predicted | Gold |
|---|---|
| `Stresemann` | `Stresemann` |

**Example 12** (doc_id: `55052`) (sent_id: `55052`)


Fischer

| Predicted | Gold |
|---|---|
| `Fischer` | `Fischer` |

**Example 13** (doc_id: `55131`) (sent_id: `55131`)


Becker

| Predicted | Gold |
|---|---|
| `Becker` | `Becker` |

**Example 14** (doc_id: `55244`) (sent_id: `55244`)


Franke

| Predicted | Gold |
|---|---|
| `Franke` | `Franke` |

**Example 15** (doc_id: `55256`) (sent_id: `55256`)


Becker

| Predicted | Gold |
|---|---|
| `Becker` | `Becker` |

**Example 16** (doc_id: `55379`) (sent_id: `55379`)


Raum

| Predicted | Gold |
|---|---|
| `Raum` | `Raum` |

**Example 17** (doc_id: `55493`) (sent_id: `55493`)


Heinkel

| Predicted | Gold |
|---|---|
| `Heinkel` | `Heinkel` |

**Example 18** (doc_id: `55527`) (sent_id: `55527`)


Schmidt

| Predicted | Gold |
|---|---|
| `Schmidt` | `Schmidt` |

**Example 19** (doc_id: `55530`) (sent_id: `55530`)


Quentin

| Predicted | Gold |
|---|---|
| `Quentin` | `Quentin` |

**Example 20** (doc_id: `55897`) (sent_id: `55897`)


Seiters

| Predicted | Gold |
|---|---|
| `Seiters` | `Seiters` |

**Example 21** (doc_id: `55926`) (sent_id: `55926`)


Suckow

| Predicted | Gold |
|---|---|
| `Suckow` | `Suckow` |

**Example 22** (doc_id: `56611`) (sent_id: `56611`)


Grube

| Predicted | Gold |
|---|---|
| `Grube` | `Grube` |

**Example 23** (doc_id: `56892`) (sent_id: `56892`)


Franke

| Predicted | Gold |
|---|---|
| `Franke` | `Franke` |

**Example 24** (doc_id: `57002`) (sent_id: `57002`)


Fritz

| Predicted | Gold |
|---|---|
| `Fritz` | `Fritz` |

**Example 25** (doc_id: `57176`) (sent_id: `57176`)


Zwanziger

| Predicted | Gold |
|---|---|
| `Zwanziger` | `Zwanziger` |

**Example 26** (doc_id: `57471`) (sent_id: `57471`)


Jäger

| Predicted | Gold |
|---|---|
| `Jäger` | `Jäger` |

**Example 27** (doc_id: `57555`) (sent_id: `57555`)


Fischer

| Predicted | Gold |
|---|---|
| `Fischer` | `Fischer` |

**Example 28** (doc_id: `57630`) (sent_id: `57630`)


Fritz

| Predicted | Gold |
|---|---|
| `Fritz` | `Fritz` |

**Example 29** (doc_id: `57727`) (sent_id: `57727`)


Roggenbuck

| Predicted | Gold |
|---|---|
| `Roggenbuck` | `Roggenbuck` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53641`) (sent_id: `53641`)


D5 EP 0 160 797 A1

**False Positives:**

- `A1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `56422`) (sent_id: `56422`)


E 9 EP 1 308 030 B1 ,

**False Positives:**

- `B1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `56560`) (sent_id: `56560`)


D18 DE 10 2009 044 546 A1

**False Positives:**

- `A1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 3** (doc_id: `56599`) (sent_id: `56599`)


E4 DE 39 30 353 A1 ,

**False Positives:**

- `A1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `56890`) (sent_id: `56890`)


D1 DE 10 2004 031 624 A1

**False Positives:**

- `A1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `57356`) (sent_id: `57356`)


D11 DE 100 34 354 A1

**False Positives:**

- `A1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 6** (doc_id: `57452`) (sent_id: `57452`)


HLNK28 WO 96/38131 A1

**False Positives:**

- `A1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 7** (doc_id: `58333`) (sent_id: `58333`)


D2 DE 199 52 004 A1 ;

**False Positives:**

- `A1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `58888`) (sent_id: `58888`)


NIK5 / NiK4 WO 97/03675 A1

**False Positives:**

- `A1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Full Names with Titles`

**F1:** 0.091 | **Precision:** 0.680 | **Recall:** 0.049  

**Format:** `regex`  
**Rule ID:** `d8741160`  
**Description:**
Captures names preceded by titles like Dr., Prof., Dipl.-Ing., etc., ensuring the title is not captured and the name is a valid surname.

**Content:**
```
(?:Dr\.?\s+|Prof\.?\s+|Dipl\.-Ing\.\s+|Dipl\.-Psych\.\s+|Richter\s+|Anwalt\s+|Rechtsanwältin\s+|Rechtsanwalt\s+)([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.680 | 0.049 | 0.091 | 25 | 17 | 8 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 17 | 8 | 331 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `54291`) (sent_id: `54291`)


In der Beschwerdesache betreffend die Designanmeldung 40 2016 201 675.4 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Merzbach` | `Merzbach` |
| `Meiser` | `Meiser` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Hacker` (PER)

**Example 1** (doc_id: `54886`) (sent_id: `54886`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2012 063 820.1 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 5. Dezember 2017 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Nielsen` | `Nielsen` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Kriener` (PER)

**Example 2** (doc_id: `55301`) (sent_id: `55301`)


Dr. Milger

| Predicted | Gold |
|---|---|
| `Milger` | `Milger` |

**Example 3** (doc_id: `55818`) (sent_id: `55818`)


In der Beschwerdesache betreffend die Patentanmeldung 10 2014 003 988.9 hat der 23. Senat ( Technischer Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 8. Mai 2018 unter Mitwirkung des Vorsitzenden Richters Dr. Strößner sowie der Richter Dr. Friedrich , Dr. Zebisch und Dr. Himmelmann beschlossen :

| Predicted | Gold |
|---|---|
| `Strößner` | `Strößner` |
| `Zebisch` | `Zebisch` |
| `Himmelmann` | `Himmelmann` |

**Missed by this rule (FN):**

- `23. Senat ( Technischer Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Friedrich` (PER)

**Example 4** (doc_id: `56015`) (sent_id: `56015`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2011 068 984.9 hat der 26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 8. Januar 2018 unter Mitwirkung der Vorsitzenden Richterin Kortge sowie der Richter Jacobi und Schödel beschlossen :

| Predicted | Gold |
|---|---|
| `Jacobi` | `Jacobi` |

**Missed by this rule (FN):**

- `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Kortge` (PER)
- `Schödel` (PER)

**Example 5** (doc_id: `59053`) (sent_id: `59053`)


c ) Unter diesen Umständen ist die Besorgnis des Beschwerdeführers nachvollziehbar , Richter Müller werde die zu entscheidenden , in hohem Maße wertungsabhängigen und von Vorverständnissen geprägten Rechtsfragen möglicherweise nicht mehr in jeder Hinsicht offen und unbefangen beurteilen können ( vgl. BVerfGE 72 , 296 < 298 > ; 95 , 189 < 192 > ; 135 , 248 < 259 Rn. 27 > ) .

| Predicted | Gold |
|---|---|
| `Müller` | `Müller` |

**Missed by this rule (FN):**

- `BVerfGE 72 , 296 < 298 > ; 95 , 189 < 192 > ; 135 , 248 < 259 Rn. 27 >` (RS)

**Example 6** (doc_id: `59509`) (sent_id: `59509`)


In der Beschwerdesache betreffend die Marke 30 2009 026 804 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 21. September 2017 unter Mitwirkung der Richter Merzbach , Dr. Meiser und Schödel beschlossen :

| Predicted | Gold |
|---|---|
| `Merzbach` | `Merzbach` |
| `Meiser` | `Meiser` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Schödel` (PER)

**Example 7** (doc_id: `59628`) (sent_id: `59628`)


In der Beschwerdesache betreffend die Marke 30 2012 041 338 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 15. November 2017 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Nielsen` | `Nielsen` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Kriener` (PER)

**Example 8** (doc_id: `59761`) (sent_id: `59761`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2015 031 519.2 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Knoll sowie der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Nielsen` | `Nielsen` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Kriener` (PER)

**Example 9** (doc_id: `59948`) (sent_id: `59948`)


In der Beschwerdesache betreffend die international registrierte Marke IR 1 160 635 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 9. November 2017 unter Mitwirkung des Vorsitzenden Richters Professor Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Hacker` | `Hacker` |
| `Merzbach` | `Merzbach` |
| `Meiser` | `Meiser` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)

**Example 10** (doc_id: `60001`) (sent_id: `60001`)


Dr. Milger

| Predicted | Gold |
|---|---|
| `Milger` | `Milger` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53397`) (sent_id: `53397`)


HLNK20 Gutachten Dr. Jay B. Saoud aus dem parallelen britischen Verfahren vom 14. April 2016 , 21 Seiten und 12 Seiten Anlagen

**False Positives:**

- `Jay` — partial — pred is substring of gold: `Jay B. Saoud`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Jay B. Saoud`(PER)

**Example 1** (doc_id: `53890`) (sent_id: `53890`)


Im Berufungsverfahren hat der zuständige Berichterstatter des LSG den Sachverständigen Prof. Dr. T. im Termin zur mündlichen Verhandlung ergänzend angehört .

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `T.`(PER)

**Example 2** (doc_id: `54018`) (sent_id: `54018`)


In der Beschwerdesache betreffend die Marke 30 2010 022 988 hat der 27. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 10. Mai 2017 durch die Vorsitzende Richterin Klante , den Richter Dr. Himmelmann und die Richterin Lachenmayr-Nikolaou beschlossen :

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `27. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Klante`(PER)
- `Himmelmann`(PER)
- `Lachenmayr-Nikolaou`(PER)

**Example 3** (doc_id: `54291`) (sent_id: `54291`)


In der Beschwerdesache betreffend die Designanmeldung 40 2016 201 675.4 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 4** (doc_id: `55088`) (sent_id: `55088`)


Zwar hat der Senat durchaus zur Kenntnis genommen , dass die Klägerin in dem von ihr benannten Schriftsatz vom 11. 2. 2016 - auf die Anhörung des LSG zur Entscheidung durch Beschluss - durch die Bezugnahme auf den Schriftsatz vom 7. 3. 2016 ihren Antrag bei Prof. Dr. T. nachzufragen , in welcher Form und in welchem Umfang er an der Erstellung des Sachverständigengutachtens beteiligt gewesen sei , wiederholt hat .

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `T.`(PER)

**Example 5** (doc_id: `55818`) (sent_id: `55818`)


In der Beschwerdesache betreffend die Patentanmeldung 10 2014 003 988.9 hat der 23. Senat ( Technischer Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 8. Mai 2018 unter Mitwirkung des Vorsitzenden Richters Dr. Strößner sowie der Richter Dr. Friedrich , Dr. Zebisch und Dr. Himmelmann beschlossen :

**False Positives:**

- `Dr` — similar text (different position): `Friedrich`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `23. Senat ( Technischer Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Strößner`(PER)
- `Friedrich`(PER)
- `Zebisch`(PER)
- `Himmelmann`(PER)

**Example 6** (doc_id: `57960`) (sent_id: `57960`)


Zu dem Sachverständigengutachten des Prof. Dr. T. sowie dessen ergänzender Stellungnahme hat die Beklagte mitgeteilt , dass sie diesem nach prüfärztlicher Stellungnahme nicht folgen könne ; der Einschätzung des Dr. S. hat sie sich hingegen angeschlossen .

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `T.`(PER)
- `S.`(PER)

**Example 7** (doc_id: `58153`) (sent_id: `58153`)


Auf die Beschwerde des Klägers hat das BSG mit Beschluss vom 14. 11. 2013 ( B 9 V 33/13 B ) das Urteil des LSG aufgehoben und die Sache zurückverwiesen , weil das LSG das Paul-Ehrlich-Institut ( Prof. Dr. C. ) nicht zu Einwänden des Klägers ergänzend befragt hat .

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG`(ORG)
- `Beschluss vom 14. 11. 2013 ( B 9 V 33/13 B )`(RS)
- `Paul-Ehrlich-Institut`(ORG)
- `C.`(PER)

</details>

---

## `Anonymized Initials with Ellipsis and Numbers`

**F1:** 0.032 | **Precision:** 0.231 | **Recall:** 0.017  

**Format:** `regex`  
**Rule ID:** `b224d30a`  
**Description:**
Captures anonymized names with an ellipsis (e.g., 'P …', 'B1 …', 'G1 …') following role indicators or in legal contexts.

**Content:**
```
\b([A-Z]\d*\s+…)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.231 | 0.017 | 0.032 | 26 | 6 | 20 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 6 | 20 | 338 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53829`) (sent_id: `53829`)


An der letzten Voraussetzung , der Eignung des an Patentanwalt K … Wohnung bzw. Kanzlei angebrachten Briefkastens zur sicheren Aufbewahrung , hat es aber zum fraglichen Zeitpunkt der Einlegung des den Prüfungsbescheid enthaltenden Umschlags gefehlt .

| Predicted | Gold |
|---|---|
| `K …` | `K …` |

**Example 1** (doc_id: `56141`) (sent_id: `56141`)


Die Widersprechende zu 2. hat mit den Schriftsätzen vom 23. September 2014 sowie - im Beschwerdeverfahren - 4. August 2017 Unterlagen zur Benutzung der Widerspruchsmarke 30 2008 062 715 NIDO einschließlich einer eidesstattlichen Versicherung des Herrn H … vom 19. September 2014 ( An-lage W13 ) vorgelegt ( Anlagen W9 - W16 sowie W21 - W30 ) .

| Predicted | Gold |
|---|---|
| `H …` | `H …` |

**Missed by this rule (FN):**

- `NIDO` (ORG)

**Example 2** (doc_id: `58746`) (sent_id: `58746`)


Auch die Anlagen zur eidesstattlichen Versicherung von Herrn N … seien zur Glaubhaftmachung einer markenmäßigen Benutzung ungeeignet .

| Predicted | Gold |
|---|---|
| `N …` | `N …` |

**Example 3** (doc_id: `58765`) (sent_id: `58765`)


Zum einen sei Herr N … erst seit Dezember 2016 Vorstandsvorsitzender der Widersprechenden und könne daher nicht mit den erklärten Verhältnissen im jeweils maßgeblichen Benutzungszeitraum vertraut gewesen sein .

| Predicted | Gold |
|---|---|
| `N …` | `N …` |

**Example 4** (doc_id: `59898`) (sent_id: `59898`)


Weiterhin macht er geltend , Patentanwalt K … sei wegen einer psychischen Erkrankung zur Zeit der Zustellversuche des DPMA geschäftsunfähig nach § 104 Abs. 2 BGB gewesen , weshalb die Zustellungen unwirksam seien .

| Predicted | Gold |
|---|---|
| `K …` | `K …` |

**Missed by this rule (FN):**

- `DPMA` (ORG)
- `§ 104 Abs. 2 BGB` (NRM)

**Example 5** (doc_id: `59964`) (sent_id: `59964`)


bb ) Der Senat war auch nicht gehalten , den Zeugen H … über das Beweisangebot der Einsprechenden hinaus dazu zu vernehmen , dass durch Messungen an dem behauptet vorbenutzten Schalter auch die weiteren , nicht im Beweisantrag aufgeführten Merkmale des erteilten Patentanspruchs 1 für den Fachmann feststellbar seien .

| Predicted | Gold |
|---|---|
| `H …` | `H …` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53472`) (sent_id: `53472`)


Sie ist dann allerdings die einzige Beschwerdeführerin , denn für die Beschwerde der R … GmbH in S … , fehlt es an der Zahlung der erforderlichen weiteren , zweiten Beschwerdegebühr , so dass diese als nicht eingelegt gilt .

**False Positives:**

- `R …` — partial — pred is substring of gold: `R … GmbH`
- `S …` — type mismatch — same span as gold: `S …`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `R … GmbH`(ORG)
- `S …`(LOC)

**Example 1** (doc_id: `54492`) (sent_id: `54492`)


Der Kontaktsockel sei auf der Messe „ Semicon Europe 2006 “ in München ausgestellt gewesen und zudem an einen Kunden , die Firma A … Inc. , geliefert worden .

**False Positives:**

- `A …` — partial — pred is substring of gold: `A … Inc.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europe`(LOC)
- `München`(LOC)
- `A … Inc.`(ORG)

**Example 2** (doc_id: `54517`) (sent_id: `54517`)


b. Ausgehend davon hat die Widersprechende zu 1 zunächst eine rechtserhaltende Benutzung der Widerspruchsmarke NIVONA gemäß Art. 15 GMV in der Gemeinschaft für den nach §§ 43 Abs. 1 Satz 2 , 125 b Nr. 4 MarkenG zum Zeitpunkt der mündlichen Verhandlung maßgeblichen Fünfjahreszeitraum September 2012 bis September 2017 ungeachtet der von ihr mit Schriftsatz vom 3. Januar 2014 für den Zeitraum 2011 bis 1. Halbjahr 2013 eingereichten Unterlagen ( Anlagen W4 - W12 ) jedenfalls mit den im Beschwerdeverfahren mit Schriftsatz vom 22. Juni 2017 ( Bl. 56 d. A. ) für den Zeitraum 2014 - 2016 eingereichten Unterlagen W14 bis W17 , insbesondere der weiteren eidesstattlichen Versicherung des Geschäftsführers W1 … vom 19. Juni 2017 ( Anlage W19 , Bl. 66 d. A. ) für die Waren „ elektrische Kaffeevollautomaten , elektrische Kaffeemühlen ; Reinigungsmittel und Entkalkungsmittel ( Reinigungstabs , CreamCleaner und Entkalker ) sowie Milchbehälter “ glaubhaft gemacht .

**False Positives:**

- `W1 …` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `NIVONA`(ORG)
- `Art. 15 GMV`(NRM)
- `§§ 43 Abs. 1 Satz 2 , 125 b Nr. 4 MarkenG`(NRM)

**Example 3** (doc_id: `54753`) (sent_id: `54753`)


Mit dieser rechtzeitigen Gebührenzahlung ist die Beschwerde der B … Aktiengesellschaft wirksam erhoben .

**False Positives:**

- `B …` — partial — pred is substring of gold: `B … Aktiengesellschaft`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `B … Aktiengesellschaft`(ORG)

**Example 4** (doc_id: `55099`) (sent_id: `55099`)


Die Gegenstände der Ansprüche 1 , 9 und 11 haben somit in der Fa. R … GmbH & Co. KG als nicht öffentlich zugänglich im Sinne des § 3 ( 1 ) , 2 PatG zu gelten , so dass insoweit keine offenkundige Vorbenutzung vorliegt .

**False Positives:**

- `R …` — partial — pred is substring of gold: `R … GmbH & Co. KG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `R … GmbH & Co. KG`(ORG)
- `§ 3 ( 1 ) , 2 PatG`(NRM)

**Example 5** (doc_id: `55422`) (sent_id: `55422`)


Anlage A3 Handzeichnung zum Ablauf von Werksbesichtigungen bei der R … GmbH & Co. KG ,

**False Positives:**

- `R …` — partial — pred is substring of gold: `R … GmbH & Co. KG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `R … GmbH & Co. KG`(ORG)

**Example 6** (doc_id: `56144`) (sent_id: `56144`)


Denn bereits zu Beginn des Lizenzzeitraums produzierte und vertrieb der M … seit fast 10 Jahren das Medikament Isentress mit dem bis 2014 einzigen ungeboosteten Integraseinhibitor ( Raltegravir ) , mit dem Umsätze in Höhe von jährlich ca. … US- $ weltweit , in Deutschland in Höhe von ca. … € erzielt worden sind ( von der Beklagten als „ Blockbuster “ bezeichnet ) .

**False Positives:**

- `M …` — type mismatch — same span as gold: `M …`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `M …`(ORG)
- `Deutschland`(LOC)

**Example 7** (doc_id: `56697`) (sent_id: `56697`)


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

**Example 8** (doc_id: `58075`) (sent_id: `58075`)


Jedoch trägt sie vor , dass sämtliche Besucher , die die Fa. R … GmbH & Co. KG besuchten , in der Vergangenheit und bis heute stets zur Geheimhaltung verpflichtet worden seien .

**False Positives:**

- `R …` — partial — pred is substring of gold: `R … GmbH & Co. KG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `R … GmbH & Co. KG`(ORG)

**Example 9** (doc_id: `58293`) (sent_id: `58293`)


Der Senat hat aufgrund einer Zusammenschau der Rechnung D4 , des zugehörigen Lieferscheins D8 und des Produktprogramms D3 keinen Zweifel daran , dass ein aus dem Produktprogramm Sicherheitstechnik der Firma E … bekannter Sicherheitsschalter am 18. / 20. 02. 2009 an die Firma C … GmbH in E … verkauft und auch geliefert wurde .

**False Positives:**

- `E …` — type mismatch — same span as gold: `E …`
- `C …` — partial — pred is substring of gold: `C … GmbH`
- `E …` — similar text (different position): `E …`

> overlaps gold: 3  |  likely missing annotation: 0

**Gold Entities:**

- `E …`(ORG)
- `C … GmbH`(ORG)
- `E …`(LOC)

**Example 10** (doc_id: `58709`) (sent_id: `58709`)


ab ) Auch die geltend gemachte offenkundige Vorbenutzung in der Fa. S2 … GmbH kann nicht berücksichtigt werden , denn sie ist hinsichtlich dessen , was angeblich offenkundig vorbenutzt wurde , nicht hinreichend substantiiert .

**False Positives:**

- `S2 …` — partial — pred is substring of gold: `S2 … GmbH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S2 … GmbH`(ORG)

**Example 11** (doc_id: `58823`) (sent_id: `58823`)


Die Anmelderin L … LLC als Vorgängerin der Beklagten sei zum Zeitpunkt der Anmeldung des Streitpatents ausweislich der Dokumente HLNK2 bis HLNK 4 Rechtsnachfolgerin der Anmelder der Prioritätsanmeldung gewesen .

**False Positives:**

- `L …` — partial — pred is substring of gold: `L … LLC`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `L … LLC`(ORG)

**Example 12** (doc_id: `59314`) (sent_id: `59314`)


Die Einsprechende stützt ihre Argumentation bezüglich der offenkundigen Vorbenutzung u. a. auf die Druckschriften D3 , D4 , D5 und D8 , die einen Verkauf eines aus dem Produktprogramm Sicherheitstechnik der Fa. E … bekannten Sicherheitsschalters ohne Geheimhaltungspflicht belegen sollen .

**False Positives:**

- `E …` — type mismatch — same span as gold: `E …`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `E …`(ORG)

**Example 13** (doc_id: `59583`) (sent_id: `59583`)


- Q1 … Q4 Überwachte Halbleiterausgänge

**False Positives:**

- `Q1 …` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Initials with Surname`

**F1:** 0.039 | **Precision:** 0.081 | **Recall:** 0.026  

**Format:** `regex`  
**Rule ID:** `b5047799`  
**Description:**
Captures names with an initial followed by a surname (e.g., 'M. Trümner', 'F. Rojahn').

**Content:**
```
\b([A-Z]\.)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.081 | 0.026 | 0.039 | 111 | 9 | 102 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 9 | 102 | 339 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `54587`) (sent_id: `54587`)


D. Reidelbach

| Predicted | Gold |
|---|---|
| `D. Reidelbach` | `D. Reidelbach` |

**Example 1** (doc_id: `55941`) (sent_id: `55941`)


K. Schmidt

| Predicted | Gold |
|---|---|
| `K. Schmidt` | `K. Schmidt` |

**Example 2** (doc_id: `56556`) (sent_id: `56556`)


D14 J. Deubener et al. , “ Induction time analysis of nucleation and crystal growth in di- and metasilicate glasses ” , Journal of Non-Crystalline Solids , 1993 , 163 , Seiten 1 bis 12

| Predicted | Gold |
|---|---|
| `J. Deubener` | `J. Deubener` |

**Example 3** (doc_id: `56970`) (sent_id: `56970`)


M. Jostes

| Predicted | Gold |
|---|---|
| `M. Jostes` | `M. Jostes` |

**Example 4** (doc_id: `57047`) (sent_id: `57047`)


K. Schmidt

| Predicted | Gold |
|---|---|
| `K. Schmidt` | `K. Schmidt` |

**Example 5** (doc_id: `58172`) (sent_id: `58172`)


J. Ratayczak

| Predicted | Gold |
|---|---|
| `J. Ratayczak` | `J. Ratayczak` |

**Example 6** (doc_id: `59171`) (sent_id: `59171`)


K. Schmidt

| Predicted | Gold |
|---|---|
| `K. Schmidt` | `K. Schmidt` |

**Example 7** (doc_id: `59274`) (sent_id: `59274`)


W. Reinfelder

| Predicted | Gold |
|---|---|
| `W. Reinfelder` | `W. Reinfelder` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53397`) (sent_id: `53397`)


HLNK20 Gutachten Dr. Jay B. Saoud aus dem parallelen britischen Verfahren vom 14. April 2016 , 21 Seiten und 12 Seiten Anlagen

**False Positives:**

- `B. Saoud` — partial — pred is substring of gold: `Jay B. Saoud`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Jay B. Saoud`(PER)

**Example 1** (doc_id: `53692`) (sent_id: `53692`)


I. Das LSG hat mit Urteil vom 11. 5. 2017 einen Zahlungsanspruch der Klägerin ( eine aus zwei Personen bestehende , im Partnerschaftsregister eingetragene Physiotherapie-Partnerschaft ) in Höhe von 7249,01 Euro für physiotherapeutische Leistungen verneint , nachdem die beklagte Krankenkasse die erbrachten Leistungen zunächst bezahlt , die Zahlungen aber wieder zurückgefordert und die Rückforderung schließlich im Wege der Aufrechnung durchgesetzt hatte .

**False Positives:**

- `I. Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `53741`) (sent_id: `53741`)


I. Erläuterung zur ersten Vorlagefrage

**False Positives:**

- `I. Erläuterung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 3** (doc_id: `53763`) (sent_id: `53763`)


Wird als Verfahrensmangel gerügt , das FG habe einen gestellten Beweisantrag übergangen ( Rüge mangelnder Sachaufklärung gemäß § 76 Abs. 1 Satz 1 FGO ) , so ist darzulegen , welche Tatfrage aufklärungsbedürftig ist , welche Beweismittel das FG zu welchem Beweisthema nicht erhoben hat , die genauen Fundstellen ( Schriftsatz mit Datum und Seitenzahl , Terminprotokoll ) , in denen die Beweismittel und Beweisthemen angeführt worden sind , das voraussichtliche Ergebnis der Beweisaufnahme , inwiefern das Urteil des FG aufgrund dessen sachlich-rechtlicher Auffassung auf der unterbliebenen Beweisaufnahme beruhen kann und dass die Nichterhebung der Beweise vor dem FG rechtzeitig gerügt worden ist oder aufgrund des Verhaltens des FG nicht mehr vor diesem gerügt werden konnte ( ständige BFH-Rechtsprechung , z.B. Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85 , und vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125 , m. w. N. ) .

**False Positives:**

- `B. Beschlüsse` — positional overlap with gold: `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 76 Abs. 1 Satz 1 FGO`(NRM)
- `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`(RS)
- `vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125`(RS)

**Example 4** (doc_id: `53767`) (sent_id: `53767`)


Ob der Erwerb eigener Anteile auf der Gesellschaftsebene entsprechend der durch das BilMoG geänderten handelsrechtlichen Vorschriften ( Einfügung des § 272 Abs. 1a und 1b HGB ) steuerrechtlich nicht mehr als Erwerbsvorgang anzusehen , sondern nunmehr als " Teilliquidation " und daher " wie " eine Kapitalherabsetzung zu behandeln ist ( so BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f. ; zustimmend z.B. Blumenberg / Lechner , DB 2014 , 141 , 147 ; kritisch z.B. Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55 ) , kann im Streitfall mangels Erheblichkeit für die Entscheidung offenbleiben .

**False Positives:**

- `B. Blumenberg` — positional overlap with gold: `Blumenberg / Lechner , DB 2014 , 141 , 147`
- `B. Gosch` — positional overlap with gold: `Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BilMoG`(NRM)
- `§ 272 Abs. 1a und 1b HGB`(NRM)
- `BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f.`(REG)
- `Blumenberg / Lechner , DB 2014 , 141 , 147`(LIT)
- `Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55`(LIT)

**Example 5** (doc_id: `53897`) (sent_id: `53897`)


Sofern die Beschwerdeführerin ferner ausführt , dass der Druckschrift D4 ( z.B. Figur 1 ) ein Anhänger zu entnehmen sei , der gemäß Merkmal M7 eine Anhängerdeichsel mit einer Anhängerkupplung und einer Drehwelle aufweise , die unterhalb der Anhängerkupplung angeordnet sei , so kann ihr darin zugestimmt werden .

**False Positives:**

- `B. Figur` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 6** (doc_id: `53940`) (sent_id: `53940`)


I. Auf das Arbeitsverhältnis finden kraft arbeitsvertraglicher Bezugnahme der Tarifvertrag zur Angleichung des Tarifrechts des Landes Berlin an das Tarifrecht der Tarifgemeinschaft deutscher Länder ( Angleichungs-TV Land Berlin ) vom 14. Oktober 2010 und gem. dessen § 2 der TV-L sowie der Tarifvertrag zur Überleitung der Beschäftigten der Länder in den TV-L und zur Regelung des Übergangsrechts ( TVÜ-Länder ) ab dem 1. November 2010 Anwendung .

**False Positives:**

- `I. Auf` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Tarifvertrag zur Angleichung des Tarifrechts des Landes Berlin an das Tarifrecht der Tarifgemeinschaft deutscher Länder`(REG)
- `Angleichungs-TV Land Berlin`(REG)
- `§ 2 der TV-L`(REG)
- `Tarifvertrag zur Überleitung der Beschäftigten der Länder in den TV-L und zur Regelung des Übergangsrechts`(REG)
- `TVÜ-Länder`(REG)

**Example 7** (doc_id: `53955`) (sent_id: `53955`)


I. Die Beschwerdeführerin ist Inhaberin des am 15. Mai 1993 angemeldeten und am 30. September 2009 erteilten europäischen Patents EP 0 835 663 ( DE 693 34 297 ) , das mittlerweile durch Zeitablauf erloschen ist .

**False Positives:**

- `I. Die Beschwerdeführerin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `54071`) (sent_id: `54071`)


C. Das Landesarbeitsgericht hat die gegen die Beendigung des Arbeitsverhältnisses der Parteien durch die außerordentliche Kündigung der Beklagten vom 28. Juli 2016 gerichtete Kündigungsschutzklage zu Recht abgewiesen .

**False Positives:**

- `C. Das Landesarbeitsgericht` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 9** (doc_id: `54143`) (sent_id: `54143`)


Dieser während des Revisionsverfahrens eingetretene Zuständigkeitswechsel führt zu einem gesetzlichen Beteiligtenwechsel ( vgl. z.B. Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109 , m. w. N. ) .

**False Positives:**

- `B. Urteil` — positional overlap with gold: `Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109`(RS)

**Example 10** (doc_id: `54169`) (sent_id: `54169`)


Unabhängig von der Frage , ob ein solcher Antrag - entgegen der Auffassung des Antragstellers - gemäß § 67 Abs. 4 VwGO dem Vertretungszwang unterliegt ( vgl. verneinend z.B. : BVerwG , Beschluss vom 6. Juni 1972 - 3 ER 404.71 - Buchholz 310 § 53 VwGO Nr. 6 ; Ziekow , in : Sodan / Ziekow , VwGO , 4. Aufl. 2014 , § 53 Rn. 20 ; Kraft , in : Eyermann , VwGO , 14. Aufl. 2014 , § 53 Rn. 14 ; Unruh , in : HK-Verwaltungsrecht , 3. Aufl. 2013 , § 53 Rn. 15 ; Scheidler , VR 2012 , 113 ; bejahend z.B. : W. - R. Schenke , in : Kopp / Schenke , VwGO , 23. Aufl. 2017 , § 53 Rn. 10 ; Redeker / von Oertzen , VwGO , 15. Aufl. 2010 , § 53 Rn. 5 ; vermittelnd z.B. : Bier , in : Schoch / Schneider / Bier , VwGO , Stand Oktober 2016 , § 53 Rn. 13 ) , ist der Antrag hier jedenfalls deshalb unzulässig , weil die gesetzlichen Voraussetzungen , unter denen das Bundesverwaltungsgericht eine Zuständigkeitsbestimmung überhaupt vornehmen darf , nicht erfüllt sind .

**False Positives:**

- `R. Schenke` — partial — pred is substring of gold: `W. - R. Schenke , in : Kopp / Schenke , VwGO , 23. Aufl. 2017 , § 53 Rn. 10`

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

**Example 11** (doc_id: `54179`) (sent_id: `54179`)


I. Mit dem angefochtenen Beschluss vom 4. November 2015 hat die Patentabteilung 43 des Deutschen Patent- und Markenamts das Patent 103 36 913 mit der Bezeichnung

**False Positives:**

- `I. Mit` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Patentabteilung 43 des Deutschen Patent- und Markenamts`(ORG)

**Example 12** (doc_id: `54337`) (sent_id: `54337`)


I. Die Verfassungsbeschwerde betrifft die Höhe des Landesblindengeldes in Schleswig-Holstein nach deren Reduzierung auf 200 Euro monatlich ab 1. Januar 2011 .

**False Positives:**

- `I. Die Verfassungsbeschwerde` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Schleswig-Holstein`(LOC)

**Example 13** (doc_id: `54348`) (sent_id: `54348`)


I. Soweit es die Verurteilung wegen der Tat vom 31. Mai 2015 betrifft , liegen dem folgende Feststellungen und Wertungen zu Grunde :

**False Positives:**

- `I. Soweit` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 14** (doc_id: `54572`) (sent_id: `54572`)


I. Die von der Beschwerdeführerin als gleichheitswidrig beanstandeten Regelungen durch den von ihr mittelbar angegriffenen § 7 Satz 2 Nr. 2 GewStG sind verfassungsgemäß ; der Gesetzgeber bewegt sich mit dieser Neuregelung des Jahres 2002 im Rahmen seiner Gestaltungsbefugnis .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 7 Satz 2 Nr. 2 GewStG`(NRM)

**Example 15** (doc_id: `54732`) (sent_id: `54732`)


I. Am 22. Mai 2013 ist das Zeichen

**False Positives:**

- `I. Am` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 16** (doc_id: `54901`) (sent_id: `54901`)


3. Die vorstehenden Erwägungen gelten entsprechend , soweit der Kläger seinen Anspruch allein auf Abschn. C. Ziff. 2.6 Abs. 1 STV stützt .

**False Positives:**

- `C. Ziff` — partial — pred is substring of gold: `Abschn. C. Ziff. 2.6 Abs. 1 STV`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Abschn. C. Ziff. 2.6 Abs. 1 STV`(REG)

**Example 17** (doc_id: `55135`) (sent_id: `55135`)


Eine unterbliebene notwendige Beiladung stellt einen vom Rechtsmittelgericht von Amts wegen zu prüfenden Verstoß gegen die Grundordnung des Verfahrens dar ( z.B. Senatsurteil vom 11. Juli 2017 I R 34/14 , juris ) .

**False Positives:**

- `B. Senatsurteil` — positional overlap with gold: `Senatsurteil vom 11. Juli 2017 I R 34/14 , juris`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Senatsurteil vom 11. Juli 2017 I R 34/14 , juris`(RS)

**Example 18** (doc_id: `55160`) (sent_id: `55160`)


I. Die Klage ist zulässig , insbesondere hinreichend bestimmt iSv. § 253 Abs. 2 Nr. 2 ZPO .

**False Positives:**

- `I. Die Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 253 Abs. 2 Nr. 2 ZPO`(NRM)

**Example 19** (doc_id: `55200`) (sent_id: `55200`)


I. Der Anspruch auf die geltend gemachte Abfindung folgt nicht aus § 11 Abs. 1 TV ATZ .

**False Positives:**

- `I. Der Anspruch` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 11 Abs. 1 TV ATZ`(REG)

**Example 20** (doc_id: `55285`) (sent_id: `55285`)


I. Das am 26. August 2013 angemeldete Zeichen Fireslim ist am 10. Januar 2014 unter der Nr. 30 2013 048 208 in das beim Deutschen Patent- und Markenamt geführte Markenregister für die nachfolgenden Waren und Dienstleistungen der Klassen 9 , 35 und 38 eingetragen worden :

**False Positives:**

- `I. Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Fireslim`(ORG)
- `Deutschen Patent- und Markenamt`(ORG)

**Example 21** (doc_id: `55398`) (sent_id: `55398`)


I. Die Beklagte ist Alleinerbin ihres am 24. Mai 2013 verstorbenen Ehemannes D. F. .

**False Positives:**

- `I. Die Beklagte` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `D. F.`(PER)

**Example 22** (doc_id: `55563`) (sent_id: `55563`)


B. Die Rechtsbeschwerde der Arbeitgeberin zu 1. ist begründet .

**False Positives:**

- `B. Die Rechtsbeschwerde` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 23** (doc_id: `55567`) (sent_id: `55567`)


Nach Durchführung einer Außenprüfung vertrat der Beklagte und Beschwerdegegner ( das Finanzamt - FA - ) in den Umsatzsteuer-Änderungsbescheiden für die Streitjahre vom 6. Mai 2014 die Auffassung , die genannten Leistungen seien von Personen in Anspruch genommen worden , denen aufgrund von Verkehrsdelikten ( z.B. Fahren unter Alkohol- oder Drogeneinfluss , Tempo- und / oder Abstandsverstöße etc. ) ihre Fahrerlaubnis entzogen worden sei , und die sich zur Wiedererlangung der Fahrerlaubnis einer medizinisch-psychologischen Untersuchung ( MPU ) i. S. des § 2 Abs. 8 des Straßenverkehrsgesetzes ( StVG ) hätten unterziehen müssen .

**False Positives:**

- `B. Fahren` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 2 Abs. 8 des Straßenverkehrsgesetzes`(NRM)
- `StVG`(NRM)

**Example 24** (doc_id: `55569`) (sent_id: `55569`)


I. Die Klägerin und Beschwerdeführerin ( Klägerin ) ist umsatzsteuerrechtlich Organgesellschaft des Organträgers N .

**False Positives:**

- `I. Die Klägerin` — similar text (different position): `N`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `N`(ORG)

**Example 25** (doc_id: `55620`) (sent_id: `55620`)


I. Die Parteien streiten über die Widerruflichkeit der auf Abschluss zweier Verbraucherdarlehensverträge gerichteten Willenserklärungen der Kläger .

**False Positives:**

- `I. Die Parteien` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 26** (doc_id: `55743`) (sent_id: `55743`)


I. Das Landgericht hat - soweit für das Revisionsverfahren bedeutsam - folgende Feststellungen getroffen :

**False Positives:**

- `I. Das Landgericht` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 27** (doc_id: `55795`) (sent_id: `55795`)


b ) Bereits aus dem Wortlaut " außergewöhnliche Belastungen " folgt - auch ohne einen Klammerverweis auf die §§ 33 bis 33b EStG - , dass § 26a Abs. 2 Satz 1 Halbsatz 1 EStG ( auch ) solche Aufwendungen erfasst , die über den Behinderten-Pauschbetrag i. S. des § 33b Abs. 1 EStG abgedeckt werden ( anders z.B. Blümich / Ettlich , § 26a EStG Rz 25 ; Pflüger in Herrmann / Heuer / Raupach - HHR - , § 26a EStG Rz 60 ; Nacke in Littmann / Bitz / Pust , Das Einkommensteuerrecht , Kommentar , § 33b Rz 51a ) .

**False Positives:**

- `B. Blümich` — positional overlap with gold: `Blümich / Ettlich , § 26a EStG Rz 25`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§§ 33 bis 33b EStG -`(NRM)
- `§ 26a Abs. 2 Satz 1 Halbsatz 1 EStG`(NRM)
- `§ 33b Abs. 1 EStG`(NRM)
- `Blümich / Ettlich , § 26a EStG Rz 25`(LIT)
- `Pflüger in Herrmann / Heuer / Raupach - HHR - , § 26a EStG Rz 60`(LIT)
- `Nacke in Littmann / Bitz / Pust , Das Einkommensteuerrecht , Kommentar , § 33b Rz 51a`(LIT)

**Example 28** (doc_id: `55939`) (sent_id: `55939`)


Denn in dem dieser Entscheidung zugrunde liegenden Fall hat sich die Einsprechende lediglich mit einem einzigen und zudem fakultativen Teilmerkmal des insgesamt vier Merkmale ( Verfahrensschritte ) umfassenden patentierten Epoxidationsverfahrens befasst , und keine weiteren Angaben gemacht , die für die Patentierungserfordernisse ( z.B. Neuheit oder erfinderische Tätigkeit ) der die gesamten bzw. nichtfakultativen Merkmale einschließenden Lehre des patentgemäßen Verfahrens von Bedeutung sein könnten .

**False Positives:**

- `B. Neuheit` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 29** (doc_id: `55974`) (sent_id: `55974`)


Teilweise beziehen sich diese Presseartikel auf Personen , die ( noch ) nicht verurteilt worden sind ( z.B. Untersuchungsgefangene ) , oder auf ein Absehen von Verfolgung insgesamt .

**False Positives:**

- `B. Untersuchungsgefangene` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

</details>

---

<details>
<summary>💣 Least Precise Rules</summary>

## `Names after 'von' (Noble/Preposition)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `030b825e`  
**Description:**
Captures names following the preposition 'von', excluding common non-name words and titles.

**Content:**
```
\bvon\s+([A-Z][a-zäöüß]+)(?!\s+(?:Frau|Herr|Dr\.?|Prof\.?|Anwalt|Gericht|Firma|Behörde|Polizei|Steuern|Umsatz|Kabel|Anlage|Umsatzsteuer|Rechtsanwältin|Rechtsanwalt|Ackerstatusrechten|Höchst-Schmerzgrenze|Marken-Beschwerdesenat|Rechtsanwalt|...|von|der|die|das|ein|eine|mit|für|auf|in|an|bei|zu|nach|vor|über|unter|durch|ohne|gegen|seit|statt|wegen|um|bis|neben|zwischen|entlang|außer|neben|ab))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 702 | 0 | 702 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 702 | 345 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53415`) (sent_id: `53415`)


Die Form der Benutzung ist insoweit jeweils glaubhaft gemacht durch die als Anlagen W14 bis W17 vorgelegten Produktkataloge aus dem vorliegend relevanten Benutzungszeitraum mit Abbildungen einer Vielzahl von ( elektrischen ) Kaffeemaschinen , Kaffeemühlen , Milchbehälter ( „ Milchcooler “ ) sowie Verpackungen / Behälter von Reinigungstabs , CreamCleaner und Entkalker , auf denen die Widerspruchsmarke NIVONA in einer ohne weitere besondere graphische Ausgestaltungselemente enthaltenden und damit in einer den kennzeichnenden Charakter der eingetragenen Marke nicht verändernden Form i. S. von § 26 Abs. 3 MarkenG deutlich sichtbar abgebildet ist .

**False Positives:**

- `Reinigungstab` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `NIVONA`(ORG)
- `§ 26 Abs. 3 MarkenG`(NRM)

**Example 1** (doc_id: `53416`) (sent_id: `53416`)


Zu dieser Zeit sei auch noch die Berücksichtigung von Zeiten wissenschaftlicher Tätigkeit streitig gewesen .

**False Positives:**

- `Zeite` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `53425`) (sent_id: `53425`)


Insbesondere hinsichtlich der Dienstleistungen „ Vermietung von Computersoftware ; Vermietung von Webservern ; Programmierung und Einstellung von Datenverarbeitungsprogrammen “ könne es sich um solche Dienstleistungen handeln , die für ein „ real Targeting “ bestimmt , geeignet und notwendig seien oder damit in unmittelbarem Zusammenhang stehen könnten .

**False Positives:**

- `Computersoftwar` — no gold match — likely missing annotation
- `Webserver` — no gold match — likely missing annotation
- `Datenverarbeitungsprogramme` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 3

**Example 3** (doc_id: `53428`) (sent_id: `53428`)


Hierzu zählt auch der Schutz vor der Erhebung und Weitergabe von Befunden über den Gesundheitszustand .

**False Positives:**

- `Befunde` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `53441`) (sent_id: `53441`)


Die Beteiligten streiten über die Erstattung bzw Übernahme von Kosten für die Entsorgung von Inkontinenzmaterial .

**False Positives:**

- `Koste` — no gold match — likely missing annotation
- `Inkontinenzmaterial` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 5** (doc_id: `53443`) (sent_id: `53443`)


Es hat einerseits angenommen , ein institutioneller Rechtsmissbrauch sei indiziert , da die Parteien seit dem 25. August 2008 insgesamt 22 befristete Arbeitsverträge abgeschlossen hätten und damit die Anzahl von Vertragsverlängerungen den in § 14 Abs. 2 Satz 1 TzBfG genannten Wert um mehr als das Fünffache überschreite .

**False Positives:**

- `Vertragsverlängerunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 14 Abs. 2 Satz 1 TzBfG`(NRM)

**Example 6** (doc_id: `53455`) (sent_id: `53455`)


c ) Die in Teilen der Literatur verbreitete Rechtsauffassung , wonach allein der Zufluss von Kindergeld auf dem Konto eines Zulageberechtigten genüge , einen Anspruch auf Kinderzulage nach § 85 Abs. 1 Satz 1 EStG a. F. zu erlangen ( so wohl Killat in Herrmann / Heuer / Raupach , § 85 EStG Rz 6 ; Myßen / Obermair , in : Kirchhof / Söhn / Mellinghoff , EStG , § 85 Rz D 18 ; Schmidt / Wacker , EStG , 36. Aufl. , § 85 Rz 2 ) , wäre hingegen missbrauchsanfällig und kann zu Ergebnissen führen , die mit dem Sinn und Zweck des § 85 Abs. 1 Satz 1 EStG a. F. nicht vereinbar wären .

**False Positives:**

- `Kindergel` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 85 Abs. 1 Satz 1 EStG a. F.`(NRM)
- `Killat in Herrmann / Heuer / Raupach , § 85 EStG Rz 6`(LIT)
- `Myßen / Obermair , in : Kirchhof / Söhn / Mellinghoff , EStG , § 85 Rz D 18`(LIT)
- `Schmidt / Wacker , EStG , 36. Aufl. , § 85 Rz 2`(LIT)
- `§ 85 Abs. 1 Satz 1 EStG a. F.`(NRM)

**Example 7** (doc_id: `53459`) (sent_id: `53459`)


Als Aufgaben sind in § 3 Nr. 5 Buchst. b der Satzung die „ Verbesserung von Einkommen und Arbeitsbedingungen durch Abschluss von Tarifverträgen und Einwirkung auf die Gesetzgebung und Behörden “ genannt .

**False Positives:**

- `Einkomme` — no gold match — likely missing annotation
- `Tarifverträge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 8** (doc_id: `53464`) (sent_id: `53464`)


Zum anderen steht der Annahme fehlender Unterscheidungskraft nicht entgegen , dass der Begriff „ ruheyoga “ bislang nicht lexikalisch nachweisbar ist ; ebenso wenig ist von Bedeutung , dass das Zusammenschreiben der Worte „ ruhe “ und „ yoga “ in einem Wort möglicherweise nicht üblich ist .

**False Positives:**

- `Bedeutun` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 9** (doc_id: `53482`) (sent_id: `53482`)


In Teil F Ziffer 3.5 ist bestimmt : " Die Partner der Gesamtverträge beschließen für Neuzulassungen von Vertragsärzten und Umwandlung der Kooperationsform Anfangs- und Übergangsregelungen .

**False Positives:**

- `Vertragsärzte` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 10** (doc_id: `53503`) (sent_id: `53503`)


Vor die ordentlichen Gerichte hingegen gehören nach § 13 GVG ua die bürgerlichen Rechtsstreitigkeiten , die Familiensachen und die Angelegenheiten der freiwilligen Gerichtsbarkeit ( Zivilsachen ) , für die nicht entweder die Zuständigkeit von Verwaltungsbehörden oder Verwaltungsgerichten begründet ist oder aufgrund von Vorschriften des Bundesrechts besondere Gerichte bestellt oder zugelassen sind .

**False Positives:**

- `Verwaltungsbehörde` — no gold match — likely missing annotation
- `Vorschrifte` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `§ 13 GVG`(NRM)

**Example 11** (doc_id: `53508`) (sent_id: `53508`)


Beruhen vielmehr die Differenzen zwischen den Auffassungen von Sachverständigen darauf , dass diese von verschiedenen tatsächlichen Annahmen ausgehen , dann muss der Tatrichter , ggf nach weiterer Aufklärung , die für seine Überzeugungsbildung maßgebenden Tatsachen feststellen oder begründen , weshalb und zu wessen Lasten sie beweislos geblieben sind ( vgl BGH Urteil vom 23. 9. 1986 - VI ZR 261/85 ) .

**False Positives:**

- `Sachverständige` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BGH Urteil vom 23. 9. 1986 - VI ZR 261/85`(RS)

**Example 12** (doc_id: `53512`) (sent_id: `53512`)


Das Landgericht hat den Angeklagten vom Vorwurf des Vorenthaltens und Veruntreuens von Arbeitsentgelt sowie der Steuerhinterziehung jeweils in 32 Fällen aus tatsächlichen Gründen freigesprochen .

**False Positives:**

- `Arbeitsentgel` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 13** (doc_id: `53514`) (sent_id: `53514`)


Auf die Revision der Klägerin wird das Urteil des Hessischen Finanzgerichts vom 24. März 2015 4 K 1187/11 aufgehoben , soweit es die Klage gegen die Nachforderung von Kapitalertragsteuer abweist ; die Nachforderungsbescheide vom 6. September 2010 über Kapitalertragsteuer für die Jahre 2005 und 2006 in Gestalt der Einspruchsentscheidung vom 8. April 2011 werden dahin geändert , dass die Kapitalertragsteuer auf 0 € festgesetzt wird .

**False Positives:**

- `Kapitalertragsteue` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Urteil des Hessischen Finanzgerichts vom 24. März 2015 4 K 1187/11`(RS)

**Example 14** (doc_id: `53523`) (sent_id: `53523`)


( Vertragsärztliche Versorgung - Ermächtigung von Sozialpädiatrischen Zentren - keine analoge Anwendung von § 118 Abs 4 SGB 5 oder § 24 Abs 3 Ärzte ZV bzgl Errichtung einer Zweigstelle bzw -praxis )

**False Positives:**

- `Sozialpädiatrische` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 118 Abs 4 SGB 5`(NRM)
- `§ 24 Abs 3 Ärzte ZV`(NRM)

**Example 15** (doc_id: `53524`) (sent_id: `53524`)


Der BFH führte aus , die handelsrechtliche Zurechnung von Vermögensgegenständen entspreche im Wesentlichen der Regelung des § 39 Abs. 2 Nr. 1 Satz 1 AO ( z.B. BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1. ; vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a ) .

**False Positives:**

- `Vermögensgegenstände` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BFH`(ORG)
- `§ 39 Abs. 2 Nr. 1 Satz 1 AO`(NRM)
- `BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1.`(RS)
- `vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a`(RS)

**Example 16** (doc_id: `53533`) (sent_id: `53533`)


Gemäß den Beispielen I-4 bis I-22 aus Tabelle III. die alle einer identischen ersten Wärmebehandlung für die Keimbildung mit 645 ° C / 1 h ausgesetzt waren , wurde je nach dem gewählten Temperaturprofil der zweiten Wärmebehandlung entweder nur Lithiummetasilicat , eine Mischung von Lithiummetasilicat und Lithiumdisilicat oder nur Lithiumdisilicat erhalten ( vgl. D3 , S. 385 , Tab. I , S. 386 Tab. III. .

**False Positives:**

- `Lithiummetasilica` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 17** (doc_id: `53535`) (sent_id: `53535`)


Das setzt einen Erfolg der Behandlungsmethode in einer für die sichere Beurteilung ausreichenden Zahl von Behandlungsfällen voraus .

**False Positives:**

- `Behandlungsfälle` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 18** (doc_id: `53559`) (sent_id: `53559`)


Amtliche Informationen kommen einem Eingriff in die Berufsfreiheit jedenfalls dann gleich , wenn sie direkt auf die Marktbedingungen konkret individualisierter Unternehmen zielen , indem sie die Grundlagen von Konsumentscheidungen zweckgerichtet beeinflussen und die Markt- und Wettbewerbssituation zum Nachteil der betroffenen Unternehmen verändern .

**False Positives:**

- `Konsumentscheidunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `53573`) (sent_id: `53573`)


Zum einen vermag die Rüge der Nichtbeachtung von Bundesrecht bei der Auslegung und Anwendung von Landesrecht die Zulassung der Grundsatzrevision nur dann zu begründen , wenn die Auslegung einer - gegenüber dem Landesrecht als korrigierender Maßstab angeführten - bundesrechtlichen Norm ihrerseits ungeklärte Fragen von grundsätzlicher Bedeutung aufwirft .

**False Positives:**

- `Bundesrech` — no gold match — likely missing annotation
- `Landesrech` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 20** (doc_id: `53605`) (sent_id: `53605`)


Klasse 35 : Einzelhandelsdienstleistungen in den Bereichen tragbare und mit der Hand gehaltene oder handbetätigte elektronische Geräte zur Übertragung , Speicherung , Verarbeitung , Aufzeichnung und Ansicht / Überprüfung von Texten , Bildern , Audios , Videos und Daten , auch über globale Computernetzwerke , drahtlose Netzwerke und elektronische Kommunikationsnetzwerke , Computer , Tablet-Computer , Lesegeräte für elektronische Bücher , Audio- und Videogeräte , elektronische persönliche Organisierer , persönliche digitale Assistenten und Geräte für globale Positionierungssysteme und elektronische und mechanische Teile und Zubehör dafür ; Computerhardware und -software , Monitore , Displays , Drähte , Kabel , Modems , Drucker , Diskettenlaufwerke , Adapter , Adapterkarten , Kabelverbinder / Kabelanschlüsse , steckbare Anschlüsse , elektrische Stromanschlüsse , Dockstationen und Laufwerke , Batterieladegeräte , Batteriepackungen , Memorykarten und Lesegeräte für Memorykarten , Kopfhörer und Ohrhörer , Lautsprecher , Mikrophone und Headsets ( Hörsprechgarnituren ) , angepasste Behälter , Abdeckungen und Gestelle für tragbare und mit der Hand gehaltene oder handbetätigte elektronische Geräte und Computer , Fernbedienungen für tragbare und mit der Hand gehaltene oder handbetätigte elektronische Geräte und Computer , Druckereierzeugnisse , gedruckte Veröffentlichungen , Periodika , Bücher , Magazine , Newsletter , Broschüren , Hefte , Pamphlete , Handbücher , Journale , Kataloge und Sticker , Hand gehaltene oder handbetätigte Anlagen zum Spielen von elektronischen Spielen , handgehaltene oder handbetätigte elektronische Spiele und Spielapparate , Spiele , elektronische Spiele und Videospiele ; Online-Einzelhandelsdienstleistungen in den Bereichen tragbare und mit der Hand gehaltene oder handbetätigte elektronische Geräte zur Übertragung , Speicherung , Verarbeitung , Aufzeichnung und Ansicht / Überprüfung von Texten , Bildern , Audios , Videos und Daten , auch über globale Computernetzwerke , drahtlose Netzwerke und elektronische Kommunikationsnetzwerke , Computer , Tablet-Computer , Lesegeräte für elektronische Bücher , Audio- und Videogeräte , elektronische persönliche Organisierer , persönliche digitale Assistenten und Geräte für globale Positionierungssysteme und elektronische und mechanische Teile und Zubehör dafür , Computerhardware und -software , Monitore , Displays , Drähte , Kabel , Modems , Drucker , Diskettenlaufwerke , Adapter , Adapterkarten , Kabelverbinder / Kabelanschlüsse , steckbare Anschlüsse , elektrische Stromanschlüsse , Dockstationen und Laufwerke , Batterieladegeräte , Batteriepackungen , Memorykarten und Lesegeräte für Memorykarten , Kopfhörer und Ohrhörer , Lautsprecher , Mikrophone und Headsets ( Hörsprechgarnituren ) , angepasste Behälter , Abdeckungen und Gestelle für tragbare und mit der Hand gehaltene oder handbetätigte elektronische Geräte und Computer , Fernbedienungen für tragbare und mit der Hand gehaltene oder handbetätigte elektronische Geräte und Computer , Druckereierzeugnisse , gedruckte Veröffentlichungen , Periodika , Bücher , Magazine , Newsletter , Broschüren , Hefte , Pamphlete , Handbücher , Journale , Kataloge und Sticker , Hand gehaltene oder handbetätigte Anlagen zum Spielen von elektronischen Spielen , handgehaltene oder handbetätigte elektronische Spiele und Spieleapparate , Spiele , elektronische Spiele und Videospiele ; Herausgabe eines Online-Handelsinformationsverzeichnisses ; Verteilung von Werbung für andere über ein elektronisches Online-Kommunikationsnetzwerk ; Herausgabe eines recherchierbaren Online Werbeführers , der Waren und Dienstleistungen von anderen zeigt ; computerisierte Datenbank-Managementdienstleistungen , nämlich Sammeln und Systematisieren von Daten in Computerdatenbanken ; Online-Bestelldienstleistungen , nämlich verwaltungstechnische Bearbeitung ;

**False Positives:**

- `Texte` — no gold match — likely missing annotation
- `Texte` — no gold match — likely missing annotation
- `Werbun` — no gold match — likely missing annotation
- `Date` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 4

**Example 21** (doc_id: `53608`) (sent_id: `53608`)


„ Klasse 9 : Wissenschaftliche , Schifffahrts- , Vermessungs- , fotografische , Film- , optische , Wäge- , Mess- , Signal- , Kontroll- , Rettungs- und Unterrichtsapparate und -instrumente ; Apparate und Instrumente zum Leiten , Schalten , Umwandeln , Speichern , Regeln und Kontrollieren von Elektrizität ; Geräte zur Aufzeichnung , Übertragung oder Wiedergabe von Sprache , Ton oder Bild ; Computerplatten , Disketten ; Bänder ; magnetische und optische Datenträger ; Magnetplatten , -disketten ; Magnetbänder ; Disketten ; Schallplatten und CDs ; Datenträger ; Speichermedien ; Speicherkarten ; Disketten , CDs , CD-ROMs und DVDs ; Flash-Aufbewahrungsvorrichtungen ; Festplatten ; Flash-basierte Festplatten ; Speicherkarten ; elektronische Massenspeichervorrichtungen ; Computerspeicher ; Verkaufsautomaten und Mechaniken für geldbetätigte Apparate ; Registrierkassen ; Rechenmaschinen ; Taschenrechner ; Datenverarbeitungsgeräte und Computer ; Datenerfassungshardware ; speicherprogrammierbare Steuerungen ; Datenverarbeitungsgeräte und Computer , insbesondere Einplatinen- oder Einschubgeräte ; elektronische Computerbauteile ; Computer-Hardware ; Computermodule ; Computer ; eingebettete Computer ; Einplatinencomputer ; Personal Computer ( PCs ) ; Kasten-PCs ; Desktop-PCs ; Wandmontage-PCs ; Laptops ; Notebooks ; Subnotebooks ; robuste Computer sowie deren Teile und Komponenten ; in Gehäuse montierte Computer ; in Gehäuse montierte Computersystemgeräte ; Industrie-PCs ; eingebettete PCs ; Schnittstellen Mensch-Maschine ( HMI ) ; Schnittstellengeräte Mensch-Maschine , bestehend aus einem Computer und einer Anzeige ; Pult-PCs ; Microclient-Computer ; Thin-Client-Computer ; interaktive Client-Computer ; Kiosk-PCs ; Spiel-PCs ; Touchpanels ; Tastfeld-PCs ; PCs in Fahrzeugen ; PCs in Kraftfahrzeugen ; Informations- und Unterhaltungs-PCs ; Telematik-PCs ; robuste Computer ; robuste Workstations ; robuste Notebooks ; tragbare PCs ; robuste Anzeigen ; Computer und Computersysteme für den Verkehr ; Zugverwaltungssysteme ; Computer und Computersysteme zur Zugverwaltung , Verkehrsregelung , industriellen Regelung , Maschinensteuerung , Eisenbahnsteuerung , digitalen Gesundheitspflege ; Computer und Anzeigen für medizinische Bildgebung , Patientenüberwachung , Informations- und Unterhaltungsgeräte , Passagierinformationssysteme ; Computer und Anzeigen für Öl und Gas enthaltende Umgebungen ; explosionsgeschützte Anzeigen ; Gateways ; Router ; automatische Prüfausrüstungen ; Luftverkehrssteuerungsausrüstungen ; Hochverfügbarkeitscomputer ; Kommunikations-Edge-Computer ; Kommunikationskerncomputer ; Computer für Verteidigungsanwendungen , Überwachungssysteme und Sicherheitssysteme ; integrierte PC-Platinen ; Elektronikplatinen ; Computerplatinen ; Computerschnittstellenkarten ; elektronische Leiterplatten ; Prozessorkarten ; Trägerkarten ; Hauptplatinen ; Faxmodemkarten für Computer ; elektronische Schnittstellenbauteile und elektronische Terminals für Speicher- , Grafik- , Bildausgabe- , Kommunikations- , Netz- und Datenerfassungsanwendungen ; CPUs ; LCD-Treiber ; LCD-TFT-Adapter ; Grafikadapter ; Treiber für Speichermedien ; Computerbildschirme und -monitore ; aktive Rückwandplatinen ; passive Rückwandplatinen ; Adapter für Computer ; Komponenten für und Teile von Computern ; Computerchassis ; Tastaturen , Mäuse , Joysticks , Bildschirme und alle anderen Peripheriegeräte für Computer ; Computerzubehör , nämlich Handgelenk- und Armauflagen , Bildschirmfilter , Mauspads ; Stromversorgung für Computer ; Tastatur-Monitor-Maus-Einheiten ( KVM-Einheiten ) ; Schalter ; industrielle Schalter / Hubs ; nicht verwaltete Schalter / Hubs ; verwaltete Schalter / Hubs ; Peripheriegeräte , Geräte , Instrumente , Zubehör und Ersatzteile für Computer ; Verbindungselemente , nämlich Reihenkoppler , Buchsen , Schaltbretter , Schaltkästen und andere Geräte ; Zeitsensoren ; Verkaufsterminals ; Lautsprecher ; Tonanlagen , nämlich Lautsprecher , Kopfhörer , Mikrofone , Empfänger , Aufzeichnungsgeräte und deren Bauteile ; Batterien ; leere Computermagnetbänder ; elektronische Planungshilfen / Organizer ; Fernsteuerungen für Computer ; Ferncursor für Computer ; Überspannungsschutz und Stromversorgungen ; elektrische Schalter ; Bänder ( Computerspiele ) ; elektronische Publikationen ; Halter für CDs ; IC-Karten ( Smartcards , Adapter und Lesegeräte für diese Karten ) ; Videokarten ; Soundkarten ; Bildverarbeitungskarten ; Bildabtasterkarten ; Grafikkarten ; Netzkarten ; Wechselsprechgeräte ; Mikrocomputer ; Minicomputer ; Mikrofone ; Computerspeicher ; Computerschnittstellen ; Datenverarbeitungsmaschinen / Computerspeichergeräte ; RAM-Schaltkreise ; Chips ; Halbleitergeräte ; integrierte Schaltkreise ; Mikroprozessoren ; elektronische Schaltkreise ; gedruckte Schaltungen ; Datenmodule ; Terminals ; Steuereinheit ; Stromrichter ; Steckverbinder ; Eingabe- und Ausgabegeräte ; Zähler ; Zeitgeber ; elektronische Test- und diagnostische Geräte ; computerbezogene Sicherheitsvorrichtungen für tragbare Computerprodukte ; Behälter und Koffer für Computer , Computerperipheriegeräte und Computerbedarf ; Tragebehältnisse für Computer ; Multiplexer ; Modems ; Datenkommunikationsterminals und Leitungsadapter ; Kommunikationsserver ; Leistungswandler , nämlich Digital-Analog-Wandler , Analog-Digital-Wandler , Vorrichtungen zum Regulieren der Spannungsstufe ; Kabel und Kabelteile ; Telekommunikationsausrüstungen ; Telefone und Telefonzubehör ; Telefonanlagen ; Fernkopiergeräte ; Tonbandgeräte ; Projektoren ; Fotoapparate ; Videokameras ; Web-Kameras ; Videospiele ; Videobildschirme ; Videorecorder ; Videobänder ; Videoanlagen ( Projektoren , Videokameras , Steuergeräte und Zubehör für Videoanlagen ) ; CD-Abspielgeräte ( Musik ) ; CDs mit Musik , Grafiken oder Computerprogrammen ; Funkrufgeräte ; Middelware-Software ; Feldbus- und industrielle Ethernet-Software ; Protokollkonvertierungssoftware und -hardware ; Software zur Anzeige ; Web-fähige Kommunikationssoftware ; Ferndiagnosesoftware ; Internet- und Intranet-Software ; Software für intelligente Peripheriemanagementschnittstellen ( IPMI ) ; Betriebssystemsoftware und Anwendersoftware für Ressourcenzuordnung , Planung , Eingabe- / Ausgabesteuerung , Datenverwaltung , Kommunikationsmanagement , Netzverwaltung , Umschreiber / Transcriber und Kombinationen von Datenverarbeitungsgeräten und Datenverarbeitungsprogrammen ; auf maschinenlesbaren Trägermedien gespeicherte Dokumentationen und Bedienungsanleitungen in Bezug auf Computer oder Computerprogramme ; Computer-Software ; Datenverarbeitungsprogramme ; Firmware ; BIOS-Software ; Computersoftware zur Verwendung mit einem globalen Computernetz ; Computersoftware für die Dokumentenverwaltung ; Computersoftware zur Verwendung für das Auffinden , Abfragen und Empfangen von Text , elektronischen Dokumenten , Grafiken und audiovisuellen Informationen in unternehmensweiten internen Computernetzen und lokalen Netzen , Fernnetzen und weltweiten Computernetzen ; Computersoftware für die Softwareentwicklung und die Erstellung von Websites ; Computer und Systeme zur Telekommunikation und Datenkommunikation ; offene Modularkommunikationsplattformen ; Systeme für drahtlosen Zugang ; Systeme für drahtlosen Edge und Kern ; Systeme für drahtgebundenen Zugang ; Systeme für Unternehmensanwendungen ; Systeme für Verkehrs- und Datenzentrumsinfrastruktur ; Basis-Sende-Empfangs-Stationen ( BTS ) ; Basisstationssteuerungen ( BSC ) ; Kommunikationsknoten ; Funknetzüberwacher ( RNC ) ; Funkvermittlungsstellen ( MSC ) ; Medien-Gateways ; Medien-Gateway-Steuerungen ( MGC ) ; Serving ( General Packet Radio Service ) GPRS Support Nodes ( SGSN ) ; Gateway GPRS Support Nodes ( GGSN ) ; Signalserver für Telekommunikationsanwendungen ; IP-Multimedia-Subsysteme ( IMS ) ; DSL-Zugangsmultiplexer ( DSLAM ) ; Schalter / Hubs ; Router ; Verkehrsfilter- / Sicherheitsvorrichtungen ; Traffic Policing- / Shaping-Vorrichtungen ; Shelf-Management-Controller ( ShMC ) ; intelligente Peripheriemanagementschnittstelle ( IPMI ) ; advancedTCA-Systeme ( Advanced Telecom Computer Architecture ) ; advancedMC-Module ( Advanced Mezzanine Card ) ; microTCA-Systeme ( Micro Telecom Computer Architecture ) ; ETSI ( Europäisches Institut für Telekommunikationsstandards ) und NEBS ( Network Equipment-Building System ) entsprechende Systeme ; Hochverfügbarkeitssysteme ; Telekommunikationsausrüstungen .

**False Positives:**

- `Elektrizitä` — no gold match — likely missing annotation
- `Sprach` — no gold match — likely missing annotation
- `Computer` — no gold match — likely missing annotation
- `Datenverarbeitungsgeräte` — no gold match — likely missing annotation
- `Tex` — no gold match — likely missing annotation
- `Website` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 6

**Gold Entities:**

- `ETSI`(ORG)
- `Europäisches Institut für Telekommunikationsstandards`(ORG)

**Example 22** (doc_id: `53616`) (sent_id: `53616`)


Wie sich aus dem Zusammenhang mit Merkmal 7.2 ergibt , dient der Satz von Befehlen dazu , die Nutzung des vom Telekommunikationsnetz angebotenen Dienstes zu ermöglichen , indem Informationen über den Dienst angezeigt werden können und mithilfe der Eingabemittel eine Auswahl bezüglich des Dienstes getroffen werden kann .

**False Positives:**

- `Befehle` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 23** (doc_id: `53632`) (sent_id: `53632`)


Denn solche eidesstattlichen Versicherungen sind von Hause aus auf das mehr oder weniger vollständige Erinnerungsvermögen des Erklärenden und die Noch-Verfügbarkeit diesbezüglicher Unterlagen angewiesen ( vgl. Ekey / Bender / Fuchs-Wissemann , a. a. O. , § 43 , Rdnr. 48 ) .

**False Positives:**

- `Haus` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Ekey / Bender / Fuchs-Wissemann , a. a. O. , § 43 , Rdnr. 48`(LIT)

**Example 24** (doc_id: `53664`) (sent_id: `53664`)


Soweit das Bundesministerium der Verteidigung darauf verweist , dass Personalgespräche dazu dienten , Soldaten über Verwendungsplanungen der Personalführung zu informieren und in diesem Zusammenhang deren persönliche Motivationslage zu ergründen , um das dienstliche Bedürfnis und die privaten Lebensumstände des Soldaten möglichst gut miteinander zu vereinbaren , trifft dies zweifellos auf Personalgespräche im Vorfeld von Personalmaßnahmen zu .

**False Positives:**

- `Personalmaßnahme` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Bundesministerium der Verteidigung`(ORG)

**Example 25** (doc_id: `53667`) (sent_id: `53667`)


Selbst wenn der Status des Lebenszeitrichters von Verfassungs wegen als Regelstatus der Berufsrichter verbindlich sein sollte , wäre ein Einsatz von Richtern auf Zeit in Ausnahmefällen , wie ihn § 18 VwGO vorsieht , verfassungsrechtlich unbedenklich ( 1. ) .

**False Positives:**

- `Verfassung` — no gold match — likely missing annotation
- `Richter` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `§ 18 VwGO`(NRM)

**Example 26** (doc_id: `53670`) (sent_id: `53670`)


Diese entscheiden , ob sie die KÄVen in die Durchführung von Impfungen einbeziehen .

**False Positives:**

- `Impfunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 27** (doc_id: `53689`) (sent_id: `53689`)


Die beanspruchte Verwendung eines Lithiumsilicatrohlings gemäß Patentanspruch 1 des Hilfsantrags 8 sei ebenfalls gegenüber D2 unter Berücksichtigung der Nacharbeitung D21 -A nicht neu , da auch in Beispiel 22 der D2 eine maschinelle Bearbeitung von Lithiummetasilicat vorgesehen sei .

**False Positives:**

- `Lithiummetasilica` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 28** (doc_id: `53711`) (sent_id: `53711`)


Ausgehend von zumindest durchschnittlicher Kennzeichnungskraft der Widerspruchsmarke , Dienstleistungsidentität bzw. hochgradiger Dienstleistungsähnlichkeit und einer nach dem Gesamteindruck der Marken bestehenden hohen Zeichenähnlichkeit , sei die Gefahr von Verwechslungen im Sinne von § 9 Abs. 1 Nr. 1 MarkenG gegeben .

**False Positives:**

- `Verwechslunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 9 Abs. 1 Nr. 1 MarkenG`(NRM)

**Example 29** (doc_id: `53715`) (sent_id: `53715`)


aa ) Obwohl die Parteien in Klausel 33 des Vertriebsvertrags die Geltung des Rechts der USA und des Staates Kalifornien vereinbart haben , ist das FG den deutschen Grundsätzen über die Auslegung von Willenserklärungen und Verträgen gefolgt .

**False Positives:**

- `Willenserklärunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `USA`(LOC)
- `Kalifornien`(LOC)

</details>

---

## `Single Letter Anonymized Names`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `c2457fe9`  
**Description:**
Captures single capital letters used as anonymized names in legal contexts (e.g., 'N', 'C', 'V', 'B').

**Content:**
```
\b([A-Z])\b(?!\s*(?:von|der|die|das|ein|eine|mit|für|auf|in|an|bei|zu|nach|vor|über|unter|durch|ohne|gegen|seit|statt|wegen|um|bis|neben|zwischen|entlang|außer|neben|ab|Gericht|Firma|Behörde|Polizei|Steuern|Umsatz|Kabel|Anlage|Umsatzsteuer|Rechtsanwältin|Rechtsanwalt|Ackerstatusrechten|Höchst-Schmerzgrenze|Marken-Beschwerdesenat|Rechtsanwalt|...|Frau|Herr|Dr\.?|Prof\.?|Anwalt|Gericht|Firma|Behörde|Polizei|Steuern|Umsatz|Kabel|Anlage|Umsatzsteuer|Rechtsanwältin|Rechtsanwalt|Ackerstatusrechten|Höchst-Schmerzgrenze|Marken-Beschwerdesenat|Rechtsanwalt|...))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 17 | 0 | 17 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 17 | 338 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53659`) (sent_id: `53659`)


Auch hieraus ergebe sich eine Verletzung des § 39 SGB X.

**False Positives:**

- `X` — partial — pred is substring of gold: `§ 39 SGB X.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 39 SGB X.`(NRM)

**Example 1** (doc_id: `55361`) (sent_id: `55361`)


Daraufhin beantragte der Beklagte die Einleitung eines Schiedsverfahrens nach § 73b Abs 4a S 1 SGB V .

**False Positives:**

- `V` — partial — pred is substring of gold: `§ 73b Abs 4a S 1 SGB V`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 73b Abs 4a S 1 SGB V`(NRM)

**Example 2** (doc_id: `55569`) (sent_id: `55569`)


I. Die Klägerin und Beschwerdeführerin ( Klägerin ) ist umsatzsteuerrechtlich Organgesellschaft des Organträgers N .

**False Positives:**

- `N` — type mismatch — same span as gold: `N`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `N`(ORG)

**Example 3** (doc_id: `56127`) (sent_id: `56127`)


§ 29a Abs. 3 Satz 2 TVÜ-Länder verweist im Klammerzusatz dabei ausdrücklich auf § 17 Abs. 4 TV-L .

**False Positives:**

- `L` — similar text (different position): `§ 29a Abs. 3 Satz 2 TVÜ-Länder`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 29a Abs. 3 Satz 2 TVÜ-Länder`(REG)
- `§ 17 Abs. 4 TV-L`(REG)

**Example 4** (doc_id: `56397`) (sent_id: `56397`)


Vertretungsberechtigt für die Holding-KG ist die C-GmbH , vertreten durch ihren Geschäftsführer D.

**False Positives:**

- `D` — partial — pred is substring of gold: `D.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C-GmbH`(ORG)
- `D.`(PER)

**Example 5** (doc_id: `56674`) (sent_id: `56674`)


I

**False Positives:**

- `I` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 6** (doc_id: `56709`) (sent_id: `56709`)


III. Die Klägerin hat über den 31. Juli 2014 hinaus Anspruch auf Vergütung nach der Entgeltgruppe 10 Stufe 5 TV-L .

**False Positives:**

- `L` — partial — pred is substring of gold: `TV-L`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `TV-L`(REG)

**Example 7** (doc_id: `56968`) (sent_id: `56968`)


Ausschnitt aus dem Signallaufplan auf Seite 30 , Abschnitt „ Restart interlock , Start-up testing “ , Kanal A

**False Positives:**

- `A` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `57256`) (sent_id: `57256`)


Vergütungsgruppe V b

**False Positives:**

- `V` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 9** (doc_id: `57914`) (sent_id: `57914`)


Der Geburtsort seiner jetzigen Ehefrau ist M.

**False Positives:**

- `M` — partial — pred is substring of gold: `M.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `M.`(LOC)

**Example 10** (doc_id: `58157`) (sent_id: `58157`)


Denn die Klägerin und die Holding-KG werden durch dieselbe Person vertreten , nämlich D.

**False Positives:**

- `D` — partial — pred is substring of gold: `D.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `D.`(PER)

**Example 11** (doc_id: `58235`) (sent_id: `58235`)


Die Klägerin ist Eigentümerin eines insgesamt 5471 m² großen Grundstücks in der Gemarkung S.

**False Positives:**

- `S` — partial — pred is substring of gold: `S.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S.`(LOC)

**Example 12** (doc_id: `58377`) (sent_id: `58377`)


Das FG ordnete mit Beschluss vom 21. April 2016 die Einholung eines Sachverständigengutachtens zu den Verkehrswerten des Gebäudes und des Grund und Bodens an und beauftragte damit den Gutachterausschuss für Grundstückswerte in der Stadt Z .

**False Positives:**

- `Z` — type mismatch — same span as gold: `Z`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Z`(LOC)

**Example 13** (doc_id: `58526`) (sent_id: `58526`)


Dies führte nach der Anlage 2 zum TVÜ-Länder iVm. § 17 Abs. 1 , § 39 Abs. 1 Angleichungs-TV Land Berlin mit Wirkung zum 1. November 2010 zu einer Überleitung in die Entgeltgruppe 10 TV-L .

**False Positives:**

- `L` — similar text (different position): `Anlage 2 zum TVÜ-Länder`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Anlage 2 zum TVÜ-Länder`(REG)
- `§ 17 Abs. 1 , § 39 Abs. 1 Angleichungs-TV Land Berlin`(REG)
- `TV-L`(REG)

**Example 14** (doc_id: `59003`) (sent_id: `59003`)


Ferner verstießen die Festsetzungen des HzV-Vertrages gegen das Gebot der Selbsttragung eines Wahltarifs nach § 53 Abs 9 SGB V .

**False Positives:**

- `V` — similar text (different position): `HzV-Vertrages`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `HzV-Vertrages`(REG)
- `§ 53 Abs 9 SGB V`(NRM)

**Example 15** (doc_id: `59098`) (sent_id: `59098`)


Der Antragsteller wendet sich gegen seine Versetzung vom ... in L. zum ... in C.

**False Positives:**

- `C` — partial — pred is substring of gold: `C.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `L.`(LOC)
- `C.`(LOC)

**Example 16** (doc_id: `59857`) (sent_id: `59857`)


Am 4. 2. 2010 beantragten die Kläger die Überprüfung " sämtlicher Bescheide den Zeitraum 1. 4. 2008 bis 30. 9. 2009 betreffend " nach § 44 SGB X .

**False Positives:**

- `X` — partial — pred is substring of gold: `§ 44 SGB X`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 44 SGB X`(NRM)

</details>

---

</details>

---

<details>
<summary>🔇 Inactive Rules</summary>

## `Hyphenated Surnames`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `acae7cb4`  
**Description:**
Captures hyphenated surnames (e.g., 'Sost-Scheible', 'Meier-Beck') only when preceded by a title or legal role.

**Content:**
```
(?:Dr\.?\s+|Prof\.?\s+|Dipl\.-Ing\.\s+|Dipl\.-Psych\.\s+|Richter\s+|Anwalt\s+|Rechtsanwältin\s+|Rechtsanwalt\s+|Angeklagten|Kläger|Zeugen|Richter|Geschädigten|Vorsitzenden|Ministerpräsidenten|Herrn|Frau)\s+([A-Z][a-zäöüß]+-[A-Z][a-zäöüß]+)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Multi-Initial Anonymized Names`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `ebd05cdc`  
**Description:**
Captures anonymized names with multiple initials (e.g., 'M. D.', 'A. A.', 'P. W.').

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

</details>

---

<details>
<summary>📋 All Rules</summary>

## `Standalone Surname`

**F1:** 0.259 | **Precision:** 0.855 | **Recall:** 0.152  

**Format:** `regex`  
**Rule ID:** `b6b7238a`  
**Description:**
Captures verified surnames that appear without titles, excluding single letters and common words.

**Content:**
```
\b(?:Franke|Haberkamp|Spelge|Schlünder|Hoch|Roloff|Sieberts|Grube|Suckow|Schneider|Roggenbuck|Sibylle|Spoo|Appl|Bredendiek|Bacher|Wemheuer|Mutzbauer|Koch|Pape|Winzenried|Pielenz|Raum|Brühler|Bender|Lohmann|Remmert|Kley|Schultz|Kleinschmidt|Kirschneck|Matter|Kapels|Reinfelder|Brown|Schäfer|Brückner|Volz|Knoll|Kriener|Nielsen|Mayen|Seiters|Busch|Linck|Leitz|Hamdorf|Fiamingo|Spaniol|Kirchhoff|Gericke|Fritz|Vogelsang|Zwanziger|Kelvin|Lauer|Zeng|Tiemann|Sander|Fischermeier|Çerikci|Kaya|Seyhan|Lorsbach|Maksymiw|Schell|Münzberg|Jäger|Peter|Eschelbach|Kortbein|Schmid|Söchtig|Hacker|Merzbach|Meiser|Ts|Arnoldi|Haupt|Niemann|Becker|Waskow|Eylert|Marx|Fischer|Stresemann|Heinkel|Hayen|Volk|Liebert|Matthias|Kayser|Klein|Maekawa|Bar|Refaeli|Josh|Duhamel|Saime|Özcan|Boolell|McMillan|Rennpferdt|Wollny|Albertshofer|Dorn|Musiol|Kuemmerle|Drüen|Klaus-Dieter|Quentin|Spinner|Schlewing|Schmidt|Limperg|Merkel|Bormann|Berg|Demir|Baykara|Shah|Bu|W|G|F|E|A|S|N|D|K|R|V|Y|I|X|H|C|L|M|O|Z|B1|G1|J1|S1|K1|N1|P1|T1|V1|Y1|A1|D1|E1|F1|G1|H1|I1|J1|K1|L1|M1|N1|O1|P1|Q1|R1|S1|T1|U1|V1|W1|X1|Y1|Z1)\b(?!\s*(?:von|der|die|das|ein|eine|mit|für|auf|in|an|bei|zu|nach|vor|über|unter|durch|ohne|gegen|seit|statt|wegen|um|bis|neben|zwischen|entlang|außer|neben|ab|Gericht|Firma|Behörde|Polizei|Steuern|Umsatz|Kabel|Anlage|Umsatzsteuer|Rechtsanwältin|Rechtsanwalt|Ackerstatusrechten|Höchst-Schmerzgrenze|Marken-Beschwerdesenat|Rechtsanwalt|...|Frau|Herr|Dr\.?|Prof\.?|Anwalt|Gericht|Firma|Behörde|Polizei|Steuern|Umsatz|Kabel|Anlage|Umsatzsteuer|Rechtsanwältin|Rechtsanwalt|Ackerstatusrechten|Höchst-Schmerzgrenze|Marken-Beschwerdesenat|Rechtsanwalt|...))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.855 | 0.152 | 0.259 | 62 | 53 | 9 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 53 | 9 | 293 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53409`) (sent_id: `53409`)


Bender

| Predicted | Gold |
|---|---|
| `Bender` | `Bender` |

**Example 1** (doc_id: `53565`) (sent_id: `53565`)


Koch

| Predicted | Gold |
|---|---|
| `Koch` | `Koch` |

**Example 2** (doc_id: `53669`) (sent_id: `53669`)


Fritz

| Predicted | Gold |
|---|---|
| `Fritz` | `Fritz` |

**Example 3** (doc_id: `54009`) (sent_id: `54009`)


Schäfer

| Predicted | Gold |
|---|---|
| `Schäfer` | `Schäfer` |

**Example 4** (doc_id: `54032`) (sent_id: `54032`)


Hayen

| Predicted | Gold |
|---|---|
| `Hayen` | `Hayen` |

**Example 5** (doc_id: `54290`) (sent_id: `54290`)


Jäger

| Predicted | Gold |
|---|---|
| `Jäger` | `Jäger` |

**Example 6** (doc_id: `54389`) (sent_id: `54389`)


Merkel

| Predicted | Gold |
|---|---|
| `Merkel` | `Merkel` |

**Example 7** (doc_id: `54416`) (sent_id: `54416`)


Eylert

| Predicted | Gold |
|---|---|
| `Eylert` | `Eylert` |

**Example 8** (doc_id: `54477`) (sent_id: `54477`)


Schäfer

| Predicted | Gold |
|---|---|
| `Schäfer` | `Schäfer` |

**Example 9** (doc_id: `54597`) (sent_id: `54597`)


Kayser

| Predicted | Gold |
|---|---|
| `Kayser` | `Kayser` |

**Example 10** (doc_id: `54825`) (sent_id: `54825`)


Spinner

| Predicted | Gold |
|---|---|
| `Spinner` | `Spinner` |

**Example 11** (doc_id: `55048`) (sent_id: `55048`)


Stresemann

| Predicted | Gold |
|---|---|
| `Stresemann` | `Stresemann` |

**Example 12** (doc_id: `55052`) (sent_id: `55052`)


Fischer

| Predicted | Gold |
|---|---|
| `Fischer` | `Fischer` |

**Example 13** (doc_id: `55131`) (sent_id: `55131`)


Becker

| Predicted | Gold |
|---|---|
| `Becker` | `Becker` |

**Example 14** (doc_id: `55244`) (sent_id: `55244`)


Franke

| Predicted | Gold |
|---|---|
| `Franke` | `Franke` |

**Example 15** (doc_id: `55256`) (sent_id: `55256`)


Becker

| Predicted | Gold |
|---|---|
| `Becker` | `Becker` |

**Example 16** (doc_id: `55379`) (sent_id: `55379`)


Raum

| Predicted | Gold |
|---|---|
| `Raum` | `Raum` |

**Example 17** (doc_id: `55493`) (sent_id: `55493`)


Heinkel

| Predicted | Gold |
|---|---|
| `Heinkel` | `Heinkel` |

**Example 18** (doc_id: `55527`) (sent_id: `55527`)


Schmidt

| Predicted | Gold |
|---|---|
| `Schmidt` | `Schmidt` |

**Example 19** (doc_id: `55530`) (sent_id: `55530`)


Quentin

| Predicted | Gold |
|---|---|
| `Quentin` | `Quentin` |

**Example 20** (doc_id: `55897`) (sent_id: `55897`)


Seiters

| Predicted | Gold |
|---|---|
| `Seiters` | `Seiters` |

**Example 21** (doc_id: `55926`) (sent_id: `55926`)


Suckow

| Predicted | Gold |
|---|---|
| `Suckow` | `Suckow` |

**Example 22** (doc_id: `56611`) (sent_id: `56611`)


Grube

| Predicted | Gold |
|---|---|
| `Grube` | `Grube` |

**Example 23** (doc_id: `56892`) (sent_id: `56892`)


Franke

| Predicted | Gold |
|---|---|
| `Franke` | `Franke` |

**Example 24** (doc_id: `57002`) (sent_id: `57002`)


Fritz

| Predicted | Gold |
|---|---|
| `Fritz` | `Fritz` |

**Example 25** (doc_id: `57176`) (sent_id: `57176`)


Zwanziger

| Predicted | Gold |
|---|---|
| `Zwanziger` | `Zwanziger` |

**Example 26** (doc_id: `57471`) (sent_id: `57471`)


Jäger

| Predicted | Gold |
|---|---|
| `Jäger` | `Jäger` |

**Example 27** (doc_id: `57555`) (sent_id: `57555`)


Fischer

| Predicted | Gold |
|---|---|
| `Fischer` | `Fischer` |

**Example 28** (doc_id: `57630`) (sent_id: `57630`)


Fritz

| Predicted | Gold |
|---|---|
| `Fritz` | `Fritz` |

**Example 29** (doc_id: `57727`) (sent_id: `57727`)


Roggenbuck

| Predicted | Gold |
|---|---|
| `Roggenbuck` | `Roggenbuck` |

**Example 30** (doc_id: `57951`) (sent_id: `57951`)


Franke

| Predicted | Gold |
|---|---|
| `Franke` | `Franke` |

**Example 31** (doc_id: `58035`) (sent_id: `58035`)


Raum

| Predicted | Gold |
|---|---|
| `Raum` | `Raum` |

**Example 32** (doc_id: `58081`) (sent_id: `58081`)


Raum

| Predicted | Gold |
|---|---|
| `Raum` | `Raum` |

**Example 33** (doc_id: `58082`) (sent_id: `58082`)


Hayen

| Predicted | Gold |
|---|---|
| `Hayen` | `Hayen` |

**Example 34** (doc_id: `58084`) (sent_id: `58084`)


Tiemann

| Predicted | Gold |
|---|---|
| `Tiemann` | `Tiemann` |

**Example 35** (doc_id: `58257`) (sent_id: `58257`)


Schmidt

| Predicted | Gold |
|---|---|
| `Schmidt` | `Schmidt` |

**Example 36** (doc_id: `58280`) (sent_id: `58280`)


Busch

| Predicted | Gold |
|---|---|
| `Busch` | `Busch` |

**Example 37** (doc_id: `58325`) (sent_id: `58325`)


Hayen

| Predicted | Gold |
|---|---|
| `Hayen` | `Hayen` |

**Example 38** (doc_id: `58419`) (sent_id: `58419`)


Merkel

| Predicted | Gold |
|---|---|
| `Merkel` | `Merkel` |

**Example 39** (doc_id: `58453`) (sent_id: `58453`)


Roggenbuck

| Predicted | Gold |
|---|---|
| `Roggenbuck` | `Roggenbuck` |

**Example 40** (doc_id: `58498`) (sent_id: `58498`)


Becker

| Predicted | Gold |
|---|---|
| `Becker` | `Becker` |

**Example 41** (doc_id: `58516`) (sent_id: `58516`)


Koch

| Predicted | Gold |
|---|---|
| `Koch` | `Koch` |

**Example 42** (doc_id: `58615`) (sent_id: `58615`)


Eschelbach

| Predicted | Gold |
|---|---|
| `Eschelbach` | `Eschelbach` |

**Example 43** (doc_id: `58686`) (sent_id: `58686`)


Grube

| Predicted | Gold |
|---|---|
| `Grube` | `Grube` |

**Example 44** (doc_id: `58861`) (sent_id: `58861`)


Vogelsang

| Predicted | Gold |
|---|---|
| `Vogelsang` | `Vogelsang` |

**Example 45** (doc_id: `58901`) (sent_id: `58901`)


Fischer

| Predicted | Gold |
|---|---|
| `Fischer` | `Fischer` |

**Example 46** (doc_id: `59158`) (sent_id: `59158`)


Raum

| Predicted | Gold |
|---|---|
| `Raum` | `Raum` |

**Example 47** (doc_id: `59305`) (sent_id: `59305`)


Schlünder

| Predicted | Gold |
|---|---|
| `Schlünder` | `Schlünder` |

**Example 48** (doc_id: `59537`) (sent_id: `59537`)


Pape

| Predicted | Gold |
|---|---|
| `Pape` | `Pape` |

**Example 49** (doc_id: `59657`) (sent_id: `59657`)


Hayen

| Predicted | Gold |
|---|---|
| `Hayen` | `Hayen` |

**Example 50** (doc_id: `59864`) (sent_id: `59864`)


Pape

| Predicted | Gold |
|---|---|
| `Pape` | `Pape` |

**Example 51** (doc_id: `59949`) (sent_id: `59949`)


Schmidt

| Predicted | Gold |
|---|---|
| `Schmidt` | `Schmidt` |

**Example 52** (doc_id: `59958`) (sent_id: `59958`)


Eschelbach

| Predicted | Gold |
|---|---|
| `Eschelbach` | `Eschelbach` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53641`) (sent_id: `53641`)


D5 EP 0 160 797 A1

**False Positives:**

- `A1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `56422`) (sent_id: `56422`)


E 9 EP 1 308 030 B1 ,

**False Positives:**

- `B1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `56560`) (sent_id: `56560`)


D18 DE 10 2009 044 546 A1

**False Positives:**

- `A1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 3** (doc_id: `56599`) (sent_id: `56599`)


E4 DE 39 30 353 A1 ,

**False Positives:**

- `A1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `56890`) (sent_id: `56890`)


D1 DE 10 2004 031 624 A1

**False Positives:**

- `A1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `57356`) (sent_id: `57356`)


D11 DE 100 34 354 A1

**False Positives:**

- `A1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 6** (doc_id: `57452`) (sent_id: `57452`)


HLNK28 WO 96/38131 A1

**False Positives:**

- `A1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 7** (doc_id: `58333`) (sent_id: `58333`)


D2 DE 199 52 004 A1 ;

**False Positives:**

- `A1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `58888`) (sent_id: `58888`)


NIK5 / NiK4 WO 97/03675 A1

**False Positives:**

- `A1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Anonymized Initials with Dots`

**F1:** 0.173 | **Precision:** 0.971 | **Recall:** 0.095  

**Format:** `regex`  
**Rule ID:** `70340c5f`  
**Description:**
Captures anonymized names consisting of a capital letter followed by a dot (e.g., 'M.', 'F.', 'K.'), often preceded by role indicators.

**Content:**
```
\b(?:Angeklagte|Angeklagten|Kläger|Klägerin|Beklagte|Beklagten|Zeuge|Zeugin|Vorsitzender|Vorsitzende|Richter|Richterin|Geschädigter|Geschädigte|Ministerpräsident|Ministerpräsidentin|Herr|Frau|Dr\.?|Prof\.?|Patentanwalt|Rechtsanwalt|Sachverständiger|Sachverständige)\s+([A-Z]\.)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.971 | 0.095 | 0.173 | 34 | 33 | 1 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 33 | 1 | 310 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53522`) (sent_id: `53522`)


Den Fällen komme gleichwohl eigenständige Bedeutung zu , weil sich für das gleichfalls - jeweils in nicht geringer Menge - gehandelte bzw. zum Handeltreiben vorgesehene Marihuana ein einheitlicher Erwerbsvorgang nicht feststellen lasse , so dass insoweit jeweils eine eigenständige Strafbarkeit des Angeklagten H. nach § 29a Abs. 1 Nr. 2 BtMG begründet werde .

| Predicted | Gold |
|---|---|
| `H.` | `H.` |

**Missed by this rule (FN):**

- `§ 29a Abs. 1 Nr. 2 BtMG` (NRM)

**Example 1** (doc_id: `53890`) (sent_id: `53890`)


Im Berufungsverfahren hat der zuständige Berichterstatter des LSG den Sachverständigen Prof. Dr. T. im Termin zur mündlichen Verhandlung ergänzend angehört .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |

**Example 2** (doc_id: `54671`) (sent_id: `54671`)


Sie hat für Dr. T. und für Dr. L. jeweils RLV berechnet und deren Summe der Klägerin als RLV zugewiesen .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |
| `L.` | `L.` |

**Example 3** (doc_id: `54771`) (sent_id: `54771`)


Der Angeklagte A. entfernte sich - ebenso wie seine Tatgenossen - von der Unfallstelle , ohne zuvor dem Zeugen K. gegenüber Angaben zu seiner Person und der Art der Unfallbeteiligung gemacht zu haben .

| Predicted | Gold |
|---|---|
| `A.` | `A.` |

**Missed by this rule (FN):**

- `K.` (PER)

**Example 4** (doc_id: `54995`) (sent_id: `54995`)


Das LSG hat vielmehr im Anschluss an die Begründung , warum es dessen sachverständige Bewertung für überzeugend hält , ausgeführt : " Hingegen hat Dr. H. lediglich auf einer Seite kurz dargestellt , dass beim Kläger seinerzeit kein KIG Grad 3 oder höher vorliege .

| Predicted | Gold |
|---|---|
| `H.` | `H.` |

**Example 5** (doc_id: `55088`) (sent_id: `55088`)


Zwar hat der Senat durchaus zur Kenntnis genommen , dass die Klägerin in dem von ihr benannten Schriftsatz vom 11. 2. 2016 - auf die Anhörung des LSG zur Entscheidung durch Beschluss - durch die Bezugnahme auf den Schriftsatz vom 7. 3. 2016 ihren Antrag bei Prof. Dr. T. nachzufragen , in welcher Form und in welchem Umfang er an der Erstellung des Sachverständigengutachtens beteiligt gewesen sei , wiederholt hat .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |

**Example 6** (doc_id: `55218`) (sent_id: `55218`)


Mit Schreiben vom 8. 7. 2009 ( eingegangen am 21. 10. 2009 ) stellte Dr. T. für die Quartale ab I / 2009 einen Antrag auf Erhöhung der Fallzahl - Anpassung des RLV wegen Jungpraxis - und beantragte die Übernahme der Fallzahl der von ihm vor seiner Anstellung im MVZ betriebenen Praxis .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |

**Example 7** (doc_id: `56131`) (sent_id: `56131`)


Der Zeuge K. hat diesen Tatentschluss umgesetzt und mit der erfolgreichen Anwerbung des Zeugen S. , der sich zum Verkauf von Cannabis des Angeklagten auf Kommissionsbasis bereiterklärte , den Handel des Angeklagten mit Betäubungsmitteln gefördert .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `S.` (PER)

**Example 8** (doc_id: `56284`) (sent_id: `56284`)


3. Der Senat setzt den Wert des Gegenstands der anwaltlichen Tätigkeit des Antragstellers zur Verteidigung des Angeklagten K. gegen die beantragte Feststellung nach § 111i Abs. 2 StPO aF antragsgemäß auf 2.006.713,43 € fest .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `§ 111i Abs. 2 StPO aF` (NRM)

**Example 9** (doc_id: `56427`) (sent_id: `56427`)


3 ) Die Einwände des Klägers gegen die Feststellung des LSG , dass alle tatsächlichen Voraussetzungen für die Eintragung der Elektronicon-GmbH gegeben waren , insbesondere dass die Eintragung auf einer richterlichen Verfügung beruhte und im Original von der Zeugin S. unterschrieben worden war , greifen nicht durch .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Missed by this rule (FN):**

- `Elektronicon-GmbH` (ORG)

**Example 10** (doc_id: `56444`) (sent_id: `56444`)


So wird ausdrücklich darauf verwiesen , dass auf der Anlage K 1. Ausdruck einer Mail von D. an D. B. vom 30. Januar 2011 ) handschriftlich " Rechnungsanschrift Help Food z. o. o. D. B. ( es folgt die postalische Anschrift der Help Food ) " vermerkt ist , auf Seite 2 der Anlage K 3 ( Präsentationsunterlage mit dem Copyright von D. und W. ) unter " Unsere Kontraktbedingungen " ein " Exklusiver Kontrakt für 2 Jahre mit Help Food " und eine " Haushaltsverfügung durch Help Food ... bis zum Ende 2011 Startphase " erwähnt werden , auf Seite 2 der Anlage K 5 ( mit dem Logo der Klägerin versehenes Protokoll eines Treffens der Beteiligten am 26. August 2011 ) von einem " Vorschlag zum Vertrag zwischen Help Food , M. D. und P. W. " die Rede ist , die Anlage K 9 ( von S. unterzeichnetes Schreiben vom 29. Dezember 2011 ) als Absender die Help Food ausweist und die Anlage K 50 ( Ausdruck einer Mail der Zeugin F. an D. und W. vom 14. September 2011 ) die Absenderadresse " m. @helpfood . eu " trägt .

| Predicted | Gold |
|---|---|
| `F.` | `F.` |

**Missed by this rule (FN):**

- `D.` (PER)
- `D. B.` (PER)
- `Help Food` (ORG)
- `Help Food` (ORG)
- `D.` (PER)
- `W.` (PER)
- `Help Food` (ORG)
- `Help Food` (ORG)
- `Help Food` (ORG)
- `M. D.` (PER)
- `P. W.` (PER)
- `S.` (PER)
- `Help Food` (ORG)
- `D.` (PER)
- `W.` (PER)

**Example 11** (doc_id: `56534`) (sent_id: `56534`)


Den Migrationshintergrund habe Dr. S. in seinem Gutachten nicht berücksichtigt , was sie an einzelnen Passagen des Gutachtens belegt .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Example 12** (doc_id: `56774`) (sent_id: `56774`)


Der Angeklagte O. räumte sodann über eine Erklärung seines Verteidigers die Tat in objektiver und subjektiver Hinsicht ein und bestätigte persönlich die Erklärung seines Verteidigers .

| Predicted | Gold |
|---|---|
| `O.` | `O.` |

**Example 13** (doc_id: `57122`) (sent_id: `57122`)


Einen Verfügungssatz , der die nachfolgend genannten " Vergleichspunktzahlvolumen , bei dessen Überschreitung eine Honorarkürzung zulässig ist " , allein auf das " Job-Sharing-Pärchen " Dr. R. und Dr. E. beziehen würde , enthält der Bescheid nicht .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |
| `E.` | `E.` |

**Example 14** (doc_id: `57131`) (sent_id: `57131`)


Der geringe Erfolg des Rechtsmittels des Angeklagten I. lässt es nicht unbillig erscheinen , ihn mit den gesamten Kosten seines Rechtsmittels zu belasten .

| Predicted | Gold |
|---|---|
| `I.` | `I.` |

**Example 15** (doc_id: `57355`) (sent_id: `57355`)


Die Taten fanden ein Ende , nachdem die Zeugin R. im Sexualkundeunterricht aufgeklärt worden war .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |

**Example 16** (doc_id: `57413`) (sent_id: `57413`)


Vielmehr bedinge das seelische Leiden , das ursächlich für die vom Kläger beschriebenen Konzentrations- und Orientierungsstörungen sei , nach den überzeugenden gutachterlichen Ausführungen des nervenärztlichen Sachverständigen G. vom 24. 7. 2017 lediglich einen Einzel-GdB von 40. Auch der Befundbericht des behandelnden Facharztes für Neurologie und Psychiatrie Dr. H. vom 12. 5. 2016 und der Entlassungsbericht der M. -Klinik vom 29. 8. 2013 rechtfertigten keine andere Beurteilung .

| Predicted | Gold |
|---|---|
| `H.` | `H.` |

**Missed by this rule (FN):**

- `G.` (PER)
- `M. -Klinik` (ORG)

**Example 17** (doc_id: `57610`) (sent_id: `57610`)


Zwar wird angeführt , dass die mitgeteilte Verurteilung vom 15. Mai 2017 erst nach der Verurteilung des Beschwerdeführers erfolgt ist ; beanstandet wird aber nur , dass die Annahme des Landgerichts auf UA 30 , der Zeuge M. sei glaubhaft , weil er sich durch seine Angaben erheblich selbst belastet habe , mit der Verurteilung vom 15. Mai 2017 nicht belegt werden könne , da diese andere Taten betreffe .

| Predicted | Gold |
|---|---|
| `M.` | `M.` |

**Example 18** (doc_id: `57665`) (sent_id: `57665`)


Mit Schreiben ihres vorinstanzlichen Prozessbevollmächtigten vom 6. Mai 2015 , das den Briefkopf " T. Ts. & Partner Rechtsanwälte " trägt , die Rechtsanwälte T. , Ts. , M. und Dr. T. auflistet und von Rechtsanwalt T. unterzeichnet wurde , wiederholte die Klägerin den Widerruf .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |
| `T.` | `T.` |

**Missed by this rule (FN):**

- `" T. Ts. & Partner Rechtsanwälte "` (ORG)
- `Ts.` (PER)
- `M.` (PER)

**Example 19** (doc_id: `57819`) (sent_id: `57819`)


Rechtsanwalt B. war dem Angeklagten , der mit der Bestellung eines Pflichtverteidigers nicht einverstanden war , mit Beschluss des Vorsitzenden vom 21. März 2016 zur Verfahrenssicherung als Pflichtverteidiger neben dem Wahlverteidiger Rechtsanwalt P. beigeordnet worden .

| Predicted | Gold |
|---|---|
| `B.` | `B.` |
| `P.` | `P.` |

**Example 20** (doc_id: `57960`) (sent_id: `57960`)


Zu dem Sachverständigengutachten des Prof. Dr. T. sowie dessen ergänzender Stellungnahme hat die Beklagte mitgeteilt , dass sie diesem nach prüfärztlicher Stellungnahme nicht folgen könne ; der Einschätzung des Dr. S. hat sie sich hingegen angeschlossen .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |
| `S.` | `S.` |

**Example 21** (doc_id: `58153`) (sent_id: `58153`)


Auf die Beschwerde des Klägers hat das BSG mit Beschluss vom 14. 11. 2013 ( B 9 V 33/13 B ) das Urteil des LSG aufgehoben und die Sache zurückverwiesen , weil das LSG das Paul-Ehrlich-Institut ( Prof. Dr. C. ) nicht zu Einwänden des Klägers ergänzend befragt hat .

| Predicted | Gold |
|---|---|
| `C.` | `C.` |

**Missed by this rule (FN):**

- `BSG` (ORG)
- `Beschluss vom 14. 11. 2013 ( B 9 V 33/13 B )` (RS)
- `Paul-Ehrlich-Institut` (ORG)

**Example 22** (doc_id: `58450`) (sent_id: `58450`)


In einer ersten Regelbeurteilung vom 23. April 2013 zum Stichtag 1. April 2013 vergab der seinerzeitige Leiter der Abteilung X des BND ( Herr Dr. A. ) das Gesamturteil 7. Auf Einwendungen des Klägers hob der BND diese dienstliche Beurteilung wegen formeller Fehler auf .

| Predicted | Gold |
|---|---|
| `A.` | `A.` |

**Missed by this rule (FN):**

- `Abteilung X des BND` (ORG)
- `BND` (ORG)

**Example 23** (doc_id: `58530`) (sent_id: `58530`)


Ab dem 1. 7. 2008 stellte die Klägerin Dr. L. ( Internist mit Schwerpunkt Hämatologie / Onkologie ) mit einem Beschäftigungsumfang von 10 Stunden / Woche an .

| Predicted | Gold |
|---|---|
| `L.` | `L.` |

**Example 24** (doc_id: `58718`) (sent_id: `58718`)


Der Geschäftsführer der Beklagten S. ist zugleich Geschäftsführer der - mittlerweile in Liquidation befindlichen - Help Food .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Missed by this rule (FN):**

- `Help Food` (ORG)

**Example 25** (doc_id: `58781`) (sent_id: `58781`)


3. Der Angeklagte R. hat die Kosten seines Rechtsmittels zu tragen .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |

**Example 26** (doc_id: `59176`) (sent_id: `59176`)


Nach Verlassen der Bar gegen 2.30 Uhr begleiteten die Angeklagten die Zeugin L. auf dem Nachhauseweg .

| Predicted | Gold |
|---|---|
| `L.` | `L.` |

**Example 27** (doc_id: `59742`) (sent_id: `59742`)


1. Dem Angeklagten A. wird auf seinen Antrag Wiedereinsetzung in den vorigen Stand wegen Versäumung der Frist zur Begründung der Revision gegen das Urteil des Landgerichts Göttingen vom 30. Mai 2017 gewährt .

| Predicted | Gold |
|---|---|
| `A.` | `A.` |

**Missed by this rule (FN):**

- `Landgerichts Göttingen` (ORG)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `57273`) (sent_id: `57273`)


Der Kläger betrieb vormals eine Anwaltssozietät mit Rechtsanwalt C. B. in F. .

**False Positives:**

- `C.` — partial — pred is substring of gold: `C. B.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C. B.`(PER)
- `F.`(LOC)

</details>

---

## `Full Names with Titles`

**F1:** 0.091 | **Precision:** 0.680 | **Recall:** 0.049  

**Format:** `regex`  
**Rule ID:** `d8741160`  
**Description:**
Captures names preceded by titles like Dr., Prof., Dipl.-Ing., etc., ensuring the title is not captured and the name is a valid surname.

**Content:**
```
(?:Dr\.?\s+|Prof\.?\s+|Dipl\.-Ing\.\s+|Dipl\.-Psych\.\s+|Richter\s+|Anwalt\s+|Rechtsanwältin\s+|Rechtsanwalt\s+)([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.680 | 0.049 | 0.091 | 25 | 17 | 8 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 17 | 8 | 331 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `54291`) (sent_id: `54291`)


In der Beschwerdesache betreffend die Designanmeldung 40 2016 201 675.4 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Merzbach` | `Merzbach` |
| `Meiser` | `Meiser` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Hacker` (PER)

**Example 1** (doc_id: `54886`) (sent_id: `54886`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2012 063 820.1 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 5. Dezember 2017 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Nielsen` | `Nielsen` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Kriener` (PER)

**Example 2** (doc_id: `55301`) (sent_id: `55301`)


Dr. Milger

| Predicted | Gold |
|---|---|
| `Milger` | `Milger` |

**Example 3** (doc_id: `55818`) (sent_id: `55818`)


In der Beschwerdesache betreffend die Patentanmeldung 10 2014 003 988.9 hat der 23. Senat ( Technischer Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 8. Mai 2018 unter Mitwirkung des Vorsitzenden Richters Dr. Strößner sowie der Richter Dr. Friedrich , Dr. Zebisch und Dr. Himmelmann beschlossen :

| Predicted | Gold |
|---|---|
| `Strößner` | `Strößner` |
| `Zebisch` | `Zebisch` |
| `Himmelmann` | `Himmelmann` |

**Missed by this rule (FN):**

- `23. Senat ( Technischer Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Friedrich` (PER)

**Example 4** (doc_id: `56015`) (sent_id: `56015`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2011 068 984.9 hat der 26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 8. Januar 2018 unter Mitwirkung der Vorsitzenden Richterin Kortge sowie der Richter Jacobi und Schödel beschlossen :

| Predicted | Gold |
|---|---|
| `Jacobi` | `Jacobi` |

**Missed by this rule (FN):**

- `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Kortge` (PER)
- `Schödel` (PER)

**Example 5** (doc_id: `59053`) (sent_id: `59053`)


c ) Unter diesen Umständen ist die Besorgnis des Beschwerdeführers nachvollziehbar , Richter Müller werde die zu entscheidenden , in hohem Maße wertungsabhängigen und von Vorverständnissen geprägten Rechtsfragen möglicherweise nicht mehr in jeder Hinsicht offen und unbefangen beurteilen können ( vgl. BVerfGE 72 , 296 < 298 > ; 95 , 189 < 192 > ; 135 , 248 < 259 Rn. 27 > ) .

| Predicted | Gold |
|---|---|
| `Müller` | `Müller` |

**Missed by this rule (FN):**

- `BVerfGE 72 , 296 < 298 > ; 95 , 189 < 192 > ; 135 , 248 < 259 Rn. 27 >` (RS)

**Example 6** (doc_id: `59509`) (sent_id: `59509`)


In der Beschwerdesache betreffend die Marke 30 2009 026 804 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 21. September 2017 unter Mitwirkung der Richter Merzbach , Dr. Meiser und Schödel beschlossen :

| Predicted | Gold |
|---|---|
| `Merzbach` | `Merzbach` |
| `Meiser` | `Meiser` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Schödel` (PER)

**Example 7** (doc_id: `59628`) (sent_id: `59628`)


In der Beschwerdesache betreffend die Marke 30 2012 041 338 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 15. November 2017 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Nielsen` | `Nielsen` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Kriener` (PER)

**Example 8** (doc_id: `59761`) (sent_id: `59761`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2015 031 519.2 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Knoll sowie der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

| Predicted | Gold |
|---|---|
| `Nielsen` | `Nielsen` |

**Missed by this rule (FN):**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Knoll` (PER)
- `Kriener` (PER)

**Example 9** (doc_id: `59948`) (sent_id: `59948`)


In der Beschwerdesache betreffend die international registrierte Marke IR 1 160 635 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 9. November 2017 unter Mitwirkung des Vorsitzenden Richters Professor Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

| Predicted | Gold |
|---|---|
| `Hacker` | `Hacker` |
| `Merzbach` | `Merzbach` |
| `Meiser` | `Meiser` |

**Missed by this rule (FN):**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts` (ORG)

**Example 10** (doc_id: `60001`) (sent_id: `60001`)


Dr. Milger

| Predicted | Gold |
|---|---|
| `Milger` | `Milger` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53397`) (sent_id: `53397`)


HLNK20 Gutachten Dr. Jay B. Saoud aus dem parallelen britischen Verfahren vom 14. April 2016 , 21 Seiten und 12 Seiten Anlagen

**False Positives:**

- `Jay` — partial — pred is substring of gold: `Jay B. Saoud`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Jay B. Saoud`(PER)

**Example 1** (doc_id: `53890`) (sent_id: `53890`)


Im Berufungsverfahren hat der zuständige Berichterstatter des LSG den Sachverständigen Prof. Dr. T. im Termin zur mündlichen Verhandlung ergänzend angehört .

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `T.`(PER)

**Example 2** (doc_id: `54018`) (sent_id: `54018`)


In der Beschwerdesache betreffend die Marke 30 2010 022 988 hat der 27. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 10. Mai 2017 durch die Vorsitzende Richterin Klante , den Richter Dr. Himmelmann und die Richterin Lachenmayr-Nikolaou beschlossen :

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `27. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Klante`(PER)
- `Himmelmann`(PER)
- `Lachenmayr-Nikolaou`(PER)

**Example 3** (doc_id: `54291`) (sent_id: `54291`)


In der Beschwerdesache betreffend die Designanmeldung 40 2016 201 675.4 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 4** (doc_id: `55088`) (sent_id: `55088`)


Zwar hat der Senat durchaus zur Kenntnis genommen , dass die Klägerin in dem von ihr benannten Schriftsatz vom 11. 2. 2016 - auf die Anhörung des LSG zur Entscheidung durch Beschluss - durch die Bezugnahme auf den Schriftsatz vom 7. 3. 2016 ihren Antrag bei Prof. Dr. T. nachzufragen , in welcher Form und in welchem Umfang er an der Erstellung des Sachverständigengutachtens beteiligt gewesen sei , wiederholt hat .

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `T.`(PER)

**Example 5** (doc_id: `55818`) (sent_id: `55818`)


In der Beschwerdesache betreffend die Patentanmeldung 10 2014 003 988.9 hat der 23. Senat ( Technischer Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 8. Mai 2018 unter Mitwirkung des Vorsitzenden Richters Dr. Strößner sowie der Richter Dr. Friedrich , Dr. Zebisch und Dr. Himmelmann beschlossen :

**False Positives:**

- `Dr` — similar text (different position): `Friedrich`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `23. Senat ( Technischer Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Strößner`(PER)
- `Friedrich`(PER)
- `Zebisch`(PER)
- `Himmelmann`(PER)

**Example 6** (doc_id: `57960`) (sent_id: `57960`)


Zu dem Sachverständigengutachten des Prof. Dr. T. sowie dessen ergänzender Stellungnahme hat die Beklagte mitgeteilt , dass sie diesem nach prüfärztlicher Stellungnahme nicht folgen könne ; der Einschätzung des Dr. S. hat sie sich hingegen angeschlossen .

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `T.`(PER)
- `S.`(PER)

**Example 7** (doc_id: `58153`) (sent_id: `58153`)


Auf die Beschwerde des Klägers hat das BSG mit Beschluss vom 14. 11. 2013 ( B 9 V 33/13 B ) das Urteil des LSG aufgehoben und die Sache zurückverwiesen , weil das LSG das Paul-Ehrlich-Institut ( Prof. Dr. C. ) nicht zu Einwänden des Klägers ergänzend befragt hat .

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG`(ORG)
- `Beschluss vom 14. 11. 2013 ( B 9 V 33/13 B )`(RS)
- `Paul-Ehrlich-Institut`(ORG)
- `C.`(PER)

</details>

---

## `Initials with Surname`

**F1:** 0.039 | **Precision:** 0.081 | **Recall:** 0.026  

**Format:** `regex`  
**Rule ID:** `b5047799`  
**Description:**
Captures names with an initial followed by a surname (e.g., 'M. Trümner', 'F. Rojahn').

**Content:**
```
\b([A-Z]\.)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.081 | 0.026 | 0.039 | 111 | 9 | 102 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 9 | 102 | 339 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `54587`) (sent_id: `54587`)


D. Reidelbach

| Predicted | Gold |
|---|---|
| `D. Reidelbach` | `D. Reidelbach` |

**Example 1** (doc_id: `55941`) (sent_id: `55941`)


K. Schmidt

| Predicted | Gold |
|---|---|
| `K. Schmidt` | `K. Schmidt` |

**Example 2** (doc_id: `56556`) (sent_id: `56556`)


D14 J. Deubener et al. , “ Induction time analysis of nucleation and crystal growth in di- and metasilicate glasses ” , Journal of Non-Crystalline Solids , 1993 , 163 , Seiten 1 bis 12

| Predicted | Gold |
|---|---|
| `J. Deubener` | `J. Deubener` |

**Example 3** (doc_id: `56970`) (sent_id: `56970`)


M. Jostes

| Predicted | Gold |
|---|---|
| `M. Jostes` | `M. Jostes` |

**Example 4** (doc_id: `57047`) (sent_id: `57047`)


K. Schmidt

| Predicted | Gold |
|---|---|
| `K. Schmidt` | `K. Schmidt` |

**Example 5** (doc_id: `58172`) (sent_id: `58172`)


J. Ratayczak

| Predicted | Gold |
|---|---|
| `J. Ratayczak` | `J. Ratayczak` |

**Example 6** (doc_id: `59171`) (sent_id: `59171`)


K. Schmidt

| Predicted | Gold |
|---|---|
| `K. Schmidt` | `K. Schmidt` |

**Example 7** (doc_id: `59274`) (sent_id: `59274`)


W. Reinfelder

| Predicted | Gold |
|---|---|
| `W. Reinfelder` | `W. Reinfelder` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53397`) (sent_id: `53397`)


HLNK20 Gutachten Dr. Jay B. Saoud aus dem parallelen britischen Verfahren vom 14. April 2016 , 21 Seiten und 12 Seiten Anlagen

**False Positives:**

- `B. Saoud` — partial — pred is substring of gold: `Jay B. Saoud`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Jay B. Saoud`(PER)

**Example 1** (doc_id: `53692`) (sent_id: `53692`)


I. Das LSG hat mit Urteil vom 11. 5. 2017 einen Zahlungsanspruch der Klägerin ( eine aus zwei Personen bestehende , im Partnerschaftsregister eingetragene Physiotherapie-Partnerschaft ) in Höhe von 7249,01 Euro für physiotherapeutische Leistungen verneint , nachdem die beklagte Krankenkasse die erbrachten Leistungen zunächst bezahlt , die Zahlungen aber wieder zurückgefordert und die Rückforderung schließlich im Wege der Aufrechnung durchgesetzt hatte .

**False Positives:**

- `I. Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `53741`) (sent_id: `53741`)


I. Erläuterung zur ersten Vorlagefrage

**False Positives:**

- `I. Erläuterung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 3** (doc_id: `53763`) (sent_id: `53763`)


Wird als Verfahrensmangel gerügt , das FG habe einen gestellten Beweisantrag übergangen ( Rüge mangelnder Sachaufklärung gemäß § 76 Abs. 1 Satz 1 FGO ) , so ist darzulegen , welche Tatfrage aufklärungsbedürftig ist , welche Beweismittel das FG zu welchem Beweisthema nicht erhoben hat , die genauen Fundstellen ( Schriftsatz mit Datum und Seitenzahl , Terminprotokoll ) , in denen die Beweismittel und Beweisthemen angeführt worden sind , das voraussichtliche Ergebnis der Beweisaufnahme , inwiefern das Urteil des FG aufgrund dessen sachlich-rechtlicher Auffassung auf der unterbliebenen Beweisaufnahme beruhen kann und dass die Nichterhebung der Beweise vor dem FG rechtzeitig gerügt worden ist oder aufgrund des Verhaltens des FG nicht mehr vor diesem gerügt werden konnte ( ständige BFH-Rechtsprechung , z.B. Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85 , und vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125 , m. w. N. ) .

**False Positives:**

- `B. Beschlüsse` — positional overlap with gold: `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 76 Abs. 1 Satz 1 FGO`(NRM)
- `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`(RS)
- `vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125`(RS)

**Example 4** (doc_id: `53767`) (sent_id: `53767`)


Ob der Erwerb eigener Anteile auf der Gesellschaftsebene entsprechend der durch das BilMoG geänderten handelsrechtlichen Vorschriften ( Einfügung des § 272 Abs. 1a und 1b HGB ) steuerrechtlich nicht mehr als Erwerbsvorgang anzusehen , sondern nunmehr als " Teilliquidation " und daher " wie " eine Kapitalherabsetzung zu behandeln ist ( so BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f. ; zustimmend z.B. Blumenberg / Lechner , DB 2014 , 141 , 147 ; kritisch z.B. Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55 ) , kann im Streitfall mangels Erheblichkeit für die Entscheidung offenbleiben .

**False Positives:**

- `B. Blumenberg` — positional overlap with gold: `Blumenberg / Lechner , DB 2014 , 141 , 147`
- `B. Gosch` — positional overlap with gold: `Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BilMoG`(NRM)
- `§ 272 Abs. 1a und 1b HGB`(NRM)
- `BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f.`(REG)
- `Blumenberg / Lechner , DB 2014 , 141 , 147`(LIT)
- `Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55`(LIT)

**Example 5** (doc_id: `53897`) (sent_id: `53897`)


Sofern die Beschwerdeführerin ferner ausführt , dass der Druckschrift D4 ( z.B. Figur 1 ) ein Anhänger zu entnehmen sei , der gemäß Merkmal M7 eine Anhängerdeichsel mit einer Anhängerkupplung und einer Drehwelle aufweise , die unterhalb der Anhängerkupplung angeordnet sei , so kann ihr darin zugestimmt werden .

**False Positives:**

- `B. Figur` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 6** (doc_id: `53940`) (sent_id: `53940`)


I. Auf das Arbeitsverhältnis finden kraft arbeitsvertraglicher Bezugnahme der Tarifvertrag zur Angleichung des Tarifrechts des Landes Berlin an das Tarifrecht der Tarifgemeinschaft deutscher Länder ( Angleichungs-TV Land Berlin ) vom 14. Oktober 2010 und gem. dessen § 2 der TV-L sowie der Tarifvertrag zur Überleitung der Beschäftigten der Länder in den TV-L und zur Regelung des Übergangsrechts ( TVÜ-Länder ) ab dem 1. November 2010 Anwendung .

**False Positives:**

- `I. Auf` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Tarifvertrag zur Angleichung des Tarifrechts des Landes Berlin an das Tarifrecht der Tarifgemeinschaft deutscher Länder`(REG)
- `Angleichungs-TV Land Berlin`(REG)
- `§ 2 der TV-L`(REG)
- `Tarifvertrag zur Überleitung der Beschäftigten der Länder in den TV-L und zur Regelung des Übergangsrechts`(REG)
- `TVÜ-Länder`(REG)

**Example 7** (doc_id: `53955`) (sent_id: `53955`)


I. Die Beschwerdeführerin ist Inhaberin des am 15. Mai 1993 angemeldeten und am 30. September 2009 erteilten europäischen Patents EP 0 835 663 ( DE 693 34 297 ) , das mittlerweile durch Zeitablauf erloschen ist .

**False Positives:**

- `I. Die Beschwerdeführerin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `54071`) (sent_id: `54071`)


C. Das Landesarbeitsgericht hat die gegen die Beendigung des Arbeitsverhältnisses der Parteien durch die außerordentliche Kündigung der Beklagten vom 28. Juli 2016 gerichtete Kündigungsschutzklage zu Recht abgewiesen .

**False Positives:**

- `C. Das Landesarbeitsgericht` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 9** (doc_id: `54143`) (sent_id: `54143`)


Dieser während des Revisionsverfahrens eingetretene Zuständigkeitswechsel führt zu einem gesetzlichen Beteiligtenwechsel ( vgl. z.B. Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109 , m. w. N. ) .

**False Positives:**

- `B. Urteil` — positional overlap with gold: `Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109`(RS)

**Example 10** (doc_id: `54169`) (sent_id: `54169`)


Unabhängig von der Frage , ob ein solcher Antrag - entgegen der Auffassung des Antragstellers - gemäß § 67 Abs. 4 VwGO dem Vertretungszwang unterliegt ( vgl. verneinend z.B. : BVerwG , Beschluss vom 6. Juni 1972 - 3 ER 404.71 - Buchholz 310 § 53 VwGO Nr. 6 ; Ziekow , in : Sodan / Ziekow , VwGO , 4. Aufl. 2014 , § 53 Rn. 20 ; Kraft , in : Eyermann , VwGO , 14. Aufl. 2014 , § 53 Rn. 14 ; Unruh , in : HK-Verwaltungsrecht , 3. Aufl. 2013 , § 53 Rn. 15 ; Scheidler , VR 2012 , 113 ; bejahend z.B. : W. - R. Schenke , in : Kopp / Schenke , VwGO , 23. Aufl. 2017 , § 53 Rn. 10 ; Redeker / von Oertzen , VwGO , 15. Aufl. 2010 , § 53 Rn. 5 ; vermittelnd z.B. : Bier , in : Schoch / Schneider / Bier , VwGO , Stand Oktober 2016 , § 53 Rn. 13 ) , ist der Antrag hier jedenfalls deshalb unzulässig , weil die gesetzlichen Voraussetzungen , unter denen das Bundesverwaltungsgericht eine Zuständigkeitsbestimmung überhaupt vornehmen darf , nicht erfüllt sind .

**False Positives:**

- `R. Schenke` — partial — pred is substring of gold: `W. - R. Schenke , in : Kopp / Schenke , VwGO , 23. Aufl. 2017 , § 53 Rn. 10`

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

**Example 11** (doc_id: `54179`) (sent_id: `54179`)


I. Mit dem angefochtenen Beschluss vom 4. November 2015 hat die Patentabteilung 43 des Deutschen Patent- und Markenamts das Patent 103 36 913 mit der Bezeichnung

**False Positives:**

- `I. Mit` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Patentabteilung 43 des Deutschen Patent- und Markenamts`(ORG)

**Example 12** (doc_id: `54337`) (sent_id: `54337`)


I. Die Verfassungsbeschwerde betrifft die Höhe des Landesblindengeldes in Schleswig-Holstein nach deren Reduzierung auf 200 Euro monatlich ab 1. Januar 2011 .

**False Positives:**

- `I. Die Verfassungsbeschwerde` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Schleswig-Holstein`(LOC)

**Example 13** (doc_id: `54348`) (sent_id: `54348`)


I. Soweit es die Verurteilung wegen der Tat vom 31. Mai 2015 betrifft , liegen dem folgende Feststellungen und Wertungen zu Grunde :

**False Positives:**

- `I. Soweit` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 14** (doc_id: `54572`) (sent_id: `54572`)


I. Die von der Beschwerdeführerin als gleichheitswidrig beanstandeten Regelungen durch den von ihr mittelbar angegriffenen § 7 Satz 2 Nr. 2 GewStG sind verfassungsgemäß ; der Gesetzgeber bewegt sich mit dieser Neuregelung des Jahres 2002 im Rahmen seiner Gestaltungsbefugnis .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 7 Satz 2 Nr. 2 GewStG`(NRM)

**Example 15** (doc_id: `54732`) (sent_id: `54732`)


I. Am 22. Mai 2013 ist das Zeichen

**False Positives:**

- `I. Am` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 16** (doc_id: `54901`) (sent_id: `54901`)


3. Die vorstehenden Erwägungen gelten entsprechend , soweit der Kläger seinen Anspruch allein auf Abschn. C. Ziff. 2.6 Abs. 1 STV stützt .

**False Positives:**

- `C. Ziff` — partial — pred is substring of gold: `Abschn. C. Ziff. 2.6 Abs. 1 STV`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Abschn. C. Ziff. 2.6 Abs. 1 STV`(REG)

**Example 17** (doc_id: `55135`) (sent_id: `55135`)


Eine unterbliebene notwendige Beiladung stellt einen vom Rechtsmittelgericht von Amts wegen zu prüfenden Verstoß gegen die Grundordnung des Verfahrens dar ( z.B. Senatsurteil vom 11. Juli 2017 I R 34/14 , juris ) .

**False Positives:**

- `B. Senatsurteil` — positional overlap with gold: `Senatsurteil vom 11. Juli 2017 I R 34/14 , juris`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Senatsurteil vom 11. Juli 2017 I R 34/14 , juris`(RS)

**Example 18** (doc_id: `55160`) (sent_id: `55160`)


I. Die Klage ist zulässig , insbesondere hinreichend bestimmt iSv. § 253 Abs. 2 Nr. 2 ZPO .

**False Positives:**

- `I. Die Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 253 Abs. 2 Nr. 2 ZPO`(NRM)

**Example 19** (doc_id: `55200`) (sent_id: `55200`)


I. Der Anspruch auf die geltend gemachte Abfindung folgt nicht aus § 11 Abs. 1 TV ATZ .

**False Positives:**

- `I. Der Anspruch` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 11 Abs. 1 TV ATZ`(REG)

**Example 20** (doc_id: `55285`) (sent_id: `55285`)


I. Das am 26. August 2013 angemeldete Zeichen Fireslim ist am 10. Januar 2014 unter der Nr. 30 2013 048 208 in das beim Deutschen Patent- und Markenamt geführte Markenregister für die nachfolgenden Waren und Dienstleistungen der Klassen 9 , 35 und 38 eingetragen worden :

**False Positives:**

- `I. Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Fireslim`(ORG)
- `Deutschen Patent- und Markenamt`(ORG)

**Example 21** (doc_id: `55398`) (sent_id: `55398`)


I. Die Beklagte ist Alleinerbin ihres am 24. Mai 2013 verstorbenen Ehemannes D. F. .

**False Positives:**

- `I. Die Beklagte` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `D. F.`(PER)

**Example 22** (doc_id: `55563`) (sent_id: `55563`)


B. Die Rechtsbeschwerde der Arbeitgeberin zu 1. ist begründet .

**False Positives:**

- `B. Die Rechtsbeschwerde` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 23** (doc_id: `55567`) (sent_id: `55567`)


Nach Durchführung einer Außenprüfung vertrat der Beklagte und Beschwerdegegner ( das Finanzamt - FA - ) in den Umsatzsteuer-Änderungsbescheiden für die Streitjahre vom 6. Mai 2014 die Auffassung , die genannten Leistungen seien von Personen in Anspruch genommen worden , denen aufgrund von Verkehrsdelikten ( z.B. Fahren unter Alkohol- oder Drogeneinfluss , Tempo- und / oder Abstandsverstöße etc. ) ihre Fahrerlaubnis entzogen worden sei , und die sich zur Wiedererlangung der Fahrerlaubnis einer medizinisch-psychologischen Untersuchung ( MPU ) i. S. des § 2 Abs. 8 des Straßenverkehrsgesetzes ( StVG ) hätten unterziehen müssen .

**False Positives:**

- `B. Fahren` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 2 Abs. 8 des Straßenverkehrsgesetzes`(NRM)
- `StVG`(NRM)

**Example 24** (doc_id: `55569`) (sent_id: `55569`)


I. Die Klägerin und Beschwerdeführerin ( Klägerin ) ist umsatzsteuerrechtlich Organgesellschaft des Organträgers N .

**False Positives:**

- `I. Die Klägerin` — similar text (different position): `N`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `N`(ORG)

**Example 25** (doc_id: `55620`) (sent_id: `55620`)


I. Die Parteien streiten über die Widerruflichkeit der auf Abschluss zweier Verbraucherdarlehensverträge gerichteten Willenserklärungen der Kläger .

**False Positives:**

- `I. Die Parteien` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 26** (doc_id: `55743`) (sent_id: `55743`)


I. Das Landgericht hat - soweit für das Revisionsverfahren bedeutsam - folgende Feststellungen getroffen :

**False Positives:**

- `I. Das Landgericht` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 27** (doc_id: `55795`) (sent_id: `55795`)


b ) Bereits aus dem Wortlaut " außergewöhnliche Belastungen " folgt - auch ohne einen Klammerverweis auf die §§ 33 bis 33b EStG - , dass § 26a Abs. 2 Satz 1 Halbsatz 1 EStG ( auch ) solche Aufwendungen erfasst , die über den Behinderten-Pauschbetrag i. S. des § 33b Abs. 1 EStG abgedeckt werden ( anders z.B. Blümich / Ettlich , § 26a EStG Rz 25 ; Pflüger in Herrmann / Heuer / Raupach - HHR - , § 26a EStG Rz 60 ; Nacke in Littmann / Bitz / Pust , Das Einkommensteuerrecht , Kommentar , § 33b Rz 51a ) .

**False Positives:**

- `B. Blümich` — positional overlap with gold: `Blümich / Ettlich , § 26a EStG Rz 25`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§§ 33 bis 33b EStG -`(NRM)
- `§ 26a Abs. 2 Satz 1 Halbsatz 1 EStG`(NRM)
- `§ 33b Abs. 1 EStG`(NRM)
- `Blümich / Ettlich , § 26a EStG Rz 25`(LIT)
- `Pflüger in Herrmann / Heuer / Raupach - HHR - , § 26a EStG Rz 60`(LIT)
- `Nacke in Littmann / Bitz / Pust , Das Einkommensteuerrecht , Kommentar , § 33b Rz 51a`(LIT)

**Example 28** (doc_id: `55939`) (sent_id: `55939`)


Denn in dem dieser Entscheidung zugrunde liegenden Fall hat sich die Einsprechende lediglich mit einem einzigen und zudem fakultativen Teilmerkmal des insgesamt vier Merkmale ( Verfahrensschritte ) umfassenden patentierten Epoxidationsverfahrens befasst , und keine weiteren Angaben gemacht , die für die Patentierungserfordernisse ( z.B. Neuheit oder erfinderische Tätigkeit ) der die gesamten bzw. nichtfakultativen Merkmale einschließenden Lehre des patentgemäßen Verfahrens von Bedeutung sein könnten .

**False Positives:**

- `B. Neuheit` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 29** (doc_id: `55974`) (sent_id: `55974`)


Teilweise beziehen sich diese Presseartikel auf Personen , die ( noch ) nicht verurteilt worden sind ( z.B. Untersuchungsgefangene ) , oder auf ein Absehen von Verfolgung insgesamt .

**False Positives:**

- `B. Untersuchungsgefangene` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 30** (doc_id: `55981`) (sent_id: `55981`)


I. Die am 6. Mai 2010 angemeldete Wort- / Bildmarke 30 2010 028 176 ist am 20. Dezember 2010 für die nachfolgend genannten Waren in das beim Deutschen Patent- und Markenamt ( DPMA ) geführte Markenregister eingetragen worden :

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschen Patent- und Markenamt`(ORG)
- `DPMA`(ORG)

**Example 31** (doc_id: `56002`) (sent_id: `56002`)


Ferner hat das Landgericht zu Recht darauf hingewiesen , dass die Kläger ausweislich ihrer Forderung nach einer Nutzungsentschädigung für die von ihnen gezahlten Vorfälligkeitsentschädigungen selbst davon ausgehen , dass die Beklagte diese Beträge wieder angelegt hat ( s. a. OLG Brandenburg , Urteil vom 4. 1. 2017 , a. a. O. Tz. 58 ) .

**False Positives:**

- `O. Tz` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `OLG Brandenburg`(ORG)

**Example 32** (doc_id: `56139`) (sent_id: `56139`)


B. Das LSG hat im Ergebnis zu Recht das Urteil des SG aufgehoben .

**False Positives:**

- `B. Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 33** (doc_id: `56197`) (sent_id: `56197`)


I. Die am 7. November 2013 angemeldete Wortfolge Rap Shot ist am 23. Januar 2014 unter der Nummer 30 2013 058 941 als Wortmarke für die nachfolgend genannten Waren und Dienstleistungen in das beim Deutschen Patent- und Markenamt geführte Markenregister eingetragen worden :

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Rap Shot`(ORG)
- `Deutschen Patent- und Markenamt`(ORG)

**Example 34** (doc_id: `56273`) (sent_id: `56273`)


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

**Example 35** (doc_id: `56276`) (sent_id: `56276`)


I. Die Revision ist zulässig , soweit der Kläger sein Schadensersatzbegehren auf die Besetzung der Stellen mit externen Bewerbern stützt .

**False Positives:**

- `I. Die Revision` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 36** (doc_id: `56478`) (sent_id: `56478`)


C. Die Verfassungsbeschwerden sind zulässig und begründet .

**False Positives:**

- `C. Die Verfassungsbeschwerden` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 37** (doc_id: `56479`) (sent_id: `56479`)


I. Nach den Feststellungen des Landgerichts kamen der Angeklagte und sein in der Türkei lebender gesondert verfolgter Bruder E. K. spätestens zu Beginn des Jahres 2015 überein , in arbeitsteiligem Zusammenwirken in der Türkei hergestellte bzw. erworbene Kleidungsstücke , die mit Schriftzügen und Labels verschiedener Markenhersteller versehen waren , unter Verletzung geschützter Gemeinschafts- bzw. Unionsmarken in Deutschland zu verkaufen , obwohl ihnen bewusst war , dass sie nicht über die für deren Verwendung erforderliche Zustimmung der Markenrechtsinhaber verfügten .

**False Positives:**

- `I. Nach` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Türkei`(LOC)
- `E. K.`(PER)
- `Türkei`(LOC)
- `Deutschland`(LOC)

**Example 38** (doc_id: `56586`) (sent_id: `56586`)


C. Die zulässigen Revisionen sind gleichfalls unbegründet , soweit sie die Zahlungsanträge gegen den Beklagten zu 3. betreffen .

**False Positives:**

- `C. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 39** (doc_id: `56618`) (sent_id: `56618`)


So sind regelmäßig auch rechtsgeschäftliche Lizenzen kündbar oder können bei Wegfall des Patents und damit der Geschäftsgrundlage angepasst werden ( vgl. z.B. Keukenschrijver , Patentnichtigkeitsverfahren , 6. Aufl. , Rn. 405 ) .

**False Positives:**

- `B. Keukenschrijver` — positional overlap with gold: `Keukenschrijver , Patentnichtigkeitsverfahren , 6. Aufl. , Rn. 405`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Keukenschrijver , Patentnichtigkeitsverfahren , 6. Aufl. , Rn. 405`(LIT)

**Example 40** (doc_id: `56752`) (sent_id: `56752`)


I. Das Wort- / Bildzeichen ist am 19. Juli 2016 für die Dienstleistungen

**False Positives:**

- `I. Das Wort` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 41** (doc_id: `56795`) (sent_id: `56795`)


I. Die Verfügungsklägerin hat beantragt , ihr für die Durchführung eines Berufungsverfahrens Prozesskostenhilfe zu gewähren .

**False Positives:**

- `I. Die Verfügungsklägerin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 42** (doc_id: `56821`) (sent_id: `56821`)


3. Das in der Beschwerdebegründung unter der Überschrift " A. Tatbestand " enthaltene Vorbringen versteht der Senat dahingehend , dass hiermit lediglich der Sachverhalt dargestellt werden soll und insoweit keine eigenständigen Zulassungsgründe dargelegt werden sollen .

**False Positives:**

- `A. Tatbestand` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 43** (doc_id: `56915`) (sent_id: `56915`)


A. Die - zur gemeinsamen Entscheidung verbundenen - Verfassungsbeschwerden betreffen die Frage , ob deutschen Beamtinnen und Beamten ein Streikrecht zusteht .

**False Positives:**

- `A. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 44** (doc_id: `56948`) (sent_id: `56948`)


I. Die Klage ist zulässig .

**False Positives:**

- `I. Die Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 45** (doc_id: `57052`) (sent_id: `57052`)


Zu solchen mittelbaren Werbungskosten gehören z.B. Depotbankgebühren , Prüfungs- und Veröffentlichungskosten oder auch Verwaltungskosten ( vgl. Köhler in Berger / Steck / Lübbehüsen , a. a. O. , § 3 Rz 99 ; Wassermeyer , Der Betrieb 2003 , 2085 , 2087 ) .

**False Positives:**

- `B. Depotbankgebühren` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Köhler in Berger / Steck / Lübbehüsen , a. a. O. , § 3 Rz 99`(LIT)
- `Wassermeyer , Der Betrieb 2003 , 2085 , 2087`(LIT)

**Example 46** (doc_id: `57069`) (sent_id: `57069`)


I. Der Kläger und Beschwerdeführer ( Kläger ) erklärte für das Streitjahr 2011 einen Gewinn aus Gewerbebetrieb von 70.300 € , den er durch Einnahmen-Überschuss-Rechnung ermittelte .

**False Positives:**

- `I. Der Kläger` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 47** (doc_id: `57079`) (sent_id: `57079`)


Denn Sinn und Zweck der Pauschbetragsregelung ist es gerade , typisierend zu unterstellen , dass bestimmten Gruppen von behinderten Menschen gewisse außergewöhnliche Belastungen erwachsen ( vgl. z.B. Blümich / K. Heger , § 33b EStG Rz 4 , m. w. N. ; Urteil des Bundesfinanzhofs - BFH - vom 28. September 1984 VI R 164/80 , BFHE 142 , 377 , BStBl II 1985 , 129 , unter 1. c , m. w. N. , und BFH-Beschluss vom 13. Juli 2011 VI B 20/11 , BFH / NV 2011 , 1863 , Rz 9 , m. w. N. ) .

**False Positives:**

- `B. Blümich` — positional overlap with gold: `Blümich / K. Heger , § 33b EStG Rz 4`
- `K. Heger` — partial — pred is substring of gold: `Blümich / K. Heger , § 33b EStG Rz 4`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Blümich / K. Heger , § 33b EStG Rz 4`(LIT)
- `Urteil des Bundesfinanzhofs - BFH - vom 28. September 1984 VI R 164/80 , BFHE 142 , 377 , BStBl II 1985 , 129 , unter 1. c`(RS)
- `BFH-Beschluss vom 13. Juli 2011 VI B 20/11 , BFH / NV 2011 , 1863 , Rz 9`(RS)

**Example 48** (doc_id: `57188`) (sent_id: `57188`)


I. Der Kläger begehrt im Wege des Zugunstenverfahrens Entschädigungsleistungen nach dem Opferentschädigungsgesetz ( OEG ) für körperliche und seelische Misshandlungen und sexuellen Missbrauch während seiner Zeit als Fürsorgezögling in verschiedenen Heimen .

**False Positives:**

- `I. Der Kläger` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Opferentschädigungsgesetz`(NRM)
- `OEG`(NRM)

**Example 49** (doc_id: `57251`) (sent_id: `57251`)


A. Die Revision der Klägerin ist begründet .

**False Positives:**

- `A. Die Revision` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 50** (doc_id: `57288`) (sent_id: `57288`)


I. Durch dem Beklagten am 15. Februar 2017 zugestelltes Urteil hat das Amtsgericht der Klage teilweise stattgegeben .

**False Positives:**

- `I. Durch` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 51** (doc_id: `57348`) (sent_id: `57348`)


B. Die Vorlage ist unzulässig .

**False Positives:**

- `B. Die Vorlage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 52** (doc_id: `57368`) (sent_id: `57368`)


Der auf allen Seiten des Schriftsatzes aufgedruckte Briefkopf dieser Berufungs- und Berufungsbegründungsschrift lautet auf " T. Ts. & Partner Rechtsanwälte " .

**False Positives:**

- `T. Ts` — partial — pred is substring of gold: `" T. Ts. & Partner Rechtsanwälte "`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `" T. Ts. & Partner Rechtsanwälte "`(ORG)

**Example 53** (doc_id: `57411`) (sent_id: `57411`)


Im Zweifel ist das den Betroffenen weniger belastende Auslegungsergebnis vorzuziehen , da er als Empfänger einer auslegungsbedürftigen Willenserklärung der Verwaltung durch etwaige Unklarheiten aus deren Sphäre nicht benachteiligt werden darf ( z.B. Senatsurteil vom 27. Oktober 2015 VIII R 59/13 , BFH / NV 2016 , 726 , m. w. N. ) .

**False Positives:**

- `B. Senatsurteil` — positional overlap with gold: `Senatsurteil vom 27. Oktober 2015 VIII R 59/13 , BFH / NV 2016 , 726`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Senatsurteil vom 27. Oktober 2015 VIII R 59/13 , BFH / NV 2016 , 726`(RS)

**Example 54** (doc_id: `57458`) (sent_id: `57458`)


I. Die klagende Verbraucherzentrale ( im Folgenden : Kläger ) nimmt das beklagte Stromversorgungsunternehmen ( im Folgenden : Beklagte ) nach dem Unterlassungsklagengesetz ( UKlaG ) in Anspruch , es bei Stromlieferungsverträgen mit Haushaltskunden außerhalb der Grundversorgung zu unterlassen , sieben näher bezeichnete Klauseln als Allgemeine Geschäftsbedingungen ( AGB ) zu verwenden und sich darauf bei Abwicklung derartiger Verträge zu berufen .

**False Positives:**

- `I. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Unterlassungsklagengesetz`(NRM)
- `UKlaG`(NRM)

**Example 55** (doc_id: `57532`) (sent_id: `57532`)


I. Streitig ist der Kindergeldanspruch für den Zeitraum Mai 2010 bis März 2012 .

**False Positives:**

- `I. Streitig` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 56** (doc_id: `57646`) (sent_id: `57646`)


Ebenso ist zu berücksichtigen , dass der Verkehr ein als Marke verwendetes Zeichen in seiner Gesamtheit mit allen seinen Bestandteilen so aufnimmt , wie es ihm entgegentritt , ohne es einer analysierenden Betrachtungsweise zu unterziehen ( EuGH GRUR 2004 , 428 Rdnr. 53 – Henkel ; BGH a. a. O. Rdnr. 10 – OUI ; a. a. O. Rdnr. 16 – for you ) .

**False Positives:**

- `O. Rdnr` — partial — pred is substring of gold: `BGH a. a. O. Rdnr. 10 – OUI`
- `O. Rdnr` — similar text (different position): `BGH a. a. O. Rdnr. 10 – OUI`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2004 , 428 Rdnr. 53 – Henkel`(RS)
- `BGH a. a. O. Rdnr. 10 – OUI`(RS)
- `a. a. O. Rdnr. 16 – for you`(RS)

**Example 57** (doc_id: `57665`) (sent_id: `57665`)


Mit Schreiben ihres vorinstanzlichen Prozessbevollmächtigten vom 6. Mai 2015 , das den Briefkopf " T. Ts. & Partner Rechtsanwälte " trägt , die Rechtsanwälte T. , Ts. , M. und Dr. T. auflistet und von Rechtsanwalt T. unterzeichnet wurde , wiederholte die Klägerin den Widerruf .

**False Positives:**

- `T. Ts` — partial — pred is substring of gold: `" T. Ts. & Partner Rechtsanwälte "`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `" T. Ts. & Partner Rechtsanwälte "`(ORG)
- `T.`(PER)
- `Ts.`(PER)
- `M.`(PER)
- `T.`(PER)
- `T.`(PER)

**Example 58** (doc_id: `57737`) (sent_id: `57737`)


Kommanditisten der Beschwerdeführerin waren neben der H. - B. Brauerei GmbH eine Stiftung , vier Kommanditgesellschaften , eine weitere Gesellschaft mit beschränkter Haftung und natürliche Personen .

**False Positives:**

- `B. Brauerei` — partial — pred is substring of gold: `H. - B. Brauerei GmbH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `H. - B. Brauerei GmbH`(ORG)

**Example 59** (doc_id: `57762`) (sent_id: `57762`)


B. Die Vorlage ist unzulässig .

**False Positives:**

- `B. Die Vorlage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 60** (doc_id: `57795`) (sent_id: `57795`)


I. Auf die am 26. Januar 2012 beim Deutschen Patent- und Markenamt eingereichte Patentanmeldung ist die Erteilung des Patents 10 2012 201 128 mit der Bezeichnung „ Verfahren , Steuergerät und Speichermedium zur Steuerung einer Harnstoffinjektion bei niedrigen Abgastemperaturen unter Berücksichtigung des Harnstoffgehalts “ am 17. Januar 2013 veröffentlicht worden .

**False Positives:**

- `I. Auf` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschen Patent- und Markenamt`(ORG)

**Example 61** (doc_id: `57822`) (sent_id: `57822`)


I. Die Klägerin und Revisionsbeklagte ( Klägerin ) ist ein Versicherungsverein auf Gegenseitigkeit .

**False Positives:**

- `I. Die Klägerin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 62** (doc_id: `57871`) (sent_id: `57871`)


B. Die Vorlage ist zulässig .

**False Positives:**

- `B. Die Vorlage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 63** (doc_id: `57890`) (sent_id: `57890`)


I. Die Buchstabenfolge

**False Positives:**

- `I. Die Buchstabenfolge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 64** (doc_id: `57977`) (sent_id: `57977`)


A. Die Anträge der Kläger auf Bewilligung von Prozesskostenhilfe und Beiordnung ihres Verfahrensbevollmächtigten für das Verfahren vor dem Bundesverwaltungsgericht werden abgelehnt , weil die Rechtsverfolgung - wie sich aus den nachstehenden Gründen ergibt - keine hinreichende Aussicht auf Erfolg bietet ( § 166 VwGO i. V. m. §§ 114 , 121 Abs. 1 ZPO ) .

**False Positives:**

- `A. Die Anträge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Bundesverwaltungsgericht`(ORG)
- `§ 166 VwGO`(NRM)
- `§§ 114 , 121 Abs. 1 ZPO`(NRM)

**Example 65** (doc_id: `58228`) (sent_id: `58228`)


I. Das am 25. Juli 2012 angemeldete Zeichen

**False Positives:**

- `I. Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 66** (doc_id: `58268`) (sent_id: `58268`)


Ferner ist es als bloßes Gestaltungsmittel , z.B. als sog. „ Eyecatcher “ werbeüblich ( vgl. BPatG , Beschluss vom 27. 01. 2009 , 27 W ( pat ) 46/09 – LIVE ! SPEAKER ; Beschluss vom 13. 09. 2006 , 29 W ( pat ) 68/04 – REZEPTE pur EINFACH ! KOCHEN ) wie auch als Ersetzung des Buchstaben „ I / i “ ( vgl. z.B. Werbeaussage „ W ! R S ! ND DABE ! “ unter www.bw-stiftung.de) .

**False Positives:**

- `B. Werbeaussage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BPatG , Beschluss vom 27. 01. 2009 , 27 W ( pat ) 46/09 – LIVE ! SPEAKER`(RS)
- `Beschluss vom 13. 09. 2006 , 29 W ( pat ) 68/04 – REZEPTE pur EINFACH ! KOCHEN`(RS)

**Example 67** (doc_id: `58311`) (sent_id: `58311`)


A. Die Verfassungsbeschwerde betrifft die Fragen , ob die Einführung der Gewerbesteuerpflicht für Gewinne aus der Veräußerung von Anteilen an einer Mitunternehmerschaft durch § 7 Satz 2 Nr. 2 GewStG im Juli 2002 und das rückwirkende Inkraftsetzen dieser Vorschrift für den Erhebungszeitraum 2002 verfassungsrechtlich zulässig sind .

**False Positives:**

- `A. Die Verfassungsbeschwerde` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 7 Satz 2 Nr. 2 GewStG`(NRM)

**Example 68** (doc_id: `58319`) (sent_id: `58319`)


I. Die Bezeichnung

**False Positives:**

- `I. Die Bezeichnung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 69** (doc_id: `58430`) (sent_id: `58430`)


Die Beigeladene zu 7. erhielt einen Befreiungsbescheid bezogen auf eine Tätigkeit bei einem Rechtsanwalt in L. Jedoch wechselten beide Beigeladenen im Dezember 1990 bzw Dezember 1992 Beschäftigung und Arbeitgeber und begannen ihre Tätigkeiten als Einzelentscheider für die Klägerin .

**False Positives:**

- `L. Jedoch` — partial — gold is substring of pred: `L.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `L.`(LOC)

**Example 70** (doc_id: `58457`) (sent_id: `58457`)


A. Die auf die Abweisung der Anträge zu 1. , zu 3. und zu 4. beschränkte Revision des Klägers ist unbegründet .

**False Positives:**

- `A. Die` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 71** (doc_id: `58476`) (sent_id: `58476`)


I. Der bei der beklagten Krankenkasse versicherte Kläger ist mit seinem Begehren auf Versorgung mit einer subkutanen Mastektomie und angleichender Liposuktion bei der Beklagten und den Vorinstanzen ohne Erfolg geblieben .

**False Positives:**

- `I. Der` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 72** (doc_id: `58606`) (sent_id: `58606`)


I. Die Verfassungsbeschwerde betrifft den Anspruch einer schwerbehinderten und in ihrer Bewegungsfähigkeit erheblich beeinträchtigten Empfängerin von Grundleistungen zum Lebensunterhalt nach § 3 Asylbewerberleistungsgesetz ( AsylbLG ) auf unentgeltliche Beförderung im öffentlichen Personennahverkehr ohne die bis 31. Dezember 2017 in § 145 Abs. 1 Satz 3 Sozialgesetzbuch Neuntes Buch ( SGB IX ) , ab 1. Januar 2018 in § 228 Abs. 2 SGB IX vorgesehene Kostenbeteiligung .

**False Positives:**

- `I. Die Verfassungsbeschwerde` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 3 Asylbewerberleistungsgesetz`(NRM)
- `AsylbLG`(NRM)
- `§ 145 Abs. 1 Satz 3 Sozialgesetzbuch Neuntes Buch`(NRM)
- `SGB IX`(NRM)
- `§ 228 Abs. 2 SGB IX`(NRM)

**Example 73** (doc_id: `58619`) (sent_id: `58619`)


D. Die Kostenentscheidung folgt aus § 23 Abs. 4 Satz 5 DesignG i. V. m. § 84 Abs. 2 Satz 2 PatG , § 97 Abs. 1 ZPO .

**False Positives:**

- `D. Die Kostenentscheidung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 23 Abs. 4 Satz 5 DesignG`(NRM)
- `§ 84 Abs. 2 Satz 2 PatG`(NRM)
- `§ 97 Abs. 1 ZPO`(NRM)

**Example 74** (doc_id: `58634`) (sent_id: `58634`)


In der Praxis laufen diese Programme ( z.B. Mailprogramme oder Datenbankprogramme ) meist gesammelt auf bestimmten Rechnern , weshalb diese Rechner umgangssprachlich auch als „ Server “ ( z.B. Mailserver , Datenbankserver ) bezeichnet werden .

**False Positives:**

- `B. Mailprogramme` — no gold match — likely missing annotation
- `B. Mailserver` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 75** (doc_id: `58638`) (sent_id: `58638`)


I. Der Kläger hat vor dem Sozialgericht ( SG ) Braunschweig Ansprüche auf Überprüfung seines Anspruchs auf Leistungen nach dem Sozialgesetzbuch Zwölftes Buch - Sozialhilfe - ( SGB XII ) gegenüber der Beklagten geltend gemacht .

**False Positives:**

- `I. Der Kläger` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Sozialgericht ( SG ) Braunschweig`(ORG)
- `Sozialgesetzbuch Zwölftes Buch - Sozialhilfe -`(NRM)
- `SGB XII`(NRM)

**Example 76** (doc_id: `58657`) (sent_id: `58657`)


I. Die Klage ist zulässig .

**False Positives:**

- `I. Die Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 77** (doc_id: `58774`) (sent_id: `58774`)


Streitgegenstand einer Vollstreckungsgegenklage ist nach wohl überwiegender Meinung die Unzulässigkeit der Zwangsvollstreckung aus dem Titel wegen der geltend gemachten Einwendungen ( vgl. BGH , Urteil vom 8. Juni 2005 - XII ZR 294/02 - , juris , Rn. 8 ; Spohnheimer , in : Wieczorek / Schütze , ZPO , 4. Aufl. 2016 , § 767 Rn. 34 und 94 ; Seiler , in : Thomas / Putzo , ZPO , 38. Aufl. 2017 , § 767 Rn. 3 , a. A. wohl z.B. Schellhammer , ZPO , 15. Aufl. 2016 , Rn. 219 ) .

**False Positives:**

- `B. Schellhammer` — positional overlap with gold: `Schellhammer , ZPO , 15. Aufl. 2016 , Rn. 219`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Urteil vom 8. Juni 2005 - XII ZR 294/02 - , juris , Rn. 8`(RS)
- `Spohnheimer , in : Wieczorek / Schütze , ZPO , 4. Aufl. 2016 , § 767 Rn. 34 und 94`(LIT)
- `Seiler , in : Thomas / Putzo , ZPO , 38. Aufl. 2017 , § 767 Rn. 3`(LIT)
- `Schellhammer , ZPO , 15. Aufl. 2016 , Rn. 219`(LIT)

**Example 78** (doc_id: `58805`) (sent_id: `58805`)


I. Der Kläger begehrt in der Hauptsache die Feststellung eines Grades der Behinderung ( GdB ) von 80 und die Zuerkennung des Merkzeichens G. Dieses Begehren hat das LSG mit Urteil vom 16. 11. 2017 verneint .

**False Positives:**

- `I. Der Kläger` — no gold match — likely missing annotation
- `G. Dieses Begehren` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 79** (doc_id: `58821`) (sent_id: `58821`)


I. Der Kläger und Revisionsbeklagte ( Kläger ) war im Jahr 2011 ( Streitjahr ) Eigentümer des Grundstücks in X , Y-Straße ... ( Grundstück ) , das er bis März 2020 steuerpflichtig an die A ( Pächterin ) verpachtet hatte .

**False Positives:**

- `I. Der Kläger` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `X`(LOC)
- `Y-Straße ...`(LOC)
- `A`(PER)

**Example 80** (doc_id: `58866`) (sent_id: `58866`)


I. Die Klägerin und Revisionsklägerin ( Klägerin ) unterhielt in den Streitjahren ( 2005 und 2006 ) einen Betrieb gewerblicher Art " Schwimmbäder " ( BgA ) .

**False Positives:**

- `I. Die Klägerin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 81** (doc_id: `59070`) (sent_id: `59070`)


I. Die Verfassungsbeschwerde betrifft die Auslieferung des Beschwerdeführers , eines serbischen Staatsangehörigen , nach Ungarn zur Strafverfolgung auf Grundlage eines Europäischen Haftbefehls vom 2. November 2017 .

**False Positives:**

- `I. Die Verfassungsbeschwerde` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Ungarn`(LOC)

**Example 82** (doc_id: `59076`) (sent_id: `59076`)


Die Präklusion nach § 767 Abs. 2 ZPO greift daher zum Beispiel nicht ein bei Titeln ohne Rechtskraftwirkung , nämlich Prozessvergleichen , vollstreckbaren Urkunden und Anwaltsvergleichen ( vgl. z.B. Herget , in : Zöller , ZPO , 32. Aufl. 2018 , § 767 Rn. 20 m. w. N. ; Seiler , in : Thomas / Putzo , ZPO , 38. Aufl. 2017 , § 767 Rn. 25 ) ; insoweit sind alle Einwendungen zulässig , auch anspruchshindernde .

**False Positives:**

- `B. Herget` — positional overlap with gold: `Herget , in : Zöller , ZPO , 32. Aufl. 2018 , § 767 Rn. 20`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 767 Abs. 2 ZPO`(NRM)
- `Herget , in : Zöller , ZPO , 32. Aufl. 2018 , § 767 Rn. 20`(LIT)
- `Seiler , in : Thomas / Putzo , ZPO , 38. Aufl. 2017 , § 767 Rn. 25`(LIT)

**Example 83** (doc_id: `59194`) (sent_id: `59194`)


Nach der Rechtsprechung des BGH kann die Annahme einer allgemeinen Werbeaussage des Markenwortes nämlich nicht auf Beispiele gestützt werden , in denen das Markenwort nicht in Alleinstellung , sondern stets im Zusammenhang mit anderen Worten benutzt wird , aus denen sich seine werbliche Bedeutung erschließt ( BGH a. a. O. Rn. 24 – OUI ) .

**False Positives:**

- `O. Rn` — partial — pred is substring of gold: `BGH a. a. O. Rn. 24 – OUI`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH`(ORG)
- `BGH a. a. O. Rn. 24 – OUI`(RS)

**Example 84** (doc_id: `59230`) (sent_id: `59230`)


I. Der Kläger begehrt eine Entschädigung wegen eines nach seiner Ansicht unangemessen langen Gerichtsverfahrens vor dem SG Gotha ( S 13 AL 118/98 ) und dem Thüringer LSG ( L 3 AL 229/00 ) .

**False Positives:**

- `I. Der Kläger` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `SG Gotha ( S 13 AL 118/98 )`(RS)
- `Thüringer LSG ( L 3 AL 229/00 )`(RS)

**Example 85** (doc_id: `59231`) (sent_id: `59231`)


C. Im Hinblick auf die aufgezeigten Bedenken , ob nach der von Rechts wegen gebotenen Aufgabe der sog. „ geschmacksmusterrechtlichen Unterkombination “ überhaupt noch eine Auslegung des Schutzgegenstands eines eingetragenen Designs auf Grundlage der Schnittmenge der allen Darstellungen gemeinsamen Merkmale in Betracht kommt , war nach § 23 Abs. 5 DesignG i. V. m. § 100 Abs. 2 Nr. 1 PatG die Zulassung der Rechtsbeschwerde veranlasst , da es sich insoweit um eine Rechtsfrage von grundsätzlicher Bedeutung handelt .

**False Positives:**

- `C. Im Hinblick` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 23 Abs. 5 DesignG`(NRM)
- `§ 100 Abs. 2 Nr. 1 PatG`(NRM)

**Example 86** (doc_id: `59324`) (sent_id: `59324`)


I. Die Klägerin und Beschwerdeführerin ( Klägerin ) erzielte mit drei Unternehmen Einkünfte aus Gewerbebetrieb .

**False Positives:**

- `I. Die Klägerin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 87** (doc_id: `59436`) (sent_id: `59436`)


Die Frage , wie die auch angesprochenen Verbraucher das angemeldete Zeichen verstehen werden , kann letztlich als nicht entscheidungserheblich dahingestellt bleiben , da wie oben dargelegt , schon das Verständnis der Fachkreise für sich genommen ausschlaggebend ist ( Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 41 ; zum Verständnis fremdsprachiger Begriffe seitens des Durchschnittsverbrauchers , das nicht zu gering veranschlagt werden darf vgl. Ströbele / Hacker , a. a. O. Rn. 168 ) .

**False Positives:**

- `O. Rn` — partial — pred is substring of gold: `Ströbele / Hacker , a. a. O. Rn. 168`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 41`(LIT)
- `Ströbele / Hacker , a. a. O. Rn. 168`(LIT)

**Example 88** (doc_id: `59559`) (sent_id: `59559`)


I. Das Wortzeichen Wohlfühlfarbe ist am 1. März 2016 zur Eintragung als Marke in das vom Deutschen Patent- und Markenamt geführte Register angemeldet worden für folgende Waren :

**False Positives:**

- `I. Das Wortzeichen Wohlfühlfarbe` — partial — gold is substring of pred: `Wohlfühlfarbe`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Wohlfühlfarbe`(ORG)
- `Deutschen Patent- und Markenamt`(ORG)

**Example 89** (doc_id: `59599`) (sent_id: `59599`)


Denn die D4 beschäftigt sich mit der Erzeugung von Feinpartikeln und erhält diese Feinpartikel durch Kollision zweier Grobpartikel-haltiger Flüssigkeitsstrahlen , die unter extrem hohen Druck aufeinander prallen und dabei in Turbulenz geraten , wobei die dafür geeignete Vorrichtung Blöcke aus abriebfesten Material wie z.B. Keramik aufweist ( vgl. D4 Sp. 1 Abs. 1 , Sp. 1/2 spaltenübergr. Abs. , Sp. 4 2. vollst. Abs. ) .

**False Positives:**

- `B. Keramik` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 90** (doc_id: `59650`) (sent_id: `59650`)


D20 I. C. Madsen et al. , “ Description and survey of methodologies for the determination of amorphous content via X-ray powder diffraction ” , Z. Kristallgr . 2011 , 226 , Seiten 944 bis 955

**False Positives:**

- `C. Madsen` — partial — pred is substring of gold: `I. C. Madsen`
- `Z. Kristallgr` — no gold match — likely missing annotation

> overlaps gold: 1  |  likely missing annotation: 1

**Gold Entities:**

- `I. C. Madsen`(PER)

**Example 91** (doc_id: `59680`) (sent_id: `59680`)


Ausgehend hiervon besitzen Wortzeichen dann keine Unterscheidungskraft , wenn ihnen die angesprochenen Verkehrskreise lediglich einen im Vordergrund stehenden beschreibenden Begriffsinhalt zuordnen ( EuGH GRUR 2004 , 674 , Rn. 86 – Postkantoor ; BGH GRUR 2012 , 270 Rn. 11 – Link economy ) oder wenn diese aus gebräuchlichen Wörtern oder Wendungen der deutschen Sprache oder einer bekannten Fremdsprache bestehen , die vom Verkehr – etwa auch wegen einer entsprechenden Verwendung in der Werbung – stets nur als solche und nicht als Unterscheidungsmittel verstanden werden ( BGH a. a. O. Rn. 12 – OUI ; GRUR 2014 , 872 Rn. 21 – Gute Laune Drops ) .

**False Positives:**

- `O. Rn` — partial — pred is substring of gold: `BGH a. a. O. Rn. 12 – OUI`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2004 , 674 , Rn. 86 – Postkantoor`(RS)
- `BGH GRUR 2012 , 270 Rn. 11 – Link economy`(RS)
- `BGH a. a. O. Rn. 12 – OUI`(RS)
- `GRUR 2014 , 872 Rn. 21 – Gute Laune Drops`(RS)

</details>

---

## `Anonymized Initials with Ellipsis and Numbers`

**F1:** 0.032 | **Precision:** 0.231 | **Recall:** 0.017  

**Format:** `regex`  
**Rule ID:** `b224d30a`  
**Description:**
Captures anonymized names with an ellipsis (e.g., 'P …', 'B1 …', 'G1 …') following role indicators or in legal contexts.

**Content:**
```
\b([A-Z]\d*\s+…)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.231 | 0.017 | 0.032 | 26 | 6 | 20 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 6 | 20 | 338 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53829`) (sent_id: `53829`)


An der letzten Voraussetzung , der Eignung des an Patentanwalt K … Wohnung bzw. Kanzlei angebrachten Briefkastens zur sicheren Aufbewahrung , hat es aber zum fraglichen Zeitpunkt der Einlegung des den Prüfungsbescheid enthaltenden Umschlags gefehlt .

| Predicted | Gold |
|---|---|
| `K …` | `K …` |

**Example 1** (doc_id: `56141`) (sent_id: `56141`)


Die Widersprechende zu 2. hat mit den Schriftsätzen vom 23. September 2014 sowie - im Beschwerdeverfahren - 4. August 2017 Unterlagen zur Benutzung der Widerspruchsmarke 30 2008 062 715 NIDO einschließlich einer eidesstattlichen Versicherung des Herrn H … vom 19. September 2014 ( An-lage W13 ) vorgelegt ( Anlagen W9 - W16 sowie W21 - W30 ) .

| Predicted | Gold |
|---|---|
| `H …` | `H …` |

**Missed by this rule (FN):**

- `NIDO` (ORG)

**Example 2** (doc_id: `58746`) (sent_id: `58746`)


Auch die Anlagen zur eidesstattlichen Versicherung von Herrn N … seien zur Glaubhaftmachung einer markenmäßigen Benutzung ungeeignet .

| Predicted | Gold |
|---|---|
| `N …` | `N …` |

**Example 3** (doc_id: `58765`) (sent_id: `58765`)


Zum einen sei Herr N … erst seit Dezember 2016 Vorstandsvorsitzender der Widersprechenden und könne daher nicht mit den erklärten Verhältnissen im jeweils maßgeblichen Benutzungszeitraum vertraut gewesen sein .

| Predicted | Gold |
|---|---|
| `N …` | `N …` |

**Example 4** (doc_id: `59898`) (sent_id: `59898`)


Weiterhin macht er geltend , Patentanwalt K … sei wegen einer psychischen Erkrankung zur Zeit der Zustellversuche des DPMA geschäftsunfähig nach § 104 Abs. 2 BGB gewesen , weshalb die Zustellungen unwirksam seien .

| Predicted | Gold |
|---|---|
| `K …` | `K …` |

**Missed by this rule (FN):**

- `DPMA` (ORG)
- `§ 104 Abs. 2 BGB` (NRM)

**Example 5** (doc_id: `59964`) (sent_id: `59964`)


bb ) Der Senat war auch nicht gehalten , den Zeugen H … über das Beweisangebot der Einsprechenden hinaus dazu zu vernehmen , dass durch Messungen an dem behauptet vorbenutzten Schalter auch die weiteren , nicht im Beweisantrag aufgeführten Merkmale des erteilten Patentanspruchs 1 für den Fachmann feststellbar seien .

| Predicted | Gold |
|---|---|
| `H …` | `H …` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53472`) (sent_id: `53472`)


Sie ist dann allerdings die einzige Beschwerdeführerin , denn für die Beschwerde der R … GmbH in S … , fehlt es an der Zahlung der erforderlichen weiteren , zweiten Beschwerdegebühr , so dass diese als nicht eingelegt gilt .

**False Positives:**

- `R …` — partial — pred is substring of gold: `R … GmbH`
- `S …` — type mismatch — same span as gold: `S …`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `R … GmbH`(ORG)
- `S …`(LOC)

**Example 1** (doc_id: `54492`) (sent_id: `54492`)


Der Kontaktsockel sei auf der Messe „ Semicon Europe 2006 “ in München ausgestellt gewesen und zudem an einen Kunden , die Firma A … Inc. , geliefert worden .

**False Positives:**

- `A …` — partial — pred is substring of gold: `A … Inc.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europe`(LOC)
- `München`(LOC)
- `A … Inc.`(ORG)

**Example 2** (doc_id: `54517`) (sent_id: `54517`)


b. Ausgehend davon hat die Widersprechende zu 1 zunächst eine rechtserhaltende Benutzung der Widerspruchsmarke NIVONA gemäß Art. 15 GMV in der Gemeinschaft für den nach §§ 43 Abs. 1 Satz 2 , 125 b Nr. 4 MarkenG zum Zeitpunkt der mündlichen Verhandlung maßgeblichen Fünfjahreszeitraum September 2012 bis September 2017 ungeachtet der von ihr mit Schriftsatz vom 3. Januar 2014 für den Zeitraum 2011 bis 1. Halbjahr 2013 eingereichten Unterlagen ( Anlagen W4 - W12 ) jedenfalls mit den im Beschwerdeverfahren mit Schriftsatz vom 22. Juni 2017 ( Bl. 56 d. A. ) für den Zeitraum 2014 - 2016 eingereichten Unterlagen W14 bis W17 , insbesondere der weiteren eidesstattlichen Versicherung des Geschäftsführers W1 … vom 19. Juni 2017 ( Anlage W19 , Bl. 66 d. A. ) für die Waren „ elektrische Kaffeevollautomaten , elektrische Kaffeemühlen ; Reinigungsmittel und Entkalkungsmittel ( Reinigungstabs , CreamCleaner und Entkalker ) sowie Milchbehälter “ glaubhaft gemacht .

**False Positives:**

- `W1 …` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `NIVONA`(ORG)
- `Art. 15 GMV`(NRM)
- `§§ 43 Abs. 1 Satz 2 , 125 b Nr. 4 MarkenG`(NRM)

**Example 3** (doc_id: `54753`) (sent_id: `54753`)


Mit dieser rechtzeitigen Gebührenzahlung ist die Beschwerde der B … Aktiengesellschaft wirksam erhoben .

**False Positives:**

- `B …` — partial — pred is substring of gold: `B … Aktiengesellschaft`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `B … Aktiengesellschaft`(ORG)

**Example 4** (doc_id: `55099`) (sent_id: `55099`)


Die Gegenstände der Ansprüche 1 , 9 und 11 haben somit in der Fa. R … GmbH & Co. KG als nicht öffentlich zugänglich im Sinne des § 3 ( 1 ) , 2 PatG zu gelten , so dass insoweit keine offenkundige Vorbenutzung vorliegt .

**False Positives:**

- `R …` — partial — pred is substring of gold: `R … GmbH & Co. KG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `R … GmbH & Co. KG`(ORG)
- `§ 3 ( 1 ) , 2 PatG`(NRM)

**Example 5** (doc_id: `55422`) (sent_id: `55422`)


Anlage A3 Handzeichnung zum Ablauf von Werksbesichtigungen bei der R … GmbH & Co. KG ,

**False Positives:**

- `R …` — partial — pred is substring of gold: `R … GmbH & Co. KG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `R … GmbH & Co. KG`(ORG)

**Example 6** (doc_id: `56144`) (sent_id: `56144`)


Denn bereits zu Beginn des Lizenzzeitraums produzierte und vertrieb der M … seit fast 10 Jahren das Medikament Isentress mit dem bis 2014 einzigen ungeboosteten Integraseinhibitor ( Raltegravir ) , mit dem Umsätze in Höhe von jährlich ca. … US- $ weltweit , in Deutschland in Höhe von ca. … € erzielt worden sind ( von der Beklagten als „ Blockbuster “ bezeichnet ) .

**False Positives:**

- `M …` — type mismatch — same span as gold: `M …`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `M …`(ORG)
- `Deutschland`(LOC)

**Example 7** (doc_id: `56697`) (sent_id: `56697`)


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

**Example 8** (doc_id: `58075`) (sent_id: `58075`)


Jedoch trägt sie vor , dass sämtliche Besucher , die die Fa. R … GmbH & Co. KG besuchten , in der Vergangenheit und bis heute stets zur Geheimhaltung verpflichtet worden seien .

**False Positives:**

- `R …` — partial — pred is substring of gold: `R … GmbH & Co. KG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `R … GmbH & Co. KG`(ORG)

**Example 9** (doc_id: `58293`) (sent_id: `58293`)


Der Senat hat aufgrund einer Zusammenschau der Rechnung D4 , des zugehörigen Lieferscheins D8 und des Produktprogramms D3 keinen Zweifel daran , dass ein aus dem Produktprogramm Sicherheitstechnik der Firma E … bekannter Sicherheitsschalter am 18. / 20. 02. 2009 an die Firma C … GmbH in E … verkauft und auch geliefert wurde .

**False Positives:**

- `E …` — type mismatch — same span as gold: `E …`
- `C …` — partial — pred is substring of gold: `C … GmbH`
- `E …` — similar text (different position): `E …`

> overlaps gold: 3  |  likely missing annotation: 0

**Gold Entities:**

- `E …`(ORG)
- `C … GmbH`(ORG)
- `E …`(LOC)

**Example 10** (doc_id: `58709`) (sent_id: `58709`)


ab ) Auch die geltend gemachte offenkundige Vorbenutzung in der Fa. S2 … GmbH kann nicht berücksichtigt werden , denn sie ist hinsichtlich dessen , was angeblich offenkundig vorbenutzt wurde , nicht hinreichend substantiiert .

**False Positives:**

- `S2 …` — partial — pred is substring of gold: `S2 … GmbH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S2 … GmbH`(ORG)

**Example 11** (doc_id: `58823`) (sent_id: `58823`)


Die Anmelderin L … LLC als Vorgängerin der Beklagten sei zum Zeitpunkt der Anmeldung des Streitpatents ausweislich der Dokumente HLNK2 bis HLNK 4 Rechtsnachfolgerin der Anmelder der Prioritätsanmeldung gewesen .

**False Positives:**

- `L …` — partial — pred is substring of gold: `L … LLC`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `L … LLC`(ORG)

**Example 12** (doc_id: `59314`) (sent_id: `59314`)


Die Einsprechende stützt ihre Argumentation bezüglich der offenkundigen Vorbenutzung u. a. auf die Druckschriften D3 , D4 , D5 und D8 , die einen Verkauf eines aus dem Produktprogramm Sicherheitstechnik der Fa. E … bekannten Sicherheitsschalters ohne Geheimhaltungspflicht belegen sollen .

**False Positives:**

- `E …` — type mismatch — same span as gold: `E …`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `E …`(ORG)

**Example 13** (doc_id: `59583`) (sent_id: `59583`)


- Q1 … Q4 Überwachte Halbleiterausgänge

**False Positives:**

- `Q1 …` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Names after 'von' (Noble/Preposition)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `030b825e`  
**Description:**
Captures names following the preposition 'von', excluding common non-name words and titles.

**Content:**
```
\bvon\s+([A-Z][a-zäöüß]+)(?!\s+(?:Frau|Herr|Dr\.?|Prof\.?|Anwalt|Gericht|Firma|Behörde|Polizei|Steuern|Umsatz|Kabel|Anlage|Umsatzsteuer|Rechtsanwältin|Rechtsanwalt|Ackerstatusrechten|Höchst-Schmerzgrenze|Marken-Beschwerdesenat|Rechtsanwalt|...|von|der|die|das|ein|eine|mit|für|auf|in|an|bei|zu|nach|vor|über|unter|durch|ohne|gegen|seit|statt|wegen|um|bis|neben|zwischen|entlang|außer|neben|ab))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 702 | 0 | 702 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 702 | 345 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53415`) (sent_id: `53415`)


Die Form der Benutzung ist insoweit jeweils glaubhaft gemacht durch die als Anlagen W14 bis W17 vorgelegten Produktkataloge aus dem vorliegend relevanten Benutzungszeitraum mit Abbildungen einer Vielzahl von ( elektrischen ) Kaffeemaschinen , Kaffeemühlen , Milchbehälter ( „ Milchcooler “ ) sowie Verpackungen / Behälter von Reinigungstabs , CreamCleaner und Entkalker , auf denen die Widerspruchsmarke NIVONA in einer ohne weitere besondere graphische Ausgestaltungselemente enthaltenden und damit in einer den kennzeichnenden Charakter der eingetragenen Marke nicht verändernden Form i. S. von § 26 Abs. 3 MarkenG deutlich sichtbar abgebildet ist .

**False Positives:**

- `Reinigungstab` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `NIVONA`(ORG)
- `§ 26 Abs. 3 MarkenG`(NRM)

**Example 1** (doc_id: `53416`) (sent_id: `53416`)


Zu dieser Zeit sei auch noch die Berücksichtigung von Zeiten wissenschaftlicher Tätigkeit streitig gewesen .

**False Positives:**

- `Zeite` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `53425`) (sent_id: `53425`)


Insbesondere hinsichtlich der Dienstleistungen „ Vermietung von Computersoftware ; Vermietung von Webservern ; Programmierung und Einstellung von Datenverarbeitungsprogrammen “ könne es sich um solche Dienstleistungen handeln , die für ein „ real Targeting “ bestimmt , geeignet und notwendig seien oder damit in unmittelbarem Zusammenhang stehen könnten .

**False Positives:**

- `Computersoftwar` — no gold match — likely missing annotation
- `Webserver` — no gold match — likely missing annotation
- `Datenverarbeitungsprogramme` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 3

**Example 3** (doc_id: `53428`) (sent_id: `53428`)


Hierzu zählt auch der Schutz vor der Erhebung und Weitergabe von Befunden über den Gesundheitszustand .

**False Positives:**

- `Befunde` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `53441`) (sent_id: `53441`)


Die Beteiligten streiten über die Erstattung bzw Übernahme von Kosten für die Entsorgung von Inkontinenzmaterial .

**False Positives:**

- `Koste` — no gold match — likely missing annotation
- `Inkontinenzmaterial` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 5** (doc_id: `53443`) (sent_id: `53443`)


Es hat einerseits angenommen , ein institutioneller Rechtsmissbrauch sei indiziert , da die Parteien seit dem 25. August 2008 insgesamt 22 befristete Arbeitsverträge abgeschlossen hätten und damit die Anzahl von Vertragsverlängerungen den in § 14 Abs. 2 Satz 1 TzBfG genannten Wert um mehr als das Fünffache überschreite .

**False Positives:**

- `Vertragsverlängerunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 14 Abs. 2 Satz 1 TzBfG`(NRM)

**Example 6** (doc_id: `53455`) (sent_id: `53455`)


c ) Die in Teilen der Literatur verbreitete Rechtsauffassung , wonach allein der Zufluss von Kindergeld auf dem Konto eines Zulageberechtigten genüge , einen Anspruch auf Kinderzulage nach § 85 Abs. 1 Satz 1 EStG a. F. zu erlangen ( so wohl Killat in Herrmann / Heuer / Raupach , § 85 EStG Rz 6 ; Myßen / Obermair , in : Kirchhof / Söhn / Mellinghoff , EStG , § 85 Rz D 18 ; Schmidt / Wacker , EStG , 36. Aufl. , § 85 Rz 2 ) , wäre hingegen missbrauchsanfällig und kann zu Ergebnissen führen , die mit dem Sinn und Zweck des § 85 Abs. 1 Satz 1 EStG a. F. nicht vereinbar wären .

**False Positives:**

- `Kindergel` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 85 Abs. 1 Satz 1 EStG a. F.`(NRM)
- `Killat in Herrmann / Heuer / Raupach , § 85 EStG Rz 6`(LIT)
- `Myßen / Obermair , in : Kirchhof / Söhn / Mellinghoff , EStG , § 85 Rz D 18`(LIT)
- `Schmidt / Wacker , EStG , 36. Aufl. , § 85 Rz 2`(LIT)
- `§ 85 Abs. 1 Satz 1 EStG a. F.`(NRM)

**Example 7** (doc_id: `53459`) (sent_id: `53459`)


Als Aufgaben sind in § 3 Nr. 5 Buchst. b der Satzung die „ Verbesserung von Einkommen und Arbeitsbedingungen durch Abschluss von Tarifverträgen und Einwirkung auf die Gesetzgebung und Behörden “ genannt .

**False Positives:**

- `Einkomme` — no gold match — likely missing annotation
- `Tarifverträge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 8** (doc_id: `53464`) (sent_id: `53464`)


Zum anderen steht der Annahme fehlender Unterscheidungskraft nicht entgegen , dass der Begriff „ ruheyoga “ bislang nicht lexikalisch nachweisbar ist ; ebenso wenig ist von Bedeutung , dass das Zusammenschreiben der Worte „ ruhe “ und „ yoga “ in einem Wort möglicherweise nicht üblich ist .

**False Positives:**

- `Bedeutun` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 9** (doc_id: `53482`) (sent_id: `53482`)


In Teil F Ziffer 3.5 ist bestimmt : " Die Partner der Gesamtverträge beschließen für Neuzulassungen von Vertragsärzten und Umwandlung der Kooperationsform Anfangs- und Übergangsregelungen .

**False Positives:**

- `Vertragsärzte` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 10** (doc_id: `53503`) (sent_id: `53503`)


Vor die ordentlichen Gerichte hingegen gehören nach § 13 GVG ua die bürgerlichen Rechtsstreitigkeiten , die Familiensachen und die Angelegenheiten der freiwilligen Gerichtsbarkeit ( Zivilsachen ) , für die nicht entweder die Zuständigkeit von Verwaltungsbehörden oder Verwaltungsgerichten begründet ist oder aufgrund von Vorschriften des Bundesrechts besondere Gerichte bestellt oder zugelassen sind .

**False Positives:**

- `Verwaltungsbehörde` — no gold match — likely missing annotation
- `Vorschrifte` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `§ 13 GVG`(NRM)

**Example 11** (doc_id: `53508`) (sent_id: `53508`)


Beruhen vielmehr die Differenzen zwischen den Auffassungen von Sachverständigen darauf , dass diese von verschiedenen tatsächlichen Annahmen ausgehen , dann muss der Tatrichter , ggf nach weiterer Aufklärung , die für seine Überzeugungsbildung maßgebenden Tatsachen feststellen oder begründen , weshalb und zu wessen Lasten sie beweislos geblieben sind ( vgl BGH Urteil vom 23. 9. 1986 - VI ZR 261/85 ) .

**False Positives:**

- `Sachverständige` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BGH Urteil vom 23. 9. 1986 - VI ZR 261/85`(RS)

**Example 12** (doc_id: `53512`) (sent_id: `53512`)


Das Landgericht hat den Angeklagten vom Vorwurf des Vorenthaltens und Veruntreuens von Arbeitsentgelt sowie der Steuerhinterziehung jeweils in 32 Fällen aus tatsächlichen Gründen freigesprochen .

**False Positives:**

- `Arbeitsentgel` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 13** (doc_id: `53514`) (sent_id: `53514`)


Auf die Revision der Klägerin wird das Urteil des Hessischen Finanzgerichts vom 24. März 2015 4 K 1187/11 aufgehoben , soweit es die Klage gegen die Nachforderung von Kapitalertragsteuer abweist ; die Nachforderungsbescheide vom 6. September 2010 über Kapitalertragsteuer für die Jahre 2005 und 2006 in Gestalt der Einspruchsentscheidung vom 8. April 2011 werden dahin geändert , dass die Kapitalertragsteuer auf 0 € festgesetzt wird .

**False Positives:**

- `Kapitalertragsteue` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Urteil des Hessischen Finanzgerichts vom 24. März 2015 4 K 1187/11`(RS)

**Example 14** (doc_id: `53523`) (sent_id: `53523`)


( Vertragsärztliche Versorgung - Ermächtigung von Sozialpädiatrischen Zentren - keine analoge Anwendung von § 118 Abs 4 SGB 5 oder § 24 Abs 3 Ärzte ZV bzgl Errichtung einer Zweigstelle bzw -praxis )

**False Positives:**

- `Sozialpädiatrische` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 118 Abs 4 SGB 5`(NRM)
- `§ 24 Abs 3 Ärzte ZV`(NRM)

**Example 15** (doc_id: `53524`) (sent_id: `53524`)


Der BFH führte aus , die handelsrechtliche Zurechnung von Vermögensgegenständen entspreche im Wesentlichen der Regelung des § 39 Abs. 2 Nr. 1 Satz 1 AO ( z.B. BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1. ; vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a ) .

**False Positives:**

- `Vermögensgegenstände` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BFH`(ORG)
- `§ 39 Abs. 2 Nr. 1 Satz 1 AO`(NRM)
- `BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1.`(RS)
- `vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a`(RS)

**Example 16** (doc_id: `53533`) (sent_id: `53533`)


Gemäß den Beispielen I-4 bis I-22 aus Tabelle III. die alle einer identischen ersten Wärmebehandlung für die Keimbildung mit 645 ° C / 1 h ausgesetzt waren , wurde je nach dem gewählten Temperaturprofil der zweiten Wärmebehandlung entweder nur Lithiummetasilicat , eine Mischung von Lithiummetasilicat und Lithiumdisilicat oder nur Lithiumdisilicat erhalten ( vgl. D3 , S. 385 , Tab. I , S. 386 Tab. III. .

**False Positives:**

- `Lithiummetasilica` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 17** (doc_id: `53535`) (sent_id: `53535`)


Das setzt einen Erfolg der Behandlungsmethode in einer für die sichere Beurteilung ausreichenden Zahl von Behandlungsfällen voraus .

**False Positives:**

- `Behandlungsfälle` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 18** (doc_id: `53559`) (sent_id: `53559`)


Amtliche Informationen kommen einem Eingriff in die Berufsfreiheit jedenfalls dann gleich , wenn sie direkt auf die Marktbedingungen konkret individualisierter Unternehmen zielen , indem sie die Grundlagen von Konsumentscheidungen zweckgerichtet beeinflussen und die Markt- und Wettbewerbssituation zum Nachteil der betroffenen Unternehmen verändern .

**False Positives:**

- `Konsumentscheidunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `53573`) (sent_id: `53573`)


Zum einen vermag die Rüge der Nichtbeachtung von Bundesrecht bei der Auslegung und Anwendung von Landesrecht die Zulassung der Grundsatzrevision nur dann zu begründen , wenn die Auslegung einer - gegenüber dem Landesrecht als korrigierender Maßstab angeführten - bundesrechtlichen Norm ihrerseits ungeklärte Fragen von grundsätzlicher Bedeutung aufwirft .

**False Positives:**

- `Bundesrech` — no gold match — likely missing annotation
- `Landesrech` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 20** (doc_id: `53605`) (sent_id: `53605`)


Klasse 35 : Einzelhandelsdienstleistungen in den Bereichen tragbare und mit der Hand gehaltene oder handbetätigte elektronische Geräte zur Übertragung , Speicherung , Verarbeitung , Aufzeichnung und Ansicht / Überprüfung von Texten , Bildern , Audios , Videos und Daten , auch über globale Computernetzwerke , drahtlose Netzwerke und elektronische Kommunikationsnetzwerke , Computer , Tablet-Computer , Lesegeräte für elektronische Bücher , Audio- und Videogeräte , elektronische persönliche Organisierer , persönliche digitale Assistenten und Geräte für globale Positionierungssysteme und elektronische und mechanische Teile und Zubehör dafür ; Computerhardware und -software , Monitore , Displays , Drähte , Kabel , Modems , Drucker , Diskettenlaufwerke , Adapter , Adapterkarten , Kabelverbinder / Kabelanschlüsse , steckbare Anschlüsse , elektrische Stromanschlüsse , Dockstationen und Laufwerke , Batterieladegeräte , Batteriepackungen , Memorykarten und Lesegeräte für Memorykarten , Kopfhörer und Ohrhörer , Lautsprecher , Mikrophone und Headsets ( Hörsprechgarnituren ) , angepasste Behälter , Abdeckungen und Gestelle für tragbare und mit der Hand gehaltene oder handbetätigte elektronische Geräte und Computer , Fernbedienungen für tragbare und mit der Hand gehaltene oder handbetätigte elektronische Geräte und Computer , Druckereierzeugnisse , gedruckte Veröffentlichungen , Periodika , Bücher , Magazine , Newsletter , Broschüren , Hefte , Pamphlete , Handbücher , Journale , Kataloge und Sticker , Hand gehaltene oder handbetätigte Anlagen zum Spielen von elektronischen Spielen , handgehaltene oder handbetätigte elektronische Spiele und Spielapparate , Spiele , elektronische Spiele und Videospiele ; Online-Einzelhandelsdienstleistungen in den Bereichen tragbare und mit der Hand gehaltene oder handbetätigte elektronische Geräte zur Übertragung , Speicherung , Verarbeitung , Aufzeichnung und Ansicht / Überprüfung von Texten , Bildern , Audios , Videos und Daten , auch über globale Computernetzwerke , drahtlose Netzwerke und elektronische Kommunikationsnetzwerke , Computer , Tablet-Computer , Lesegeräte für elektronische Bücher , Audio- und Videogeräte , elektronische persönliche Organisierer , persönliche digitale Assistenten und Geräte für globale Positionierungssysteme und elektronische und mechanische Teile und Zubehör dafür , Computerhardware und -software , Monitore , Displays , Drähte , Kabel , Modems , Drucker , Diskettenlaufwerke , Adapter , Adapterkarten , Kabelverbinder / Kabelanschlüsse , steckbare Anschlüsse , elektrische Stromanschlüsse , Dockstationen und Laufwerke , Batterieladegeräte , Batteriepackungen , Memorykarten und Lesegeräte für Memorykarten , Kopfhörer und Ohrhörer , Lautsprecher , Mikrophone und Headsets ( Hörsprechgarnituren ) , angepasste Behälter , Abdeckungen und Gestelle für tragbare und mit der Hand gehaltene oder handbetätigte elektronische Geräte und Computer , Fernbedienungen für tragbare und mit der Hand gehaltene oder handbetätigte elektronische Geräte und Computer , Druckereierzeugnisse , gedruckte Veröffentlichungen , Periodika , Bücher , Magazine , Newsletter , Broschüren , Hefte , Pamphlete , Handbücher , Journale , Kataloge und Sticker , Hand gehaltene oder handbetätigte Anlagen zum Spielen von elektronischen Spielen , handgehaltene oder handbetätigte elektronische Spiele und Spieleapparate , Spiele , elektronische Spiele und Videospiele ; Herausgabe eines Online-Handelsinformationsverzeichnisses ; Verteilung von Werbung für andere über ein elektronisches Online-Kommunikationsnetzwerk ; Herausgabe eines recherchierbaren Online Werbeführers , der Waren und Dienstleistungen von anderen zeigt ; computerisierte Datenbank-Managementdienstleistungen , nämlich Sammeln und Systematisieren von Daten in Computerdatenbanken ; Online-Bestelldienstleistungen , nämlich verwaltungstechnische Bearbeitung ;

**False Positives:**

- `Texte` — no gold match — likely missing annotation
- `Texte` — no gold match — likely missing annotation
- `Werbun` — no gold match — likely missing annotation
- `Date` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 4

**Example 21** (doc_id: `53608`) (sent_id: `53608`)


„ Klasse 9 : Wissenschaftliche , Schifffahrts- , Vermessungs- , fotografische , Film- , optische , Wäge- , Mess- , Signal- , Kontroll- , Rettungs- und Unterrichtsapparate und -instrumente ; Apparate und Instrumente zum Leiten , Schalten , Umwandeln , Speichern , Regeln und Kontrollieren von Elektrizität ; Geräte zur Aufzeichnung , Übertragung oder Wiedergabe von Sprache , Ton oder Bild ; Computerplatten , Disketten ; Bänder ; magnetische und optische Datenträger ; Magnetplatten , -disketten ; Magnetbänder ; Disketten ; Schallplatten und CDs ; Datenträger ; Speichermedien ; Speicherkarten ; Disketten , CDs , CD-ROMs und DVDs ; Flash-Aufbewahrungsvorrichtungen ; Festplatten ; Flash-basierte Festplatten ; Speicherkarten ; elektronische Massenspeichervorrichtungen ; Computerspeicher ; Verkaufsautomaten und Mechaniken für geldbetätigte Apparate ; Registrierkassen ; Rechenmaschinen ; Taschenrechner ; Datenverarbeitungsgeräte und Computer ; Datenerfassungshardware ; speicherprogrammierbare Steuerungen ; Datenverarbeitungsgeräte und Computer , insbesondere Einplatinen- oder Einschubgeräte ; elektronische Computerbauteile ; Computer-Hardware ; Computermodule ; Computer ; eingebettete Computer ; Einplatinencomputer ; Personal Computer ( PCs ) ; Kasten-PCs ; Desktop-PCs ; Wandmontage-PCs ; Laptops ; Notebooks ; Subnotebooks ; robuste Computer sowie deren Teile und Komponenten ; in Gehäuse montierte Computer ; in Gehäuse montierte Computersystemgeräte ; Industrie-PCs ; eingebettete PCs ; Schnittstellen Mensch-Maschine ( HMI ) ; Schnittstellengeräte Mensch-Maschine , bestehend aus einem Computer und einer Anzeige ; Pult-PCs ; Microclient-Computer ; Thin-Client-Computer ; interaktive Client-Computer ; Kiosk-PCs ; Spiel-PCs ; Touchpanels ; Tastfeld-PCs ; PCs in Fahrzeugen ; PCs in Kraftfahrzeugen ; Informations- und Unterhaltungs-PCs ; Telematik-PCs ; robuste Computer ; robuste Workstations ; robuste Notebooks ; tragbare PCs ; robuste Anzeigen ; Computer und Computersysteme für den Verkehr ; Zugverwaltungssysteme ; Computer und Computersysteme zur Zugverwaltung , Verkehrsregelung , industriellen Regelung , Maschinensteuerung , Eisenbahnsteuerung , digitalen Gesundheitspflege ; Computer und Anzeigen für medizinische Bildgebung , Patientenüberwachung , Informations- und Unterhaltungsgeräte , Passagierinformationssysteme ; Computer und Anzeigen für Öl und Gas enthaltende Umgebungen ; explosionsgeschützte Anzeigen ; Gateways ; Router ; automatische Prüfausrüstungen ; Luftverkehrssteuerungsausrüstungen ; Hochverfügbarkeitscomputer ; Kommunikations-Edge-Computer ; Kommunikationskerncomputer ; Computer für Verteidigungsanwendungen , Überwachungssysteme und Sicherheitssysteme ; integrierte PC-Platinen ; Elektronikplatinen ; Computerplatinen ; Computerschnittstellenkarten ; elektronische Leiterplatten ; Prozessorkarten ; Trägerkarten ; Hauptplatinen ; Faxmodemkarten für Computer ; elektronische Schnittstellenbauteile und elektronische Terminals für Speicher- , Grafik- , Bildausgabe- , Kommunikations- , Netz- und Datenerfassungsanwendungen ; CPUs ; LCD-Treiber ; LCD-TFT-Adapter ; Grafikadapter ; Treiber für Speichermedien ; Computerbildschirme und -monitore ; aktive Rückwandplatinen ; passive Rückwandplatinen ; Adapter für Computer ; Komponenten für und Teile von Computern ; Computerchassis ; Tastaturen , Mäuse , Joysticks , Bildschirme und alle anderen Peripheriegeräte für Computer ; Computerzubehör , nämlich Handgelenk- und Armauflagen , Bildschirmfilter , Mauspads ; Stromversorgung für Computer ; Tastatur-Monitor-Maus-Einheiten ( KVM-Einheiten ) ; Schalter ; industrielle Schalter / Hubs ; nicht verwaltete Schalter / Hubs ; verwaltete Schalter / Hubs ; Peripheriegeräte , Geräte , Instrumente , Zubehör und Ersatzteile für Computer ; Verbindungselemente , nämlich Reihenkoppler , Buchsen , Schaltbretter , Schaltkästen und andere Geräte ; Zeitsensoren ; Verkaufsterminals ; Lautsprecher ; Tonanlagen , nämlich Lautsprecher , Kopfhörer , Mikrofone , Empfänger , Aufzeichnungsgeräte und deren Bauteile ; Batterien ; leere Computermagnetbänder ; elektronische Planungshilfen / Organizer ; Fernsteuerungen für Computer ; Ferncursor für Computer ; Überspannungsschutz und Stromversorgungen ; elektrische Schalter ; Bänder ( Computerspiele ) ; elektronische Publikationen ; Halter für CDs ; IC-Karten ( Smartcards , Adapter und Lesegeräte für diese Karten ) ; Videokarten ; Soundkarten ; Bildverarbeitungskarten ; Bildabtasterkarten ; Grafikkarten ; Netzkarten ; Wechselsprechgeräte ; Mikrocomputer ; Minicomputer ; Mikrofone ; Computerspeicher ; Computerschnittstellen ; Datenverarbeitungsmaschinen / Computerspeichergeräte ; RAM-Schaltkreise ; Chips ; Halbleitergeräte ; integrierte Schaltkreise ; Mikroprozessoren ; elektronische Schaltkreise ; gedruckte Schaltungen ; Datenmodule ; Terminals ; Steuereinheit ; Stromrichter ; Steckverbinder ; Eingabe- und Ausgabegeräte ; Zähler ; Zeitgeber ; elektronische Test- und diagnostische Geräte ; computerbezogene Sicherheitsvorrichtungen für tragbare Computerprodukte ; Behälter und Koffer für Computer , Computerperipheriegeräte und Computerbedarf ; Tragebehältnisse für Computer ; Multiplexer ; Modems ; Datenkommunikationsterminals und Leitungsadapter ; Kommunikationsserver ; Leistungswandler , nämlich Digital-Analog-Wandler , Analog-Digital-Wandler , Vorrichtungen zum Regulieren der Spannungsstufe ; Kabel und Kabelteile ; Telekommunikationsausrüstungen ; Telefone und Telefonzubehör ; Telefonanlagen ; Fernkopiergeräte ; Tonbandgeräte ; Projektoren ; Fotoapparate ; Videokameras ; Web-Kameras ; Videospiele ; Videobildschirme ; Videorecorder ; Videobänder ; Videoanlagen ( Projektoren , Videokameras , Steuergeräte und Zubehör für Videoanlagen ) ; CD-Abspielgeräte ( Musik ) ; CDs mit Musik , Grafiken oder Computerprogrammen ; Funkrufgeräte ; Middelware-Software ; Feldbus- und industrielle Ethernet-Software ; Protokollkonvertierungssoftware und -hardware ; Software zur Anzeige ; Web-fähige Kommunikationssoftware ; Ferndiagnosesoftware ; Internet- und Intranet-Software ; Software für intelligente Peripheriemanagementschnittstellen ( IPMI ) ; Betriebssystemsoftware und Anwendersoftware für Ressourcenzuordnung , Planung , Eingabe- / Ausgabesteuerung , Datenverwaltung , Kommunikationsmanagement , Netzverwaltung , Umschreiber / Transcriber und Kombinationen von Datenverarbeitungsgeräten und Datenverarbeitungsprogrammen ; auf maschinenlesbaren Trägermedien gespeicherte Dokumentationen und Bedienungsanleitungen in Bezug auf Computer oder Computerprogramme ; Computer-Software ; Datenverarbeitungsprogramme ; Firmware ; BIOS-Software ; Computersoftware zur Verwendung mit einem globalen Computernetz ; Computersoftware für die Dokumentenverwaltung ; Computersoftware zur Verwendung für das Auffinden , Abfragen und Empfangen von Text , elektronischen Dokumenten , Grafiken und audiovisuellen Informationen in unternehmensweiten internen Computernetzen und lokalen Netzen , Fernnetzen und weltweiten Computernetzen ; Computersoftware für die Softwareentwicklung und die Erstellung von Websites ; Computer und Systeme zur Telekommunikation und Datenkommunikation ; offene Modularkommunikationsplattformen ; Systeme für drahtlosen Zugang ; Systeme für drahtlosen Edge und Kern ; Systeme für drahtgebundenen Zugang ; Systeme für Unternehmensanwendungen ; Systeme für Verkehrs- und Datenzentrumsinfrastruktur ; Basis-Sende-Empfangs-Stationen ( BTS ) ; Basisstationssteuerungen ( BSC ) ; Kommunikationsknoten ; Funknetzüberwacher ( RNC ) ; Funkvermittlungsstellen ( MSC ) ; Medien-Gateways ; Medien-Gateway-Steuerungen ( MGC ) ; Serving ( General Packet Radio Service ) GPRS Support Nodes ( SGSN ) ; Gateway GPRS Support Nodes ( GGSN ) ; Signalserver für Telekommunikationsanwendungen ; IP-Multimedia-Subsysteme ( IMS ) ; DSL-Zugangsmultiplexer ( DSLAM ) ; Schalter / Hubs ; Router ; Verkehrsfilter- / Sicherheitsvorrichtungen ; Traffic Policing- / Shaping-Vorrichtungen ; Shelf-Management-Controller ( ShMC ) ; intelligente Peripheriemanagementschnittstelle ( IPMI ) ; advancedTCA-Systeme ( Advanced Telecom Computer Architecture ) ; advancedMC-Module ( Advanced Mezzanine Card ) ; microTCA-Systeme ( Micro Telecom Computer Architecture ) ; ETSI ( Europäisches Institut für Telekommunikationsstandards ) und NEBS ( Network Equipment-Building System ) entsprechende Systeme ; Hochverfügbarkeitssysteme ; Telekommunikationsausrüstungen .

**False Positives:**

- `Elektrizitä` — no gold match — likely missing annotation
- `Sprach` — no gold match — likely missing annotation
- `Computer` — no gold match — likely missing annotation
- `Datenverarbeitungsgeräte` — no gold match — likely missing annotation
- `Tex` — no gold match — likely missing annotation
- `Website` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 6

**Gold Entities:**

- `ETSI`(ORG)
- `Europäisches Institut für Telekommunikationsstandards`(ORG)

**Example 22** (doc_id: `53616`) (sent_id: `53616`)


Wie sich aus dem Zusammenhang mit Merkmal 7.2 ergibt , dient der Satz von Befehlen dazu , die Nutzung des vom Telekommunikationsnetz angebotenen Dienstes zu ermöglichen , indem Informationen über den Dienst angezeigt werden können und mithilfe der Eingabemittel eine Auswahl bezüglich des Dienstes getroffen werden kann .

**False Positives:**

- `Befehle` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 23** (doc_id: `53632`) (sent_id: `53632`)


Denn solche eidesstattlichen Versicherungen sind von Hause aus auf das mehr oder weniger vollständige Erinnerungsvermögen des Erklärenden und die Noch-Verfügbarkeit diesbezüglicher Unterlagen angewiesen ( vgl. Ekey / Bender / Fuchs-Wissemann , a. a. O. , § 43 , Rdnr. 48 ) .

**False Positives:**

- `Haus` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Ekey / Bender / Fuchs-Wissemann , a. a. O. , § 43 , Rdnr. 48`(LIT)

**Example 24** (doc_id: `53664`) (sent_id: `53664`)


Soweit das Bundesministerium der Verteidigung darauf verweist , dass Personalgespräche dazu dienten , Soldaten über Verwendungsplanungen der Personalführung zu informieren und in diesem Zusammenhang deren persönliche Motivationslage zu ergründen , um das dienstliche Bedürfnis und die privaten Lebensumstände des Soldaten möglichst gut miteinander zu vereinbaren , trifft dies zweifellos auf Personalgespräche im Vorfeld von Personalmaßnahmen zu .

**False Positives:**

- `Personalmaßnahme` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Bundesministerium der Verteidigung`(ORG)

**Example 25** (doc_id: `53667`) (sent_id: `53667`)


Selbst wenn der Status des Lebenszeitrichters von Verfassungs wegen als Regelstatus der Berufsrichter verbindlich sein sollte , wäre ein Einsatz von Richtern auf Zeit in Ausnahmefällen , wie ihn § 18 VwGO vorsieht , verfassungsrechtlich unbedenklich ( 1. ) .

**False Positives:**

- `Verfassung` — no gold match — likely missing annotation
- `Richter` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `§ 18 VwGO`(NRM)

**Example 26** (doc_id: `53670`) (sent_id: `53670`)


Diese entscheiden , ob sie die KÄVen in die Durchführung von Impfungen einbeziehen .

**False Positives:**

- `Impfunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 27** (doc_id: `53689`) (sent_id: `53689`)


Die beanspruchte Verwendung eines Lithiumsilicatrohlings gemäß Patentanspruch 1 des Hilfsantrags 8 sei ebenfalls gegenüber D2 unter Berücksichtigung der Nacharbeitung D21 -A nicht neu , da auch in Beispiel 22 der D2 eine maschinelle Bearbeitung von Lithiummetasilicat vorgesehen sei .

**False Positives:**

- `Lithiummetasilica` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 28** (doc_id: `53711`) (sent_id: `53711`)


Ausgehend von zumindest durchschnittlicher Kennzeichnungskraft der Widerspruchsmarke , Dienstleistungsidentität bzw. hochgradiger Dienstleistungsähnlichkeit und einer nach dem Gesamteindruck der Marken bestehenden hohen Zeichenähnlichkeit , sei die Gefahr von Verwechslungen im Sinne von § 9 Abs. 1 Nr. 1 MarkenG gegeben .

**False Positives:**

- `Verwechslunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 9 Abs. 1 Nr. 1 MarkenG`(NRM)

**Example 29** (doc_id: `53715`) (sent_id: `53715`)


aa ) Obwohl die Parteien in Klausel 33 des Vertriebsvertrags die Geltung des Rechts der USA und des Staates Kalifornien vereinbart haben , ist das FG den deutschen Grundsätzen über die Auslegung von Willenserklärungen und Verträgen gefolgt .

**False Positives:**

- `Willenserklärunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `USA`(LOC)
- `Kalifornien`(LOC)

**Example 30** (doc_id: `53721`) (sent_id: `53721`)


Die Regelung ist daher geeignet , die Befristung von Arbeitsverträgen mit programmgestaltenden Mitarbeitern bei Rundfunkanstalten zu rechtfertigen ( BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101 ) .

**False Positives:**

- `Arbeitsverträge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101`(RS)

**Example 31** (doc_id: `53754`) (sent_id: `53754`)


Die Gewährung von Alg für die Zeit ab 4. 9. 2012 beruhte auf § 145 Abs 1 Satz 1 SGB III .

**False Positives:**

- `Al` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 145 Abs 1 Satz 1 SGB III`(NRM)

**Example 32** (doc_id: `53775`) (sent_id: `53775`)


nicht nur das erstmalige Legen eines Hausanschlusses , sondern auch Arbeiten zur Erneuerung oder zur Reduzierung von Wasseranschlüssen unter die Steuerermäßigung fallen ( vgl. BGH-Urteil in HFR 2012 , 1110 , Rz 20 ) .

**False Positives:**

- `Wasseranschlüsse` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BGH-Urteil in HFR 2012 , 1110 , Rz 20`(RS)

**Example 33** (doc_id: `53796`) (sent_id: `53796`)


Sie enthalten einerseits normative Vorgaben für die Durchführung von Schutzimpfungen im Rahmen der vertragsärztlichen Versorgung und regeln andererseits - der Tradition des Vertragsarztrechts entsprechend - auf der gesetzlichen Grundlage des § 106 Abs 2 S 4 SGB V aF die Zuständigkeit der Wirtschaftlichkeitsprüfungsgremien für die Durchsetzung einer wirtschaftlichen Behandlungs- und Verordnungsweise auch im Rahmen des Impfens .

**False Positives:**

- `Schutzimpfunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 106 Abs 2 S 4 SGB V aF`(NRM)

**Example 34** (doc_id: `53808`) (sent_id: `53808`)


Zur Begründung führte das Gericht aus , der Antrag sei nicht eilbedürftig , denn eine Beschwer folge aus der Versagung von Lockerungen erst ab Januar 2016 .

**False Positives:**

- `Lockerunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 35** (doc_id: `53816`) (sent_id: `53816`)


Gespeicherte Computerprogramme ( Software ) , insbesondere CAD / CAM- Programme , Finanzprogramme , Bilderstellungs- , Bildbearbeitungs- und Bildverwaltungsprogramme , Vermessungsprogramme für 3D -Vermessungen , Programme für die Rekonstruktion von 3D -Daten aus 2D -Aufnahmen ( Tomographie ) , Programme für die Erstellung und Verwaltung von Röntgenaufnahmen ; alle vorstehenden Waren insbesondere für zahnmedizinische Restaurationen .

**False Positives:**

- `Röntgenaufnahme` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 36** (doc_id: `53826`) (sent_id: `53826`)


Mit diesem Vertrag , der nach seiner Klausel 33 ausschließlich dem Recht der Vereinigten Staaten von Amerika ( USA ) und des Staates Kalifornien unterliegt , gewährte die Klägerin ( als Eigentümerin ) VU das Recht , die Urheberrechte an dem Film für eine Lizenzzeit von 17 Jahren bis zum 23. Dezember 2021 umfassend zu verwerten .

**False Positives:**

- `Amerik` — partial — pred is substring of gold: `Vereinigten Staaten von Amerika`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Vereinigten Staaten von Amerika`(LOC)
- `USA`(LOC)
- `Kalifornien`(LOC)

**Example 37** (doc_id: `53827`) (sent_id: `53827`)


Klasse 42 : Technische Beratung bezüglich Schließanlagen , Türverriegelungs-anlagen , Türsteuerungen , Schlössern , Schließzylindern , Panik- / Antipanikschlössern , Codeschlössern und Türantrieben ; technische Planung von Schließanlagen , Türverriegelungsanlagen , Tür-steuerungen , Schlössern , Schließzylindern , Panik- / Antipanikschlössern , Codeschlössern und Türantrieben ;

**False Positives:**

- `Schließanlage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 38** (doc_id: `53831`) (sent_id: `53831`)


Nach Auswertung strafrechtlicher Ermittlungsergebnisse im Rahmen einer Betriebsprüfung forderte die Beklagte von ihr für die ( illegale ) Beschäftigung des Beigeladenen zu 1. die Zahlung von Sozialversicherungs- und Umlagebeiträgen sowie Säumniszuschlägen iHv zusammen 76 960,14 Euro ( Bescheid vom 21. 4. 2015 in der Gestalt des Widerspruchsbescheids vom 3. 9. 2015 ) .

**False Positives:**

- `Sozialversicherungs` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 39** (doc_id: `53843`) (sent_id: `53843`)


Da die Begriffsdefinitionen des Reisekostenrechts auf die dienstliche Tätigkeit von Beamten zugeschnitten sind und nicht ohne Weiteres davon ausgegangen werden kann , dass sie in jeder Hinsicht mit Normen und Grundsätzen des Personalvertretungsrechts im Einklang stehen , gebietet § 45 Abs. 1 Satz 2 SächsPersVG die entsprechende Anwendung des § 1 Abs. 2 SächsRKG und der dort in Bezug genommenen Bestimmungen ( vgl. BVerwG , Beschlüsse vom 25. November 2004 - 6 P 6.04 - Buchholz 251.7 § 40 NWPersVG Nr. 3 S. 5 ; vom 21. Mai 2007 - 6 P 5.06 - Buchholz 251.5 § 42 HePersVG Nr. 1 Rn. 24 und vom 28. November 2012 - 6 P 3.12 - Buchholz 262 § 9 TGV Nr. 1 Rn. 15 m. w. N. , vgl. auch SächsLT- Drs. 5/4071 ) .

**False Positives:**

- `Beamte` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 45 Abs. 1 Satz 2 SächsPersVG`(NRM)
- `§ 1 Abs. 2 SächsRKG`(NRM)
- `BVerwG , Beschlüsse vom 25. November 2004 - 6 P 6.04 - Buchholz 251.7 § 40 NWPersVG Nr. 3 S. 5`(RS)
- `vom 21. Mai 2007 - 6 P 5.06 - Buchholz 251.5 § 42 HePersVG Nr. 1 Rn. 24`(RS)
- `vom 28. November 2012 - 6 P 3.12 - Buchholz 262 § 9 TGV Nr. 1 Rn. 15`(RS)
- `SächsLT- Drs. 5/4071`(LIT)

**Example 40** (doc_id: `53852`) (sent_id: `53852`)


Mit den hergebrachten Grundsätzen des Berufsbeamtentums im Sinne des Art. 33 Abs. 5 GG ist der Kernbestand von Strukturprinzipien gemeint , die allgemein oder doch ganz überwiegend während eines längeren , traditionsbildenden Zeitraums , insbesondere unter der Reichsverfassung von Weimar , als verbindlich anerkannt und gewahrt worden sind ( vgl. BVerfGE 8 , 332 < 343 > ; 46 , 97 < 117 > ; 58 , 68 < 76 f. > ; 83 , 89 < 98 > ; 106 , 225 < 232 > ; 107 , 218 < 237 > ; 117 , 330 < 344 f. > ; 117 , 372 < 379 > ; 121 , 205 < 219 > ; ohne Bezug auf die Weimarer Reichsverfassung BVerfGE 145 , 1 < 8 Rn. 16 > ) .

**False Positives:**

- `Strukturprinzipie` — no gold match — likely missing annotation
- `Weima` — partial — pred is substring of gold: `Reichsverfassung von Weimar`

> overlaps gold: 1  |  likely missing annotation: 1

**Gold Entities:**

- `Art. 33 Abs. 5 GG`(NRM)
- `Reichsverfassung von Weimar`(NRM)
- `BVerfGE 8 , 332 < 343 > ; 46 , 97 < 117 > ; 58 , 68 < 76 f. > ; 83 , 89 < 98 > ; 106 , 225 < 232 > ; 107 , 218 < 237 > ; 117 , 330 < 344 f. > ; 117 , 372 < 379 > ; 121 , 205 < 219 >`(RS)
- `Weimarer Reichsverfassung`(NRM)
- `BVerfGE 145 , 1 < 8 Rn. 16 >`(RS)

**Example 41** (doc_id: `53879`) (sent_id: `53879`)


Bemüht sich jemand , der ein Statusfeststellungsverfahren einleitet , zeitnah um private Eigenvorsorge , so kann er diese für den Fall , dass das Statusfeststellungsverfahren entgegen seinen Vorstellungen zu einer Feststellung von Versicherungspflicht führt , möglicherweise gar nicht mehr oder nur mit erheblichem Aufwand rückabwickeln ( zu diesen Konsequenzen siehe LSG Berlin-Brandenburg Urteil vom 18. 9. 2013 - L 9 KR 384/11 - Juris RdNr 38 ) .

**False Positives:**

- `Versicherungspflich` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `LSG Berlin-Brandenburg Urteil vom 18. 9. 2013 - L 9 KR 384/11 - Juris RdNr 38`(RS)

**Example 42** (doc_id: `53880`) (sent_id: `53880`)


Die bloße Nichtergreifung von Maßnahmen durch den Kläger gegen seine Abberufung von der Beklagten konnte bei dieser nicht die begründete Erwartung hervorrufen , sie werde nicht mehr auf das Bestehen eines Arbeitsverhältnisses in Anspruch genommen .

**False Positives:**

- `Maßnahme` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 43** (doc_id: `53881`) (sent_id: `53881`)


Ob dies der Fall ist , richtet sich nach den Umständen des Einzelfalls , bei denen darauf abzustellen ist , wie das Hoheitszeichen im Rahmen der Designgestaltung konkret verwendet ist ( vgl. BPatG GRUR 2002 , 337 - Schlüsselanhänger ; Eichmann / von Falckenstein / Kühne , a. a. O. , § 3 Rdn. 26 ff. ) .

**False Positives:**

- `Falckenstei` — partial — pred is substring of gold: `Eichmann / von Falckenstein / Kühne , a. a. O. , § 3 Rdn. 26 ff.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG GRUR 2002 , 337 - Schlüsselanhänger`(RS)
- `Eichmann / von Falckenstein / Kühne , a. a. O. , § 3 Rdn. 26 ff.`(LIT)

**Example 44** (doc_id: `53883`) (sent_id: `53883`)


Denn ungeachtet von Differenzen im Einzelnen verlangt die in beiden Vorschriften enthaltene Verantwortbarkeitsklausel eine Wahrscheinlichkeitsprognose für eine Legalbewährung in Freiheit , wobei die Anforderungen an die Aussicht auf künftige Straffreiheit umso höher anzusetzen sind , je schwerer die in Betracht kommenden Taten wiegen ( zu den rechtlichen Maßstäben des § 88 Abs. 1 JGG vgl. OLG Karlsruhe , Beschluss vom 24. Juli 2006 - 3 Ws 213/06 , StV 2007 , 12 , 13 ; Brunner / Dölling , JGG , 13. Aufl. , § 88 Rn. 5 ; HK-JGG / Kern , 2. Aufl. , § 88 Rn. 26 mwN ; zu den rechtlichen Maßstäben des § 57 Abs. 1 Satz 1 Nr. 2 , Satz 2 StGB vgl. BGH , Beschlüsse vom 25. April 2003 - StB 4/03 , BGHR StGB § 57 Abs. 1 Erprobung 2 ; vom 4. Oktober 2011 - StB 14/11 , NStZ-RR 2012 , 8 ; vom 10. April 2014 - StB 4/14 , juris Rn. 3 ) .

**False Positives:**

- `Differenze` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 88 Abs. 1 JGG`(NRM)
- `OLG Karlsruhe , Beschluss vom 24. Juli 2006 - 3 Ws 213/06 , StV 2007 , 12 , 13`(RS)
- `Brunner / Dölling , JGG , 13. Aufl. , § 88 Rn. 5`(LIT)
- `HK-JGG / Kern , 2. Aufl. , § 88 Rn. 26`(LIT)
- `§ 57 Abs. 1 Satz 1 Nr. 2 , Satz 2 StGB`(NRM)
- `BGH , Beschlüsse vom 25. April 2003 - StB 4/03 , BGHR StGB § 57 Abs. 1 Erprobung 2`(RS)
- `vom 4. Oktober 2011 - StB 14/11 , NStZ-RR 2012 , 8`(RS)
- `10. April 2014 - StB 4/14 , juris Rn. 3`(RS)

**Example 45** (doc_id: `53884`) (sent_id: `53884`)


bb ) Bei der Ausgestaltung von Regelungen zur Bestimmung der Bemessungsgrundlage einer Steuer verfügt der Gesetzgeber über einen weiten Spielraum .

**False Positives:**

- `Regelunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 46** (doc_id: `53913`) (sent_id: `53913`)


Die unterschiedliche Behandlung der Gewinne aus der Veräußerung von Mitunternehmeranteilen in Abhängigkeit von der Gesellschafterstruktur führe zu willkürlichen Ergebnissen .

**False Positives:**

- `Mitunternehmeranteile` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 47** (doc_id: `53933`) (sent_id: `53933`)


Weder die Studie der Holy Fashion Group vom 24. Februar 2016 ( Bl. 49/50 d. A. ) noch die im Amtsverfahren vorgelegten Unterlagen zu Marktforschungsergebnissen einer „ Brigitte “ -Studie , Internetausdrucken zu Showrooms und Verkaufsstätten von Waren der Marke „ JOOP “ oder die beigefügten Urteile sind geeignet , einen entsprechenden Benutzungsnachweis für mit der Marke gekennzeichnete Dienstleistungen zu erbringen .

**False Positives:**

- `Ware` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Holy Fashion Group`(ORG)
- `„ JOOP “`(ORG)

**Example 48** (doc_id: `53934`) (sent_id: `53934`)


a ) Das Landesarbeitsgericht hat zu Unrecht angenommen , dass durch einen Personalüberleitungsvertrag für einen Arbeitgeber , der nicht an dem Vertrag beteiligt ist , eine dynamische Anwendbarkeit von Tarifverträgen ohne seine Zustimmung vereinbart werden könne .

**False Positives:**

- `Tarifverträge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 49** (doc_id: `53938`) (sent_id: `53938`)


In dem besonderen Fall der Sanktionierung von Verstößen gegen die Verordnung [ … ] wurden jedoch die straf- oder bußgeldbewehrten Vorschriften der Verordnung [ … ] durch das Inkrafttreten der Sanktionsvorschriften vor dem Anwendungszeitpunkt der bewehrten EU-Verordnung bereits ab dem 2. Juli 2016 in Deutschland für anwendbar erklärt .

**False Positives:**

- `Verstöße` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschland`(LOC)

**Example 50** (doc_id: `53943`) (sent_id: `53943`)


Der Kläger hat - soweit für die Revision noch von Belang - beantragt ,

**False Positives:**

- `Belan` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 51** (doc_id: `53944`) (sent_id: `53944`)


Nach diesen Grundsätzen ist eine markenrechtlich relevante unmittelbare Gefahr von Verwechslungen zwischen den Vergleichsmarken für die von der angegriffenen Marke zu den Klassen 38 und 45 beanspruchten Dienstleistungen zu besorgen , weshalb die angegriffene Marke insoweit nach § 43 Abs. 2 Satz 1 MarkenG zu löschen ist .

**False Positives:**

- `Verwechslunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 43 Abs. 2 Satz 1 MarkenG`(NRM)

**Example 52** (doc_id: `53947`) (sent_id: `53947`)


Die Angaben der Klägerin zur nicht vertragskonformen Inanspruchnahme von Leistungen seien nicht belegt und nicht nachvollziehbar .

**False Positives:**

- `Leistunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 53** (doc_id: `53960`) (sent_id: `53960`)


Das Landgericht bejahte aufgrund der Chat-Korrespondenz einen Anfangsverdacht wegen sexuellen Missbrauchs von Kindern gemäß § 176 StGB in der im Jahr 1997 gültigen Fassung .

**False Positives:**

- `Kinder` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 176 StGB`(NRM)

**Example 54** (doc_id: `53962`) (sent_id: `53962`)


Klasse 32 : Biere ; Mineralwässer ; kohlensäurehaltige Wässer ; alkoholfreie Getränke ; Fruchtgetränke ; Fruchtsäfte ; Sirupe für die Zubereitung von Getränken ; Präparate für die Zubereitung von Getränken .

**False Positives:**

- `Getränke` — no gold match — likely missing annotation
- `Getränken` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 55** (doc_id: `53978`) (sent_id: `53978`)


Es sei schon zweifelhaft , ob allein wegen des Vorliegens von Arbeitslosigkeit Berufsmäßigkeit bejaht werden könne .

**False Positives:**

- `Arbeitslosigkei` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 56** (doc_id: `54045`) (sent_id: `54045`)


Es gebe mittlerweile eine immer größere Zahl von Gebäuden , die sich nach Bauart , Bauweise , Konstruktion oder Objektgröße von den damals vorhandenen Gebäuden so sehr unterschieden , dass ihre Bewertung nicht mehr mit einer den verfassungsrechtlichen Anforderungen entsprechenden Genauigkeit und Überprüfbarkeit möglich sei .

**False Positives:**

- `Gebäude` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 57** (doc_id: `54055`) (sent_id: `54055`)


Verwirklicht eine Körperschaft ihre satzungsmäßig festgelegten gemeinnützigen Ziele nicht , scheitert die Anerkennung bereits an der fehlenden Übereinstimmung von Satzung und tatsächlicher Geschäftsführung ( § 63 Abs. 1 AO ) .

**False Positives:**

- `Satzun` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 63 Abs. 1 AO`(NRM)

**Example 58** (doc_id: `54063`) (sent_id: `54063`)


Die Erstellung von Sonderbeurteilungen sei erforderlich geworden , weil der Antragsteller wegen seines Dienstzeitendes ... nicht mehr planmäßig zu beurteilen gewesen sei .

**False Positives:**

- `Sonderbeurteilunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 59** (doc_id: `54086`) (sent_id: `54086`)


Sie zeigt die Möglichkeit einer Verletzung von Grundrechten nicht hinreichend auf ( § 23 Abs. 1 Satz 2 , § 92 BVerfGG ) .

**False Positives:**

- `Grundrechte` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 23 Abs. 1 Satz 2 , § 92 BVerfGG`(NRM)

**Example 60** (doc_id: `54110`) (sent_id: `54110`)


C eine zur Messung von Ausgangsgrößen der Spannungsbereitstellungsschaltung ( Gleichrichter- und H-Brückenschaltung , „ PIM “ ) bestimmte Sensorik ( „ Current transformer “ ) ,

**False Positives:**

- `Ausgangsgröße` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 61** (doc_id: `54111`) (sent_id: `54111`)


Das Landgericht hat den Angeklagten wegen Mordes in zwei Fällen , jeweils in Tateinheit mit unerlaubtem Führen einer halbautomatischen Kurzwaffe zum Verschießen von Patronenmunition und Besitz von Munition , zu einer lebenslangen Freiheitsstrafe als Gesamtstrafe verurteilt .

**False Positives:**

- `Patronenmunitio` — no gold match — likely missing annotation
- `Munitio` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 62** (doc_id: `54116`) (sent_id: `54116`)


Sie hat ihre Kollegen angewiesen , „ die Verfolgung des Angeklagten aufzunehmen und aufgrund von Gefahr im Verzug , bevor die vermeintlichen Drogen in Umlauf gelangten , den Angeklagten anzuhalten und ihn zu durchsuchen . “

**False Positives:**

- `Gefah` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 63** (doc_id: `54133`) (sent_id: `54133`)


aa ) Ergibt sich aus dem Vortrag der Parteien im Rechtsstreit , dass die normative Wirkung eines Tarifvertrags nach § 4 Abs. 1 , § 5 Abs. 4 TVG in Betracht kommt , muss das Gericht diese Normen nach § 293 ZPO von Amts wegen ermitteln ( BAG 25. Januar 2017 - 4 AZR 520/15 - Rn. 29 mwN ) .

**False Positives:**

- `Amt` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 4 Abs. 1 , § 5 Abs. 4 TVG`(NRM)
- `§ 293 ZPO`(NRM)
- `BAG 25. Januar 2017 - 4 AZR 520/15 - Rn. 29`(RS)

**Example 64** (doc_id: `54147`) (sent_id: `54147`)


( c ) Nur die vorliegende Auslegung führt auch zu einem sachgerechten , zweckorientierten und praktisch brauchbaren Verständnis von Nr. 3.4.2 GBV BSAV .

**False Positives:**

- `Nr` — partial — pred is substring of gold: `Nr. 3.4.2 GBV BSAV`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Nr. 3.4.2 GBV BSAV`(REG)

**Example 65** (doc_id: `54148`) (sent_id: `54148`)


Es wird unterstellt , dass Besuchern die strittige Werkzeuganordnung zum Verschrauben von Schraubentellerfedern auch im Detail dargestellt wurde .

**False Positives:**

- `Schraubentellerfeder` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 66** (doc_id: `54165`) (sent_id: `54165`)


b ) Ebenfalls von Verfassungs wegen nicht zu beanstanden ist die Argumentation der Gerichte , dass die Verweigerung einer Stellungnahme von dem Betroffenen nicht im Einzelfall gerechtfertigt werden müsse .

**False Positives:**

- `Verfassung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 67** (doc_id: `54166`) (sent_id: `54166`)


Sein Verständnis wird durch den Umstand gefördert , dass CAD-Anwendungen umfangreich in Laboren eingesetzt werden , etwa zur Konstruktion von Dental- oder orthopädischen Prothesen oder von Webseiten mit Animationen .

**False Positives:**

- `Dental` — no gold match — likely missing annotation
- `Webseite` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Example 68** (doc_id: `54168`) (sent_id: `54168`)


Daher kann der Senat anhand der Urteilsfeststellungen in keinem der Fälle der Hinterziehung von Umsatzsteuer nachprüfen , ob Tatvollendung eingetreten ist .

**False Positives:**

- `Umsatzsteue` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 69** (doc_id: `54169`) (sent_id: `54169`)


Unabhängig von der Frage , ob ein solcher Antrag - entgegen der Auffassung des Antragstellers - gemäß § 67 Abs. 4 VwGO dem Vertretungszwang unterliegt ( vgl. verneinend z.B. : BVerwG , Beschluss vom 6. Juni 1972 - 3 ER 404.71 - Buchholz 310 § 53 VwGO Nr. 6 ; Ziekow , in : Sodan / Ziekow , VwGO , 4. Aufl. 2014 , § 53 Rn. 20 ; Kraft , in : Eyermann , VwGO , 14. Aufl. 2014 , § 53 Rn. 14 ; Unruh , in : HK-Verwaltungsrecht , 3. Aufl. 2013 , § 53 Rn. 15 ; Scheidler , VR 2012 , 113 ; bejahend z.B. : W. - R. Schenke , in : Kopp / Schenke , VwGO , 23. Aufl. 2017 , § 53 Rn. 10 ; Redeker / von Oertzen , VwGO , 15. Aufl. 2010 , § 53 Rn. 5 ; vermittelnd z.B. : Bier , in : Schoch / Schneider / Bier , VwGO , Stand Oktober 2016 , § 53 Rn. 13 ) , ist der Antrag hier jedenfalls deshalb unzulässig , weil die gesetzlichen Voraussetzungen , unter denen das Bundesverwaltungsgericht eine Zuständigkeitsbestimmung überhaupt vornehmen darf , nicht erfüllt sind .

**False Positives:**

- `Oertze` — partial — pred is substring of gold: `Redeker / von Oertzen , VwGO , 15. Aufl. 2010 , § 53 Rn. 5`

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

**Example 70** (doc_id: `54175`) (sent_id: `54175`)


Zutreffend führt das LSG in diesem Zusammenhang zudem aus , dass es nicht darauf ankommt , ob Arbeitseinsätze im Rahmen eines Dauerarbeitsverhältnisses von vorneherein feststehen oder von Mal zu Mal vereinbart werden .

**False Positives:**

- `Ma` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 71** (doc_id: `54189`) (sent_id: `54189`)


3. Das vorbezeichnete Schutzhindernis nach § 8 Abs. 2 Nr. 1 MarkenG betrifft zunächst unmittelbar die beanspruchten Dienstleistungen der Klassen 35 und 36 , die regelmäßig von Einkaufsfinanzierern angeboten oder vermittelt werden .

**False Positives:**

- `Einkaufsfinanzierer` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 8 Abs. 2 Nr. 1 MarkenG`(NRM)

**Example 72** (doc_id: `54193`) (sent_id: `54193`)


Im vorliegenden Fall war die Erstellung von Sonderbeurteilungen ( Nr. 206 ZDv A- 1340/50 ) für beide Bewerber zulässig und geboten , weil aktuelle planmäßige Beurteilungen mit einem vergleichbaren Beurteilungszeitraum nicht vorlagen .

**False Positives:**

- `Sonderbeurteilunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Nr. 206 ZDv A- 1340/50`(REG)

**Example 73** (doc_id: `54229`) (sent_id: `54229`)


Der Sache nach hat der Beklagte einen Regress wegen unwirtschaftlicher Verordnung von Impfstoffen und keinen verschuldensabhängigen Ersatz wegen der Verursachung eines " sonstigen Schadens " festgesetzt ( c ) .

**False Positives:**

- `Impfstoffe` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 74** (doc_id: `54235`) (sent_id: `54235`)


Die Ausführbarkeit werde auch nicht durch die Nacharbeitungen gemäß den Anlagenkonvoluten D19C und D21B in Frage gestellt , da hier die Verfahrensparameter von Beispiel 13 des Streitpatents , insbesondere in Bezug auf die Wärmebehandlung , nicht eingehalten worden seien .

**False Positives:**

- `Beispie` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 75** (doc_id: `54236`) (sent_id: `54236`)


Es wird jedoch nicht ausgeführt , inwieweit die Leistung der Verleger mit derjenigen von Tonträger- und Filmherstellern vergleichbar ist .

**False Positives:**

- `Tonträger` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 76** (doc_id: `54250`) (sent_id: `54250`)


Die piktogrammartige Verwendung von Herzsymbolen habe eine weltweite Verbreitung erfahren und sei in der modernen Werbung zu einem allgegenwärtigen Gestaltungselement geworden .

**False Positives:**

- `Herzsymbole` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 77** (doc_id: `54252`) (sent_id: `54252`)


aa ) Der Aufhebungsvertrag regelt nach Nr. 1 Abs. 2 die Rechte und Pflichten der Vertragsparteien während der Übergangsphasen von der Beendigung des Arbeitsverhältnisses bis zum Bezug von Sozialversicherungsrente .

**False Positives:**

- `Sozialversicherungsrente` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 78** (doc_id: `54269`) (sent_id: `54269`)


Dies wäre jedoch angesichts der Bandbreite der in §§ 174 ff. StGB geregelten Straftaten , deren Strafrahmen von Geldstrafe bis zu lebenslanger Freiheitstrafe reicht , geboten gewesen .

**False Positives:**

- `Geldstraf` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§§ 174 ff. StGB`(NRM)

**Example 79** (doc_id: `54274`) (sent_id: `54274`)


Jedenfalls aber kann in diesem Zusammenhang nicht auf die zur Kompetenzabgrenzung von Betriebsrat und Gesamtbetriebsrat maßgebenden Kriterien ( dazu BAG 3. Mai 2006 - 1 ABR 15/05 - BAGE 118 , 131 [ hauptsächlich zum Aufstellen eines Sozialplans ] ; 11. Dezember 2001 - 1 AZR 193/01 - BAGE 100 , 60 ; 8. Juni 1999 - 1 AZR 831/98 - BAGE 92 , 11 ; 24. Januar 1996 - 1 AZR 542/95 - BAGE 82 , 79 ; 17. Februar 1981 - 1 AZR 290/78 - BAGE 35 , 80 ) zurückgegriffen werden .

**False Positives:**

- `Betriebsra` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BAG 3. Mai 2006 - 1 ABR 15/05 - BAGE 118 , 131 [ hauptsächlich zum Aufstellen eines Sozialplans ]`(RS)
- `11. Dezember 2001 - 1 AZR 193/01 - BAGE 100 , 60`(RS)
- `8. Juni 1999 - 1 AZR 831/98 - BAGE 92 , 11`(RS)
- `24. Januar 1996 - 1 AZR 542/95 - BAGE 82 , 79`(RS)
- `17. Februar 1981 - 1 AZR 290/78 - BAGE 35 , 80`(RS)

**Example 80** (doc_id: `54278`) (sent_id: `54278`)


In der Modebranche sei es üblich , dass bestimmte Kollektionen unter leicht abgewandeltem Namen herausgebracht würden , weshalb die angesprochenen Verkehrskreise denken würden , bei der angefochtenen Marke handele es sich um eine Abwandlung bzw. eine besondere Form der älteren Marke für eine bestimmte Art / Kollektion von Waren .

**False Positives:**

- `Waren` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 81** (doc_id: `54281`) (sent_id: `54281`)


Die Beklagte bewilligte der Klägerin für die ersten zwölf Lebensmonate des Kindes Elterngeld auf der Grundlage des im Zeitraum von März 2014 bis Februar 2015 monatlich gezahlten Brutto-Festgehalts , ohne die Quartalsprovisionen zu berücksichtigten ( Bescheid vom 17. 8. 2015 ) .

**False Positives:**

- `Mär` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 82** (doc_id: `54282`) (sent_id: `54282`)


a ) Nach § 76 Abs. 1 Satz 1 FGO erforscht das Gericht den Sachverhalt von Amts wegen .

**False Positives:**

- `Amt` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 76 Abs. 1 Satz 1 FGO`(NRM)

**Example 83** (doc_id: `54295`) (sent_id: `54295`)


So reicht es für den Fortbestand des Anspruchs auf Investitionszulage nicht aus , dass ein gefördertes Wirtschaftsgut zu Beginn des Bindungszeitraums die Wirtschaftstätigkeit im Fördergebiet erheblich gefördert und zur Schaffung von Arbeitsplätzen beigetragen hat , wenn es während des Bindungszeitraums in einen Betrieb außerhalb des Fördergebiets verlagert wird .

**False Positives:**

- `Arbeitsplätze` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 84** (doc_id: `54315`) (sent_id: `54315`)


Dabei erwächst der gesteigerte Unwert der Tat aus dem groben Missverhältnis von Mittel und Zweck , indem der Täter das Leben eines anderen Menschen der Befriedigung eigener Geschlechtslust unterordnet ( BGH , Urteil vom 22. April 2005 - 2 StR 310/04 , BGHSt 50 , 80 , 86 ) .

**False Positives:**

- `Mitte` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BGH , Urteil vom 22. April 2005 - 2 StR 310/04 , BGHSt 50 , 80 , 86`(RS)

**Example 85** (doc_id: `54328`) (sent_id: `54328`)


Meist wiesen solche Forstanhänger dabei Lastbewegungsvorrichtungen in Form von Kranarmen mit Greifern auf , welche durch Hydrauliksysteme betätigbar seien .

**False Positives:**

- `Kranarme` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 86** (doc_id: `54340`) (sent_id: `54340`)


( 2 ) Die Erzielung von Veräußerungserlösen ist generell mit dem Gleichheitssatz aus Art. 3 Abs. 1 GG vereinbar .

**False Positives:**

- `Veräußerungserlöse` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Art. 3 Abs. 1 GG`(NRM)

**Example 87** (doc_id: `54344`) (sent_id: `54344`)


Vertragsärztliche Versorgung - Berufsausübungsgemeinschaft - Festlegung der Leistungsbegrenzung bei Eintritt eines Arztes im Wege des Job-Sharings - Bedarfsplanungs-Richtlinie - Zulässigkeit der Saldierung von Punktzahlen innerhalb des Jahresbezugs - kein Ermessensspielraum der Kassenärztlichen Vereinigung - Neuberechnung - Faktor für die jährliche Anpassung der Job-Sharing-Obergrenze in Sonderfällen

**False Positives:**

- `Punktzahle` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 88** (doc_id: `54353`) (sent_id: `54353`)


Die Gewährung von Rechtsschutz und die Eröffnung des nach der Prozessordnung dafür vorgesehenen Instanzenzuges hängen insbesondere nicht vom Zeitpunkt der Erledigung der Maßnahme ab ( vgl. BVerfGK 6 , 303 < 309 > ) .

**False Positives:**

- `Rechtsschut` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BVerfGK 6 , 303 < 309 >`(RS)

**Example 89** (doc_id: `54398`) (sent_id: `54398`)


a ) Mit seinem Leistungsantrag will der Betriebsrat die Arbeitgeberin dazu verpflichten , bei der Berechnung der monatlichen Arbeitszeit im Rahmen „ der Erstellung von Dienstplänen “ Tage , an denen ein Freizeitausgleich für geleistete Arbeitszeiten an Sonn- oder Feiertagen iSd. § 12 Abs. 7 DRK-TV LSA gewährt wird , als Arbeitszeit mit einzubeziehen , weil sich dies auf die Arbeitszeitkonten und die höchstzulässigen Schwankungsbreiten nach den geschlossenen Betriebsvereinbarungen auswirke .

**False Positives:**

- `Dienstpläne` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 12 Abs. 7 DRK-TV LSA`(REG)

**Example 90** (doc_id: `54407`) (sent_id: `54407`)


Denn die Tätigkeit von Tagesmüttern und -vätern , die fremde Kinder in ihrem Haushalt , im Haushalt des Personensorgeberechtigten oder in anderen geeigneten Räumen betreuen und fördern und die Tätigkeit der genannten Personengruppen , die diese Leistungen in Kindertageseinrichtungen erbringen , sind vergleichbar .

**False Positives:**

- `Tagesmütter` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 91** (doc_id: `54409`) (sent_id: `54409`)


Die freiwillige Herausgabe von Gegenständen lasse vorliegend die Notwendigkeit einer Durchsuchung nicht entfallen , da die Ermittlungsbehörden bei einer etwaigen Herausgabe nicht hätten feststellen können , ob tatsächlich alle Gegenstände herausgegeben worden seien .

**False Positives:**

- `Gegenstände` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 92** (doc_id: `54435`) (sent_id: `54435`)


Denn in der D1 und D2 ist weder ein Überbrückungseinsatz , der anstelle des gewöhnlich vorhandenen Ventileinsatzes in die Ventilaufnahme des Saug-Spül-Handgriffs einsetzbar sein soll , noch ein Fußventil , das mit dem Saug-Spül-Handgriff verbindbar ist und zum Öffnen und Schließen von Saug- und Spülkanal dienen soll , genannt .

**False Positives:**

- `Saug` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 93** (doc_id: `54440`) (sent_id: `54440`)


Der BFH ließ es daher dahingestellt , ob als Rechtsgrundlage für die steuerrechtliche Zurechnung von Wirtschaftsgütern die handelsrechtlichen GoB oder unmittelbar § 39 AO heranzuziehen war ( BFH-Urteil vom 25. April 2006 X R 57/04 , BFH / NV 2006 , 1819 , unter II. 2. c ) .

**False Positives:**

- `Wirtschaftsgüter` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BFH`(ORG)
- `§ 39 AO`(NRM)
- `BFH-Urteil vom 25. April 2006 X R 57/04 , BFH / NV 2006 , 1819 , unter II. 2. c`(RS)

**Example 94** (doc_id: `54443`) (sent_id: `54443`)


Soweit eine Heranziehung zu den Kosten des Mittagessens festgesetzt wird , liegt eine Entscheidung vor , mit der iS des § 44 Abs 1 Satz 1 SGB X die Erbringung von Sozialleistungen geregelt wird .

**False Positives:**

- `Sozialleistunge` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 44 Abs 1 Satz 1 SGB X`(NRM)

**Example 95** (doc_id: `54448`) (sent_id: `54448`)


Welchen Einfluss die aufrechterhaltene Stationierung von Atomwaffen in Büchel für das Verhalten von Terroristen ( und im Konflikt mit NATO-Staaten stehenden Drittstaaten ) habe , entziehe sich einer gerichtlichen Feststellung .

**False Positives:**

- `Atomwaffe` — no gold match — likely missing annotation
- `Terroriste` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `Büchel`(LOC)

**Example 96** (doc_id: `54452`) (sent_id: `54452`)


Die Vorschrift beschränke sich ausdrücklich nicht auf die vom Gesetzgeber angeführten Missbrauchsfälle ; eine Besteuerung trete auch dann ein , wenn zuvor keine steuerneutrale Veräußerung von Wirtschaftsgütern erfolgt sei .

**False Positives:**

- `Wirtschaftsgüter` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 97** (doc_id: `54457`) (sent_id: `54457`)


Er rügt eine Verletzung von Art. 3 Abs. 1 und Art. 19 Abs. 4 Satz 1 in Verbindung mit Art. 2 Abs. 2 Satz 1 GG .

**False Positives:**

- `Art` — partial — pred is substring of gold: `Art. 3 Abs. 1 und Art. 19 Abs. 4 Satz 1 in Verbindung mit Art. 2 Abs. 2 Satz 1 GG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 3 Abs. 1 und Art. 19 Abs. 4 Satz 1 in Verbindung mit Art. 2 Abs. 2 Satz 1 GG`(NRM)

**Example 98** (doc_id: `54461`) (sent_id: `54461`)


Zu diesen Gruppen gehören die Empfänger der in der Vorschrift aufgezählten bedürftigkeitsabhängigen laufenden Leistungen zum Lebensunterhalt , insbesondere von Arbeitslosengeld II. Sozialgeld oder Sozialhilfe ( vgl. jeweils Nr. 2 der Vorschrift ) .

**False Positives:**

- `Arbeitslosengel` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 99** (doc_id: `54462`) (sent_id: `54462`)


Der Fachmann stellt etwa beim Test von Schaltkreisen für Hochstrom- oder Poweranwendungen ( vgl. Streitpatentschrift , Absatz 0005 ) andere Anforderungen an die Übergangswiderstände zwischen den Kontaktfederteilen als beim Test mit niedrigen Strömen .

**False Positives:**

- `Schaltkreise` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Hyphenated Surnames`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `acae7cb4`  
**Description:**
Captures hyphenated surnames (e.g., 'Sost-Scheible', 'Meier-Beck') only when preceded by a title or legal role.

**Content:**
```
(?:Dr\.?\s+|Prof\.?\s+|Dipl\.-Ing\.\s+|Dipl\.-Psych\.\s+|Richter\s+|Anwalt\s+|Rechtsanwältin\s+|Rechtsanwalt\s+|Angeklagten|Kläger|Zeugen|Richter|Geschädigten|Vorsitzenden|Ministerpräsidenten|Herrn|Frau)\s+([A-Z][a-zäöüß]+-[A-Z][a-zäöüß]+)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Single Letter Anonymized Names`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `c2457fe9`  
**Description:**
Captures single capital letters used as anonymized names in legal contexts (e.g., 'N', 'C', 'V', 'B').

**Content:**
```
\b([A-Z])\b(?!\s*(?:von|der|die|das|ein|eine|mit|für|auf|in|an|bei|zu|nach|vor|über|unter|durch|ohne|gegen|seit|statt|wegen|um|bis|neben|zwischen|entlang|außer|neben|ab|Gericht|Firma|Behörde|Polizei|Steuern|Umsatz|Kabel|Anlage|Umsatzsteuer|Rechtsanwältin|Rechtsanwalt|Ackerstatusrechten|Höchst-Schmerzgrenze|Marken-Beschwerdesenat|Rechtsanwalt|...|Frau|Herr|Dr\.?|Prof\.?|Anwalt|Gericht|Firma|Behörde|Polizei|Steuern|Umsatz|Kabel|Anlage|Umsatzsteuer|Rechtsanwältin|Rechtsanwalt|Ackerstatusrechten|Höchst-Schmerzgrenze|Marken-Beschwerdesenat|Rechtsanwalt|...))
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 17 | 0 | 17 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 17 | 338 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53659`) (sent_id: `53659`)


Auch hieraus ergebe sich eine Verletzung des § 39 SGB X.

**False Positives:**

- `X` — partial — pred is substring of gold: `§ 39 SGB X.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 39 SGB X.`(NRM)

**Example 1** (doc_id: `55361`) (sent_id: `55361`)


Daraufhin beantragte der Beklagte die Einleitung eines Schiedsverfahrens nach § 73b Abs 4a S 1 SGB V .

**False Positives:**

- `V` — partial — pred is substring of gold: `§ 73b Abs 4a S 1 SGB V`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 73b Abs 4a S 1 SGB V`(NRM)

**Example 2** (doc_id: `55569`) (sent_id: `55569`)


I. Die Klägerin und Beschwerdeführerin ( Klägerin ) ist umsatzsteuerrechtlich Organgesellschaft des Organträgers N .

**False Positives:**

- `N` — type mismatch — same span as gold: `N`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `N`(ORG)

**Example 3** (doc_id: `56127`) (sent_id: `56127`)


§ 29a Abs. 3 Satz 2 TVÜ-Länder verweist im Klammerzusatz dabei ausdrücklich auf § 17 Abs. 4 TV-L .

**False Positives:**

- `L` — similar text (different position): `§ 29a Abs. 3 Satz 2 TVÜ-Länder`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 29a Abs. 3 Satz 2 TVÜ-Länder`(REG)
- `§ 17 Abs. 4 TV-L`(REG)

**Example 4** (doc_id: `56397`) (sent_id: `56397`)


Vertretungsberechtigt für die Holding-KG ist die C-GmbH , vertreten durch ihren Geschäftsführer D.

**False Positives:**

- `D` — partial — pred is substring of gold: `D.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C-GmbH`(ORG)
- `D.`(PER)

**Example 5** (doc_id: `56674`) (sent_id: `56674`)


I

**False Positives:**

- `I` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 6** (doc_id: `56709`) (sent_id: `56709`)


III. Die Klägerin hat über den 31. Juli 2014 hinaus Anspruch auf Vergütung nach der Entgeltgruppe 10 Stufe 5 TV-L .

**False Positives:**

- `L` — partial — pred is substring of gold: `TV-L`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `TV-L`(REG)

**Example 7** (doc_id: `56968`) (sent_id: `56968`)


Ausschnitt aus dem Signallaufplan auf Seite 30 , Abschnitt „ Restart interlock , Start-up testing “ , Kanal A

**False Positives:**

- `A` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `57256`) (sent_id: `57256`)


Vergütungsgruppe V b

**False Positives:**

- `V` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 9** (doc_id: `57914`) (sent_id: `57914`)


Der Geburtsort seiner jetzigen Ehefrau ist M.

**False Positives:**

- `M` — partial — pred is substring of gold: `M.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `M.`(LOC)

**Example 10** (doc_id: `58157`) (sent_id: `58157`)


Denn die Klägerin und die Holding-KG werden durch dieselbe Person vertreten , nämlich D.

**False Positives:**

- `D` — partial — pred is substring of gold: `D.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `D.`(PER)

**Example 11** (doc_id: `58235`) (sent_id: `58235`)


Die Klägerin ist Eigentümerin eines insgesamt 5471 m² großen Grundstücks in der Gemarkung S.

**False Positives:**

- `S` — partial — pred is substring of gold: `S.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S.`(LOC)

**Example 12** (doc_id: `58377`) (sent_id: `58377`)


Das FG ordnete mit Beschluss vom 21. April 2016 die Einholung eines Sachverständigengutachtens zu den Verkehrswerten des Gebäudes und des Grund und Bodens an und beauftragte damit den Gutachterausschuss für Grundstückswerte in der Stadt Z .

**False Positives:**

- `Z` — type mismatch — same span as gold: `Z`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Z`(LOC)

**Example 13** (doc_id: `58526`) (sent_id: `58526`)


Dies führte nach der Anlage 2 zum TVÜ-Länder iVm. § 17 Abs. 1 , § 39 Abs. 1 Angleichungs-TV Land Berlin mit Wirkung zum 1. November 2010 zu einer Überleitung in die Entgeltgruppe 10 TV-L .

**False Positives:**

- `L` — similar text (different position): `Anlage 2 zum TVÜ-Länder`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Anlage 2 zum TVÜ-Länder`(REG)
- `§ 17 Abs. 1 , § 39 Abs. 1 Angleichungs-TV Land Berlin`(REG)
- `TV-L`(REG)

**Example 14** (doc_id: `59003`) (sent_id: `59003`)


Ferner verstießen die Festsetzungen des HzV-Vertrages gegen das Gebot der Selbsttragung eines Wahltarifs nach § 53 Abs 9 SGB V .

**False Positives:**

- `V` — similar text (different position): `HzV-Vertrages`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `HzV-Vertrages`(REG)
- `§ 53 Abs 9 SGB V`(NRM)

**Example 15** (doc_id: `59098`) (sent_id: `59098`)


Der Antragsteller wendet sich gegen seine Versetzung vom ... in L. zum ... in C.

**False Positives:**

- `C` — partial — pred is substring of gold: `C.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `L.`(LOC)
- `C.`(LOC)

**Example 16** (doc_id: `59857`) (sent_id: `59857`)


Am 4. 2. 2010 beantragten die Kläger die Überprüfung " sämtlicher Bescheide den Zeitraum 1. 4. 2008 bis 30. 9. 2009 betreffend " nach § 44 SGB X .

**False Positives:**

- `X` — partial — pred is substring of gold: `§ 44 SGB X`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 44 SGB X`(NRM)

</details>

---

## `Multi-Initial Anonymized Names`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `ebd05cdc`  
**Description:**
Captures anonymized names with multiple initials (e.g., 'M. D.', 'A. A.', 'P. W.').

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

</details>

---


# Rule Evaluation Report — Qwen/Qwen3.5-35B-A3B

Generated on: 2026-06-23T07:20:08.008174

---

<details>
<summary>Configuration</summary>

Results can be reproduced by running this command: 
```
 python benchmark.py --config reports/germanler/Qwen_Qwen3.5-35B-A3B/PER/2026-06-23_v2/config.yaml 
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
| Validation sentences | 468 |
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
| Best Batch Idx | 9 |
| Best Batch F1 | 0.1968838526912181 |
| Best Rules Serialized | [{'id': '84421f67', 'name': 'Full names with titles', 'description': "Captures full names preceded by titles like Dr., Prof., or Dipl.-Ing., ensuring multi-part names with middle initials (e.g., 'Jay B. Saoud') are captured as a single entity.", 'format': 'regex', 'content': '(?:Dr\\.?\\s+|Prof\\.?\\s+|Dipl\\.-Ing\\.\\s+|Dipl\\.-Psych\\.\\s+|Dipl\\.-Ing\\.\\s+Univ\\.\\s+)([A-Z][a-zäöüß]+(?:\\s+[A-Z][a-zäöüß]+)*|([A-Z]\\.)\\s+[A-Z][a-zäöüß]+(?:\\s+[A-Z][a-zäöüß]+)*)', 'priority': 10, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:08:59.296110', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'af28d6a7', 'name': 'Anonymized initials with dots', 'description': "Captures anonymized person identifiers consisting of a single capital letter followed by a dot (e.g., 'T.', 'F.', 'S.'), ensuring the dot is included.", 'format': 'regex', 'content': '\\b([A-Z]\\.)\\b', 'priority': 15, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:07:29.378403', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'a3f99de3', 'name': 'Initials with dots and spaces (e.g., T. D.)', 'description': 'Captures sequences of initial-dot-space-initial-dot patterns.', 'format': 'regex', 'content': '\\b([A-Z]\\.[ ]+[A-Z]\\.)\\b', 'priority': 8, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:02:17.824003', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '0599e43e', 'name': "Names after 'Herr'", 'description': "Captures names immediately following the title 'Herr' (including 'Herrn').", 'format': 'regex', 'content': '\\b(?:Herr|Herrn)\\s+([A-Z][a-zäöüß]+(?:\\s+[A-Z][a-zäöüß]+)*)\\b', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:02:17.823824', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '4993cd07', 'name': "Names after 'Angeklagte' or 'Klägerin'", 'description': "Captures names following legal role indicators like 'Angeklagte' or 'Klägerin'.", 'format': 'regex', 'content': '\\b(?:Angeklagte|Angeklagten|Klägerin|Kläger|Zeugin|Zeuge|Geschädigte|Gutachter|Gutachterin)\\s+([A-Z][a-zäöüß]+(?:\\s+[A-Z][a-zäöüß]+)*)\\b', 'priority': 6, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:02:17.824138', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '4f43c0ca', 'name': "Names after 'Richter' or 'Vorsitzender'", 'description': "Captures names following judicial titles like 'Richter', 'Vorsitzender', 'Richterin', 'Vorsitzende Richterin', ensuring the name is captured even if preceded by 'Dr.' or 'Prof.'.", 'format': 'regex', 'content': '\\b(?:Richter|Vorsitzender|Richterin|Vorsitzende Richterin|Vorsitzenden Richters)\\s+(?:Dr\\.?\\s+|Prof\\.?\\s+|Dipl\\.-[A-Za-z]+\\s+)?([A-Z][a-zäöüß]+(?:-[A-Z][a-zäöüß]+)*)', 'priority': 11, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:07:29.379556', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '6e133f27', 'name': "Names after 'Dr.' or 'Prof.'", 'description': 'Captures names following titles like Dr. or Prof., handling both full names and initials.', 'format': 'regex', 'content': '\\b(?:Dr\\.?\\s+|Prof\\.?\\s+)([A-Z][a-zäöüß]+(?:\\s+[A-Z][a-zäöüß]+)*|[A-Z]\\.[ ]+[A-Z]\\.|[A-Z]\\.)\\b', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:01:57.254521', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '6e0b7cc1', 'name': "Names after 'Rechtsanwältin' or 'Rechtsanwalt'", 'description': 'Captures names following legal profession titles.', 'format': 'regex', 'content': '\\b(?:Rechtsanwältin|Rechtsanwalt)\\s+(?:Dr\\.?\\s+|Prof\\.?\\s+)?([A-Z][a-zäöüß]+(?:\\s+[A-Z][a-zäöüß]+)*)\\b', 'priority': 6, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:07:11.569621', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '6a893240', 'name': 'Names with dots in middle (e.g., B1 …)', 'description': "Captures anonymized names with dots and ellipses or spaces (e.g., 'B1 …', 'K …', 'K1 …').", 'format': 'regex', 'content': '\\b([A-Z]\\d?\\s+…+)\\b', 'priority': 7, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:02:30.348163', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '88d3e859', 'name': "Names after 'Dr.' or 'Prof.' (standalone)", 'description': 'Captures names following titles like Dr. or Prof. when not part of a longer title sequence, handling both full names and initials.', 'format': 'regex', 'content': '\\b(?:Dr\\.?\\s+|Prof\\.?\\s+|Professor\\s+)([A-Z][a-zäöüß]+(?:\\s+[A-Z][a-zäöüß]+)*|\\w+\\.?)\\b', 'priority': 10, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:08:59.298169', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'daa8797d', 'name': 'Anonymized names with ellipses', 'description': "Captures anonymized names with ellipses like 'K …', 'B1 …', 'T. D.', 'L. …', 'Ch. …' ensuring no trailing spaces are included.", 'format': 'regex', 'content': '\\b([A-Z]\\s+…|[A-Z]\\d+\\s+…|T\\.\\s+D\\.|B1\\s+…|K1\\s+…|H\\.\\s+…|L\\.\\s+…|Ch\\.\\s+…|T\\.)\\b', 'priority': 12, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:04:56.725154', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '12d202b6', 'name': "Names after 'Herr' or 'Herrn'", 'description': "Captures names immediately following the title 'Herr' or 'Herrn', including single initials.", 'format': 'regex', 'content': '\\b(?:Herr|Herrn)\\s+([A-Z][a-zäöüß]+(?:\\s+[A-Z][a-zäöüß]+)*|([A-Z])\\.)', 'priority': 10, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:05:14.298723', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '97b86ba2', 'name': 'Hyphenated surnames', 'description': "Captures hyphenated surnames like 'Schmidt-Räntsch' only when preceded by a title or in a list of names to avoid matching court names.", 'format': 'regex', 'content': '(?:Dr\\.?\\s+|Prof\\.?\\s+|Richter\\s+|Vorsitzender\\s+|und\\s+|sowie\\s+|der\\s+|die\\s+|des\\s+)([A-Z][a-zäöüß]+(?:-[A-Z][a-zäöüß]+)+)', 'priority': 11, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:04:56.725880', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'ae91774f', 'name': 'Anonymized initials with dots (multi-initial)', 'description': "Captures multi-initial anonymized names like 'A. S.' or 'R. C.' as a single entity.", 'format': 'regex', 'content': '(?:[A-Z]\\.)\\s+(?:[A-Z]\\.)', 'priority': 9, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:04:56.725844', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '549dd13e', 'name': "Surnames after 'Generalanwalt' or 'Generalanwältin'", 'description': "Captures names following 'Generalanwalt' or 'Generalanwältin' titles.", 'format': 'regex', 'content': '\\b(?:Generalanwalt|Generalanwältin)\\s+([A-Z][a-zäöüß]+(?:\\s+[A-Z][a-zäöüß]+)*)\\b', 'priority': 10, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:04:56.725487', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '68df845b', 'name': "Surnames after 'Richter' or 'Vorsitzender' (refined)", 'description': "Captures names following judicial titles, ensuring the name is captured correctly even if preceded by 'Dipl.' or 'Prof.'.", 'format': 'regex', 'content': '\\b(?:Richter|Vorsitzender)\\s+(?:Dipl\\.-[a-z]+\\s+)?([A-Z][a-zäöüß]+(?:\\s+[A-Z][a-zäöüß]+)*)\\b', 'priority': 10, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:04:56.725602', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'e91ec092', 'name': "Surnames after 'Rechtsanwalt' or 'Rechtsanwältin' (refined)", 'description': "Captures names following legal profession titles, handling potential titles like 'Dr.' or 'Prof.' before the name.", 'format': 'regex', 'content': '\\b(?:Rechtsanwalt|Rechtsanwältin)\\s+(?:Dr\\.?\\s+|Prof\\.?\\s+)?([A-Z][a-zäöüß]+(?:\\s+[A-Z][a-zäöüß]+)*)\\b', 'priority': 6, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:04:56.725712', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '2eb9f1cc', 'name': 'Anonymized initials with ellipses after roles', 'description': "Captures anonymized names with ellipses (e.g., 'K …', 'B1 …') following legal roles or titles.", 'format': 'regex', 'content': '\\b(?:Angeklagte|Angeklagten|Kläger|Klägerin|Zeuge|Zeugin|Geschädigte|Geschädigten|Rechtsanwalt|Rechtsanwältin|Patentanwalt|Vorsitzender|Richter|Richterin|Herr|Herrn)\\s+([A-Z]\\s+…|[A-Z]\\d+\\s+…)', 'priority': 14, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:07:53.409226', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '2980bda7', 'name': 'Known surnames after legal roles', 'description': "Captures specific known surnames (e.g., 'Knoll', 'Kriener', 'Schmid') when they follow legal role indicators, ensuring they are treated as PER.", 'format': 'regex', 'content': '\\b(?:Angeklagte|Angeklagten|Kläger|Klägerin|Zeuge|Zeugin|Geschädigte|Geschädigten|Rechtsanwalt|Rechtsanwältin|Vorsitzender|Richter|Richterin|Herr|Herrn)\\s+(?:Schäfer|Brückner|Volz|Treber|Knoll|Kriener|Nielsen|Kortbein|Schmid|Söchtig|Kortge|Jacobi|Schödel|Fiamingo|Kelvin|Schilling|Zimmermann|Dölp|Wollensak|Heinkel|Wemheuer|Sost-Scheible|Kirchhof|Paul Kirchhof|Koch|Brune|Grüneberg|Hayen|Lipphaus|Merkel|Eylert|Krumbiegel|Kayser|Jäger|Merzbach|Hacker|Meiser|Becker|Zeng|Merk|Beji Caid Essebsi|Gericke|Franke|Busch|Bender|Augat|Tiemann|Löffler|Schlünder|Schmidt|Schmalz|Melzacks|Edda Redeker|Rosen|Kortbein|Schmid|Söchtig|Kortge|Jacobi|Schödel|Linck|Schultz|Bellay|Leitz|Fieback|Rachor|Cosima|Hoch|Appl|Berger|Quentin|Roloff|Lohmann|Raum|Spinner|St|Br\\.|B1|S3|S4|Karaçay|Schramm|Egerer|Kätker|Wismeth|Freudenreich|Schwitzer|Enerji Yapi-Yol Sen|Schwabe|Paffrath|Derstadt|Gallner|Herrmann|Shah|Krasshöfer|Limperg|Mosbacher|Schneider|Niemann|Zwanziger|Brenneisen|Hausmann|Kazele|Hohoff|Roggenbuck|Hamdorf|Grabinski|Krehl|Kosziol|Sunder|Mayen|Seiters|Schlewing|Spaniol|Kirchhoff|Fritz|Vogelsang|Lauer|Mutzbauer|Cierniak|Müller|Ahrendt|Dö.|Widuch|Menezes|Sander|Fischermeier|Hoffmann|Kleinschmidt|Kirschneck|Matter|Kapels|Jostes|Da.|Maksymiw|Schell|Münzberg|D7|Peter Lorsbach|Lorsbach|D1)\\b', 'priority': 13, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:08:59.297356', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '5ae5e583', 'name': 'Initials with dots (standalone context)', 'description': "Captures single initials with dots (e.g., 'A.', 'S.') when they appear in contexts suggesting a name, such as after 'Dr.', 'Prof.', 'Herr', or at the start of a sentence followed by a verb.", 'format': 'regex', 'content': '(?:Dr\\.?\\s+|Prof\\.?\\s+|Professor\\s+|Herr\\s+|Herrn\\s+|^|\\b)([A-Z]\\.)\\b', 'priority': 12, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:07:53.410426', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '266c1518', 'name': 'Full names with initials (e.g., K. Schmidt)', 'description': "Captures full names consisting of an initial and a surname (e.g., 'K. Schmidt', 'M. Rennpferdt').", 'format': 'regex', 'content': '\\b([A-Z]\\.)\\s+([A-Z][a-zäöüß]+(?:\\s+[A-Z][a-zäöüß]+)*)\\b', 'priority': 11, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:07:53.410538', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '76281285', 'name': 'Multi-initial anonymized names', 'description': "Captures multi-initial anonymized names like 'M. D.' or 'A. S.'.", 'format': 'regex', 'content': '\\b([A-Z]\\.)\\s+([A-Z]\\.)\\b', 'priority': 10, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:07:53.410643', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '381264c3', 'name': 'Anonymized initials after legal roles', 'description': "Captures single-letter anonymized names (e.g., 'M.', 'F.', 'R', 'A') immediately following legal role indicators like 'Angeklagte', 'Kläger', 'Zeuge', etc., including those with or without dots.", 'format': 'regex', 'content': '\\b(?:Angeklagte|Angeklagten|Mitangeklagte|Mitangeklagten|Kläger|Klägerin|Klägers|Klägerin|Zeuge|Zeugin|Geschädigte|Geschädigten|Beteiligte|Beteiligten|Antragsteller|Antragstellerin|Antragstellers|Antragstellerin|Vorsitzender|Richter|Richterin|Rechtsanwalt|Rechtsanwältin|Gutachter|Gutachterin|Sachverständige|Sachverständigen|Herr|Herrn)\\s+([A-Z]\\.?)\\b', 'priority': 15, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:08:59.296128', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': 'a8cfecf8', 'name': 'Isolated anonymized initials', 'description': "Captures single-letter anonymized names (e.g., 'A', 'K', 'E', 'S') appearing in legal contexts such as after prepositions ('von', 'zu', 'in'), after 'der/die/das', or at the start of a sentence, with or without dots.", 'format': 'regex', 'content': '(?:\\b(?:von|zu|in|an|auf|bei|nach|vor|mit|ohne|für|gegen|durch|über|unter|neben|zwischen|hinter|vor|nach|um|bis|seit|während|trotz|wegen|statt|außer|neben|gegenüber|entlang|laut|gemäß|inklusive|exklusive|sowie|wie|als|obwohl|wenn|falls|da|denn|dass|ob|obwohl|obgleich|indem|ohne|ohne\\s+dass|solange|sobald|bis|seit|während|trotz|wegen|statt|außer|neben|gegenüber|entlang|laut|gemäß|inklusive|exklusive|sowie|wie|als|obwohl|wenn|falls|da|denn|dass|ob|obwohl|obgleich|indem)\\s+|\\b(?:der|die|das|den|dem|des|ein|eine|einem|einen|einer|eines|mein|dein|sein|ihr|unser|euer|ihr|mein|dein|sein|ihr|unser|euer|ihr|der|die|das|den|dem|des|ein|eine|einem|einen|einer|eines)\\s+|\\b(?:Kläger|Angeklagter|Angeklagte|Zeuge|Zeugin|Geschädigte|Geschädigter|Beteiligte|Beteiligter|Antragsteller|Antragstellerin|Vorsitzender|Richter|Richterin|Rechtsanwalt|Rechtsanwältin|Gutachter|Gutachterin|Sachverständige|Sachverständiger|Herr|Frau)\\s+|\\b(?:in|zu|von|an|auf|bei|nach|vor|mit|ohne|für|gegen|durch|über|unter|neben|zwischen|hinter|vor|nach|um|bis|seit|während|trotz|wegen|statt|außer|neben|gegenüber|entlang|laut|gemäß|inklusive|exklusive|sowie|wie|als|obwohl|wenn|falls|da|denn|dass|ob|obwohl|obgleich|indem)\\s+|\\b(?:der|die|das|den|dem|des|ein|eine|einem|einen|einer|eines)\\s+|^)([A-Z]\\.?)\\b', 'priority': 12, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:08:59.296140', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}, {'id': '295b59c5', 'name': 'Isolated known surnames', 'description': "Captures specific known surnames (e.g., 'Knoll', 'Kriener', 'Schmid') appearing in isolation or at the start of a sentence, treating them as PER.", 'format': 'regex', 'content': '\\b(?:Schäfer|Brückner|Volz|Treber|Knoll|Kriener|Nielsen|Kortbein|Schmid|Söchtig|Kortge|Jacobi|Schödel|Fiamingo|Kelvin|Schilling|Zimmermann|Dölp|Wollensak|Heinkel|Wemheuer|Sost-Scheible|Kirchhof|Paul Kirchhof|Koch|Brune|Grüneberg|Hayen|Lipphaus|Merkel|Eylert|Krumbiegel|Kayser|Jäger|Merzbach|Hacker|Meiser|Becker|Zeng|Merk|Beji Caid Essebsi|Gericke|Franke|Busch|Bender|Augat|Tiemann|Löffler|Schlünder|Schmidt|Schmalz|Melzacks|Edda Redeker|Rosen|Kortbein|Schmid|Söchtig|Kortge|Jacobi|Schödel|Linck|Schultz|Bellay|Leitz|Fieback|Rachor|Cosima|Hoch|Appl|Berger|Quentin|Roloff|Lohmann|Raum|Spinner|St|Br\\.|B1|S3|S4|Karaçay|Schramm|Egerer|Kätker|Wismeth|Freudenreich|Schwitzer|Enerji Yapi-Yol Sen|Schwabe|Paffrath|Derstadt|Gallner|Herrmann|Shah|Krasshöfer|Limperg|Mosbacher|Schneider|Niemann|Zwanziger|Brenneisen|Hausmann|Kazele|Hohoff|Roggenbuck|Hamdorf|Grabinski|Krehl|Kosziol|Sunder|Mayen|Seiters|Schlewing|Spaniol|Kirchhoff|Fritz|Vogelsang|Lauer|Mutzbauer|Cierniak|Müller|Ahrendt|Dö.|Widuch|Menezes|Sander|Fischermeier|Hoffmann|Kleinschmidt|Kirschneck|Matter|Kapels|Jostes|Da.|Maksymiw|Schell|Münzberg|D7|Peter Lorsbach|Lorsbach|D1)\\b', 'priority': 14, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-23T07:08:59.297369', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'PER'}, 'output_key': 'entities'}] |

<details>
<summary>Results</summary>

| Metric | Value |
|---|---|
| Accuracy (exact match) | 87.5% |
| True Positives | 156 |
| False Positives | 799 |
| False Negatives | 168 |
| Total Gold Entities | 324 |
| Micro Precision | 16.3% |
| Micro Recall | 48.1% |
| Micro F1 | 24.4% |
| Macro F1 | 24.4% |

</details>

---

<details>
<summary>📊 Summary</summary>

| Rule | F1 | Precision | Recall | Total Predicted | True Positives | False Positives |
|---|---|---|---|---|---|---|
| `Anonymized initials with ellipses after roles` | 3.0% | 100.0% | 1.5% | 5 | 5 | 0 |
| `Anonymized initials after legal roles` | 11.1% | 100.0% | 5.9% | 19 | 19 | 0 |
| `Names after 'Richter' or 'Vorsitzender'` | 1.8% | 50.0% | 0.9% | 6 | 3 | 3 |
| `Names after 'Herr' or 'Herrn'` | 0.6% | 33.3% | 0.3% | 3 | 1 | 2 |
| `Anonymized initials with dots (multi-initial)` | 4.1% | 33.3% | 2.2% | 21 | 7 | 14 |
| `Isolated anonymized initials` | 9.2% | 26.1% | 5.6% | 69 | 18 | 51 |
| `Isolated known surnames` | 19.1% | 14.6% | 27.5% | 609 | 89 | 520 |
| `Full names with titles` | 0.6% | 11.1% | 0.3% | 9 | 1 | 8 |
| `Full names with initials (e.g., K. Schmidt)` | 5.4% | 9.8% | 3.7% | 122 | 12 | 110 |
| `Hyphenated surnames` | 0.5% | 1.5% | 0.3% | 65 | 1 | 64 |
| `Anonymized initials with dots` | 0.0% | 0.0% | 0.0% | 5 | 0 | 5 |
| `Initials with dots and spaces (e.g., T. D.)` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Names after 'Herr'` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Names after 'Angeklagte' or 'Klägerin'` | 0.0% | 0.0% | 0.0% | 12 | 0 | 12 |
| `Names after 'Dr.' or 'Prof.'` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Names after 'Rechtsanwältin' or 'Rechtsanwalt'` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Names with dots in middle (e.g., B1 …)` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Anonymized names with ellipses` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Surnames after 'Generalanwalt' or 'Generalanwältin'` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Surnames after 'Richter' or 'Vorsitzender' (refined)` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Surnames after 'Rechtsanwalt' or 'Rechtsanwältin' (refined)` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Known surnames after legal roles` | 0.0% | 0.0% | 0.0% | 10 | 0 | 10 |
| `Initials with dots (standalone context)` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Multi-initial anonymized names` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |
| `Names after 'Dr.' or 'Prof.' (standalone)` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 |

</details>

---

<details>
<summary>🏆 Most Precise Rules</summary>

## `Anonymized initials after legal roles`

**F1:** 0.111 | **Precision:** 1.000 | **Recall:** 0.059  

**Format:** `regex`  
**Rule ID:** `4d75d5d9`  
**Description:**
Captures single-letter anonymized names (e.g., 'A', 'K', 'E', 'S') immediately following legal role indicators like 'Angeklagte', 'Kläger', 'Zeuge', etc., ensuring the dot is included if present.

**Content:**
```
\b(?:Angeklagte|Angeklagten|Mitangeklagte|Mitangeklagten|Kläger|Klägerin|Klägers|Klägerin|Zeuge|Zeugin|Geschädigte|Geschädigten|Beteiligte|Beteiligten|Antragsteller|Antragstellerin|Antragstellers|Antragstellerin|Vorsitzender|Richter|Richterin|Rechtsanwalt|Rechtsanwältin|Gutachter|Gutachterin|Sachverständige|Sachverständigen|Herr|Herrn|Dr\.?\s+|Prof\.?\s+|Dipl\.-[A-Za-z]+\s+)([A-Z]\.)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.059 | 0.111 | 19 | 19 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 19 | 0 | 302 |

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

**Example 5** (doc_id: `61586`) (sent_id: `61586`)


Die Klägerin hatte ihren Sitz zum Zeitpunkt des Eintritts der Job-Sharing-Partnerin Dr. E. im Bezirk der KÄV Nordbaden .

| Predicted | Gold |
|---|---|
| `E.` | `E.` |

**Missed by this rule (FN):**

- `KÄV Nordbaden` (ORG)

**Example 6** (doc_id: `61864`) (sent_id: `61864`)


Zu dem Sachverständigengutachten des Prof. Dr. T. sowie dessen ergänzender Stellungnahme hat die Beklagte mitgeteilt , dass sie diesem nach prüfärztlicher Stellungnahme nicht folgen könne ; der Einschätzung des Dr. S. hat sie sich hingegen angeschlossen .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |
| `S.` | `S.` |

**Example 7** (doc_id: `62635`) (sent_id: `62635`)


Schließlich wird das Stellen eines ordnungsgemäßen Beweisantrags mit der Beschwerdebegründung nicht dargelegt , soweit die Klägerin die Sachaufklärungspflicht des LSG dadurch verletzt sieht , dass dieses keine ergänzende gutachterliche Äußerung Dr. R. eingeholt hat .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |

**Example 8** (doc_id: `62752`) (sent_id: `62752`)


Die Implantation der Coils als alleiniger Grund für die stationäre Behandlung der Versicherten sei nach dem überzeugenden MDK-Gutachten ( Dr. S. ) nicht erforderlich gewesen .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Example 9** (doc_id: `63754`) (sent_id: `63754`)


Soweit das LSG Ausführungen von Prof. Dr. S. referiert , dass sich im Zusammenhang mit den Pneumonien auch immer wieder Septitiden entwickelt hätten , macht das LSG nicht deutlich , dass es diese als festgestellt ansieht , und erst recht nicht , von welchem Begriff oder Schweregrad der Sepsis es im Anschluss an Prof. Dr. S. aufgrund welcher Befunde dabei ausgeht ( ältere Klassifizierungen : Systemisches inflammatorisches Response-Syndrom < SIRS > , Sepsis , schwere Sepsis und septischer Schock ; zur darauf aufbauenden DRG-Kodierung vgl Hanser in Zaiß , DRG : Verschlüsseln leicht gemacht , 14. Aufl 2016 , S 101 ff ; neuere Klassifizierungen : Sepsis-related organ failure assessment score < SOFA > , insbesondere für Intensivstationen , und quickSOFA außerhalb von Intensivstationen ) .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |
| `S.` | `S.` |

**Example 10** (doc_id: `63885`) (sent_id: `63885`)


Nach den Ausführungen des im Verfahren von Amts wegen gehörten Sachverständigen Prof. Dr. T. hätten die vom Kläger vorgetragenen Gewalterfahrungen während seiner Heimaufenthalte ab April 1959 nicht die entscheidende Traumatisierung dargestellt .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |

**Example 11** (doc_id: `63999`) (sent_id: `63999`)


Der weitere vom LSG beauftragte Sachverständige Dr. S. ( Neurologe und Psychiater / Psychotherapeut ) hat die quantitative Leistungsfähigkeit der Klägerin mit mindestens 6 Stunden für leichte Arbeiten mit qualitativen Einschränkungen beurteilt .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Example 12** (doc_id: `64291`) (sent_id: `64291`)


Ferner hat sie beantragt , Prof. Dr. B. , Zentralinstitut für seelische Gesundheit M. , Institut für Psychiatrie und Psychosomatische Psychotherapie , als sachverständigen Zeugen zu vernehmen sowie den Sachverständigen Prof. Dr. S. zu den mit Schriftsatz vom 21. 3. 2017 erhobenen Einwendungen anzuhören .

| Predicted | Gold |
|---|---|
| `B.` | `B.` |
| `S.` | `S.` |

**Missed by this rule (FN):**

- `Zentralinstitut für seelische Gesundheit M. , Institut für Psychiatrie und Psychosomatische Psychotherapie` (ORG)

**Example 13** (doc_id: `64530`) (sent_id: `64530`)


Das LSG hat vielmehr im Anschluss an die Begründung , warum es dessen sachverständige Bewertung für überzeugend hält , ausgeführt : " Hingegen hat Dr. H. lediglich auf einer Seite kurz dargestellt , dass beim Kläger seinerzeit kein KIG Grad 3 oder höher vorliege .

| Predicted | Gold |
|---|---|
| `H.` | `H.` |

**Example 14** (doc_id: `65228`) (sent_id: `65228`)


Statt einer beantragten orthopädischen Begutachtung unter Berücksichtigung der Schmerzsymptomatik sei eine Begutachtung durch Dr. N. angeordnet worden , obwohl er ( der Kläger ) auf neurologischem Gebiet völlig gesund sei .

| Predicted | Gold |
|---|---|
| `N.` | `N.` |

**Example 15** (doc_id: `66269`) (sent_id: `66269`)


Die Beklagte hat sich hierzu nicht geäußert und nach der Übersendung des Sachverständigengutachtens des Dr. B. ohne weitere inhaltliche Einlassung mit einer Entscheidung ohne mündliche Verhandlung einverstanden erklärt .

| Predicted | Gold |
|---|---|
| `B.` | `B.` |

</details>

---

## `Anonymized initials with dots (multi-initial)`

**F1:** 0.041 | **Precision:** 0.333 | **Recall:** 0.022  

**Format:** `regex`  
**Rule ID:** `ae91774f`  
**Description:**
Captures multi-initial anonymized names like 'A. S.' or 'R. C.' as a single entity.

**Content:**
```
(?:[A-Z]\.)\s+(?:[A-Z]\.)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.333 | 0.022 | 0.041 | 21 | 7 | 14 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 7 | 14 | 305 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60448`) (sent_id: `60448`)


Der Kläger nahm Arabisch-Unterricht bei einem T. H. , der der salafistischen Szene zuzurechnen ist und Kontakte zu Personen pflegte , die nach Syrien ausgereist waren oder dies versucht hatten .

| Predicted | Gold |
|---|---|
| `T. H.` | `T. H.` |

**Missed by this rule (FN):**

- `Syrien` (LOC)

**Example 1** (doc_id: `61032`) (sent_id: `61032`)


b ) Der Antragsteller hatte aber bei der Feststellung der beschränkten Dienstfähigkeit der Lehrerin K. B. und der Herabsetzung ihrer Arbeitszeit gemäß § 27 BeamtStG in analoger Anwendung des § 68 Abs. 1 Nr. 6 PersVG BB mitzuwirken .

| Predicted | Gold |
|---|---|
| `K. B.` | `K. B.` |

**Missed by this rule (FN):**

- `§ 27 BeamtStG` (NRM)
- `§ 68 Abs. 1 Nr. 6 PersVG BB` (NRM)

**Example 2** (doc_id: `61243`) (sent_id: `61243`)


Doch auch wenn die Taten 21 bis 23 und 33 der Urteilsgründe sich damit im Hinblick auf das Marihuana als selbständige Umsatzgeschäfte darstellen , fallen die darauf bezogenen Handlungen des Angeklagten mit denjenigen zusammen , die dem Absatz des zugleich in diesen Fällen gehandelten Amphetamins dienten , hinsichtlich dessen aufgrund des einheitlichen Erwerbs im Fall 27 der Urteilsgründe von einer Bewertungseinheit und damit von einer Tat im Rechtssinne auszugehen ist : Im Fall 21 der Urteilsgründe nahm der Angeklagte die Bestellung beider Betäubungsmittel einheitlich entgegen , in den Fällen 22 und 23 der Urteilsgründe lieferte er sie auch gleichzeitig an die Angeklagte A. A. .

| Predicted | Gold |
|---|---|
| `A. A.` | `A. A.` |

**Example 3** (doc_id: `61423`) (sent_id: `61423`)


H. N. ging auf ihn zu , hielt ihm in einer Entfernung von ca. einem Meter ein etwa 22 cm langes Küchenmesser mit ungefähr 11 cm langer Klinge vor die Brust und forderte ihn auf , ihm das auf dem Tresen bzw. in der offenen Kasse liegende Geld zu übergeben .

| Predicted | Gold |
|---|---|
| `H. N.` | `H. N.` |

**Example 4** (doc_id: `62293`) (sent_id: `62293`)


Das Landgericht hat den Angeklagten T. D. wegen schweren sexuellen Missbrauchs eines Kindes in Tateinheit mit sexuellem Missbrauch eines Schutzbefohlenen in zwei Fällen sowie schweren sexuellen Missbrauchs einer widerstandsunfähigen Person in fünf Fällen zu einer Gesamtfreiheitsstrafe von vier Jahren und drei Monaten verurteilt .

| Predicted | Gold |
|---|---|
| `T. D.` | `T. D.` |

**Example 5** (doc_id: `63047`) (sent_id: `63047`)


1. Die Verurteilung des Angeklagten T. D. wegen schweren sexuellen Missbrauchs einer widerstandsunfähigen Person in den Fällen III. 3 bis 7 der Urteilsgründe nach dem zur Tatzeit geltenden § 179 Abs. 1 Nr. 1 , Abs. 5 Nr. 1 StGB in der bis zum 9. November 2016 geltenden Fassung hält revisionsrechtlicher Überprüfung nicht stand , weil die Urteilsgründe eine Widerstandsunfähigkeit des Nebenklägers nicht belegen .

| Predicted | Gold |
|---|---|
| `T. D.` | `T. D.` |

**Missed by this rule (FN):**

- `§ 179 Abs. 1 Nr. 1 , Abs. 5 Nr. 1 StGB` (NRM)

**Example 6** (doc_id: `64492`) (sent_id: `64492`)


Schlussendlich erklärte er : " Ich plane dann mit C. W.

| Predicted | Gold |
|---|---|
| `C. W.` | `C. W.` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60442`) (sent_id: `60442`)


Den unmittelbar Geschädigten W. und M. L. wurde für die Wegnahme ihres landwirtschaftlichen Vermögens in P. eine Hauptentschädigung zuerkannt .

**False Positives:**

- `M. L.` — partial — pred is substring of gold: `W. und M. L.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `W. und M. L.`(PER)
- `P.`(LOC)

**Example 1** (doc_id: `60597`) (sent_id: `60597`)


Aufgrund der Landesverordnung über das Naturschutzgebiet " Liether Kalkgrube " vom 18. 10. 1991 ( GVOBl Schl-H 1992 , S 2 ; in Zukunft : LandesVO L. K. ) seien die Nutzungsmöglichkeiten des Waldes für den Kläger in einem so erheblichen Ausmaß eingeschränkt , dass objektiv keine Bewirtschaftungsmöglichkeit bestehe , die die Vermutung einer forstwirtschaftlichen Tätigkeit rechtfertigen könne ( Urteil vom 27. 6. 2012 ) .

**False Positives:**

- `L. K.` — partial — pred is substring of gold: `Landesverordnung über das Naturschutzgebiet " Liether Kalkgrube " vom 18. 10. 1991 ( GVOBl Schl-H 1992 , S 2 ; in Zukunft : LandesVO L. K. )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Landesverordnung über das Naturschutzgebiet " Liether Kalkgrube " vom 18. 10. 1991 ( GVOBl Schl-H 1992 , S 2 ; in Zukunft : LandesVO L. K. )`(NRM)

**Example 2** (doc_id: `61399`) (sent_id: `61399`)


Zuletzt legt der Kläger einen Auszug aus dem niederländischen Handelsregister vor , aus dem sich die Auflösung der C- B. V. am ... Mai 2006 ergibt .

**False Positives:**

- `B. V.` — partial — pred is substring of gold: `C- B. V.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C- B. V.`(ORG)

**Example 3** (doc_id: `61627`) (sent_id: `61627`)


D15 P. W. McMillan et al. , “ The Structure and Properties of a Lithium Zinc Silicate Glass-Ceramic ” , Journal of Materials Science , 1966 , 1 , Seiten 269 bis 279

**False Positives:**

- `P. W.` — partial — pred is substring of gold: `P. W. McMillan`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `P. W. McMillan`(PER)

**Example 4** (doc_id: `62115`) (sent_id: `62115`)


A. I. 1. Der Beschwerdeführer ist auf Grundlage des Urteils des Landgerichts Lübeck vom 7. Oktober 2014 gemäß § 63 StGB wegen Mordes in einem psychiatrischen Krankenhaus der AMEOS Krankenhausgesellschaft Holstein mbH untergebracht , nachdem er im schuldunfähigen Zustand auf Grund einer wahnhaften Störung im Januar 2014 seine vierjährige Tochter und seinen sechs Jahre alten Sohn getötet hatte .

**False Positives:**

- `A. I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Landgerichts Lübeck`(ORG)
- `§ 63 StGB`(NRM)
- `AMEOS Krankenhausgesellschaft Holstein mbH`(ORG)

**Example 5** (doc_id: `63055`) (sent_id: `63055`)


Die Markenstelle verweist zur Begründung der Schutzunfähigkeit des Wortes „ wir “ unter anderem auf die sehr alte BPatG-Entscheidung zur Anmeldung „ W. I. R Zeitarbeit “ ( Beschluss vom 17. 07. 1996 , 29 W ( pat ) 9/95 ) .

**False Positives:**

- `W. I.` — partial — pred is substring of gold: `BPatG-Entscheidung zur Anmeldung „ W. I. R Zeitarbeit “ ( Beschluss vom 17. 07. 1996 , 29 W ( pat ) 9/95 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG-Entscheidung zur Anmeldung „ W. I. R Zeitarbeit “ ( Beschluss vom 17. 07. 1996 , 29 W ( pat ) 9/95 )`(RS)

**Example 6** (doc_id: `63105`) (sent_id: `63105`)


Weiterhin legt er Presseartikel vor , wonach am ... Juni 2003 , am ... August 2003 und am ... März 2004 von verschiedener Stelle aus öffentlich Zweifel an der Bonität der C- B. V. geäußert werden .

**False Positives:**

- `B. V.` — partial — pred is substring of gold: `C- B. V.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C- B. V.`(ORG)

**Example 7** (doc_id: `63990`) (sent_id: `63990`)


Vielmehr müssen die steuerlichen Vorteile der Typisierung im rechten Verhältnis zu der mit der Typisierung notwendig verbundenen Ungleichheit der steuerlichen Belastung stehen ( vgl. z.B. BVerfG-Urteil vom 20. April 2004 1 BvR 1748/99 , 1 BvR 905/00 , BVerfGE 110 , 274 ; BVerfG-Beschluss vom 15. Januar 2008 1 BvL 2/04 , BVerfGE 120 , 1 , unter C. I. 2. a aa ) .

**False Positives:**

- `C. I.` — partial — pred is substring of gold: `BVerfG-Beschluss vom 15. Januar 2008 1 BvL 2/04 , BVerfGE 120 , 1 , unter C. I. 2. a aa`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG-Urteil vom 20. April 2004 1 BvR 1748/99 , 1 BvR 905/00 , BVerfGE 110 , 274`(RS)
- `BVerfG-Beschluss vom 15. Januar 2008 1 BvL 2/04 , BVerfGE 120 , 1 , unter C. I. 2. a aa`(RS)

**Example 8** (doc_id: `64059`) (sent_id: `64059`)


( c ) Ob für die in § 7 Satz 2 Hs. 2 GewStG geschaffene Privilegierung der auf unmittelbar beteiligte natürliche Personen entfallenden Veräußerungsgewinne daneben noch weitere Motive des Gesetzgebers eine Rolle gespielt haben - wie etwa die Schonung des Mittelstandes ( vgl. Bericht der Bundesregierung vom 18. April 2001 a. a. O. unter B. I. 6 ; s. o. A I 3 b aa ( 2 ) ) - ist neben den tragfähigen Überlegungen zur Umgehungsverhinderung nicht erheblich .

**False Positives:**

- `B. I.` — partial — pred is substring of gold: `Bericht der Bundesregierung vom 18. April 2001 a. a. O. unter B. I. 6`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 7 Satz 2 Hs. 2 GewStG`(NRM)
- `Bericht der Bundesregierung vom 18. April 2001 a. a. O. unter B. I. 6`(LIT)

**Example 9** (doc_id: `64559`) (sent_id: `64559`)


aa ) Die unter Beweis gestellte fehlende Werthaltigkeit der Forderung der A-GbR gegen die C- B. V. zum 31. Dezember 2004 ist entscheidungserheblich .

**False Positives:**

- `B. V.` — partial — pred is substring of gold: `C- B. V.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `A-GbR`(ORG)
- `C- B. V.`(ORG)

**Example 10** (doc_id: `65862`) (sent_id: `65862`)


In der mündlichen Verhandlung vor dem FG beantragte der Prozessbevollmächtigte des Klägers , die in einem Schriftsatz zuvor benannten Zeugen zu den dort genannten Beweisthemen zu vernehmen und rügte die Rechtsverletzung des Klägers durch Unterlassen weiterer Sachaufklärung und Zeugenvernehmung , insbesondere zur wirtschaftlichen Situation der C- B. V.

**False Positives:**

- `B. V.` — partial — pred is substring of gold: `C- B. V.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C- B. V.`(ORG)

**Example 11** (doc_id: `66066`) (sent_id: `66066`)


Hierfür erhält sie von der Prinzipalin , der E K S. A. R. L. G - einer Schwestergesellschaft - eine umsatzbezogene Vergütung .

**False Positives:**

- `S. A.` — partial — pred is substring of gold: `E K S. A. R. L. G`
- `R. L.` — partial — pred is substring of gold: `E K S. A. R. L. G`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `E K S. A. R. L. G`(ORG)

**Example 12** (doc_id: `66113`) (sent_id: `66113`)


Bei anschließenden Internetrecherchen wurde ein mit großer Wahrscheinlichkeit dem Kläger zuzuordnendes ask . fm-Profil eines " C. J. " aufgefunden , das die Flagge des sogenannten Islamischen Staates zeigte und weitere salafistische Inhalte aufwies .

**False Positives:**

- `C. J.` — partial — pred is substring of gold: `" C. J. "`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `" C. J. "`(PER)
- `Islamischen Staates`(ORG)

</details>

---

## `Isolated anonymized initials`

**F1:** 0.092 | **Precision:** 0.261 | **Recall:** 0.056  

**Format:** `regex`  
**Rule ID:** `959f7758`  
**Description:**
Captures single-letter anonymized names (e.g., 'A', 'K', 'E', 'S') appearing in legal contexts such as after prepositions ('von', 'zu', 'in'), after 'der/die/das', or at the start of a sentence, with a dot.

**Content:**
```
(?:\b(?:von|zu|in|an|auf|bei|nach|vor|mit|ohne|für|gegen|durch|über|unter|neben|zwischen|hinter|vor|nach|um|bis|seit|während|trotz|wegen|statt|außer|neben|gegenüber|entlang|laut|gemäß|inklusive|exklusive|sowie|wie|als|obwohl|wenn|falls|da|denn|dass|ob|obwohl|obgleich|indem)\s+|\b(?:der|die|das|den|dem|des|ein|eine|einem|einen|einer|eines|mein|dein|sein|ihr|unser|euer|ihr|mein|dein|sein|ihr|unser|euer|ihr|der|die|das|den|dem|des|ein|eine|einem|einen|einer|eines)\s+|\b(?:Kläger|Angeklagter|Angeklagte|Zeuge|Zeugin|Geschädigte|Geschädigter|Beteiligte|Beteiligter|Antragsteller|Antragstellerin|Vorsitzender|Richter|Richterin|Rechtsanwalt|Rechtsanwältin|Gutachter|Gutachterin|Sachverständige|Sachverständiger|Herr|Frau)\s+|\b(?:in|zu|von|an|auf|bei|nach|vor|mit|ohne|für|gegen|durch|über|unter|neben|zwischen|hinter|vor|nach|um|bis|seit|während|trotz|wegen|statt|außer|neben|gegenüber|entlang|laut|gemäß|inklusive|exklusive|sowie|wie|als|obwohl|wenn|falls|da|denn|dass|ob|obwohl|obgleich|indem)\s+|\b(?:der|die|das|den|dem|des|ein|eine|einem|einen|einer|eines)\s+|^)([A-Z]\.)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.261 | 0.056 | 0.092 | 69 | 18 | 51 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 18 | 51 | 305 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60270`) (sent_id: `60270`)


Auch der M. wollte nach Angaben des Klägers einen Anschlag auf Zivilisten planen ; hierzu erklärte sich der Kläger ohne Einschränkungen bereit .

| Predicted | Gold |
|---|---|
| `M.` | `M.` |

**Example 1** (doc_id: `60282`) (sent_id: `60282`)


Diese sind rechtswidrig und beschweren die Klägerin , soweit sie Honorar für RLV-Leistungen nicht auch unter Anwendung eines arztpraxisbezogenen RLV , sondern lediglich unter Zugrundelegung einer Obergrenze zuerkennen , deren Höhe von der Zahl der durch S. im streitbefangenen Quartal tatsächlich behandelten Patienten abhängt .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Example 2** (doc_id: `60683`) (sent_id: `60683`)


Ab dem Inkrafttreten des Bundessozialhilfegesetzes ( BSHG ) in den neuen Bundesländern zum 1. 1. 1991 erbrachte das Land B. als der nach Landesrecht zuständige überörtliche Träger der Sozialhilfe Leistungen der Eingliederungshilfe an K.

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `Bundessozialhilfegesetzes` (NRM)
- `BSHG` (NRM)
- `B.` (LOC)

**Example 3** (doc_id: `61486`) (sent_id: `61486`)


Das Gericht wies am dritten Hauptverhandlungstag im Zusammenhang mit einem Antrag von Rechtsanwalt P. , den dieser unter Bezugnahme auf das zuvor genannte Schreiben begründet hatte , unter anderem darauf hin , dass sich in der Akte ein „ Terminverlegungsantrag vom 12. April 2016 “ befinde .

| Predicted | Gold |
|---|---|
| `P.` | `P.` |

**Example 4** (doc_id: `61657`) (sent_id: `61657`)


Dort ist im Einzelnen dargelegt , dass die Anlagen K 1. K 3 , K 5 , K 9 und K 50 jeweils Hinweise darauf enthalten , dass S. die Verhandlungen für die Help Food und nicht für die Beklagte führte .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Missed by this rule (FN):**

- `Help Food` (ORG)

**Example 5** (doc_id: `61871`) (sent_id: `61871`)


Als die Geschädigte während dieses Geschehens von der Zeugin K. angerufen wurde , riss M. der Geschädigten das Mobiltelefon aus der Hand und nahm es im Einverständnis mit dem Angeklagten R. an sich , um zu verhindern , dass die Geschädigte um Hilfe rief .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `M.` (PER)
- `R.` (PER)

**Example 6** (doc_id: `62034`) (sent_id: `62034`)


Die entsprechenden Feststellungen wird das LSG allerdings nur dann nachzuholen haben , wenn K. nicht ohnedies während ihrer Teilnahme am Modellprojekt Enthospitalisierung in Wi. und damit im Zuständigkeitsbereich des Klägers , ihren letzten gewöhnlichen Aufenthalt vor Aufnahme in die Außenwohngruppe im Jahr 2005 begründet hat .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `Wi.` (ORG)

**Example 7** (doc_id: `62385`) (sent_id: `62385`)


Der Nebenkläger war nämlich durch M. hinreichend geschützt .

| Predicted | Gold |
|---|---|
| `M.` | `M.` |

**Example 8** (doc_id: `62485`) (sent_id: `62485`)


Die Kammer sei auch nicht in der Lage , Spruchreife herzustellen , weil dazu Erbscheine der Erbeserben des S. erforderlich seien .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Example 9** (doc_id: `62987`) (sent_id: `62987`)


Zugleich hat der Kläger die Richtigkeit des gesamten bisherigen Vorbringens des Beklagten zum tatsächlichen Verwaltungsaufwand und zur Minutenberechnung erneut ausdrücklich bestritten und eine vom Beklagten als Anlage zu einem Vermerk vom 22. Juli 2013 vorgelegte " Zeiterfassung bei der Bearbeitung repräsentativer Fälle durch Frau B. " , aus der sich angeblich eine mittlere Bearbeitungszeit von ca. 27,25 bis 27,625 Minuten ergebe , als nicht nachvollziehbar bezeichnet .

| Predicted | Gold |
|---|---|
| `B.` | `B.` |

**Example 10** (doc_id: `63898`) (sent_id: `63898`)


Dem Kläger dürfte dies zumindest außerhalb von P. auch ohne Freunde oder Verwandte möglich sein , zumal nicht alle Vermieter nur an ethnische Russen vermieten .

| Predicted | Gold |
|---|---|
| `P.` | `P.` |

**Example 11** (doc_id: `64271`) (sent_id: `64271`)


Der Antrag des Klägers , ihm für das Verfahren der Beschwerde gegen die Nichtzulassung der Revision im Urteil des Landessozialgerichts Niedersachsen-Bremen vom 16. November 2017 Prozesskostenhilfe zu bewilligen und Rechtsanwältin K. aus H. beizuordnen , wird abgelehnt .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `Landessozialgerichts Niedersachsen-Bremen` (ORG)
- `H.` (LOC)

**Example 12** (doc_id: `65166`) (sent_id: `65166`)


Am 10. Oktober 2016 fuhren A. , F. und Z. gemeinsam nach F. , der Angeklagte brachte mit einem Mietwagen drei Fahrräder nach Deutschland .

| Predicted | Gold |
|---|---|
| `F.` | `F.` |

**Missed by this rule (FN):**

- `A.` (PER)
- `Z.` (PER)
- `Deutschland` (LOC)

**Example 13** (doc_id: `65282`) (sent_id: `65282`)


Die Taten fanden ein Ende , nachdem die Zeugin R. im Sexualkundeunterricht aufgeklärt worden war .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |

**Example 14** (doc_id: `65905`) (sent_id: `65905`)


2. Im Zuge eines von J. betriebenen Verfahrens der einstweiligen Verfügung verurteilten das Landgericht F. ( P. ) und letztinstanzlich das Pfälzische Oberlandesgericht Zweibrücken die Beschwerdeführerin antragsgemäß zum Abdruck der folgenden Gegendarstellung , wobei die Größe des Wortes " Gegendarstellung " der Größe der Schrift der Worte " Sterbedrama um seinen besten Freund " und der Text der Gegendarstellung im Übrigen der Schriftgröße der Zeile " Hätte er ihn damals retten können ? " zu entsprechen hatten :

| Predicted | Gold |
|---|---|
| `J.` | `J.` |

**Missed by this rule (FN):**

- `Landgericht F. ( P. )` (ORG)
- `Pfälzische Oberlandesgericht Zweibrücken` (ORG)

**Example 15** (doc_id: `66575`) (sent_id: `66575`)


Die vorliegenden Eingangsrechnungen der Gaststätte lauten teilweise auf den Namen der Gaststätte , teilweise auf die Klägerin und teilweise auf A.

| Predicted | Gold |
|---|---|
| `A.` | `A.` |

**Example 16** (doc_id: `66608`) (sent_id: `66608`)


Nach den getroffenen Feststellungen ist unzweifelhaft , dass der Zeuge K. , der im Fall II. 1. der Urteilsgründe selbst Cannabis vom Angeklagten erhielt und weiterverkaufte , dabei auch in der Vorstellung , den Betäubungsmittelhandel des Angeklagten zu fördern , tätig wurde .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Example 17** (doc_id: `66649`) (sent_id: `66649`)


M. hielt der Zeugin Mund und Nase zu , so dass sie nicht mehr schreien konnte .

| Predicted | Gold |
|---|---|
| `M.` | `M.` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60116`) (sent_id: `60116`)


Die Herstellung der dentalen Restauration erfolgt gemäß Beispiel 26 durch Heißpressen ( vgl. D13 , S. 16 , Bsp. 26 i. V. m. S. 9 , [ 0155 ] bis S. 10 , [ 0162 ] , S. 11/12 Bsp. 6 ) .

**False Positives:**

- `S.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `60183`) (sent_id: `60183`)


Auf der Flucht legten sie an einem zuvor bestimmten Platz am Teich des Kurparks die Rucksäcke mit der Tatbeute ab und fuhren mit dem Zug nach F. .

**False Positives:**

- `F.` — type mismatch — same span as gold: `F.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `F.`(LOC)

**Example 2** (doc_id: `60310`) (sent_id: `60310`)


Läge hingegen ein Fall des ambulant-betreuten Wohnens vor , hätte K. in Wi. , dem Ort , an dem die Wohngemeinschaft belegen war , ihren letzten gewöhnlichen Aufenthalt vor der Wiederaufnahme in das A. -Zentrum im November 1994 begründet ; § 109 SGB XII bzw § 109 BSHG stehen nur bei einem stationären Aufenthalt der Begründung eines gewöhnlichen Aufenthalts am Anstalts- bzw Einrichtungsort entgegen .

**False Positives:**

- `A.` — partial — pred is substring of gold: `A. -Zentrum`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K.`(PER)
- `Wi.`(ORG)
- `A. -Zentrum`(ORG)
- `§ 109 SGB XII`(NRM)
- `§ 109 BSHG`(NRM)

**Example 3** (doc_id: `60400`) (sent_id: `60400`)


Die disziplinarische Ahndung des Verhaltens des Beschwerdeführers zu I. sowie der Beschwerdeführerinnen zu II. bis IV. durch Verfügungen ihrer Dienstherren und deren disziplinargerichtliche Bestätigung durch die angegriffenen Gerichtsentscheidungen begrenzen die Möglichkeit zur Teilnahme an einem Arbeitskampf .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `60404`) (sent_id: `60404`)


3. Wegen dieser Berichterstattung betrieben der Kläger , die P. AG und die H. AG jeweils Unterlassungsverfahren gegen die Beschwerdeführerin ; im Fall des Klägers verbunden mit einer Klage auf Richtigstellung .

**False Positives:**

- `P.` — partial — pred is substring of gold: `P. AG`
- `H.` — partial — pred is substring of gold: `H. AG`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `P. AG`(ORG)
- `H. AG`(ORG)

**Example 5** (doc_id: `60442`) (sent_id: `60442`)


Den unmittelbar Geschädigten W. und M. L. wurde für die Wegnahme ihres landwirtschaftlichen Vermögens in P. eine Hauptentschädigung zuerkannt .

**False Positives:**

- `P.` — type mismatch — same span as gold: `P.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `W. und M. L.`(PER)
- `P.`(LOC)

**Example 6** (doc_id: `60654`) (sent_id: `60654`)


Die Fachgerichte sind jedoch durch Art. 100 Abs. 1 GG nicht gehindert , schon vor der im Hauptsacheverfahren einzuholenden Entscheidung des BVerfG auf der Grundlage ihrer Rechtsauffassung vorläufigen Rechtsschutz zu gewähren , wenn dies im Interesse eines effektiven Rechtsschutzes geboten erscheint und die Hauptsacheentscheidung dadurch nicht vorweggenommen wird ( vgl. BVerfG-Beschluss vom 24. Juni 1992 1 BvR 1028/91 , BVerfGE 86 , 382 , unter B. II. 2. b ; BFH-Beschluss in BFHE 204 , 39 , BStBl II 2004 , 367 ) .

**False Positives:**

- `B.` — partial — pred is substring of gold: `BVerfG-Beschluss vom 24. Juni 1992 1 BvR 1028/91 , BVerfGE 86 , 382 , unter B. II. 2. b`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 100 Abs. 1 GG`(NRM)
- `BVerfG`(ORG)
- `BVerfG-Beschluss vom 24. Juni 1992 1 BvR 1028/91 , BVerfGE 86 , 382 , unter B. II. 2. b`(RS)
- `BFH-Beschluss in BFHE 204 , 39 , BStBl II 2004 , 367`(RS)

**Example 7** (doc_id: `60666`) (sent_id: `60666`)


Nach der vorliegenden Erkenntnislage war es dem Kläger bei Abschiebung grundsätzlich möglich und zumutbar , in der Russischen Föderation etwa in der weiteren , ländlicheren Umgebung von P. legal Wohnsitz zu nehmen und insbesondere registriert zu werden .

**False Positives:**

- `P.` — type mismatch — same span as gold: `P.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Russischen Föderation`(LOC)
- `P.`(LOC)

**Example 8** (doc_id: `60742`) (sent_id: `60742`)


Von einem Bedienhebel nach M4 und Teilmerkmal M5 ist auf den S. 244 - 253 des Fachtagungsbuches keine Rede , denn die Fig. 5 zeigt nur symbolische Darstellungen für die Funksteuerung , das Bedienpult oder das Laptop und lässt allenfalls den Schluss auf Tasten zu , was auch in Übereinstimmung mit den Ausführungen zur Drehzahlregelung über zwei Taster ( Rechts / Links ) steht ( vgl. EI ( D1 ) : S. 249 : „ Funktion der SPCD “ ) .

**False Positives:**

- `S.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 9** (doc_id: `60796`) (sent_id: `60796`)


I. 1. Gegen den Beschwerdeführer wurde bei der Staatsanwaltschaft München I ein Ermittlungsverfahren wegen Betruges geführt .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Staatsanwaltschaft München I`(ORG)

**Example 10** (doc_id: `60890`) (sent_id: `60890`)


I. 1. Das Streitpatent betrifft die Bereitstellung einer den hochselektiven PDE5 -Inhibitor Tadalafil enthaltenden Einheitsdosiszusammensetzung für die Behandlung sexueller Dysfunktion ( vgl. NIK1.3 / NiK1 S. 2 Abs. [ 0002 ] sowie Patentansprüche 1 und 10 ) .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 11** (doc_id: `60942`) (sent_id: `60942`)


Soweit sich gleichwohl aufgrund eines späteren gewillkürten Versicherungsbeginns Nachteile im Versicherungsschutz Betroffener realisieren können , etwa weil infolge der Nichtberücksichtigung von Versicherungszeiten möglicherweise die Voraussetzungen für einen Rentenanspruch nicht erfüllt sind , sollte der spätere Eintritt der Versicherungspflicht außerdem nach § 7a Abs 6 S 1 Nr 1 SGB IV von der Zustimmung des Beschäftigten abhängig gemacht werden ( vgl dazu näher Beschlussempfehlung und Bericht des Ausschusses für Arbeit und Sozialordnung < 11. Ausschuss > , BT-Drucks 14/2046 S 1 unter A. , S 2 unter B. , S 5 unter II. , S 10 < BDA , DAG > und S 13 < Koalitionsfraktionen > ) .

**False Positives:**

- `A.` — partial — pred is substring of gold: `Beschlussempfehlung und Bericht des Ausschusses für Arbeit und Sozialordnung < 11. Ausschuss > , BT-Drucks 14/2046 S 1 unter A. , S 2 unter B. , S 5 unter II. , S 10 < BDA , DAG > und S 13 < Koalitionsfraktionen >`
- `B.` — partial — pred is substring of gold: `Beschlussempfehlung und Bericht des Ausschusses für Arbeit und Sozialordnung < 11. Ausschuss > , BT-Drucks 14/2046 S 1 unter A. , S 2 unter B. , S 5 unter II. , S 10 < BDA , DAG > und S 13 < Koalitionsfraktionen >`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 7a Abs 6 S 1 Nr 1 SGB IV`(NRM)
- `Beschlussempfehlung und Bericht des Ausschusses für Arbeit und Sozialordnung < 11. Ausschuss > , BT-Drucks 14/2046 S 1 unter A. , S 2 unter B. , S 5 unter II. , S 10 < BDA , DAG > und S 13 < Koalitionsfraktionen >`(LIT)

**Example 12** (doc_id: `61021`) (sent_id: `61021`)


Dass der Kläger nach dem Gutachten des Dr. K. noch nicht wie ein Erwachsener wirkt und ihm nach Beobachtungen von Pflegern in der B. er Klinik " jegliche Alltagspraxis " fehle , rechtfertigte bei seiner Abschiebung keine andere Prognose .

**False Positives:**

- `B.` — partial — pred is substring of gold: `B. er Klinik`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K.`(PER)
- `B. er Klinik`(ORG)

**Example 13** (doc_id: `61069`) (sent_id: `61069`)


Nach Zurückverweisung hat das LSG Dr. K. , Institut für neurologisch psychiatrische Begutachtung in B. , mit der Erstellung eines neurologisch-psychiatrischen Gutachtens nach ambulanter Untersuchung des Klägers beauftragt .

**False Positives:**

- `B.` — partial — pred is substring of gold: `Institut für neurologisch psychiatrische Begutachtung in B.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K.`(PER)
- `Institut für neurologisch psychiatrische Begutachtung in B.`(ORG)

**Example 14** (doc_id: `61101`) (sent_id: `61101`)


X.

**False Positives:**

- `X.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 15** (doc_id: `61357`) (sent_id: `61357`)


Mit beim Anwaltsgerichtshof am 6. Oktober 2017 eingegangenem Schreiben vom 5. Oktober 2017 bat der Kläger erneut um Übersendung der Verwaltungsakte an sein Büro in F. .

**False Positives:**

- `F.` — type mismatch — same span as gold: `F.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `F.`(LOC)

**Example 16** (doc_id: `61613`) (sent_id: `61613`)


2. Aufgrund dieses Rauschgiftfunds beantragte Kriminaloberkommissarin Dö. unter Einbindung des zuständigen Staatsanwalts bei dem Ermittlungsrichter des Amtsgerichts Offenbach am Main den Erlass eines Durchsuchungsbeschlusses für die Wohnung des Angeklagten in F. .

**False Positives:**

- `F.` — type mismatch — same span as gold: `F.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Dö.`(PER)
- `Amtsgerichts Offenbach am Main`(ORG)
- `F.`(LOC)

**Example 17** (doc_id: `61961`) (sent_id: `61961`)


I. 1. Nach den Feststellungen des Landgerichts gab der zur Tatzeit 22 Jahre alte Angeklagte , der zuvor Alkohol und Marihuana konsumiert hatte , am späten Abend des 19. November 2015 von einer Telefonzelle aus bei einem Pizza-Lieferservice unter falschem Namen und Angabe einer nicht auf ihn zugelassenen Rufnummer eine Bestellung auf .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 18** (doc_id: `62076`) (sent_id: `62076`)


Am 8. April 2012 fuhr er in B. unter Einfluss von Marihuana mit dem Auto .

**False Positives:**

- `B.` — type mismatch — same span as gold: `B.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `B.`(LOC)

**Example 19** (doc_id: `62082`) (sent_id: `62082`)


Der Ventileinsatz ( valve body 20 ) weist ein Ventil zum Öffnen und Schließen der Saug- und Spülkanäle auf ( vgl. S. 11 Z. 19 bis S. 12 Z. 2 : „ … The transversal through-going bore 48 and the branching-off bore 50 serve the purpose of establishing direct connection between the through-going bore of the tube 12 and the through-going holes of one of the tubular fittings 26 and 30 in a specific activation position . … ” ) [ = Merkmal M6 ] .

**False Positives:**

- `S.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 20** (doc_id: `62169`) (sent_id: `62169`)


Ab August 2002 absolvierte er an einer Berufsfachschule für Sozialassistenz mit Schwerpunkt Sozialpädagogik in H. eine auf zwei Jahre angelegte Ausbildung zum staatlich geprüften Sozialassistenten , die er krankheitsbedingt erst im Juli 2005 abschloss .

**False Positives:**

- `H.` — type mismatch — same span as gold: `H.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `H.`(LOC)

**Example 21** (doc_id: `62343`) (sent_id: `62343`)


I. § 7 Abs. 1 Satz 2 TV AKS 2012 verweist auf § 1 TV AKS 2012 .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 7 Abs. 1 Satz 2 TV AKS 2012`(REG)
- `§ 1 TV AKS 2012`(REG)

**Example 22** (doc_id: `62483`) (sent_id: `62483`)


Gegen 14.45 Uhr rief dieser den Angeklagten an und zitierte ihn zu seinem Garten in M. bei O. , wo der Angeklagte um 15.35 Uhr eintraf .

**False Positives:**

- `M.` — type mismatch — same span as gold: `M.`
- `O.` — type mismatch — same span as gold: `O.`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `M.`(LOC)
- `O.`(LOC)

**Example 23** (doc_id: `62931`) (sent_id: `62931`)


Deshalb können nur solche Aufwendungen als Werbungskosten i. S. des § 9 Abs. 1 EStG abgezogen werden , welche die persönliche Leistungsfähigkeit des Steuerpflichtigen mindern ( ständige Rechtsprechung , z.B. Beschluss des Großen Senats des BFH in BFHE 189 , 160 , BStBl II 1999 , 782 , unter C. IV. 1. b ; BFH-Urteil vom 15. November 2005 IX R 25/03 , BFHE 211 , 318 , BStBl II 2006 , 623 , m. w. N. ; Senatsurteil vom 13. März 1996 VI R 103/95 , BFHE 180 , 139 , BStBl II 1996 , 375 ) .

**False Positives:**

- `C.` — partial — pred is substring of gold: `Beschluss des Großen Senats des BFH in BFHE 189 , 160 , BStBl II 1999 , 782 , unter C. IV. 1. b`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 9 Abs. 1 EStG`(NRM)
- `Beschluss des Großen Senats des BFH in BFHE 189 , 160 , BStBl II 1999 , 782 , unter C. IV. 1. b`(RS)
- `BFH-Urteil vom 15. November 2005 IX R 25/03 , BFHE 211 , 318 , BStBl II 2006 , 623`(RS)
- `Senatsurteil vom 13. März 1996 VI R 103/95 , BFHE 180 , 139 , BStBl II 1996 , 375`(RS)

**Example 24** (doc_id: `62959`) (sent_id: `62959`)


Das FG hat auf S. 9 des Urteils in plausibler Weise begründet , dass in der Differenz zwischen Batterieladung und Batterieentladung keine unternehmerische Nutzung zu sehen ist , weil es sich insoweit nicht um gespeicherten Strom handele , sondern um während des Speichervorgangs entstehende Energieverluste , die für eine ( unternehmerische oder nichtunternehmerische ) Nutzung nicht zur Verfügung stehen .

**False Positives:**

- `S.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 25** (doc_id: `62980`) (sent_id: `62980`)


Hinzu komme , dass die Klägerin gegenüber der Rechtsanwaltskammer erklärt habe , eine gutgehende Anwaltskanzlei in P. zu führen .

**False Positives:**

- `P.` — type mismatch — same span as gold: `P.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `P.`(LOC)

**Example 26** (doc_id: `63011`) (sent_id: `63011`)


I. 1. Die Beschwerdeführerin , die Verwaltungs-GmbH einer nicht rechtsfähigen Stiftung , wendet sich gegen den am 6. Dezember 2013 ( BGBl I S. 1386 ) in Kraft getretenen § 6a Bundesjagdgesetz ( BJagdG ) .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BGBl I S. 1386`(LIT)
- `§ 6a Bundesjagdgesetz`(NRM)
- `BJagdG`(NRM)

**Example 27** (doc_id: `63073`) (sent_id: `63073`)


Als Anschlagsort hatte M. das E. in Q. ins Auge gefasst , da es " [ d ] er meist besuchteste Ort Europas " sei und sich dort viele " kuffar " aufhielten .

**False Positives:**

- `E.` — type mismatch — same span as gold: `E.`
- `Q.` — type mismatch — same span as gold: `Q.`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `M.`(PER)
- `E.`(LOC)
- `Q.`(LOC)
- `Europas`(LOC)

**Example 28** (doc_id: `63198`) (sent_id: `63198`)


Unbegründet erweist sich die Rechtsbeschwerde hingegen insoweit , als der Antragsteller die Gewährung dieser Vergütung ohne Anrechnung der Wegstrecke von seinem Wohnsitz - der Wohnung - zu dem bisherigen Dienstort in S. erstrebt ( 2. ) .

**False Positives:**

- `S.` — type mismatch — same span as gold: `S.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S.`(LOC)

**Example 29** (doc_id: `63253`) (sent_id: `63253`)


Sowohl die Pflichtmitgliedschaft in der berufsständischen Kammer als auch ( in der Folge ) die Mitgliedschaft in der Versorgungsanstalt stehen nach den einschlägigen landesrechtlichen Vorschriften nicht zur Disposition des Betroffenen ( vgl dazu bereits die Ausführungen unter I. 1. ) .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Isolated known surnames`

**F1:** 0.191 | **Precision:** 0.146 | **Recall:** 0.275  

**Format:** `regex`  
**Rule ID:** `325e1c8e`  
**Description:**
Captures specific known surnames appearing in isolation, ensuring they are treated as PER. Excludes common German words like 'Das', 'Der', 'Die'.

**Content:**
```
\b(?:Schäfer|Brückner|Volz|Treber|Knoll|Kriener|Nielsen|Kortbein|Schmid|Söchtig|Kortge|Jacobi|Schödel|Fiamingo|Kelvin|Schilling|Zimmermann|Dölp|Wollensak|Heinkel|Wemheuer|Sost-Scheible|Kirchhof|Paul Kirchhof|Koch|Brune|Grüneberg|Hayen|Lipphaus|Merkel|Eylert|Krumbiegel|Kayser|Jäger|Merzbach|Hacker|Meiser|Becker|Zeng|Merk|Beji Caid Essebsi|Gericke|Franke|Busch|Bender|Augat|Tiemann|Löffler|Schlünder|Schmidt|Schmalz|Melzacks|Edda Redeker|Rosen|Kortbein|Schmid|Söchtig|Kortge|Jacobi|Schödel|Linck|Schultz|Bellay|Leitz|Fieback|Rachor|Cosima|Hoch|Appl|Berger|Quentin|Roloff|Lohmann|Raum|Spinner|St|Br\.|B1|S3|S4|Karaçay|Schramm|Egerer|Kätker|Wismeth|Freudenreich|Schwitzer|Enerji Yapi-Yol Sen|Schwabe|Paffrath|Derstadt|Gallner|Herrmann|Shah|Krasshöfer|Limperg|Mosbacher|Schneider|Niemann|Zwanziger|Brenneisen|Hausmann|Kazele|Hohoff|Roggenbuck|Hamdorf|Grabinski|Krehl|Kosziol|Sunder|Mayen|Seiters|Schlewing|Spaniol|Kirchhoff|Fritz|Vogelsang|Lauer|Mutzbauer|Cierniak|Müller|Ahrendt|Dö.|Widuch|Menezes|Sander|Fischermeier|Hoffmann|Kleinschmidt|Kirschneck|Matter|Kapels|Jostes|Da.|Maksymiw|Schell|Münzberg|D7|Peter Lorsbach|Lorsbach|D1|Stresemann|Feddersen|Botur|Galke|Grupp|Naumann|Suckow|Cirener|Grube|Rahmstorf|Gehrlein|von Pentz)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.146 | 0.275 | 0.191 | 609 | 89 | 520 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 89 | 520 | 235 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60150`) (sent_id: `60150`)


Gallner

| Predicted | Gold |
|---|---|
| `Gallner` | `Gallner` |

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

**Example 3** (doc_id: `60504`) (sent_id: `60504`)


Dölp

| Predicted | Gold |
|---|---|
| `Dölp` | `Dölp` |

**Example 4** (doc_id: `60546`) (sent_id: `60546`)


Sost-Scheible

| Predicted | Gold |
|---|---|
| `Sost-Scheible` | `Sost-Scheible` |

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

**Example 7** (doc_id: `60781`) (sent_id: `60781`)


Mosbacher

| Predicted | Gold |
|---|---|
| `Mosbacher` | `Mosbacher` |

**Example 8** (doc_id: `60821`) (sent_id: `60821`)


Gallner

| Predicted | Gold |
|---|---|
| `Gallner` | `Gallner` |

**Example 9** (doc_id: `60994`) (sent_id: `60994`)


Hayen

| Predicted | Gold |
|---|---|
| `Hayen` | `Hayen` |

**Example 10** (doc_id: `61019`) (sent_id: `61019`)


Wemheuer

| Predicted | Gold |
|---|---|
| `Wemheuer` | `Wemheuer` |

**Example 11** (doc_id: `61083`) (sent_id: `61083`)


Wemheuer

| Predicted | Gold |
|---|---|
| `Wemheuer` | `Wemheuer` |

**Example 12** (doc_id: `61123`) (sent_id: `61123`)


Vogelsang

| Predicted | Gold |
|---|---|
| `Vogelsang` | `Vogelsang` |

**Example 13** (doc_id: `61174`) (sent_id: `61174`)


Hohoff

| Predicted | Gold |
|---|---|
| `Hohoff` | `Hohoff` |

**Example 14** (doc_id: `61183`) (sent_id: `61183`)


Schmidt

| Predicted | Gold |
|---|---|
| `Schmidt` | `Schmidt` |

**Example 15** (doc_id: `61238`) (sent_id: `61238`)


Becker

| Predicted | Gold |
|---|---|
| `Becker` | `Becker` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60070`) (sent_id: `60070`)


Eine Mindestentfernung zwischen Haupt- und beruflicher Zweitwohnung bestimmt das Einkommensteuergesetz nicht ( Geserich , in : Kirchhof / Söhn / Mellinghoff , EStG , § 9 Rz G 60 ) .

**False Positives:**

- `Kirchhof` — partial — pred is substring of gold: `Geserich , in : Kirchhof / Söhn / Mellinghoff , EStG , § 9 Rz G 60`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Einkommensteuergesetz`(NRM)
- `Geserich , in : Kirchhof / Söhn / Mellinghoff , EStG , § 9 Rz G 60`(LIT)

**Example 1** (doc_id: `60075`) (sent_id: `60075`)


Da weder Art. 19 Abs. 4 noch Art. 3 Abs. 1 GG zur Regelung einer vom jeweiligen Landesrecht unabhängigen einheitlichen Normenkontrollzuständigkeit eines gemeinsamen Obergerichts verpflichten , gebieten sie auch nicht , eine solche Zuständigkeit nach Maßgabe der großzügigsten in den beteiligten Ländern getroffenen Regelung vorzusehen .

**False Positives:**

- `Da ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Art. 19 Abs. 4 noch Art. 3 Abs. 1 GG`(NRM)

**Example 2** (doc_id: `60095`) (sent_id: `60095`)


Das Ziel des Sterbens sei doch lediglich , ins Paradies zu kommen , das wolle er .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 3** (doc_id: `60100`) (sent_id: `60100`)


Der in D1 zudem verwendete Keramik-Dauerfilter werde nicht näher beschrieben , so dass davon auszugehen sei , dass er lediglich eine Filterfunktion zur Verfügung stelle und insbesondere nicht mit der Erzeugung der Verwirbelungen des Wassers im Zusammenhang stehe .

**False Positives:**

- `D1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `60121`) (sent_id: `60121`)


Das Landesarbeitsgericht hat angenommen , die von der Klägerin gehaltenen Lehrveranstaltungen hätten einen wissenschaftlichen Zuschnitt , da andernfalls das Ausbildungsziel , die Kompetenz zu wissenschaftlicher Arbeit mit Literaturtexten zu vermitteln , nicht zu erreichen sei .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `60141`) (sent_id: `60141`)


Da diese Fragen sowohl in tatsächlicher als auch in rechtlicher Hinsicht der Klärung im Hauptsacheverfahren vorzubehalten sind , fällt die im Rahmen des § 123 VwGO zu treffende Abwägung wegen des Gewichts der möglicherweise im Raum stehenden öffentlichen Belange des Geheimnisschutzes sowie berechtigter schutzwürdiger Interessen Privater an der Vertraulichkeit zulasten des Antragstellers aus .

**False Positives:**

- `Da ` — no gold match — likely missing annotation
- `Raum` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `§ 123 VwGO`(NRM)

**Example 6** (doc_id: `60159`) (sent_id: `60159`)


5. Das Landgericht wies die Klinik mit Beschluss vom 29. Juli 2016 an , den Ausdruck der auf dem Klinikrechner gespeicherten Datei des Beschwerdeführers aus der Krankenakte zu entfernen .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 7** (doc_id: `60174`) (sent_id: `60174`)


Das LSG hat die Beklagte zur Zahlung von 80 Euro nebst Zinsen in Höhe von 5 % über dem jeweiligen Basiszinssatz seit dem 7. 3. 2014 verurteilt und im Übrigen die Berufung der Klägerin zurückgewiesen : Abgesehen von den zu Unrecht mit aufgerechneten 80 Euro Selbstbeteiligung stehe der Beklagten ein Anspruch auf Erstattung des übrigen Rechnungsbetrags zu .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `60181`) (sent_id: `60181`)


Das Gewaltverbot beinhaltet lediglich eine Unterlassungspflicht , vermittelt jedoch keinen Anspruch auf Unterlassung .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 9** (doc_id: `60195`) (sent_id: `60195`)


Das AG hob diese Anordnung durch Beschluss vom 7. August 2017 auf , da der Insolvenzplan bindend sei und die geltend gemachten Forderungen von diesem umfasst seien .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 10** (doc_id: `60202`) (sent_id: `60202`)


Das FG hat zu Recht entschieden , dass der Kläger die Voraussetzungen für die Zuerkennung der Gemeinnützigkeit nicht erfüllte und das FA daher die Anerkennung als gemeinnütziger Verein widerrufen durfte .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 11** (doc_id: `60247`) (sent_id: `60247`)


1. Das Landgericht hat dazu im Wesentlichen Folgendes festgestellt :

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 12** (doc_id: `60259`) (sent_id: `60259`)


Da sich der maßgebliche Sachverhalt des vorliegenden Rechtsstreits nach dem Zeitpunkt des Inkrafttretens dieses Beschlusses zugetragen hat , unterliegt er aus zeitlichen Gründen allein der VO ( EG ) Nr 883/2004 und der VO ( EG ) Nr 987/2009 ( vgl Art 90 Abs 1 Buchst c VO < EG > Nr 883/2004 und Art 96 Abs 1 Buchst c VO < EG > Nr 987/2009 jeweils iVm dem Beschluss Nr 1/2012 ) .

**False Positives:**

- `Da ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `VO ( EG ) Nr 883/2004`(NRM)
- `VO ( EG ) Nr 987/2009`(NRM)
- `Art 90 Abs 1 Buchst c VO < EG > Nr 883/2004`(NRM)
- `Art 96 Abs 1 Buchst c VO < EG > Nr 987/2009`(NRM)

**Example 13** (doc_id: `60260`) (sent_id: `60260`)


Das trifft auf die genannte Prüfung schon deshalb nicht zu , weil der Kläger an dieser ( Abschluss- ) Prüfung als Schüler der Fachschule teilgenommen hat .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 14** (doc_id: `60261`) (sent_id: `60261`)


3. Das Streitpatent betrifft einen Forstanhänger mit einer Knickdeichsel , der gemäß der Absätze [ 0004 ] und [ 0006 ] der Streitpatentschrift , im folgenden SPS genannt , beispielsweise zum Laden und Liefern von Bäumen dient und von einer Zugmaschine in Form eines Traktors gezogen wird .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 15** (doc_id: `60290`) (sent_id: `60290`)


Dieser Disilicatrohling wird dann maschinell zu entsprechenden dentalen Restaurationen weiterverarbeitet ( vgl. D7 , Patentanspruch 12 , S. 4 , [ 0036 ] ) .

**False Positives:**

- `D7` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 16** (doc_id: `60297`) (sent_id: `60297`)


Das Landgericht hat im Rahmen der Strafzumessung rechtsfehlerhaft das Gesamtstrafübel für die Angeklagte nicht in den Blick genommen , das - infolge der Zäsurwirkung des Urteils des Amtsgerichts Kulmbach vom 6. August 2015 - aus der obligatorischen Bildung von zwei Gesamtstrafen resultierte .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Amtsgerichts Kulmbach`(ORG)

**Example 17** (doc_id: `60323`) (sent_id: `60323`)


Das LSG habe ihm den barrierefreien Zugang zur mündlichen Verhandlung in seiner Sache verwehrt und ihm dadurch die Möglichkeit abgeschnitten , im Rahmen der Verhandlung sich zu Wort zu melden und ggf darin neue Beweisanträge zu stellen .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 18** (doc_id: `60324`) (sent_id: `60324`)


Soweit die Widersprechende sich darauf beruft , dass auch kennzeichnungsschwache Marken zumindest Schutz gegen eine identische Übernahme beanspruchen könnten , führt dieser grundsätzlich zutreffende Einwand gleichfalls nicht zur Bejahung der Verwechslungsgefahr , da sich die hier zu vergleichenden Zeichen – wie nachfolgend unter Ziffer 1. 3. dargelegt – erheblich unterscheiden ( vgl. im Übrigen zum Schutzumfang zu Unrecht eingetragener , materiell schutzunfähiger Marken Ströbele / Hacker , MarkenG , 11. Aufl. , § 9 Rn. 194 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele / Hacker , MarkenG , 11. Aufl. , § 9 Rn. 194`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 9 Rn. 194`(LIT)

**Example 19** (doc_id: `60329`) (sent_id: `60329`)


Die strafschärfende Berücksichtigung der hierin liegenden Schuldsteigerung gerate weder mit dem in § 46 Abs. 3 StGB verankerten Doppelverwertungsverbot von Tatbestandsmerkmalen ( SSW-StGB / Eschelbach , aaO , § 46 Rn. 93 , 185 ; von Heintschel-Heinegg , Streng-FS 2017 S. 229 , 239 ) noch mit dem Gedanken in Konflikt , dass es sich um das Regeltatbild des Totschlags handele ( Fahl , JR 2017 , 391 , 393 ; MüKo / Schneider , aaO , § 212 Rn. 82 ; Tomiak , HRRS 2017 , 225 ff. ) .

**False Positives:**

- `Schneider` — partial — pred is substring of gold: `MüKo / Schneider , aaO , § 212 Rn. 82`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 46 Abs. 3 StGB`(NRM)
- `SSW-StGB / Eschelbach , aaO , § 46 Rn. 93 , 185`(LIT)
- `von Heintschel-Heinegg , Streng-FS 2017 S. 229 , 239`(LIT)
- `Fahl , JR 2017 , 391 , 393`(LIT)
- `MüKo / Schneider , aaO , § 212 Rn. 82`(LIT)
- `Tomiak , HRRS 2017 , 225 ff.`(LIT)

**Example 20** (doc_id: `60334`) (sent_id: `60334`)


1.1 Das Patent betrifft die Verwendung eines Rohlings aus einem Lithiumsilicatmaterial , das durch maschinelle Verarbeitung einfach geformt und anschließend zu dentalen Restaurationen von hoher Festigkeit umgewandelt werden kann ( vgl. Streitpatentschrift , Patentansprüche 1 , 13 , 17 und 19 , S. 2 , [ 0001 ] ) .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 21** (doc_id: `60346`) (sent_id: `60346`)


Das Urteil beruht auf diesem Verfahrensfehler ( dazu b ) .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 22** (doc_id: `60348`) (sent_id: `60348`)


Das FA lehnte die Änderung ab .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 23** (doc_id: `60351`) (sent_id: `60351`)


Das Gesetz verschaffe aber keinen nachträglichen Zugang zu einem Zusatzversorgungssystem , das den Beschäftigten unabhängig von einer politischen Verfolgung aufgrund der restriktiven Einbeziehungspraxis der DDR vorenthalten worden sei .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `DDR`(LOC)

**Example 24** (doc_id: `60353`) (sent_id: `60353`)


Das Verwaltungsgericht habe die Abweisung der Klage als offensichtlich unbegründet jedoch allein damit begründet , dass es den Vortrag zum individuellen Verfolgungsschicksal als krass widersprüchlich und damit unglaubhaft eingestuft , und das Offensichtlichkeitsurteil auf § 30 Abs. 1 AsylG und § 30 Abs. 3 Nr. 1 AsylG gestützt habe .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 30 Abs. 1 AsylG`(NRM)
- `§ 30 Abs. 3 Nr. 1 AsylG`(NRM)

**Example 25** (doc_id: `60395`) (sent_id: `60395`)


Das Beitragsaufkommen ist nach § 3 Abs. 2 Satz 2 und 3 RFinStV gedeckelt .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 3 Abs. 2 Satz 2 und 3 RFinStV`(REG)

**Example 26** (doc_id: `60425`) (sent_id: `60425`)


IV. Das Land Baden-Württemberg hat dem Beschwerdeführer gemäß § 34a Abs. 2 BVerfGG die notwendigen Auslagen zu erstatten .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Baden-Württemberg`(LOC)
- `§ 34a Abs. 2 BVerfGG`(NRM)

**Example 27** (doc_id: `60439`) (sent_id: `60439`)


Das " gelebte " Vertragsverhältnis entspricht dem formell vereinbarten Vertrag über ein selbstständiges Dienstverhältnis .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 28** (doc_id: `60471`) (sent_id: `60471`)


Das Truppendienstgericht ist nicht befugt , im Rahmen der Entscheidung , ob einer Nichtzulassungsbeschwerde abgeholfen wird , den angefochtenen Beschluss nachzubessern und gerügte Verfahrensmängel zu beheben .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 29** (doc_id: `60478`) (sent_id: `60478`)


Das FA beantragt , das Urteil des FG aufzuheben und die Klage abzuweisen .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Full names with initials (e.g., K. Schmidt)`

**F1:** 0.054 | **Precision:** 0.098 | **Recall:** 0.037  

**Format:** `regex`  
**Rule ID:** `266c1518`  
**Description:**
Captures full names consisting of an initial and a surname (e.g., 'K. Schmidt', 'M. Rennpferdt').

**Content:**
```
\b([A-Z]\.)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)\b
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

- `I. Die Klägerin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `60117`) (sent_id: `60117`)


I. Die Befristungskontrollklage ist unbegründet .

**False Positives:**

- `I. Die Befristungskontrollklage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `60238`) (sent_id: `60238`)


V. Die Klage ist nicht abweisungsreif ( vgl. § 563 Abs. 3 ZPO ) .

**False Positives:**

- `V. Die Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 563 Abs. 3 ZPO`(NRM)

**Example 3** (doc_id: `60477`) (sent_id: `60477`)


I. Die Würdigung des Landesarbeitsgerichts , das beklagte Königreich sei im vorliegenden Rechtsstreit grundsätzlich nicht der deutschen Gerichtsbarkeit unterworfen , sondern genieße - sollte es darauf nicht verzichtet haben - Staatenimmunität , ist revisionsrechtlich nicht zu beanstanden .

**False Positives:**

- `I. Die Würdigung` — no gold match — likely missing annotation

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

- `I. Die Antragsgegnerin` — no gold match — likely missing annotation

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

- `I. Die Kläger` — no gold match — likely missing annotation

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

- `B. Not` — no gold match — likely missing annotation

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

- `I. Der Feststellungsantrag` — no gold match — likely missing annotation

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

- `I. Die Anmelderin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschen Patent- und Markenamt`(ORG)
- `A-ÖFFNER`(ORG)

**Example 18** (doc_id: `61516`) (sent_id: `61516`)


I. Der Kläger und Revisionskläger ( Kläger ) war in den Streitjahren ( 1995 bis 1997 ) u. a. als Steuerberater in einer Einzelkanzlei tätig .

**False Positives:**

- `I. Der Kläger` — no gold match — likely missing annotation

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

- `I. Die Bezeichnung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `MAM Munich Asset Management`(ORG)
- `Deutschen Patent- und Markenamt`(ORG)
- `DPMA`(ORG)

**Example 25** (doc_id: `61932`) (sent_id: `61932`)


V. Die Kostenentscheidung beruht auf § 90 Satz 2 EnWG , die Festsetzung des Gegenstandswerts auf § 50 Abs. 1 Satz 1 Nr. 2 GKG und § 3 ZPO .

**False Positives:**

- `V. Die Kostenentscheidung` — no gold match — likely missing annotation

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

- `A. Die Richtervorlage` — no gold match — likely missing annotation

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

- `I. Der Kläger` — no gold match — likely missing annotation

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

## `Hyphenated surnames`

**F1:** 0.005 | **Precision:** 0.015 | **Recall:** 0.003  

**Format:** `regex`  
**Rule ID:** `97b86ba2`  
**Description:**
Captures hyphenated surnames like 'Schmidt-Räntsch' only when preceded by a title or in a list of names to avoid matching court names.

**Content:**
```
(?:Dr\.?\s+|Prof\.?\s+|Richter\s+|Vorsitzender\s+|und\s+|sowie\s+|der\s+|die\s+|des\s+)([A-Z][a-zäöüß]+(?:-[A-Z][a-zäöüß]+)+)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.015 | 0.003 | 0.005 | 65 | 1 | 64 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 1 | 64 | 323 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `62802`) (sent_id: `62802`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2014 004 201.0 hat der 29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 29. November 2017 durch die Vorsitzende Richterin Dr. Mittenberger-Huber , die Richterin Akintche und die Richterin Seyfarth beschlossen :

| Predicted | Gold |
|---|---|
| `Mittenberger-Huber` | `Mittenberger-Huber` |

**Missed by this rule (FN):**

- `29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Akintche` (PER)
- `Seyfarth` (PER)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60065`) (sent_id: `60065`)


- durch deren Betätigung ein durch die Antriebseinrichtung bewirktes Öffnen des Schiebeflügels zur Freigabe eines Flucht- und Rettungswegs auslösbar ist ( Seite 3 , Abschnitt 1.1.4 , 4. Spiegelstrich , 7. Punkt : „ wenn zusätzlich Fluchttüranforderung besteht “ , Seite 4 , Abschnitt 2.1 , 5. Spiegelstrich : „ Automatische Schiebetür … zum Einsatz in Rettungswegen “ und Seite 4 , Abschnitt 2.1 , letzter Absatz : „ … darf die Rauchschutz-Schiebetür nur durch Betätigung der NOT-AUF-Taster … für den Durchgang von Personen geöffnet werden . “ ; Merkmal 1.8) ;

**False Positives:**

- `Rauchschutz-Schiebetür` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `60101`) (sent_id: `60101`)


3. Als zuständigen Fachmann sieht der Senat – in Übereinstimmung mit der Patentabteilung im Einspruchsbeschluss – einen Diplomingenieur der Elektrotechnik mit mehrjähriger Berufserfahrung auf dem Gebiet der Hardware- und Software-Entwicklung und des Betreibens von Sicherheitsschaltern .

**False Positives:**

- `Software-Entwicklung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `60155`) (sent_id: `60155`)


Die Beschwerde hält in diesem Zusammenhang für klärungsbedürftig , ob die in § 68 Abs. 1 Satz 1 , 3 und 4 , § 68a Satz 1 AufenthG bundesgesetzlich geregelte Geltungsdauer für Verpflichtungserklärungen durch landesinterne Vorgaben ( hier : Aufnahmeanordnung des Landes Rheinland-Pfalz vom 30. August 2013 i. V. m. den zugehörigen Anwendungshinweisen ) eingeschränkt werden kann , soweit davon Leistungen in der Verantwortung des Bundes ( hier : Leistungen der Grundsicherung für Arbeitsuchende nach dem Zweiten Buch Sozialgesetzbuch - SGB II - in der Trägerschaft der Bundesagentur für Arbeit nach § 6 Abs. 1 Satz 1 Nr. 1 SGB II ) betroffen wären .

**False Positives:**

- `Rheinland-Pfalz` — type mismatch — same span as gold: `Rheinland-Pfalz`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 68 Abs. 1 Satz 1 , 3 und 4 , § 68a Satz 1 AufenthG`(NRM)
- `Rheinland-Pfalz`(LOC)
- `Zweiten Buch Sozialgesetzbuch`(NRM)
- `SGB II`(NRM)
- `Bundesagentur für Arbeit`(ORG)
- `§ 6 Abs. 1 Satz 1 Nr. 1 SGB II`(NRM)

**Example 3** (doc_id: `60167`) (sent_id: `60167`)


Den Bescheid vom 3. 3. 2009 korrigierte die Beklagte zugunsten der Klägerin mit weiterem Bescheid vom 7. 7. 2009 und setzte die Rückforderung wegen Überschreitung der Job-Sharing-Grenzen für die drei genannten Quartale auf insgesamt 9125,83 Euro fest .

**False Positives:**

- `Job-Sharing-Grenzen` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `60196`) (sent_id: `60196`)


Dies bedeutet , dass sich auch die Trennstrecke außerhalb des Lichtbogen-Brennraums befindet .

**False Positives:**

- `Lichtbogen-Brennraums` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `60263`) (sent_id: `60263`)


Mit diesem Bescheid sei die Punktzahlobergrenze im Rahmen des Job-Sharings bindend festgesetzt worden .

**False Positives:**

- `Job-Sharings` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 6** (doc_id: `60367`) (sent_id: `60367`)


Um 21:37 Uhr befuhr er die Hans-Böckler-Straße und anschließend deren Verlängerung , die Nordstraße , in stadtauswärtiger Richtung .

**False Positives:**

- `Hans-Böckler-Straße` — type mismatch — same span as gold: `Hans-Böckler-Straße`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Hans-Böckler-Straße`(LOC)
- `Nordstraße`(LOC)

**Example 7** (doc_id: `60469`) (sent_id: `60469`)


Der Senat ist insoweit an die unter Anwendung des Baden-Württembergischen Landesrechts getroffene Entscheidung des LSG gebunden ( § 202 SGG iVm § 560 ZPO ) .

**False Positives:**

- `Baden-Württembergischen` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 202 SGG`(NRM)
- `§ 560 ZPO`(NRM)

**Example 8** (doc_id: `60475`) (sent_id: `60475`)


Vorführung von Waren für Werbezwecke , insbesondere Präsentation von Waren im Teleshoppingbereich ; das Zusammenstellen verschiedener Waren [ ausgenommen deren Transport ] für Dritte , um über Websites oder Teleshopping-Sendungen den Verbrauchern Ansicht und Erwerb dieser Waren zu erleichtern .

**False Positives:**

- `Teleshopping-Sendungen` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 9** (doc_id: `60602`) (sent_id: `60602`)


Sie verwiesen auf das Urteil des Bundesfinanzhofs ( BFH ) vom 12. Mai 2015 VIII R 4/15 ( BFHE 250 , 75 , BStBl II 2015 , 835 ) , wonach die Erlöse aus der Auslieferung des Xetra-Goldes nicht im Rahmen der Einkünfte aus Kapitalvermögen steuerbar seien .

**False Positives:**

- `Xetra-Goldes` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Urteil des Bundesfinanzhofs ( BFH ) vom 12. Mai 2015 VIII R 4/15 ( BFHE 250 , 75 , BStBl II 2015 , 835 )`(RS)

**Example 10** (doc_id: `60790`) (sent_id: `60790`)


Diese kann zwar , wie § 73 Abs. 1 Satz 2 ArbGG deutlich macht , nicht auf die Versäumung der Fünf-Monats-Frist gestützt werden .

**False Positives:**

- `Fünf-Monats-Frist` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 73 Abs. 1 Satz 2 ArbGG`(NRM)

**Example 11** (doc_id: `60800`) (sent_id: `60800`)


Hier reicht die Zündhilfselektrode bei einer Definition des Lichtbogen-Brennraums durch die von den Distanzhaltern ( 21 ) und der Isolierung gebildete Linie nicht in den Lichtbogen-Brennraum hinein .

**False Positives:**

- `Lichtbogen-Brennraums` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 12** (doc_id: `60875`) (sent_id: `60875`)


Eine Novelle der 35. Verordnung zur Durchführung des Bundes-Immissionsschutzgesetzes sei in diesem Zusammenhang wünschenswert , aber nicht notwendig .

**False Positives:**

- `Bundes-Immissionsschutzgesetzes` — partial — pred is substring of gold: `35. Verordnung zur Durchführung des Bundes-Immissionsschutzgesetzes`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `35. Verordnung zur Durchführung des Bundes-Immissionsschutzgesetzes`(NRM)

**Example 13** (doc_id: `60935`) (sent_id: `60935`)


Klasse 42 : Aktualisieren von Computer-Software ; Aktualisierung und Design von Computer-Software ; Aktualisierung und Wartung von Computer-Software ; Aktualisierung von Software-Datenbanken ; Aktualisierung von Speicherbanken [ Software ] von Computersystemen ; Beratung auf dem Gebiet von Computerhardware und -software ; Beratung in Bezug auf Computer und Software ; Beratung in Bezug auf Computernetze mit unterschiedlichen Softwareumgebungen ; Beratungsdienste auf dem Gebiet von Computerhardware und -software ; Computerhardware- und -softwareberatungsdienstleistungen ; Computerprogrammierung und Softwareentwicklung ; Consulting und Beratung auf dem Gebiet der Computerhardware und -software ; Design von Computer-Software ; Designdienstleistungen für Computer-Software ; Dienstleistungen für den Entwurf von Software für die elektronische Datenverarbeitung ; Dienstleistungen für die Gestaltung von Computer-Software ; Entwickeln von Software ; Entwicklung von Computer-Software ; Entwicklung von Software ; Entwicklung von Software für Computer ; Entwicklung von Software für Rechner ; Entwicklung von Softwarelösungen für Internet-Provider und Internet-Nutzer ; Entwicklung , Programmierung und Implementierung von Software ; Entwurf und Entwicklung von Computerhardware und -software ; Entwurf , Entwicklung und Implementierung von Software ; Erstellung von Datenverarbeitungsprogrammen [ Software ] ; Erstellung , Wartung , Pflege und Anpassung von Software ; Hosting-Dienste , Software as a Service ( SaaS ) und Vermietung von Software ; Installation von Software ; Installation , Wartung und Reparatur von Software für Computer ; Installation , Wartung und Reparatur von Software für Computersysteme ; Kundenspezifische Gestaltung von Softwarepaketen ; Kundenspezifische Softwareanpassung ; Kundenspezifisches Design von Softwarepaketen ; Reparatur [ Wartung und Aktualisierung ] von Software ; Software as a Service [ SaaS ] ; Softwaredesign ; Softwaredesign und -entwicklung ; Softwareengineering ; Softwareentwicklung ; Softwareentwicklungsdienste ; Softwareentwicklungsleistungen ; Softwareerstellung ; Softwareerstellungsleistungen ; Softwarevermietung für Computer ; Technischer Support im Softwarebereich ; Vermietung von Computer-Software ; Vermietung von Software für Computer ; Vermietung von Software für Rechner ; Wartung und Aktualisierung von Software ; Wartung und Reparatur von Software ;

**False Positives:**

- `Internet-Nutzer` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 14** (doc_id: `60945`) (sent_id: `60945`)


Ein RLV in diesem Sinne war die von einem Arzt oder der Arztpraxis in einem bestimmten Zeitraum abrechenbare Menge vertragsärztlicher Leistungen , die mit den in der Euro-Gebührenordnung enthaltenen und für den Arzt oder die Arztpraxis geltenden Preisen zu vergüten war ( § 87b Abs 2 S 2 SGB V aF ) .

**False Positives:**

- `Euro-Gebührenordnung` — type mismatch — same span as gold: `Euro-Gebührenordnung`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Euro-Gebührenordnung`(NRM)
- `§ 87b Abs 2 S 2 SGB V aF`(NRM)

**Example 15** (doc_id: `61037`) (sent_id: `61037`)


Das Verbot betraf danach sämtliche Fußballstadien in Deutschland hinsichtlich nationaler und internationaler Fußballveranstaltungen von Vereinen beziehungsweise Tochtergesellschaften der Fußball-Bundesligen und der Fußballregionalligen sowie des Deutschen Fußball-Bundes .

**False Positives:**

- `Fußball-Bundesligen` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschland`(LOC)
- `Deutschen Fußball-Bundes`(ORG)

**Example 16** (doc_id: `61307`) (sent_id: `61307`)


Die Berechnung der Job-Sharing-Obergrenze sei zutreffend unter Heranziehung der Gruppe der Internisten mit dem Schwerpunkt Kardiologie erfolgt .

**False Positives:**

- `Job-Sharing-Obergrenze` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 17** (doc_id: `61367`) (sent_id: `61367`)


Zudem liege in einem Anstieg der Quadratmeter-Miete von 4,95 Euro im September 2008 auf 5,18 Euro im vierten Quartal 2011 kein unvorhergesehener Preissprung , sondern eine normale Preisentwicklung .

**False Positives:**

- `Quadratmeter-Miete` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 18** (doc_id: `61526`) (sent_id: `61526`)


Diese Angaben sind nur insoweit gegenständlich merkmalsbildend , als dass die beanspruchte Biegeschiene raumkörperlich so ausgebildet sein muss , dass beim bestimmungsgemäßen Anlegen an einen Patientenfuß die Schwenkachse der Gelenkeinrichtung in etwa der Gelenkachse des Großzehengrundgelenks in der Flexion-Extensionsrichtung entspricht ( vgl. Figur 1 u. 2 ) .

**False Positives:**

- `Flexion-Extensionsrichtung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `61586`) (sent_id: `61586`)


Die Klägerin hatte ihren Sitz zum Zeitpunkt des Eintritts der Job-Sharing-Partnerin Dr. E. im Bezirk der KÄV Nordbaden .

**False Positives:**

- `Job-Sharing-Partnerin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `E.`(PER)
- `KÄV Nordbaden`(ORG)

**Example 20** (doc_id: `61608`) (sent_id: `61608`)


Auch weitere – sowohl aktuelle , als auch vor dem Anmeldezeitpunkt datierende – Verwendungsbeispiele beziehen sich auf den Bereich der Raum- und Farbgestaltung : darin ist von „ Wohlfühlfarben : Natürliches Flair “ ( www.livingathome.de) , von „ Sanfte ( n ) Tönen im Wohnbereich – Wohlfühlfarben “ ( www.zuhausewohnen.de) , „ Trendtöne ( n ) : Wohlfühlfarben ... “ ( www.wunderweib.de) die Rede ( vgl. hierzu die Google-Recherche zum Stichwort „ Wohlfühlfarben “ nebst Anlagen ; siehe im Übrigen auch schon BPatG , a. a. O. , 30 W ( pat ) 530/13 - WohlFühlFarben , mit weiteren Verwendungsbeispielen ) .

**False Positives:**

- `Google-Recherche` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BPatG , a. a. O. , 30 W ( pat ) 530/13 - WohlFühlFarben`(RS)

**Example 21** (doc_id: `61788`) (sent_id: `61788`)


Nicht anwendbar ist entgegen der Auffassung des Landesarbeitsgerichts hingegen der Tarifvertrag zur Übernahme des Tarifrechts des Landes Berlin für die Beschäftigten der Berlin-Brandenburgischen Akademie der Wissenschaften ( ÜTV BBAW ) vom 30. Mai 2011 .

**False Positives:**

- `Berlin-Brandenburgischen` — partial — pred is substring of gold: `Tarifvertrag zur Übernahme des Tarifrechts des Landes Berlin für die Beschäftigten der Berlin-Brandenburgischen Akademie der Wissenschaften`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Tarifvertrag zur Übernahme des Tarifrechts des Landes Berlin für die Beschäftigten der Berlin-Brandenburgischen Akademie der Wissenschaften`(REG)
- `ÜTV BBAW`(REG)

**Example 22** (doc_id: `61948`) (sent_id: `61948`)


Durch die Durchschnittsbildung beim " fachgleichen Pärchen " werde einer etwaigen pflichtwidrigen Fehlzuordnung von Leistungen zum Zwecke der Umgehung der Leistungsobergrenze im Rahmen des Job-Sharings vorgebeugt .

**False Positives:**

- `Job-Sharings` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 23** (doc_id: `61974`) (sent_id: `61974`)


Im Streitfall war der Kläger indessen schon kein beherrschender Gesellschafter-Geschäftsführer der GmbH .

**False Positives:**

- `Gesellschafter-Geschäftsführer` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 24** (doc_id: `62353`) (sent_id: `62353`)


Als Beendigung der Rechtsfähigkeit des Betriebs ist der 3. 7. 1990 , als Rechtsnachfolger sind die Electronicon-GmbH G. und die B. Kondensatoren-GmbH eingetragen .

**False Positives:**

- `Electronicon-Gmb` — partial — pred is substring of gold: `Electronicon-GmbH G.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Electronicon-GmbH G.`(ORG)
- `B. Kondensatoren-GmbH`(ORG)

**Example 25** (doc_id: `62671`) (sent_id: `62671`)


Es kann dahinstehen , ob es fachüblich ist , vor der ( Vierleiter- ) Testung an jedem Anschlusskontakt des Bauelements in einer Schleife über die beiden Kontaktfedern der Kelvin-Kontaktierung den Übergangswiderstand der Kontaktierung zu den Kontaktfedern eines Kontaktfederpaares zu bestimmen und später zur Korrektur der Messergebnisse zu verwenden , vgl. Patentschrift , Absatz 0020 , denn der Senat kann auch im Zusammenhang mit einer derartigen Messung des Übergangswiderstands keine Veranlassung des Fachmanns erkennen , die Kontaktfedern C entsprechend der Anweisungen in den Merkmalen M3 bis M3.2 lamelliert für eine hohe Stromtragfähigkeit auszubilden .

**False Positives:**

- `Kelvin-Kontaktierung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 26** (doc_id: `62721`) (sent_id: `62721`)


Seit 16. 2. 2013 ist der Kläger Pflichtmitglied der Landestierärztekammer Baden-Württemberg ( im Folgenden : Landestierärztekammer ) und Pflichtmitglied der Baden-Württembergischen Versorgungsanstalt für Ärzte , Zahnärzte und Tierärzte ( Beigeladene zu 1. ) , die ihren Teilnehmern und deren Hinterbliebenen Altersruhegeld , Ruhegeld bei Berufsunfähigkeit sowie eine Hinterbliebenenversorgung gewährt .

**False Positives:**

- `Baden-Württembergischen` — partial — pred is substring of gold: `Baden-Württembergischen Versorgungsanstalt für Ärzte , Zahnärzte und Tierärzte`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Landestierärztekammer Baden-Württemberg`(ORG)
- `Baden-Württembergischen Versorgungsanstalt für Ärzte , Zahnärzte und Tierärzte`(ORG)

**Example 27** (doc_id: `62742`) (sent_id: `62742`)


1. Zunächst ist festzustellen , dass die Voraussetzung für die Durchführung des Löschungsverfahrens mit inhaltlicher Prüfung nach § 54 Abs. 2 Satz 3 MarkenG erfüllt ist , nachdem die Markeninhaberin dem ihr am 7. Mai 2013 zugestellten Löschungsantrag mit am 4. Juli 2013 beim DPMA eingegangenem Schriftsatz fristgerecht innerhalb der Zwei-Monats-Frist des § 54 Abs. 2 Satz 2 MarkenG widersprochen hat .

**False Positives:**

- `Zwei-Monats-Frist` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 54 Abs. 2 Satz 3 MarkenG`(NRM)
- `DPMA`(ORG)
- `§ 54 Abs. 2 Satz 2 MarkenG`(NRM)

**Example 28** (doc_id: `62814`) (sent_id: `62814`)


Auf die Beschwerde des Klägers gegen die Nichtzulassung der Revision wird das Urteil des Schleswig-Holsteinischen Landessozialgerichts vom 12. Dezember 2016 aufgehoben .

**False Positives:**

- `Schleswig-Holsteinischen` — partial — pred is substring of gold: `Schleswig-Holsteinischen Landessozialgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Schleswig-Holsteinischen Landessozialgerichts`(ORG)

**Example 29** (doc_id: `62923`) (sent_id: `62923`)


Im Rahmen der Umsatzbesteuerung unterlag er der Ist-Besteuerung .

**False Positives:**

- `Ist-Besteuerung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Names after 'Angeklagte' or 'Klägerin'`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `4993cd07`  
**Description:**
Captures names following legal role indicators like 'Angeklagte' or 'Klägerin'.

**Content:**
```
\b(?:Angeklagte|Angeklagten|Klägerin|Kläger|Zeugin|Zeuge|Geschädigte|Gutachter|Gutachterin)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 12 | 0 | 12 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 12 | 324 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60099`) (sent_id: `60099`)


Jedenfalls beruht das Ergebnis des Gutachtens auf einer umfassenden Auswertung des dem Gutachter zur Verfügung gestellten Aktenmaterials , aus dem der Gutachter Schlüsse zieht , die auch unabhängig von den dem Senat im Einzelnen nicht bekannten Prognosemanualen nachvollziehbar erscheinen und im Einklang mit der ordnungsrechtlichen Gefahrenbewertung stehen , wie sie auch nach dem Akteninhalt im Übrigen veranlasst ist .

**False Positives:**

- `Schlüsse` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `60750`) (sent_id: `60750`)


Der Bescheid der Beklagten ist rechtmäßig , soweit der Klägerin Insg von nicht mehr als 3927,71 Euro bewilligt worden ist .

**False Positives:**

- `Insg` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `60944`) (sent_id: `60944`)


Nachdem er vom FA auf den fehlenden Nachweis einer regelmäßigen Summenziehung hingewiesen worden sei , habe der Kläger Erfassungsprotokolle beim FG eingereicht , die eine chronologische Auflistung der Geschäftsvorfälle ohne Angabe von Belegnummern enthalten hätten .

**False Positives:**

- `Erfassungsprotokolle` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 3** (doc_id: `61328`) (sent_id: `61328`)


Am 18. November 2015 hat der Kläger Klage erhoben und zur Begründung geltend gemacht , der Jahresmittelgrenzwert für Stickstoffdioxid ( NO2 ) sei im Jahr 2013 an allen verkehrsnahen Messstationen zum Teil um mehr als das Doppelte überschritten worden und habe auch im Jahr 2014 an bestimmten Messstationen deutlich über den Grenzwerten gelegen .

**False Positives:**

- `Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `62164`) (sent_id: `62164`)


Gegen dieses Urteil hat die Klägerin Revision eingelegt .

**False Positives:**

- `Revision` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `62721`) (sent_id: `62721`)


Seit 16. 2. 2013 ist der Kläger Pflichtmitglied der Landestierärztekammer Baden-Württemberg ( im Folgenden : Landestierärztekammer ) und Pflichtmitglied der Baden-Württembergischen Versorgungsanstalt für Ärzte , Zahnärzte und Tierärzte ( Beigeladene zu 1. ) , die ihren Teilnehmern und deren Hinterbliebenen Altersruhegeld , Ruhegeld bei Berufsunfähigkeit sowie eine Hinterbliebenenversorgung gewährt .

**False Positives:**

- `Pflichtmitglied` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Landestierärztekammer Baden-Württemberg`(ORG)
- `Baden-Württembergischen Versorgungsanstalt für Ärzte , Zahnärzte und Tierärzte`(ORG)

**Example 6** (doc_id: `63009`) (sent_id: `63009`)


Hiergegen hat der Kläger Klage zum SG erhoben , das durch Urteil vom 2. 10. 2012 den Bescheid der Beklagten vom 18. 4. 2011 in der Gestalt des Widerspruchsbescheids vom 8. 6. 2011 aufgehoben hat , weil das Grundstück des Klägers aufgrund der anzuwendenden Ausnahmevorschrift des § 123 Abs 2 SGB VII als versicherungsfreier Haus- und Ziergarten einzuordnen sei .

**False Positives:**

- `Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 123 Abs 2 SGB VII`(NRM)

**Example 7** (doc_id: `63032`) (sent_id: `63032`)


III. Sollte die neue Verhandlung ergeben , dass die Klägerin Tätigkeiten mit nicht unwesentlichem Einfluss auf die Programmgestaltung schuldete , wird das Landesarbeitsgericht im Rahmen der Prüfung , ob die Befristung zum 31. Mai 2014 mit der Rundfunkfreiheit gerechtfertigt werden kann , eine erneute einzelfallbezogene Abwägung der Belange des Beklagten und der Klägerin vorzunehmen haben .

**False Positives:**

- `Tätigkeiten` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `63492`) (sent_id: `63492`)


Mit der Revision rügen die Kläger Verletzung formellen und materiellen Rechts .

**False Positives:**

- `Verletzung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 9** (doc_id: `64047`) (sent_id: `64047`)


In einem weiteren Fall öffnete der Angeklagte Knopf und Reißverschluss seiner Hose und forderte die Zeugin sinngemäß auf , an seinem Glied zu reiben .

**False Positives:**

- `Knopf` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 10** (doc_id: `66649`) (sent_id: `66649`)


M. hielt der Zeugin Mund und Nase zu , so dass sie nicht mehr schreien konnte .

**False Positives:**

- `Mund` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `M.`(PER)

**Example 11** (doc_id: `66655`) (sent_id: `66655`)


Am 18. November 2015 hat der Kläger Klage erhoben und zur Begründung geltend gemacht , die anhaltende Überschreitung der Grenzwerte sei ein Indiz dafür , dass die bisherigen Maßnahmen nicht geeignet seien , die Überschreitungszeiträume so kurz wie möglich zu halten .

**False Positives:**

- `Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Known surnames after legal roles`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `2980bda7`  
**Description:**
Captures specific known surnames (e.g., 'Knoll', 'Kriener', 'Schmid') when they follow legal role indicators, ensuring they are treated as PER.

**Content:**
```
\b(?:Angeklagte|Angeklagten|Kläger|Klägerin|Zeuge|Zeugin|Geschädigte|Geschädigten|Rechtsanwalt|Rechtsanwältin|Vorsitzender|Richter|Richterin|Herr|Herrn)\s+(?:Schäfer|Brückner|Volz|Treber|Knoll|Kriener|Nielsen|Kortbein|Schmid|Söchtig|Kortge|Jacobi|Schödel|Fiamingo|Kelvin|Schilling|Zimmermann|Dölp|Wollensak|Heinkel|Wemheuer|Sost-Scheible|Kirchhof|Paul Kirchhof|Koch|Brune|Grüneberg|Hayen|Lipphaus|Merkel|Eylert|Krumbiegel|Kayser|Jäger|Merzbach|Hacker|Meiser|Becker|Zeng|Merk|Beji Caid Essebsi|Gericke|Franke|Busch|Bender|Augat|Tiemann|Löffler|Schlünder|Schmidt|Schmalz|Melzacks|Edda Redeker|Rosen|Kortbein|Schmid|Söchtig|Kortge|Jacobi|Schödel|Linck|Schultz|Bellay|Leitz|Fieback|Rachor|Cosima|Hoch|Appl|Berger|Quentin|Roloff|Lohmann|Raum|Spinner|St|Br\.|B1|S3|S4|Karaçay|Schramm|Egerer|Kätker|Wismeth|Freudenreich|Schwitzer|Enerji Yapi-Yol Sen|Schwabe|Paffrath|Derstadt|Gallner|Herrmann|Shah|Krasshöfer|Limperg|Mosbacher|Schneider|Niemann|Zwanziger|Brenneisen|Hausmann|Kazele|Hohoff|Roggenbuck|Hamdorf|Grabinski|Krehl|Kosziol|Sunder|Mayen|Seiters|Schlewing|Spaniol|Kirchhoff|Fritz|Vogelsang|Lauer|Mutzbauer|Cierniak|Müller|Ahrendt|Dö.|Widuch|Menezes|Sander|Fischermeier|Hoffmann|Kleinschmidt|Kirschneck|Matter|Kapels|Jostes|Da.|Maksymiw|Schell|Münzberg|D7|Peter Lorsbach|Lorsbach|D1)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 10 | 0 | 10 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 10 | 246 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `61787`) (sent_id: `61787`)


In der Beschwerdesache betreffend die Marke 30 2009 056 266 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 14. September 2017 unter Mitwirkung des Vorsitzenden Richters Knoll sowie der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

**False Positives:**

- `` — similar text (different position): `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Knoll`(PER)
- `Kriener`(PER)
- `Nielsen`(PER)

**Example 1** (doc_id: `61969`) (sent_id: `61969`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2013 036 234.9 hat der 26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 16. Oktober 2017 unter Mitwirkung der Vorsitzenden Richterin Kortge sowie der Richter Jacobi und Schödel beschlossen :

**False Positives:**

- `` — similar text (different position): `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`
- `` — similar text (different position): `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Kortge`(PER)
- `Jacobi`(PER)
- `Schödel`(PER)

**Example 2** (doc_id: `62983`) (sent_id: `62983`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2011 068 984.9 hat der 26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 8. Januar 2018 unter Mitwirkung der Vorsitzenden Richterin Kortge sowie der Richter Jacobi und Schödel beschlossen :

**False Positives:**

- `` — similar text (different position): `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`
- `` — similar text (different position): `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Kortge`(PER)
- `Jacobi`(PER)
- `Schödel`(PER)

**Example 3** (doc_id: `64394`) (sent_id: `64394`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2013 053 470.0 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 9. November 2017 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

**False Positives:**

- `` — similar text (different position): `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Knoll`(PER)
- `Kriener`(PER)
- `Nielsen`(PER)

**Example 4** (doc_id: `64527`) (sent_id: `64527`)


In der Beschwerdesache betreffend die Marke 30 2011 035 856 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `` — similar text (different position): `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 5** (doc_id: `64925`) (sent_id: `64925`)


In der Beschwerdesache betreffend die Marke 30 2010 003 649 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `` — similar text (different position): `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 6** (doc_id: `64967`) (sent_id: `64967`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2015 034 000.6 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 11. Januar 2018 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

**False Positives:**

- `` — similar text (different position): `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Knoll`(PER)
- `Kriener`(PER)
- `Nielsen`(PER)

**Example 7** (doc_id: `65500`) (sent_id: `65500`)


In der Beschwerdesache betreffend die Designanmeldung .... ( hier : Antrag auf Verfahrenskostenhilfe ) hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `` — similar text (different position): `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

</details>

---

## `Full names with titles`

**F1:** 0.006 | **Precision:** 0.111 | **Recall:** 0.003  

**Format:** `regex`  
**Rule ID:** `84421f67`  
**Description:**
Captures full names preceded by titles like Dr., Prof., or Dipl.-Ing., ensuring multi-part names with middle initials (e.g., 'Jay B. Saoud') are captured as a single entity.

**Content:**
```
(?:Dr\.?\s+|Prof\.?\s+|Dipl\.-Ing\.\s+|Dipl\.-Psych\.\s+|Dipl\.-Ing\.\s+Univ\.\s+)([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*|([A-Z]\.)\s+[A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.111 | 0.003 | 0.006 | 9 | 1 | 8 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 1 | 8 | 257 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `61554`) (sent_id: `61554`)


Dr. Achilles

| Predicted | Gold |
|---|---|
| `Achilles` | `Achilles` |

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

</details>

---

## `Names after 'Richter' or 'Vorsitzender'`

**F1:** 0.018 | **Precision:** 0.500 | **Recall:** 0.009  

**Format:** `regex`  
**Rule ID:** `4f43c0ca`  
**Description:**
Captures names following judicial titles like 'Richter', 'Vorsitzender', 'Richterin', 'Vorsitzende Richterin', ensuring the name is captured even if preceded by 'Dr.' or 'Prof.'.

**Content:**
```
\b(?:Richter|Vorsitzender|Richterin|Vorsitzende Richterin|Vorsitzenden Richters)\s+(?:Dr\.?\s+|Prof\.?\s+|Dipl\.-[A-Za-z]+\s+)?([A-Z][a-zäöüß]+(?:-[A-Z][a-zäöüß]+)*)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.500 | 0.009 | 0.018 | 6 | 3 | 3 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 3 | 3 | 303 |

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

**Example 1** (doc_id: `62802`) (sent_id: `62802`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2014 004 201.0 hat der 29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 29. November 2017 durch die Vorsitzende Richterin Dr. Mittenberger-Huber , die Richterin Akintche und die Richterin Seyfarth beschlossen :

| Predicted | Gold |
|---|---|
| `Akintche` | `Akintche` |
| `Seyfarth` | `Seyfarth` |

**Missed by this rule (FN):**

- `29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Mittenberger-Huber` (PER)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `64527`) (sent_id: `64527`)


In der Beschwerdesache betreffend die Marke 30 2011 035 856 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 1** (doc_id: `64925`) (sent_id: `64925`)


In der Beschwerdesache betreffend die Marke 30 2010 003 649 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 2** (doc_id: `65500`) (sent_id: `65500`)


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

## `Names after 'Herr'`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `0599e43e`  
**Description:**
Captures names immediately following the title 'Herr' (including 'Herrn').

**Content:**
```
\b(?:Herr|Herrn)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Names after 'Dr.' or 'Prof.'`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `6e133f27`  
**Description:**
Captures names following titles like Dr. or Prof., handling both full names and initials.

**Content:**
```
\b(?:Dr\.?\s+|Prof\.?\s+)([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*|[A-Z]\.[ ]+[A-Z]\.|[A-Z]\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Names after 'Rechtsanwältin' or 'Rechtsanwalt'`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `6e0b7cc1`  
**Description:**
Captures names following legal profession titles.

**Content:**
```
\b(?:Rechtsanwältin|Rechtsanwalt)\s+(?:Dr\.?\s+|Prof\.?\s+)?([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Names with dots in middle (e.g., B1 …)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `6a893240`  
**Description:**
Captures anonymized names with dots and ellipses or spaces (e.g., 'B1 …', 'K …', 'K1 …').

**Content:**
```
\b([A-Z]\d?\s+…+)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Anonymized names with ellipses`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `daa8797d`  
**Description:**
Captures anonymized names with ellipses like 'K …', 'B1 …', 'T. D.', 'L. …', 'Ch. …' ensuring no trailing spaces are included.

**Content:**
```
\b([A-Z]\s+…|[A-Z]\d+\s+…|T\.\s+D\.|B1\s+…|K1\s+…|H\.\s+…|L\.\s+…|Ch\.\s+…|T\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Surnames after 'Generalanwalt' or 'Generalanwältin'`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `549dd13e`  
**Description:**
Captures names following 'Generalanwalt' or 'Generalanwältin' titles.

**Content:**
```
\b(?:Generalanwalt|Generalanwältin)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Surnames after 'Richter' or 'Vorsitzender' (refined)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `68df845b`  
**Description:**
Captures names following judicial titles, ensuring the name is captured correctly even if preceded by 'Dipl.' or 'Prof.'.

**Content:**
```
\b(?:Richter|Vorsitzender)\s+(?:Dipl\.-[a-z]+\s+)?([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Surnames after 'Rechtsanwalt' or 'Rechtsanwältin' (refined)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `e91ec092`  
**Description:**
Captures names following legal profession titles, handling potential titles like 'Dr.' or 'Prof.' before the name.

**Content:**
```
\b(?:Rechtsanwalt|Rechtsanwältin)\s+(?:Dr\.?\s+|Prof\.?\s+)?([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Initials with dots (standalone context)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `5ae5e583`  
**Description:**
Captures single initials with dots (e.g., 'A.', 'S.') when they appear in contexts suggesting a name, such as after 'Dr.', 'Prof.', 'Herr', or at the start of a sentence followed by a verb.

**Content:**
```
(?:Dr\.?\s+|Prof\.?\s+|Professor\s+|Herr\s+|Herrn\s+|^|\b)([A-Z]\.)\b
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

## `Isolated known surnames`

**F1:** 0.191 | **Precision:** 0.146 | **Recall:** 0.275  

**Format:** `regex`  
**Rule ID:** `325e1c8e`  
**Description:**
Captures specific known surnames appearing in isolation, ensuring they are treated as PER. Excludes common German words like 'Das', 'Der', 'Die'.

**Content:**
```
\b(?:Schäfer|Brückner|Volz|Treber|Knoll|Kriener|Nielsen|Kortbein|Schmid|Söchtig|Kortge|Jacobi|Schödel|Fiamingo|Kelvin|Schilling|Zimmermann|Dölp|Wollensak|Heinkel|Wemheuer|Sost-Scheible|Kirchhof|Paul Kirchhof|Koch|Brune|Grüneberg|Hayen|Lipphaus|Merkel|Eylert|Krumbiegel|Kayser|Jäger|Merzbach|Hacker|Meiser|Becker|Zeng|Merk|Beji Caid Essebsi|Gericke|Franke|Busch|Bender|Augat|Tiemann|Löffler|Schlünder|Schmidt|Schmalz|Melzacks|Edda Redeker|Rosen|Kortbein|Schmid|Söchtig|Kortge|Jacobi|Schödel|Linck|Schultz|Bellay|Leitz|Fieback|Rachor|Cosima|Hoch|Appl|Berger|Quentin|Roloff|Lohmann|Raum|Spinner|St|Br\.|B1|S3|S4|Karaçay|Schramm|Egerer|Kätker|Wismeth|Freudenreich|Schwitzer|Enerji Yapi-Yol Sen|Schwabe|Paffrath|Derstadt|Gallner|Herrmann|Shah|Krasshöfer|Limperg|Mosbacher|Schneider|Niemann|Zwanziger|Brenneisen|Hausmann|Kazele|Hohoff|Roggenbuck|Hamdorf|Grabinski|Krehl|Kosziol|Sunder|Mayen|Seiters|Schlewing|Spaniol|Kirchhoff|Fritz|Vogelsang|Lauer|Mutzbauer|Cierniak|Müller|Ahrendt|Dö.|Widuch|Menezes|Sander|Fischermeier|Hoffmann|Kleinschmidt|Kirschneck|Matter|Kapels|Jostes|Da.|Maksymiw|Schell|Münzberg|D7|Peter Lorsbach|Lorsbach|D1|Stresemann|Feddersen|Botur|Galke|Grupp|Naumann|Suckow|Cirener|Grube|Rahmstorf|Gehrlein|von Pentz)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.146 | 0.275 | 0.191 | 609 | 89 | 520 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 89 | 520 | 235 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60150`) (sent_id: `60150`)


Gallner

| Predicted | Gold |
|---|---|
| `Gallner` | `Gallner` |

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

**Example 3** (doc_id: `60504`) (sent_id: `60504`)


Dölp

| Predicted | Gold |
|---|---|
| `Dölp` | `Dölp` |

**Example 4** (doc_id: `60546`) (sent_id: `60546`)


Sost-Scheible

| Predicted | Gold |
|---|---|
| `Sost-Scheible` | `Sost-Scheible` |

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

**Example 7** (doc_id: `60781`) (sent_id: `60781`)


Mosbacher

| Predicted | Gold |
|---|---|
| `Mosbacher` | `Mosbacher` |

**Example 8** (doc_id: `60821`) (sent_id: `60821`)


Gallner

| Predicted | Gold |
|---|---|
| `Gallner` | `Gallner` |

**Example 9** (doc_id: `60994`) (sent_id: `60994`)


Hayen

| Predicted | Gold |
|---|---|
| `Hayen` | `Hayen` |

**Example 10** (doc_id: `61019`) (sent_id: `61019`)


Wemheuer

| Predicted | Gold |
|---|---|
| `Wemheuer` | `Wemheuer` |

**Example 11** (doc_id: `61083`) (sent_id: `61083`)


Wemheuer

| Predicted | Gold |
|---|---|
| `Wemheuer` | `Wemheuer` |

**Example 12** (doc_id: `61123`) (sent_id: `61123`)


Vogelsang

| Predicted | Gold |
|---|---|
| `Vogelsang` | `Vogelsang` |

**Example 13** (doc_id: `61174`) (sent_id: `61174`)


Hohoff

| Predicted | Gold |
|---|---|
| `Hohoff` | `Hohoff` |

**Example 14** (doc_id: `61183`) (sent_id: `61183`)


Schmidt

| Predicted | Gold |
|---|---|
| `Schmidt` | `Schmidt` |

**Example 15** (doc_id: `61238`) (sent_id: `61238`)


Becker

| Predicted | Gold |
|---|---|
| `Becker` | `Becker` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60070`) (sent_id: `60070`)


Eine Mindestentfernung zwischen Haupt- und beruflicher Zweitwohnung bestimmt das Einkommensteuergesetz nicht ( Geserich , in : Kirchhof / Söhn / Mellinghoff , EStG , § 9 Rz G 60 ) .

**False Positives:**

- `Kirchhof` — partial — pred is substring of gold: `Geserich , in : Kirchhof / Söhn / Mellinghoff , EStG , § 9 Rz G 60`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Einkommensteuergesetz`(NRM)
- `Geserich , in : Kirchhof / Söhn / Mellinghoff , EStG , § 9 Rz G 60`(LIT)

**Example 1** (doc_id: `60075`) (sent_id: `60075`)


Da weder Art. 19 Abs. 4 noch Art. 3 Abs. 1 GG zur Regelung einer vom jeweiligen Landesrecht unabhängigen einheitlichen Normenkontrollzuständigkeit eines gemeinsamen Obergerichts verpflichten , gebieten sie auch nicht , eine solche Zuständigkeit nach Maßgabe der großzügigsten in den beteiligten Ländern getroffenen Regelung vorzusehen .

**False Positives:**

- `Da ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Art. 19 Abs. 4 noch Art. 3 Abs. 1 GG`(NRM)

**Example 2** (doc_id: `60095`) (sent_id: `60095`)


Das Ziel des Sterbens sei doch lediglich , ins Paradies zu kommen , das wolle er .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 3** (doc_id: `60100`) (sent_id: `60100`)


Der in D1 zudem verwendete Keramik-Dauerfilter werde nicht näher beschrieben , so dass davon auszugehen sei , dass er lediglich eine Filterfunktion zur Verfügung stelle und insbesondere nicht mit der Erzeugung der Verwirbelungen des Wassers im Zusammenhang stehe .

**False Positives:**

- `D1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `60121`) (sent_id: `60121`)


Das Landesarbeitsgericht hat angenommen , die von der Klägerin gehaltenen Lehrveranstaltungen hätten einen wissenschaftlichen Zuschnitt , da andernfalls das Ausbildungsziel , die Kompetenz zu wissenschaftlicher Arbeit mit Literaturtexten zu vermitteln , nicht zu erreichen sei .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `60141`) (sent_id: `60141`)


Da diese Fragen sowohl in tatsächlicher als auch in rechtlicher Hinsicht der Klärung im Hauptsacheverfahren vorzubehalten sind , fällt die im Rahmen des § 123 VwGO zu treffende Abwägung wegen des Gewichts der möglicherweise im Raum stehenden öffentlichen Belange des Geheimnisschutzes sowie berechtigter schutzwürdiger Interessen Privater an der Vertraulichkeit zulasten des Antragstellers aus .

**False Positives:**

- `Da ` — no gold match — likely missing annotation
- `Raum` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `§ 123 VwGO`(NRM)

**Example 6** (doc_id: `60159`) (sent_id: `60159`)


5. Das Landgericht wies die Klinik mit Beschluss vom 29. Juli 2016 an , den Ausdruck der auf dem Klinikrechner gespeicherten Datei des Beschwerdeführers aus der Krankenakte zu entfernen .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 7** (doc_id: `60174`) (sent_id: `60174`)


Das LSG hat die Beklagte zur Zahlung von 80 Euro nebst Zinsen in Höhe von 5 % über dem jeweiligen Basiszinssatz seit dem 7. 3. 2014 verurteilt und im Übrigen die Berufung der Klägerin zurückgewiesen : Abgesehen von den zu Unrecht mit aufgerechneten 80 Euro Selbstbeteiligung stehe der Beklagten ein Anspruch auf Erstattung des übrigen Rechnungsbetrags zu .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `60181`) (sent_id: `60181`)


Das Gewaltverbot beinhaltet lediglich eine Unterlassungspflicht , vermittelt jedoch keinen Anspruch auf Unterlassung .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 9** (doc_id: `60195`) (sent_id: `60195`)


Das AG hob diese Anordnung durch Beschluss vom 7. August 2017 auf , da der Insolvenzplan bindend sei und die geltend gemachten Forderungen von diesem umfasst seien .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 10** (doc_id: `60202`) (sent_id: `60202`)


Das FG hat zu Recht entschieden , dass der Kläger die Voraussetzungen für die Zuerkennung der Gemeinnützigkeit nicht erfüllte und das FA daher die Anerkennung als gemeinnütziger Verein widerrufen durfte .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 11** (doc_id: `60247`) (sent_id: `60247`)


1. Das Landgericht hat dazu im Wesentlichen Folgendes festgestellt :

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 12** (doc_id: `60259`) (sent_id: `60259`)


Da sich der maßgebliche Sachverhalt des vorliegenden Rechtsstreits nach dem Zeitpunkt des Inkrafttretens dieses Beschlusses zugetragen hat , unterliegt er aus zeitlichen Gründen allein der VO ( EG ) Nr 883/2004 und der VO ( EG ) Nr 987/2009 ( vgl Art 90 Abs 1 Buchst c VO < EG > Nr 883/2004 und Art 96 Abs 1 Buchst c VO < EG > Nr 987/2009 jeweils iVm dem Beschluss Nr 1/2012 ) .

**False Positives:**

- `Da ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `VO ( EG ) Nr 883/2004`(NRM)
- `VO ( EG ) Nr 987/2009`(NRM)
- `Art 90 Abs 1 Buchst c VO < EG > Nr 883/2004`(NRM)
- `Art 96 Abs 1 Buchst c VO < EG > Nr 987/2009`(NRM)

**Example 13** (doc_id: `60260`) (sent_id: `60260`)


Das trifft auf die genannte Prüfung schon deshalb nicht zu , weil der Kläger an dieser ( Abschluss- ) Prüfung als Schüler der Fachschule teilgenommen hat .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 14** (doc_id: `60261`) (sent_id: `60261`)


3. Das Streitpatent betrifft einen Forstanhänger mit einer Knickdeichsel , der gemäß der Absätze [ 0004 ] und [ 0006 ] der Streitpatentschrift , im folgenden SPS genannt , beispielsweise zum Laden und Liefern von Bäumen dient und von einer Zugmaschine in Form eines Traktors gezogen wird .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 15** (doc_id: `60290`) (sent_id: `60290`)


Dieser Disilicatrohling wird dann maschinell zu entsprechenden dentalen Restaurationen weiterverarbeitet ( vgl. D7 , Patentanspruch 12 , S. 4 , [ 0036 ] ) .

**False Positives:**

- `D7` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 16** (doc_id: `60297`) (sent_id: `60297`)


Das Landgericht hat im Rahmen der Strafzumessung rechtsfehlerhaft das Gesamtstrafübel für die Angeklagte nicht in den Blick genommen , das - infolge der Zäsurwirkung des Urteils des Amtsgerichts Kulmbach vom 6. August 2015 - aus der obligatorischen Bildung von zwei Gesamtstrafen resultierte .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Amtsgerichts Kulmbach`(ORG)

**Example 17** (doc_id: `60323`) (sent_id: `60323`)


Das LSG habe ihm den barrierefreien Zugang zur mündlichen Verhandlung in seiner Sache verwehrt und ihm dadurch die Möglichkeit abgeschnitten , im Rahmen der Verhandlung sich zu Wort zu melden und ggf darin neue Beweisanträge zu stellen .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 18** (doc_id: `60324`) (sent_id: `60324`)


Soweit die Widersprechende sich darauf beruft , dass auch kennzeichnungsschwache Marken zumindest Schutz gegen eine identische Übernahme beanspruchen könnten , führt dieser grundsätzlich zutreffende Einwand gleichfalls nicht zur Bejahung der Verwechslungsgefahr , da sich die hier zu vergleichenden Zeichen – wie nachfolgend unter Ziffer 1. 3. dargelegt – erheblich unterscheiden ( vgl. im Übrigen zum Schutzumfang zu Unrecht eingetragener , materiell schutzunfähiger Marken Ströbele / Hacker , MarkenG , 11. Aufl. , § 9 Rn. 194 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Ströbele / Hacker , MarkenG , 11. Aufl. , § 9 Rn. 194`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 9 Rn. 194`(LIT)

**Example 19** (doc_id: `60329`) (sent_id: `60329`)


Die strafschärfende Berücksichtigung der hierin liegenden Schuldsteigerung gerate weder mit dem in § 46 Abs. 3 StGB verankerten Doppelverwertungsverbot von Tatbestandsmerkmalen ( SSW-StGB / Eschelbach , aaO , § 46 Rn. 93 , 185 ; von Heintschel-Heinegg , Streng-FS 2017 S. 229 , 239 ) noch mit dem Gedanken in Konflikt , dass es sich um das Regeltatbild des Totschlags handele ( Fahl , JR 2017 , 391 , 393 ; MüKo / Schneider , aaO , § 212 Rn. 82 ; Tomiak , HRRS 2017 , 225 ff. ) .

**False Positives:**

- `Schneider` — partial — pred is substring of gold: `MüKo / Schneider , aaO , § 212 Rn. 82`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 46 Abs. 3 StGB`(NRM)
- `SSW-StGB / Eschelbach , aaO , § 46 Rn. 93 , 185`(LIT)
- `von Heintschel-Heinegg , Streng-FS 2017 S. 229 , 239`(LIT)
- `Fahl , JR 2017 , 391 , 393`(LIT)
- `MüKo / Schneider , aaO , § 212 Rn. 82`(LIT)
- `Tomiak , HRRS 2017 , 225 ff.`(LIT)

**Example 20** (doc_id: `60334`) (sent_id: `60334`)


1.1 Das Patent betrifft die Verwendung eines Rohlings aus einem Lithiumsilicatmaterial , das durch maschinelle Verarbeitung einfach geformt und anschließend zu dentalen Restaurationen von hoher Festigkeit umgewandelt werden kann ( vgl. Streitpatentschrift , Patentansprüche 1 , 13 , 17 und 19 , S. 2 , [ 0001 ] ) .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 21** (doc_id: `60346`) (sent_id: `60346`)


Das Urteil beruht auf diesem Verfahrensfehler ( dazu b ) .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 22** (doc_id: `60348`) (sent_id: `60348`)


Das FA lehnte die Änderung ab .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 23** (doc_id: `60351`) (sent_id: `60351`)


Das Gesetz verschaffe aber keinen nachträglichen Zugang zu einem Zusatzversorgungssystem , das den Beschäftigten unabhängig von einer politischen Verfolgung aufgrund der restriktiven Einbeziehungspraxis der DDR vorenthalten worden sei .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `DDR`(LOC)

**Example 24** (doc_id: `60353`) (sent_id: `60353`)


Das Verwaltungsgericht habe die Abweisung der Klage als offensichtlich unbegründet jedoch allein damit begründet , dass es den Vortrag zum individuellen Verfolgungsschicksal als krass widersprüchlich und damit unglaubhaft eingestuft , und das Offensichtlichkeitsurteil auf § 30 Abs. 1 AsylG und § 30 Abs. 3 Nr. 1 AsylG gestützt habe .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 30 Abs. 1 AsylG`(NRM)
- `§ 30 Abs. 3 Nr. 1 AsylG`(NRM)

**Example 25** (doc_id: `60395`) (sent_id: `60395`)


Das Beitragsaufkommen ist nach § 3 Abs. 2 Satz 2 und 3 RFinStV gedeckelt .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 3 Abs. 2 Satz 2 und 3 RFinStV`(REG)

**Example 26** (doc_id: `60425`) (sent_id: `60425`)


IV. Das Land Baden-Württemberg hat dem Beschwerdeführer gemäß § 34a Abs. 2 BVerfGG die notwendigen Auslagen zu erstatten .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Baden-Württemberg`(LOC)
- `§ 34a Abs. 2 BVerfGG`(NRM)

**Example 27** (doc_id: `60439`) (sent_id: `60439`)


Das " gelebte " Vertragsverhältnis entspricht dem formell vereinbarten Vertrag über ein selbstständiges Dienstverhältnis .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 28** (doc_id: `60471`) (sent_id: `60471`)


Das Truppendienstgericht ist nicht befugt , im Rahmen der Entscheidung , ob einer Nichtzulassungsbeschwerde abgeholfen wird , den angefochtenen Beschluss nachzubessern und gerügte Verfahrensmängel zu beheben .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 29** (doc_id: `60478`) (sent_id: `60478`)


Das FA beantragt , das Urteil des FG aufzuheben und die Klage abzuweisen .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 30** (doc_id: `60497`) (sent_id: `60497`)


Da die Klägerin in beiden Varianten letztlich das alleinige Steuersubjekt für denselben Betrag an Kapitalertragsteuer ist , handelt es sich lediglich um verschiedene Begründungen des einheitlichen Nachforderungsbescheids und nicht um mehrere Verwaltungsakte in einem Sammelbescheid .

**False Positives:**

- `Da ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 31** (doc_id: `60498`) (sent_id: `60498`)


1. a ) Das Grundrecht aus Art. 9 Abs. 3 GG ist für jedermann und für alle Berufe gewährleistet und umfasst auch die Koalition als solche und ihr Recht , durch spezifisch koalitionsgemäße Betätigung die in Art. 9 Abs. 3 GG genannten Zwecke zu verfolgen , nämlich die Arbeits- und Wirtschaftsbedingungen zu wahren und zu fördern ( vgl. BVerfGE 4 , 96 < 107 > ; 17 , 319 < 333 > ; 18 , 18 < 25 f. > ; 50 , 290 < 367 > ) .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Art. 9 Abs. 3 GG`(NRM)
- `Art. 9 Abs. 3 GG`(NRM)
- `BVerfGE 4 , 96 < 107 > ; 17 , 319 < 333 > ; 18 , 18 < 25 f. > ; 50 , 290 < 367 >`(RS)

**Example 32** (doc_id: `60507`) (sent_id: `60507`)


Das hiermit einhergehende Fehlen einer zusätzlichen Belastung der Versichertengemeinschaft trotz tatsächlichen Rentenbezugs entspricht wirtschaftlich betrachtet dem Fall des " nicht mehr " Inanspruchnehmens im Sinne des § 77 Abs 3 S 3 Nr 1 SGB VI .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 77 Abs 3 S 3 Nr 1 SGB VI`(NRM)

**Example 33** (doc_id: `60508`) (sent_id: `60508`)


Das FA beurteilte die Vereinbarungen als bloße Finanzierungsvereinbarungen und rechnete der KG einerseits Zinsanteile aus den Leasingzahlungen zu und zog andererseits als Aufwand insbesondere Zinsen für den Lieferantenkredit sowie weitere Aufwendungen ab .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 34** (doc_id: `60518`) (sent_id: `60518`)


Das Berufungsgericht hat den Wiedereinsetzungsantrag zurückgewiesen , die Berufung als unzulässig verworfen und eine Gegenvorstellung der Klägerin zurückgewiesen .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 35** (doc_id: `60534`) (sent_id: `60534`)


Das LSG hat die Berufung gegen das klageabweisende Urteil des SG zu Recht zurückgewiesen .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 36** (doc_id: `60535`) (sent_id: `60535`)


Das erfordert jedoch einen schriftlich abzufassenden und der Geschäftsstelle zu übergebenden Beschluss ( § 153 Abs 1 iVm § 142 Abs 1 und § 134 SGG ) , der der Zustellung an die Beteiligten ( § 153 Abs 1 iVm § 142 Abs 1 und § 133 Satz 2 SGG ) bedarf ( BSG vom 27. 4. 2010 - B 2 U 344/09 B - SozR 4 - 1500 § 153 Nr 8 RdNr 7 ; BSG vom 24. 10. 2013 - B 13 R 240/12 B - juris RdNr 9 ) .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 153 Abs 1 iVm § 142 Abs 1 und § 134 SGG`(NRM)
- `§ 153 Abs 1 iVm § 142 Abs 1 und § 133 Satz 2 SGG`(NRM)
- `BSG vom 27. 4. 2010 - B 2 U 344/09 B - SozR 4 - 1500 § 153 Nr 8 RdNr 7`(RS)
- `BSG vom 24. 10. 2013 - B 13 R 240/12 B - juris RdNr 9`(RS)

**Example 37** (doc_id: `60556`) (sent_id: `60556`)


Das Urteil erweise sich auch aus anderen Gründen als richtig .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 38** (doc_id: `60565`) (sent_id: `60565`)


Da der Anspruch auf die Zahlung einer Abfindung erst mit der rechtlichen Beendigung des Arbeitsverhältnisses entstehe , seien Änderungen der Berechnungsfaktoren während des Altersteilzeitarbeitsverhältnisses in Anwendung des § 11 Abs. 1 Satz 2 TV ATZ zu berücksichtigen .

**False Positives:**

- `Da ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 11 Abs. 1 Satz 2 TV ATZ`(REG)

**Example 39** (doc_id: `60577`) (sent_id: `60577`)


Hingegen ist die Zulassung wegen Divergenz gegen eine Entscheidung eines anderen obersten Gerichtshofes des Bundes oder des EuGH nicht zulässig ( vgl Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160 RdNr 11 mwN ) .

**False Positives:**

- `Schmidt` — partial — pred is substring of gold: `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160 RdNr 11`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH`(ORG)
- `Leitherer in Meyer-Ladewig / Keller / Leitherer / Schmidt , SGG , 12. Aufl 2017 , § 160 RdNr 11`(LIT)

**Example 40** (doc_id: `60634`) (sent_id: `60634`)


Das Landgericht hätte sich an dieser Stelle daher auch damit auseinandersetzen müssen , dass der Angeklagte am 12. Juli 2015 einen Diebstahl „ im besonders schweren Fall “ beging , wofür er am 6. Juli 2016 vom Amtsgericht Frankfurt am Main - Außenstelle Höchst - verurteilt wurde .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Amtsgericht Frankfurt am Main - Außenstelle Höchst -`(ORG)

**Example 41** (doc_id: `60665`) (sent_id: `60665`)


1.7.2 Hierzu hat die Einsprechende das Beispiel 22 der D1 nachgearbeitet und diese Nacharbeitung als Anlagenkonvolut D21A eingereicht , um dadurch zu belegen , dass der Rohling nach der ersten Wärmebehandlung Lithiummetasilicat als eine Hauptkristallphase entsprechend den Merkmalen des erteilten Patentanspruchs 1 enthält .

**False Positives:**

- `D1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 42** (doc_id: `60680`) (sent_id: `60680`)


Das SG hat die hiergegen erhobene Klage abgewiesen ( Urteil vom 19. 1. 2016 ) .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 43** (doc_id: `60689`) (sent_id: `60689`)


Das ergibt sich aber hinreichend aus dessen eigenen Angaben im Chatverlauf , wonach er " nach Syrien oder Iraq " wollte und deshalb eine Verfügung mit räumlicher Geltungsbeschränkung seines Personalausweises auf das Inland erhalten hat , sowie aus der Äußerung , dass sein Plan " der beste in der Geschichte von dawla sei " ; Dawla werde in die Geschichte eingehen .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Syrien`(LOC)
- `Iraq`(LOC)

**Example 44** (doc_id: `60706`) (sent_id: `60706`)


Das Deutsche Patent- und Markenamt , Markenstelle für Klasse 7 , hat die Anmeldung nach vorangegangener Beanstandung vom 8. Februar 2013 mit Beschlüssen vom 16. April 2013 und vom 11. August 2015 , von denen letzterer im Erinnerungsverfahren ergangen ist , zurückgewiesen .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutsche Patent- und Markenamt , Markenstelle für Klasse 7`(ORG)

**Example 45** (doc_id: `60714`) (sent_id: `60714`)


Da allein das Fehlen jeglicher Unterscheidungskraft ein Eintragungshindernis begründet , ist nach der Rechtsprechung des Bundesgerichtshofs ein großzügiger Maßstab anzulegen , so dass jede auch noch so geringe Unterscheidungskraft genügt , um das Schutzhindernis zu überwinden ( vgl. BGH GRUR 2012 , 1143 , Rdnr. 7 - Starsat ; GRUR 2012 , 1044 , Rdnr. 9 - Neuschwanstein ; GRUR 2012 , 270 , Rdnr. 8 - Link economy ) .

**False Positives:**

- `Da ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Bundesgerichtshofs`(ORG)
- `BGH GRUR 2012 , 1143 , Rdnr. 7 - Starsat`(RS)
- `GRUR 2012 , 1044 , Rdnr. 9 - Neuschwanstein`(RS)
- `GRUR 2012 , 270 , Rdnr. 8 - Link economy`(RS)

**Example 46** (doc_id: `60734`) (sent_id: `60734`)


Da sich die Klägerin das Verschulden ihres Prozessbevollmächtigten , der eine Berufungs- und Berufungsbegründungsschrift dem Gericht über einen Erklärungsboten zuleitet , nach § 85 Abs. 2 ZPO zurechnen lassen muss ( BGH , Beschluss vom 19. Juni 2007 - VI ZB 81/05 , FamRZ 2007 , 1638 Rn. 6 ) , war der Mangel der Form auch nicht unverschuldet .

**False Positives:**

- `Da ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 85 Abs. 2 ZPO`(NRM)
- `BGH , Beschluss vom 19. Juni 2007 - VI ZB 81/05 , FamRZ 2007 , 1638 Rn. 6`(RS)

**Example 47** (doc_id: `60742`) (sent_id: `60742`)


Von einem Bedienhebel nach M4 und Teilmerkmal M5 ist auf den S. 244 - 253 des Fachtagungsbuches keine Rede , denn die Fig. 5 zeigt nur symbolische Darstellungen für die Funksteuerung , das Bedienpult oder das Laptop und lässt allenfalls den Schluss auf Tasten zu , was auch in Übereinstimmung mit den Ausführungen zur Drehzahlregelung über zwei Taster ( Rechts / Links ) steht ( vgl. EI ( D1 ) : S. 249 : „ Funktion der SPCD “ ) .

**False Positives:**

- `D1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 48** (doc_id: `60768`) (sent_id: `60768`)


Da diese nicht aufzufinden gewesen sei , habe die Kanzleimitarbeiterin jedoch nichts unternommen .

**False Positives:**

- `Da ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 49** (doc_id: `60782`) (sent_id: `60782`)


a ) Das Beschwerdegericht hat das Vorbringen des Antragsgegners als unsubstantiiert angesehen , weil er nicht dargelegt habe , inwieweit das Stromnetz derzeit und nach dem geplanten Anschluss einer Kläranlage in Anspruch genommen werde und welche Kapazität noch frei sei bzw. frei bleibe .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 50** (doc_id: `60797`) (sent_id: `60797`)


Das beklagte Jobcenter gab bezogen auf dieses Mietverhältnis gegenüber der die Vermieterin vertretenden Rechtsanwaltskanzlei am 24. 2. 2015 eine Bürgschaftserklärung ab .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 51** (doc_id: `60798`) (sent_id: `60798`)


Im Ladungsschreiben müssen die Gründe für die Nichtverlängerung nicht mitgeteilt werden ( Bolwin / Sponer Bühnen- und Orchesterrecht Stand November 2017 Teil A I § 61 NV Bühne Rn. 61 ; Schneider in Nix / Hegemann / Hemke Normalvertrag Bühne 2. Aufl. § 61 Rn. 35 ; vgl. zu § 24 Abs. 4 Normalvertrag Tanz BAG 18. April 1986 - 7 AZR 114/85 - zu III der Gründe , BAGE 51 , 375 ) .

**False Positives:**

- `Schneider` — partial — pred is substring of gold: `Schneider in Nix / Hegemann / Hemke Normalvertrag Bühne 2. Aufl. § 61 Rn. 35`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bolwin / Sponer Bühnen- und Orchesterrecht Stand November 2017 Teil A I § 61 NV Bühne Rn. 61`(LIT)
- `Schneider in Nix / Hegemann / Hemke Normalvertrag Bühne 2. Aufl. § 61 Rn. 35`(LIT)
- `§ 24 Abs. 4 Normalvertrag Tanz`(REG)
- `BAG 18. April 1986 - 7 AZR 114/85 - zu III der Gründe , BAGE 51 , 375`(RS)

**Example 52** (doc_id: `60825`) (sent_id: `60825`)


Das LSG hat zu Unrecht ein Prozessurteil statt eines Sachurteils erlassen .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 53** (doc_id: `60870`) (sent_id: `60870`)


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

**Example 54** (doc_id: `60874`) (sent_id: `60874`)


1. Das Berufungsgericht hat gemeint , der Kläger sei nicht verpflichtet gewesen , sich durch das Gesundheitsamt auf seine Arbeitsfähigkeit untersuchen zu lassen .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 55** (doc_id: `60878`) (sent_id: `60878`)


Das Berufungsgericht wird jedoch zu erwägen haben , ob die Beklagte eine sekundäre Darlegungslast trifft .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 56** (doc_id: `60890`) (sent_id: `60890`)


I. 1. Das Streitpatent betrifft die Bereitstellung einer den hochselektiven PDE5 -Inhibitor Tadalafil enthaltenden Einheitsdosiszusammensetzung für die Behandlung sexueller Dysfunktion ( vgl. NIK1.3 / NiK1 S. 2 Abs. [ 0002 ] sowie Patentansprüche 1 und 10 ) .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 57** (doc_id: `60900`) (sent_id: `60900`)


D7 DE 369 381

**False Positives:**

- `D7` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 58** (doc_id: `60979`) (sent_id: `60979`)


Da jedoch die vom Kläger gerügten Handlungen des FG-Präsidenten nach dem Absenden der Urteilsausfertigungen lagen , scheidet eine Gehörsverletzung des Klägers aus .

**False Positives:**

- `Da ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 59** (doc_id: `60993`) (sent_id: `60993`)


a ) Das Oberlandesgericht hat plausibel begründet , dass der in § 1791b Abs. 1 Satz 1 BGB normierte Vorrang der ehrenamtlichen Einzelvormundschaft vor der Amtsvormundschaft nur in Bezug auf einen geeigneten Einzelvormund gelte .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 1791b Abs. 1 Satz 1 BGB`(NRM)

**Example 60** (doc_id: `61000`) (sent_id: `61000`)


aa ) Das FA hat keinen Verfahrensfehler dadurch begangen , dass es die Einsprüche gegen die Schätzungsbescheide für die Streitjahre als unbegründet zurückgewiesen hat , nachdem die Klägerin keine Steuererklärungen eingereicht hatte .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 61** (doc_id: `61005`) (sent_id: `61005`)


Das heißt , Δt wird bestimmt , zumindest die Antwortzeit zu sein und derart , dass die Zeit , die zum Beginn der Entladung erforderlich ist , mehrere Sekunden lang wird ( zum Beispiel etwa 2 Sekunden ) .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 62** (doc_id: `61006`) (sent_id: `61006`)


Der Gegenstand der erteilten Patentansprüche werde auch nicht durch die implizite Offenbarung der D1 neuheitsschädlich vorweggenommen .

**False Positives:**

- `D1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 63** (doc_id: `61007`) (sent_id: `61007`)


Im Allgemeinen lassen unter anderem Angaben über Werbeaufwendungen Schlüsse auf die Verkehrsbekanntheit einer Marke zu ( BGH GRUR 2013 , 833 , 836 Rn. 41 – Culinaria / Villa Culinaria ; Hacker , a. a. O. , § 9 Rn. 160 ) .

**False Positives:**

- `Hacker` — partial — pred is substring of gold: `Hacker , a. a. O. , § 9 Rn. 160`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH GRUR 2013 , 833 , 836 Rn. 41 – Culinaria / Villa Culinaria`(RS)
- `Hacker , a. a. O. , § 9 Rn. 160`(LIT)

**Example 64** (doc_id: `61010`) (sent_id: `61010`)


Das Landesarbeitsgericht hat die Berufung des Beklagten zurückgewiesen .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 65** (doc_id: `61011`) (sent_id: `61011`)


Das Geständnis des Beschwerdeführers , die inkriminierten Äußerungen stammten von ihm , bezieht sich nur auf den Blogeintrag und ist daher für den ebenfalls vom Anfangsverdacht umfassten Kommentar auf der Webseite " D … " unbeachtlich .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 66** (doc_id: `61016`) (sent_id: `61016`)


Das Bundesverwaltungsgericht hat bereits entschieden , dass der Vorausbau einer gemeinschaftlichen Anlage nicht automatisch zum Erlass einer Anordnung nach § 36 Abs. 1 FlurbG berechtigt , dass ihm jedoch für die geforderte Dringlichkeit erhebliches Gewicht zukommt .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Bundesverwaltungsgericht`(ORG)
- `§ 36 Abs. 1 FlurbG`(NRM)

**Example 67** (doc_id: `61026`) (sent_id: `61026`)


Das begriffliche Verständnis der Gesamtmarke Wohlfühlfarbe im Sinn von „ Farbe für das Wohlfühlen “ bereitet dem angesprochenen Publikum somit keinerlei Schwierigkeiten .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 68** (doc_id: `61028`) (sent_id: `61028`)


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

**Example 69** (doc_id: `61031`) (sent_id: `61031`)


4. Das LSG wird die gebotenen Feststellungen nachzuholen haben .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 70** (doc_id: `61033`) (sent_id: `61033`)


Das Patent betrifft ein Verfahren zur Senkung des Blutglukosespiegels bei Säugern durch die Verabreichung sogenannter DP IV-Inhibitoren .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 71** (doc_id: `61037`) (sent_id: `61037`)


Das Verbot betraf danach sämtliche Fußballstadien in Deutschland hinsichtlich nationaler und internationaler Fußballveranstaltungen von Vereinen beziehungsweise Tochtergesellschaften der Fußball-Bundesligen und der Fußballregionalligen sowie des Deutschen Fußball-Bundes .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschland`(LOC)
- `Deutschen Fußball-Bundes`(ORG)

**Example 72** (doc_id: `61088`) (sent_id: `61088`)


Das LSG hat die Berufung zurückgewiesen ( Urteil vom 30. 9. 2015 ) .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 73** (doc_id: `61108`) (sent_id: `61108`)


Das ist dann nicht mehr der Fall , wenn ein Arzt in demselben Planungsbereich bereits in einem die Aufbauphase übersteigenden Zeitraum vertragsärztlich tätig war .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 74** (doc_id: `61125`) (sent_id: `61125`)


D21 -A Anlagenkonvolut : Nacharbeitung des Beispiels 22 der Entgegenhaltung D1 , 4 Seiten , 21. 9. 2017

**False Positives:**

- `D1` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 75** (doc_id: `61131`) (sent_id: `61131`)


Das Lebenszeitprinzip hat - im Zusammenspiel mit dem die amtsangemessene Besoldung sichernden Alimentationsprinzip - die Funktion , die Unabhängigkeit der Beamten im Interesse einer rechtsstaatlichen Verwaltung zu gewährleisten .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 76** (doc_id: `61133`) (sent_id: `61133`)


Das Einspruchsverfahren wurde in der Folgezeit mit dem Kläger fortgeführt , den das FA als Gesamtrechtsnachfolger betrachtete .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 77** (doc_id: `61182`) (sent_id: `61182`)


Das Landgericht kam zu dem Schluss , dass die Durchsuchungsanordnung nur bezüglich des Tatvorwurfs des sexuellen Missbrauchs von Kindern den formalen Anforderungen an einen Durchsuchungsbeschluss genüge , während der Tatvorwurf des Besitzes kinderpornographischer Schriften weder benannt noch hinreichend konkret umschrieben werde .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 78** (doc_id: `61185`) (sent_id: `61185`)


Das Fortbestehen dieser übereinstimmenden Absicht über die gesamte Dauer der gestuften Berufsausbildung zeigt sich an der Bezeichnung des Berufsausbildungsvertrags vom 29. August 2014 als „ Anschlussvertrag “ , der nur kurzen zeitlichen Unterbrechung zwischen den beiden Stufen der Berufsausbildung und daran , dass der Zeitraum der Berufsausbildung zum Zimmerer bei der Festlegung der Vergütung als „ 3. Ausbildungsjahr “ definiert worden ist .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 79** (doc_id: `61195`) (sent_id: `61195`)


Das gilt selbst dann , wenn die von ihnen getroffene Regelung für die Arbeitnehmer günstiger ist als diejenige der Tarifvertragsparteien ( vgl. BAG 30. Mai 2006 - 1 AZR 111/05 - Rn. 27 , BAGE 118 , 211 ) .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BAG 30. Mai 2006 - 1 AZR 111/05 - Rn. 27 , BAGE 118 , 211`(RS)

**Example 80** (doc_id: `61199`) (sent_id: `61199`)


Das FA beantragt , das angefochtene Urteil aufzuheben und die Klage abzuweisen .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 81** (doc_id: `61225`) (sent_id: `61225`)


Das rechtliche Gehör des Antragstellers ist nicht dadurch verletzt worden , dass das Oberverwaltungsgericht bestimmte Schlussfolgerungen aus den beigezogenen Verwaltungsakten ( Beiakten I und II ) gezogen hat , ohne den Antragsteller zuvor ausdrücklich darauf hinzuweisen .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 82** (doc_id: `61241`) (sent_id: `61241`)


Das Berufungsgericht habe seiner Entscheidung einen unzutreffenden Maßstab zugrunde gelegt , indem es die Prüfung einer tierärztlichen Berufsausübung allein an den landesrechtlichen kammer- und versorgungsrechtlichen Normen ausgerichtet und insbesondere § 1 Abs 1 BTÄO als nicht einschlägig erachtet habe .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 1 Abs 1 BTÄO`(NRM)

**Example 83** (doc_id: `61259`) (sent_id: `61259`)


( 2 ) Das Meldeformular , das von der Ausbildungskostenausgleichskasse zur Verfügung gestellt wird , ist zu unterschreiben .

**False Positives:**

- `Das` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Anonymized initials after legal roles`

**F1:** 0.111 | **Precision:** 1.000 | **Recall:** 0.059  

**Format:** `regex`  
**Rule ID:** `4d75d5d9`  
**Description:**
Captures single-letter anonymized names (e.g., 'A', 'K', 'E', 'S') immediately following legal role indicators like 'Angeklagte', 'Kläger', 'Zeuge', etc., ensuring the dot is included if present.

**Content:**
```
\b(?:Angeklagte|Angeklagten|Mitangeklagte|Mitangeklagten|Kläger|Klägerin|Klägers|Klägerin|Zeuge|Zeugin|Geschädigte|Geschädigten|Beteiligte|Beteiligten|Antragsteller|Antragstellerin|Antragstellers|Antragstellerin|Vorsitzender|Richter|Richterin|Rechtsanwalt|Rechtsanwältin|Gutachter|Gutachterin|Sachverständige|Sachverständigen|Herr|Herrn|Dr\.?\s+|Prof\.?\s+|Dipl\.-[A-Za-z]+\s+)([A-Z]\.)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.059 | 0.111 | 19 | 19 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 19 | 0 | 302 |

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

**Example 5** (doc_id: `61586`) (sent_id: `61586`)


Die Klägerin hatte ihren Sitz zum Zeitpunkt des Eintritts der Job-Sharing-Partnerin Dr. E. im Bezirk der KÄV Nordbaden .

| Predicted | Gold |
|---|---|
| `E.` | `E.` |

**Missed by this rule (FN):**

- `KÄV Nordbaden` (ORG)

**Example 6** (doc_id: `61864`) (sent_id: `61864`)


Zu dem Sachverständigengutachten des Prof. Dr. T. sowie dessen ergänzender Stellungnahme hat die Beklagte mitgeteilt , dass sie diesem nach prüfärztlicher Stellungnahme nicht folgen könne ; der Einschätzung des Dr. S. hat sie sich hingegen angeschlossen .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |
| `S.` | `S.` |

**Example 7** (doc_id: `62635`) (sent_id: `62635`)


Schließlich wird das Stellen eines ordnungsgemäßen Beweisantrags mit der Beschwerdebegründung nicht dargelegt , soweit die Klägerin die Sachaufklärungspflicht des LSG dadurch verletzt sieht , dass dieses keine ergänzende gutachterliche Äußerung Dr. R. eingeholt hat .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |

**Example 8** (doc_id: `62752`) (sent_id: `62752`)


Die Implantation der Coils als alleiniger Grund für die stationäre Behandlung der Versicherten sei nach dem überzeugenden MDK-Gutachten ( Dr. S. ) nicht erforderlich gewesen .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Example 9** (doc_id: `63754`) (sent_id: `63754`)


Soweit das LSG Ausführungen von Prof. Dr. S. referiert , dass sich im Zusammenhang mit den Pneumonien auch immer wieder Septitiden entwickelt hätten , macht das LSG nicht deutlich , dass es diese als festgestellt ansieht , und erst recht nicht , von welchem Begriff oder Schweregrad der Sepsis es im Anschluss an Prof. Dr. S. aufgrund welcher Befunde dabei ausgeht ( ältere Klassifizierungen : Systemisches inflammatorisches Response-Syndrom < SIRS > , Sepsis , schwere Sepsis und septischer Schock ; zur darauf aufbauenden DRG-Kodierung vgl Hanser in Zaiß , DRG : Verschlüsseln leicht gemacht , 14. Aufl 2016 , S 101 ff ; neuere Klassifizierungen : Sepsis-related organ failure assessment score < SOFA > , insbesondere für Intensivstationen , und quickSOFA außerhalb von Intensivstationen ) .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |
| `S.` | `S.` |

**Example 10** (doc_id: `63885`) (sent_id: `63885`)


Nach den Ausführungen des im Verfahren von Amts wegen gehörten Sachverständigen Prof. Dr. T. hätten die vom Kläger vorgetragenen Gewalterfahrungen während seiner Heimaufenthalte ab April 1959 nicht die entscheidende Traumatisierung dargestellt .

| Predicted | Gold |
|---|---|
| `T.` | `T.` |

**Example 11** (doc_id: `63999`) (sent_id: `63999`)


Der weitere vom LSG beauftragte Sachverständige Dr. S. ( Neurologe und Psychiater / Psychotherapeut ) hat die quantitative Leistungsfähigkeit der Klägerin mit mindestens 6 Stunden für leichte Arbeiten mit qualitativen Einschränkungen beurteilt .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Example 12** (doc_id: `64291`) (sent_id: `64291`)


Ferner hat sie beantragt , Prof. Dr. B. , Zentralinstitut für seelische Gesundheit M. , Institut für Psychiatrie und Psychosomatische Psychotherapie , als sachverständigen Zeugen zu vernehmen sowie den Sachverständigen Prof. Dr. S. zu den mit Schriftsatz vom 21. 3. 2017 erhobenen Einwendungen anzuhören .

| Predicted | Gold |
|---|---|
| `B.` | `B.` |
| `S.` | `S.` |

**Missed by this rule (FN):**

- `Zentralinstitut für seelische Gesundheit M. , Institut für Psychiatrie und Psychosomatische Psychotherapie` (ORG)

**Example 13** (doc_id: `64530`) (sent_id: `64530`)


Das LSG hat vielmehr im Anschluss an die Begründung , warum es dessen sachverständige Bewertung für überzeugend hält , ausgeführt : " Hingegen hat Dr. H. lediglich auf einer Seite kurz dargestellt , dass beim Kläger seinerzeit kein KIG Grad 3 oder höher vorliege .

| Predicted | Gold |
|---|---|
| `H.` | `H.` |

**Example 14** (doc_id: `65228`) (sent_id: `65228`)


Statt einer beantragten orthopädischen Begutachtung unter Berücksichtigung der Schmerzsymptomatik sei eine Begutachtung durch Dr. N. angeordnet worden , obwohl er ( der Kläger ) auf neurologischem Gebiet völlig gesund sei .

| Predicted | Gold |
|---|---|
| `N.` | `N.` |

**Example 15** (doc_id: `66269`) (sent_id: `66269`)


Die Beklagte hat sich hierzu nicht geäußert und nach der Übersendung des Sachverständigengutachtens des Dr. B. ohne weitere inhaltliche Einlassung mit einer Entscheidung ohne mündliche Verhandlung einverstanden erklärt .

| Predicted | Gold |
|---|---|
| `B.` | `B.` |

</details>

---

## `Isolated anonymized initials`

**F1:** 0.092 | **Precision:** 0.261 | **Recall:** 0.056  

**Format:** `regex`  
**Rule ID:** `959f7758`  
**Description:**
Captures single-letter anonymized names (e.g., 'A', 'K', 'E', 'S') appearing in legal contexts such as after prepositions ('von', 'zu', 'in'), after 'der/die/das', or at the start of a sentence, with a dot.

**Content:**
```
(?:\b(?:von|zu|in|an|auf|bei|nach|vor|mit|ohne|für|gegen|durch|über|unter|neben|zwischen|hinter|vor|nach|um|bis|seit|während|trotz|wegen|statt|außer|neben|gegenüber|entlang|laut|gemäß|inklusive|exklusive|sowie|wie|als|obwohl|wenn|falls|da|denn|dass|ob|obwohl|obgleich|indem)\s+|\b(?:der|die|das|den|dem|des|ein|eine|einem|einen|einer|eines|mein|dein|sein|ihr|unser|euer|ihr|mein|dein|sein|ihr|unser|euer|ihr|der|die|das|den|dem|des|ein|eine|einem|einen|einer|eines)\s+|\b(?:Kläger|Angeklagter|Angeklagte|Zeuge|Zeugin|Geschädigte|Geschädigter|Beteiligte|Beteiligter|Antragsteller|Antragstellerin|Vorsitzender|Richter|Richterin|Rechtsanwalt|Rechtsanwältin|Gutachter|Gutachterin|Sachverständige|Sachverständiger|Herr|Frau)\s+|\b(?:in|zu|von|an|auf|bei|nach|vor|mit|ohne|für|gegen|durch|über|unter|neben|zwischen|hinter|vor|nach|um|bis|seit|während|trotz|wegen|statt|außer|neben|gegenüber|entlang|laut|gemäß|inklusive|exklusive|sowie|wie|als|obwohl|wenn|falls|da|denn|dass|ob|obwohl|obgleich|indem)\s+|\b(?:der|die|das|den|dem|des|ein|eine|einem|einen|einer|eines)\s+|^)([A-Z]\.)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.261 | 0.056 | 0.092 | 69 | 18 | 51 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 18 | 51 | 305 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60270`) (sent_id: `60270`)


Auch der M. wollte nach Angaben des Klägers einen Anschlag auf Zivilisten planen ; hierzu erklärte sich der Kläger ohne Einschränkungen bereit .

| Predicted | Gold |
|---|---|
| `M.` | `M.` |

**Example 1** (doc_id: `60282`) (sent_id: `60282`)


Diese sind rechtswidrig und beschweren die Klägerin , soweit sie Honorar für RLV-Leistungen nicht auch unter Anwendung eines arztpraxisbezogenen RLV , sondern lediglich unter Zugrundelegung einer Obergrenze zuerkennen , deren Höhe von der Zahl der durch S. im streitbefangenen Quartal tatsächlich behandelten Patienten abhängt .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Example 2** (doc_id: `60683`) (sent_id: `60683`)


Ab dem Inkrafttreten des Bundessozialhilfegesetzes ( BSHG ) in den neuen Bundesländern zum 1. 1. 1991 erbrachte das Land B. als der nach Landesrecht zuständige überörtliche Träger der Sozialhilfe Leistungen der Eingliederungshilfe an K.

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `Bundessozialhilfegesetzes` (NRM)
- `BSHG` (NRM)
- `B.` (LOC)

**Example 3** (doc_id: `61486`) (sent_id: `61486`)


Das Gericht wies am dritten Hauptverhandlungstag im Zusammenhang mit einem Antrag von Rechtsanwalt P. , den dieser unter Bezugnahme auf das zuvor genannte Schreiben begründet hatte , unter anderem darauf hin , dass sich in der Akte ein „ Terminverlegungsantrag vom 12. April 2016 “ befinde .

| Predicted | Gold |
|---|---|
| `P.` | `P.` |

**Example 4** (doc_id: `61657`) (sent_id: `61657`)


Dort ist im Einzelnen dargelegt , dass die Anlagen K 1. K 3 , K 5 , K 9 und K 50 jeweils Hinweise darauf enthalten , dass S. die Verhandlungen für die Help Food und nicht für die Beklagte führte .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Missed by this rule (FN):**

- `Help Food` (ORG)

**Example 5** (doc_id: `61871`) (sent_id: `61871`)


Als die Geschädigte während dieses Geschehens von der Zeugin K. angerufen wurde , riss M. der Geschädigten das Mobiltelefon aus der Hand und nahm es im Einverständnis mit dem Angeklagten R. an sich , um zu verhindern , dass die Geschädigte um Hilfe rief .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `M.` (PER)
- `R.` (PER)

**Example 6** (doc_id: `62034`) (sent_id: `62034`)


Die entsprechenden Feststellungen wird das LSG allerdings nur dann nachzuholen haben , wenn K. nicht ohnedies während ihrer Teilnahme am Modellprojekt Enthospitalisierung in Wi. und damit im Zuständigkeitsbereich des Klägers , ihren letzten gewöhnlichen Aufenthalt vor Aufnahme in die Außenwohngruppe im Jahr 2005 begründet hat .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `Wi.` (ORG)

**Example 7** (doc_id: `62385`) (sent_id: `62385`)


Der Nebenkläger war nämlich durch M. hinreichend geschützt .

| Predicted | Gold |
|---|---|
| `M.` | `M.` |

**Example 8** (doc_id: `62485`) (sent_id: `62485`)


Die Kammer sei auch nicht in der Lage , Spruchreife herzustellen , weil dazu Erbscheine der Erbeserben des S. erforderlich seien .

| Predicted | Gold |
|---|---|
| `S.` | `S.` |

**Example 9** (doc_id: `62987`) (sent_id: `62987`)


Zugleich hat der Kläger die Richtigkeit des gesamten bisherigen Vorbringens des Beklagten zum tatsächlichen Verwaltungsaufwand und zur Minutenberechnung erneut ausdrücklich bestritten und eine vom Beklagten als Anlage zu einem Vermerk vom 22. Juli 2013 vorgelegte " Zeiterfassung bei der Bearbeitung repräsentativer Fälle durch Frau B. " , aus der sich angeblich eine mittlere Bearbeitungszeit von ca. 27,25 bis 27,625 Minuten ergebe , als nicht nachvollziehbar bezeichnet .

| Predicted | Gold |
|---|---|
| `B.` | `B.` |

**Example 10** (doc_id: `63898`) (sent_id: `63898`)


Dem Kläger dürfte dies zumindest außerhalb von P. auch ohne Freunde oder Verwandte möglich sein , zumal nicht alle Vermieter nur an ethnische Russen vermieten .

| Predicted | Gold |
|---|---|
| `P.` | `P.` |

**Example 11** (doc_id: `64271`) (sent_id: `64271`)


Der Antrag des Klägers , ihm für das Verfahren der Beschwerde gegen die Nichtzulassung der Revision im Urteil des Landessozialgerichts Niedersachsen-Bremen vom 16. November 2017 Prozesskostenhilfe zu bewilligen und Rechtsanwältin K. aus H. beizuordnen , wird abgelehnt .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Missed by this rule (FN):**

- `Landessozialgerichts Niedersachsen-Bremen` (ORG)
- `H.` (LOC)

**Example 12** (doc_id: `65166`) (sent_id: `65166`)


Am 10. Oktober 2016 fuhren A. , F. und Z. gemeinsam nach F. , der Angeklagte brachte mit einem Mietwagen drei Fahrräder nach Deutschland .

| Predicted | Gold |
|---|---|
| `F.` | `F.` |

**Missed by this rule (FN):**

- `A.` (PER)
- `Z.` (PER)
- `Deutschland` (LOC)

**Example 13** (doc_id: `65282`) (sent_id: `65282`)


Die Taten fanden ein Ende , nachdem die Zeugin R. im Sexualkundeunterricht aufgeklärt worden war .

| Predicted | Gold |
|---|---|
| `R.` | `R.` |

**Example 14** (doc_id: `65905`) (sent_id: `65905`)


2. Im Zuge eines von J. betriebenen Verfahrens der einstweiligen Verfügung verurteilten das Landgericht F. ( P. ) und letztinstanzlich das Pfälzische Oberlandesgericht Zweibrücken die Beschwerdeführerin antragsgemäß zum Abdruck der folgenden Gegendarstellung , wobei die Größe des Wortes " Gegendarstellung " der Größe der Schrift der Worte " Sterbedrama um seinen besten Freund " und der Text der Gegendarstellung im Übrigen der Schriftgröße der Zeile " Hätte er ihn damals retten können ? " zu entsprechen hatten :

| Predicted | Gold |
|---|---|
| `J.` | `J.` |

**Missed by this rule (FN):**

- `Landgericht F. ( P. )` (ORG)
- `Pfälzische Oberlandesgericht Zweibrücken` (ORG)

**Example 15** (doc_id: `66575`) (sent_id: `66575`)


Die vorliegenden Eingangsrechnungen der Gaststätte lauten teilweise auf den Namen der Gaststätte , teilweise auf die Klägerin und teilweise auf A.

| Predicted | Gold |
|---|---|
| `A.` | `A.` |

**Example 16** (doc_id: `66608`) (sent_id: `66608`)


Nach den getroffenen Feststellungen ist unzweifelhaft , dass der Zeuge K. , der im Fall II. 1. der Urteilsgründe selbst Cannabis vom Angeklagten erhielt und weiterverkaufte , dabei auch in der Vorstellung , den Betäubungsmittelhandel des Angeklagten zu fördern , tätig wurde .

| Predicted | Gold |
|---|---|
| `K.` | `K.` |

**Example 17** (doc_id: `66649`) (sent_id: `66649`)


M. hielt der Zeugin Mund und Nase zu , so dass sie nicht mehr schreien konnte .

| Predicted | Gold |
|---|---|
| `M.` | `M.` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60116`) (sent_id: `60116`)


Die Herstellung der dentalen Restauration erfolgt gemäß Beispiel 26 durch Heißpressen ( vgl. D13 , S. 16 , Bsp. 26 i. V. m. S. 9 , [ 0155 ] bis S. 10 , [ 0162 ] , S. 11/12 Bsp. 6 ) .

**False Positives:**

- `S.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `60183`) (sent_id: `60183`)


Auf der Flucht legten sie an einem zuvor bestimmten Platz am Teich des Kurparks die Rucksäcke mit der Tatbeute ab und fuhren mit dem Zug nach F. .

**False Positives:**

- `F.` — type mismatch — same span as gold: `F.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `F.`(LOC)

**Example 2** (doc_id: `60310`) (sent_id: `60310`)


Läge hingegen ein Fall des ambulant-betreuten Wohnens vor , hätte K. in Wi. , dem Ort , an dem die Wohngemeinschaft belegen war , ihren letzten gewöhnlichen Aufenthalt vor der Wiederaufnahme in das A. -Zentrum im November 1994 begründet ; § 109 SGB XII bzw § 109 BSHG stehen nur bei einem stationären Aufenthalt der Begründung eines gewöhnlichen Aufenthalts am Anstalts- bzw Einrichtungsort entgegen .

**False Positives:**

- `A.` — partial — pred is substring of gold: `A. -Zentrum`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K.`(PER)
- `Wi.`(ORG)
- `A. -Zentrum`(ORG)
- `§ 109 SGB XII`(NRM)
- `§ 109 BSHG`(NRM)

**Example 3** (doc_id: `60400`) (sent_id: `60400`)


Die disziplinarische Ahndung des Verhaltens des Beschwerdeführers zu I. sowie der Beschwerdeführerinnen zu II. bis IV. durch Verfügungen ihrer Dienstherren und deren disziplinargerichtliche Bestätigung durch die angegriffenen Gerichtsentscheidungen begrenzen die Möglichkeit zur Teilnahme an einem Arbeitskampf .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `60404`) (sent_id: `60404`)


3. Wegen dieser Berichterstattung betrieben der Kläger , die P. AG und die H. AG jeweils Unterlassungsverfahren gegen die Beschwerdeführerin ; im Fall des Klägers verbunden mit einer Klage auf Richtigstellung .

**False Positives:**

- `P.` — partial — pred is substring of gold: `P. AG`
- `H.` — partial — pred is substring of gold: `H. AG`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `P. AG`(ORG)
- `H. AG`(ORG)

**Example 5** (doc_id: `60442`) (sent_id: `60442`)


Den unmittelbar Geschädigten W. und M. L. wurde für die Wegnahme ihres landwirtschaftlichen Vermögens in P. eine Hauptentschädigung zuerkannt .

**False Positives:**

- `P.` — type mismatch — same span as gold: `P.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `W. und M. L.`(PER)
- `P.`(LOC)

**Example 6** (doc_id: `60654`) (sent_id: `60654`)


Die Fachgerichte sind jedoch durch Art. 100 Abs. 1 GG nicht gehindert , schon vor der im Hauptsacheverfahren einzuholenden Entscheidung des BVerfG auf der Grundlage ihrer Rechtsauffassung vorläufigen Rechtsschutz zu gewähren , wenn dies im Interesse eines effektiven Rechtsschutzes geboten erscheint und die Hauptsacheentscheidung dadurch nicht vorweggenommen wird ( vgl. BVerfG-Beschluss vom 24. Juni 1992 1 BvR 1028/91 , BVerfGE 86 , 382 , unter B. II. 2. b ; BFH-Beschluss in BFHE 204 , 39 , BStBl II 2004 , 367 ) .

**False Positives:**

- `B.` — partial — pred is substring of gold: `BVerfG-Beschluss vom 24. Juni 1992 1 BvR 1028/91 , BVerfGE 86 , 382 , unter B. II. 2. b`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 100 Abs. 1 GG`(NRM)
- `BVerfG`(ORG)
- `BVerfG-Beschluss vom 24. Juni 1992 1 BvR 1028/91 , BVerfGE 86 , 382 , unter B. II. 2. b`(RS)
- `BFH-Beschluss in BFHE 204 , 39 , BStBl II 2004 , 367`(RS)

**Example 7** (doc_id: `60666`) (sent_id: `60666`)


Nach der vorliegenden Erkenntnislage war es dem Kläger bei Abschiebung grundsätzlich möglich und zumutbar , in der Russischen Föderation etwa in der weiteren , ländlicheren Umgebung von P. legal Wohnsitz zu nehmen und insbesondere registriert zu werden .

**False Positives:**

- `P.` — type mismatch — same span as gold: `P.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Russischen Föderation`(LOC)
- `P.`(LOC)

**Example 8** (doc_id: `60742`) (sent_id: `60742`)


Von einem Bedienhebel nach M4 und Teilmerkmal M5 ist auf den S. 244 - 253 des Fachtagungsbuches keine Rede , denn die Fig. 5 zeigt nur symbolische Darstellungen für die Funksteuerung , das Bedienpult oder das Laptop und lässt allenfalls den Schluss auf Tasten zu , was auch in Übereinstimmung mit den Ausführungen zur Drehzahlregelung über zwei Taster ( Rechts / Links ) steht ( vgl. EI ( D1 ) : S. 249 : „ Funktion der SPCD “ ) .

**False Positives:**

- `S.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 9** (doc_id: `60796`) (sent_id: `60796`)


I. 1. Gegen den Beschwerdeführer wurde bei der Staatsanwaltschaft München I ein Ermittlungsverfahren wegen Betruges geführt .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Staatsanwaltschaft München I`(ORG)

**Example 10** (doc_id: `60890`) (sent_id: `60890`)


I. 1. Das Streitpatent betrifft die Bereitstellung einer den hochselektiven PDE5 -Inhibitor Tadalafil enthaltenden Einheitsdosiszusammensetzung für die Behandlung sexueller Dysfunktion ( vgl. NIK1.3 / NiK1 S. 2 Abs. [ 0002 ] sowie Patentansprüche 1 und 10 ) .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 11** (doc_id: `60942`) (sent_id: `60942`)


Soweit sich gleichwohl aufgrund eines späteren gewillkürten Versicherungsbeginns Nachteile im Versicherungsschutz Betroffener realisieren können , etwa weil infolge der Nichtberücksichtigung von Versicherungszeiten möglicherweise die Voraussetzungen für einen Rentenanspruch nicht erfüllt sind , sollte der spätere Eintritt der Versicherungspflicht außerdem nach § 7a Abs 6 S 1 Nr 1 SGB IV von der Zustimmung des Beschäftigten abhängig gemacht werden ( vgl dazu näher Beschlussempfehlung und Bericht des Ausschusses für Arbeit und Sozialordnung < 11. Ausschuss > , BT-Drucks 14/2046 S 1 unter A. , S 2 unter B. , S 5 unter II. , S 10 < BDA , DAG > und S 13 < Koalitionsfraktionen > ) .

**False Positives:**

- `A.` — partial — pred is substring of gold: `Beschlussempfehlung und Bericht des Ausschusses für Arbeit und Sozialordnung < 11. Ausschuss > , BT-Drucks 14/2046 S 1 unter A. , S 2 unter B. , S 5 unter II. , S 10 < BDA , DAG > und S 13 < Koalitionsfraktionen >`
- `B.` — partial — pred is substring of gold: `Beschlussempfehlung und Bericht des Ausschusses für Arbeit und Sozialordnung < 11. Ausschuss > , BT-Drucks 14/2046 S 1 unter A. , S 2 unter B. , S 5 unter II. , S 10 < BDA , DAG > und S 13 < Koalitionsfraktionen >`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 7a Abs 6 S 1 Nr 1 SGB IV`(NRM)
- `Beschlussempfehlung und Bericht des Ausschusses für Arbeit und Sozialordnung < 11. Ausschuss > , BT-Drucks 14/2046 S 1 unter A. , S 2 unter B. , S 5 unter II. , S 10 < BDA , DAG > und S 13 < Koalitionsfraktionen >`(LIT)

**Example 12** (doc_id: `61021`) (sent_id: `61021`)


Dass der Kläger nach dem Gutachten des Dr. K. noch nicht wie ein Erwachsener wirkt und ihm nach Beobachtungen von Pflegern in der B. er Klinik " jegliche Alltagspraxis " fehle , rechtfertigte bei seiner Abschiebung keine andere Prognose .

**False Positives:**

- `B.` — partial — pred is substring of gold: `B. er Klinik`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K.`(PER)
- `B. er Klinik`(ORG)

**Example 13** (doc_id: `61069`) (sent_id: `61069`)


Nach Zurückverweisung hat das LSG Dr. K. , Institut für neurologisch psychiatrische Begutachtung in B. , mit der Erstellung eines neurologisch-psychiatrischen Gutachtens nach ambulanter Untersuchung des Klägers beauftragt .

**False Positives:**

- `B.` — partial — pred is substring of gold: `Institut für neurologisch psychiatrische Begutachtung in B.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K.`(PER)
- `Institut für neurologisch psychiatrische Begutachtung in B.`(ORG)

**Example 14** (doc_id: `61101`) (sent_id: `61101`)


X.

**False Positives:**

- `X.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 15** (doc_id: `61357`) (sent_id: `61357`)


Mit beim Anwaltsgerichtshof am 6. Oktober 2017 eingegangenem Schreiben vom 5. Oktober 2017 bat der Kläger erneut um Übersendung der Verwaltungsakte an sein Büro in F. .

**False Positives:**

- `F.` — type mismatch — same span as gold: `F.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `F.`(LOC)

**Example 16** (doc_id: `61613`) (sent_id: `61613`)


2. Aufgrund dieses Rauschgiftfunds beantragte Kriminaloberkommissarin Dö. unter Einbindung des zuständigen Staatsanwalts bei dem Ermittlungsrichter des Amtsgerichts Offenbach am Main den Erlass eines Durchsuchungsbeschlusses für die Wohnung des Angeklagten in F. .

**False Positives:**

- `F.` — type mismatch — same span as gold: `F.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Dö.`(PER)
- `Amtsgerichts Offenbach am Main`(ORG)
- `F.`(LOC)

**Example 17** (doc_id: `61961`) (sent_id: `61961`)


I. 1. Nach den Feststellungen des Landgerichts gab der zur Tatzeit 22 Jahre alte Angeklagte , der zuvor Alkohol und Marihuana konsumiert hatte , am späten Abend des 19. November 2015 von einer Telefonzelle aus bei einem Pizza-Lieferservice unter falschem Namen und Angabe einer nicht auf ihn zugelassenen Rufnummer eine Bestellung auf .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 18** (doc_id: `62076`) (sent_id: `62076`)


Am 8. April 2012 fuhr er in B. unter Einfluss von Marihuana mit dem Auto .

**False Positives:**

- `B.` — type mismatch — same span as gold: `B.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `B.`(LOC)

**Example 19** (doc_id: `62082`) (sent_id: `62082`)


Der Ventileinsatz ( valve body 20 ) weist ein Ventil zum Öffnen und Schließen der Saug- und Spülkanäle auf ( vgl. S. 11 Z. 19 bis S. 12 Z. 2 : „ … The transversal through-going bore 48 and the branching-off bore 50 serve the purpose of establishing direct connection between the through-going bore of the tube 12 and the through-going holes of one of the tubular fittings 26 and 30 in a specific activation position . … ” ) [ = Merkmal M6 ] .

**False Positives:**

- `S.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 20** (doc_id: `62169`) (sent_id: `62169`)


Ab August 2002 absolvierte er an einer Berufsfachschule für Sozialassistenz mit Schwerpunkt Sozialpädagogik in H. eine auf zwei Jahre angelegte Ausbildung zum staatlich geprüften Sozialassistenten , die er krankheitsbedingt erst im Juli 2005 abschloss .

**False Positives:**

- `H.` — type mismatch — same span as gold: `H.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `H.`(LOC)

**Example 21** (doc_id: `62343`) (sent_id: `62343`)


I. § 7 Abs. 1 Satz 2 TV AKS 2012 verweist auf § 1 TV AKS 2012 .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 7 Abs. 1 Satz 2 TV AKS 2012`(REG)
- `§ 1 TV AKS 2012`(REG)

**Example 22** (doc_id: `62483`) (sent_id: `62483`)


Gegen 14.45 Uhr rief dieser den Angeklagten an und zitierte ihn zu seinem Garten in M. bei O. , wo der Angeklagte um 15.35 Uhr eintraf .

**False Positives:**

- `M.` — type mismatch — same span as gold: `M.`
- `O.` — type mismatch — same span as gold: `O.`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `M.`(LOC)
- `O.`(LOC)

**Example 23** (doc_id: `62931`) (sent_id: `62931`)


Deshalb können nur solche Aufwendungen als Werbungskosten i. S. des § 9 Abs. 1 EStG abgezogen werden , welche die persönliche Leistungsfähigkeit des Steuerpflichtigen mindern ( ständige Rechtsprechung , z.B. Beschluss des Großen Senats des BFH in BFHE 189 , 160 , BStBl II 1999 , 782 , unter C. IV. 1. b ; BFH-Urteil vom 15. November 2005 IX R 25/03 , BFHE 211 , 318 , BStBl II 2006 , 623 , m. w. N. ; Senatsurteil vom 13. März 1996 VI R 103/95 , BFHE 180 , 139 , BStBl II 1996 , 375 ) .

**False Positives:**

- `C.` — partial — pred is substring of gold: `Beschluss des Großen Senats des BFH in BFHE 189 , 160 , BStBl II 1999 , 782 , unter C. IV. 1. b`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 9 Abs. 1 EStG`(NRM)
- `Beschluss des Großen Senats des BFH in BFHE 189 , 160 , BStBl II 1999 , 782 , unter C. IV. 1. b`(RS)
- `BFH-Urteil vom 15. November 2005 IX R 25/03 , BFHE 211 , 318 , BStBl II 2006 , 623`(RS)
- `Senatsurteil vom 13. März 1996 VI R 103/95 , BFHE 180 , 139 , BStBl II 1996 , 375`(RS)

**Example 24** (doc_id: `62959`) (sent_id: `62959`)


Das FG hat auf S. 9 des Urteils in plausibler Weise begründet , dass in der Differenz zwischen Batterieladung und Batterieentladung keine unternehmerische Nutzung zu sehen ist , weil es sich insoweit nicht um gespeicherten Strom handele , sondern um während des Speichervorgangs entstehende Energieverluste , die für eine ( unternehmerische oder nichtunternehmerische ) Nutzung nicht zur Verfügung stehen .

**False Positives:**

- `S.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 25** (doc_id: `62980`) (sent_id: `62980`)


Hinzu komme , dass die Klägerin gegenüber der Rechtsanwaltskammer erklärt habe , eine gutgehende Anwaltskanzlei in P. zu führen .

**False Positives:**

- `P.` — type mismatch — same span as gold: `P.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `P.`(LOC)

**Example 26** (doc_id: `63011`) (sent_id: `63011`)


I. 1. Die Beschwerdeführerin , die Verwaltungs-GmbH einer nicht rechtsfähigen Stiftung , wendet sich gegen den am 6. Dezember 2013 ( BGBl I S. 1386 ) in Kraft getretenen § 6a Bundesjagdgesetz ( BJagdG ) .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BGBl I S. 1386`(LIT)
- `§ 6a Bundesjagdgesetz`(NRM)
- `BJagdG`(NRM)

**Example 27** (doc_id: `63073`) (sent_id: `63073`)


Als Anschlagsort hatte M. das E. in Q. ins Auge gefasst , da es " [ d ] er meist besuchteste Ort Europas " sei und sich dort viele " kuffar " aufhielten .

**False Positives:**

- `E.` — type mismatch — same span as gold: `E.`
- `Q.` — type mismatch — same span as gold: `Q.`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `M.`(PER)
- `E.`(LOC)
- `Q.`(LOC)
- `Europas`(LOC)

**Example 28** (doc_id: `63198`) (sent_id: `63198`)


Unbegründet erweist sich die Rechtsbeschwerde hingegen insoweit , als der Antragsteller die Gewährung dieser Vergütung ohne Anrechnung der Wegstrecke von seinem Wohnsitz - der Wohnung - zu dem bisherigen Dienstort in S. erstrebt ( 2. ) .

**False Positives:**

- `S.` — type mismatch — same span as gold: `S.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S.`(LOC)

**Example 29** (doc_id: `63253`) (sent_id: `63253`)


Sowohl die Pflichtmitgliedschaft in der berufsständischen Kammer als auch ( in der Folge ) die Mitgliedschaft in der Versorgungsanstalt stehen nach den einschlägigen landesrechtlichen Vorschriften nicht zur Disposition des Betroffenen ( vgl dazu bereits die Ausführungen unter I. 1. ) .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 30** (doc_id: `63335`) (sent_id: `63335`)


Soweit der Antragsteller die Dringlichkeit der Dienstbesetzung in Frage stellt , kann er damit nicht durchdringen , weil dem Dienstherrn insoweit ein im Wesentlichen von militärischen Zweckmäßigkeitserwägungen geprägter Einschätzungsvorrang hinsichtlich der Erforderlichkeit und Priorität von Personalmaßnahmen zukommt ; dessen Grenzen sind mit den Erwägungen des Leiters des ... , die der Versetzung des Antragstellers nach C. zugrunde liegen , nicht überschritten .

**False Positives:**

- `C.` — type mismatch — same span as gold: `C.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `...`(ORG)
- `C.`(LOC)

**Example 31** (doc_id: `64039`) (sent_id: `64039`)


II. 1. a ) Der 1951 geborene Beschwerdeführer zu I. wurde im Jahr 1981 zum Beamten auf Lebenszeit ernannt und war als Lehrer im Schuldienst des Landes Niedersachsen tätig .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Niedersachsen`(LOC)

**Example 32** (doc_id: `64124`) (sent_id: `64124`)


" Anfang 2009 habe ihn ein P. -Mitarbeiter [ Mitarbeiter der P. AG ] gebeten , spätabends zum Seiteneingang der H. -Zentrale in der H. Innenstadt zu kommen , um einen heiklen Spezialauftrag auszuführen .

**False Positives:**

- `P.` — similar text (different position): `P. AG`
- `P.` — partial — pred is substring of gold: `P. AG`
- `H.` — partial — pred is substring of gold: `H. -Zentrale`

> overlaps gold: 3  |  likely missing annotation: 0

**Gold Entities:**

- `P. AG`(ORG)
- `H. -Zentrale`(ORG)
- `H. Innenstadt`(LOC)

**Example 33** (doc_id: `64174`) (sent_id: `64174`)


Der gerichtliche Sachverständige Dr. von M. habe in der erstinstanzlichen mündlichen Verhandlung mitgeteilt , dass ihm keine Daten zur Marktentwicklung in Delmenhorst vorlägen .

**False Positives:**

- `M.` — partial — pred is substring of gold: `von M.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `von M.`(PER)
- `Delmenhorst`(LOC)

**Example 34** (doc_id: `64306`) (sent_id: `64306`)


D. h. , nur unter Anwendung der bisherigen Verfahrensweise kann eine Anpassung im Umfang der Ruhegeldempfänger sichergestellt werden .

**False Positives:**

- `D.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 35** (doc_id: `64704`) (sent_id: `64704`)


I. Der Zulässigkeit der Verfassungsbeschwerden in den Verfahren 2 BvR 1738/12 und 2 BvR 1068/14 steht nicht entgegen , dass die Beschwerdeführerin zu III. bereits während des fachgerichtlichen Verfahrens und damit vor Erhebung der Verfassungsbeschwerde auf eigenen Wunsch aus dem Beamtenverhältnis ausgeschieden ist und der Beschwerdeführer zu I. während des Verfassungsbeschwerdeverfahrens die Altersgrenze des § 35 Abs. 1 Satz 2 , Abs. 2 des Niedersächsischen Beamtengesetzes erreicht hat und in den Ruhestand getreten ist .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Verfahren 2 BvR 1738/12 und 2 BvR 1068/14`(RS)
- `§ 35 Abs. 1 Satz 2 , Abs. 2 des Niedersächsischen Beamtengesetzes`(NRM)

**Example 36** (doc_id: `64962`) (sent_id: `64962`)


Die Klägerin ist Inhaberin eines Hostels in der K. straße , K. , mit nach ihren Angaben 40 beitragspflichtigen Gästezimmern ohne sozialversicherungspflichtige Mitarbeiter .

**False Positives:**

- `K.` — partial — pred is substring of gold: `K. straße`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K. straße`(LOC)
- `K.`(LOC)

**Example 37** (doc_id: `65104`) (sent_id: `65104`)


Ob im Hinblick auf die weitergehenden Ausführungen auf S. 19 f. der Begründung eine Auslegung im o. a. Sinn geboten ist , kann letztlich dahinstehen , da für die Prüfung des Senats nur die Rechtslage zum Zeitpunkt des angefochtenen Verwaltungsakts maßgeblich ist .

**False Positives:**

- `S.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 38** (doc_id: `65172`) (sent_id: `65172`)


Er ging davon aus , dass für die Zurechnung nach den §§ 240 , 242 HGB nichts anderes gelte als für die nach § 39 AO ( BFH-Urteil in BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a ) .

**False Positives:**

- `I.` — partial — pred is substring of gold: `BFH-Urteil in BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§§ 240 , 242 HGB`(NRM)
- `§ 39 AO`(NRM)
- `BFH-Urteil in BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a`(RS)

**Example 39** (doc_id: `65234`) (sent_id: `65234`)


Die Beigeladene zu 7. nahm im Oktober 1990 eine Beschäftigung als Rechtsanwältin bei einem Rechtsanwalt in L. auf .

**False Positives:**

- `L.` — type mismatch — same span as gold: `L.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `L.`(LOC)

**Example 40** (doc_id: `65351`) (sent_id: `65351`)


Ernstliche Zweifel können auch verfassungsrechtliche Zweifel an der Gültigkeit einer dem angefochtenen Verwaltungsakt zugrunde liegenden Norm sein ( ständige Rechtsprechung , z.B. BVerfG-Urteil vom 21. Februar 1961 1 BvR 314/60 , BVerfGE 12 , 180 , BStBl I 1961 , 63 , unter B. II. ; BFH-Beschlüsse vom 5. März 2001 IX B 90/00 , BFHE 195 , 205 , BStBl II 2001 , 405 ; vom 22. Dezember 2003 IX B 177/02 , BFHE 204 , 39 , BStBl II 2004 , 367 ) .

**False Positives:**

- `B.` — partial — pred is substring of gold: `BVerfG-Urteil vom 21. Februar 1961 1 BvR 314/60 , BVerfGE 12 , 180 , BStBl I 1961 , 63 , unter B. II.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG-Urteil vom 21. Februar 1961 1 BvR 314/60 , BVerfGE 12 , 180 , BStBl I 1961 , 63 , unter B. II.`(RS)
- `BFH-Beschlüsse vom 5. März 2001 IX B 90/00 , BFHE 195 , 205 , BStBl II 2001 , 405`(RS)
- `vom 22. Dezember 2003 IX B 177/02 , BFHE 204 , 39 , BStBl II 2004 , 367`(RS)

**Example 41** (doc_id: `66023`) (sent_id: `66023`)


I. 1. Nach bisheriger Rechtsprechung des Bundesgerichtshofs wurde es überwiegend als ein Verstoß gegen das in § 46 Abs. 3 StGB verankerte Verbot der Doppelverwertung von Tatbestandsmerkmalen und damit als rechtsfehlerhaft angesehen , wenn der Tatrichter das subjektive Tatbestandsmerkmal direkten Tötungsvorsatzes strafschärfend berücksichtigt ( vgl. BGH , Beschluss vom 11. März 2015 - 1 StR 3/15 , NStZ-RR 2015 , 171 ( Ls. ) ; Senat , Beschlüsse vom 25. Juni 2015 - 2 StR 83/15 , BGHR StGB § 46 Abs. 3 Tötungsvorsatz 7 , vom 21. Januar 2004 - 2 StR 449/03 , vom 23. Oktober 1992 - 2 StR 483/92 , StV 1993 , 72 und vom 1. Dezember 1989 - 2 StR 555/89 , BGHR StGB § 46 Abs. 3 Tötungsvorsatz 3 ; BGH , Beschlüsse vom 5. Oktober 1977 - 3 StR 369/77 , vom 8. Februar 1978 - 3 StR 425/77 und vom 13. Mai 1981 - 3 StR 126/81 , NJW 1981 , 2204 ; BGH , Urteil vom 28. Juni 1968 - 4 StR 226/68 ; Beschlüsse vom 16. September 1986 - 4 StR 457/86 , BGHR StGB § 46 Abs. 3 Tötungsvorsatz 1 , vom 26. April 1988 - 4 StR 157/88 , NStE Nr. 41 zu § 46 StGB , vom 30. Juli 1998 - 4 StR 346/98 , NStZ 1999 , 23 , vom 3. Februar 2004 - 4 StR 403/03 und vom 14. Oktober 2015 - 5 StR 355/15 , NStZ-RR 2016 , 8 ) .

**False Positives:**

- `I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

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

**Example 42** (doc_id: `66057`) (sent_id: `66057`)


Von September 2007 bis September 2012 arbeitete der Kläger als Erzieher in einer Kindertagesstätte in D.

**False Positives:**

- `D.` — type mismatch — same span as gold: `D.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `D.`(LOC)

**Example 43** (doc_id: `66433`) (sent_id: `66433`)


Nachdem der Europäische Gerichtshof für Menschenrechte ( EGMR ) eine auf Antrag des Klägers am 31. Juli 2017 erlassene vorläufige Untersagung der Abschiebung am 29. August 2017 wieder aufgehoben hatte , wurde der Kläger am 4. September 2017 nach P. ( Russische Föderation ) abgeschoben .

**False Positives:**

- `P.` — type mismatch — same span as gold: `P.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Europäische Gerichtshof für Menschenrechte`(ORG)
- `EGMR`(ORG)
- `P.`(LOC)
- `Russische Föderation`(LOC)

**Example 44** (doc_id: `66706`) (sent_id: `66706`)


In den unter IV. der Urteilsgründe zusammengefassten zwei Fällen ( Beiseiteschaffen von Fahrzeugen und Gerätschaften aus dem Vermögen der M. GmbH und aus dem Vermögen des nicht revidierenden Mitangeklagten Ma. ) hat das Landgericht den Angeklagten jeweils wegen Beihilfe zum Bankrott verurteilt und den Strafrahmen des § 283 Abs. 1 StGB jeweils gemäß § 27 Abs. 2 , § 49 Abs. 1 StGB gemildert .

**False Positives:**

- `M.` — partial — pred is substring of gold: `M. GmbH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `M. GmbH`(ORG)
- `Ma.`(PER)
- `§ 283 Abs. 1 StGB`(NRM)
- `§ 27 Abs. 2 , § 49 Abs. 1 StGB`(NRM)

</details>

---

## `Full names with initials (e.g., K. Schmidt)`

**F1:** 0.054 | **Precision:** 0.098 | **Recall:** 0.037  

**Format:** `regex`  
**Rule ID:** `266c1518`  
**Description:**
Captures full names consisting of an initial and a surname (e.g., 'K. Schmidt', 'M. Rennpferdt').

**Content:**
```
\b([A-Z]\.)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)\b
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

- `I. Die Klägerin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `60117`) (sent_id: `60117`)


I. Die Befristungskontrollklage ist unbegründet .

**False Positives:**

- `I. Die Befristungskontrollklage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `60238`) (sent_id: `60238`)


V. Die Klage ist nicht abweisungsreif ( vgl. § 563 Abs. 3 ZPO ) .

**False Positives:**

- `V. Die Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 563 Abs. 3 ZPO`(NRM)

**Example 3** (doc_id: `60477`) (sent_id: `60477`)


I. Die Würdigung des Landesarbeitsgerichts , das beklagte Königreich sei im vorliegenden Rechtsstreit grundsätzlich nicht der deutschen Gerichtsbarkeit unterworfen , sondern genieße - sollte es darauf nicht verzichtet haben - Staatenimmunität , ist revisionsrechtlich nicht zu beanstanden .

**False Positives:**

- `I. Die Würdigung` — no gold match — likely missing annotation

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

- `I. Die Antragsgegnerin` — no gold match — likely missing annotation

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

- `I. Die Kläger` — no gold match — likely missing annotation

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

- `B. Not` — no gold match — likely missing annotation

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

- `I. Der Feststellungsantrag` — no gold match — likely missing annotation

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

- `I. Die Anmelderin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschen Patent- und Markenamt`(ORG)
- `A-ÖFFNER`(ORG)

**Example 18** (doc_id: `61516`) (sent_id: `61516`)


I. Der Kläger und Revisionskläger ( Kläger ) war in den Streitjahren ( 1995 bis 1997 ) u. a. als Steuerberater in einer Einzelkanzlei tätig .

**False Positives:**

- `I. Der Kläger` — no gold match — likely missing annotation

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

- `I. Die Bezeichnung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `MAM Munich Asset Management`(ORG)
- `Deutschen Patent- und Markenamt`(ORG)
- `DPMA`(ORG)

**Example 25** (doc_id: `61932`) (sent_id: `61932`)


V. Die Kostenentscheidung beruht auf § 90 Satz 2 EnWG , die Festsetzung des Gegenstandswerts auf § 50 Abs. 1 Satz 1 Nr. 2 GKG und § 3 ZPO .

**False Positives:**

- `V. Die Kostenentscheidung` — no gold match — likely missing annotation

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

- `A. Die Richtervorlage` — no gold match — likely missing annotation

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

- `I. Der Kläger` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `X`(LOC)
- `Y-Straße ...`(LOC)
- `A`(PER)

**Example 30** (doc_id: `62232`) (sent_id: `62232`)


D. Der Kläger hat gem. § 97 Abs. 1 ZPO die Kosten seiner erfolglosen Revision zu tragen .

**False Positives:**

- `D. Der Kläger` — no gold match — likely missing annotation

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

- `I. Der Kläger` — no gold match — likely missing annotation

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

- `I. Die Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 41** (doc_id: `62878`) (sent_id: `62878`)


I. Die Verfassungsbeschwerde betrifft eine Entscheidung über den von der Beschwerdeführerin geltend gemachten Anspruch auf Zugewinnausgleich .

**False Positives:**

- `I. Die Verfassungsbeschwerde` — no gold match — likely missing annotation

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

- `I. Die Ablehnungsgesuche` — no gold match — likely missing annotation

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

- `C. Das Landesarbeitsgericht` — no gold match — likely missing annotation

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

- `I. Das Landesarbeitsgericht` — no gold match — likely missing annotation

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

- `I. Die Beschwerde` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 55** (doc_id: `63750`) (sent_id: `63750`)


I. Die Klage ist zulässig , insbesondere hinreichend bestimmt iSv. § 253 Abs. 2 Nr. 2 ZPO .

**False Positives:**

- `I. Die Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 253 Abs. 2 Nr. 2 ZPO`(NRM)

**Example 56** (doc_id: `63877`) (sent_id: `63877`)


I. Mit Beschluss vom 25. September 2017 X B 79/17 hatte der Senat zum einen eine Beschwerde des Kostenschuldners , Erinnerungsführers und Rügeführers ( Rügeführer ) gegen die Verwerfung einer Anhörungsrüge durch das Finanzgericht als unzulässig seinerseits als unzulässig verworfen , zum anderen eine Beschwerde gegen die Ablehnung eines Antrags auf Akteneinsicht als unbegründet zurückgewiesen und die Kosten des Beschwerdeverfahrens dem Rügeführer auferlegt .

**False Positives:**

- `I. Mit Beschluss` — positional overlap with gold: `Beschluss vom 25. September 2017 X B 79/17`

> overlaps gold: 1  |  likely missing annotation: 0

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

- `I. Die Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 67** (doc_id: `64704`) (sent_id: `64704`)


I. Der Zulässigkeit der Verfassungsbeschwerden in den Verfahren 2 BvR 1738/12 und 2 BvR 1068/14 steht nicht entgegen , dass die Beschwerdeführerin zu III. bereits während des fachgerichtlichen Verfahrens und damit vor Erhebung der Verfassungsbeschwerde auf eigenen Wunsch aus dem Beamtenverhältnis ausgeschieden ist und der Beschwerdeführer zu I. während des Verfassungsbeschwerdeverfahrens die Altersgrenze des § 35 Abs. 1 Satz 2 , Abs. 2 des Niedersächsischen Beamtengesetzes erreicht hat und in den Ruhestand getreten ist .

**False Positives:**

- `I. Der Zulässigkeit` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Verfahren 2 BvR 1738/12 und 2 BvR 1068/14`(RS)
- `§ 35 Abs. 1 Satz 2 , Abs. 2 des Niedersächsischen Beamtengesetzes`(NRM)

**Example 68** (doc_id: `64768`) (sent_id: `64768`)


I. Die Klägerin begehrt die Gewährung einer Rente wegen Erwerbsminderung .

**False Positives:**

- `I. Die Klägerin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 69** (doc_id: `64780`) (sent_id: `64780`)


I. Der Kläger und Beschwerdeführer ( Kläger ) , ein Heilpraktiker und approbierter Psychotherapeut , führte im Rahmen seiner psychotherapeutischen Leistungen in den Streitjahren ( 2010 bis 2012 ) u. a. auch verkehrspsychologische Behandlungen durch .

**False Positives:**

- `I. Der Kläger` — no gold match — likely missing annotation

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

- `I. Die Kläger` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 73** (doc_id: `65021`) (sent_id: `65021`)


B. Die Rechtsbeschwerde der Antragsteller ist begründet .

**False Positives:**

- `B. Die Rechtsbeschwerde` — no gold match — likely missing annotation

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

- `I. Die Revision` — no gold match — likely missing annotation

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

- `C. Im Hinblick` — no gold match — likely missing annotation

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

- `I. Die Beschwerdeführerin` — no gold match — likely missing annotation

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

- `R. Physiotherapie` — positional overlap with gold: `T. R.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `S.`(PER)
- `T. R.`(PER)

**Example 82** (doc_id: `65328`) (sent_id: `65328`)


I. Die Bezeichnung

**False Positives:**

- `I. Die Bezeichnung` — no gold match — likely missing annotation

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

- `I. Die Klägerin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 86** (doc_id: `65416`) (sent_id: `65416`)


I. Die Klägerin , Revisionsklägerin und Revisionsbeklagte ( Klägerin ) unterhielt bis zum Streitjahr 2002 den Regiebetrieb " X " ( im Folgenden : BgA ) , einen Betrieb gewerblicher Art i. S. des § 4 des Körperschaftsteuergesetzes ( KStG ) , für den der Beklagte , Revisionskläger und Revisionsbeklagte ( das Finanzamt - FA - ) gegenüber der Klägerin im Zusammenhang mit der Einbringung des BgA in eine Kapitalgesellschaft für den Anmeldungszeitraum 2002 Kapitalertragsteuer zuzüglich Solidaritätszuschlag festsetzte .

**False Positives:**

- `I. Die Klägerin` — no gold match — likely missing annotation

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

## `Anonymized initials with dots (multi-initial)`

**F1:** 0.041 | **Precision:** 0.333 | **Recall:** 0.022  

**Format:** `regex`  
**Rule ID:** `ae91774f`  
**Description:**
Captures multi-initial anonymized names like 'A. S.' or 'R. C.' as a single entity.

**Content:**
```
(?:[A-Z]\.)\s+(?:[A-Z]\.)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.333 | 0.022 | 0.041 | 21 | 7 | 14 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 7 | 14 | 305 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60448`) (sent_id: `60448`)


Der Kläger nahm Arabisch-Unterricht bei einem T. H. , der der salafistischen Szene zuzurechnen ist und Kontakte zu Personen pflegte , die nach Syrien ausgereist waren oder dies versucht hatten .

| Predicted | Gold |
|---|---|
| `T. H.` | `T. H.` |

**Missed by this rule (FN):**

- `Syrien` (LOC)

**Example 1** (doc_id: `61032`) (sent_id: `61032`)


b ) Der Antragsteller hatte aber bei der Feststellung der beschränkten Dienstfähigkeit der Lehrerin K. B. und der Herabsetzung ihrer Arbeitszeit gemäß § 27 BeamtStG in analoger Anwendung des § 68 Abs. 1 Nr. 6 PersVG BB mitzuwirken .

| Predicted | Gold |
|---|---|
| `K. B.` | `K. B.` |

**Missed by this rule (FN):**

- `§ 27 BeamtStG` (NRM)
- `§ 68 Abs. 1 Nr. 6 PersVG BB` (NRM)

**Example 2** (doc_id: `61243`) (sent_id: `61243`)


Doch auch wenn die Taten 21 bis 23 und 33 der Urteilsgründe sich damit im Hinblick auf das Marihuana als selbständige Umsatzgeschäfte darstellen , fallen die darauf bezogenen Handlungen des Angeklagten mit denjenigen zusammen , die dem Absatz des zugleich in diesen Fällen gehandelten Amphetamins dienten , hinsichtlich dessen aufgrund des einheitlichen Erwerbs im Fall 27 der Urteilsgründe von einer Bewertungseinheit und damit von einer Tat im Rechtssinne auszugehen ist : Im Fall 21 der Urteilsgründe nahm der Angeklagte die Bestellung beider Betäubungsmittel einheitlich entgegen , in den Fällen 22 und 23 der Urteilsgründe lieferte er sie auch gleichzeitig an die Angeklagte A. A. .

| Predicted | Gold |
|---|---|
| `A. A.` | `A. A.` |

**Example 3** (doc_id: `61423`) (sent_id: `61423`)


H. N. ging auf ihn zu , hielt ihm in einer Entfernung von ca. einem Meter ein etwa 22 cm langes Küchenmesser mit ungefähr 11 cm langer Klinge vor die Brust und forderte ihn auf , ihm das auf dem Tresen bzw. in der offenen Kasse liegende Geld zu übergeben .

| Predicted | Gold |
|---|---|
| `H. N.` | `H. N.` |

**Example 4** (doc_id: `62293`) (sent_id: `62293`)


Das Landgericht hat den Angeklagten T. D. wegen schweren sexuellen Missbrauchs eines Kindes in Tateinheit mit sexuellem Missbrauch eines Schutzbefohlenen in zwei Fällen sowie schweren sexuellen Missbrauchs einer widerstandsunfähigen Person in fünf Fällen zu einer Gesamtfreiheitsstrafe von vier Jahren und drei Monaten verurteilt .

| Predicted | Gold |
|---|---|
| `T. D.` | `T. D.` |

**Example 5** (doc_id: `63047`) (sent_id: `63047`)


1. Die Verurteilung des Angeklagten T. D. wegen schweren sexuellen Missbrauchs einer widerstandsunfähigen Person in den Fällen III. 3 bis 7 der Urteilsgründe nach dem zur Tatzeit geltenden § 179 Abs. 1 Nr. 1 , Abs. 5 Nr. 1 StGB in der bis zum 9. November 2016 geltenden Fassung hält revisionsrechtlicher Überprüfung nicht stand , weil die Urteilsgründe eine Widerstandsunfähigkeit des Nebenklägers nicht belegen .

| Predicted | Gold |
|---|---|
| `T. D.` | `T. D.` |

**Missed by this rule (FN):**

- `§ 179 Abs. 1 Nr. 1 , Abs. 5 Nr. 1 StGB` (NRM)

**Example 6** (doc_id: `64492`) (sent_id: `64492`)


Schlussendlich erklärte er : " Ich plane dann mit C. W.

| Predicted | Gold |
|---|---|
| `C. W.` | `C. W.` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60442`) (sent_id: `60442`)


Den unmittelbar Geschädigten W. und M. L. wurde für die Wegnahme ihres landwirtschaftlichen Vermögens in P. eine Hauptentschädigung zuerkannt .

**False Positives:**

- `M. L.` — partial — pred is substring of gold: `W. und M. L.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `W. und M. L.`(PER)
- `P.`(LOC)

**Example 1** (doc_id: `60597`) (sent_id: `60597`)


Aufgrund der Landesverordnung über das Naturschutzgebiet " Liether Kalkgrube " vom 18. 10. 1991 ( GVOBl Schl-H 1992 , S 2 ; in Zukunft : LandesVO L. K. ) seien die Nutzungsmöglichkeiten des Waldes für den Kläger in einem so erheblichen Ausmaß eingeschränkt , dass objektiv keine Bewirtschaftungsmöglichkeit bestehe , die die Vermutung einer forstwirtschaftlichen Tätigkeit rechtfertigen könne ( Urteil vom 27. 6. 2012 ) .

**False Positives:**

- `L. K.` — partial — pred is substring of gold: `Landesverordnung über das Naturschutzgebiet " Liether Kalkgrube " vom 18. 10. 1991 ( GVOBl Schl-H 1992 , S 2 ; in Zukunft : LandesVO L. K. )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Landesverordnung über das Naturschutzgebiet " Liether Kalkgrube " vom 18. 10. 1991 ( GVOBl Schl-H 1992 , S 2 ; in Zukunft : LandesVO L. K. )`(NRM)

**Example 2** (doc_id: `61399`) (sent_id: `61399`)


Zuletzt legt der Kläger einen Auszug aus dem niederländischen Handelsregister vor , aus dem sich die Auflösung der C- B. V. am ... Mai 2006 ergibt .

**False Positives:**

- `B. V.` — partial — pred is substring of gold: `C- B. V.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C- B. V.`(ORG)

**Example 3** (doc_id: `61627`) (sent_id: `61627`)


D15 P. W. McMillan et al. , “ The Structure and Properties of a Lithium Zinc Silicate Glass-Ceramic ” , Journal of Materials Science , 1966 , 1 , Seiten 269 bis 279

**False Positives:**

- `P. W.` — partial — pred is substring of gold: `P. W. McMillan`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `P. W. McMillan`(PER)

**Example 4** (doc_id: `62115`) (sent_id: `62115`)


A. I. 1. Der Beschwerdeführer ist auf Grundlage des Urteils des Landgerichts Lübeck vom 7. Oktober 2014 gemäß § 63 StGB wegen Mordes in einem psychiatrischen Krankenhaus der AMEOS Krankenhausgesellschaft Holstein mbH untergebracht , nachdem er im schuldunfähigen Zustand auf Grund einer wahnhaften Störung im Januar 2014 seine vierjährige Tochter und seinen sechs Jahre alten Sohn getötet hatte .

**False Positives:**

- `A. I.` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Landgerichts Lübeck`(ORG)
- `§ 63 StGB`(NRM)
- `AMEOS Krankenhausgesellschaft Holstein mbH`(ORG)

**Example 5** (doc_id: `63055`) (sent_id: `63055`)


Die Markenstelle verweist zur Begründung der Schutzunfähigkeit des Wortes „ wir “ unter anderem auf die sehr alte BPatG-Entscheidung zur Anmeldung „ W. I. R Zeitarbeit “ ( Beschluss vom 17. 07. 1996 , 29 W ( pat ) 9/95 ) .

**False Positives:**

- `W. I.` — partial — pred is substring of gold: `BPatG-Entscheidung zur Anmeldung „ W. I. R Zeitarbeit “ ( Beschluss vom 17. 07. 1996 , 29 W ( pat ) 9/95 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG-Entscheidung zur Anmeldung „ W. I. R Zeitarbeit “ ( Beschluss vom 17. 07. 1996 , 29 W ( pat ) 9/95 )`(RS)

**Example 6** (doc_id: `63105`) (sent_id: `63105`)


Weiterhin legt er Presseartikel vor , wonach am ... Juni 2003 , am ... August 2003 und am ... März 2004 von verschiedener Stelle aus öffentlich Zweifel an der Bonität der C- B. V. geäußert werden .

**False Positives:**

- `B. V.` — partial — pred is substring of gold: `C- B. V.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C- B. V.`(ORG)

**Example 7** (doc_id: `63990`) (sent_id: `63990`)


Vielmehr müssen die steuerlichen Vorteile der Typisierung im rechten Verhältnis zu der mit der Typisierung notwendig verbundenen Ungleichheit der steuerlichen Belastung stehen ( vgl. z.B. BVerfG-Urteil vom 20. April 2004 1 BvR 1748/99 , 1 BvR 905/00 , BVerfGE 110 , 274 ; BVerfG-Beschluss vom 15. Januar 2008 1 BvL 2/04 , BVerfGE 120 , 1 , unter C. I. 2. a aa ) .

**False Positives:**

- `C. I.` — partial — pred is substring of gold: `BVerfG-Beschluss vom 15. Januar 2008 1 BvL 2/04 , BVerfGE 120 , 1 , unter C. I. 2. a aa`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG-Urteil vom 20. April 2004 1 BvR 1748/99 , 1 BvR 905/00 , BVerfGE 110 , 274`(RS)
- `BVerfG-Beschluss vom 15. Januar 2008 1 BvL 2/04 , BVerfGE 120 , 1 , unter C. I. 2. a aa`(RS)

**Example 8** (doc_id: `64059`) (sent_id: `64059`)


( c ) Ob für die in § 7 Satz 2 Hs. 2 GewStG geschaffene Privilegierung der auf unmittelbar beteiligte natürliche Personen entfallenden Veräußerungsgewinne daneben noch weitere Motive des Gesetzgebers eine Rolle gespielt haben - wie etwa die Schonung des Mittelstandes ( vgl. Bericht der Bundesregierung vom 18. April 2001 a. a. O. unter B. I. 6 ; s. o. A I 3 b aa ( 2 ) ) - ist neben den tragfähigen Überlegungen zur Umgehungsverhinderung nicht erheblich .

**False Positives:**

- `B. I.` — partial — pred is substring of gold: `Bericht der Bundesregierung vom 18. April 2001 a. a. O. unter B. I. 6`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 7 Satz 2 Hs. 2 GewStG`(NRM)
- `Bericht der Bundesregierung vom 18. April 2001 a. a. O. unter B. I. 6`(LIT)

**Example 9** (doc_id: `64559`) (sent_id: `64559`)


aa ) Die unter Beweis gestellte fehlende Werthaltigkeit der Forderung der A-GbR gegen die C- B. V. zum 31. Dezember 2004 ist entscheidungserheblich .

**False Positives:**

- `B. V.` — partial — pred is substring of gold: `C- B. V.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `A-GbR`(ORG)
- `C- B. V.`(ORG)

**Example 10** (doc_id: `65862`) (sent_id: `65862`)


In der mündlichen Verhandlung vor dem FG beantragte der Prozessbevollmächtigte des Klägers , die in einem Schriftsatz zuvor benannten Zeugen zu den dort genannten Beweisthemen zu vernehmen und rügte die Rechtsverletzung des Klägers durch Unterlassen weiterer Sachaufklärung und Zeugenvernehmung , insbesondere zur wirtschaftlichen Situation der C- B. V.

**False Positives:**

- `B. V.` — partial — pred is substring of gold: `C- B. V.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `C- B. V.`(ORG)

**Example 11** (doc_id: `66066`) (sent_id: `66066`)


Hierfür erhält sie von der Prinzipalin , der E K S. A. R. L. G - einer Schwestergesellschaft - eine umsatzbezogene Vergütung .

**False Positives:**

- `S. A.` — partial — pred is substring of gold: `E K S. A. R. L. G`
- `R. L.` — partial — pred is substring of gold: `E K S. A. R. L. G`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `E K S. A. R. L. G`(ORG)

**Example 12** (doc_id: `66113`) (sent_id: `66113`)


Bei anschließenden Internetrecherchen wurde ein mit großer Wahrscheinlichkeit dem Kläger zuzuordnendes ask . fm-Profil eines " C. J. " aufgefunden , das die Flagge des sogenannten Islamischen Staates zeigte und weitere salafistische Inhalte aufwies .

**False Positives:**

- `C. J.` — partial — pred is substring of gold: `" C. J. "`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `" C. J. "`(PER)
- `Islamischen Staates`(ORG)

</details>

---

## `Anonymized initials with ellipses after roles`

**F1:** 0.030 | **Precision:** 1.000 | **Recall:** 0.015  

**Format:** `regex`  
**Rule ID:** `2eb9f1cc`  
**Description:**
Captures anonymized names with ellipses (e.g., 'K …', 'B1 …') following legal roles or titles.

**Content:**
```
\b(?:Angeklagte|Angeklagten|Kläger|Klägerin|Zeuge|Zeugin|Geschädigte|Geschädigten|Rechtsanwalt|Rechtsanwältin|Patentanwalt|Vorsitzender|Richter|Richterin|Herr|Herrn)\s+([A-Z]\s+…|[A-Z]\d+\s+…)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 1.000 | 0.015 | 0.030 | 5 | 5 | 0 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 5 | 0 | 290 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `60716`) (sent_id: `60716`)


Daraufhin ist das Anschreiben vom 31. Juli 2012 zusammen mit dem Bescheid nochmals mit Zustellungsurkunde an Patentanwalt K … verschickt und ausweislich der Zustellungsurkunde am 11. August 2012 durch Einlegen des Schriftstücks in den zur Wohnung gehörenden Briefkasten oder in eine ähnliche Vorrichtung zugestellt worden .

| Predicted | Gold |
|---|---|
| `K …` | `K …` |

**Example 1** (doc_id: `63808`) (sent_id: `63808`)


Aufgrund der dargelegten Sachlage hätte die Prüfungsstelle die Unwirksamkeit der Zustellungen erkennen können , insbesondere nachdem sie durch die Mitteilung von Patentanwalt B1 … vom 2. Mai 2013 Kenntnis von dem Bescheid der Patentanwaltskammer vom 4. April 2013 und damit von der Tatsache erhalten hat , dass die Kanzlei des beigeordneten Patentanwalts K … jedenfalls zum Zeit- punkt der vermeintlichen Zustellung der Fristverlängerung mit Beschlussankündigung am 14. März 2013 schon seit einiger Zeit verwaist war .

| Predicted | Gold |
|---|---|
| `B1 …` | `B1 …` |

**Missed by this rule (FN):**

- `K …` (PER)

**Example 2** (doc_id: `64657`) (sent_id: `64657`)


Sie hat insbesondere ausgesagt : Herr F … nahm das betreffende Kuvert und bestand darauf , das Kuvert selbst abzuliefern .

| Predicted | Gold |
|---|---|
| `F …` | `F …` |

**Example 3** (doc_id: `64795`) (sent_id: `64795`)


Weiterhin macht er geltend , Patentanwalt K … sei wegen einer psychischen Erkrankung zur Zeit der Zustellversuche des DPMA geschäftsunfähig nach § 104 Abs. 2 BGB gewesen , weshalb die Zustellungen unwirksam seien .

| Predicted | Gold |
|---|---|
| `K …` | `K …` |

**Missed by this rule (FN):**

- `DPMA` (ORG)
- `§ 104 Abs. 2 BGB` (NRM)

**Example 4** (doc_id: `65305`) (sent_id: `65305`)


c ) Damit erledigt sich der Antrag auf Gewährung von Prozesskostenhilfe und Beiordnung von Rechtsanwältin H … für das Verfahren auf Erlass einer einstweiligen Anordnung .

| Predicted | Gold |
|---|---|
| `H …` | `H …` |

</details>

---

## `Names after 'Richter' or 'Vorsitzender'`

**F1:** 0.018 | **Precision:** 0.500 | **Recall:** 0.009  

**Format:** `regex`  
**Rule ID:** `4f43c0ca`  
**Description:**
Captures names following judicial titles like 'Richter', 'Vorsitzender', 'Richterin', 'Vorsitzende Richterin', ensuring the name is captured even if preceded by 'Dr.' or 'Prof.'.

**Content:**
```
\b(?:Richter|Vorsitzender|Richterin|Vorsitzende Richterin|Vorsitzenden Richters)\s+(?:Dr\.?\s+|Prof\.?\s+|Dipl\.-[A-Za-z]+\s+)?([A-Z][a-zäöüß]+(?:-[A-Z][a-zäöüß]+)*)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.500 | 0.009 | 0.018 | 6 | 3 | 3 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 3 | 3 | 303 |

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

**Example 1** (doc_id: `62802`) (sent_id: `62802`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2014 004 201.0 hat der 29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 29. November 2017 durch die Vorsitzende Richterin Dr. Mittenberger-Huber , die Richterin Akintche und die Richterin Seyfarth beschlossen :

| Predicted | Gold |
|---|---|
| `Akintche` | `Akintche` |
| `Seyfarth` | `Seyfarth` |

**Missed by this rule (FN):**

- `29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Mittenberger-Huber` (PER)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `64527`) (sent_id: `64527`)


In der Beschwerdesache betreffend die Marke 30 2011 035 856 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 1** (doc_id: `64925`) (sent_id: `64925`)


In der Beschwerdesache betreffend die Marke 30 2010 003 649 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `Dr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 2** (doc_id: `65500`) (sent_id: `65500`)


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

## `Names after 'Herr' or 'Herrn'`

**F1:** 0.006 | **Precision:** 0.333 | **Recall:** 0.003  

**Format:** `regex`  
**Rule ID:** `12d202b6`  
**Description:**
Captures names immediately following the title 'Herr' or 'Herrn', including single initials.

**Content:**
```
\b(?:Herr|Herrn)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*|([A-Z])\.)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.333 | 0.003 | 0.006 | 3 | 1 | 2 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 1 | 2 | 241 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `66350`) (sent_id: `66350`)


Im Berufungsverfahren fand zunächst ein Erörterungstermin am 7. 9. 2016 statt , in dem der Berichterstatter des LSG die Klägerin persönlich anhörte und Herrn G. als Zeugen vernahm .

| Predicted | Gold |
|---|---|
| `G.` | `G.` |

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

## `Full names with titles`

**F1:** 0.006 | **Precision:** 0.111 | **Recall:** 0.003  

**Format:** `regex`  
**Rule ID:** `84421f67`  
**Description:**
Captures full names preceded by titles like Dr., Prof., or Dipl.-Ing., ensuring multi-part names with middle initials (e.g., 'Jay B. Saoud') are captured as a single entity.

**Content:**
```
(?:Dr\.?\s+|Prof\.?\s+|Dipl\.-Ing\.\s+|Dipl\.-Psych\.\s+|Dipl\.-Ing\.\s+Univ\.\s+)([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*|([A-Z]\.)\s+[A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.111 | 0.003 | 0.006 | 9 | 1 | 8 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 1 | 8 | 257 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `61554`) (sent_id: `61554`)


Dr. Achilles

| Predicted | Gold |
|---|---|
| `Achilles` | `Achilles` |

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

</details>

---

## `Hyphenated surnames`

**F1:** 0.005 | **Precision:** 0.015 | **Recall:** 0.003  

**Format:** `regex`  
**Rule ID:** `97b86ba2`  
**Description:**
Captures hyphenated surnames like 'Schmidt-Räntsch' only when preceded by a title or in a list of names to avoid matching court names.

**Content:**
```
(?:Dr\.?\s+|Prof\.?\s+|Richter\s+|Vorsitzender\s+|und\s+|sowie\s+|der\s+|die\s+|des\s+)([A-Z][a-zäöüß]+(?:-[A-Z][a-zäöüß]+)+)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.015 | 0.003 | 0.005 | 65 | 1 | 64 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 1 | 64 | 323 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `62802`) (sent_id: `62802`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2014 004 201.0 hat der 29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 29. November 2017 durch die Vorsitzende Richterin Dr. Mittenberger-Huber , die Richterin Akintche und die Richterin Seyfarth beschlossen :

| Predicted | Gold |
|---|---|
| `Mittenberger-Huber` | `Mittenberger-Huber` |

**Missed by this rule (FN):**

- `29. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts` (ORG)
- `Akintche` (PER)
- `Seyfarth` (PER)

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60065`) (sent_id: `60065`)


- durch deren Betätigung ein durch die Antriebseinrichtung bewirktes Öffnen des Schiebeflügels zur Freigabe eines Flucht- und Rettungswegs auslösbar ist ( Seite 3 , Abschnitt 1.1.4 , 4. Spiegelstrich , 7. Punkt : „ wenn zusätzlich Fluchttüranforderung besteht “ , Seite 4 , Abschnitt 2.1 , 5. Spiegelstrich : „ Automatische Schiebetür … zum Einsatz in Rettungswegen “ und Seite 4 , Abschnitt 2.1 , letzter Absatz : „ … darf die Rauchschutz-Schiebetür nur durch Betätigung der NOT-AUF-Taster … für den Durchgang von Personen geöffnet werden . “ ; Merkmal 1.8) ;

**False Positives:**

- `Rauchschutz-Schiebetür` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `60101`) (sent_id: `60101`)


3. Als zuständigen Fachmann sieht der Senat – in Übereinstimmung mit der Patentabteilung im Einspruchsbeschluss – einen Diplomingenieur der Elektrotechnik mit mehrjähriger Berufserfahrung auf dem Gebiet der Hardware- und Software-Entwicklung und des Betreibens von Sicherheitsschaltern .

**False Positives:**

- `Software-Entwicklung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `60155`) (sent_id: `60155`)


Die Beschwerde hält in diesem Zusammenhang für klärungsbedürftig , ob die in § 68 Abs. 1 Satz 1 , 3 und 4 , § 68a Satz 1 AufenthG bundesgesetzlich geregelte Geltungsdauer für Verpflichtungserklärungen durch landesinterne Vorgaben ( hier : Aufnahmeanordnung des Landes Rheinland-Pfalz vom 30. August 2013 i. V. m. den zugehörigen Anwendungshinweisen ) eingeschränkt werden kann , soweit davon Leistungen in der Verantwortung des Bundes ( hier : Leistungen der Grundsicherung für Arbeitsuchende nach dem Zweiten Buch Sozialgesetzbuch - SGB II - in der Trägerschaft der Bundesagentur für Arbeit nach § 6 Abs. 1 Satz 1 Nr. 1 SGB II ) betroffen wären .

**False Positives:**

- `Rheinland-Pfalz` — type mismatch — same span as gold: `Rheinland-Pfalz`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 68 Abs. 1 Satz 1 , 3 und 4 , § 68a Satz 1 AufenthG`(NRM)
- `Rheinland-Pfalz`(LOC)
- `Zweiten Buch Sozialgesetzbuch`(NRM)
- `SGB II`(NRM)
- `Bundesagentur für Arbeit`(ORG)
- `§ 6 Abs. 1 Satz 1 Nr. 1 SGB II`(NRM)

**Example 3** (doc_id: `60167`) (sent_id: `60167`)


Den Bescheid vom 3. 3. 2009 korrigierte die Beklagte zugunsten der Klägerin mit weiterem Bescheid vom 7. 7. 2009 und setzte die Rückforderung wegen Überschreitung der Job-Sharing-Grenzen für die drei genannten Quartale auf insgesamt 9125,83 Euro fest .

**False Positives:**

- `Job-Sharing-Grenzen` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `60196`) (sent_id: `60196`)


Dies bedeutet , dass sich auch die Trennstrecke außerhalb des Lichtbogen-Brennraums befindet .

**False Positives:**

- `Lichtbogen-Brennraums` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `60263`) (sent_id: `60263`)


Mit diesem Bescheid sei die Punktzahlobergrenze im Rahmen des Job-Sharings bindend festgesetzt worden .

**False Positives:**

- `Job-Sharings` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 6** (doc_id: `60367`) (sent_id: `60367`)


Um 21:37 Uhr befuhr er die Hans-Böckler-Straße und anschließend deren Verlängerung , die Nordstraße , in stadtauswärtiger Richtung .

**False Positives:**

- `Hans-Böckler-Straße` — type mismatch — same span as gold: `Hans-Böckler-Straße`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Hans-Böckler-Straße`(LOC)
- `Nordstraße`(LOC)

**Example 7** (doc_id: `60469`) (sent_id: `60469`)


Der Senat ist insoweit an die unter Anwendung des Baden-Württembergischen Landesrechts getroffene Entscheidung des LSG gebunden ( § 202 SGG iVm § 560 ZPO ) .

**False Positives:**

- `Baden-Württembergischen` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 202 SGG`(NRM)
- `§ 560 ZPO`(NRM)

**Example 8** (doc_id: `60475`) (sent_id: `60475`)


Vorführung von Waren für Werbezwecke , insbesondere Präsentation von Waren im Teleshoppingbereich ; das Zusammenstellen verschiedener Waren [ ausgenommen deren Transport ] für Dritte , um über Websites oder Teleshopping-Sendungen den Verbrauchern Ansicht und Erwerb dieser Waren zu erleichtern .

**False Positives:**

- `Teleshopping-Sendungen` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 9** (doc_id: `60602`) (sent_id: `60602`)


Sie verwiesen auf das Urteil des Bundesfinanzhofs ( BFH ) vom 12. Mai 2015 VIII R 4/15 ( BFHE 250 , 75 , BStBl II 2015 , 835 ) , wonach die Erlöse aus der Auslieferung des Xetra-Goldes nicht im Rahmen der Einkünfte aus Kapitalvermögen steuerbar seien .

**False Positives:**

- `Xetra-Goldes` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Urteil des Bundesfinanzhofs ( BFH ) vom 12. Mai 2015 VIII R 4/15 ( BFHE 250 , 75 , BStBl II 2015 , 835 )`(RS)

**Example 10** (doc_id: `60790`) (sent_id: `60790`)


Diese kann zwar , wie § 73 Abs. 1 Satz 2 ArbGG deutlich macht , nicht auf die Versäumung der Fünf-Monats-Frist gestützt werden .

**False Positives:**

- `Fünf-Monats-Frist` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 73 Abs. 1 Satz 2 ArbGG`(NRM)

**Example 11** (doc_id: `60800`) (sent_id: `60800`)


Hier reicht die Zündhilfselektrode bei einer Definition des Lichtbogen-Brennraums durch die von den Distanzhaltern ( 21 ) und der Isolierung gebildete Linie nicht in den Lichtbogen-Brennraum hinein .

**False Positives:**

- `Lichtbogen-Brennraums` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 12** (doc_id: `60875`) (sent_id: `60875`)


Eine Novelle der 35. Verordnung zur Durchführung des Bundes-Immissionsschutzgesetzes sei in diesem Zusammenhang wünschenswert , aber nicht notwendig .

**False Positives:**

- `Bundes-Immissionsschutzgesetzes` — partial — pred is substring of gold: `35. Verordnung zur Durchführung des Bundes-Immissionsschutzgesetzes`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `35. Verordnung zur Durchführung des Bundes-Immissionsschutzgesetzes`(NRM)

**Example 13** (doc_id: `60935`) (sent_id: `60935`)


Klasse 42 : Aktualisieren von Computer-Software ; Aktualisierung und Design von Computer-Software ; Aktualisierung und Wartung von Computer-Software ; Aktualisierung von Software-Datenbanken ; Aktualisierung von Speicherbanken [ Software ] von Computersystemen ; Beratung auf dem Gebiet von Computerhardware und -software ; Beratung in Bezug auf Computer und Software ; Beratung in Bezug auf Computernetze mit unterschiedlichen Softwareumgebungen ; Beratungsdienste auf dem Gebiet von Computerhardware und -software ; Computerhardware- und -softwareberatungsdienstleistungen ; Computerprogrammierung und Softwareentwicklung ; Consulting und Beratung auf dem Gebiet der Computerhardware und -software ; Design von Computer-Software ; Designdienstleistungen für Computer-Software ; Dienstleistungen für den Entwurf von Software für die elektronische Datenverarbeitung ; Dienstleistungen für die Gestaltung von Computer-Software ; Entwickeln von Software ; Entwicklung von Computer-Software ; Entwicklung von Software ; Entwicklung von Software für Computer ; Entwicklung von Software für Rechner ; Entwicklung von Softwarelösungen für Internet-Provider und Internet-Nutzer ; Entwicklung , Programmierung und Implementierung von Software ; Entwurf und Entwicklung von Computerhardware und -software ; Entwurf , Entwicklung und Implementierung von Software ; Erstellung von Datenverarbeitungsprogrammen [ Software ] ; Erstellung , Wartung , Pflege und Anpassung von Software ; Hosting-Dienste , Software as a Service ( SaaS ) und Vermietung von Software ; Installation von Software ; Installation , Wartung und Reparatur von Software für Computer ; Installation , Wartung und Reparatur von Software für Computersysteme ; Kundenspezifische Gestaltung von Softwarepaketen ; Kundenspezifische Softwareanpassung ; Kundenspezifisches Design von Softwarepaketen ; Reparatur [ Wartung und Aktualisierung ] von Software ; Software as a Service [ SaaS ] ; Softwaredesign ; Softwaredesign und -entwicklung ; Softwareengineering ; Softwareentwicklung ; Softwareentwicklungsdienste ; Softwareentwicklungsleistungen ; Softwareerstellung ; Softwareerstellungsleistungen ; Softwarevermietung für Computer ; Technischer Support im Softwarebereich ; Vermietung von Computer-Software ; Vermietung von Software für Computer ; Vermietung von Software für Rechner ; Wartung und Aktualisierung von Software ; Wartung und Reparatur von Software ;

**False Positives:**

- `Internet-Nutzer` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 14** (doc_id: `60945`) (sent_id: `60945`)


Ein RLV in diesem Sinne war die von einem Arzt oder der Arztpraxis in einem bestimmten Zeitraum abrechenbare Menge vertragsärztlicher Leistungen , die mit den in der Euro-Gebührenordnung enthaltenen und für den Arzt oder die Arztpraxis geltenden Preisen zu vergüten war ( § 87b Abs 2 S 2 SGB V aF ) .

**False Positives:**

- `Euro-Gebührenordnung` — type mismatch — same span as gold: `Euro-Gebührenordnung`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Euro-Gebührenordnung`(NRM)
- `§ 87b Abs 2 S 2 SGB V aF`(NRM)

**Example 15** (doc_id: `61037`) (sent_id: `61037`)


Das Verbot betraf danach sämtliche Fußballstadien in Deutschland hinsichtlich nationaler und internationaler Fußballveranstaltungen von Vereinen beziehungsweise Tochtergesellschaften der Fußball-Bundesligen und der Fußballregionalligen sowie des Deutschen Fußball-Bundes .

**False Positives:**

- `Fußball-Bundesligen` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutschland`(LOC)
- `Deutschen Fußball-Bundes`(ORG)

**Example 16** (doc_id: `61307`) (sent_id: `61307`)


Die Berechnung der Job-Sharing-Obergrenze sei zutreffend unter Heranziehung der Gruppe der Internisten mit dem Schwerpunkt Kardiologie erfolgt .

**False Positives:**

- `Job-Sharing-Obergrenze` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 17** (doc_id: `61367`) (sent_id: `61367`)


Zudem liege in einem Anstieg der Quadratmeter-Miete von 4,95 Euro im September 2008 auf 5,18 Euro im vierten Quartal 2011 kein unvorhergesehener Preissprung , sondern eine normale Preisentwicklung .

**False Positives:**

- `Quadratmeter-Miete` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 18** (doc_id: `61526`) (sent_id: `61526`)


Diese Angaben sind nur insoweit gegenständlich merkmalsbildend , als dass die beanspruchte Biegeschiene raumkörperlich so ausgebildet sein muss , dass beim bestimmungsgemäßen Anlegen an einen Patientenfuß die Schwenkachse der Gelenkeinrichtung in etwa der Gelenkachse des Großzehengrundgelenks in der Flexion-Extensionsrichtung entspricht ( vgl. Figur 1 u. 2 ) .

**False Positives:**

- `Flexion-Extensionsrichtung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `61586`) (sent_id: `61586`)


Die Klägerin hatte ihren Sitz zum Zeitpunkt des Eintritts der Job-Sharing-Partnerin Dr. E. im Bezirk der KÄV Nordbaden .

**False Positives:**

- `Job-Sharing-Partnerin` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `E.`(PER)
- `KÄV Nordbaden`(ORG)

**Example 20** (doc_id: `61608`) (sent_id: `61608`)


Auch weitere – sowohl aktuelle , als auch vor dem Anmeldezeitpunkt datierende – Verwendungsbeispiele beziehen sich auf den Bereich der Raum- und Farbgestaltung : darin ist von „ Wohlfühlfarben : Natürliches Flair “ ( www.livingathome.de) , von „ Sanfte ( n ) Tönen im Wohnbereich – Wohlfühlfarben “ ( www.zuhausewohnen.de) , „ Trendtöne ( n ) : Wohlfühlfarben ... “ ( www.wunderweib.de) die Rede ( vgl. hierzu die Google-Recherche zum Stichwort „ Wohlfühlfarben “ nebst Anlagen ; siehe im Übrigen auch schon BPatG , a. a. O. , 30 W ( pat ) 530/13 - WohlFühlFarben , mit weiteren Verwendungsbeispielen ) .

**False Positives:**

- `Google-Recherche` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BPatG , a. a. O. , 30 W ( pat ) 530/13 - WohlFühlFarben`(RS)

**Example 21** (doc_id: `61788`) (sent_id: `61788`)


Nicht anwendbar ist entgegen der Auffassung des Landesarbeitsgerichts hingegen der Tarifvertrag zur Übernahme des Tarifrechts des Landes Berlin für die Beschäftigten der Berlin-Brandenburgischen Akademie der Wissenschaften ( ÜTV BBAW ) vom 30. Mai 2011 .

**False Positives:**

- `Berlin-Brandenburgischen` — partial — pred is substring of gold: `Tarifvertrag zur Übernahme des Tarifrechts des Landes Berlin für die Beschäftigten der Berlin-Brandenburgischen Akademie der Wissenschaften`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Tarifvertrag zur Übernahme des Tarifrechts des Landes Berlin für die Beschäftigten der Berlin-Brandenburgischen Akademie der Wissenschaften`(REG)
- `ÜTV BBAW`(REG)

**Example 22** (doc_id: `61948`) (sent_id: `61948`)


Durch die Durchschnittsbildung beim " fachgleichen Pärchen " werde einer etwaigen pflichtwidrigen Fehlzuordnung von Leistungen zum Zwecke der Umgehung der Leistungsobergrenze im Rahmen des Job-Sharings vorgebeugt .

**False Positives:**

- `Job-Sharings` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 23** (doc_id: `61974`) (sent_id: `61974`)


Im Streitfall war der Kläger indessen schon kein beherrschender Gesellschafter-Geschäftsführer der GmbH .

**False Positives:**

- `Gesellschafter-Geschäftsführer` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 24** (doc_id: `62353`) (sent_id: `62353`)


Als Beendigung der Rechtsfähigkeit des Betriebs ist der 3. 7. 1990 , als Rechtsnachfolger sind die Electronicon-GmbH G. und die B. Kondensatoren-GmbH eingetragen .

**False Positives:**

- `Electronicon-Gmb` — partial — pred is substring of gold: `Electronicon-GmbH G.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Electronicon-GmbH G.`(ORG)
- `B. Kondensatoren-GmbH`(ORG)

**Example 25** (doc_id: `62671`) (sent_id: `62671`)


Es kann dahinstehen , ob es fachüblich ist , vor der ( Vierleiter- ) Testung an jedem Anschlusskontakt des Bauelements in einer Schleife über die beiden Kontaktfedern der Kelvin-Kontaktierung den Übergangswiderstand der Kontaktierung zu den Kontaktfedern eines Kontaktfederpaares zu bestimmen und später zur Korrektur der Messergebnisse zu verwenden , vgl. Patentschrift , Absatz 0020 , denn der Senat kann auch im Zusammenhang mit einer derartigen Messung des Übergangswiderstands keine Veranlassung des Fachmanns erkennen , die Kontaktfedern C entsprechend der Anweisungen in den Merkmalen M3 bis M3.2 lamelliert für eine hohe Stromtragfähigkeit auszubilden .

**False Positives:**

- `Kelvin-Kontaktierung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 26** (doc_id: `62721`) (sent_id: `62721`)


Seit 16. 2. 2013 ist der Kläger Pflichtmitglied der Landestierärztekammer Baden-Württemberg ( im Folgenden : Landestierärztekammer ) und Pflichtmitglied der Baden-Württembergischen Versorgungsanstalt für Ärzte , Zahnärzte und Tierärzte ( Beigeladene zu 1. ) , die ihren Teilnehmern und deren Hinterbliebenen Altersruhegeld , Ruhegeld bei Berufsunfähigkeit sowie eine Hinterbliebenenversorgung gewährt .

**False Positives:**

- `Baden-Württembergischen` — partial — pred is substring of gold: `Baden-Württembergischen Versorgungsanstalt für Ärzte , Zahnärzte und Tierärzte`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Landestierärztekammer Baden-Württemberg`(ORG)
- `Baden-Württembergischen Versorgungsanstalt für Ärzte , Zahnärzte und Tierärzte`(ORG)

**Example 27** (doc_id: `62742`) (sent_id: `62742`)


1. Zunächst ist festzustellen , dass die Voraussetzung für die Durchführung des Löschungsverfahrens mit inhaltlicher Prüfung nach § 54 Abs. 2 Satz 3 MarkenG erfüllt ist , nachdem die Markeninhaberin dem ihr am 7. Mai 2013 zugestellten Löschungsantrag mit am 4. Juli 2013 beim DPMA eingegangenem Schriftsatz fristgerecht innerhalb der Zwei-Monats-Frist des § 54 Abs. 2 Satz 2 MarkenG widersprochen hat .

**False Positives:**

- `Zwei-Monats-Frist` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 54 Abs. 2 Satz 3 MarkenG`(NRM)
- `DPMA`(ORG)
- `§ 54 Abs. 2 Satz 2 MarkenG`(NRM)

**Example 28** (doc_id: `62814`) (sent_id: `62814`)


Auf die Beschwerde des Klägers gegen die Nichtzulassung der Revision wird das Urteil des Schleswig-Holsteinischen Landessozialgerichts vom 12. Dezember 2016 aufgehoben .

**False Positives:**

- `Schleswig-Holsteinischen` — partial — pred is substring of gold: `Schleswig-Holsteinischen Landessozialgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Schleswig-Holsteinischen Landessozialgerichts`(ORG)

**Example 29** (doc_id: `62923`) (sent_id: `62923`)


Im Rahmen der Umsatzbesteuerung unterlag er der Ist-Besteuerung .

**False Positives:**

- `Ist-Besteuerung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 30** (doc_id: `63002`) (sent_id: `63002`)


Die Beschwerde der Kläger wegen Nichtzulassung der Revision gegen das Urteil des Finanzgerichts des Landes Sachsen-Anhalt vom 26. Mai 2017 5 K 1166/10 wird als unbegründet zurückgewiesen .

**False Positives:**

- `Sachsen-Anhalt` — partial — pred is substring of gold: `Urteil des Finanzgerichts des Landes Sachsen-Anhalt vom 26. Mai 2017 5 K 1166/10`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Finanzgerichts des Landes Sachsen-Anhalt vom 26. Mai 2017 5 K 1166/10`(RS)

**Example 31** (doc_id: `63011`) (sent_id: `63011`)


I. 1. Die Beschwerdeführerin , die Verwaltungs-GmbH einer nicht rechtsfähigen Stiftung , wendet sich gegen den am 6. Dezember 2013 ( BGBl I S. 1386 ) in Kraft getretenen § 6a Bundesjagdgesetz ( BJagdG ) .

**False Positives:**

- `Verwaltungs-Gmb` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BGBl I S. 1386`(LIT)
- `§ 6a Bundesjagdgesetz`(NRM)
- `BJagdG`(NRM)

**Example 32** (doc_id: `63043`) (sent_id: `63043`)


1. Der Beschluss des Schleswig-Holsteinischen Oberlandesgerichts vom 7. Februar 2017 - 1 VollzWs 479/16 ( 271/16 ) - verletzt den Beschwerdeführer in seinem Grundrecht aus Artikel 2 Absatz 1 in Verbindung mit Artikel 1 Absatz 1 des Grundgesetzes .

**False Positives:**

- `Schleswig-Holsteinischen` — partial — pred is substring of gold: `Beschluss des Schleswig-Holsteinischen Oberlandesgerichts vom 7. Februar 2017 - 1 VollzWs 479/16 ( 271/16 ) -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Beschluss des Schleswig-Holsteinischen Oberlandesgerichts vom 7. Februar 2017 - 1 VollzWs 479/16 ( 271/16 ) -`(RS)
- `Artikel 2 Absatz 1 in Verbindung mit Artikel 1 Absatz 1 des Grundgesetzes`(NRM)

**Example 33** (doc_id: `63061`) (sent_id: `63061`)


Nach § 3 Abs. 1 der 39. Verordnung zur Durchführung des Bundes-Immissionsschutzgesetzes - Verordnung über Luftqualitätsstandards und Emissionshöchstmengen ( 39. BImSchV ) vom 2. August 2010 ( BGBl. I S. 1065 ) , zuletzt geändert durch Art. 1 des Gesetzes vom 10. Oktober 2016 ( BGBl. I S. 2244 ) , beträgt zum Schutz der menschlichen Gesundheit der über eine volle Stunde gemittelte Immissionsgrenzwert für Stickstoffdioxid ( NO2 ) 200 µg / m³ bei 18 zugelassenen Überschreitungen im Kalenderjahr .

**False Positives:**

- `Bundes-Immissionsschutzgesetzes` — partial — pred is substring of gold: `§ 3 Abs. 1 der 39. Verordnung zur Durchführung des Bundes-Immissionsschutzgesetzes - Verordnung über Luftqualitätsstandards und Emissionshöchstmengen ( 39. BImSchV ) vom 2. August 2010 ( BGBl. I S. 1065 ) , zuletzt geändert durch Art. 1 des Gesetzes vom 10. Oktober 2016 ( BGBl. I S. 2244 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 3 Abs. 1 der 39. Verordnung zur Durchführung des Bundes-Immissionsschutzgesetzes - Verordnung über Luftqualitätsstandards und Emissionshöchstmengen ( 39. BImSchV ) vom 2. August 2010 ( BGBl. I S. 1065 ) , zuletzt geändert durch Art. 1 des Gesetzes vom 10. Oktober 2016 ( BGBl. I S. 2244 )`(NRM)

**Example 34** (doc_id: `63086`) (sent_id: `63086`)


Nach Art. 16 Abs. 6 der Serviceverträge der Parteien müsse die - schriftliche - Kündigung eine Begründung enthalten , die objektiv und transparent sei , um sicherzustellen , dass die Kündigung nicht wegen Verhaltensweisen des Vertragspartners erfolge , die nach der Kfz-Gruppenfreistellungsverordnung 2002 nicht eingeschränkt werden dürften .

**False Positives:**

- `Kfz-Gruppenfreistellungsverordnung` — partial — pred is substring of gold: `Kfz-Gruppenfreistellungsverordnung 2002`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 16 Abs. 6 der Serviceverträge`(REG)
- `Kfz-Gruppenfreistellungsverordnung 2002`(NRM)

**Example 35** (doc_id: `63427`) (sent_id: `63427`)


3. Der Anspruch ergibt sich hingegen aus § 2.6 Abs. 2 TV Sonderzahlungen 2006 , einer Sonderregelung für Beschäftigte , die unter anderem zwecks Inanspruchnahme eines vorgezogenen Altersruhegeldes aus dem Beruf ausscheiden , auch wenn ihr Arbeitsverhältnis vor dem Auszahlungstag am 1. Dezember endet ( BAG 5. August 1992 - 10 AZR 208/91 - [ zu einer gleichlautenden Tarifvorschrift für das metallverarbeitende Handwerk in Nordrhein-Westfalen ] ; 15. Januar 2014 - 10 AZR 297/13 - [ zu einer gleichlautenden Tarifvorschrift für die Metall- und Elektroindustrie in Südbaden und Südwürttemberg-Hohenzollern ] ) .

**False Positives:**

- `Südwürttemberg-Hohenzollern` — partial — pred is substring of gold: `15. Januar 2014 - 10 AZR 297/13 - [ zu einer gleichlautenden Tarifvorschrift für die Metall- und Elektroindustrie in Südbaden und Südwürttemberg-Hohenzollern ]`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 2.6 Abs. 2 TV Sonderzahlungen 2006`(REG)
- `BAG 5. August 1992 - 10 AZR 208/91 - [ zu einer gleichlautenden Tarifvorschrift für das metallverarbeitende Handwerk in Nordrhein-Westfalen ]`(RS)
- `15. Januar 2014 - 10 AZR 297/13 - [ zu einer gleichlautenden Tarifvorschrift für die Metall- und Elektroindustrie in Südbaden und Südwürttemberg-Hohenzollern ]`(RS)

**Example 36** (doc_id: `63443`) (sent_id: `63443`)


An diesem Tag zählte die kreisfreie Stadt Sch. 127815 Einwohner , der Kreis Sch. Land einschließlich der kreisangehörigen Stadt C. 33997 Einwohner ( Statistisches Landesamt Mecklenburg-Vorpommern , Statistische Berichte , Unterreihe A. IS , Bevölkerungsstand der Kreise und Gemeinden des Landes Mecklenburg-Vorpommern , Sch. 1990 ) .

**False Positives:**

- `Mecklenburg-Vorpommern` — similar text (different position): `Statistisches Landesamt Mecklenburg-Vorpommern`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Sch.`(LOC)
- `Sch. Land`(LOC)
- `C.`(LOC)
- `Statistisches Landesamt Mecklenburg-Vorpommern`(ORG)
- `Mecklenburg-Vorpommern`(LOC)
- `Sch.`(LOC)

**Example 37** (doc_id: `63653`) (sent_id: `63653`)


Als sachkundige Auskunftsperson hat sich in der mündlichen Verhandlung Prof. Dr. Klaus-Dieter Drüen geäußert .

**False Positives:**

- `Klaus-Dieter` — partial — pred is substring of gold: `Klaus-Dieter Drüen`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Klaus-Dieter Drüen`(PER)

**Example 38** (doc_id: `63926`) (sent_id: `63926`)


Die weitere Begründung , die Dauer der Abwesenheit des Dublin-Rückkehrers könne Einfluss auf die Frage haben , ob dieser sein ursprüngliches Verfahren fortsetzen oder wiederaufnehmen könne ( OVG Münster , Urteil vom 13. Oktober 2017 - 11 A 78 / 17. A - juris Rn. 69 ) , ist nicht selbständig tragend .

**False Positives:**

- `Dublin-Rückkehrers` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `OVG Münster , Urteil vom 13. Oktober 2017 - 11 A 78 / 17. A - juris Rn. 69`(RS)

**Example 39** (doc_id: `63970`) (sent_id: `63970`)


a ) § 7 Satz 2 GewStG wurde durch Art. 5 des Fünften Gesetzes zur Änderung des Steuerbeamten-Ausbildungsgesetzes und zur Änderung von Steuergesetzen ( StBAÄG ) vom 23. Juli 2002 , verkündet am 26. Juli 2002 und in Kraft getreten am 27. Juli 2002 ( BGBl I S. 2715 ) , in das Gewerbesteuergesetz eingefügt .

**False Positives:**

- `Steuerbeamten-Ausbildungsgesetzes` — partial — pred is substring of gold: `Art. 5 des Fünften Gesetzes zur Änderung des Steuerbeamten-Ausbildungsgesetzes und zur Änderung von Steuergesetzen ( StBAÄG ) vom 23. Juli 2002 , verkündet am 26. Juli 2002 und in Kraft getreten am 27. Juli 2002 ( BGBl I S. 2715 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 7 Satz 2 GewStG`(NRM)
- `Art. 5 des Fünften Gesetzes zur Änderung des Steuerbeamten-Ausbildungsgesetzes und zur Änderung von Steuergesetzen ( StBAÄG ) vom 23. Juli 2002 , verkündet am 26. Juli 2002 und in Kraft getreten am 27. Juli 2002 ( BGBl I S. 2715 )`(NRM)
- `Gewerbesteuergesetz`(NRM)

**Example 40** (doc_id: `64052`) (sent_id: `64052`)


Zwar lassen die derzeit geltenden Regelungen des Bundes-Immissionsschutzrechts für sich genommen derartige Verkehrsverbote nicht zu .

**False Positives:**

- `Bundes-Immissionsschutzrechts` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 41** (doc_id: `64263`) (sent_id: `64263`)


Es hat jedoch nicht über den mit dem Schriftsatz vom 8. Dezember 2017 vorgeschalteten , nach herrschender Auffassung nicht an Fristen oder Antragsteller-Quoren gebundenen ( vgl. für die Wahl zum Gesamtvertrauenspersonenausschuss Höges , SBG , Stand März 2018 , § 47 Rn. 20 ) ( Haupt- ) Antrag , die Nichtigkeit der Wahl festzustellen , entschieden .

**False Positives:**

- `Antragsteller-Quoren` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Wahl zum Gesamtvertrauenspersonenausschuss Höges , SBG , Stand März 2018 , § 47 Rn. 20`(LIT)

**Example 42** (doc_id: `64471`) (sent_id: `64471`)


Mit Bescheiden vom 3. 3. 2009 und vom 29. 6. 2009 verfügte die Beklagte sachlich-rechnerische Richtigstellungen wegen Überschreitung der Job-Sharing-Obergrenzen in den Quartalen II. 2008 bis IV / 2008 .

**False Positives:**

- `Job-Sharing-Obergrenzen` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 43** (doc_id: `64527`) (sent_id: `64527`)


In der Beschwerdesache betreffend die Marke 30 2011 035 856 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `Design-Beschwerdesenat` — partial — pred is substring of gold: `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 44** (doc_id: `64564`) (sent_id: `64564`)


In Anspruch 1 nach Hilfsantrag 1 ist zusätzlich angegeben , dass zur Ausführung des Verfahrens ein Zentral-Server und mehrere Client-Systeme verwendet werden , wobei die Client-Systeme die Zusatzinformationen für die jeweiligen E-Shop-Systeme verwalten und über ein Netzwerk mit den E-Shop-Systemen jeweils verbunden sind ( Merkmale M1.1 ( 1 ) , M1.1 .1 ( 1 ) und M1.1 .2 ( 1 ) ) .

**False Positives:**

- `Client-Systeme` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 45** (doc_id: `64649`) (sent_id: `64649`)


2. Am 19. / 20. November 2010 verabschiedeten die NATO-Bündnispartner ein neues strategisches Konzept , welches sich erstmals auf das Ziel einer nuklearwaffenfreien Welt festlegte , zugleich aber das Prinzip der nuklearen Abschreckung bis zur vollständigen Vernichtung aller Nuklearwaffen auf der Welt bestätigte ( vgl. Strategisches Konzept für die Verteidigung und Sicherheit der Mitglieder der Nordatlantikvertrags-Organisation , 2010 ) .

**False Positives:**

- `Nordatlantikvertrags-Organisation` — type mismatch — same span as gold: `Nordatlantikvertrags-Organisation`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Nordatlantikvertrags-Organisation`(ORG)

**Example 46** (doc_id: `64774`) (sent_id: `64774`)


4. Verstößt es gegen den Grundsatz der Effektivität oder gegen die Dassonville-Formel seit dem Urteil des 11. Juli 1974 , Rs 8 - 74 , einem Unternehmer die Berufung auf die Gleichbehandlung mit Konkurrenten zu verweigern mit dem Hinweis darauf , der Leistungserbringer hätte eine Klage erheben müssen mit dem Ziel , dass auch die Konkurrenten mit Umsatzsteuer belegt werden ?

**False Positives:**

- `Dassonville-Formel` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 47** (doc_id: `64895`) (sent_id: `64895`)


M1.7 ( 6 ) Einbetten der Zusatzinformation , insbesondere der Links auf die Zusatzinformationen durch das E-Shop-System in die E-Shop-Seite und übermitteln dieser an das Endgerät , wobei das Client-System als Zusatzinformation neben der Media-URL auch das Link-Ziel der Zusatzinformation übermittelt , die Bestimmung des Link-Ziels erfolgt vollautomatisiert , indem eine Suchanfrage in den entsprechenden Produkt-Identifikatoren eingesetzt wird und als Link-Ziel für das E-Shop-System mit einem oder mehreren Produkt-Identifikatoren vorbesetzt wird .

**False Positives:**

- `Link-Ziels` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 48** (doc_id: `64925`) (sent_id: `64925`)


In der Beschwerdesache betreffend die Marke 30 2010 003 649 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `Design-Beschwerdesenat` — partial — pred is substring of gold: `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 49** (doc_id: `64936`) (sent_id: `64936`)


Es sei ferner nicht zulässig , die Verbotsentscheidung auf die Stadionverbots-Richtlinien zu stützen .

**False Positives:**

- `Stadionverbots-Richtlinien` — type mismatch — same span as gold: `Stadionverbots-Richtlinien`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Stadionverbots-Richtlinien`(REG)

**Example 50** (doc_id: `64983`) (sent_id: `64983`)


Aus dem Waldgesetz des Landes Schleswig-Holstein ergäbe sich ua eine Bewirtschaftungspflicht .

**False Positives:**

- `Schleswig-Holstein` — partial — pred is substring of gold: `Waldgesetz des Landes Schleswig-Holstein`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Waldgesetz des Landes Schleswig-Holstein`(NRM)

**Example 51** (doc_id: `65035`) (sent_id: `65035`)


So wird schon nicht substantiiert dargetan , dass die Klägerin gerade nur den terminlich verhinderten Rechtsanwalt bevollmächtigt hat und nicht auch die anderen zur Vertretung des Verhinderten in Betracht kommenden Kollegen der Rechtsanwalts-GbR ; die Klägerin geht in ihrem Vorbringen insoweit nicht darauf ein , dass sich die von ihr selbst zu den Akten gereichte und nicht erkennbar im Verfahrensablauf geänderte Vollmacht vom 30. 7. 2010 ausdrücklich auf alle damaligen Kollegen der Rechtsanwaltssozietät bezieht .

**False Positives:**

- `Rechtsanwalts-Gb` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 52** (doc_id: `65177`) (sent_id: `65177`)


Da hiernach Basis der Vergütung der " Wachstumsärztin " S. nicht die Fallzahl aus dem Vorjahresquartal , sondern die im Abrechnungsquartal tatsächlich erreichte Fallzahl ( maximal bis zum Fachgruppendurchschnitt ) gewesen sei , habe in der Vorab-Mitteilung eine feste Gesamt-Obergrenze nicht angegeben werden können .

**False Positives:**

- `Vorab-Mitteilung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `S.`(PER)

**Example 53** (doc_id: `65181`) (sent_id: `65181`)


Die Klägerin , die im Jahr 2003 das Studium der Germanistik / Allgemeine Sprachwissenschaften mit dem Magister abgeschlossen hatte , wurde von dem beklagten Land in der Zeit vom 1. April 2010 bis zum 30. September 2015 auf der Grundlage von drei aufeinanderfolgenden befristeten Arbeitsverträgen an der Ernst-Moritz-Arndt-Universität Greifswald beschäftigt .

**False Positives:**

- `Ernst-Moritz-Arndt-Universität` — partial — pred is substring of gold: `Ernst-Moritz-Arndt-Universität Greifswald`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Ernst-Moritz-Arndt-Universität Greifswald`(ORG)

**Example 54** (doc_id: `65199`) (sent_id: `65199`)


Aufgrund der fachlichen Weisungsbefugnis des Wing-Commanders der 80th Flying Training Wing und der durch die Dienstpostenbeschreibung vorgenommenen Tätigkeitszuweisung werde die tägliche bzw. wöchentliche Arbeitszeit für den vom Antragsteller bekleideten Dienstposten nicht in nationaler Verantwortung bestimmt , sondern aufgrund spezieller internationaler Abstimmungen zwischen der Bundeswehr und dem aufnehmenden Bereich .

**False Positives:**

- `Wing-Commanders` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Bundeswehr`(ORG)

**Example 55** (doc_id: `65287`) (sent_id: `65287`)


b ) Verpachtet ein Unternehmer ein Grundstück an einen Landwirt , der seine Umsätze gemäß § 24 Abs. 1 UStG nach Durchschnittssätzen versteuert , kann der Verpächter nicht auf die Steuerfreiheit seiner Umsätze nach § 9 Abs. 2 Satz 1 UStG verzichten ( zutreffend Nieuwenhuis , a. a. O. , § 9 UStG Rz 78 ; Schüler-Täsch in Sölch / Ringleb , Umsatzsteuer , § 9 Rz 47 , und Mehrwertsteuerrecht 2013 , 540 , 543 ; Stadie in Stadie , UStG , 3. Aufl. , § 9 Rz 28 und § 24 Rz 41 ; a. M. Abschn. 9.2 Abs. 2 des Umsatzsteuer-Anwendungserlasses - UStAE - , sowie Lange in Offerhaus / Söhn / Lange , § 24 UStG Rz 456 , und Widmann in Schwarz / Widmann / Radeisen , UStG , § 9 Rz 171 ) .

**False Positives:**

- `Umsatzsteuer-Anwendungserlasses` — partial — pred is substring of gold: `Abschn. 9.2 Abs. 2 des Umsatzsteuer-Anwendungserlasses - UStAE`

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

**Example 56** (doc_id: `65500`) (sent_id: `65500`)


In der Beschwerdesache betreffend die Designanmeldung .... ( hier : Antrag auf Verfahrenskostenhilfe ) hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `Design-Beschwerdesenat` — partial — pred is substring of gold: `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 57** (doc_id: `65927`) (sent_id: `65927`)


Hierfür kommt neben der Bürgschaft auch ein Schuldbeitritt des Gesellschafter-Geschäftsführers in Betracht .

**False Positives:**

- `Gesellschafter-Geschäftsführers` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 58** (doc_id: `66065`) (sent_id: `66065`)


Dem stehen auch die Vorgaben der " Maßnahme zur Änderung der Soll-Organisation " ( MÄSO ) des Kommandos Sanitätsdienst der Bundeswehr ... nicht entgegen .

**False Positives:**

- `Soll-Organisation` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Bundeswehr ...`(ORG)

**Example 59** (doc_id: `66203`) (sent_id: `66203`)


Bei der vom Kläger erbrachten , in der Aufhebungsvereinbarung vom 4. März 2011 als " Entschädigung " bezeichneten Gegenleistung hat es sich - wovon das FG zutreffend ausgegangen ist und auch der Verwaltungsauffassung in Abschn. 1. 3. Abs. 13 Satz 1 des Umsatzsteuer-Anwendungserlasses entspricht - um ein Leistungsentgelt für die Bereitschaft der Pächter , die Vertragslaufzeit von März 2020 auf den 30. April 2012 zu verkürzen , gehandelt .

**False Positives:**

- `Umsatzsteuer-Anwendungserlasses` — partial — pred is substring of gold: `Abschn. 1. 3. Abs. 13 Satz 1 des Umsatzsteuer-Anwendungserlasses`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Abschn. 1. 3. Abs. 13 Satz 1 des Umsatzsteuer-Anwendungserlasses`(REG)

**Example 60** (doc_id: `66326`) (sent_id: `66326`)


Durch die Übertragung der Produkt-Identifikatoren gemeinsam mit der Session-ID soll das Suchverhalten des Kunden , d. h. die für eine Werbeeinblendung benötigte Information , ermittelt werden , um dem Kunden eine angepasste Information anzeigen zu können .

**False Positives:**

- `Produkt-Identifikatoren` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 61** (doc_id: `66377`) (sent_id: `66377`)


3. Mit der Einführung des mit der Verfassungsbeschwerde mittelbar angegriffenen § 7 Satz 2 GewStG durch das Fünfte Gesetz zur Änderung des Steuerbeamten-Ausbildungsgesetzes und zur Änderung von Steuergesetzen hat der Gesetzgeber diese Rechtslage für Mitunternehmerschaften beendet und bei ihnen auch die Gewinne aus der Veräußerung ihres Betriebs , eines Teilbetriebs oder von Anteilen eines Gesellschafters , der als Mitunternehmer anzusehen ist , weitgehend der Gewerbesteuer unterworfen .

**False Positives:**

- `Steuerbeamten-Ausbildungsgesetzes` — partial — pred is substring of gold: `Fünfte Gesetz zur Änderung des Steuerbeamten-Ausbildungsgesetzes und zur Änderung von Steuergesetzen`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 7 Satz 2 GewStG`(NRM)
- `Fünfte Gesetz zur Änderung des Steuerbeamten-Ausbildungsgesetzes und zur Änderung von Steuergesetzen`(NRM)

**Example 62** (doc_id: `66440`) (sent_id: `66440`)


Bei der Umsatzermittlung ist nicht auf die Bemessungsgrundlage abzustellen , sondern auf die vom Unternehmer vereinnahmten Bruttobeträge ( vergleiche dazu zum Beispiel Tehler , Umsatzsteuer- und Verkehrsteuer-Recht - UVR - 2016 , 345 ; Bunjes / Korn , UStG , 16. Aufl. , § 19 Rz 29 ) .

**False Positives:**

- `Verkehrsteuer-Recht` — partial — pred is substring of gold: `Tehler , Umsatzsteuer- und Verkehrsteuer-Recht - UVR - 2016 , 345`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Tehler , Umsatzsteuer- und Verkehrsteuer-Recht - UVR - 2016 , 345`(LIT)
- `Bunjes / Korn , UStG , 16. Aufl. , § 19 Rz 29`(LIT)

**Example 63** (doc_id: `66697`) (sent_id: `66697`)


Das Vertrauen sei auch deshalb uneingeschränkt schutzwürdig , weil das Rechtsgeschäft am 1. Februar 2002 , also noch vor der Verkündung des Steuerbeamten-Ausbildungsgesetzes , mit dem Closing vollständig abgewickelt worden sei .

**False Positives:**

- `Steuerbeamten-Ausbildungsgesetzes` — type mismatch — same span as gold: `Steuerbeamten-Ausbildungsgesetzes`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Steuerbeamten-Ausbildungsgesetzes`(NRM)

</details>

---

## `Anonymized initials with dots`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `af28d6a7`  
**Description:**
Captures anonymized person identifiers consisting of a single capital letter followed by a dot (e.g., 'T.', 'F.', 'S.'), ensuring the dot is included.

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

## `Names after 'Herr'`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `0599e43e`  
**Description:**
Captures names immediately following the title 'Herr' (including 'Herrn').

**Content:**
```
\b(?:Herr|Herrn)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Names after 'Angeklagte' or 'Klägerin'`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `4993cd07`  
**Description:**
Captures names following legal role indicators like 'Angeklagte' or 'Klägerin'.

**Content:**
```
\b(?:Angeklagte|Angeklagten|Klägerin|Kläger|Zeugin|Zeuge|Geschädigte|Gutachter|Gutachterin)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 12 | 0 | 12 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 12 | 324 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `60099`) (sent_id: `60099`)


Jedenfalls beruht das Ergebnis des Gutachtens auf einer umfassenden Auswertung des dem Gutachter zur Verfügung gestellten Aktenmaterials , aus dem der Gutachter Schlüsse zieht , die auch unabhängig von den dem Senat im Einzelnen nicht bekannten Prognosemanualen nachvollziehbar erscheinen und im Einklang mit der ordnungsrechtlichen Gefahrenbewertung stehen , wie sie auch nach dem Akteninhalt im Übrigen veranlasst ist .

**False Positives:**

- `Schlüsse` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `60750`) (sent_id: `60750`)


Der Bescheid der Beklagten ist rechtmäßig , soweit der Klägerin Insg von nicht mehr als 3927,71 Euro bewilligt worden ist .

**False Positives:**

- `Insg` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `60944`) (sent_id: `60944`)


Nachdem er vom FA auf den fehlenden Nachweis einer regelmäßigen Summenziehung hingewiesen worden sei , habe der Kläger Erfassungsprotokolle beim FG eingereicht , die eine chronologische Auflistung der Geschäftsvorfälle ohne Angabe von Belegnummern enthalten hätten .

**False Positives:**

- `Erfassungsprotokolle` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 3** (doc_id: `61328`) (sent_id: `61328`)


Am 18. November 2015 hat der Kläger Klage erhoben und zur Begründung geltend gemacht , der Jahresmittelgrenzwert für Stickstoffdioxid ( NO2 ) sei im Jahr 2013 an allen verkehrsnahen Messstationen zum Teil um mehr als das Doppelte überschritten worden und habe auch im Jahr 2014 an bestimmten Messstationen deutlich über den Grenzwerten gelegen .

**False Positives:**

- `Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `62164`) (sent_id: `62164`)


Gegen dieses Urteil hat die Klägerin Revision eingelegt .

**False Positives:**

- `Revision` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `62721`) (sent_id: `62721`)


Seit 16. 2. 2013 ist der Kläger Pflichtmitglied der Landestierärztekammer Baden-Württemberg ( im Folgenden : Landestierärztekammer ) und Pflichtmitglied der Baden-Württembergischen Versorgungsanstalt für Ärzte , Zahnärzte und Tierärzte ( Beigeladene zu 1. ) , die ihren Teilnehmern und deren Hinterbliebenen Altersruhegeld , Ruhegeld bei Berufsunfähigkeit sowie eine Hinterbliebenenversorgung gewährt .

**False Positives:**

- `Pflichtmitglied` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Landestierärztekammer Baden-Württemberg`(ORG)
- `Baden-Württembergischen Versorgungsanstalt für Ärzte , Zahnärzte und Tierärzte`(ORG)

**Example 6** (doc_id: `63009`) (sent_id: `63009`)


Hiergegen hat der Kläger Klage zum SG erhoben , das durch Urteil vom 2. 10. 2012 den Bescheid der Beklagten vom 18. 4. 2011 in der Gestalt des Widerspruchsbescheids vom 8. 6. 2011 aufgehoben hat , weil das Grundstück des Klägers aufgrund der anzuwendenden Ausnahmevorschrift des § 123 Abs 2 SGB VII als versicherungsfreier Haus- und Ziergarten einzuordnen sei .

**False Positives:**

- `Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 123 Abs 2 SGB VII`(NRM)

**Example 7** (doc_id: `63032`) (sent_id: `63032`)


III. Sollte die neue Verhandlung ergeben , dass die Klägerin Tätigkeiten mit nicht unwesentlichem Einfluss auf die Programmgestaltung schuldete , wird das Landesarbeitsgericht im Rahmen der Prüfung , ob die Befristung zum 31. Mai 2014 mit der Rundfunkfreiheit gerechtfertigt werden kann , eine erneute einzelfallbezogene Abwägung der Belange des Beklagten und der Klägerin vorzunehmen haben .

**False Positives:**

- `Tätigkeiten` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `63492`) (sent_id: `63492`)


Mit der Revision rügen die Kläger Verletzung formellen und materiellen Rechts .

**False Positives:**

- `Verletzung` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 9** (doc_id: `64047`) (sent_id: `64047`)


In einem weiteren Fall öffnete der Angeklagte Knopf und Reißverschluss seiner Hose und forderte die Zeugin sinngemäß auf , an seinem Glied zu reiben .

**False Positives:**

- `Knopf` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 10** (doc_id: `66649`) (sent_id: `66649`)


M. hielt der Zeugin Mund und Nase zu , so dass sie nicht mehr schreien konnte .

**False Positives:**

- `Mund` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `M.`(PER)

**Example 11** (doc_id: `66655`) (sent_id: `66655`)


Am 18. November 2015 hat der Kläger Klage erhoben und zur Begründung geltend gemacht , die anhaltende Überschreitung der Grenzwerte sei ein Indiz dafür , dass die bisherigen Maßnahmen nicht geeignet seien , die Überschreitungszeiträume so kurz wie möglich zu halten .

**False Positives:**

- `Klage` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Names after 'Dr.' or 'Prof.'`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `6e133f27`  
**Description:**
Captures names following titles like Dr. or Prof., handling both full names and initials.

**Content:**
```
\b(?:Dr\.?\s+|Prof\.?\s+)([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*|[A-Z]\.[ ]+[A-Z]\.|[A-Z]\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Names after 'Rechtsanwältin' or 'Rechtsanwalt'`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `6e0b7cc1`  
**Description:**
Captures names following legal profession titles.

**Content:**
```
\b(?:Rechtsanwältin|Rechtsanwalt)\s+(?:Dr\.?\s+|Prof\.?\s+)?([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Names with dots in middle (e.g., B1 …)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `6a893240`  
**Description:**
Captures anonymized names with dots and ellipses or spaces (e.g., 'B1 …', 'K …', 'K1 …').

**Content:**
```
\b([A-Z]\d?\s+…+)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Anonymized names with ellipses`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `daa8797d`  
**Description:**
Captures anonymized names with ellipses like 'K …', 'B1 …', 'T. D.', 'L. …', 'Ch. …' ensuring no trailing spaces are included.

**Content:**
```
\b([A-Z]\s+…|[A-Z]\d+\s+…|T\.\s+D\.|B1\s+…|K1\s+…|H\.\s+…|L\.\s+…|Ch\.\s+…|T\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Surnames after 'Generalanwalt' or 'Generalanwältin'`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `549dd13e`  
**Description:**
Captures names following 'Generalanwalt' or 'Generalanwältin' titles.

**Content:**
```
\b(?:Generalanwalt|Generalanwältin)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Surnames after 'Richter' or 'Vorsitzender' (refined)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `68df845b`  
**Description:**
Captures names following judicial titles, ensuring the name is captured correctly even if preceded by 'Dipl.' or 'Prof.'.

**Content:**
```
\b(?:Richter|Vorsitzender)\s+(?:Dipl\.-[a-z]+\s+)?([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Surnames after 'Rechtsanwalt' or 'Rechtsanwältin' (refined)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `e91ec092`  
**Description:**
Captures names following legal profession titles, handling potential titles like 'Dr.' or 'Prof.' before the name.

**Content:**
```
\b(?:Rechtsanwalt|Rechtsanwältin)\s+(?:Dr\.?\s+|Prof\.?\s+)?([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Known surnames after legal roles`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `2980bda7`  
**Description:**
Captures specific known surnames (e.g., 'Knoll', 'Kriener', 'Schmid') when they follow legal role indicators, ensuring they are treated as PER.

**Content:**
```
\b(?:Angeklagte|Angeklagten|Kläger|Klägerin|Zeuge|Zeugin|Geschädigte|Geschädigten|Rechtsanwalt|Rechtsanwältin|Vorsitzender|Richter|Richterin|Herr|Herrn)\s+(?:Schäfer|Brückner|Volz|Treber|Knoll|Kriener|Nielsen|Kortbein|Schmid|Söchtig|Kortge|Jacobi|Schödel|Fiamingo|Kelvin|Schilling|Zimmermann|Dölp|Wollensak|Heinkel|Wemheuer|Sost-Scheible|Kirchhof|Paul Kirchhof|Koch|Brune|Grüneberg|Hayen|Lipphaus|Merkel|Eylert|Krumbiegel|Kayser|Jäger|Merzbach|Hacker|Meiser|Becker|Zeng|Merk|Beji Caid Essebsi|Gericke|Franke|Busch|Bender|Augat|Tiemann|Löffler|Schlünder|Schmidt|Schmalz|Melzacks|Edda Redeker|Rosen|Kortbein|Schmid|Söchtig|Kortge|Jacobi|Schödel|Linck|Schultz|Bellay|Leitz|Fieback|Rachor|Cosima|Hoch|Appl|Berger|Quentin|Roloff|Lohmann|Raum|Spinner|St|Br\.|B1|S3|S4|Karaçay|Schramm|Egerer|Kätker|Wismeth|Freudenreich|Schwitzer|Enerji Yapi-Yol Sen|Schwabe|Paffrath|Derstadt|Gallner|Herrmann|Shah|Krasshöfer|Limperg|Mosbacher|Schneider|Niemann|Zwanziger|Brenneisen|Hausmann|Kazele|Hohoff|Roggenbuck|Hamdorf|Grabinski|Krehl|Kosziol|Sunder|Mayen|Seiters|Schlewing|Spaniol|Kirchhoff|Fritz|Vogelsang|Lauer|Mutzbauer|Cierniak|Müller|Ahrendt|Dö.|Widuch|Menezes|Sander|Fischermeier|Hoffmann|Kleinschmidt|Kirschneck|Matter|Kapels|Jostes|Da.|Maksymiw|Schell|Münzberg|D7|Peter Lorsbach|Lorsbach|D1)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 10 | 0 | 10 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `PER` | 0 | 10 | 246 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `61787`) (sent_id: `61787`)


In der Beschwerdesache betreffend die Marke 30 2009 056 266 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 14. September 2017 unter Mitwirkung des Vorsitzenden Richters Knoll sowie der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

**False Positives:**

- `` — similar text (different position): `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Knoll`(PER)
- `Kriener`(PER)
- `Nielsen`(PER)

**Example 1** (doc_id: `61969`) (sent_id: `61969`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2013 036 234.9 hat der 26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 16. Oktober 2017 unter Mitwirkung der Vorsitzenden Richterin Kortge sowie der Richter Jacobi und Schödel beschlossen :

**False Positives:**

- `` — similar text (different position): `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`
- `` — similar text (different position): `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Kortge`(PER)
- `Jacobi`(PER)
- `Schödel`(PER)

**Example 2** (doc_id: `62983`) (sent_id: `62983`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2011 068 984.9 hat der 26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts am 8. Januar 2018 unter Mitwirkung der Vorsitzenden Richterin Kortge sowie der Richter Jacobi und Schödel beschlossen :

**False Positives:**

- `` — similar text (different position): `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`
- `` — similar text (different position): `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `26. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Kortge`(PER)
- `Jacobi`(PER)
- `Schödel`(PER)

**Example 3** (doc_id: `64394`) (sent_id: `64394`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2013 053 470.0 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 9. November 2017 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

**False Positives:**

- `` — similar text (different position): `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Knoll`(PER)
- `Kriener`(PER)
- `Nielsen`(PER)

**Example 4** (doc_id: `64527`) (sent_id: `64527`)


In der Beschwerdesache betreffend die Marke 30 2011 035 856 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `` — similar text (different position): `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 5** (doc_id: `64925`) (sent_id: `64925`)


In der Beschwerdesache betreffend die Marke 30 2010 003 649 hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 12. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `` — similar text (different position): `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

**Example 6** (doc_id: `64967`) (sent_id: `64967`)


In der Beschwerdesache betreffend die Markenanmeldung 30 2015 034 000.6 hat der 25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts auf die mündliche Verhandlung vom 11. Januar 2018 unter Mitwirkung des Vorsitzenden Richters Knoll , der Richterin Kriener und des Richters Dr. Nielsen beschlossen :

**False Positives:**

- `` — similar text (different position): `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `25. Senat ( Marken-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Knoll`(PER)
- `Kriener`(PER)
- `Nielsen`(PER)

**Example 7** (doc_id: `65500`) (sent_id: `65500`)


In der Beschwerdesache betreffend die Designanmeldung .... ( hier : Antrag auf Verfahrenskostenhilfe ) hat der 30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts in der Sitzung vom 5. Oktober 2017 unter Mitwirkung des Vorsitzenden Richters Prof. Dr. Hacker sowie der Richter Merzbach und Dr. Meiser beschlossen :

**False Positives:**

- `` — similar text (different position): `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `30. Senat ( Marken- und Design-Beschwerdesenat ) des Bundespatentgerichts`(ORG)
- `Hacker`(PER)
- `Merzbach`(PER)
- `Meiser`(PER)

</details>

---

## `Initials with dots (standalone context)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `5ae5e583`  
**Description:**
Captures single initials with dots (e.g., 'A.', 'S.') when they appear in contexts suggesting a name, such as after 'Dr.', 'Prof.', 'Herr', or at the start of a sentence followed by a verb.

**Content:**
```
(?:Dr\.?\s+|Prof\.?\s+|Professor\s+|Herr\s+|Herrn\s+|^|\b)([A-Z]\.)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

</details>

---

## `Multi-initial anonymized names`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `76281285`  
**Description:**
Captures multi-initial anonymized names like 'M. D.' or 'A. S.'.

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

## `Names after 'Dr.' or 'Prof.' (standalone)`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `ebde2cdc`  
**Description:**
Captures names following titles like Dr. or Prof. when not part of a longer title sequence, handling both full names and initials.

**Content:**
```
\b(?:Dr\.?\s+|Prof\.?\s+|Professor\s+)([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*|([A-Z]\.)\s+[A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)*)
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


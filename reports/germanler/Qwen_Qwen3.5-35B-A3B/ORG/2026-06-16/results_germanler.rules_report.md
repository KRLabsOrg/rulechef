# Rule Evaluation Report — Qwen/Qwen3.5-35B-A3B

Generated on: 2026-06-16T13:13:34.058058

---

<details>
<summary>Configuration</summary>

Results can be reproduced by running this command: 
```
 python benchmark.py --config reports/germanler/Qwen_Qwen3.5-35B-A3B/ORG/2026-06-16/config.yaml 
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
| Synthesis strategy | per_class |
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
| Best Batch Idx | 21 |
| Best Batch F1 | 0.413053613053613 |
| Best Rules Serialized | [{'id': '99bc8170', 'name': 'Specific Court Departments and Senates', 'description': 'Matches specific German court names and abbreviations that were previously in the long list, now using a more structured approach for common courts and abbreviations.', 'format': 'regex', 'content': '\\b(?:BVerfG|BGH|BFH|BVerwG|BAG|BSG|EuGH|DPMA|BPatG|ZDS|ZIV|GBA|GEW|X-EWIV|GmSOGB|EGMR|NATO|EU|MDK|BA|VCS|Google|RAPAMUNE|nivo|KAEFER|ARD|ZDF|Deutschlandradio|HBV|K\\u00c4V|ver\\.di|DB|BND|RVA|TdL|InWIS|DRV|KOSMICA|Monster\\s*Abwehr\\s*Spray|Haus\\u00e4rzteverband|Bundesvereinigung\\s*der\\s*Arbeitgeberverb\\u00e4nde|Deutsche\\s*Rentenversicherung\\s*Bund|Ausw\\u00e4rtige\\s*Amt|Bundesamt\\s*f\\u00fcr\\s*Migration\\s*und\\s*Fl\\u00fcchtlinge|BMF|Internationalen\\s*Gerichtshofs|Europ\\u00e4ischen\\s*Kommission|Justizkommission\\s*zu\\s*Tunis|Bundeswehr|Fliegerhorst\\s*B\\u00fcchel|St\\u00e4dtischen\\s*Klinikum\\s*K\\.|Kernkraftwerks\\s*Kr\\u00fcmmel|Landesarbeitsgericht\\s*Berlin-Brandenburg|S\\u00e4chsische\\s*Bildungsagentur|Gemeinsamen\\s*Bundesausschusses|Landesaussch\\u00fcsse|Obersten\\s*Verwaltungsgerichts\\s*der\\s*Republik\\s*Bulgarien|Bundeskasse\\s*Halle\\s*/\\s*Saale\\s*-\\s*Dienstsitz\\s*Weiden\\s*/\\s*Oberpfalz|Diakonischen\\s*Werkes\\s*der\\s*Evangelischen\\s*Kirche\\s*in\\s*Deutschland\\s*e\\.\\s*V\\.|Evangelischen\\s*Entwicklungsdienstes\\s*e\\.\\s*V\\.|Haus\\u00e4rztlliche\\s*Vertragsgemeinschaft\\s*Aktiengesellschaft|H\\u00c4VG-Rechenzentrum\\s*AG|H\\u00e4VG-Rechenzentrum\\s*GmbH|B\\s*\\u2026\\s*Patentanwaltsgesellschaft\\s*mbH|C\\s*\\u2026\\s*GmbH|D\\s*P\\s*T\\s*S\\s*GmbH|K\\s*\\u2026\\s*GmbH|B\\s*\\u2026\\s*AG|S\\s*\\u2026|C-\\s*B\\.\\s*V\\.|A-Fonds|BgA\\s*X|InEK|Monster\\s*Abwehr\\s*Spray|Haus\\u00e4rzteverband|Bundesvereinigung\\s*der\\s*Arbeitgeberverb\\u00e4nde|Deutsche\\s*Rentenversicherung\\s*Bund|Ausw\\u00e4rtige\\s*Amt|Bundesamt\\s*f\\u00fcr\\s*Migration\\s*und\\s*Fl\\u00fcchtlinge|BMF|Internationalen\\s*Gerichtshofs|Europ\\u00e4ischen\\s*Kommission|Justizkommission\\s*zu\\s*Tunis|Bundeswehr|Fliegerhorst\\s*B\\u00fcchel|St\\u00e4dtischen\\s*Klinikum\\s*K\\.|Kernkraftwerks\\s*Kr\\u00fcmmel|Landesarbeitsgericht\\s*Berlin-Brandenburg|S\\u00e4chsische\\s*Bildungsagentur|Gemeinsamen\\s*Bundesausschusses|Landesaussch\\u00fcsse|Obersten\\s*Verwaltungsgerichts\\s*der\\s*Republik\\s*Bulgarien|Bundeskasse\\s*Halle\\s*/\\s*Saale\\s*-\\s*Dienstsitz\\s*Weiden\\s*/\\s*Oberpfalz|Diakonischen\\s*Werkes\\s*der\\s*Evangelischen\\s*Kirche\\s*in\\s*Deutschland\\s*e\\.\\s*V\\.|Evangelischen\\s*Entwicklungsdienstes\\s*e\\.\\s*V\\.|Haus\\u00e4rztlliche\\s*Vertragsgemeinschaft\\s*Aktiengesellschaft|H\\u00c4VG-Rechenzentrum\\s*AG|H\\u00e4VG-Rechenzentrum\\s*GmbH|B\\s*\\u2026\\s*Patentanwaltsgesellschaft\\s*mbH|C\\s*\\u2026\\s*GmbH|D\\s*P\\s*T\\s*S\\s*GmbH|K\\s*\\u2026\\s*GmbH|B\\s*\\u2026\\s*AG|S\\s*\\u2026|C-\\s*B\\.\\s*V\\.|A-Fonds|BgA\\s*X|InEK|Bundesregierung|Bundesministerium\\s+der\\s+Finanzen|Bundesamts\\s+f\\u00fcr\\s+Justiz|Justizministerium\\s+des\\s+Landes\\s+Nordrhein-Westfalen|Neurologischen\\s+Klinik\\s+B|Amtsgericht\\s+O\\.|Handwerksverband\\s+Metallbau\\s+und\\s+Feinwerktechnik\\s+Baden-W\\u00fcrttemberg|Industriegewerkschaft\\s+Metall|VEB\\s+[A-Z][a-zA-Z\\s]+|nieders\\u00e4chsische\\s+Landesschulbeh\\u00f6rde|EON-Konzerns|EON-Konzern|dbb\\s+beamtenbund\\s+und\\s+tarifunion|ADAC|Kernkraftwerks\\s+Biblis|Kernkraftwerks\\s+M\\u00fclheim-K\\u00e4rlich|Deutschen\\s*Botschaft|Finanzamt\\s+[A-Za-z\\u00e4\\u00f6\\u00fc\\u00df\\s]+|Bundesministeriums\\s+der\\s+Verteidigung\\s+-\\s+R\\s+II\\s+2\\s+-|Bundesamts\\s+f\\u00fcr\\s+das\\s+Personalmanagement|A\\s+Lebensversicherung\\s+AG|Europ\\u00e4ische\\s+Gerichtshof\\s+f\\u00fcr\\s+Menschenrechte|Th\\u00fcringer\\s+Landessozialgerichts|Deutsche\\s+Patent-\\s+und\\s+Markenamt|Deutschen\\s+Patent-\\s+und\\s+Markenamt|Deutschen\\s+Patent-\\s+und\\s+Markenamtes|K\\u00c4V\\s+Brandenburg|Neue\\s+Richtervereinigung|Generalstaatsanwaltschaft\\s+des\\s+Landes\\s+Schleswig-Holstein|K-Klinik|H\\u00c4VG|Schott|PreussenElektra\\s+GmbH|E\\.\\s+ON\\s+Kernkraft\\s+GmbH|G-Gruppe|Vereinigung\\s+der\\s+kommunalen\\s*Arbeitgeberverb\\u00e4nde|Staatskasse|Kernkraftwerks\\s+Gundremmingen|Bundesministeriums\\s+der\\s+Verteidigung|Bundesministerium\\s+der\\s+Verteidigung|Bundesverfassungsgerichts|Bundesverwaltungsgerichts|Bundesgerichtshofs|Bundesfinanzhofs|Bundesarbeitsgerichts|Bundessozialgerichts|Landgerichts\\s+Darmstadt|Landgerichts\\s+D\\u00fcsseldorf|Landgerichts\\s+Hamburg|Landgerichts\\s+Bremen|Landgerichts\\s+Oldenburg|Landgerichts\\s+Karlsruhe|Landgerichts\\s+Potsdam|Landgerichts\\s+F\\.\\s*\\(\\s*P\\.\\s*\\)|Pf\\u00e4lzische\\s+Oberlandesgericht\\s+Zweibr\\u00fccken|Oberlandesgerichts\\s+M\\u00fcnchen|Oberlandesgerichts\\s+Hamm|Amtsgerichts\\s+O\\.|Finanzamts\\s+[A-Za-z\\u00e4\\u00f6\\u00fc\\u00df\\s]+|Bundesamts\\s+f\\u00fcr\\s+das\\s+Personalmanagement\\s+der\\s+Bundeswehr|Markenstelle\\s*f\\u00fcr\\s*Klasse\\s*\\d+\\s*des\\s*Deutschen\\s*Patent-\\s*und\\s*Markenamts|Patentabteilung\\s*\\d+\\.\\d+\\s*des\\s*Deutschen\\s*Patent-\\s*und\\s*Markenamts|Designabteilung\\s*\\d+\\.\\d+\\s*des\\s*Deutschen\\s*Patent-\\s*und\\s*Markenamts|Markenstelle\\s*f\\u00fcr\\s*Klasse\\s*\\d+\\s*des\\s*DPMA|Patentabteilung\\s*\\d+\\.\\d+\\s*des\\s*DPMA|Designabteilung\\s*\\d+\\.\\d+\\s*des\\s*DPMA|Deutschen\\s*Patent-\\s*und\\s*Markenamts|Deutschen\\s*Patent-\\s*und\\s*Markenamt|Deutsche\\s*Patent-\\s*und\\s*Markenamt|Deutsche\\s*Patent-\\s*und\\s*Markenamts|Statistischen\\s*Bundesamt|Deutschen\\s*Bundestages|Spitzenverband\\s*Bund\\s*der\\s*KKn|Verband\\s*der\\s*Privaten\\s*Krankenversicherung|Deutsche\\s*Krankenhausgesellschaft|VKDA|Gro\\u00dfen\\s*Senat\\s*des\\s*BFH|VIII\\.\\s*Senat\\s*des\\s*BFH|VIII\\.\\s*Senats\\s*des\\s*BFH|LSG\\s+Berlin-Brandenburg|RWE-Konzerns|B\\.\\s*GmbH|w\\s*GmbH|w\\s*Holding\\s*GmbH|P\\s*GmbH|G\\s*GmbH|M-GmbH\\s*&\\s*atypisch\\s*Still|Gewerkschaft\\s*ver\\.di|Kreiskrankenh\\u00e4user\\s*M\\s+und\\s*R|NIVONA|fluege\\.de|CHECK24\\.de|Jaguar|Land\\s*Rover|\\u00d6z\\s*Gaziantep\\s*Dilim\\s*Baklavalari|Wohnungsbau-\\s*und\\s*Kommissionsgesellschaft\\s*Reichenstra\\u00dfen|A-Fonds|B-Fonds|A-AG|B-AG|C-AG|D-AG|E-AG|F-AG|G-AG|H-AG|I-AG|J-AG|K-AG|L-AG|M-AG|N-AG|O-AG|P-AG|Q-AG|R-AG|S-AG|T-AG|U-AG|V-AG|W-AG|X-AG|Y-AG|Z-AG|A\\s*Lebensversicherung\\s*AG|B\\s*Lebensversicherung\\s*AG|C\\s*Lebensversicherung\\s*AG|D\\s*Lebensversicherung\\s*AG|E\\s*Lebensversicherung\\s*AG|F\\s*Lebensversicherung\\s*AG|G\\s*Lebensversicherung\\s*AG|H\\s*Lebensversicherung\\s*AG|I\\s*Lebensversicherung\\s*AG|J\\s*Lebensversicherung\\s*AG|K\\s*Lebensversicherung\\s*AG|L\\s*Lebensversicherung\\s*AG|M\\s*Lebensversicherung\\s*AG|N\\s*Lebensversicherung\\s*AG|O\\s*Lebensversicherung\\s*AG|P\\s*Lebensversicherung\\s*AG|Q\\s*Lebensversicherung\\s*AG|R\\s*Lebensversicherung\\s*AG|S\\s*Lebensversicherung\\s*AG|T\\s*Lebensversicherung\\s*AG|U\\s*Lebensversicherung\\s*AG|V\\s*Lebensversicherung\\s*AG|W\\s*Lebensversicherung\\s*AG|X\\s*Lebensversicherung\\s*AG|Y\\s*Lebensversicherung\\s*AG|Z\\s*Lebensversicherung\\s*AG)\\b', 'priority': 20, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-16T10:52:52.956422', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'ORG'}, 'output_key': 'entities'}, {'id': 'e294a159', 'name': 'Court with Location Genitive', 'description': "Matches court names in genitive case (e.g., 'des Landgerichts') but extracts only the court name, handling compound state names correctly (e.g., 'Sachsen-Anhalt', 'Nordrhein-Westfalen').", 'format': 'regex', 'content': '(?<=\\s(?:des|der|dem|die|den)\\s)(Landgerichts|Oberlandesgerichts|Bundesgerichtshofs|Bundesverwaltungsgerichts|Bundesfinanzhofs|Bundessozialgerichts|Landessozialgerichts|Verwaltungsgerichts|Finanzgerichts|Arbeitsgerichts|Amtsgerichts|Sozialgerichts|Gerichtshofs|Kammer|Amt|Dienst|Beh\\u00f6rde|Ministeriums|Amtes|Bundeswehr|Bundesagentur|Staatsanwaltschaft|Landratsamt|Generalstaatsanwaltschaft|Finanzamt|Klinik|Krankenhaus|Firma|Unternehmen|Vereinigung|Verband|Kanzlei|Kammer|Senat|Abteilung|Stelle|Justizvollzugsanstalt|Patentabteilung|Markenstelle)\\s+(?:[A-Z][a-z\\u00e4\\u00f6\\u00fc\\u00df]+(?:\\s+-\\s+[A-Z][a-z\\u00e4\\u00f6\\u00fc\\u00df]+)?(?:\\s+am\\s+Main|\\s+am\\s+Neckar|\\s+in\\s+der\\s+Freien\\s+und\\s+Hansestadt\\s+Hamburg|\\s+Zweibr\\u00fccken|\\s+Duisburg|\\s+Wiesbaden|\\s+Dresden|\\s+Braunschweig|\\s+Sachsen-Anhalt|\\s+Berlin-Brandenburg|\\s+Berlin|\\s+Frankfurt\\s+am\\s+Main|\\s+H\\u00f6chst|\\s+D\\u00fcsseldorf|\\s+M\\u00fcnchen|\\s+Pfalz|\\s+Saarl\\u00e4ndischen|\\s+Mecklenburg-Vorpommern|\\s+Rheinland-Pfalz|\\s+Nordrhein-Westfalen|\\s+Offenburg|\\s+K\\.|\\s+M\\.|\\s+O\\.|\\s+D\\.|\\s+K\\s+\\u2026|\\s+M\\s+\\u2026|\\s+O\\s+\\u2026|\\s+D\\s+\\u2026|\\s+K\\s+\\u2026\\s+GmbH|\\s+M\\s+\\u2026\\s+GmbH|\\s+O\\s+\\u2026\\s+GmbH|\\s+D\\s+\\u2026\\s+GmbH|\\s+K\\s+\\u2026\\s+AG|\\s+M\\s+\\u2026\\s+AG|\\s+O\\s+\\u2026\\s+AG|\\s+D\\s+\\u2026\\s+AG|\\s+K\\s+\\u2026\\s+mbH|\\s+M\\s+\\u2026\\s+mbH|\\s+O\\s+\\u2026\\s+mbH|\\s+D\\s+\\u2026\\s+mbH)?)', 'priority': 14, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-16T10:52:52.964098', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'ORG'}, 'output_key': 'entities'}, {'id': '5b6bf80a', 'name': 'Organization with Location/Type', 'description': "Matches organizations with specific descriptors like 'Schulzentrum für Technik' or 'Senat des ...', ensuring full capture.", 'format': 'regex', 'content': '\\b[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*\\s+(?:Schulzentrum\\s+f\\u00fcr\\s+[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*|Senat\\s+des\\s+[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*|Kammer\\s+des|Abteilung\\s+des|Zweig|Niederlassung|Gesch\\u00e4ftsf\\u00fchrer|Bundesministerium\\s+f\\u00fcr\\s+[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*|Bundesagentur\\s+f\\u00fcr\\s+[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)\\b', 'priority': 12, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-16T10:52:52.964126', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'ORG'}, 'output_key': 'entities'}, {'id': 'c1f44934', 'name': 'Quoted Organization Names', 'description': "Matches organization names enclosed in German quotation marks, but only when preceded by context indicating an organization (e.g., 'Firma', 'Marke', 'Name', 'der', 'des') to avoid matching product names or streets.", 'format': 'regex', 'content': '(?:Firma|Marke|Name|der|des|bei|von|aus|in)\\s*\\u201e\\s*([A-Z][a-zA-Z\\u00e4\\u00f6\\u00fc\\u00df\\s]+(?:\\s+[A-Z][a-zA-Z\\u00e4\\u00f6\\u00fc\\u00df\\s]+)*)\\s*\\u201c', 'priority': 18, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-16T10:52:52.964141', 'output_template': {'text': '$1', 'start': '$start', 'end': '$end', 'type': 'ORG'}, 'output_key': 'entities'}, {'id': '4d8f42a0', 'name': 'Specific Court Abbreviations and Full Names', 'description': 'Matches common German court abbreviations (BVerfG, BGH, BFH, etc.) with strict word boundaries to avoid partial matches and false positives.', 'format': 'regex', 'content': '\\b(?:BVerfG|BGH|BFH|BVerwG|BAG|BSG|EuGH|DPMA|BPatG|ZDS|ZIV|GBA|GEW|X-EWIV|GmSOGB|EGMR|NATO|EU|MDK|BA|VCS|Google|RAPAMUNE|nivo|KAEFER|ARD|ZDF|Deutschlandradio|HBV|K\\u00c4V|K\\u00c4V\\s+Brandenburg|KJM|EWDE|Luftwaffe|Bundestag|AOK-Bayern|Deutschen\\s*Post|Deutsche\\s*Post|Deutsche\\s*Emissionshandelsstelle|Energie-\\s*und\\s*Klimafonds|Medizinischen\\s*Dienstes\\s*der\\s*Krankenversicherung|Bayerische\\s*Verwaltungsgerichtshof|S\\s*\\u2026|H\\s*AG|I\\s*AG|P\\s*\\u2026\\s*GmbH\\s*&\\s*Co\\.\\s*KG|HSG\\s*Z\\s*\\u2026|V\\.|M\\s*\\u2026|F\\s*AG|Fl\\.\\s*AG|BEAST|Dignitas\\s*Deutschland|FC\\s*Bayern\\s*M\\u00fcnchen|Verwaltungsgericht\\s*Stuttgart|Schleswig-Holsteinische\\s*Verwaltungsgericht|Ausschuss\\s*f\\u00fcr\\s*Arbeit\\s*und\\s*Sozialordnung\\s*des\\s*Bundestages|Bund\\s*Deutscher\\s*Verwaltungsrichter\\s*und\\s*Verwaltungsrichterinnen|BDVR|Gemeinsamen\\s*Senats\\s*der\\s*obersten\\s*Gerichtsh\\u00f6fe\\s*des\\s*Bundes|Gro\\u00dfen\\s*Senats\\s*des\\s*BFH|I\\.\\s*Senates\\s*des\\s*BFH|3\\.\\s*Senats\\s*des\\s*BSG|14\\.\\s*Senat\\s*\\(\\s*Technischer\\s*Beschwerdesenat\\s*\\)\\s*des\\s*Bundespatentgerichts|30\\.\\s*Senat\\s*\\(\\s*Marken-\\s*und\\s*Design-Beschwerdesenat\\s*\\)\\s*des\\s*Bundespatentgerichts|19\\.\\s*Senat\\s*\\(\\s*Technischer\\s*Beschwerdesenat\\s*\\)\\s*des\\s*Bundespatentgerichts|28\\.\\s*Senat\\s*\\(\\s*Marken-Beschwerdesenat\\s*\\)\\s*des\\s*Bundespatentgerichts|25\\.\\s*Senat\\s*\\(\\s*Marken-Beschwerdesenat\\s*\\)\\s*des\\s*Bundespatentgerichts|17\\.\\s*Senat\\s*\\(\\s*Technischer\\s*Beschwerdesenat\\s*\\)\\s*des\\s*Bundespatentgerichts|X\\.\\s*Zivilsenats\\s*des\\s*Bundesgerichtshofs|Schleswig-Holsteinische\\s+Oberlandesgericht|Th\\u00fcringer\\s+Finanzgericht|Nieders\\u00e4chsische\\s+Landesschulbeh\\u00f6rde|Diakonissenhaus\\s+[A-Z]\\.|Deutschen\\s*Patent-\\s*und\\s*Markenamts\\s*,\\s*Markenstelle\\s*f\\u00fcr\\s*Klasse\\s*\\d+|Pr\\u00fcfungsstelle\\s*f\\u00fcr\\s*Klasse\\s*[A-Z0-9]+\\s*des\\s*Deutschen\\s*Patent-\\s*und\\s*Markenamts|Bundesrat|Europ\\u00e4ische\\s*Union|Europ\\u00e4ischer\\s*Gerichtshof|Gerichtshof\\s*der\\s*Europ\\u00e4ischen\\s*Union|Bundesverfassungsgericht|Bundesverfassungsgerichts|Bundesgerichtshof|Europ\\u00e4ischen\\s*Gerichtshofs\\s*f\\u00fcr\\s*Menschenrechte|Europ\\u00e4ischen\\s*Gerichtshof|Europ\\u00e4ischen\\s*Gerichtshofs|Europ\\u00e4ischen\\s*Gerichtshofs\\s*f\\u00fcr\\s*Menschenrechte|Bundesverwaltungsgericht|Bundesarbeitsgericht|Bundessozialgericht|Bundesfinanzhof|Bundespatentgericht|Oberlandesgericht\\s+M\\u00fcnchen|Oberlandesgericht\\s+Hamm|Landgericht\\s+Hamburg|Landgericht\\s+Bremen|Landgericht\\s+Oldenburg|Landgericht\\s+Karlsruhe|Landgericht\\s+Darmstadt|Landgericht\\s+D\\u00fcsseldorf|Landgericht\\s+Potsdam|Landgericht\\s+F\\.\\s*\\(\\s*P\\.\\s*\\)|Pf\\u00e4lzische\\s+Oberlandesgericht\\s+Zweibr\\u00fccken|FG\\s+M\\u00fcnster|FG\\s+M\\u00fcnchen|Staatsanwaltschaft\\s+Duisburg|Landratsamt\\s+D|Kundenniederlassung\\s+Spezial\\s+in\\s+S|1\\.\\s*Senat\\s*des\\s*Nieders\\u00e4chsischen\\s+Anwaltsgerichtshofs|Ersten\\s*Senats\\s*des\\s*Bundesarbeitsgerichts|4\\.\\s*Senat\\s*des\\s*BSG|29\\.\\s*Zivilkammer\\s*des\\s*Landgerichts\\s*K\\u00f6ln|Zivilkammer\\s*des\\s*Landgerichts\\s*Berlin|ausw\\u00e4rtigen\\s*gro\\u00dfen\\s*Strafkammer\\s*des\\s*Landgerichts\\s*Kleve\\s*in\\s*Moers|Justizvollzugsanstalt\\s*Offenburg|Landesarbeitsgericht\\s*Berlin-Brandenburg|S\\u00e4chsische\\s*Bildungsagentur|Gemeinsamen\\s*Bundesausschusses|Landesaussch\\u00fcsse|Obersten\\s*Verwaltungsgerichts\\s*der\\s*Republik\\s*Bulgarien|Bundeskasse\\s*Halle\\s*/\\s*Saale\\s*-\\s*Dienstsitz\\s*Weiden\\s*/\\s*Oberpfalz|Diakonischen\\s*Werkes\\s*der\\s*Evangelischen\\s*Kirche\\s*in\\s*Deutschland\\s*e\\.\\s*V\\.|Evangelischen\\s*Entwicklungsdienstes\\s*e\\.\\s*V\\.|Haus\\u00e4rztlliche\\s*Vertragsgemeinschaft\\s*Aktiengesellschaft|H\\u00c4VG-Rechenzentrum\\s*AG|H\\u00e4VG-Rechenzentrum\\s*GmbH|B\\s*\\u2026\\s*Patentanwaltsgesellschaft\\s*mbH|C\\s*\\u2026\\s*GmbH|D\\s*P\\s*T\\s*S\\s*GmbH|K\\s*\\u2026\\s*GmbH|B\\s*\\u2026\\s*AG|S\\s*\\u2026|C-\\s*B\\.\\s*V\\.|A-Fonds|BgA\\s*X|InEK|ver\\.di|DB|BND|RVA|TdL|InWIS|DRV|KOSMICA|Monster\\s*Abwehr\\s*Spray|Haus\\u00e4rzteverband|Bundesvereinigung\\s*der\\s*Arbeitgeberverb\\u00e4nde|Deutsche\\s*Rentenversicherung\\s*Bund|Ausw\\u00e4rtige\\s*Amt|Bundesamt\\s*f\\u00fcr\\s*Migration\\s*und\\s*Fl\\u00fcchtlinge|BMF|Internationalen\\s*Gerichtshofs|Europ\\u00e4ischen\\s*Kommission|Justizkommission\\s*zu\\s*Tunis|Bundeswehr|Fliegerhorst\\s*B\\u00fcchel|St\\u00e4dtischen\\s*Klinikum\\s*K.|Kernkraftwerks\\s*Kr\\u00fcmmel|Landesarbeitsgericht\\s*Berlin-Brandenburg|S2\\s*\\u2026\\s*GmbH|S\\s*\\u2026|C-\\s*B\\.\\s*V\\.|Software\\s*f\\u00fcr\\s*Ihren\\s*Erfolg|A-Fonds|BgA\\s*X|Bundesregierung|Bundesministerium\\s+der\\s+Finanzen|Bundesamts\\s+f\\u00fcr\\s+Justiz|Justizministerium\\s+des\\s+Landes\\s+Nordrhein-Westfalen|Neurologischen\\s+Klinik\\s+B|Amtsgericht\\s+O\\.|Handwerksverband\\s+Metallbau\\s+und\\s+Feinwerktechnik\\s+Baden-W\\u00fcrttemberg|Industriegewerkschaft\\s+Metall|VEB\\s+[A-Z][a-zA-Z\\s]+|nieders\\u00e4chsische\\s+Landesschulbeh\\u00f6rde|EON-Konzerns|EON-Konzern|dbb\\s+beamtenbund\\s+und\\s+tarifunion|ADAC|Kernkraftwerks\\s+Biblis|Kernkraftwerks\\s+M\\u00fclheim-K\\u00e4rlich|Deutschen\\s*Botschaft|Finanzamt\\s+[A-Za-z\\u00e4\\u00f6\\u00fc\\u00df\\s]+|Bundesministeriums\\s+der\\s+Verteidigung\\s+-\\s+R\\s+II\\s+2\\s+-|Bundesamts\\s+f\\u00fcr\\s+das\\s+Personalmanagement|A\\s+Lebensversicherung\\s+AG|Europ\\u00e4ische\\s+Gerichtshof\\s+f\\u00fcr\\s+Menschenrechte|Th\\u00fcringer\\s+Landessozialgerichts|Deutsche\\s+Patent-\\s+und\\s+Markenamt|Deutschen\\s+Patent-\\s+und\\s+Markenamt|Deutschen\\s+Patent-\\s+und\\s+Markenamtes|K\\u00c4V\\s+Brandenburg|Neue\\s+Richtervereinigung|Generalstaatsanwaltschaft\\s+des\\s+Landes\\s+Schleswig-Holstein|K-Klinik|H\\u00c4VG|Schott|PreussenElektra\\s+GmbH|E\\.\\s+ON\\s+Kernkraft\\s+GmbH|G-Gruppe|Vereinigung\\s+der\\s+kommunalen\\s*Arbeitgeberverb\\u00e4nde|Staatskasse|Kernkraftwerks\\s+Gundremmingen|Bundesministeriums\\s+der\\s+Verteidigung|Bundesministerium\\s+der\\s+Verteidigung|Bundesverfassungsgerichts|Bundesverwaltungsgerichts|Bundesgerichtshofs|Bundesfinanzhofs|Bundesarbeitsgerichts|Bundessozialgerichts|Landgerichts\\s+Darmstadt|Landgerichts\\s+D\\u00fcsseldorf|Landgerichts\\s+Hamburg|Landgerichts\\s+Bremen|Landgerichts\\s+Oldenburg|Landgerichts\\s+Karlsruhe|Landgerichts\\s+Potsdam|Landgerichts\\s+F\\.\\s*\\(\\s*P\\.\\s*\\)|Pf\\u00e4lzische\\s+Oberlandesgericht\\s+Zweibr\\u00fccken|Oberlandesgerichts\\s+M\\u00fcnchen|Oberlandesgerichts\\s+Hamm|Amtsgerichts\\s+O\\.|Finanzamts\\s+[A-Za-z\\u00e4\\u00f6\\u00fc\\u00df\\s]+|Bundesamts\\s+f\\u00fcr\\s+das\\s+Personalmanagement\\s+der\\s+Bundeswehr|Markenstelle\\s*f\\u00fcr\\s*Klasse\\s*\\d+\\s*des\\s*Deutschen\\s*Patent-\\s*und\\s*Markenamts|Patentabteilung\\s*\\d+\\.\\d+\\s*des\\s*Deutschen\\s*Patent-\\s*und\\s*Markenamts|Designabteilung\\s*\\d+\\.\\d+\\s*des\\s*Deutschen\\s*Patent-\\s*und\\s*Markenamts|Markenstelle\\s*f\\u00fcr\\s*Klasse\\s*\\d+\\s*des\\s*DPMA|Patentabteilung\\s*\\d+\\.\\d+\\s*des\\s*DPMA|Designabteilung\\s*\\d+\\.\\d+\\s*des\\s*DPMA|Deutschen\\s*Patent-\\s*und\\s*Markenamts|Deutschen\\s*Patent-\\s*und\\s*Markenamt|Deutsche\\s*Patent-\\s*und\\s*Markenamt|Deutsche\\s*Patent-\\s*und\\s*Markenamts|Statistischen\\s*Bundesamt|Deutschen\\s*Bundestages|Spitzenverband\\s*Bund\\s*der\\s*KKn|Verband\\s*der\\s*Privaten\\s*Krankenversicherung|Deutsche\\s*Krankenhausgesellschaft|VKDA|Gro\\u00dfen\\s*Senat\\s*des\\s*BFH|VIII\\.\\s*Senat\\s*des\\s*BFH|VIII\\.\\s*Senats\\s*des\\s*BFH|LSG\\s+Berlin-Brandenburg|RWE-Konzerns|B\\.\\s*GmbH|w\\s*GmbH|w\\s*Holding\\s*GmbH|P\\s*GmbH|G\\s*GmbH|M-GmbH\\s*&\\s*atypisch\\s*Still|Gewerkschaft\\s*ver\\.di|Kreiskrankenh\\u00e4user\\s*M\\s+und\\s*R|NIVONA|fluege\\.de|CHECK24\\.de|Jaguar|Land\\s*Rover|\\u00d6z\\s*Gaziantep\\s*Dilim\\s*Baklavalari|Wohnungsbau-\\s*und\\s*Kommissionsgesellschaft\\s*Reichenstra\\u00dfen|A-Fonds|B-Fonds|A-AG|B-AG|C-AG|D-AG|E-AG|F-AG|G-AG|H-AG|I-AG|J-AG|K-AG|L-AG|M-AG|N-AG|O-AG|P-AG|Q-AG|R-AG|S-AG|T-AG|U-AG|V-AG|W-AG|X-AG|Y-AG|Z-AG|A\\s*Lebensversicherung\\s*AG|B\\s*Lebensversicherung\\s*AG|C\\s*Lebensversicherung\\s*AG|D\\s*Lebensversicherung\\s*AG|E\\s*Lebensversicherung\\s*AG|F\\s*Lebensversicherung\\s*AG|G\\s*Lebensversicherung\\s*AG|H\\s*Lebensversicherung\\s*AG|I\\s*Lebensversicherung\\s*AG|J\\s*Lebensversicherung\\s*AG|K\\s*Lebensversicherung\\s*AG|L\\s*Lebensversicherung\\s*AG|M\\s*Lebensversicherung\\s*AG|N\\s*Lebensversicherung\\s*AG|O\\s*Lebensversicherung\\s*AG|P\\s*Lebensversicherung\\s*AG|Q\\s*Lebensversicherung\\s*AG|R\\s*Lebensversicherung\\s*AG|S\\s*Lebensversicherung\\s*AG|T\\s*Lebensversicherung\\s*AG|U\\s*Lebensversicherung\\s*AG|V\\s*Lebensversicherung\\s*AG|W\\s*Lebensversicherung\\s*AG|X\\s*Lebensversicherung\\s*AG|Y\\s*Lebensversicherung\\s*AG|Z\\s*Lebensversicherung\\s*AG)\\b', 'priority': 15, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-16T10:52:52.974219', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'ORG'}, 'output_key': 'entities'}, {'id': '70b265e1', 'name': 'Generic Organization Patterns', 'description': "Matches generic organization patterns like 'Firma X GmbH', 'Senat des Y', 'Bundesministerium', 'Landesamt', etc., using structural patterns to generalize.", 'format': 'regex', 'content': '\\b(?:Firma|Gesellschaft|AG|GmbH|KG|e\\.\\s*V\\.|Verband|Vereinigung|Dienst|Amt|Beh\\u00f6rde|Ministerium|Klinik|Krankenhaus|Schule|Schulzentrum|Senat|Kammer|Abteilung|Stelle|Gericht|Landesgericht|Oberlandesgericht|Bundesgericht|Finanzgericht|Arbeitsgericht|Sozialgericht|Verwaltungsgericht|Amtsgericht|Staatsanwaltschaft|Landratsamt|Post|Botschaft|Konsulat|Kanzlei|Korporation|Konzern|Gruppe|Fonds|Institut|Akademie|Hochschule|Universität|Bundesagentur|Bundesamt|Landesamt|Staat|Republik|Union|Kommission|Parlament|Bundestag|Landtag|Senat|Kabinett|Regierung|Verwaltung|Organisation|Institution|Behörde|Dienstleistung|Gewerkschaft|Arbeitsgeberverband|Arbeitnehmerverband|Krankenkasse|Rentenversicherung|Sozialversicherung|Versicherung|Bank|Finanzinstitut|Holding|Investment|Fonds|Stiftung|Organisation|Institution|Behörde|Dienstleistung|Gewerkschaft|Arbeitsgeberverband|Arbeitnehmerverband|Krankenkasse|Rentenversicherung|Sozialversicherung|Versicherung|Bank|Finanzinstitut|Holding|Investment|Fonds|Stiftung)\\s+(?:[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*|\\d+|\\u2026|\\.)', 'priority': 10, 'confidence': 0.5, 'times_applied': 0, 'successes': 0, 'failures': 0, 'created_at': '2026-06-16T10:52:52.973293', 'output_template': {'text': '$0', 'start': '$start', 'end': '$end', 'type': 'ORG'}, 'output_key': 'entities'}] |

<details>
<summary>Results</summary>

| Metric | Value |
|---|---|
| Accuracy (exact match) | 86.8% |
| True Positives | 443 |
| False Positives | 895 |
| False Negatives | 364 |
| Total Gold Entities | 807 |
| Micro Precision | 33.1% |
| Micro Recall | 54.9% |
| Micro F1 | 41.3% |
| Macro F1 | 41.3% |

</details>

---

<details>
<summary>📊 Summary</summary>

| Rule | F1 | Precision | Recall | Total Predicted | True Positives | False Positives |
|---|---|---|---|---|---|---|
| `Court with Location Genitive` | 4.9% | 46.7% | 2.6% | 45 | 21 | 24 |
| `Specific Court Abbreviations and Full Names` | 41.3% | 34.2% | 52.0% | 1228 | 420 | 808 |
| `Specific Court Departments and Senates` | 31.7% | 27.4% | 37.4% | 1101 | 302 | 799 |
| `Generic Organization Patterns` | 0.9% | 10.3% | 0.5% | 39 | 4 | 35 |
| `Organization with Location/Type` | 0.0% | 0.0% | 0.0% | 1 | 0 | 1 |
| `Quoted Organization Names` | 0.0% | 0.0% | 0.0% | 31 | 0 | 31 |

</details>

---

<details>
<summary>🏆 Most Precise Rules</summary>

## `Court with Location Genitive`

**F1:** 0.049 | **Precision:** 0.467 | **Recall:** 0.026  

**Format:** `regex`  
**Rule ID:** `e294a159`  
**Description:**
Matches court names in genitive case (e.g., 'des Landgerichts') but extracts only the court name, handling compound state names correctly (e.g., 'Sachsen-Anhalt', 'Nordrhein-Westfalen').

**Content:**
```
(?<=\s(?:des|der|dem|die|den)\s)(Landgerichts|Oberlandesgerichts|Bundesgerichtshofs|Bundesverwaltungsgerichts|Bundesfinanzhofs|Bundessozialgerichts|Landessozialgerichts|Verwaltungsgerichts|Finanzgerichts|Arbeitsgerichts|Amtsgerichts|Sozialgerichts|Gerichtshofs|Kammer|Amt|Dienst|Beh\u00f6rde|Ministeriums|Amtes|Bundeswehr|Bundesagentur|Staatsanwaltschaft|Landratsamt|Generalstaatsanwaltschaft|Finanzamt|Klinik|Krankenhaus|Firma|Unternehmen|Vereinigung|Verband|Kanzlei|Kammer|Senat|Abteilung|Stelle|Justizvollzugsanstalt|Patentabteilung|Markenstelle)\s+(?:[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+-\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)?(?:\s+am\s+Main|\s+am\s+Neckar|\s+in\s+der\s+Freien\s+und\s+Hansestadt\s+Hamburg|\s+Zweibr\u00fccken|\s+Duisburg|\s+Wiesbaden|\s+Dresden|\s+Braunschweig|\s+Sachsen-Anhalt|\s+Berlin-Brandenburg|\s+Berlin|\s+Frankfurt\s+am\s+Main|\s+H\u00f6chst|\s+D\u00fcsseldorf|\s+M\u00fcnchen|\s+Pfalz|\s+Saarl\u00e4ndischen|\s+Mecklenburg-Vorpommern|\s+Rheinland-Pfalz|\s+Nordrhein-Westfalen|\s+Offenburg|\s+K\.|\s+M\.|\s+O\.|\s+D\.|\s+K\s+\u2026|\s+M\s+\u2026|\s+O\s+\u2026|\s+D\s+\u2026|\s+K\s+\u2026\s+GmbH|\s+M\s+\u2026\s+GmbH|\s+O\s+\u2026\s+GmbH|\s+D\s+\u2026\s+GmbH|\s+K\s+\u2026\s+AG|\s+M\s+\u2026\s+AG|\s+O\s+\u2026\s+AG|\s+D\s+\u2026\s+AG|\s+K\s+\u2026\s+mbH|\s+M\s+\u2026\s+mbH|\s+O\s+\u2026\s+mbH|\s+D\s+\u2026\s+mbH)?)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.467 | 0.026 | 0.049 | 45 | 21 | 24 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 21 | 24 | 742 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53725`) (sent_id: `53725`)


4. Auf die sofortige Beschwerde der Staatsanwaltschaft Düsseldorf hob das Oberlandesgericht Hamm mit angegriffenem Beschluss vom 28. Juli 2015 den Beschluss des Landgerichts Paderborn auf und ordnete die weitere Vollstreckung der Unterbringung des Beschwerdeführers in der Sicherungsverwahrung an .

| Predicted | Gold |
|---|---|
| `Staatsanwaltschaft Düsseldorf` | `Staatsanwaltschaft Düsseldorf` |
| `Landgerichts Paderborn` | `Landgerichts Paderborn` |

**Missed by this rule (FN):**

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

**Example 4** (doc_id: `55533`) (sent_id: `55533`)


Auf die Revision des Angeklagten wird das Urteil des Landgerichts Berlin vom 27. Juni 2017 mit den Feststellungen aufgehoben .

| Predicted | Gold |
|---|---|
| `Landgerichts Berlin` | `Landgerichts Berlin` |

**Example 5** (doc_id: `55722`) (sent_id: `55722`)


Vielmehr hat sich das LSG die Feststellungen aus dem Strafurteil nach eigener Prüfung zu eigen gemacht und dabei neben den Feststellungen aus dem Strafurteil nicht nur den Inhalt des Erstattungsbescheides aus dem Jahr 2013 zur Honorarrückforderung in Höhe von 216 492,33 Euro berücksichtigt , sondern auch den Inhalt der Strafakten ausgewertet , aus denen hervorgeht , dass sich das Urteil des Amtsgerichts Fürth nicht allein auf das Geständnis des Klägers stützt , sondern auch auf das Ergebnis einer umfangreichen Beweisaufnahme .

| Predicted | Gold |
|---|---|
| `Amtsgerichts Fürth` | `Amtsgerichts Fürth` |

**Example 6** (doc_id: `56530`) (sent_id: `56530`)


Die Beklagte beantragt , die Urteile des Bayerischen Landessozialgerichts vom 16. 12. 2015 und des Sozialgerichts München vom 16. 5. 2014 aufzuheben und die Klage abzuweisen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts München` | `Sozialgerichts München` |

**Missed by this rule (FN):**

- `Bayerischen Landessozialgerichts` (ORG)

**Example 7** (doc_id: `56596`) (sent_id: `56596`)


Hierfür wären die Urteile des Landgerichts Cottbus vom ... und des Brandenburgischen Oberlandesgerichts vom ... zu beachten , wonach der Kläger durch Kündigung vom ... 2005 wirksam aus der A-GbR ausgeschlossen wurde .

| Predicted | Gold |
|---|---|
| `Landgerichts Cottbus` | `Landgerichts Cottbus` |

**Missed by this rule (FN):**

- `Brandenburgischen Oberlandesgerichts` (ORG)
- `A-GbR` (ORG)

**Example 8** (doc_id: `56616`) (sent_id: `56616`)


Die Beklagte beantragt , das Urteil des Bayerischen Landessozialgerichts vom 21. Oktober 2015 aufzuheben und das Urteil des Sozialgerichts Nürnberg vom 25. Juni 2013 bezüglich der Beigeladenen zu 1. und 3. aufzuheben und die Klage abzuweisen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts Nürnberg` | `Sozialgerichts Nürnberg` |

**Missed by this rule (FN):**

- `Bayerischen Landessozialgerichts` (ORG)

**Example 9** (doc_id: `56951`) (sent_id: `56951`)


Auf die Revision des Angeklagten wird das Urteil des Landgerichts Ravensburg vom 1. August 2017 , soweit es ihn betrifft , mit den Feststellungen aufgehoben .

| Predicted | Gold |
|---|---|
| `Landgerichts Ravensburg` | `Landgerichts Ravensburg` |

**Example 10** (doc_id: `57130`) (sent_id: `57130`)


1. Auf die Revision des Angeklagten gegen das Urteil des Landgerichts Aachen vom 7. Juli 2016 wird

| Predicted | Gold |
|---|---|
| `Landgerichts Aachen` | `Landgerichts Aachen` |

**Example 11** (doc_id: `57570`) (sent_id: `57570`)


Die Klägerin beantragt , den Beschluss des Thüringer Landessozialgerichts vom 21. Juli 2016 und das Urteil des Sozialgerichts Meiningen vom 7. Januar 2015 aufzuheben sowie den Bescheid der Beklagten vom 15. April 2013 in der Gestalt des Widerspruchsbescheids vom 17. Mai 2013 abzuändern und die Beklagte zu verurteilen , ihr für die Zeit vom 1. Januar bis 28. März 2012 höheres Insolvenzgeld zu zahlen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts Meiningen` | `Sozialgerichts Meiningen` |

**Missed by this rule (FN):**

- `Thüringer Landessozialgerichts` (ORG)

**Example 12** (doc_id: `58013`) (sent_id: `58013`)


das Urteil des Bayerischen Landessozialgerichts vom 3. Juni 2016 und den Gerichtsbescheid des Sozialgerichts Nürnberg vom 30. August 2013 sowie den Bescheid der Beklagten vom 19. März 2012 und den Widerspruchsbescheid vom 18. Juni 2012 aufzuheben und die Beklagte zu verurteilen , unter Rücknahme des Bescheides vom 4. Februar 2005 und des Widerspruchsbescheides vom 22. April 2005 die Zeit vom 1. September 1973 bis 30. Juni 1990 als Zeit der Zugehörigkeit zum Zusatzversorgungssystem der technischen Intelligenz und die hierin erzielten Arbeitsentgelte festzustellen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts Nürnberg` | `Sozialgerichts Nürnberg` |

**Missed by this rule (FN):**

- `Bayerischen Landessozialgerichts` (ORG)

**Example 13** (doc_id: `58301`) (sent_id: `58301`)


Der Beklagte beantragt , das Urteil des Sächsischen Landessozialgerichts vom 9. Februar 2017 aufzuheben und die Berufungen der Kläger gegen das Urteil des Sozialgerichts Dresden vom 10. Februar 2014 zurückzuweisen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts Dresden` | `Sozialgerichts Dresden` |

**Missed by this rule (FN):**

- `Sächsischen Landessozialgerichts` (ORG)

**Example 14** (doc_id: `58405`) (sent_id: `58405`)


3. Mit Schriftsatz vom 8. März 2018 beantragt die Beschwerdeführerin durch ihren Bevollmächtigten , " die Vollstreckbarkeit " der Beschlüsse des Landgerichts Potsdam vom 11. März 2014 und vom " 20. Juli 2017 " ( gemeint wohl 17. Juli 2017 ) vorläufig auszusetzen .

| Predicted | Gold |
|---|---|
| `Landgerichts Potsdam` | `Landgerichts Potsdam` |

**Example 15** (doc_id: `59356`) (sent_id: `59356`)


Über die sofortige Beschwerde der Nebenklägerin K. T. gegen die im Urteil des Landgerichts Frankfurt am Main vom 15. August 2017 getroffene Kostenentscheidung hat das Oberlandesgericht Frankfurt am Main zu entscheiden .

| Predicted | Gold |
|---|---|
| `Landgerichts Frankfurt am Main` | `Landgerichts Frankfurt am Main` |

**Missed by this rule (FN):**

- `K. T.` (PER)
- `Oberlandesgericht Frankfurt am Main` (ORG)

**Example 16** (doc_id: `59568`) (sent_id: `59568`)


Die Revision des Angeklagten gegen das Urteil des Landgerichts Essen vom 10. April 2017 wird als unbegründet verworfen , da die Nachprüfung des Urteils auf Grund der Revisionsrechtfertigung keinen Rechtsfehler zum Nachteil des Angeklagten ergeben hat ( § 349 Abs. 2 StPO ) .

| Predicted | Gold |
|---|---|
| `Landgerichts Essen` | `Landgerichts Essen` |

**Missed by this rule (FN):**

- `§ 349 Abs. 2 StPO` (NRM)

**Example 17** (doc_id: `59658`) (sent_id: `59658`)


A. I. 1. Der Beschwerdeführer ist auf Grundlage des Urteils des Landgerichts Lübeck vom 7. Oktober 2014 gemäß § 63 StGB wegen Mordes in einem psychiatrischen Krankenhaus der AMEOS Krankenhausgesellschaft Holstein mbH untergebracht , nachdem er im schuldunfähigen Zustand auf Grund einer wahnhaften Störung im Januar 2014 seine vierjährige Tochter und seinen sechs Jahre alten Sohn getötet hatte .

| Predicted | Gold |
|---|---|
| `Landgerichts Lübeck` | `Landgerichts Lübeck` |

**Missed by this rule (FN):**

- `§ 63 StGB` (NRM)
- `AMEOS Krankenhausgesellschaft Holstein mbH` (ORG)

**Example 18** (doc_id: `59742`) (sent_id: `59742`)


1. Dem Angeklagten A. wird auf seinen Antrag Wiedereinsetzung in den vorigen Stand wegen Versäumung der Frist zur Begründung der Revision gegen das Urteil des Landgerichts Göttingen vom 30. Mai 2017 gewährt .

| Predicted | Gold |
|---|---|
| `Landgerichts Göttingen` | `Landgerichts Göttingen` |

**Missed by this rule (FN):**

- `A.` (PER)

**Example 19** (doc_id: `59823`) (sent_id: `59823`)


Erstens würden anderen Gefangenen in vergleichbaren Situationen vollzugsöffnende Maßnahmen gewährt , und zweitens habe das Oberlandesgericht in einem anderen Verfahren vertreten , dass auch die Justizvollzugsanstalt Bruchsal Möglichkeiten der Diagnose vorhalten müsse und der Grundsatz der bestmöglichen Sachaufklärung die Einholung gutachterlicher Expertise gebiete .

| Predicted | Gold |
|---|---|
| `Justizvollzugsanstalt Bruchsal` | `Justizvollzugsanstalt Bruchsal` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `54385`) (sent_id: `54385`)


die Urteile des Landessozialgerichts Sachsen-Anhalt vom 9. März 2017 und des Sozialgerichts Dessau-Roßlau vom 2. Dezember 2013 sowie den Bescheid des Beklagten vom 16. Februar 2010 in der Gestalt des Widerspruchsbescheids vom 31. Mai 2010 aufzuheben .

**False Positives:**

- `Landessozialgerichts Sachsen` — partial — pred is substring of gold: `Landessozialgerichts Sachsen-Anhalt`
- `Sozialgerichts Dessau` — partial — pred is substring of gold: `Sozialgerichts Dessau-Roßlau`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Landessozialgerichts Sachsen-Anhalt`(ORG)
- `Sozialgerichts Dessau-Roßlau`(ORG)

**Example 1** (doc_id: `54438`) (sent_id: `54438`)


Der Haftbefehl des Amtsgerichts Neu-Ulm vom 31. Juli 2017 könne unter Berücksichtigung der Bedeutung und Tragweite des Grundrechts auf Freiheit der Person des Beschwerdeführers keinen Bestand haben .

**False Positives:**

- `Amtsgerichts Neu` — partial — pred is substring of gold: `Amtsgerichts Neu-Ulm`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Amtsgerichts Neu-Ulm`(ORG)

**Example 2** (doc_id: `55082`) (sent_id: `55082`)


Demgemäß hat der Senat Schüler , die im häuslichen Bereich unterrichtsvorbereitend ein Werkstück erstellen ( BSG vom 1. 2. 1979 - 2 RU 107/77 - SozR 2200 § 539 Nr 54 ) , ebenso wenig für versichert erachtet wie solche , die für die schulische Foto-AG in der Altstadt ohne weitere Aufsicht fotografieren ( BSG vom 30. 5. 1988 - 2 RU 5/88 - Juris ) .

**False Positives:**

- `Senat Schüler` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG vom 1. 2. 1979 - 2 RU 107/77 - SozR 2200 § 539 Nr 54`(RS)
- `Foto-AG`(ORG)
- `BSG vom 30. 5. 1988 - 2 RU 5/88 - Juris`(RS)

**Example 3** (doc_id: `55122`) (sent_id: `55122`)


So bietet beispielsweise die Firma Pointer „ Wohlfühlfarben für die Wohnung “ an ; in einem der Anmelderin übersandten Internetausdruck heißt es hierzu : „ Farben gezielt einsetzen .

**False Positives:**

- `Firma Pointer` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `55265`) (sent_id: `55265`)


Die Revision des Klägers gegen das Urteil des Landessozialgerichts Rheinland-Pfalz vom 9. Juni 2016 wird zurückgewiesen .

**False Positives:**

- `Landessozialgerichts Rheinland` — partial — pred is substring of gold: `Landessozialgerichts Rheinland-Pfalz`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Landessozialgerichts Rheinland-Pfalz`(ORG)

**Example 5** (doc_id: `55511`) (sent_id: `55511`)


Die Beschwerde der Antragstellerin gegen den Beschluss des Finanzgerichts Hamburg vom 15. August 2016 1 V 41/16 wird als unbegründet zurückgewiesen .

**False Positives:**

- `Finanzgerichts Hamburg` — partial — pred is substring of gold: `Beschluss des Finanzgerichts Hamburg vom 15. August 2016 1 V 41/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Beschluss des Finanzgerichts Hamburg vom 15. August 2016 1 V 41/16`(RS)

**Example 6** (doc_id: `55659`) (sent_id: `55659`)


Die Beschwerde des Klägers wegen Nichtzulassung der Revision gegen das Urteil des Finanzgerichts Münster vom 12. September 2017 15 K 3562/14 U wird als unbegründet zurückgewiesen .

**False Positives:**

- `Finanzgerichts Münster` — partial — pred is substring of gold: `Urteil des Finanzgerichts Münster vom 12. September 2017 15 K 3562/14 U`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Finanzgerichts Münster vom 12. September 2017 15 K 3562/14 U`(RS)

**Example 7** (doc_id: `55764`) (sent_id: `55764`)


Schließlich vertrieb die Firma Köhnlein am 10. November 2009 über das Internet ein „ Drei-Bolzen-Sicherheitsautomatikschloss mit A-Öffner “ , das sich durch das automatische Öffnen mit dem Komfort der automatischen Verriegelung auszeichnet ( Anlagen 8a und 8c ) .

**False Positives:**

- `Firma Köhnlein` — partial — gold is substring of pred: `Köhnlein`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Köhnlein`(ORG)

**Example 8** (doc_id: `56170`) (sent_id: `56170`)


2. Die Berufung des Beklagten gegen das Urteil des Arbeitsgerichts Oberhausen vom 15. Dezember 2016 - 4 Ca 1318/16 - wird zurückgewiesen .

**False Positives:**

- `Arbeitsgerichts Oberhausen` — partial — pred is substring of gold: `Urteil des Arbeitsgerichts Oberhausen vom 15. Dezember 2016 - 4 Ca 1318/16 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Arbeitsgerichts Oberhausen vom 15. Dezember 2016 - 4 Ca 1318/16 -`(RS)

**Example 9** (doc_id: `56331`) (sent_id: `56331`)


Auf die Berufung der Beklagten wird - unter Zurückweisung der Anschlussberufung des Klägers - das Urteil des Arbeitsgerichts Bonn vom 11. November 2015 - 4 Ca 1615/15 - abgeändert und die Klage abgewiesen .

**False Positives:**

- `Arbeitsgerichts Bonn` — partial — pred is substring of gold: `Urteil des Arbeitsgerichts Bonn vom 11. November 2015 - 4 Ca 1615/15 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Arbeitsgerichts Bonn vom 11. November 2015 - 4 Ca 1615/15 -`(RS)

**Example 10** (doc_id: `56780`) (sent_id: `56780`)


Die Revision der Beklagten gegen das Urteil des Finanzgerichts Berlin-Brandenburg vom 18. Januar 2017 3 K 3219/16 wird als unbegründet zurückgewiesen .

**False Positives:**

- `Finanzgerichts Berlin` — partial — pred is substring of gold: `Urteil des Finanzgerichts Berlin-Brandenburg vom 18. Januar 2017 3 K 3219/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Finanzgerichts Berlin-Brandenburg vom 18. Januar 2017 3 K 3219/16`(RS)

**Example 11** (doc_id: `56966`) (sent_id: `56966`)


Dies betrifft sowohl die angeordneten Tätigkeiten in der Abteilung Standesamt und Gerichtliche Angelegenheiten als auch diejenigen für die Visaabteilung .

**False Positives:**

- `Abteilung Standesamt` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 12** (doc_id: `57051`) (sent_id: `57051`)


Die Beschwerde der Klägerin gegen die Nichtzulassung der Revision im Urteil des Landessozialgerichts Nordrhein-Westfalen vom 22. Juni 2017 wird zurückgewiesen .

**False Positives:**

- `Landessozialgerichts Nordrhein` — partial — pred is substring of gold: `Landessozialgerichts Nordrhein-Westfalen`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Landessozialgerichts Nordrhein-Westfalen`(ORG)

**Example 13** (doc_id: `57287`) (sent_id: `57287`)


Die Beschwerde des Klägers gegen die Nichtzulassung der Revision in dem Beschluss des Landessozialgerichts Niedersachsen-Bremen vom 9. November 2017 wird als unzulässig verworfen .

**False Positives:**

- `Landessozialgerichts Niedersachsen` — partial — pred is substring of gold: `Landessozialgerichts Niedersachsen-Bremen`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Landessozialgerichts Niedersachsen-Bremen`(ORG)

**Example 14** (doc_id: `57361`) (sent_id: `57361`)


Auf die Revision der Beklagten wird das Urteil des Landessozialgerichts Nordrhein-Westfalen vom 17. Dezember 2014 aufgehoben , soweit das Bestehen von Rentenversicherungspflicht des Klägers wegen Beschäftigung bei der Beigeladenen zu 1. für die Zeit ab 10. Juli 2008 verneint wird .

**False Positives:**

- `Landessozialgerichts Nordrhein` — partial — pred is substring of gold: `Landessozialgerichts Nordrhein-Westfalen`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Landessozialgerichts Nordrhein-Westfalen`(ORG)

**Example 15** (doc_id: `57953`) (sent_id: `57953`)


2. Die Berufung des Klägers gegen das Urteil des Arbeitsgerichts Dortmund vom 27. März 2014 - 6 Ca 3695/11 - wird zurückgewiesen .

**False Positives:**

- `Arbeitsgerichts Dortmund` — partial — pred is substring of gold: `Urteil des Arbeitsgerichts Dortmund vom 27. März 2014 - 6 Ca 3695/11 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Arbeitsgerichts Dortmund vom 27. März 2014 - 6 Ca 3695/11 -`(RS)

**Example 16** (doc_id: `58297`) (sent_id: `58297`)


In Bezug auf den gerügten Haftbefehl des Amtsgerichts Neu-Ulm vom 31. Juli 2017 sei die Verfassungsbeschwerde wegen des Grundsatzes der Subsidiarität der Verfassungsbeschwerde hingegen unzulässig , da eine abschließende Sachprüfung durch das Oberlandesgericht München noch nicht stattgefunden habe .

**False Positives:**

- `Amtsgerichts Neu` — partial — pred is substring of gold: `Amtsgerichts Neu-Ulm`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Amtsgerichts Neu-Ulm`(ORG)
- `Oberlandesgericht München`(ORG)

**Example 17** (doc_id: `58399`) (sent_id: `58399`)


Die Revision der Klägerin gegen das Urteil des Finanzgerichts Hamburg vom 21. April 2017 4 K 186/16 wird als unbegründet zurückgewiesen .

**False Positives:**

- `Finanzgerichts Hamburg` — partial — pred is substring of gold: `Urteil des Finanzgerichts Hamburg vom 21. April 2017 4 K 186/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Finanzgerichts Hamburg vom 21. April 2017 4 K 186/16`(RS)

**Example 18** (doc_id: `58546`) (sent_id: `58546`)


Auf die Revision des Beklagten wird das Urteil des Finanzgerichts München vom 29. März 2017 3 K 2565/16 aufgehoben .

**False Positives:**

- `Finanzgerichts München` — partial — pred is substring of gold: `Urteil des Finanzgerichts München vom 29. März 2017 3 K 2565/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Finanzgerichts München vom 29. März 2017 3 K 2565/16`(RS)

**Example 19** (doc_id: `58915`) (sent_id: `58915`)


Die Beschwerde gegen die Nichtzulassung der Revision in dem Urteil des 7. Zivilsenats des Oberlandesgerichts Frankfurt am Main vom 8. Juni 2017 wird auf Kosten des Klägers als unzulässig verworfen .

**False Positives:**

- `Oberlandesgerichts Frankfurt am Main` — partial — pred is substring of gold: `Urteil des 7. Zivilsenats des Oberlandesgerichts Frankfurt am Main`

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

**Example 21** (doc_id: `59091`) (sent_id: `59091`)


Auf die Revision des Klägers wird der Beschluss des Landessozialgerichts Baden-Württemberg vom 9. Februar 2015 aufgehoben und die Sache zur erneuten Verhandlung und Entscheidung an dieses Gericht zurückverwiesen .

**False Positives:**

- `Landessozialgerichts Baden` — partial — pred is substring of gold: `Landessozialgerichts Baden-Württemberg`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Landessozialgerichts Baden-Württemberg`(ORG)

**Example 22** (doc_id: `59490`) (sent_id: `59490`)


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

## `Specific Court Abbreviations and Full Names`

**F1:** 0.413 | **Precision:** 0.342 | **Recall:** 0.520  

**Format:** `regex`  
**Rule ID:** `4d8f42a0`  
**Description:**
Matches common German court abbreviations (BVerfG, BGH, BFH, etc.) with strict word boundaries to avoid partial matches and false positives.

**Content:**
```
\b(?:BVerfG|BGH|BFH|BVerwG|BAG|BSG|EuGH|DPMA|BPatG|ZDS|ZIV|GBA|GEW|X-EWIV|GmSOGB|EGMR|NATO|EU|MDK|BA|VCS|Google|RAPAMUNE|nivo|KAEFER|ARD|ZDF|Deutschlandradio|HBV|K\u00c4V|K\u00c4V\s+Brandenburg|KJM|EWDE|Luftwaffe|Bundestag|AOK-Bayern|Deutschen\s*Post|Deutsche\s*Post|Deutsche\s*Emissionshandelsstelle|Energie-\s*und\s*Klimafonds|Medizinischen\s*Dienstes\s*der\s*Krankenversicherung|Bayerische\s*Verwaltungsgerichtshof|S\s*\u2026|H\s*AG|I\s*AG|P\s*\u2026\s*GmbH\s*&\s*Co\.\s*KG|HSG\s*Z\s*\u2026|V\.|M\s*\u2026|F\s*AG|Fl\.\s*AG|BEAST|Dignitas\s*Deutschland|FC\s*Bayern\s*M\u00fcnchen|Verwaltungsgericht\s*Stuttgart|Schleswig-Holsteinische\s*Verwaltungsgericht|Ausschuss\s*f\u00fcr\s*Arbeit\s*und\s*Sozialordnung\s*des\s*Bundestages|Bund\s*Deutscher\s*Verwaltungsrichter\s*und\s*Verwaltungsrichterinnen|BDVR|Gemeinsamen\s*Senats\s*der\s*obersten\s*Gerichtsh\u00f6fe\s*des\s*Bundes|Gro\u00dfen\s*Senats\s*des\s*BFH|I\.\s*Senates\s*des\s*BFH|3\.\s*Senats\s*des\s*BSG|14\.\s*Senat\s*\(\s*Technischer\s*Beschwerdesenat\s*\)\s*des\s*Bundespatentgerichts|30\.\s*Senat\s*\(\s*Marken-\s*und\s*Design-Beschwerdesenat\s*\)\s*des\s*Bundespatentgerichts|19\.\s*Senat\s*\(\s*Technischer\s*Beschwerdesenat\s*\)\s*des\s*Bundespatentgerichts|28\.\s*Senat\s*\(\s*Marken-Beschwerdesenat\s*\)\s*des\s*Bundespatentgerichts|25\.\s*Senat\s*\(\s*Marken-Beschwerdesenat\s*\)\s*des\s*Bundespatentgerichts|17\.\s*Senat\s*\(\s*Technischer\s*Beschwerdesenat\s*\)\s*des\s*Bundespatentgerichts|X\.\s*Zivilsenats\s*des\s*Bundesgerichtshofs|Schleswig-Holsteinische\s+Oberlandesgericht|Th\u00fcringer\s+Finanzgericht|Nieders\u00e4chsische\s+Landesschulbeh\u00f6rde|Diakonissenhaus\s+[A-Z]\.|Deutschen\s*Patent-\s*und\s*Markenamts\s*,\s*Markenstelle\s*f\u00fcr\s*Klasse\s*\d+|Pr\u00fcfungsstelle\s*f\u00fcr\s*Klasse\s*[A-Z0-9]+\s*des\s*Deutschen\s*Patent-\s*und\s*Markenamts|Bundesrat|Europ\u00e4ische\s*Union|Europ\u00e4ischer\s*Gerichtshof|Gerichtshof\s*der\s*Europ\u00e4ischen\s*Union|Bundesverfassungsgericht|Bundesverfassungsgerichts|Bundesgerichtshof|Europ\u00e4ischen\s*Gerichtshofs\s*f\u00fcr\s*Menschenrechte|Europ\u00e4ischen\s*Gerichtshof|Europ\u00e4ischen\s*Gerichtshofs|Europ\u00e4ischen\s*Gerichtshofs\s*f\u00fcr\s*Menschenrechte|Bundesverwaltungsgericht|Bundesarbeitsgericht|Bundessozialgericht|Bundesfinanzhof|Bundespatentgericht|Oberlandesgericht\s+M\u00fcnchen|Oberlandesgericht\s+Hamm|Landgericht\s+Hamburg|Landgericht\s+Bremen|Landgericht\s+Oldenburg|Landgericht\s+Karlsruhe|Landgericht\s+Darmstadt|Landgericht\s+D\u00fcsseldorf|Landgericht\s+Potsdam|Landgericht\s+F\.\s*\(\s*P\.\s*\)|Pf\u00e4lzische\s+Oberlandesgericht\s+Zweibr\u00fccken|FG\s+M\u00fcnster|FG\s+M\u00fcnchen|Staatsanwaltschaft\s+Duisburg|Landratsamt\s+D|Kundenniederlassung\s+Spezial\s+in\s+S|1\.\s*Senat\s*des\s*Nieders\u00e4chsischen\s+Anwaltsgerichtshofs|Ersten\s*Senats\s*des\s*Bundesarbeitsgerichts|4\.\s*Senat\s*des\s*BSG|29\.\s*Zivilkammer\s*des\s*Landgerichts\s*K\u00f6ln|Zivilkammer\s*des\s*Landgerichts\s*Berlin|ausw\u00e4rtigen\s*gro\u00dfen\s*Strafkammer\s*des\s*Landgerichts\s*Kleve\s*in\s*Moers|Justizvollzugsanstalt\s*Offenburg|Landesarbeitsgericht\s*Berlin-Brandenburg|S\u00e4chsische\s*Bildungsagentur|Gemeinsamen\s*Bundesausschusses|Landesaussch\u00fcsse|Obersten\s*Verwaltungsgerichts\s*der\s*Republik\s*Bulgarien|Bundeskasse\s*Halle\s*/\s*Saale\s*-\s*Dienstsitz\s*Weiden\s*/\s*Oberpfalz|Diakonischen\s*Werkes\s*der\s*Evangelischen\s*Kirche\s*in\s*Deutschland\s*e\.\s*V\.|Evangelischen\s*Entwicklungsdienstes\s*e\.\s*V\.|Haus\u00e4rztlliche\s*Vertragsgemeinschaft\s*Aktiengesellschaft|H\u00c4VG-Rechenzentrum\s*AG|H\u00e4VG-Rechenzentrum\s*GmbH|B\s*\u2026\s*Patentanwaltsgesellschaft\s*mbH|C\s*\u2026\s*GmbH|D\s*P\s*T\s*S\s*GmbH|K\s*\u2026\s*GmbH|B\s*\u2026\s*AG|S\s*\u2026|C-\s*B\.\s*V\.|A-Fonds|BgA\s*X|InEK|ver\.di|DB|BND|RVA|TdL|InWIS|DRV|KOSMICA|Monster\s*Abwehr\s*Spray|Haus\u00e4rzteverband|Bundesvereinigung\s*der\s*Arbeitgeberverb\u00e4nde|Deutsche\s*Rentenversicherung\s*Bund|Ausw\u00e4rtige\s*Amt|Bundesamt\s*f\u00fcr\s*Migration\s*und\s*Fl\u00fcchtlinge|BMF|Internationalen\s*Gerichtshofs|Europ\u00e4ischen\s*Kommission|Justizkommission\s*zu\s*Tunis|Bundeswehr|Fliegerhorst\s*B\u00fcchel|St\u00e4dtischen\s*Klinikum\s*K.|Kernkraftwerks\s*Kr\u00fcmmel|Landesarbeitsgericht\s*Berlin-Brandenburg|S2\s*\u2026\s*GmbH|S\s*\u2026|C-\s*B\.\s*V\.|Software\s*f\u00fcr\s*Ihren\s*Erfolg|A-Fonds|BgA\s*X|Bundesregierung|Bundesministerium\s+der\s+Finanzen|Bundesamts\s+f\u00fcr\s+Justiz|Justizministerium\s+des\s+Landes\s+Nordrhein-Westfalen|Neurologischen\s+Klinik\s+B|Amtsgericht\s+O\.|Handwerksverband\s+Metallbau\s+und\s+Feinwerktechnik\s+Baden-W\u00fcrttemberg|Industriegewerkschaft\s+Metall|VEB\s+[A-Z][a-zA-Z\s]+|nieders\u00e4chsische\s+Landesschulbeh\u00f6rde|EON-Konzerns|EON-Konzern|dbb\s+beamtenbund\s+und\s+tarifunion|ADAC|Kernkraftwerks\s+Biblis|Kernkraftwerks\s+M\u00fclheim-K\u00e4rlich|Deutschen\s*Botschaft|Finanzamt\s+[A-Za-z\u00e4\u00f6\u00fc\u00df\s]+|Bundesministeriums\s+der\s+Verteidigung\s+-\s+R\s+II\s+2\s+-|Bundesamts\s+f\u00fcr\s+das\s+Personalmanagement|A\s+Lebensversicherung\s+AG|Europ\u00e4ische\s+Gerichtshof\s+f\u00fcr\s+Menschenrechte|Th\u00fcringer\s+Landessozialgerichts|Deutsche\s+Patent-\s+und\s+Markenamt|Deutschen\s+Patent-\s+und\s+Markenamt|Deutschen\s+Patent-\s+und\s+Markenamtes|K\u00c4V\s+Brandenburg|Neue\s+Richtervereinigung|Generalstaatsanwaltschaft\s+des\s+Landes\s+Schleswig-Holstein|K-Klinik|H\u00c4VG|Schott|PreussenElektra\s+GmbH|E\.\s+ON\s+Kernkraft\s+GmbH|G-Gruppe|Vereinigung\s+der\s+kommunalen\s*Arbeitgeberverb\u00e4nde|Staatskasse|Kernkraftwerks\s+Gundremmingen|Bundesministeriums\s+der\s+Verteidigung|Bundesministerium\s+der\s+Verteidigung|Bundesverfassungsgerichts|Bundesverwaltungsgerichts|Bundesgerichtshofs|Bundesfinanzhofs|Bundesarbeitsgerichts|Bundessozialgerichts|Landgerichts\s+Darmstadt|Landgerichts\s+D\u00fcsseldorf|Landgerichts\s+Hamburg|Landgerichts\s+Bremen|Landgerichts\s+Oldenburg|Landgerichts\s+Karlsruhe|Landgerichts\s+Potsdam|Landgerichts\s+F\.\s*\(\s*P\.\s*\)|Pf\u00e4lzische\s+Oberlandesgericht\s+Zweibr\u00fccken|Oberlandesgerichts\s+M\u00fcnchen|Oberlandesgerichts\s+Hamm|Amtsgerichts\s+O\.|Finanzamts\s+[A-Za-z\u00e4\u00f6\u00fc\u00df\s]+|Bundesamts\s+f\u00fcr\s+das\s+Personalmanagement\s+der\s+Bundeswehr|Markenstelle\s*f\u00fcr\s*Klasse\s*\d+\s*des\s*Deutschen\s*Patent-\s*und\s*Markenamts|Patentabteilung\s*\d+\.\d+\s*des\s*Deutschen\s*Patent-\s*und\s*Markenamts|Designabteilung\s*\d+\.\d+\s*des\s*Deutschen\s*Patent-\s*und\s*Markenamts|Markenstelle\s*f\u00fcr\s*Klasse\s*\d+\s*des\s*DPMA|Patentabteilung\s*\d+\.\d+\s*des\s*DPMA|Designabteilung\s*\d+\.\d+\s*des\s*DPMA|Deutschen\s*Patent-\s*und\s*Markenamts|Deutschen\s*Patent-\s*und\s*Markenamt|Deutsche\s*Patent-\s*und\s*Markenamt|Deutsche\s*Patent-\s*und\s*Markenamts|Statistischen\s*Bundesamt|Deutschen\s*Bundestages|Spitzenverband\s*Bund\s*der\s*KKn|Verband\s*der\s*Privaten\s*Krankenversicherung|Deutsche\s*Krankenhausgesellschaft|VKDA|Gro\u00dfen\s*Senat\s*des\s*BFH|VIII\.\s*Senat\s*des\s*BFH|VIII\.\s*Senats\s*des\s*BFH|LSG\s+Berlin-Brandenburg|RWE-Konzerns|B\.\s*GmbH|w\s*GmbH|w\s*Holding\s*GmbH|P\s*GmbH|G\s*GmbH|M-GmbH\s*&\s*atypisch\s*Still|Gewerkschaft\s*ver\.di|Kreiskrankenh\u00e4user\s*M\s+und\s*R|NIVONA|fluege\.de|CHECK24\.de|Jaguar|Land\s*Rover|\u00d6z\s*Gaziantep\s*Dilim\s*Baklavalari|Wohnungsbau-\s*und\s*Kommissionsgesellschaft\s*Reichenstra\u00dfen|A-Fonds|B-Fonds|A-AG|B-AG|C-AG|D-AG|E-AG|F-AG|G-AG|H-AG|I-AG|J-AG|K-AG|L-AG|M-AG|N-AG|O-AG|P-AG|Q-AG|R-AG|S-AG|T-AG|U-AG|V-AG|W-AG|X-AG|Y-AG|Z-AG|A\s*Lebensversicherung\s*AG|B\s*Lebensversicherung\s*AG|C\s*Lebensversicherung\s*AG|D\s*Lebensversicherung\s*AG|E\s*Lebensversicherung\s*AG|F\s*Lebensversicherung\s*AG|G\s*Lebensversicherung\s*AG|H\s*Lebensversicherung\s*AG|I\s*Lebensversicherung\s*AG|J\s*Lebensversicherung\s*AG|K\s*Lebensversicherung\s*AG|L\s*Lebensversicherung\s*AG|M\s*Lebensversicherung\s*AG|N\s*Lebensversicherung\s*AG|O\s*Lebensversicherung\s*AG|P\s*Lebensversicherung\s*AG|Q\s*Lebensversicherung\s*AG|R\s*Lebensversicherung\s*AG|S\s*Lebensversicherung\s*AG|T\s*Lebensversicherung\s*AG|U\s*Lebensversicherung\s*AG|V\s*Lebensversicherung\s*AG|W\s*Lebensversicherung\s*AG|X\s*Lebensversicherung\s*AG|Y\s*Lebensversicherung\s*AG|Z\s*Lebensversicherung\s*AG)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.342 | 0.520 | 0.413 | 1228 | 420 | 808 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 420 | 808 | 387 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53387`) (sent_id: `53387`)


7. Die hiergegen eingelegte Nichtzulassungsbeschwerde sowie eine Anhörungsrüge der Beschwerdeführerin wies der Bundesgerichtshof zurück .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Example 1** (doc_id: `53400`) (sent_id: `53400`)


Die Vorgabe des Bundesfinanzhofs ( BFH ) , dass die Einsteller ( Leistungsempfänger ) keine Land- und Forstwirte sein müssten , die von ihnen empfangenen Leistungen ( Zuchtleistungen als einheitliche Leistung ) aber gleichwohl zu landwirtschaftlichen Zwecken genutzt werden müssten , erscheine nicht eindeutig und auf den ersten Blick widersprüchlich .

| Predicted | Gold |
|---|---|
| `Bundesfinanzhofs` | `Bundesfinanzhofs` |
| `BFH` | `BFH` |

**Example 2** (doc_id: `53402`) (sent_id: `53402`)


Bei offenem Ausgang des Verfassungsbeschwerdeverfahrens muss das Bundesverfassungsgericht die Folgen abwägen , die eintreten würden , wenn die einstweilige Anordnung nicht erginge , die Verfassungsbeschwerde aber Erfolg hätte , gegenüber den Nachteilen , die entstünden , wenn die begehrte einstweilige Anordnung erlassen würde , der Verfassungsbeschwerde aber der Erfolg zu versagen wäre ( vgl. BVerfGE 76 , 253 < 255 > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `BVerfGE 76 , 253 < 255 >` (RS)

**Example 3** (doc_id: `53403`) (sent_id: `53403`)


Vor diesem Hintergrund erweist sich schon die Auseinandersetzung des Beschwerdeführers mit den vom Bundesverfassungsgericht - wenn auch zu Art. 19 Abs. 4 GG , den der Beschwerdeführer nicht rügt - entwickelten verfassungsrechtlichen Maßstäben als unzureichend ; umso weniger ist unter diesen Umständen eine mögliche Willkür der angegriffenen Entscheidung plausibel dargelegt .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Art. 19 Abs. 4 GG` (NRM)

**Example 4** (doc_id: `53404`) (sent_id: `53404`)


Hiergegen richtet sich die mit Schriftsatz vom 11. Februar 2014 beim Deutschen Patent- und Markenamt eingelegte Beschwerde der Anmelderin , die ihr Patentbegehren mit den mit Schreiben vom 25. Februar 2011 eingereichten Unterlagen weiterverfolgt .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Example 5** (doc_id: `53408`) (sent_id: `53408`)


Ein solcher enger sachlicher Zusammenhang mit der gesetzlichen Tätigkeit als Träger der Grundsicherung für Arbeitsuchende ist bei der BA immer dann anzunehmen , wenn sie - objektiv betrachtet - in dem zugrunde liegenden Verfahren eine Aufgabe im Rahmen des Vollzugs des SGB II wahrnimmt .

| Predicted | Gold |
|---|---|
| `BA` | `BA` |

**Missed by this rule (FN):**

- `SGB II` (NRM)

**Example 6** (doc_id: `53415`) (sent_id: `53415`)


Die Form der Benutzung ist insoweit jeweils glaubhaft gemacht durch die als Anlagen W14 bis W17 vorgelegten Produktkataloge aus dem vorliegend relevanten Benutzungszeitraum mit Abbildungen einer Vielzahl von ( elektrischen ) Kaffeemaschinen , Kaffeemühlen , Milchbehälter ( „ Milchcooler “ ) sowie Verpackungen / Behälter von Reinigungstabs , CreamCleaner und Entkalker , auf denen die Widerspruchsmarke NIVONA in einer ohne weitere besondere graphische Ausgestaltungselemente enthaltenden und damit in einer den kennzeichnenden Charakter der eingetragenen Marke nicht verändernden Form i. S. von § 26 Abs. 3 MarkenG deutlich sichtbar abgebildet ist .

| Predicted | Gold |
|---|---|
| `NIVONA` | `NIVONA` |

**Missed by this rule (FN):**

- `§ 26 Abs. 3 MarkenG` (NRM)

**Example 7** (doc_id: `53419`) (sent_id: `53419`)


Hiergegen richtet sich die Beschwerde der Anmelderin vom 29. November 2016 , mit der sie sinngemäß beantragt , den Beschluss der Markenstelle für Klasse 35 des Deutschen Patent- und Markenamts vom 11. November 2016 aufzuheben .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 35 des Deutschen Patent- und Markenamts` | `Markenstelle für Klasse 35 des Deutschen Patent- und Markenamts` |

**Example 8** (doc_id: `53424`) (sent_id: `53424`)


Die X-EWIV wäre dem Kläger in vollem Umfang auskunfts- und rechenschaftspflichtig gewesen .

| Predicted | Gold |
|---|---|
| `X-EWIV` | `X-EWIV` |

**Example 9** (doc_id: `53430`) (sent_id: `53430`)


II. Die zulässige Beschwerde der Löschungsantragsgegnerin hat sich durch die zwischenzeitlich mit Beschluss des DPMA vom 27. September 2017 angeordnete Verfallslöschung der angegriffenen Marke in der Hauptsache erledigt .

| Predicted | Gold |
|---|---|
| `DPMA` | `DPMA` |

**Example 10** (doc_id: `53466`) (sent_id: `53466`)


Die PreussenElektra GmbH war Beschwerdeführerin im Verfassungsbeschwerdeverfahren 1 BvR 2821/11 ( BVerfGE 143 , 246 ) , das sich ebenfalls gegen die hier angegriffenen Regelungen des Atomgesetzes richtete .

| Predicted | Gold |
|---|---|
| `PreussenElektra GmbH` | `PreussenElektra GmbH` |

**Missed by this rule (FN):**

- `Verfassungsbeschwerdeverfahren 1 BvR 2821/11 ( BVerfGE 143 , 246 )` (RS)
- `Atomgesetzes` (NRM)

**Example 11** (doc_id: `53524`) (sent_id: `53524`)


Der BFH führte aus , die handelsrechtliche Zurechnung von Vermögensgegenständen entspreche im Wesentlichen der Regelung des § 39 Abs. 2 Nr. 1 Satz 1 AO ( z.B. BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1. ; vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a ) .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Missed by this rule (FN):**

- `§ 39 Abs. 2 Nr. 1 Satz 1 AO` (NRM)
- `BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1.` (RS)
- `vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a` (RS)

**Example 12** (doc_id: `53539`) (sent_id: `53539`)


cc ) Stellt ein Bewerber nicht innerhalb eines Monats nach Zugang der Abbruchmitteilung einen Antrag auf Erlass einer einstweiligen Verfügung , darf der Dienstherr nach der Rechtsprechung des Bundesverwaltungsgerichts darauf vertrauen , dass der Bewerber den Abbruch des Auswahlverfahrens nicht angreift , sondern sein Begehren im Rahmen einer neuen Ausschreibung weiterverfolgt ( vgl. BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14 ) .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Missed by this rule (FN):**

- `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14` (RS)

**Example 13** (doc_id: `53549`) (sent_id: `53549`)


aa ) Der GBA hat eine IVIG-Therapie zur Behandlung des sekundären Immunglobulinmangels mit MGUS und rezidivierenden Pneumonien nicht empfohlen .

| Predicted | Gold |
|---|---|
| `GBA` | `GBA` |

**Example 14** (doc_id: `53574`) (sent_id: `53574`)


Die Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts hat diese unter der Nummer 30 2014 034 614.1 geführte Anmeldung mit Beschluss vom 25. November 2014 wegen fehlender Unterscheidungskraft zurückgewiesen .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts` | `Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts` |

**Example 15** (doc_id: `53586`) (sent_id: `53586`)


d ) Mit dem mit der Verfassungsbeschwerde ebenfalls angegriffenen Beschluss vom 8. März 2011 wies der Bundesfinanzhof die von der Beschwerdeführerin erhobene Anhörungsrüge und Gegenvorstellung zurück .

| Predicted | Gold |
|---|---|
| `Bundesfinanzhof` | `Bundesfinanzhof` |

**Example 16** (doc_id: `53594`) (sent_id: `53594`)


Die von der Krankenkasse nicht getragenen Kosten der stationären Behandlung zahlte der Kläger zu 1. Die Therapie der E in der K-Klinik war im Vorfeld durch den Medizinischen Dienst der Krankenversicherungen ( MDK ) auf ihre medizinische Notwendigkeit geprüft und befürwortet worden .

| Predicted | Gold |
|---|---|
| `K-Klinik` | `K-Klinik` |
| `MDK` | `MDK` |

**Missed by this rule (FN):**

- `E` (PER)
- `Medizinischen Dienst der Krankenversicherungen` (ORG)

**Example 17** (doc_id: `53619`) (sent_id: `53619`)


( 2 ) Ob die Umwandlung der Todesstrafe in eine lebenslange Freiheitsstrafe bereits zwingend aus dem seit 1991 praktizierten Moratorium folgt , wie es das Bundesverwaltungsgericht angenommen hat , kann dahinstehen .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Example 18** (doc_id: `53630`) (sent_id: `53630`)


Die Verträge seien derart angelegt gewesen , dass die KG am Ende der Laufzeit ihr " Andienungsrecht " ausübe und die P GmbH die Leasingobjekte zurückerwerben müsse .

| Predicted | Gold |
|---|---|
| `P GmbH` | `P GmbH` |

**Example 19** (doc_id: `53656`) (sent_id: `53656`)


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

**Example 20** (doc_id: `53664`) (sent_id: `53664`)


Soweit das Bundesministerium der Verteidigung darauf verweist , dass Personalgespräche dazu dienten , Soldaten über Verwendungsplanungen der Personalführung zu informieren und in diesem Zusammenhang deren persönliche Motivationslage zu ergründen , um das dienstliche Bedürfnis und die privaten Lebensumstände des Soldaten möglichst gut miteinander zu vereinbaren , trifft dies zweifellos auf Personalgespräche im Vorfeld von Personalmaßnahmen zu .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Example 21** (doc_id: `53690`) (sent_id: `53690`)


An dieser Regelung hielt die Bundesregierung fest , obwohl der Bundesrat die Einbeziehung der einmaligen Einnahmen vorschlug ( BT-Drucks 16/2454 S 11 ) .

| Predicted | Gold |
|---|---|
| `Bundesregierung` | `Bundesregierung` |
| `Bundesrat` | `Bundesrat` |

**Missed by this rule (FN):**

- `BT-Drucks 16/2454 S 11` (LIT)

**Example 22** (doc_id: `53700`) (sent_id: `53700`)


3. Das Bundesverfassungsgericht überprüft die Vereinbarkeit eines nationalen Gesetzes mit dem Grundgesetz auch , wenn zugleich Zweifel an der Vereinbarkeit des Gesetzes mit Sekundärrecht der Europäischen Union bestehen .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Grundgesetz` (NRM)
- `Europäischen Union` (ORG)

**Example 23** (doc_id: `53725`) (sent_id: `53725`)


4. Auf die sofortige Beschwerde der Staatsanwaltschaft Düsseldorf hob das Oberlandesgericht Hamm mit angegriffenem Beschluss vom 28. Juli 2015 den Beschluss des Landgerichts Paderborn auf und ordnete die weitere Vollstreckung der Unterbringung des Beschwerdeführers in der Sicherungsverwahrung an .

| Predicted | Gold |
|---|---|
| `Oberlandesgericht Hamm` | `Oberlandesgericht Hamm` |

**Missed by this rule (FN):**

- `Staatsanwaltschaft Düsseldorf` (ORG)
- `Landgerichts Paderborn` (ORG)

**Example 24** (doc_id: `53746`) (sent_id: `53746`)


Das hat zur Folge , dass beim BSG über Kostenerinnerungen nach § 189 Abs 2 S 2 SGG der Senat in der Besetzung mit drei Berufsrichtern zu befinden hat .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Missed by this rule (FN):**

- `§ 189 Abs 2 S 2 SGG` (NRM)

**Example 25** (doc_id: `53773`) (sent_id: `53773`)


Vielmehr hat das LSG nach erfolgter Zurückverweisung durch das BSG nach umfangreicher weiterer Sachaufklärung den Anspruch des Klägers erneut geprüft und eine Entscheidung gefällt .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 26** (doc_id: `53777`) (sent_id: `53777`)


Am 14. November 2000 wurden den Aktionären der A-AG vereinbarungsgemäß ... Stück vinkulierte Namensaktien an der B-AG übertragen .

| Predicted | Gold |
|---|---|
| `A-AG` | `A-AG` |
| `B-AG` | `B-AG` |

**Example 27** (doc_id: `53791`) (sent_id: `53791`)


Zwischen den Vergleichszeichen besteht hinsichtlich der angegriffenen Waren Verwechslungsgefahr gemäß § 9 Abs. 1 Nr. 2 MarkenG , so dass der Widerspruch aus der Marke 30 2011 047 766 vom Deutschen Patent- und Markenamt zu Unrecht zurückgewiesen wurde .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `§ 9 Abs. 1 Nr. 2 MarkenG` (NRM)

**Example 28** (doc_id: `53844`) (sent_id: `53844`)


Der im Juli 1950 geborene Kläger war seit dem 1. März 1971 bei einer Rechtsvorgängerin der Beklagten , der H AG ( im Folgenden H AG alt ) als Arbeitnehmer tätig .

| Predicted | Gold |
|---|---|
| `H AG` | `H AG` |

**Missed by this rule (FN):**

- `H AG alt` (ORG)

**Example 29** (doc_id: `53855`) (sent_id: `53855`)


aa ) In organisatorischer Hinsicht ist beim Bundesverfassungsgericht , anders als bei den Fachgerichten , eine Kapazitätsausweitung zur Verkürzung der Verfahrensdauer als Reaktion auf gesteigerte Eingangszahlen grundsätzlich nicht möglich , da die Struktur des Gerichts durch seine Funktion bedingt und durch die Verfassung und das Bundesverfassungsgerichtsgesetz vorgegeben ist ( BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21 ; vgl. BTDrucks 17/3802 , S. 26 ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Bundesverfassungsgerichtsgesetz` (NRM)
- `BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21` (RS)
- `BTDrucks 17/3802 , S. 26` (LIT)

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

**Example 1** (doc_id: `53422`) (sent_id: `53422`)


Bei der vom FG zu beurteilenden gesetzlichen Prozessführungsbefugnis ( Beteiligtenstellung ) handelt es sich um eine Sachurteilsvoraussetzung , deren fehlerhafte Beurteilung einen Verfahrensmangel darstellt ( BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943 ; vom 23. November 2010 V B 133/09 , BFH / NV 2011 , 612 , Rz 5 ; jeweils m. w. N. ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`
- `BFH` — partial — pred is substring of gold: `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`
- `BFH` — similar text (different position): `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`

> overlaps gold: 3  |  likely missing annotation: 0

**Gold Entities:**

- `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`(RS)
- `vom 23. November 2010 V B 133/09 , BFH / NV 2011 , 612 , Rz 5`(RS)

**Example 2** (doc_id: `53446`) (sent_id: `53446`)


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

**Example 3** (doc_id: `53468`) (sent_id: `53468`)


Wie das vorliegende Verfahren zeigt , bietet die Anhörungsrüge die Möglichkeit , durch Zuordnungsschwierigkeiten entstehende Probleme zu bewältigen ( vgl. BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15 ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15`(RS)

**Example 4** (doc_id: `53474`) (sent_id: `53474`)


Beim Leasing ist - wie dargestellt - darauf abzustellen , ob der Herausgabeanspruch des Leasinggebers ( zivilrechtlichen Eigentümers ) noch eine wirtschaftliche Bedeutung hat ( grundlegend BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1. ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1.`(RS)

**Example 5** (doc_id: `53481`) (sent_id: `53481`)


Diese können insbesondere dann vorliegen , wenn die Gesamtstrafe sich nicht innerhalb des gesetzlichen Strafrahmens bewegt , die gebotene Begründung für die Gesamtstrafe fehlt , oder wenn die Besorgnis besteht , der Tatrichter habe sich von der Summe der Einzelstrafen leiten lassen ( BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris`(RS)

**Example 6** (doc_id: `53485`) (sent_id: `53485`)


Soweit die Elemente eines Bildzeichens nur die typischen Merkmale der in Rede stehenden Waren und Dienstleistungen darstellen oder sich in einfachen dekorativen Gestaltungsmitteln erschöpfen , an die sich der Verkehr etwa durch häufige Verwendung gewöhnt hat , wird diesem Zeichen im Allgemeinen wegen seines bloß beschreibenden Inhalts ebenfalls die konkrete Eignung fehlen , die mit ihm gekennzeichneten Waren oder Dienstleistungen von denjenigen anderer Herkunft zu unterscheiden ( vgl. BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband ; GRUR 2005 , 257 , 258 – Bürogebäude ; GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel ; GRUR 2001 , 239f. – Zahnpastastrang ; GRUR 2001 , 734 , 735 – Westie-Kopf ; BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband`
- `BPatG` — partial — pred is substring of gold: `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband`(RS)
- `GRUR 2005 , 257 , 258 – Bürogebäude`(RS)
- `GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel`(RS)
- `GRUR 2001 , 239f. – Zahnpastastrang`(RS)
- `GRUR 2001 , 734 , 735 – Westie-Kopf`(RS)
- `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`(RS)

**Example 7** (doc_id: `53490`) (sent_id: `53490`)


Es müssen letztlich besondere Verhaltensweisen sowohl des Berechtigten als auch des Verpflichteten vorliegen , die es rechtfertigen , die späte Geltendmachung des Rechts als mit Treu und Glauben unvereinbar und für den Verpflichteten als unzumutbar anzusehen ( vgl. BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27 mwN ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27`(RS)

**Example 8** (doc_id: `53508`) (sent_id: `53508`)


Beruhen vielmehr die Differenzen zwischen den Auffassungen von Sachverständigen darauf , dass diese von verschiedenen tatsächlichen Annahmen ausgehen , dann muss der Tatrichter , ggf nach weiterer Aufklärung , die für seine Überzeugungsbildung maßgebenden Tatsachen feststellen oder begründen , weshalb und zu wessen Lasten sie beweislos geblieben sind ( vgl BGH Urteil vom 23. 9. 1986 - VI ZR 261/85 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH Urteil vom 23. 9. 1986 - VI ZR 261/85`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH Urteil vom 23. 9. 1986 - VI ZR 261/85`(RS)

**Example 9** (doc_id: `53524`) (sent_id: `53524`)


Der BFH führte aus , die handelsrechtliche Zurechnung von Vermögensgegenständen entspreche im Wesentlichen der Regelung des § 39 Abs. 2 Nr. 1 Satz 1 AO ( z.B. BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1. ; vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a ) .

**False Positives:**

- `BFH` — similar text (different position): `BFH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH`(ORG)
- `§ 39 Abs. 2 Nr. 1 Satz 1 AO`(NRM)
- `BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1.`(RS)
- `vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a`(RS)

**Example 10** (doc_id: `53539`) (sent_id: `53539`)


cc ) Stellt ein Bewerber nicht innerhalb eines Monats nach Zugang der Abbruchmitteilung einen Antrag auf Erlass einer einstweiligen Verfügung , darf der Dienstherr nach der Rechtsprechung des Bundesverwaltungsgerichts darauf vertrauen , dass der Bewerber den Abbruch des Auswahlverfahrens nicht angreift , sondern sein Begehren im Rahmen einer neuen Ausschreibung weiterverfolgt ( vgl. BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14 ) .

**False Positives:**

- `BVerwG` — partial — pred is substring of gold: `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesverwaltungsgerichts`(ORG)
- `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14`(RS)

**Example 11** (doc_id: `53551`) (sent_id: `53551`)


Die Beklagte wird das der Klägerin zustehende Honorar unter Zugrundelegung eines festen RLV für die gesamte BAG neu zu ermitteln haben .

**False Positives:**

- `BAG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 12** (doc_id: `53582`) (sent_id: `53582`)


3. Wird einem Beteiligten das rechtliche Gehör dadurch versagt , dass es ihm nicht ermöglicht wird , an der mündlichen Verhandlung teilzunehmen , so ist davon auszugehen , dass dies für eine aufgrund dieser Verhandlung ergangenen Entscheidung ursächlich geworden ist ( vgl BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7 ) .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7`(RS)

**Example 13** (doc_id: `53583`) (sent_id: `53583`)


Denn die Hauptfunktion einer Marke liegt darin , die Ursprungsidentität der gekennzeichneten Waren und Dienstleistungen zu gewährleisten ( vgl. u. a. EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel ; BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006 ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel`
- `BGH` — partial — pred is substring of gold: `BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel`(RS)
- `BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006`(RS)

**Example 14** (doc_id: `53599`) (sent_id: `53599`)


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

**Example 15** (doc_id: `53600`) (sent_id: `53600`)


Auch danach setzt die Bejahung der Unterscheidungskraft unverändert voraus , dass das Zeichen geeignet sein muss , die beanspruchten Produkte als von einem bestimmten Unternehmen stammend zu kennzeichnen ( vgl. EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK`(RS)

**Example 16** (doc_id: `53614`) (sent_id: `53614`)


Soweit Personen dezentral untergebracht sind , ist es für die Bejahung einer Einrichtung erforderlich , dass die dezentrale Unterkunft zu den Räumlichkeiten der Einrichtung gehört , der Hilfebedürftige also in die Räumlichkeiten des Einrichtungsträgers eingegliedert ist ( vgl BSG SozR 4 - 3500 § 98 Nr 3 ) .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG SozR 4 - 3500 § 98 Nr 3`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG SozR 4 - 3500 § 98 Nr 3`(RS)

**Example 17** (doc_id: `53618`) (sent_id: `53618`)


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

**Example 18** (doc_id: `53623`) (sent_id: `53623`)


Nach der von der Einsprechenden zitierten BGH-Entscheidung „ Kommunikationskanal “ , muss für den Fachmann die im Anspruch bezeichnete Lehre „ unmittelbar und eindeutig “ als mögliche Ausführungsform der Erfindung den Ursprungsunterlagen entnehmbar sein .

**False Positives:**

- `BGH` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `53637`) (sent_id: `53637`)


aa ) Das Berufungsgericht ist zu der Einschätzung gelangt , die Klägerin habe nicht dargelegt , dass die Zulassung als Jaguar- und Land-Rover-Vertragswerkstatt eine Ressource darstelle , ohne die der Zugang zu dem nachgelagerten Endkundenmarkt nicht oder nicht sinnvoll möglich sei .

**False Positives:**

- `Jaguar` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 20** (doc_id: `53680`) (sent_id: `53680`)


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

**Example 21** (doc_id: `53688`) (sent_id: `53688`)


Allerdings beschränkt sich die Geltung des Grundsatzes der Bestenauslese im Bereich der Verwendungsentscheidungen auf Entscheidungen über - wie hier - höherwertige , die Beförderung in einen höheren Dienstgrad oder die Einweisung in die Planstelle einer höheren Besoldungsgruppe vorprägende Verwendungen ( vgl. klarstellend BVerwG , Beschluss vom 30. Januar 2014 - 1 WB 1.13 - juris Rn. 32 ) .

**False Positives:**

- `BVerwG` — partial — pred is substring of gold: `BVerwG , Beschluss vom 30. Januar 2014 - 1 WB 1.13 - juris Rn. 32`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerwG , Beschluss vom 30. Januar 2014 - 1 WB 1.13 - juris Rn. 32`(RS)

**Example 22** (doc_id: `53701`) (sent_id: `53701`)


Für eine solche Prognose des Arbeitgebers bedarf es ausreichend konkreter Anhaltspunkte ( BAG 27. Juli 2016 - 7 AZR 545/14 - Rn. 19 ; 24. September 2014 - 7 AZR 987/12 - Rn. 18 ; 7. Mai 2008 - 7 AZR 146/07 - Rn. 15 ; 7. April 2004 - 7 AZR 441/03 - zu II 2 a aa der Gründe ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 27. Juli 2016 - 7 AZR 545/14 - Rn. 19`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 27. Juli 2016 - 7 AZR 545/14 - Rn. 19`(RS)
- `24. September 2014 - 7 AZR 987/12 - Rn. 18`(RS)
- `7. Mai 2008 - 7 AZR 146/07 - Rn. 15`(RS)
- `7. April 2004 - 7 AZR 441/03 - zu II 2 a aa der Gründe`(RS)

**Example 23** (doc_id: `53707`) (sent_id: `53707`)


Da eine so weitgehende Selbstentäußerung des ausländischen Staates im Zweifel nicht zu vermuten ist ( BGH 30. Januar 2013 - III ZB 40/12 - Rn. 19 ) , dürfen die Umstände des Falles hinsichtlich des Vorliegens und der Reichweite eines Verzichts keinen Zweifel lassen ( BAG 29. Juni 2017 - 2 AZR 759/16 - Rn. 20 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH 30. Januar 2013 - III ZB 40/12 - Rn. 19`
- `BAG` — partial — pred is substring of gold: `BAG 29. Juni 2017 - 2 AZR 759/16 - Rn. 20`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH 30. Januar 2013 - III ZB 40/12 - Rn. 19`(RS)
- `BAG 29. Juni 2017 - 2 AZR 759/16 - Rn. 20`(RS)

**Example 24** (doc_id: `53717`) (sent_id: `53717`)


Bei Anerkennungsbeträgen handelt es sich um eine jener Massenerscheinungen , die ein typisierendes und pauschalierendes Vorgehen auch der Verwaltung rechtfertigen ( vgl. BVerfG , Beschluss vom 31. Mai 1988 - 1 BvR 520/83 - BVerfGE 78 , 214 < 227 > ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Beschluss vom 31. Mai 1988 - 1 BvR 520/83 - BVerfGE 78 , 214 < 227 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG , Beschluss vom 31. Mai 1988 - 1 BvR 520/83 - BVerfGE 78 , 214 < 227 >`(RS)

**Example 25** (doc_id: `53721`) (sent_id: `53721`)


Die Regelung ist daher geeignet , die Befristung von Arbeitsverträgen mit programmgestaltenden Mitarbeitern bei Rundfunkanstalten zu rechtfertigen ( BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101 ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101`(RS)

**Example 26** (doc_id: `53760`) (sent_id: `53760`)


Etwas anderes gilt insbesondere dann , wenn der Arbeitgeber seine Tarifgebundenheit in einer dem Arbeitnehmer hinreichend erkennbaren Weise zur auflösenden Bedingung der Bezugnahme gemacht hat ( BAG 18. April 2007 - 4 AZR 652/05 - Rn. 32 , 40 , BAGE 122 , 74 ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 18. April 2007 - 4 AZR 652/05 - Rn. 32 , 40 , BAGE 122 , 74`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 18. April 2007 - 4 AZR 652/05 - Rn. 32 , 40 , BAGE 122 , 74`(RS)

**Example 27** (doc_id: `53763`) (sent_id: `53763`)


Wird als Verfahrensmangel gerügt , das FG habe einen gestellten Beweisantrag übergangen ( Rüge mangelnder Sachaufklärung gemäß § 76 Abs. 1 Satz 1 FGO ) , so ist darzulegen , welche Tatfrage aufklärungsbedürftig ist , welche Beweismittel das FG zu welchem Beweisthema nicht erhoben hat , die genauen Fundstellen ( Schriftsatz mit Datum und Seitenzahl , Terminprotokoll ) , in denen die Beweismittel und Beweisthemen angeführt worden sind , das voraussichtliche Ergebnis der Beweisaufnahme , inwiefern das Urteil des FG aufgrund dessen sachlich-rechtlicher Auffassung auf der unterbliebenen Beweisaufnahme beruhen kann und dass die Nichterhebung der Beweise vor dem FG rechtzeitig gerügt worden ist oder aufgrund des Verhaltens des FG nicht mehr vor diesem gerügt werden konnte ( ständige BFH-Rechtsprechung , z.B. Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85 , und vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125 , m. w. N. ) .

**False Positives:**

- `BFH` — similar text (different position): `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`
- `BFH` — similar text (different position): `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 76 Abs. 1 Satz 1 FGO`(NRM)
- `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`(RS)
- `vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125`(RS)

**Example 28** (doc_id: `53767`) (sent_id: `53767`)


Ob der Erwerb eigener Anteile auf der Gesellschaftsebene entsprechend der durch das BilMoG geänderten handelsrechtlichen Vorschriften ( Einfügung des § 272 Abs. 1a und 1b HGB ) steuerrechtlich nicht mehr als Erwerbsvorgang anzusehen , sondern nunmehr als " Teilliquidation " und daher " wie " eine Kapitalherabsetzung zu behandeln ist ( so BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f. ; zustimmend z.B. Blumenberg / Lechner , DB 2014 , 141 , 147 ; kritisch z.B. Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55 ) , kann im Streitfall mangels Erheblichkeit für die Entscheidung offenbleiben .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f.`
- `DB` — partial — pred is substring of gold: `Blumenberg / Lechner , DB 2014 , 141 , 147`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BilMoG`(NRM)
- `§ 272 Abs. 1a und 1b HGB`(NRM)
- `BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f.`(REG)
- `Blumenberg / Lechner , DB 2014 , 141 , 147`(LIT)
- `Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55`(LIT)

**Example 29** (doc_id: `53775`) (sent_id: `53775`)


nicht nur das erstmalige Legen eines Hausanschlusses , sondern auch Arbeiten zur Erneuerung oder zur Reduzierung von Wasseranschlüssen unter die Steuerermäßigung fallen ( vgl. BGH-Urteil in HFR 2012 , 1110 , Rz 20 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH-Urteil in HFR 2012 , 1110 , Rz 20`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH-Urteil in HFR 2012 , 1110 , Rz 20`(RS)

</details>

---

## `Specific Court Departments and Senates`

**F1:** 0.317 | **Precision:** 0.274 | **Recall:** 0.374  

**Format:** `regex`  
**Rule ID:** `99bc8170`  
**Description:**
Matches specific German court names and abbreviations that were previously in the long list, now using a more structured approach for common courts and abbreviations.

**Content:**
```
\b(?:BVerfG|BGH|BFH|BVerwG|BAG|BSG|EuGH|DPMA|BPatG|ZDS|ZIV|GBA|GEW|X-EWIV|GmSOGB|EGMR|NATO|EU|MDK|BA|VCS|Google|RAPAMUNE|nivo|KAEFER|ARD|ZDF|Deutschlandradio|HBV|K\u00c4V|ver\.di|DB|BND|RVA|TdL|InWIS|DRV|KOSMICA|Monster\s*Abwehr\s*Spray|Haus\u00e4rzteverband|Bundesvereinigung\s*der\s*Arbeitgeberverb\u00e4nde|Deutsche\s*Rentenversicherung\s*Bund|Ausw\u00e4rtige\s*Amt|Bundesamt\s*f\u00fcr\s*Migration\s*und\s*Fl\u00fcchtlinge|BMF|Internationalen\s*Gerichtshofs|Europ\u00e4ischen\s*Kommission|Justizkommission\s*zu\s*Tunis|Bundeswehr|Fliegerhorst\s*B\u00fcchel|St\u00e4dtischen\s*Klinikum\s*K\.|Kernkraftwerks\s*Kr\u00fcmmel|Landesarbeitsgericht\s*Berlin-Brandenburg|S\u00e4chsische\s*Bildungsagentur|Gemeinsamen\s*Bundesausschusses|Landesaussch\u00fcsse|Obersten\s*Verwaltungsgerichts\s*der\s*Republik\s*Bulgarien|Bundeskasse\s*Halle\s*/\s*Saale\s*-\s*Dienstsitz\s*Weiden\s*/\s*Oberpfalz|Diakonischen\s*Werkes\s*der\s*Evangelischen\s*Kirche\s*in\s*Deutschland\s*e\.\s*V\.|Evangelischen\s*Entwicklungsdienstes\s*e\.\s*V\.|Haus\u00e4rztlliche\s*Vertragsgemeinschaft\s*Aktiengesellschaft|H\u00c4VG-Rechenzentrum\s*AG|H\u00e4VG-Rechenzentrum\s*GmbH|B\s*\u2026\s*Patentanwaltsgesellschaft\s*mbH|C\s*\u2026\s*GmbH|D\s*P\s*T\s*S\s*GmbH|K\s*\u2026\s*GmbH|B\s*\u2026\s*AG|S\s*\u2026|C-\s*B\.\s*V\.|A-Fonds|BgA\s*X|InEK|Monster\s*Abwehr\s*Spray|Haus\u00e4rzteverband|Bundesvereinigung\s*der\s*Arbeitgeberverb\u00e4nde|Deutsche\s*Rentenversicherung\s*Bund|Ausw\u00e4rtige\s*Amt|Bundesamt\s*f\u00fcr\s*Migration\s*und\s*Fl\u00fcchtlinge|BMF|Internationalen\s*Gerichtshofs|Europ\u00e4ischen\s*Kommission|Justizkommission\s*zu\s*Tunis|Bundeswehr|Fliegerhorst\s*B\u00fcchel|St\u00e4dtischen\s*Klinikum\s*K\.|Kernkraftwerks\s*Kr\u00fcmmel|Landesarbeitsgericht\s*Berlin-Brandenburg|S\u00e4chsische\s*Bildungsagentur|Gemeinsamen\s*Bundesausschusses|Landesaussch\u00fcsse|Obersten\s*Verwaltungsgerichts\s*der\s*Republik\s*Bulgarien|Bundeskasse\s*Halle\s*/\s*Saale\s*-\s*Dienstsitz\s*Weiden\s*/\s*Oberpfalz|Diakonischen\s*Werkes\s*der\s*Evangelischen\s*Kirche\s*in\s*Deutschland\s*e\.\s*V\.|Evangelischen\s*Entwicklungsdienstes\s*e\.\s*V\.|Haus\u00e4rztlliche\s*Vertragsgemeinschaft\s*Aktiengesellschaft|H\u00c4VG-Rechenzentrum\s*AG|H\u00e4VG-Rechenzentrum\s*GmbH|B\s*\u2026\s*Patentanwaltsgesellschaft\s*mbH|C\s*\u2026\s*GmbH|D\s*P\s*T\s*S\s*GmbH|K\s*\u2026\s*GmbH|B\s*\u2026\s*AG|S\s*\u2026|C-\s*B\.\s*V\.|A-Fonds|BgA\s*X|InEK|Bundesregierung|Bundesministerium\s+der\s+Finanzen|Bundesamts\s+f\u00fcr\s+Justiz|Justizministerium\s+des\s+Landes\s+Nordrhein-Westfalen|Neurologischen\s+Klinik\s+B|Amtsgericht\s+O\.|Handwerksverband\s+Metallbau\s+und\s+Feinwerktechnik\s+Baden-W\u00fcrttemberg|Industriegewerkschaft\s+Metall|VEB\s+[A-Z][a-zA-Z\s]+|nieders\u00e4chsische\s+Landesschulbeh\u00f6rde|EON-Konzerns|EON-Konzern|dbb\s+beamtenbund\s+und\s+tarifunion|ADAC|Kernkraftwerks\s+Biblis|Kernkraftwerks\s+M\u00fclheim-K\u00e4rlich|Deutschen\s*Botschaft|Finanzamt\s+[A-Za-z\u00e4\u00f6\u00fc\u00df\s]+|Bundesministeriums\s+der\s+Verteidigung\s+-\s+R\s+II\s+2\s+-|Bundesamts\s+f\u00fcr\s+das\s+Personalmanagement|A\s+Lebensversicherung\s+AG|Europ\u00e4ische\s+Gerichtshof\s+f\u00fcr\s+Menschenrechte|Th\u00fcringer\s+Landessozialgerichts|Deutsche\s+Patent-\s+und\s+Markenamt|Deutschen\s+Patent-\s+und\s+Markenamt|Deutschen\s+Patent-\s+und\s+Markenamtes|K\u00c4V\s+Brandenburg|Neue\s+Richtervereinigung|Generalstaatsanwaltschaft\s+des\s+Landes\s+Schleswig-Holstein|K-Klinik|H\u00c4VG|Schott|PreussenElektra\s+GmbH|E\.\s+ON\s+Kernkraft\s+GmbH|G-Gruppe|Vereinigung\s+der\s+kommunalen\s*Arbeitgeberverb\u00e4nde|Staatskasse|Kernkraftwerks\s+Gundremmingen|Bundesministeriums\s+der\s+Verteidigung|Bundesministerium\s+der\s+Verteidigung|Bundesverfassungsgerichts|Bundesverwaltungsgerichts|Bundesgerichtshofs|Bundesfinanzhofs|Bundesarbeitsgerichts|Bundessozialgerichts|Landgerichts\s+Darmstadt|Landgerichts\s+D\u00fcsseldorf|Landgerichts\s+Hamburg|Landgerichts\s+Bremen|Landgerichts\s+Oldenburg|Landgerichts\s+Karlsruhe|Landgerichts\s+Potsdam|Landgerichts\s+F\.\s*\(\s*P\.\s*\)|Pf\u00e4lzische\s+Oberlandesgericht\s+Zweibr\u00fccken|Oberlandesgerichts\s+M\u00fcnchen|Oberlandesgerichts\s+Hamm|Amtsgerichts\s+O\.|Finanzamts\s+[A-Za-z\u00e4\u00f6\u00fc\u00df\s]+|Bundesamts\s+f\u00fcr\s+das\s+Personalmanagement\s+der\s+Bundeswehr|Markenstelle\s*f\u00fcr\s*Klasse\s*\d+\s*des\s*Deutschen\s*Patent-\s*und\s*Markenamts|Patentabteilung\s*\d+\.\d+\s*des\s*Deutschen\s*Patent-\s*und\s*Markenamts|Designabteilung\s*\d+\.\d+\s*des\s*Deutschen\s*Patent-\s*und\s*Markenamts|Markenstelle\s*f\u00fcr\s*Klasse\s*\d+\s*des\s*DPMA|Patentabteilung\s*\d+\.\d+\s*des\s*DPMA|Designabteilung\s*\d+\.\d+\s*des\s*DPMA|Deutschen\s*Patent-\s*und\s*Markenamts|Deutschen\s*Patent-\s*und\s*Markenamt|Deutsche\s*Patent-\s*und\s*Markenamt|Deutsche\s*Patent-\s*und\s*Markenamts|Statistischen\s*Bundesamt|Deutschen\s*Bundestages|Spitzenverband\s*Bund\s*der\s*KKn|Verband\s*der\s*Privaten\s*Krankenversicherung|Deutsche\s*Krankenhausgesellschaft|VKDA|Gro\u00dfen\s*Senat\s*des\s*BFH|VIII\.\s*Senat\s*des\s*BFH|VIII\.\s*Senats\s*des\s*BFH|LSG\s+Berlin-Brandenburg|RWE-Konzerns|B\.\s*GmbH|w\s*GmbH|w\s*Holding\s*GmbH|P\s*GmbH|G\s*GmbH|M-GmbH\s*&\s*atypisch\s*Still|Gewerkschaft\s*ver\.di|Kreiskrankenh\u00e4user\s*M\s+und\s*R|NIVONA|fluege\.de|CHECK24\.de|Jaguar|Land\s*Rover|\u00d6z\s*Gaziantep\s*Dilim\s*Baklavalari|Wohnungsbau-\s*und\s*Kommissionsgesellschaft\s*Reichenstra\u00dfen|A-Fonds|B-Fonds|A-AG|B-AG|C-AG|D-AG|E-AG|F-AG|G-AG|H-AG|I-AG|J-AG|K-AG|L-AG|M-AG|N-AG|O-AG|P-AG|Q-AG|R-AG|S-AG|T-AG|U-AG|V-AG|W-AG|X-AG|Y-AG|Z-AG|A\s*Lebensversicherung\s*AG|B\s*Lebensversicherung\s*AG|C\s*Lebensversicherung\s*AG|D\s*Lebensversicherung\s*AG|E\s*Lebensversicherung\s*AG|F\s*Lebensversicherung\s*AG|G\s*Lebensversicherung\s*AG|H\s*Lebensversicherung\s*AG|I\s*Lebensversicherung\s*AG|J\s*Lebensversicherung\s*AG|K\s*Lebensversicherung\s*AG|L\s*Lebensversicherung\s*AG|M\s*Lebensversicherung\s*AG|N\s*Lebensversicherung\s*AG|O\s*Lebensversicherung\s*AG|P\s*Lebensversicherung\s*AG|Q\s*Lebensversicherung\s*AG|R\s*Lebensversicherung\s*AG|S\s*Lebensversicherung\s*AG|T\s*Lebensversicherung\s*AG|U\s*Lebensversicherung\s*AG|V\s*Lebensversicherung\s*AG|W\s*Lebensversicherung\s*AG|X\s*Lebensversicherung\s*AG|Y\s*Lebensversicherung\s*AG|Z\s*Lebensversicherung\s*AG)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.274 | 0.374 | 0.317 | 1101 | 302 | 799 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 302 | 799 | 504 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53400`) (sent_id: `53400`)


Die Vorgabe des Bundesfinanzhofs ( BFH ) , dass die Einsteller ( Leistungsempfänger ) keine Land- und Forstwirte sein müssten , die von ihnen empfangenen Leistungen ( Zuchtleistungen als einheitliche Leistung ) aber gleichwohl zu landwirtschaftlichen Zwecken genutzt werden müssten , erscheine nicht eindeutig und auf den ersten Blick widersprüchlich .

| Predicted | Gold |
|---|---|
| `Bundesfinanzhofs` | `Bundesfinanzhofs` |
| `BFH` | `BFH` |

**Example 1** (doc_id: `53404`) (sent_id: `53404`)


Hiergegen richtet sich die mit Schriftsatz vom 11. Februar 2014 beim Deutschen Patent- und Markenamt eingelegte Beschwerde der Anmelderin , die ihr Patentbegehren mit den mit Schreiben vom 25. Februar 2011 eingereichten Unterlagen weiterverfolgt .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Example 2** (doc_id: `53408`) (sent_id: `53408`)


Ein solcher enger sachlicher Zusammenhang mit der gesetzlichen Tätigkeit als Träger der Grundsicherung für Arbeitsuchende ist bei der BA immer dann anzunehmen , wenn sie - objektiv betrachtet - in dem zugrunde liegenden Verfahren eine Aufgabe im Rahmen des Vollzugs des SGB II wahrnimmt .

| Predicted | Gold |
|---|---|
| `BA` | `BA` |

**Missed by this rule (FN):**

- `SGB II` (NRM)

**Example 3** (doc_id: `53415`) (sent_id: `53415`)


Die Form der Benutzung ist insoweit jeweils glaubhaft gemacht durch die als Anlagen W14 bis W17 vorgelegten Produktkataloge aus dem vorliegend relevanten Benutzungszeitraum mit Abbildungen einer Vielzahl von ( elektrischen ) Kaffeemaschinen , Kaffeemühlen , Milchbehälter ( „ Milchcooler “ ) sowie Verpackungen / Behälter von Reinigungstabs , CreamCleaner und Entkalker , auf denen die Widerspruchsmarke NIVONA in einer ohne weitere besondere graphische Ausgestaltungselemente enthaltenden und damit in einer den kennzeichnenden Charakter der eingetragenen Marke nicht verändernden Form i. S. von § 26 Abs. 3 MarkenG deutlich sichtbar abgebildet ist .

| Predicted | Gold |
|---|---|
| `NIVONA` | `NIVONA` |

**Missed by this rule (FN):**

- `§ 26 Abs. 3 MarkenG` (NRM)

**Example 4** (doc_id: `53419`) (sent_id: `53419`)


Hiergegen richtet sich die Beschwerde der Anmelderin vom 29. November 2016 , mit der sie sinngemäß beantragt , den Beschluss der Markenstelle für Klasse 35 des Deutschen Patent- und Markenamts vom 11. November 2016 aufzuheben .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 35 des Deutschen Patent- und Markenamts` | `Markenstelle für Klasse 35 des Deutschen Patent- und Markenamts` |

**Example 5** (doc_id: `53424`) (sent_id: `53424`)


Die X-EWIV wäre dem Kläger in vollem Umfang auskunfts- und rechenschaftspflichtig gewesen .

| Predicted | Gold |
|---|---|
| `X-EWIV` | `X-EWIV` |

**Example 6** (doc_id: `53430`) (sent_id: `53430`)


II. Die zulässige Beschwerde der Löschungsantragsgegnerin hat sich durch die zwischenzeitlich mit Beschluss des DPMA vom 27. September 2017 angeordnete Verfallslöschung der angegriffenen Marke in der Hauptsache erledigt .

| Predicted | Gold |
|---|---|
| `DPMA` | `DPMA` |

**Example 7** (doc_id: `53466`) (sent_id: `53466`)


Die PreussenElektra GmbH war Beschwerdeführerin im Verfassungsbeschwerdeverfahren 1 BvR 2821/11 ( BVerfGE 143 , 246 ) , das sich ebenfalls gegen die hier angegriffenen Regelungen des Atomgesetzes richtete .

| Predicted | Gold |
|---|---|
| `PreussenElektra GmbH` | `PreussenElektra GmbH` |

**Missed by this rule (FN):**

- `Verfassungsbeschwerdeverfahren 1 BvR 2821/11 ( BVerfGE 143 , 246 )` (RS)
- `Atomgesetzes` (NRM)

**Example 8** (doc_id: `53524`) (sent_id: `53524`)


Der BFH führte aus , die handelsrechtliche Zurechnung von Vermögensgegenständen entspreche im Wesentlichen der Regelung des § 39 Abs. 2 Nr. 1 Satz 1 AO ( z.B. BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1. ; vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a ) .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Missed by this rule (FN):**

- `§ 39 Abs. 2 Nr. 1 Satz 1 AO` (NRM)
- `BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1.` (RS)
- `vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a` (RS)

**Example 9** (doc_id: `53539`) (sent_id: `53539`)


cc ) Stellt ein Bewerber nicht innerhalb eines Monats nach Zugang der Abbruchmitteilung einen Antrag auf Erlass einer einstweiligen Verfügung , darf der Dienstherr nach der Rechtsprechung des Bundesverwaltungsgerichts darauf vertrauen , dass der Bewerber den Abbruch des Auswahlverfahrens nicht angreift , sondern sein Begehren im Rahmen einer neuen Ausschreibung weiterverfolgt ( vgl. BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14 ) .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Missed by this rule (FN):**

- `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14` (RS)

**Example 10** (doc_id: `53549`) (sent_id: `53549`)


aa ) Der GBA hat eine IVIG-Therapie zur Behandlung des sekundären Immunglobulinmangels mit MGUS und rezidivierenden Pneumonien nicht empfohlen .

| Predicted | Gold |
|---|---|
| `GBA` | `GBA` |

**Example 11** (doc_id: `53574`) (sent_id: `53574`)


Die Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts hat diese unter der Nummer 30 2014 034 614.1 geführte Anmeldung mit Beschluss vom 25. November 2014 wegen fehlender Unterscheidungskraft zurückgewiesen .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts` | `Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts` |

**Example 12** (doc_id: `53594`) (sent_id: `53594`)


Die von der Krankenkasse nicht getragenen Kosten der stationären Behandlung zahlte der Kläger zu 1. Die Therapie der E in der K-Klinik war im Vorfeld durch den Medizinischen Dienst der Krankenversicherungen ( MDK ) auf ihre medizinische Notwendigkeit geprüft und befürwortet worden .

| Predicted | Gold |
|---|---|
| `K-Klinik` | `K-Klinik` |
| `MDK` | `MDK` |

**Missed by this rule (FN):**

- `E` (PER)
- `Medizinischen Dienst der Krankenversicherungen` (ORG)

**Example 13** (doc_id: `53630`) (sent_id: `53630`)


Die Verträge seien derart angelegt gewesen , dass die KG am Ende der Laufzeit ihr " Andienungsrecht " ausübe und die P GmbH die Leasingobjekte zurückerwerben müsse .

| Predicted | Gold |
|---|---|
| `P GmbH` | `P GmbH` |

**Example 14** (doc_id: `53656`) (sent_id: `53656`)


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

**Example 15** (doc_id: `53664`) (sent_id: `53664`)


Soweit das Bundesministerium der Verteidigung darauf verweist , dass Personalgespräche dazu dienten , Soldaten über Verwendungsplanungen der Personalführung zu informieren und in diesem Zusammenhang deren persönliche Motivationslage zu ergründen , um das dienstliche Bedürfnis und die privaten Lebensumstände des Soldaten möglichst gut miteinander zu vereinbaren , trifft dies zweifellos auf Personalgespräche im Vorfeld von Personalmaßnahmen zu .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Example 16** (doc_id: `53690`) (sent_id: `53690`)


An dieser Regelung hielt die Bundesregierung fest , obwohl der Bundesrat die Einbeziehung der einmaligen Einnahmen vorschlug ( BT-Drucks 16/2454 S 11 ) .

| Predicted | Gold |
|---|---|
| `Bundesregierung` | `Bundesregierung` |

**Missed by this rule (FN):**

- `Bundesrat` (ORG)
- `BT-Drucks 16/2454 S 11` (LIT)

**Example 17** (doc_id: `53746`) (sent_id: `53746`)


Das hat zur Folge , dass beim BSG über Kostenerinnerungen nach § 189 Abs 2 S 2 SGG der Senat in der Besetzung mit drei Berufsrichtern zu befinden hat .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Missed by this rule (FN):**

- `§ 189 Abs 2 S 2 SGG` (NRM)

**Example 18** (doc_id: `53773`) (sent_id: `53773`)


Vielmehr hat das LSG nach erfolgter Zurückverweisung durch das BSG nach umfangreicher weiterer Sachaufklärung den Anspruch des Klägers erneut geprüft und eine Entscheidung gefällt .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 19** (doc_id: `53777`) (sent_id: `53777`)


Am 14. November 2000 wurden den Aktionären der A-AG vereinbarungsgemäß ... Stück vinkulierte Namensaktien an der B-AG übertragen .

| Predicted | Gold |
|---|---|
| `A-AG` | `A-AG` |
| `B-AG` | `B-AG` |

**Example 20** (doc_id: `53791`) (sent_id: `53791`)


Zwischen den Vergleichszeichen besteht hinsichtlich der angegriffenen Waren Verwechslungsgefahr gemäß § 9 Abs. 1 Nr. 2 MarkenG , so dass der Widerspruch aus der Marke 30 2011 047 766 vom Deutschen Patent- und Markenamt zu Unrecht zurückgewiesen wurde .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `§ 9 Abs. 1 Nr. 2 MarkenG` (NRM)

**Example 21** (doc_id: `53862`) (sent_id: `53862`)


Patentansprüche 1 bis 13 vom 24. November 2017 , beim BPatG als 6. Hilfsantrag per Fax eingegangen am 27. November 2017

| Predicted | Gold |
|---|---|
| `BPatG` | `BPatG` |

**Example 22** (doc_id: `53867`) (sent_id: `53867`)


Zwar kann das LSG von einer Entscheidung ua des BSG auch dann abweichen , wenn es einen der höchstrichterlichen Rechtsprechung widersprechenden Rechtssatz nur sinngemäß und in scheinbar fallbezogene Ausführungen gekleidet entwickelt .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 23** (doc_id: `53939`) (sent_id: `53939`)


Dort hat der BFH lediglich ausgeführt , die Vergütung für die Hingabe eines partiarischen Darlehens könne auch umsatzabhängig ausgestaltet werden .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Example 24** (doc_id: `54007`) (sent_id: `54007`)


Das Unterrichtungsschreiben vom 14. November 2005 enthält weder Angaben zum Sitz der D P T S GmbH noch zu deren Anschrift , zum zuständigen Registergericht und zur Registernummer .

| Predicted | Gold |
|---|---|
| `D P T S GmbH` | `D P T S GmbH` |

**Example 25** (doc_id: `54059`) (sent_id: `54059`)


Wie der Kläger selbst vorträgt , wollte sich das LSG vielmehr nach seinem eigenen Verständnis im Rahmen der bereits vorliegenden Rechtsprechung des BSG halten , ist diesen aber in dem von ihm entschiedenen Fall lediglich deshalb nicht gefolgt , weil es - vom Kläger als unzutreffend gerügte - wesentliche Sachverhaltsunterschiede zu den bereits entschiedenen Fällen angenommen hat .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 26** (doc_id: `54064`) (sent_id: `54064`)


Der BFH prüft insofern nur , ob sie gegen Denkgesetze und Erfahrungssätze oder die anerkannten Auslegungsregeln verstößt .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Example 27** (doc_id: `54130`) (sent_id: `54130`)


b ) In Leasingfällen ( auch beim " Lease " im Rahmen eines " Sale-and-lease-back " ) geht der BFH - wie bereits in dem Urteil in BFHE 255 , 386 , BStBl II 2018 , 81 dargestellt - bei Anwendung des § 39 Abs. 2 Nr. 1 Satz 1 AO insbesondere von folgenden Grundsätzen aus :

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Missed by this rule (FN):**

- `Urteil in BFHE 255 , 386 , BStBl II 2018 , 81` (RS)
- `§ 39 Abs. 2 Nr. 1 Satz 1 AO` (NRM)

**Example 28** (doc_id: `54150`) (sent_id: `54150`)


Die Prüfung der Markenanmeldung muss daher nach der maßgeblichen Rechtsprechung des EuGH grundsätzlich streng und vollständig sein , um ungerechtfertigte Eintragungen zu vermeiden ( vgl. EuGH , GRUR 2003 , 604 Rn. 57 , 60 – Libertel ; BGH , GRUR 2014 , 565 Rn. 17 – smartbook ; Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 158 , 159 ) .

| Predicted | Gold |
|---|---|
| `EuGH` | `EuGH` |

**Missed by this rule (FN):**

- `EuGH , GRUR 2003 , 604 Rn. 57 , 60 – Libertel` (RS)
- `BGH , GRUR 2014 , 565 Rn. 17 – smartbook` (RS)
- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 158 , 159` (LIT)

**Example 29** (doc_id: `54212`) (sent_id: `54212`)


Ist es dem Kläger im Rahmen seiner deshalb nötigen Ermittlungen aufgrund des Verhaltens des FG-Präsidenten nicht möglich , diesen Verfahrensmangel zu substantiieren , so hat dies allein zur Folge , dass der BFH insoweit einen geringeren Maßstab der Darlegung des Verfahrensmangels genügen lassen muss .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

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

**Example 1** (doc_id: `53422`) (sent_id: `53422`)


Bei der vom FG zu beurteilenden gesetzlichen Prozessführungsbefugnis ( Beteiligtenstellung ) handelt es sich um eine Sachurteilsvoraussetzung , deren fehlerhafte Beurteilung einen Verfahrensmangel darstellt ( BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943 ; vom 23. November 2010 V B 133/09 , BFH / NV 2011 , 612 , Rz 5 ; jeweils m. w. N. ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`
- `BFH` — partial — pred is substring of gold: `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`
- `BFH` — similar text (different position): `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`

> overlaps gold: 3  |  likely missing annotation: 0

**Gold Entities:**

- `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`(RS)
- `vom 23. November 2010 V B 133/09 , BFH / NV 2011 , 612 , Rz 5`(RS)

**Example 2** (doc_id: `53446`) (sent_id: `53446`)


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

**Example 3** (doc_id: `53468`) (sent_id: `53468`)


Wie das vorliegende Verfahren zeigt , bietet die Anhörungsrüge die Möglichkeit , durch Zuordnungsschwierigkeiten entstehende Probleme zu bewältigen ( vgl. BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15 ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15`(RS)

**Example 4** (doc_id: `53474`) (sent_id: `53474`)


Beim Leasing ist - wie dargestellt - darauf abzustellen , ob der Herausgabeanspruch des Leasinggebers ( zivilrechtlichen Eigentümers ) noch eine wirtschaftliche Bedeutung hat ( grundlegend BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1. ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1.`(RS)

**Example 5** (doc_id: `53481`) (sent_id: `53481`)


Diese können insbesondere dann vorliegen , wenn die Gesamtstrafe sich nicht innerhalb des gesetzlichen Strafrahmens bewegt , die gebotene Begründung für die Gesamtstrafe fehlt , oder wenn die Besorgnis besteht , der Tatrichter habe sich von der Summe der Einzelstrafen leiten lassen ( BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris`(RS)

**Example 6** (doc_id: `53485`) (sent_id: `53485`)


Soweit die Elemente eines Bildzeichens nur die typischen Merkmale der in Rede stehenden Waren und Dienstleistungen darstellen oder sich in einfachen dekorativen Gestaltungsmitteln erschöpfen , an die sich der Verkehr etwa durch häufige Verwendung gewöhnt hat , wird diesem Zeichen im Allgemeinen wegen seines bloß beschreibenden Inhalts ebenfalls die konkrete Eignung fehlen , die mit ihm gekennzeichneten Waren oder Dienstleistungen von denjenigen anderer Herkunft zu unterscheiden ( vgl. BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband ; GRUR 2005 , 257 , 258 – Bürogebäude ; GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel ; GRUR 2001 , 239f. – Zahnpastastrang ; GRUR 2001 , 734 , 735 – Westie-Kopf ; BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband`
- `BPatG` — partial — pred is substring of gold: `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband`(RS)
- `GRUR 2005 , 257 , 258 – Bürogebäude`(RS)
- `GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel`(RS)
- `GRUR 2001 , 239f. – Zahnpastastrang`(RS)
- `GRUR 2001 , 734 , 735 – Westie-Kopf`(RS)
- `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`(RS)

**Example 7** (doc_id: `53490`) (sent_id: `53490`)


Es müssen letztlich besondere Verhaltensweisen sowohl des Berechtigten als auch des Verpflichteten vorliegen , die es rechtfertigen , die späte Geltendmachung des Rechts als mit Treu und Glauben unvereinbar und für den Verpflichteten als unzumutbar anzusehen ( vgl. BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27 mwN ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27`(RS)

**Example 8** (doc_id: `53508`) (sent_id: `53508`)


Beruhen vielmehr die Differenzen zwischen den Auffassungen von Sachverständigen darauf , dass diese von verschiedenen tatsächlichen Annahmen ausgehen , dann muss der Tatrichter , ggf nach weiterer Aufklärung , die für seine Überzeugungsbildung maßgebenden Tatsachen feststellen oder begründen , weshalb und zu wessen Lasten sie beweislos geblieben sind ( vgl BGH Urteil vom 23. 9. 1986 - VI ZR 261/85 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH Urteil vom 23. 9. 1986 - VI ZR 261/85`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH Urteil vom 23. 9. 1986 - VI ZR 261/85`(RS)

**Example 9** (doc_id: `53524`) (sent_id: `53524`)


Der BFH führte aus , die handelsrechtliche Zurechnung von Vermögensgegenständen entspreche im Wesentlichen der Regelung des § 39 Abs. 2 Nr. 1 Satz 1 AO ( z.B. BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1. ; vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a ) .

**False Positives:**

- `BFH` — similar text (different position): `BFH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH`(ORG)
- `§ 39 Abs. 2 Nr. 1 Satz 1 AO`(NRM)
- `BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1.`(RS)
- `vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a`(RS)

**Example 10** (doc_id: `53539`) (sent_id: `53539`)


cc ) Stellt ein Bewerber nicht innerhalb eines Monats nach Zugang der Abbruchmitteilung einen Antrag auf Erlass einer einstweiligen Verfügung , darf der Dienstherr nach der Rechtsprechung des Bundesverwaltungsgerichts darauf vertrauen , dass der Bewerber den Abbruch des Auswahlverfahrens nicht angreift , sondern sein Begehren im Rahmen einer neuen Ausschreibung weiterverfolgt ( vgl. BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14 ) .

**False Positives:**

- `BVerwG` — partial — pred is substring of gold: `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesverwaltungsgerichts`(ORG)
- `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14`(RS)

**Example 11** (doc_id: `53551`) (sent_id: `53551`)


Die Beklagte wird das der Klägerin zustehende Honorar unter Zugrundelegung eines festen RLV für die gesamte BAG neu zu ermitteln haben .

**False Positives:**

- `BAG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 12** (doc_id: `53582`) (sent_id: `53582`)


3. Wird einem Beteiligten das rechtliche Gehör dadurch versagt , dass es ihm nicht ermöglicht wird , an der mündlichen Verhandlung teilzunehmen , so ist davon auszugehen , dass dies für eine aufgrund dieser Verhandlung ergangenen Entscheidung ursächlich geworden ist ( vgl BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7 ) .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7`(RS)

**Example 13** (doc_id: `53583`) (sent_id: `53583`)


Denn die Hauptfunktion einer Marke liegt darin , die Ursprungsidentität der gekennzeichneten Waren und Dienstleistungen zu gewährleisten ( vgl. u. a. EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel ; BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006 ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel`
- `BGH` — partial — pred is substring of gold: `BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel`(RS)
- `BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006`(RS)

**Example 14** (doc_id: `53599`) (sent_id: `53599`)


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

**Example 15** (doc_id: `53600`) (sent_id: `53600`)


Auch danach setzt die Bejahung der Unterscheidungskraft unverändert voraus , dass das Zeichen geeignet sein muss , die beanspruchten Produkte als von einem bestimmten Unternehmen stammend zu kennzeichnen ( vgl. EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK`(RS)

**Example 16** (doc_id: `53614`) (sent_id: `53614`)


Soweit Personen dezentral untergebracht sind , ist es für die Bejahung einer Einrichtung erforderlich , dass die dezentrale Unterkunft zu den Räumlichkeiten der Einrichtung gehört , der Hilfebedürftige also in die Räumlichkeiten des Einrichtungsträgers eingegliedert ist ( vgl BSG SozR 4 - 3500 § 98 Nr 3 ) .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG SozR 4 - 3500 § 98 Nr 3`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG SozR 4 - 3500 § 98 Nr 3`(RS)

**Example 17** (doc_id: `53618`) (sent_id: `53618`)


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

**Example 18** (doc_id: `53623`) (sent_id: `53623`)


Nach der von der Einsprechenden zitierten BGH-Entscheidung „ Kommunikationskanal “ , muss für den Fachmann die im Anspruch bezeichnete Lehre „ unmittelbar und eindeutig “ als mögliche Ausführungsform der Erfindung den Ursprungsunterlagen entnehmbar sein .

**False Positives:**

- `BGH` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `53637`) (sent_id: `53637`)


aa ) Das Berufungsgericht ist zu der Einschätzung gelangt , die Klägerin habe nicht dargelegt , dass die Zulassung als Jaguar- und Land-Rover-Vertragswerkstatt eine Ressource darstelle , ohne die der Zugang zu dem nachgelagerten Endkundenmarkt nicht oder nicht sinnvoll möglich sei .

**False Positives:**

- `Jaguar` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 20** (doc_id: `53680`) (sent_id: `53680`)


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

**Example 21** (doc_id: `53688`) (sent_id: `53688`)


Allerdings beschränkt sich die Geltung des Grundsatzes der Bestenauslese im Bereich der Verwendungsentscheidungen auf Entscheidungen über - wie hier - höherwertige , die Beförderung in einen höheren Dienstgrad oder die Einweisung in die Planstelle einer höheren Besoldungsgruppe vorprägende Verwendungen ( vgl. klarstellend BVerwG , Beschluss vom 30. Januar 2014 - 1 WB 1.13 - juris Rn. 32 ) .

**False Positives:**

- `BVerwG` — partial — pred is substring of gold: `BVerwG , Beschluss vom 30. Januar 2014 - 1 WB 1.13 - juris Rn. 32`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerwG , Beschluss vom 30. Januar 2014 - 1 WB 1.13 - juris Rn. 32`(RS)

**Example 22** (doc_id: `53701`) (sent_id: `53701`)


Für eine solche Prognose des Arbeitgebers bedarf es ausreichend konkreter Anhaltspunkte ( BAG 27. Juli 2016 - 7 AZR 545/14 - Rn. 19 ; 24. September 2014 - 7 AZR 987/12 - Rn. 18 ; 7. Mai 2008 - 7 AZR 146/07 - Rn. 15 ; 7. April 2004 - 7 AZR 441/03 - zu II 2 a aa der Gründe ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 27. Juli 2016 - 7 AZR 545/14 - Rn. 19`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 27. Juli 2016 - 7 AZR 545/14 - Rn. 19`(RS)
- `24. September 2014 - 7 AZR 987/12 - Rn. 18`(RS)
- `7. Mai 2008 - 7 AZR 146/07 - Rn. 15`(RS)
- `7. April 2004 - 7 AZR 441/03 - zu II 2 a aa der Gründe`(RS)

**Example 23** (doc_id: `53707`) (sent_id: `53707`)


Da eine so weitgehende Selbstentäußerung des ausländischen Staates im Zweifel nicht zu vermuten ist ( BGH 30. Januar 2013 - III ZB 40/12 - Rn. 19 ) , dürfen die Umstände des Falles hinsichtlich des Vorliegens und der Reichweite eines Verzichts keinen Zweifel lassen ( BAG 29. Juni 2017 - 2 AZR 759/16 - Rn. 20 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH 30. Januar 2013 - III ZB 40/12 - Rn. 19`
- `BAG` — partial — pred is substring of gold: `BAG 29. Juni 2017 - 2 AZR 759/16 - Rn. 20`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH 30. Januar 2013 - III ZB 40/12 - Rn. 19`(RS)
- `BAG 29. Juni 2017 - 2 AZR 759/16 - Rn. 20`(RS)

**Example 24** (doc_id: `53717`) (sent_id: `53717`)


Bei Anerkennungsbeträgen handelt es sich um eine jener Massenerscheinungen , die ein typisierendes und pauschalierendes Vorgehen auch der Verwaltung rechtfertigen ( vgl. BVerfG , Beschluss vom 31. Mai 1988 - 1 BvR 520/83 - BVerfGE 78 , 214 < 227 > ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Beschluss vom 31. Mai 1988 - 1 BvR 520/83 - BVerfGE 78 , 214 < 227 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG , Beschluss vom 31. Mai 1988 - 1 BvR 520/83 - BVerfGE 78 , 214 < 227 >`(RS)

**Example 25** (doc_id: `53721`) (sent_id: `53721`)


Die Regelung ist daher geeignet , die Befristung von Arbeitsverträgen mit programmgestaltenden Mitarbeitern bei Rundfunkanstalten zu rechtfertigen ( BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101 ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101`(RS)

**Example 26** (doc_id: `53760`) (sent_id: `53760`)


Etwas anderes gilt insbesondere dann , wenn der Arbeitgeber seine Tarifgebundenheit in einer dem Arbeitnehmer hinreichend erkennbaren Weise zur auflösenden Bedingung der Bezugnahme gemacht hat ( BAG 18. April 2007 - 4 AZR 652/05 - Rn. 32 , 40 , BAGE 122 , 74 ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 18. April 2007 - 4 AZR 652/05 - Rn. 32 , 40 , BAGE 122 , 74`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 18. April 2007 - 4 AZR 652/05 - Rn. 32 , 40 , BAGE 122 , 74`(RS)

**Example 27** (doc_id: `53763`) (sent_id: `53763`)


Wird als Verfahrensmangel gerügt , das FG habe einen gestellten Beweisantrag übergangen ( Rüge mangelnder Sachaufklärung gemäß § 76 Abs. 1 Satz 1 FGO ) , so ist darzulegen , welche Tatfrage aufklärungsbedürftig ist , welche Beweismittel das FG zu welchem Beweisthema nicht erhoben hat , die genauen Fundstellen ( Schriftsatz mit Datum und Seitenzahl , Terminprotokoll ) , in denen die Beweismittel und Beweisthemen angeführt worden sind , das voraussichtliche Ergebnis der Beweisaufnahme , inwiefern das Urteil des FG aufgrund dessen sachlich-rechtlicher Auffassung auf der unterbliebenen Beweisaufnahme beruhen kann und dass die Nichterhebung der Beweise vor dem FG rechtzeitig gerügt worden ist oder aufgrund des Verhaltens des FG nicht mehr vor diesem gerügt werden konnte ( ständige BFH-Rechtsprechung , z.B. Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85 , und vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125 , m. w. N. ) .

**False Positives:**

- `BFH` — similar text (different position): `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`
- `BFH` — similar text (different position): `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 76 Abs. 1 Satz 1 FGO`(NRM)
- `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`(RS)
- `vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125`(RS)

**Example 28** (doc_id: `53767`) (sent_id: `53767`)


Ob der Erwerb eigener Anteile auf der Gesellschaftsebene entsprechend der durch das BilMoG geänderten handelsrechtlichen Vorschriften ( Einfügung des § 272 Abs. 1a und 1b HGB ) steuerrechtlich nicht mehr als Erwerbsvorgang anzusehen , sondern nunmehr als " Teilliquidation " und daher " wie " eine Kapitalherabsetzung zu behandeln ist ( so BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f. ; zustimmend z.B. Blumenberg / Lechner , DB 2014 , 141 , 147 ; kritisch z.B. Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55 ) , kann im Streitfall mangels Erheblichkeit für die Entscheidung offenbleiben .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f.`
- `DB` — partial — pred is substring of gold: `Blumenberg / Lechner , DB 2014 , 141 , 147`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BilMoG`(NRM)
- `§ 272 Abs. 1a und 1b HGB`(NRM)
- `BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f.`(REG)
- `Blumenberg / Lechner , DB 2014 , 141 , 147`(LIT)
- `Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55`(LIT)

**Example 29** (doc_id: `53775`) (sent_id: `53775`)


nicht nur das erstmalige Legen eines Hausanschlusses , sondern auch Arbeiten zur Erneuerung oder zur Reduzierung von Wasseranschlüssen unter die Steuerermäßigung fallen ( vgl. BGH-Urteil in HFR 2012 , 1110 , Rz 20 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH-Urteil in HFR 2012 , 1110 , Rz 20`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH-Urteil in HFR 2012 , 1110 , Rz 20`(RS)

</details>

---

</details>

---

<details>
<summary>💣 Least Precise Rules</summary>

## `Generic Organization Patterns`

**F1:** 0.009 | **Precision:** 0.103 | **Recall:** 0.005  

**Format:** `regex`  
**Rule ID:** `70b265e1`  
**Description:**
Matches generic organization patterns like 'Firma X GmbH', 'Senat des Y', 'Bundesministerium', 'Landesamt', etc., using structural patterns to generalize.

**Content:**
```
\b(?:Firma|Gesellschaft|AG|GmbH|KG|e\.\s*V\.|Verband|Vereinigung|Dienst|Amt|Beh\u00f6rde|Ministerium|Klinik|Krankenhaus|Schule|Schulzentrum|Senat|Kammer|Abteilung|Stelle|Gericht|Landesgericht|Oberlandesgericht|Bundesgericht|Finanzgericht|Arbeitsgericht|Sozialgericht|Verwaltungsgericht|Amtsgericht|Staatsanwaltschaft|Landratsamt|Post|Botschaft|Konsulat|Kanzlei|Korporation|Konzern|Gruppe|Fonds|Institut|Akademie|Hochschule|Universität|Bundesagentur|Bundesamt|Landesamt|Staat|Republik|Union|Kommission|Parlament|Bundestag|Landtag|Senat|Kabinett|Regierung|Verwaltung|Organisation|Institution|Behörde|Dienstleistung|Gewerkschaft|Arbeitsgeberverband|Arbeitnehmerverband|Krankenkasse|Rentenversicherung|Sozialversicherung|Versicherung|Bank|Finanzinstitut|Holding|Investment|Fonds|Stiftung|Organisation|Institution|Behörde|Dienstleistung|Gewerkschaft|Arbeitsgeberverband|Arbeitnehmerverband|Krankenkasse|Rentenversicherung|Sozialversicherung|Versicherung|Bank|Finanzinstitut|Holding|Investment|Fonds|Stiftung)\s+(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|\d+|\u2026|\.)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.103 | 0.005 | 0.009 | 39 | 4 | 35 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 4 | 35 | 781 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53725`) (sent_id: `53725`)


4. Auf die sofortige Beschwerde der Staatsanwaltschaft Düsseldorf hob das Oberlandesgericht Hamm mit angegriffenem Beschluss vom 28. Juli 2015 den Beschluss des Landgerichts Paderborn auf und ordnete die weitere Vollstreckung der Unterbringung des Beschwerdeführers in der Sicherungsverwahrung an .

| Predicted | Gold |
|---|---|
| `Oberlandesgericht Hamm` | `Oberlandesgericht Hamm` |

**Missed by this rule (FN):**

- `Staatsanwaltschaft Düsseldorf` (ORG)
- `Landgerichts Paderborn` (ORG)

**Example 1** (doc_id: `54861`) (sent_id: `54861`)


Das Arbeitsgericht Zwickau verurteilte die Beklagte am 22. April 2015 ( - 9 Ca 146/15 - ) , das abgebrochene Stellenbesetzungsverfahren 01/2014 fortzuführen und über die Bewerbung des Klägers erneut zu entscheiden .

| Predicted | Gold |
|---|---|
| `Arbeitsgericht Zwickau` | `Arbeitsgericht Zwickau` |

**Missed by this rule (FN):**

- `22. April 2015 ( - 9 Ca 146/15 - )` (RS)

**Example 2** (doc_id: `56732`) (sent_id: `56732`)


Dieser half das Amtsgericht Luckenwalde mit Beschluss vom 11. November 2013 nicht ab .

| Predicted | Gold |
|---|---|
| `Amtsgericht Luckenwalde` | `Amtsgericht Luckenwalde` |

**Example 3** (doc_id: `57841`) (sent_id: `57841`)


2. Das Amtsgericht Dieburg gab der Klage mit Urteil vom 7. Dezember 2012 statt , erklärte die Zwangsvollstreckung aus dem Vollstreckungsbescheid insgesamt für unzulässig und verurteilte den Beklagten , die vollstreckbare Ausfertigung an den Beschwerdeführer herauszugeben ; alle Forderungen des Beklagten gegen den Beschwerdeführer seien getilgt .

| Predicted | Gold |
|---|---|
| `Amtsgericht Dieburg` | `Amtsgericht Dieburg` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53560`) (sent_id: `53560`)


Welches Volljährigkeitsalter nach dem Recht der Republik Guinea gilt , wird in der obergerichtlichen Rechtsprechung uneinheitlich beantwortet .

**False Positives:**

- `Republik Guinea` — type mismatch — same span as gold: `Republik Guinea`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Republik Guinea`(LOC)

**Example 1** (doc_id: `53675`) (sent_id: `53675`)


Sie endet bei Wegfall der Erwerbsminderungsrente aus der gesetzlichen Rentenversicherung .

**False Positives:**

- `Rentenversicherung .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `53856`) (sent_id: `53856`)


c ) Die nach § 9 Abs. 1 Satz 3 DVO.EKD aF erfolgte Zuordnung der Klägerin zu höchstens Stufe 2 der Entgeltgruppe 14 DVO.EKD verstieß nicht gegen das Recht der Europäischen Union .

**False Positives:**

- `Union .` — positional overlap with gold: `Europäischen Union`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 9 Abs. 1 Satz 3 DVO.EKD aF`(REG)
- `DVO.EKD`(REG)
- `Europäischen Union`(ORG)

**Example 3** (doc_id: `53889`) (sent_id: `53889`)


Er zielt sachlich auf einen objektiv bestehenden Bedarf an zusätzlichem richterlichem Personal bei einem konkreten Verwaltungsgericht .

**False Positives:**

- `Verwaltungsgericht .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `54197`) (sent_id: `54197`)


Gleiches gilt für die Klagebefugnis des Empfangsbevollmächtigten einer atypisch stillen Gesellschaft .

**False Positives:**

- `Gesellschaft .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `54451`) (sent_id: `54451`)


Die vom Berufungsgericht herangezogene Instanzrechtsprechung ( vgl. LG Mönchengladbach , Urteile vom 11. Juli 2006 - 2 S 176/05 , juris , und vom 7. April 2006 - 2 S 172/05 , juris ; LG Lübeck , NJW-RR 1999 , 1655 ; LG Mainz , NJW-RR 1998 , 631 ; AG Donaueschingen , Urteil vom 25. Juli 2002 - 31 C 176/02 , juris ; AG Köpenick , NJW 1996 , 1005 ) bezieht sich im Übrigen nicht auf Verträge über die Schaltung einer Werbeanzeige unter einer konkret bezeichneten Domain .

**False Positives:**

- `AG Donaueschingen` — partial — pred is substring of gold: `AG Donaueschingen , Urteil vom 25. Juli 2002 - 31 C 176/02 , juris`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `LG Mönchengladbach , Urteile vom 11. Juli 2006 - 2 S 176/05 , juris`(RS)
- `vom 7. April 2006 - 2 S 172/05 , juris`(RS)
- `LG Lübeck , NJW-RR 1999 , 1655`(RS)
- `LG Mainz , NJW-RR 1998 , 631`(RS)
- `AG Donaueschingen , Urteil vom 25. Juli 2002 - 31 C 176/02 , juris`(RS)
- `AG Köpenick , NJW 1996 , 1005`(RS)

**Example 6** (doc_id: `54500`) (sent_id: `54500`)


Die Sache wird an das Finanzgericht Rheinland-Pfalz zurückverwiesen .

**False Positives:**

- `Finanzgericht Rheinland` — partial — pred is substring of gold: `Finanzgericht Rheinland-Pfalz`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Finanzgericht Rheinland-Pfalz`(ORG)

**Example 7** (doc_id: `54697`) (sent_id: `54697`)


Zwei dieser Gruppen umfassen nur ein Kriterium ( Gruppe 1 : Sanitätsstabsoffizier Zahnarzt < Rang 1 > ; Gruppe 3 : Leiter einer Zahnärztlichen Behandlungseinrichtung < Rang 3 > ) , eine Gruppe besteht aus vier Kriterien ( Gruppe 2 : Fachzahnarzt Oralchirurgie < Rang 2 > , Curriculare Fortbildung Parodontologie < Rang 4 > , Curriculare Fortbildung Prothetik < Rang 5 > und Curriculare Fortbildung CMD < Rang 6 > ) .

**False Positives:**

- `Gruppe 1` — no gold match — likely missing annotation
- `Gruppe 3` — no gold match — likely missing annotation
- `Gruppe 2` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 3

**Example 8** (doc_id: `54703`) (sent_id: `54703`)


b ) Beim Richter auf Zeit nach § 18 VwGO ergibt sich eine spezifische , mit Art. 97 Abs. 1 GG unvereinbare Möglichkeit der vermeidbaren Einflussnahme durch die Exekutive auf seine richterliche Tätigkeit aus der durch den Richterstatus nur vorübergehend gesicherten persönlichen Unabhängigkeit im Sinne von Art. 97 Abs. 2 GG und der danach absehbar ( wieder ) bestehenden stärkeren Abhängigkeit der beruflichen Karriere des Richters gerade vom Staat .

**False Positives:**

- `Staat .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 18 VwGO`(NRM)
- `Art. 97 Abs. 1 GG`(NRM)
- `Art. 97 Abs. 2 GG`(NRM)

**Example 9** (doc_id: `54738`) (sent_id: `54738`)


b ) Die gegen die Disziplinarverfügung gerichtete Klage wies das Verwaltungsgericht Osnabrück mit Urteil vom 19. August 2011 - 9 A 1/11 - ab .

**False Positives:**

- `Verwaltungsgericht Osnabr` — partial — pred is substring of gold: `Verwaltungsgericht Osnabrück mit Urteil vom 19. August 2011 - 9 A 1/11 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Verwaltungsgericht Osnabrück mit Urteil vom 19. August 2011 - 9 A 1/11 -`(RS)

**Example 10** (doc_id: `54827`) (sent_id: `54827`)


2 ) Soweit die Klägerin weiter rügt , das LSG habe gegen § 103 SGG verstoßen , weil das Gericht Beweisanträgen ohne hinreichende Begründung nicht gefolgt sei , genügt sie mit ihrer Beschwerdebegründung ebenfalls nicht den Anforderungen des § 160a Abs 2 S 3 SGG .

**False Positives:**

- `Gericht Beweisantr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 103 SGG`(NRM)
- `§ 160a Abs 2 S 3 SGG`(NRM)

**Example 11** (doc_id: `55082`) (sent_id: `55082`)


Demgemäß hat der Senat Schüler , die im häuslichen Bereich unterrichtsvorbereitend ein Werkstück erstellen ( BSG vom 1. 2. 1979 - 2 RU 107/77 - SozR 2200 § 539 Nr 54 ) , ebenso wenig für versichert erachtet wie solche , die für die schulische Foto-AG in der Altstadt ohne weitere Aufsicht fotografieren ( BSG vom 30. 5. 1988 - 2 RU 5/88 - Juris ) .

**False Positives:**

- `Senat Sch` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG vom 1. 2. 1979 - 2 RU 107/77 - SozR 2200 § 539 Nr 54`(RS)
- `Foto-AG`(ORG)
- `BSG vom 30. 5. 1988 - 2 RU 5/88 - Juris`(RS)

**Example 12** (doc_id: `55122`) (sent_id: `55122`)


So bietet beispielsweise die Firma Pointer „ Wohlfühlfarben für die Wohnung “ an ; in einem der Anmelderin übersandten Internetausdruck heißt es hierzu : „ Farben gezielt einsetzen .

**False Positives:**

- `Firma Pointer` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 13** (doc_id: `55334`) (sent_id: `55334`)


Mit Wirkung vom 1. März 2011 ernannte ihn die Ministerin für Wissenschaft , Forschung und Kultur erneut unter Berufung in das Beamtenverhältnis auf Zeit für die Dauer von sechs Jahren zum Kanzler der Hochschule .

**False Positives:**

- `Hochschule .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 14** (doc_id: `55615`) (sent_id: `55615`)


Neben der Beklagten zu 1. , in deren Betrieb ein Betriebsrat gewählt war , gehören weitere rechtlich eigenständige Standortgesellschaften zur sog. w Gruppe .

**False Positives:**

- `Gruppe .` — positional overlap with gold: `w Gruppe`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `w Gruppe`(ORG)

**Example 15** (doc_id: `55719`) (sent_id: `55719`)


Dies gilt auch für die Strukturprinzipien des Art. 33 Abs. 5 GG , die einem Ausgleich mit anderen Gütern nicht von vornherein verschlossen sind ( vgl. Kees , Der Staat 54 < 2015 > , S. 63 < 75 > ) .

**False Positives:**

- `Staat 54` — partial — pred is substring of gold: `Kees , Der Staat 54 < 2015 > , S. 63 < 75 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 33 Abs. 5 GG`(NRM)
- `Kees , Der Staat 54 < 2015 > , S. 63 < 75 >`(LIT)

**Example 16** (doc_id: `56089`) (sent_id: `56089`)


Der Endoskopkopf 1 weist auch eine Ventilaufnahme ( zylindrische Kammer 4 ) auf , durch die die beiden Kanäle ( erster Einlass 2 - erster Auslass 5 ; zweiter Einlass 3 - zweiter Auslass 6 ) führen [ = Merkmal M4 ] .

**False Positives:**

- `Kammer 4` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 17** (doc_id: `56323`) (sent_id: `56323`)


aa ) Mit Urteil vom 24. 3. 2016 ( B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 ) hat der Senat bereits entschieden , dass es für den Beginn der aufgeschobenen Versicherungspflicht nach § 7a Abs 6 S 1 SGB IV - mit Wirkung für alle Zweige der Sozialversicherung - auf die Bekanntgabe einer ( ersten ) Entscheidung der Deutschen Rentenversicherung Bund über das Bestehen von " Beschäftigung " ankommt und nicht auf eine ( spätere ) - diese unzulässige Elementenfeststellung korrigierende - Entscheidung über " Versicherungspflicht wegen Beschäftigung " .

**False Positives:**

- `Rentenversicherung Bund` — partial — pred is substring of gold: `Deutschen Rentenversicherung Bund`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil vom 24. 3. 2016 ( B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 )`(RS)
- `§ 7a Abs 6 S 1 SGB IV`(NRM)
- `Deutschen Rentenversicherung Bund`(ORG)

**Example 18** (doc_id: `56966`) (sent_id: `56966`)


Dies betrifft sowohl die angeordneten Tätigkeiten in der Abteilung Standesamt und Gerichtliche Angelegenheiten als auch diejenigen für die Visaabteilung .

**False Positives:**

- `Abteilung Standesamt` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `57172`) (sent_id: `57172`)


( b ) Gemessen hieran ist das Oberlandesgericht ohne ausreichende Ermittlungen zu dem Ergebnis gelangt , dass die Volljährigkeit ( auch ) nach dem Recht der Republik Guinea mit der Vollendung des 18. Lebensjahres eintrete .

**False Positives:**

- `Republik Guinea` — type mismatch — same span as gold: `Republik Guinea`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Republik Guinea`(LOC)

**Example 20** (doc_id: `57238`) (sent_id: `57238`)


Am 7. April 1999 leistete diese auf die noch ausstehende Stammeinlage für den an sie abgetretenen Geschäftsanteil den offenen Betrag von 12.500 DM an die Gesellschaft .

**False Positives:**

- `Gesellschaft .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 21** (doc_id: `57554`) (sent_id: `57554`)


festzustellen , dass auf das Arbeitsverhältnis der Parteien der Kirchliche Arbeitnehmerinnen Tarifvertrag , abgeschlossen zwischen dem Verband kirchlicher und diakonischer Anstellungsträger Nordelbien sowie der Gewerkschaft Kirche und Diakonie und ver.di , Landesbezirke Hamburg und Nord , andererseits vom 1. Dezember 2006 Anwendung finde .

**False Positives:**

- `Gewerkschaft Kirche` — partial — pred is substring of gold: `Gewerkschaft Kirche und Diakonie`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Kirchliche Arbeitnehmerinnen Tarifvertrag`(REG)
- `Verband kirchlicher und diakonischer Anstellungsträger Nordelbien`(ORG)
- `Gewerkschaft Kirche und Diakonie`(ORG)
- `ver.di , Landesbezirke Hamburg und Nord`(ORG)

**Example 22** (doc_id: `57565`) (sent_id: `57565`)


b ) Entgegen der Auffassung des Senats ist es im Hinblick auf das äußere Bild der Neutralität und Unparteilichkeit nicht nur bedenklich , wenn ein Richter auf Zeit in Verfahren entscheiden würde , in denen die Stammbehörde des Richters oder eine dieser vorgesetzte Behörde Beteiligte ist .

**False Positives:**

- `Behörde Beteiligte` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 23** (doc_id: `57748`) (sent_id: `57748`)


Sie verfügten bei Aufnahme der Tätigkeit jeweils über eine Befreiungsentscheidung der Bundesversicherungsanstalt für Angestellte als Rechtsvorgängerin der beklagten Deutschen Rentenversicherung Bund .

**False Positives:**

- `Rentenversicherung Bund` — partial — pred is substring of gold: `Deutschen Rentenversicherung Bund`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Deutschen Rentenversicherung Bund`(ORG)

**Example 24** (doc_id: `58275`) (sent_id: `58275`)


Dabei ist der Ausgangspunkt jeweils identisch , wonach gemäß dem - bislang nicht ausdrücklich aufgehobenen - Art. 443 des Code Civil der Republik Guinea die Volljährigkeit auf das vollendete 21. Lebensjahr festgesetzt wird .

**False Positives:**

- `Republik Guinea` — partial — pred is substring of gold: `Art. 443 des Code Civil der Republik Guinea`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 443 des Code Civil der Republik Guinea`(NRM)

**Example 25** (doc_id: `58309`) (sent_id: `58309`)


a ) Das Landesarbeitsgericht hat angenommen , mit der in Punkt „ Siebtens “ des Arbeitsvertrags vereinbarten Anwendbarkeit der arbeitsrechtlichen Vorschriften der deutschen Gesetzgebung sei auch § 4 KSchG vereinbart und mithin das Erfordernis einer fristgerechten Klageerhebung vor einem deutschen Gericht .

**False Positives:**

- `Gericht .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 4 KSchG`(NRM)

**Example 26** (doc_id: `58567`) (sent_id: `58567`)


Darüber hinaus ist darauf hinzuweisen , dass über den bloßen , auch in der Druckschrift D10 zum Ausdruck gebrachten Wunsch hinaus auch die Streitpatentschrift weder in den Patentansprüchen noch an anderer Stelle Angaben dazu macht , welche Maßnahmen ergriffen werden sollen , um polymerisolierte Kabel für HGÜ-Zwecke verwenden zu können .

**False Positives:**

- `Stelle Angaben` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 27** (doc_id: `58736`) (sent_id: `58736`)


Daher kann dahinstehen , ob die Vorschrift - wie das Landesarbeitsgericht meint - generell eine Spezialregelung gegenüber der Ausschlussfrist des § 37 Abs. 1 TVöD darstellt ( so wohl auch Breier / Dassau / Kiefer / Lang / Langenbrinck TVöD Stand 1/2018 B 2.2 § 26 TVÜ-Bund Erl. 2 Rn. 6 ; Litschen in Adam / Bauer / Bettenhausen / ua. Tarifrecht der Beschäftigten im öffentlichen Dienst Stand Dezember 2015 Teil II § 26 TVÜ-Bund B I Rn. 4 ; Clemens / Scheuring / Steingen / Wiese TVöD Stand Januar 2018 Teil IV / 3 Rn. 372 ) , oder ob sich diese lediglich auf den Wechsel in das neue tarifliche Entgeltsystem bezieht und es hinsichtlich der sich aus der Ausübung des Antragsrechts folgenden Zahlungsansprüche bei der allgemeinen Ausschlussfrist des § 37 Abs. 1 TVöD verbleibt ( so für § 29a Abs. 4 Satz 1 TVÜ-Länder BeckOK TV-L / Dannenberg Stand 1. Januar 2013 TVÜ-Länder § 29a Rn. 38 ; Augustin ZTR 2012 , 484 ) und wann diese ggf. fällig werden .

**False Positives:**

- `Dienst Stand Dezember` — partial — pred is substring of gold: `Litschen in Adam / Bauer / Bettenhausen / ua. Tarifrecht der Beschäftigten im öffentlichen Dienst Stand Dezember 2015 Teil II § 26 TVÜ-Bund B I Rn. 4`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 37 Abs. 1 TVöD`(REG)
- `Breier / Dassau / Kiefer / Lang / Langenbrinck TVöD Stand 1/2018 B 2.2 § 26 TVÜ-Bund Erl. 2 Rn. 6`(LIT)
- `Litschen in Adam / Bauer / Bettenhausen / ua. Tarifrecht der Beschäftigten im öffentlichen Dienst Stand Dezember 2015 Teil II § 26 TVÜ-Bund B I Rn. 4`(LIT)
- `Clemens / Scheuring / Steingen / Wiese TVöD Stand Januar 2018 Teil IV / 3 Rn. 372`(LIT)
- `§ 37 Abs. 1 TVöD`(REG)
- `§ 29a Abs. 4 Satz 1 TVÜ-Länder`(REG)
- `BeckOK TV-L / Dannenberg Stand 1. Januar 2013 TVÜ-Länder § 29a Rn. 38`(LIT)
- `Augustin ZTR 2012 , 484`(LIT)

**Example 28** (doc_id: `58927`) (sent_id: `58927`)


Der in § 1 KAT erwähnte Kirchliche Tarifvertrag Diakonie ( KTD , GVOBl. NEK S. 317 ) wurde bereits am 15. August 2002 zwischen dem Verband kirchlicher und diakonischer Anstellungsträger Nordelbien einerseits und der Gewerkschaft Kirche und Diakonie , der IG Bauen-Agrar-Umwelt , Bundesvorstand , sowie von ver.di andererseits geschlossen .

**False Positives:**

- `Gewerkschaft Kirche` — partial — pred is substring of gold: `Gewerkschaft Kirche und Diakonie`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 1 KAT`(REG)
- `Kirchliche Tarifvertrag Diakonie ( KTD , GVOBl. NEK S. 317 )`(REG)
- `Verband kirchlicher und diakonischer Anstellungsträger Nordelbien`(ORG)
- `Gewerkschaft Kirche und Diakonie`(ORG)
- `IG Bauen-Agrar-Umwelt , Bundesvorstand`(ORG)
- `ver.di`(ORG)

**Example 29** (doc_id: `59299`) (sent_id: `59299`)


Im Streitfall war der Kläger indessen schon kein beherrschender Gesellschafter-Geschäftsführer der GmbH .

**False Positives:**

- `GmbH .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Quoted Organization Names`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `c1f44934`  
**Description:**
Matches organization names enclosed in German quotation marks, but only when preceded by context indicating an organization (e.g., 'Firma', 'Marke', 'Name', 'der', 'des') to avoid matching product names or streets.

**Content:**
```
(?:Firma|Marke|Name|der|des|bei|von|aus|in)\s*\u201e\s*([A-Z][a-zA-Z\u00e4\u00f6\u00fc\u00df\s]+(?:\s+[A-Z][a-zA-Z\u00e4\u00f6\u00fc\u00df\s]+)*)\s*\u201c
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 31 | 0 | 31 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 0 | 31 | 774 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53634`) (sent_id: `53634`)


Aus deren Sicht könne das Wort „ targeting “ auch andere Bedeutungen haben , wie etwa „ Zielausrichtung “ oder „ Zielbestimmung “ .

**False Positives:**

- `Zielbestimmung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `53933`) (sent_id: `53933`)


Weder die Studie der Holy Fashion Group vom 24. Februar 2016 ( Bl. 49/50 d. A. ) noch die im Amtsverfahren vorgelegten Unterlagen zu Marktforschungsergebnissen einer „ Brigitte “ -Studie , Internetausdrucken zu Showrooms und Verkaufsstätten von Waren der Marke „ JOOP “ oder die beigefügten Urteile sind geeignet , einen entsprechenden Benutzungsnachweis für mit der Marke gekennzeichnete Dienstleistungen zu erbringen .

**False Positives:**

- `JOOP ` — partial — pred is substring of gold: `„ JOOP “`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Holy Fashion Group`(ORG)
- `„ JOOP “`(ORG)

**Example 2** (doc_id: `54038`) (sent_id: `54038`)


Er hat insbesondere angegeben , dass er am 16. Mai 2006 als Entwicklungsleiter im Bereich Sicherheitstechnik das Vertriebsfreigabedokument E4a für diese Produktfamilie unterzeichnet habe ( vgl. auch die beiden Felder „ Date “ und „ Signature ISC spokesperson “ am Ende des Dokuments E4a ) .

**False Positives:**

- `Date ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 3** (doc_id: `54137`) (sent_id: `54137`)


Dagegen beanspruche die Anmelderin vorliegend im Wesentlichen „ chemische Hilfsmaterialien “ , welche sich an Handwerker richteten , die die Waren alleine aufgrund ihrer ( technischen ) Funktion und Qualität erwerben würden , für die ein „ Wohlfühleffekt “ keine Rolle spiele .

**False Positives:**

- `Wohlfühleffekt ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `54316`) (sent_id: `54316`)


Bereits in der Internet-Werbung der „ Kooperationskasse “ fand der qualifizierte Nachrang Erwähnung , auch wenn die mit ihr verbundene Bedingung ( „ Rückforderung darf nicht zur Insolvenz führen “ ) als „ theoretisch “ bezeichnet wurde .

**False Positives:**

- `Kooperationskasse ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `54681`) (sent_id: `54681`)


Ein „ Klosterhof “ ist die den deutschen Verbrauchern ohne weiteres verständliche Bezeichnung einer Gebäudeteils einer Klosteranlage und eines Hofes im Sinn eines Anwesens und bäuerlich landwirtschaftlichen Betriebs , das sich durch die Bewirtschaftung durch oder die Zugehörigkeit zu einem Kloster auszeichnet .

**False Positives:**

- `Klosterhof ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 6** (doc_id: `54871`) (sent_id: `54871`)


Zudem könne der Schutzgegenstand in Anwendung der BGH-Entscheidung „ Weinkaraffe “ ( GRUR 2012 , 1139 ) auch durch Auslegung unter Berücksichtigung der Beschreibung sowie der Erzeugnisangabe aus der „ Schnittmenge “ der gemeinsamen Merkmale ermittelt werden .

**False Positives:**

- `Schnittmenge ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BGH-Entscheidung „ Weinkaraffe “ ( GRUR 2012 , 1139 )`(RS)

**Example 7** (doc_id: `54903`) (sent_id: `54903`)


Die angesprochenen Verkehrskreise seien durch Marken wie „ Facebook “ ( Jahrbuch ) , „ Soundcloud “ ( Klangwolke ) oder „ My Space “ ( Mein Raum ) gewohnt , dass markenrechtlich geschützte Bezeichnungen für bestimmte Netzwerke durch die Zusammensetzung und Neukreation an sich beschreibender Einzelelemente entstünden .

**False Positives:**

- `My Space ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `54947`) (sent_id: `54947`)


Soweit die Widersprechende sich darauf beruft , dass sie den größten Online-Shop der Welt betreibe , hat sie nicht vorgetragen , dass dies unter der Marke „ Fire “ geschehe , so dass diese Tatsache für eine etwaige Steigerung der Kennzeichnungskraft ohne Relevanz ist .

**False Positives:**

- `Fire ` — partial — pred is substring of gold: `„ Fire “`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `„ Fire “`(ORG)

**Example 9** (doc_id: `55014`) (sent_id: `55014`)


Voraussetzung hierfür ist ein entsprechendes „ Einvernehmen zwischen Arbeitgeber und Arbeitnehmer “ .

**False Positives:**

- `Einvernehmen zwischen Arbeitgeber und Arbeitnehmer ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 10** (doc_id: `55132`) (sent_id: `55132`)


a ) Wie das Deutsche Patent- und Markenamt zutreffend ausgeführt hat , ist das Anmeldezeichen zum Zeitpunkt seiner Anmeldung am 20. Januar 2016 von den überwiegend angesprochenen Fachverkehrskreisen , aber - insbesondere in Bezug auf die Dienstleistungen der Klassen 40 und 44 - auch von Auftraggebern oder Empfängern zahntechnischer oder -medizinischer Dienstleistungen im Sinne von „ CAD Labor “ oder „ Labor , das CAD einsetzt “ verstanden worden .

**False Positives:**

- `CAD Labor ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutsche Patent- und Markenamt`(ORG)

**Example 11** (doc_id: `55566`) (sent_id: `55566`)


ccc ) „ Music ” entspricht dem deutschen Wort „ Musik ” ( Langenscheidts Schulwörterbuch Englisch , 1986 ) und hat die Bedeutung „ Tonkunst “ , „ Komposition “ oder „ Musikstücke “ ( Duden - Die deutsche Rechtschreibung , 26. Aufl. 2013 ) .

**False Positives:**

- `Musikstücke ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Duden`(ORG)

**Example 12** (doc_id: `55629`) (sent_id: `55629`)


Zwar hat das Arbeitsgericht im Tatbestand , den das Landesarbeitsgericht in Bezug genommen hat , ausgeführt , die maximale Wasserverdrängung der „ G “ betrage 6,5 m³ .

**False Positives:**

- `G ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 13** (doc_id: `55768`) (sent_id: `55768`)


Schließlich ließ sich der „ Vorstand von Neudeutschland “ die Berechtigung einräumen , die Einzahlung für die Verwirklichung eines Projekts zusammenzulegen .

**False Positives:**

- `Vorstand von Neudeutschland ` — partial — gold is substring of pred: `Neudeutschland`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Neudeutschland`(ORG)

**Example 14** (doc_id: `56153`) (sent_id: `56153`)


III. Sollten nach den vom Landesarbeitsgericht noch zu treffenden Feststellungen die Befähigung des Klägers , die von ihm befahrene Binnenwasserstraße und die technische Ausstattung der „ G “ den tariflichen Anforderungen entsprechen , kommt ein Vergütungsanspruch des Klägers nach der Entgeltgruppe 8 der Anlage 1 zum TV EntgO Bund grundsätzlich in Betracht .

**False Positives:**

- `G ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Anlage 1 zum TV EntgO Bund`(REG)

**Example 15** (doc_id: `56195`) (sent_id: `56195`)


Vergleichbar gebildete Bezeichnungen wie „ Frau und Wirtschaft “ , „ Erfahrung ist Zukunft “ , „ Technik und Wirtschaft “ oder „ Recycling ist Zukunft “ würden bereits beschreibend verwendet .

**False Positives:**

- `Recycling ist Zukunft ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 16** (doc_id: `56254`) (sent_id: `56254`)


Ein großer Teil des angesprochenen Verkehrskreises sei der französischen Sprache insoweit mächtig , dass er die Bedeutung von „ Petit Filou “ , nämlich „ kleiner Schlingel “ , „ kleiner Spitzbub “ , verstehe , zumindest sei der Ausdruck im deutschsprachigen Raum bekannt .

**False Positives:**

- `Petit Filou ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 17** (doc_id: `56784`) (sent_id: `56784`)


c ) Der Senat teilt die Auffassung der Markenstelle , dass das angesprochene Publikum das umgedrehte Ausrufezeichen ohne analysierende Betrachtung und gedankliche Zwischenschritte zwanglos als Ersetzung des Buchstaben „ I / i “ lesen und damit als „ WIR “ oder „ WiR “ auffassen wird .

**False Positives:**

- `WiR ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 18** (doc_id: `56845`) (sent_id: `56845`)


Infolge der Intransparenz des Merkmals der „ Geeignetheit “ und der ersatzlosen Streichung der Regelung in Ziff. 2.3 Unterabs. 3 Satz 2 des Erlasses gab es keinen Anknüpfungspunkt mehr für die Frage , ob eine „ Eignung “ nur für ein Fach oder für zwei Fächer bestand , so dass auch keine Herabgruppierung bei „ Eignung “ nur für ein Fach mehr erfolgen konnte .

**False Positives:**

- `Geeignetheit ` — no gold match — likely missing annotation
- `Eignung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `Ziff. 2.3 Unterabs. 3 Satz 2 des Erlasses`(REG)

**Example 19** (doc_id: `57367`) (sent_id: `57367`)


Dass der Begriff des „ Kontors “ für verschiedenste Dienstleistungen - wie etwa Spirituosenherstellung , Versicherungswesen , Werbung oder Immobilienwesen - aktuell Verwendung findet , hat das Deutsche Patent- und Markenamt überzeugend dargetan .

**False Positives:**

- `Kontors ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutsche Patent- und Markenamt`(ORG)

**Example 20** (doc_id: `58231`) (sent_id: `58231`)


Wie bei den Begriffen „ DIE Limmer Schleuse “ oder „ DIE Limmer Kanu Regatta “ würden die angesprochenen Verkehrskreise auch das Anmeldezeichen im Sinne von „ DAS Limmer Kontor “ einem bestimmten Anbieter bzw. einer Institution zuordnen können .

**False Positives:**

- `DIE Limmer Kanu Regatta ` — similar text (different position): `Limmer`
- `DAS Limmer Kontor ` — similar text (different position): `Limmer`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Limmer`(LOC)
- `Limmer`(LOC)
- `Limmer`(LOC)

**Example 21** (doc_id: `58364`) (sent_id: `58364`)


Der angesprochene Verkehr würde die Wortfolge der angegriffenen Marke „ ARROW AND BEAST “ nicht auf „ ARROW “ verkürzen .

**False Positives:**

- `ARROW AND BEAST ` — partial — pred is substring of gold: `„ ARROW AND BEAST “`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `„ ARROW AND BEAST “`(ORG)
- `„ ARROW “`(ORG)

**Example 22** (doc_id: `58381`) (sent_id: `58381`)


Soweit der Gerichtshof in den beiden Entscheidungen darauf abstellt , dass ein Grundpatent im Sinne der Art. 1 ( c ) und 3 ( a ) AMVO einen Wirkstoff nur dann „ als solchen “ schützt , wenn er den Gegenstand der von dem Patent geschützten Erfindung bildet ( EuGH , GRUR Int. 2015 , 446 , Rnd. 38 – Actavis / Boehringer ) , wertet dies der Senat als Bestätigung der in „ Medeva “ und „ Eli Lilly “ niedergelegten Grundsätze .

**False Positives:**

- `Medeva ` — partial — pred is substring of gold: `„ Medeva “`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 1 ( c ) und 3 ( a ) AMVO`(NRM)
- `EuGH , GRUR Int. 2015 , 446 , Rnd. 38 – Actavis / Boehringer`(RS)
- `„ Medeva “`(ORG)
- `„ Eli Lilly “`(ORG)

**Example 23** (doc_id: `58483`) (sent_id: `58483`)


b ) Für die Annahme der Beklagten , unter „ ununterbrochenem Einsatz “ sei die Summe der im Kundenbetrieb geleisteten Arbeitstage zu verstehen , weil mit dem Branchenzuschlag ein „ Erfahrungszuschlag “ gewährt werde , gibt es im Tarifvertrag keine Anhaltspunkte .

**False Positives:**

- `Erfahrungszuschlag ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 24** (doc_id: `58525`) (sent_id: `58525`)


Weder in der Bedeutung „ Schlingel / Schelm / Schlawiner “ noch in der von „ Gauner “ handelt es sich – entgegen der Auffassung der Markeninhaberin um eine spezielle Zielgruppe .

**False Positives:**

- `Gauner ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 25** (doc_id: `58544`) (sent_id: `58544`)


Der Begriff der „ Normalleistung “ hat keinen Eingang in den Wortlaut des Mindestlohngesetzes gefunden ( im Einzelnen : BAG 21. Dezember 2016 - 5 AZR 374/16 - Rn. 21 , BAGE 157 , 356 ; zust .

**False Positives:**

- `Normalleistung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Mindestlohngesetzes`(NRM)
- `BAG 21. Dezember 2016 - 5 AZR 374/16 - Rn. 21 , BAGE 157 , 356`(RS)

**Example 26** (doc_id: `58936`) (sent_id: `58936`)


Kein Teil der angegriffenen Marke sei stärker prägend als der andere , Auch der Bedeutungsgehalt von „ Shot “ im Sinn von „ Schuss , etwas Schnelles , Kurzes “ erleichtere das Auseinanderhalten der Vergleichszeichen und eigne sich dazu , Hör- und Merkfehler zu vermeiden .

**False Positives:**

- `Shot ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 27** (doc_id: `59193`) (sent_id: `59193`)


Zwar mag der Name „ Albert Einstein “ bzw. der Nachname „ Einstein “ als solcher im Einzelfall als Synonym für „ Genie “ verwendet werden .

**False Positives:**

- `Albert Einstein ` — partial — pred is substring of gold: `„ Albert Einstein “`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `„ Albert Einstein “`(PER)
- `„ Einstein “`(PER)

**Example 28** (doc_id: `60041`) (sent_id: `60041`)


HLNK39 Fachinformation zu Yohimbin „ Spiegel “ , Stand : September 2008 , 5 Seiten

**False Positives:**

- `Spiegel ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

</details>

---

<details>
<summary>🔇 Inactive Rules</summary>

</details>

---

<details>
<summary>📋 All Rules</summary>

## `Specific Court Abbreviations and Full Names`

**F1:** 0.413 | **Precision:** 0.342 | **Recall:** 0.520  

**Format:** `regex`  
**Rule ID:** `4d8f42a0`  
**Description:**
Matches common German court abbreviations (BVerfG, BGH, BFH, etc.) with strict word boundaries to avoid partial matches and false positives.

**Content:**
```
\b(?:BVerfG|BGH|BFH|BVerwG|BAG|BSG|EuGH|DPMA|BPatG|ZDS|ZIV|GBA|GEW|X-EWIV|GmSOGB|EGMR|NATO|EU|MDK|BA|VCS|Google|RAPAMUNE|nivo|KAEFER|ARD|ZDF|Deutschlandradio|HBV|K\u00c4V|K\u00c4V\s+Brandenburg|KJM|EWDE|Luftwaffe|Bundestag|AOK-Bayern|Deutschen\s*Post|Deutsche\s*Post|Deutsche\s*Emissionshandelsstelle|Energie-\s*und\s*Klimafonds|Medizinischen\s*Dienstes\s*der\s*Krankenversicherung|Bayerische\s*Verwaltungsgerichtshof|S\s*\u2026|H\s*AG|I\s*AG|P\s*\u2026\s*GmbH\s*&\s*Co\.\s*KG|HSG\s*Z\s*\u2026|V\.|M\s*\u2026|F\s*AG|Fl\.\s*AG|BEAST|Dignitas\s*Deutschland|FC\s*Bayern\s*M\u00fcnchen|Verwaltungsgericht\s*Stuttgart|Schleswig-Holsteinische\s*Verwaltungsgericht|Ausschuss\s*f\u00fcr\s*Arbeit\s*und\s*Sozialordnung\s*des\s*Bundestages|Bund\s*Deutscher\s*Verwaltungsrichter\s*und\s*Verwaltungsrichterinnen|BDVR|Gemeinsamen\s*Senats\s*der\s*obersten\s*Gerichtsh\u00f6fe\s*des\s*Bundes|Gro\u00dfen\s*Senats\s*des\s*BFH|I\.\s*Senates\s*des\s*BFH|3\.\s*Senats\s*des\s*BSG|14\.\s*Senat\s*\(\s*Technischer\s*Beschwerdesenat\s*\)\s*des\s*Bundespatentgerichts|30\.\s*Senat\s*\(\s*Marken-\s*und\s*Design-Beschwerdesenat\s*\)\s*des\s*Bundespatentgerichts|19\.\s*Senat\s*\(\s*Technischer\s*Beschwerdesenat\s*\)\s*des\s*Bundespatentgerichts|28\.\s*Senat\s*\(\s*Marken-Beschwerdesenat\s*\)\s*des\s*Bundespatentgerichts|25\.\s*Senat\s*\(\s*Marken-Beschwerdesenat\s*\)\s*des\s*Bundespatentgerichts|17\.\s*Senat\s*\(\s*Technischer\s*Beschwerdesenat\s*\)\s*des\s*Bundespatentgerichts|X\.\s*Zivilsenats\s*des\s*Bundesgerichtshofs|Schleswig-Holsteinische\s+Oberlandesgericht|Th\u00fcringer\s+Finanzgericht|Nieders\u00e4chsische\s+Landesschulbeh\u00f6rde|Diakonissenhaus\s+[A-Z]\.|Deutschen\s*Patent-\s*und\s*Markenamts\s*,\s*Markenstelle\s*f\u00fcr\s*Klasse\s*\d+|Pr\u00fcfungsstelle\s*f\u00fcr\s*Klasse\s*[A-Z0-9]+\s*des\s*Deutschen\s*Patent-\s*und\s*Markenamts|Bundesrat|Europ\u00e4ische\s*Union|Europ\u00e4ischer\s*Gerichtshof|Gerichtshof\s*der\s*Europ\u00e4ischen\s*Union|Bundesverfassungsgericht|Bundesverfassungsgerichts|Bundesgerichtshof|Europ\u00e4ischen\s*Gerichtshofs\s*f\u00fcr\s*Menschenrechte|Europ\u00e4ischen\s*Gerichtshof|Europ\u00e4ischen\s*Gerichtshofs|Europ\u00e4ischen\s*Gerichtshofs\s*f\u00fcr\s*Menschenrechte|Bundesverwaltungsgericht|Bundesarbeitsgericht|Bundessozialgericht|Bundesfinanzhof|Bundespatentgericht|Oberlandesgericht\s+M\u00fcnchen|Oberlandesgericht\s+Hamm|Landgericht\s+Hamburg|Landgericht\s+Bremen|Landgericht\s+Oldenburg|Landgericht\s+Karlsruhe|Landgericht\s+Darmstadt|Landgericht\s+D\u00fcsseldorf|Landgericht\s+Potsdam|Landgericht\s+F\.\s*\(\s*P\.\s*\)|Pf\u00e4lzische\s+Oberlandesgericht\s+Zweibr\u00fccken|FG\s+M\u00fcnster|FG\s+M\u00fcnchen|Staatsanwaltschaft\s+Duisburg|Landratsamt\s+D|Kundenniederlassung\s+Spezial\s+in\s+S|1\.\s*Senat\s*des\s*Nieders\u00e4chsischen\s+Anwaltsgerichtshofs|Ersten\s*Senats\s*des\s*Bundesarbeitsgerichts|4\.\s*Senat\s*des\s*BSG|29\.\s*Zivilkammer\s*des\s*Landgerichts\s*K\u00f6ln|Zivilkammer\s*des\s*Landgerichts\s*Berlin|ausw\u00e4rtigen\s*gro\u00dfen\s*Strafkammer\s*des\s*Landgerichts\s*Kleve\s*in\s*Moers|Justizvollzugsanstalt\s*Offenburg|Landesarbeitsgericht\s*Berlin-Brandenburg|S\u00e4chsische\s*Bildungsagentur|Gemeinsamen\s*Bundesausschusses|Landesaussch\u00fcsse|Obersten\s*Verwaltungsgerichts\s*der\s*Republik\s*Bulgarien|Bundeskasse\s*Halle\s*/\s*Saale\s*-\s*Dienstsitz\s*Weiden\s*/\s*Oberpfalz|Diakonischen\s*Werkes\s*der\s*Evangelischen\s*Kirche\s*in\s*Deutschland\s*e\.\s*V\.|Evangelischen\s*Entwicklungsdienstes\s*e\.\s*V\.|Haus\u00e4rztlliche\s*Vertragsgemeinschaft\s*Aktiengesellschaft|H\u00c4VG-Rechenzentrum\s*AG|H\u00e4VG-Rechenzentrum\s*GmbH|B\s*\u2026\s*Patentanwaltsgesellschaft\s*mbH|C\s*\u2026\s*GmbH|D\s*P\s*T\s*S\s*GmbH|K\s*\u2026\s*GmbH|B\s*\u2026\s*AG|S\s*\u2026|C-\s*B\.\s*V\.|A-Fonds|BgA\s*X|InEK|ver\.di|DB|BND|RVA|TdL|InWIS|DRV|KOSMICA|Monster\s*Abwehr\s*Spray|Haus\u00e4rzteverband|Bundesvereinigung\s*der\s*Arbeitgeberverb\u00e4nde|Deutsche\s*Rentenversicherung\s*Bund|Ausw\u00e4rtige\s*Amt|Bundesamt\s*f\u00fcr\s*Migration\s*und\s*Fl\u00fcchtlinge|BMF|Internationalen\s*Gerichtshofs|Europ\u00e4ischen\s*Kommission|Justizkommission\s*zu\s*Tunis|Bundeswehr|Fliegerhorst\s*B\u00fcchel|St\u00e4dtischen\s*Klinikum\s*K.|Kernkraftwerks\s*Kr\u00fcmmel|Landesarbeitsgericht\s*Berlin-Brandenburg|S2\s*\u2026\s*GmbH|S\s*\u2026|C-\s*B\.\s*V\.|Software\s*f\u00fcr\s*Ihren\s*Erfolg|A-Fonds|BgA\s*X|Bundesregierung|Bundesministerium\s+der\s+Finanzen|Bundesamts\s+f\u00fcr\s+Justiz|Justizministerium\s+des\s+Landes\s+Nordrhein-Westfalen|Neurologischen\s+Klinik\s+B|Amtsgericht\s+O\.|Handwerksverband\s+Metallbau\s+und\s+Feinwerktechnik\s+Baden-W\u00fcrttemberg|Industriegewerkschaft\s+Metall|VEB\s+[A-Z][a-zA-Z\s]+|nieders\u00e4chsische\s+Landesschulbeh\u00f6rde|EON-Konzerns|EON-Konzern|dbb\s+beamtenbund\s+und\s+tarifunion|ADAC|Kernkraftwerks\s+Biblis|Kernkraftwerks\s+M\u00fclheim-K\u00e4rlich|Deutschen\s*Botschaft|Finanzamt\s+[A-Za-z\u00e4\u00f6\u00fc\u00df\s]+|Bundesministeriums\s+der\s+Verteidigung\s+-\s+R\s+II\s+2\s+-|Bundesamts\s+f\u00fcr\s+das\s+Personalmanagement|A\s+Lebensversicherung\s+AG|Europ\u00e4ische\s+Gerichtshof\s+f\u00fcr\s+Menschenrechte|Th\u00fcringer\s+Landessozialgerichts|Deutsche\s+Patent-\s+und\s+Markenamt|Deutschen\s+Patent-\s+und\s+Markenamt|Deutschen\s+Patent-\s+und\s+Markenamtes|K\u00c4V\s+Brandenburg|Neue\s+Richtervereinigung|Generalstaatsanwaltschaft\s+des\s+Landes\s+Schleswig-Holstein|K-Klinik|H\u00c4VG|Schott|PreussenElektra\s+GmbH|E\.\s+ON\s+Kernkraft\s+GmbH|G-Gruppe|Vereinigung\s+der\s+kommunalen\s*Arbeitgeberverb\u00e4nde|Staatskasse|Kernkraftwerks\s+Gundremmingen|Bundesministeriums\s+der\s+Verteidigung|Bundesministerium\s+der\s+Verteidigung|Bundesverfassungsgerichts|Bundesverwaltungsgerichts|Bundesgerichtshofs|Bundesfinanzhofs|Bundesarbeitsgerichts|Bundessozialgerichts|Landgerichts\s+Darmstadt|Landgerichts\s+D\u00fcsseldorf|Landgerichts\s+Hamburg|Landgerichts\s+Bremen|Landgerichts\s+Oldenburg|Landgerichts\s+Karlsruhe|Landgerichts\s+Potsdam|Landgerichts\s+F\.\s*\(\s*P\.\s*\)|Pf\u00e4lzische\s+Oberlandesgericht\s+Zweibr\u00fccken|Oberlandesgerichts\s+M\u00fcnchen|Oberlandesgerichts\s+Hamm|Amtsgerichts\s+O\.|Finanzamts\s+[A-Za-z\u00e4\u00f6\u00fc\u00df\s]+|Bundesamts\s+f\u00fcr\s+das\s+Personalmanagement\s+der\s+Bundeswehr|Markenstelle\s*f\u00fcr\s*Klasse\s*\d+\s*des\s*Deutschen\s*Patent-\s*und\s*Markenamts|Patentabteilung\s*\d+\.\d+\s*des\s*Deutschen\s*Patent-\s*und\s*Markenamts|Designabteilung\s*\d+\.\d+\s*des\s*Deutschen\s*Patent-\s*und\s*Markenamts|Markenstelle\s*f\u00fcr\s*Klasse\s*\d+\s*des\s*DPMA|Patentabteilung\s*\d+\.\d+\s*des\s*DPMA|Designabteilung\s*\d+\.\d+\s*des\s*DPMA|Deutschen\s*Patent-\s*und\s*Markenamts|Deutschen\s*Patent-\s*und\s*Markenamt|Deutsche\s*Patent-\s*und\s*Markenamt|Deutsche\s*Patent-\s*und\s*Markenamts|Statistischen\s*Bundesamt|Deutschen\s*Bundestages|Spitzenverband\s*Bund\s*der\s*KKn|Verband\s*der\s*Privaten\s*Krankenversicherung|Deutsche\s*Krankenhausgesellschaft|VKDA|Gro\u00dfen\s*Senat\s*des\s*BFH|VIII\.\s*Senat\s*des\s*BFH|VIII\.\s*Senats\s*des\s*BFH|LSG\s+Berlin-Brandenburg|RWE-Konzerns|B\.\s*GmbH|w\s*GmbH|w\s*Holding\s*GmbH|P\s*GmbH|G\s*GmbH|M-GmbH\s*&\s*atypisch\s*Still|Gewerkschaft\s*ver\.di|Kreiskrankenh\u00e4user\s*M\s+und\s*R|NIVONA|fluege\.de|CHECK24\.de|Jaguar|Land\s*Rover|\u00d6z\s*Gaziantep\s*Dilim\s*Baklavalari|Wohnungsbau-\s*und\s*Kommissionsgesellschaft\s*Reichenstra\u00dfen|A-Fonds|B-Fonds|A-AG|B-AG|C-AG|D-AG|E-AG|F-AG|G-AG|H-AG|I-AG|J-AG|K-AG|L-AG|M-AG|N-AG|O-AG|P-AG|Q-AG|R-AG|S-AG|T-AG|U-AG|V-AG|W-AG|X-AG|Y-AG|Z-AG|A\s*Lebensversicherung\s*AG|B\s*Lebensversicherung\s*AG|C\s*Lebensversicherung\s*AG|D\s*Lebensversicherung\s*AG|E\s*Lebensversicherung\s*AG|F\s*Lebensversicherung\s*AG|G\s*Lebensversicherung\s*AG|H\s*Lebensversicherung\s*AG|I\s*Lebensversicherung\s*AG|J\s*Lebensversicherung\s*AG|K\s*Lebensversicherung\s*AG|L\s*Lebensversicherung\s*AG|M\s*Lebensversicherung\s*AG|N\s*Lebensversicherung\s*AG|O\s*Lebensversicherung\s*AG|P\s*Lebensversicherung\s*AG|Q\s*Lebensversicherung\s*AG|R\s*Lebensversicherung\s*AG|S\s*Lebensversicherung\s*AG|T\s*Lebensversicherung\s*AG|U\s*Lebensversicherung\s*AG|V\s*Lebensversicherung\s*AG|W\s*Lebensversicherung\s*AG|X\s*Lebensversicherung\s*AG|Y\s*Lebensversicherung\s*AG|Z\s*Lebensversicherung\s*AG)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.342 | 0.520 | 0.413 | 1228 | 420 | 808 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 420 | 808 | 387 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53387`) (sent_id: `53387`)


7. Die hiergegen eingelegte Nichtzulassungsbeschwerde sowie eine Anhörungsrüge der Beschwerdeführerin wies der Bundesgerichtshof zurück .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Example 1** (doc_id: `53400`) (sent_id: `53400`)


Die Vorgabe des Bundesfinanzhofs ( BFH ) , dass die Einsteller ( Leistungsempfänger ) keine Land- und Forstwirte sein müssten , die von ihnen empfangenen Leistungen ( Zuchtleistungen als einheitliche Leistung ) aber gleichwohl zu landwirtschaftlichen Zwecken genutzt werden müssten , erscheine nicht eindeutig und auf den ersten Blick widersprüchlich .

| Predicted | Gold |
|---|---|
| `Bundesfinanzhofs` | `Bundesfinanzhofs` |
| `BFH` | `BFH` |

**Example 2** (doc_id: `53402`) (sent_id: `53402`)


Bei offenem Ausgang des Verfassungsbeschwerdeverfahrens muss das Bundesverfassungsgericht die Folgen abwägen , die eintreten würden , wenn die einstweilige Anordnung nicht erginge , die Verfassungsbeschwerde aber Erfolg hätte , gegenüber den Nachteilen , die entstünden , wenn die begehrte einstweilige Anordnung erlassen würde , der Verfassungsbeschwerde aber der Erfolg zu versagen wäre ( vgl. BVerfGE 76 , 253 < 255 > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `BVerfGE 76 , 253 < 255 >` (RS)

**Example 3** (doc_id: `53403`) (sent_id: `53403`)


Vor diesem Hintergrund erweist sich schon die Auseinandersetzung des Beschwerdeführers mit den vom Bundesverfassungsgericht - wenn auch zu Art. 19 Abs. 4 GG , den der Beschwerdeführer nicht rügt - entwickelten verfassungsrechtlichen Maßstäben als unzureichend ; umso weniger ist unter diesen Umständen eine mögliche Willkür der angegriffenen Entscheidung plausibel dargelegt .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Art. 19 Abs. 4 GG` (NRM)

**Example 4** (doc_id: `53404`) (sent_id: `53404`)


Hiergegen richtet sich die mit Schriftsatz vom 11. Februar 2014 beim Deutschen Patent- und Markenamt eingelegte Beschwerde der Anmelderin , die ihr Patentbegehren mit den mit Schreiben vom 25. Februar 2011 eingereichten Unterlagen weiterverfolgt .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Example 5** (doc_id: `53408`) (sent_id: `53408`)


Ein solcher enger sachlicher Zusammenhang mit der gesetzlichen Tätigkeit als Träger der Grundsicherung für Arbeitsuchende ist bei der BA immer dann anzunehmen , wenn sie - objektiv betrachtet - in dem zugrunde liegenden Verfahren eine Aufgabe im Rahmen des Vollzugs des SGB II wahrnimmt .

| Predicted | Gold |
|---|---|
| `BA` | `BA` |

**Missed by this rule (FN):**

- `SGB II` (NRM)

**Example 6** (doc_id: `53415`) (sent_id: `53415`)


Die Form der Benutzung ist insoweit jeweils glaubhaft gemacht durch die als Anlagen W14 bis W17 vorgelegten Produktkataloge aus dem vorliegend relevanten Benutzungszeitraum mit Abbildungen einer Vielzahl von ( elektrischen ) Kaffeemaschinen , Kaffeemühlen , Milchbehälter ( „ Milchcooler “ ) sowie Verpackungen / Behälter von Reinigungstabs , CreamCleaner und Entkalker , auf denen die Widerspruchsmarke NIVONA in einer ohne weitere besondere graphische Ausgestaltungselemente enthaltenden und damit in einer den kennzeichnenden Charakter der eingetragenen Marke nicht verändernden Form i. S. von § 26 Abs. 3 MarkenG deutlich sichtbar abgebildet ist .

| Predicted | Gold |
|---|---|
| `NIVONA` | `NIVONA` |

**Missed by this rule (FN):**

- `§ 26 Abs. 3 MarkenG` (NRM)

**Example 7** (doc_id: `53419`) (sent_id: `53419`)


Hiergegen richtet sich die Beschwerde der Anmelderin vom 29. November 2016 , mit der sie sinngemäß beantragt , den Beschluss der Markenstelle für Klasse 35 des Deutschen Patent- und Markenamts vom 11. November 2016 aufzuheben .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 35 des Deutschen Patent- und Markenamts` | `Markenstelle für Klasse 35 des Deutschen Patent- und Markenamts` |

**Example 8** (doc_id: `53424`) (sent_id: `53424`)


Die X-EWIV wäre dem Kläger in vollem Umfang auskunfts- und rechenschaftspflichtig gewesen .

| Predicted | Gold |
|---|---|
| `X-EWIV` | `X-EWIV` |

**Example 9** (doc_id: `53430`) (sent_id: `53430`)


II. Die zulässige Beschwerde der Löschungsantragsgegnerin hat sich durch die zwischenzeitlich mit Beschluss des DPMA vom 27. September 2017 angeordnete Verfallslöschung der angegriffenen Marke in der Hauptsache erledigt .

| Predicted | Gold |
|---|---|
| `DPMA` | `DPMA` |

**Example 10** (doc_id: `53466`) (sent_id: `53466`)


Die PreussenElektra GmbH war Beschwerdeführerin im Verfassungsbeschwerdeverfahren 1 BvR 2821/11 ( BVerfGE 143 , 246 ) , das sich ebenfalls gegen die hier angegriffenen Regelungen des Atomgesetzes richtete .

| Predicted | Gold |
|---|---|
| `PreussenElektra GmbH` | `PreussenElektra GmbH` |

**Missed by this rule (FN):**

- `Verfassungsbeschwerdeverfahren 1 BvR 2821/11 ( BVerfGE 143 , 246 )` (RS)
- `Atomgesetzes` (NRM)

**Example 11** (doc_id: `53524`) (sent_id: `53524`)


Der BFH führte aus , die handelsrechtliche Zurechnung von Vermögensgegenständen entspreche im Wesentlichen der Regelung des § 39 Abs. 2 Nr. 1 Satz 1 AO ( z.B. BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1. ; vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a ) .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Missed by this rule (FN):**

- `§ 39 Abs. 2 Nr. 1 Satz 1 AO` (NRM)
- `BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1.` (RS)
- `vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a` (RS)

**Example 12** (doc_id: `53539`) (sent_id: `53539`)


cc ) Stellt ein Bewerber nicht innerhalb eines Monats nach Zugang der Abbruchmitteilung einen Antrag auf Erlass einer einstweiligen Verfügung , darf der Dienstherr nach der Rechtsprechung des Bundesverwaltungsgerichts darauf vertrauen , dass der Bewerber den Abbruch des Auswahlverfahrens nicht angreift , sondern sein Begehren im Rahmen einer neuen Ausschreibung weiterverfolgt ( vgl. BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14 ) .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Missed by this rule (FN):**

- `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14` (RS)

**Example 13** (doc_id: `53549`) (sent_id: `53549`)


aa ) Der GBA hat eine IVIG-Therapie zur Behandlung des sekundären Immunglobulinmangels mit MGUS und rezidivierenden Pneumonien nicht empfohlen .

| Predicted | Gold |
|---|---|
| `GBA` | `GBA` |

**Example 14** (doc_id: `53574`) (sent_id: `53574`)


Die Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts hat diese unter der Nummer 30 2014 034 614.1 geführte Anmeldung mit Beschluss vom 25. November 2014 wegen fehlender Unterscheidungskraft zurückgewiesen .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts` | `Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts` |

**Example 15** (doc_id: `53586`) (sent_id: `53586`)


d ) Mit dem mit der Verfassungsbeschwerde ebenfalls angegriffenen Beschluss vom 8. März 2011 wies der Bundesfinanzhof die von der Beschwerdeführerin erhobene Anhörungsrüge und Gegenvorstellung zurück .

| Predicted | Gold |
|---|---|
| `Bundesfinanzhof` | `Bundesfinanzhof` |

**Example 16** (doc_id: `53594`) (sent_id: `53594`)


Die von der Krankenkasse nicht getragenen Kosten der stationären Behandlung zahlte der Kläger zu 1. Die Therapie der E in der K-Klinik war im Vorfeld durch den Medizinischen Dienst der Krankenversicherungen ( MDK ) auf ihre medizinische Notwendigkeit geprüft und befürwortet worden .

| Predicted | Gold |
|---|---|
| `K-Klinik` | `K-Klinik` |
| `MDK` | `MDK` |

**Missed by this rule (FN):**

- `E` (PER)
- `Medizinischen Dienst der Krankenversicherungen` (ORG)

**Example 17** (doc_id: `53619`) (sent_id: `53619`)


( 2 ) Ob die Umwandlung der Todesstrafe in eine lebenslange Freiheitsstrafe bereits zwingend aus dem seit 1991 praktizierten Moratorium folgt , wie es das Bundesverwaltungsgericht angenommen hat , kann dahinstehen .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgericht` | `Bundesverwaltungsgericht` |

**Example 18** (doc_id: `53630`) (sent_id: `53630`)


Die Verträge seien derart angelegt gewesen , dass die KG am Ende der Laufzeit ihr " Andienungsrecht " ausübe und die P GmbH die Leasingobjekte zurückerwerben müsse .

| Predicted | Gold |
|---|---|
| `P GmbH` | `P GmbH` |

**Example 19** (doc_id: `53656`) (sent_id: `53656`)


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

**Example 20** (doc_id: `53664`) (sent_id: `53664`)


Soweit das Bundesministerium der Verteidigung darauf verweist , dass Personalgespräche dazu dienten , Soldaten über Verwendungsplanungen der Personalführung zu informieren und in diesem Zusammenhang deren persönliche Motivationslage zu ergründen , um das dienstliche Bedürfnis und die privaten Lebensumstände des Soldaten möglichst gut miteinander zu vereinbaren , trifft dies zweifellos auf Personalgespräche im Vorfeld von Personalmaßnahmen zu .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Example 21** (doc_id: `53690`) (sent_id: `53690`)


An dieser Regelung hielt die Bundesregierung fest , obwohl der Bundesrat die Einbeziehung der einmaligen Einnahmen vorschlug ( BT-Drucks 16/2454 S 11 ) .

| Predicted | Gold |
|---|---|
| `Bundesregierung` | `Bundesregierung` |
| `Bundesrat` | `Bundesrat` |

**Missed by this rule (FN):**

- `BT-Drucks 16/2454 S 11` (LIT)

**Example 22** (doc_id: `53700`) (sent_id: `53700`)


3. Das Bundesverfassungsgericht überprüft die Vereinbarkeit eines nationalen Gesetzes mit dem Grundgesetz auch , wenn zugleich Zweifel an der Vereinbarkeit des Gesetzes mit Sekundärrecht der Europäischen Union bestehen .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Grundgesetz` (NRM)
- `Europäischen Union` (ORG)

**Example 23** (doc_id: `53725`) (sent_id: `53725`)


4. Auf die sofortige Beschwerde der Staatsanwaltschaft Düsseldorf hob das Oberlandesgericht Hamm mit angegriffenem Beschluss vom 28. Juli 2015 den Beschluss des Landgerichts Paderborn auf und ordnete die weitere Vollstreckung der Unterbringung des Beschwerdeführers in der Sicherungsverwahrung an .

| Predicted | Gold |
|---|---|
| `Oberlandesgericht Hamm` | `Oberlandesgericht Hamm` |

**Missed by this rule (FN):**

- `Staatsanwaltschaft Düsseldorf` (ORG)
- `Landgerichts Paderborn` (ORG)

**Example 24** (doc_id: `53746`) (sent_id: `53746`)


Das hat zur Folge , dass beim BSG über Kostenerinnerungen nach § 189 Abs 2 S 2 SGG der Senat in der Besetzung mit drei Berufsrichtern zu befinden hat .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Missed by this rule (FN):**

- `§ 189 Abs 2 S 2 SGG` (NRM)

**Example 25** (doc_id: `53773`) (sent_id: `53773`)


Vielmehr hat das LSG nach erfolgter Zurückverweisung durch das BSG nach umfangreicher weiterer Sachaufklärung den Anspruch des Klägers erneut geprüft und eine Entscheidung gefällt .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 26** (doc_id: `53777`) (sent_id: `53777`)


Am 14. November 2000 wurden den Aktionären der A-AG vereinbarungsgemäß ... Stück vinkulierte Namensaktien an der B-AG übertragen .

| Predicted | Gold |
|---|---|
| `A-AG` | `A-AG` |
| `B-AG` | `B-AG` |

**Example 27** (doc_id: `53791`) (sent_id: `53791`)


Zwischen den Vergleichszeichen besteht hinsichtlich der angegriffenen Waren Verwechslungsgefahr gemäß § 9 Abs. 1 Nr. 2 MarkenG , so dass der Widerspruch aus der Marke 30 2011 047 766 vom Deutschen Patent- und Markenamt zu Unrecht zurückgewiesen wurde .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `§ 9 Abs. 1 Nr. 2 MarkenG` (NRM)

**Example 28** (doc_id: `53844`) (sent_id: `53844`)


Der im Juli 1950 geborene Kläger war seit dem 1. März 1971 bei einer Rechtsvorgängerin der Beklagten , der H AG ( im Folgenden H AG alt ) als Arbeitnehmer tätig .

| Predicted | Gold |
|---|---|
| `H AG` | `H AG` |

**Missed by this rule (FN):**

- `H AG alt` (ORG)

**Example 29** (doc_id: `53855`) (sent_id: `53855`)


aa ) In organisatorischer Hinsicht ist beim Bundesverfassungsgericht , anders als bei den Fachgerichten , eine Kapazitätsausweitung zur Verkürzung der Verfahrensdauer als Reaktion auf gesteigerte Eingangszahlen grundsätzlich nicht möglich , da die Struktur des Gerichts durch seine Funktion bedingt und durch die Verfassung und das Bundesverfassungsgerichtsgesetz vorgegeben ist ( BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21 ; vgl. BTDrucks 17/3802 , S. 26 ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `Bundesverfassungsgerichtsgesetz` (NRM)
- `BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21` (RS)
- `BTDrucks 17/3802 , S. 26` (LIT)

**Example 30** (doc_id: `53862`) (sent_id: `53862`)


Patentansprüche 1 bis 13 vom 24. November 2017 , beim BPatG als 6. Hilfsantrag per Fax eingegangen am 27. November 2017

| Predicted | Gold |
|---|---|
| `BPatG` | `BPatG` |

**Example 31** (doc_id: `53867`) (sent_id: `53867`)


Zwar kann das LSG von einer Entscheidung ua des BSG auch dann abweichen , wenn es einen der höchstrichterlichen Rechtsprechung widersprechenden Rechtssatz nur sinngemäß und in scheinbar fallbezogene Ausführungen gekleidet entwickelt .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 32** (doc_id: `53903`) (sent_id: `53903`)


Sie genügen den Anforderungen des Europäischen Gerichtshofs für Menschenrechte an die Überprüfbarkeit einer lebenslangen Freiheitsstrafe .

| Predicted | Gold |
|---|---|
| `Europäischen Gerichtshofs für Menschenrechte` | `Europäischen Gerichtshofs für Menschenrechte` |

**Example 33** (doc_id: `53939`) (sent_id: `53939`)


Dort hat der BFH lediglich ausgeführt , die Vergütung für die Hingabe eines partiarischen Darlehens könne auch umsatzabhängig ausgestaltet werden .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Example 34** (doc_id: `53957`) (sent_id: `53957`)


Über keine der anhängigen Rechtsbeschwerden hat der Bundesgerichtshof bislang entschieden .

| Predicted | Gold |
|---|---|
| `Bundesgerichtshof` | `Bundesgerichtshof` |

**Example 35** (doc_id: `54007`) (sent_id: `54007`)


Das Unterrichtungsschreiben vom 14. November 2005 enthält weder Angaben zum Sitz der D P T S GmbH noch zu deren Anschrift , zum zuständigen Registergericht und zur Registernummer .

| Predicted | Gold |
|---|---|
| `D P T S GmbH` | `D P T S GmbH` |

**Example 36** (doc_id: `54010`) (sent_id: `54010`)


Auf die Beschwerde der Anmelderin werden die Beschlüsse des Deutschen Patent- und Markenamts , Markenstelle für Klasse 41 , vom 3. Juli 2014 und vom 3. Dezember 2015 aufgehoben , soweit die Anmeldung in Bezug auf die nachfolgend genannten Dienstleistungen zurückgewiesen worden ist :

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamts , Markenstelle für Klasse 41` | `Deutschen Patent- und Markenamts , Markenstelle für Klasse 41` |

**Example 37** (doc_id: `54059`) (sent_id: `54059`)


Wie der Kläger selbst vorträgt , wollte sich das LSG vielmehr nach seinem eigenen Verständnis im Rahmen der bereits vorliegenden Rechtsprechung des BSG halten , ist diesen aber in dem von ihm entschiedenen Fall lediglich deshalb nicht gefolgt , weil es - vom Kläger als unzutreffend gerügte - wesentliche Sachverhaltsunterschiede zu den bereits entschiedenen Fällen angenommen hat .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 38** (doc_id: `54064`) (sent_id: `54064`)


Der BFH prüft insofern nur , ob sie gegen Denkgesetze und Erfahrungssätze oder die anerkannten Auslegungsregeln verstößt .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Example 39** (doc_id: `54126`) (sent_id: `54126`)


1. Aus besonderem Grund , namentlich im Interesse einer verlässlichen Finanz- und Haushaltsplanung und eines gleichmäßigen Verwaltungsvollzugs für Zeiträume einer weitgehend schon abgeschlossenen Veranlagung , hat das Bundesverfassungsgericht wiederholt die weitere Anwendbarkeit verfassungswidriger Normen binnen der dem Gesetzgeber bis zu einer Neuregelung gesetzten Frist oder spätestens bis zur Neuregelung für gerechtfertigt erklärt ( vgl. etwa BVerfGE 87 , 153 < 178 > ; 93 , 121 < 148 f. > ; 123 , 1 < 38 > ; 125 , 175 < 258 > ; 138 , 136 < 251 Rn. 287 > ; 139 , 285 < 319 Rn. 89 > ) .

| Predicted | Gold |
|---|---|
| `Bundesverfassungsgericht` | `Bundesverfassungsgericht` |

**Missed by this rule (FN):**

- `BVerfGE 87 , 153 < 178 > ; 93 , 121 < 148 f. > ; 123 , 1 < 38 > ; 125 , 175 < 258 > ; 138 , 136 < 251 Rn. 287 > ; 139 , 285 < 319 Rn. 89 >` (RS)

**Example 40** (doc_id: `54130`) (sent_id: `54130`)


b ) In Leasingfällen ( auch beim " Lease " im Rahmen eines " Sale-and-lease-back " ) geht der BFH - wie bereits in dem Urteil in BFHE 255 , 386 , BStBl II 2018 , 81 dargestellt - bei Anwendung des § 39 Abs. 2 Nr. 1 Satz 1 AO insbesondere von folgenden Grundsätzen aus :

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Missed by this rule (FN):**

- `Urteil in BFHE 255 , 386 , BStBl II 2018 , 81` (RS)
- `§ 39 Abs. 2 Nr. 1 Satz 1 AO` (NRM)

**Example 41** (doc_id: `54150`) (sent_id: `54150`)


Die Prüfung der Markenanmeldung muss daher nach der maßgeblichen Rechtsprechung des EuGH grundsätzlich streng und vollständig sein , um ungerechtfertigte Eintragungen zu vermeiden ( vgl. EuGH , GRUR 2003 , 604 Rn. 57 , 60 – Libertel ; BGH , GRUR 2014 , 565 Rn. 17 – smartbook ; Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 158 , 159 ) .

| Predicted | Gold |
|---|---|
| `EuGH` | `EuGH` |

**Missed by this rule (FN):**

- `EuGH , GRUR 2003 , 604 Rn. 57 , 60 – Libertel` (RS)
- `BGH , GRUR 2014 , 565 Rn. 17 – smartbook` (RS)
- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 158 , 159` (LIT)

**Example 42** (doc_id: `54169`) (sent_id: `54169`)


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

**Example 1** (doc_id: `53422`) (sent_id: `53422`)


Bei der vom FG zu beurteilenden gesetzlichen Prozessführungsbefugnis ( Beteiligtenstellung ) handelt es sich um eine Sachurteilsvoraussetzung , deren fehlerhafte Beurteilung einen Verfahrensmangel darstellt ( BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943 ; vom 23. November 2010 V B 133/09 , BFH / NV 2011 , 612 , Rz 5 ; jeweils m. w. N. ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`
- `BFH` — partial — pred is substring of gold: `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`
- `BFH` — similar text (different position): `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`

> overlaps gold: 3  |  likely missing annotation: 0

**Gold Entities:**

- `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`(RS)
- `vom 23. November 2010 V B 133/09 , BFH / NV 2011 , 612 , Rz 5`(RS)

**Example 2** (doc_id: `53446`) (sent_id: `53446`)


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

**Example 3** (doc_id: `53468`) (sent_id: `53468`)


Wie das vorliegende Verfahren zeigt , bietet die Anhörungsrüge die Möglichkeit , durch Zuordnungsschwierigkeiten entstehende Probleme zu bewältigen ( vgl. BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15 ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15`(RS)

**Example 4** (doc_id: `53474`) (sent_id: `53474`)


Beim Leasing ist - wie dargestellt - darauf abzustellen , ob der Herausgabeanspruch des Leasinggebers ( zivilrechtlichen Eigentümers ) noch eine wirtschaftliche Bedeutung hat ( grundlegend BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1. ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1.`(RS)

**Example 5** (doc_id: `53481`) (sent_id: `53481`)


Diese können insbesondere dann vorliegen , wenn die Gesamtstrafe sich nicht innerhalb des gesetzlichen Strafrahmens bewegt , die gebotene Begründung für die Gesamtstrafe fehlt , oder wenn die Besorgnis besteht , der Tatrichter habe sich von der Summe der Einzelstrafen leiten lassen ( BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris`(RS)

**Example 6** (doc_id: `53485`) (sent_id: `53485`)


Soweit die Elemente eines Bildzeichens nur die typischen Merkmale der in Rede stehenden Waren und Dienstleistungen darstellen oder sich in einfachen dekorativen Gestaltungsmitteln erschöpfen , an die sich der Verkehr etwa durch häufige Verwendung gewöhnt hat , wird diesem Zeichen im Allgemeinen wegen seines bloß beschreibenden Inhalts ebenfalls die konkrete Eignung fehlen , die mit ihm gekennzeichneten Waren oder Dienstleistungen von denjenigen anderer Herkunft zu unterscheiden ( vgl. BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband ; GRUR 2005 , 257 , 258 – Bürogebäude ; GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel ; GRUR 2001 , 239f. – Zahnpastastrang ; GRUR 2001 , 734 , 735 – Westie-Kopf ; BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband`
- `BPatG` — partial — pred is substring of gold: `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband`(RS)
- `GRUR 2005 , 257 , 258 – Bürogebäude`(RS)
- `GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel`(RS)
- `GRUR 2001 , 239f. – Zahnpastastrang`(RS)
- `GRUR 2001 , 734 , 735 – Westie-Kopf`(RS)
- `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`(RS)

**Example 7** (doc_id: `53490`) (sent_id: `53490`)


Es müssen letztlich besondere Verhaltensweisen sowohl des Berechtigten als auch des Verpflichteten vorliegen , die es rechtfertigen , die späte Geltendmachung des Rechts als mit Treu und Glauben unvereinbar und für den Verpflichteten als unzumutbar anzusehen ( vgl. BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27 mwN ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27`(RS)

**Example 8** (doc_id: `53508`) (sent_id: `53508`)


Beruhen vielmehr die Differenzen zwischen den Auffassungen von Sachverständigen darauf , dass diese von verschiedenen tatsächlichen Annahmen ausgehen , dann muss der Tatrichter , ggf nach weiterer Aufklärung , die für seine Überzeugungsbildung maßgebenden Tatsachen feststellen oder begründen , weshalb und zu wessen Lasten sie beweislos geblieben sind ( vgl BGH Urteil vom 23. 9. 1986 - VI ZR 261/85 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH Urteil vom 23. 9. 1986 - VI ZR 261/85`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH Urteil vom 23. 9. 1986 - VI ZR 261/85`(RS)

**Example 9** (doc_id: `53524`) (sent_id: `53524`)


Der BFH führte aus , die handelsrechtliche Zurechnung von Vermögensgegenständen entspreche im Wesentlichen der Regelung des § 39 Abs. 2 Nr. 1 Satz 1 AO ( z.B. BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1. ; vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a ) .

**False Positives:**

- `BFH` — similar text (different position): `BFH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH`(ORG)
- `§ 39 Abs. 2 Nr. 1 Satz 1 AO`(NRM)
- `BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1.`(RS)
- `vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a`(RS)

**Example 10** (doc_id: `53539`) (sent_id: `53539`)


cc ) Stellt ein Bewerber nicht innerhalb eines Monats nach Zugang der Abbruchmitteilung einen Antrag auf Erlass einer einstweiligen Verfügung , darf der Dienstherr nach der Rechtsprechung des Bundesverwaltungsgerichts darauf vertrauen , dass der Bewerber den Abbruch des Auswahlverfahrens nicht angreift , sondern sein Begehren im Rahmen einer neuen Ausschreibung weiterverfolgt ( vgl. BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14 ) .

**False Positives:**

- `BVerwG` — partial — pred is substring of gold: `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesverwaltungsgerichts`(ORG)
- `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14`(RS)

**Example 11** (doc_id: `53551`) (sent_id: `53551`)


Die Beklagte wird das der Klägerin zustehende Honorar unter Zugrundelegung eines festen RLV für die gesamte BAG neu zu ermitteln haben .

**False Positives:**

- `BAG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 12** (doc_id: `53582`) (sent_id: `53582`)


3. Wird einem Beteiligten das rechtliche Gehör dadurch versagt , dass es ihm nicht ermöglicht wird , an der mündlichen Verhandlung teilzunehmen , so ist davon auszugehen , dass dies für eine aufgrund dieser Verhandlung ergangenen Entscheidung ursächlich geworden ist ( vgl BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7 ) .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7`(RS)

**Example 13** (doc_id: `53583`) (sent_id: `53583`)


Denn die Hauptfunktion einer Marke liegt darin , die Ursprungsidentität der gekennzeichneten Waren und Dienstleistungen zu gewährleisten ( vgl. u. a. EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel ; BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006 ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel`
- `BGH` — partial — pred is substring of gold: `BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel`(RS)
- `BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006`(RS)

**Example 14** (doc_id: `53599`) (sent_id: `53599`)


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

**Example 15** (doc_id: `53600`) (sent_id: `53600`)


Auch danach setzt die Bejahung der Unterscheidungskraft unverändert voraus , dass das Zeichen geeignet sein muss , die beanspruchten Produkte als von einem bestimmten Unternehmen stammend zu kennzeichnen ( vgl. EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK`(RS)

**Example 16** (doc_id: `53614`) (sent_id: `53614`)


Soweit Personen dezentral untergebracht sind , ist es für die Bejahung einer Einrichtung erforderlich , dass die dezentrale Unterkunft zu den Räumlichkeiten der Einrichtung gehört , der Hilfebedürftige also in die Räumlichkeiten des Einrichtungsträgers eingegliedert ist ( vgl BSG SozR 4 - 3500 § 98 Nr 3 ) .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG SozR 4 - 3500 § 98 Nr 3`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG SozR 4 - 3500 § 98 Nr 3`(RS)

**Example 17** (doc_id: `53618`) (sent_id: `53618`)


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

**Example 18** (doc_id: `53623`) (sent_id: `53623`)


Nach der von der Einsprechenden zitierten BGH-Entscheidung „ Kommunikationskanal “ , muss für den Fachmann die im Anspruch bezeichnete Lehre „ unmittelbar und eindeutig “ als mögliche Ausführungsform der Erfindung den Ursprungsunterlagen entnehmbar sein .

**False Positives:**

- `BGH` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `53637`) (sent_id: `53637`)


aa ) Das Berufungsgericht ist zu der Einschätzung gelangt , die Klägerin habe nicht dargelegt , dass die Zulassung als Jaguar- und Land-Rover-Vertragswerkstatt eine Ressource darstelle , ohne die der Zugang zu dem nachgelagerten Endkundenmarkt nicht oder nicht sinnvoll möglich sei .

**False Positives:**

- `Jaguar` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 20** (doc_id: `53680`) (sent_id: `53680`)


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

**Example 21** (doc_id: `53688`) (sent_id: `53688`)


Allerdings beschränkt sich die Geltung des Grundsatzes der Bestenauslese im Bereich der Verwendungsentscheidungen auf Entscheidungen über - wie hier - höherwertige , die Beförderung in einen höheren Dienstgrad oder die Einweisung in die Planstelle einer höheren Besoldungsgruppe vorprägende Verwendungen ( vgl. klarstellend BVerwG , Beschluss vom 30. Januar 2014 - 1 WB 1.13 - juris Rn. 32 ) .

**False Positives:**

- `BVerwG` — partial — pred is substring of gold: `BVerwG , Beschluss vom 30. Januar 2014 - 1 WB 1.13 - juris Rn. 32`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerwG , Beschluss vom 30. Januar 2014 - 1 WB 1.13 - juris Rn. 32`(RS)

**Example 22** (doc_id: `53701`) (sent_id: `53701`)


Für eine solche Prognose des Arbeitgebers bedarf es ausreichend konkreter Anhaltspunkte ( BAG 27. Juli 2016 - 7 AZR 545/14 - Rn. 19 ; 24. September 2014 - 7 AZR 987/12 - Rn. 18 ; 7. Mai 2008 - 7 AZR 146/07 - Rn. 15 ; 7. April 2004 - 7 AZR 441/03 - zu II 2 a aa der Gründe ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 27. Juli 2016 - 7 AZR 545/14 - Rn. 19`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 27. Juli 2016 - 7 AZR 545/14 - Rn. 19`(RS)
- `24. September 2014 - 7 AZR 987/12 - Rn. 18`(RS)
- `7. Mai 2008 - 7 AZR 146/07 - Rn. 15`(RS)
- `7. April 2004 - 7 AZR 441/03 - zu II 2 a aa der Gründe`(RS)

**Example 23** (doc_id: `53707`) (sent_id: `53707`)


Da eine so weitgehende Selbstentäußerung des ausländischen Staates im Zweifel nicht zu vermuten ist ( BGH 30. Januar 2013 - III ZB 40/12 - Rn. 19 ) , dürfen die Umstände des Falles hinsichtlich des Vorliegens und der Reichweite eines Verzichts keinen Zweifel lassen ( BAG 29. Juni 2017 - 2 AZR 759/16 - Rn. 20 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH 30. Januar 2013 - III ZB 40/12 - Rn. 19`
- `BAG` — partial — pred is substring of gold: `BAG 29. Juni 2017 - 2 AZR 759/16 - Rn. 20`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH 30. Januar 2013 - III ZB 40/12 - Rn. 19`(RS)
- `BAG 29. Juni 2017 - 2 AZR 759/16 - Rn. 20`(RS)

**Example 24** (doc_id: `53717`) (sent_id: `53717`)


Bei Anerkennungsbeträgen handelt es sich um eine jener Massenerscheinungen , die ein typisierendes und pauschalierendes Vorgehen auch der Verwaltung rechtfertigen ( vgl. BVerfG , Beschluss vom 31. Mai 1988 - 1 BvR 520/83 - BVerfGE 78 , 214 < 227 > ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Beschluss vom 31. Mai 1988 - 1 BvR 520/83 - BVerfGE 78 , 214 < 227 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG , Beschluss vom 31. Mai 1988 - 1 BvR 520/83 - BVerfGE 78 , 214 < 227 >`(RS)

**Example 25** (doc_id: `53721`) (sent_id: `53721`)


Die Regelung ist daher geeignet , die Befristung von Arbeitsverträgen mit programmgestaltenden Mitarbeitern bei Rundfunkanstalten zu rechtfertigen ( BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101 ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101`(RS)

**Example 26** (doc_id: `53760`) (sent_id: `53760`)


Etwas anderes gilt insbesondere dann , wenn der Arbeitgeber seine Tarifgebundenheit in einer dem Arbeitnehmer hinreichend erkennbaren Weise zur auflösenden Bedingung der Bezugnahme gemacht hat ( BAG 18. April 2007 - 4 AZR 652/05 - Rn. 32 , 40 , BAGE 122 , 74 ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 18. April 2007 - 4 AZR 652/05 - Rn. 32 , 40 , BAGE 122 , 74`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 18. April 2007 - 4 AZR 652/05 - Rn. 32 , 40 , BAGE 122 , 74`(RS)

**Example 27** (doc_id: `53763`) (sent_id: `53763`)


Wird als Verfahrensmangel gerügt , das FG habe einen gestellten Beweisantrag übergangen ( Rüge mangelnder Sachaufklärung gemäß § 76 Abs. 1 Satz 1 FGO ) , so ist darzulegen , welche Tatfrage aufklärungsbedürftig ist , welche Beweismittel das FG zu welchem Beweisthema nicht erhoben hat , die genauen Fundstellen ( Schriftsatz mit Datum und Seitenzahl , Terminprotokoll ) , in denen die Beweismittel und Beweisthemen angeführt worden sind , das voraussichtliche Ergebnis der Beweisaufnahme , inwiefern das Urteil des FG aufgrund dessen sachlich-rechtlicher Auffassung auf der unterbliebenen Beweisaufnahme beruhen kann und dass die Nichterhebung der Beweise vor dem FG rechtzeitig gerügt worden ist oder aufgrund des Verhaltens des FG nicht mehr vor diesem gerügt werden konnte ( ständige BFH-Rechtsprechung , z.B. Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85 , und vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125 , m. w. N. ) .

**False Positives:**

- `BFH` — similar text (different position): `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`
- `BFH` — similar text (different position): `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 76 Abs. 1 Satz 1 FGO`(NRM)
- `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`(RS)
- `vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125`(RS)

**Example 28** (doc_id: `53767`) (sent_id: `53767`)


Ob der Erwerb eigener Anteile auf der Gesellschaftsebene entsprechend der durch das BilMoG geänderten handelsrechtlichen Vorschriften ( Einfügung des § 272 Abs. 1a und 1b HGB ) steuerrechtlich nicht mehr als Erwerbsvorgang anzusehen , sondern nunmehr als " Teilliquidation " und daher " wie " eine Kapitalherabsetzung zu behandeln ist ( so BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f. ; zustimmend z.B. Blumenberg / Lechner , DB 2014 , 141 , 147 ; kritisch z.B. Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55 ) , kann im Streitfall mangels Erheblichkeit für die Entscheidung offenbleiben .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f.`
- `DB` — partial — pred is substring of gold: `Blumenberg / Lechner , DB 2014 , 141 , 147`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BilMoG`(NRM)
- `§ 272 Abs. 1a und 1b HGB`(NRM)
- `BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f.`(REG)
- `Blumenberg / Lechner , DB 2014 , 141 , 147`(LIT)
- `Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55`(LIT)

**Example 29** (doc_id: `53775`) (sent_id: `53775`)


nicht nur das erstmalige Legen eines Hausanschlusses , sondern auch Arbeiten zur Erneuerung oder zur Reduzierung von Wasseranschlüssen unter die Steuerermäßigung fallen ( vgl. BGH-Urteil in HFR 2012 , 1110 , Rz 20 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH-Urteil in HFR 2012 , 1110 , Rz 20`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH-Urteil in HFR 2012 , 1110 , Rz 20`(RS)

**Example 30** (doc_id: `53811`) (sent_id: `53811`)


Diese Grundsätze gelten ebenso für die Anwendung der hergebrachten Grundsätze des Berufsbeamtentums im Sinne des Art. 33 Abs. 5 GG ( BVerfG , Kammerbeschluss vom 7. Oktober 2015 - 2 BvR 413/15 - NVwZ 2016 , 56 Rn. 24 f. m. w. N. ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Kammerbeschluss vom 7. Oktober 2015 - 2 BvR 413/15 - NVwZ 2016 , 56 Rn. 24 f.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 33 Abs. 5 GG`(NRM)
- `BVerfG , Kammerbeschluss vom 7. Oktober 2015 - 2 BvR 413/15 - NVwZ 2016 , 56 Rn. 24 f.`(RS)

**Example 31** (doc_id: `53820`) (sent_id: `53820`)


Wenn es - wie vorliegend - an einer ausdrücklichen Sonderzuweisung für den zuständigen Rechtsweg fehlt , bestimmt sich die gerichtliche Zuständigkeit nach der Natur des Rechtsverhältnisses , aus dem der Klageanspruch hergeleitet wird ( stRspr ; Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2 ; GmSOGB vom 10. 4. 1986 - GmS-OGB 1/85 - BGHZ 97 , 312 = SozR 1500 § 51 Nr 39 ; GmSOGB vom 29. 10. 1987 - GmS-OGB 1/86 - BGHZ 102 , 280 = SozR 1500 § 51 Nr 47 ; GmSOGB vom 10. 7. 1989 - GmS-OGB 1/88 - BGHZ 108 , 284 = SozR 1500 § 51 Nr 53 ; zum Rechtsverhältnis zwischen den Beteiligten als entscheidendes Kriterium zur Beurteilung des Rechtswegs vgl letztens etwa BSG vom 21. 7. 2014 - B 14 SF 1/14 R - SozR 4 - 1500 § 51 Nr 12 RdNr 8 ; BSG vom 25. 10. 2017 - B 7 SF 1/16 R - juris , RdNr 6 ) .

**False Positives:**

- `GmSOGB` — partial — pred is substring of gold: `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`
- `GmSOGB` — similar text (different position): `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`
- `GmSOGB` — similar text (different position): `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`
- `GmSOGB` — similar text (different position): `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`
- `BSG` — similar text (different position): `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`
- `BSG` — similar text (different position): `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`

> overlaps gold: 6  |  likely missing annotation: 0

**Gold Entities:**

- `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`(RS)
- `GmSOGB vom 10. 4. 1986 - GmS-OGB 1/85 - BGHZ 97 , 312 = SozR 1500 § 51 Nr 39`(RS)
- `GmSOGB vom 29. 10. 1987 - GmS-OGB 1/86 - BGHZ 102 , 280 = SozR 1500 § 51 Nr 47`(RS)
- `GmSOGB vom 10. 7. 1989 - GmS-OGB 1/88 - BGHZ 108 , 284 = SozR 1500 § 51 Nr 53`(RS)
- `BSG vom 21. 7. 2014 - B 14 SF 1/14 R - SozR 4 - 1500 § 51 Nr 12 RdNr 8`(RS)
- `BSG vom 25. 10. 2017 - B 7 SF 1/16 R - juris , RdNr 6`(RS)

**Example 32** (doc_id: `53822`) (sent_id: `53822`)


Dies folgt aus § 73b Abs 5 S 4 SGB V , der Abweichungen von den Vorschriften des Vierten Kapitels und damit auch von dem in § 71 Abs 1 S 1 SGB V verankerten Grundsatz der Beitragssatzstabilität zulässt ( BSG Urteil vom 25. 3. 2015 - B 6 KA 9/14 R - BSGE 118 , 164 = SozR 4 - 2500 § 73b Nr 1 , RdNr 72 ) .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG Urteil vom 25. 3. 2015 - B 6 KA 9/14 R - BSGE 118 , 164 = SozR 4 - 2500 § 73b Nr 1 , RdNr 72`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 73b Abs 5 S 4 SGB V`(NRM)
- `§ 71 Abs 1 S 1 SGB V`(NRM)
- `BSG Urteil vom 25. 3. 2015 - B 6 KA 9/14 R - BSGE 118 , 164 = SozR 4 - 2500 § 73b Nr 1 , RdNr 72`(RS)

**Example 33** (doc_id: `53843`) (sent_id: `53843`)


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

**Example 34** (doc_id: `53844`) (sent_id: `53844`)


Der im Juli 1950 geborene Kläger war seit dem 1. März 1971 bei einer Rechtsvorgängerin der Beklagten , der H AG ( im Folgenden H AG alt ) als Arbeitnehmer tätig .

**False Positives:**

- `H AG` — similar text (different position): `H AG`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `H AG`(ORG)
- `H AG alt`(ORG)

**Example 35** (doc_id: `53848`) (sent_id: `53848`)


Vielmehr setzt der Sinn und Zweck der Vorschrift voraus , dass auch das konkrete Verfahren von dem Sozialleistungsträger gerade in dieser Eigenschaft geführt wird ; das Verfahren muss also einen engen sachlichen Zusammenhang zu der gesetzlichen Tätigkeit als Träger der in der Vorschrift genannten Sozialleistungen haben ( BGH Beschluss vom 10. 11. 2005 - IX ZR 189/02 - Juris RdNr 6 f ; BGH Beschluss vom 28. 9. 2016 - XII ZB 251/16 - Juris RdNr 30 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH Beschluss vom 10. 11. 2005 - IX ZR 189/02 - Juris RdNr 6 f`
- `BGH` — similar text (different position): `BGH Beschluss vom 10. 11. 2005 - IX ZR 189/02 - Juris RdNr 6 f`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH Beschluss vom 10. 11. 2005 - IX ZR 189/02 - Juris RdNr 6 f`(RS)
- `BGH Beschluss vom 28. 9. 2016 - XII ZB 251/16 - Juris RdNr 30`(RS)

**Example 36** (doc_id: `53854`) (sent_id: `53854`)


Die insoweit verfrüht erhobene Einrede entfaltet auch mit dem Ablauf der maßgeblichen Frist ( am 12. Juni 2014 ) nicht die Rechtswirkung einer zulässigen Einrede ( BPatG GRUR 2005 , 773 , 775 - Blue Bull / RED BULL ; Ströbele / Hacker , MarkenG , 11. Aufl. , § 43 Rn. 29 m. w. N. ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG GRUR 2005 , 773 , 775 - Blue Bull / RED BULL`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG GRUR 2005 , 773 , 775 - Blue Bull / RED BULL`(RS)
- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 43 Rn. 29`(LIT)

**Example 37** (doc_id: `53855`) (sent_id: `53855`)


aa ) In organisatorischer Hinsicht ist beim Bundesverfassungsgericht , anders als bei den Fachgerichten , eine Kapazitätsausweitung zur Verkürzung der Verfahrensdauer als Reaktion auf gesteigerte Eingangszahlen grundsätzlich nicht möglich , da die Struktur des Gerichts durch seine Funktion bedingt und durch die Verfassung und das Bundesverfassungsgerichtsgesetz vorgegeben ist ( BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21 ; vgl. BTDrucks 17/3802 , S. 26 ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesverfassungsgericht`(ORG)
- `Bundesverfassungsgerichtsgesetz`(NRM)
- `BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21`(RS)
- `BTDrucks 17/3802 , S. 26`(LIT)

**Example 38** (doc_id: `53879`) (sent_id: `53879`)


Bemüht sich jemand , der ein Statusfeststellungsverfahren einleitet , zeitnah um private Eigenvorsorge , so kann er diese für den Fall , dass das Statusfeststellungsverfahren entgegen seinen Vorstellungen zu einer Feststellung von Versicherungspflicht führt , möglicherweise gar nicht mehr oder nur mit erheblichem Aufwand rückabwickeln ( zu diesen Konsequenzen siehe LSG Berlin-Brandenburg Urteil vom 18. 9. 2013 - L 9 KR 384/11 - Juris RdNr 38 ) .

**False Positives:**

- `LSG Berlin-Brandenburg` — partial — pred is substring of gold: `LSG Berlin-Brandenburg Urteil vom 18. 9. 2013 - L 9 KR 384/11 - Juris RdNr 38`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `LSG Berlin-Brandenburg Urteil vom 18. 9. 2013 - L 9 KR 384/11 - Juris RdNr 38`(RS)

**Example 39** (doc_id: `53881`) (sent_id: `53881`)


Ob dies der Fall ist , richtet sich nach den Umständen des Einzelfalls , bei denen darauf abzustellen ist , wie das Hoheitszeichen im Rahmen der Designgestaltung konkret verwendet ist ( vgl. BPatG GRUR 2002 , 337 - Schlüsselanhänger ; Eichmann / von Falckenstein / Kühne , a. a. O. , § 3 Rdn. 26 ff. ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG GRUR 2002 , 337 - Schlüsselanhänger`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG GRUR 2002 , 337 - Schlüsselanhänger`(RS)
- `Eichmann / von Falckenstein / Kühne , a. a. O. , § 3 Rdn. 26 ff.`(LIT)

**Example 40** (doc_id: `53883`) (sent_id: `53883`)


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

**Example 41** (doc_id: `53898`) (sent_id: `53898`)


Das gilt jedenfalls uneingeschränkt für das Elterngeld als fürsorgerische Leistung der Familienförderung , die über die bloße Sicherung des Existenzminimums hinausgeht ( zum Elterngeld vgl BVerfG < Kammer > Beschluss vom 9. 11. 2011 - 1 BvR 1853/11 - BVerfGK 19 , 186 ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG < Kammer > Beschluss vom 9. 11. 2011 - 1 BvR 1853/11 - BVerfGK 19 , 186`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG < Kammer > Beschluss vom 9. 11. 2011 - 1 BvR 1853/11 - BVerfGK 19 , 186`(RS)

**Example 42** (doc_id: `53910`) (sent_id: `53910`)


Zur Vermeidung einer mittelbaren Diskriminierung wegen Behinderung sei er nach der Entscheidung des Gerichtshofs der Europäischen Union ( EuGH ) vom 6. Dezember 2012 ( - C- 152/11 - [ Odar ] ) wie ein nicht schwerbehinderter Arbeitnehmer zu behandeln .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `Entscheidung des Gerichtshofs der Europäischen Union ( EuGH ) vom 6. Dezember 2012 ( - C- 152/11 - [ Odar ] )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Entscheidung des Gerichtshofs der Europäischen Union ( EuGH ) vom 6. Dezember 2012 ( - C- 152/11 - [ Odar ] )`(RS)

**Example 43** (doc_id: `53937`) (sent_id: `53937`)


Da allein das Fehlen jeglicher Unterscheidungskraft ein Eintragungshindernis begründet , ist ein großzügiger Maßstab anzulegen , so dass jede auch noch so geringe Unterscheidungskraft genügt , um das Schutzhindernis zu überwinden ( BGH a. a. O. – OUI ; a. a. O. – for you ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH a. a. O. – OUI`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH a. a. O. – OUI`(RS)
- `a. a. O. – for you`(RS)

**Example 44** (doc_id: `53938`) (sent_id: `53938`)


In dem besonderen Fall der Sanktionierung von Verstößen gegen die Verordnung [ … ] wurden jedoch die straf- oder bußgeldbewehrten Vorschriften der Verordnung [ … ] durch das Inkrafttreten der Sanktionsvorschriften vor dem Anwendungszeitpunkt der bewehrten EU-Verordnung bereits ab dem 2. Juli 2016 in Deutschland für anwendbar erklärt .

**False Positives:**

- `EU` — similar text (different position): `Deutschland`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Deutschland`(LOC)

**Example 45** (doc_id: `53963`) (sent_id: `53963`)


Der Gesamteindruck aber kann auch bei Übernahme der geschützten „ Schnittmenge “ durch Hinzufügung weiterer Merkmale im Einzelfall erheblich verändert werden ( vgl. OLG Köln , 6 U 128/09 v. 8. Januar 2010 ; veröffentlicht in juris ; Klawitter in : Festschrift 50 Jahre Bundespatentgericht , S. 1071 , 1079 ) .

**False Positives:**

- `Bundespatentgericht` — partial — pred is substring of gold: `Klawitter in : Festschrift 50 Jahre Bundespatentgericht , S. 1071 , 1079`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `OLG Köln , 6 U 128/09 v. 8. Januar 2010 ; veröffentlicht in juris`(RS)
- `Klawitter in : Festschrift 50 Jahre Bundespatentgericht , S. 1071 , 1079`(LIT)

**Example 46** (doc_id: `53981`) (sent_id: `53981`)


Dies ergibt sich z.B. aus der Begründung zum Entwurf der Bundesregierung eines Gesetzes zur Einführung einer Musterwiderrufsinformation für Verbraucherdarlehensverträge , zur Änderung der Vorschriften über das Widerrufsrecht bei Verbraucherdarlehensverträgen und zur Änderung des Darlehensvermittlungsrechts vom 29. 4. 2010 ( BT-Drs. 17/1394 , S. 15 ) .

**False Positives:**

- `Bundesregierung` — partial — pred is substring of gold: `Begründung zum Entwurf der Bundesregierung eines Gesetzes zur Einführung einer Musterwiderrufsinformation für Verbraucherdarlehensverträge , zur Änderung der Vorschriften über das Widerrufsrecht bei Verbraucherdarlehensverträgen und zur Änderung des Darlehensvermittlungsrechts vom 29. 4. 2010 ( BT-Drs. 17/1394 , S. 15 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Begründung zum Entwurf der Bundesregierung eines Gesetzes zur Einführung einer Musterwiderrufsinformation für Verbraucherdarlehensverträge , zur Änderung der Vorschriften über das Widerrufsrecht bei Verbraucherdarlehensverträgen und zur Änderung des Darlehensvermittlungsrechts vom 29. 4. 2010 ( BT-Drs. 17/1394 , S. 15 )`(LIT)

**Example 47** (doc_id: `53982`) (sent_id: `53982`)


Der Rundfunkbeitrag wird erhoben , um den individuellen Nutzungsvorteil abzugelten ( BVerwG , Urteil vom 18. März 2016 - 6 C 6.15 - BVerwGE 154 , 275 Rn. 25 ff. ) .

**False Positives:**

- `BVerwG` — partial — pred is substring of gold: `BVerwG , Urteil vom 18. März 2016 - 6 C 6.15 - BVerwGE 154 , 275 Rn. 25 ff.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerwG , Urteil vom 18. März 2016 - 6 C 6.15 - BVerwGE 154 , 275 Rn. 25 ff.`(RS)

**Example 48** (doc_id: `53991`) (sent_id: `53991`)


Nicht das tatsächliche Verhalten des Arbeitgebers im Lohnsteuerabzugsverfahren bindet dessen Beteiligte ( vgl BSG Urteil vom 23. 5. 2017 - B 12 KR 6/16 R - SozR 4 - 5376 § 1 Nr 1 RdNr 23 ; BSG Urteil vom 26. 3. 2014 - B 10 EG 14/13 R - BSGE 115 , 198 = SozR 4 - 7837 § 2 Nr 25 , RdNr 26 f ) , wohl aber die Rechtsfolgen , die AO und EStG daran knüpfen .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG Urteil vom 23. 5. 2017 - B 12 KR 6/16 R - SozR 4 - 5376 § 1 Nr 1 RdNr 23`
- `BSG` — similar text (different position): `BSG Urteil vom 23. 5. 2017 - B 12 KR 6/16 R - SozR 4 - 5376 § 1 Nr 1 RdNr 23`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Urteil vom 23. 5. 2017 - B 12 KR 6/16 R - SozR 4 - 5376 § 1 Nr 1 RdNr 23`(RS)
- `BSG Urteil vom 26. 3. 2014 - B 10 EG 14/13 R - BSGE 115 , 198 = SozR 4 - 7837 § 2 Nr 25 , RdNr 26 f`(RS)
- `AO`(NRM)
- `EStG`(NRM)

**Example 49** (doc_id: `53995`) (sent_id: `53995`)


Sind beide Anträge - wie hier - Gegenstand desselben Rechtsstreits , kann über sie gleichzeitig verhandelt und entschieden werden ( vgl. BAG 27. April 2006 - 2 AZR 360/05 - Rn. 19 , aaO ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 27. April 2006 - 2 AZR 360/05 - Rn. 19 , aaO`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 27. April 2006 - 2 AZR 360/05 - Rn. 19 , aaO`(RS)

**Example 50** (doc_id: `54027`) (sent_id: `54027`)


Dies gilt auch in den Fällen einer Erkrankung mit einer nur noch begrenzten Lebenserwartung , da die Regelung des § 64 Abs. 1 Satz 1 Nr. 1 EStDV i. d. F. des StVereinfG 2011 keine Differenzierung zwischen verschiedenen Krankheitskosten enthält ( BFH-Urteil vom 25. April 2017 VIII R 52/13 , BFHE 258 , 53 , BStBl II 2017 , 949 ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH-Urteil vom 25. April 2017 VIII R 52/13 , BFHE 258 , 53 , BStBl II 2017 , 949`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 64 Abs. 1 Satz 1 Nr. 1 EStDV`(NRM)
- `StVereinfG 2011`(NRM)
- `BFH-Urteil vom 25. April 2017 VIII R 52/13 , BFHE 258 , 53 , BStBl II 2017 , 949`(RS)

**Example 51** (doc_id: `54030`) (sent_id: `54030`)


Die vermeintliche Unrichtigkeit einer Entscheidung des Berufungsgerichts eröffnet aber nicht die Revisionsinstanz ( vgl BSG SozR 1500 § 160a Nr 7 ) .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG SozR 1500 § 160a Nr 7`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG SozR 1500 § 160a Nr 7`(RS)

**Example 52** (doc_id: `54041`) (sent_id: `54041`)


3.1 Zunächst hat der Gerichtshof im Hinblick auf die Auslegungskriterien zu Art. 3 ( a ) AMVO festgestellt , dass es unzulässig ist , ein ergänzendes Schutzzertifikat für solche Wirkstoffe zu erteilen , die in den Ansprüchen des Grundpatents nicht genannt sind ( EuGH , GRUR 2012 , 257 , Rnd. 28 – Medeva ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH , GRUR 2012 , 257 , Rnd. 28 – Medeva`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 3 ( a ) AMVO`(NRM)
- `EuGH , GRUR 2012 , 257 , Rnd. 28 – Medeva`(RS)

**Example 53** (doc_id: `54108`) (sent_id: `54108`)


Mit Beschluss vom 23. September 2015 hat die Patentabteilung des DPMA den Antrag zurückgewiesen .

**False Positives:**

- `DPMA` — partial — pred is substring of gold: `Patentabteilung des DPMA`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Patentabteilung des DPMA`(ORG)

**Example 54** (doc_id: `54121`) (sent_id: `54121`)


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

**Example 55** (doc_id: `54133`) (sent_id: `54133`)


aa ) Ergibt sich aus dem Vortrag der Parteien im Rechtsstreit , dass die normative Wirkung eines Tarifvertrags nach § 4 Abs. 1 , § 5 Abs. 4 TVG in Betracht kommt , muss das Gericht diese Normen nach § 293 ZPO von Amts wegen ermitteln ( BAG 25. Januar 2017 - 4 AZR 520/15 - Rn. 29 mwN ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 25. Januar 2017 - 4 AZR 520/15 - Rn. 29`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 4 Abs. 1 , § 5 Abs. 4 TVG`(NRM)
- `§ 293 ZPO`(NRM)
- `BAG 25. Januar 2017 - 4 AZR 520/15 - Rn. 29`(RS)

**Example 56** (doc_id: `54143`) (sent_id: `54143`)


Dieser während des Revisionsverfahrens eingetretene Zuständigkeitswechsel führt zu einem gesetzlichen Beteiligtenwechsel ( vgl. z.B. Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109 , m. w. N. ) .

**False Positives:**

- `Bundesfinanzhofs` — partial — pred is substring of gold: `Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109`
- `BFH` — partial — pred is substring of gold: `Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109`(RS)

**Example 57** (doc_id: `54150`) (sent_id: `54150`)


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

**Example 58** (doc_id: `54160`) (sent_id: `54160`)


aa ) Die Jahressonderzuwendung hat - ähnlich wie die Jahressonderzahlung nach § 20 TV-L / TVöD ( vgl. dazu BAG 12. Dezember 2012 - 10 AZR 922/11 - Rn. 20 , BAGE 144 , 117 ) - mehrere erkennbare Zwecke .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 12. Dezember 2012 - 10 AZR 922/11 - Rn. 20 , BAGE 144 , 117`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 20 TV-L / TVöD`(REG)
- `BAG 12. Dezember 2012 - 10 AZR 922/11 - Rn. 20 , BAGE 144 , 117`(RS)

**Example 59** (doc_id: `54169`) (sent_id: `54169`)


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

**Example 60** (doc_id: `54179`) (sent_id: `54179`)


I. Mit dem angefochtenen Beschluss vom 4. November 2015 hat die Patentabteilung 43 des Deutschen Patent- und Markenamts das Patent 103 36 913 mit der Bezeichnung

**False Positives:**

- `Deutschen Patent- und Markenamts` — partial — pred is substring of gold: `Patentabteilung 43 des Deutschen Patent- und Markenamts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Patentabteilung 43 des Deutschen Patent- und Markenamts`(ORG)

**Example 61** (doc_id: `54192`) (sent_id: `54192`)


- 100 mg Granulat zur Zubereitung oral einzunehmender Suspensionen unter der Nummer EU / 1 / 07 / 436 / 005 .

**False Positives:**

- `EU` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 62** (doc_id: `54202`) (sent_id: `54202`)


Eine Vertragslücke , die einer Schließung durch den Rückgriff auf dispositives Gesetzesrecht oder einer ergänzenden Vertragsauslegung bedurft hätte ( BAG 15. Dezember 2016 - 6 AZR 478/15 - Rn. 31 , BAGE 157 , 284 ) , bestand nicht .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 15. Dezember 2016 - 6 AZR 478/15 - Rn. 31 , BAGE 157 , 284`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 15. Dezember 2016 - 6 AZR 478/15 - Rn. 31 , BAGE 157 , 284`(RS)

</details>

---

## `Specific Court Departments and Senates`

**F1:** 0.317 | **Precision:** 0.274 | **Recall:** 0.374  

**Format:** `regex`  
**Rule ID:** `99bc8170`  
**Description:**
Matches specific German court names and abbreviations that were previously in the long list, now using a more structured approach for common courts and abbreviations.

**Content:**
```
\b(?:BVerfG|BGH|BFH|BVerwG|BAG|BSG|EuGH|DPMA|BPatG|ZDS|ZIV|GBA|GEW|X-EWIV|GmSOGB|EGMR|NATO|EU|MDK|BA|VCS|Google|RAPAMUNE|nivo|KAEFER|ARD|ZDF|Deutschlandradio|HBV|K\u00c4V|ver\.di|DB|BND|RVA|TdL|InWIS|DRV|KOSMICA|Monster\s*Abwehr\s*Spray|Haus\u00e4rzteverband|Bundesvereinigung\s*der\s*Arbeitgeberverb\u00e4nde|Deutsche\s*Rentenversicherung\s*Bund|Ausw\u00e4rtige\s*Amt|Bundesamt\s*f\u00fcr\s*Migration\s*und\s*Fl\u00fcchtlinge|BMF|Internationalen\s*Gerichtshofs|Europ\u00e4ischen\s*Kommission|Justizkommission\s*zu\s*Tunis|Bundeswehr|Fliegerhorst\s*B\u00fcchel|St\u00e4dtischen\s*Klinikum\s*K\.|Kernkraftwerks\s*Kr\u00fcmmel|Landesarbeitsgericht\s*Berlin-Brandenburg|S\u00e4chsische\s*Bildungsagentur|Gemeinsamen\s*Bundesausschusses|Landesaussch\u00fcsse|Obersten\s*Verwaltungsgerichts\s*der\s*Republik\s*Bulgarien|Bundeskasse\s*Halle\s*/\s*Saale\s*-\s*Dienstsitz\s*Weiden\s*/\s*Oberpfalz|Diakonischen\s*Werkes\s*der\s*Evangelischen\s*Kirche\s*in\s*Deutschland\s*e\.\s*V\.|Evangelischen\s*Entwicklungsdienstes\s*e\.\s*V\.|Haus\u00e4rztlliche\s*Vertragsgemeinschaft\s*Aktiengesellschaft|H\u00c4VG-Rechenzentrum\s*AG|H\u00e4VG-Rechenzentrum\s*GmbH|B\s*\u2026\s*Patentanwaltsgesellschaft\s*mbH|C\s*\u2026\s*GmbH|D\s*P\s*T\s*S\s*GmbH|K\s*\u2026\s*GmbH|B\s*\u2026\s*AG|S\s*\u2026|C-\s*B\.\s*V\.|A-Fonds|BgA\s*X|InEK|Monster\s*Abwehr\s*Spray|Haus\u00e4rzteverband|Bundesvereinigung\s*der\s*Arbeitgeberverb\u00e4nde|Deutsche\s*Rentenversicherung\s*Bund|Ausw\u00e4rtige\s*Amt|Bundesamt\s*f\u00fcr\s*Migration\s*und\s*Fl\u00fcchtlinge|BMF|Internationalen\s*Gerichtshofs|Europ\u00e4ischen\s*Kommission|Justizkommission\s*zu\s*Tunis|Bundeswehr|Fliegerhorst\s*B\u00fcchel|St\u00e4dtischen\s*Klinikum\s*K\.|Kernkraftwerks\s*Kr\u00fcmmel|Landesarbeitsgericht\s*Berlin-Brandenburg|S\u00e4chsische\s*Bildungsagentur|Gemeinsamen\s*Bundesausschusses|Landesaussch\u00fcsse|Obersten\s*Verwaltungsgerichts\s*der\s*Republik\s*Bulgarien|Bundeskasse\s*Halle\s*/\s*Saale\s*-\s*Dienstsitz\s*Weiden\s*/\s*Oberpfalz|Diakonischen\s*Werkes\s*der\s*Evangelischen\s*Kirche\s*in\s*Deutschland\s*e\.\s*V\.|Evangelischen\s*Entwicklungsdienstes\s*e\.\s*V\.|Haus\u00e4rztlliche\s*Vertragsgemeinschaft\s*Aktiengesellschaft|H\u00c4VG-Rechenzentrum\s*AG|H\u00e4VG-Rechenzentrum\s*GmbH|B\s*\u2026\s*Patentanwaltsgesellschaft\s*mbH|C\s*\u2026\s*GmbH|D\s*P\s*T\s*S\s*GmbH|K\s*\u2026\s*GmbH|B\s*\u2026\s*AG|S\s*\u2026|C-\s*B\.\s*V\.|A-Fonds|BgA\s*X|InEK|Bundesregierung|Bundesministerium\s+der\s+Finanzen|Bundesamts\s+f\u00fcr\s+Justiz|Justizministerium\s+des\s+Landes\s+Nordrhein-Westfalen|Neurologischen\s+Klinik\s+B|Amtsgericht\s+O\.|Handwerksverband\s+Metallbau\s+und\s+Feinwerktechnik\s+Baden-W\u00fcrttemberg|Industriegewerkschaft\s+Metall|VEB\s+[A-Z][a-zA-Z\s]+|nieders\u00e4chsische\s+Landesschulbeh\u00f6rde|EON-Konzerns|EON-Konzern|dbb\s+beamtenbund\s+und\s+tarifunion|ADAC|Kernkraftwerks\s+Biblis|Kernkraftwerks\s+M\u00fclheim-K\u00e4rlich|Deutschen\s*Botschaft|Finanzamt\s+[A-Za-z\u00e4\u00f6\u00fc\u00df\s]+|Bundesministeriums\s+der\s+Verteidigung\s+-\s+R\s+II\s+2\s+-|Bundesamts\s+f\u00fcr\s+das\s+Personalmanagement|A\s+Lebensversicherung\s+AG|Europ\u00e4ische\s+Gerichtshof\s+f\u00fcr\s+Menschenrechte|Th\u00fcringer\s+Landessozialgerichts|Deutsche\s+Patent-\s+und\s+Markenamt|Deutschen\s+Patent-\s+und\s+Markenamt|Deutschen\s+Patent-\s+und\s+Markenamtes|K\u00c4V\s+Brandenburg|Neue\s+Richtervereinigung|Generalstaatsanwaltschaft\s+des\s+Landes\s+Schleswig-Holstein|K-Klinik|H\u00c4VG|Schott|PreussenElektra\s+GmbH|E\.\s+ON\s+Kernkraft\s+GmbH|G-Gruppe|Vereinigung\s+der\s+kommunalen\s*Arbeitgeberverb\u00e4nde|Staatskasse|Kernkraftwerks\s+Gundremmingen|Bundesministeriums\s+der\s+Verteidigung|Bundesministerium\s+der\s+Verteidigung|Bundesverfassungsgerichts|Bundesverwaltungsgerichts|Bundesgerichtshofs|Bundesfinanzhofs|Bundesarbeitsgerichts|Bundessozialgerichts|Landgerichts\s+Darmstadt|Landgerichts\s+D\u00fcsseldorf|Landgerichts\s+Hamburg|Landgerichts\s+Bremen|Landgerichts\s+Oldenburg|Landgerichts\s+Karlsruhe|Landgerichts\s+Potsdam|Landgerichts\s+F\.\s*\(\s*P\.\s*\)|Pf\u00e4lzische\s+Oberlandesgericht\s+Zweibr\u00fccken|Oberlandesgerichts\s+M\u00fcnchen|Oberlandesgerichts\s+Hamm|Amtsgerichts\s+O\.|Finanzamts\s+[A-Za-z\u00e4\u00f6\u00fc\u00df\s]+|Bundesamts\s+f\u00fcr\s+das\s+Personalmanagement\s+der\s+Bundeswehr|Markenstelle\s*f\u00fcr\s*Klasse\s*\d+\s*des\s*Deutschen\s*Patent-\s*und\s*Markenamts|Patentabteilung\s*\d+\.\d+\s*des\s*Deutschen\s*Patent-\s*und\s*Markenamts|Designabteilung\s*\d+\.\d+\s*des\s*Deutschen\s*Patent-\s*und\s*Markenamts|Markenstelle\s*f\u00fcr\s*Klasse\s*\d+\s*des\s*DPMA|Patentabteilung\s*\d+\.\d+\s*des\s*DPMA|Designabteilung\s*\d+\.\d+\s*des\s*DPMA|Deutschen\s*Patent-\s*und\s*Markenamts|Deutschen\s*Patent-\s*und\s*Markenamt|Deutsche\s*Patent-\s*und\s*Markenamt|Deutsche\s*Patent-\s*und\s*Markenamts|Statistischen\s*Bundesamt|Deutschen\s*Bundestages|Spitzenverband\s*Bund\s*der\s*KKn|Verband\s*der\s*Privaten\s*Krankenversicherung|Deutsche\s*Krankenhausgesellschaft|VKDA|Gro\u00dfen\s*Senat\s*des\s*BFH|VIII\.\s*Senat\s*des\s*BFH|VIII\.\s*Senats\s*des\s*BFH|LSG\s+Berlin-Brandenburg|RWE-Konzerns|B\.\s*GmbH|w\s*GmbH|w\s*Holding\s*GmbH|P\s*GmbH|G\s*GmbH|M-GmbH\s*&\s*atypisch\s*Still|Gewerkschaft\s*ver\.di|Kreiskrankenh\u00e4user\s*M\s+und\s*R|NIVONA|fluege\.de|CHECK24\.de|Jaguar|Land\s*Rover|\u00d6z\s*Gaziantep\s*Dilim\s*Baklavalari|Wohnungsbau-\s*und\s*Kommissionsgesellschaft\s*Reichenstra\u00dfen|A-Fonds|B-Fonds|A-AG|B-AG|C-AG|D-AG|E-AG|F-AG|G-AG|H-AG|I-AG|J-AG|K-AG|L-AG|M-AG|N-AG|O-AG|P-AG|Q-AG|R-AG|S-AG|T-AG|U-AG|V-AG|W-AG|X-AG|Y-AG|Z-AG|A\s*Lebensversicherung\s*AG|B\s*Lebensversicherung\s*AG|C\s*Lebensversicherung\s*AG|D\s*Lebensversicherung\s*AG|E\s*Lebensversicherung\s*AG|F\s*Lebensversicherung\s*AG|G\s*Lebensversicherung\s*AG|H\s*Lebensversicherung\s*AG|I\s*Lebensversicherung\s*AG|J\s*Lebensversicherung\s*AG|K\s*Lebensversicherung\s*AG|L\s*Lebensversicherung\s*AG|M\s*Lebensversicherung\s*AG|N\s*Lebensversicherung\s*AG|O\s*Lebensversicherung\s*AG|P\s*Lebensversicherung\s*AG|Q\s*Lebensversicherung\s*AG|R\s*Lebensversicherung\s*AG|S\s*Lebensversicherung\s*AG|T\s*Lebensversicherung\s*AG|U\s*Lebensversicherung\s*AG|V\s*Lebensversicherung\s*AG|W\s*Lebensversicherung\s*AG|X\s*Lebensversicherung\s*AG|Y\s*Lebensversicherung\s*AG|Z\s*Lebensversicherung\s*AG)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.274 | 0.374 | 0.317 | 1101 | 302 | 799 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 302 | 799 | 504 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53400`) (sent_id: `53400`)


Die Vorgabe des Bundesfinanzhofs ( BFH ) , dass die Einsteller ( Leistungsempfänger ) keine Land- und Forstwirte sein müssten , die von ihnen empfangenen Leistungen ( Zuchtleistungen als einheitliche Leistung ) aber gleichwohl zu landwirtschaftlichen Zwecken genutzt werden müssten , erscheine nicht eindeutig und auf den ersten Blick widersprüchlich .

| Predicted | Gold |
|---|---|
| `Bundesfinanzhofs` | `Bundesfinanzhofs` |
| `BFH` | `BFH` |

**Example 1** (doc_id: `53404`) (sent_id: `53404`)


Hiergegen richtet sich die mit Schriftsatz vom 11. Februar 2014 beim Deutschen Patent- und Markenamt eingelegte Beschwerde der Anmelderin , die ihr Patentbegehren mit den mit Schreiben vom 25. Februar 2011 eingereichten Unterlagen weiterverfolgt .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Example 2** (doc_id: `53408`) (sent_id: `53408`)


Ein solcher enger sachlicher Zusammenhang mit der gesetzlichen Tätigkeit als Träger der Grundsicherung für Arbeitsuchende ist bei der BA immer dann anzunehmen , wenn sie - objektiv betrachtet - in dem zugrunde liegenden Verfahren eine Aufgabe im Rahmen des Vollzugs des SGB II wahrnimmt .

| Predicted | Gold |
|---|---|
| `BA` | `BA` |

**Missed by this rule (FN):**

- `SGB II` (NRM)

**Example 3** (doc_id: `53415`) (sent_id: `53415`)


Die Form der Benutzung ist insoweit jeweils glaubhaft gemacht durch die als Anlagen W14 bis W17 vorgelegten Produktkataloge aus dem vorliegend relevanten Benutzungszeitraum mit Abbildungen einer Vielzahl von ( elektrischen ) Kaffeemaschinen , Kaffeemühlen , Milchbehälter ( „ Milchcooler “ ) sowie Verpackungen / Behälter von Reinigungstabs , CreamCleaner und Entkalker , auf denen die Widerspruchsmarke NIVONA in einer ohne weitere besondere graphische Ausgestaltungselemente enthaltenden und damit in einer den kennzeichnenden Charakter der eingetragenen Marke nicht verändernden Form i. S. von § 26 Abs. 3 MarkenG deutlich sichtbar abgebildet ist .

| Predicted | Gold |
|---|---|
| `NIVONA` | `NIVONA` |

**Missed by this rule (FN):**

- `§ 26 Abs. 3 MarkenG` (NRM)

**Example 4** (doc_id: `53419`) (sent_id: `53419`)


Hiergegen richtet sich die Beschwerde der Anmelderin vom 29. November 2016 , mit der sie sinngemäß beantragt , den Beschluss der Markenstelle für Klasse 35 des Deutschen Patent- und Markenamts vom 11. November 2016 aufzuheben .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 35 des Deutschen Patent- und Markenamts` | `Markenstelle für Klasse 35 des Deutschen Patent- und Markenamts` |

**Example 5** (doc_id: `53424`) (sent_id: `53424`)


Die X-EWIV wäre dem Kläger in vollem Umfang auskunfts- und rechenschaftspflichtig gewesen .

| Predicted | Gold |
|---|---|
| `X-EWIV` | `X-EWIV` |

**Example 6** (doc_id: `53430`) (sent_id: `53430`)


II. Die zulässige Beschwerde der Löschungsantragsgegnerin hat sich durch die zwischenzeitlich mit Beschluss des DPMA vom 27. September 2017 angeordnete Verfallslöschung der angegriffenen Marke in der Hauptsache erledigt .

| Predicted | Gold |
|---|---|
| `DPMA` | `DPMA` |

**Example 7** (doc_id: `53466`) (sent_id: `53466`)


Die PreussenElektra GmbH war Beschwerdeführerin im Verfassungsbeschwerdeverfahren 1 BvR 2821/11 ( BVerfGE 143 , 246 ) , das sich ebenfalls gegen die hier angegriffenen Regelungen des Atomgesetzes richtete .

| Predicted | Gold |
|---|---|
| `PreussenElektra GmbH` | `PreussenElektra GmbH` |

**Missed by this rule (FN):**

- `Verfassungsbeschwerdeverfahren 1 BvR 2821/11 ( BVerfGE 143 , 246 )` (RS)
- `Atomgesetzes` (NRM)

**Example 8** (doc_id: `53524`) (sent_id: `53524`)


Der BFH führte aus , die handelsrechtliche Zurechnung von Vermögensgegenständen entspreche im Wesentlichen der Regelung des § 39 Abs. 2 Nr. 1 Satz 1 AO ( z.B. BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1. ; vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a ) .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Missed by this rule (FN):**

- `§ 39 Abs. 2 Nr. 1 Satz 1 AO` (NRM)
- `BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1.` (RS)
- `vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a` (RS)

**Example 9** (doc_id: `53539`) (sent_id: `53539`)


cc ) Stellt ein Bewerber nicht innerhalb eines Monats nach Zugang der Abbruchmitteilung einen Antrag auf Erlass einer einstweiligen Verfügung , darf der Dienstherr nach der Rechtsprechung des Bundesverwaltungsgerichts darauf vertrauen , dass der Bewerber den Abbruch des Auswahlverfahrens nicht angreift , sondern sein Begehren im Rahmen einer neuen Ausschreibung weiterverfolgt ( vgl. BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14 ) .

| Predicted | Gold |
|---|---|
| `Bundesverwaltungsgerichts` | `Bundesverwaltungsgerichts` |

**Missed by this rule (FN):**

- `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14` (RS)

**Example 10** (doc_id: `53549`) (sent_id: `53549`)


aa ) Der GBA hat eine IVIG-Therapie zur Behandlung des sekundären Immunglobulinmangels mit MGUS und rezidivierenden Pneumonien nicht empfohlen .

| Predicted | Gold |
|---|---|
| `GBA` | `GBA` |

**Example 11** (doc_id: `53574`) (sent_id: `53574`)


Die Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts hat diese unter der Nummer 30 2014 034 614.1 geführte Anmeldung mit Beschluss vom 25. November 2014 wegen fehlender Unterscheidungskraft zurückgewiesen .

| Predicted | Gold |
|---|---|
| `Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts` | `Markenstelle für Klasse 42 des Deutschen Patent- und Markenamts` |

**Example 12** (doc_id: `53594`) (sent_id: `53594`)


Die von der Krankenkasse nicht getragenen Kosten der stationären Behandlung zahlte der Kläger zu 1. Die Therapie der E in der K-Klinik war im Vorfeld durch den Medizinischen Dienst der Krankenversicherungen ( MDK ) auf ihre medizinische Notwendigkeit geprüft und befürwortet worden .

| Predicted | Gold |
|---|---|
| `K-Klinik` | `K-Klinik` |
| `MDK` | `MDK` |

**Missed by this rule (FN):**

- `E` (PER)
- `Medizinischen Dienst der Krankenversicherungen` (ORG)

**Example 13** (doc_id: `53630`) (sent_id: `53630`)


Die Verträge seien derart angelegt gewesen , dass die KG am Ende der Laufzeit ihr " Andienungsrecht " ausübe und die P GmbH die Leasingobjekte zurückerwerben müsse .

| Predicted | Gold |
|---|---|
| `P GmbH` | `P GmbH` |

**Example 14** (doc_id: `53656`) (sent_id: `53656`)


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

**Example 15** (doc_id: `53664`) (sent_id: `53664`)


Soweit das Bundesministerium der Verteidigung darauf verweist , dass Personalgespräche dazu dienten , Soldaten über Verwendungsplanungen der Personalführung zu informieren und in diesem Zusammenhang deren persönliche Motivationslage zu ergründen , um das dienstliche Bedürfnis und die privaten Lebensumstände des Soldaten möglichst gut miteinander zu vereinbaren , trifft dies zweifellos auf Personalgespräche im Vorfeld von Personalmaßnahmen zu .

| Predicted | Gold |
|---|---|
| `Bundesministerium der Verteidigung` | `Bundesministerium der Verteidigung` |

**Example 16** (doc_id: `53690`) (sent_id: `53690`)


An dieser Regelung hielt die Bundesregierung fest , obwohl der Bundesrat die Einbeziehung der einmaligen Einnahmen vorschlug ( BT-Drucks 16/2454 S 11 ) .

| Predicted | Gold |
|---|---|
| `Bundesregierung` | `Bundesregierung` |

**Missed by this rule (FN):**

- `Bundesrat` (ORG)
- `BT-Drucks 16/2454 S 11` (LIT)

**Example 17** (doc_id: `53746`) (sent_id: `53746`)


Das hat zur Folge , dass beim BSG über Kostenerinnerungen nach § 189 Abs 2 S 2 SGG der Senat in der Besetzung mit drei Berufsrichtern zu befinden hat .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Missed by this rule (FN):**

- `§ 189 Abs 2 S 2 SGG` (NRM)

**Example 18** (doc_id: `53773`) (sent_id: `53773`)


Vielmehr hat das LSG nach erfolgter Zurückverweisung durch das BSG nach umfangreicher weiterer Sachaufklärung den Anspruch des Klägers erneut geprüft und eine Entscheidung gefällt .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 19** (doc_id: `53777`) (sent_id: `53777`)


Am 14. November 2000 wurden den Aktionären der A-AG vereinbarungsgemäß ... Stück vinkulierte Namensaktien an der B-AG übertragen .

| Predicted | Gold |
|---|---|
| `A-AG` | `A-AG` |
| `B-AG` | `B-AG` |

**Example 20** (doc_id: `53791`) (sent_id: `53791`)


Zwischen den Vergleichszeichen besteht hinsichtlich der angegriffenen Waren Verwechslungsgefahr gemäß § 9 Abs. 1 Nr. 2 MarkenG , so dass der Widerspruch aus der Marke 30 2011 047 766 vom Deutschen Patent- und Markenamt zu Unrecht zurückgewiesen wurde .

| Predicted | Gold |
|---|---|
| `Deutschen Patent- und Markenamt` | `Deutschen Patent- und Markenamt` |

**Missed by this rule (FN):**

- `§ 9 Abs. 1 Nr. 2 MarkenG` (NRM)

**Example 21** (doc_id: `53862`) (sent_id: `53862`)


Patentansprüche 1 bis 13 vom 24. November 2017 , beim BPatG als 6. Hilfsantrag per Fax eingegangen am 27. November 2017

| Predicted | Gold |
|---|---|
| `BPatG` | `BPatG` |

**Example 22** (doc_id: `53867`) (sent_id: `53867`)


Zwar kann das LSG von einer Entscheidung ua des BSG auch dann abweichen , wenn es einen der höchstrichterlichen Rechtsprechung widersprechenden Rechtssatz nur sinngemäß und in scheinbar fallbezogene Ausführungen gekleidet entwickelt .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 23** (doc_id: `53939`) (sent_id: `53939`)


Dort hat der BFH lediglich ausgeführt , die Vergütung für die Hingabe eines partiarischen Darlehens könne auch umsatzabhängig ausgestaltet werden .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Example 24** (doc_id: `54007`) (sent_id: `54007`)


Das Unterrichtungsschreiben vom 14. November 2005 enthält weder Angaben zum Sitz der D P T S GmbH noch zu deren Anschrift , zum zuständigen Registergericht und zur Registernummer .

| Predicted | Gold |
|---|---|
| `D P T S GmbH` | `D P T S GmbH` |

**Example 25** (doc_id: `54059`) (sent_id: `54059`)


Wie der Kläger selbst vorträgt , wollte sich das LSG vielmehr nach seinem eigenen Verständnis im Rahmen der bereits vorliegenden Rechtsprechung des BSG halten , ist diesen aber in dem von ihm entschiedenen Fall lediglich deshalb nicht gefolgt , weil es - vom Kläger als unzutreffend gerügte - wesentliche Sachverhaltsunterschiede zu den bereits entschiedenen Fällen angenommen hat .

| Predicted | Gold |
|---|---|
| `BSG` | `BSG` |

**Example 26** (doc_id: `54064`) (sent_id: `54064`)


Der BFH prüft insofern nur , ob sie gegen Denkgesetze und Erfahrungssätze oder die anerkannten Auslegungsregeln verstößt .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Example 27** (doc_id: `54130`) (sent_id: `54130`)


b ) In Leasingfällen ( auch beim " Lease " im Rahmen eines " Sale-and-lease-back " ) geht der BFH - wie bereits in dem Urteil in BFHE 255 , 386 , BStBl II 2018 , 81 dargestellt - bei Anwendung des § 39 Abs. 2 Nr. 1 Satz 1 AO insbesondere von folgenden Grundsätzen aus :

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Missed by this rule (FN):**

- `Urteil in BFHE 255 , 386 , BStBl II 2018 , 81` (RS)
- `§ 39 Abs. 2 Nr. 1 Satz 1 AO` (NRM)

**Example 28** (doc_id: `54150`) (sent_id: `54150`)


Die Prüfung der Markenanmeldung muss daher nach der maßgeblichen Rechtsprechung des EuGH grundsätzlich streng und vollständig sein , um ungerechtfertigte Eintragungen zu vermeiden ( vgl. EuGH , GRUR 2003 , 604 Rn. 57 , 60 – Libertel ; BGH , GRUR 2014 , 565 Rn. 17 – smartbook ; Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 158 , 159 ) .

| Predicted | Gold |
|---|---|
| `EuGH` | `EuGH` |

**Missed by this rule (FN):**

- `EuGH , GRUR 2003 , 604 Rn. 57 , 60 – Libertel` (RS)
- `BGH , GRUR 2014 , 565 Rn. 17 – smartbook` (RS)
- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 8 Rn. 158 , 159` (LIT)

**Example 29** (doc_id: `54212`) (sent_id: `54212`)


Ist es dem Kläger im Rahmen seiner deshalb nötigen Ermittlungen aufgrund des Verhaltens des FG-Präsidenten nicht möglich , diesen Verfahrensmangel zu substantiieren , so hat dies allein zur Folge , dass der BFH insoweit einen geringeren Maßstab der Darlegung des Verfahrensmangels genügen lassen muss .

| Predicted | Gold |
|---|---|
| `BFH` | `BFH` |

**Example 30** (doc_id: `54257`) (sent_id: `54257`)


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

**Example 1** (doc_id: `53422`) (sent_id: `53422`)


Bei der vom FG zu beurteilenden gesetzlichen Prozessführungsbefugnis ( Beteiligtenstellung ) handelt es sich um eine Sachurteilsvoraussetzung , deren fehlerhafte Beurteilung einen Verfahrensmangel darstellt ( BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943 ; vom 23. November 2010 V B 133/09 , BFH / NV 2011 , 612 , Rz 5 ; jeweils m. w. N. ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`
- `BFH` — partial — pred is substring of gold: `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`
- `BFH` — similar text (different position): `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`

> overlaps gold: 3  |  likely missing annotation: 0

**Gold Entities:**

- `BFH-Beschlüsse vom 15. März 2002 VII B 120/01 , BFH / NV 2002 , 943`(RS)
- `vom 23. November 2010 V B 133/09 , BFH / NV 2011 , 612 , Rz 5`(RS)

**Example 2** (doc_id: `53446`) (sent_id: `53446`)


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

**Example 3** (doc_id: `53468`) (sent_id: `53468`)


Wie das vorliegende Verfahren zeigt , bietet die Anhörungsrüge die Möglichkeit , durch Zuordnungsschwierigkeiten entstehende Probleme zu bewältigen ( vgl. BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15 ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG , Beschluss der 1. Kammer des Zweiten Senats vom 12. Dezember 2012 - 2 BvR 1294/10 - , a. a. O. , Rn. 15`(RS)

**Example 4** (doc_id: `53474`) (sent_id: `53474`)


Beim Leasing ist - wie dargestellt - darauf abzustellen , ob der Herausgabeanspruch des Leasinggebers ( zivilrechtlichen Eigentümers ) noch eine wirtschaftliche Bedeutung hat ( grundlegend BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1. ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH-Urteil vom 26. Januar 1970 IV R 144/66 , BFHE 97 , 466 , BStBl II 1970 , 264 , unter C. III. 1.`(RS)

**Example 5** (doc_id: `53481`) (sent_id: `53481`)


Diese können insbesondere dann vorliegen , wenn die Gesamtstrafe sich nicht innerhalb des gesetzlichen Strafrahmens bewegt , die gebotene Begründung für die Gesamtstrafe fehlt , oder wenn die Besorgnis besteht , der Tatrichter habe sich von der Summe der Einzelstrafen leiten lassen ( BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , Beschluss vom 10. November 2016 - 1 StR 417/16 , juris`(RS)

**Example 6** (doc_id: `53485`) (sent_id: `53485`)


Soweit die Elemente eines Bildzeichens nur die typischen Merkmale der in Rede stehenden Waren und Dienstleistungen darstellen oder sich in einfachen dekorativen Gestaltungsmitteln erschöpfen , an die sich der Verkehr etwa durch häufige Verwendung gewöhnt hat , wird diesem Zeichen im Allgemeinen wegen seines bloß beschreibenden Inhalts ebenfalls die konkrete Eignung fehlen , die mit ihm gekennzeichneten Waren oder Dienstleistungen von denjenigen anderer Herkunft zu unterscheiden ( vgl. BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband ; GRUR 2005 , 257 , 258 – Bürogebäude ; GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel ; GRUR 2001 , 239f. – Zahnpastastrang ; GRUR 2001 , 734 , 735 – Westie-Kopf ; BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband`
- `BPatG` — partial — pred is substring of gold: `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH , GRUR 2011 , 158 Rn. 8 – Hefteinband`(RS)
- `GRUR 2005 , 257 , 258 – Bürogebäude`(RS)
- `GRUR 2004 , 683 , 684 – Farbige Arzneimittelkapsel`(RS)
- `GRUR 2001 , 239f. – Zahnpastastrang`(RS)
- `GRUR 2001 , 734 , 735 – Westie-Kopf`(RS)
- `BPatG , Beschluss vom 06. 08. 2015 25 W ( pat ) 14/14 – Abbildung eines bunten aus Stoffbahnen bestehenden Zeltes`(RS)

**Example 7** (doc_id: `53490`) (sent_id: `53490`)


Es müssen letztlich besondere Verhaltensweisen sowohl des Berechtigten als auch des Verpflichteten vorliegen , die es rechtfertigen , die späte Geltendmachung des Rechts als mit Treu und Glauben unvereinbar und für den Verpflichteten als unzumutbar anzusehen ( vgl. BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27 mwN ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 17. Oktober 2013 - 8 AZR 974/12 - Rn. 27`(RS)

**Example 8** (doc_id: `53508`) (sent_id: `53508`)


Beruhen vielmehr die Differenzen zwischen den Auffassungen von Sachverständigen darauf , dass diese von verschiedenen tatsächlichen Annahmen ausgehen , dann muss der Tatrichter , ggf nach weiterer Aufklärung , die für seine Überzeugungsbildung maßgebenden Tatsachen feststellen oder begründen , weshalb und zu wessen Lasten sie beweislos geblieben sind ( vgl BGH Urteil vom 23. 9. 1986 - VI ZR 261/85 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH Urteil vom 23. 9. 1986 - VI ZR 261/85`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH Urteil vom 23. 9. 1986 - VI ZR 261/85`(RS)

**Example 9** (doc_id: `53524`) (sent_id: `53524`)


Der BFH führte aus , die handelsrechtliche Zurechnung von Vermögensgegenständen entspreche im Wesentlichen der Regelung des § 39 Abs. 2 Nr. 1 Satz 1 AO ( z.B. BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1. ; vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a ) .

**False Positives:**

- `BFH` — similar text (different position): `BFH`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BFH`(ORG)
- `§ 39 Abs. 2 Nr. 1 Satz 1 AO`(NRM)
- `BFH-Urteile in BFHE 166 , 49 , BStBl II 1992 , 182 , unter 1.`(RS)
- `vom 14. Mai 2002 VIII R 30/98 , BFHE 199 , 181 , BStBl II 2002 , 741 , unter I. 1. a`(RS)

**Example 10** (doc_id: `53539`) (sent_id: `53539`)


cc ) Stellt ein Bewerber nicht innerhalb eines Monats nach Zugang der Abbruchmitteilung einen Antrag auf Erlass einer einstweiligen Verfügung , darf der Dienstherr nach der Rechtsprechung des Bundesverwaltungsgerichts darauf vertrauen , dass der Bewerber den Abbruch des Auswahlverfahrens nicht angreift , sondern sein Begehren im Rahmen einer neuen Ausschreibung weiterverfolgt ( vgl. BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14 ) .

**False Positives:**

- `BVerwG` — partial — pred is substring of gold: `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesverwaltungsgerichts`(ORG)
- `BVerwG 3. Dezember 2014 - 2 A 3.13 - Rn. 24 , BVerwGE 151 , 14`(RS)

**Example 11** (doc_id: `53551`) (sent_id: `53551`)


Die Beklagte wird das der Klägerin zustehende Honorar unter Zugrundelegung eines festen RLV für die gesamte BAG neu zu ermitteln haben .

**False Positives:**

- `BAG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 12** (doc_id: `53582`) (sent_id: `53582`)


3. Wird einem Beteiligten das rechtliche Gehör dadurch versagt , dass es ihm nicht ermöglicht wird , an der mündlichen Verhandlung teilzunehmen , so ist davon auszugehen , dass dies für eine aufgrund dieser Verhandlung ergangenen Entscheidung ursächlich geworden ist ( vgl BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7 ) .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Beschluss vom 26. 6. 2007 - B 2 U 55/07 B - SozR 4 - 1750 § 227 Nr 1 , RdNr 7`(RS)

**Example 13** (doc_id: `53583`) (sent_id: `53583`)


Denn die Hauptfunktion einer Marke liegt darin , die Ursprungsidentität der gekennzeichneten Waren und Dienstleistungen zu gewährleisten ( vgl. u. a. EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel ; BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006 ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel`
- `BGH` — partial — pred is substring of gold: `BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2004 , 428 Rn. 30 , 31 – Henkel`(RS)
- `BGH GRUR 2006 , 850 Rn. 17 – FUSSBALL WM 2006`(RS)

**Example 14** (doc_id: `53599`) (sent_id: `53599`)


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

**Example 15** (doc_id: `53600`) (sent_id: `53600`)


Auch danach setzt die Bejahung der Unterscheidungskraft unverändert voraus , dass das Zeichen geeignet sein muss , die beanspruchten Produkte als von einem bestimmten Unternehmen stammend zu kennzeichnen ( vgl. EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `EuGH GRUR 2010 , 228 Rn. 44 – VORSPRUNG DURCH TECHNIK`(RS)

**Example 16** (doc_id: `53614`) (sent_id: `53614`)


Soweit Personen dezentral untergebracht sind , ist es für die Bejahung einer Einrichtung erforderlich , dass die dezentrale Unterkunft zu den Räumlichkeiten der Einrichtung gehört , der Hilfebedürftige also in die Räumlichkeiten des Einrichtungsträgers eingegliedert ist ( vgl BSG SozR 4 - 3500 § 98 Nr 3 ) .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG SozR 4 - 3500 § 98 Nr 3`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG SozR 4 - 3500 § 98 Nr 3`(RS)

**Example 17** (doc_id: `53618`) (sent_id: `53618`)


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

**Example 18** (doc_id: `53623`) (sent_id: `53623`)


Nach der von der Einsprechenden zitierten BGH-Entscheidung „ Kommunikationskanal “ , muss für den Fachmann die im Anspruch bezeichnete Lehre „ unmittelbar und eindeutig “ als mögliche Ausführungsform der Erfindung den Ursprungsunterlagen entnehmbar sein .

**False Positives:**

- `BGH` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `53637`) (sent_id: `53637`)


aa ) Das Berufungsgericht ist zu der Einschätzung gelangt , die Klägerin habe nicht dargelegt , dass die Zulassung als Jaguar- und Land-Rover-Vertragswerkstatt eine Ressource darstelle , ohne die der Zugang zu dem nachgelagerten Endkundenmarkt nicht oder nicht sinnvoll möglich sei .

**False Positives:**

- `Jaguar` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 20** (doc_id: `53680`) (sent_id: `53680`)


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

**Example 21** (doc_id: `53688`) (sent_id: `53688`)


Allerdings beschränkt sich die Geltung des Grundsatzes der Bestenauslese im Bereich der Verwendungsentscheidungen auf Entscheidungen über - wie hier - höherwertige , die Beförderung in einen höheren Dienstgrad oder die Einweisung in die Planstelle einer höheren Besoldungsgruppe vorprägende Verwendungen ( vgl. klarstellend BVerwG , Beschluss vom 30. Januar 2014 - 1 WB 1.13 - juris Rn. 32 ) .

**False Positives:**

- `BVerwG` — partial — pred is substring of gold: `BVerwG , Beschluss vom 30. Januar 2014 - 1 WB 1.13 - juris Rn. 32`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerwG , Beschluss vom 30. Januar 2014 - 1 WB 1.13 - juris Rn. 32`(RS)

**Example 22** (doc_id: `53701`) (sent_id: `53701`)


Für eine solche Prognose des Arbeitgebers bedarf es ausreichend konkreter Anhaltspunkte ( BAG 27. Juli 2016 - 7 AZR 545/14 - Rn. 19 ; 24. September 2014 - 7 AZR 987/12 - Rn. 18 ; 7. Mai 2008 - 7 AZR 146/07 - Rn. 15 ; 7. April 2004 - 7 AZR 441/03 - zu II 2 a aa der Gründe ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 27. Juli 2016 - 7 AZR 545/14 - Rn. 19`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 27. Juli 2016 - 7 AZR 545/14 - Rn. 19`(RS)
- `24. September 2014 - 7 AZR 987/12 - Rn. 18`(RS)
- `7. Mai 2008 - 7 AZR 146/07 - Rn. 15`(RS)
- `7. April 2004 - 7 AZR 441/03 - zu II 2 a aa der Gründe`(RS)

**Example 23** (doc_id: `53707`) (sent_id: `53707`)


Da eine so weitgehende Selbstentäußerung des ausländischen Staates im Zweifel nicht zu vermuten ist ( BGH 30. Januar 2013 - III ZB 40/12 - Rn. 19 ) , dürfen die Umstände des Falles hinsichtlich des Vorliegens und der Reichweite eines Verzichts keinen Zweifel lassen ( BAG 29. Juni 2017 - 2 AZR 759/16 - Rn. 20 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH 30. Januar 2013 - III ZB 40/12 - Rn. 19`
- `BAG` — partial — pred is substring of gold: `BAG 29. Juni 2017 - 2 AZR 759/16 - Rn. 20`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH 30. Januar 2013 - III ZB 40/12 - Rn. 19`(RS)
- `BAG 29. Juni 2017 - 2 AZR 759/16 - Rn. 20`(RS)

**Example 24** (doc_id: `53717`) (sent_id: `53717`)


Bei Anerkennungsbeträgen handelt es sich um eine jener Massenerscheinungen , die ein typisierendes und pauschalierendes Vorgehen auch der Verwaltung rechtfertigen ( vgl. BVerfG , Beschluss vom 31. Mai 1988 - 1 BvR 520/83 - BVerfGE 78 , 214 < 227 > ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Beschluss vom 31. Mai 1988 - 1 BvR 520/83 - BVerfGE 78 , 214 < 227 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG , Beschluss vom 31. Mai 1988 - 1 BvR 520/83 - BVerfGE 78 , 214 < 227 >`(RS)

**Example 25** (doc_id: `53721`) (sent_id: `53721`)


Die Regelung ist daher geeignet , die Befristung von Arbeitsverträgen mit programmgestaltenden Mitarbeitern bei Rundfunkanstalten zu rechtfertigen ( BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101 ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 18. Mai 2016 - 7 AZR 533/14 - Rn. 18 , BAGE 155 , 101`(RS)

**Example 26** (doc_id: `53760`) (sent_id: `53760`)


Etwas anderes gilt insbesondere dann , wenn der Arbeitgeber seine Tarifgebundenheit in einer dem Arbeitnehmer hinreichend erkennbaren Weise zur auflösenden Bedingung der Bezugnahme gemacht hat ( BAG 18. April 2007 - 4 AZR 652/05 - Rn. 32 , 40 , BAGE 122 , 74 ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 18. April 2007 - 4 AZR 652/05 - Rn. 32 , 40 , BAGE 122 , 74`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 18. April 2007 - 4 AZR 652/05 - Rn. 32 , 40 , BAGE 122 , 74`(RS)

**Example 27** (doc_id: `53763`) (sent_id: `53763`)


Wird als Verfahrensmangel gerügt , das FG habe einen gestellten Beweisantrag übergangen ( Rüge mangelnder Sachaufklärung gemäß § 76 Abs. 1 Satz 1 FGO ) , so ist darzulegen , welche Tatfrage aufklärungsbedürftig ist , welche Beweismittel das FG zu welchem Beweisthema nicht erhoben hat , die genauen Fundstellen ( Schriftsatz mit Datum und Seitenzahl , Terminprotokoll ) , in denen die Beweismittel und Beweisthemen angeführt worden sind , das voraussichtliche Ergebnis der Beweisaufnahme , inwiefern das Urteil des FG aufgrund dessen sachlich-rechtlicher Auffassung auf der unterbliebenen Beweisaufnahme beruhen kann und dass die Nichterhebung der Beweise vor dem FG rechtzeitig gerügt worden ist oder aufgrund des Verhaltens des FG nicht mehr vor diesem gerügt werden konnte ( ständige BFH-Rechtsprechung , z.B. Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85 , und vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125 , m. w. N. ) .

**False Positives:**

- `BFH` — similar text (different position): `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`
- `BFH` — similar text (different position): `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `§ 76 Abs. 1 Satz 1 FGO`(NRM)
- `Beschlüsse vom 24. Juli 2002 V B 25/02 , BFHE 199 , 85`(RS)
- `vom 17. März 2000 VII B 1/00 , BFH / NV 2000 , 1125`(RS)

**Example 28** (doc_id: `53767`) (sent_id: `53767`)


Ob der Erwerb eigener Anteile auf der Gesellschaftsebene entsprechend der durch das BilMoG geänderten handelsrechtlichen Vorschriften ( Einfügung des § 272 Abs. 1a und 1b HGB ) steuerrechtlich nicht mehr als Erwerbsvorgang anzusehen , sondern nunmehr als " Teilliquidation " und daher " wie " eine Kapitalherabsetzung zu behandeln ist ( so BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f. ; zustimmend z.B. Blumenberg / Lechner , DB 2014 , 141 , 147 ; kritisch z.B. Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55 ) , kann im Streitfall mangels Erheblichkeit für die Entscheidung offenbleiben .

**False Positives:**

- `BMF` — partial — pred is substring of gold: `BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f.`
- `DB` — partial — pred is substring of gold: `Blumenberg / Lechner , DB 2014 , 141 , 147`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BilMoG`(NRM)
- `§ 272 Abs. 1a und 1b HGB`(NRM)
- `BMF-Schreiben in BStBl I 2013 , 1615 , Rz 8 f.`(REG)
- `Blumenberg / Lechner , DB 2014 , 141 , 147`(LIT)
- `Gosch in Kirchhof , a. a. O. , § 17 EStG Rz 55`(LIT)

**Example 29** (doc_id: `53775`) (sent_id: `53775`)


nicht nur das erstmalige Legen eines Hausanschlusses , sondern auch Arbeiten zur Erneuerung oder zur Reduzierung von Wasseranschlüssen unter die Steuerermäßigung fallen ( vgl. BGH-Urteil in HFR 2012 , 1110 , Rz 20 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH-Urteil in HFR 2012 , 1110 , Rz 20`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH-Urteil in HFR 2012 , 1110 , Rz 20`(RS)

**Example 30** (doc_id: `53811`) (sent_id: `53811`)


Diese Grundsätze gelten ebenso für die Anwendung der hergebrachten Grundsätze des Berufsbeamtentums im Sinne des Art. 33 Abs. 5 GG ( BVerfG , Kammerbeschluss vom 7. Oktober 2015 - 2 BvR 413/15 - NVwZ 2016 , 56 Rn. 24 f. m. w. N. ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Kammerbeschluss vom 7. Oktober 2015 - 2 BvR 413/15 - NVwZ 2016 , 56 Rn. 24 f.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 33 Abs. 5 GG`(NRM)
- `BVerfG , Kammerbeschluss vom 7. Oktober 2015 - 2 BvR 413/15 - NVwZ 2016 , 56 Rn. 24 f.`(RS)

**Example 31** (doc_id: `53820`) (sent_id: `53820`)


Wenn es - wie vorliegend - an einer ausdrücklichen Sonderzuweisung für den zuständigen Rechtsweg fehlt , bestimmt sich die gerichtliche Zuständigkeit nach der Natur des Rechtsverhältnisses , aus dem der Klageanspruch hergeleitet wird ( stRspr ; Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2 ; GmSOGB vom 10. 4. 1986 - GmS-OGB 1/85 - BGHZ 97 , 312 = SozR 1500 § 51 Nr 39 ; GmSOGB vom 29. 10. 1987 - GmS-OGB 1/86 - BGHZ 102 , 280 = SozR 1500 § 51 Nr 47 ; GmSOGB vom 10. 7. 1989 - GmS-OGB 1/88 - BGHZ 108 , 284 = SozR 1500 § 51 Nr 53 ; zum Rechtsverhältnis zwischen den Beteiligten als entscheidendes Kriterium zur Beurteilung des Rechtswegs vgl letztens etwa BSG vom 21. 7. 2014 - B 14 SF 1/14 R - SozR 4 - 1500 § 51 Nr 12 RdNr 8 ; BSG vom 25. 10. 2017 - B 7 SF 1/16 R - juris , RdNr 6 ) .

**False Positives:**

- `GmSOGB` — partial — pred is substring of gold: `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`
- `GmSOGB` — similar text (different position): `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`
- `GmSOGB` — similar text (different position): `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`
- `GmSOGB` — similar text (different position): `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`
- `BSG` — similar text (different position): `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`
- `BSG` — similar text (different position): `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`

> overlaps gold: 6  |  likely missing annotation: 0

**Gold Entities:**

- `Gemeinsamer Senat der obersten Gerichtshöfe des Bundes < GmSOGB > vom 4. 6. 1974 - GmS-OGB 2/73 - BSGE 37 , 292 = SozR 1500 § 51 Nr 2`(RS)
- `GmSOGB vom 10. 4. 1986 - GmS-OGB 1/85 - BGHZ 97 , 312 = SozR 1500 § 51 Nr 39`(RS)
- `GmSOGB vom 29. 10. 1987 - GmS-OGB 1/86 - BGHZ 102 , 280 = SozR 1500 § 51 Nr 47`(RS)
- `GmSOGB vom 10. 7. 1989 - GmS-OGB 1/88 - BGHZ 108 , 284 = SozR 1500 § 51 Nr 53`(RS)
- `BSG vom 21. 7. 2014 - B 14 SF 1/14 R - SozR 4 - 1500 § 51 Nr 12 RdNr 8`(RS)
- `BSG vom 25. 10. 2017 - B 7 SF 1/16 R - juris , RdNr 6`(RS)

**Example 32** (doc_id: `53822`) (sent_id: `53822`)


Dies folgt aus § 73b Abs 5 S 4 SGB V , der Abweichungen von den Vorschriften des Vierten Kapitels und damit auch von dem in § 71 Abs 1 S 1 SGB V verankerten Grundsatz der Beitragssatzstabilität zulässt ( BSG Urteil vom 25. 3. 2015 - B 6 KA 9/14 R - BSGE 118 , 164 = SozR 4 - 2500 § 73b Nr 1 , RdNr 72 ) .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG Urteil vom 25. 3. 2015 - B 6 KA 9/14 R - BSGE 118 , 164 = SozR 4 - 2500 § 73b Nr 1 , RdNr 72`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 73b Abs 5 S 4 SGB V`(NRM)
- `§ 71 Abs 1 S 1 SGB V`(NRM)
- `BSG Urteil vom 25. 3. 2015 - B 6 KA 9/14 R - BSGE 118 , 164 = SozR 4 - 2500 § 73b Nr 1 , RdNr 72`(RS)

**Example 33** (doc_id: `53843`) (sent_id: `53843`)


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

**Example 34** (doc_id: `53848`) (sent_id: `53848`)


Vielmehr setzt der Sinn und Zweck der Vorschrift voraus , dass auch das konkrete Verfahren von dem Sozialleistungsträger gerade in dieser Eigenschaft geführt wird ; das Verfahren muss also einen engen sachlichen Zusammenhang zu der gesetzlichen Tätigkeit als Träger der in der Vorschrift genannten Sozialleistungen haben ( BGH Beschluss vom 10. 11. 2005 - IX ZR 189/02 - Juris RdNr 6 f ; BGH Beschluss vom 28. 9. 2016 - XII ZB 251/16 - Juris RdNr 30 ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH Beschluss vom 10. 11. 2005 - IX ZR 189/02 - Juris RdNr 6 f`
- `BGH` — similar text (different position): `BGH Beschluss vom 10. 11. 2005 - IX ZR 189/02 - Juris RdNr 6 f`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BGH Beschluss vom 10. 11. 2005 - IX ZR 189/02 - Juris RdNr 6 f`(RS)
- `BGH Beschluss vom 28. 9. 2016 - XII ZB 251/16 - Juris RdNr 30`(RS)

**Example 35** (doc_id: `53854`) (sent_id: `53854`)


Die insoweit verfrüht erhobene Einrede entfaltet auch mit dem Ablauf der maßgeblichen Frist ( am 12. Juni 2014 ) nicht die Rechtswirkung einer zulässigen Einrede ( BPatG GRUR 2005 , 773 , 775 - Blue Bull / RED BULL ; Ströbele / Hacker , MarkenG , 11. Aufl. , § 43 Rn. 29 m. w. N. ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG GRUR 2005 , 773 , 775 - Blue Bull / RED BULL`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG GRUR 2005 , 773 , 775 - Blue Bull / RED BULL`(RS)
- `Ströbele / Hacker , MarkenG , 11. Aufl. , § 43 Rn. 29`(LIT)

**Example 36** (doc_id: `53855`) (sent_id: `53855`)


aa ) In organisatorischer Hinsicht ist beim Bundesverfassungsgericht , anders als bei den Fachgerichten , eine Kapazitätsausweitung zur Verkürzung der Verfahrensdauer als Reaktion auf gesteigerte Eingangszahlen grundsätzlich nicht möglich , da die Struktur des Gerichts durch seine Funktion bedingt und durch die Verfassung und das Bundesverfassungsgerichtsgesetz vorgegeben ist ( BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21 ; vgl. BTDrucks 17/3802 , S. 26 ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Bundesverfassungsgericht`(ORG)
- `Bundesverfassungsgerichtsgesetz`(NRM)
- `BVerfG , Beschluss der Beschwerdekammer vom 30. August 2016 - 2 BvC 26/14 - Vz 1/16 - , juris , Rn. 21`(RS)
- `BTDrucks 17/3802 , S. 26`(LIT)

**Example 37** (doc_id: `53879`) (sent_id: `53879`)


Bemüht sich jemand , der ein Statusfeststellungsverfahren einleitet , zeitnah um private Eigenvorsorge , so kann er diese für den Fall , dass das Statusfeststellungsverfahren entgegen seinen Vorstellungen zu einer Feststellung von Versicherungspflicht führt , möglicherweise gar nicht mehr oder nur mit erheblichem Aufwand rückabwickeln ( zu diesen Konsequenzen siehe LSG Berlin-Brandenburg Urteil vom 18. 9. 2013 - L 9 KR 384/11 - Juris RdNr 38 ) .

**False Positives:**

- `LSG Berlin-Brandenburg` — partial — pred is substring of gold: `LSG Berlin-Brandenburg Urteil vom 18. 9. 2013 - L 9 KR 384/11 - Juris RdNr 38`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `LSG Berlin-Brandenburg Urteil vom 18. 9. 2013 - L 9 KR 384/11 - Juris RdNr 38`(RS)

**Example 38** (doc_id: `53881`) (sent_id: `53881`)


Ob dies der Fall ist , richtet sich nach den Umständen des Einzelfalls , bei denen darauf abzustellen ist , wie das Hoheitszeichen im Rahmen der Designgestaltung konkret verwendet ist ( vgl. BPatG GRUR 2002 , 337 - Schlüsselanhänger ; Eichmann / von Falckenstein / Kühne , a. a. O. , § 3 Rdn. 26 ff. ) .

**False Positives:**

- `BPatG` — partial — pred is substring of gold: `BPatG GRUR 2002 , 337 - Schlüsselanhänger`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BPatG GRUR 2002 , 337 - Schlüsselanhänger`(RS)
- `Eichmann / von Falckenstein / Kühne , a. a. O. , § 3 Rdn. 26 ff.`(LIT)

**Example 39** (doc_id: `53883`) (sent_id: `53883`)


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

**Example 40** (doc_id: `53898`) (sent_id: `53898`)


Das gilt jedenfalls uneingeschränkt für das Elterngeld als fürsorgerische Leistung der Familienförderung , die über die bloße Sicherung des Existenzminimums hinausgeht ( zum Elterngeld vgl BVerfG < Kammer > Beschluss vom 9. 11. 2011 - 1 BvR 1853/11 - BVerfGK 19 , 186 ) .

**False Positives:**

- `BVerfG` — partial — pred is substring of gold: `BVerfG < Kammer > Beschluss vom 9. 11. 2011 - 1 BvR 1853/11 - BVerfGK 19 , 186`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerfG < Kammer > Beschluss vom 9. 11. 2011 - 1 BvR 1853/11 - BVerfGK 19 , 186`(RS)

**Example 41** (doc_id: `53910`) (sent_id: `53910`)


Zur Vermeidung einer mittelbaren Diskriminierung wegen Behinderung sei er nach der Entscheidung des Gerichtshofs der Europäischen Union ( EuGH ) vom 6. Dezember 2012 ( - C- 152/11 - [ Odar ] ) wie ein nicht schwerbehinderter Arbeitnehmer zu behandeln .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `Entscheidung des Gerichtshofs der Europäischen Union ( EuGH ) vom 6. Dezember 2012 ( - C- 152/11 - [ Odar ] )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Entscheidung des Gerichtshofs der Europäischen Union ( EuGH ) vom 6. Dezember 2012 ( - C- 152/11 - [ Odar ] )`(RS)

**Example 42** (doc_id: `53937`) (sent_id: `53937`)


Da allein das Fehlen jeglicher Unterscheidungskraft ein Eintragungshindernis begründet , ist ein großzügiger Maßstab anzulegen , so dass jede auch noch so geringe Unterscheidungskraft genügt , um das Schutzhindernis zu überwinden ( BGH a. a. O. – OUI ; a. a. O. – for you ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH a. a. O. – OUI`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH a. a. O. – OUI`(RS)
- `a. a. O. – for you`(RS)

**Example 43** (doc_id: `53938`) (sent_id: `53938`)


In dem besonderen Fall der Sanktionierung von Verstößen gegen die Verordnung [ … ] wurden jedoch die straf- oder bußgeldbewehrten Vorschriften der Verordnung [ … ] durch das Inkrafttreten der Sanktionsvorschriften vor dem Anwendungszeitpunkt der bewehrten EU-Verordnung bereits ab dem 2. Juli 2016 in Deutschland für anwendbar erklärt .

**False Positives:**

- `EU` — similar text (different position): `Deutschland`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Deutschland`(LOC)

**Example 44** (doc_id: `53981`) (sent_id: `53981`)


Dies ergibt sich z.B. aus der Begründung zum Entwurf der Bundesregierung eines Gesetzes zur Einführung einer Musterwiderrufsinformation für Verbraucherdarlehensverträge , zur Änderung der Vorschriften über das Widerrufsrecht bei Verbraucherdarlehensverträgen und zur Änderung des Darlehensvermittlungsrechts vom 29. 4. 2010 ( BT-Drs. 17/1394 , S. 15 ) .

**False Positives:**

- `Bundesregierung` — partial — pred is substring of gold: `Begründung zum Entwurf der Bundesregierung eines Gesetzes zur Einführung einer Musterwiderrufsinformation für Verbraucherdarlehensverträge , zur Änderung der Vorschriften über das Widerrufsrecht bei Verbraucherdarlehensverträgen und zur Änderung des Darlehensvermittlungsrechts vom 29. 4. 2010 ( BT-Drs. 17/1394 , S. 15 )`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Begründung zum Entwurf der Bundesregierung eines Gesetzes zur Einführung einer Musterwiderrufsinformation für Verbraucherdarlehensverträge , zur Änderung der Vorschriften über das Widerrufsrecht bei Verbraucherdarlehensverträgen und zur Änderung des Darlehensvermittlungsrechts vom 29. 4. 2010 ( BT-Drs. 17/1394 , S. 15 )`(LIT)

**Example 45** (doc_id: `53982`) (sent_id: `53982`)


Der Rundfunkbeitrag wird erhoben , um den individuellen Nutzungsvorteil abzugelten ( BVerwG , Urteil vom 18. März 2016 - 6 C 6.15 - BVerwGE 154 , 275 Rn. 25 ff. ) .

**False Positives:**

- `BVerwG` — partial — pred is substring of gold: `BVerwG , Urteil vom 18. März 2016 - 6 C 6.15 - BVerwGE 154 , 275 Rn. 25 ff.`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BVerwG , Urteil vom 18. März 2016 - 6 C 6.15 - BVerwGE 154 , 275 Rn. 25 ff.`(RS)

**Example 46** (doc_id: `53991`) (sent_id: `53991`)


Nicht das tatsächliche Verhalten des Arbeitgebers im Lohnsteuerabzugsverfahren bindet dessen Beteiligte ( vgl BSG Urteil vom 23. 5. 2017 - B 12 KR 6/16 R - SozR 4 - 5376 § 1 Nr 1 RdNr 23 ; BSG Urteil vom 26. 3. 2014 - B 10 EG 14/13 R - BSGE 115 , 198 = SozR 4 - 7837 § 2 Nr 25 , RdNr 26 f ) , wohl aber die Rechtsfolgen , die AO und EStG daran knüpfen .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG Urteil vom 23. 5. 2017 - B 12 KR 6/16 R - SozR 4 - 5376 § 1 Nr 1 RdNr 23`
- `BSG` — similar text (different position): `BSG Urteil vom 23. 5. 2017 - B 12 KR 6/16 R - SozR 4 - 5376 § 1 Nr 1 RdNr 23`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `BSG Urteil vom 23. 5. 2017 - B 12 KR 6/16 R - SozR 4 - 5376 § 1 Nr 1 RdNr 23`(RS)
- `BSG Urteil vom 26. 3. 2014 - B 10 EG 14/13 R - BSGE 115 , 198 = SozR 4 - 7837 § 2 Nr 25 , RdNr 26 f`(RS)
- `AO`(NRM)
- `EStG`(NRM)

**Example 47** (doc_id: `53995`) (sent_id: `53995`)


Sind beide Anträge - wie hier - Gegenstand desselben Rechtsstreits , kann über sie gleichzeitig verhandelt und entschieden werden ( vgl. BAG 27. April 2006 - 2 AZR 360/05 - Rn. 19 , aaO ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 27. April 2006 - 2 AZR 360/05 - Rn. 19 , aaO`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 27. April 2006 - 2 AZR 360/05 - Rn. 19 , aaO`(RS)

**Example 48** (doc_id: `54010`) (sent_id: `54010`)


Auf die Beschwerde der Anmelderin werden die Beschlüsse des Deutschen Patent- und Markenamts , Markenstelle für Klasse 41 , vom 3. Juli 2014 und vom 3. Dezember 2015 aufgehoben , soweit die Anmeldung in Bezug auf die nachfolgend genannten Dienstleistungen zurückgewiesen worden ist :

**False Positives:**

- `Deutschen Patent- und Markenamts` — partial — pred is substring of gold: `Deutschen Patent- und Markenamts , Markenstelle für Klasse 41`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Deutschen Patent- und Markenamts , Markenstelle für Klasse 41`(ORG)

**Example 49** (doc_id: `54027`) (sent_id: `54027`)


Dies gilt auch in den Fällen einer Erkrankung mit einer nur noch begrenzten Lebenserwartung , da die Regelung des § 64 Abs. 1 Satz 1 Nr. 1 EStDV i. d. F. des StVereinfG 2011 keine Differenzierung zwischen verschiedenen Krankheitskosten enthält ( BFH-Urteil vom 25. April 2017 VIII R 52/13 , BFHE 258 , 53 , BStBl II 2017 , 949 ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH-Urteil vom 25. April 2017 VIII R 52/13 , BFHE 258 , 53 , BStBl II 2017 , 949`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 64 Abs. 1 Satz 1 Nr. 1 EStDV`(NRM)
- `StVereinfG 2011`(NRM)
- `BFH-Urteil vom 25. April 2017 VIII R 52/13 , BFHE 258 , 53 , BStBl II 2017 , 949`(RS)

**Example 50** (doc_id: `54030`) (sent_id: `54030`)


Die vermeintliche Unrichtigkeit einer Entscheidung des Berufungsgerichts eröffnet aber nicht die Revisionsinstanz ( vgl BSG SozR 1500 § 160a Nr 7 ) .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG SozR 1500 § 160a Nr 7`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG SozR 1500 § 160a Nr 7`(RS)

**Example 51** (doc_id: `54041`) (sent_id: `54041`)


3.1 Zunächst hat der Gerichtshof im Hinblick auf die Auslegungskriterien zu Art. 3 ( a ) AMVO festgestellt , dass es unzulässig ist , ein ergänzendes Schutzzertifikat für solche Wirkstoffe zu erteilen , die in den Ansprüchen des Grundpatents nicht genannt sind ( EuGH , GRUR 2012 , 257 , Rnd. 28 – Medeva ) .

**False Positives:**

- `EuGH` — partial — pred is substring of gold: `EuGH , GRUR 2012 , 257 , Rnd. 28 – Medeva`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 3 ( a ) AMVO`(NRM)
- `EuGH , GRUR 2012 , 257 , Rnd. 28 – Medeva`(RS)

**Example 52** (doc_id: `54108`) (sent_id: `54108`)


Mit Beschluss vom 23. September 2015 hat die Patentabteilung des DPMA den Antrag zurückgewiesen .

**False Positives:**

- `DPMA` — partial — pred is substring of gold: `Patentabteilung des DPMA`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Patentabteilung des DPMA`(ORG)

**Example 53** (doc_id: `54121`) (sent_id: `54121`)


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

**Example 54** (doc_id: `54133`) (sent_id: `54133`)


aa ) Ergibt sich aus dem Vortrag der Parteien im Rechtsstreit , dass die normative Wirkung eines Tarifvertrags nach § 4 Abs. 1 , § 5 Abs. 4 TVG in Betracht kommt , muss das Gericht diese Normen nach § 293 ZPO von Amts wegen ermitteln ( BAG 25. Januar 2017 - 4 AZR 520/15 - Rn. 29 mwN ) .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 25. Januar 2017 - 4 AZR 520/15 - Rn. 29`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 4 Abs. 1 , § 5 Abs. 4 TVG`(NRM)
- `§ 293 ZPO`(NRM)
- `BAG 25. Januar 2017 - 4 AZR 520/15 - Rn. 29`(RS)

**Example 55** (doc_id: `54143`) (sent_id: `54143`)


Dieser während des Revisionsverfahrens eingetretene Zuständigkeitswechsel führt zu einem gesetzlichen Beteiligtenwechsel ( vgl. z.B. Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109 , m. w. N. ) .

**False Positives:**

- `Bundesfinanzhofs` — partial — pred is substring of gold: `Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109`
- `BFH` — partial — pred is substring of gold: `Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Bundesfinanzhofs - BFH - vom 22. August 2007 X R 2/04 , BFHE 218 , 533 , BStBl II 2008 , 109`(RS)

**Example 56** (doc_id: `54150`) (sent_id: `54150`)


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

**Example 57** (doc_id: `54160`) (sent_id: `54160`)


aa ) Die Jahressonderzuwendung hat - ähnlich wie die Jahressonderzahlung nach § 20 TV-L / TVöD ( vgl. dazu BAG 12. Dezember 2012 - 10 AZR 922/11 - Rn. 20 , BAGE 144 , 117 ) - mehrere erkennbare Zwecke .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 12. Dezember 2012 - 10 AZR 922/11 - Rn. 20 , BAGE 144 , 117`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 20 TV-L / TVöD`(REG)
- `BAG 12. Dezember 2012 - 10 AZR 922/11 - Rn. 20 , BAGE 144 , 117`(RS)

**Example 58** (doc_id: `54169`) (sent_id: `54169`)


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

**Example 59** (doc_id: `54179`) (sent_id: `54179`)


I. Mit dem angefochtenen Beschluss vom 4. November 2015 hat die Patentabteilung 43 des Deutschen Patent- und Markenamts das Patent 103 36 913 mit der Bezeichnung

**False Positives:**

- `Deutschen Patent- und Markenamts` — partial — pred is substring of gold: `Patentabteilung 43 des Deutschen Patent- und Markenamts`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Patentabteilung 43 des Deutschen Patent- und Markenamts`(ORG)

**Example 60** (doc_id: `54192`) (sent_id: `54192`)


- 100 mg Granulat zur Zubereitung oral einzunehmender Suspensionen unter der Nummer EU / 1 / 07 / 436 / 005 .

**False Positives:**

- `EU` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 61** (doc_id: `54202`) (sent_id: `54202`)


Eine Vertragslücke , die einer Schließung durch den Rückgriff auf dispositives Gesetzesrecht oder einer ergänzenden Vertragsauslegung bedurft hätte ( BAG 15. Dezember 2016 - 6 AZR 478/15 - Rn. 31 , BAGE 157 , 284 ) , bestand nicht .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 15. Dezember 2016 - 6 AZR 478/15 - Rn. 31 , BAGE 157 , 284`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 15. Dezember 2016 - 6 AZR 478/15 - Rn. 31 , BAGE 157 , 284`(RS)

**Example 62** (doc_id: `54208`) (sent_id: `54208`)


Hierdurch ist das Verfahren über das Ablehnungsgesuch abgeschlossen worden , denn erst zu diesem Zeitpunkt war das Gericht an seine Entscheidung gebunden ( vgl. auch Beschluss des Bundesfinanzhofs - BFH - vom 17. Juli 2008 I B 22/08 , juris , unter II. 2. zum Abschluss des Ablehnungsverfahrens im Zeitpunkt der Absendung der Entscheidung durch die Geschäftsstelle ) .

**False Positives:**

- `Bundesfinanzhofs` — partial — pred is substring of gold: `Beschluss des Bundesfinanzhofs - BFH - vom 17. Juli 2008 I B 22/08 , juris , unter II. 2.`
- `BFH` — partial — pred is substring of gold: `Beschluss des Bundesfinanzhofs - BFH - vom 17. Juli 2008 I B 22/08 , juris , unter II. 2.`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Beschluss des Bundesfinanzhofs - BFH - vom 17. Juli 2008 I B 22/08 , juris , unter II. 2.`(RS)

**Example 63** (doc_id: `54213`) (sent_id: `54213`)


( c ) Zu den Wertverhältnissen gehören nach der im Verfahren der Normenkontrolle grundsätzlich bindenden Auffassung der Fachgerichte schließlich auch Miet- und Belegungsbindungen aufgrund einer öffentlichen Förderung des Wohnungsbaus ( Vorlagebeschluss vom 17. Dezember 2014 - II R 14/13 - , juris , Rn. 15 in dem Verfahren 1 BvL 1/15 unter Bezugnahme auf die BFH-Urteile vom 26. Juli 1989 - II R 65/86 - , BFHE 158 , 87 , und vom 5. Mai 1993 - II R 71/90 - ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH-Urteile vom 26. Juli 1989 - II R 65/86 - , BFHE 158 , 87`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Vorlagebeschluss vom 17. Dezember 2014 - II R 14/13 - , juris , Rn. 15 in dem Verfahren 1 BvL 1/15`(RS)
- `BFH-Urteile vom 26. Juli 1989 - II R 65/86 - , BFHE 158 , 87`(RS)
- `vom 5. Mai 1993 - II R 71/90 -`(RS)

**Example 64** (doc_id: `54217`) (sent_id: `54217`)


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

**Example 65** (doc_id: `54228`) (sent_id: `54228`)


Da allein das Fehlen jeglicher Unterscheidungskraft ein Eintragungshindernis begründet , ist ein großzügiger Maßstab anzulegen , so dass jede auch noch so geringe Unterscheidungskraft genügt , um das Schutzhindernis zu überwinden ( BGH a. a. O. – OUI ; a. a. O. – for you ) .

**False Positives:**

- `BGH` — partial — pred is substring of gold: `BGH a. a. O. – OUI`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BGH a. a. O. – OUI`(RS)
- `a. a. O. – for you`(RS)

**Example 66** (doc_id: `54233`) (sent_id: `54233`)


Weder im deutschen AsylG noch in einem anderen deutschen Regelungswerk gebe es eine Norm , in der stehe oder aus der abgeleitet werden könne , die Gewährung subsidiären Schutzes in einem anderen EU-Mitgliedstaat stünde der Auslieferung der betroffenen Person durch die Bundesrepublik Deutschland entgegen .

**False Positives:**

- `EU` — similar text (different position): `Bundesrepublik Deutschland`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `AsylG`(NRM)
- `Bundesrepublik Deutschland`(LOC)

**Example 67** (doc_id: `54237`) (sent_id: `54237`)


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

**Example 68** (doc_id: `54255`) (sent_id: `54255`)


Die bundesrechtlich geforderte Zuweisung eines einheitlichen RLV an eine von mehreren Ärzten gebildete Arztpraxis ( BAG , MVZ ) hat zur Folge , dass innerhalb dieser Arztpraxis bei Beachtung der Fachgebietsgrenzen sowie qualifikationsgebundener Genehmigungen zur Leistungserbringung weitgehende Flexibilität herrscht .

**False Positives:**

- `BAG` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 69** (doc_id: `54274`) (sent_id: `54274`)


Jedenfalls aber kann in diesem Zusammenhang nicht auf die zur Kompetenzabgrenzung von Betriebsrat und Gesamtbetriebsrat maßgebenden Kriterien ( dazu BAG 3. Mai 2006 - 1 ABR 15/05 - BAGE 118 , 131 [ hauptsächlich zum Aufstellen eines Sozialplans ] ; 11. Dezember 2001 - 1 AZR 193/01 - BAGE 100 , 60 ; 8. Juni 1999 - 1 AZR 831/98 - BAGE 92 , 11 ; 24. Januar 1996 - 1 AZR 542/95 - BAGE 82 , 79 ; 17. Februar 1981 - 1 AZR 290/78 - BAGE 35 , 80 ) zurückgegriffen werden .

**False Positives:**

- `BAG` — partial — pred is substring of gold: `BAG 3. Mai 2006 - 1 ABR 15/05 - BAGE 118 , 131 [ hauptsächlich zum Aufstellen eines Sozialplans ]`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BAG 3. Mai 2006 - 1 ABR 15/05 - BAGE 118 , 131 [ hauptsächlich zum Aufstellen eines Sozialplans ]`(RS)
- `11. Dezember 2001 - 1 AZR 193/01 - BAGE 100 , 60`(RS)
- `8. Juni 1999 - 1 AZR 831/98 - BAGE 92 , 11`(RS)
- `24. Januar 1996 - 1 AZR 542/95 - BAGE 82 , 79`(RS)
- `17. Februar 1981 - 1 AZR 290/78 - BAGE 35 , 80`(RS)

**Example 70** (doc_id: `54276`) (sent_id: `54276`)


Nicht die Unrichtigkeit der Entscheidung im Einzelfall , sondern die Nichtübereinstimmung im Grundsätzlichen begründet die Zulassung der Revision wegen Divergenz ( stRspr ; vgl BSG SozR 1500 § 160a Nr 14 , 21 , 29 und 67 ) .

**False Positives:**

- `BSG` — partial — pred is substring of gold: `BSG SozR 1500 § 160a Nr 14 , 21 , 29 und 67`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `BSG SozR 1500 § 160a Nr 14 , 21 , 29 und 67`(RS)

**Example 71** (doc_id: `54314`) (sent_id: `54314`)


e ) Nur ausnahmsweise kann auf einen früheren Zeitpunkt abgestellt werden , etwa wenn die Eröffnung des Konkurs- oder Insolvenzverfahrens mangels Masse abgelehnt worden ist ( BFH-Urteil in BFHE 194 , 108 , BStBl II 2001 , 385 ; BFH-Beschlüsse vom 27. November 1995 VIII B 16/95 , BFH / NV 1996 , 406 ; vom 4. Oktober 2007 VIII S 3/07 ( PKH ) , BFH / NV 2008 , 209 ) oder wenn aus anderen Gründen feststeht , dass die Gesellschaft bereits im Zeitpunkt eines Auflösungsbeschlusses vermögenslos war ( BFH-Urteil vom 4. November 1997 VIII R 18/94 , BFHE 184 , 374 , BStBl II 1999 , 344 ) .

**False Positives:**

- `BFH` — partial — pred is substring of gold: `BFH-Urteil in BFHE 194 , 108 , BStBl II 2001 , 385`
- `BFH` — similar text (different position): `BFH-Urteil in BFHE 194 , 108 , BStBl II 2001 , 385`
- `BFH` — similar text (different position): `BFH-Urteil in BFHE 194 , 108 , BStBl II 2001 , 385`
- `BFH` — similar text (different position): `BFH-Urteil in BFHE 194 , 108 , BStBl II 2001 , 385`
- `BFH` — similar text (different position): `BFH-Urteil in BFHE 194 , 108 , BStBl II 2001 , 385`

> overlaps gold: 5  |  likely missing annotation: 0

**Gold Entities:**

- `BFH-Urteil in BFHE 194 , 108 , BStBl II 2001 , 385`(RS)
- `BFH-Beschlüsse vom 27. November 1995 VIII B 16/95 , BFH / NV 1996 , 406`(RS)
- `vom 4. Oktober 2007 VIII S 3/07 ( PKH ) , BFH / NV 2008 , 209`(RS)
- `BFH-Urteil vom 4. November 1997 VIII R 18/94 , BFHE 184 , 374 , BStBl II 1999 , 344`(RS)

</details>

---

## `Court with Location Genitive`

**F1:** 0.049 | **Precision:** 0.467 | **Recall:** 0.026  

**Format:** `regex`  
**Rule ID:** `e294a159`  
**Description:**
Matches court names in genitive case (e.g., 'des Landgerichts') but extracts only the court name, handling compound state names correctly (e.g., 'Sachsen-Anhalt', 'Nordrhein-Westfalen').

**Content:**
```
(?<=\s(?:des|der|dem|die|den)\s)(Landgerichts|Oberlandesgerichts|Bundesgerichtshofs|Bundesverwaltungsgerichts|Bundesfinanzhofs|Bundessozialgerichts|Landessozialgerichts|Verwaltungsgerichts|Finanzgerichts|Arbeitsgerichts|Amtsgerichts|Sozialgerichts|Gerichtshofs|Kammer|Amt|Dienst|Beh\u00f6rde|Ministeriums|Amtes|Bundeswehr|Bundesagentur|Staatsanwaltschaft|Landratsamt|Generalstaatsanwaltschaft|Finanzamt|Klinik|Krankenhaus|Firma|Unternehmen|Vereinigung|Verband|Kanzlei|Kammer|Senat|Abteilung|Stelle|Justizvollzugsanstalt|Patentabteilung|Markenstelle)\s+(?:[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+(?:\s+-\s+[A-Z][a-z\u00e4\u00f6\u00fc\u00df]+)?(?:\s+am\s+Main|\s+am\s+Neckar|\s+in\s+der\s+Freien\s+und\s+Hansestadt\s+Hamburg|\s+Zweibr\u00fccken|\s+Duisburg|\s+Wiesbaden|\s+Dresden|\s+Braunschweig|\s+Sachsen-Anhalt|\s+Berlin-Brandenburg|\s+Berlin|\s+Frankfurt\s+am\s+Main|\s+H\u00f6chst|\s+D\u00fcsseldorf|\s+M\u00fcnchen|\s+Pfalz|\s+Saarl\u00e4ndischen|\s+Mecklenburg-Vorpommern|\s+Rheinland-Pfalz|\s+Nordrhein-Westfalen|\s+Offenburg|\s+K\.|\s+M\.|\s+O\.|\s+D\.|\s+K\s+\u2026|\s+M\s+\u2026|\s+O\s+\u2026|\s+D\s+\u2026|\s+K\s+\u2026\s+GmbH|\s+M\s+\u2026\s+GmbH|\s+O\s+\u2026\s+GmbH|\s+D\s+\u2026\s+GmbH|\s+K\s+\u2026\s+AG|\s+M\s+\u2026\s+AG|\s+O\s+\u2026\s+AG|\s+D\s+\u2026\s+AG|\s+K\s+\u2026\s+mbH|\s+M\s+\u2026\s+mbH|\s+O\s+\u2026\s+mbH|\s+D\s+\u2026\s+mbH)?)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.467 | 0.026 | 0.049 | 45 | 21 | 24 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 21 | 24 | 742 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53725`) (sent_id: `53725`)


4. Auf die sofortige Beschwerde der Staatsanwaltschaft Düsseldorf hob das Oberlandesgericht Hamm mit angegriffenem Beschluss vom 28. Juli 2015 den Beschluss des Landgerichts Paderborn auf und ordnete die weitere Vollstreckung der Unterbringung des Beschwerdeführers in der Sicherungsverwahrung an .

| Predicted | Gold |
|---|---|
| `Staatsanwaltschaft Düsseldorf` | `Staatsanwaltschaft Düsseldorf` |
| `Landgerichts Paderborn` | `Landgerichts Paderborn` |

**Missed by this rule (FN):**

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

**Example 4** (doc_id: `55533`) (sent_id: `55533`)


Auf die Revision des Angeklagten wird das Urteil des Landgerichts Berlin vom 27. Juni 2017 mit den Feststellungen aufgehoben .

| Predicted | Gold |
|---|---|
| `Landgerichts Berlin` | `Landgerichts Berlin` |

**Example 5** (doc_id: `55722`) (sent_id: `55722`)


Vielmehr hat sich das LSG die Feststellungen aus dem Strafurteil nach eigener Prüfung zu eigen gemacht und dabei neben den Feststellungen aus dem Strafurteil nicht nur den Inhalt des Erstattungsbescheides aus dem Jahr 2013 zur Honorarrückforderung in Höhe von 216 492,33 Euro berücksichtigt , sondern auch den Inhalt der Strafakten ausgewertet , aus denen hervorgeht , dass sich das Urteil des Amtsgerichts Fürth nicht allein auf das Geständnis des Klägers stützt , sondern auch auf das Ergebnis einer umfangreichen Beweisaufnahme .

| Predicted | Gold |
|---|---|
| `Amtsgerichts Fürth` | `Amtsgerichts Fürth` |

**Example 6** (doc_id: `56530`) (sent_id: `56530`)


Die Beklagte beantragt , die Urteile des Bayerischen Landessozialgerichts vom 16. 12. 2015 und des Sozialgerichts München vom 16. 5. 2014 aufzuheben und die Klage abzuweisen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts München` | `Sozialgerichts München` |

**Missed by this rule (FN):**

- `Bayerischen Landessozialgerichts` (ORG)

**Example 7** (doc_id: `56596`) (sent_id: `56596`)


Hierfür wären die Urteile des Landgerichts Cottbus vom ... und des Brandenburgischen Oberlandesgerichts vom ... zu beachten , wonach der Kläger durch Kündigung vom ... 2005 wirksam aus der A-GbR ausgeschlossen wurde .

| Predicted | Gold |
|---|---|
| `Landgerichts Cottbus` | `Landgerichts Cottbus` |

**Missed by this rule (FN):**

- `Brandenburgischen Oberlandesgerichts` (ORG)
- `A-GbR` (ORG)

**Example 8** (doc_id: `56616`) (sent_id: `56616`)


Die Beklagte beantragt , das Urteil des Bayerischen Landessozialgerichts vom 21. Oktober 2015 aufzuheben und das Urteil des Sozialgerichts Nürnberg vom 25. Juni 2013 bezüglich der Beigeladenen zu 1. und 3. aufzuheben und die Klage abzuweisen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts Nürnberg` | `Sozialgerichts Nürnberg` |

**Missed by this rule (FN):**

- `Bayerischen Landessozialgerichts` (ORG)

**Example 9** (doc_id: `56951`) (sent_id: `56951`)


Auf die Revision des Angeklagten wird das Urteil des Landgerichts Ravensburg vom 1. August 2017 , soweit es ihn betrifft , mit den Feststellungen aufgehoben .

| Predicted | Gold |
|---|---|
| `Landgerichts Ravensburg` | `Landgerichts Ravensburg` |

**Example 10** (doc_id: `57130`) (sent_id: `57130`)


1. Auf die Revision des Angeklagten gegen das Urteil des Landgerichts Aachen vom 7. Juli 2016 wird

| Predicted | Gold |
|---|---|
| `Landgerichts Aachen` | `Landgerichts Aachen` |

**Example 11** (doc_id: `57570`) (sent_id: `57570`)


Die Klägerin beantragt , den Beschluss des Thüringer Landessozialgerichts vom 21. Juli 2016 und das Urteil des Sozialgerichts Meiningen vom 7. Januar 2015 aufzuheben sowie den Bescheid der Beklagten vom 15. April 2013 in der Gestalt des Widerspruchsbescheids vom 17. Mai 2013 abzuändern und die Beklagte zu verurteilen , ihr für die Zeit vom 1. Januar bis 28. März 2012 höheres Insolvenzgeld zu zahlen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts Meiningen` | `Sozialgerichts Meiningen` |

**Missed by this rule (FN):**

- `Thüringer Landessozialgerichts` (ORG)

**Example 12** (doc_id: `58013`) (sent_id: `58013`)


das Urteil des Bayerischen Landessozialgerichts vom 3. Juni 2016 und den Gerichtsbescheid des Sozialgerichts Nürnberg vom 30. August 2013 sowie den Bescheid der Beklagten vom 19. März 2012 und den Widerspruchsbescheid vom 18. Juni 2012 aufzuheben und die Beklagte zu verurteilen , unter Rücknahme des Bescheides vom 4. Februar 2005 und des Widerspruchsbescheides vom 22. April 2005 die Zeit vom 1. September 1973 bis 30. Juni 1990 als Zeit der Zugehörigkeit zum Zusatzversorgungssystem der technischen Intelligenz und die hierin erzielten Arbeitsentgelte festzustellen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts Nürnberg` | `Sozialgerichts Nürnberg` |

**Missed by this rule (FN):**

- `Bayerischen Landessozialgerichts` (ORG)

**Example 13** (doc_id: `58301`) (sent_id: `58301`)


Der Beklagte beantragt , das Urteil des Sächsischen Landessozialgerichts vom 9. Februar 2017 aufzuheben und die Berufungen der Kläger gegen das Urteil des Sozialgerichts Dresden vom 10. Februar 2014 zurückzuweisen .

| Predicted | Gold |
|---|---|
| `Sozialgerichts Dresden` | `Sozialgerichts Dresden` |

**Missed by this rule (FN):**

- `Sächsischen Landessozialgerichts` (ORG)

**Example 14** (doc_id: `58405`) (sent_id: `58405`)


3. Mit Schriftsatz vom 8. März 2018 beantragt die Beschwerdeführerin durch ihren Bevollmächtigten , " die Vollstreckbarkeit " der Beschlüsse des Landgerichts Potsdam vom 11. März 2014 und vom " 20. Juli 2017 " ( gemeint wohl 17. Juli 2017 ) vorläufig auszusetzen .

| Predicted | Gold |
|---|---|
| `Landgerichts Potsdam` | `Landgerichts Potsdam` |

**Example 15** (doc_id: `59356`) (sent_id: `59356`)


Über die sofortige Beschwerde der Nebenklägerin K. T. gegen die im Urteil des Landgerichts Frankfurt am Main vom 15. August 2017 getroffene Kostenentscheidung hat das Oberlandesgericht Frankfurt am Main zu entscheiden .

| Predicted | Gold |
|---|---|
| `Landgerichts Frankfurt am Main` | `Landgerichts Frankfurt am Main` |

**Missed by this rule (FN):**

- `K. T.` (PER)
- `Oberlandesgericht Frankfurt am Main` (ORG)

**Example 16** (doc_id: `59568`) (sent_id: `59568`)


Die Revision des Angeklagten gegen das Urteil des Landgerichts Essen vom 10. April 2017 wird als unbegründet verworfen , da die Nachprüfung des Urteils auf Grund der Revisionsrechtfertigung keinen Rechtsfehler zum Nachteil des Angeklagten ergeben hat ( § 349 Abs. 2 StPO ) .

| Predicted | Gold |
|---|---|
| `Landgerichts Essen` | `Landgerichts Essen` |

**Missed by this rule (FN):**

- `§ 349 Abs. 2 StPO` (NRM)

**Example 17** (doc_id: `59658`) (sent_id: `59658`)


A. I. 1. Der Beschwerdeführer ist auf Grundlage des Urteils des Landgerichts Lübeck vom 7. Oktober 2014 gemäß § 63 StGB wegen Mordes in einem psychiatrischen Krankenhaus der AMEOS Krankenhausgesellschaft Holstein mbH untergebracht , nachdem er im schuldunfähigen Zustand auf Grund einer wahnhaften Störung im Januar 2014 seine vierjährige Tochter und seinen sechs Jahre alten Sohn getötet hatte .

| Predicted | Gold |
|---|---|
| `Landgerichts Lübeck` | `Landgerichts Lübeck` |

**Missed by this rule (FN):**

- `§ 63 StGB` (NRM)
- `AMEOS Krankenhausgesellschaft Holstein mbH` (ORG)

**Example 18** (doc_id: `59742`) (sent_id: `59742`)


1. Dem Angeklagten A. wird auf seinen Antrag Wiedereinsetzung in den vorigen Stand wegen Versäumung der Frist zur Begründung der Revision gegen das Urteil des Landgerichts Göttingen vom 30. Mai 2017 gewährt .

| Predicted | Gold |
|---|---|
| `Landgerichts Göttingen` | `Landgerichts Göttingen` |

**Missed by this rule (FN):**

- `A.` (PER)

**Example 19** (doc_id: `59823`) (sent_id: `59823`)


Erstens würden anderen Gefangenen in vergleichbaren Situationen vollzugsöffnende Maßnahmen gewährt , und zweitens habe das Oberlandesgericht in einem anderen Verfahren vertreten , dass auch die Justizvollzugsanstalt Bruchsal Möglichkeiten der Diagnose vorhalten müsse und der Grundsatz der bestmöglichen Sachaufklärung die Einholung gutachterlicher Expertise gebiete .

| Predicted | Gold |
|---|---|
| `Justizvollzugsanstalt Bruchsal` | `Justizvollzugsanstalt Bruchsal` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `54385`) (sent_id: `54385`)


die Urteile des Landessozialgerichts Sachsen-Anhalt vom 9. März 2017 und des Sozialgerichts Dessau-Roßlau vom 2. Dezember 2013 sowie den Bescheid des Beklagten vom 16. Februar 2010 in der Gestalt des Widerspruchsbescheids vom 31. Mai 2010 aufzuheben .

**False Positives:**

- `Landessozialgerichts Sachsen` — partial — pred is substring of gold: `Landessozialgerichts Sachsen-Anhalt`
- `Sozialgerichts Dessau` — partial — pred is substring of gold: `Sozialgerichts Dessau-Roßlau`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Landessozialgerichts Sachsen-Anhalt`(ORG)
- `Sozialgerichts Dessau-Roßlau`(ORG)

**Example 1** (doc_id: `54438`) (sent_id: `54438`)


Der Haftbefehl des Amtsgerichts Neu-Ulm vom 31. Juli 2017 könne unter Berücksichtigung der Bedeutung und Tragweite des Grundrechts auf Freiheit der Person des Beschwerdeführers keinen Bestand haben .

**False Positives:**

- `Amtsgerichts Neu` — partial — pred is substring of gold: `Amtsgerichts Neu-Ulm`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Amtsgerichts Neu-Ulm`(ORG)

**Example 2** (doc_id: `55082`) (sent_id: `55082`)


Demgemäß hat der Senat Schüler , die im häuslichen Bereich unterrichtsvorbereitend ein Werkstück erstellen ( BSG vom 1. 2. 1979 - 2 RU 107/77 - SozR 2200 § 539 Nr 54 ) , ebenso wenig für versichert erachtet wie solche , die für die schulische Foto-AG in der Altstadt ohne weitere Aufsicht fotografieren ( BSG vom 30. 5. 1988 - 2 RU 5/88 - Juris ) .

**False Positives:**

- `Senat Schüler` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG vom 1. 2. 1979 - 2 RU 107/77 - SozR 2200 § 539 Nr 54`(RS)
- `Foto-AG`(ORG)
- `BSG vom 30. 5. 1988 - 2 RU 5/88 - Juris`(RS)

**Example 3** (doc_id: `55122`) (sent_id: `55122`)


So bietet beispielsweise die Firma Pointer „ Wohlfühlfarben für die Wohnung “ an ; in einem der Anmelderin übersandten Internetausdruck heißt es hierzu : „ Farben gezielt einsetzen .

**False Positives:**

- `Firma Pointer` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `55265`) (sent_id: `55265`)


Die Revision des Klägers gegen das Urteil des Landessozialgerichts Rheinland-Pfalz vom 9. Juni 2016 wird zurückgewiesen .

**False Positives:**

- `Landessozialgerichts Rheinland` — partial — pred is substring of gold: `Landessozialgerichts Rheinland-Pfalz`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Landessozialgerichts Rheinland-Pfalz`(ORG)

**Example 5** (doc_id: `55511`) (sent_id: `55511`)


Die Beschwerde der Antragstellerin gegen den Beschluss des Finanzgerichts Hamburg vom 15. August 2016 1 V 41/16 wird als unbegründet zurückgewiesen .

**False Positives:**

- `Finanzgerichts Hamburg` — partial — pred is substring of gold: `Beschluss des Finanzgerichts Hamburg vom 15. August 2016 1 V 41/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Beschluss des Finanzgerichts Hamburg vom 15. August 2016 1 V 41/16`(RS)

**Example 6** (doc_id: `55659`) (sent_id: `55659`)


Die Beschwerde des Klägers wegen Nichtzulassung der Revision gegen das Urteil des Finanzgerichts Münster vom 12. September 2017 15 K 3562/14 U wird als unbegründet zurückgewiesen .

**False Positives:**

- `Finanzgerichts Münster` — partial — pred is substring of gold: `Urteil des Finanzgerichts Münster vom 12. September 2017 15 K 3562/14 U`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Finanzgerichts Münster vom 12. September 2017 15 K 3562/14 U`(RS)

**Example 7** (doc_id: `55764`) (sent_id: `55764`)


Schließlich vertrieb die Firma Köhnlein am 10. November 2009 über das Internet ein „ Drei-Bolzen-Sicherheitsautomatikschloss mit A-Öffner “ , das sich durch das automatische Öffnen mit dem Komfort der automatischen Verriegelung auszeichnet ( Anlagen 8a und 8c ) .

**False Positives:**

- `Firma Köhnlein` — partial — gold is substring of pred: `Köhnlein`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Köhnlein`(ORG)

**Example 8** (doc_id: `56170`) (sent_id: `56170`)


2. Die Berufung des Beklagten gegen das Urteil des Arbeitsgerichts Oberhausen vom 15. Dezember 2016 - 4 Ca 1318/16 - wird zurückgewiesen .

**False Positives:**

- `Arbeitsgerichts Oberhausen` — partial — pred is substring of gold: `Urteil des Arbeitsgerichts Oberhausen vom 15. Dezember 2016 - 4 Ca 1318/16 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Arbeitsgerichts Oberhausen vom 15. Dezember 2016 - 4 Ca 1318/16 -`(RS)

**Example 9** (doc_id: `56331`) (sent_id: `56331`)


Auf die Berufung der Beklagten wird - unter Zurückweisung der Anschlussberufung des Klägers - das Urteil des Arbeitsgerichts Bonn vom 11. November 2015 - 4 Ca 1615/15 - abgeändert und die Klage abgewiesen .

**False Positives:**

- `Arbeitsgerichts Bonn` — partial — pred is substring of gold: `Urteil des Arbeitsgerichts Bonn vom 11. November 2015 - 4 Ca 1615/15 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Arbeitsgerichts Bonn vom 11. November 2015 - 4 Ca 1615/15 -`(RS)

**Example 10** (doc_id: `56780`) (sent_id: `56780`)


Die Revision der Beklagten gegen das Urteil des Finanzgerichts Berlin-Brandenburg vom 18. Januar 2017 3 K 3219/16 wird als unbegründet zurückgewiesen .

**False Positives:**

- `Finanzgerichts Berlin` — partial — pred is substring of gold: `Urteil des Finanzgerichts Berlin-Brandenburg vom 18. Januar 2017 3 K 3219/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Finanzgerichts Berlin-Brandenburg vom 18. Januar 2017 3 K 3219/16`(RS)

**Example 11** (doc_id: `56966`) (sent_id: `56966`)


Dies betrifft sowohl die angeordneten Tätigkeiten in der Abteilung Standesamt und Gerichtliche Angelegenheiten als auch diejenigen für die Visaabteilung .

**False Positives:**

- `Abteilung Standesamt` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 12** (doc_id: `57051`) (sent_id: `57051`)


Die Beschwerde der Klägerin gegen die Nichtzulassung der Revision im Urteil des Landessozialgerichts Nordrhein-Westfalen vom 22. Juni 2017 wird zurückgewiesen .

**False Positives:**

- `Landessozialgerichts Nordrhein` — partial — pred is substring of gold: `Landessozialgerichts Nordrhein-Westfalen`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Landessozialgerichts Nordrhein-Westfalen`(ORG)

**Example 13** (doc_id: `57287`) (sent_id: `57287`)


Die Beschwerde des Klägers gegen die Nichtzulassung der Revision in dem Beschluss des Landessozialgerichts Niedersachsen-Bremen vom 9. November 2017 wird als unzulässig verworfen .

**False Positives:**

- `Landessozialgerichts Niedersachsen` — partial — pred is substring of gold: `Landessozialgerichts Niedersachsen-Bremen`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Landessozialgerichts Niedersachsen-Bremen`(ORG)

**Example 14** (doc_id: `57361`) (sent_id: `57361`)


Auf die Revision der Beklagten wird das Urteil des Landessozialgerichts Nordrhein-Westfalen vom 17. Dezember 2014 aufgehoben , soweit das Bestehen von Rentenversicherungspflicht des Klägers wegen Beschäftigung bei der Beigeladenen zu 1. für die Zeit ab 10. Juli 2008 verneint wird .

**False Positives:**

- `Landessozialgerichts Nordrhein` — partial — pred is substring of gold: `Landessozialgerichts Nordrhein-Westfalen`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Landessozialgerichts Nordrhein-Westfalen`(ORG)

**Example 15** (doc_id: `57953`) (sent_id: `57953`)


2. Die Berufung des Klägers gegen das Urteil des Arbeitsgerichts Dortmund vom 27. März 2014 - 6 Ca 3695/11 - wird zurückgewiesen .

**False Positives:**

- `Arbeitsgerichts Dortmund` — partial — pred is substring of gold: `Urteil des Arbeitsgerichts Dortmund vom 27. März 2014 - 6 Ca 3695/11 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Arbeitsgerichts Dortmund vom 27. März 2014 - 6 Ca 3695/11 -`(RS)

**Example 16** (doc_id: `58297`) (sent_id: `58297`)


In Bezug auf den gerügten Haftbefehl des Amtsgerichts Neu-Ulm vom 31. Juli 2017 sei die Verfassungsbeschwerde wegen des Grundsatzes der Subsidiarität der Verfassungsbeschwerde hingegen unzulässig , da eine abschließende Sachprüfung durch das Oberlandesgericht München noch nicht stattgefunden habe .

**False Positives:**

- `Amtsgerichts Neu` — partial — pred is substring of gold: `Amtsgerichts Neu-Ulm`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Amtsgerichts Neu-Ulm`(ORG)
- `Oberlandesgericht München`(ORG)

**Example 17** (doc_id: `58399`) (sent_id: `58399`)


Die Revision der Klägerin gegen das Urteil des Finanzgerichts Hamburg vom 21. April 2017 4 K 186/16 wird als unbegründet zurückgewiesen .

**False Positives:**

- `Finanzgerichts Hamburg` — partial — pred is substring of gold: `Urteil des Finanzgerichts Hamburg vom 21. April 2017 4 K 186/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Finanzgerichts Hamburg vom 21. April 2017 4 K 186/16`(RS)

**Example 18** (doc_id: `58546`) (sent_id: `58546`)


Auf die Revision des Beklagten wird das Urteil des Finanzgerichts München vom 29. März 2017 3 K 2565/16 aufgehoben .

**False Positives:**

- `Finanzgerichts München` — partial — pred is substring of gold: `Urteil des Finanzgerichts München vom 29. März 2017 3 K 2565/16`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil des Finanzgerichts München vom 29. März 2017 3 K 2565/16`(RS)

**Example 19** (doc_id: `58915`) (sent_id: `58915`)


Die Beschwerde gegen die Nichtzulassung der Revision in dem Urteil des 7. Zivilsenats des Oberlandesgerichts Frankfurt am Main vom 8. Juni 2017 wird auf Kosten des Klägers als unzulässig verworfen .

**False Positives:**

- `Oberlandesgerichts Frankfurt am Main` — partial — pred is substring of gold: `Urteil des 7. Zivilsenats des Oberlandesgerichts Frankfurt am Main`

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

**Example 21** (doc_id: `59091`) (sent_id: `59091`)


Auf die Revision des Klägers wird der Beschluss des Landessozialgerichts Baden-Württemberg vom 9. Februar 2015 aufgehoben und die Sache zur erneuten Verhandlung und Entscheidung an dieses Gericht zurückverwiesen .

**False Positives:**

- `Landessozialgerichts Baden` — partial — pred is substring of gold: `Landessozialgerichts Baden-Württemberg`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Landessozialgerichts Baden-Württemberg`(ORG)

**Example 22** (doc_id: `59490`) (sent_id: `59490`)


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

## `Generic Organization Patterns`

**F1:** 0.009 | **Precision:** 0.103 | **Recall:** 0.005  

**Format:** `regex`  
**Rule ID:** `70b265e1`  
**Description:**
Matches generic organization patterns like 'Firma X GmbH', 'Senat des Y', 'Bundesministerium', 'Landesamt', etc., using structural patterns to generalize.

**Content:**
```
\b(?:Firma|Gesellschaft|AG|GmbH|KG|e\.\s*V\.|Verband|Vereinigung|Dienst|Amt|Beh\u00f6rde|Ministerium|Klinik|Krankenhaus|Schule|Schulzentrum|Senat|Kammer|Abteilung|Stelle|Gericht|Landesgericht|Oberlandesgericht|Bundesgericht|Finanzgericht|Arbeitsgericht|Sozialgericht|Verwaltungsgericht|Amtsgericht|Staatsanwaltschaft|Landratsamt|Post|Botschaft|Konsulat|Kanzlei|Korporation|Konzern|Gruppe|Fonds|Institut|Akademie|Hochschule|Universität|Bundesagentur|Bundesamt|Landesamt|Staat|Republik|Union|Kommission|Parlament|Bundestag|Landtag|Senat|Kabinett|Regierung|Verwaltung|Organisation|Institution|Behörde|Dienstleistung|Gewerkschaft|Arbeitsgeberverband|Arbeitnehmerverband|Krankenkasse|Rentenversicherung|Sozialversicherung|Versicherung|Bank|Finanzinstitut|Holding|Investment|Fonds|Stiftung|Organisation|Institution|Behörde|Dienstleistung|Gewerkschaft|Arbeitsgeberverband|Arbeitnehmerverband|Krankenkasse|Rentenversicherung|Sozialversicherung|Versicherung|Bank|Finanzinstitut|Holding|Investment|Fonds|Stiftung)\s+(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|\d+|\u2026|\.)
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.103 | 0.005 | 0.009 | 39 | 4 | 35 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 4 | 35 | 781 |

</details>

---

<details>
<summary>✅ Worked</summary>

**Example 0** (doc_id: `53725`) (sent_id: `53725`)


4. Auf die sofortige Beschwerde der Staatsanwaltschaft Düsseldorf hob das Oberlandesgericht Hamm mit angegriffenem Beschluss vom 28. Juli 2015 den Beschluss des Landgerichts Paderborn auf und ordnete die weitere Vollstreckung der Unterbringung des Beschwerdeführers in der Sicherungsverwahrung an .

| Predicted | Gold |
|---|---|
| `Oberlandesgericht Hamm` | `Oberlandesgericht Hamm` |

**Missed by this rule (FN):**

- `Staatsanwaltschaft Düsseldorf` (ORG)
- `Landgerichts Paderborn` (ORG)

**Example 1** (doc_id: `54861`) (sent_id: `54861`)


Das Arbeitsgericht Zwickau verurteilte die Beklagte am 22. April 2015 ( - 9 Ca 146/15 - ) , das abgebrochene Stellenbesetzungsverfahren 01/2014 fortzuführen und über die Bewerbung des Klägers erneut zu entscheiden .

| Predicted | Gold |
|---|---|
| `Arbeitsgericht Zwickau` | `Arbeitsgericht Zwickau` |

**Missed by this rule (FN):**

- `22. April 2015 ( - 9 Ca 146/15 - )` (RS)

**Example 2** (doc_id: `56732`) (sent_id: `56732`)


Dieser half das Amtsgericht Luckenwalde mit Beschluss vom 11. November 2013 nicht ab .

| Predicted | Gold |
|---|---|
| `Amtsgericht Luckenwalde` | `Amtsgericht Luckenwalde` |

**Example 3** (doc_id: `57841`) (sent_id: `57841`)


2. Das Amtsgericht Dieburg gab der Klage mit Urteil vom 7. Dezember 2012 statt , erklärte die Zwangsvollstreckung aus dem Vollstreckungsbescheid insgesamt für unzulässig und verurteilte den Beklagten , die vollstreckbare Ausfertigung an den Beschwerdeführer herauszugeben ; alle Forderungen des Beklagten gegen den Beschwerdeführer seien getilgt .

| Predicted | Gold |
|---|---|
| `Amtsgericht Dieburg` | `Amtsgericht Dieburg` |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53560`) (sent_id: `53560`)


Welches Volljährigkeitsalter nach dem Recht der Republik Guinea gilt , wird in der obergerichtlichen Rechtsprechung uneinheitlich beantwortet .

**False Positives:**

- `Republik Guinea` — type mismatch — same span as gold: `Republik Guinea`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Republik Guinea`(LOC)

**Example 1** (doc_id: `53675`) (sent_id: `53675`)


Sie endet bei Wegfall der Erwerbsminderungsrente aus der gesetzlichen Rentenversicherung .

**False Positives:**

- `Rentenversicherung .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 2** (doc_id: `53856`) (sent_id: `53856`)


c ) Die nach § 9 Abs. 1 Satz 3 DVO.EKD aF erfolgte Zuordnung der Klägerin zu höchstens Stufe 2 der Entgeltgruppe 14 DVO.EKD verstieß nicht gegen das Recht der Europäischen Union .

**False Positives:**

- `Union .` — positional overlap with gold: `Europäischen Union`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 9 Abs. 1 Satz 3 DVO.EKD aF`(REG)
- `DVO.EKD`(REG)
- `Europäischen Union`(ORG)

**Example 3** (doc_id: `53889`) (sent_id: `53889`)


Er zielt sachlich auf einen objektiv bestehenden Bedarf an zusätzlichem richterlichem Personal bei einem konkreten Verwaltungsgericht .

**False Positives:**

- `Verwaltungsgericht .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `54197`) (sent_id: `54197`)


Gleiches gilt für die Klagebefugnis des Empfangsbevollmächtigten einer atypisch stillen Gesellschaft .

**False Positives:**

- `Gesellschaft .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `54451`) (sent_id: `54451`)


Die vom Berufungsgericht herangezogene Instanzrechtsprechung ( vgl. LG Mönchengladbach , Urteile vom 11. Juli 2006 - 2 S 176/05 , juris , und vom 7. April 2006 - 2 S 172/05 , juris ; LG Lübeck , NJW-RR 1999 , 1655 ; LG Mainz , NJW-RR 1998 , 631 ; AG Donaueschingen , Urteil vom 25. Juli 2002 - 31 C 176/02 , juris ; AG Köpenick , NJW 1996 , 1005 ) bezieht sich im Übrigen nicht auf Verträge über die Schaltung einer Werbeanzeige unter einer konkret bezeichneten Domain .

**False Positives:**

- `AG Donaueschingen` — partial — pred is substring of gold: `AG Donaueschingen , Urteil vom 25. Juli 2002 - 31 C 176/02 , juris`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `LG Mönchengladbach , Urteile vom 11. Juli 2006 - 2 S 176/05 , juris`(RS)
- `vom 7. April 2006 - 2 S 172/05 , juris`(RS)
- `LG Lübeck , NJW-RR 1999 , 1655`(RS)
- `LG Mainz , NJW-RR 1998 , 631`(RS)
- `AG Donaueschingen , Urteil vom 25. Juli 2002 - 31 C 176/02 , juris`(RS)
- `AG Köpenick , NJW 1996 , 1005`(RS)

**Example 6** (doc_id: `54500`) (sent_id: `54500`)


Die Sache wird an das Finanzgericht Rheinland-Pfalz zurückverwiesen .

**False Positives:**

- `Finanzgericht Rheinland` — partial — pred is substring of gold: `Finanzgericht Rheinland-Pfalz`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Finanzgericht Rheinland-Pfalz`(ORG)

**Example 7** (doc_id: `54697`) (sent_id: `54697`)


Zwei dieser Gruppen umfassen nur ein Kriterium ( Gruppe 1 : Sanitätsstabsoffizier Zahnarzt < Rang 1 > ; Gruppe 3 : Leiter einer Zahnärztlichen Behandlungseinrichtung < Rang 3 > ) , eine Gruppe besteht aus vier Kriterien ( Gruppe 2 : Fachzahnarzt Oralchirurgie < Rang 2 > , Curriculare Fortbildung Parodontologie < Rang 4 > , Curriculare Fortbildung Prothetik < Rang 5 > und Curriculare Fortbildung CMD < Rang 6 > ) .

**False Positives:**

- `Gruppe 1` — no gold match — likely missing annotation
- `Gruppe 3` — no gold match — likely missing annotation
- `Gruppe 2` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 3

**Example 8** (doc_id: `54703`) (sent_id: `54703`)


b ) Beim Richter auf Zeit nach § 18 VwGO ergibt sich eine spezifische , mit Art. 97 Abs. 1 GG unvereinbare Möglichkeit der vermeidbaren Einflussnahme durch die Exekutive auf seine richterliche Tätigkeit aus der durch den Richterstatus nur vorübergehend gesicherten persönlichen Unabhängigkeit im Sinne von Art. 97 Abs. 2 GG und der danach absehbar ( wieder ) bestehenden stärkeren Abhängigkeit der beruflichen Karriere des Richters gerade vom Staat .

**False Positives:**

- `Staat .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 18 VwGO`(NRM)
- `Art. 97 Abs. 1 GG`(NRM)
- `Art. 97 Abs. 2 GG`(NRM)

**Example 9** (doc_id: `54738`) (sent_id: `54738`)


b ) Die gegen die Disziplinarverfügung gerichtete Klage wies das Verwaltungsgericht Osnabrück mit Urteil vom 19. August 2011 - 9 A 1/11 - ab .

**False Positives:**

- `Verwaltungsgericht Osnabr` — partial — pred is substring of gold: `Verwaltungsgericht Osnabrück mit Urteil vom 19. August 2011 - 9 A 1/11 -`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Verwaltungsgericht Osnabrück mit Urteil vom 19. August 2011 - 9 A 1/11 -`(RS)

**Example 10** (doc_id: `54827`) (sent_id: `54827`)


2 ) Soweit die Klägerin weiter rügt , das LSG habe gegen § 103 SGG verstoßen , weil das Gericht Beweisanträgen ohne hinreichende Begründung nicht gefolgt sei , genügt sie mit ihrer Beschwerdebegründung ebenfalls nicht den Anforderungen des § 160a Abs 2 S 3 SGG .

**False Positives:**

- `Gericht Beweisantr` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 103 SGG`(NRM)
- `§ 160a Abs 2 S 3 SGG`(NRM)

**Example 11** (doc_id: `55082`) (sent_id: `55082`)


Demgemäß hat der Senat Schüler , die im häuslichen Bereich unterrichtsvorbereitend ein Werkstück erstellen ( BSG vom 1. 2. 1979 - 2 RU 107/77 - SozR 2200 § 539 Nr 54 ) , ebenso wenig für versichert erachtet wie solche , die für die schulische Foto-AG in der Altstadt ohne weitere Aufsicht fotografieren ( BSG vom 30. 5. 1988 - 2 RU 5/88 - Juris ) .

**False Positives:**

- `Senat Sch` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BSG vom 1. 2. 1979 - 2 RU 107/77 - SozR 2200 § 539 Nr 54`(RS)
- `Foto-AG`(ORG)
- `BSG vom 30. 5. 1988 - 2 RU 5/88 - Juris`(RS)

**Example 12** (doc_id: `55122`) (sent_id: `55122`)


So bietet beispielsweise die Firma Pointer „ Wohlfühlfarben für die Wohnung “ an ; in einem der Anmelderin übersandten Internetausdruck heißt es hierzu : „ Farben gezielt einsetzen .

**False Positives:**

- `Firma Pointer` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 13** (doc_id: `55334`) (sent_id: `55334`)


Mit Wirkung vom 1. März 2011 ernannte ihn die Ministerin für Wissenschaft , Forschung und Kultur erneut unter Berufung in das Beamtenverhältnis auf Zeit für die Dauer von sechs Jahren zum Kanzler der Hochschule .

**False Positives:**

- `Hochschule .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 14** (doc_id: `55615`) (sent_id: `55615`)


Neben der Beklagten zu 1. , in deren Betrieb ein Betriebsrat gewählt war , gehören weitere rechtlich eigenständige Standortgesellschaften zur sog. w Gruppe .

**False Positives:**

- `Gruppe .` — positional overlap with gold: `w Gruppe`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `w Gruppe`(ORG)

**Example 15** (doc_id: `55719`) (sent_id: `55719`)


Dies gilt auch für die Strukturprinzipien des Art. 33 Abs. 5 GG , die einem Ausgleich mit anderen Gütern nicht von vornherein verschlossen sind ( vgl. Kees , Der Staat 54 < 2015 > , S. 63 < 75 > ) .

**False Positives:**

- `Staat 54` — partial — pred is substring of gold: `Kees , Der Staat 54 < 2015 > , S. 63 < 75 >`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 33 Abs. 5 GG`(NRM)
- `Kees , Der Staat 54 < 2015 > , S. 63 < 75 >`(LIT)

**Example 16** (doc_id: `56089`) (sent_id: `56089`)


Der Endoskopkopf 1 weist auch eine Ventilaufnahme ( zylindrische Kammer 4 ) auf , durch die die beiden Kanäle ( erster Einlass 2 - erster Auslass 5 ; zweiter Einlass 3 - zweiter Auslass 6 ) führen [ = Merkmal M4 ] .

**False Positives:**

- `Kammer 4` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 17** (doc_id: `56323`) (sent_id: `56323`)


aa ) Mit Urteil vom 24. 3. 2016 ( B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 ) hat der Senat bereits entschieden , dass es für den Beginn der aufgeschobenen Versicherungspflicht nach § 7a Abs 6 S 1 SGB IV - mit Wirkung für alle Zweige der Sozialversicherung - auf die Bekanntgabe einer ( ersten ) Entscheidung der Deutschen Rentenversicherung Bund über das Bestehen von " Beschäftigung " ankommt und nicht auf eine ( spätere ) - diese unzulässige Elementenfeststellung korrigierende - Entscheidung über " Versicherungspflicht wegen Beschäftigung " .

**False Positives:**

- `Rentenversicherung Bund` — partial — pred is substring of gold: `Deutschen Rentenversicherung Bund`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Urteil vom 24. 3. 2016 ( B 12 R 3/14 R - SozR 4 - 2400 § 7a Nr 5 )`(RS)
- `§ 7a Abs 6 S 1 SGB IV`(NRM)
- `Deutschen Rentenversicherung Bund`(ORG)

**Example 18** (doc_id: `56966`) (sent_id: `56966`)


Dies betrifft sowohl die angeordneten Tätigkeiten in der Abteilung Standesamt und Gerichtliche Angelegenheiten als auch diejenigen für die Visaabteilung .

**False Positives:**

- `Abteilung Standesamt` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 19** (doc_id: `57172`) (sent_id: `57172`)


( b ) Gemessen hieran ist das Oberlandesgericht ohne ausreichende Ermittlungen zu dem Ergebnis gelangt , dass die Volljährigkeit ( auch ) nach dem Recht der Republik Guinea mit der Vollendung des 18. Lebensjahres eintrete .

**False Positives:**

- `Republik Guinea` — type mismatch — same span as gold: `Republik Guinea`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Republik Guinea`(LOC)

**Example 20** (doc_id: `57238`) (sent_id: `57238`)


Am 7. April 1999 leistete diese auf die noch ausstehende Stammeinlage für den an sie abgetretenen Geschäftsanteil den offenen Betrag von 12.500 DM an die Gesellschaft .

**False Positives:**

- `Gesellschaft .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 21** (doc_id: `57554`) (sent_id: `57554`)


festzustellen , dass auf das Arbeitsverhältnis der Parteien der Kirchliche Arbeitnehmerinnen Tarifvertrag , abgeschlossen zwischen dem Verband kirchlicher und diakonischer Anstellungsträger Nordelbien sowie der Gewerkschaft Kirche und Diakonie und ver.di , Landesbezirke Hamburg und Nord , andererseits vom 1. Dezember 2006 Anwendung finde .

**False Positives:**

- `Gewerkschaft Kirche` — partial — pred is substring of gold: `Gewerkschaft Kirche und Diakonie`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Kirchliche Arbeitnehmerinnen Tarifvertrag`(REG)
- `Verband kirchlicher und diakonischer Anstellungsträger Nordelbien`(ORG)
- `Gewerkschaft Kirche und Diakonie`(ORG)
- `ver.di , Landesbezirke Hamburg und Nord`(ORG)

**Example 22** (doc_id: `57565`) (sent_id: `57565`)


b ) Entgegen der Auffassung des Senats ist es im Hinblick auf das äußere Bild der Neutralität und Unparteilichkeit nicht nur bedenklich , wenn ein Richter auf Zeit in Verfahren entscheiden würde , in denen die Stammbehörde des Richters oder eine dieser vorgesetzte Behörde Beteiligte ist .

**False Positives:**

- `Behörde Beteiligte` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 23** (doc_id: `57748`) (sent_id: `57748`)


Sie verfügten bei Aufnahme der Tätigkeit jeweils über eine Befreiungsentscheidung der Bundesversicherungsanstalt für Angestellte als Rechtsvorgängerin der beklagten Deutschen Rentenversicherung Bund .

**False Positives:**

- `Rentenversicherung Bund` — partial — pred is substring of gold: `Deutschen Rentenversicherung Bund`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Deutschen Rentenversicherung Bund`(ORG)

**Example 24** (doc_id: `58275`) (sent_id: `58275`)


Dabei ist der Ausgangspunkt jeweils identisch , wonach gemäß dem - bislang nicht ausdrücklich aufgehobenen - Art. 443 des Code Civil der Republik Guinea die Volljährigkeit auf das vollendete 21. Lebensjahr festgesetzt wird .

**False Positives:**

- `Republik Guinea` — partial — pred is substring of gold: `Art. 443 des Code Civil der Republik Guinea`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 443 des Code Civil der Republik Guinea`(NRM)

**Example 25** (doc_id: `58309`) (sent_id: `58309`)


a ) Das Landesarbeitsgericht hat angenommen , mit der in Punkt „ Siebtens “ des Arbeitsvertrags vereinbarten Anwendbarkeit der arbeitsrechtlichen Vorschriften der deutschen Gesetzgebung sei auch § 4 KSchG vereinbart und mithin das Erfordernis einer fristgerechten Klageerhebung vor einem deutschen Gericht .

**False Positives:**

- `Gericht .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `§ 4 KSchG`(NRM)

**Example 26** (doc_id: `58567`) (sent_id: `58567`)


Darüber hinaus ist darauf hinzuweisen , dass über den bloßen , auch in der Druckschrift D10 zum Ausdruck gebrachten Wunsch hinaus auch die Streitpatentschrift weder in den Patentansprüchen noch an anderer Stelle Angaben dazu macht , welche Maßnahmen ergriffen werden sollen , um polymerisolierte Kabel für HGÜ-Zwecke verwenden zu können .

**False Positives:**

- `Stelle Angaben` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 27** (doc_id: `58736`) (sent_id: `58736`)


Daher kann dahinstehen , ob die Vorschrift - wie das Landesarbeitsgericht meint - generell eine Spezialregelung gegenüber der Ausschlussfrist des § 37 Abs. 1 TVöD darstellt ( so wohl auch Breier / Dassau / Kiefer / Lang / Langenbrinck TVöD Stand 1/2018 B 2.2 § 26 TVÜ-Bund Erl. 2 Rn. 6 ; Litschen in Adam / Bauer / Bettenhausen / ua. Tarifrecht der Beschäftigten im öffentlichen Dienst Stand Dezember 2015 Teil II § 26 TVÜ-Bund B I Rn. 4 ; Clemens / Scheuring / Steingen / Wiese TVöD Stand Januar 2018 Teil IV / 3 Rn. 372 ) , oder ob sich diese lediglich auf den Wechsel in das neue tarifliche Entgeltsystem bezieht und es hinsichtlich der sich aus der Ausübung des Antragsrechts folgenden Zahlungsansprüche bei der allgemeinen Ausschlussfrist des § 37 Abs. 1 TVöD verbleibt ( so für § 29a Abs. 4 Satz 1 TVÜ-Länder BeckOK TV-L / Dannenberg Stand 1. Januar 2013 TVÜ-Länder § 29a Rn. 38 ; Augustin ZTR 2012 , 484 ) und wann diese ggf. fällig werden .

**False Positives:**

- `Dienst Stand Dezember` — partial — pred is substring of gold: `Litschen in Adam / Bauer / Bettenhausen / ua. Tarifrecht der Beschäftigten im öffentlichen Dienst Stand Dezember 2015 Teil II § 26 TVÜ-Bund B I Rn. 4`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 37 Abs. 1 TVöD`(REG)
- `Breier / Dassau / Kiefer / Lang / Langenbrinck TVöD Stand 1/2018 B 2.2 § 26 TVÜ-Bund Erl. 2 Rn. 6`(LIT)
- `Litschen in Adam / Bauer / Bettenhausen / ua. Tarifrecht der Beschäftigten im öffentlichen Dienst Stand Dezember 2015 Teil II § 26 TVÜ-Bund B I Rn. 4`(LIT)
- `Clemens / Scheuring / Steingen / Wiese TVöD Stand Januar 2018 Teil IV / 3 Rn. 372`(LIT)
- `§ 37 Abs. 1 TVöD`(REG)
- `§ 29a Abs. 4 Satz 1 TVÜ-Länder`(REG)
- `BeckOK TV-L / Dannenberg Stand 1. Januar 2013 TVÜ-Länder § 29a Rn. 38`(LIT)
- `Augustin ZTR 2012 , 484`(LIT)

**Example 28** (doc_id: `58927`) (sent_id: `58927`)


Der in § 1 KAT erwähnte Kirchliche Tarifvertrag Diakonie ( KTD , GVOBl. NEK S. 317 ) wurde bereits am 15. August 2002 zwischen dem Verband kirchlicher und diakonischer Anstellungsträger Nordelbien einerseits und der Gewerkschaft Kirche und Diakonie , der IG Bauen-Agrar-Umwelt , Bundesvorstand , sowie von ver.di andererseits geschlossen .

**False Positives:**

- `Gewerkschaft Kirche` — partial — pred is substring of gold: `Gewerkschaft Kirche und Diakonie`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `§ 1 KAT`(REG)
- `Kirchliche Tarifvertrag Diakonie ( KTD , GVOBl. NEK S. 317 )`(REG)
- `Verband kirchlicher und diakonischer Anstellungsträger Nordelbien`(ORG)
- `Gewerkschaft Kirche und Diakonie`(ORG)
- `IG Bauen-Agrar-Umwelt , Bundesvorstand`(ORG)
- `ver.di`(ORG)

**Example 29** (doc_id: `59299`) (sent_id: `59299`)


Im Streitfall war der Kläger indessen schon kein beherrschender Gesellschafter-Geschäftsführer der GmbH .

**False Positives:**

- `GmbH .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 30** (doc_id: `59356`) (sent_id: `59356`)


Über die sofortige Beschwerde der Nebenklägerin K. T. gegen die im Urteil des Landgerichts Frankfurt am Main vom 15. August 2017 getroffene Kostenentscheidung hat das Oberlandesgericht Frankfurt am Main zu entscheiden .

**False Positives:**

- `Oberlandesgericht Frankfurt` — partial — pred is substring of gold: `Oberlandesgericht Frankfurt am Main`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `K. T.`(PER)
- `Landgerichts Frankfurt am Main`(ORG)
- `Oberlandesgericht Frankfurt am Main`(ORG)

**Example 31** (doc_id: `59656`) (sent_id: `59656`)


Hier liegt der Fall in zweierlei Hinsicht anders : Zum einen ist kein Vorgesetzter des Klägers aus dem Beurteilungszeitraum mehr im Dienst .

**False Positives:**

- `Dienst .` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 32** (doc_id: `59676`) (sent_id: `59676`)


Hiervon entfielen auf die GmbH 142.957 € ( 3 % ) und auf die natürliche Person 4.622.283 € ( 97 % ) .

**False Positives:**

- `GmbH 142` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

## `Organization with Location/Type`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `5b6bf80a`  
**Description:**
Matches organizations with specific descriptors like 'Schulzentrum für Technik' or 'Senat des ...', ensuring full capture.

**Content:**
```
\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Schulzentrum\s+f\u00fcr\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|Senat\s+des\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|Kammer\s+des|Abteilung\s+des|Zweig|Niederlassung|Gesch\u00e4ftsf\u00fchrer|Bundesministerium\s+f\u00fcr\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|Bundesagentur\s+f\u00fcr\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 1 | 0 | 1 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 0 | 1 | 171 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `58718`) (sent_id: `58718`)


Der Geschäftsführer der Beklagten S. ist zugleich Geschäftsführer der - mittlerweile in Liquidation befindlichen - Help Food .

**False Positives:**

- `Der Geschäftsführer` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `S.`(PER)
- `Help Food`(ORG)

</details>

---

## `Quoted Organization Names`

**F1:** 0.000 | **Precision:** 0.000 | **Recall:** 0.000  

**Format:** `regex`  
**Rule ID:** `c1f44934`  
**Description:**
Matches organization names enclosed in German quotation marks, but only when preceded by context indicating an organization (e.g., 'Firma', 'Marke', 'Name', 'der', 'des') to avoid matching product names or streets.

**Content:**
```
(?:Firma|Marke|Name|der|des|bei|von|aus|in)\s*\u201e\s*([A-Z][a-zA-Z\u00e4\u00f6\u00fc\u00df\s]+(?:\s+[A-Z][a-zA-Z\u00e4\u00f6\u00fc\u00df\s]+)*)\s*\u201c
```

<details>
<summary>📊 Detailed Metrics</summary>

| Precision | Recall | F1 | Total Predicted | TP | FP |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 31 | 0 | 31 |

**Per-Class Breakdown**

| Class | TP | FP | FN |
|---|---|---|---|
| `ORG` | 0 | 31 | 774 |

</details>

---

<details>
<summary>⚠️ False Positives</summary>

**Example 0** (doc_id: `53634`) (sent_id: `53634`)


Aus deren Sicht könne das Wort „ targeting “ auch andere Bedeutungen haben , wie etwa „ Zielausrichtung “ oder „ Zielbestimmung “ .

**False Positives:**

- `Zielbestimmung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 1** (doc_id: `53933`) (sent_id: `53933`)


Weder die Studie der Holy Fashion Group vom 24. Februar 2016 ( Bl. 49/50 d. A. ) noch die im Amtsverfahren vorgelegten Unterlagen zu Marktforschungsergebnissen einer „ Brigitte “ -Studie , Internetausdrucken zu Showrooms und Verkaufsstätten von Waren der Marke „ JOOP “ oder die beigefügten Urteile sind geeignet , einen entsprechenden Benutzungsnachweis für mit der Marke gekennzeichnete Dienstleistungen zu erbringen .

**False Positives:**

- `JOOP ` — partial — pred is substring of gold: `„ JOOP “`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Holy Fashion Group`(ORG)
- `„ JOOP “`(ORG)

**Example 2** (doc_id: `54038`) (sent_id: `54038`)


Er hat insbesondere angegeben , dass er am 16. Mai 2006 als Entwicklungsleiter im Bereich Sicherheitstechnik das Vertriebsfreigabedokument E4a für diese Produktfamilie unterzeichnet habe ( vgl. auch die beiden Felder „ Date “ und „ Signature ISC spokesperson “ am Ende des Dokuments E4a ) .

**False Positives:**

- `Date ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 3** (doc_id: `54137`) (sent_id: `54137`)


Dagegen beanspruche die Anmelderin vorliegend im Wesentlichen „ chemische Hilfsmaterialien “ , welche sich an Handwerker richteten , die die Waren alleine aufgrund ihrer ( technischen ) Funktion und Qualität erwerben würden , für die ein „ Wohlfühleffekt “ keine Rolle spiele .

**False Positives:**

- `Wohlfühleffekt ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 4** (doc_id: `54316`) (sent_id: `54316`)


Bereits in der Internet-Werbung der „ Kooperationskasse “ fand der qualifizierte Nachrang Erwähnung , auch wenn die mit ihr verbundene Bedingung ( „ Rückforderung darf nicht zur Insolvenz führen “ ) als „ theoretisch “ bezeichnet wurde .

**False Positives:**

- `Kooperationskasse ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 5** (doc_id: `54681`) (sent_id: `54681`)


Ein „ Klosterhof “ ist die den deutschen Verbrauchern ohne weiteres verständliche Bezeichnung einer Gebäudeteils einer Klosteranlage und eines Hofes im Sinn eines Anwesens und bäuerlich landwirtschaftlichen Betriebs , das sich durch die Bewirtschaftung durch oder die Zugehörigkeit zu einem Kloster auszeichnet .

**False Positives:**

- `Klosterhof ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 6** (doc_id: `54871`) (sent_id: `54871`)


Zudem könne der Schutzgegenstand in Anwendung der BGH-Entscheidung „ Weinkaraffe “ ( GRUR 2012 , 1139 ) auch durch Auslegung unter Berücksichtigung der Beschreibung sowie der Erzeugnisangabe aus der „ Schnittmenge “ der gemeinsamen Merkmale ermittelt werden .

**False Positives:**

- `Schnittmenge ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `BGH-Entscheidung „ Weinkaraffe “ ( GRUR 2012 , 1139 )`(RS)

**Example 7** (doc_id: `54903`) (sent_id: `54903`)


Die angesprochenen Verkehrskreise seien durch Marken wie „ Facebook “ ( Jahrbuch ) , „ Soundcloud “ ( Klangwolke ) oder „ My Space “ ( Mein Raum ) gewohnt , dass markenrechtlich geschützte Bezeichnungen für bestimmte Netzwerke durch die Zusammensetzung und Neukreation an sich beschreibender Einzelelemente entstünden .

**False Positives:**

- `My Space ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 8** (doc_id: `54947`) (sent_id: `54947`)


Soweit die Widersprechende sich darauf beruft , dass sie den größten Online-Shop der Welt betreibe , hat sie nicht vorgetragen , dass dies unter der Marke „ Fire “ geschehe , so dass diese Tatsache für eine etwaige Steigerung der Kennzeichnungskraft ohne Relevanz ist .

**False Positives:**

- `Fire ` — partial — pred is substring of gold: `„ Fire “`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `„ Fire “`(ORG)

**Example 9** (doc_id: `55014`) (sent_id: `55014`)


Voraussetzung hierfür ist ein entsprechendes „ Einvernehmen zwischen Arbeitgeber und Arbeitnehmer “ .

**False Positives:**

- `Einvernehmen zwischen Arbeitgeber und Arbeitnehmer ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 10** (doc_id: `55132`) (sent_id: `55132`)


a ) Wie das Deutsche Patent- und Markenamt zutreffend ausgeführt hat , ist das Anmeldezeichen zum Zeitpunkt seiner Anmeldung am 20. Januar 2016 von den überwiegend angesprochenen Fachverkehrskreisen , aber - insbesondere in Bezug auf die Dienstleistungen der Klassen 40 und 44 - auch von Auftraggebern oder Empfängern zahntechnischer oder -medizinischer Dienstleistungen im Sinne von „ CAD Labor “ oder „ Labor , das CAD einsetzt “ verstanden worden .

**False Positives:**

- `CAD Labor ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutsche Patent- und Markenamt`(ORG)

**Example 11** (doc_id: `55566`) (sent_id: `55566`)


ccc ) „ Music ” entspricht dem deutschen Wort „ Musik ” ( Langenscheidts Schulwörterbuch Englisch , 1986 ) und hat die Bedeutung „ Tonkunst “ , „ Komposition “ oder „ Musikstücke “ ( Duden - Die deutsche Rechtschreibung , 26. Aufl. 2013 ) .

**False Positives:**

- `Musikstücke ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Duden`(ORG)

**Example 12** (doc_id: `55629`) (sent_id: `55629`)


Zwar hat das Arbeitsgericht im Tatbestand , den das Landesarbeitsgericht in Bezug genommen hat , ausgeführt , die maximale Wasserverdrängung der „ G “ betrage 6,5 m³ .

**False Positives:**

- `G ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 13** (doc_id: `55768`) (sent_id: `55768`)


Schließlich ließ sich der „ Vorstand von Neudeutschland “ die Berechtigung einräumen , die Einzahlung für die Verwirklichung eines Projekts zusammenzulegen .

**False Positives:**

- `Vorstand von Neudeutschland ` — partial — gold is substring of pred: `Neudeutschland`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Neudeutschland`(ORG)

**Example 14** (doc_id: `56153`) (sent_id: `56153`)


III. Sollten nach den vom Landesarbeitsgericht noch zu treffenden Feststellungen die Befähigung des Klägers , die von ihm befahrene Binnenwasserstraße und die technische Ausstattung der „ G “ den tariflichen Anforderungen entsprechen , kommt ein Vergütungsanspruch des Klägers nach der Entgeltgruppe 8 der Anlage 1 zum TV EntgO Bund grundsätzlich in Betracht .

**False Positives:**

- `G ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Anlage 1 zum TV EntgO Bund`(REG)

**Example 15** (doc_id: `56195`) (sent_id: `56195`)


Vergleichbar gebildete Bezeichnungen wie „ Frau und Wirtschaft “ , „ Erfahrung ist Zukunft “ , „ Technik und Wirtschaft “ oder „ Recycling ist Zukunft “ würden bereits beschreibend verwendet .

**False Positives:**

- `Recycling ist Zukunft ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 16** (doc_id: `56254`) (sent_id: `56254`)


Ein großer Teil des angesprochenen Verkehrskreises sei der französischen Sprache insoweit mächtig , dass er die Bedeutung von „ Petit Filou “ , nämlich „ kleiner Schlingel “ , „ kleiner Spitzbub “ , verstehe , zumindest sei der Ausdruck im deutschsprachigen Raum bekannt .

**False Positives:**

- `Petit Filou ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 17** (doc_id: `56784`) (sent_id: `56784`)


c ) Der Senat teilt die Auffassung der Markenstelle , dass das angesprochene Publikum das umgedrehte Ausrufezeichen ohne analysierende Betrachtung und gedankliche Zwischenschritte zwanglos als Ersetzung des Buchstaben „ I / i “ lesen und damit als „ WIR “ oder „ WiR “ auffassen wird .

**False Positives:**

- `WiR ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 18** (doc_id: `56845`) (sent_id: `56845`)


Infolge der Intransparenz des Merkmals der „ Geeignetheit “ und der ersatzlosen Streichung der Regelung in Ziff. 2.3 Unterabs. 3 Satz 2 des Erlasses gab es keinen Anknüpfungspunkt mehr für die Frage , ob eine „ Eignung “ nur für ein Fach oder für zwei Fächer bestand , so dass auch keine Herabgruppierung bei „ Eignung “ nur für ein Fach mehr erfolgen konnte .

**False Positives:**

- `Geeignetheit ` — no gold match — likely missing annotation
- `Eignung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 2

**Gold Entities:**

- `Ziff. 2.3 Unterabs. 3 Satz 2 des Erlasses`(REG)

**Example 19** (doc_id: `57367`) (sent_id: `57367`)


Dass der Begriff des „ Kontors “ für verschiedenste Dienstleistungen - wie etwa Spirituosenherstellung , Versicherungswesen , Werbung oder Immobilienwesen - aktuell Verwendung findet , hat das Deutsche Patent- und Markenamt überzeugend dargetan .

**False Positives:**

- `Kontors ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Deutsche Patent- und Markenamt`(ORG)

**Example 20** (doc_id: `58231`) (sent_id: `58231`)


Wie bei den Begriffen „ DIE Limmer Schleuse “ oder „ DIE Limmer Kanu Regatta “ würden die angesprochenen Verkehrskreise auch das Anmeldezeichen im Sinne von „ DAS Limmer Kontor “ einem bestimmten Anbieter bzw. einer Institution zuordnen können .

**False Positives:**

- `DIE Limmer Kanu Regatta ` — similar text (different position): `Limmer`
- `DAS Limmer Kontor ` — similar text (different position): `Limmer`

> overlaps gold: 2  |  likely missing annotation: 0

**Gold Entities:**

- `Limmer`(LOC)
- `Limmer`(LOC)
- `Limmer`(LOC)

**Example 21** (doc_id: `58364`) (sent_id: `58364`)


Der angesprochene Verkehr würde die Wortfolge der angegriffenen Marke „ ARROW AND BEAST “ nicht auf „ ARROW “ verkürzen .

**False Positives:**

- `ARROW AND BEAST ` — partial — pred is substring of gold: `„ ARROW AND BEAST “`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `„ ARROW AND BEAST “`(ORG)
- `„ ARROW “`(ORG)

**Example 22** (doc_id: `58381`) (sent_id: `58381`)


Soweit der Gerichtshof in den beiden Entscheidungen darauf abstellt , dass ein Grundpatent im Sinne der Art. 1 ( c ) und 3 ( a ) AMVO einen Wirkstoff nur dann „ als solchen “ schützt , wenn er den Gegenstand der von dem Patent geschützten Erfindung bildet ( EuGH , GRUR Int. 2015 , 446 , Rnd. 38 – Actavis / Boehringer ) , wertet dies der Senat als Bestätigung der in „ Medeva “ und „ Eli Lilly “ niedergelegten Grundsätze .

**False Positives:**

- `Medeva ` — partial — pred is substring of gold: `„ Medeva “`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `Art. 1 ( c ) und 3 ( a ) AMVO`(NRM)
- `EuGH , GRUR Int. 2015 , 446 , Rnd. 38 – Actavis / Boehringer`(RS)
- `„ Medeva “`(ORG)
- `„ Eli Lilly “`(ORG)

**Example 23** (doc_id: `58483`) (sent_id: `58483`)


b ) Für die Annahme der Beklagten , unter „ ununterbrochenem Einsatz “ sei die Summe der im Kundenbetrieb geleisteten Arbeitstage zu verstehen , weil mit dem Branchenzuschlag ein „ Erfahrungszuschlag “ gewährt werde , gibt es im Tarifvertrag keine Anhaltspunkte .

**False Positives:**

- `Erfahrungszuschlag ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 24** (doc_id: `58525`) (sent_id: `58525`)


Weder in der Bedeutung „ Schlingel / Schelm / Schlawiner “ noch in der von „ Gauner “ handelt es sich – entgegen der Auffassung der Markeninhaberin um eine spezielle Zielgruppe .

**False Positives:**

- `Gauner ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 25** (doc_id: `58544`) (sent_id: `58544`)


Der Begriff der „ Normalleistung “ hat keinen Eingang in den Wortlaut des Mindestlohngesetzes gefunden ( im Einzelnen : BAG 21. Dezember 2016 - 5 AZR 374/16 - Rn. 21 , BAGE 157 , 356 ; zust .

**False Positives:**

- `Normalleistung ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Gold Entities:**

- `Mindestlohngesetzes`(NRM)
- `BAG 21. Dezember 2016 - 5 AZR 374/16 - Rn. 21 , BAGE 157 , 356`(RS)

**Example 26** (doc_id: `58936`) (sent_id: `58936`)


Kein Teil der angegriffenen Marke sei stärker prägend als der andere , Auch der Bedeutungsgehalt von „ Shot “ im Sinn von „ Schuss , etwas Schnelles , Kurzes “ erleichtere das Auseinanderhalten der Vergleichszeichen und eigne sich dazu , Hör- und Merkfehler zu vermeiden .

**False Positives:**

- `Shot ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

**Example 27** (doc_id: `59193`) (sent_id: `59193`)


Zwar mag der Name „ Albert Einstein “ bzw. der Nachname „ Einstein “ als solcher im Einzelfall als Synonym für „ Genie “ verwendet werden .

**False Positives:**

- `Albert Einstein ` — partial — pred is substring of gold: `„ Albert Einstein “`

> overlaps gold: 1  |  likely missing annotation: 0

**Gold Entities:**

- `„ Albert Einstein “`(PER)
- `„ Einstein “`(PER)

**Example 28** (doc_id: `60041`) (sent_id: `60041`)


HLNK39 Fachinformation zu Yohimbin „ Spiegel “ , Stand : September 2008 , 5 Seiten

**False Positives:**

- `Spiegel ` — no gold match — likely missing annotation

> overlaps gold: 0  |  likely missing annotation: 1

</details>

---

</details>

---


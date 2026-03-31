
PROMPT_EXTRACTION_V1 = """
You are an assistant that extracts structured bibliographic data from a German patent document.

Your task:
Extract the following fields and return them as a single, valid JSON object:

- title (string, in the original text, DO NOT translate)
- inventors (list of objects with fields: name, address)
- assignees (list of objects with fields: name, address)
- pub_date_application (YYYY-MM-DD or null)
- pub_date_publication (YYYY-MM-DD or null)
- pub_date_foreign (YYYY-MM-DD or null)
- classification (string or null)
- industrial_field (short classification category, in the original text, DO NOT translate)

Rules:
- ALWAYS output a valid JSON object.
- If a field is missing, use null.
- Each inventor and assignee should, whenever possible, include both name and address.
- Addresses must be formatted as they appear in the document.
- Dates must be in YYYY-MM-DD format
- pub_date_application is the date the application was filed.
- pub_date_publication is the publication/issue date this document was published. If several, choose the oldest publication/issue date.
- pub_date_foreign is the earliest foreign priority date, if multiple foreign priorities exist, choose the oldest; otherwise null.
- The title and industrial_field MUST NOT be translated or altered.
- Do not add any comments or explanations outside the JSON.
- If multiple publication dates appear in the document, choose the OLDEST date by default as pub_date_publication.
- Fill in industrial_field ONLY if it is EXPLICITLY stated in the document after the classification number; otherwise use null.

Self-check before final output:
- Ensure pub_date_application is not later than pub_date_publication (if both exist). If it is later, re-check selection and fix.
- Ensure pub_date_foreign is not later than pub_date_application (if both exist). If it is later, re-check or set pub_date_foreign = null.


Example 1:
{{
  "title": "Dispositif de chauffage solaire à collecteurs modulaires",
  "inventors": [
    {{"name": "Dr. Alice Montreux", "address": "Genf"}},
    {{"name": "Marc-André Keller", "address": "Zürich"}}
  ],
  "assignees": [
    {{"name": "HelioTech SA", "address": "Lausanne (Schweiz)"}}
  ],
  "pub_date_application": "1974-06-14",
  "pub_date_publication": "1976-02-02",
  "pub_date_foreign": null,
  "classification": "15b",
  "industrial_field": "Erneuerbare Energiesysteme"
}}
Example 2:
{{
  "title": "Vorrichtung zur automatischen Schmierung eines Kettenantriebs mit Dosierventil",
  "inventors": [
    {{"name": "Hans-Peter Vogel", "address": "Stuttgart (Deutschland)"}},
    {{"name": "Claire Dubois", "address": "Strasbourg (France)"}}
  ],
  "assignees": [
    {{"name": "Kettenwerk GmbH", "address": "München (Deutschland)"}}
  ],
  "pub_date_application": "1968-11-05",
  "pub_date_publication": "1970-04-22",
  "pub_date_foreign": "1969-02-18",
  "classification": "47c",
  "industrial_field": "Maschinenbau und Antriebstechnik"
}}
Example 3:
{{
  "title": "Verfahren zur Verbesserung der Papierbahnführung in Schnelllaufmaschinen",
  "inventors": null,
  "assignees": [
    {{"name": "Müller & Söhne", "address": "Mannheim (Deutschland)"}}
  ],
  "pub_date_application": "1959-03-17",
  "pub_date_publication": "1961-09-08",
  "pub_date_foreign": null,
  "classification": "42e",
  "industrial_field": "Papiertechnik und industrielle Fertigung"
}}

Text:
{text}
"""

PROMPT_EXTRACTION_V2 = """
Du bist ein Assistent, der strukturierte bibliographische Daten aus einem deutschen Patentdokument extrahiert.

Deine Aufgabe:
Extrahiere die folgenden Felder und gib sie als ein einzelnes, gültiges JSON-Objekt zurück:

- title (string, im Originaltext, NICHT übersetzen)
- inventors (Liste von Objekten mit den Feldern: name, address)
- assignees (Liste von Objekten mit den Feldern: name, address)
- pub_date_application (YYYY-MM-DD oder null)
- pub_date_publication (YYYY-MM-DD oder null)
- pub_date_foreign (YYYY-MM-DD oder null)
- classification (string oder null)
- industrial_field (kurze Klassifikationskategorie, im Originaltext, NICHT übersetzen)

Regeln:
- Gib IMMER ein gültiges JSON-Objekt aus.
- Wenn ein Feld fehlt, verwende null.
- Führen Sie nur diejenigen Personen auf, die im Text ausdrücklich als „Erfinder“ genannt werden. Enthält ein Name ein kaufmännisches Und-Zeichen (&) oder ist er mit einem Ortsnamen verknüpft (z. B. „Name & Name in der Stadt“), gilt er als assignee.
- Jeder Erfinder und jeder Anmelder soll nach Möglichkeit sowohl name als auch address enthalten.
- Adressen müssen so formatiert sein, wie sie im Dokument erscheinen.
- Datumsangaben müssen im Format YYYY-MM-DD stehen.
- pub_date_application ist das Datum, an dem die Anmeldung eingereicht wurde. Ist es nicht vorhanden, verwende null.“
- pub_date_publication ist das Veröffentlichungs-/Ausgabedatum, an dem dieses Dokument veröffentlicht wurde. Wenn mehrere vorhanden sind, wähle das älteste Veröffentlichungs-/Ausgabedatum.
- pub_date_foreign ist das früheste ausländische Prioritätsdatum; wenn mehrere ausländische Prioritäten existieren, wähle das älteste; andernfalls null. Ist es nicht vorhanden, verwende null.“
- Wenn nur ein Datum angegeben ist, handelt es sich um das Veröffentlichungsdatum.
- Der Titel und industrial_field dürfen NICHT übersetzt oder verändert werden.
- Füge keine Kommentare oder Erklärungen außerhalb des JSON hinzu.
- Wenn im Dokument mehrere Veröffentlichungsdaten erscheinen, wähle STANDARDMÄSSIG das ÄLTESTE Datum als pub_date_publication.
- Trage industrial_field NUR ein, wenn es im Dokument EXPLIZIT nach der Klassifikationsnummer angegeben ist; andernfalls verwende null.


Beispielausgabe 1:
{{
  "title": "Dispositif de chauffage solaire à collecteurs modulaires",
  "inventors": [
    {{"name": "Dr. Alice Montreux", "address": "Genf"}},
    {{"name": "Marc-André Keller", "address": "Zürich"}}
  ],
  "assignees": [
    {{"name": "HelioTech SA", "address": "Lausanne (Schweiz)"}}
  ],
  "pub_date_application": "1974-06-14",
  "pub_date_publication": "1976-02-02",
  "pub_date_foreign": null,
  "classification": "15b",
  "industrial_field": "Erneuerbare Energiesysteme"
}}

Beispielausgabe 2:
{{
  "title": "Verfahren zur Verbesserung der Papierbahnführung in Schnelllaufmaschinen",
  "inventors": null,
  "assignees": [
    {{"name": "Müller & Söhne", "address": "Mannheim (Deutschland)"}}
  ],
  "pub_date_application": "1959-03-17",
  "pub_date_publication": "1961-09-08",
  "pub_date_foreign": "1958-12-05",
  "classification": "42e",
  "industrial_field": "Papiertechnik und industrielle Fertigung"
}}

Beispielausgabe 3:
{{
  "title": "Vorrichtung zur automatischen Schmierung eines Kettenantriebs mit Dosierventil",
  "inventors": [
    {{"name": "Hans-Peter Vogel", "address": "Stuttgart (Deutschland)"}},
    {{"name": "Claire Dubois", "address": "Strasbourg (France)"}}
  ],
  "assignees": null,
  "pub_date_application": "null",
  "pub_date_publication": "1970-04-22",
  "pub_date_foreign": "null",
  "classification": "null",
  "industrial_field": "null"
}}
Text:
{text}

"""

PROMPT_EXTRACTION_V3 = """
You are an assistant that extracts structured bibliographic data from a German patent document.

Return one valid JSON object with these fields:
- title
- inventors
- assignees
- pub_date_application
- pub_date_publication
- pub_date_foreign
- classification
- industrial_field

Field rules:
- title: string in the original language, do not translate.
- inventors: list of objects with fields name and address, or null.
- assignees: list of objects with fields name and address, or null.
- pub_date_application / pub_date_publication / pub_date_foreign: YYYY-MM-DD or null.
- classification: string or null.
- industrial_field: short field/category exactly as written in the document, or null.

General rules:
- Output ONLY one valid JSON object.
- If a field is missing, use null.
- Keep names and addresses exactly as written when possible.
- Do not translate title or industrial_field.
- If several publication dates appear, choose the oldest publication date.
- pub_date_application must not be later than pub_date_publication.
- pub_date_foreign must not be later than pub_date_application.
- Fill industrial_field only when it is explicitly written after the classification.

CRITICAL — Inventors:
- Extract every person explicitly named as inventor.
- Scan beyond the header: inventors may appear after phrases such as "ist als Erfinder genannt worden" or "Als Erfinder benannt".
- If an inventor address is missing, keep the inventor with address = null.
- Never omit an inventor because the address is missing.
- Do not include attorneys, agents, representatives, or assignees as inventors.

CRITICAL — Assignees (high precision, conservative):
- Assignees are rights-holders/applicants explicitly identified in the bibliographic header.
- Use assignees ONLY when the document clearly presents a separate rights-holder/applicant/owner.
- Keywords that may indicate assignees: "Anmelder", "Anmelderin", "Patentanmelder", "Patentinhaber", "Inhaber", "assignee", "cessionnaire", "übertragen an".
- Never copy "Vertreter", "Patentanwalt", attorney, agent, or representative into assignees.
- Never infer an assignee from the technical description body.
- If you are unsure whether an assignee exists, prefer assignees = null.

VERY IMPORTANT — old German patent pattern:
- Many documents contain:
  "Anmelder: <person name>"
  and later
  "<same person> ist als Erfinder genannt worden"
  or
  "Als Erfinder benannt: <same person>"
- In that case, DO NOT copy that natural person into assignees.
- If the only applicant named is the same natural person who is also the inventor, return assignees = null.
- Keep that person only in inventors.

Assignee self-check before final output:
- Is each assignee explicitly shown as applicant/owner in the header?
- Is any assignee merely the inventor copied again with no separate legal entity? If yes, set assignees to null for that case.
- Did you remove duplicate assignee names?
- Did you avoid representatives and patent attorneys?

Example 1:
{{
  "title": "Drehkolbenmaschine mit bewegbaren Kolben",
  "inventors": [
    {{"name": "Dipl.-Ing. Paul Schmidt", "address": "München 54, Riesstr. 18"}}
  ],
  "assignees": null,
  "pub_date_application": "1955-05-11",
  "pub_date_publication": "1957-01-17",
  "pub_date_foreign": null,
  "classification": "88b",
  "industrial_field": null
}}

Example 2:
{{
  "title": "Vorrichtung zum kontinuierlichen Herstellen von Brotteig",
  "inventors": [
    {{"name": "David King Baker", "address": "Peterborough, Northants (Großbritannien)"}}
  ],
  "assignees": [
    {{"name": "Baker Perkins Holdings Limited", "address": "Peterborough, Norihants (Großbritannien)"}}
  ],
  "pub_date_application": "1960-10-26",
  "pub_date_publication": "1964-01-30",
  "pub_date_foreign": "1959-10-27",
  "classification": "2b-4",
  "industrial_field": null
}}

Example 3:
{{
  "title": "Verfahren zur biologischen Reinigung von Abwasser und Rücklaufschlamm",
  "inventors": [
    {{"name": "Dr.-Ing. Hellmut Geiger", "address": "Karlsruhe, Hardeckstr. 3"}},
    {{"name": "Dr.-Ing. Plümer", "address": "Viersen (Rhld.)"}}
  ],
  "assignees": null,
  "pub_date_application": "1958-10-20",
  "pub_date_publication": "1962-06-20",
  "pub_date_foreign": null,
  "classification": "C 02",
  "industrial_field": null
}}

Text:
{text}
"""

PROMPT_EXTRACTION_V4 = """
You are an assistant that extracts structured bibliographic data from a German patent document.

Return one valid JSON object with these fields:
- title
- inventors
- assignees
- pub_date_application
- pub_date_publication
- pub_date_foreign
- classification
- industrial_field

Field rules:
- title: string in the original language, do not translate.
- inventors: list of objects with fields name and address, or null.
- assignees: list of objects with fields name and address, or null.
- pub_date_application / pub_date_publication / pub_date_foreign: YYYY-MM-DD or null.
- classification: string or null.
- industrial_field: short field/category exactly as written in the document, or null.

General rules:
- Output ONLY one valid JSON object.
- If a field is missing, use null.
- Keep names and addresses exactly as written when possible.
- Do not translate title or industrial_field.
- Fill industrial_field only when it is explicitly written after the classification.

CRITICAL — Inventors:
- Extract every person explicitly named as inventor.
- Scan beyond the header: inventors may appear after phrases such as "ist als Erfinder genannt worden" or "Als Erfinder benannt".
- If an inventor address is missing, keep the inventor with address = null.
- Never omit an inventor because the address is missing.
- Do not include attorneys, agents, representatives, or assignees as inventors.

CRITICAL — Assignees (high precision, conservative):
- Assignees are rights-holders/applicants explicitly identified in the bibliographic header.
- Use assignees ONLY when the document clearly presents a separate rights-holder/applicant/owner.
- Keywords that may indicate assignees: "Anmelder", "Anmelderin", "Patentanmelder", "Patentinhaber", "Inhaber", "assignee", "cessionnaire", "übertragen an".
- Never copy "Vertreter", "Patentanwalt", attorney, agent, or representative into assignees.
- Never infer an assignee from the technical description body.
- If you are unsure whether an assignee exists, prefer assignees = null.

VERY IMPORTANT — old German patent pattern:
- Many documents contain:
  "Anmelder: <person name>"
  and later
  "<same person> ist als Erfinder genannt worden"
  or
  "Als Erfinder benannt: <same person>"
- In that case, DO NOT copy that natural person into assignees.
- If the only applicant named is the same natural person who is also the inventor, return assignees = null.
- Keep that person only in inventors.

CRITICAL — Dates (high precision, label-driven):
- Extract dates from the bibliographic header only. Ignore dates from the technical description body.
- Map each date by its label, not by global oldest/newest date selection.
- pub_date_application = filing/application date.
- pub_date_publication = publication/issue date of this German document.
- pub_date_foreign = earliest true foreign priority date, otherwise null.

Date label mapping:
- Use as pub_date_application when clearly tied to labels such as:
  "Anmeldetag", "Anmeldung", "angemeldet am", "Patentiert im Deutschen Reiche vom ... ab".
- Use as pub_date_publication when clearly tied to labels such as:
  "Ausgabe der Auslegeschrift", "Auslegeschrift:", "Patentschrift vom", "Offenlegungstag", "Bekanntmachung".
- Use as pub_date_foreign ONLY when the document explicitly states a foreign priority, foreign filing, or foreign country-linked priority.

Date safety rules:
- If a date is incomplete, OCR-corrupted, or uncertain, use null instead of guessing.
- Do not invent a missing year or day from a neighboring date.
- If several publication dates appear, choose the one explicitly attached to the publication label.
- If several foreign priorities appear, choose the earliest true foreign priority date.
- Exhibition/show/fair dates are NOT foreign priority dates.
- "Schaustellung", "Ausstellung", "Messe", "Deutsche Industriemesse" are NOT pub_date_foreign unless a real foreign priority is also explicitly stated.
- pub_date_application must not be later than pub_date_publication.
- pub_date_foreign must not be later than pub_date_application.

Assignee self-check before final output:
- Is each assignee explicitly shown as applicant/owner in the header?
- Is any assignee merely the inventor copied again with no separate legal entity? If yes, set assignees to null for that case.
- Did you remove duplicate assignee names?
- Did you avoid representatives and patent attorneys?

Date self-check before final output:
- Did you assign each date from its own label, rather than from overall chronology alone?
- Did you avoid using exhibition/show/fair dates as foreign priority?
- Did you avoid using a publication date as application date, or vice versa?
- If a printed date is unreadable or incomplete, did you set it to null?

Example 1:
{{
  "title": "Drehkolbenmaschine mit bewegbaren Kolben",
  "inventors": [
    {{"name": "Dipl.-Ing. Paul Schmidt", "address": "München 54, Riesstr. 18"}}
  ],
  "assignees": null,
  "pub_date_application": "1955-05-11",
  "pub_date_publication": "1957-01-17",
  "pub_date_foreign": null,
  "classification": "88b",
  "industrial_field": null
}}

Example 2:
{{
  "title": "Vorrichtung zum kontinuierlichen Herstellen von Brotteig",
  "inventors": [
    {{"name": "David King Baker", "address": "Peterborough, Northants (Großbritannien)"}}
  ],
  "assignees": [
    {{"name": "Baker Perkins Holdings Limited", "address": "Peterborough, Norihants (Großbritannien)"}}
  ],
  "pub_date_application": "1960-10-26",
  "pub_date_publication": "1964-01-30",
  "pub_date_foreign": "1959-10-27",
  "classification": "2b-4",
  "industrial_field": null
}}

Example 3:
{{
  "title": "Gabel für Verladezwecke",
  "inventors": [
    {{"name": "Erwin Baas", "address": "Hamburg-Hochkamp, Up de Schanz 66"}}
  ],
  "assignees": null,
  "pub_date_application": "1952-10-25",
  "pub_date_publication": "1957-03-28",
  "pub_date_foreign": null,
  "classification": "81 e",
  "industrial_field": null
}}

Text:
{text}
"""

PROMPTS = {
  "v1": PROMPT_EXTRACTION_V1,
  "v2": PROMPT_EXTRACTION_V2,
  "v3": PROMPT_EXTRACTION_V3,
  "v4": PROMPT_EXTRACTION_V4,
}

# PROMPT_EXTRACTION = """You are an assistant extracting structured bibliographic data from a patent document.

# Your task:
# Extract the following fields as a single valid JSON object:

# - title (string, original language)
# - inventors (list of objects with fields: name, address)
# - assignees (list of objects with fields: name, address)
# - pub_date_application (YYYY-MM-DD or null)
# - pub_date_publication (YYYY-MM-DD or null)
# - pub_date_foreign (YYYY-MM-DD or null)
# - classification (string or null)
# - industrial_field (short English category)

# Rules:
# - Always output valid JSON.
# - If a field is missing, use null.
# - Each inventor/assignee must have both name and address fields if possible.
# - Addresses must be translated in English, formatted as "City (Country)".
# - Dates must be in YYYY-MM-DD format.
# - The title must NOT be translated.

# Example output:
# {{
#   "title": "Dispositif de chauffage solaire à collecteurs modulaires",
#   "inventors": [
#     {{"name": "Dr. Alice Montreux", "address": "Geneva (Switzerland)"}},
#     {{"name": "Marc-André Keller", "address": "Zurich (Switzerland)"}}
#   ],
#   "assignees": [
#     {{"name": "HelioTech SA", "address": "Lausanne (Switzerland)"}}
#   ],
#   "pub_date_application": "1974-06-14",
#   "pub_date_publication": "1976-02-02",
#   "pub_date_foreign": null,
#   "classification": "15b",
#   "industrial_field": "Renewable energy systems"
# }}


# Text:
# {text}
# """
# PROMPT_EXTRACTION = """You are an assistant extracting structured bibliographic data from a patent document.

# Your task:
# 1. Identify the patent number/code → this is the "identifier".
# 2. Identify the invention title (the actual name of the invention, not administrative labels).
# 3. Extract inventors' names.
# 4. Extract assignee (the company owning the patent).
# 5. Extract relevant dates:
#    - pub_date_application = filing/application date
#    - pub_date_publication = publication date
#    - pub_date_foreign = foreign priority date (if any)
# 6. Extract address: the location of inventor(s) or assignee(s). Translate it in English.
# 7. Infer the industrial_field in English as a concise category.

# Constraints:
# - Use exactly these fields:
#   identifier, title, inventors, assignee, pub_date_application, pub_date_publication, pub_date_foreign, address, industrial_field
# - If data is missing, put null.
# - Dates must be in YYYY-MM-DD.
# - Title must stay in the original language (do NOT translate).
# - Adress must be in english.
# - industrial_field must be a short category in English (e.g. "Electronics", "Chemistry", "Food processing machinery").
# - The output must be one valid JSON object and nothing else.

# Text:
# {text}

# You are an assistant extracting structured bibliographic data from a patent document.

# Steps you must follow:
# 1. Extract the real patent number/code → this is the "identifier". Use formats like "CH-132767" or "wo 1327/67". Ignore administrative labels ("Mémoire exposé", etc.).
# 2. Find the invention title. This is the name of the object/procedure described (e.g. "Horloge électronique"). Do NOT pick the first header or administrative text.
# 3. Extract inventors' names as a single string. If multiple, separate with "; ".
# 4. Extract the assignee (company owning the patent, NO address, NO city or NO country).
# 5. Extract the following dates:
#    - pub_date_application = filing/application date
#    - pub_date_publication = publication date
#    - pub_date_foreign = foreign priority date (if any)
# 6. Extract the address: inventor or assignee location(s). Format as "City (Country)". If multiple, separate with "; ". Translate it in English.
# 7. Extract industrial_field: a short English category (3–4 words max, e.g. "Electronics", "Medical devices", "Food processing machinery"). If no field is obvious, infer it from the description.

# Constraints:
# - Use exactly these fields:
#   identifier, title, inventors, assignee, pub_date_application, pub_date_publication, pub_date_foreign, address, industrial_field
# - Do not add extra fields.
# - If a field is not found, set its value to null.
# - Dates must be formatted as YYYY-MM-DD.
# - Title must be in the original language (do NOT translate).
# - Adress must be in English.
# - Output must be one valid JSON object and nothing else.

# Text:
# {text}
# """

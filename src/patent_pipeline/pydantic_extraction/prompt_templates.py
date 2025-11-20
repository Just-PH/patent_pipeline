PROMPT_EXTRACTION = """You are an assistant extracting structured bibliographic data from a patent document.

Your task:
Extract the following fields as a single valid JSON object:

- title (string, original language)
- inventors (list of objects with fields: name, address)
- assignees (list of objects with fields: name, address)
- pub_date_application (YYYY-MM-DD or null)
- pub_date_publication (YYYY-MM-DD or null)
- pub_date_foreign (YYYY-MM-DD or null)
- classification (string or null)
- industrial_field (short class category, in original language)

Rules:
- Always output valid JSON.
- If a field is missing, use null.
- Each inventor/assignee must have both name and address fields if possible.
- Addresses, formatted as in the document.
- Dates must be in YYYY-MM-DD format.
- The title must NOT be translated.
- The industrial_field must NOT be translated.

Example output:
{{
  "title": "Dispositif de chauffage solaire à collecteurs modulaires",
  "inventors": [
    {{"name": "Dr. Alice Montreux", "address": "Geneva"}},
    {{"name": "Marc-André Keller", "address": "Zurich"}}
  ],
  "assignees": [
    {{"name": "HelioTech SA", "address": "Lausanne (Switzerland)"}}
  ],
  "pub_date_application": "1974-06-14",
  "pub_date_publication": "1976-02-02",
  "pub_date_foreign": null,
  "classification": "15b",
  "industrial_field": "Renewable energy systems"
}}


Text:
{text}
"""


PROMPT_EXTRACTION = """
Du bist ein Assistent, der strukturierte bibliographische Daten aus einem deutschen Patentdokument extrahiert.

Deine Aufgabe:
Extrahiere die folgenden Felder und gib sie als ein einzelnes, gültiges JSON-Objekt zurück:

- title (string, im Originaltext, NICHT übersetzen)
- inventors (Liste von Objekten mit Feldern: name, address)
- assignees (Liste von Objekten mit Feldern: name, address)
- pub_date_application (YYYY-MM-DD oder null)
- pub_date_publication (YYYY-MM-DD oder null)
- pub_date_foreign (YYYY-MM-DD oder null)
- classification (string oder null)
- industrial_field (kurze Klassifikationskategorie, im Originaltext, NICHT übersetzen)

Regeln:
- Gib IMMER ein gültiges JSON-Objekt aus.
- Wenn ein Feld fehlt, verwende null.
- Jeder Erfinder (inventor) und Anmelder (assignee) soll nach Möglichkeit sowohl name als auch address enthalten.
- Adressen sollen so formatiert sein, wie sie im Dokument erscheinen.
- Datumsangaben müssen im Format YYYY-MM-DD stehen.
- Der Titel und das industrielle Fachgebiet dürfen NICHT übersetzt oder verändert werden.
- Verwende keine zusätzlichen Kommentare oder Erklärungen außerhalb des JSON.
- Falls im Dokument mehrere Veröffentlichungsdaten erscheinen, wähle GRUNDSÄTZLICH das ÄLTESTE Datum als pub_date_publication.
- Trage das Feld industrial_field NUR ein, wenn es im Dokument EXPLIZIT nach der Klassifikationsnummer angegeben ist; andernfalls verwende null.


Beispielausgabe:
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

Text:
{text}
"""


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

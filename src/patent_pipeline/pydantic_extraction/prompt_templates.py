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
- industrial_field (short English category)

Rules:
- Always output valid JSON.
- If a field is missing, use null.
- Each inventor/assignee must have both name and address fields if possible.
- Addresses must be translated in English, formatted as "City (Country)".
- Dates must be in YYYY-MM-DD format.
- The title must NOT be translated.

Example output:
{{
  "title": "Dispositif de chauffage solaire à collecteurs modulaires",
  "inventors": [
    {{"name": "Dr. Alice Montreux", "address": "Geneva (Switzerland)"}},
    {{"name": "Marc-André Keller", "address": "Zurich (Switzerland)"}}
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

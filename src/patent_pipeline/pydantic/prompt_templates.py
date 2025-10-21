PROMPT_EXTRACTION = """You are an information extraction model for patents.
Return ONLY one valid JSON object with fields:
{
  "identifier": str,
  "title": Optional[str],
  "inventors": Optional[str],
  "assignee": Optional[str],
  "pub_date_application": Optional[str],  // YYYY-MM-DD
  "pub_date_publication": Optional[str],
  "pub_date_foreign": Optional[str],
  "address": Optional[str],
  "industrial_field": Optional[str]
}
Rules:
- Output JSON only (no prose).
- Dates are YYYY-MM-DD.
- 'title' stays original language.
- 'address' and 'industrial_field' short in English.

Text:
{text}
"""

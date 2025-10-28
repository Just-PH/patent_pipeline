# 📄 src/patent_pipeline/pydantic/models.py
from datetime import date, datetime
from typing import Optional, List, Union
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------

class Inventor(BaseModel):
    name: str = Field(..., description="Inventor's full name")
    address: Optional[str] = Field(None, description="City and country, e.g. 'Zurich (Switzerland)'")


class Assignee(BaseModel):
    name: str = Field(..., description="Company or assignee name")
    address: Optional[str] = Field(None, description="Headquarters or origin location")


# ---------------------------------------------------------------------
# Main models
# ---------------------------------------------------------------------

class PatentMetadata(BaseModel):
    """
    Structured patent metadata extracted from OCR text.
    """
    title: Optional[str] = Field(None, description="Title of the invention (original language)")

    # richer fields
    inventors: Optional[List[Inventor]] = Field(None, description="List of inventors with name and address")
    assignees: Optional[List[Assignee]] = Field(None, description="List of assignees (companies or individuals)")

    pub_date_application: Optional[date] = Field(None, description="Application filing date (YYYY-MM-DD)")
    pub_date_publication: Optional[date] = Field(None, description="Publication date (YYYY-MM-DD)")
    pub_date_foreign: Optional[date] = Field(None, description="Foreign priority date (if any)")

    classification: Optional[str] = Field(None, description="Patent classification code (e.g., 'G04C 17/00')")
    industrial_field: Optional[str] = Field(None, description="Short English category (e.g. 'Electronics')")

    # ---------------------------------------------------------------
    # Validators
    # ---------------------------------------------------------------
    @field_validator("pub_date_application", "pub_date_publication", "pub_date_foreign", mode="before")
    def parse_date(cls, v):
        """Convert strings like '1926-01-15' or '15 Jan 1926' to date objects."""
        if isinstance(v, date):
            return v
        if not v:
            return None
        try:
            return date.fromisoformat(v)
        except Exception:
            try:
                return datetime.strptime(v, "%d %b %Y").date()
            except Exception:
                return None

    @field_validator("inventors", "assignees", mode="before")
    def clean_empty_entities(cls, v):
        """
        Remove invalid or null entries from inventors/assignees lists.
        """
        if not v:
            return None
        if isinstance(v, list):
            cleaned = []
            for item in v:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                # Skip if name is missing or None
                if not name:
                    continue
                cleaned.append(item)
            return cleaned or None
        return None


class PatentExtraction(BaseModel):
    """Full OCR → LLM extraction result for one patent document."""
    ocr_text: str = Field(..., description="Raw OCR text")
    model: str = Field(..., description="LLM model name (e.g. 'mlx-community/Mistral-7B-Instruct')")
    prediction: PatentMetadata = Field(..., description="Structured metadata extracted from OCR text")

from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Inventor(BaseModel):
    name: str = Field(..., description="Inventor full name.")
    address: Optional[str] = Field(None, description="Inventor address as written in the document.")


class Assignee(BaseModel):
    name: str = Field(..., description="Assignee or company name.")
    address: Optional[str] = Field(None, description="Assignee address as written in the document.")


class PatentMetadata(BaseModel):
    title: Optional[str] = Field(None, description="Title of the invention in the original language.")
    inventors: Optional[List[Inventor]] = Field(None, description="Inventor list.")
    assignees: Optional[List[Assignee]] = Field(None, description="Assignee list.")
    pub_date_application: Optional[date] = Field(None, description="Application date as YYYY-MM-DD.")
    pub_date_publication: Optional[date] = Field(None, description="Publication date as YYYY-MM-DD.")
    pub_date_foreign: Optional[date] = Field(None, description="Foreign priority date as YYYY-MM-DD.")
    classification: Optional[str] = Field(None, description="Patent classification.")
    industrial_field: Optional[str] = Field(None, description="Industrial field or category.")

    @field_validator("pub_date_application", "pub_date_publication", "pub_date_foreign", mode="before")
    @classmethod
    def parse_date(cls, value):
        if isinstance(value, date):
            return value
        if not value:
            return None
        try:
            return date.fromisoformat(value)
        except Exception:
            try:
                return datetime.strptime(value, "%d %b %Y").date()
            except Exception:
                return None

    @field_validator("inventors", "assignees", mode="before")
    @classmethod
    def clean_empty_entities(cls, value):
        if not value:
            return None
        if isinstance(value, list):
            cleaned = []
            for item in value:
                if not isinstance(item, dict):
                    continue
                if not item.get("name"):
                    continue
                cleaned.append(item)
            return cleaned or None
        return None


class PatentExtraction(BaseModel):
    ocr_text: str = Field(..., description="Raw OCR text.")
    model: str = Field(..., description="Model used for extraction.")
    prediction: PatentMetadata = Field(..., description="Structured prediction.")


__all__ = ["Assignee", "Inventor", "PatentExtraction", "PatentMetadata"]

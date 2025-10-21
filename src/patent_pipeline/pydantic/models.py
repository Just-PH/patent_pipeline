from datetime import date
from typing import Optional
from pydantic import BaseModel

class PatentMetadata(BaseModel):
    identifier: str
    title: Optional[str] = None
    inventors: Optional[str] = None
    assignee: Optional[str] = None
    pub_date_application: Optional[date] = None
    pub_date_publication: Optional[date] = None
    pub_date_foreign: Optional[date] = None
    address: Optional[str] = None
    industrial_field: Optional[str] = None

class PatentExtraction(BaseModel):
    ocr_text: str
    model: str
    prediction: PatentMetadata

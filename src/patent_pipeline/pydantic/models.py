from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


class PatentMetadata(BaseModel):
    """
    Représente les métadonnées structurées d'un brevet extraites depuis du texte OCR.
    """
    identifier: str = Field(..., description="Identifiant du brevet, ex: 'CH-16799-A'")
    title: Optional[str] = Field(None, description="Titre du brevet")
    inventors: Optional[str] = Field(None, description="Nom(s) des inventeurs listés")
    assignee: Optional[str] = Field(None, description="Nom du déposant / détenteur du brevet")
    pub_date_application: Optional[date] = Field(None, description="Date de dépôt de la demande de brevet")
    pub_date_publication: Optional[date] = Field(None, description="Date de publication du brevet")
    pub_date_foreign: Optional[date] = Field(None, description="Date de publication à l’étranger (si applicable)")
    address: Optional[str] = Field(None, description="Adresse ou lieu de l’inventeur ou du déposant")
    industrial_field: Optional[str] = Field(None, description="Champ industriel ou domaine d’application")


class PatentExtraction(BaseModel):
    """Résultat complet d’un run d’extraction depuis un texte OCR."""
    ocr_text: str = Field(..., description="Texte OCR brut du brevet")
    model: str = Field(..., description="Nom du modèle LLM utilisé (ex: mistral:7b-instruct)")
    prediction: PatentMetadata = Field(..., description="Résultat structuré de l’extraction")

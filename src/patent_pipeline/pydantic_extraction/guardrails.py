from __future__ import annotations

from typing import Literal, Optional

from .models import PatentMetadata
from . import postprocess


GuardrailProfile = Literal["auto", "off", "de_legacy_self_applicant"]

GUARDRAIL_PROFILES = {"auto", "off", "de_legacy_self_applicant"}


def looks_like_de_legacy_self_applicant_case(ocr_text: str) -> bool:
    text = str(ocr_text or "").lower()
    return (
        "anmelder" in text
        and (
            "ist als erfinder genannt worden" in text
            or "als erfinder benannt" in text
            or "ist als erfinder benannt" in text
        )
    )


def should_apply_de_legacy_self_applicant_guardrail(
    *,
    prompt_id: Optional[str],
    guardrail_profile: str,
) -> bool:
    if guardrail_profile == "off":
        return False
    if guardrail_profile == "de_legacy_self_applicant":
        return True
    return prompt_id in {"v3", "v4"}


def apply_de_legacy_self_applicant_guardrail(
    metadata: PatentMetadata,
    ocr_text: str,
    *,
    prompt_id: Optional[str],
    guardrail_profile: str,
    verbose: bool = True,
) -> PatentMetadata:
    if not should_apply_de_legacy_self_applicant_guardrail(
        prompt_id=prompt_id,
        guardrail_profile=guardrail_profile,
    ):
        return metadata

    inventors = metadata.inventors or []
    assignees = metadata.assignees or []
    if len(inventors) != 1 or len(assignees) != 1:
        return metadata

    inventor_name = str(inventors[0].name or "").strip()
    assignee_name = str(assignees[0].name or "").strip()
    if not inventor_name or not assignee_name:
        return metadata
    if postprocess.is_company_name(assignee_name):
        return metadata
    if not postprocess.same_person_identity(inventor_name, assignee_name):
        return metadata
    if not looks_like_de_legacy_self_applicant_case(ocr_text):
        return metadata

    corrected = metadata.model_dump(mode="json")
    corrected["assignees"] = None
    if verbose:
        print("🔧 Prompt correction: dropped assignee duplicated from sole inventor")
    return PatentMetadata(**corrected)


def apply_guardrails(
    metadata: PatentMetadata,
    ocr_text: str,
    *,
    prompt_id: Optional[str],
    guardrail_profile: str,
    verbose: bool = True,
) -> PatentMetadata:
    return apply_de_legacy_self_applicant_guardrail(
        metadata,
        ocr_text,
        prompt_id=prompt_id,
        guardrail_profile=guardrail_profile,
        verbose=verbose,
    )

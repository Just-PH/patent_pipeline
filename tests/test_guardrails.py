import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from patent_pipeline.pydantic_extraction import guardrails
from patent_pipeline.pydantic_extraction.models import PatentMetadata


class TestGuardrails(unittest.TestCase):
    def test_auto_profile_applies_for_v4(self):
        metadata = PatentMetadata(
            title="Drehkolbenmaschine",
            inventors=[{"name": "Dipl.-Ing. Paul Schmidt", "address": "München 54, Riesstr. 18"}],
            assignees=[{"name": "Dipl.-Ing. Paul Schmidt", "address": "München 54, Riesstr. 18"}],
            pub_date_application="1955-05-11",
            pub_date_publication="1957-01-17",
        )
        text = (
            "Anmelder: Dipl.-Ing. Paul Schmidt, München 54, Riesstr. 18\n"
            "Dipl.-Ing. Paul Schmidt, München, ist als Erfinder genannt worden"
        )

        fixed = guardrails.apply_guardrails(
            metadata,
            text,
            prompt_id="v4",
            guardrail_profile="auto",
            verbose=False,
        )
        self.assertIsNone(fixed.assignees)

    def test_explicit_profile_applies_with_external_prompt(self):
        metadata = PatentMetadata(
            title="Drehkolbenmaschine",
            inventors=[{"name": "Dipl.-Ing. Paul Schmidt", "address": "München 54, Riesstr. 18"}],
            assignees=[{"name": "Dipl.-Ing. Paul Schmidt", "address": "München 54, Riesstr. 18"}],
            pub_date_application="1955-05-11",
            pub_date_publication="1957-01-17",
        )
        text = (
            "Anmelder: Dipl.-Ing. Paul Schmidt, München 54, Riesstr. 18\n"
            "Dipl.-Ing. Paul Schmidt, München, ist als Erfinder genannt worden"
        )

        fixed = guardrails.apply_guardrails(
            metadata,
            text,
            prompt_id=None,
            guardrail_profile="de_legacy_self_applicant",
            verbose=False,
        )
        self.assertIsNone(fixed.assignees)

    def test_off_profile_disables_guardrail(self):
        metadata = PatentMetadata(
            title="Drehkolbenmaschine",
            inventors=[{"name": "Dipl.-Ing. Paul Schmidt", "address": "München 54, Riesstr. 18"}],
            assignees=[{"name": "Dipl.-Ing. Paul Schmidt", "address": "München 54, Riesstr. 18"}],
            pub_date_application="1955-05-11",
            pub_date_publication="1957-01-17",
        )
        text = (
            "Anmelder: Dipl.-Ing. Paul Schmidt, München 54, Riesstr. 18\n"
            "Dipl.-Ing. Paul Schmidt, München, ist als Erfinder genannt worden"
        )

        fixed = guardrails.apply_guardrails(
            metadata,
            text,
            prompt_id="v4",
            guardrail_profile="off",
            verbose=False,
        )
        self.assertIsNotNone(fixed.assignees)


if __name__ == "__main__":
    unittest.main()

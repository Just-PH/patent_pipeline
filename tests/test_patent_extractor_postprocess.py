import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from patent_pipeline.pydantic_extraction.models import PatentMetadata
from patent_pipeline.pydantic_extraction import postprocess


class TestPatentExtractorPostprocess(unittest.TestCase):
    def test_parse_and_validate_dedupes_duplicate_assignees(self):
        raw = """
        {
          "title": "Beispiel",
          "inventors": [{"name": "Alice Example", "address": "Paris"}],
          "assignees": [
            {"name": "Müller & Mann", "address": "Berlin"},
            {"name": "Müller & Mann", "address": "Berlin"}
          ],
          "pub_date_application": "1920-01-01",
          "pub_date_publication": "1920-02-01",
          "pub_date_foreign": null,
          "classification": "G01",
          "industrial_field": null
        }
        """
        meta = postprocess.parse_and_validate(raw)
        self.assertEqual(len(meta.assignees or []), 1)
        self.assertEqual(meta.assignees[0].name, "Müller & Mann")

    def test_parse_and_validate_normalizes_legacy_fields_and_moves_companies(self):
        raw = """
        {
          "title": "Beispiel",
          "inventor": [
            {"name": "ACME GmbH", "address": "Berlin"},
            {"name": "Alice Example", "address": "Paris"}
          ],
          "assignee": null,
          "class": "G01",
          "pub_date_application": "1920-01-01",
          "pub_date_publication": "1920-02-01",
          "pub_date_foreign": null,
          "industrial_field": null
        }
        """
        meta = postprocess.parse_and_validate(raw)
        self.assertEqual(meta.classification, "G01")
        self.assertEqual(len(meta.inventors or []), 1)
        self.assertEqual(meta.inventors[0].name, "Alice Example")
        self.assertEqual(len(meta.assignees or []), 1)
        self.assertEqual(meta.assignees[0].name, "ACME GmbH")

    def test_merge_metadata_candidates_enforces_date_order_under_vote_policy(self):
        candidates = [
            PatentMetadata(
                title="Title A",
                pub_date_application="1920-03-01",
                pub_date_publication="1920-02-01",
            ),
            PatentMetadata(
                title="Title A",
                pub_date_application="1920-01-01",
                pub_date_publication="1920-02-01",
            ),
        ]
        merged = postprocess.merge_metadata_candidates(candidates, policy="prefer_first")
        self.assertIsNone(merged.pub_date_application)
        self.assertEqual(str(merged.pub_date_publication), "1920-02-01")


if __name__ == "__main__":
    unittest.main()

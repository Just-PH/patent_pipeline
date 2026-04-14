import unittest
from pathlib import Path

from patent_extraction.extractor import PatentExtractor
from patent_extraction.profiles import DEFAULT_PROFILE_NAME, load_profile, resolve_profile_path
from patent_extraction.prompt_templates import PROMPT_EXTRACTION_V4


class PatentExtractionProfilesTests(unittest.TestCase):
    def test_can_load_packaged_de_profile(self):
        profile = load_profile(name=DEFAULT_PROFILE_NAME)
        self.assertEqual(profile.name, "de_legacy_v4")
        self.assertEqual(profile.extraction.backend, "vllm")
        self.assertEqual(profile.extraction.strategy.name, "two_pass_targeted")
        self.assertTrue(profile.extraction.vllm.enable_prefix_caching)
        self.assertEqual(profile.extraction.guardrail_profile, "de_legacy_self_applicant")
        self.assertIsNotNone(profile.extraction.prompt_path)
        self.assertTrue(profile.extraction.prompt_path.exists())
        self.assertIn("German patent document", profile.read_prompt_text())

    def test_profile_overrides_apply_cleanly(self):
        profile = load_profile(
            name=DEFAULT_PROFILE_NAME,
            overrides={
                "strategy": "baseline",
                "doc_batch_size": 64,
                "save_raw_output": True,
                "guardrail_profile": "de_legacy_self_applicant",
                "quantization": "bitsandbytes",
            },
        )
        self.assertEqual(profile.extraction.strategy.name, "baseline")
        self.assertEqual(profile.extraction.vllm.doc_batch_size, 64)
        self.assertTrue(profile.extraction.save_raw_output)
        self.assertEqual(profile.extraction.guardrail_profile, "de_legacy_self_applicant")
        self.assertEqual(profile.extraction.vllm.quantization, "bitsandbytes")

    def test_resolve_profile_path_points_to_json_definition(self):
        path = resolve_profile_path(DEFAULT_PROFILE_NAME)
        self.assertEqual(path.name, "de_legacy_v4.json")
        self.assertEqual(path.suffix, ".json")
        self.assertTrue(path.exists())
        self.assertIsInstance(path, Path)

    def test_packaged_de_prompt_renders_exactly_like_embedded_v4(self):
        profile = load_profile(name=DEFAULT_PROFILE_NAME)
        file_prompt = profile.read_prompt_text()
        self.assertIsNotNone(file_prompt)
        rendered_from_file = PatentExtractor._render_prompt_template(file_prompt, "TEXT")
        rendered_embedded = PatentExtractor._render_prompt_template(PROMPT_EXTRACTION_V4, "TEXT")
        self.assertEqual(rendered_from_file, rendered_embedded)

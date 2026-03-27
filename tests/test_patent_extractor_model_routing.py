import unittest
from unittest.mock import patch
import sys
from pathlib import Path
import types
import importlib

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def _load_module_with_fake_transformers():
    fake = types.ModuleType("transformers")
    fake.AutoConfig = object
    fake.AutoModelForCausalLM = object
    fake.AutoTokenizer = object
    fake.BitsAndBytesConfig = object

    def _fake_pipeline(*args, **kwargs):
        return None

    fake.pipeline = _fake_pipeline
    fake.Mistral3ForCausalLM = object
    fake.Mistral3ForConditionalGeneration = object

    sys.modules["transformers"] = fake
    if "patent_pipeline.pydantic_extraction.runtime" in sys.modules:
        del sys.modules["patent_pipeline.pydantic_extraction.runtime"]
    if "patent_pipeline.pydantic_extraction.patent_extractor" in sys.modules:
        del sys.modules["patent_pipeline.pydantic_extraction.patent_extractor"]
    return importlib.import_module("patent_pipeline.pydantic_extraction.patent_extractor")


pe = _load_module_with_fake_transformers()


class DummyConfig:
    model_type = "qwen2"
    auto_map = None


class DummyMistral3Config:
    model_type = "mistral3"
    auto_map = None


class TestPatentExtractorModelRouting(unittest.TestCase):
    def test_mistral3_uses_causallm_route_when_available(self):
        cfg = DummyMistral3Config()
        with patch.object(pe, "Mistral3ForConditionalGeneration", object()), patch.object(
            pe, "Mistral3ForCausalLM", object()
        ):
            route = pe.PatentExtractor._resolve_model_route(cfg)
        self.assertEqual(route["route_name"], "Mistral3ForCausalLM")
        self.assertEqual(route["pipeline_task"], "text-generation")

    def test_mistral3_falls_back_to_conditional_direct_generate(self):
        cfg = DummyMistral3Config()
        with patch.object(pe, "Mistral3ForConditionalGeneration", object()), patch.object(
            pe, "Mistral3ForCausalLM", None
        ):
            route = pe.PatentExtractor._resolve_model_route(cfg)
        self.assertEqual(route["route_name"], "Mistral3ForConditionalGeneration")
        self.assertEqual(route["pipeline_task"], "direct-generate")
        self.assertFalse(route["use_pipeline"])

    def test_classic_causallm_route_unchanged(self):
        cfg = DummyConfig()
        route = pe.PatentExtractor._resolve_model_route(cfg)
        self.assertEqual(route["route_name"], "AutoModelForCausalLM")
        self.assertEqual(route["pipeline_task"], "text-generation")


if __name__ == "__main__":
    unittest.main()

import importlib
import sys
import types
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def _load_module_with_fake_transformers():
    fake = types.ModuleType("transformers")
    fake.AutoConfig = object
    fake.AutoModelForCausalLM = object
    fake.AutoTokenizer = object
    fake.pipeline = lambda *args, **kwargs: None
    fake.Mistral3ForCausalLM = object
    fake.Mistral3ForConditionalGeneration = object
    sys.modules["transformers"] = fake
    if "patent_pipeline.pydantic_extraction.patent_extractor" in sys.modules:
        del sys.modules["patent_pipeline.pydantic_extraction.patent_extractor"]
    return importlib.import_module("patent_pipeline.pydantic_extraction.patent_extractor")


pe = _load_module_with_fake_transformers()


class TestPatentExtractorGenerateKwargs(unittest.TestCase):
    def _mk_extractor(self, pipeline_task: str):
        ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
        ex.backend = "pytorch"
        ex.max_new_tokens = 64
        ex.temperature = 0.0
        ex.do_sample = False
        ex.pipeline_task = pipeline_task
        return ex

    def test_text2text_generation_drops_return_full_text(self):
        ex = self._mk_extractor("text2text-generation")
        seen = {}

        def fake_pipe(_prompt, **kwargs):
            seen.update(kwargs)
            return [{"generated_text": "ok"}]

        ex.pipe = fake_pipe
        out = ex._generate("prompt")
        self.assertEqual(out, "ok")
        self.assertNotIn("return_full_text", seen)

    def test_text_generation_keeps_return_full_text(self):
        ex = self._mk_extractor("text-generation")
        seen = {}

        def fake_pipe(_prompt, **kwargs):
            seen.update(kwargs)
            return [{"generated_text": "ok"}]

        ex.pipe = fake_pipe
        out = ex._generate("prompt")
        self.assertEqual(out, "ok")
        self.assertIn("return_full_text", seen)
        self.assertFalse(seen["return_full_text"])
        self.assertNotIn("temperature", seen)

    def test_direct_generate_path_calls_model_generate_without_return_full_text(self):
        ex = self._mk_extractor("direct-generate")
        seen = {}

        class _Tok:
            def __call__(self, _prompt, return_tensors=None):
                import torch

                return {"input_ids": torch.tensor([[1, 2]])}

            def decode(self, ids, skip_special_tokens=True):
                return "ok"

        class _Model:
            device = "cpu"

            def generate(self, **kwargs):
                import torch

                seen.update(kwargs)
                return torch.tensor([[1, 2, 3]])

        ex.tokenizer = _Tok()
        ex.model = _Model()
        ex.pipe = None
        out = ex._generate("prompt")
        self.assertEqual(out, "ok")
        self.assertNotIn("return_full_text", seen)


if __name__ == "__main__":
    unittest.main()

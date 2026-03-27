import importlib
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

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

    class _BitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake.BitsAndBytesConfig = _BitsAndBytesConfig
    sys.modules["transformers"] = fake
    for mod_name in [
        "patent_pipeline.pydantic_extraction.runtime",
        "patent_pipeline.pydantic_extraction.patent_extractor",
    ]:
        if mod_name in sys.modules:
            del sys.modules[mod_name]
    return importlib.import_module("patent_pipeline.pydantic_extraction.patent_extractor")


pe = _load_module_with_fake_transformers()


class TestPatentExtractorRuntimeOptions(unittest.TestCase):
    def test_vllm_backend_rejects_pytorch_only_cache_option(self):
        with patch.object(pe.PatentExtractor, "_load_model", lambda self: None):
            with self.assertRaises(ValueError):
                pe.PatentExtractor(backend="vllm", cache_implementation="dynamic")

    def test_build_quantization_config_for_bnb_8bit(self):
        ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
        ex.quantization = "bnb_8bit"

        class _BitsAndBytesConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        with patch.object(pe.runtime_mod, "BitsAndBytesConfig", _BitsAndBytesConfig):
            cfg = ex._build_quantization_config(ex, dtype=torch.bfloat16)
        self.assertEqual(cfg.kwargs, {"load_in_8bit": True})

    def test_build_quantization_config_for_bnb_4bit_uses_compute_dtype(self):
        ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
        ex.quantization = "bnb_4bit"

        class _BitsAndBytesConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        with patch.object(pe.runtime_mod, "BitsAndBytesConfig", _BitsAndBytesConfig):
            cfg = ex._build_quantization_config(ex, dtype=torch.bfloat16)
        self.assertTrue(cfg.kwargs["load_in_4bit"])
        self.assertEqual(cfg.kwargs["bnb_4bit_quant_type"], "nf4")
        self.assertEqual(cfg.kwargs["bnb_4bit_compute_dtype"], torch.bfloat16)

    def test_resolve_attn_implementation_rejects_flash_attention_without_cuda(self):
        ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
        ex.attn_implementation = "flash_attention_2"
        ex.device = "cpu"

        with self.assertRaises(ValueError):
            ex._resolve_attn_implementation(ex, dtype=torch.bfloat16)

    def test_resolve_cache_implementation_rejects_static_without_compiler(self):
        ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
        ex.cache_implementation = "static"

        with patch.object(pe.runtime_mod, "has_c_compiler", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                ex._resolve_cache_implementation(ex)
        self.assertIn("cache_implementation=static", str(ctx.exception))

    def test_resolve_cache_implementation_allows_static_with_compiler(self):
        ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
        ex.cache_implementation = "static"

        with patch.object(pe.runtime_mod, "has_c_compiler", return_value=True):
            self.assertEqual(ex._resolve_cache_implementation(ex), "static")

    def test_extract_from_file_records_error_detail(self):
        ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
        ex.strategy = "baseline"
        ex.prompt_id = None
        ex.prompt_hash = "hash"
        ex.timings = "off"

        def _boom(_ocr_text):
            raise RuntimeError("boom detail")

        ex.extract = _boom

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_text("hello", encoding="utf-8")
            rec = ex.extract_from_file(path)

        self.assertEqual(rec["error"], "exception: RuntimeError")
        self.assertEqual(rec["error_type"], "RuntimeError")
        self.assertEqual(rec["error_detail"], "boom detail")


if __name__ == "__main__":
    unittest.main()

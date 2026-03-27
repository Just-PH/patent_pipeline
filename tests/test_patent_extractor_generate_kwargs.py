import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

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
    fake.pipeline = lambda *args, **kwargs: None
    fake.Mistral3ForCausalLM = object
    fake.Mistral3ForConditionalGeneration = object
    sys.modules["transformers"] = fake
    if "patent_pipeline.pydantic_extraction.runtime" in sys.modules:
        del sys.modules["patent_pipeline.pydantic_extraction.runtime"]
    if "patent_pipeline.pydantic_extraction.patent_extractor" in sys.modules:
        del sys.modules["patent_pipeline.pydantic_extraction.patent_extractor"]
    return importlib.import_module("patent_pipeline.pydantic_extraction.patent_extractor")


pe = _load_module_with_fake_transformers()


class _FakeProgress:
    def __init__(self, iterable=None):
        self.iterable = iterable

    def __iter__(self):
        if self.iterable is None:
            return iter(())
        return iter(self.iterable)

    def update(self, _n=1):
        return None

    def close(self):
        return None


def _fake_tqdm(iterable=None, **kwargs):
    return _FakeProgress(iterable)


class TestPatentExtractorGenerateKwargs(unittest.TestCase):
    def _mk_extractor(self, pipeline_task: str):
        ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
        ex.backend = "pytorch"
        ex.max_new_tokens = 64
        ex.temperature = 0.0
        ex.do_sample = False
        ex.pipeline_task = pipeline_task
        ex.cache_implementation = "auto"
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

    def test_generation_overrides_do_not_mutate_extractor_state(self):
        ex = self._mk_extractor("text-generation")
        seen = {}

        def fake_pipe(_prompt, **kwargs):
            seen.update(kwargs)
            return [{"generated_text": "ok"}]

        ex.pipe = fake_pipe
        out = ex._generate("prompt", temperature_override=0.4, do_sample_override=True)
        self.assertEqual(out, "ok")
        self.assertTrue(seen["do_sample"])
        self.assertEqual(seen["temperature"], 0.4)
        self.assertEqual(ex.temperature, 0.0)
        self.assertFalse(ex.do_sample)

    def test_static_cache_is_forwarded_to_generate(self):
        ex = self._mk_extractor("text-generation")
        ex.cache_implementation = "static"
        seen = {}

        def fake_pipe(_prompt, **kwargs):
            seen.update(kwargs)
            return [{"generated_text": "ok"}]

        ex.pipe = fake_pipe
        with patch.object(pe.runtime_mod, "has_c_compiler", return_value=True):
            out = ex._generate("prompt")
        self.assertEqual(out, "ok")
        self.assertEqual(seen["cache_implementation"], "static")

    def test_vllm_generate_returns_first_candidate_text(self):
        ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
        ex.backend = "vllm"
        ex.max_new_tokens = 64
        ex.temperature = 0.0
        ex.do_sample = False
        seen = {}

        class _Model:
            def generate(self, prompts, sampling_params=None, use_tqdm=False):
                seen["prompts"] = prompts
                seen["sampling_params"] = sampling_params
                seen["use_tqdm"] = use_tqdm
                return [SimpleNamespace(outputs=[SimpleNamespace(text="ok from vllm")])]

        ex.model = _Model()
        with patch.object(pe.runtime_mod, "build_vllm_sampling_params", return_value="sampling"):
            out = ex._generate("prompt")
        self.assertEqual(out, "ok from vllm")
        self.assertEqual(seen["prompts"], ["prompt"])
        self.assertEqual(seen["sampling_params"], "sampling")
        self.assertFalse(seen["use_tqdm"])

    def test_vllm_batch_baseline_extracts_json_from_markdown_fences(self):
        ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
        ex.backend = "vllm"
        ex.strategy = "baseline"
        ex.prompt_template = "{text}"
        ex.prompt_suffix = ""
        ex.max_ocr_chars = 10000
        ex.timings = "off"
        ex.prompt_hash = "hash"
        ex.prompt_id = None
        ex.model_name = "mistral"
        ex.save_strategy_meta = False
        ex.save_raw_output = False
        ex.merge_policy = "prefer_non_null"
        ex.vllm_doc_batch_size = 32
        ex.vllm_sort_by_prompt_length = True
        ex.compute_confidence = pe.PatentExtractor.compute_confidence
        ex._build_vllm_sampling_params = lambda owner: "sampling"

        raw_output = """```json
{
  "title": "Fernschreibsender",
  "inventors": [{"name": "Benno Hausmann", "address": "München"}],
  "assignees": [{"name": "Siemens & Halske Aktiengesellschaft", "address": "München"}],
  "pub_date_application": "1956-03-03",
  "pub_date_publication": "1957-09-05",
  "pub_date_foreign": null,
  "classification": "H 045; 1",
  "industrial_field": null
}
```"""

        class _Model:
            def generate(self, prompts, sampling_params=None, use_tqdm=False):
                return [SimpleNamespace(outputs=[SimpleNamespace(text=raw_output)])]

        ex.model = _Model()

        with TemporaryDirectory() as tmpdir, patch.object(pe, "tqdm", _fake_tqdm):
            txt_dir = Path(tmpdir) / "texts"
            txt_dir.mkdir()
            (txt_dir / "DE-1015044-B.txt").write_text("ocr text", encoding="utf-8")
            out_file = Path(tmpdir) / "preds.jsonl"

            count = ex._batch_extract_vllm_baseline(txt_dir, out_file)
            self.assertEqual(count, 1)

            rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            pred = rows[0]["prediction"]
            self.assertEqual(pred["identifier"], "DE-1015044-B")
            self.assertEqual(pred["title"], "Fernschreibsender")
            self.assertEqual(pred["pub_date_application"], "1956-03-03")
            self.assertEqual(pred["inventors"][0]["name"], "Benno Hausmann")
            self.assertEqual(pred["assignees"][0]["name"], "Siemens & Halske Aktiengesellschaft")

    def test_vllm_batch_baseline_splits_micro_batches_and_preserves_file_order(self):
        ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
        ex.backend = "vllm"
        ex.strategy = "baseline"
        ex.prompt_template = "{text}"
        ex.prompt_suffix = ""
        ex.max_ocr_chars = 10000
        ex.timings = "off"
        ex.prompt_hash = "hash"
        ex.prompt_id = None
        ex.model_name = "mistral"
        ex.save_strategy_meta = False
        ex.save_raw_output = False
        ex.merge_policy = "prefer_non_null"
        ex.vllm_doc_batch_size = 2
        ex.vllm_sort_by_prompt_length = True
        ex.compute_confidence = pe.PatentExtractor.compute_confidence
        ex._build_vllm_sampling_params = lambda owner: "sampling"
        seen_batches = []

        class _Model:
            def generate(self, prompts, sampling_params=None, use_tqdm=False):
                seen_batches.append(list(prompts))
                outs = []
                for prompt in prompts:
                    outs.append(
                        SimpleNamespace(
                            outputs=[
                                SimpleNamespace(
                                    text=json.dumps(
                                        {
                                            "title": prompt,
                                            "inventors": None,
                                            "assignees": None,
                                            "pub_date_application": None,
                                            "pub_date_publication": None,
                                            "pub_date_foreign": None,
                                            "classification": None,
                                            "industrial_field": None,
                                        },
                                        ensure_ascii=False,
                                    )
                                )
                            ]
                        )
                    )
                return outs

        ex.model = _Model()

        with TemporaryDirectory() as tmpdir, patch.object(pe, "tqdm", _fake_tqdm):
            txt_dir = Path(tmpdir) / "texts"
            txt_dir.mkdir()
            (txt_dir / "doc_a.txt").write_text("a", encoding="utf-8")
            (txt_dir / "doc_b.txt").write_text("bbbbb", encoding="utf-8")
            (txt_dir / "doc_c.txt").write_text("ccc", encoding="utf-8")
            out_file = Path(tmpdir) / "preds.jsonl"

            count = ex._batch_extract_vllm_baseline(txt_dir, out_file)
            self.assertEqual(count, 3)

            self.assertEqual(len(seen_batches), 2)
            self.assertEqual([len(x) for x in seen_batches], [2, 1])
            self.assertEqual([len(prompt) for prompt in seen_batches[0]], [5, 3])

            rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([row["file_name"] for row in rows], ["doc_a.txt", "doc_b.txt", "doc_c.txt"])
            self.assertEqual(rows[0]["prediction"]["title"], "a")
            self.assertEqual(rows[1]["prediction"]["title"], "bbbbb")
            self.assertEqual(rows[2]["prediction"]["title"], "ccc")

    def test_vllm_batch_baseline_uses_request_metrics_for_doc_timing(self):
        ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
        ex.backend = "vllm"
        ex.strategy = "baseline"
        ex.prompt_template = "{text}"
        ex.prompt_suffix = ""
        ex.max_ocr_chars = 10000
        ex.timings = "detailed"
        ex.prompt_hash = "hash"
        ex.prompt_id = None
        ex.model_name = "mistral"
        ex.save_strategy_meta = False
        ex.save_raw_output = False
        ex.merge_policy = "prefer_non_null"
        ex.vllm_doc_batch_size = 32
        ex.vllm_sort_by_prompt_length = True
        ex.compute_confidence = pe.PatentExtractor.compute_confidence
        ex._build_vllm_sampling_params = lambda owner: "sampling"

        raw_output = """{
  "title": "Fernschreibsender",
  "inventors": [{"name": "Benno Hausmann", "address": "München"}],
  "assignees": [{"name": "Siemens & Halske Aktiengesellschaft", "address": "München"}],
  "pub_date_application": "1956-03-03",
  "pub_date_publication": "1957-09-05",
  "pub_date_foreign": null,
  "classification": "H 045; 1",
  "industrial_field": null
}"""

        metrics = SimpleNamespace(
            arrival_time=10.0,
            first_scheduled_time=10.2,
            first_token_time=10.5,
            finished_time=11.0,
            time_in_queue=0.2,
            scheduler_time=0.05,
            model_forward_time=0.3,
            model_execute_time=0.4,
        )

        class _Model:
            def generate(self, prompts, sampling_params=None, use_tqdm=False):
                return [SimpleNamespace(outputs=[SimpleNamespace(text=raw_output)], metrics=metrics)]

        ex.model = _Model()

        with TemporaryDirectory() as tmpdir, patch.object(pe, "tqdm", _fake_tqdm):
            txt_dir = Path(tmpdir) / "texts"
            txt_dir.mkdir()
            (txt_dir / "DE-1015044-B.txt").write_text("ocr text", encoding="utf-8")
            out_file = Path(tmpdir) / "preds.jsonl"

            count = ex._batch_extract_vllm_baseline(txt_dir, out_file)
            self.assertEqual(count, 1)

            rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            timing = rows[0]["timing"]
            self.assertEqual(timing["t_vllm_request_s"], 1.0)
            self.assertEqual(timing["t_vllm_queue_s"], 0.2)
            self.assertEqual(timing["t_vllm_ttft_s"], 0.5)
            self.assertEqual(timing["t_vllm_decode_s"], 0.5)
            self.assertEqual(timing["t_vllm_scheduler_s"], 0.05)
            self.assertEqual(timing["t_vllm_model_forward_s"], 0.3)
            self.assertEqual(timing["t_vllm_model_execute_s"], 0.4)
            self.assertIn("t_prebatch_wait_s", timing)
            self.assertGreaterEqual(timing["t_total_file_s"], 1.0)
            self.assertLess(timing["t_total_file_s"], 2.0)


if __name__ == "__main__":
    unittest.main()

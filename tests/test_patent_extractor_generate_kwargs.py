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
    def _mk_vllm_batch_extractor(self, strategy: str):
        ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
        ex.backend = "vllm"
        ex.strategy = strategy
        ex.prompt_template = "{text}"
        ex.prompt_suffix = ""
        ex.max_ocr_chars = 10000
        ex.timings = "off"
        ex.prompt_hash = "hash"
        ex.prompt_id = None
        ex.guardrail_profile = "auto"
        ex.model_name = "mistral"
        ex.save_strategy_meta = False
        ex.save_raw_output = False
        ex.merge_policy = "prefer_non_null"
        ex.vllm_doc_batch_size = 32
        ex.vllm_sort_by_prompt_length = False
        ex.compute_confidence = pe.PatentExtractor.compute_confidence
        ex._build_vllm_sampling_params = lambda owner, **kwargs: "sampling"
        ex.extraction_mode = "auto"
        ex.chunk_size_chars = 7000
        ex.chunk_overlap_chars = 800
        ex.extraction_passes = 2
        ex.header_lines = 1
        ex.targeted_rerun_threshold = 0.6
        ex.self_consistency_n = 2
        ex.self_consistency_temp = 0.2
        return ex

    def _mk_extractor(self, pipeline_task: str):
        ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
        ex.backend = "pytorch"
        ex.max_new_tokens = 64
        ex.temperature = 0.0
        ex.do_sample = False
        ex.pipeline_task = pipeline_task
        ex.cache_implementation = "auto"
        ex.guardrail_profile = "auto"
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

    def test_render_prompt_template_supports_raw_and_escaped_json_braces(self):
        raw_template = 'Example:\n{"title": null}\nText:\n{text}'
        escaped_template = 'Example:\n{{"title": null}}\nText:\n{text}'

        raw_rendered = pe.PatentExtractor._render_prompt_template(raw_template, "hello")
        escaped_rendered = pe.PatentExtractor._render_prompt_template(escaped_template, "hello")

        self.assertIn('{"title": null}', raw_rendered)
        self.assertIn('{"title": null}', escaped_rendered)
        self.assertIn("hello", raw_rendered)
        self.assertIn("hello", escaped_rendered)
        self.assertNotIn('{{"title"', raw_rendered)
        self.assertNotIn('{{"title"', escaped_rendered)

    def test_prompt_v3_drops_self_assignee_for_old_german_self_applicant_pattern(self):
        metadata = pe.PatentMetadata(
            title="Drehkolbenmaschine",
            inventors=[{"name": "Dipl.-Ing. Paul Schmidt", "address": "München 54, Riesstr. 18"}],
            assignees=[{"name": "Dipl.-Ing. Paul Schmidt", "address": "München 54, Riesstr. 18"}],
            pub_date_application="1955-05-11",
            pub_date_publication="1957-01-17",
        )
        text = (
            "Anmelder: Dipl.-Ing. Paul Schmidt, München 54, Riesstr. 18\\n"
            "Dipl.-Ing. Paul Schmidt, München, ist als Erfinder genannt worden"
        )

        for prompt_id in ("v3", "v4"):
            with self.subTest(prompt_id=prompt_id):
                ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
                ex.prompt_id = prompt_id
                ex.guardrail_profile = "auto"
                fixed = ex._apply_de_legacy_self_applicant_guardrail(metadata, text)
                self.assertIsNone(fixed.assignees)
                self.assertEqual(fixed.inventors[0].name, "Dipl.-Ing. Paul Schmidt")

    def test_prompt_v3_keeps_company_assignee_even_when_inventor_name_overlaps(self):
        metadata = pe.PatentMetadata(
            title="Beispiel",
            inventors=[{"name": "Otto Junker", "address": "Dessau"}],
            assignees=[{"name": "Fa. Otto Junker", "address": "Dessau"}],
            pub_date_application="1955-05-11",
            pub_date_publication="1957-01-17",
        )
        text = (
            "Anmelder: Fa. Otto Junker\\n"
            "Otto Junker ist als Erfinder genannt worden"
        )

        for prompt_id in ("v3", "v4"):
            with self.subTest(prompt_id=prompt_id):
                ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
                ex.prompt_id = prompt_id
                ex.guardrail_profile = "auto"
                fixed = ex._apply_de_legacy_self_applicant_guardrail(metadata, text)
                self.assertIsNotNone(fixed.assignees)
                self.assertEqual(fixed.assignees[0].name, "Fa. Otto Junker")

    def test_explicit_guardrail_profile_works_with_external_prompt(self):
        ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
        ex.prompt_id = None
        ex.guardrail_profile = "de_legacy_self_applicant"

        metadata = pe.PatentMetadata(
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

        fixed = ex._apply_de_legacy_self_applicant_guardrail(metadata, text)
        self.assertIsNone(fixed.assignees)

    def test_guardrail_profile_off_disables_prompt_default_guardrail(self):
        ex = pe.PatentExtractor.__new__(pe.PatentExtractor)
        ex.prompt_id = "v4"
        ex.guardrail_profile = "off"

        metadata = pe.PatentMetadata(
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

        fixed = ex._apply_de_legacy_self_applicant_guardrail(metadata, text)
        self.assertIsNotNone(fixed.assignees)

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
        ex.guardrail_profile = "auto"
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

    def test_vllm_batch_chunked_keeps_candidate_order_inside_doc(self):
        ex = self._mk_vllm_batch_extractor("chunked")
        ex.merge_policy = "prefer_first"
        ex.chunk_size_chars = 4
        ex.chunk_overlap_chars = 0
        ex.extraction_passes = 1
        ex.vllm_doc_batch_size = 8
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
            (txt_dir / "doc_a.txt").write_text("abcdefgh", encoding="utf-8")
            (txt_dir / "doc_b.txt").write_text("wxyzuv", encoding="utf-8")
            out_file = Path(tmpdir) / "preds.jsonl"

            count = ex.batch_extract(txt_dir, out_file)
            self.assertEqual(count, 2)
            self.assertEqual(len(seen_batches), 1)
            self.assertEqual(seen_batches[0], ["abcd", "efgh", "wxyz", "uv"])

            rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([row["prediction"]["title"] for row in rows], ["abcd", "wxyz"])

    def test_vllm_batch_header_first_batches_header_then_fallback_subset(self):
        ex = self._mk_vllm_batch_extractor("header_first")
        seen_batches = []

        class _Model:
            def generate(self, prompts, sampling_params=None, use_tqdm=False):
                seen_batches.append(list(prompts))
                outs = []
                for prompt in prompts:
                    if prompt == "HEADER_OK":
                        payload = {
                            "title": "Header Winner",
                            "inventors": [{"name": "Alice", "address": None}],
                            "assignees": [{"name": "ACME", "address": None}],
                            "pub_date_application": "1960-01-01",
                            "pub_date_publication": "1961-01-01",
                            "pub_date_foreign": None,
                            "classification": None,
                            "industrial_field": None,
                        }
                    elif prompt == "HEADER_BAD":
                        payload = {
                            "title": "Header Incomplete",
                            "inventors": None,
                            "assignees": None,
                            "pub_date_application": None,
                            "pub_date_publication": None,
                            "pub_date_foreign": None,
                            "classification": None,
                            "industrial_field": None,
                        }
                    else:
                        payload = {
                            "title": "Full Rescue",
                            "inventors": [{"name": "Bob", "address": None}],
                            "assignees": [{"name": "Fallback Corp", "address": None}],
                            "pub_date_application": "1960-02-02",
                            "pub_date_publication": "1961-02-02",
                            "pub_date_foreign": None,
                            "classification": None,
                            "industrial_field": None,
                        }
                    outs.append(SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps(payload, ensure_ascii=False))]))
                return outs

        ex.model = _Model()

        with TemporaryDirectory() as tmpdir, patch.object(pe, "tqdm", _fake_tqdm):
            txt_dir = Path(tmpdir) / "texts"
            txt_dir.mkdir()
            (txt_dir / "doc_a.txt").write_text("HEADER_OK\nBODY", encoding="utf-8")
            (txt_dir / "doc_b.txt").write_text("HEADER_BAD\nBODY", encoding="utf-8")
            out_file = Path(tmpdir) / "preds.jsonl"

            count = ex.batch_extract(txt_dir, out_file)
            self.assertEqual(count, 2)
            self.assertEqual(len(seen_batches), 2)
            self.assertEqual(seen_batches[0], ["HEADER_OK", "HEADER_BAD"])
            self.assertEqual(seen_batches[1], ["HEADER_BAD\nBODY"])

            rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(rows[0]["prediction"]["inventors"][0]["name"], "Alice")
            self.assertEqual(rows[1]["prediction"]["inventors"][0]["name"], "Bob")
            self.assertEqual(rows[1]["prediction"]["assignees"][0]["name"], "Fallback Corp")

    def test_vllm_batch_two_pass_targeted_reruns_only_low_conf_subset(self):
        ex = self._mk_vllm_batch_extractor("two_pass_targeted")
        ex.compute_confidence = lambda metadata, raw_text: ((0.2, {"mock": 0.2}) if raw_text == "LOW" else (0.9, {"mock": 0.9}))
        seen_batches = []

        class _Model:
            def generate(self, prompts, sampling_params=None, use_tqdm=False):
                seen_batches.append(list(prompts))
                outs = []
                for prompt in prompts:
                    if "Correction mode" in prompt:
                        payload = {
                            "title": "Low Pass 2",
                            "inventors": [{"name": "Bob", "address": None}],
                            "assignees": [{"name": "Fixed Corp", "address": None}],
                            "pub_date_application": "1960-02-02",
                            "pub_date_publication": "1961-02-02",
                            "pub_date_foreign": None,
                            "classification": None,
                            "industrial_field": None,
                        }
                    elif prompt == "HIGH":
                        payload = {
                            "title": "High Pass 1",
                            "inventors": [{"name": "Alice", "address": None}],
                            "assignees": [{"name": "ACME", "address": None}],
                            "pub_date_application": "1960-01-01",
                            "pub_date_publication": "1961-01-01",
                            "pub_date_foreign": None,
                            "classification": None,
                            "industrial_field": None,
                        }
                    else:
                        payload = {
                            "title": "Low Pass 1",
                            "inventors": [{"name": "Bob", "address": None}],
                            "assignees": None,
                            "pub_date_application": "1960-02-02",
                            "pub_date_publication": "1961-02-02",
                            "pub_date_foreign": None,
                            "classification": None,
                            "industrial_field": None,
                        }
                    outs.append(SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps(payload, ensure_ascii=False))]))
                return outs

        ex.model = _Model()

        with TemporaryDirectory() as tmpdir, patch.object(pe, "tqdm", _fake_tqdm):
            txt_dir = Path(tmpdir) / "texts"
            txt_dir.mkdir()
            (txt_dir / "doc_a.txt").write_text("HIGH", encoding="utf-8")
            (txt_dir / "doc_b.txt").write_text("LOW", encoding="utf-8")
            out_file = Path(tmpdir) / "preds.jsonl"

            count = ex.batch_extract(txt_dir, out_file)
            self.assertEqual(count, 2)
            self.assertEqual(len(seen_batches), 2)
            self.assertEqual(seen_batches[0], ["HIGH", "LOW"])
            self.assertEqual(len(seen_batches[1]), 1)
            self.assertIn("Correction mode", seen_batches[1][0])

            rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(rows[0]["prediction"]["assignees"][0]["name"], "ACME")
            self.assertEqual(rows[1]["prediction"]["assignees"][0]["name"], "Fixed Corp")

    def test_vllm_batch_self_consistency_batches_each_round(self):
        ex = self._mk_vllm_batch_extractor("self_consistency")
        ex.self_consistency_n = 2
        ex.self_consistency_temp = 0.2
        seen_batches = []

        class _Model:
            def generate(self, prompts, sampling_params=None, use_tqdm=False):
                seen_batches.append(list(prompts))
                outs = []
                for prompt in prompts:
                    inventor_name = "Alice" if prompt == "A" else "Bob"
                    payload = {
                        "title": f"title:{prompt}",
                        "inventors": [{"name": inventor_name, "address": None}],
                        "assignees": None,
                        "pub_date_application": None,
                        "pub_date_publication": None,
                        "pub_date_foreign": None,
                        "classification": None,
                        "industrial_field": None,
                    }
                    outs.append(SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps(payload, ensure_ascii=False))]))
                return outs

        ex.model = _Model()

        with TemporaryDirectory() as tmpdir, patch.object(pe, "tqdm", _fake_tqdm):
            txt_dir = Path(tmpdir) / "texts"
            txt_dir.mkdir()
            (txt_dir / "doc_a.txt").write_text("A", encoding="utf-8")
            (txt_dir / "doc_b.txt").write_text("B", encoding="utf-8")
            out_file = Path(tmpdir) / "preds.jsonl"

            count = ex.batch_extract(txt_dir, out_file)
            self.assertEqual(count, 2)
            self.assertEqual(len(seen_batches), 2)
            self.assertEqual(seen_batches[0], ["A", "B"])
            self.assertEqual(seen_batches[1], ["A", "B"])

            rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(rows[0]["prediction"]["inventors"][0]["name"], "Alice")
            self.assertEqual(rows[1]["prediction"]["inventors"][0]["name"], "Bob")


if __name__ == "__main__":
    unittest.main()

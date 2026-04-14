import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from patent_extraction.config import ExtractionConfig, ProfileConfig, StrategyConfig, VLLMConfig
from patent_extraction.extractor import PatentExtractionRunner


class _FakeExtractor:
    def __init__(self):
        self.calls = []

    def batch_extract(self, *, txt_dir, out_file, limit=None, raw_output_dir=None):
        self.calls.append(
            {
                "txt_dir": Path(txt_dir),
                "out_file": Path(out_file),
                "limit": limit,
                "raw_output_dir": None if raw_output_dir is None else Path(raw_output_dir),
            }
        )
        Path(out_file).write_text('{"file_name":"doc1.txt","status":"ok"}\n', encoding="utf-8")
        return 1


class PatentExtractionRunnerTests(unittest.TestCase):
    def test_runner_writes_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            texts_dir = root / "texts"
            texts_dir.mkdir()
            (texts_dir / "doc1.txt").write_text("hello", encoding="utf-8")
            prompt_path = root / "prompt.txt"
            prompt_path.write_text("Prompt\n{text}\n", encoding="utf-8")

            profile = ProfileConfig(
                name="test_profile",
                description="test",
                extraction=ExtractionConfig(
                    backend="vllm",
                    prompt_path=prompt_path,
                    guardrail_profile="auto",
                    save_raw_output=False,
                    vllm=VLLMConfig(enable_prefix_caching=True, doc_batch_size=8),
                    strategy=StrategyConfig(name="baseline"),
                ),
            )
            runner = PatentExtractionRunner(profile=profile)
            fake = _FakeExtractor()

            with patch.object(PatentExtractionRunner, "build_extractor", return_value=fake):
                artifacts = runner.batch_extract(
                    texts_dir=texts_dir,
                    out_root=root / "runs",
                    run_name="smoke",
                    limit=1,
                    force=True,
                )

            run_meta = json.loads(artifacts.run_meta_path.read_text(encoding="utf-8"))
            self.assertEqual(run_meta["profile_name"], "test_profile")
            self.assertEqual(run_meta["docs_written"], 1)
            self.assertEqual(run_meta["docs_limit"], 1)
            self.assertEqual(run_meta["config"]["strategy"]["name"], "baseline")
            self.assertEqual(run_meta["config"]["vllm"]["doc_batch_size"], 8)
            self.assertEqual(run_meta["config"]["vllm"]["quantization"], "none")
            self.assertTrue(artifacts.preds_path.exists())
            self.assertIsNone(artifacts.raw_outputs_dir)
            self.assertEqual(fake.calls[0]["limit"], 1)

    def test_runner_refuses_to_overwrite_run_dir_without_force(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            texts_dir = root / "texts"
            texts_dir.mkdir()
            (texts_dir / "doc1.txt").write_text("hello", encoding="utf-8")
            prompt_path = root / "prompt.txt"
            prompt_path.write_text("Prompt\n{text}\n", encoding="utf-8")

            profile = ProfileConfig(
                name="test_profile",
                extraction=ExtractionConfig(
                    backend="vllm",
                    prompt_path=prompt_path,
                    strategy=StrategyConfig(name="baseline"),
                ),
            )
            runner = PatentExtractionRunner(profile=profile)
            run_dir = root / "runs" / "smoke"
            run_dir.mkdir(parents=True)

            with self.assertRaises(FileExistsError):
                runner.batch_extract(
                    texts_dir=texts_dir,
                    out_root=root / "runs",
                    run_name="smoke",
                    force=False,
                )

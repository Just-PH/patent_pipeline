import json
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_bench_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Avoid importing the real PatentExtractor stack (transformers/mlx) in smoke tests.
    fake_mod = types.ModuleType("patent_pipeline.pydantic_extraction.patent_extractor")
    fake_mod.PatentExtractor = FakeExtractor
    sys.modules["patent_pipeline.pydantic_extraction.patent_extractor"] = fake_mod

    mod_path = repo_root / "scripts" / "bench_run_extract.py"
    spec = importlib.util.spec_from_file_location("bench_run_extract_mod", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load bench_run_extract module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeExtractor:
    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.get("model_name") or "fake-model"
        self.backend = kwargs.get("backend", "pytorch")
        self.device = kwargs.get("device", "cpu")
        self.prompt_id = kwargs.get("prompt_id")
        self.prompt_template_source = "fake_prompt"
        self.prompt_hash = "fake_hash"

    def batch_extract(self, txt_dir: Path, out_file: Path, limit=None, raw_output_dir=None):
        txt_files = sorted(txt_dir.glob("*.txt"))
        if limit is not None:
            txt_files = txt_files[:limit]
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            for p in txt_files:
                rec = {"file_name": p.name, "ocr_path": str(p), "prediction": {"identifier": p.stem}}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return len(txt_files)


class TestBenchRunExtractSmoke(unittest.TestCase):
    def test_cli_limit_1_writes_preds_and_run_meta(self):
        bench_run_extract = _load_bench_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            texts = root / "texts"
            texts.mkdir(parents=True, exist_ok=True)
            (texts / "doc1.txt").write_text("dummy ocr text", encoding="utf-8")

            out_root = root / "out"
            run_name = "smoke_run"

            argv = [
                "bench_run_extract.py",
                "--texts-dir",
                str(texts),
                "--out-root",
                str(out_root),
                "--run-name",
                run_name,
                "--backend",
                "pytorch",
                "--quantization",
                "none",
                "--attn-implementation",
                "sdpa",
                "--cache-implementation",
                "static",
                "--limit",
                "1",
                "--force",
            ]

            with patch.object(bench_run_extract, "PatentExtractor", FakeExtractor), patch(
                "sys.argv", argv
            ):
                bench_run_extract.main()

            run_dir = out_root / run_name
            self.assertTrue((run_dir / "preds.jsonl").exists())
            self.assertTrue((run_dir / "run.json").exists())


if __name__ == "__main__":
    unittest.main()

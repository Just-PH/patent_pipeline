import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_compare_module():
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "scripts" / "bench_compare_hf_vllm.py"
    spec = importlib.util.spec_from_file_location("bench_compare_hf_vllm_mod", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load bench_compare_hf_vllm module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBenchCompareHfVllmSmoke(unittest.TestCase):
    def test_compare_writes_summary(self):
        mod = _load_compare_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            texts = root / "texts"
            texts.mkdir(parents=True, exist_ok=True)
            (texts / "doc1.txt").write_text("dummy ocr text", encoding="utf-8")

            out_root = root / "out"
            compare_name = "cmp_smoke"

            def _fake_run(cmd, check=True):
                self.assertTrue(check)
                args = list(cmd[2:])
                out_dir = Path(args[args.index("--out-root") + 1])
                run_name = args[args.index("--run-name") + 1]
                backend = args[args.index("--backend") + 1]
                run_dir = out_dir / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "preds.jsonl").write_text(
                    json.dumps(
                        {
                            "file_name": "doc1.txt",
                            "prediction": {"identifier": "doc1"},
                            "confidence_score": 0.8 if backend == "vllm" else 0.7,
                        },
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (run_dir / "run.json").write_text(
                    json.dumps(
                        {
                            "config": {"backend": backend},
                            "strategy_stats": {
                                "docs": 1,
                                "mean_confidence": 0.8 if backend == "vllm" else 0.7,
                            },
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

            argv = [
                "bench_compare_hf_vllm.py",
                "--texts-dir",
                str(texts),
                "--out-root",
                str(out_root),
                "--compare-name",
                compare_name,
                "--limit",
                "1",
                "--force",
            ]

            with patch("subprocess.run", side_effect=_fake_run), patch("sys.argv", argv):
                mod.main()

            compare_json = out_root / compare_name / "compare.json"
            self.assertTrue(compare_json.exists())
            summary = json.loads(compare_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["runs"]["transformers"]["docs"], 1)
            self.assertEqual(summary["runs"]["vllm"]["docs"], 1)
            self.assertEqual(summary["runs"]["transformers"]["valid_json_docs"], 1)
            self.assertEqual(summary["runs"]["vllm"]["valid_json_docs"], 1)


if __name__ == "__main__":
    unittest.main()

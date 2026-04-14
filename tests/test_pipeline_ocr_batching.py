import tempfile
import unittest
from pathlib import Path

from PIL import Image

from patent_pipeline.patent_ocr.pipeline_modular import PipelineOCRConfig, Pipeline_OCR


class _FakeGpuBatchBackend:
    def __init__(self):
        self.calls = []

    @property
    def is_gpu(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "fake-gpu-batch"

    def run_blocks_ocr(self, block_imgs, ocr_config):
        self.calls.append(len(block_imgs))
        return [f"text-{idx}" for idx, _ in enumerate(block_imgs, start=1)]


class PipelineOCRBatchingTests(unittest.TestCase):
    def test_backend_mode_batches_multiple_docs_for_gpu_backends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            out_dir = root / "texts"
            report_file = root / "ocr_report.csv"
            raw_dir.mkdir()

            for idx in range(3):
                img = Image.new("RGB", (32, 24), color=(255, 255, 255))
                img.save(raw_dir / f"doc{idx + 1}.png")

            backend = _FakeGpuBatchBackend()
            pipeline = Pipeline_OCR(ocr_backend=backend)
            cfg = PipelineOCRConfig(
                raw_dir=raw_dir,
                out_dir=out_dir,
                report_file=report_file,
                segmentation_mode="backend",
                deskew=False,
                workers=1,
                parallel="none",
                ocr_config={"batch_size": 2},
                force=True,
            )

            rows = pipeline.run(cfg)

            self.assertEqual(backend.calls, [2, 1])
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row.status == "ok" for row in rows))
            self.assertTrue((out_dir / "doc1.txt").exists())
            self.assertTrue((out_dir / "doc2.txt").exists())
            self.assertTrue((out_dir / "doc3.txt").exists())


if __name__ == "__main__":
    unittest.main()

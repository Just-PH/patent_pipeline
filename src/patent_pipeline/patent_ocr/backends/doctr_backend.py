# src/patent_pipeline/patent_ocr/backends/doctr_backend.py

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Dict, List

from PIL import Image

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from patent_pipeline.patent_ocr.utils.image_preprocess import preprocess_pil


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    """
    Convert a PIL image to PNG bytes in memory.

    Why:
      - Some docTR versions (including yours) accept file paths or bytes in
        DocumentFile.from_images, but NOT numpy arrays.
      - Using bytes avoids writing temp files to disk.
    """
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _page_to_text(page) -> str:
    """
    Robust text extraction from a docTR page object.

    We explicitly walk blocks -> lines -> words instead of relying on page.render(),
    because render() behavior can vary across docTR versions / settings.
    """
    out_lines: List[str] = []

    for block in page.blocks:
        for line in block.lines:
            line_txt = " ".join((w.value or "").strip() for w in line.words).strip()
            if line_txt:
                out_lines.append(line_txt)

        # Optional blank line between blocks for readability
        if out_lines and out_lines[-1] != "":
            out_lines.append("")

    # Trim trailing empty lines
    while out_lines and out_lines[-1] == "":
        out_lines.pop()

    return "\n".join(out_lines).strip()


# ---------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------
@dataclass
class DocTROcrBackend:
    """
    docTR OCR backend (backend-first architecture).

    Contract:
      - run_blocks_ocr(block_imgs, ocr_config) -> list[str] (same length/order)

    Notes:
      - Works with segmentation_mode="backend" (page-level OCR).
      - Also works with segmentation_mode="custom" (cropped blocks), though you
        may lose some layout cues docTR could use at page-level.
      - Uses shared preprocess_pil(...) to keep preprocessing consistent across backends.
    """

    det_arch: str = "db_resnet50"
    reco_arch: str = "crnn_vgg16_bn"
    pretrained: bool = True
    device: str = "cpu"
    name_: str = "doctr"

    def __post_init__(self) -> None:
        self.device = (self.device or "cpu").lower().strip()

        self.model = ocr_predictor(
            det_arch=self.det_arch,
            reco_arch=self.reco_arch,
            pretrained=self.pretrained,
        )

        if self.device == "cuda":
            self.model = self.model.cuda()
        elif self.device == "mps":
            self.model = self.model.to("mps")

        self.model.eval()

    @property
    def is_gpu(self) -> bool:
        return self.device in {"cuda", "mps"}

    @property
    def name(self) -> str:
        return self.name_

    def run_blocks_ocr(self, block_imgs: List[Any], ocr_config: Dict[str, Any]) -> List[str]:
        """
        Run docTR OCR on a list of images (pages or blocks).

        Inputs:
          - block_imgs: list of PIL.Image (or array-like convertible to PIL)
          - ocr_config:
              - preprocess: "none" | "gray" | "light" (optional; shared util)
              - det_arch / reco_arch (optional override) [not reloaded here by default]

        Output:
          - list[str], one per input image
        """
        if not block_imgs:
            return []

        preprocess_mode = ocr_config.get("preprocess", "none")

        # Convert inputs to PIL -> shared preprocess -> PNG bytes for docTR
        files: List[bytes] = []
        for im in block_imgs:
            pil = im if isinstance(im, Image.Image) else Image.fromarray(im).convert("RGB")
            pil = preprocess_pil(pil, mode=preprocess_mode)
            files.append(_pil_to_png_bytes(pil))

        doc = DocumentFile.from_images(files)
        result = self.model(doc)

        texts: List[str] = []
        for page in result.pages:
            texts.append(_page_to_text(page))

        if len(texts) != len(block_imgs):
            raise RuntimeError(f"DocTR backend returned {len(texts)} texts for {len(block_imgs)} images")

        return texts


    def validate_ocr_config(self, ocr_config: Dict[str, Any]) -> None:
        # preprocess is shared; accept default
        mode = str(ocr_config.get("preprocess", "none")).lower().strip()
        allowed = {"none", "gray", "light"}
        if mode not in allowed:
            raise ValueError(
                f"[DoctrBackend] Invalid ocr_config['preprocess']={mode!r}. "
                f"Allowed: {sorted(allowed)}"
            )

        # if your DoctrBackend has model_name or predictor objects, validate them:
        if getattr(self, "model_name", None) is not None and not isinstance(self.model_name, str):
            raise TypeError("[DoctrBackend] self.model_name must be a string if provided.")

# src/patent_pipeline/patent_ocr/backends/doctr_backend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# docTR imports
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


@dataclass
class DocTROcrBackend:
    """
    docTR end-to-end OCR backend.

    Contract (strict):
      - run_blocks_ocr(block_imgs, ocr_config) -> list[str] of same length

    Intended usage:
      - segmentation_mode="backend" (pipeline sends [page_img])
      - We still support len(block_imgs) > 1 (batch pages or block images),
        but docTR shines when it receives full pages.
    """

    # Defaults that can be overridden via ocr_config
    det_arch: str = "db_resnet50"
    reco_arch: str = "crnn_vgg16_bn"
    pretrained: bool = True

    # device: "cpu" | "cuda" | "mps"
    device: str = "cpu"

    # Optional: if you want to limit threads or similar later
    name_: str = "doctr"

    def __post_init__(self) -> None:
        self.device = (self.device or "cpu").lower().strip()

        self.model = ocr_predictor(
            det_arch=self.det_arch,
            reco_arch=self.reco_arch,
            pretrained=self.pretrained,
        )

        # Move to device
        if self.device == "cuda":
            self.model = self.model.cuda()
        elif self.device == "mps":
            self.model = self.model.to("mps")
        else:
            # cpu
            pass

    @property
    def is_gpu(self) -> bool:
        return self.device in {"cuda", "mps"}

    @property
    def name(self) -> str:
        return self.name_

    def run_blocks_ocr(self, block_imgs: List[Any], ocr_config: Dict[str, Any]) -> List[str]:
        """
        Runs docTR OCR for each image in block_imgs.

        Notes:
          - For segmentation_mode="backend", block_imgs is typically [page_img].
          - If you pass multiple images, docTR will run them as a DocumentFile batch.
          - Output is one string per input image, in the same order.
        """
        if not block_imgs:
            return []

        # Allow overriding architectures at runtime (optional)
        # (We keep it minimal: rebuild model only if user requests different arch)
        det_arch = str(ocr_config.get("det_arch", self.det_arch))
        reco_arch = str(ocr_config.get("reco_arch", self.reco_arch))
        if det_arch != self.det_arch or reco_arch != self.reco_arch:
            # Rebuild model with requested arches (rare; but useful for experiments)
            self.det_arch = det_arch
            self.reco_arch = reco_arch
            self.model = ocr_predictor(det_arch=self.det_arch, reco_arch=self.reco_arch, pretrained=self.pretrained)
            if self.device == "cuda":
                self.model = self.model.cuda()
            elif self.device == "mps":
                self.model = self.model.to("mps")

        # Convert PIL images to numpy arrays
        # docTR expects HxWxC uint8 (RGB) typically.
        np_imgs = []
        for img in block_imgs:
            # assume PIL.Image
            arr = np.array(img)
            if arr.ndim == 2:
                # grayscale -> add channel
                arr = np.stack([arr, arr, arr], axis=-1)
            np_imgs.append(arr)

        doc = DocumentFile.from_images(np_imgs)

        # Forward pass
        result = self.model(doc)

        # docTR returns a Document object; render() gives full text for each page.
        # result.pages is aligned with inputs.
        texts: List[str] = []
        for page in result.pages:
            # page.render() exists; result.render() renders all pages at once.
            # We'll use page.render() to keep strict per-input mapping.
            txt = page.render()
            texts.append((txt or "").strip())

        return texts

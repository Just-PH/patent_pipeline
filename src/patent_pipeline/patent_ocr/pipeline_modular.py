# src/patent_pipeline/patent_ocr/pipeline_modular.py
from __future__ import annotations

"""
Pipeline OCR — orchestration stricte :

LOGIQUE (source de vérité) :
  1) Charger la page en PIL (RGB)
  2) Deskew UNE FOIS (page-level) si cfg.deskew=True et deskewer fourni
  3) Selon segmentation_mode :
      - "custom"  : appeler segmenter.process(page_img, deskew=False) pour obtenir boxes_ordered
                   puis cropper les blocs
      - "backend" : passer la page entière comme unique bloc [page_img]
  4) OCR via UNE SEULE API : backend.run_blocks_ocr(block_imgs, ocr_config) -> list[str]
  5) join + write + report

Objectif :
  - éviter les redondances (deskew n'est PAS couplé à la segmentation)
  - garder une API OCR unique et stricte (run_blocks_ocr)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union, Literal
import csv
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


# -----------------------
# Types
# -----------------------
Box = List[int]   # [x1, y1, x2, y2]
Boxes = List[Box]


# -----------------------
# Protocols (interfaces)
# -----------------------
class Segmenter(Protocol):
    def process(
        self,
        img_or_path: Union[str, Path, Any],
        *,
        deskew: bool = True,
        deskew_max_angle: Optional[float] = None,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Contract segmenter:
          - returns dict with at least:
              "image": PIL.Image
              "boxes_ordered": Boxes
          - optional:
              "deskew_angle": float (ignored by pipeline if pipeline deskews)
        """
        ...


class OcrBackend(Protocol):
    """
    STRICT backend contract:

    Pipeline calls ONLY:
        run_blocks_ocr(block_imgs, ocr_config) -> list[str]

    Rules:
      - Must return list[str] of SAME LENGTH as block_imgs
      - If segmentation_mode='backend', pipeline sends [page_img] (len=1),
        backend returns ["full page text"] (len=1).
      - If segmentation_mode='custom', pipeline sends many crops,
        backend returns one text per crop.
    """
    def run_blocks_ocr(self, block_imgs: List[Any], ocr_config: Dict[str, Any]) -> List[str]:
        ...

    @property
    def is_gpu(self) -> bool:
        ...

    @property
    def name(self) -> str:
        ...


class OcrEngineBackend:
    """
    Adapter strict au-dessus d'un engine.

    Engine contract (strict):
        engine.run_ocr(PIL.Image, config) -> str

    Backend contract (strict):
        run_blocks_ocr(list[PIL.Image], config) -> list[str]

    Batching "par défaut" = boucle.
    Un backend SOTA pourra implémenter run_blocks_ocr avec vrai batch GPU.
    """
    def __init__(self, engine: Any, *, name: Optional[str] = None, is_gpu: bool = False):
        self._engine = engine
        self._name = name or getattr(engine, "ocr_name", "ocr_engine")
        self._is_gpu = bool(is_gpu)

    def run_blocks_ocr(self, block_imgs: List[Any], ocr_config: Dict[str, Any]) -> List[str]:
        texts: List[str] = []
        for img in block_imgs:
            txt = self._engine.run_ocr(img, ocr_config)
            if not isinstance(txt, str):
                raise TypeError(
                    f"{self._name}.run_ocr must return str, got {type(txt)}. "
                    "Keep run_ocr strict."
                )
            texts.append(txt)
        return texts

    @property
    def is_gpu(self) -> bool:
        return self._is_gpu

    @property
    def name(self) -> str:
        return self._name


# -----------------------
# Config / results
# -----------------------
SegmentationMode = Literal["custom", "backend"]


@dataclass
class PipelineOCRConfig:
    raw_dir: Path
    out_dir: Path
    report_file: Path

    # Segmentation strategy:
    # - "custom": use segmenter -> boxes -> crop blocks
    # - "backend": send full page as single block; backend handles detection/order internally
    segmentation_mode: SegmentationMode = "custom"

    # Deskew strategy: ALWAYS handled by the pipeline (page-level) if enabled.
    deskew: bool = True
    deskew_max_angle: float = 20.0

    # Free-form config for OCR backend/engine
    ocr_config: Dict[str, Any] = field(default_factory=dict)

    # Parallelism
    workers: int = 1
    parallel: str = "threads"   # "threads" | "processes" | "none"

    # IO behavior
    force: bool = False
    limit: Optional[int] = None
    keep_empty_docs: bool = True
    join_with: str = "\n\n"


@dataclass
class DocumentReport:
    file_name: str
    status: str
    n_blocks: int
    n_blocks_kept: int
    deskew_angle: float
    out_txt: str
    error: str = ""


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}


# -----------------------
# IO helpers
# -----------------------
def iter_docs(raw_dir: Path) -> List[Path]:
    return sorted([p for p in raw_dir.glob("**/*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])


def write_report_csv(path: Path, rows: List[DocumentReport]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["file_name", "status", "n_blocks", "n_blocks_kept", "deskew_angle", "out_txt", "error"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "file_name": r.file_name,
                    "status": r.status,
                    "n_blocks": r.n_blocks,
                    "n_blocks_kept": r.n_blocks_kept,
                    "deskew_angle": f"{r.deskew_angle:.4f}",
                    "out_txt": r.out_txt,
                    "error": r.error,
                }
            )


def load_page_as_pil(img_or_path: Union[str, Path, Any]) -> Any:
    """
    Charge une page sous forme PIL.Image RGB.

    - images (png/jpg/...) : PIL.Image.open
    - pdf : pdf2image (1ère page)

    Note: Si tu as déjà converti tes PDFs en PNGs, tu ne passes jamais ici en PDF.
    """
    from PIL import Image

    if not isinstance(img_or_path, (str, Path)):
        return img_or_path

    p = Path(img_or_path)
    ext = p.suffix.lower()

    if ext == ".pdf":
        try:
            from pdf2image import convert_from_path
        except Exception as e:
            raise RuntimeError(
                "PDF input detected but pdf2image is not available. "
                "Install pdf2image + poppler, or pre-extract PDFs to images."
            ) from e

        pages = convert_from_path(str(p), first_page=1, last_page=1)
        if not pages:
            raise RuntimeError(f"Could not render PDF to image: {p}")
        return pages[0].convert("RGB")

    return Image.open(p).convert("RGB")


def deskew_page_if_needed(page_img: Any, *, deskewer: Optional[Any], max_angle: float, enabled: bool) -> tuple[Any, float]:
    """
    Deskew page-level (UNE FOIS).

    deskewer contract expected:
        deskewer.deskew(gray=np.ndarray, max_angle=float) -> (gray_deskewed, angle_deg)

    Returns:
        (page_img_deskewed_RGB, angle_deg)
    """
    if not enabled or deskewer is None:
        return page_img, 0.0

    from PIL import Image
    import numpy as np

    # convert PIL RGB -> gray uint8
    gray = np.array(page_img.convert("L"))
    gray_desk, angle = deskewer.deskew(gray=gray, max_angle=float(max_angle))

    # back to RGB for downstream code consistency
    page_desk = Image.fromarray(gray_desk).convert("RGB")
    return page_desk, float(angle)


# -----------------------
# Main pipeline
# -----------------------
class Pipeline_OCR:
    def __init__(
        self,
        *,
        segmenter: Optional[Segmenter] = None,
        deskewer: Optional[Any] = None,
        ocr_backend: OcrBackend,
    ):
        """
        segmenter:
          - required for segmentation_mode="custom"
          - ignored for segmentation_mode="backend"

        deskewer:
          - optional page-level deskewer (recommended for both modes)
          - if None or cfg.deskew=False => no deskew
        """
        self.segmenter = segmenter
        self.deskewer = deskewer
        self.ocr_backend = ocr_backend

    def _process_one(self, doc_path: Path, cfg: PipelineOCRConfig) -> DocumentReport:
        out_txt_path = cfg.out_dir / f"{doc_path.stem}.txt"

        # Skip if exists (unless force)
        if out_txt_path.exists() and not cfg.force:
            return DocumentReport(
                file_name=str(doc_path),
                status="skipped",
                n_blocks=0,
                n_blocks_kept=0,
                deskew_angle=0.0,
                out_txt=str(out_txt_path),
            )

        try:
            # ------------------------------------------------------------
            # 1) Load page (PIL RGB)
            # ------------------------------------------------------------
            page_img = load_page_as_pil(doc_path)

            # ------------------------------------------------------------
            # 2) Deskew ONCE (page-level)
            # ------------------------------------------------------------
            page_img, deskew_angle = deskew_page_if_needed(
                page_img,
                deskewer=self.deskewer,
                max_angle=cfg.deskew_max_angle,
                enabled=cfg.deskew,
            )

            # ------------------------------------------------------------
            # 3) Segmentation strategy
            # ------------------------------------------------------------
            if cfg.segmentation_mode == "custom":
                if self.segmenter is None:
                    raise ValueError("segmentation_mode='custom' requires a segmenter instance")

                # IMPORTANT: pipeline already deskewed the page
                # -> we explicitly disable deskew inside the segmenter
                res = self.segmenter.process(
                    page_img,
                    deskew=False,
                    deskew_max_angle=cfg.deskew_max_angle,
                    return_debug=False,
                )

                # We trust our deskewed page as source of truth
                page_img = res.get("image", page_img)
                boxes: Boxes = res["boxes_ordered"]

                block_imgs: List[Any] = [
                    page_img.crop((x1, y1, x2, y2)) for (x1, y1, x2, y2) in boxes
                ]

            elif cfg.segmentation_mode == "backend":
                # backend wants the whole (deskewed) page as a single "block"
                w, h = page_img.size
                boxes = [[0, 0, w, h]]
                block_imgs = [page_img]

            else:
                raise ValueError(f"Unknown segmentation_mode: {cfg.segmentation_mode!r}")

            # ------------------------------------------------------------
            # 4) OCR in ONE unified call (STRICT)
            # ------------------------------------------------------------
            texts: List[str] = self.ocr_backend.run_blocks_ocr(block_imgs, cfg.ocr_config)
            if len(texts) != len(block_imgs):
                raise ValueError(
                    f"OCR backend returned {len(texts)} results for {len(block_imgs)} blocks "
                    f"(backend={self.ocr_backend.name})"
                )

            # ------------------------------------------------------------
            # 5) Normalize + keep non-empty
            # ------------------------------------------------------------
            kept = 0
            blocks_text: List[str] = []
            for txt in texts:
                if not isinstance(txt, str):
                    raise TypeError(f"OCR backend must return list[str], got element type {type(txt)}")
                t = txt.strip()
                if t:
                    blocks_text.append(t)
                    kept += 1

            full_text = cfg.join_with.join(blocks_text).strip()

            # ------------------------------------------------------------
            # 6) Write output
            # ------------------------------------------------------------
            cfg.out_dir.mkdir(parents=True, exist_ok=True)
            if full_text or cfg.keep_empty_docs:
                out_txt_path.write_text(full_text, encoding="utf-8")

            status = "ok" if full_text else "empty"
            return DocumentReport(
                file_name=str(doc_path),
                status=status,
                n_blocks=len(boxes),
                n_blocks_kept=kept,
                deskew_angle=deskew_angle,
                out_txt=str(out_txt_path),
            )

        except Exception as e:
            cfg.out_dir.mkdir(parents=True, exist_ok=True)
            if cfg.keep_empty_docs:
                out_txt_path.write_text("", encoding="utf-8")

            return DocumentReport(
                file_name=str(doc_path),
                status="error",
                n_blocks=0,
                n_blocks_kept=0,
                deskew_angle=0.0,
                out_txt=str(out_txt_path),
                error="".join(traceback.format_exception_only(type(e), e)).strip(),
            )

    def run(self, cfg: PipelineOCRConfig) -> List[DocumentReport]:
        docs = iter_docs(cfg.raw_dir)
        if cfg.limit is not None:
            docs = docs[: int(cfg.limit)]

        # ------------------------------------------------------------
        # Parallelism heuristic WITHOUT mutating cfg
        # ------------------------------------------------------------
        workers_eff = int(cfg.workers)
        parallel_eff = str(cfg.parallel).lower().strip()

        # --- Fail-fast: backend-specific config validation ---
        if hasattr(self.ocr_backend, "validate_ocr_config"):
            self.ocr_backend.validate_ocr_config(cfg.ocr_config or {})

        # GPU backends: avoid naive parallelism
        if self.ocr_backend.is_gpu and workers_eff > 1:
            workers_eff = 1
            parallel_eff = "none"

        rows: List[DocumentReport] = []

        if parallel_eff == "none" or workers_eff <= 1:
            for p in docs:
                r = self._process_one(p, cfg)
                rows.append(r)
                print(f" - {p.name}: {r.status} ({r.n_blocks_kept}/{r.n_blocks})")
            return rows

        Executor = ThreadPoolExecutor if parallel_eff == "threads" else ProcessPoolExecutor

        with Executor(max_workers=workers_eff) as ex:
            futs = [ex.submit(self._process_one, p, cfg) for p in docs]
            for fut in as_completed(futs):
                r = fut.result()
                rows.append(r)
                print(f" - {Path(r.file_name).name}: {r.status} ({r.n_blocks_kept}/{r.n_blocks})")

        return rows

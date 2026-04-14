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
from typing import Any, Callable, Dict, List, Optional, Protocol, Union, Literal
import csv
import traceback
import time
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
TimingsMode = Literal["off", "basic", "detailed"]


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

    # Timings:
    #  - "off": no timing measurements
    #  - "basic": measure only t_total_s and t_ocr_s
    #  - "detailed": measure load/deskew/segment/ocr/write/total
    timings: TimingsMode = "off"


@dataclass
class DocumentReport:
    file_name: str
    status: str
    n_blocks: int
    n_blocks_kept: int
    deskew_angle: float
    out_txt: str
    error: str = ""

    # Timings (seconds). Always present in CSV for stable schema.
    t_load_s: float = 0.0
    t_deskew_s: float = 0.0
    t_segment_s: float = 0.0
    t_ocr_s: float = 0.0
    t_write_s: float = 0.0
    t_total_s: float = 0.0


@dataclass
class _PreparedBackendDoc:
    doc_path: Path
    out_txt_path: Path
    page_img: Any
    deskew_angle: float
    t_load_s: float = 0.0
    t_deskew_s: float = 0.0
    t_segment_s: float = 0.0


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
            fieldnames=[
                "file_name",
                "status",
                "n_blocks",
                "n_blocks_kept",
                "deskew_angle",
                "out_txt",
                "error",
                # timings
                "t_load_s",
                "t_deskew_s",
                "t_segment_s",
                "t_ocr_s",
                "t_write_s",
                "t_total_s",
            ],
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
                    "t_load_s": f"{r.t_load_s:.6f}",
                    "t_deskew_s": f"{r.t_deskew_s:.6f}",
                    "t_segment_s": f"{r.t_segment_s:.6f}",
                    "t_ocr_s": f"{r.t_ocr_s:.6f}",
                    "t_write_s": f"{r.t_write_s:.6f}",
                    "t_total_s": f"{r.t_total_s:.6f}",
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

    def _backend_doc_batch_size(self, cfg: PipelineOCRConfig) -> int:
        if cfg.segmentation_mode != "backend":
            return 1
        if not self.ocr_backend.is_gpu:
            return 1
        try:
            batch_size = int((cfg.ocr_config or {}).get("batch_size", 1))
        except (TypeError, ValueError):
            return 1
        return max(1, batch_size)

    def _prepare_backend_doc(
        self,
        doc_path: Path,
        cfg: PipelineOCRConfig,
    ) -> tuple[Optional[_PreparedBackendDoc], Optional[DocumentReport]]:
        out_txt_path = cfg.out_dir / f"{doc_path.stem}.txt"

        timings_mode: TimingsMode = getattr(cfg, "timings", "off")
        want_basic = timings_mode in ("basic", "detailed")
        want_detailed = timings_mode == "detailed"
        t_load_s = t_deskew_s = t_segment_s = t_total_s = 0.0
        t0_total = time.perf_counter() if want_basic else 0.0

        if out_txt_path.exists() and not cfg.force:
            return None, DocumentReport(
                file_name=str(doc_path),
                status="skipped",
                n_blocks=0,
                n_blocks_kept=0,
                deskew_angle=0.0,
                out_txt=str(out_txt_path),
                t_total_s=t_total_s,
            )

        try:
            if want_detailed:
                t0 = time.perf_counter()
                page_img = load_page_as_pil(doc_path)
                t_load_s = time.perf_counter() - t0
            else:
                page_img = load_page_as_pil(doc_path)

            if want_detailed:
                t0 = time.perf_counter()
                page_img, deskew_angle = deskew_page_if_needed(
                    page_img,
                    deskewer=self.deskewer,
                    max_angle=cfg.deskew_max_angle,
                    enabled=cfg.deskew,
                )
                t_deskew_s = time.perf_counter() - t0
            else:
                page_img, deskew_angle = deskew_page_if_needed(
                    page_img,
                    deskewer=self.deskewer,
                    max_angle=cfg.deskew_max_angle,
                    enabled=cfg.deskew,
                )

            if want_basic:
                t_total_s = time.perf_counter() - t0_total

            return _PreparedBackendDoc(
                doc_path=doc_path,
                out_txt_path=out_txt_path,
                page_img=page_img,
                deskew_angle=deskew_angle,
                t_load_s=t_load_s,
                t_deskew_s=t_deskew_s,
                t_segment_s=t_segment_s,
            ), None
        except Exception as e:
            cfg.out_dir.mkdir(parents=True, exist_ok=True)
            if cfg.keep_empty_docs:
                out_txt_path.write_text("", encoding="utf-8")

            if want_basic:
                t_total_s = time.perf_counter() - t0_total

            return None, DocumentReport(
                file_name=str(doc_path),
                status="error",
                n_blocks=0,
                n_blocks_kept=0,
                deskew_angle=0.0,
                out_txt=str(out_txt_path),
                error="".join(traceback.format_exception_only(type(e), e)).strip(),
                t_load_s=t_load_s,
                t_deskew_s=t_deskew_s,
                t_segment_s=t_segment_s,
                t_total_s=t_total_s,
            )

    def _finalize_backend_doc(
        self,
        prepared: _PreparedBackendDoc,
        text: str,
        cfg: PipelineOCRConfig,
        *,
        t_ocr_s: float,
    ) -> DocumentReport:
        timings_mode: TimingsMode = getattr(cfg, "timings", "off")
        want_basic = timings_mode in ("basic", "detailed")
        want_detailed = timings_mode == "detailed"

        kept = 0
        full_text = (text or "").strip()
        if full_text:
            kept = 1

        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        t_write_s = 0.0
        if want_detailed:
            t0 = time.perf_counter()
            if full_text or cfg.keep_empty_docs:
                prepared.out_txt_path.write_text(full_text, encoding="utf-8")
            t_write_s = time.perf_counter() - t0
        else:
            if full_text or cfg.keep_empty_docs:
                prepared.out_txt_path.write_text(full_text, encoding="utf-8")

        t_total_s = 0.0
        if want_basic:
            t_total_s = prepared.t_load_s + prepared.t_deskew_s + prepared.t_segment_s + t_ocr_s + t_write_s

        return DocumentReport(
            file_name=str(prepared.doc_path),
            status="ok" if full_text else "empty",
            n_blocks=1,
            n_blocks_kept=kept,
            deskew_angle=prepared.deskew_angle,
            out_txt=str(prepared.out_txt_path),
            t_load_s=prepared.t_load_s,
            t_deskew_s=prepared.t_deskew_s,
            t_segment_s=prepared.t_segment_s,
            t_ocr_s=t_ocr_s,
            t_write_s=t_write_s,
            t_total_s=t_total_s,
        )

    def _process_backend_doc_batch(
        self,
        prepared_docs: List[_PreparedBackendDoc],
        cfg: PipelineOCRConfig,
    ) -> List[DocumentReport]:
        if not prepared_docs:
            return []

        timings_mode: TimingsMode = getattr(cfg, "timings", "off")
        want_basic = timings_mode in ("basic", "detailed")

        imgs = [doc.page_img for doc in prepared_docs]
        batch_ocr_s = 0.0

        try:
            if want_basic:
                t0 = time.perf_counter()
                texts = self.ocr_backend.run_blocks_ocr(imgs, cfg.ocr_config)
                batch_ocr_s = time.perf_counter() - t0
            else:
                texts = self.ocr_backend.run_blocks_ocr(imgs, cfg.ocr_config)

            if len(texts) != len(prepared_docs):
                raise ValueError(
                    f"OCR backend returned {len(texts)} results for {len(prepared_docs)} backend docs "
                    f"(backend={self.ocr_backend.name})"
                )

            per_doc_ocr_s = (batch_ocr_s / len(prepared_docs)) if want_basic else 0.0
            return [
                self._finalize_backend_doc(prepared, txt, cfg, t_ocr_s=per_doc_ocr_s)
                for prepared, txt in zip(prepared_docs, texts)
            ]
        except Exception:
            rows: List[DocumentReport] = []
            for prepared in prepared_docs:
                item_ocr_s = 0.0
                try:
                    if want_basic:
                        t0 = time.perf_counter()
                        texts = self.ocr_backend.run_blocks_ocr([prepared.page_img], cfg.ocr_config)
                        item_ocr_s = time.perf_counter() - t0
                    else:
                        texts = self.ocr_backend.run_blocks_ocr([prepared.page_img], cfg.ocr_config)

                    if len(texts) != 1:
                        raise ValueError(
                            f"OCR backend returned {len(texts)} results for 1 backend doc "
                            f"(backend={self.ocr_backend.name})"
                        )
                    rows.append(self._finalize_backend_doc(prepared, texts[0], cfg, t_ocr_s=item_ocr_s))
                except Exception as e:
                    if cfg.keep_empty_docs:
                        cfg.out_dir.mkdir(parents=True, exist_ok=True)
                        prepared.out_txt_path.write_text("", encoding="utf-8")
                    total_s = prepared.t_load_s + prepared.t_deskew_s + prepared.t_segment_s + item_ocr_s
                    rows.append(
                        DocumentReport(
                            file_name=str(prepared.doc_path),
                            status="error",
                            n_blocks=1,
                            n_blocks_kept=0,
                            deskew_angle=prepared.deskew_angle,
                            out_txt=str(prepared.out_txt_path),
                            error="".join(traceback.format_exception_only(type(e), e)).strip(),
                            t_load_s=prepared.t_load_s,
                            t_deskew_s=prepared.t_deskew_s,
                            t_segment_s=prepared.t_segment_s,
                            t_ocr_s=item_ocr_s,
                            t_total_s=total_s if want_basic else 0.0,
                        )
                    )
            return rows

    def _process_one(self, doc_path: Path, cfg: PipelineOCRConfig) -> DocumentReport:
        out_txt_path = cfg.out_dir / f"{doc_path.stem}.txt"

        timings_mode: TimingsMode = getattr(cfg, "timings", "off")
        want_basic = timings_mode in ("basic", "detailed")
        want_detailed = timings_mode == "detailed"

        # timers (seconds)
        t_load_s = t_deskew_s = t_segment_s = t_ocr_s = t_write_s = t_total_s = 0.0
        t0_total = time.perf_counter() if want_basic else 0.0

        # Skip if exists (unless force)
        if out_txt_path.exists() and not cfg.force:
            return DocumentReport(
                file_name=str(doc_path),
                status="skipped",
                n_blocks=0,
                n_blocks_kept=0,
                deskew_angle=0.0,
                out_txt=str(out_txt_path),
                t_load_s=t_load_s,
                t_deskew_s=t_deskew_s,
                t_segment_s=t_segment_s,
                t_ocr_s=t_ocr_s,
                t_write_s=t_write_s,
                t_total_s=t_total_s,
            )

        try:
            # ------------------------------------------------------------
            # 1) Load page (PIL RGB)
            # ------------------------------------------------------------
            if want_detailed:
                t0 = time.perf_counter()
                page_img = load_page_as_pil(doc_path)
                t_load_s = time.perf_counter() - t0
            else:
                page_img = load_page_as_pil(doc_path)

            # ------------------------------------------------------------
            # 2) Deskew ONCE (page-level)
            # ------------------------------------------------------------
            if want_detailed:
                t0 = time.perf_counter()
                page_img, deskew_angle = deskew_page_if_needed(
                    page_img,
                    deskewer=self.deskewer,
                    max_angle=cfg.deskew_max_angle,
                    enabled=cfg.deskew,
                )
                t_deskew_s = time.perf_counter() - t0
            else:
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
                if want_detailed:
                    t0 = time.perf_counter()
                    res = self.segmenter.process(
                        page_img,
                        deskew=False,
                        deskew_max_angle=cfg.deskew_max_angle,
                        return_debug=False,
                    )
                    t_segment_s = time.perf_counter() - t0
                else:
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

                # Detailed timings: segmentation is basically a no-op
                if want_detailed:
                    t_segment_s = 0.0

            else:
                raise ValueError(f"Unknown segmentation_mode: {cfg.segmentation_mode!r}")

            # ------------------------------------------------------------
            # 4) OCR in ONE unified call (STRICT)
            # ------------------------------------------------------------
            if want_basic:
                t0 = time.perf_counter()
                texts: List[str] = self.ocr_backend.run_blocks_ocr(block_imgs, cfg.ocr_config)
                t_ocr_s = time.perf_counter() - t0
            else:
                texts = self.ocr_backend.run_blocks_ocr(block_imgs, cfg.ocr_config)
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
            if want_detailed:
                t0 = time.perf_counter()
                if full_text or cfg.keep_empty_docs:
                    out_txt_path.write_text(full_text, encoding="utf-8")
                t_write_s = time.perf_counter() - t0
            else:
                if full_text or cfg.keep_empty_docs:
                    out_txt_path.write_text(full_text, encoding="utf-8")

            if want_basic:
                t_total_s = time.perf_counter() - t0_total

            status = "ok" if full_text else "empty"
            return DocumentReport(
                file_name=str(doc_path),
                status=status,
                n_blocks=len(boxes),
                n_blocks_kept=kept,
                deskew_angle=deskew_angle,
                out_txt=str(out_txt_path),
                t_load_s=t_load_s,
                t_deskew_s=t_deskew_s,
                t_segment_s=t_segment_s,
                t_ocr_s=t_ocr_s,
                t_write_s=t_write_s,
                t_total_s=t_total_s,
            )

        except Exception as e:
            cfg.out_dir.mkdir(parents=True, exist_ok=True)
            if cfg.keep_empty_docs:
                out_txt_path.write_text("", encoding="utf-8")

            if want_basic:
                t_total_s = time.perf_counter() - t0_total

            return DocumentReport(
                file_name=str(doc_path),
                status="error",
                n_blocks=0,
                n_blocks_kept=0,
                deskew_angle=0.0,
                out_txt=str(out_txt_path),
                error="".join(traceback.format_exception_only(type(e), e)).strip(),
                t_load_s=t_load_s,
                t_deskew_s=t_deskew_s,
                t_segment_s=t_segment_s,
                t_ocr_s=t_ocr_s,
                t_write_s=t_write_s,
                t_total_s=t_total_s,
            )

    def run(
        self,
        cfg: PipelineOCRConfig,
        progress_callback: Callable[[DocumentReport, int, int], None] | None = None,
    ) -> List[DocumentReport]:
        docs = iter_docs(cfg.raw_dir)
        if cfg.limit is not None:
            docs = docs[: int(cfg.limit)]

        total_docs = len(docs)

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

        backend_doc_batch_size = self._backend_doc_batch_size(cfg)
        if parallel_eff == "none" and workers_eff <= 1 and backend_doc_batch_size > 1:
            prepared_batch: List[_PreparedBackendDoc] = []
            for p in docs:
                prepared, direct_row = self._prepare_backend_doc(p, cfg)
                if direct_row is not None:
                    rows.append(direct_row)
                    print(f" - {p.name}: {direct_row.status} ({direct_row.n_blocks_kept}/{direct_row.n_blocks})")
                    if progress_callback:
                        progress_callback(direct_row, len(rows), total_docs)
                    continue

                if prepared is None:
                    continue

                prepared_batch.append(prepared)
                if len(prepared_batch) < backend_doc_batch_size:
                    continue

                for row in self._process_backend_doc_batch(prepared_batch, cfg):
                    rows.append(row)
                    print(f" - {Path(row.file_name).name}: {row.status} ({row.n_blocks_kept}/{row.n_blocks})")
                    if progress_callback:
                        progress_callback(row, len(rows), total_docs)
                prepared_batch = []

            if prepared_batch:
                for row in self._process_backend_doc_batch(prepared_batch, cfg):
                    rows.append(row)
                    print(f" - {Path(row.file_name).name}: {row.status} ({row.n_blocks_kept}/{row.n_blocks})")
                    if progress_callback:
                        progress_callback(row, len(rows), total_docs)
            return rows

        if parallel_eff == "none" or workers_eff <= 1:
            for p in docs:
                r = self._process_one(p, cfg)
                rows.append(r)
                print(f" - {p.name}: {r.status} ({r.n_blocks_kept}/{r.n_blocks})")
                if progress_callback:
                    progress_callback(r, len(rows), total_docs)
            return rows

        Executor = ThreadPoolExecutor if parallel_eff == "threads" else ProcessPoolExecutor

        with Executor(max_workers=workers_eff) as ex:
            futs = [ex.submit(self._process_one, p, cfg) for p in docs]
            for fut in as_completed(futs):
                r = fut.result()
                rows.append(r)
                print(f" - {Path(r.file_name).name}: {r.status} ({r.n_blocks_kept}/{r.n_blocks})")
                if progress_callback:
                    progress_callback(r, len(rows), total_docs)

        return rows

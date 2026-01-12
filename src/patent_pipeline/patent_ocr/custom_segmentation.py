"""
preproc_doc_before_ocr.py

CustomSegmentation
===================

Module de pré-traitement d'une page de document avant OCR.

Ce fichier expose une seule classe : `CustomSegmentation`.

Fonctions principales
---------------------
- Charger une image (PNG/JPG/TIFF/...) ou la 1ère page d'un PDF.
- Redresser l'image (deskew) en estimant un angle global.
- Segmenter la page en blocs (bounding boxes).
- Ordonner les blocs dans un ordre de lecture géométrique.
- Générer des images de debug (rectangles + indices).

Dépendances
-----------
  pip install pillow numpy opencv-python pdf2image matplotlib

PDF (pdf2image)
---------------
pdf2image nécessite poppler.
- macOS: brew install poppler
- Linux: apt install poppler-utils
- Windows: installer poppler et configurer PATH

Exemples
--------
(1) Pipeline
    >>> proc = CustomSegmentation()
    >>> res = proc.process("page.pdf", deskew_max_angle=20.0, return_debug=False)
    >>> img = proc.render_overlay(res)
    >>> img.save("debug.png")

(2) Démo sur un dossier
    $ python preproc_doc_before_ocr.py "data/gold_standard_DE/PNGs_extracted/" --n 6 --seed 86857199 --max-angle 20
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2
from PIL import Image, ImageDraw
from patent_pipeline.patent_ocr.deskewer import Deskewer
try:
    from pdf2image import convert_from_path
except Exception:  # pragma: no cover
    convert_from_path = None


Box = List[int]   # [x1, y1, x2, y2]
Boxes = List[Box]


@dataclass(frozen=True)
class SegmentationDebug:
    """Conteneur d'images intermédiaires utiles pour inspecter la segmentation."""
    gray: np.ndarray
    blurred: np.ndarray
    blocks: np.ndarray
    closed: np.ndarray
    lines: np.ndarray
    ink_clean: np.ndarray


class CustomSegmentation:
    """
    Pré-traitement complet d'une page avant OCR.

    API recommandée
    ---------------
    - load(path) -> PIL.Image
    - process(img_or_path, ...) -> dict avec image deskew + boxes + ordering + debug optionnel
    - render_overlay(result) -> PIL.Image (visualisation)
    - demo_folder(folder, ...) -> matplotlib preview
    """

    def __init__(
        self,
        *,
        pdf_dpi: int = 300,
        deskewer: Optional[Deskewer] = None,
        deskew_pad: int = 150,
        deskew_max_angle: float = 20.0,
        min_contour_area: int = 3000,
        iou_thresh: float = 0.7,
        big_min_w: int = 100,
        big_min_h: int = 100,
        fallback_area_ratio: float = 0.55,
        small_min_ink_ratio: float = 0.0018,
        small_min_blob_area: int = 25,
        small_min_significant_blobs: int = 2,
        inner_frac: float = 0.08,
        segment_debug_print: bool = False,
        order_vertical_overlap_threshold: float = 0.3,
        draw_box_color: str = "red",
        draw_box_width: int = 3,
        draw_number_font_size: int = 160,
        draw_number_color: str = "blue",
    ):
        """Stocke la config et prépare les caches utilisés sur chaque page."""
        self.pdf_dpi = pdf_dpi
        self.deskewer = deskewer or Deskewer(method="hough", pad=deskew_pad)
        self.deskew_pad = deskew_pad
        self.deskew_max_angle = deskew_max_angle
        self.min_contour_area = min_contour_area
        self.iou_thresh = iou_thresh
        self.big_min_w = big_min_w
        self.big_min_h = big_min_h
        self.fallback_area_ratio = fallback_area_ratio
        self.small_min_ink_ratio = small_min_ink_ratio
        self.small_min_blob_area = small_min_blob_area
        self.small_min_significant_blobs = small_min_significant_blobs
        self.inner_frac = inner_frac
        self.segment_debug_print = segment_debug_print

        self.order_vertical_overlap_threshold = order_vertical_overlap_threshold

        self.draw_box_color = draw_box_color
        self.draw_box_width = draw_box_width
        self.draw_number_font_size = draw_number_font_size
        self.draw_number_color = draw_number_color

        # Kernels OpenCV (créés une seule fois)
        self._k_open_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self._k_close_25 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        self._k_line_50x25 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 25))

        # Cache de polices (par taille)
        self._font_cache: Dict[int, Any] = {}
        self._font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\arialbd.ttf",
            "arial.ttf",
        ]

    # ---------------------------
    # IO
    # ---------------------------
    def load(self, path: Union[str, Path]) -> Image.Image:
        """Charge une image ou la première page d'un PDF et renvoie une PIL.Image RGB."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))

        suf = p.suffix.lower()
        if suf in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]:
            return Image.open(p).convert("RGB")

        if suf == ".pdf":
            if convert_from_path is None:
                raise ImportError("pdf2image indisponible. Installe pdf2image + poppler.")
            pages = convert_from_path(str(p), dpi=self.pdf_dpi, first_page=1, last_page=1)
            if not pages:
                raise ValueError(f"PDF vide: {p}")
            return pages[0].convert("RGB")

        raise ValueError(f"Format non supporté: {p.suffix}")

    # ---------------------------
    # Pipeline
    # ---------------------------
    def process(
        self,
        img_or_path: Union[Image.Image, str, Path],
        *,
        deskew: bool = True,
        deskew_max_angle: Optional[float] = None,
        return_debug: bool = True,
    ) -> Dict[str, Any]:
        """
        Pipeline complet sur une page.

        IMPORTANT (nouvelle architecture):
        - Le deskew page-level peut être fait en amont par Pipeline_OCR.
        - Dans ce cas, le pipeline appelle process(..., deskew=False)
            pour éviter toute redondance.

        Retourne un dict avec:
        - image: PIL.Image (RGB) (deskewée si deskew=True, sinon originale)
        - deskew_angle: float
        - boxes_ordered: list[[x1,y1,x2,y2], ...]
        - debug: ...
        """

        # 1) Load image (RGB)
        if isinstance(img_or_path, (str, Path)):
            pil_img = Image.open(img_or_path).convert("RGB")
            path = Path(img_or_path)
        else:
            pil_img = img_or_path.convert("RGB")
            path = None

        # 2) Toujours définir un gray de base (source de vérité pour _segment)
        deskew_gray = np.array(pil_img.convert("L"))

        # 3) Deskew optionnel (produit deskew_gray + deskew_angle)
        deskew_angle = 0.0

        if deskew:
            max_angle = float(deskew_max_angle if deskew_max_angle is not None else self.deskew_max_angle)
            if self.deskewer is not None and max_angle > 0:
                deskew_gray, deskew_angle = self.deskewer.deskew(gray=deskew_gray, max_angle=max_angle)

        # 4) Reconstruire une image RGB cohérente pour overlay/debug + crops
        deskewed_pil = Image.fromarray(deskew_gray).convert("RGB")

        # 5) Segmentation (sur gray deskewé)

        dbg, boxes = self._segment(
            pil_img=deskewed_pil,
            gray=deskew_gray,
            return_debug=return_debug,
        )
        ordered = self.order_boxes(boxes)

        return {
            "path": path,
            "image": deskewed_pil,
            "deskew_angle": float(deskew_angle),
            "boxes": boxes,
            "boxes_ordered": ordered,
            "debug": dbg,
        }

    # ---------------------------
    # Ordering
    # ---------------------------
    def order_boxes(self, boxes: Boxes) -> Boxes:
        """Ordonne les boxes en groupant par 'ligne' (overlap vertical) puis tri gauche->droite."""
        thr = self.order_vertical_overlap_threshold
        if not boxes:
            return []
        if len(boxes) == 1:
            return boxes

        boxes_sorted = sorted(boxes, key=lambda b: b[1])

        lines: List[Boxes] = []
        current: Boxes = [boxes_sorted[0]]

        for box in boxes_sorted[1:]:
            x1, y1, x2, y2 = box
            has_overlap = False

            for ref in current:
                X1, Y1, X2, Y2 = ref
                oy1 = max(y1, Y1)
                oy2 = min(y2, Y2)
                oh = max(0, oy2 - oy1)

                h1 = y2 - y1
                h2 = Y2 - Y1
                mh = min(h1, h2)

                if mh > 0 and (oh / mh) >= thr:
                    has_overlap = True
                    break

            if has_overlap:
                current.append(box)
            else:
                lines.append(current)
                current = [box]

        lines.append(current)

        out: Boxes = []
        for line in lines:
            out.extend(sorted(line, key=lambda b: b[0]))
        return out

    # ---------------------------
    # Drawing
    # ---------------------------
    def draw_boxes(
        self,
        pil_img: Image.Image,
        boxes: Boxes,
        *,
        color: Optional[str] = None,
        width: Optional[int] = None,
    ) -> Image.Image:
        """Retourne une copie de l'image avec les rectangles des boxes."""
        color = color or self.draw_box_color
        width = width if width is not None else self.draw_box_width

        out = pil_img.copy()
        draw = ImageDraw.Draw(out)
        for (x1, y1, x2, y2) in boxes:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        return out

    def draw_box_indices(
        self,
        pil_img: Image.Image,
        boxes: Boxes,
        *,
        font_size: Optional[int] = None,
        color: Optional[str] = None,
    ) -> Image.Image:
        """Retourne une copie de l'image avec les indices (1..N) au centre de chaque box."""
        font_size = font_size if font_size is not None else self.draw_number_font_size
        color = color or self.draw_number_color

        out = pil_img.copy()
        draw = ImageDraw.Draw(out)
        font = self._get_font(font_size)

        for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            text = str(idx)

            if font:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            else:
                tw = len(text) * 20
                th = 30

            tx = cx - tw // 2
            ty = cy - th // 2
            pad = 10

            draw.rectangle(
                [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
                fill="white",
                outline="black",
                width=2,
            )
            if font:
                draw.text((tx, ty), text, fill=color, font=font)
            else:
                draw.text((tx, ty), text, fill=color)

        return out

    def render_overlay(self, result: Dict[str, Any]) -> Image.Image:
        """Dessine rectangles + indices à partir du dict retourné par process()."""
        overlay = self.draw_boxes(result["image"], result["boxes"])
        return self.draw_box_indices(overlay, result["boxes_ordered"])

    # ---------------------------
    # Demo (matplotlib)
    # ---------------------------
    def demo_folder(
        self,
        folder: Union[str, Path],
        *,
        n: int = 6,
        seed: Optional[int] = None,
        max_angle: float = 20.0,
        return_debug: bool = False,
    ) -> None:
        """
        Démo matplotlib sur un dossier: sample n fichiers, process(), overlay, affichage en grille.
        """
        import random
        import matplotlib.pyplot as plt

        folder = Path(folder)
        files = self._list_images(folder)
        if not files:
            raise ValueError(f"Aucun fichier image/PDF dans {folder}")

        rng = random.Random(seed)
        sample = rng.sample(files, k=min(n, len(files)))

        cols = 3
        rows = int(np.ceil(len(sample) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
        axes = np.array(axes).reshape(-1)

        for ax, p in zip(axes, sample):
            res = self.process(p, deskew_max_angle=max_angle, return_debug=return_debug)
            img = self.render_overlay(res)

            ax.imshow(img)
            ax.set_title(f"{p.name}\nangle: {res['deskew_angle']:+.2f}°")
            ax.axis("off")

        for ax in axes[len(sample):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    # ============================================================
    # Internals: conversions / utilitaires
    # ============================================================
    def _pil_to_rgb_np(self, pil_img: Image.Image) -> np.ndarray:
        """Convertit une PIL.Image en np.ndarray RGB uint8 (H, W, 3)."""
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return np.asarray(pil_img)

    def _rgb_to_gray(self, rgb: np.ndarray) -> np.ndarray:
        """Convertit un RGB numpy (H,W,3) en grayscale numpy (H,W)."""
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def _get_font(self, font_size: int):
        """Charge une police (ou fallback) et la met en cache pour cette taille."""
        if font_size in self._font_cache:
            return self._font_cache[font_size]

        try:
            from PIL import ImageFont
            font = None
            for fp in self._font_paths:
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except Exception:
                    continue
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = None

        self._font_cache[font_size] = font
        return font

    def _list_images(self, folder: Path) -> List[Path]:
        """Liste récursivement les fichiers image/PDF supportés dans un dossier."""
        exts = {".png", ".jpg", ".jpeg", ".pdf", ".tif", ".tiff", ".bmp", ".webp"}
        return [p for p in folder.glob("**/*") if p.is_file() and p.suffix.lower() in exts]

    # ============================================================
    # Internals: deskew
    # ============================================================
    def deskew(self, *, gray: np.ndarray, max_angle: float) -> Tuple[np.ndarray, float]:
       return self.deskewer.deskew(gray=gray, max_angle=max_angle)


    # ============================================================
    # Internals: segmentation
    # ============================================================
    def _segment(
        self,
        *,
        pil_img: Image.Image,
        gray: np.ndarray,
        return_debug: bool,
    ) -> Tuple[Optional[SegmentationDebug], Boxes]:
        """Applique la segmentation et renvoie (debug optionnel, boxes finales)."""
        H, W = gray.shape

        ink_clean, labels_cc = self._ink_and_cc(gray)
        raw_boxes, blurred, blocks, closed, fused = self._vision_boxes(gray)

        nested_cleaned = self._remove_nested_boxes(raw_boxes, iou_thresh=self.iou_thresh)
        filtered = self._filter_boxes(nested_cleaned, ink_clean=ink_clean, labels_cc=labels_cc)

        merged = self._merge_boxes_brutal(filtered, overlap_thresh=0.15)
        final_sorted = self._merge_boxes_horizontally(merged, min_vertical_overlap=0.2)

        if self._needs_fallback(final_sorted, H, W, area_ratio=self.fallback_area_ratio):
            if self.segment_debug_print:
                print("Fallback split triggered")
            final_sorted = self._split_dominant_block(pil_img, final_sorted[0])

        if self.segment_debug_print:
            print(len(raw_boxes), "boxes found")
            print(len(nested_cleaned), "after nested removal")
            print(len(filtered), "after small filtering")
            print(len(merged), "after merge")
            print(len(final_sorted), "final sorted")

        dbg = None
        if return_debug:
            dbg = SegmentationDebug(
                gray=gray,
                blurred=blurred,
                blocks=blocks,
                closed=closed,
                lines=fused,
                ink_clean=ink_clean,
            )
        return dbg, final_sorted

    def _ink_and_cc(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Construit un masque d'encre (0/1) et une carte de labels CC (0=background)."""
        _, bw_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = (bw_otsu == 0).astype(np.uint8)
        ink_clean = cv2.morphologyEx(ink, cv2.MORPH_OPEN, self._k_open_2, iterations=1)

        ink255 = (ink_clean * 255).astype(np.uint8)
        _num_cc, labels_cc, _stats_cc, _ = cv2.connectedComponentsWithStats(ink255, connectivity=8)
        return ink_clean, labels_cc

    def _vision_boxes(self, gray: np.ndarray) -> Tuple[Boxes, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Détecte des boxes candidates via threshold adaptatif + morpho + contours."""
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        inv = 255 - gray_blur
        blurred = cv2.GaussianBlur(inv, (41, 41), 0)

        blocks = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            101, -10
        )

        closed = cv2.morphologyEx(blocks, cv2.MORPH_CLOSE, self._k_close_25)
        fused = cv2.dilate(closed, self._k_line_50x25, iterations=1)

        contours, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: Boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < self.min_contour_area:
                continue
            boxes.append([int(x), int(y), int(x + w), int(y + h)])

        return boxes, blurred, blocks, closed, fused

    def _filter_boxes(self, boxes: Boxes, *, ink_clean: np.ndarray, labels_cc: np.ndarray) -> Boxes:
        """Conserve toutes les grosses boxes et valide les petites via densité d'encre + blobs CC."""
        out: Boxes = []
        for b in boxes:
            bw, bh = self._box_wh(b)

            if bw > self.big_min_w and bh > self.big_min_h:
                out.append(b)
                continue

            if self._validate_small_box(
                b,
                ink_clean=ink_clean,
                labels_cc=labels_cc,
                min_ink_ratio=self.small_min_ink_ratio,
                min_blob_area=self.small_min_blob_area,
                min_significant_blobs=self.small_min_significant_blobs,
                inner_frac=self.inner_frac,
            ):
                out.append(b)
        return out

    def _box_area(self, b: Box) -> int:
        """Retourne l'aire (pixels²) d'une box."""
        x1, y1, x2, y2 = b
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _box_wh(self, b: Box) -> Tuple[int, int]:
        """Retourne (w, h) d'une box."""
        x1, y1, x2, y2 = b
        return max(0, x2 - x1), max(0, y2 - y1)

    def _inner_crop_coords(self, x1: int, y1: int, x2: int, y2: int, *, inner_frac: float) -> Tuple[int, int, int, int]:
        """Calcule un rectangle interne à une box (pour ignorer les bords)."""
        w = x2 - x1
        h = y2 - y1
        mx = int(inner_frac * w)
        my = int(inner_frac * h)
        xx1 = min(x2 - 1, x1 + mx)
        yy1 = min(y2 - 1, y1 + my)
        xx2 = max(xx1 + 1, x2 - mx)
        yy2 = max(yy1 + 1, y2 - my)
        return xx1, yy1, xx2, yy2

    def _validate_small_box(
        self,
        b: Box,
        *,
        ink_clean: np.ndarray,
        labels_cc: np.ndarray,
        min_ink_ratio: float,
        min_blob_area: int,
        min_significant_blobs: int,
        inner_frac: float,
    ) -> bool:
        """Retourne True si une petite box a assez d'encre et assez de blobs CC significatifs."""
        x1, y1, x2, y2 = b
        xx1, yy1, xx2, yy2 = self._inner_crop_coords(x1, y1, x2, y2, inner_frac=inner_frac)

        crop_ink = ink_clean[yy1:yy2, xx1:xx2]
        if crop_ink.size == 0:
            return False
        if float(crop_ink.mean()) < min_ink_ratio:
            return False

        roi_labels = labels_cc[yy1:yy2, xx1:xx2]
        roi_labels = roi_labels[roi_labels > 0]
        if roi_labels.size == 0:
            return False

        _, counts = np.unique(roi_labels, return_counts=True)
        significant = int(np.sum(counts >= min_blob_area))
        return significant >= min_significant_blobs

    def _remove_nested_boxes(self, boxes: Boxes, *, iou_thresh: float) -> Boxes:
        """Supprime une box si elle est majoritairement incluse dans une autre plus grande."""
        out: Boxes = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            areaA = self._box_area([x1, y1, x2, y2])
            if areaA <= 0:
                continue

            keep = True
            for j, (X1, Y1, X2, Y2) in enumerate(boxes):
                if i == j:
                    continue

                areaB = self._box_area([X1, Y1, X2, Y2])
                if areaB <= 0:
                    continue

                ix1 = max(x1, X1)
                iy1 = max(y1, Y1)
                ix2 = min(x2, X2)
                iy2 = min(y2, Y2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

                if (inter / areaA) > iou_thresh and areaA < areaB:
                    keep = False
                    break

            if keep:
                out.append([x1, y1, x2, y2])
        return out

    def _merge_boxes_horizontally(self, boxes: Boxes, *, min_vertical_overlap: float) -> Boxes:
        """Groupe les boxes par overlap vertical et fusionne chaque groupe en une box englobante."""
        if not boxes:
            return []

        boxes_sorted = sorted(boxes, key=lambda b: b[1])

        bands: List[Boxes] = []
        current: Boxes = [boxes_sorted[0]]

        for box in boxes_sorted[1:]:
            x1, y1, x2, y2 = box
            has_overlap = False

            for ref in current:
                X1, Y1, X2, Y2 = ref
                oy1 = max(y1, Y1)
                oy2 = min(y2, Y2)
                oh = max(0, oy2 - oy1)
                h1 = y2 - y1
                h2 = Y2 - Y1
                mh = min(h1, h2)

                if mh > 0 and (oh / mh) >= min_vertical_overlap:
                    has_overlap = True
                    break

            if has_overlap:
                current.append(box)
            else:
                bands.append(current)
                current = [box]

        bands.append(current)

        merged: Boxes = []
        for band in bands:
            x1m = min(b[0] for b in band)
            y1m = min(b[1] for b in band)
            x2m = max(b[2] for b in band)
            y2m = max(b[3] for b in band)
            merged.append([x1m, y1m, x2m, y2m])

        return merged

    def _merge_boxes_brutal(self, boxes: Boxes, *, overlap_thresh: float) -> Boxes:
        """Fusionne itérativement les boxes si elles se recouvrent assez ou si elles se touchent en Y (horizontalement)."""
        merged = True
        boxes = [list(b) for b in boxes]

        while merged:
            merged = False
            new: Boxes = []
            used = set()

            for i in range(len(boxes)):
                if i in used:
                    continue

                B = boxes[i][:]

                for j in range(i + 1, len(boxes)):
                    if j in used:
                        continue

                    X1, Y1, X2, Y2 = boxes[j]

                    inter_w = max(0, min(B[2], X2) - max(B[0], X1))
                    inter_h = max(0, min(B[3], Y2) - max(B[1], Y1))
                    inter = inter_w * inter_h

                    areaA = self._box_area(B)
                    areaB = self._box_area([X1, Y1, X2, Y2])

                    if inter > 0 and areaA > 0 and areaB > 0 and (
                        inter / areaA > overlap_thresh or inter / areaB > overlap_thresh
                    ):
                        B = [min(B[0], X1), min(B[1], Y1), max(B[2], X2), max(B[3], Y2)]
                        used.add(j)
                        merged = True

                    if abs(B[1] - Y2) < 20 or abs(B[3] - Y1) < 20:
                        B = [min(B[0], X1), min(B[1], Y1), max(B[2], X2), max(B[3], Y2)]
                        used.add(j)
                        merged = True

                used.add(i)
                new.append(B)

            boxes = new

        return boxes

    def _needs_fallback(self, boxes: Boxes, H: int, W: int, *, area_ratio: float) -> bool:
        """Retourne True si la segmentation n'a qu'une box qui couvre une grande partie de la page."""
        if len(boxes) == 1:
            x1, y1, x2, y2 = boxes[0]
            area = (x2 - x1) * (y2 - y1)
            return (area / float(H * W)) > area_ratio
        return False

    def _split_dominant_block(
        self,
        pil_img: Image.Image,
        block: Box,
        *,
        title_ratio: float = 0.22,
        valley_rel_thresh: float = 0.75,
    ) -> Boxes:
        """Découpe une grosse box en [titre] + [corps] ou [titre] + [col_gauche] + [col_droite]."""
        x1, y1, x2, y2 = block
        crop = np.array(pil_img.crop((x1, y1, x2, y2)).convert("L"))
        H, W = crop.shape

        title_h = max(60, int(H * title_ratio))
        title_h = min(title_h, H - 60)
        title_box = [x1, y1, x2, y1 + title_h]

        body = crop[title_h:, :]
        body_y0 = y1 + title_h
        Hb, Wb = body.shape

        vproj = np.sum(body < 200, axis=0)

        margin = max(5, int(0.10 * Wb))
        inner = vproj[margin: Wb - margin]
        if inner.size == 0:
            return [title_box, [x1, body_y0, x2, y2]]

        max_val = inner.max()
        min_idx = int(inner.argmin()) + margin
        min_val = float(inner[min_idx - margin])

        has_two_cols = (max_val > 0) and (min_val / float(max_val) < valley_rel_thresh)

        out: Boxes = [title_box]
        if has_two_cols:
            split_x = min_idx
            out.append([x1, body_y0, x1 + split_x, y2])
            out.append([x1 + split_x, body_y0, x2, y2])
        else:
            out.append([x1, body_y0, x2, y2])

        return out


# ------------------------------------------------------------
# CLI demo
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=str, help="Dossier contenant images/PDF")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-angle", type=float, default=20.0)
    ap.add_argument("--seg-debug-print", action="store_true")
    ap.add_argument("--return-debug", action="store_true", help="Retourne debug arrays (RAM++)")
    args = ap.parse_args()

    proc = CustomSegmentation(segment_debug_print=args.seg_debug_print)
    proc.demo_folder(
        args.folder,
        n=args.n,
        seed=args.seed,
        max_angle=args.max_angle,
        return_debug=args.return_debug,
    )

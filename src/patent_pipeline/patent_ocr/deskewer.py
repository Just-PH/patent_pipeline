from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import cv2


@dataclass
class Deskewer:
    """
    Deskew global d'une page (mono-canal) avec choix de méthode via `method`.

    method:
      - "hough"   : angles via HoughLinesP (robuste quand il y a des lignes)
      - "minarea" : orientation via minAreaRect sur pixels d'encre (souvent bon sur blocs denses)
      - "none"    : ne fait rien
    """
    method: str = "hough"

    # crop-to-content
    pad: int = 150
    crop_min_area: int = 500
    crop_min_ink_frac: float = 0.001

    # hough params (tu peux affiner plus tard)
    hough_canny1: int = 50
    hough_canny2: int = 150
    hough_threshold: int = 120
    hough_min_line_len_frac: float = 0.25
    hough_max_line_gap: int = 20
    hough_min_angles: int = 5

    # minarea params
    minarea_sample_max_points: int = 200_000  # pour éviter de traiter des millions de pixels d'encre

    def deskew(self, *, gray: np.ndarray, max_angle: float) -> Tuple[np.ndarray, float]:
        """
        Applique un deskew sur une image gray (H,W) uint8.
        Retourne (gray_deskewed, angle_degrees).
        """
        method = (self.method or "hough").lower().strip()
        if method in {"none", "off", "false", "0"}:
            return gray, 0.0

        crop = self._crop_to_content(gray, pad=self.pad)

        if method == "hough":
            angle = self._estimate_skew_angle_hough(crop, max_angle=max_angle)
        elif method in {"minarea", "min_area", "minarearect"}:
            angle = self._estimate_skew_angle_minarea(crop, max_angle=max_angle)
        else:
            raise ValueError(f"Unknown deskew method: {self.method!r}")

        if abs(angle) < 0.1 or abs(angle) > max_angle:
            return gray, 0.0

        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(
            gray,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated, float(angle)

    # -----------------------
    # Internals
    # -----------------------
    def _crop_to_content(
        self,
        gray: np.ndarray,
        *,
        pad: int,
    ) -> np.ndarray:
        """Crop contenu central via CC d'encre (évite d'estimer l'angle sur des marges)."""
        h, w = gray.shape

        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = (bw == 0).astype(np.uint8)

        if float(ink.mean()) < self.crop_min_ink_frac:
            return gray

        num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

        keep = np.zeros_like(ink)
        for i in range(1, num):
            x, y, ww, hh, area = stats[i]
            if area < self.crop_min_area:
                continue
            touches_border = (x == 0) or (y == 0) or (x + ww >= w) or (y + hh >= h)
            if touches_border:
                continue
            keep[labels == i] = 1

        if keep.sum() == 0:
            return gray

        ys, xs = np.where(keep > 0)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)

        return gray[y1 : y2 + 1, x1 : x2 + 1]

    def _estimate_skew_angle_hough(self, gray: np.ndarray, *, max_angle: float) -> float:
        """Médiane des angles des segments quasi-horizontaux via HoughLinesP."""
        g = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(g, self.hough_canny1, self.hough_canny2, apertureSize=3)

        min_line_len = int(min(gray.shape) * self.hough_min_line_len_frac)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=min_line_len,
            maxLineGap=self.hough_max_line_gap,
        )
        if lines is None:
            return 0.0

        angles: List[float] = []
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = (x2 - x1), (y2 - y1)
            if dx == 0:
                continue
            ang = float(np.degrees(np.arctan2(dy, dx)))
            if abs(ang) <= max_angle:
                angles.append(ang)

        if len(angles) < self.hough_min_angles:
            return 0.0

        return float(np.median(angles))

    def _estimate_skew_angle_minarea(self, gray: np.ndarray, *, max_angle: float) -> float:
        """
        Estimation d'angle via minAreaRect sur les pixels d'encre.
        Bon sur pages denses / colonnes, parfois meilleur que Hough.
        """
        # binaire encre
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = (bw == 0).astype(np.uint8)

        ys, xs = np.where(ink > 0)
        if xs.size < 2000:
            return 0.0

        # subsample si énorme
        if xs.size > self.minarea_sample_max_points:
            idx = np.random.choice(xs.size, self.minarea_sample_max_points, replace=False)
            xs = xs[idx]
            ys = ys[idx]

        pts = np.column_stack([xs, ys]).astype(np.float32)
        rect = cv2.minAreaRect(pts)  # ((cx,cy),(w,h), angle)
        angle = float(rect[-1])

        # OpenCV: angle in [-90, 0). Normalisation pour obtenir un angle "petit" autour de 0
        # Convention courante:
        #  - si angle < -45 => angle = 90 + angle
        if angle < -45.0:
            angle = 90.0 + angle

        # à ce stade angle est ~[-45, +45]
        if abs(angle) > max_angle:
            return 0.0
        return angle

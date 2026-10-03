"""
core/segmentation.py
---------------------
7 métodos clásicos de segmentación de imágenes refactorizados como
una clase unificada orientada a imágenes agrícolas / satelitales.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from sklearn.cluster import KMeans


class ClassicSegmenter:
    """
    Encapsula los 7 métodos clásicos de segmentación.
    """

    _METHODS: dict[str, str] = {
        "otsu":           "_otsu",
        "canny":          "_canny",
        "region_growing": "_region_growing",
        "watershed":      "_watershed",
        "kmeans":         "_kmeans",
        "meanshift":      "_meanshift",
        "grabcut":        "_grabcut",
    }

    def run(
        self,
        method_name: str,
        image_rgb: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        key = method_name.lower().strip()
        if key not in self._METHODS:
            available = ", ".join(self._METHODS)
            raise ValueError(
                f"Método '{method_name}' no reconocido. "
                f"Disponibles: {available}"
            )

        image_rgb = self._validate(image_rgb)
        method_fn = getattr(self, self._METHODS[key])
        return method_fn(image_rgb, **params)

    @staticmethod
    def available_methods() -> list[str]:
        return list(ClassicSegmenter._METHODS.keys())

    @staticmethod
    def _validate(image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.ndim == 2:
            image_rgb = np.stack([image_rgb] * 3, axis=-1)
        if image_rgb.shape[2] > 3:
            image_rgb = image_rgb[:, :, :3]
        return image_rgb.astype(np.uint8)

    @staticmethod
    def _to_gray(image_rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # ------------------------------------------------------------------
    # 1. OTSU — Umbralización
    # ------------------------------------------------------------------
    def _otsu(
        self,
        image_rgb: np.ndarray,
        blur_kernel: int = 5,
        **kwargs
    ) -> np.ndarray:
        gray = self._to_gray(image_rgb)
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        _, binary = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return self._gray_to_rgb(binary)

    # ------------------------------------------------------------------
    # 2. CANNY — Detección de bordes (Compatible con sigma y **kwargs)
    # ------------------------------------------------------------------
    def _canny(
        self,
        image_rgb: np.ndarray,
        sigma: float = 1.0,
        low_threshold: int | None = None,
        high_threshold: int | None = None,
        **kwargs
    ) -> np.ndarray:
        gray = self._to_gray(image_rgb)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        
        if low_threshold is None or high_threshold is None:
            v = np.median(blurred)
            lower = int(max(0, (1.0 - 0.33) * v))
            upper = int(min(255, (1.0 + 0.33) * v))
        else:
            lower = low_threshold
            upper = high_threshold
            
        edges = cv2.Canny(blurred, lower, upper)
        return self._gray_to_rgb(edges)

    # ------------------------------------------------------------------
    # 3. REGION GROWING — Crecimiento de regiones
    # ------------------------------------------------------------------
    def _region_growing(
        self,
        image_rgb: np.ndarray,
        seed: tuple[int, int] | None = None,
        tolerance: float = 15.0,
        **kwargs
    ) -> np.ndarray:
        gray = self._to_gray(image_rgb)
        h, w = gray.shape

        if tolerance <= 1.0:
            tolerance = tolerance * 255.0

        if seed is None:
            seed = (w // 2, h // 2)

        x0, y0 = int(seed[0]), int(seed[1])
        x0 = np.clip(x0, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)

        mask = np.zeros((h, w), dtype=np.uint8)
        visited = np.zeros((h, w), dtype=bool)

        pending: list[tuple[int, int]] = [(x0, y0)]
        visited[y0, x0] = True
        mask[y0, x0] = 255

        total = float(gray[y0, x0])
        count = 1

        neighbors = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1),
        ]

        while pending:
            cx, cy = pending.pop()
            mean = total / count

            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                if visited[ny, nx]:
                    continue
                visited[ny, nx] = True

                val = float(gray[ny, nx])
                if abs(val - mean) <= tolerance:
                    mask[ny, nx] = 255
                    pending.append((nx, ny))
                    total += val
                    count += 1

        return self._gray_to_rgb(mask)

    # ------------------------------------------------------------------
    # 4. WATERSHED
    # ------------------------------------------------------------------
    def _watershed(
        self,
        image_rgb: np.ndarray,
        compactness: float = 0.01,
        **kwargs
    ) -> np.ndarray:
        gray = self._to_gray(image_rgb)

        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        thresh_val = np.clip(0.5 + (compactness * 5), 0.1, 0.9)
        _, sure_fg = cv2.threshold(dist, thresh_val * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        markers = cv2.watershed(bgr, markers)

        result = image_rgb.copy()
        result[markers == -1] = [255, 0, 0]
        return result

    # ------------------------------------------------------------------
    # 5. K-MEANS
    # ------------------------------------------------------------------
    def _kmeans(
        self,
        image_rgb: np.ndarray,
        n_clusters: int = 4,
        random_state: int = 42,
        **kwargs
    ) -> np.ndarray:
        k = int(np.clip(n_clusters, 2, 20))
        pixels = image_rgb.reshape(-1, 3).astype(np.float32)

        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        )
        labels = model.fit_predict(pixels)
        centers = np.uint8(model.cluster_centers_)

        result = centers[labels].reshape(image_rgb.shape)
        return result.astype(np.uint8)

    # ------------------------------------------------------------------
    # 6. MEAN-SHIFT
    # ------------------------------------------------------------------
    def _meanshift(
        self,
        image_rgb: np.ndarray,
        spatial_radius: int = 15,
        color_radius: int = 30,
        **kwargs
    ) -> np.ndarray:
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        filtered = cv2.pyrMeanShiftFiltering(bgr, sp=spatial_radius, sr=color_radius)
        return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # 7. GRABCUT
    # ------------------------------------------------------------------
    def _grabcut(
        self,
        image_rgb: np.ndarray,
        margin_frac: float = 0.10,
        iterCount: int = 5,
        **kwargs
    ) -> np.ndarray:
        h, w = image_rgb.shape[:2]

        margin_x = max(5, int(w * margin_frac))
        margin_y = max(5, int(h * margin_frac))

        rect = (
            margin_x,
            margin_y,
            w - 2 * margin_x,
            h - 2 * margin_y,
        )

        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        mask = np.zeros((h, w), np.uint8)
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            bgr, mask, rect,
            bg_model, fg_model,
            iterCount,
            cv2.GC_INIT_WITH_RECT,
        )

        binary_mask = np.where(
            (mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
            0, 255,
        ).astype(np.uint8)

        result = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)
        return result.astype(np.uint8)
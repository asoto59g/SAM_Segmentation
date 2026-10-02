"""
core/segmentation.py
---------------------
7 métodos clásicos de segmentación de imágenes refactorizados como
una clase unificada orientada a imágenes agrícolas / satelitales.

Métodos disponibles
-------------------
- otsu          : Umbralización automática (Otsu)
- canny         : Detección de bordes (Canny)
- region_growing: Crecimiento de regiones
- watershed     : Segmentación Watershed
- kmeans        : Clustering K-Means por color
- meanshift     : Filtrado Mean-Shift (OpenCV pyrMeanShiftFiltering)
- grabcut       : Separación fondo / primer plano (GrabCut)

Todos los métodos reciben y devuelven arrays NumPy uint8 (H, W, 3) RGB.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class ClassicSegmenter:
    """
    Encapsula los 7 métodos clásicos de segmentación.

    Uso básico
    ----------
    >>> segmenter = ClassicSegmenter()
    >>> result = segmenter.run("kmeans", image_rgb, k=5)
    """

    # Mapa nombre → método interno
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
        """
        Ejecuta el método de segmentación indicado.

        Parameters
        ----------
        method_name : str
            Nombre del método (ver ``ClassicSegmenter._METHODS``).
        image_rgb : np.ndarray
            Imagen de entrada uint8 (H, W, 3) en orden RGB.
        **params : Any
            Parámetros específicos de cada método.

        Returns
        -------
        np.ndarray
            Resultado uint8 (H, W, 3) RGB.

        Raises
        ------
        ValueError
            Si el nombre de método no está registrado.
        """
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
        """Retorna la lista de nombres de métodos disponibles."""
        return list(ClassicSegmenter._METHODS.keys())

    # ------------------------------------------------------------------
    # Validación interna
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(image_rgb: np.ndarray) -> np.ndarray:
        """Asegura que la imagen sea uint8 (H, W, 3)."""
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
    ) -> np.ndarray:
        """
        Umbralización automática de Otsu.

        Parameters
        ----------
        blur_kernel : int
            Tamaño del kernel Gaussiano de suavizado (impar ≥ 1).
        """
        gray = self._to_gray(image_rgb)

        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        _thresh, binary = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return self._gray_to_rgb(binary)

    # ------------------------------------------------------------------
    # 2. CANNY — Detección de bordes
    # ------------------------------------------------------------------

    def _canny(
        self,
        image_rgb: np.ndarray,
        low_threshold: int = 100,
        high_threshold: int = 200,
    ) -> np.ndarray:
        """
        Detección de bordes de Canny.

        Parameters
        ----------
        low_threshold : int
            Umbral bajo para histéresis.
        high_threshold : int
            Umbral alto para histéresis.
        """
        gray = self._to_gray(image_rgb)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return self._gray_to_rgb(edges)

    # ------------------------------------------------------------------
    # 3. REGION GROWING — Crecimiento de regiones
    # ------------------------------------------------------------------

    def _region_growing(
        self,
        image_rgb: np.ndarray,
        seed: tuple[int, int] | None = None,
        tolerance: float = 15.0,
    ) -> np.ndarray:
        """
        Crecimiento de regiones desde una semilla.

        Parameters
        ----------
        seed : tuple[int, int] | None
            Punto inicial (col, row). Si es None, usa el centro.
        tolerance : float
            Máxima diferencia de intensidad aceptada para agregar un píxel.
        """
        gray = self._to_gray(image_rgb)
        h, w = gray.shape

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
    ) -> np.ndarray:
        """
        Segmentación Watershed con marcadores derivados de la
        transformada de distancia.
        """
        gray = self._to_gray(image_rgb)

        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        markers = cv2.watershed(bgr, markers)

        result = image_rgb.copy()
        result[markers == -1] = [255, 0, 0]  # fronteras en rojo
        return result

    # ------------------------------------------------------------------
    # 5. K-MEANS
    # ------------------------------------------------------------------

    def _kmeans(
        self,
        image_rgb: np.ndarray,
        k: int = 4,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Segmentación por clustering K-Means sobre color RGB.

        Parameters
        ----------
        k : int
            Número de clusters (2 ≤ k ≤ 16).
        random_state : int
            Semilla para reproducibilidad.
        """
        k = int(np.clip(k, 2, 16))
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
        sp: int = 15,
        sr: int = 30,
    ) -> np.ndarray:
        """
        Segmentación Mean-Shift con ``cv2.pyrMeanShiftFiltering``.

        Parameters
        ----------
        sp : int
            Tamaño espacial de la ventana (radio en píxeles).
        sr : int
            Radio de búsqueda de color (rango de color).
        """
        # pyrMeanShiftFiltering opera en BGR
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        filtered = cv2.pyrMeanShiftFiltering(bgr, sp=sp, sr=sr)
        return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # 7. GRABCUT
    # ------------------------------------------------------------------

    def _grabcut(
        self,
        image_rgb: np.ndarray,
        margin_frac: float = 0.10,
        iterations: int = 5,
    ) -> np.ndarray:
        """
        Separación fondo / primer plano con GrabCut.

        Un rectángulo automático ocupa el (1 - 2·margin_frac) de la imagen.

        Parameters
        ----------
        margin_frac : float
            Fracción de margen respecto al ancho/alto total (0.05 – 0.30).
        iterations : int
            Número de iteraciones del algoritmo GrabCut.
        """
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
            iterations,
            cv2.GC_INIT_WITH_RECT,
        )

        # 0 y 2 = fondo definitivo / probable fondo → negro
        # 1 y 3 = primer plano definitivo / probable → conservar
        binary_mask = np.where(
            (mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
            0, 255,
        ).astype(np.uint8)

        result = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)
        return result.astype(np.uint8)

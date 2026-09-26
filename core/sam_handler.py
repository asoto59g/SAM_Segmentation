"""
core/sam_handler.py
--------------------
Integración de Segment Anything Model (SAM) ViT-L.

Modos disponibles
-----------------
- auto_segment   : Genera máscaras automáticamente sobre toda la imagen.
- point_segment  : El usuario proporciona puntos (x, y) como prompts.

La carga del modelo es lazy (se hace solo la primera vez que se necesita).
Usa CUDA automáticamente si está disponible; de lo contrario, usa CPU.

Dependencias
------------
    pip install torch torchvision
    pip install git+https://github.com/facebookresearch/segment-anything.git

Checkpoint recomendado: sam_vit_l_0b3195.pth (~1.2 GB)
https://github.com/facebookresearch/segment-anything#model-checkpoints
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MODEL_TYPE = "vit_l"

# Parámetros optimizados para imágenes agrícolas / satelitales:
# - Menos puntos por lado → regiones más grandes (campos, parcelas)
# - IOU alto → menos máscaras de baja calidad
# - Área mínima → ignora píxeles aislados / ruido
_AUTO_PARAMS = {
    "points_per_side": 16,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.92,
    "min_mask_region_area": 500,
}


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class SAMHandler:
    """
    Gestiona la carga y la inferencia de SAM ViT-L.

    Attributes
    ----------
    checkpoint_path : Path | None
        Ruta al archivo .pth del checkpoint.
    _model : sam_model_registry | None
        Instancia del modelo SAM (None hasta que se cargue).
    _device : str
        ``'cuda'`` o ``'cpu'``.
    _loaded : bool
        True si el modelo ya está en memoria.
    """

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self.checkpoint_path: Path | None = (
            Path(checkpoint_path) if checkpoint_path else None
        )
        self._model = None
        self._device: str = self._detect_device()
        self._loaded: bool = False
        self._error_msg: str | None = None

    # ------------------------------------------------------------------
    # Propiedades públicas
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True si SAM está instalado y el checkpoint existe."""
        return self._sam_installed() and self._checkpoint_ok()

    @property
    def is_loaded(self) -> bool:
        """True si el modelo ya está cargado en memoria."""
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    @property
    def error_message(self) -> str | None:
        return self._error_msg

    # ------------------------------------------------------------------
    # Carga del modelo
    # ------------------------------------------------------------------

    def load_model(
        self,
        checkpoint_path: str | Path | None = None,
    ) -> bool:
        """
        Carga el modelo SAM en memoria.

        Parameters
        ----------
        checkpoint_path : str | Path | None
            Si se proporciona, sobreescribe el path configurado en ``__init__``.

        Returns
        -------
        bool
            True si la carga fue exitosa.
        """
        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)

        if not self._sam_installed():
            self._error_msg = (
                "SAM no está instalado. Ejecuta:\n"
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
            return False

        if not self._checkpoint_ok():
            self._error_msg = (
                f"Checkpoint no encontrado: {self.checkpoint_path}\n\n"
                "Descárgalo desde:\n"
                "https://github.com/facebookresearch/segment-anything#model-checkpoints\n\n"
                "Coloca 'sam_vit_l_0b3195.pth' en la carpeta models/"
            )
            return False

        try:
            from segment_anything import sam_model_registry
            self._model = sam_model_registry[MODEL_TYPE](
                checkpoint=str(self.checkpoint_path)
            )
            self._model.to(self._device)
            self._loaded = True
            self._error_msg = None
            return True
        except Exception as exc:
            self._error_msg = f"Error al cargar SAM: {exc}"
            self._loaded = False
            return False

    # ------------------------------------------------------------------
    # Modo automático
    # ------------------------------------------------------------------

    def auto_segment(
        self,
        image_rgb: np.ndarray,
        points_per_side: int = _AUTO_PARAMS["points_per_side"],
        pred_iou_thresh: float = _AUTO_PARAMS["pred_iou_thresh"],
        stability_score_thresh: float = _AUTO_PARAMS["stability_score_thresh"],
        min_mask_region_area: int = _AUTO_PARAMS["min_mask_region_area"],
    ) -> tuple[list[dict] | None, str | None]:
        """
        Genera máscaras automáticamente sobre toda la imagen.

        Parameters
        ----------
        image_rgb : np.ndarray
            Imagen uint8 (H, W, 3) RGB.
        points_per_side : int
            Densidad de la grilla de puntos de muestreo.
        pred_iou_thresh : float
            Umbral mínimo de IoU predicho para aceptar una máscara.
        stability_score_thresh : float
            Umbral de estabilidad de máscara.
        min_mask_region_area : int
            Área mínima en píxeles para filtrar regiones pequeñas.

        Returns
        -------
        masks : list[dict] | None
            Lista de diccionarios SAM con clave ``'segmentation'`` (array bool).
            None si ocurrió un error.
        error : str | None
            Mensaje de error (None si fue exitoso).
        """
        if not self._loaded:
            ok = self.load_model()
            if not ok:
                return None, self._error_msg

        try:
            from segment_anything import SamAutomaticMaskGenerator
            generator = SamAutomaticMaskGenerator(
                model=self._model,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
            )
            image_uint8 = np.asarray(image_rgb, dtype=np.uint8)
            masks = generator.generate(image_uint8)
            # Ordenar de mayor a menor área
            masks.sort(key=lambda m: m["area"], reverse=True)
            return masks, None
        except Exception as exc:
            return None, f"Error en auto_segment: {exc}"

    # ------------------------------------------------------------------
    # Modo por clic (predictor con puntos)
    # ------------------------------------------------------------------

    def point_segment(
        self,
        image_rgb: np.ndarray,
        points: list[tuple[int, int]],
        labels: list[int],
    ) -> tuple[list[dict] | None, str | None]:
        """
        Segmenta la imagen usando puntos del usuario como prompts.

        Parameters
        ----------
        image_rgb : np.ndarray
            Imagen uint8 (H, W, 3) RGB.
        points : list[tuple[int, int]]
            Lista de coordenadas (x, y) en píxeles sobre la imagen mostrada.
        labels : list[int]
            1 = punto positivo (objeto), 0 = punto negativo (fondo).

        Returns
        -------
        masks : list[dict] | None
            Lista de máscaras predichas (ordenadas por score descendente).
            Cada dict tiene claves ``'segmentation'``, ``'score'``.
        error : str | None
        """
        if not points:
            return None, "No hay puntos definidos. Haz clic sobre la imagen."

        if len(points) != len(labels):
            return None, "El número de puntos y etiquetas debe coincidir."

        if not self._loaded:
            ok = self.load_model()
            if not ok:
                return None, self._error_msg

        try:
            from segment_anything import SamPredictor
            predictor = SamPredictor(self._model)
            image_uint8 = np.asarray(image_rgb, dtype=np.uint8)
            predictor.set_image(image_uint8)

            input_points = np.array(points, dtype=np.float32)
            input_labels = np.array(labels, dtype=np.int32)

            masks_arr, scores, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )

            # Convertir a lista de dicts al mismo formato que auto_segment
            result: list[dict] = []
            for mask, score in zip(masks_arr, scores):
                result.append({
                    "segmentation": mask.astype(bool),
                    "score": float(score),
                    "area": int(mask.sum()),
                })

            # Ordenar por score descendente
            result.sort(key=lambda m: m["score"], reverse=True)
            return result, None

        except Exception as exc:
            return None, f"Error en point_segment: {exc}"

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        """Retorna 'cuda' si PyTorch + CUDA están disponibles, 'cpu' si no."""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @staticmethod
    def _sam_installed() -> bool:
        try:
            import segment_anything  # noqa: F401
            return True
        except ImportError:
            return False

    def _checkpoint_ok(self) -> bool:
        if self.checkpoint_path is None:
            return False
        return Path(self.checkpoint_path).exists()

    # ------------------------------------------------------------------
    # Utilidades de diagnóstico
    # ------------------------------------------------------------------

    def status_info(self) -> dict:
        """
        Retorna un diccionario con información de estado para mostrar en UI.

        Returns
        -------
        dict
            Claves: ``sam_installed``, ``checkpoint_found``, ``model_loaded``,
            ``device``, ``checkpoint_path``, ``error``.
        """
        return {
            "sam_installed": self._sam_installed(),
            "checkpoint_found": self._checkpoint_ok(),
            "model_loaded": self._loaded,
            "device": self._device,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else "No configurado",
            "error": self._error_msg,
        }

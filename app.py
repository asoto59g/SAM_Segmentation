"""
core/sam_handler.py
-------------------
Manejo de la inferencia y carga de modelos Segment Anything (SAM).
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    _SAM_INSTALLED = True
except ImportError:
    _SAM_INSTALLED = False


class SAMHandler:
    def __init__(self, checkpoint_path: str, model_type: str = "vit_b"):
        """
        Inicializa el handler de SAM.
        
        Args:
            checkpoint_path: Ruta al archivo .pth del modelo.
            model_type: Tipo de modelo ("vit_b", "vit_l", "vit_h"). Por defecto "vit_b".
        """
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = None
        self.mask_generator = None
        self.predictor = None

    def status_info(self) -> dict:
        """Retorna información sobre la disponibilidad de SAM y el checkpoint."""
        checkpoint_exists = Path(self.checkpoint_path).exists()
        file_size_ok = (
            checkpoint_exists and Path(self.checkpoint_path).stat().st_size > 50_000_000
        )
        return {
            "sam_installed": _SAM_INSTALLED,
            "checkpoint_found": file_size_ok,
            "model_loaded": self.sam is not None,
            "device": self.device,
        }

    def _load_model(self) -> str | None:
        """Carga los pesos del modelo en memoria si no están cargados."""
        if not _SAM_INSTALLED:
            return "La librería 'segment-anything' no está instalada."

        if self.sam is not None:
            return None

        if not Path(self.checkpoint_path).exists():
            return f"Checkpoint no encontrado en: {self.checkpoint_path}"

        try:
            # Sincronizado con la arquitectura 'vit_b' (375 MB)
            self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            self.sam.to(device=self.device)
            self.predictor = SamPredictor(self.sam)
            return None
        except Exception as exc:
            return f"Error al cargar SAM ({self.model_type}): {exc}"

    def auto_segment(
        self,
        image_rgb: np.ndarray,
        points_per_side: int = 12,
        pred_iou_thresh: float = 0.88,
        min_mask_region_area: int = 500,
    ) -> tuple[list[dict] | None, str | None]:
        """Genera máscaras automáticas en toda la imagen."""
        err = self._load_model()
        if err:
            return None, err

        try:
            generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                min_mask_region_area=min_mask_region_area,
            )
            masks = generator.generate(image_rgb)
            return masks, None
        except Exception as exc:
            return None, f"Error durante segmentación automática: {exc}"

    def point_segment(
        self,
        image_rgb: np.ndarray,
        points: list[tuple[int, int]],
        labels: list[int],
    ) -> tuple[list[dict] | None, str | None]:
        """Genera máscaras a partir de puntos (clics) interactivos."""
        err = self._load_model()
        if err:
            return None, err

        if not points:
            return None, "No se proporcionaron puntos de interés."

        try:
            self.predictor.set_image(image_rgb)
            input_points = np.array(points)
            input_labels = np.array(labels)

            masks_raw, scores, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )

            # Seleccionar la máscara con mayor puntuación
            best_idx = np.argmax(scores)
            selected_mask = masks_raw[best_idx]

            formatted_masks = [{"segmentation": selected_mask, "area": np.sum(selected_mask)}]
            return formatted_masks, None
        except Exception as exc:
            return None, f"Error durante segmentación por puntos: {exc}"
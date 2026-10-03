"""
sam_handler.py
--------------
Módulo encargado de la inicialización y ejecución del modelo Segment Anything (SAM).
Soporta arquitecturas ViT-B, ViT-L y ViT-H dinámicamente según el archivo .pth seleccionado.
"""

from __future__ import annotations

import traceback
from pathlib import Path
import numpy as np
import torch

try:
    from segment_anything import (
        SamAutomaticMaskGenerator,
        SamPredictor,
        sam_model_registry,
    )
    _SAM_AVAILABLE = True
except ImportError:
    _SAM_AVAILABLE = False


class SAMHandler:
    """Clase manejadora para cargar el modelo SAM y realizar inferencias."""

    def __init__(self, checkpoint_path: str, model_type: str = "vit_b"):
        """
        Inicializa el handler de SAM.

        Args:
            checkpoint_path (str): Ruta al archivo de pesos (.pth).
            model_type (str): Tipo de arquitectura ('vit_b', 'vit_l', 'vit_h').
        """
        if not _SAM_AVAILABLE:
            raise ImportError(
                "La librería 'segment_anything' no está instalada. "
                "Instálala con: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        self.checkpoint_path = str(checkpoint_path)
        self.model_type = model_type.lower()

        if self.model_type not in sam_model_registry:
            raise ValueError(
                f"Tipo de modelo '{self.model_type}' no reconocido. "
                f"Opciones válidas: {list(sam_model_registry.keys())}"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def auto_segment(
        self,
        image_rgb: np.ndarray,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 100,
        **kwargs,
    ) -> tuple[list[dict] | None, str | None]:
        """
        Genera una segmentación automática en toda la imagen.
        """
        try:
            mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
                crop_n_layers=1 if points_per_side >= 32 else 0,
            )
            masks = mask_generator.generate(image_rgb)
            return masks, None
        except Exception:
            err_msg = f"Error durante la segmentación automática SAM:\n{traceback.format_exc()}"
            return None, err_msg

    def point_segment(
        self,
        image_rgb: np.ndarray,
        points: list[tuple[int, int]],
        labels: list[int],
    ) -> tuple[list[dict] | None, str | None]:
        """
        Genera segmentación basada en clics/puntos (prompting).
        """
        if not points:
            return None, "No se proporcionaron puntos para la segmentación."

        try:
            self.predictor.set_image(image_rgb)

            input_points = np.array(points, dtype=np.float32)
            input_labels = np.array(labels, dtype=np.int32)

            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )

            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]

            formatted_masks = [
                {
                    "segmentation": best_mask,
                    "area": int(np.sum(best_mask)),
                    "predicted_iou": float(scores[best_idx]),
                }
            ]

            return formatted_masks, None
        except Exception:
            err_msg = f"Error durante la segmentación por clics SAM:\n{traceback.format_exc()}"
            return None, err_msg
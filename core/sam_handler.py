"""
core/sam_handler.py
-------------------
Manejo de la inferencia y carga de modelos Segment Anything (SAM).
Optimizado para bajo consumo de memoria RAM y CPU en la nube.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
import numpy as np

# Configurar variables de entorno antes de cualquier cálculo
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

# Limitar hilos en PyTorch de forma segura para evitar throttling
try:
    torch.set_num_threads(1)
except Exception:
    pass

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    _SAM_INSTALLED = True
except ImportError:
    _SAM_INSTALLED = False

class SAMHandler:
    def __init__(self, checkpoint_path: str, model_type: str = "vit_b"):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = None
        self.mask_generator = None
        self.predictor = None

    def _load_model(self) -> str | None:
        if not _SAM_INSTALLED:
            return "La librería 'segment-anything' no está instalada."

        if self.sam is not None:
            return None

        if not Path(self.checkpoint_path).exists():
            return f"Checkpoint no encontrado en: {self.checkpoint_path}"

        try:
            self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            self.sam.to(device=self.device)
            self.sam.eval()
            self.predictor = SamPredictor(self.sam)
            return None
        except Exception as exc:
            return f"Error al cargar SAM ({self.model_type}): {exc}"

    def auto_segment(
        self,
        image_rgb: np.ndarray,
        points_per_side: int = 8,  # Mantenido bajo para proteger CPU
        pred_iou_thresh: float = 0.88,
        min_mask_region_area: int = 500,
    ) -> tuple[list[dict] | None, str | None]:
        err = self._load_model()
        if err:
            return None, err

        try:
            with torch.no_grad(): # Liberar memoria de gradientes
                generator = SamAutomaticMaskGenerator(
                    model=self.sam,
                    points_per_side=points_per_side,
                    pred_iou_thresh=pred_iou_thresh,
                    min_mask_region_area=min_mask_region_area,
                )
                masks = generator.generate(image_rgb)
            
            gc.collect() # Limpieza forzada de RAM
            return masks, None
        except Exception as exc:
            gc.collect()
            return None, f"Error durante segmentación automática: {exc}"

    def point_segment(
        self,
        image_rgb: np.ndarray,
        points: list[tuple[int, int]],
        labels: list[int],
    ) -> tuple[list[dict] | None, str | None]:
        err = self._load_model()
        if err:
            return None, err

        if not points:
            return None, "No se proporcionaron puntos de interés."

        try:
            with torch.no_grad(): # Liberar memoria de gradientes
                self.predictor.set_image(image_rgb)
                input_points = np.array(points)
                input_labels = np.array(labels)

                masks_raw, scores, _ = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True,
                )

            best_idx = np.argmax(scores)
            selected_mask = masks_raw[best_idx]
            formatted_masks = [{"segmentation": selected_mask, "area": np.sum(selected_mask)}]
            
            gc.collect()
            return formatted_masks, None
        except Exception as exc:
            gc.collect()
            return None, f"Error durante segmentación por puntos: {exc}"
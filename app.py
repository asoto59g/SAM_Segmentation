"""
app.py
------
Aplicación Streamlit para segmentación con Segment Anything (SAM).
Incluye interfaz gráfica de usuario y optimización de hilos CPU / caché para evitar throttling.
"""

from __future__ import annotations

import os
from pathlib import Path

# 1. Configurar variables de entorno antes de cargar PyTorch/NumPy para controlar hilos
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np
from PIL import Image
import torch
import streamlit as st

# 2. Configuración segura de hilos en PyTorch
try:
    torch.set_num_threads(2)
except Exception:
    pass

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    _SAM_INSTALLED = True
except ImportError:
    _SAM_INSTALLED = False


class SAMHandler:
    def __init__(self, checkpoint_path: str, model_type: str = "vit_b"):
        """Inicializa el handler de SAM."""
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
        points_per_side: int = 8,
        pred_iou_thresh: float = 0.88,
        min_mask_region_area: int = 500,
    ) -> tuple[list[dict] | None, str | None]:
        """Genera máscaras automáticas en toda la imagen liberando gradientes de RAM."""
        err = self._load_model()
        if err:
            return None, err

        try:
            with torch.no_grad():
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


@st.cache_resource
def get_sam_handler(checkpoint_path: str, model_type: str = "vit_b") -> SAMHandler:
    """Instancia y carga el modelo SAM manteniéndolo en la memoria caché de Streamlit."""
    handler = SAMHandler(checkpoint_path=checkpoint_path, model_type=model_type)
    err = handler._load_model()
    if err:
        st.error(f"Error cargando SAM: {err}")
    return handler


# -----------------------------------------------------------------------------
# INTERFAZ DE USUARIO (STREAMLIT UI)
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Segment Anything (SAM)", layout="wide")
    st.title("Segment Anything (SAM) App")

    # Panel lateral para seleccionar opciones
    st.sidebar.header("Configuración del Modelo")
    model_type = st.sidebar.selectbox("Tipo de Modelo", ["vit_b", "vit_l", "vit_h"], index=0)
    
    # Ruta del archivo de checkpoint (.pth)
    default_ckpt = f"sam_{model_type}.pth"
    checkpoint_path = st.sidebar.text_input("Ruta del Checkpoint (.pth)", value=default_ckpt)

    # Carga del Handler optimizado
    sam_handler = get_sam_handler(checkpoint_path=checkpoint_path, model_type=model_type)
    
    # Mostrar estado del sistema
    status = sam_handler.status_info()
    with st.sidebar.expander("Estado del Entorno", expanded=False):
        st.json(status)

    st.header("Carga de Imagen")
    uploaded_file = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagen Original")
            st.image(image, use_container_width=True)

        # Ajuste de parámetros de inferencia
        st.sidebar.subheader("Parámetros de Segmentación")
        points_per_side = st.sidebar.slider("Puntos por lado (points_per_side)", 4, 16, 8)
        pred_iou_thresh = st.sidebar.slider("Umbral IOU (pred_iou_thresh)", 0.5, 1.0, 0.88)

        if st.button("Ejecutar Segmentación Automática"):
            with st.spinner("Procesando segmentación en CPU..."):
                masks, err = sam_handler.auto_segment(
                    image_np,
                    points_per_side=points_per_side,
                    pred_iou_thresh=pred_iou_thresh,
                )

            if err:
                st.error(err)
            elif masks:
                st.success(f"Se generaron {len(masks)} máscaras.")
                with col2:
                    st.subheader("Resultado de Segmentación")
                    # Crear superposición visual de máscaras
                    overlay = image_np.copy()
                    for mask in masks:
                        m = mask["segmentation"]
                        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
                        overlay[m] = overlay[m] * 0.5 + color * 0.5
                    
                    st.image(overlay, use_container_width=True)


if __name__ == "__main__":
    main()
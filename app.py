"""
app.py
------
Aplicación Streamlit para segmentación de imágenes con múltiples algoritmos
(Watershed, Otsu, Felzenszwalb, SLIC, Canny, Chan-Vese, SAM).
Optimizado para evitar throttling de CPU, errores de hilos en PyTorch y pantallas en blanco.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import streamlit as st

# Algoritmos clásicos y de scikit-image
from skimage.color import rgb2gray
from skimage.filters import sobel, threshold_otsu
from skimage.segmentation import watershed, felzenszwalb, slic, chan_vese
from skimage.feature import canny

# 1. Variables de entorno ANTES de importar PyTorch/NumPy
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import torch

# 2. Manejo seguro de hilos de PyTorch
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
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = None
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
        points_per_side: int = 8,
        pred_iou_thresh: float = 0.88,
        min_mask_region_area: int = 500,
    ) -> tuple[list[dict] | None, str | None]:
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
    handler = SAMHandler(checkpoint_path=checkpoint_path, model_type=model_type)
    err = handler._load_model()
    if err:
        st.error(f"Error cargando SAM: {err}")
    return handler


# -----------------------------------------------------------------------------
# FUNCIONES DE SEGMENTACIÓN TRADICIONAL
# -----------------------------------------------------------------------------
def run_watershed(image_rgb: np.ndarray, markers_count: int = 16) -> np.ndarray:
    gray = rgb2gray(image_rgb)
    elevation_map = sobel(gray)
    markers = np.zeros_like(gray, dtype=int)
    side = int(np.sqrt(markers_count))
    if side < 1:
        side = 1
    grid_x = np.linspace(0, gray.shape[0] - 1, side, dtype=int)
    grid_y = np.linspace(0, gray.shape[1] - 1, side, dtype=int)
    count = 1
    for x in grid_x:
        for y in grid_y:
            markers[x, y] = count
            count += 1
    segmentation = watershed(elevation_map, markers)
    return segmentation


def run_otsu(image_rgb: np.ndarray) -> np.ndarray:
    gray_float = rgb2gray(image_rgb)
    gray_uint8 = (gray_float * 255).astype(np.uint8)
    thresh = threshold_otsu(gray_uint8)
    binary = (gray_uint8 > thresh).astype(np.uint8) * 255
    return binary


def run_felzenszwalb(image_rgb: np.ndarray, scale: float = 100, sigma: float = 0.5, min_size: int = 50) -> np.ndarray:
    return felzenszwalb(image_rgb, scale=scale, sigma=sigma, min_size=min_size)


def run_slic(image_rgb: np.ndarray, n_segments: int = 100, compactness: float = 10.0) -> np.ndarray:
    return slic(image_rgb, n_segments=n_segments, compactness=compactness, start_label=1)


def run_canny(image_rgb: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    gray = rgb2gray(image_rgb)
    return canny(gray, sigma=sigma).astype(np.uint8) * 255


def run_chan_vese(image_rgb: np.ndarray, max_num_iter: int = 100) -> np.ndarray:
    gray = rgb2gray(image_rgb)
    return chan_vese(gray, max_num_iter=max_num_iter).astype(np.uint8) * 255


# -----------------------------------------------------------------------------
# INTERFAZ DE USUARIO (STREAMLIT UI)
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Herramienta de Segmentación de Imágenes", layout="wide")
    st.title("Procesamiento y Segmentación de Imágenes")

    st.sidebar.header("Selección de Algoritmo")
    model_choice = st.sidebar.selectbox(
        "Selecciona el modelo/algoritmo:",
        [
            "Watershed",
            "Otsu Thresholding",
            "Felzenszwalb",
            "SLIC (Superpixels)",
            "Canny Edge Detector",
            "Chan-Vese",
            "Segment Anything (SAM)",
        ],
    )

    st.header("1. Carga de Imagen")
    uploaded_file = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagen Original")
            st.image(image, use_container_width=True)

        st.sidebar.subheader("Parámetros del Algoritmo")

        if model_choice == "Watershed":
            markers_count = st.sidebar.slider("Número de marcadores", 4, 100, 16)
            if st.button("Ejecutar Watershed"):
                res = run_watershed(image_np, markers_count)
                with col2:
                    st.subheader("Resultado Watershed")
                    norm_res = ((res - res.min()) / (res.max() - res.min() + 1e-8) * 255).astype(np.uint8)
                    st.image(norm_res, use_container_width=True)

        elif model_choice == "Otsu Thresholding":
            if st.button("Ejecutar Otsu"):
                res = run_otsu(image_np)
                with col2:
                    st.subheader("Resultado Otsu")
                    st.image(res, use_container_width=True)

        elif model_choice == "Felzenszwalb":
            scale = st.sidebar.slider("Escala", 10, 500, 100)
            sigma = st.sidebar.slider("Sigma", 0.1, 3.0, 0.5)
            min_size = st.sidebar.slider("Tamaño Mínimo", 10, 200, 50)
            if st.button("Ejecutar Felzenszwalb"):
                res = run_felzenszwalb(image_np, scale, sigma, min_size)
                with col2:
                    st.subheader("Resultado Felzenszwalb")
                    norm_res = ((res - res.min()) / (res.max() - res.min() + 1e-8) * 255).astype(np.uint8)
                    st.image(norm_res, use_container_width=True)

        elif model_choice == "SLIC (Superpixels)":
            n_segments = st.sidebar.slider("Número de segmentos", 20, 500, 100)
            compactness = st.sidebar.slider("Compacidad", 1.0, 50.0, 10.0)
            if st.button("Ejecutar SLIC"):
                res = run_slic(image_np, n_segments, compactness)
                with col2:
                    st.subheader("Resultado SLIC")
                    norm_res = ((res - res.min()) / (res.max() - res.min() + 1e-8) * 255).astype(np.uint8)
                    st.image(norm_res, use_container_width=True)

        elif model_choice == "Canny Edge Detector":
            sigma = st.sidebar.slider("Sigma", 0.5, 5.0, 1.0)
            if st.button("Ejecutar Canny"):
                res = run_canny(image_np, sigma)
                with col2:
                    st.subheader("Bordes Canny")
                    st.image(res, use_container_width=True)

        elif model_choice == "Chan-Vese":
            max_iter = st.sidebar.slider("Máx Iteraciones", 10, 200, 50)
            if st.button("Ejecutar Chan-Vese"):
                res = run_chan_vese(image_np, max_iter)
                with col2:
                    st.subheader("Resultado Chan-Vese")
                    st.image(res, use_container_width=True)

        elif model_choice == "Segment Anything (SAM)":
            sam_type = st.sidebar.selectbox("Tipo SAM", ["vit_b", "vit_l", "vit_h"], index=0)
            ckpt_path = st.sidebar.text_input("Ruta Checkpoint", value=f"sam_{sam_type}.pth")
            points_per_side = st.sidebar.slider("Puntos por lado", 4, 16, 8)

            if st.button("Ejecutar SAM"):
                sam_handler = get_sam_handler(ckpt_path, sam_type)
                with st.spinner("Procesando en CPU..."):
                    masks, err = sam_handler.auto_segment(image_np, points_per_side=points_per_side)
                if err:
                    st.error(err)
                elif masks:
                    st.success(f"Se generaron {len(masks)} máscaras.")
                    with col2:
                        overlay = image_np.copy()
                        for mask in masks:
                            m = mask["segmentation"]
                            color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
                            overlay[m] = overlay[m] * 0.5 + color * 0.5
                        st.subheader("Resultado SAM")
                        st.image(overlay, use_container_width=True)


if __name__ == "__main__":
    main()
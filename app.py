"""
app.py
------
Aplicación Streamlit para segmentación con múltiples modelos.
Diseñado con Lazy Loading para evitar colapsos de memoria (Segfault/Oh No) en Streamlit Cloud.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import traceback

# Configuración de recursos a nivel de SO
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from PIL import Image
import streamlit as st

# La configuración de página DEBE ser el primer comando de Streamlit
st.set_page_config(page_title="Herramienta de Segmentación", layout="wide")

# Rutas del sistema
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))


# Importaciones perezosas para evitar colapsos de PyTorch al arrancar
def load_skimage_modules():
    from skimage.color import rgb2gray
    from skimage.filters import sobel, threshold_otsu
    from skimage.segmentation import watershed, felzenszwalb, slic, chan_vese
    from skimage.feature import canny
    return rgb2gray, sobel, threshold_otsu, watershed, felzenszwalb, slic, chan_vese, canny


# Cargar SAM bajo demanda con caché de Streamlit
@st.cache_resource
def get_sam_handler(checkpoint_path: str, model_type: str = "vit_b"):
    import torch
    torch.set_num_threads(1)
    from core.sam_handler import SAMHandler
    
    handler = SAMHandler(checkpoint_path=checkpoint_path, model_type=model_type)
    err = handler._load_model()
    if err:
        st.error(f"Error cargando SAM: {err}")
    return handler


def main():
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

    uploaded_file = st.file_uploader("Cargar imagen...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagen Original")
            st.image(image, width="stretch")

        if model_choice == "Segment Anything (SAM)":
            sam_type = st.sidebar.selectbox("Tipo SAM", ["vit_b", "vit_l", "vit_h"], index=0)
            ckpt_path = st.sidebar.text_input("Ruta Checkpoint", value=f"sam_{sam_type}.pth")
            points_per_side = st.sidebar.slider("Puntos por lado", 4, 16, 8)

            if st.button("Ejecutar SAM"):
                with st.spinner("Cargando modelo SAM en CPU..."):
                    sam_handler = get_sam_handler(ckpt_path, sam_type)
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
                        st.image(overlay, width="stretch")

        else:
            if st.button(f"Ejecutar {model_choice}"):
                rgb2gray, sobel, threshold_otsu, watershed, felzenszwalb, slic, chan_vese, canny = load_skimage_modules()
                
                with col2:
                    st.subheader(f"Resultado {model_choice}")
                    if model_choice == "Otsu Thresholding":
                        gray_float = rgb2gray(image_np)
                        gray_uint8 = (gray_float * 255).astype(np.uint8)
                        thresh = threshold_otsu(gray_uint8)
                        res = (gray_uint8 > thresh).astype(np.uint8) * 255
                        st.image(res, width="stretch")

                    elif model_choice == "Canny Edge Detector":
                        gray = rgb2gray(image_np)
                        res = canny(gray, sigma=1.0).astype(np.uint8) * 255
                        st.image(res, width="stretch")

                    elif model_choice == "SLIC (Superpixels)":
                        res = slic(image_np, n_segments=100, compactness=10.0, start_label=1)
                        norm_res = ((res - res.min()) / (res.max() - res.min() + 1e-8) * 255).astype(np.uint8)
                        st.image(norm_res, width="stretch")

                    elif model_choice == "Felzenszwalb":
                        res = felzenszwalb(image_np, scale=100, sigma=0.5, min_size=50)
                        norm_res = ((res - res.min()) / (res.max() - res.min() + 1e-8) * 255).astype(np.uint8)
                        st.image(norm_res, width="stretch")

                    elif model_choice == "Watershed":
                        gray = rgb2gray(image_np)
                        elevation_map = sobel(gray)
                        markers = np.zeros_like(gray, dtype=int)
                        markers[0, 0] = 1
                        markers[-1, -1] = 2
                        res = watershed(elevation_map, markers)
                        norm_res = ((res - res.min()) / (res.max() - res.min() + 1e-8) * 255).astype(np.uint8)
                        st.image(norm_res, width="stretch")

                    elif model_choice == "Chan-Vese":
                        gray = rgb2gray(image_np)
                        res = chan_vese(gray, max_num_iter=50).astype(np.uint8) * 255
                        st.image(res, width="stretch")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Error inesperado:")
        st.code(traceback.format_exc())
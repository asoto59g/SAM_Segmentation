"""
app.py
------
Punto de entrada para la app de segmentación en Streamlit Cloud.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import traceback

# 1. Ajuste estricto de variables de entorno para hilos CPU
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

# 2. Asegurar rutas del proyecto
_APP_DIR = Path(__file__).resolve().parent
_CWD = Path.cwd().resolve()

for _path in [_APP_DIR, _CWD]:
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import streamlit as st

# Configuración de página DEBE ser la primera instrucción de Streamlit
st.set_page_config(page_title="Herramienta de Segmentación", layout="wide")

try:
    import torch
    try:
        torch.set_num_threads(2)
    except Exception:
        pass

    from PIL import Image
    import numpy as np

    from core.segmentation import ClassicSegmenter
    from core.sam_handler import SAMHandler
except Exception as import_err:
    st.error("Error al importar librerías:")
    st.code(traceback.format_exc())
    st.stop()


@st.cache_resource
def get_sam_handler(checkpoint_path: str, model_type: str = "vit_b") -> SAMHandler:
    handler = SAMHandler(checkpoint_path=checkpoint_path, model_type=model_type)
    err = handler._load_model()
    if err:
        st.error(f"Error cargando SAM: {err}")
    return handler


def main():
    st.title("Procesamiento y Segmentación de Imágenes")

    st.sidebar.header("Opciones de Segmentación")
    model_choice = st.sidebar.selectbox(
        "Selecciona el algoritmo:",
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
                        st.image(overlay, width="stretch")
        else:
            segmenter = ClassicSegmenter()
            if st.button("Ejecutar Segmentación"):
                with st.spinner("Procesando..."):
                    res = segmenter.run(model_choice, image_np)
                with col2:
                    st.subheader(f"Resultado {model_choice}")
                    st.image(res, width="stretch")


if __name__ == "__main__":
    main()
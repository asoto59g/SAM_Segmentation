"""
app.py
------
App web de segmentación de imágenes agrícolas / satelitales.
Construida con Streamlit + PyTorch + SAM.
Diseñado con protección de importación para evitar fallos de compilación en Streamlit Cloud.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# 1. Ajuste estricto de variables de entorno para hilos CPU antes de cualquier cálculo
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st

# La configuración de página DEBE ser la primera instrucción de Streamlit
st.set_page_config(
    page_title="Segmentación Agrícola",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ajuste de path para módulos locales
_APP_DIR = Path(__file__).parent.resolve()
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# ---------------------------------------------------------------------------
# Verificación e importación segura de librerías esenciales
# ---------------------------------------------------------------------------
try:
    import numpy as np
    from PIL import Image
except Exception as init_err:
    st.error("❌ Fallo crítico al iniciar las librerías base:")
    st.code(traceback.format_exc())
    st.stop()

APP_TITLE = "🌾 Segmentación de Imágenes Agrícolas"
APP_ICON = "🌿"
DEFAULT_SAM_CHECKPOINT = str(Path("/tmp/sam_models") / "sam_vit_b.pth")

# ---------------------------------------------------------------------------
# Mapeos de Métodos
# ---------------------------------------------------------------------------
METHOD_LABELS: dict[str, str] = {
    "1 · Otsu (Umbralización)":       "otsu",
    "2 · Canny (Bordes)":             "canny",
    "3 · Region Growing":             "region_growing",
    "4 · Watershed":                  "watershed",
    "5 · K-Means (Clustering)":       "kmeans",
    "6 · Mean-Shift":                 "meanshift",
    "7 · GrabCut":                    "grabcut",
    "8 · SAM – Automático":           "sam_auto",
    "8 · SAM – Por clic":             "sam_click",
}

SAM_METHODS = {"sam_auto", "sam_click"}

COMPARE_OPTIONS: list[str] = [
    "Original",
    "1 · Otsu",
    "2 · Canny",
    "3 · Region Growing",
    "4 · Watershed",
    "5 · K-Means",
    "6 · Mean-Shift",
    "7 · GrabCut",
]

COMPARE_METHOD_MAP: dict[str, str] = {
    "Original":            "original",
    "1 · Otsu":            "otsu",
    "2 · Canny":           "canny",
    "3 · Region Growing":  "region_growing",
    "4 · Watershed":       "watershed",
    "5 · K-Means":         "kmeans",
    "6 · Mean-Shift":      "meanshift",
    "7 · GrabCut":         "grabcut",
}


def _init_state() -> None:
    defaults = {
        "image_rgb":       None,
        "metadata":        None,
        "filename":        None,
        "result_rgb":      None,
        "result_method":   None,
        "sam_masks":       None,
        "click_points":    [],
        "click_labels":    [],
        "compare_results": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# Carga diferida (Lazy Load) de SAM con Caché
@st.cache_resource(show_spinner=False)
def get_sam_handler(checkpoint_path: str):
    import torch
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    from core.sam_handler import SAMHandler
    return SAMHandler(checkpoint_path=checkpoint_path)


@st.cache_resource(show_spinner=False)
def get_segmenter():
    from core.segmentation import ClassicSegmenter
    return ClassicSegmenter()


def _build_filename(ext: str) -> str:
    fname = st.session_state.get("filename") or "imagen"
    stem  = Path(fname).stem
    mkey  = st.session_state.get("result_method") or "resultado"
    return f"{stem}_{mkey}.{ext}"


def safe_run_grabcut(segmenter, image_rgb: np.ndarray, margin_frac: float = 0.1) -> np.ndarray:
    import cv2
    h, w = image_rgb.shape[:2]
    max_dim = max(h, w)
    
    if max_dim > 800:
        scale = 800.0 / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        small_img = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_small = segmenter.run("grabcut", small_img, margin_frac=margin_frac)
        mask_full = cv2.resize(mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return mask_full
    else:
        return segmenter.run("grabcut", image_rgb, margin_frac=margin_frac)


# ---------------------------------------------------------------------------
# INTERFAZ - SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"## {APP_TITLE}")
    st.markdown("---")

    st.markdown("📂 **Imagen de entrada**")
    uploaded = st.file_uploader(
        label="Sube una imagen",
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        file_bytes = uploaded.read()
        try:
            from utils.image_io import image_from_bytes
            image_rgb, metadata = image_from_bytes(file_bytes, uploaded.name)
            st.session_state["image_rgb"]   = image_rgb
            st.session_state["metadata"]    = metadata
            st.session_state["filename"]    = uploaded.name
            st.session_state["result_rgb"]  = None
            st.session_state["sam_masks"]   = None
            st.session_state["click_points"] = []
            st.session_state["click_labels"] = []
            st.session_state["compare_results"] = {}
        except Exception as e:
            st.error(f"Error al cargar la imagen: {e}")

    if st.session_state["image_rgb"] is not None:
        h, w = st.session_state["image_rgb"].shape[:2]
        meta = st.session_state["metadata"]
        geo_tag = " · GeoTIFF ✓" if meta and meta.get("crs") else ""
        st.caption(f"📐 {w}×{h} px · {st.session_state['filename']}{geo_tag}")

    st.markdown("🔬 **Método de segmentación**")
    method_label = st.selectbox(
        "Método",
        options=list(METHOD_LABELS.keys()),
        index=4,
        label_visibility="collapsed",
    )
    method_key = METHOD_LABELS[method_label]

    st.markdown("⚙️ **Parámetros**")
    params: dict = {}

    if method_key == "otsu":
        params["blur_kernel"] = st.slider("Tamaño suavizado (px)", 1, 21, 5, step=2)

    elif method_key == "canny":
        col_a, col_b = st.columns(2)
        with col_a:
            params["low_threshold"] = st.number_input("Umbral bajo", 10, 500, 100, step=10)
        with col_b:
            params["high_threshold"] = st.number_input("Umbral alto", 10, 500, 200, step=10)

    elif method_key == "region_growing":
        params["tolerance"] = st.slider("Tolerancia de intensidad", 1, 80, 15)

    elif method_key == "kmeans":
        params["k"] = st.slider("Número de clusters (k)", 2, 12, 5)

    elif method_key == "meanshift":
        col_a, col_b = st.columns(2)
        with col_a:
            params["sp"] = st.slider("Radio espacial", 5, 40, 15)
        with col_b:
            params["sr"] = st.slider("Radio de color", 5, 80, 30)

    elif method_key == "grabcut":
        params["margin_frac"] = st.slider("Margen (%)", 5, 30, 10) / 100.0

    elif method_key == "sam_auto":
        params["points_per_side"] = st.slider("Puntos por lado (grilla)", 4, 16, 8)
        params["pred_iou_thresh"] = st.slider("Umbral IoU mínimo", 0.70, 0.98, 0.88, step=0.01)
        params["min_mask_region_area"] = st.slider("Área mínima (px²)", 100, 5000, 500, step=100)

    if method_key in SAM_METHODS:
        st.markdown("🤖 **Estado de SAM**")
        sam_path = st.text_input("Ruta checkpoint .pth", value=DEFAULT_SAM_CHECKPOINT)
        ckpt_exists = Path(sam_path).exists()
        if ckpt_exists:
            st.success("✅ Checkpoint detectado en disco")
        else:
            st.info("💡 Coloca el archivo checkpoint .pth para habilitar SAM.")

    st.markdown("---")
    st.caption("ABC Geomática Agrícola SRL · 2026")


# ---------------------------------------------------------------------------
# INTERFAZ PRINCIPAL
# ---------------------------------------------------------------------------
st.title(APP_TITLE)
st.markdown("Segmentación interactiva de imágenes agrícolas y satelitales.")

if st.session_state["image_rgb"] is None:
    st.info("👋 Por favor, sube una imagen en el panel lateral para empezar.")
    st.stop()

tab_seg, tab_compare = st.tabs(["🔬 Segmentación", "📊 Comparar Métodos"])

# ---------------------------------------------------------------------------
# PESTAÑA 1: SEGMENTACIÓN
# ---------------------------------------------------------------------------
with tab_seg:
    image_rgb: np.ndarray = st.session_state["image_rgb"]
    metadata: dict        = st.session_state["metadata"]

    col_orig, col_result = st.columns(2, gap="medium")

    with col_orig:
        st.subheader("📷 Imagen original")

        if method_key == "sam_click" and st.session_state["click_points"]:
            from utils.visualization import draw_points
            display_img = draw_points(
                image_rgb,
                st.session_state["click_points"],
                st.session_state["click_labels"],
            )
        else:
            display_img = image_rgb

        if method_key == "sam_click":
            try:
                from streamlit_image_coordinates import streamlit_image_coordinates
                click_value = streamlit_image_coordinates(Image.fromarray(display_img), key="coords")
            except Exception:
                st.image(display_img, width="stretch")
                click_value = None
        else:
            st.image(display_img, width="stretch")
            click_value = None

        if method_key == "sam_click":
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button("✅ Positivo", width="stretch"):
                    st.session_state["next_label"] = 1
            with btn_col2:
                if st.button("❌ Fondo", width="stretch"):
                    st.session_state["next_label"] = 0
            with btn_col3:
                if st.button("🗑️ Limpiar", width="stretch"):
                    st.session_state["click_points"] = []
                    st.session_state["click_labels"] = []
                    st.session_state["sam_masks"]    = None
                    st.session_state["result_rgb"]   = None
                    st.rerun()

            if click_value is not None:
                cx, cy = int(click_value["x"]), int(click_value["y"])
                existing = st.session_state["click_points"]
                if not existing or existing[-1] != (cx, cy):
                    st.session_state["click_points"].append((cx, cy))
                    st.session_state["click_labels"].append(st.session_state.get("next_label", 1))
                    st.rerun()

    with col_result:
        st.subheader(f"🎯 Resultado: {method_label}")

        run_disabled = (method_key == "sam_click" and len(st.session_state["click_points"]) == 0)
        run_btn = st.button("▶ Segmentar", type="primary", width="stretch", disabled=run_disabled)

        if run_btn:
            with st.spinner(f"Procesando con {method_label}…"):
                try:
                    from utils.visualization import overlay_masks, colorize_result

                    if method_key in SAM_METHODS:
                        handler = get_sam_handler(DEFAULT_SAM_CHECKPOINT)
                        if method_key == "sam_auto":
                            masks, err = handler.auto_segment(
                                image_rgb,
                                points_per_side=params.get("points_per_side", 8),
                                pred_iou_thresh=params.get("pred_iou_thresh", 0.88),
                                min_mask_region_area=params.get("min_mask_region_area", 500),
                            )
                        else:
                            masks, err = handler.point_segment(
                                image_rgb,
                                points=st.session_state["click_points"],
                                labels=st.session_state["click_labels"],
                            )

                        if err:
                            st.error(err)
                        elif masks is not None:
                            st.session_state["sam_masks"]  = masks
                            st.session_state["result_rgb"] = overlay_masks(image_rgb, masks)
                            st.session_state["result_method"] = method_key
                    else:
                        segmenter = get_segmenter()
                        if method_key == "grabcut":
                            result_raw = safe_run_grabcut(segmenter, image_rgb, margin_frac=params.get("margin_frac", 0.1))
                        else:
                            result_raw = segmenter.run(method_key, image_rgb, **params)

                        st.session_state["result_rgb"]   = colorize_result(result_raw, method_key)
                        st.session_state["result_method"] = method_key
                        st.session_state["sam_masks"]    = None

                except Exception as exc:
                    st.error("Error durante la segmentación:")
                    st.code(traceback.format_exc())

        result_rgb: np.ndarray | None = st.session_state["result_rgb"]

        if result_rgb is not None:
            st.image(result_rgb, width="stretch")

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                from utils.image_io import image_to_bytes
                png_bytes = image_to_bytes(result_rgb, fmt="png")
                st.download_button("⬇️ PNG", png_bytes, _build_filename("png"), "image/png", width="stretch")

            with dl_col2:
                from utils.image_io import geotiff_to_bytes
                has_geo = metadata and metadata.get("crs") is not None
                tiff_bytes = geotiff_to_bytes(result_rgb, metadata or {})
                st.download_button("⬇️ GeoTIFF", tiff_bytes, _build_filename("tif"), "image/tiff", width="stretch")


# ---------------------------------------------------------------------------
# PESTAÑA 2: COMPARAR MÉTODOS
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("📊 Comparación de métodos")
    image_rgb_c: np.ndarray = st.session_state["image_rgb"]

    selected_labels = st.multiselect(
        "Métodos a comparar",
        options=COMPARE_OPTIONS,
        default=["Original", "5 · K-Means", "4 · Watershed", "2 · Canny"],
        max_selections=4,
    )

    run_compare = st.button("▶ Ejecutar comparación", type="primary", disabled=len(selected_labels) < 2)

    if run_compare and len(selected_labels) >= 2:
        with st.spinner("Procesando comparación…"):
            try:
                from utils.visualization import colorize_result, create_comparison_grid
                segmenter_c = get_segmenter()
                compare_store: dict[str, np.ndarray] = {}

                for label in selected_labels:
                    mkey = COMPARE_METHOD_MAP[label]
                    if mkey == "original":
                        compare_store[label] = image_rgb_c.copy()
                    elif mkey == "grabcut":
                        raw = safe_run_grabcut(segmenter_c, image_rgb_c)
                        compare_store[label] = colorize_result(raw, mkey)
                    else:
                        raw = segmenter_c.run(mkey, image_rgb_c)
                        compare_store[label] = colorize_result(raw, mkey)

                ordered_imgs   = [compare_store[lbl] for lbl in selected_labels if lbl in compare_store]
                ordered_titles = [lbl for lbl in selected_labels if lbl in compare_store]

                grid = create_comparison_grid(ordered_imgs, ordered_titles, cols=min(2, len(ordered_imgs)))
                st.image(grid, width="stretch", caption="Grid comparativo")
            except Exception as comp_err:
                st.error("Error durante la comparación:")
                st.code(traceback.format_exc())
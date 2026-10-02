"""
app.py
------
App web de segmentación de imágenes agrícolas / satelitales.
Construida con Streamlit + PyTorch + SAM ViT-L.

Ejecutar:
    streamlit run app.py
"""

from __future__ import annotations

import concurrent.futures
import io
import os
import sys
import threading as _threading
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Configuración de recursos del sistema (EVITA THROTTLING DE CPU EN CLOUD)
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np
import streamlit as st
from PIL import Image

# Configuración del path local
_APP_DIR = Path(__file__).parent.resolve()
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# Importaciones locales
from core.segmentation import ClassicSegmenter
from core.sam_handler import SAMHandler
from utils.image_io import (
    image_from_bytes,
    image_to_bytes,
    geotiff_to_bytes,
)
from utils.visualization import (
    overlay_masks,
    colorize_result,
    draw_points,
    create_comparison_grid,
)

# ---------------------------------------------------------------------------
# Constantes de la interfaz
# ---------------------------------------------------------------------------

APP_TITLE = "🌾 Segmentación de Imágenes Agrícolas"
APP_ICON = "🌿"

# Google Drive file ID del checkpoint SAM ViT-L
_SAM_GDRIVE_FILE_ID = "1Uj9-Shntka1a3k4e49ZFbHHwDV_dC2_d"
_SAM_FILENAME = "sam_vit_l_0b3195.pth"


def _dir_is_writable(path: Path) -> bool:
    """Prueba escritura real creando y borrando un archivo temporal."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        test = path / ".write_test"
        test.write_text("x")
        test.unlink()
        return True
    except Exception:
        return False


_LOCAL_MODELS_DIR = _APP_DIR / "models"
if _dir_is_writable(_LOCAL_MODELS_DIR):
    _SAM_CHECKPOINT_DIR = _LOCAL_MODELS_DIR
else:
    _SAM_CHECKPOINT_DIR = Path("/tmp/sam_models")

DEFAULT_SAM_CHECKPOINT = str(_SAM_CHECKPOINT_DIR / _SAM_FILENAME)


# ---------------------------------------------------------------------------
# Descarga automática del checkpoint SAM desde Google Drive
# ---------------------------------------------------------------------------

def _ensure_sam_checkpoint():
    """Descarga el checkpoint SAM ViT-L si no existe en disco."""
    import traceback

    checkpoint_path = Path(DEFAULT_SAM_CHECKPOINT)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 100_000_000:
        st.session_state["sam_download_error"] = None
        return str(checkpoint_path)

    st.session_state["sam_download_error"] = None

    try:
        import gdown
        url = f"https://drive.google.com/uc?id={_SAM_GDRIVE_FILE_ID}"
        result = gdown.download(url, str(checkpoint_path), quiet=False, fuzzy=True)

        if result is None:
            st.session_state["sam_download_error"] = (
                "gdown retornó None. Asegúrate de que el enlace de Drive sea público."
            )
        elif checkpoint_path.exists() and checkpoint_path.stat().st_size < 100_000_000:
            mb = checkpoint_path.stat().st_size // 1024 // 1024
            st.session_state["sam_download_error"] = (
                f"Descarga incompleta: {mb} MB obtenidos (se esperan ~1200 MB)."
            )

    except Exception:
        st.session_state["sam_download_error"] = (
            f"Error en descarga:\n{traceback.format_exc()}"
        )

    return str(checkpoint_path)


# Mapeo de métodos
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
    "8 · SAM – Automático",
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
    "8 · SAM – Automático":"sam_auto",
}

# ---------------------------------------------------------------------------
# Configuración Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Segmentación Agrícola",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inicialización de Estado y Fondo
# ---------------------------------------------------------------------------

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
        "sam_download_error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# Hilo en background para descarga de SAM
def _background_download():
    _ensure_sam_checkpoint()

if "sam_download_started" not in st.session_state:
    st.session_state["sam_download_started"] = True
    _t = _threading.Thread(target=_background_download, daemon=True)
    _t.start()

_resolved_sam_path = DEFAULT_SAM_CHECKPOINT

# Caches optimizados
@st.cache_resource(show_spinner=False)
def get_segmenter() -> ClassicSegmenter:
    return ClassicSegmenter()

@st.cache_resource(show_spinner=False)
def get_sam_handler(checkpoint_path: str) -> SAMHandler:
    return SAMHandler(checkpoint_path=checkpoint_path)


def _build_filename(ext: str) -> str:
    fname = st.session_state.get("filename") or "imagen"
    stem  = Path(fname).stem
    mkey  = st.session_state.get("result_method") or "resultado"
    return f"{stem}_{mkey}.{ext}"


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
        params["points_per_side"] = st.slider("Puntos por lado (grilla)", 4, 24, 8)
        params["pred_iou_thresh"] = st.slider("Umbral IoU mínimo", 0.70, 0.98, 0.88, step=0.01)
        params["min_mask_region_area"] = st.slider("Área mínima (px²)", 100, 5000, 500, step=100)

    if method_key in SAM_METHODS:
        st.markdown("🤖 **Estado de SAM**")
        sam_path = st.text_input("Ruta checkpoint .pth", value=_resolved_sam_path)
        handler = get_sam_handler(sam_path)
        info = handler.status_info()

        if info["model_loaded"]:
            st.success("✅ Modelo cargado en memoria")
        elif info["checkpoint_found"] and info["sam_installed"]:
            st.warning("⏳ Checkpoint encontrado — se cargará al segmentar")
        elif not info["sam_installed"]:
            st.error("❌ SAM no instalado")
        else:
            st.info("⏬ Descargando checkpoint SAM desde Google Drive...")

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
                    if method_key in SAM_METHODS:
                        handler = get_sam_handler(_resolved_sam_path)
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
                        result_raw = segmenter.run(method_key, image_rgb, **params)
                        st.session_state["result_rgb"]   = colorize_result(result_raw, method_key)
                        st.session_state["result_method"] = method_key
                        st.session_state["sam_masks"]    = None

                except Exception as exc:
                    st.error(f"Error durante la segmentación: {exc}")

        result_rgb: np.ndarray | None = st.session_state["result_rgb"]

        if result_rgb is not None:
            st.image(result_rgb, width="stretch")

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                png_bytes = image_to_bytes(result_rgb, fmt="png")
                st.download_button("⬇️ PNG", png_bytes, _build_filename("png"), "image/png", width="stretch")

            with dl_col2:
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
        segmenter_c = get_segmenter()
        compare_store: dict[str, np.ndarray] = {}

        with st.spinner("Procesando comparación…"):
            for label in selected_labels:
                mkey = COMPARE_METHOD_MAP[label]
                if mkey == "original":
                    compare_store[label] = image_rgb_c.copy()
                elif mkey == "sam_auto":
                    handler_c = get_sam_handler(_resolved_sam_path)
                    masks_c, err_c = handler_c.auto_segment(image_rgb_c, points_per_side=8)
                    if masks_c:
                        compare_store[label] = overlay_masks(image_rgb_c, masks_c)
                    else:
                        compare_store[label] = image_rgb_c.copy()
                else:
                    raw = segmenter_c.run(mkey, image_rgb_c)
                    compare_store[label] = colorize_result(raw, mkey)

        ordered_imgs   = [compare_store[lbl] for lbl in selected_labels if lbl in compare_store]
        ordered_titles = [lbl for lbl in selected_labels if lbl in compare_store]

        grid = create_comparison_grid(ordered_imgs, ordered_titles, cols=min(2, len(ordered_imgs)))
        st.image(grid, width="stretch", caption="Grid comparativo")
"""
app.py
------
App web de segmentación de imágenes agrícolas / satelitales.
Optimizado para ejecución local y Streamlit Cloud.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Forzar la raíz del proyecto en sys.path para Streamlit Cloud
# ---------------------------------------------------------------------------
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import concurrent.futures
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image

# Importaciones de módulos locales (deben ir después de ajustar sys.path)
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
# Constantes de la interfaz y modelos
# ---------------------------------------------------------------------------

APP_TITLE = "🌾 Segmentación de Imágenes Agrícolas"
APP_ICON = "🌿"

# Usar SAM ViT-B (~375 MB) por defecto para evitar agotamiento de RAM en Streamlit Cloud
_SAM_OFFICIAL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
_SAM_FILENAME = "sam_vit_b_01ec64.pth"

def _dir_is_writable(path: Path) -> bool:
    """Prueba si el directorio permite escritura en el sistema de archivos actual."""
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
# Gestión de Checkpoint SAM
# ---------------------------------------------------------------------------

def _ensure_sam_checkpoint(checkpoint_path_str: str) -> bool:
    """
    Verifica o descarga el checkpoint de SAM desde la fuente oficial.
    Retorna True si el archivo existe y es válido.
    """
    checkpoint_path = Path(checkpoint_path_str)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 50_000_000:
        return True

    try:
        with st.spinner("Descargando checkpoint SAM (~375 MB)..."):
            urllib.request.urlretrieve(_SAM_OFFICIAL_URL, str(checkpoint_path))
        return checkpoint_path.exists() and checkpoint_path.stat().st_size > 50_000_000
    except Exception as exc:
        st.error(f"Error al descargar el modelo: {exc}")
        return False

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
# Configuración de Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Segmentación Agrícola",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    .sidebar-section {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #58a6ff;
        margin-top: 1.2rem;
        margin-bottom: 0.3rem;
    }
    .sam-status-ok   { color: #56d364; font-weight: 600; }
    .sam-status-warn { color: #e3b341; font-weight: 600; }
    .sam-status-err  { color: #f85149; font-weight: 600; }
    .img-info { font-size: 0.75rem; color: #8b949e; margin-top: 0.2rem; }
    div[data-testid="stButton"] > button[kind="primary"] { width: 100%; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Inicialización de estado de sesión
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "image_rgb":          None,
        "metadata":           None,
        "filename":           None,
        "result_rgb":         None,
        "result_method":      None,
        "sam_masks":          None,
        "click_points":       [],
        "click_labels":       [],
        "compare_results":    {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ---------------------------------------------------------------------------
# Carga de recursos en caché
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_segmenter() -> ClassicSegmenter:
    return ClassicSegmenter()

@st.cache_resource(show_spinner=False)
def get_sam_handler(checkpoint_path: str) -> SAMHandler:
    return SAMHandler(checkpoint_path=checkpoint_path)

# ---------------------------------------------------------------------------
# PANEL LATERAL
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"## {APP_TITLE}")
    st.markdown("---")

    st.markdown('<p class="sidebar-section">📂 Imagen de entrada</p>', unsafe_allow_html=True)
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
        st.markdown(
            f'<p class="img-info">📐 {w}×{h} px · {st.session_state["filename"]}{geo_tag}</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<p class="sidebar-section">🔬 Método de segmentación</p>', unsafe_allow_html=True)
    method_label = st.selectbox(
        "Método",
        options=list(METHOD_LABELS.keys()),
        index=4,
        label_visibility="collapsed",
    )
    method_key = METHOD_LABELS[method_label]

    st.markdown('<p class="sidebar-section">⚙️ Parámetros</p>', unsafe_allow_html=True)
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
        st.info("🌱 Semilla automática: centro de la imagen.")
    elif method_key == "watershed":
        st.info("Watershed usa marcadores automáticos derivados de la transformada de distancia.")
    elif method_key == "kmeans":
        params["k"] = st.slider("Número de clusters (k)", 2, 12, 5)
    elif method_key == "meanshift":
        col_a, col_b = st.columns(2)
        with col_a:
            params["sp"] = st.slider("Radio espacial", 5, 40, 15)
        with col_b:
            params["sr"] = st.slider("Radio de color", 5, 80, 30)
    elif method_key == "grabcut":
        params["margin_frac"] = st.slider("Margen del recuadro (%)", 5, 30, 10) / 100.0
    elif method_key == "sam_auto":
        params["points_per_side"] = st.slider("Puntos por lado (grilla)", 8, 32, 12)
        params["pred_iou_thresh"] = st.slider("Umbral IoU mínimo", 0.70, 0.98, 0.88, step=0.01)
        params["min_mask_region_area"] = st.slider("Área mínima de región (px²)", 100, 5000, 500, step=100)
    elif method_key == "sam_click":
        st.info("👆 Haz clic sobre la imagen para agregar puntos de interés.")

    if method_key in SAM_METHODS:
        st.markdown('<p class="sidebar-section">🤖 Estado de SAM</p>', unsafe_allow_html=True)
        sam_path = st.text_input(
            "Ruta checkpoint .pth",
            value=DEFAULT_SAM_CHECKPOINT,
            label_visibility="visible",
        )
        
        checkpoint_ready = Path(sam_path).exists() and Path(sam_path).stat().st_size > 50_000_000
        if checkpoint_ready:
            st.markdown('<p class="sam-status-ok">✅ Modelo disponible en disco</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="sam-status-warn">⏳ El modelo se descargará al ejecutar la segmentación</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("ABC Geomática Agrícola SRL · 2026")

# ---------------------------------------------------------------------------
# ÁREA PRINCIPAL
# ---------------------------------------------------------------------------

st.title(APP_TITLE)
st.markdown("Segmentación interactiva de imágenes agrícolas y satelitales.")

if st.session_state["image_rgb"] is None:
    st.markdown("---")
    col_info, col_img = st.columns([3, 2])
    with col_info:
        st.markdown(
            """
            ### ¿Cómo usar la app?
            1. **Sube una imagen** en el panel lateral (JPG, PNG o GeoTIFF)
            2. **Elige el método** de segmentación que quieres probar
            3. **Ajusta los parámetros** según tu imagen
            4. **Descarga** el resultado en PNG o GeoTIFF
            """
        )
    with col_img:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Colorized_image_of_a_corn_field_in_Iowa.jpg/640px-Colorized_image_of_a_corn_field_in_Iowa.jpg",
            caption="Ejemplo: imagen agrícola satelital", width="stretch",
        )
    st.stop()

tab_seg, tab_compare = st.tabs(["🔬 Segmentación", "📊 Comparar Métodos"])

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
                click_value = streamlit_image_coordinates(
                    Image.fromarray(display_img),
                    key="image_coords",
                )
            except ImportError:
                st.image(display_img, width="stretch")
                click_value = None
        else:
            st.image(display_img, width="stretch")
            click_value = None

        if method_key == "sam_click":
            st.markdown("**Agregar punto:**")
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                add_positive = st.button("✅ Punto positivo", width="stretch")
            with btn_col2:
                add_negative = st.button("❌ Marcar fondo", width="stretch")
            with btn_col3:
                clear_points = st.button("🗑️ Limpiar", width="stretch")

            if "next_label" not in st.session_state:
                st.session_state["next_label"] = 1

            if add_positive:
                st.session_state["next_label"] = 1
            if add_negative:
                st.session_state["next_label"] = 0
            if clear_points:
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
                    st.session_state["click_labels"].append(st.session_state["next_label"])
                    st.rerun()

    with col_result:
        st.subheader(f"🎯 Resultado: {method_label}")

        run_label = "▶ Segmentar selección" if method_key == "sam_click" else "▶ Segmentar"
        run_disabled = (method_key == "sam_click" and len(st.session_state["click_points"]) == 0)

        run_btn = st.button(run_label, type="primary", width="stretch", disabled=run_disabled)

        if run_btn:
            with st.spinner(f"Procesando con {method_label}…"):
                try:
                    if method_key in SAM_METHODS:
                        if _ensure_sam_checkpoint(DEFAULT_SAM_CHECKPOINT):
                            handler = get_sam_handler(DEFAULT_SAM_CHECKPOINT)

                            if method_key == "sam_auto":
                                masks, err = handler.auto_segment(
                                    image_rgb,
                                    points_per_side=params.get("points_per_side", 12),
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
                                masks = None

                            if masks is not None:
                                st.session_state["sam_masks"]  = masks
                                result = overlay_masks(image_rgb, masks)
                                st.session_state["result_rgb"] = result
                                st.session_state["result_method"] = method_key
                            else:
                                st.session_state["result_rgb"] = None
                        else:
                            st.error("No se pudo obtener el checkpoint del modelo SAM.")
                    else:
                        segmenter = get_segmenter()
                        result_raw = segmenter.run(method_key, image_rgb, **params)
                        result = colorize_result(result_raw, method_key)
                        st.session_state["result_rgb"]   = result
                        st.session_state["result_method"] = method_key
                        st.session_state["sam_masks"]    = None

                except Exception as exc:
                    st.error(f"Error durante la segmentación: {exc}")

        result_rgb: np.ndarray | None = st.session_state["result_rgb"]

        if result_rgb is not None:
            st.image(result_rgb, width="stretch")
            if st.session_state["sam_masks"] is not None:
                n_masks = len(st.session_state["sam_masks"])
                st.success(f"✅ SAM detectó **{n_masks}** regiones.")

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

        def _run_single(label: str) -> tuple[str, np.ndarray]:
            mkey = COMPARE_METHOD_MAP[label]
            if mkey == "original":
                return label, image_rgb_c.copy()

            if mkey == "sam_auto":
                if _ensure_sam_checkpoint(DEFAULT_SAM_CHECKPOINT):
                    handler_c = get_sam_handler(DEFAULT_SAM_CHECKPOINT)
                    masks_c, err_c = handler_c.auto_segment(image_rgb_c, points_per_side=12)
                    if err_c or masks_c is None:
                        return label, image_rgb_c.copy()
                    return label, overlay_masks(image_rgb_c, masks_c)
                return label, image_rgb_c.copy()

            raw = segmenter_c.run(mkey, image_rgb_c)
            return label, colorize_result(raw, mkey)

        with st.spinner("Ejecutando métodos en paralelo…"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(selected_labels))) as executor:
                futures = {executor.submit(_run_single, lbl): lbl for lbl in selected_labels}
                compare_store: dict[str, np.ndarray] = {}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        lbl, img_r = future.result()
                        compare_store[lbl] = img_r
                    except Exception as exc:
                        st.warning(f"Error en {futures[future]}: {exc}")

        ordered_imgs   = [compare_store[lbl] for lbl in selected_labels if lbl in compare_store]
        ordered_titles = [lbl for lbl in selected_labels if lbl in compare_store]

        st.session_state["compare_results"] = {"images": ordered_imgs, "titles": ordered_titles}

    cmp_data = st.session_state.get("compare_results", {})
    if cmp_data and "images" in cmp_data:
        imgs  = cmp_data["images"]
        titls = cmp_data["titles"]
        grid = create_comparison_grid(imgs, titls, cols=min(2, len(imgs)), cell_width=520, cell_height=520)
        st.image(grid, width="stretch", caption="Grid comparativo")
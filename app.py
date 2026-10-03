"""
app.py
------
App web de segmentación de imágenes agrícolas / satelitales.
Construida con Streamlit + PyTorch + SAM.
Soporta descarga automática de modelos SAM (ViT-B, ViT-L, ViT-H) en entorno local y nube,
así como el ajuste interactivo de todos sus parámetros con protección de memoria RAM.
"""

from __future__ import annotations

import os
import sys
import gc
import urllib.request
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Configuración de recursos del sistema (EVITA THROTTLING EN CLOUD / CPU)
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np
from PIL import Image
import streamlit as st

# Configuración de la página
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

APP_TITLE = "🌾 Segmentación de Imágenes Agrícolas"
APP_ICON = "🌿"

# ---------------------------------------------------------------------------
# Configuración Dinámica de la Ruta de Modelos (Local vs Streamlit Cloud)
# ---------------------------------------------------------------------------
_LOCAL_MODELS_DIR = Path(
    r"C:\Users\AlejandroSotoBarquer\OneDrive - ABC Geomática Agricola SRL\Documentos\ABC_Gis_Activos\01_Clientes\2026\Segmentacion Imagenes\models"
)
_PROJECT_MODELS_DIR = _APP_DIR / "models"
_CLOUD_MODELS_DIR = Path("/tmp/sam_models")

if _LOCAL_MODELS_DIR.exists():
    MODELS_DIR = _LOCAL_MODELS_DIR
elif _PROJECT_MODELS_DIR.exists():
    MODELS_DIR = _PROJECT_MODELS_DIR
else:
    MODELS_DIR = _CLOUD_MODELS_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)

SAM_MODEL_CONFIGS = {
    "ViT-B (Base - Rápido)": {
        "filename": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "type": "vit_b",
    },
    "ViT-L (Large - Balanceado)": {
        "filename": "sam_vit_l_0b3195.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "type": "vit_l",
    },
    "ViT-H (Huge - Alta Precisión)": {
        "filename": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "type": "vit_h",
    },
}

DEFAULT_MODEL_KEY = "ViT-B (Base - Rápido)"

# Límite de dimensión para downscale opt-in
MAX_PROCESSING_DIM = 2048


def get_model_type_from_filename(checkpoint_path: str) -> str:
    name = Path(checkpoint_path).name.lower()
    if "vit_h" in name:
        return "vit_h"
    elif "vit_l" in name:
        return "vit_l"
    else:
        return "vit_b"


def download_checkpoint_with_progress(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    progress_bar = st.progress(0)
    status_text = st.empty()

    def _reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(1.0, downloaded / total_size)
            dl_mb = downloaded / (1024 * 1024)
            tot_mb = total_size / (1024 * 1024)
            progress_bar.progress(percent)
            status_text.text(f"📥 Descargando modelo SAM: {dl_mb:.1f} MB / {tot_mb:.1f} MB ({percent*100:.0f}%)")

    urllib.request.urlretrieve(url, str(dest_path), reporthook=_reporthook)
    progress_bar.empty()
    status_text.empty()


def _ensure_sam_checkpoint_by_key(model_key: str) -> str:
    config = SAM_MODEL_CONFIGS[model_key]
    target_path = MODELS_DIR / config["filename"]

    if target_path.exists() and target_path.stat().st_size > 50_000_000:
        return str(target_path)

    try:
        st.info(f"⚡ Inicializando descarga automática de {model_key} en Streamlit Cloud...")
        download_checkpoint_with_progress(config["url"], target_path)
        if target_path.exists() and target_path.stat().st_size > 50_000_000:
            st.success("✅ Descarga del modelo SAM completada con éxito.")
            return str(target_path)
        else:
            raise RuntimeError("El archivo descargado no es válido o está incompleto.")
    except Exception as exc:
        st.error(f"❌ Error al descargar el modelo {model_key}: {exc}")
        st.code(traceback.format_exc())
        st.stop()


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
        "sam_download_error": None,
        "loaded_sam_path":    None,
        # Cache de bytes de descarga
        "cached_png_bytes":   None,
        "cached_tiff_bytes":  None,
        "cached_result_id":   None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


@st.cache_resource(show_spinner=False, max_entries=1, ttl=3600)
def load_sam_handler_instance(checkpoint_path: str, model_type: str):
    import torch
    try:
        torch.set_num_threads(2)
    except Exception:
        pass
    from core.sam_handler import SAMHandler
    
    try:
        return SAMHandler(checkpoint_path=checkpoint_path, model_type=model_type)
    except TypeError:
        return SAMHandler(checkpoint_path=checkpoint_path)


def get_sam_handler(checkpoint_path: str):
    model_type = get_model_type_from_filename(checkpoint_path)
    
    if st.session_state.get("loaded_sam_path") != checkpoint_path:
        gc.collect()
        load_sam_handler_instance.clear()
        st.session_state["loaded_sam_path"] = checkpoint_path
        
    return load_sam_handler_instance(checkpoint_path, model_type)


@st.cache_resource(show_spinner=False, max_entries=1, ttl=3600)
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


def downscale_if_needed(
    image_rgb: np.ndarray,
    max_dim: int = MAX_PROCESSING_DIM,
) -> tuple[np.ndarray, float | None]:
    """
    Redimensiona la imagen si supera max_dim, manteniendo proporción.
    Retorna (imagen_procesada, scale_factor).
    Si no hay downscale, scale_factor = None.
    """
    h, w = image_rgb.shape[:2]
    current_max = max(h, w)
    if current_max <= max_dim:
        return image_rgb, None
    scale = max_dim / current_max
    new_w, new_h = int(w * scale), int(h * scale)
    import cv2
    small_img = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small_img, scale


def upscale_mask(mask: np.ndarray, orig_shape: tuple[int, int], scale: float) -> np.ndarray:
    """Escala una máscara booleana/uint8 al tamaño original con INTER_NEAREST."""
    import cv2
    h, w = orig_shape[:2]
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


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
        index=7,
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
        st.caption("Ajustes de sensibilidad de detección:")
        params["points_per_side"] = st.slider(
            "Puntos por lado (Grilla)", 8, 64, 16, step=8,
            help="16 o 24 es ideal para CPU. 32 o más exige alta GPU."
        )
        params["pred_iou_thresh"] = st.slider(
            "Umbral IoU mínimo", 0.50, 0.98, 0.86, step=0.01,
            help="Disminuirlo conserva más máscaras detectadas."
        )
        params["stability_score_thresh"] = st.slider(
            "Umbral de Estabilidad", 0.60, 0.99, 0.92, step=0.01,
            help="Disminuirlo detecta objetos con bordes difusos."
        )
        params["min_mask_region_area"] = st.slider(
            "Área mínima (px²)", 0, 5000, 100, step=50,
            help="Filtra elementos más pequeños que este área."
        )

    if method_key in SAM_METHODS:
        st.markdown("🤖 **Modelo SAM**")
        
        selected_model_key = st.selectbox(
            "Arquitectura del Modelo",
            options=list(SAM_MODEL_CONFIGS.keys()),
            index=0,
        )

        cfg = SAM_MODEL_CONFIGS[selected_model_key]
        expected_path = MODELS_DIR / cfg["filename"]

        if expected_path.exists() and expected_path.stat().st_size > 50_000_000:
            mb = expected_path.stat().st_size // (1024 * 1024)
            st.success(f"✅ Modelo listo ({mb} MB)")
        else:
            st.info("☁️ Se descargará automáticamente al ejecutar.")

        # Advertencia ViT-H en cloud
        if selected_model_key == "ViT-H (Huge - Alta Precisión)" and MODELS_DIR == _CLOUD_MODELS_DIR:
            st.warning("⚠️ ViT-H (~2.5 GB) puede exceder el límite de RAM en Streamlit Cloud. ")
            st.caption("   Recomendado: ViT-B o ViT-L en cloud.")

    # Downscale opt-in
    st.markdown("⚡ **Rendimiento**")
    limit_size = st.checkbox(
        "Limitar tamaño de procesamiento a 2048 px",
        value=False,
        help="Reduce la imagen antes de procesar (upscale final con INTER_NEAREST). ")
        
    st.session_state["limit_processing_size"] = limit_size

    if st.session_state["image_rgb"] is not None:
        h, w = st.session_state["image_rgb"].shape[:2]
        if max(h, w) > MAX_PROCESSING_DIM:
            st.warning(f"⚠️ Imagen grande ({w}×{h} px). Activar 'Limitar tamaño' acelera y ahorra RAM.")

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

                    # Downscale opt-in
                    limit_size = st.session_state.get("limit_processing_size", False)
                    proc_image = image_rgb
                    scale = None
                    if limit_size:
                        proc_image, scale = downscale_if_needed(image_rgb, MAX_PROCESSING_DIM)
                        if scale is not None:
                            st.info(f"⚡ Procesando versión reducida ({proc_image.shape[1]}×{proc_image.shape[0]} px)")

                    if method_key in SAM_METHODS:
                        # Forzar la recolección de basura de memoria RAM
                        gc.collect()

                        valid_ckpt_path = _ensure_sam_checkpoint_by_key(selected_model_key)
                        handler = get_sam_handler(valid_ckpt_path)
                        
                        if method_key == "sam_auto":
                            masks, err = handler.auto_segment(
                                proc_image,
                                points_per_side=params.get("points_per_side", 16),
                                pred_iou_thresh=params.get("pred_iou_thresh", 0.86),
                                stability_score_thresh=params.get("stability_score_thresh", 0.92),
                                min_mask_region_area=params.get("min_mask_region_area", 100),
                            )
                        else:
                            masks, err = handler.point_segment(
                                proc_image,
                                points=st.session_state["click_points"],
                                labels=st.session_state["click_labels"],
                            )

                        if err:
                            st.error(err)
                        elif masks is not None:
                            # Upscale máscaras si hubo downscale
                            if scale is not None:
                                for m in masks:
                                    if "segmentation" in m:
                                        m["segmentation"] = upscale_mask(m["segmentation"], image_rgb.shape, scale)
                            st.session_state["sam_masks"]  = masks
                            st.session_state["result_rgb"] = overlay_masks(image_rgb, masks)
                            st.session_state["result_method"] = method_key
                    else:
                        segmenter = get_segmenter()
                        if method_key == "grabcut":
                            # GrabCut ya tiene su propio downscale interno
                            result_raw = safe_run_grabcut(segmenter, image_rgb, margin_frac=params.get("margin_frac", 0.1))
                        elif limit_size and scale is not None:
                            # Procesar en versión reducida y upscale resultado
                            result_raw = segmenter.run(method_key, proc_image, **params)
                            import cv2
                            h, w = image_rgb.shape[:2]
                            result_raw = cv2.resize(result_raw, (w, h), interpolation=cv2.INTER_NEAREST)
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

            # Cache de bytes de descarga: solo recalcular si cambió el resultado
            current_result_id = id(result_rgb)
            if st.session_state["cached_result_id"] != current_result_id:
                from utils.image_io import image_to_bytes, geotiff_to_bytes
                st.session_state["cached_png_bytes"] = image_to_bytes(result_rgb, fmt="png")
                st.session_state["cached_tiff_bytes"] = geotiff_to_bytes(result_rgb, metadata or {})
                st.session_state["cached_result_id"] = current_result_id

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button("⬇️ PNG", st.session_state["cached_png_bytes"], _build_filename("png"), "image/png", width="stretch")

            with dl_col2:
                st.download_button("⬇️ GeoTIFF", st.session_state["cached_tiff_bytes"], _build_filename("tif"), "image/tiff", width="stretch")


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
                from utils.visualization import colorize_result, create_comparison_grid, overlay_masks
                segmenter_c = get_segmenter()
                compare_store: dict[str, np.ndarray] = {}

                for label in selected_labels:
                    mkey = COMPARE_METHOD_MAP[label]
                    if mkey == "original":
                        compare_store[label] = image_rgb_c.copy()
                    elif mkey == "grabcut":
                        raw = safe_run_grabcut(segmenter_c, image_rgb_c)
                        compare_store[label] = colorize_result(raw, mkey)
                    elif mkey == "sam_auto":
                        valid_ckpt_path = _ensure_sam_checkpoint_by_key(DEFAULT_MODEL_KEY)
                        handler_c = get_sam_handler(valid_ckpt_path)
                        masks_c, err_c = handler_c.auto_segment(image_rgb_c, points_per_side=16)
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
            except Exception as comp_err:
                st.error("Error durante la comparación:")
                st.code(traceback.format_exc())
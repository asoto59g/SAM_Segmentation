"""
app.py
------
Aplicación web de segmentación de imágenes agrícolas / satelitales.
Integra Lazy Loading, protección anti-OOM y descarga oficial para Streamlit Cloud.
"""

from __future__ import annotations

import os
import sys
import gc
import urllib.request
import traceback
from pathlib import Path

# 1. Configuración de recursos para evitar Throttling en Streamlit Cloud
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from PIL import Image
import streamlit as st

# La configuración de página DEBE ser la primera instrucción
st.set_page_config(
    page_title="Segmentación Agrícola",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Ajuste absoluto de path para módulos locales en la Nube
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

APP_TITLE = "🌾 Segmentación de Imágenes Agrícolas"

# 3. Configuración Dinámica de Rutas (Local vs Streamlit Cloud)
_LOCAL_MODELS_DIR = ROOT_DIR / "models"
_CLOUD_MODELS_DIR = Path("/tmp/sam_models")

MODELS_DIR = _LOCAL_MODELS_DIR if os.access(ROOT_DIR, os.W_OK) else _CLOUD_MODELS_DIR
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Enlaces de descarga directos (Meta / Facebook AI Research)
SAM_MODEL_CONFIGS = {
    "ViT-B (Base - Rápido - 375 MB)": {
        "filename": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "type": "vit_b",
    },
    "ViT-L (Large - Balanceado - 1.2 GB)": {
        "filename": "sam_vit_l_0b3195.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "type": "vit_l",
    },
    "ViT-H (Huge - Alta Precisión - 2.5 GB)": {
        "filename": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "type": "vit_h",
    },
}

DEFAULT_MODEL_KEY = "ViT-B (Base - Rápido - 375 MB)"

def download_checkpoint_with_progress(url: str, dest_path: Path) -> None:
    """Descarga un archivo con una barra de progreso nativa en Streamlit."""
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
            status_text.text(f"📥 Descargando modelo: {dl_mb:.1f} MB / {tot_mb:.1f} MB ({percent*100:.0f}%)")

    urllib.request.urlretrieve(url, str(dest_path), reporthook=_reporthook)
    progress_bar.empty()
    status_text.empty()

def _ensure_sam_checkpoint_by_key(model_key: str) -> str:
    """Garantiza el checkpoint SAM, descargándolo si no existe localmente."""
    config = SAM_MODEL_CONFIGS[model_key]
    target_path = MODELS_DIR / config["filename"]

    if target_path.exists() and target_path.stat().st_size > 50_000_000:
        return str(target_path)

    try:
        st.info(f"⚡ Inicializando descarga de {model_key} desde servidores oficiales...")
        download_checkpoint_with_progress(config["url"], target_path)
        if target_path.exists() and target_path.stat().st_size > 50_000_000:
            st.success("✅ Descarga completada con éxito.")
            return str(target_path)
        else:
            raise RuntimeError("Archivo descargado inválido o incompleto.")
    except Exception as exc:
        st.error(f"❌ Error al descargar el modelo: {exc}")
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
        "loaded_sam_path":    None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

@st.cache_resource(show_spinner=False)
def load_sam_handler_instance(checkpoint_path: str, model_type: str):
    import torch
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    from core.sam_handler import SAMHandler
    return SAMHandler(checkpoint_path=checkpoint_path, model_type=model_type)

def get_sam_handler(checkpoint_path: str, model_type: str):
    if st.session_state.get("loaded_sam_path") != checkpoint_path:
        load_sam_handler_instance.clear()
        st.session_state["loaded_sam_path"] = checkpoint_path
    return load_sam_handler_instance(checkpoint_path, model_type)

@st.cache_resource(show_spinner=False)
def get_segmenter():
    from core.segmentation import ClassicSegmenter
    return ClassicSegmenter()

def safe_run_grabcut(segmenter, image_rgb: np.ndarray, margin_frac: float = 0.1) -> np.ndarray:
    """Escalado preventivo para GrabCut evitando colapso OOM."""
    import cv2
    h, w = image_rgb.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        small_img = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_small = segmenter.run("grabcut", small_img, margin_frac=margin_frac)
        return cv2.resize(mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return segmenter.run("grabcut", image_rgb, margin_frac=margin_frac)

# ---------------------------------------------------------------------------
# INTERFAZ - SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"## {APP_TITLE}")
    st.markdown("---")
    st.markdown("📂 **Imagen de entrada**")
    
    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"], label_visibility="collapsed")
    
    if uploaded is not None:
        file_bytes = uploaded.read()
        try:
            from utils.image_io import image_from_bytes
            image_rgb, metadata = image_from_bytes(file_bytes, uploaded.name)
            
            # Reescalar si la imagen es gigantesca para proteger memoria (OOM)
            max_input_dim = 1024
            orig_h, orig_w = image_rgb.shape[:2]
            if max(orig_h, orig_w) > max_input_dim:
                st.warning(f"Imagen escalada a {max_input_dim}px para proteger la memoria de Streamlit.")
                scale_in = max_input_dim / float(max(orig_h, orig_w))
                image_rgb = np.array(Image.fromarray(image_rgb).resize((int(orig_w * scale_in), int(orig_h * scale_in)), Image.Resampling.LANCZOS))
                
            st.session_state["image_rgb"] = image_rgb
            st.session_state["metadata"] = metadata
            st.session_state["filename"] = uploaded.name
            st.session_state["result_rgb"] = None
            st.session_state["sam_masks"] = None
            st.session_state["click_points"] = []
            st.session_state["click_labels"] = []
        except Exception as e:
            st.error(f"Error cargando imagen: {e}")

    st.markdown("🔬 **Método de segmentación**")
    method_label = st.selectbox("Método", options=list(METHOD_LABELS.keys()), index=4, label_visibility="collapsed")
    method_key = METHOD_LABELS[method_label]

    st.markdown("⚙️ **Parámetros**")
    params: dict = {}
    if method_key == "otsu":
        params["blur_kernel"] = st.slider("Tamaño suavizado (px)", 1, 21, 5, step=2)
    elif method_key == "sam_auto":
        params["points_per_side"] = st.slider("Puntos por lado", 4, 16, 8)
        params["pred_iou_thresh"] = st.slider("Umbral IoU", 0.70, 0.98, 0.88)
    
    if method_key in SAM_METHODS:
        st.markdown("🤖 **Arquitectura SAM**")
        selected_model_key = st.selectbox("Modelo", options=list(SAM_MODEL_CONFIGS.keys()), index=0)

# ---------------------------------------------------------------------------
# INTERFAZ PRINCIPAL
# ---------------------------------------------------------------------------
st.title(APP_TITLE)
if st.session_state["image_rgb"] is None:
    st.info("👋 Por favor, sube una imagen en el panel lateral para empezar.")
    st.stop()

tab_seg, _ = st.tabs(["🔬 Segmentación", "📊 Comparar Métodos"])

with tab_seg:
    image_rgb: np.ndarray = st.session_state["image_rgb"]
    col_orig, col_result = st.columns(2, gap="medium")

    with col_orig:
        st.subheader("📷 Imagen original")
        display_img = image_rgb

        if method_key == "sam_click":
            try:
                from streamlit_image_coordinates import streamlit_image_coordinates
                if st.session_state["click_points"]:
                    from utils.visualization import draw_points
                    display_img = draw_points(image_rgb, st.session_state["click_points"], st.session_state["click_labels"])
                
                click_value = streamlit_image_coordinates(Image.fromarray(display_img), key="coords")
                
                c1, c2, c3 = st.columns(3)
                with c1: 
                    if st.button("✅ Positivo", use_container_width=True): st.session_state["next_label"] = 1
                with c2: 
                    if st.button("❌ Fondo", use_container_width=True): st.session_state["next_label"] = 0
                with c3:
                    if st.button("🗑️ Limpiar", use_container_width=True):
                        st.session_state["click_points"] = []
                        st.session_state["click_labels"] = []
                        st.rerun()

                if click_value:
                    cx, cy = int(click_value["x"]), int(click_value["y"])
                    if not st.session_state["click_points"] or st.session_state["click_points"][-1] != (cx, cy):
                        st.session_state["click_points"].append((cx, cy))
                        st.session_state["click_labels"].append(st.session_state.get("next_label", 1))
                        st.rerun()
            except Exception:
                st.image(display_img, use_container_width=True)
        else:
            st.image(display_img, use_container_width=True)

    with col_result:
        st.subheader(f"🎯 Resultado: {method_label}")
        run_disabled = (method_key == "sam_click" and len(st.session_state["click_points"]) == 0)
        
        if st.button("▶ Segmentar", type="primary", use_container_width=True, disabled=run_disabled):
            with st.spinner(f"Procesando con {method_label}…"):
                try:
                    from utils.visualization import overlay_masks, colorize_result

                    if method_key in SAM_METHODS:
                        # Descarga diferida: Solo descarga y carga en memoria si el usuario presiona "Segmentar"
                        valid_ckpt_path = _ensure_sam_checkpoint_by_key(selected_model_key)
                        cfg = SAM_MODEL_CONFIGS[selected_model_key]
                        handler = get_sam_handler(valid_ckpt_path, cfg["type"])
                        
                        if method_key == "sam_auto":
                            masks, err = handler.auto_segment(image_rgb, points_per_side=params.get("points_per_side", 8))
                        else:
                            masks, err = handler.point_segment(image_rgb, st.session_state["click_points"], st.session_state["click_labels"])

                        if not err and masks:
                            st.session_state["result_rgb"] = overlay_masks(image_rgb, masks)
                        else:
                            st.error(err)
                            
                        # Limpiar instancia explícitamente para proteger la RAM
                        del handler
                        gc.collect()

                    else:
                        segmenter = get_segmenter()
                        if method_key == "grabcut":
                            result_raw = safe_run_grabcut(segmenter, image_rgb)
                        else:
                            result_raw = segmenter.run(method_key, image_rgb, **params)
                        st.session_state["result_rgb"] = colorize_result(result_raw, method_key)

                except Exception as exc:
                    st.error("Error durante la segmentación:")
                    st.code(traceback.format_exc())

        if st.session_state["result_rgb"] is not None:
            st.image(st.session_state["result_rgb"], use_container_width=True)
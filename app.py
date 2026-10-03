"""
app.py
------
Aplicación web de segmentación de imágenes agrícolas / satelitales.
Integra Lazy Loading, protección anti-OOM, controles para todos los modelos, escalado y descargas.
"""

from __future__ import annotations

import os
import sys
import gc
import urllib.request
import traceback
import io
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
        "last_processed_click": None,
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

def safe_run_grabcut(segmenter, image_rgb: np.ndarray, **kwargs) -> np.ndarray:
    """Escalado preventivo para GrabCut evitando colapso OOM."""
    import cv2
    h, w = image_rgb.shape[:2]
    max_dim = 800
    margin_frac = kwargs.pop("margin_frac", 0.1)
    
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        small_img = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_small = segmenter.run("grabcut", small_img, margin_frac=margin_frac, **kwargs)
        return cv2.resize(mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return segmenter.run("grabcut", image_rgb, margin_frac=margin_frac, **kwargs)

# ---------------------------------------------------------------------------
# INTERFAZ - SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"## {APP_TITLE}")
    st.markdown("---")
    
    st.markdown("📏 **Resolución máxima de entrada**")
    scale_option = st.selectbox(
        "Escalar imagen a:",
        options=["1024 px (Recomendado Cloud)", "2048 px", "Original (Sin escalar)", "Personalizado"],
        index=0,
        help="Las imágenes satelitales/agrícolas suelen ser enormes. Reducir su tamaño previene errores de memoria."
    )
    
    max_input_dim = None
    if scale_option == "1024 px (Recomendado Cloud)":
        max_input_dim = 1024
    elif scale_option == "2048 px":
        max_input_dim = 2048
    elif scale_option == "Personalizado":
        max_input_dim = st.number_input("Dimensión máxima (px)", min_value=256, max_value=10000, value=1500, step=100)
    
    st.markdown("📂 **Imagen de entrada**")
    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"], label_visibility="collapsed")
    
    if uploaded is not None:
        file_bytes = uploaded.read()
        try:
            from utils.image_io import image_from_bytes
            image_rgb, metadata = image_from_bytes(file_bytes, uploaded.name)
            
            orig_h, orig_w = image_rgb.shape[:2]
            
            if max_input_dim is not None and max(orig_h, orig_w) > max_input_dim:
                if scale_option != "1024 px (Recomendado Cloud)":
                    st.warning(f"Imagen escalada a {max_input_dim}px. ⚠️ Tamaños muy grandes pueden causar estrangulamiento de CPU (Throttling).")
                
                scale_in = max_input_dim / float(max(orig_h, orig_w))
                image_rgb = np.array(Image.fromarray(image_rgb).resize((int(orig_w * scale_in), int(orig_h * scale_in)), Image.Resampling.LANCZOS))
            elif max_input_dim is None:
                st.warning("⚠️ Mantener el tamaño original en imágenes muy grandes probablemente provocará un colapso en la nube al ejecutar SAM.")
                
            st.session_state["image_rgb"] = image_rgb
            st.session_state["metadata"] = metadata
            st.session_state["filename"] = uploaded.name
            st.session_state["result_rgb"] = None
            st.session_state["sam_masks"] = None
            st.session_state["click_points"] = []
            st.session_state["click_labels"] = []
            st.session_state["last_processed_click"] = None
        except Exception as e:
            st.error(f"Error cargando imagen: {e}")

    st.markdown("🔬 **Método de segmentación**")
    method_label = st.selectbox("Método", options=list(METHOD_LABELS.keys()), index=7, label_visibility="collapsed")
    method_key = METHOD_LABELS[method_label]

    st.markdown("⚙ **Parámetros de Configuración**")
    params: dict = {}
    
    if method_key == "otsu":
        params["blur_kernel"] = st.slider("Tamaño suavizado (px)", 1, 31, 5, step=2)
    elif method_key == "canny":
        params["sigma"] = st.slider("Sigma (suavizado)", 0.5, 5.0, 1.0, step=0.1)
    elif method_key == "kmeans":
        params["n_clusters"] = st.slider("Número de clústeres (K)", 2, 20, 5)
    elif method_key == "meanshift":
        params["spatial_radius"] = st.slider("Radio espacial", 1, 50, 15)
        params["color_radius"] = st.slider("Radio de color", 1, 50, 15)
    elif method_key == "watershed":
        params["compactness"] = st.slider("Compacidad", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
    elif method_key == "region_growing":
        params["tolerance"] = st.slider("Tolerancia", 0.01, 0.5, 0.1, step=0.01)
    elif method_key == "grabcut":
        params["margin_frac"] = st.slider("Margen inicial (%)", 1, 40, 10) / 100.0
        params["iterCount"] = st.slider("Iteraciones", 1, 10, 5)
    elif method_key == "sam_auto":
        params["points_per_side"] = st.slider("Puntos por lado", 4, 32, 8, help="Más puntos = más detalle, pero requiere muchísimo más procesamiento de CPU.")
        params["pred_iou_thresh"] = st.slider("Umbral IoU", 0.50, 0.98, 0.88, help="Filtra máscaras de baja confianza.")
        params["min_mask_region_area"] = st.number_input("Área mínima de región (px)", min_value=0, value=500, step=100)
    elif method_key == "sam_click":
        st.info("💡 Haz clic en la imagen original para agregar puntos. Usa los botones debajo de la imagen para cambiar entre Positivo / Fondo o Limpiar.")
    
    if method_key in SAM_METHODS:
        st.markdown("🤖 **Arquitectura SAM**")
        selected_model_key = st.selectbox(
            "Modelo", 
            options=list(SAM_MODEL_CONFIGS.keys()), 
            index=0,
            help="⚠️ En Streamlit Cloud Free usa ViT-B. Modelos más grandes causarán Out of Memory (OOM)."
        )

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
                    if st.button("✅ Positivo", use_container_width=True): 
                        st.session_state["next_label"] = 1
                with c2: 
                    if st.button("❌ Fondo", use_container_width=True): 
                        st.session_state["next_label"] = 0
                with c3:
                    if st.button("🗑️ Limpiar", use_container_width=True):
                        st.session_state["click_points"] = []
                        st.session_state["click_labels"] = []
                        st.session_state["last_processed_click"] = None
                        st.rerun()

                if click_value:
                    cx, cy = int(click_value["x"]), int(click_value["y"])
                    click_sig = (cx, cy)
                    if click_sig != st.session_state.get("last_processed_click"):
                        st.session_state["last_processed_click"] = click_sig
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
            with st.spinner(f"Procesando con {method_label}… (Esto puede tardar en Streamlit Cloud)"):
                try:
                    from utils.visualization import overlay_masks, colorize_result

                    if method_key in SAM_METHODS:
                        valid_ckpt_path = _ensure_sam_checkpoint_by_key(selected_model_key)
                        cfg = SAM_MODEL_CONFIGS[selected_model_key]
                        handler = get_sam_handler(valid_ckpt_path, cfg["type"])
                        
                        if method_key == "sam_auto":
                            masks, err = handler.auto_segment(
                                image_rgb, 
                                points_per_side=params.get("points_per_side", 8),
                                pred_iou_thresh=params.get("pred_iou_thresh", 0.88),
                                min_mask_region_area=params.get("min_mask_region_area", 500)
                            )
                        else:
                            masks, err = handler.point_segment(image_rgb, st.session_state["click_points"], st.session_state["click_labels"])

                        if not err and masks:
                            st.session_state["result_rgb"] = overlay_masks(image_rgb, masks)
                        else:
                            st.error(err)
                            
                        del handler
                        gc.collect()

                    else:
                        segmenter = get_segmenter()
                        if method_key == "grabcut":
                            result_raw = safe_run_grabcut(segmenter, image_rgb, **params)
                        else:
                            result_raw = segmenter.run(method_key, image_rgb, **params)
                        st.session_state["result_rgb"] = colorize_result(result_raw, method_key)

                except Exception as exc:
                    st.error("Error durante la segmentación:")
                    st.code(traceback.format_exc())

        if st.session_state["result_rgb"] is not None:
            st.image(st.session_state["result_rgb"], use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 💾 Descargar Resultados")
            col_d1, col_d2 = st.columns(2)
            
            buf_png = io.BytesIO()
            Image.fromarray(st.session_state["result_rgb"]).save(buf_png, format="PNG")
            
            with col_d1:
                st.download_button(
                    label="⬇️ Descargar PNG",
                    data=buf_png.getvalue(),
                    file_name=f"seg_{st.session_state['filename'].split('.')[0]}.png",
                    mime="image/png",
                    use_container_width=True
                )
                
            with col_d2:
                try:
                    import rasterio
                    from rasterio.io import MemoryFile
                    
                    meta = st.session_state.get("metadata", {})
                    transform = meta.get("transform") if isinstance(meta, dict) else None
                    crs = meta.get("crs") if isinstance(meta, dict) else None
                    
                    img_res = st.session_state["result_rgb"]
                    h, w = img_res.shape[:2]
                    channels = img_res.shape[2] if len(img_res.shape) == 3 else 1
                    
                    with MemoryFile() as memfile:
                        kwargs = {
                            'driver': 'GTiff',
                            'height': h,
                            'width': w,
                            'count': channels,
                            'dtype': img_res.dtype,
                        }
                        if transform: kwargs['transform'] = transform
                        if crs: kwargs['crs'] = crs
                        
                        with memfile.open(**kwargs) as dataset:
                            if channels == 1:
                                dataset.write(img_res, 1)
                            else:
                                for i in range(channels):
                                    dataset.write(img_res[:, :, i], i + 1)
                                    
                        tif_bytes = memfile.read()
                        
                    st.download_button(
                        label="⬇️ Descargar GeoTIFF",
                        data=tif_bytes,
                        file_name=f"seg_{st.session_state['filename'].split('.')[0]}.tif",
                        mime="image/tiff",
                        use_container_width=True
                    )
                except Exception as e:
                    st.warning(f"No se pudo habilitar GeoTIFF: {e}")
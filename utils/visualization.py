"""
utils/visualization.py
-----------------------
Funciones de composición visual para la app de segmentación.

Funciones públicas
------------------
- overlay_masks        : Superpone máscaras SAM con colores semitransparentes.
- colorize_result      : Aplica colormap adecuado según el método clásico.
- draw_points          : Dibuja marcadores de clic sobre la imagen.
- create_comparison_grid: Genera un grid PNG comparativo de varios métodos.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paleta de colores predefinida (visualmente distinguible para agri)
# ---------------------------------------------------------------------------

# Colores diseñados para ser visibles sobre imágenes de vegetación
_AGRI_PALETTE = np.array([
    [230,  25,  75],   # Rojo vivo
    [ 60, 180,  75],   # Verde
    [255, 225,  25],   # Amarillo
    [  0, 130, 200],   # Azul
    [245, 130,  48],   # Naranja
    [145,  30, 180],   # Púrpura
    [ 70, 240, 240],   # Cian
    [240,  50, 230],   # Magenta
    [210, 245,  60],   # Lima
    [250, 190, 212],   # Rosa pastel
    [  0, 128, 128],   # Verde azulado
    [220, 190, 255],   # Lavanda
    [170, 110,  40],   # Marrón
    [255, 250, 200],   # Crema
    [128,   0,   0],   # Granate
    [ 10, 210, 160],   # Menta
], dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. Overlay de máscaras SAM
# ---------------------------------------------------------------------------

def overlay_masks(
    image_rgb: np.ndarray,
    masks: list[dict],
    alpha: float = 0.45,
    max_masks: int = 200,
) -> np.ndarray:
    """
    Superpone las máscaras SAM sobre la imagen original con colores
    semitransparentes.

    Optimización: usa el 'bbox' [x, y, w, h] que provee SAM para limitar
    el blending al rectángulo de la máscara, en lugar de toda la imagen.

    Parameters
    ----------
    image_rgb : np.ndarray
        Imagen base uint8 (H, W, 3) RGB.
    masks : list[dict]
        Lista de dicts SAM con claves:
        - ``'segmentation'``: bool array H×W
        - ``'bbox'``: [x, y, w, h] (opcional, pero presente en SAM)
    alpha : float
        Opacidad del color de relleno (0=transparente, 1=sólido).
    max_masks : int
        Máximo número de máscaras a mostrar (las primeras por área).

    Returns
    -------
    np.ndarray
        Imagen compuesta uint8 (H, W, 3) RGB.
    """
    if not masks:
        return image_rgb.copy()

    result = image_rgb.astype(np.float32)
    n_palette = len(_AGRI_PALETTE)

    visible_masks = masks[:max_masks]
    rng = np.random.default_rng(42)

    for idx, mask_data in enumerate(visible_masks):
        seg = mask_data.get("segmentation")
        if seg is None:
            continue

        seg_bool = np.asarray(seg, dtype=bool)
        if not seg_bool.any():
            continue

        # Usar bbox si está disponible para limitar la operación
        bbox = mask_data.get("bbox")
        if bbox is not None:
            x, y, w, h = map(int, bbox)
            # Recortar a los límites de la imagen
            x = max(0, x)
            y = max(0, y)
            w = min(w, result.shape[1] - x)
            h = min(h, result.shape[0] - y)
            if w <= 0 or h <= 0:
                continue
            roi_bool = seg_bool[y:y+h, x:x+w]
            if not roi_bool.any():
                continue
            roi = result[y:y+h, x:x+w]
        else:
            # Fallback: imagen completa (caso legacy)
            roi_bool = seg_bool
            roi = result

        if idx < n_palette:
            color = _AGRI_PALETTE[idx].astype(np.float32)
        else:
            color = rng.integers(64, 230, size=3, dtype=np.uint8).astype(np.float32)

        # Blending vectorizado solo en la ROI
        roi[roi_bool] = (1.0 - alpha) * roi[roi_bool] + alpha * color

    return result.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Colorizar resultado clásico
# ---------------------------------------------------------------------------

def colorize_result(
    result: np.ndarray,
    method_name: str,
) -> np.ndarray:
    """
    Normaliza y aplica un colormap adecuado según el método.

    - Otsu / Canny / Region Growing  →  escala de grises (ya son binarios)
    - Watershed / K-Means / Mean-Shift / GrabCut  →  imagen RGB directa
    Si el resultado ya es RGB, se devuelve sin cambios.

    Parameters
    ----------
    result : np.ndarray
        Array uint8 (H, W) o (H, W, 3).
    method_name : str
        Nombre del método ('otsu', 'canny', etc.).

    Returns
    -------
    np.ndarray
        Array uint8 (H, W, 3) RGB listo para visualizar.
    """
    method_name = method_name.lower().strip()

    # Si es 2D, convertir a RGB
    if result.ndim == 2:
        return cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Ya es RGB
    if result.ndim == 3 and result.shape[2] == 3:
        return result.astype(np.uint8)

    # Caso raro: >3 canales
    return result[:, :, :3].astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. Dibujar puntos de clic SAM
# ---------------------------------------------------------------------------

def draw_points(
    image_rgb: np.ndarray,
    points: list[tuple[int, int]],
    labels: list[int],
    radius: int = 8,
    thickness: int = 2,
) -> np.ndarray:
    """
    Dibuja los puntos de clic sobre la imagen.

    - Puntos positivos (label=1)  →  círculo verde sólido con borde blanco
    - Puntos negativos (label=0)  →  círculo rojo sólido con borde blanco

    Parameters
    ----------
    image_rgb : np.ndarray
        Imagen uint8 (H, W, 3) RGB.
    points : list[tuple[int, int]]
        Lista de coordenadas (x, y).
    labels : list[int]
        1 = positivo, 0 = negativo.
    radius : int
        Radio del círculo en píxeles.
    thickness : int
        Grosor del borde blanco exterior.

    Returns
    -------
    np.ndarray
        Copia de la imagen con los puntos dibujados.
    """
    if not points:
        return image_rgb.copy()

    canvas = image_rgb.copy()

    for (x, y), label in zip(points, labels):
        color_fill = (0, 200, 0) if label == 1 else (200, 0, 0)
        center = (int(x), int(y))

        # Borde blanco exterior
        cv2.circle(canvas, center, radius + thickness, (255, 255, 255), -1)
        # Relleno de color
        cv2.circle(canvas, center, radius, color_fill, -1)
        # Borde oscuro interior para contraste
        cv2.circle(canvas, center, radius, (0, 0, 0), 1)

    return canvas


# ---------------------------------------------------------------------------
# 4. Grid comparativo multi-método
# ---------------------------------------------------------------------------

def create_comparison_grid(
    images: list[np.ndarray],
    titles: list[str],
    cols: int = 2,
    cell_width: int = 512,
    cell_height: int = 512,
    bg_color: tuple[int, int, int] = (20, 20, 20),
    title_height: int = 36,
    font_scale: float = 0.65,
) -> np.ndarray:
    """
    Genera un grid comparativo de múltiples resultados de segmentación.

    Parameters
    ----------
    images : list[np.ndarray]
        Lista de imágenes uint8 (H, W, 3) RGB.
    titles : list[str]
        Títulos correspondientes a cada imagen.
    cols : int
        Número de columnas en el grid (2 o 3 recomendado).
    cell_width : int
        Ancho de cada celda en píxeles.
    cell_height : int
        Alto de cada celda en píxeles (sin título).
    bg_color : tuple
        Color de fondo del grid (R, G, B).
    title_height : int
        Alto del área de título en píxeles.
    font_scale : float
        Escala de la fuente del título.

    Returns
    -------
    np.ndarray
        Array uint8 (H_total, W_total, 3) RGB del grid compuesto.
    """
    n = len(images)
    if n == 0:
        # Grid vacío
        placeholder = np.full((cell_height, cell_width, 3), bg_color, dtype=np.uint8)
        return placeholder

    rows = math.ceil(n / cols)
    grid_w = cols * cell_width
    grid_h = rows * (cell_height + title_height)

    # Fondo oscuro
    grid = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)

    for idx, (img, title) in enumerate(zip(images, titles)):
        row = idx // cols
        col = idx % cols

        x_offset = col * cell_width
        y_offset = row * (cell_height + title_height)

        # Redimensionar imagen al tamaño de la celda
        resized = _resize_fit(img, cell_width, cell_height)

        # Centrar dentro de la celda si es más pequeña
        y_img = y_offset + title_height
        x_img = x_offset

        h_img, w_img = resized.shape[:2]
        # Añadir relleno horizontal si es necesario
        pad_x = (cell_width - w_img) // 2
        pad_y = (cell_height - h_img) // 2

        grid[
            y_img + pad_y : y_img + pad_y + h_img,
            x_img + pad_x : x_img + pad_x + w_img,
        ] = resized

        # Dibujar título
        _draw_title(
            grid,
            title,
            x_offset,
            y_offset,
            cell_width,
            title_height,
            font_scale,
        )

        # Borde de celda
        cv2.rectangle(
            grid,
            (x_offset, y_offset),
            (x_offset + cell_width - 1, y_offset + title_height + cell_height - 1),
            (80, 80, 80),
            1,
        )

    return grid


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _resize_fit(
    image: np.ndarray,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """
    Redimensiona la imagen para que quepa en target_w × target_h
    manteniendo la proporción (letterbox).
    """
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def _draw_title(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    width: int,
    height: int,
    font_scale: float,
) -> None:
    """Escribe el título centrado en el área reservada."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    # Truncar si no cabe
    while text_w > width - 10 and len(text) > 4:
        text = text[:-1]
        (text_w, _), _ = cv2.getTextSize(text + "…", font, font_scale, thickness)
        if text_w <= width - 10:
            text += "…"
            break

    text_x = x + (width - text_w) // 2
    text_y = y + (height + text_h) // 2 - baseline

    # Sombra para legibilidad
    cv2.putText(
        canvas, text,
        (text_x + 1, text_y + 1),
        font, font_scale, (0, 0, 0), thickness + 1,
        cv2.LINE_AA,
    )
    # Texto en blanco
    cv2.putText(
        canvas, text,
        (text_x, text_y),
        font, font_scale, (240, 240, 240), thickness,
        cv2.LINE_AA,
    )

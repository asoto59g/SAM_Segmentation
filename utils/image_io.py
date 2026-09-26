"""
utils/image_io.py
-----------------
Carga y guardado de imágenes agrícolas / satelitales.

Soporta:
  - JPG, PNG, BMP  →  cargados con OpenCV / Pillow
  - TIF / TIFF (GeoTIFF)  →  cargados con rasterio

Retorna siempre arrays RGB uint8 de 3 canales.
Conserva la georreferenciación cuando el archivo de entrada es GeoTIFF.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# rasterio se importa de forma condicional para que la app funcione
# aunque el usuario no lo tenga instalado (degradación elegante).
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.enums import ColorInterp
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Tipos internos
# ---------------------------------------------------------------------------

METADATA_EMPTY: dict = {
    "crs": None,
    "transform": None,
    "source_format": None,
    "original_shape": None,
}


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> tuple[np.ndarray, dict]:
    """
    Carga una imagen desde disco.

    Parameters
    ----------
    path : str | Path
        Ruta al archivo (JPG, PNG, TIF, TIFF, BMP).

    Returns
    -------
    image_rgb : np.ndarray
        Array NumPy uint8 con forma (H, W, 3) en orden RGB.
    metadata : dict
        Información de georreferenciación:
        - ``crs``        : objeto CRS de rasterio (o None)
        - ``transform``  : objeto Affine de rasterio (o None)
        - ``source_format`` : 'geotiff' | 'standard'
        - ``original_shape`` : (H, W)

    Raises
    ------
    FileNotFoundError
        Si la ruta no existe.
    ValueError
        Si el archivo no puede interpretarse como imagen.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {path}")

    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"} and _RASTERIO_AVAILABLE:
        return _load_geotiff(path)
    else:
        return _load_standard(path)


def image_from_bytes(
    file_bytes: bytes,
    filename: str,
) -> tuple[np.ndarray, dict]:
    """
    Carga una imagen desde bytes (integración con st.file_uploader).

    Parameters
    ----------
    file_bytes : bytes
        Contenido binario del archivo subido por Streamlit.
    filename : str
        Nombre original del archivo (usado para detectar GeoTIFF).

    Returns
    -------
    image_rgb : np.ndarray
        Array NumPy uint8 (H, W, 3) en orden RGB.
    metadata : dict
        Igual que en :func:`load_image`.
    """
    suffix = Path(filename).suffix.lower()

    if suffix in {".tif", ".tiff"} and _RASTERIO_AVAILABLE:
        # rasterio puede leer desde un objeto de tipo fichero (MemoryFile)
        with rasterio.MemoryFile(file_bytes) as memfile:
            with memfile.open() as dataset:
                return _parse_rasterio_dataset(dataset)
    else:
        # OpenCV / Pillow para el resto
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            # Fallback con Pillow
            img_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            image_rgb = np.array(img_pil, dtype=np.uint8)
        else:
            image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        meta = dict(METADATA_EMPTY)
        meta["source_format"] = "standard"
        meta["original_shape"] = image_rgb.shape[:2]
        return image_rgb, meta


def save_result(
    image_rgb: np.ndarray,
    metadata: dict,
    output_path: str | Path,
    fmt: str = "png",
) -> Path:
    """
    Guarda el resultado de segmentación en disco.

    Parameters
    ----------
    image_rgb : np.ndarray
        Array uint8 (H, W, 3) en orden RGB.
    metadata : dict
        Metadata de la imagen original (retornada por :func:`load_image`).
    output_path : str | Path
        Ruta destino (sin extensión, la función la agrega).
    fmt : str
        ``'png'`` o ``'geotiff'`` (o ``'tiff'``).

    Returns
    -------
    Path
        Ruta efectiva del archivo guardado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower()

    if fmt in {"geotiff", "tiff"} and _RASTERIO_AVAILABLE and metadata.get("crs"):
        out = output_path.with_suffix(".tif")
        _save_geotiff(image_rgb, metadata, out)
    else:
        out = output_path.with_suffix(".png")
        _save_png(image_rgb, out)

    return out


def image_to_bytes(image_rgb: np.ndarray, fmt: str = "png") -> bytes:
    """
    Convierte un array RGB a bytes PNG o JPEG (para st.download_button).

    Parameters
    ----------
    image_rgb : np.ndarray
        Array uint8 (H, W, 3).
    fmt : str
        ``'png'`` o ``'jpeg'``.

    Returns
    -------
    bytes
        Contenido binario listo para descarga.
    """
    fmt_lower = fmt.lower()
    if fmt_lower == "png":
        pil_fmt = "PNG"
        suffix = ".png"
    else:
        pil_fmt = "JPEG"
        suffix = ".jpg"

    img_pil = Image.fromarray(image_rgb.astype(np.uint8))
    buffer = io.BytesIO()
    img_pil.save(buffer, format=pil_fmt)
    return buffer.getvalue()


def geotiff_to_bytes(image_rgb: np.ndarray, metadata: dict) -> bytes:
    """
    Convierte resultado a GeoTIFF en memoria (para st.download_button).

    Si no hay georreferenciación disponible, genera un GeoTIFF simple
    sin CRS/transform.

    Parameters
    ----------
    image_rgb : np.ndarray
        Array uint8 (H, W, 3).
    metadata : dict
        Metadata de georreferenciación.

    Returns
    -------
    bytes
        Contenido GeoTIFF listo para descarga.
    """
    if not _RASTERIO_AVAILABLE:
        # Degradación elegante: devuelve PNG si rasterio no está disponible
        return image_to_bytes(image_rgb, fmt="png")

    h, w = image_rgb.shape[:2]
    transform = metadata.get("transform")
    crs = metadata.get("crs")

    # Si no hay transform, crear uno ficticio para que sea un GeoTIFF válido
    if transform is None:
        transform = from_bounds(0, 0, w, h, w, h)

    buffer = io.BytesIO()
    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=h,
            width=w,
            count=3,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
        ) as dst:
            # rasterio usa orden (banda, filas, columnas)
            for i in range(3):
                dst.write(image_rgb[:, :, i], i + 1)

        buffer.write(memfile.read())

    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _load_standard(path: Path) -> tuple[np.ndarray, dict]:
    """Carga JPG/PNG/BMP con OpenCV."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        # Intentar con Pillow (maneja más formatos edge-case)
        try:
            img_pil = Image.open(path).convert("RGB")
            image_rgb = np.array(img_pil, dtype=np.uint8)
        except Exception as exc:
            raise ValueError(
                f"No se pudo cargar la imagen: {path}\n{exc}"
            ) from exc
    else:
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    meta = dict(METADATA_EMPTY)
    meta["source_format"] = "standard"
    meta["original_shape"] = image_rgb.shape[:2]
    return image_rgb, meta


def _load_geotiff(path: Path) -> tuple[np.ndarray, dict]:
    """Carga GeoTIFF con rasterio."""
    with rasterio.open(path) as dataset:
        return _parse_rasterio_dataset(dataset)


def _parse_rasterio_dataset(
    dataset: "rasterio.DatasetReader",
) -> tuple[np.ndarray, dict]:
    """
    Extrae array RGB y metadata de un dataset rasterio abierto.

    Maneja imágenes multibanda (>3 bandas) tomando las 3 primeras
    o buscando las bandas R, G, B por interpretación de color.
    """
    count = dataset.count
    ci = dataset.colorinterp

    # Determinar qué bandas usar como R, G, B
    band_r = band_g = band_b = None

    for idx, interp in enumerate(ci, start=1):
        if interp == ColorInterp.red:
            band_r = idx
        elif interp == ColorInterp.green:
            band_g = idx
        elif interp == ColorInterp.blue:
            band_b = idx

    # Si no hay interpretación de color, usar las 3 primeras bandas
    if None in (band_r, band_g, band_b):
        if count >= 3:
            band_r, band_g, band_b = 1, 2, 3
        elif count == 1:
            # Imagen en escala de grises → duplicar a 3 canales
            gray = dataset.read(1).astype(np.uint8)
            gray_normalized = _normalize_to_uint8(gray)
            image_rgb = np.stack([gray_normalized] * 3, axis=-1)
            meta = _build_meta(dataset, "geotiff")
            return image_rgb, meta
        else:
            # 2 bandas: usar las dos primeras y rellenar azul con 0
            r = _normalize_to_uint8(dataset.read(1))
            g = _normalize_to_uint8(dataset.read(2))
            b = np.zeros_like(r)
            image_rgb = np.stack([r, g, b], axis=-1)
            meta = _build_meta(dataset, "geotiff")
            return image_rgb, meta

    r = _normalize_to_uint8(dataset.read(band_r))
    g = _normalize_to_uint8(dataset.read(band_g))
    b = _normalize_to_uint8(dataset.read(band_b))
    image_rgb = np.stack([r, g, b], axis=-1)

    meta = _build_meta(dataset, "geotiff")
    return image_rgb, meta


def _build_meta(dataset: "rasterio.DatasetReader", fmt: str) -> dict:
    return {
        "crs": dataset.crs,
        "transform": dataset.transform,
        "source_format": fmt,
        "original_shape": (dataset.height, dataset.width),
    }


def _normalize_to_uint8(band: np.ndarray) -> np.ndarray:
    """
    Normaliza una banda a rango [0, 255] uint8.

    Maneja float32, uint16, int16 y uint8.
    """
    band = band.astype(np.float32)
    mn, mx = band.min(), band.max()
    if mx == mn:
        return np.zeros_like(band, dtype=np.uint8)
    normalized = (band - mn) / (mx - mn) * 255.0
    return normalized.clip(0, 255).astype(np.uint8)


def _save_png(image_rgb: np.ndarray, path: Path) -> None:
    bgr = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _save_geotiff(
    image_rgb: np.ndarray,
    metadata: dict,
    path: Path,
) -> None:
    h, w = image_rgb.shape[:2]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=3,
        dtype=np.uint8,
        crs=metadata.get("crs"),
        transform=metadata.get("transform"),
    ) as dst:
        for i in range(3):
            dst.write(image_rgb[:, :, i], i + 1)

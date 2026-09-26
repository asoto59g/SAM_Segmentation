"""
core/checkpoint.py
------------------
Descarga y cache del checkpoint de SAM desde los servidores oficiales de Meta.

El estado se guarda en variables de módulo protegidas por un lock, de forma que
la descarga puede ejecutarse en un hilo de background sin tocar
``st.session_state`` (que no existe fuera del hilo del script de Streamlit).
"""

from __future__ import annotations

import os
import shutil
import threading
import urllib.request
from pathlib import Path

_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything"

# model_type -> (nombre de archivo, tamaño aproximado en bytes)
CHECKPOINTS: dict[str, tuple[str, int]] = {
    "vit_b": ("sam_vit_b_01ec64.pth", 375_000_000),
    "vit_l": ("sam_vit_l_0b3195.pth", 1_250_000_000),
    "vit_h": ("sam_vit_h_4b8939.pth", 2_560_000_000),
}

DEFAULT_MODEL_TYPE = "vit_b"

_lock = threading.Lock()
_state: dict[str, object] = {"status": "idle", "error": None}
_thread: threading.Thread | None = None


def resolve_model_type(preferred: str | None = None) -> str:
    """Modelo a usar: argumento > variable de entorno SAM_MODEL_TYPE > vit_b."""
    candidate = preferred or os.environ.get("SAM_MODEL_TYPE") or DEFAULT_MODEL_TYPE
    candidate = candidate.strip().lower()
    return candidate if candidate in CHECKPOINTS else DEFAULT_MODEL_TYPE


def model_type_from_path(path: str | Path) -> str:
    """Deduce el model_type a partir del nombre del archivo del checkpoint."""
    name = Path(path).name
    for model_type, (filename, _) in CHECKPOINTS.items():
        if name == filename or model_type in name:
            return model_type
    return DEFAULT_MODEL_TYPE


def expected_size(model_type: str) -> int:
    return CHECKPOINTS[resolve_model_type(model_type)][1]


def checkpoints_dir() -> Path:
    """Directorio con permiso real de escritura (en Cloud /mount/src es read-only)."""
    local = Path(__file__).resolve().parent.parent / "models"
    try:
        local.mkdir(parents=True, exist_ok=True)
        probe = local / ".write_test"
        probe.write_text("x")
        probe.unlink()
        return local
    except Exception:
        fallback = Path("/tmp/sam_models")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def checkpoint_path(model_type: str) -> Path:
    filename, _ = CHECKPOINTS[resolve_model_type(model_type)]
    return checkpoints_dir() / filename


def is_complete(model_type: str) -> bool:
    path = checkpoint_path(model_type)
    if not path.exists():
        return False
    return path.stat().st_size >= expected_size(model_type) * 0.95


def downloaded_bytes(model_type: str) -> int:
    path = checkpoint_path(model_type)
    partial = path.with_suffix(path.suffix + ".part")
    for candidate in (path, partial):
        if candidate.exists():
            return candidate.stat().st_size
    return 0


def status(model_type: str) -> dict:
    """Estado actual de la descarga, seguro de llamar desde el script Streamlit."""
    with _lock:
        snapshot = dict(_state)
    if is_complete(model_type):
        snapshot["status"] = "ready"
        snapshot["error"] = None
    snapshot["downloaded_bytes"] = downloaded_bytes(model_type)
    snapshot["expected_bytes"] = expected_size(model_type)
    return snapshot


def _download(model_type: str) -> None:
    path = checkpoint_path(model_type)
    partial = path.with_suffix(path.suffix + ".part")
    filename, _ = CHECKPOINTS[model_type]
    url = f"{_BASE_URL}/{filename}"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "sam-segmentation-app"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as fh:
            shutil.copyfileobj(response, fh, length=1024 * 1024)
        partial.replace(path)
        if not is_complete(model_type):
            size_mb = path.stat().st_size // 1024 // 1024
            raise RuntimeError(
                f"Descarga incompleta: {size_mb} MB "
                f"(se esperan ~{expected_size(model_type) // 1024 // 1024} MB)"
            )
        with _lock:
            _state.update(status="ready", error=None)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        with _lock:
            _state.update(status="error", error=f"{type(exc).__name__}: {exc}\nURL: {url}")


def start_download(model_type: str, force: bool = False) -> None:
    """Lanza la descarga en background una sola vez por proceso."""
    global _thread
    model_type = resolve_model_type(model_type)

    if force:
        checkpoint_path(model_type).unlink(missing_ok=True)

    if is_complete(model_type):
        with _lock:
            _state.update(status="ready", error=None)
        return

    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        if _state["status"] == "error" and not force:
            return
        _state.update(status="downloading", error=None)
        _thread = threading.Thread(target=_download, args=(model_type,), daemon=True)
        _thread.start()

"""
core/ - Módulos de segmentación
"""
from .segmentation import ClassicSegmenter
from .sam_handler import SAMHandler

__all__ = ["ClassicSegmenter", "SAMHandler"]

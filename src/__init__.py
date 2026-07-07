# src/__init__.py
"""GrammaLang Hybrid v0.5.0 — Онтологический анализатор с темпоральной кардиограммой"""

from .fast_parser import fast_parse
from .deep_interpreter import (
    deep_interpret,
    deep_interpret_heidegger,
    deep_interpret_dostoevsky,
    HeideggerAnalysis,
    HeideggerResult,
    DostoevskyResult
)
from .fusion import apply_fusion_layer, build_temporal_cardiogram, generate_textual_analysis
from .cardiogram import plot_cardiogram, save_cardiogram_png

__version__ = "0.5.0"
__all__ = [
    "fast_parse",
    "deep_interpret",
    "deep_interpret_heidegger",
    "deep_interpret_dostoevsky",
    "HeideggerAnalysis",
    "HeideggerResult",
    "DostoevskyResult",
    "apply_fusion_layer",
    "build_temporal_cardiogram",
    "generate_textual_analysis",
    "plot_cardiogram",
    "save_cardiogram_png"
]

"""
cardiogram.py
Визуализация темпоральной кардиограммы через matplotlib.
"""

import matplotlib
matplotlib.use('Agg')  # Без GUI-окна, только сохранение в файл
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, Any
from pathlib import Path


# Стили
BG_COLOR = '#1a1a2e'
GRID_COLOR = '#2d2d44'
LINE_COLOR = '#e94560'
FILL_COLOR = '#e94560'
SAFE_ZONE_COLOR = '#0f3460'
WARNING_ZONE_COLOR = '#533483'
DANGER_ZONE_COLOR = '#1a1a2e'
TEXT_COLOR = '#eee'
TREND_COLOR = '#16c79a'


def plot_cardiogram(cardiogram_data: Dict[str, Any], title: str = "Темпоральная кардиограмма") -> plt.Figure:
    """
    Строит график темпоральной кардиограммы.
    
    Args:
        cardiogram_data: результат build_temporal_cardiogram()
        title: заголовок графика
    
    Returns:
        matplotlib Figure
    """
    indices = cardiogram_data.get("timeline", [])
    if not indices:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "Нет данных", ha='center', va='center', fontsize=16, color=TEXT_COLOR)
        return fig
    
    x = list(range(1, len(indices) + 1))
    
    # Создаём фигуру
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Зоны риска
    ax.axhspan(-1.0, -0.5, alpha=0.3, color=DANGER_ZONE_COLOR, label='Зона слома (< -0.5)')
    ax.axhspan(-0.5, -0.3, alpha=0.2, color=WARNING_ZONE_COLOR, label='Критическая зона (-0.5 — -0.3)')
    ax.axhspan(0.3, 1.0, alpha=0.15, color=SAFE_ZONE_COLOR, label='Зона устойчивости (> 0.3)')
    
    # Основная линия
    ax.plot(x, indices, color=LINE_COLOR, linewidth=2.5, marker='o', markersize=6,
            markerfacecolor=LINE_COLOR, markeredgecolor='white', markeredgewidth=1, zorder=5)
    
    # Заливка под графиком
    ax.fill_between(x, indices, alpha=0.15, color=FILL_COLOR)
    
    # Горизонтальные линии
    ax.axhline(y=0, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=cardiogram_data.get("mean", 0), color=TREND_COLOR, linestyle=':', linewidth=1.2,
               label=f'Среднее: {cardiogram_data.get("mean", 0):+.3f}')
    
    # Критические точки
    for pt in cardiogram_data.get("stability_break_points", []):
        if pt < len(indices):
            ax.plot(pt + 1, indices[pt], 'x', color='red', markersize=14, markeredgewidth=3, zorder=10)
    
    # Трендовая линия
    if len(indices) >= 2:
        slope = cardiogram_data.get("slope", 0)
        intercept = cardiogram_data.get("mean", 0) - slope * (len(indices) - 1) / 2
        trend_y = [slope * i + intercept for i in range(len(indices))]
        ax.plot(x, trend_y, color=TREND_COLOR, linestyle='-', linewidth=1.5, alpha=0.7,
                label=f'Тренд: {cardiogram_data.get("trend", "stable")}')
    
    # Оформление
    ax.set_xlabel("Предложение", fontsize=13, color=TEXT_COLOR)
    ax.set_ylabel("Индекс воли", fontsize=13, color=TEXT_COLOR)
    ax.set_title(title, fontsize=16, color=TEXT_COLOR, fontweight='bold', pad=15)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0.5, len(indices) + 0.5)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    ax.legend(loc='upper right', facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
    
    # Аннотации для пиков и долин
    if indices:
        peak_idx = indices.index(max(indices))
        valley_idx = indices.index(min(indices))
        ax.annotate(f'Пик: {indices[peak_idx]:+.3f}', xy=(peak_idx + 1, indices[peak_idx]),
                    xytext=(peak_idx + 1, indices[peak_idx] + 0.15), fontsize=9, color=TREND_COLOR,
                    ha='center', arrowprops=dict(arrowstyle='->', color=TREND_COLOR, lw=1))
        ax.annotate(f'Спад: {indices[valley_idx]:+.3f}', xy=(valley_idx + 1, indices[valley_idx]),
                    xytext=(valley_idx + 1, indices[valley_idx] - 0.15), fontsize=9, color=LINE_COLOR,
                    ha='center', arrowprops=dict(arrowstyle='->', color=LINE_COLOR, lw=1))
    
    plt.tight_layout()
    return fig


def save_cardiogram_png(cardiogram_data: Dict[str, Any], output_path: str = "cardiogram.png", title: str = "Темпоральная кардиограмма") -> str:
    """
    Сохраняет кардиограмму в PNG-файл.
    
    Args:
        cardiogram_data: результат build_temporal_cardiogram()
        output_path: путь для сохранения
        title: заголовок
    
    Returns:
        путь к сохранённому файлу
    """
    fig = plot_cardiogram(cardiogram_data, title)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return str(path.resolve())

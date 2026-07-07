"""
fusion.py
Fusion Layer: слияние синтаксического и семантического слоёв.
Генерирует JSON + текстовый разбор + темпоральную кардиограмму.
v0.5.0 — добавлена build_temporal_cardiogram()
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("fusion")

# Попытка импорта HeideggerAnalysis (может быть в deep_interpreter)
try:
    from src.deep_interpreter import HeideggerAnalysis
except ImportError:
    HeideggerAnalysis = None


def apply_fusion_layer(
    syntax_data: Dict[str, Any],
    heidegger: Optional[Any] = None,
    dostoevsky: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Применяет слой слияния к синтаксическим и семантическим данным.
    
    Args:
        syntax_data: результат fast_parse()
        heidegger: результат deep_interpret_heidegger() (опционально)
        dostoevsky: результат deep_interpret_dostoevsky() (опционально)
    
    Returns:
        dict с final_index, health_status, analysis_text и метаданными
    """
    
    # Извлечение синтаксических данных
    lexical_index = syntax_data.get("lexical_index", 0.0)
    syntax_type = syntax_data.get("syntax_type", "assertion")
    depth = syntax_data.get("depth", 1)
    modal_modifier = syntax_data.get("modal_modifier", 0.0)
    node_types = syntax_data.get("node_types", [])
    lexical_markers = syntax_data.get("lexical_markers", [])

    # Модификаторы синтаксического типа
    SYNTAX_MODIFIERS = {
        "imperative": 0.40,
        "definition": 0.30,
        "assertion": 0.00,
        "rhetorical_question": -0.30,
        "counterfactual": -0.35,
        "optative": -0.30
    }
    
    syntax_modifier = SYNTAX_MODIFIERS.get(syntax_type, 0.0)
    depth_penalty = -0.05 * max(0, depth - 1)
    base_index = lexical_index + syntax_modifier + modal_modifier + depth_penalty

    # Семантическая корректировка
    final_index_adjustment = 0.0
    hold_break_risk = False
    hold_break_type = None
    dasein_mode = "unknown"
    interpretation = {}
    
    if heidegger is not None:
        if hasattr(heidegger, 'final_index_adjustment'):
            final_index_adjustment = heidegger.final_index_adjustment
        if hasattr(heidegger, 'hold_break_risk'):
            hold_break_risk = heidegger.hold_break_risk
        if hasattr(heidegger, 'hold_break_type'):
            hold_break_type = heidegger.hold_break_type
        if hasattr(heidegger, 'dasein_mode'):
            dasein_mode = heidegger.dasein_mode
        if hasattr(heidegger, 'model_dump'):
            interpretation = heidegger.model_dump()
        elif hasattr(heidegger, '__dict__'):
            interpretation = heidegger.__dict__
    
    final_index = base_index + final_index_adjustment

    if hold_break_risk:
        final_index -= 0.25

    final_index = round(max(-1.0, min(1.0, final_index)), 3)
    
    # Статус здоровья
    health_status = "critical" if hold_break_risk else "stable"
    if final_index < -0.5:
        health_status = "critical"
    elif final_index > 0.8:
        health_status = "high_stability"

    # Текстовый разбор
    analysis_text = generate_textual_analysis(
        syntax_type=syntax_type,
        node_types=node_types,
        lexical_markers=lexical_markers,
        depth=depth,
        modal_modifier=modal_modifier,
        syntax_modifier=syntax_modifier,
        base_index=base_index,
        final_index=final_index,
        health_status=health_status,
        heidegger=heidegger,
        dostoevsky=dostoevsky
    )

    result = {
        "final_index": final_index,
        "health_status": health_status,
        "dasein_mode": dasein_mode,
        "hold_break_detected": hold_break_risk,
        "hold_break_type": hold_break_type,
        "interpretation": interpretation,
        "syntax": {
            "type": syntax_type,
            "depth": depth,
            "lexical_index": lexical_index,
            "syntax_modifier": syntax_modifier,
            "modal_modifier": modal_modifier,
            "depth_penalty": depth_penalty,
            "base_index": round(base_index, 3)
        },
        "analysis_text": analysis_text
    }
    
    logger.info(f"Fusion result: final_index={final_index}, health={health_status}")
    return result


def generate_textual_analysis(
    syntax_type: str = "assertion",
    node_types: Optional[List[str]] = None,
    lexical_markers: Optional[List[str]] = None,
    depth: int = 1,
    modal_modifier: float = 0.0,
    syntax_modifier: float = 0.0,
    base_index: float = 0.0,
    final_index: float = 0.0,
    health_status: str = "stable",
    heidegger: Optional[Any] = None,
    dostoevsky: Optional[Any] = None
) -> str:
    """Генерирует читаемый текстовый разбор."""
    
    if node_types is None:
        node_types = []
    if lexical_markers is None:
        lexical_markers = []
    
    syntax_names = {
        "imperative": "Императив (повеление, призыв к действию)",
        "definition": "Определение (установление сущности)",
        "assertion": "Утверждение (констатация факта)",
        "rhetorical_question": "Риторический вопрос (скрытое утверждение)",
        "counterfactual": "Контрфактуал (предположение о несостоявшемся)",
        "optative": "Оптатив (пожелание, надежда)"
    }
    syntax_name = syntax_names.get(syntax_type, syntax_type)

    dasein_names = {
        "ahead-of-itself": "впереди-себя (проективность, устремлённость в будущее)",
        "being-in-the-world": "бытие-в-мире (погружённость в контекст)",
        "fallenness": "падение (захваченность миром, отчуждение от себя)",
        "questioning": "вопрошание (поиск, подвешенность)",
        "unknown": "не определён"
    }
    
    lines = ["=" * 60, "ОНТОЛОГИЧЕСКИЙ РАЗБОР ТЕКСТА", "=" * 60]
    lines.append("\n1. СИНТАКСИЧЕСКИЙ АНАЛИЗ")
    lines.append(f"   Тип высказывания: {syntax_name}")
    lines.append(f"   Глубина конструкции: {depth} (от 1 до 5)")
    
    if node_types:
        lines.append(f"   Онтологические узлы: {', '.join(node_types)}")
    if lexical_markers:
        lines.append(f"   Ключевые маркеры: {', '.join(lexical_markers)}")
    
    modal_desc = "нейтральная"
    if modal_modifier > 0.15:
        modal_desc = "усиливающая (повышает вес высказывания)"
    elif modal_modifier < -0.15:
        modal_desc = "ослабляющая (снижает вес высказывания)"
    lines.append(f"   Модальность: {modal_desc} ({modal_modifier:+.2f})")
    
    lines.append("\n2. СЕМАНТИЧЕСКИЙ АНАЛИЗ")
    
    if heidegger is not None:
        dasein_mode = getattr(heidegger, 'dasein_mode', 'unknown')
        dasein_name = dasein_names.get(dasein_mode, dasein_mode)
        lines.append(f"   Модус Dasein: {dasein_name}")
        
        existentiale_type = getattr(heidegger, 'existentiale_type', None)
        if existentiale_type:
            lines.append(f"   Экзистенциал: {existentiale_type}")
        
        reference = getattr(heidegger, 'reference', None)
        if reference:
            lines.append(f"   Отсылка: {reference}")
        
        is_ironic = getattr(heidegger, 'is_ironic', False)
        if is_ironic:
            lines.append("   Ирония: обнаружена (текст содержит скрытый смысл)")
        
        hold_break_risk = getattr(heidegger, 'hold_break_risk', False)
        hold_break_type = getattr(heidegger, 'hold_break_type', None)
        if hold_break_risk:
            lines.append(f"   Риск HOLD_BREAK: ОБНАРУЖЕН ({hold_break_type or 'неуточнённый тип'})")
            lines.append("   → Текст подрывает собственную онтологическую основу")
        else:
            lines.append("   Риск HOLD_BREAK: отсутствует")
        
        confidence = getattr(heidegger, 'confidence', 0.0)
        lines.append(f"   Уверенность интерпретации: {confidence:.0%}")
    
    if dostoevsky is not None:
        config = getattr(dostoevsky, 'polyphonic_config', None)
        if config:
            lines.append(f"   Полифоническая конфигурация: {config}")
        operator = getattr(dostoevsky, 'operator', None)
        if operator:
            lines.append(f"   Оператор поэтики: {operator}")
    
    lines.append("\n3. ЧИСЛОВЫЕ ПОКАЗАТЕЛИ")
    lines.append(f"   Базовый индекс (синтаксис): {base_index:+.3f}")
    
    if heidegger is not None:
        adjustment = getattr(heidegger, 'final_index_adjustment', 0.0)
        lines.append(f"   Корректировка от LLM: {adjustment:+.3f}")
        if getattr(heidegger, 'hold_break_risk', False):
            lines.append("   Штраф HOLD_BREAK: -0.250")
    
    lines.append(f"   ИТОГОВЫЙ ИНДЕКС ВОЛИ: {final_index:+.3f}")
    
    lines.append("\n4. ИНТЕРПРЕТАЦИЯ")
    lines.append(f"   Статус: {health_status}")
    
    if health_status == "high_stability":
        lines.append("   Текст обладает высокой онтологической устойчивостью.")
    elif health_status == "stable":
        lines.append("   Текст онтологически стабилен.")
    elif health_status == "critical":
        lines.append("   Текст находится в критическом состоянии.")
    
    if final_index > 0.5:
        lines.append("   Преобладает воля к утверждению и действию.")
    elif final_index < -0.3:
        lines.append("   Преобладает воля к вопрошанию и демонтажу.")
    else:
        lines.append("   Баланс между утверждением и вопрошанием.")
    
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def build_temporal_cardiogram(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Строит темпоральную кардиограмму по предложениям.
    
    Анализирует последовательность результатов анализа и строит
    временной ряд индекса воли с выявлением трендов, паттернов и зон риска.
    
    Args:
        results: список результатов анализа (каждый с полем final_index)
    
    Returns:
        dict с ключами:
            - timeline: список значений индекса по предложениям
            - syntax_types: типы высказываний
            - health_sequence: статусы здоровья
            - mean: средний индекс
            - peak: максимальный индекс
            - valley: минимальный индекс
            - volatility: размах (peak - valley)
            - trend: "rising" | "falling" | "stable"
            - slope: наклон линии тренда
            - total_points: количество точек
            - critical_points: индексы точек < -0.3
            - stability_break_points: индексы точек < -0.5
            - patterns: найденные паттерны (sharp_rise, sharp_drop)
    """
    indices = [r.get("final_index", 0.0) for r in results]
    health_statuses = [r.get("health_status", "unknown") for r in results]
    
    # Извлечение syntax_types
    syntax_types = []
    for r in results:
        syntax = r.get("syntax", {})
        if isinstance(syntax, dict):
            syntax_types.append(syntax.get("type", "unknown"))
        else:
            syntax_types.append("unknown")
    
    if not indices:
        return {"error": "Нет данных", "timeline": [], "total_points": 0}
    
    # Метрики
    mean_idx = sum(indices) / len(indices)
    peak = max(indices)
    valley = min(indices)
    volatility = peak - valley
    
    # Зоны риска
    critical_points = [i for i, v in enumerate(indices) if v < -0.3]
    stability_break_points = [i for i, v in enumerate(indices) if v < -0.5]
    
    # Тренд (линейная регрессия)
    if len(indices) >= 2:
        x = list(range(len(indices)))
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(indices)
        sum_xy = sum(x[i] * indices[i] for i in range(n))
        sum_xx = sum(x[i] * x[i] for i in range(n))
        
        denom = n * sum_xx - sum_x * sum_x
        if denom != 0:
            slope = (n * sum_xy - sum_x * sum_y) / denom
        else:
            slope = 0.0
        
        if slope > 0.02:
            trend = "rising"
        elif slope < -0.02:
            trend = "falling"
        else:
            trend = "stable"
    else:
        trend = "stable"
        slope = 0.0
    
    # Паттерны (резкие скачки)
    patterns = []
    for i in range(len(indices) - 1):
        delta = indices[i+1] - indices[i]
        if delta > 0.3:
            patterns.append({
                "type": "sharp_rise",
                "from": i,
                "to": i + 1,
                "delta": round(delta, 3)
            })
        elif delta < -0.3:
            patterns.append({
                "type": "sharp_drop",
                "from": i,
                "to": i + 1,
                "delta": round(delta, 3)
            })
    
    # Типы смены состояний
    transitions = []
    for i in range(len(health_statuses) - 1):
        if health_statuses[i] != health_statuses[i+1]:
            transitions.append({
                "from": i,
                "to": i + 1,
                "from_status": health_statuses[i],
                "to_status": health_statuses[i+1]
            })
    
    return {
        "timeline": indices,
        "syntax_types": syntax_types,
        "health_sequence": health_statuses,
        "mean": round(mean_idx, 3),
        "peak": round(peak, 3),
        "valley": round(valley, 3),
        "volatility": round(volatility, 3),
        "trend": trend,
        "slope": round(slope, 4),
        "total_points": len(indices),
        "critical_points": critical_points,
        "stability_break_points": stability_break_points,
        "patterns": patterns,
        "transitions": transitions
    }


def build_ascii_cardiogram(cardiogram_data: Dict[str, Any]) -> str:
    """
    Строит ASCII-график кардиограммы для вывода в консоль/текст.
    
    Args:
        cardiogram_data: результат build_temporal_cardiogram()
    
    Returns:
        str: многострочный ASCII-график
    """
    indices = cardiogram_data.get("timeline", [])
    if not indices:
        return "Нет данных для построения кардиограммы."
    
    max_width = 40  # ширина графика в символах
    min_idx = -1.0
    max_idx = 1.0
    
    lines = []
    lines.append("=" * 60)
    lines.append("ТЕМПОРАЛЬНАЯ КАРДИОГРАММА (ASCII)")
    lines.append("=" * 60)
    lines.append(f"  Тренд: {cardiogram_data.get('trend', 'N/A')} | "
                 f"Среднее: {cardiogram_data.get('mean', 0):+.3f} | "
                 f"Волатильность: {cardiogram_data.get('volatility', 0):.3f}")
    lines.append("-" * 60)
    
    for i, idx in enumerate(indices):
        # Нормализация в диапазон 0..max_width
        norm = int(((idx - min_idx) / (max_idx - min_idx)) * max_width)
        norm = max(0, min(max_width, norm))
        
        # Выбор символа в зависимости от зоны
        if idx < -0.5:
            bar_char = "▓"
            status_icon = "🔴"
        elif idx < -0.3:
            bar_char = "▒"
            status_icon = "🟡"
        elif idx > 0.5:
            bar_char = "█"
            status_icon = "🟢"
        else:
            bar_char = "░"
            status_icon = "⚪"
        
        bar = bar_char * norm
        if norm == 0:
            bar = "·"
        
        # Номер предложения
        num = str(i + 1).rjust(2)
        lines.append(f"{num} {status_icon} {bar:<{max_width}} {idx:+.3f}")
    
    lines.append("-" * 60)
    
    # Легенда
    lines.append("Легенда:  🔴 < -0.5  |  🟡 -0.5..-0.3  |  ⚪ -0.3..+0.5  |  🟢 > +0.5")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# Экспорты
__all__ = [
    "apply_fusion_layer",
    "generate_textual_analysis",
    "build_temporal_cardiogram",
    "build_ascii_cardiogram"
]

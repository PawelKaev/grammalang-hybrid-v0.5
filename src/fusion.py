"""
fusion.py
Fusion Layer: слияние синтаксического и семантического слоёв.
Генерирует JSON + текстовый разбор.
"""

import logging
from typing import Dict, Any

from src.deep_interpreter import HeideggerAnalysis

logger = logging.getLogger("fusion")


def apply_fusion_layer(syntax_data: Dict[str, Any], heidegger: HeideggerAnalysis) -> Dict[str, Any]:
    """
    Применяет слой слияния и генерирует текстовый разбор.
    """
    # Извлечение компонентов
    lexical_index = syntax_data.get("lexical_index", 0.0)
    syntax_type = syntax_data.get("syntax_type", "assertion")
    depth = syntax_data.get("depth", 1)
    modal_modifier = syntax_data.get("modal_modifier", 0.0)
    node_types = syntax_data.get("node_types", [])
    lexical_markers = syntax_data.get("lexical_markers", [])

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
    final_index = base_index + heidegger.final_index_adjustment

    if heidegger.hold_break_risk:
        final_index -= 0.25

    final_index = round(final_index, 3)
    health_status = "critical" if heidegger.hold_break_risk else "stable"
    if final_index < -0.5:
        health_status = "critical"
    elif final_index > 0.8:
        health_status = "high_stability"

    # ========== ГЕНЕРАЦИЯ ТЕКСТОВОГО РАЗБОРА ==========
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
        heidegger=heidegger
    )

    result = {
        "final_index": final_index,
        "health_status": health_status,
        "dasein_mode": heidegger.dasein_mode,
        "hold_break_detected": heidegger.hold_break_risk,
        "hold_break_type": heidegger.hold_break_type,
        "interpretation": heidegger.model_dump(),
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
    syntax_type: str,
    node_types: list,
    lexical_markers: list,
    depth: int,
    modal_modifier: float,
    syntax_modifier: float,
    base_index: float,
    final_index: float,
    health_status: str,
    heidegger: HeideggerAnalysis
) -> str:
    """Генерирует читаемый текстовый разбор."""

    # Карта русских названий
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
    dasein_name = dasein_names.get(heidegger.dasein_mode, heidegger.dasein_mode)

    lines = []
    lines.append("=" * 60)
    lines.append("ОНТОЛОГИЧЕСКИЙ РАЗБОР ТЕКСТА")
    lines.append("=" * 60)

    # Часть 1: Синтаксис
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

    # Часть 2: Семантика
    lines.append("\n2. СЕМАНТИЧЕСКИЙ АНАЛИЗ")
    lines.append(f"   Модус Dasein: {dasein_name}")
    if heidegger.existentiale_type:
        lines.append(f"   Экзистенциал: {heidegger.existentiale_type}")
    if heidegger.reference:
        lines.append(f"   Отсылка: {heidegger.reference}")
    if heidegger.is_ironic:
        lines.append("   Ирония: обнаружена (текст содержит скрытый смысл)")

    if heidegger.hold_break_risk:
        lines.append(f"   Риск HOLD_BREAK: ОБНАРУЖЕН ({heidegger.hold_break_type or 'неуточнённый тип'})")
        lines.append("   → Текст подрывает собственную онтологическую основу")
    else:
        lines.append("   Риск HOLD_BREAK: отсутствует")

    lines.append(f"   Уверенность интерпретации: {heidegger.confidence:.0%}")

    # Часть 3: Индексы
    lines.append("\n3. ЧИСЛОВЫЕ ПОКАЗАТЕЛИ")
    lines.append(f"   Базовый индекс (синтаксис): {base_index:+.3f}")
    lines.append(f"   Корректировка от LLM: {heidegger.final_index_adjustment:+.3f}")
    if heidegger.hold_break_risk:
        lines.append(f"   Штраф HOLD_BREAK: -0.250")
    lines.append(f"   ИТОГОВЫЙ ИНДЕКС ВОЛИ: {final_index:+.3f}")

    # Часть 4: Интерпретация
    lines.append("\n4. ИНТЕРПРЕТАЦИЯ")
    lines.append(f"   Статус: {health_status}")

    if health_status == "high_stability":
        lines.append("   Текст обладает высокой онтологической устойчивостью.")
        lines.append("   Высказывание утверждает, строит, движется вперёд.")
    elif health_status == "stable":
        lines.append("   Текст онтологически стабилен.")
        lines.append("   Высказывание удерживает свою позицию без угрозы распада.")
    elif health_status == "critical":
        lines.append("   Текст находится в критическом состоянии.")
        lines.append("   Высказывание содержит внутреннее напряжение или противоречие.")

    if final_index > 0.5:
        lines.append("   Преобладает воля к утверждению и действию.")
    elif final_index < -0.3:
        lines.append("   Преобладает воля к вопрошанию и демонтажу.")
    else:
        lines.append("   Баланс между утверждением и вопрошанием.")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)

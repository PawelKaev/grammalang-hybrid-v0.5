"""
deep_interpreter.py
Семантический интерпретатор GrammaLang.
Поддерживает Qwen, DeepSeek, три режима анализа.
v0.5.1 — исправлена очистка <think>, увеличен max_tokens.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger("deep_interpreter")

# Пути к моделям
MODEL_PATH_14B = os.environ.get(
    "DEEP_MODEL_PATH",
    "C:\\Projects\\grammalang_hybrid\\models\\Qwen3-14B-Q4_K_M.gguf"
)
MODEL_PATH_16B = os.environ.get(
    "DEEP_MODEL_PATH_16B",
    "C:\\Projects\\grammalang_hybrid\\models\\Qwen3-16B-Q4_K_M.gguf"
)

# Системные промпты
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_model = None


class HeideggerAnalysis:
    """Результат анализа в режиме Аристотеля."""
    def __init__(self, dasein_mode="unknown", existentiale_type=None,
                 hold_break_risk=False, hold_break_type=None,
                 final_index_adjustment=0.0, reference=None,
                 is_ironic=False, confidence=0.5, **kwargs):
        self.dasein_mode = dasein_mode
        self.existentiale_type = existentiale_type
        self.hold_break_risk = hold_break_risk
        self.hold_break_type = hold_break_type
        self.final_index_adjustment = final_index_adjustment
        self.reference = reference
        self.is_ironic = is_ironic
        self.confidence = confidence
    
    def model_dump(self) -> dict:
        return {
            "dasein_mode": self.dasein_mode,
            "existentiale_type": self.existentiale_type,
            "hold_break_risk": self.hold_break_risk,
            "hold_break_type": self.hold_break_type,
            "final_index_adjustment": self.final_index_adjustment,
            "reference": self.reference,
            "is_ironic": self.is_ironic,
            "confidence": self.confidence
        }


class HeideggerResult:
    """Результат анализа в режиме Хайдеггера."""
    def __init__(self, gesture_of_destruction="null", reactor_phase="null",
                 region_of_destruction="null", holding_of_questioning=False,
                 confidence=0.5, **kwargs):
        self.gesture_of_destruction = gesture_of_destruction
        self.reactor_phase = reactor_phase
        self.region_of_destruction = region_of_destruction
        self.holding_of_questioning = holding_of_questioning
        self.confidence = confidence
    
    def model_dump(self) -> dict:
        return {
            "gesture_of_destruction": self.gesture_of_destruction,
            "reactor_phase": self.reactor_phase,
            "region_of_destruction": self.region_of_destruction,
            "holding_of_questioning": self.holding_of_questioning,
            "confidence": self.confidence
        }


class DostoevskyResult:
    """Результат анализа в режиме Достоевского."""
    def __init__(self, polyphonic_configuration="null", operator_type="null",
                 detonation_phase="null", unresolved_tension=False,
                 dominant_ideologeme="null", confidence=0.5, **kwargs):
        self.polyphonic_configuration = polyphonic_configuration
        self.operator_type = operator_type
        self.detonation_phase = detonation_phase
        self.unresolved_tension = unresolved_tension
        self.dominant_ideologeme = dominant_ideologeme
        self.confidence = confidence
    
    def model_dump(self) -> dict:
        return {
            "polyphonic_configuration": self.polyphonic_configuration,
            "operator_type": self.operator_type,
            "detonation_phase": self.detonation_phase,
            "unresolved_tension": self.unresolved_tension,
            "dominant_ideologeme": self.dominant_ideologeme,
            "confidence": self.confidence
        }


def get_model():
    """Загружает семантическую модель (Qwen 14B или 16B)."""
    global _model
    if _model is not None:
        return _model
    
    # Выбор модели: 16B приоритетнее
    if os.path.exists(MODEL_PATH_16B):
        model_path = MODEL_PATH_16B
        logger.info(f"Loading Qwen 3 16B from {model_path}...")
    elif os.path.exists(MODEL_PATH_14B):
        model_path = MODEL_PATH_14B
        logger.info(f"Loading Qwen 3 14B from {model_path}...")
    else:
        raise FileNotFoundError(
            f"Модель не найдена. Проверьте:\n"
            f"  {MODEL_PATH_16B}\n"
            f"  {MODEL_PATH_14B}"
        )
    
    try:
        from local_model import LocalModel
        _model = LocalModel(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=-1
        )
        logger.info("Qwen model loaded")
        return _model
    except ImportError:
        raise ImportError("local_model не найден. Проверьте путь к local-ai-one-button.")


def _clean_response(raw: str) -> str:
    """Очищает ответ модели от технических артефактов."""
    # Удаляем блок <think>...</think>
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    
    # Удаляем маркеры кода
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    
    return raw.strip()


def _parse_json(raw: str, default_result: Any) -> Any:
    """Безопасный парсинг JSON."""
    raw = _clean_response(raw)
    
    if not raw:
        logger.error("Empty response after cleaning")
        return default_result
    
    try:
        data = json.loads(raw)
        logger.info(f"Parsed: {data}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Raw response (first 200 chars): {raw[:200]}")
        return default_result


def _load_prompt(filename: str) -> str:
    """Загружает системный промпт из файла."""
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    logger.warning(f"Prompt file not found: {path}")
    return ""


def deep_interpret(text: str, syntax_data: Dict[str, Any], use_cache: bool = True) -> HeideggerAnalysis:
    """
    Режим Аристотеля: определяет модус Dasein и индекс воли.
    """
    model = get_model()
    
    system_prompt = _load_prompt("deep_interpreter_system.txt")
    if not system_prompt:
        system_prompt = "Ты — герменевтический аналитик. Определи модус Dasein и риск HOLD_BREAK. Верни ТОЛЬКО JSON."
    
    user_prompt = (
        f"Текст: {text}\n"
        f"Синтаксические данные: {json.dumps(syntax_data, ensure_ascii=False)}\n\n"
        f"Верни ТОЛЬКО JSON с полями: dasein_mode, existentiale_type, hold_break_risk, "
        f"hold_break_type, final_index_adjustment, reference, is_ironic, confidence."
    )
    
    logger.info(f"Deep interpreting: {text[:80]}...")
    
    raw = model.chat(
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        temperature=0.15,
        max_tokens=1024
    )
    
    data = _parse_json(raw, {})
    return HeideggerAnalysis(**data) if data else HeideggerAnalysis()


def deep_interpret_heidegger(text: str, syntax_data: Dict[str, Any]) -> HeideggerResult:
    """
    Режим Хайдеггера: определяет жест деструкции и фазу реактора.
    """
    model = get_model()
    
    system_prompt = _load_prompt("heidegger_system.txt")
    if not system_prompt:
        system_prompt = (
            "Ты — анализатор хайдеггеровской деструкции. "
            "Определи жест деструкции, фазу реактора, область и удержание вопрошания. "
            "Верни ТОЛЬКО JSON."
        )
    
    user_prompt = (
        f"Текст: {text}\n"
        f"Синтаксические данные: {json.dumps(syntax_data, ensure_ascii=False)}\n\n"
        f"Верни ТОЛЬКО JSON с полями: gesture_of_destruction, reactor_phase, "
        f"region_of_destruction, holding_of_questioning, confidence."
    )
    
    logger.info(f"Interpreting with custom prompt: {text[:80]}...")
    
    raw = model.chat(
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        temperature=0.15,
        max_tokens=1024
    )
    
    data = _parse_json(raw, {
        "gesture_of_destruction": "null",
        "reactor_phase": "null",
        "region_of_destruction": "null",
        "holding_of_questioning": False,
        "confidence": 0.5
    })
    return HeideggerResult(**data)


def deep_interpret_dostoevsky(text: str, syntax_data: Dict[str, Any]) -> DostoevskyResult:
    """
    Режим Достоевского: определяет полифоническую конфигурацию.
    """
    model = get_model()
    
    system_prompt = _load_prompt("dostoevsky_system.txt")
    if not system_prompt:
        system_prompt = (
            "Ты — анализатор полифонии Достоевского. "
            "Определи конфигурацию голосов, оператор поэтики, фазу детонации. "
            "Верни ТОЛЬКО JSON."
        )
    
    user_prompt = (
        f"Текст: {text}\n"
        f"Синтаксические данные: {json.dumps(syntax_data, ensure_ascii=False)}\n\n"
        f"Верни ТОЛЬКО JSON с полями: polyphonic_configuration, operator_type, "
        f"detonation_phase, unresolved_tension, dominant_ideologeme, confidence."
    )
    
    logger.info(f"Dostoevsky interpreting: {text[:80]}...")
    
    raw = model.chat(
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        temperature=0.15,
        max_tokens=1024
    )
    
    data = _parse_json(raw, {
        "polyphonic_configuration": "null",
        "operator_type": "null",
        "detonation_phase": "null",
        "unresolved_tension": False,
        "dominant_ideologeme": "null",
        "confidence": 0.5
    })
    return DostoevskyResult(**data)

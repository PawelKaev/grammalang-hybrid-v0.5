"""
deep_interpreter.py
Слой Qwen 3 14B: глубокая онтологическая интерпретация через llama-cpp-python.
Поддерживает три режима: Аристотель, Хайдеггер, Достоевский.
"""

import json
import re
import hashlib
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ValidationError

LOCAL_AI_PATH = "C:\\Projects\\local-ai-one-button"
if LOCAL_AI_PATH not in sys.path:
    sys.path.insert(0, LOCAL_AI_PATH)

from local_model import LocalModel

logger = logging.getLogger("deep_interpreter")

PROMPT_FILE = Path("prompts/deep_interpreter_system.txt")
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8") if PROMPT_FILE.exists() else ""


# ==================== МОДЕЛИ ДАННЫХ ====================

class HeideggerAnalysis(BaseModel):
    """Аристотелевская модель: dasein, hold_break, reference."""
    dasein_mode: str = "unknown"
    existentiale_type: Optional[str] = None
    hold_break_risk: bool = False
    hold_break_type: Optional[str] = None
    final_index_adjustment: float = 0.0
    reference: Optional[str] = None
    is_ironic: bool = False
    confidence: float = 0.5


class HeideggerResult(BaseModel):
    """Хайдеггеровская модель: жесты деструкции, фазы реактора."""
    gesture_of_destruction: str = "null"
    reactor_phase: str = "null"
    region_of_destruction: str = "null"
    holding_of_questioning: bool = False
    confidence: float = 0.5


class DostoevskyResult(BaseModel):
    """Модель Достоевского: полифония, операторы, детонация."""
    polyphonic_configuration: str = "null"
    operator_type: str = "null"
    detonation_phase: str = "null"
    unresolved_tension: bool = False
    dominant_ideologeme: str = "null"
    confidence: float = 0.5


DEFAULT_ANALYSIS = HeideggerAnalysis()

# Путь к модели Qwen 3 14B
MODEL_PATH = os.environ.get(
    "DEEP_MODEL_PATH",
    "C:\\Projects\\grammalang_hybrid\\models\\Qwen3-14B-Q4_K_M.gguf"
)

_model = None


def get_model():
    """Ленивая загрузка модели."""
    global _model
    if _model is None:
        logger.info(f"Loading Qwen 3 14B from {MODEL_PATH}...")
        _model = LocalModel(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=12,
            n_gpu_layers=-1
        )
        logger.info("Qwen 3 14B loaded")
    return _model


# ==================== КЭШ ====================

def get_cache_key(text: str, syntax_data: dict) -> str:
    content = text + json.dumps(syntax_data, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_from_cache(text: str, syntax_data: dict) -> HeideggerAnalysis | None:
    key = get_cache_key(text, syntax_data)
    cache_file = CACHE_DIR / f"deep_{key}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return HeideggerAnalysis(**data.get("result", {}))
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
        return None


def save_to_cache(text: str, syntax_data: dict, result: HeideggerAnalysis) -> None:
    key = get_cache_key(text, syntax_data)
    cache_file = CACHE_DIR / f"deep_{key}.json"
    try:
        data = {
            "_timestamp": datetime.now().isoformat(),
            "result": result.model_dump()
        }
        cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Cached (hash: {key[:8]}...)")
    except Exception as e:
        logger.warning(f"Cache write error: {e}")


# ==================== АРИСТОТЕЛЕВСКИЙ РЕЖИМ ====================

def deep_interpret(text: str, syntax_data: dict, use_cache: bool = True) -> HeideggerAnalysis:
    """Глубокая онтологическая интерпретация через Qwen 3 14B (Аристотель)."""
    text = text.strip()
    if not text:
        logger.warning("Empty text received")
        return DEFAULT_ANALYSIS

    if use_cache:
        cached = load_from_cache(text, syntax_data)
        if cached is not None:
            return cached

    logger.info(f"Deep interpreting: {text[:50]}...")

    try:
        user_prompt = (
            f"Текст: {text}\n\n"
            f"Синтаксические данные:\n"
            f"{json.dumps(syntax_data, ensure_ascii=False, indent=2)}"
        )

        model = get_model()
        raw = model.chat(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=SYSTEM_PROMPT,
            temperature=0.15,
            max_tokens=512
        )

        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
        raw = re.sub(r'<thinking>.*?</thinking>', '', raw, flags=re.DOTALL)
        raw = re.sub(r'<thought>.*?</thought>', '', raw, flags=re.DOTALL)
        raw = raw.strip()

        data = json.loads(raw)
        result = HeideggerAnalysis(**data)

        if use_cache:
            save_to_cache(text, syntax_data, result)

        logger.info(
            f"Deep interpretation: dasein_mode={result.dasein_mode}, "
            f"hold_break_risk={result.hold_break_risk}"
        )
        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return DEFAULT_ANALYSIS
    except ValidationError as e:
        logger.error(f"Pydantic validation error: {e}")
        return DEFAULT_ANALYSIS
    except Exception as e:
        logger.error(f"Deep interpreter error: {e}")
        return DEFAULT_ANALYSIS


# ==================== УНИВЕРСАЛЬНЫЙ МЕТОД ====================

def _interpret_with_model(text: str, syntax_data: dict, system_prompt: str, model_class):
    """Универсальный метод интерпретации с указанной моделью."""
    global _model
    if _model is None:
        get_model()

    text = text.strip()
    if not text:
        return model_class()

    logger.info(f"Interpreting with custom prompt: {text[:50]}...")

    try:
        user_prompt = (
            f"Текст: {text}\n\n"
            f"Синтаксические данные:\n"
            f"{json.dumps(syntax_data, ensure_ascii=False, indent=2)}"
        )

        raw = _model.chat(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            temperature=0.15,
            max_tokens=512
        )

        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
        raw = re.sub(r'<thinking>.*?</thinking>', '', raw, flags=re.DOTALL)
        raw = raw.strip()

        data = json.loads(raw)
        return model_class(**data)

    except Exception as e:
        logger.error(f"Interpret error: {e}")
        return model_class()


def deep_interpret_with_prompt(text: str, syntax_data: dict, system_prompt: str, use_cache: bool = False) -> HeideggerAnalysis:
    """Глубокая интерпретация с переданным промптом (Аристотель)."""
    return _interpret_with_model(text, syntax_data, system_prompt, HeideggerAnalysis)


def deep_interpret_heidegger(text: str, syntax_data: dict, use_cache: bool = False) -> HeideggerResult:
    """Хайдеггеровский анализ."""
    prompt = Path("prompts/heidegger_system.txt").read_text(encoding="utf-8")
    return _interpret_with_model(text, syntax_data, prompt, HeideggerResult)


def deep_interpret_dostoevsky(text: str, syntax_data: dict, use_cache: bool = False) -> DostoevskyResult:
    """Анализ по Достоевскому."""
    prompt = Path("prompts/dostoevsky_system.txt").read_text(encoding="utf-8")
    return _interpret_with_model(text, syntax_data, prompt, DostoevskyResult)

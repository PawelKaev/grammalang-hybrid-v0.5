"""
deep_interpreter.py
Семантический интерпретатор GrammaLang.
Поддерживает Qwen, DeepSeek, три режима анализа.
v0.5.1 — локальный импорт local_model.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any

# Добавляем путь к local-ai-one-button
sys.path.insert(0, "C:\\Projects\\local-ai-one-button")

logger = logging.getLogger("deep_interpreter")

MODEL_PATH_14B = os.environ.get("DEEP_MODEL_PATH", "C:\\Projects\\grammalang_hybrid\\models\\Qwen3-14B-Q4_K_M.gguf")
MODEL_PATH_16B = os.environ.get("DEEP_MODEL_PATH_16B", "C:\\Projects\\grammalang_hybrid\\models\\Qwen3-16B-Q4_K_M.gguf")

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_model = None


class HeideggerAnalysis:
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
        return {k: v for k, v in self.__dict__.items()}


class HeideggerResult:
    def __init__(self, gesture_of_destruction="null", reactor_phase="null",
                 region_of_destruction="null", holding_of_questioning=False,
                 confidence=0.5, **kwargs):
        self.gesture_of_destruction = gesture_of_destruction
        self.reactor_phase = reactor_phase
        self.region_of_destruction = region_of_destruction
        self.holding_of_questioning = holding_of_questioning
        self.confidence = confidence
    
    def model_dump(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class DostoevskyResult:
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
        return {k: v for k, v in self.__dict__.items()}


def get_model():
    global _model
    if _model is not None:
        return _model
    
    if os.path.exists(MODEL_PATH_16B):
        model_path = MODEL_PATH_16B
        logger.info(f"Loading Qwen 3 16B from {model_path}...")
    elif os.path.exists(MODEL_PATH_14B):
        model_path = MODEL_PATH_14B
        logger.info(f"Loading Qwen 3 14B from {model_path}...")
    else:
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH_16B} или {MODEL_PATH_14B}")
    
    try:
        from local_model import LocalModel
    except ImportError:
        from src.local_model import LocalModel
    
    _model = LocalModel(model_path=model_path, n_ctx=2048, n_threads=8, n_gpu_layers=-1)
    logger.info("Qwen model loaded")
    return _model


def _clean_response(raw: str) -> str:
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


def _parse_json(raw: str, default_result: Any) -> Any:
    raw = _clean_response(raw)
    if not raw:
        return default_result
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default_result


def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8") if path.exists() else ""


def deep_interpret(text: str, syntax_data: Dict[str, Any], use_cache: bool = True) -> HeideggerAnalysis:
    model = get_model()
    system_prompt = _load_prompt("deep_interpreter_system.txt")
    user_prompt = f"Текст: {text}\nСинтаксис: {json.dumps(syntax_data, ensure_ascii=False)}\nВерни ТОЛЬКО JSON."
    
    raw = model.chat(messages=[{"role": "user", "content": user_prompt}],
                     system_prompt=system_prompt, temperature=0.15, max_tokens=1024)
    data = _parse_json(raw, {})
    return HeideggerAnalysis(**data) if data else HeideggerAnalysis()


def deep_interpret_heidegger(text: str, syntax_data: Dict[str, Any]) -> HeideggerResult:
    model = get_model()
    system_prompt = _load_prompt("heidegger_system.txt")
    user_prompt = f"Текст: {text}\nСинтаксис: {json.dumps(syntax_data, ensure_ascii=False)}\nВерни ТОЛЬКО JSON."
    
    raw = model.chat(messages=[{"role": "user", "content": user_prompt}],
                     system_prompt=system_prompt, temperature=0.15, max_tokens=1024)
    data = _parse_json(raw, {"gesture_of_destruction": "null", "reactor_phase": "null",
                              "region_of_destruction": "null", "holding_of_questioning": False, "confidence": 0.5})
    return HeideggerResult(**data)


def deep_interpret_dostoevsky(text: str, syntax_data: Dict[str, Any]) -> DostoevskyResult:
    model = get_model()
    system_prompt = _load_prompt("dostoevsky_system.txt")
    user_prompt = f"Текст: {text}\nСинтаксис: {json.dumps(syntax_data, ensure_ascii=False)}\nВерни ТОЛЬКО JSON."
    
    raw = model.chat(messages=[{"role": "user", "content": user_prompt}],
                     system_prompt=system_prompt, temperature=0.15, max_tokens=1024)
    data = _parse_json(raw, {"polyphonic_configuration": "null", "operator_type": "null",
                              "detonation_phase": "null", "unresolved_tension": False,
                              "dominant_ideologeme": "null", "confidence": 0.5})
    return DostoevskyResult(**data)

"""
fast_parser.py
Синтаксический анализатор GrammaLang.
Использует HY-MT1.5-7B для классификации типов высказываний.
v0.5.1 — автосоздание папки logs, локальный импорт local_model.
"""

import json
import logging
import os
import sys
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

# Добавляем путь к local-ai-one-button
sys.path.insert(0, "C:\\Projects\\local-ai-one-button")

logger = logging.getLogger("fast_parser")

# Автосоздание папки logs
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "grammalang.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Путь к модели
MODEL_PATH = os.environ.get(
    "FAST_MODEL_PATH",
    "C:\\Projects\\grammalang_hybrid\\models\\HY-MT1.5-7B.Q4_K_M.gguf"
)

_model = None
_cache: Dict[str, Dict[str, Any]] = {}

LEXICAL_MARKERS = {
    "absolute": ["всегда", "вечно", "абсолютно", "несомненно", "безусловно", "воистину"],
    "doubt": ["возможно", "кажется", "вероятно", "может быть", "скорее всего"],
    "negation": ["не", "нет", "никто", "ничто", "никогда", "нигде"],
    "question": ["почему", "зачем", "как", "что", "где", "когда", "кто", "разве", "неужели"],
    "imperative": ["должен", "надо", "нужно", "следует", "обязан", "познай", "делай"],
    "ontology": ["бытие", "сущее", "сущность", "существование", "присутствие", "Dasein"],
    "deconstruction": ["деструкция", "демонтаж", "разбор", "анализ", "критика"],
}

NODE_TYPES_MAP = {
    "absolute": "Absolute", "doubt": "Doubt", "negation": "Negation",
    "question": "Question", "imperative": "Imperative",
    "ontology": "Ontology", "deconstruction": "Deconstruction",
}


def get_model():
    global _model
    if _model is not None:
        return _model
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")
    
    logger.info(f"Loading HY-MT model from {MODEL_PATH}...")
    try:
        from local_model import LocalModel
        _model = LocalModel(model_path=MODEL_PATH, n_ctx=512, n_threads=4, n_gpu_layers=-1)
        logger.info("HY-MT model loaded")
        return _model
    except ImportError:
        try:
            from src.local_model import LocalModel
            _model = LocalModel(model_path=MODEL_PATH, n_ctx=512, n_threads=4, n_gpu_layers=-1)
            logger.info("HY-MT model loaded (from src)")
            return _model
        except ImportError:
            logger.error("local_model not found")
            raise ImportError("local_model не найден.")


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _quick_lexical_analysis(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    markers_found = []
    node_types_found = []
    
    for category, words in LEXICAL_MARKERS.items():
        for word in words:
            if word.lower() in text_lower:
                markers_found.append(word)
                node_name = NODE_TYPES_MAP.get(category, category)
                if node_name not in [nt[1] for nt in node_types_found]:
                    node_types_found.append((category, node_name))
                break
    
    syntax_type = "assertion"
    if any(w in text_lower for w in ["разве", "неужели", "?"]):
        syntax_type = "rhetorical_question"
    elif any(w in text_lower for w in ["должен", "надо", "нужно", "следует", "обязан", "познай", "делай", "!"]):
        syntax_type = "imperative"
    elif any(w in text_lower for w in ["есть", "является", "определяется", "суть"]):
        syntax_type = "definition"
    elif any(w in text_lower for w in ["если бы", "был бы", "мог бы"]):
        syntax_type = "counterfactual"
    elif any(w in text_lower for w in ["пусть", "да будет", "желаю"]):
        syntax_type = "optative"
    
    modal_modifier = 0.0
    if any(w in text_lower for w in LEXICAL_MARKERS["absolute"]):
        modal_modifier += 0.3
    if any(w in text_lower for w in LEXICAL_MARKERS["doubt"]):
        modal_modifier -= 0.2
    if any(w in text_lower for w in LEXICAL_MARKERS["negation"]):
        modal_modifier -= 0.15
    
    depth = 1
    if len(text) > 100:
        depth = 2
    if len(text) > 200:
        depth = 3
    if any(w in text_lower for w in ["который", "поэтому", "следовательно", "однако"]):
        depth = max(depth, 3)
    
    lexical_index = 0.0
    if syntax_type == "imperative":
        lexical_index = 0.4
    elif syntax_type == "definition":
        lexical_index = 0.3
    elif syntax_type == "rhetorical_question":
        lexical_index = -0.3
    elif syntax_type == "counterfactual":
        lexical_index = -0.35
    elif syntax_type == "optative":
        lexical_index = -0.3
    
    return {
        "syntax_type": syntax_type,
        "node_types": [nt[1] for nt in node_types_found],
        "lexical_markers": markers_found[:5],
        "depth": depth,
        "modal_modifier": round(modal_modifier, 2),
        "lexical_index": round(lexical_index, 2)
    }


def fast_parse(text: str, use_cache: bool = True) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {"syntax_type": "assertion", "node_types": [], "lexical_markers": [],
                "depth": 1, "modal_modifier": 0.0, "lexical_index": 0.0}
    
    text_hash = _text_hash(text)
    if use_cache and text_hash in _cache:
        logger.info(f"Cache hit (hash: {text_hash}...)")
        return _cache[text_hash].copy()
    
    result = _quick_lexical_analysis(text)
    
    if len(text) > 50 or result["syntax_type"] == "assertion":
        try:
            model = get_model()
            logger.info(f"Processing: {text[:80]}...")
            
            prompt = (
                "Определи тип высказывания и онтологические узлы. Верни ТОЛЬКО JSON.\n\n"
                f"Текст: {text[:200]}\n\n"
                'Формат: {"syntax_type": "assertion|imperative|definition|rhetorical_question|counterfactual|optative", '
                '"node_types": ["Absolute","Doubt","Negation","Question","Imperative","Ontology","Deconstruction"], '
                '"lexical_markers": ["word1","word2"], "depth": 1-5, "modal_modifier": -0.5..0.5}'
            )
            
            raw = model.chat(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="Ты — синтаксический анализатор. Верни только JSON.",
                temperature=0.1, max_tokens=256
            )
            
            raw = raw.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.endswith("```"):
                raw = raw[:-3]
            
            model_result = json.loads(raw)
            result.update(model_result)
            logger.info(f"Result: syntax_type={result.get('syntax_type')}, depth={result.get('depth')}")
        except Exception as e:
            logger.warning(f"Model parse failed: {e}")
    
    if use_cache:
        _cache[text_hash] = result.copy()
        logger.info(f"Cached (hash: {text_hash}...)")
    
    return result


def clear_cache():
    global _cache
    _cache = {}
    logger.info("Cache cleared")


__all__ = ["fast_parse", "get_model", "clear_cache"]


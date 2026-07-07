"""
fast_parser.py
Слой HY-MT1.5-7B: быстрый синтаксический анализ через llama-cpp-python.
"""

import json
import hashlib
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Добавляем путь к local_model из проекта локального ИИ
LOCAL_AI_PATH = "C:\\Projects\\local-ai-one-button"
if LOCAL_AI_PATH not in sys.path:
    sys.path.insert(0, LOCAL_AI_PATH)

from local_model import LocalModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/grammalang.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fast_parser")

PROMPT_FILE = Path("prompts/fast_parser_system.txt")
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8") if PROMPT_FILE.exists() else ""

DEFAULT_RESPONSE = {
    "syntax_type": "assertion",
    "node_types": [],
    "lexical_markers": [],
    "depth": 1,
    "modal_modifier": 0.0
}

# Путь к модели HY-MT
MODEL_PATH = os.environ.get(
    "HY_MT_MODEL_PATH",
    "C:\\Projects\\grammalang_hybrid\\models\\HY-MT1.5-7B.Q4_K_M.gguf"
)

# Ленивая загрузка модели
_model = None

def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading HY-MT model from {MODEL_PATH}...")
        _model = LocalModel(model_path=MODEL_PATH, n_ctx=1024, n_threads=8, n_gpu_layers=-1)
        logger.info("HY-MT model loaded")
    return _model

def get_cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_from_cache(text: str) -> dict | None:
    key = get_cache_key(text)
    cache_file = CACHE_DIR / f"fast_{key}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        timestamp = datetime.fromisoformat(data.get("_timestamp", "2000-01-01T00:00:00"))
        if (datetime.now() - timestamp).total_seconds() < 24 * 3600:
            logger.info(f"Cache hit (hash: {key[:8]}...)")
            return data.get("result", DEFAULT_RESPONSE)
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
    return None

def save_to_cache(text: str, result: dict) -> None:
    key = get_cache_key(text)
    cache_file = CACHE_DIR / f"fast_{key}.json"
    try:
        data = {"_timestamp": datetime.now().isoformat(), "result": result}
        cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Cached (hash: {key[:8]}...)")
    except Exception as e:
        logger.warning(f"Cache write error: {e}")

def fast_parse(text: str, use_cache: bool = True) -> dict:
    text = text.strip()
    if not text:
        logger.warning("Empty text received")
        return DEFAULT_RESPONSE

    if use_cache:
        cached = load_from_cache(text)
        if cached is not None:
            return cached

    logger.info(f"Processing: {text[:50]}...")

    try:
        model = get_model()
        raw = model.chat(
            messages=[{"role": "user", "content": text}],
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=256
        )
        # Очистка
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        result = json.loads(raw)

        if "syntax_type" not in result:
            result["syntax_type"] = "assertion"
        if "node_types" not in result:
            result["node_types"] = []
        if "lexical_markers" not in result:
            result["lexical_markers"] = []
        if "depth" not in result:
            result["depth"] = 1
        if "modal_modifier" not in result:
            result["modal_modifier"] = 0.0

        try:
            result["depth"] = max(1, min(5, int(result["depth"])))
        except (ValueError, TypeError):
            result["depth"] = 1

        if use_cache:
            save_to_cache(text, result)

        logger.info(f"Result: syntax_type={result['syntax_type']}, depth={result['depth']}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return DEFAULT_RESPONSE
    except Exception as e:
        logger.error(f"Fast parser error: {e}")
        return DEFAULT_RESPONSE

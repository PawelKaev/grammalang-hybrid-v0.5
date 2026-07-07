#!/usr/bin/env python3
"""
GrammaLang Hybrid v0.4.2
Гибридный онтологический анализатор: HY-MT1.5-7B + Qwen 3 14B
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.fast_parser import fast_parse
from src.deep_interpreter import deep_interpret
from src.fusion import apply_fusion_layer


def setup_logging(verbose: bool = False):
    """Настройка логирования."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/grammalang.log"),
            logging.StreamHandler()
        ]
    )


def split_sentences(text: str) -> list[str]:
    """Простое разбиение на предложения (для MVP)."""
    delimiters = ['.', '!', '?', ';']
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in delimiters:
            if current.strip():
                sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


def main():
    parser = argparse.ArgumentParser(
        description="GrammaLang Hybrid — онтологический анализатор текста"
    )
    parser.add_argument("--input", "-i", required=True, help="Путь к входному текстовому файлу")
    parser.add_argument("--output", "-o", default="result.json", help="Путь к выходному JSON-файлу")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод в консоль")
    parser.add_argument("--no-cache", action="store_true", help="Отключить кэширование")
    parser.add_argument("--batch", action="store_true", help="Обрабатывать текст по предложениям")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("main")
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        logger.error("Input file is empty")
        sys.exit(1)
    
    logger.info(f"Processing file: {args.input} ({len(text)} chars)")
    use_cache = not args.no_cache
    
    if args.batch:
        sentences = split_sentences(text)
        logger.info(f"Batch mode: {len(sentences)} sentences detected")
        results = []
        for i, sentence in enumerate(sentences):
            if not sentence:
                continue
            logger.info(f"Processing sentence {i+1}/{len(sentences)}: {sentence[:50]}...")
            syntax = fast_parse(sentence, use_cache=use_cache)
            heidegger = deep_interpret(sentence, syntax, use_cache=use_cache)
            fusion = apply_fusion_layer(syntax, heidegger)
            fusion["sentence"] = sentence
            fusion["sentence_index"] = i
            results.append(fusion)
        
        if results:
            indices = [r.get("final_index", 0.0) for r in results]
            summary = {
                "total_sentences": len(results),
                "mean_final_index": round(sum(indices) / len(results), 3),
                "min_final_index": round(min(indices), 3),
                "max_final_index": round(max(indices), 3),
                "critical_sentences": sum(1 for r in results if r.get("health_status") == "critical"),
                "stable_sentences": sum(1 for r in results if r.get("health_status") == "stable")
            }
            output = {"mode": "batch", "summary": summary, "sentences": results}
        else:
            output = {"error": "No sentences processed"}
    else:
        syntax = fast_parse(text, use_cache=use_cache)
        heidegger = deep_interpret(text, syntax, use_cache=use_cache)
        fusion = apply_fusion_layer(syntax, heidegger)
        fusion["text"] = text
        output = {"mode": "single", "result": fusion}
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Result saved to: {args.output}")
    
    if args.verbose:
        print("\n" + "="*60)
        print("RESULT:")
        print(json.dumps(output, ensure_ascii=False, indent=2))
        print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

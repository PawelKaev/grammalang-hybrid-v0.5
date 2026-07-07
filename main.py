"""
main.py
GrammaLang Hybrid v0.5.0 — CLI с поддержкой темпоральной кардиограммы.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from src.fast_parser import fast_parse
from src.deep_interpreter import deep_interpret, deep_interpret_heidegger, deep_interpret_dostoevsky
from src.fusion import apply_fusion_layer, build_temporal_cardiogram, build_ascii_cardiogram
from src.cardiogram import save_cardiogram_png


def split_sentences(text: str) -> list:
    """Разбивает текст на предложения."""
    delimiters = ['.', '!', '?', ';', '\n']
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in delimiters:
            stripped = current.strip()
            if stripped and len(stripped) > 1:
                sentences.append(stripped)
            current = ""
    if current.strip() and len(current.strip()) > 1:
        sentences.append(current.strip())
    return sentences


def analyze_single(text: str, mode: str = "aristotle") -> dict:
    """Анализ одного текста."""
    syntax = fast_parse(text)
    
    if mode == "heidegger":
        h = deep_interpret_heidegger(text, syntax)
        fusion = apply_fusion_layer(syntax, heidegger=h)
    elif mode == "dostoevsky":
        d = deep_interpret_dostoevsky(text, syntax)
        fusion = apply_fusion_layer(syntax, dostoevsky=d)
    else:
        h = deep_interpret(text, syntax)
        fusion = apply_fusion_layer(syntax, heidegger=h)
    
    return fusion


def analyze_batch(text: str, mode: str = "aristotle", verbose: bool = False) -> list:
    """Пакетный анализ по предложениям."""
    sentences = split_sentences(text)
    results = []
    
    for i, sentence in enumerate(sentences, 1):
        if verbose:
            print(f"[{i}/{len(sentences)}] {sentence[:60]}...")
        
        try:
            syntax = fast_parse(sentence)
            
            if mode == "heidegger":
                h = deep_interpret_heidegger(sentence, syntax)
                fusion = apply_fusion_layer(syntax, heidegger=h)
            elif mode == "dostoevsky":
                d = deep_interpret_dostoevsky(sentence, syntax)
                fusion = apply_fusion_layer(syntax, dostoevsky=d)
            else:
                h = deep_interpret(sentence, syntax)
                fusion = apply_fusion_layer(syntax, heidegger=h)
            
            fusion["sentence"] = sentence
            fusion["sentence_num"] = i
            results.append(fusion)
        except Exception as e:
            if verbose:
                print(f"  ⚠ Ошибка: {e}")
            results.append({
                "sentence": sentence,
                "sentence_num": i,
                "final_index": 0.0,
                "health_status": "error",
                "syntax": {"type": "unknown"},
                "error": str(e)
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="GrammaLang Hybrid v0.5.0 — Онтологический анализатор с темпоральной кардиограммой",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main.py --input text.txt --output result.json
  python main.py --input text.txt --batch --cardiogram
  python main.py --input text.txt --mode heidegger --verbose
  python main.py --input text.txt --batch --cardiogram-png plot.png
        """
    )
    
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Путь к входному .txt файлу")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Путь для сохранения JSON-результата")
    parser.add_argument("--mode", "-m", type=str, default="aristotle",
                        choices=["aristotle", "heidegger", "dostoevsky"],
                        help="Режим анализа (по умолчанию: aristotle)")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Пакетный режим: анализ по предложениям")
    parser.add_argument("--cardiogram", "-c", action="store_true",
                        help="Построить ASCII-кардиограмму (требует --batch)")
    parser.add_argument("--cardiogram-png", type=str, default=None,
                        help="Сохранить визуальную кардиограмму в PNG (требует --batch)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Подробный вывод")
    parser.add_argument("--no-cache", action="store_true",
                        help="Отключить кэш")
    parser.add_argument("--version", action="store_true",
                        help="Показать версию")
    
    args = parser.parse_args()
    
    # Версия
    if args.version:
        from src import __version__
        print(f"GrammaLang Hybrid v{__version__}")
        return
    
    # Чтение файла
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Ошибка: файл '{args.input}' не найден.")
        sys.exit(1)
    
    text = input_path.read_text(encoding="utf-8")
    
    if args.verbose:
        print(f"📖 Файл: {args.input}")
        print(f"📏 Символов: {len(text)}")
        print(f"🎯 Режим: {args.mode}")
        print(f"📦 Пакетный: {'да' if args.batch else 'нет'}")
        print("-" * 50)
    
    start_time = time.time()
    
    # Анализ
    if args.batch:
        sentences = split_sentences(text)
        if args.verbose:
            print(f"📝 Предложений: {len(sentences)}")
        
        results = analyze_batch(text, mode=args.mode, verbose=args.verbose)
        
        # Вывод результатов
        print(f"\n{'='*60}")
        print(f"ПАКЕТНЫЙ АНАЛИЗ: {len(sentences)} предложений")
        print(f"{'='*60}")
        
        for r in results:
            status = r.get("health_status", "unknown")
            icon = "🟢" if status in ("stable", "high_stability") else "🔴" if status == "critical" else "⚪"
            idx = r.get("final_index", 0)
            sent = r.get("sentence", "")[:60]
            print(f"{icon} [{r.get('sentence_num', '?'):2d}] {idx:+.3f} | {sent}...")
        
        print(f"{'='*60}")
        
        # Кардиограмма ASCII
        if args.cardiogram:
            cardiogram_data = build_temporal_cardiogram(results)
            ascii_art = build_ascii_cardiogram(cardiogram_data)
            print("\n")
            print(ascii_art)
        
        # Кардиограмма PNG
        if args.cardiogram_png:
            cardiogram_data = build_temporal_cardiogram(results)
            path = save_cardiogram_png(cardiogram_data, args.cardiogram_png)
            print(f"\n💾 Кардиограмма сохранена: {path}")
        
        # Сохранение JSON
        if args.output:
            output_data = {
                "version": "0.5.0",
                "mode": args.mode,
                "input_file": str(input_path),
                "total_sentences": len(sentences),
                "results": results
            }
            
            if args.cardiogram or args.cardiogram_png:
                cardiogram_data = build_temporal_cardiogram(results)
                output_data["cardiogram"] = cardiogram_data
            
            output_path = Path(args.output)
            output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.verbose:
                print(f"📄 Результат сохранён: {args.output}")
    
    else:
        # Одиночный анализ
        result = analyze_single(text, mode=args.mode)
        
        print("\n" + result.get("analysis_text", ""))
        
        if args.output:
            output_data = {
                "version": "0.5.0",
                "mode": args.mode,
                "input_file": str(input_path),
                "result": result
            }
            output_path = Path(args.output)
            output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.verbose:
                print(f"\n📄 Результат сохранён: {args.output}")
    
    elapsed = time.time() - start_time
    if args.verbose:
        print(f"\n⏱ Время выполнения: {elapsed:.2f} сек.")


if __name__ == "__main__":
    main()

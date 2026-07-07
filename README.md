# GrammaLang Hybrid v0.4.2

**Гибридный онтологический анализатор:** HY-MT1.5-7B + Qwen 3 14B

## Архитектура

[Текст] -> HY-MT1.5-7B -> [синтаксическая разметка] -> Qwen 3 14B -> [онтологическая интерпретация] -> Fusion Layer -> [итоговый JSON]

## Требования
- Python 3.10+
- Ollama с моделями hy-mt1.5-7b и qwen3:14b
- 16+ ГБ VRAM

## Установка
pip install -r requirements.txt

text

## Запуск
python main.py --input input.txt --output result.json --verbose

text

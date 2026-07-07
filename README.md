# GrammaLang Hybrid v0.5.0

**Онтологический анализатор текста с темпоральной кардиограммой.**

## Что нового в v0.5

- **Темпоральная кардиограмма** — динамика индекса воли по предложениям
- **Визуальный график** — отдельное окно с matplotlib и тулбаром
- **ASCII-кардиограмма** — текстовая версия для консоли
- **Тёмная тема GUI** — современный интерфейс на Tkinter

## Быстрый старт

### Установка
\\\ash
git clone https://github.com/PawelKaev/grammalang-hybrid-v0.5.git
cd grammalang-hybrid-v0.5
pip install -r requirements.txt
\\\

### Запуск GUI
\\\ash
python gui_app.py
\\\

### Запуск CLI с кардиограммой
\\\ash
python main.py --input text.txt --batch --cardiogram --cardiogram-png plot.png
\\\

## Документация
- [Changelog v0.5.0](docs/v0.5_changelog.md)
- [Оригинальный репозиторий v0.4.2](https://github.com/PawelKaev/grammalang-hybrid)

## Лицензия
MIT

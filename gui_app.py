"""
gui_app.py
GUI для GrammaLang Hybrid с тремя режимами анализа:
Аристотель (сборка), Хайдеггер (демонтаж), Достоевский (полифонический детонатор).
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import json
import sys
import os
import re
from pathlib import Path

sys.path.insert(0, "C:\\Projects\\local-ai-one-button")

from src.fast_parser import fast_parse, get_model as get_fast_model
from src.deep_interpreter import (
    HeideggerAnalysis, deep_interpret_with_prompt,
    deep_interpret_heidegger, deep_interpret_dostoevsky,
    HeideggerResult, DostoevskyResult
)
from src.fusion import apply_fusion_layer


class GrammaLangGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GrammaLang Hybrid v0.4.2 — Онтологический анализатор")
        self.root.geometry("1100x900")
        self.root.minsize(900, 700)

        self.deep_model = None
        self.backend_choice = tk.StringVar(value="local_qwen")
        self.api_key = tk.StringVar(value="")
        self.output_mode = tk.StringVar(value="full")
        self.analysis_mode = tk.StringVar(value="aristotle")

        self._create_widgets()
        self._loading_screen()
        threading.Thread(target=self._load_fast_model, daemon=True).start()

    def _loading_screen(self):
        self.status_var.set("Загрузка HY-MT...")
        self.btn_analyze.configure(state='disabled')
        self.btn_batch.configure(state='disabled')

    def _load_fast_model(self):
        try:
            self._log_gui("Загрузка HY-MT1.5-7B (синтаксис)...")
            get_fast_model()
            self.root.after(0, lambda: self.status_var.set(
                "HY-MT готов. Выберите режим анализа и нажмите «Переключить»."
            ))
            self.root.after(0, lambda: self.btn_switch.configure(state='normal'))
        except Exception as e:
            self.root.after(0, lambda: self._on_error(str(e)))

    def _create_widgets(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        # Выбор режима анализа
        mode_frame = ttk.Frame(top_frame)
        mode_frame.pack(fill=tk.X, pady=5)

        ttk.Label(mode_frame, text="Режим анализа:",
                  font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.mode_combo = ttk.Combobox(
            mode_frame, textvariable=self.analysis_mode,
            values=["aristotle", "heidegger", "dostoevsky"],
            state="readonly", width=30
        )
        self.mode_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(mode_frame, text="Аристотель — сборка | Хайдеггер — демонтаж | Достоевский — полифония",
                  font=("Segoe UI", 9, "italic"), foreground="gray").pack(side=tk.LEFT, padx=10)

        # Выбор бэкенда
        backend_frame = ttk.Frame(top_frame)
        backend_frame.pack(fill=tk.X, pady=5)

        ttk.Label(backend_frame, text="Семантический движок:",
                  font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=5)
        self.backend_combo = ttk.Combobox(
            backend_frame, textvariable=self.backend_choice,
            values=["local_qwen", "local_deepseek", "deepseek_api"],
            state="readonly", width=25
        )
        self.backend_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(backend_frame, text="API ключ:",
                  font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=5)
        self.api_entry = ttk.Entry(backend_frame, textvariable=self.api_key, width=35, show="*")
        self.api_entry.pack(side=tk.LEFT, padx=5)

        self.btn_switch = ttk.Button(backend_frame, text="Переключить",
                                     command=self._on_switch_backend, state='disabled')
        self.btn_switch.pack(side=tk.LEFT, padx=5)

        # Режим вывода
        ttk.Label(backend_frame, text="Вывод:",
                  font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(20, 5))
        self.mode_out_combo = ttk.Combobox(
            backend_frame, textvariable=self.output_mode,
            values=["full", "text_only", "json_only"],
            state="readonly", width=15
        )
        self.mode_out_combo.pack(side=tk.LEFT, padx=5)

        # Поле ввода
        ttk.Label(top_frame, text="Введите текст для анализа:",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(10, 0))

        input_frame = ttk.Frame(top_frame)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.input_text = tk.Text(input_frame, height=8, wrap=tk.WORD, font=("Segoe UI", 12))
        input_scrollbar = ttk.Scrollbar(input_frame, command=self.input_text.yview)
        self.input_text.configure(yscrollcommand=input_scrollbar.set)
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        input_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Кнопки
        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        self.btn_analyze = ttk.Button(btn_frame, text="🔍 Анализировать",
                                      command=self._on_analyze, state='disabled')
        self.btn_analyze.pack(side=tk.LEFT, padx=5)

        self.btn_batch = ttk.Button(btn_frame, text="📊 По предложениям",
                                    command=self._on_analyze_batch, state='disabled')
        self.btn_batch.pack(side=tk.LEFT, padx=5)

        self.btn_paste = ttk.Button(btn_frame, text="📋 Вставить", command=self._on_paste)
        self.btn_paste.pack(side=tk.LEFT, padx=5)

        self.btn_clear = ttk.Button(btn_frame, text="Очистить", command=self._on_clear)
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        # Привязка Ctrl+V
        self.root.bind('<Control-v>', lambda e: self._on_paste())
        self.root.bind('<Control-V>', lambda e: self._on_paste())

        # Область вывода
        output_frame = ttk.Frame(self.root)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        ttk.Label(output_frame, text="Результат анализа:",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")

        output_text_frame = ttk.Frame(output_frame)
        output_text_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.output_text = tk.Text(output_text_frame, wrap=tk.WORD, font=("Consolas", 11))
        output_scrollbar = ttk.Scrollbar(output_text_frame, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=output_scrollbar.set)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Статус-бар
        self.status_var = tk.StringVar(value="Загрузка...")
        self.status_bar = ttk.Label(
            self.root, textvariable=self.status_var,
            relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ==================== ВСТАВКА ====================

    def _on_paste(self):
        try:
            text = self.root.clipboard_get()
            self.input_text.insert(tk.INSERT, text)
        except:
            pass

    # ==================== ТЕКСТОВЫЕ РАЗБОРЫ ====================

    def _generate_heidegger_analysis_text(self, syntax_data, h: HeideggerResult):
        syntax_type = syntax_data.get("syntax_type", "assertion")
        depth = syntax_data.get("depth", 1)
        node_types = syntax_data.get("node_types", [])

        gesture_names = {
            "destruction_of_question": "Деструкция вопроса",
            "destruction_of_subjectivity": "Деструкция субъектности",
            "destruction_of_copula": "Деструкция связки",
            "destruction_of_spatiality": "Деструкция пространственности",
            "null": "Не определён"
        }
        phase_names = {
            "injection": "Инжекция (вброс понятия)",
            "irradiation": "Облучение (раскрытие смыслов)",
            "plasma": "Плазма (предельное напряжение)",
            "crystallization": "Кристаллизация (новый смысл)",
            "null": "Не определена"
        }
        region_names = {
            "question_of_being": "Вопрос о бытии (§1-4)",
            "dasein_man": "Dasein и Man (§25-27)",
            "being_in_world": "Бытие-в-мире (§12-24)",
            "temporality": "Временность (§65-83)",
            "null": "Не определена"
        }

        lines = ["=" * 60, "ХАЙДЕГГЕРОВСКИЙ АНАЛИЗ: СЕМАНТИЧЕСКИЙ РЕАКТОР", "=" * 60]
        lines.append(f"\n1. ЖЕСТ ДЕСТРУКЦИИ: {gesture_names.get(h.gesture_of_destruction, h.gesture_of_destruction)}")
        lines.append(f"2. ФАЗА РЕАКТОРА: {phase_names.get(h.reactor_phase, h.reactor_phase)}")
        lines.append(f"3. ОБЛАСТЬ ДЕСТРУКЦИИ: {region_names.get(h.region_of_destruction, h.region_of_destruction)}")
        lines.append(f"4. УДЕРЖАНИЕ ВОПРОШАНИЯ: {'Да' if h.holding_of_questioning else 'Нет'}")
        lines.append(f"\n5. СИНТАКСИЧЕСКАЯ ОСНОВА: тип={syntax_type}, глубина={depth}")
        if node_types:
            lines.append(f"   Узлы: {', '.join(node_types)}")
        lines.append(f"\n6. УВЕРЕННОСТЬ: {h.confidence:.0%}")
        lines.append("\n" + "=" * 60 + "\nДЕМОНТАЖ ЗАВЕРШЁН. ВОПРОШАНИЕ УДЕРЖАНО.\n" + "=" * 60)
        return "\n".join(lines)

    def _generate_dostoevsky_analysis_text(self, syntax_data, d: DostoevskyResult):
        syntax_type = syntax_data.get("syntax_type", "assertion")
        depth = syntax_data.get("depth", 1)
        node_types = syntax_data.get("node_types", [])

        config_names = {
            "double_voice": "Двойной голос",
            "mirror_pair": "Зеркальная пара",
            "chorus": "Хор голосов",
            "internal_dialogue": "Внутренний диалог",
            "null": "Не определена"
        }
        operator_names = {
            "condensation": "Конденсация (сгущение смысла)",
            "nadryv_vdrug": "Надрыв и «вдруг»",
            "double_self": "Двойник (расщепление)",
            "odnako_zhe": "«Однако же» (удержание конфликта)",
            "null": "Не определён"
        }
        det_phase_names = {
            "condensation": "Конденсация (напряжение)",
            "interference": "Интерференция (столкновение)",
            "detonation": "Детонация (взрыв)",
            "redistribution": "Перераспределение",
            "null": "Не определена"
        }
        ideologeme_names = {
            "свобода и преступление": "Свобода и преступление",
            "вера и неверие": "Вера и неверие",
            "страдание и искупление": "Страдание и искупление",
            "гордость и смирение": "Гордость и смирение",
            "красота и безобразие": "Красота и безобразие",
            "null": "Не определена"
        }

        lines = ["=" * 60, "ДОСТОЕВСКИЙ: ПОЛИФОНИЧЕСКИЙ ДЕТОНАТОР", "=" * 60]
        lines.append(f"\n1. ПОЛИФОНИЧЕСКАЯ КОНФИГУРАЦИЯ: {config_names.get(d.polyphonic_configuration, d.polyphonic_configuration)}")
        lines.append(f"2. ОПЕРАТОР: {operator_names.get(d.operator_type, d.operator_type)}")
        lines.append(f"3. ФАЗА ДЕТОНАЦИИ: {det_phase_names.get(d.detonation_phase, d.detonation_phase)}")
        lines.append(f"4. НЕРАЗРЕШИМОЕ НАПРЯЖЕНИЕ: {'Да' if d.unresolved_tension else 'Нет'}")
        if d.unresolved_tension:
            lines.append("   Конфликт остаётся продуктивным. Смысл не собирается в синтез.")
        lines.append(f"5. ДОМИНАНТНАЯ ИДЕОЛОГЕМА: {ideologeme_names.get(d.dominant_ideologeme, d.dominant_ideologeme)}")
        lines.append(f"\n6. СИНТАКСИЧЕСКАЯ ОСНОВА: тип={syntax_type}, глубина={depth}")
        if node_types:
            lines.append(f"   Узлы: {', '.join(node_types)}")
        lines.append(f"\n7. УВЕРЕННОСТЬ: {d.confidence:.0%}")
        lines.append("\n" + "=" * 60 + "\nНАПРЯЖЕНИЕ УДЕРЖАНО. ГОЛОСА В РАВНОВЕСИИ.\n" + "=" * 60)
        return "\n".join(lines)

    # ==================== ОБЩИЕ МЕТОДЫ ====================

    def _on_switch_backend(self):
        backend = self.backend_choice.get()
        self.btn_analyze.configure(state='disabled')
        self.btn_batch.configure(state='disabled')
        self.status_var.set(f"Переключение на {backend}...")
        threading.Thread(target=self._load_backend, args=(backend,), daemon=True).start()

    def _load_backend(self, backend):
        try:
            if backend == "local_qwen":
                self._log_gui("Загрузка Qwen 3 14B...")
                from src.deep_interpreter import get_model as get_qwen
                self.deep_model = get_qwen()
            elif backend == "local_deepseek":
                self._log_gui("Загрузка DeepSeek локально...")
                from local_model import LocalModel
                self.deep_model = LocalModel(
                    model_path="C:\\Projects\\grammalang_hybrid\\models\\deepseek-multilingual-q4_k_m.gguf",
                    n_ctx=2048, n_threads=12, n_gpu_layers=-1
                )
            elif backend == "deepseek_api":
                self._log_gui("DeepSeek API — готов.")
                self.deep_model = None
            self.root.after(0, lambda: self._on_backend_ready(backend))
        except Exception as e:
            self.root.after(0, lambda: self._on_error(str(e)))

    def _on_backend_ready(self, backend):
        mode_names = {
            "aristotle": "Аристотель (сборка)",
            "heidegger": "Хайдеггер (демонтаж)",
            "dostoevsky": "Достоевский (полифония)"
        }
        mode_name = mode_names.get(self.analysis_mode.get(), self.analysis_mode.get())
        self.status_var.set(f"Готов. Режим: {mode_name} | Движок: {backend}")
        self.btn_analyze.configure(state='normal')
        self.btn_batch.configure(state='normal')

    def _call_semantic_model(self, text, syntax_data):
        backend = self.backend_choice.get()
        mode = self.analysis_mode.get()
        if backend == "deepseek_api":
            return self._call_deepseek_api(text, syntax_data, mode)
        else:
            if mode == "heidegger":
                return deep_interpret_heidegger(text, syntax_data)
            elif mode == "dostoevsky":
                return deep_interpret_dostoevsky(text, syntax_data)
            else:
                from src.deep_interpreter import deep_interpret
                return deep_interpret(text, syntax_data, use_cache=False)

    def _call_deepseek_api(self, text, syntax_data, mode="aristotle"):
        api_key = self.api_key.get().strip()
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            self._log_gui("API ключ не указан.")
            return HeideggerAnalysis()
        try:
            import requests
            if mode == "heidegger":
                prompt = Path("prompts/heidegger_system.txt").read_text(encoding="utf-8")
                prompt += f"\n\nТекст: {text}\nСинтаксис: {json.dumps(syntax_data, ensure_ascii=False)}"
            elif mode == "dostoevsky":
                prompt = Path("prompts/dostoevsky_system.txt").read_text(encoding="utf-8")
                prompt += f"\n\nТекст: {text}\nСинтаксис: {json.dumps(syntax_data, ensure_ascii=False)}"
            else:
                prompt = f"Ты — герменевтический аналитик. Верни ТОЛЬКО JSON:\nТекст: {text}\nСинтаксис: {json.dumps(syntax_data, ensure_ascii=False)}\n"
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.15, "max_tokens": 256, "response_format": {"type": "json_object"}},
                timeout=15
            )
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                return HeideggerAnalysis(**json.loads(content))
            else:
                self._log_gui(f"API error: {response.status_code}")
                return HeideggerAnalysis()
        except Exception as e:
            self._log_gui(f"API exception: {e}")
            return HeideggerAnalysis()

    def _on_analyze(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Пустой ввод", "Введите текст для анализа.")
            return
        self.status_var.set("Анализ...")
        self.btn_analyze.configure(state='disabled')
        self.btn_batch.configure(state='disabled')
        self.output_text.delete("1.0", tk.END)
        threading.Thread(target=self._run_analysis, args=(text, False), daemon=True).start()

    def _on_analyze_batch(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Пустой ввод", "Введите текст для анализа.")
            return
        self.status_var.set("Батч-анализ...")
        self.btn_analyze.configure(state='disabled')
        self.btn_batch.configure(state='disabled')
        self.output_text.delete("1.0", tk.END)
        threading.Thread(target=self._run_analysis, args=(text, True), daemon=True).start()

    def _run_analysis(self, text, batch_mode):
        try:
            result = self._batch_analysis(text) if batch_mode else self._single_analysis(text)
            self.root.after(0, lambda r=result: self._show_result(r))
        except Exception as e:
            self.root.after(0, lambda: self._on_error(str(e)))

    def _single_analysis(self, text):
        mode = self.analysis_mode.get()
        self._log_gui("▸ Синтаксический анализ (HY-MT1.5-7B)...")
        syntax = fast_parse(text, use_cache=False)
        self._log_gui(f"  Тип: {syntax['syntax_type']}, глубина: {syntax['depth']}")

        if mode == "heidegger":
            self._log_gui("▸ Хайдеггеровский анализ (демонтаж)...")
            h = self._call_semantic_model(text, syntax)
            self._log_gui(f"  Жест: {h.gesture_of_destruction}, Фаза: {h.reactor_phase}")
            return {"analysis_text": self._generate_heidegger_analysis_text(syntax, h), "mode": "heidegger", "syntax": syntax, "heidegger": h.model_dump()}
        elif mode == "dostoevsky":
            self._log_gui("▸ Анализ по Достоевскому (полифония)...")
            d = self._call_semantic_model(text, syntax)
            self._log_gui(f"  Конфигурация: {d.polyphonic_configuration}, Оператор: {d.operator_type}")
            return {"analysis_text": self._generate_dostoevsky_analysis_text(syntax, d), "mode": "dostoevsky", "syntax": syntax, "dostoevsky": d.model_dump()}
        else:
            self._log_gui("▸ Семантический анализ (Аристотель)...")
            h = self._call_semantic_model(text, syntax)
            self._log_gui(f"  Dasein: {h.dasein_mode}, HOLD_BREAK: {h.hold_break_risk}")
            self._log_gui("▸ Слияние слоёв...")
            return apply_fusion_layer(syntax, h)

    def _batch_analysis(self, text):
        mode = self.analysis_mode.get()
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
        results = []
        for i, sentence in enumerate(sentences, 1):
            self._log_gui(f"▸ Предложение {i}/{len(sentences)}: {sentence[:40]}...")
            syntax = fast_parse(sentence, use_cache=False)
            h = self._call_semantic_model(sentence, syntax)
            if mode == "heidegger":
                results.append({"sentence": sentence, "syntax_type": syntax["syntax_type"],
                                "gesture": h.gesture_of_destruction,
                                "phase": h.reactor_phase,
                                "holding": h.holding_of_questioning})
            elif mode == "dostoevsky":
                results.append({"sentence": sentence, "syntax_type": syntax["syntax_type"],
                                "config": d.polyphonic_configuration,
                                "operator": d.operator_type,
                                "det_phase": d.detonation_phase,
                                "unresolved": d.unresolved_tension})
            else:
                fusion = apply_fusion_layer(syntax, h)
                results.append({"sentence": sentence, "syntax_type": syntax["syntax_type"],
                                "final_index": fusion["final_index"], "health_status": fusion["health_status"],
                                "dasein_mode": h.dasein_mode})
        return {"mode": mode, "total_sentences": len(sentences), "sentences": results}

    def _format_output(self, data):
        output_mode = self.output_mode.get()
        if output_mode == "json_only":
            return json.dumps(data, ensure_ascii=False, indent=2)
        if "analysis_text" in data:
            return data["analysis_text"] if output_mode == "text_only" else data["analysis_text"] + "\n\n" + "=" * 60 + "\nJSON:\n" + "=" * 60 + "\n" + json.dumps(data, ensure_ascii=False, indent=2)
        if "sentences" in data:
            lines = [f"БАТЧ-АНАЛИЗ: {data['total_sentences']} предложений\n"]
            for s in data["sentences"]:
                lines.append(f"\n{s['sentence']}")
                if "final_index" in s:
                    lines.append(f"  Индекс: {s['final_index']:+.3f} | Статус: {s['health_status']}")
                elif "gesture" in s:
                    lines.append(f"  Жест: {s['gesture']} | Фаза: {s['phase']} | Удержание: {s['holding']}")
                elif "config" in s:
                    lines.append(f"  Конфигурация: {s['config']} | Оператор: {s['operator']} | Фаза: {s['det_phase']}")
            return "\n".join(lines)
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _show_result(self, data):
        self.output_text.insert("1.0", self._format_output(data))
        self.status_var.set("Анализ завершён.")
        self.btn_analyze.configure(state='normal')
        self.btn_batch.configure(state='normal')

    def _log_gui(self, msg):
        self.root.after(0, lambda: self._append_log(msg))

    def _append_log(self, msg):
        self.output_text.insert(tk.END, msg + "\n")
        self.output_text.see(tk.END)

    def _on_clear(self):
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)

    def _on_error(self, msg):
        self.status_var.set(f"Ошибка: {msg}")
        messagebox.showerror("Ошибка", msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = GrammaLangGUI(root)
    root.mainloop()

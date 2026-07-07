"""
gui_app.py
GrammaLang Hybrid v0.5.1 — Онтологический анализатор.
Три режима: Аристотель, Хайдеггер, Достоевский.
Темпоральная кардиограмма с визуализацией.
Поддержка DeepSeek API.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import traceback
import threading
import os
import requests
from pathlib import Path

from src.fast_parser import fast_parse, get_model as get_fast_model
from src.deep_interpreter import (
    deep_interpret_heidegger, deep_interpret_dostoevsky,
    get_model as get_deep_model,
    HeideggerResult, DostoevskyResult, HeideggerAnalysis
)
from src.fusion import apply_fusion_layer, build_temporal_cardiogram, build_ascii_cardiogram
from src.cardiogram import plot_cardiogram, save_cardiogram_png


class GrammaLangGUI:
    
    def __init__(self, root):
        self.root = root
        self.root.title("GrammaLang Hybrid v0.5.1")
        self.root.geometry("1100x850")
        
        self.mode_var = tk.StringVar(value="aristotle")
        self.engine_var = tk.StringVar(value="local_qwen")
        self.api_key_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Нажмите «Переключить» для загрузки моделей.")
        self.cardiogram_status = tk.StringVar(value="Выполните пакетный анализ в режиме Аристотеля.")
        
        self._models_loaded = False
        self._batch_results = None
        self._loading = False
        
        self._build_ui()
        print("[GrammaLang] Запущен.")
    
    def _build_ui(self):
        # Верхняя панель
        top = tk.Frame(self.root, bg='#1e1e2e')
        top.pack(fill=tk.X, padx=10, pady=10)
        
        # Первая строка: режим + движок + переключить
        row1 = tk.Frame(top, bg='#1e1e2e')
        row1.pack(fill=tk.X, pady=2)
        
        tk.Label(row1, text="Режим:", bg='#1e1e2e', fg='white', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.mode_combo = ttk.Combobox(row1, textvariable=self.mode_var, values=['aristotle','heidegger','dostoevsky'], state='readonly', width=15)
        self.mode_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row1, text="Движок:", bg='#1e1e2e', fg='white', font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(20, 0))
        self.engine_combo = ttk.Combobox(row1, textvariable=self.engine_var, values=['local_qwen', 'deepseek_api'], state='readonly', width=15)
        self.engine_combo.pack(side=tk.LEFT, padx=5)
        self.engine_combo.bind('<<ComboboxSelected>>', self._on_engine_change)
        
        self.btn_switch = tk.Button(row1, text="Переключить", command=self._on_switch, bg='#45475a', fg='white', font=('Segoe UI', 10))
        self.btn_switch.pack(side=tk.LEFT, padx=20)
        
        # Вторая строка: API ключ
        row2 = tk.Frame(top, bg='#1e1e2e')
        row2.pack(fill=tk.X, pady=2)
        
        self.api_label = tk.Label(row2, text="API ключ:", bg='#1e1e2e', fg='#6c7086', font=('Segoe UI', 10))
        self.api_label.pack(side=tk.LEFT)
        self.api_entry = tk.Entry(row2, textvariable=self.api_key_var, width=40, bg='#313244', fg='white', insertbackground='white', font=('Segoe UI', 10), show='*')
        self.api_entry.pack(side=tk.LEFT, padx=5)
        
        # Вкладки
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.analysis_tab = tk.Frame(self.notebook, bg='#1e1e2e')
        self.notebook.add(self.analysis_tab, text="Анализ")
        self._build_analysis_tab()
        
        self.cardiogram_tab = tk.Frame(self.notebook, bg='#1e1e2e')
        self.notebook.add(self.cardiogram_tab, text="Кардиограмма")
        self._build_cardiogram_tab()
        
        tk.Label(self.root, textvariable=self.status_var, bg='#1e1e2e', fg='#a6adc8', font=('Segoe UI', 9), anchor=tk.W).pack(fill=tk.X, padx=10, pady=5)
    
    def _build_analysis_tab(self):
        self.input_text = tk.Text(self.analysis_tab, height=8, bg='#313244', fg='white', insertbackground='white', font=('Consolas', 11))
        self.input_text.pack(fill=tk.X, padx=10, pady=5)
        self.input_text.insert('1.0', 'Разве Бог не умер? Мы убили его. Как утешимся мы?')
        
        btns = tk.Frame(self.analysis_tab, bg='#1e1e2e')
        btns.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_analyze = tk.Button(btns, text="Анализировать", command=self._on_analyze, bg='#45475a', fg='white', font=('Segoe UI', 10))
        self.btn_analyze.pack(side=tk.LEFT, padx=3)
        
        self.btn_batch = tk.Button(btns, text="По предложениям", command=self._on_batch, bg='#45475a', fg='white', font=('Segoe UI', 10))
        self.btn_batch.pack(side=tk.LEFT, padx=3)
        
        tk.Button(btns, text="Очистить", command=self._on_clear, bg='#45475a', fg='white', font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=3)
        
        self.output_text = tk.Text(self.analysis_tab, height=20, bg='#1a1a2e', fg='white', font=('Consolas', 11))
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def _build_cardiogram_tab(self):
        tk.Label(self.cardiogram_tab, text="Темпоральная кардиограмма", bg='#1e1e2e', fg='white', font=('Segoe UI', 14, 'bold')).pack(pady=10)
        tk.Label(self.cardiogram_tab, text="Динамика индекса воли по предложениям.", bg='#1e1e2e', fg='#a6adc8', font=('Segoe UI', 9)).pack()
        
        btns = tk.Frame(self.cardiogram_tab, bg='#1e1e2e')
        btns.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(btns, text="Построить кардиограмму", command=self._on_cardiogram, bg='#45475a', fg='white', font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=3)
        tk.Button(btns, text="Сохранить PNG", command=self._on_save_png, bg='#45475a', fg='white', font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=3)
        tk.Button(btns, text="Показать ASCII", command=self._on_ascii, bg='#45475a', fg='white', font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=3)
        
        tk.Label(self.cardiogram_tab, textvariable=self.cardiogram_status, bg='#1e1e2e', fg='#a6adc8', font=('Segoe UI', 9)).pack(pady=5)
        tk.Label(self.cardiogram_tab, text="\n\n\nГрафик откроется в отдельном окне\nпосле построения кардиограммы.\n\n\n", bg='#1e1e2e', fg='#6c7086', font=('Segoe UI', 11)).pack(expand=True)
    
    # ============================================================
    # Переключение движка
    # ============================================================
    
    def _on_engine_change(self, event=None):
        engine = self.engine_var.get()
        if engine == "deepseek_api":
            self.api_label.configure(fg='white')
            self.btn_switch.configure(state='disabled', text='API готов')
            self._models_loaded = True  # API не требует загрузки
            self._status("DeepSeek API выбран. Введите API ключ.")
        else:
            self.api_label.configure(fg='#6c7086')
            self._models_loaded = False
            self.btn_switch.configure(state='normal', text='Переключить')
            self._status("Локальный Qwen выбран. Нажмите «Переключить».")
    
    def _on_switch(self):
        engine = self.engine_var.get()
        if engine == "deepseek_api":
            self._status("DeepSeek API уже готов. Введите ключ.")
            return
        if self._models_loaded:
            self._status("Модели уже загружены.")
            return
        if self._loading:
            return
        self._loading = True
        self._status("Загрузка моделей...")
        self.btn_switch.configure(state='disabled', text='Загрузка...')
        threading.Thread(target=self._load_models, daemon=True).start()
    
    def _load_models(self):
        try:
            self._status("HY-MT1.5-7B (синтаксис)...")
            get_fast_model()
            self._status("Qwen 14B (семантика)...")
            get_deep_model()
            self._models_loaded = True
            self._status("Готово. Выберите режим и нажмите «Анализировать».")
        except Exception as e:
            self._status(f"Ошибка загрузки: {e}")
        finally:
            self._loading = False
            self.root.after(0, lambda: self.btn_switch.configure(text='Загружено'))
    
    # ============================================================
    # Вызов модели (локальной или API)
    # ============================================================
    
    def _call_semantic(self, text, syntax, mode):
        """Вызывает семантическую модель в зависимости от движка."""
        engine = self.engine_var.get()
        
        if engine == "deepseek_api":
            return self._call_deepseek_api(text, syntax, mode)
        else:
            return self._call_local_model(text, syntax, mode)
    
    def _call_local_model(self, text, syntax, mode):
        """Вызов локальной Qwen."""
        if mode == "heidegger":
            return deep_interpret_heidegger(text, syntax)
        elif mode == "dostoevsky":
            return deep_interpret_dostoevsky(text, syntax)
        else:
            from src.deep_interpreter import deep_interpret
            return deep_interpret(text, syntax)
    
    def _call_deepseek_api(self, text, syntax, mode):
        """Вызов DeepSeek API."""
        api_key = self.api_key_var.get().strip()
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            self._status("API ключ не указан!")
            if mode == "heidegger":
                return HeideggerResult()
            elif mode == "dostoevsky":
                return DostoevskyResult()
            else:
                return HeideggerAnalysis()
        
        try:
            # Выбор промпта
            if mode == "heidegger":
                system_prompt = Path("prompts/heidegger_system.txt").read_text(encoding="utf-8")
                user_prompt = f"Текст: {text}\nСинтаксис: {json.dumps(syntax, ensure_ascii=False)}"
            elif mode == "dostoevsky":
                system_prompt = Path("prompts/dostoevsky_system.txt").read_text(encoding="utf-8")
                user_prompt = f"Текст: {text}\nСинтаксис: {json.dumps(syntax, ensure_ascii=False)}"
            else:
                system_prompt = Path("prompts/deep_interpreter_system.txt").read_text(encoding="utf-8")
                user_prompt = f"Текст: {text}\nСинтаксис: {json.dumps(syntax, ensure_ascii=False)}"
            
            self._status("DeepSeek API: запрос...")
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.15,
                    "max_tokens": 512,
                    "response_format": {"type": "json_object"}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                # Очистка
                import re
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                
                result = json.loads(content)
                self._status("DeepSeek API: ответ получен.")
                
                if mode == "heidegger":
                    return HeideggerResult(**result)
                elif mode == "dostoevsky":
                    return DostoevskyResult(**result)
                else:
                    return HeideggerAnalysis(**result)
            else:
                self._status(f"API ошибка: {response.status_code}")
                if mode == "heidegger":
                    return HeideggerResult()
                elif mode == "dostoevsky":
                    return DostoevskyResult()
                else:
                    return HeideggerAnalysis()
                    
        except Exception as e:
            self._status(f"API исключение: {e}")
            if mode == "heidegger":
                return HeideggerResult()
            elif mode == "dostoevsky":
                return DostoevskyResult()
            else:
                return HeideggerAnalysis()
    
    # ============================================================
    # Анализ
    # ============================================================
    
    def _on_analyze(self):
        engine = self.engine_var.get()
        if engine == "local_qwen" and not self._models_loaded:
            messagebox.showinfo("Модели не загружены", "Нажмите «Переключить».")
            return
        if engine == "deepseek_api" and not self.api_key_var.get().strip():
            messagebox.showinfo("API ключ", "Введите API ключ DeepSeek.")
            return
        
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            return
        
        mode = self.mode_var.get()
        self._status(f"Анализ ({mode})...")
        self.output_text.delete("1.0", tk.END)
        self.btn_analyze.configure(state='disabled')
        self.btn_batch.configure(state='disabled')
        
        threading.Thread(target=self._run_analysis, args=(text, mode), daemon=True).start()
    
    def _run_analysis(self, text, mode):
        try:
            text_short = text[:300]
            syntax = fast_parse(text_short)
            
            if mode == "heidegger":
                h = self._call_semantic(text_short, syntax, mode)
                result = self._format_heidegger(syntax, h)
            elif mode == "dostoevsky":
                d = self._call_semantic(text_short, syntax, mode)
                result = self._format_dostoevsky(syntax, d)
            else:
                h = self._call_semantic(text_short, syntax, mode)
                fusion = apply_fusion_layer(syntax, h)
                result = fusion.get("analysis_text", "")
                if result:
                    result += "\n\n"
                result += json.dumps({k: v for k, v in fusion.items() if k != "analysis_text"}, ensure_ascii=False, indent=2)
            
            self.root.after(0, lambda: self._show_result(result))
        except Exception as e:
            self.root.after(0, lambda: self._show_error(f"{e}\n\n{traceback.format_exc()}"))
    
    def _on_batch(self):
        engine = self.engine_var.get()
        if engine == "local_qwen" and not self._models_loaded:
            messagebox.showinfo("Модели не загружены", "Нажмите «Переключить».")
            return
        if engine == "deepseek_api" and not self.api_key_var.get().strip():
            messagebox.showinfo("API ключ", "Введите API ключ DeepSeek.")
            return
        
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            return
        
        mode = self.mode_var.get()
        self._status(f"Батч-анализ ({mode})...")
        self.output_text.delete("1.0", tk.END)
        self._batch_results = None
        self.btn_analyze.configure(state='disabled')
        self.btn_batch.configure(state='disabled')
        
        threading.Thread(target=self._run_batch, args=(text, mode), daemon=True).start()
    
    def _run_batch(self, text, mode):
        try:
            sentences = []
            cur = ""
            for ch in text:
                cur += ch
                if ch in '.!?;\n':
                    s = cur.strip()
                    if s and len(s) > 1:
                        sentences.append(s)
                    cur = ""
            if cur.strip() and len(cur.strip()) > 1:
                sentences.append(cur.strip())
            
            if not sentences:
                self.root.after(0, lambda: self._show_error("Не удалось разбить текст на предложения."))
                return
            
            results = []
            for i, s in enumerate(sentences, 1):
                self._status(f"Предложение {i}/{len(sentences)}...")
                try:
                    s_short = s[:200]
                    syntax = fast_parse(s_short)
                    
                    if mode == "heidegger":
                        h = self._call_semantic(s_short, syntax, mode)
                        results.append({"sentence": s[:60], "num": i, "gesture": h.gesture_of_destruction})
                    elif mode == "dostoevsky":
                        d = self._call_semantic(s_short, syntax, mode)
                        results.append({"sentence": s[:60], "num": i, "config": d.polyphonic_configuration})
                    else:
                        h = self._call_semantic(s_short, syntax, mode)
                        fusion = apply_fusion_layer(syntax, h)
                        fusion["sentence"] = s[:60]
                        fusion["num"] = i
                        results.append(fusion)
                except Exception as e:
                    results.append({"sentence": s[:60], "num": i, "error": str(e)})
            
            self._batch_results = results
            
            lines = [f"ПАКЕТНЫЙ АНАЛИЗ: {len(sentences)} предложений (режим: {mode})", "=" * 50]
            for r in results:
                num = r.get('num', '?')
                if "final_index" in r:
                    lines.append(f"[{num:2d}] {r.get('final_index', 0):+.3f} | {r.get('sentence', '')}")
                elif "gesture" in r:
                    lines.append(f"[{num:2d}] Жест: {r.get('gesture')} | {r.get('sentence', '')}")
                elif "config" in r:
                    lines.append(f"[{num:2d}] Конфиг: {r.get('config')} | {r.get('sentence', '')}")
                elif "error" in r:
                    lines.append(f"[{num:2d}] ОШИБКА | {r.get('sentence', '')}")
            lines.append("=" * 50)
            
            if mode == "aristotle":
                lines.append("\n✅ Перейдите на вкладку «Кардиограмма» для визуализации.")
            
            self.root.after(0, lambda: self._show_result("\n".join(lines)))
        except Exception as e:
            self.root.after(0, lambda: self._show_error(f"{e}\n\n{traceback.format_exc()}"))
    
    # ============================================================
    # Кардиограмма
    # ============================================================
    
    def _on_cardiogram(self):
        if not self._batch_results:
            messagebox.showwarning("Нет данных", "Сначала выполните пакетный анализ в режиме Аристотеля.")
            return
        
        data = [r for r in self._batch_results if "final_index" in r]
        if not data:
            messagebox.showwarning("Не те данные", "Нужен режим Аристотеля.")
            return
        
        try:
            cardiogram_data = build_temporal_cardiogram(data)
            self._display_cardiogram(cardiogram_data)
            self.cardiogram_status.set(f"Тренд: {cardiogram_data.get('trend', 'N/A')} | Среднее: {cardiogram_data.get('mean', 0):+.3f}")
        except Exception as e:
            traceback.print_exc()
    
    def _on_save_png(self):
        if not self._batch_results:
            return
        data = [r for r in self._batch_results if "final_index" in r]
        if not data:
            return
        fp = filedialog.asksaveasfilename(defaultextension=".png")
        if fp:
            save_cardiogram_png(build_temporal_cardiogram(data), fp)
    
    def _on_ascii(self):
        if not self._batch_results:
            return
        data = [r for r in self._batch_results if "final_index" in r]
        if not data:
            return
        art = build_ascii_cardiogram(build_temporal_cardiogram(data))
        w = tk.Toplevel(self.root)
        w.title("ASCII-кардиограмма")
        w.geometry("950x700")
        t = tk.Text(w, bg='#1a1a2e', fg='white', font=('Consolas', 11))
        t.pack(fill=tk.BOTH, expand=True)
        t.insert("1.0", art)
        t.configure(state='disabled')
    
    def _display_cardiogram(self, data):
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            w = tk.Toplevel(self.root)
            w.title("Темпоральная кардиограмма")
            w.geometry("1200x750")
            fig = plot_cardiogram(data)
            canvas = FigureCanvasTkAgg(fig, master=w)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            NavigationToolbar2Tk(canvas, w).pack(fill=tk.X)
            tk.Button(w, text="Закрыть", command=w.destroy, bg='#45475a', fg='white').pack(pady=5)
        except:
            pass
    
    # ============================================================
    # Форматирование
    # ============================================================
    
    def _format_heidegger(self, syntax, h: HeideggerResult):
        if h.gesture_of_destruction == "null" and h.confidence <= 0.5:
            return "\n".join([
                "ХАЙДЕГГЕРОВСКИЙ АНАЛИЗ", "=" * 50, "",
                "⚠ Модель вернула пустой ответ.",
                "", "=" * 50,
                json.dumps(h.model_dump(), ensure_ascii=False, indent=2)
            ])
        
        g = {"destruction_of_question": "Деструкция вопроса",
             "destruction_of_subjectivity": "Деструкция субъектности",
             "destruction_of_copula": "Деструкция связки",
             "destruction_of_spatiality": "Деструкция пространственности"}
        p = {"injection": "Инжекция", "irradiation": "Облучение",
             "plasma": "Плазма", "crystallization": "Кристаллизация"}
        r = {"question_of_being": "Вопрос о бытии", "dasein_man": "Dasein и Man",
             "being_in_world": "Бытие-в-мире", "temporality": "Временность"}
        
        return "\n".join([
            "=" * 50, "ХАЙДЕГГЕРОВСКИЙ АНАЛИЗ: СЕМАНТИЧЕСКИЙ РЕАКТОР", "=" * 50, "",
            f"Жест деструкции: {g.get(h.gesture_of_destruction, h.gesture_of_destruction)}",
            f"Фаза реактора: {p.get(h.reactor_phase, h.reactor_phase)}",
            f"Область: {r.get(h.region_of_destruction, h.region_of_destruction)}",
            f"Удержание: {'Да' if h.holding_of_questioning else 'Нет'}",
            f"Уверенность: {h.confidence:.0%}",
            "", "=" * 50,
            json.dumps(h.model_dump(), ensure_ascii=False, indent=2)
        ])
    
    def _format_dostoevsky(self, syntax, d: DostoevskyResult):
        if d.polyphonic_configuration == "null" and d.confidence <= 0.5:
            return "\n".join([
                "ДОСТОЕВСКИЙ: ПОЛИФОНИЧЕСКИЙ АНАЛИЗ", "=" * 50, "",
                "⚠ Модель вернула пустой ответ.",
                "", "=" * 50,
                json.dumps(d.model_dump(), ensure_ascii=False, indent=2)
            ])
        
        c = {"double_voice": "Двойной голос", "mirror_pair": "Зеркальная пара",
             "chorus": "Хор голосов", "internal_dialogue": "Внутренний диалог"}
        o = {"condensation": "Конденсация", "nadryv_vdrug": "Надрыв и «вдруг»",
             "double_self": "Двойник", "odnako_zhe": "«Однако же»"}
        p = {"condensation": "Конденсация", "interference": "Интерференция",
             "detonation": "Детонация", "redistribution": "Перераспределение"}
        
        return "\n".join([
            "=" * 50, "ДОСТОЕВСКИЙ: ПОЛИФОНИЧЕСКИЙ ДЕТОНАТОР", "=" * 50, "",
            f"Конфигурация: {c.get(d.polyphonic_configuration, d.polyphonic_configuration)}",
            f"Оператор: {o.get(d.operator_type, d.operator_type)}",
            f"Фаза: {p.get(d.detonation_phase, d.detonation_phase)}",
            f"Напряжение: {'Да' if d.unresolved_tension else 'Нет'}",
            f"Уверенность: {d.confidence:.0%}",
            "", "=" * 50,
            json.dumps(d.model_dump(), ensure_ascii=False, indent=2)
        ])
    
    # ============================================================
    def _on_clear(self):
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)
        self._status("Готов.")
    
    def _show_result(self, text):
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", text)
        self._status("Анализ завершён.")
        self.btn_analyze.configure(state='normal')
        self.btn_batch.configure(state='normal')
    
    def _show_error(self, msg):
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", f"ОШИБКА:\n{msg}")
        self._status("Ошибка.")
        self.btn_analyze.configure(state='normal')
        self.btn_batch.configure(state='normal')
    
    def _status(self, msg):
        print(f"[GUI] {msg}")
        self.root.after(0, lambda: self.status_var.set(msg))


def main():
    root = tk.Tk()
    GrammaLangGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

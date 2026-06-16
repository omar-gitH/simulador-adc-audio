
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from adc_module import generar_senal, muestrear_y_cuantizar
from audio_module import grabar_audio

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "TP_Integrador_Audio"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_WAV = OUTPUT_DIR / "output.wav"


class TPIntegradorApp:
    def _aplicar_tamano_ventana(self):
        self.root.update_idletasks()

        ancho_pantalla = max(1, self.root.winfo_screenwidth())
        alto_pantalla = max(1, self.root.winfo_screenheight())

        ancho = int(min(1800, max(1180, ancho_pantalla * 0.94)))
        alto = int(min(1080, max(760, alto_pantalla * 0.92)))

        ancho = min(ancho, ancho_pantalla)
        alto = min(alto, alto_pantalla)

        x = max(0, (ancho_pantalla - ancho) // 2)
        y = max(0, (alto_pantalla - alto) // 2)

        self.root.geometry(f"{ancho}x{alto}+{x}+{y}")

    def __init__(self, root):
        # 1. CAMBIO A MODO CLARO Y TEMA AZUL MODERNO
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.root = root
        self.root.title("TP Integrador – Simulador ADC y Conversor de Audio")
        
        # Fondo gris claro para la ventana principal (estilo Tailwind Slate-50)
        self.root.configure(fg_color="#f8fafc")
        self.root.resizable(True, True)
        self.root.minsize(1180, 760)
        self._aplicar_tamano_ventana()

        # 2. CONFIGURACIÓN DE MATPLOTLIB PARA ENTORNO CLARO
        rcParams['font.family'] = 'DejaVu Sans'
        rcParams['axes.titleweight'] = 'bold'
        rcParams['axes.labelcolor'] = '#334155'  # Gris oscuro
        rcParams['xtick.color'] = '#64748b'
        rcParams['ytick.color'] = '#64748b'
        rcParams['text.color'] = '#0f172a'
        rcParams['axes.edgecolor'] = '#e2e8f0'   # Bordes suaves
        rcParams['axes.facecolor'] = '#ffffff'   # Fondo blanco de gráfica
        rcParams['figure.facecolor'] = '#ffffff' # Fondo blanco de figura
        rcParams['savefig.facecolor'] = '#ffffff'

        self.recording = False
        self._build_ui()

    def _build_ui(self):
        # Título principal con contraste oscuro
        title = ctk.CTkLabel(
            self.root,
            text="Propuestas del TP Integrador – Simulador de Señales y Conversor de Audio",
            font=("Segoe UI", 18, "bold"),
            text_color="#0f172a",
        )
        title.pack(pady=(16, 4))

        subtitle = ctk.CTkLabel(
            self.root,
            text="Interfaz unificada inspirada en diseño de paneles limpios y conversión analógico-digital.",
            font=("Segoe UI", 11),
            text_color="#475569",
        )
        subtitle.pack(pady=(0, 14))

        # Estilos personalizados para los componentes nativos de TK/TTK
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#f8fafc', borderwidth=0)
        
        # Configuración de los combobox nativos para que luzcan limpios en fondo blanco
        style.configure('TCombobox', fieldbackground='#ffffff', background='#e2e8f0', foreground='#0f172a')
        style.configure('TEntry', fieldbackground='#ffffff', foreground='#0f172a')
        
        # Estilos de botones de herramientas auxiliares (Pan, Reset)
        style.configure('Modern.TButton', background='#e2e8f0', foreground='#334155', borderwidth=0, padding=[8, 4])
        style.map('Modern.TButton', background=[('active', '#cbd5e1')])

        # Tabview con colores claros y pestañas estilizadas en azul
        notebook = ctk.CTkTabview(
            self.root, 
            corner_radius=12, 
            fg_color="#f1f5f9",
            segmented_button_fg_color="#e2e8f0",
            segmented_button_selected_color="#2563eb",
            segmented_button_selected_hover_color="#1d4ed8",
            segmented_button_unselected_color="#e2e8f0",
            text_color="#ffffff"
        )
        notebook.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        notebook.add("Simulador ADC")
        notebook.add("Conversor de Audio")

        self.adc_frame = notebook.tab("Simulador ADC")
        self.audio_frame = notebook.tab("Conversor de Audio")
        self.adc_frame.configure(fg_color="#f1f5f9")
        self.audio_frame.configure(fg_color="#f1f5f9")

        self._build_adc_tab()
        self._build_audio_tab()

    def _build_adc_tab(self):
        # Panel Izquierdo: Tarjeta blanca limpia con bordes suaves
        left = ctk.CTkFrame(self.adc_frame, corner_radius=12, border_width=1, border_color="#e2e8f0", fg_color="#ffffff")
        left.pack(side="left", fill="y", padx=(12, 8), pady=12)
        
        ctk.CTkLabel(left, text="Parámetros del simulador ADC", font=("Segoe UI", 14, "bold"), text_color="#0f172a").grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(14, 10))

        # Helper para etiquetas consistentes
        def crear_label(master, text, r, c):
            lbl = tk.Label(master, text=text, bg="#ffffff", fg="#334155", font=("Segoe UI", 10))
            lbl.grid(row=r, column=c, sticky="w", padx=14, pady=6)
            return lbl

        crear_label(left, "Tipo de señal", 1, 0)
        self.signal_type = tk.StringVar(value="sinusoidal")
        ttk.Combobox(left, textvariable=self.signal_type, values=["sinusoidal", "cuadrada", "ruido"], state="readonly", width=18).grid(row=1, column=1, padx=14, pady=6)

        crear_label(left, "Frecuencia (Hz)", 2, 0)
        self.frequency = tk.StringVar(value="1000")
        self.frequency_combo = ttk.Combobox(left, textvariable=self.frequency, values=["100", "250", "500", "1000", "2000", "5000"], width=18, state="normal")
        self.frequency_combo.grid(row=2, column=1, padx=14, pady=6)
        # Deshabilitar la selección de frecuencia cuando se elija 'ruido'
        def actualizar_estado_frecuencia(*args):
            try:
                if self.signal_type.get() == "ruido":
                    self.frequency_combo.configure(state="disabled")
                else:
                    self.frequency_combo.configure(state="readonly")
            except Exception:
                pass

        # Vincular la función al cambio de la variable
        try:
            self.signal_type.trace_add("write", actualizar_estado_frecuencia)
        except Exception:
            # Compatibilidad con versiones antiguas de tkinter
            self.signal_type.trace("w", actualizar_estado_frecuencia)

        crear_label(left, "Tasa de muestreo (Hz)", 3, 0)
        self.sample_rate = tk.StringVar(value="44100")
        self.sample_combo = ttk.Combobox(left, textvariable=self.sample_rate, values=["8000", "11025", "22050", "44100"], width=18, state="normal")
        self.sample_combo.grid(row=3, column=1, padx=14, pady=6)

        crear_label(left, "Bits de cuantización", 4, 0)
        self.bits = tk.StringVar(value="8")
        self.bits_combo = ttk.Combobox(left, textvariable=self.bits, values=["4", "8", "16"], width=18, state="normal")
        self.bits_combo.grid(row=4, column=1, padx=14, pady=6)

        # Botones con colores vivos de acento azul (estilo mockup)
        ctk.CTkButton(left, text="Ejecutar Simulación", command=self.generar_adc, fg_color="#2563eb", hover_color="#1d4ed8", text_color="#ffffff", corner_radius=8, height=34).grid(row=5, column=0, columnspan=2, sticky="ew", padx=14, pady=(14, 4))
        ctk.CTkButton(left, text="Demostrar Aliasing", command=self.simular_aliasing, fg_color="#475569", hover_color="#334155", text_color="#ffffff", corner_radius=8, height=34).grid(row=6, column=0, columnspan=2, sticky="ew", padx=14, pady=4)
        ctk.CTkButton(left, text="Limpiar", command=self.limpiar_adc, fg_color="#e2e8f0", hover_color="#cbd5e1", text_color="#334155", corner_radius=8, height=34).grid(row=7, column=0, columnspan=2, sticky="ew", padx=14, pady=4)

        # Caja de logs adaptada al entorno claro
        self.info_adc = tk.Text(left, height=10, width=38, bg="#f8fafc", fg="#334155", relief="flat", highlightthickness=1, highlightbackground="#e2e8f0", font=("Consolas", 9))
        self.info_adc.grid(row=8, column=0, columnspan=2, padx=14, pady=14, sticky="nsew")
        self.info_adc.insert("end", "Resultados de Nyquist, cuantización y análisis de aliasing.\n")

        left.columnconfigure(1, weight=1)

        # Panel Derecho: Gráfica dentro de tarjeta blanca
        right = ctk.CTkFrame(self.adc_frame, corner_radius=12, border_width=1, border_color="#e2e8f0", fg_color="#ffffff")
        right.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=12)
        ctk.CTkLabel(right, text="Señal Original vs. Señal Digitalizada", font=("Segoe UI", 14, "bold"), text_color="#0f172a").pack(anchor="w", padx=16, pady=(14, 0))

        # Controladores de Zoom / Pan integrados sutilmente
        zoom_bar = tk.Frame(right, bg="#ffffff")
        zoom_bar.pack(fill='x', padx=16, pady=(6, 0))
        tk.Label(zoom_bar, text="Mover: ", bg="#ffffff", fg="#64748b").pack(side='left')
        ttk.Button(zoom_bar, text="Pan", command=self.activar_pan_adc, style='Modern.TButton').pack(side='left', padx=(0, 8))
        tk.Label(zoom_bar, text="Zoom: ", bg="#ffffff", fg="#64748b").pack(side='left')
        self.adc_zoom_var = tk.DoubleVar(value=1.0)
        ttk.Scale(zoom_bar, from_=1.0, to=10.0, variable=self.adc_zoom_var, command=self.aplicar_zoom_adc).pack(side='left', fill='x', expand=True, padx=(0, 8))
        ttk.Button(zoom_bar, text="Reset", command=self.reset_zoom_adc, style='Modern.TButton').pack(side='right')

        self.adc_figure = plt.Figure(figsize=(9.6, 6.4), dpi=120)
        self.adc_canvas = FigureCanvasTkAgg(self.adc_figure, master=right)
        self.adc_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self.adc_toolbar = NavigationToolbar2Tk(self.adc_canvas, right, pack_toolbar=False)
        self.adc_toolbar.update()

        self.adc_ax = self.adc_figure.add_subplot(111)
        self.adc_ax.set_xlabel("Tiempo (s)")
        self.adc_ax.set_ylabel("Amplitud")
        self.adc_ax.grid(True, alpha=0.2, color="#94a3b8")
        # Evitar que los cambios de X re-escalen automáticamente el eje Y
        try:
            self.adc_ax.set_autoscaley_on(False)
        except Exception:
            pass
        self.adc_zoom = 1.0
        self.adc_base_xlim = None
        self.adc_base_ylim = None
        self.adc_canvas.draw()

    def _build_audio_tab(self):
        # Panel Izquierdo: Configuración del conversor de audio
        left = ctk.CTkFrame(self.audio_frame, corner_radius=12, border_width=1, border_color="#e2e8f0", fg_color="#ffffff")
        left.pack(side="left", fill="y", padx=(12, 8), pady=12)
        ctk.CTkLabel(left, text="Parámetros del conversor", font=("Segoe UI", 14, "bold"), text_color="#0f172a").grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(14, 10))

        def crear_label_audio(master, text, r, c):
            lbl = tk.Label(master, text=text, bg="#ffffff", fg="#334155", font=("Segoe UI", 10))
            lbl.grid(row=r, column=c, sticky="w", padx=14, pady=6)
            return lbl

        crear_label_audio(left, "Duración (s)", 1, 0)
        self.duration = tk.StringVar(value="3")
        self.duration_entry = tk.Entry(left, textvariable=self.duration, width=16, bg="#ffffff", fg="#0f172a", relief="solid", bd=1)
        self.duration_entry.grid(row=1, column=1, padx=14, pady=6)

        crear_label_audio(left, "Frecuencia (Hz)", 2, 0)
        self.audio_fs = tk.StringVar(value="44100")
        ttk.Combobox(left, textvariable=self.audio_fs, values=["8000", "16000", "22050", "44100"], state="readonly", width=14).grid(row=2, column=1, padx=14, pady=6)

        crear_label_audio(left, "Profundidad", 3, 0)
        self.audio_bits = tk.StringVar(value="16")
        ttk.Combobox(left, textvariable=self.audio_bits, values=["8", "16"], state="readonly", width=14).grid(row=3, column=1, padx=14, pady=6)

        self.audio_record_btn = ctk.CTkButton(left, text="Grabar Audio", command=self.grabar_audio, fg_color="#ef4444", hover_color="#dc2626", text_color="#ffffff", corner_radius=8, height=34)
        self.audio_record_btn.grid(row=4, column=0, columnspan=2, sticky="ew", padx=14, pady=(14, 4))
        
        ctk.CTkButton(left, text="Mostrar Espectro", command=self.mostrar_espectro, fg_color="#2563eb", hover_color="#1d4ed8", text_color="#ffffff", corner_radius=8, height=34).grid(row=5, column=0, columnspan=2, sticky="ew", padx=14, pady=4)
        ctk.CTkButton(left, text="Ver onda original", command=self.mostrar_onda_original, fg_color="#10b981", hover_color="#059669", text_color="#ffffff", corner_radius=8, height=34).grid(row=6, column=0, columnspan=2, sticky="ew", padx=14, pady=4)
        ctk.CTkButton(left, text="Ver Archivo WAV", command=self.ver_wav, fg_color="#e2e8f0", hover_color="#cbd5e1", text_color="#334155", corner_radius=8, height=34).grid(row=7, column=0, columnspan=2, sticky="ew", padx=14, pady=4)

        self.audio_status = tk.StringVar(value="Listo para grabar audio.")
        tk.Label(left, textvariable=self.audio_status, wraplength=240, bg="#ffffff", fg="#2563eb", font=("Segoe UI", 9, "italic")).grid(row=7, column=0, columnspan=2, sticky="w", padx=14, pady=6)

        self.audio_log = tk.Text(left, height=12, width=38, bg="#f8fafc", fg="#334155", relief="flat", highlightthickness=1, highlightbackground="#e2e8f0", font=("Consolas", 9))
        self.audio_log.grid(row=8, column=0, columnspan=2, padx=14, pady=14, sticky="nsew")
        self.audio_log.insert("end", "Registro del proceso de exportación...\n")

        # Panel Derecho: Gráfica de Audio / Espectro
        right = ctk.CTkFrame(self.audio_frame, corner_radius=12, border_width=1, border_color="#e2e8f0", fg_color="#ffffff")
        right.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=12)
        ctk.CTkLabel(right, text="Análisis de Onda del Audio", font=("Segoe UI", 14, "bold"), text_color="#0f172a").pack(anchor="w", padx=16, pady=(14, 0))

        zoom_bar = tk.Frame(right, bg="#ffffff")
        zoom_bar.pack(fill='x', padx=16, pady=(6, 0))
        tk.Label(zoom_bar, text="Mover: ", bg="#ffffff", fg="#64748b").pack(side='left')
        ttk.Button(zoom_bar, text="Pan", command=self.activar_pan_audio, style='Modern.TButton').pack(side='left', padx=(0, 8))
        tk.Label(zoom_bar, text="Zoom: ", bg="#ffffff", fg="#64748b").pack(side='left')
        self.audio_zoom_var = tk.DoubleVar(value=1.0)
        ttk.Scale(zoom_bar, from_=1.0, to=10.0, variable=self.audio_zoom_var, command=self.aplicar_zoom_audio).pack(side='left', fill='x', expand=True, padx=(0, 8))
        # Opción para activar zoom proporcional en X e Y
        self.audio_proportional_var = tk.BooleanVar(value=False)
        tk.Checkbutton(zoom_bar, text="Proporcional Y↕", bg="#ffffff", fg="#334155", variable=self.audio_proportional_var).pack(side='right', padx=(6, 0))
        ttk.Button(zoom_bar, text="Reset", command=self.reset_zoom_audio, style='Modern.TButton').pack(side='right')

        self.audio_figure = plt.Figure(figsize=(9.6, 6.4), dpi=120)
        self.audio_canvas = FigureCanvasTkAgg(self.audio_figure, master=right)
        self.audio_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self.audio_toolbar = NavigationToolbar2Tk(self.audio_canvas, right, pack_toolbar=False)
        self.audio_toolbar.update()

        self.audio_ax = self.audio_figure.add_subplot(111)
        self.audio_ax.set_xlabel("Tiempo / Frecuencia")
        self.audio_ax.set_ylabel("Amplitud")
        self.audio_ax.grid(True, alpha=0.2, color="#94a3b8")
        # Evitar que los cambios de X re-escalen automáticamente el eje Y
        try:
            self.audio_ax.set_autoscaley_on(False)
        except Exception:
            pass
        self.audio_zoom = 1.0
        self.audio_base_xlim = None
        self.audio_base_ylim = None
        self.audio_canvas.draw()

    # --- LÓGICA DE CONTROLADORES AUXILIARES (SE MANTIENE IGUAL) ---
    def activar_pan_adc(self):
        try:
            if hasattr(self, 'adc_toolbar'):
                self.adc_toolbar.pan()
        except Exception:
            pass

    def aplicar_zoom_adc(self, value):
        valor = float(value)
        self.adc_zoom = valor
        if self.adc_base_xlim is None:
            self.adc_base_xlim = self.adc_ax.get_xlim()
        # Asegurarse de tener también el límite vertical base para no reescalar Y
        if self.adc_base_ylim is None:
            self.adc_base_ylim = self.adc_ax.get_ylim()
        width = (self.adc_base_xlim[1] - self.adc_base_xlim[0]) / valor
        x0 = (self.adc_base_xlim[0] + self.adc_base_xlim[1]) / 2 - width / 2
        x1 = x0 + width
        self.adc_ax.set_xlim(x0, x1)
        # Restaurar límites Y base para evitar deformaciones por autoscale
        try:
            self.adc_ax.set_ylim(*self.adc_base_ylim)
        except Exception:
            pass
        self.adc_canvas.draw_idle()

    def reset_zoom_adc(self):
        self.adc_zoom_var.set(1.0)
        self.adc_zoom = 1.0
        if self.adc_base_xlim is not None:
            self.adc_ax.set_xlim(*self.adc_base_xlim)
        self.adc_canvas.draw_idle()

    def activar_pan_audio(self):
        try:
            if hasattr(self, 'audio_toolbar'):
                self.audio_toolbar.pan()
        except Exception:
            pass

    def aplicar_zoom_audio(self, value):
        valor = float(value)
        self.audio_zoom = valor
        # Asegurar límites base X/Y
        if self.audio_base_xlim is None:
            self.audio_base_xlim = self.audio_ax.get_xlim()
        if self.audio_base_ylim is None:
            self.audio_base_ylim = self.audio_ax.get_ylim()

        # Si el usuario solicita zoom proporcional, escalamos ambos ejes alrededor del centro
        if getattr(self, 'audio_proportional_var', None) and self.audio_proportional_var.get():
            base_x0, base_x1 = self.audio_base_xlim
            base_y0, base_y1 = self.audio_base_ylim
            width = (base_x1 - base_x0) / valor
            height = (base_y1 - base_y0) / valor
            x0 = (base_x0 + base_x1) / 2 - width / 2
            x1 = x0 + width
            y0 = (base_y0 + base_y1) / 2 - height / 2
            y1 = y0 + height
            self.audio_ax.set_xlim(x0, x1)
            try:
                self.audio_ax.set_ylim(y0, y1)
            except Exception:
                pass
        else:
            # Comportamiento tradicional: zoom en X y mantener Y fija
            width = (self.audio_base_xlim[1] - self.audio_base_xlim[0]) / valor
            x0 = (self.audio_base_xlim[0] + self.audio_base_xlim[1]) / 2 - width / 2
            x1 = x0 + width
            self.audio_ax.set_xlim(x0, x1)
            try:
                self.audio_ax.set_ylim(*self.audio_base_ylim)
            except Exception:
                pass

        self.audio_canvas.draw_idle()

    def reset_zoom_audio(self):
        self.audio_zoom_var.set(1.0)
        self.audio_zoom = 1.0
        if self.audio_base_xlim is not None:
            self.audio_ax.set_xlim(*self.audio_base_xlim)
        self.audio_canvas.draw_idle()

    def _append_adc_info(self, text):
        self.info_adc.insert("end", text + "\n")
        self.info_adc.see("end")

    def _append_audio_log(self, text):
        self.audio_log.insert("end", text + "\n")
        self.audio_log.see("end")

    def generar_adc(self):
        try:
            frec = float(self.frequency.get())
            fs = int(self.sample_rate.get())
            bits = int(self.bits.get())
            tipo = self.signal_type.get()

            t, señal = generar_senal(tipo=tipo, frecuencia=frec, duracion=0.01, fs=fs)
            digital = muestrear_y_cuantizar(señal, fs=fs, bits=bits)

            nyquist = fs / 2.0
            alias = "Sí" if frec > nyquist else "No"
            error = float(np.mean(np.abs(señal - digital)))

            self.adc_ax.clear()
            # Colores limpios y brillantes para modo claro
            self.adc_ax.plot(t, señal, label="Analógica", linewidth=2, color="#2563eb")
            self.adc_ax.step(t, digital, where="mid", label="Digitalizada", linewidth=2, color="#ef4444")
            self.adc_ax.set_title(f"Simulación ADC — {tipo.capitalize()} @ {frec:.0f} Hz", color="#0f172a")
            self.adc_ax.set_xlabel("Tiempo (s)")
            self.adc_ax.set_ylabel("Amplitud")
            self.adc_ax.legend(facecolor='#ffffff', edgecolor='#e2e8f0')
            self.adc_ax.grid(True, alpha=0.2, color="#94a3b8")
            self.adc_base_xlim = tuple(self.adc_ax.get_xlim())
            self.adc_base_ylim = tuple(self.adc_ax.get_ylim())
            self.adc_zoom_var.set(1.0)
            self.adc_zoom = 1.0
            self.adc_canvas.draw()

            self.info_adc.delete("1.0", "end")
            self.info_adc.insert("end", f"Tipo de señal: {tipo}\n")
            self.info_adc.insert("end", f"Frecuencia: {frec:.2f} Hz\n")
            self.info_adc.insert("end", f"Frecuencia de muestreo: {fs} Hz\n")
            self.info_adc.insert("end", f"Bits de cuantización: {bits}\n")
            self.info_adc.insert("end", f"Límite de Nyquist: {nyquist:.2f} Hz\n")
            self.info_adc.insert("end", f"¿Puede aparecer aliasing?: {alias}\n")
            self.info_adc.insert("end", f"Error medio de cuantización: {error:.4f}\n")

        except Exception as exc:
            messagebox.showerror("Error de simulación", f"No se pudo generar la señal: {exc}")

    def simular_aliasing(self):
        try:
            fs = int(self.sample_rate.get())
            frec_base = float(self.frequency.get()) if self.frequency.get() else 1000.0
            bits = int(self.bits.get())
            tipo = self.signal_type.get()

            frec_demo = frec_base if frec_base > fs / 2 else fs * 0.75
            alias_freq = abs(frec_demo % fs)
            if alias_freq > fs / 2:
                alias_freq = fs - alias_freq

            t, señal = generar_senal(tipo=tipo, frecuencia=frec_demo, duracion=0.02, fs=fs)
            digital = muestrear_y_cuantizar(señal, fs=fs, bits=bits)

            self.adc_ax.clear()
            self.adc_ax.plot(t, señal, label=f"Analógica ({frec_demo:.1f} Hz)", linewidth=2, color="#2563eb")
            self.adc_ax.step(t, digital, where="mid", label="Digitalizada", linewidth=2, color="#ef4444")
            self.adc_ax.set_title("Demostración de aliasing: señal por encima de Nyquist", color="#0f172a")
            self.adc_ax.set_xlabel("Tiempo (s)")
            self.adc_ax.set_ylabel("Amplitud")
            self.adc_ax.legend(facecolor='#ffffff', edgecolor='#e2e8f0')
            self.adc_ax.grid(True, alpha=0.2, color="#94a3b8")
            self.adc_base_xlim = tuple(self.adc_ax.get_xlim())
            self.adc_base_ylim = tuple(self.adc_ax.get_ylim())
            self.adc_zoom_var.set(1.0)
            self.adc_zoom = 1.0
            self.adc_canvas.draw()

            self.info_adc.delete("1.0", "end")
            self.info_adc.insert("end", f"Frecuencia real usada: {frec_demo:.1f} Hz\n")
            self.info_adc.insert("end", f"Límite de Nyquist: {fs / 2:.1f} Hz\n")
            self.info_adc.insert("end", f"Frecuencia aparente (alias): {alias_freq:.1f} Hz\n")
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo simular aliasing: {exc}")

    def limpiar_adc(self):
        self.adc_ax.clear()
        self.adc_ax.set_title("Señal analógica vs. señal digitalizada", color="#0f172a")
        self.adc_ax.grid(True, alpha=0.2, color="#94a3b8")
        self.adc_canvas.draw()
        self.info_adc.delete("1.0", "end")
        self.info_adc.insert("end", "Panel reiniciado. Elija parámetros.\n")

    def grabar_audio(self):
        try:
            duracion = float(self.duration.get())
            fs = int(self.audio_fs.get())
            bits = int(self.audio_bits.get())
            if duracion <= 0:
                raise ValueError("La duración debe ser mayor que cero.")

            self.recording = True
            self.audio_status.set("🎤 Grabando… espere.")
            self._append_audio_log(f"Inicio de captura: {duracion:.1f} s, {fs} Hz")
            self._animar_grabacion()
            thread = threading.Thread(target=self._grabar_audio_thread, args=(duracion, fs, bits), daemon=True)
            thread.start()
        except Exception as exc:
            messagebox.showerror("Error de grabación", f"Datos inválidos: {exc}")

    def _animar_grabacion(self, frame=0):
        if self.recording:
            frames = ["🎤 Grabando", "🎤 Grabando.", "🎤 Grabando..", "🎤 Grabando..."]
            self.audio_record_btn.configure(text=frames[frame % 4])
            self.root.after(500, lambda: self._animar_grabacion((frame + 1) % 4))

    def _grabar_audio_thread(self, duracion, fs, bits):
        try:
            audio, fs_out = grabar_audio(duracion=duracion, fs=fs, bits=bits, filename=str(OUTPUT_WAV))
            audio_f = audio.astype(np.float64)
            rms = np.sqrt(np.mean(audio_f ** 2))
            peak = np.max(np.abs(audio_f))
            self.root.after(0, lambda: self._finalizar_audio(audio, fs_out, rms, peak, bits))
        except Exception as exc:
            self.root.after(0, lambda: self._error_audio(str(exc)))

    def _finalizar_audio(self, audio, fs, rms, peak, bits):
        self.recording = False
        self.audio_record_btn.configure(text="Grabar Audio", fg_color="#ef4444")
        self.audio_status.set("Grabación finalizada con éxito.")
        self._append_audio_log(f"Muestras: {len(audio)} | RMS: {rms:.4f}")

        self.audio_ax.clear()
        t = np.arange(len(audio)) / fs
        # Guardar la última captura para poder re-dibujar la onda original
        try:
            self.last_audio = audio.copy()
        except Exception:
            self.last_audio = np.array(audio)
        self.last_audio_fs = fs

        self.audio_ax.plot(t, audio[:, 0] if audio.ndim > 1 else audio, linewidth=1, color="#2563eb")
        self.audio_ax.set_title("Forma de onda capturada", color="#0f172a")
        self.audio_ax.grid(True, alpha=0.2, color="#94a3b8")
        # Guardar límites X/Y base para permitir resetear volumen y escala
        self.audio_base_xlim = tuple(self.audio_ax.get_xlim())
        self.audio_base_ylim = tuple(self.audio_ax.get_ylim())
        self.audio_canvas.draw()

    def _error_audio(self, mensaje):
        self.recording = False
        self.audio_record_btn.configure(text="Grabar Audio", fg_color="#ef4444")
        self.audio_status.set("No fue posible grabar.")
        self._append_audio_log("Error: " + mensaje)

    def mostrar_espectro(self):
        if not OUTPUT_WAV.exists():
            messagebox.showinfo("Sin audio", "Primero grabe una muestra.")
            return

        import wave
        with wave.open(str(OUTPUT_WAV), 'rb') as wav:
            fs = wav.getframerate()
            data = wav.readframes(wav.getnframes())
            audio = np.frombuffer(data, dtype=np.int16)

        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(fft), 1 / fs)
        magnitude = np.abs(fft)[:len(freqs) // 2]
        x = freqs[:len(freqs) // 2]

        self.audio_ax.clear()
        # Gráfica de espectro rellena azul suave (estilo la imagen mockup 2)
        self.audio_ax.fill_between(x, magnitude, color="#3b82f6", alpha=0.6)
        self.audio_ax.plot(x, magnitude, color="#1d4ed8", linewidth=1.2)
        self.audio_ax.set_title("Espectro de frecuencia del audio", color="#0f172a")
        self.audio_ax.set_xlabel("Frecuencia (Hz)")
        self.audio_ax.set_ylabel("Magnitud")
        self.audio_ax.grid(True, alpha=0.2, color="#94a3b8")
        # Guardar límites base del espectro para poder restaurarlos
        self.audio_base_xlim = tuple(self.audio_ax.get_xlim())
        self.audio_base_ylim = tuple(self.audio_ax.get_ylim())
        self.audio_canvas.draw()

    def ver_wav(self):
        if OUTPUT_WAV.exists():
            messagebox.showinfo("Archivo WAV", f"Ubicación:\n{OUTPUT_WAV.resolve()}")
        else:
            messagebox.showinfo("Sin archivo", "Aún no existe una captura.")

    def mostrar_onda_original(self):
        # Re-dibuja la última onda capturada (desde memoria o desde el WAV)
        audio = None
        fs = None
        if hasattr(self, 'last_audio') and getattr(self, 'last_audio') is not None:
            audio = self.last_audio
            fs = getattr(self, 'last_audio_fs', None)
        elif OUTPUT_WAV.exists():
            try:
                import wave
                with wave.open(str(OUTPUT_WAV), 'rb') as wav:
                    fs = wav.getframerate()
                    data = wav.readframes(wav.getnframes())
                    audio = np.frombuffer(data, dtype=np.int16)
            except Exception as exc:
                messagebox.showerror("Error", f"No se pudo leer el archivo WAV: {exc}")
                return
        else:
            messagebox.showinfo("Sin audio", "No hay audio grabado para mostrar la onda original.")
            return

        # Dibujo de la onda
        try:
            self.audio_ax.clear()
            t = np.arange(len(audio)) / fs
            # Manejar audio estéreo si aplica
            y = audio[:, 0] if getattr(audio, 'ndim', 1) > 1 else audio
            self.audio_ax.plot(t, y, linewidth=1, color="#2563eb")
            self.audio_ax.set_title("Forma de onda original", color="#0f172a")
            self.audio_ax.set_xlabel("Tiempo (s)")
            self.audio_ax.set_ylabel("Amplitud")
            self.audio_ax.grid(True, alpha=0.2, color="#94a3b8")
            # Guardar límites base y actualizar lienzo
            self.audio_base_xlim = tuple(self.audio_ax.get_xlim())
            self.audio_base_ylim = tuple(self.audio_ax.get_ylim())
            self.audio_canvas.draw()
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo mostrar la onda original: {exc}")


if __name__ == "__main__":
    root = ctk.CTk()
    TPIntegradorApp(root)
    root.mainloop()
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.io.wavfile import write

import audio_module

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "TP_Integrador_Audio"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_WAV = OUTPUT_DIR / "output_convertido.wav"


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
        # 1. MODO CLARO Y TEMA AZUL MODERNO
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.root = root
        self.root.title("TP Integrador – Conversor y Analizador de Audio")
        
        # Fondo gris claro para la ventana principal (estilo Tailwind Slate-50)
        self.root.configure(fg_color="#f8fafc")
        self.root.resizable(True, True)
        self.root.minsize(1180, 760)
        
        # Variables de estado para el audio original (Fiel/Nativo)
        self.audio_alta_fidelidad = None
        self.fs_nativa = 44100
        
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
            text="Propuestas del TP Integrador – Conversor Digital de Audio",
            font=("Segoe UI", 18, "bold"),
            text_color="#0f172a",
        )
        title.pack(pady=(16, 4))

        subtitle = ctk.CTkLabel(
            self.root,
            text="Interfaz para grabación en tiempo real, reducción de resolución digital y análisis de Nyquist/Aliasing.",
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

        # Contenedor principal con diseño de panel limpio
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=12, fg_color="#f1f5f9")
        self.main_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self._build_audio_tab()

    def _build_audio_tab(self):
        # Panel Izquierdo: Configuración del conversor de audio
        left = ctk.CTkFrame(self.main_frame, corner_radius=12, border_width=1, border_color="#e2e8f0", fg_color="#ffffff")
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

        crear_label_audio(left, "Tasa Destino (Hz)", 2, 0)
        self.audio_fs = tk.StringVar(value="8000")
        ttk.Combobox(left, textvariable=self.audio_fs, values=["8000", "16000", "22050", "44100"], state="readonly", width=14).grid(row=2, column=1, padx=14, pady=6)

        crear_label_audio(left, "Profundidad Destino", 3, 0)
        self.audio_bits = tk.StringVar(value="8")
        ttk.Combobox(left, textvariable=self.audio_bits, values=["8", "16"], state="readonly", width=14).grid(row=3, column=1, padx=14, pady=6)

        # --- BLOQUE DE BOTONES ORDENADO ---
        
        # Fila 4: Botón de grabación
        self.audio_record_btn = ctk.CTkButton(left, text="Grabar Audio", command=self.grabar_audio, fg_color="#ef4444", hover_color="#dc2626", text_color="#ffffff", corner_radius=8, height=34)
        self.audio_record_btn.grid(row=4, column=0, columnspan=2, sticky="ew", padx=14, pady=(14, 4))
        
        # Fila 5 y 6: Comparaciones simples (1 a 1)
        ctk.CTkButton(left, text="📉 Comparar Ondas (Tiempo)", command=self.mostrar_onda_original, fg_color="#10b981", hover_color="#059669", text_color="#ffffff", corner_radius=8, height=34).grid(row=5, column=0, columnspan=2, sticky="ew", padx=14, pady=4)
        
        ctk.CTkButton(left, text="📊 Comparar Espectros (Freq)", command=self.mostrar_espectro, fg_color="#2563eb", hover_color="#1d4ed8", text_color="#ffffff", corner_radius=8, height=34).grid(row=6, column=0, columnspan=2, sticky="ew", padx=14, pady=4)
        
        # Fila 7 y 8: Comparaciones múltiples (Nuevas funciones)
        ctk.CTkButton(left, text="🌈 Múltiples Espectros (Freq)", command=self.mostrar_multiples_espectros, fg_color="#8b5cf6", hover_color="#7c3aed", text_color="#ffffff", corner_radius=8, height=34).grid(row=7, column=0, columnspan=2, sticky="ew", padx=14, pady=4)
        
        ctk.CTkButton(left, text="📈 Múltiples Ondas (Tiempo)", command=self.mostrar_multiples_ondas, fg_color="#f59e0b", hover_color="#d97706", text_color="#ffffff", corner_radius=8, height=34).grid(row=8, column=0, columnspan=2, sticky="ew", padx=14, pady=4)

        # Fila 9: Ver archivo
        ctk.CTkButton(left, text="Ver Archivo WAV", command=self.ver_wav, fg_color="#e2e8f0", hover_color="#cbd5e1", text_color="#334155", corner_radius=8, height=34).grid(row=9, column=0, columnspan=2, sticky="ew", padx=14, pady=4)

        # Fila 10: Estado
        self.audio_status = tk.StringVar(value="Listo para grabar audio.")
        tk.Label(left, textvariable=self.audio_status, wraplength=240, bg="#ffffff", fg="#2563eb", font=("Segoe UI", 9, "italic")).grid(row=10, column=0, columnspan=2, sticky="w", padx=14, pady=6)

        # Fila 11: Consola/Log
        self.audio_log = tk.Text(left, height=12, width=38, bg="#f8fafc", fg="#334155", relief="flat", highlightthickness=1, highlightbackground="#e2e8f0", font=("Consolas", 9))
        self.audio_log.grid(row=11, column=0, columnspan=2, padx=14, pady=14, sticky="nsew")
        self.audio_log.insert("end", "Registro del proceso de exportación y remuestreo...\n")

        # Panel Derecho: Gráfica de Audio / Espectro
        right = ctk.CTkFrame(self.main_frame, corner_radius=12, border_width=1, border_color="#e2e8f0", fg_color="#ffffff")
        right.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=12)
        ctk.CTkLabel(right, text="Análisis de Conversión de Audio", font=("Segoe UI", 14, "bold"), text_color="#0f172a").pack(anchor="w", padx=16, pady=(14, 0))

        zoom_bar = tk.Frame(right, bg="#ffffff")
        zoom_bar.pack(fill='x', padx=16, pady=(6, 0))
        tk.Label(zoom_bar, text="Mover: ", bg="#ffffff", fg="#64748b").pack(side='left')
        ttk.Button(zoom_bar, text="Pan", command=self.activar_pan_audio, style='Modern.TButton').pack(side='left', padx=(0, 8))
        tk.Label(zoom_bar, text="Zoom: ", bg="#ffffff", fg="#64748b").pack(side='left')
        self.audio_zoom_var = tk.DoubleVar(value=1.0)
        ttk.Scale(zoom_bar, from_=1.0, to=10.0, variable=self.audio_zoom_var, command=self.aplicar_zoom_audio).pack(side='left', fill='x', expand=True, padx=(0, 8))
        
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
        
        try:
            self.audio_ax.set_autoscaley_on(False)
        except Exception:
            pass
        self.audio_zoom = 1.0
        self.audio_base_xlim = None
        self.audio_base_ylim = None
        self.audio_canvas.draw()

    def activar_pan_audio(self):
        try:
            if hasattr(self, 'audio_toolbar'):
                self.audio_toolbar.pan()
        except Exception:
            pass

    def aplicar_zoom_audio(self, value):
        valor = float(value)
        self.audio_zoom = valor
        # Asegurarse de tener límites base válidos (recalcular si no existen)
        if self.audio_base_xlim is None:
            try:
                self.audio_ax.relim()
                self.audio_ax.autoscale_view()
            except Exception:
                pass
            self.audio_base_xlim = self.audio_ax.get_xlim()
        if self.audio_base_ylim is None:
            try:
                self.audio_ax.relim()
                self.audio_ax.autoscale_view()
            except Exception:
                pass
            self.audio_base_ylim = self.audio_ax.get_ylim()

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
        if self.audio_base_ylim is not None:
            self.audio_ax.set_ylim(*self.audio_base_ylim)
        self.audio_canvas.draw_idle()

    def _append_audio_log(self, text):
        self.audio_log.insert("end", text + "\n")
        self.audio_log.see("end")


    


    def _animar_grabacion(self, frame=0):
        if self.recording:
            frames = ["🎤 Grabando", "🎤 Grabando.", "🎤 Grabando..", "🎤 Grabando..."]
            self.audio_record_btn.configure(text=frames[frame % 4])
            self.root.after(500, lambda: self._animar_grabacion((frame + 1) % 4))

    def _grabar_audio_thread(self, duracion, fs):
        try:
            # Graba a calidad nativa sin reducir bits ni hertz todavía
            audio, fs_out = audio_module.grabar_audio(duracion=duracion, fs=fs)
            self.audio_alta_fidelidad = audio
            
            audio_f = audio.astype(np.float64)
            rms = np.sqrt(np.mean(audio_f ** 2))
            
            self.root.after(0, lambda: self._finalizar_audio(rms))
        except Exception as exc:
            self.root.after(0, lambda: self._error_audio(str(exc)))

    def _finalizar_audio(self, rms):
        self.recording = False
        self.audio_record_btn.configure(text="Grabar Audio", fg_color="#ef4444")
        self.audio_status.set("Audio en memoria. Configure destino y analice.")
        self._append_audio_log(f"Captura finalizada exitosamente en memoria. RMS: {rms:.4f}")
        
        # Muestra por defecto la onda recién capturada
        self.mostrar_onda_original()

    def _error_audio(self, mensaje):
        self.recording = False
        self.audio_record_btn.configure(text="Grabar Audio", fg_color="#ef4444")
        self.audio_status.set("No fue posible grabar.")
        self._append_audio_log("Error: " + mensaje)


    def grabar_audio(self):
        try:
            duracion = float(self.duration.get())
            if duracion <= 0:
                raise ValueError("La duración debe ser mayor que cero.")

            # Capturamos los valores seleccionados en la interfaz
            freq_elegida = self.audio_fs.get()
            bits_elegidos = self.audio_bits.get()

            self.recording = True
            self.audio_status.set("🎤 Grabando… hable ahora.")
            
            # Creamos el mensaje dinámico
            mensaje = f"Grabando {duracion:.1f}s | Maestra: {self.fs_nativa}Hz -> Destino: {freq_elegida}Hz, {bits_elegidos} bits"
            
            # Mostramos en log y terminal
            self._append_audio_log(mensaje)
            print(f"[I/O HANDLER] {mensaje}")
            
            self._animar_grabacion()
            
            # Se graba de forma nativa a máxima calidad fija
            thread = threading.Thread(target=self._grabar_audio_thread, args=(duracion, self.fs_nativa), daemon=True)
            thread.start()
            
        except Exception as exc:
            messagebox.showerror("Error de grabación", f"Datos inválidos: {exc}")

    def procesar_audio_seleccionado(self):
        """
        [CAPA LÓGICA] Aplica las reducciones de resolución configuradas.
        
        Procesa el audio grabado aplicando:
        1. Remuestreo digital (diezmado a fs_destino)
        2. Cuantización digital (reducción a bits_destino)
        3. Guardado en WAV
        
        Retorna:
            tuple: (audio_convertido, fs_destino, bits_destino)
        """
        if self.audio_alta_fidelidad is None:
            return None, None, None
            
        fs_destino = int(self.audio_fs.get())
        bits_destino = int(self.audio_bits.get())
        
        # 1. Remuestreo Digital (Core Engine)
        audio_remuestreado = audio_module.remuestrear_audio(
            self.audio_alta_fidelidad, self.fs_nativa, fs_destino
        )
        
        # 2. Cuantización Digital (Core Engine)
        audio_convertido = audio_module.cuantizar_audio(audio_remuestreado, bits_destino)
        
        # 3. Persistencia física (I/O Handler)
        write(str(OUTPUT_WAV), fs_destino, audio_convertido)
        
        # Log de Nyquist
        info_nyquist = audio_module.detectar_aliasing_potencial(
            self.fs_nativa, fs_destino
        )
        self._append_audio_log(
            f"Nyquist: {info_nyquist['nyquist_destino']:.0f}Hz "
            f"(factor: {info_nyquist['relacion']:.3f})"
        )
        
        return audio_convertido, fs_destino, bits_destino

    def mostrar_onda_original(self):
        if self.audio_alta_fidelidad is None:
            messagebox.showinfo("Sin audio", "Primero grabe una muestra de audio.")
            return
        self.audio_base_xlim = None
        self.audio_base_ylim = None
        audio_conv, fs_conv, bits_conv = self.procesar_audio_seleccionado()
        self._append_audio_log(f"Procesando Tiempo -> Destino: {fs_conv}Hz, {bits_conv} bits.")

        # Limpiar eje y volver a dibujar los datos recalculando escalas
        self.audio_ax.clear()

        # Vectores de tiempo independientes
        t_orig = np.arange(len(self.audio_alta_fidelidad), dtype=float) / float(self.fs_nativa)
        t_conv = np.arange(len(audio_conv), dtype=float) / float(fs_conv)
        """
        # Ventana de tiempo (Zoom dinámico de 50ms para apreciar los escalones)
        max_tiempo = 0.05
        muestras_orig = max(1, int(max_tiempo * self.fs_nativa))
        muestras_conv = max(1, int(max_tiempo * fs_conv))

        # Gráfica original en azul suave (usar float para evitar artefactos por enteros)
        self.audio_ax.plot(t_orig[:muestras_orig], self.audio_alta_fidelidad[:muestras_orig].astype(float),
                   label="Original (44.1 kHz, 16 bits)", color="#2563eb", alpha=0.5, linewidth=1.5)

        # Gráfica digitalizada escalonada (Simulación ADC) en rojo
        self.audio_ax.step(t_conv[:muestras_conv], audio_conv[:muestras_conv].astype(float), where="mid",
                   label=f"Convertido ({fs_conv} Hz, {bits_conv} bits)", color="#ef4444", linewidth=1.5)
        """
        # --- CÓDIGO CORREGIDO ---
        # Gráfica digitalizada escalonada en rojo (Se dibuja primero, en el fondo, y semi-transparente)
        # Gráfica digitalizada escalonada en rojo (Dibuja la convertida primero, semi-transparente)
        self.audio_ax.step(t_conv, audio_conv.astype(float), where="mid",
                           label=f"Convertido ({fs_conv} Hz, {bits_conv} bits)", 
                           color="#640D0D", linewidth=1.5, alpha=0.6, zorder=2)

        # Gráfica original en azul suave (Dibuja la original al frente con zorder=3, pero MUY traslúcida)
        self.audio_ax.plot(t_orig, self.audio_alta_fidelidad.astype(float),
                           label="Original (44.1 kHz, 16 bits)", 
                           color="#504E73", linewidth=1.5, alpha=0.5, zorder=3)
        self.audio_ax.set_title("Comparación en el Tiempo (Zoom 50ms)", color="#0f172a")
        self.audio_ax.set_xlabel("Tiempo (s)")
        self.audio_ax.set_ylabel("Amplitud Int16")
        self.audio_ax.legend(facecolor='#ffffff', edgecolor='#e2e8f0')
        self.audio_ax.grid(True, alpha=0.2, color="#94a3b8")
        
        # Recalcular límites según los datos actuales para evitar que queden obsoletos
        try:
            self.audio_ax.relim()
            self.audio_ax.autoscale_view()
        except Exception:
            pass

        self.audio_base_xlim = tuple(self.audio_ax.get_xlim())
        self.audio_base_ylim = tuple(self.audio_ax.get_ylim())
        self.audio_zoom_var.set(1.0)
        self.audio_canvas.draw()

    def mostrar_espectro(self):
        if self.audio_alta_fidelidad is None:
            messagebox.showinfo("Sin audio", "Primero grabe una muestra de audio.")
            return
        self.audio_base_xlim = None
        self.audio_base_ylim = None

        audio_conv, fs_conv, bits_conv = self.procesar_audio_seleccionado()
        self._append_audio_log(f"Procesando Espectro -> Nyquist Destino: {fs_conv/2} Hz.")
        
        # Limpiar eje y dibujar espectros, luego recalcular escalas
        self.audio_ax.clear()

        # FFT de la señal original
        fft_orig = np.fft.fft(self.audio_alta_fidelidad.astype(float))
        freqs_orig = np.fft.fftfreq(len(fft_orig), 1.0 / float(self.fs_nativa))
        mitad_orig = len(freqs_orig) // 2
        magnitude_orig = np.abs(fft_orig)[:mitad_orig]

        # FFT de la señal remuestreada y cuantizada
        fft_conv = np.fft.fft(audio_conv.astype(float))
        freqs_conv = np.fft.fftfreq(len(fft_conv), 1.0 / float(fs_conv))
        mitad_conv = len(freqs_conv) // 2
        magnitude_conv = np.abs(fft_conv)[:mitad_conv]

        # Espectro original relleno de azul traslúcido
        self.audio_ax.fill_between(freqs_orig[:mitad_orig], magnitude_orig, color="#3b82f6", alpha=0.3, label="Espectro Fiel (44.1kHz)")
        self.audio_ax.plot(freqs_orig[:mitad_orig], magnitude_orig, color="#1d4ed8", linewidth=1)

        # Espectro Convertido superpuesto en contorno rojo (muestra el Aliasing si se pasa de Nyquist)
        self.audio_ax.plot(freqs_conv[:mitad_conv], magnitude_conv, color="#ef4444", alpha=0.8, linewidth=1.5,
                   label=f"Espectro Convertido ({fs_conv} Hz)")
        
        self.audio_ax.set_title("Comparación de Espectros de Frecuencia (Aliasing)", color="#0f172a")
        self.audio_ax.set_xlabel("Frecuencia (Hz)")
        self.audio_ax.set_ylabel("Magnitud / Densidad")
        self.audio_ax.legend(facecolor='#ffffff', edgecolor='#e2e8f0')
        self.audio_ax.grid(True, alpha=0.2, color="#94a3b8")
        
        try:
            self.audio_ax.relim()
            self.audio_ax.autoscale_view()
        except Exception:
            pass

        self.audio_base_xlim = tuple(self.audio_ax.get_xlim())
        self.audio_base_ylim = tuple(self.audio_ax.get_ylim())
        self.audio_zoom_var.set(1.0)
        self.audio_canvas.draw()

    def ver_wav(self):
        if OUTPUT_WAV.exists():
            messagebox.showinfo("Archivo WAV Convertido", f"Ubicación:\n{OUTPUT_WAV.resolve()}")
        else:
            messagebox.showinfo("Sin archivo", "Aún no has procesado ninguna conversión.")
    
    def mostrar_multiples_espectros(self):
        if self.audio_alta_fidelidad is None:
            messagebox.showinfo("Sin audio", "Primero grabe una muestra de audio.")
            return

        self.audio_base_xlim = None
        self.audio_base_ylim = None
        self.audio_ax.clear()

        # 1. Graficar el espectro ORIGINAL como fondo sólido gris/azulado
        fft_orig = np.fft.fft(self.audio_alta_fidelidad.astype(float))
        freqs_orig = np.fft.fftfreq(len(fft_orig), 1.0 / float(self.fs_nativa))
        mitad_orig = len(freqs_orig) // 2
        magnitude_orig = np.abs(fft_orig)[:mitad_orig]
        
        self.audio_ax.fill_between(freqs_orig[:mitad_orig], magnitude_orig, color="#e2e8f0", alpha=0.7, label=f"Original ({self.fs_nativa} Hz)")

        # 2. Definir las frecuencias a comparar y sus colores
        tasas_destino = [22050, 16000, 8000]
        colores = ["#3b82f6", "#10b981", "#ef4444"] # Azul, Verde, Rojo

        # 3. Iterar, remuestrear y graficar cada una
        bits_destino = int(self.audio_bits.get()) # Mantenemos los bits seleccionados en la UI

        for fs_dest, color in zip(tasas_destino, colores):
            # Remuestrear usando tu módulo
            audio_rem = audio_module.remuestrear_audio(self.audio_alta_fidelidad, self.fs_nativa, fs_dest)
            audio_conv = audio_module.cuantizar_audio(audio_rem, bits_destino)

            # Calcular FFT
            fft_conv = np.fft.fft(audio_conv.astype(float))
            freqs_conv = np.fft.fftfreq(len(fft_conv), 1.0 / float(fs_dest))
            mitad_conv = len(freqs_conv) // 2
            magnitude_conv = np.abs(fft_conv)[:mitad_conv]

            # Graficar solo el contorno para que se vean las demás
            self.audio_ax.plot(freqs_conv[:mitad_conv], magnitude_conv, color=color, alpha=0.9, linewidth=1.5, label=f"Destino ({fs_dest} Hz)")

        # 4. Configuración visual de la gráfica
        self.audio_ax.set_title("Comparación Simultánea de Frecuencias de Muestreo", color="#0f172a")
        self.audio_ax.set_xlabel("Frecuencia (Hz)")
        self.audio_ax.set_ylabel("Magnitud / Densidad")
        self.audio_ax.legend(facecolor='#ffffff', edgecolor='#e2e8f0', loc='upper right')
        self.audio_ax.grid(True, alpha=0.2, color="#94a3b8")

        # Ajuste dinámico de la vista
        try:
            self.audio_ax.relim()
            self.audio_ax.autoscale_view()
            # Opcional: limitar el eje X hasta los 22kHz para no ver el espacio vacío
            self.audio_ax.set_xlim(0, self.fs_nativa / 2) 
        except Exception:
            pass

        self.audio_base_xlim = tuple(self.audio_ax.get_xlim())
        self.audio_base_ylim = tuple(self.audio_ax.get_ylim())
        self.audio_zoom_var.set(1.0)
        self.audio_canvas.draw()
        
        self._append_audio_log("Análisis múltiple de frecuencias completado.")

    def mostrar_multiples_ondas(self):
        if self.audio_alta_fidelidad is None:
            messagebox.showinfo("Sin audio", "Primero grabe una muestra de audio.")
            return

        self.audio_base_xlim = None
        self.audio_base_ylim = None
        self.audio_ax.clear()

        # 1. Vectores de tiempo y datos para la señal original
        t_orig = np.arange(len(self.audio_alta_fidelidad), dtype=float) / float(self.fs_nativa)
        self.audio_ax.plot(t_orig, self.audio_alta_fidelidad.astype(float),
                           label=f"Original ({self.fs_nativa} Hz)", color="#64748b", alpha=0.4, linewidth=2)

        # 2. Configurar las frecuencias a comparar, colores y estilos
        tasas_destino = [22050, 16000, 8000]
        colores = ["#3b82f6", "#10b981", "#ef4444"] # Azul, Verde, Rojo
        bits_destino = int(self.audio_bits.get())

        for fs_dest, color in zip(tasas_destino, colores):
            # Remuestrear y cuantizar usando tu módulo de audio
            audio_rem = audio_module.remuestrear_audio(self.audio_alta_fidelidad, self.fs_nativa, fs_dest)
            audio_conv = audio_module.cuantizar_audio(audio_rem, bits_destino)
            
            # Crear su propio vector de tiempo basado en su frecuencia
            t_conv = np.arange(len(audio_conv), dtype=float) / float(fs_dest)
            
            # Graficar con .step para simular el retenedor de orden cero (conversión digital)
            self.audio_ax.step(t_conv, audio_conv.astype(float), where="mid",
                               label=f"{fs_dest} Hz ({bits_destino} bits)", color=color, linewidth=1.5, alpha=0.85)

        # 3. AUTO-ZOOM AL PICO MÁS ALTO (Crucial para ver los escalones planos)
        indice_pico = np.argmax(np.abs(self.audio_alta_fidelidad))
        tiempo_pico = indice_pico / float(self.fs_nativa)
        
        # Ventana de 8 milisegundos (4ms a la izquierda, 4ms a la derecha)
        margen = 0.004 
        self.audio_ax.set_xlim(max(0, tiempo_pico - margen), tiempo_pico + margen)

        # 4. Ajustes estéticos de la gráfica
        self.audio_ax.set_title("Degradación de la Onda según Frecuencia de Muestreo (Zoom 8ms)", color="#0f172a")
        self.audio_ax.set_xlabel("Tiempo (s)")
        self.audio_ax.set_ylabel("Amplitud Int16")
        self.audio_ax.legend(facecolor='#ffffff', edgecolor='#e2e8f0', loc='upper right')
        self.audio_ax.grid(True, alpha=0.2, color="#94a3b8")

        try:
            self.audio_ax.relim()
            self.audio_ax.autoscale_view()
        except Exception:
            pass

        self.audio_base_xlim = tuple(self.audio_ax.get_xlim())
        self.audio_base_ylim = tuple(self.audio_ax.get_ylim())
        self.audio_zoom_var.set(1.0)
        self.audio_canvas.draw()
        
        self._append_audio_log("Análisis comparativo de formas de onda completado.")



if __name__ == "__main__":
    root = ctk.CTk()
    TPIntegradorApp(root)
    root.mainloop()


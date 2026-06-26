"""
CAPA DE LÓGICA (CORE ENGINE) - Procesamiento Digital de Señales (DSP)

Módulo encargado de operaciones de procesamiento de audio:
- Grabación desde micrófono (Entrada)
- Remuestreo digital (diezmado)
- Cuantización digital (reducción de profundidad de bits)
- Análisis de Fourier (FFT) para espectro de frecuencias
- Cálculo de parámetros de Nyquist/Aliasing
"""

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.signal import resample_poly
from math import gcd


def grabar_audio(duracion=5, fs=44100, bits=16, filename="output.wav"):
    """
    Captura audio desde el micrófono en tiempo real.
    
    Parámetros:
        duracion (float): Duración de la grabación en segundos
        fs (int): Frecuencia de muestreo en Hz (44100 por defecto)
        bits (int): Profundidad de bits (16 por defecto)
        filename (str): Nombre del archivo (no utilizado internamente)
    
    Retorna:
        tuple: (audio_array, frecuencia_muestreo)
            - audio_array: Array NumPy con muestras int16
            - frecuencia_muestreo: Frecuencia de grabación efectiva
    """
    print(f"[I/O HANDLER] Grabando a {fs}Hz, {bits} bits, duración {duracion}s...")
    audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("[I/O HANDLER] Grabación finalizada correctamente.")
    # Array plano 1D para evitar errores de matriz columna
    return audio.flatten(), fs


def remuestrear_audio(audio_original, fs_original, fs_destino):
    """
    Remuestrea audio a una nueva frecuencia de muestreo (DIEZMADO).
    
    Utiliza resample_poly de SciPy para preservar calidad mediante
    interpolación polifásica (no naive downsampling).
    
    Parámetros:
        audio_original: Array de muestras de audio (int16)
        fs_original (int): Frecuencia de muestreo original en Hz
        fs_destino (int): Frecuencia de muestreo destino en Hz
    
    Retorna:
        audio_resampled: Array de muestras remuestreadas (int16)
    
    Nota: Si fs_original == fs_destino, retorna copia del original.
    """
    if fs_original == fs_destino:
        return audio_original.copy()

    # Convertir a float para evitar pérdida de precisión en cálculos
    audio_flat = np.asarray(audio_original, dtype=np.float64).flatten()
    
    # Calcular factores de remuestreo para resample_poly
    g = gcd(fs_destino, fs_original)
    up = fs_destino // g
    down = fs_original // g

    # Remuestreo polifásico (método profesional)
    audio_resampled = resample_poly(audio_flat, up, down)
    
    # Asegurar rango válido de int16 (-32768 a 32767)
    audio_resampled = np.clip(audio_resampled, -32768, 32767)

    return audio_resampled.astype(np.int16)


def cuantizar_audio(audio_original, bits_destino):
    """
    Reduce la profundidad de bits del audio (CUANTIZACIÓN).
    
    Mapea muestras int16 a menos niveles de cuantización.
    Utiliza rango dinámico de la señal para máxima resolución efectiva.
    
    Parámetros:
        audio_original: Array de muestras (int16)
        bits_destino (int): 8 o 16 (bits destino)
    
    Retorna:
        audio_cuantizado: Array remapped a nueva profundidad (int16)
    """
    if bits_destino == 16:
        return audio_original.copy()

    elif bits_destino == 8:
        audio_float = np.asarray(audio_original, dtype=np.float64).flatten()

        # Usar RANGO REAL de la señal para máxima utilización de niveles
        # (en lugar de asumir rango teórico completo de int16)
        min_val = audio_float.min()
        max_val = audio_float.max()
        rango = max_val - min_val

        if rango == 0:
            return audio_original.copy()

        # 8 bits = 256 niveles
        niveles = 2 ** bits_destino
        delta = rango / (niveles - 1)

        # Mapeo: [min_val, max_val] → [0, 255] → mapear de vuelta
        audio_en_escalones = np.round((audio_float - min_val) / delta)
        audio_en_escalones = np.clip(audio_en_escalones, 0, niveles - 1)
        audio_cuantizado = (audio_en_escalones * delta) + min_val

        return audio_cuantizado.astype(np.int16)

    return audio_original


def calcular_espectro_frecuencia(audio, fs):
    """
    Calcula el espectro de frecuencia mediante FFT (ANÁLISIS FRECUENCIAL).
    
    Parámetros:
        audio: Array de muestras de audio
        fs (int): Frecuencia de muestreo en Hz
    
    Retorna:
        tuple: (frecuencias, magnitudes)
            - frecuencias: Array de valores de frecuencia (Hz)
            - magnitudes: Array de magnitudes del espectro
    """
    # Convertir a float para FFT
    audio_float = np.asarray(audio, dtype=np.float64).flatten()
    
    # FFT: Transformada Rápida de Fourier
    fft = np.fft.fft(audio_float)
    freqs = np.fft.fftfreq(len(fft), 1.0 / float(fs))
    
    # Tomar solo mitad positiva (simetría en señales reales)
    mitad = len(freqs) // 2
    magnitudes = np.abs(fft)[:mitad]
    
    return freqs[:mitad], magnitudes


def calcular_frecuencia_nyquist(fs):
    """
    Calcula la frecuencia de Nyquist (límite teórico de Aliasing).
    
    Parámetros:
        fs (int): Frecuencia de muestreo en Hz
    
    Retorna:
        float: Frecuencia de Nyquist = fs/2
    
    Nota: Frecuencias por encima de Nyquist generan aliasing.
    """
    return fs / 2.0


def detectar_aliasing_potencial(fs_original, fs_destino, frecuencias_audio_max=None):
    """
    Detecta si hay riesgo de aliasing al cambiar frecuencia de muestreo.
    
    Parámetros:
        fs_original (int): Frecuencia de muestreo original
        fs_destino (int): Frecuencia de muestreo destino
        frecuencias_audio_max (float): Frecuencia máxima presente en audio (opcional)
    
    Retorna:
        dict: Información de aliasing:
            - tiene_riesgo: bool
            - nyquist_original: float
            - nyquist_destino: float
            - relacion: float (fs_destino / fs_original)
    """
    nyquist_orig = calcular_frecuencia_nyquist(fs_original)
    nyquist_dest = calcular_frecuencia_nyquist(fs_destino)
    
    tiene_riesgo = fs_destino < fs_original and nyquist_dest < nyquist_orig
    
    return {
        'tiene_riesgo': tiene_riesgo,
        'nyquist_original': nyquist_orig,
        'nyquist_destino': nyquist_dest,
        'relacion': fs_destino / fs_original,
    }


def normalizar_amplitud(audio, bits_destino):
    """
    Normaliza amplitud de audio según profundidad de bits destino.
    
    Parámetros:
        audio: Array de muestras
        bits_destino (int): 8 o 16
    
    Retorna:
        Array normalizado en rango [-1, 1]
    """
    audio_float = np.asarray(audio, dtype=np.float64).flatten()
    
    max_val = np.abs(audio_float).max()
    if max_val == 0:
        return audio_float
    
    return audio_float / max_val


# ============ FUNCIONES AUXILIARES (DEMO Y TESTING) ============

def mostrar_espectro(audio, fs):
    """Visualiza el espectro de Fourier (función de prueba)."""
    freqs, magnitudes = calcular_espectro_frecuencia(audio, fs)
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, magnitudes)
    plt.title("Espectro de Frecuencia del Audio")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.show()


def demo_audio():
    """Demo: graba audio, muestra espectro."""
    audio, fs = grabar_audio(duracion=3)
    mostrar_espectro(audio, fs)
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.signal import resample

from scipy.signal import resample_poly
from math import gcd

def grabar_audio(duracion=5, fs=44100, bits=16, filename="output.wav"):
    print(f"Grabando a {fs}Hz de forma nativa...")
    audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Grabación nativa finalizada.")
    # Forzamos un vector plano de 1 dimensión para evitar errores de matriz columna
    return audio.flatten(), fs

def cuantizar_audio(audio_original, bits_destino):
    if bits_destino == 16:
        return audio_original.copy()

    elif bits_destino == 8:
        audio_float = np.asarray(audio_original, dtype=np.float64).flatten()

        # Usamos el rango REAL de la señal. Ahora que remuestrear_audio ya
        # no genera outliers por ringing (resample_poly + clip), esto es
        # seguro, y además es lo correcto: reparte los 256 niveles sobre la
        # amplitud que el audio realmente usa, no sobre todo el rango
        # teórico de int16 que una voz grabada nunca llega a ocupar entero.
        min_val = audio_float.min()
        max_val = audio_float.max()
        rango = max_val - min_val

        if rango == 0:
            return audio_original.copy()

        niveles = 2 ** bits_destino  # 256
        delta = rango / (niveles - 1)

        audio_en_escalones = np.round((audio_float - min_val) / delta)
        audio_en_escalones = np.clip(audio_en_escalones, 0, niveles - 1)  # por seguridad
        audio_cuantizado = (audio_en_escalones * delta) + min_val

        return audio_cuantizado.astype(np.int16)

    return audio_original

def remuestrear_audio(audio_original, fs_original, fs_destino):
    if fs_original == fs_destino:
        return audio_original.copy()

    audio_flat = np.asarray(audio_original, dtype=np.float64).flatten()
    g = gcd(fs_destino, fs_original)
    up = fs_destino // g
    down = fs_original // g

    audio_resampled = resample_poly(audio_flat, up, down)
    audio_resampled = np.clip(audio_resampled, -32768, 32767)

    return audio_resampled.astype(np.int16)

def mostrar_espectro(audio, fs):
    fft = np.fft.fft(audio.flatten())
    freqs = np.fft.fftfreq(len(fft), 1/fs)
    plt.figure(figsize=(10,4))
    plt.plot(freqs[:len(freqs)//2], np.abs(fft)[:len(freqs)//2])
    plt.title("Espectro de Frecuencia del Audio")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.show()

def demo_audio():
    audio, fs = grabar_audio(duracion=3)
    mostrar_espectro(audio, fs)
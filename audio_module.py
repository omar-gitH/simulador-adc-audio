import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

def grabar_audio(duracion=5, fs=44100, bits=16, filename="output.wav"):
    print("Grabando...")
    audio = sd.rec(int(duracion*fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, audio)
    print("Grabación finalizada y guardada en", filename)
    return audio, fs

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

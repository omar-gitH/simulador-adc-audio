import numpy as np
import matplotlib.pyplot as plt

def generar_senal(tipo="sinusoidal", frecuencia=1000, duracion=0.01, fs=44100):
    t = np.linspace(0, duracion, int(fs*duracion), endpoint=False)
    if tipo == "sinusoidal":
        señal = np.sin(2*np.pi*frecuencia*t)
    elif tipo == "cuadrada":
        señal = np.sign(np.sin(2*np.pi*frecuencia*t))
    elif tipo == "ruido":
        # Ruido blanco gaussiano normal(0,1). No depende de 'frecuencia'.
        señal = np.random.normal(0.0, 1.0, size=len(t))
        # Normalizar a rango [-1, 1] para mantener consistencia visual
        max_val = np.max(np.abs(señal))
        if max_val > 0:
            señal = señal / max_val
    else:
        raise ValueError("Tipo de señal no soportado")
    return t, señal

def muestrear_y_cuantizar(señal, fs, bits):
    niveles = 2**bits
    # Asegurar que la señal está en [-1, 1]
    señal_clipped = np.clip(señal, -1.0, 1.0)
    señal_cuantizada = np.round(señal_clipped * (niveles/2)) / (niveles/2)
    return señal_cuantizada

def demo_adc():
    t, señal = generar_senal("sinusoidal", frecuencia=1000)
    señal_cuantizada = muestrear_y_cuantizar(señal, fs=44100, bits=8)

    plt.figure(figsize=(10,4))
    plt.plot(t, señal, label="Señal Original")
    plt.step(t, señal_cuantizada, label="Señal Digitalizada", where="mid")
    plt.title("Simulación ADC – Nyquist y Cuantización")
    plt.legend()
    plt.show()

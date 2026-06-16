# TP Integrador – Simulador ADC y Conversor de Audio

Esta aplicación muestra dos módulos:

1. Simulador de conversión de señales (ADC)
2. Conversor de audio analógico a digital

## Requisitos

- Python 3.10 o superior
- Micrófono disponible si se quiere usar la grabación de audio

## Instalación

```bash
python -m pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

## Notas importantes

- El audio grabado se guarda en el archivo local output.wav.
- Para evaluar el proyecto en GitHub, clona este repositorio y ejecuta la aplicación con las dependencias listadas arriba.
- Si el entorno de evaluación no tiene micrófono, la parte de grabación puede no funcionar, pero la interfaz y la simulación ADC sí deben abrirse.

- Por otro lado, si se dispone de un entorno externo a las funcionalidades operativas estandar de Microsoft (en particular entornos Linux) se debera abrir una terminal en la cual se deberan ejecutar los siguientes comandos:

```bash
   sudo apt update && sudo apt install python3-tk libportaudio2
```

- A diferencia de otros sistemas, los entornos Linux requieren la instalación manual de ciertas dependencias nativas del sistema operativo para dar soporte a la interfaz gráfica (tkinter) y al backend del motor de audio (portaudio).
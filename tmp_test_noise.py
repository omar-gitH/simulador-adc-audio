from adc_module import generar_senal

t,s = generar_senal('ruido', 1000, 0.01, 44100)
print(len(t), s.min(), s.max())

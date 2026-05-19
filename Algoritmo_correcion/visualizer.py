import numpy as np
import matplotlib.pyplot as plt

def dibujar_analisis_completo(fc, mag_medida, coh_medida, correccion, mag_objetivo, data_alta_res=None, titulo="Micrófono de Medición"):
    """
    Genera el panel de 3 gráficas (Coherencia, Magnitud y Corrección) para un micrófono específico.
    
    Parámetros:
        fc: Frecuencias centrales (eje X).
        mag_medida: Magnitud en 1/3 de octava del micro medido.
        coh_medida: Coherencia en 1/3 de octava del micro medido.
        correccion: Valores de EQ calculados por el algoritmo.
        mag_objetivo: La curva del FOH que queremos alcanzar.
        data_alta_res: Tupla (frecuencias, coherencia) opcional para la línea gris de fondo.
        titulo: Nombre del micrófono (aparecerá en las gráficas).
    """
    
    # Formato de etiquetas para frecuencias (redondeo y formato legible)
    etiquetas_x = [str(int(f)) if f == int(f) else str(f) for f in fc]
    
    plt.figure(figsize=(14, 12))

    # --- 1. COHERENCIA ---
    ax1 = plt.subplot(3, 1, 1)
    # Si pasamos los datos de alta resolución (el array FT_micro1), los pintamos de fondo
    if data_alta_res is not None:
        ax1.semilogx(data_alta_res[0], data_alta_res[1], label=f'Coherencia Alta Res ({titulo})', color='lightgray', alpha=0.7)
    
    ax1.semilogx(fc, coh_medida, marker='o', label='Coherencia 1/3 Octava', color='red')
    ax1.axhline(0.7, color='black', linestyle='--', label='Umbral de Seguridad (0.7)')
    ax1.set_title(f'1. Análisis de Coherencia [{titulo}]: ¿Es fiable la medición?')
    ax1.set_ylabel('Coherencia (0 a 1)')
    ax1.set_ylim([0, 1.1])
    ax1.set_xlim([20, 20000])
    ax1.set_xticks(fc)
    ax1.set_xticklabels(etiquetas_x, rotation=45, ha='right', fontsize=8)
    ax1.grid(True, which='major', linestyle='--', alpha=0.5)
    ax1.legend()

    # --- 2. MAGNITUDES ---
    ax2 = plt.subplot(3, 1, 2)
    ax2.semilogx(fc, mag_objetivo, marker='s', label='Objetivo (FOH)', color='blue')
    ax2.semilogx(fc, mag_medida, marker='^', label=f'Medición ({titulo})', color='orange')
    ax2.set_title(f'2. Comparativa de Magnitudes: {titulo} vs Objetivo')
    ax2.set_ylabel('Magnitud (dB)')
    ax2.set_xlim([20, 20000])
    ax2.autoscale(enable=True, axis='y')
    ax2.set_xticks(fc)
    ax2.set_xticklabels(etiquetas_x, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, which='major', linestyle='--', alpha=0.5)
    ax2.legend()

    # --- 3. CORRECCIÓN (EQ) ---
    ax3 = plt.subplot(3, 1, 3)
    eje_x_barras = np.arange(len(fc))
    ax3.bar(eje_x_barras, correccion, color='green', alpha=0.7, label='Faders EQ Aplicados')
    ax3.set_title(f'3. Ecualización Final sugerida para {titulo}')
    ax3.set_ylabel('Ajuste (dB)')
    ax3.set_ylim([-15, 10])
    ax3.set_xticks(eje_x_barras)
    ax3.set_xticklabels(etiquetas_x, rotation=45, ha='right', fontsize=8)
    ax3.axhline(0, color='black', linewidth=1)
    ax3.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax3.legend()

    plt.tight_layout()
    plt.show()
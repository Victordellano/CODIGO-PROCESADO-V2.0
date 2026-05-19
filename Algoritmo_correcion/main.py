import audio_io
import dsp_core
import visualizer
import numpy as np
import pandas as pd

# =============PARAMETROS GLOBALES =============

curva_objetivo_guardada = None
SOLICITAR_CAPTURA_FOH = True  
UMBRAL_COH = 0.7
MAX_CUT = -6.0
MAX_BOOST = 6.0

# ==============================================

# Listado de microfonos
lista_micros_sala = {
    "Micro 1 (Seccion oeste)": "algoritmo_correcion/recinto.J02.wav",
    "Micro 2 (Seccion norte)": "algoritmo_correcion/recinto.J03.wav",
    "Micro 3 (Seccion este)": "algoritmo_correcion/recinto.J04.wav"
}

def analizar_microfono_individual(nombre_micro, archivo_audio, referencia, fs, curva_objetivo):

    print(f"  -> Analizando: {nombre_micro}...")
    
    # Lectura del output wav de Odeon
    fs_ir, ir_medicion = audio_io.wav_to_code(archivo_audio)

    if fs != fs_ir:
        print(f"     [!] ADVERTENCIA: La fs del barrido ({fs}) no coincide con la IR ({fs_ir})")
    
    #convolucion con el sweep
    micro_convolucionado = dsp_core.convolucionar_ir(referencia, ir_medicion)

    ref_alineada, med_alineada = dsp_core.alinear_senales(referencia, micro_convolucionado)

    # Calculo de funcion de transferencia y coherencia
    frec, mag, fase, coh = dsp_core.funcion_transferencia(ref_alineada, med_alineada, fs)
    fc, mag_1_3, _, coh_1_3 = dsp_core.to_1_3_oct(frec, mag, fase, coh)
    
    # 3. Calculo de la correccion EQ respecto a la curva objetivo
    correccion = dsp_core.algoritmo_correccion_eq(mag_medida=mag_1_3, coh_medida=coh_1_3, mag_objetivo=curva_objetivo, umbral_coh=UMBRAL_COH, max_cut=MAX_CUT, max_boost=MAX_BOOST)
    
    visualizer.dibujar_analisis_completo(fc, mag_1_3, coh_1_3, correccion, curva_objetivo, titulo=nombre_micro)
    
    return correccion

def ciclo_principal(referencia, fs):

    global curva_objetivo_guardada, SOLICITAR_CAPTURA_FOH
    
    print("\n=== INICIANDO CICLO DE ANÁLISIS ===")
    
    # Captura de la curva objetivo (FOH) o uso de la guardada en memoria
    if SOLICITAR_CAPTURA_FOH or curva_objetivo_guardada is None:

        print("Capturando curva objetivo desde el FOH...")

        fs_ir, ir_foh = audio_io.wav_to_code("algoritmo_correcion/recinto.J01.wav")

        foh_convolucionado = dsp_core.convolucionar_ir(referencia, ir_foh)

        ref_alineada, med_alineada = dsp_core.alinear_senales(referencia, foh_convolucionado)
        
        frec_foh, mag_foh, fase_foh, coh_foh = dsp_core.funcion_transferencia(ref_alineada, med_alineada, fs)
        _, mag_foh_1_3, _, _ = dsp_core.to_1_3_oct(frec_foh, mag_foh, fase_foh, coh_foh)

        curva_objetivo_guardada = mag_foh_1_3
        SOLICITAR_CAPTURA_FOH = False
    else:
        print("Usando curva objetivo en memoria.")

    # Recorre el diccionario de micrófonos y analiza cada uno, guardando los resultados en un nuevo diccionario
    resultados_totales = {}
    
    for nombre, archivo in lista_micros_sala.items():
        
        ajuste_micro = analizar_microfono_individual(nombre, archivo, referencia, fs, curva_objetivo_guardada)
        resultados_totales[nombre] = ajuste_micro

    print("=== CICLO COMPLETADO ===")
    return resultados_totales

if __name__ == "__main__":

    FS_SISTEMA = 44100
    sweep_referencia = audio_io.generador_ruido_rosa(30.0, FS_SISTEMA)

    while True:
        correcciones_del_recinto = ciclo_principal(sweep_referencia, FS_SISTEMA)

        for nombre_micro in correcciones_del_recinto.keys():
            correccion = correcciones_del_recinto[nombre_micro]
            print(f"\nCorrecciones para {nombre_micro}:")
            for banda, ajuste in zip(dsp_core.FC, correccion):
                print(f"  Banda {banda} Hz: {ajuste:.2f} dB")
            correcciones_del_recinto[nombre_micro] = correccion.tolist()
        
        pd.DataFrame(correcciones_del_recinto).set_index(np.asarray(dsp_core.FC, dtype=np.int16)).to_csv("correcciones_calculadas.csv", float_format="%.2f", index_label="Frecuencia (Hz)")

        input("\nPresiona Enter para repetir el ciclo de análisis...")

    
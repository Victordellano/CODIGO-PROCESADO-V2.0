import V2_audio_io
import V2_dsp_core
import visualizer

# =============PARAMETROS GLOBALES =============

#curva_objetivo_guardada = None 
solicitar_captura_foh = True  
umbral_coh = 0.7
max_cut = -12.0
max_boost = 6.0

# ==============================================

# Listado de microfonos
lista_micros_sala = {
    "Micro 1 (Seccion oeste)": "micro_medicion_1.wav",
    "Micro 2 (Seccion norte)": "micro_medicion_2.wav",
    "Micro 3 (Seccion este)": "micro_medicion_3.wav"
}

def analizar_microfono_individual(nombre_micro, archivo_audio, referencia, fs, curva_objetivo):

    print(f"  -> Analizando: {nombre_micro}...")
    
    # Lectura del output wav de Odeon
    _, micro_medicion = V2_audio_io.wav_to_code(archivo_audio)
    
    # Calculo de funcion de transferencia y coherencia
    frec, mag, fase, coh = V2_dsp_core.funcion_transferencia(referencia, micro_medicion, fs)
    fc, mag_1_3, _, coh_1_3 = V2_dsp_core.to_1_3_oct(frec, mag, fase, coh)
    
    # 3. Calculo de la correccion EQ respecto a la curva objetivo
    correccion = V2_dsp_core.algoritmo_correccion_eq(mag_medida=mag_1_3, coh_medida=coh_1_3, mag_objetivo=curva_objetivo, umbral_coh=0.7, max_cut=-12.0, max_boost=6.0)
    
    visualizer.dibujar_analisis_completo(fc, mag_1_3, coh_1_3, correccion, curva_objetivo, titulo=nombre_micro)
    
    return correccion

def ciclo_principal():

    global curva_objetivo_guardada, solicitar_captura_foh
    
    print("\n=== INICIANDO CICLO DE ANÁLISIS ===")
    
    # Lectura del audio de referencia (mesa FOH)
    fs, referencia = V2_audio_io.wav_to_code("referencia_radiohead.wav")
    
    # Captura de la curva objetivo (FOH) o uso de la guardada en memoria
    if solicitar_captura_foh or curva_objetivo_guardada is None:

        print("Capturando curva objetivo desde el FOH...")

        _, micro_foh = V2_audio_io.wav_to_code("micro_FOH.wav")

        frec_foh, mag_foh, fase_foh, coh_foh = V2_dsp_core.funcion_transferencia(referencia, micro_foh, fs)
        _, mag_foh_1_3, _, _ = V2_dsp_core.to_1_3_oct(frec_foh, mag_foh, fase_foh, coh_foh)

        curva_objetivo_guardada = mag_foh_1_3
        solicitar_captura_foh = False
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

    correcciones_del_recinto = ciclo_principal()

    print(correcciones_del_recinto)
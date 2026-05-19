import numpy as np
import scipy.signal as signal

FC = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 
                   400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 
                   4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000])

def funcion_transferencia(ref_signal, mic_signal, sample_rate, ventana_fft=65536):
    """
    Calcula la Función de Transferencia H(f) y la Coherencia entre una señal 
    de referencia (mesa) y una de medición (micrófono).
    
    Parámetros:
        ref_signal: Array NumPy con el audio de la mesa (FOH). (señal normalizada)
        mic_signal: Array NumPy con el audio del micrófono de medición. (señal normalizada)
        sample_rate: Frecuencia de muestreo.
        ventana_fft: Tamaño de la ventana (frame) para el promediado de Welch.
    
    Retorna:
        frecuencias: Array frecuencias en Hz. (eje x)
        magnitud_db: Array diferencia de EQ en dB. (eje y)
        fase_grados: Array con la diferencia de fase en Grados.
        coherencia:  Array con la causalidad (valores de 0.0 a 1.0).
    """
    
    # Comprobacion de que sean del mismo tamaño
    min_length = min(len(ref_signal), len(mic_signal))
    ref_recortada = ref_signal[:min_length]
    mic_recortada = mic_signal[:min_length]
    
    # Espectro de Potencia de la Referencia (Pxx)
    frecuencias, Pxx_ref = signal.welch(ref_recortada, fs=sample_rate, nperseg=ventana_fft)
    
    # Espectro Cruzado (Pxy) - Lo que tienen en común
    _, Pxy = signal.csd(ref_recortada, mic_recortada, fs=sample_rate, nperseg=ventana_fft)
    
    # Función de Transferencia H_f
    H_f = Pxy / (Pxx_ref + 1e-12) # Evita división por cero con un pequeño valor añadido
    
    # Magnitud (EQ) pasado a dB
    magnitud_db = 20 * np.log10(np.abs(H_f) + 1e-12)
    
    # Fase (Retardo) pasado a grados
    fase_grados = np.angle(H_f, deg=True)
    
    # Calculo de la Coherencia
    _, coherencia = signal.coherence(ref_recortada, mic_recortada, fs=sample_rate, nperseg=ventana_fft)
    
    return frecuencias, magnitud_db, fase_grados, coherencia

def to_1_3_oct(frecuencias, magnitud_db, fase_grados, coherencia):
    """
    Agrupa los datos lineales de una FFT de alta resolución en las 31 bandas 
    estándar de 1/3 de octava (ISO 266).

    Parámetros: (iguales a la SALIDA de funcion_transferencia)
        frecuencias: Array frecuencias en Hz. (eje x)
        magnitud_db: Array diferencia de EQ en dB. (eje y)
        fase_grados: Array con la diferencia de fase en Grados.
        coherencia:  Array con la causalidad (valores de 0.0 a 1.0).
    
    Retorna:
        fc: frecuencias centrales de los tercios de octava.
        mag_1_3: magnitud de los tercios de octava.
        fase_1_3: fase de los tercios de octava.
        coh_1_3: coherencia de los tercios de octava.
    """
    
    # Frecuencias centrales ISO 266
    fc = FC

    mag_1_3 = []
    fase_1_3 = []
    coh_1_3 = []

    # Conversion dB a lineal para promediar la energía acústica correctamente
    mag_lineal = 10 ** (magnitud_db / 20)

    # Desenvolver la fase (unwrap) evita que el promedio falle cuando la fase salta bruscamente de +180 a -180
    fase_unwrapped = np.unwrap(np.deg2rad(fase_grados))

    # Agrupacion datos ("Bins" de la FFT) en cada banda
    for f in fc:
        f_min = f / (2 ** (1/6))
        f_max = f * (2 ** (1/6))

        # Encontrar los índices de las frecuencias que caen dentro de esta banda
        indices = np.where((frecuencias >= f_min) & (frecuencias < f_max))[0]

        if len(indices) > 0:
            # Promediado de magnitud y paso a dB
            mag_promedio_lin = np.mean(mag_lineal[indices])
            mag_1_3.append(20 * np.log10(mag_promedio_lin))
            
            # Promediado de fase y paso a grados
            fase_promedio_rad = np.mean(fase_unwrapped[indices])
            fase_1_3.append(np.rad2deg(fase_promedio_rad))
            
            # Promediado de coherencia
            coh_1_3.append(np.mean(coherencia[indices]))
        else:
            # Si en graves extremos la resolución de la FFT es baja y no hay datos,
            # buscamos el punto de la FFT de alta resolución más cercano a f para evitar dejar un bin vacío con valor 0.
            idx_cercano = np.argmin(np.abs(frecuencias - f))
            
            # En lugar de 0, usamos los datos reales de ese punto exacto
            mag_1_3.append(magnitud_db[idx_cercano])
            fase_1_3.append(fase_grados[idx_cercano])
            coh_1_3.append(coherencia[idx_cercano])

    # Volver a envolver la fase entre -180 y +180 grados
    fase_1_3 = (np.array(fase_1_3) + 180) % 360 - 180

    return fc, np.array(mag_1_3), np.array(fase_1_3), np.array(coh_1_3)

def algoritmo_correccion_eq(mag_medida, coh_medida, mag_objetivo=None, umbral_coh=0.7, max_cut=-12.0, max_boost=6.0):
    """
    Calcula los valores de ecualización (en dB) necesarios para que la 
    señal medida alcance la curva objetivo, utilizando la coherencia 
    como filtro de seguridad.

    Parámetros:
        mag_medida: Array de 31 valores (magnitud en dB de la sección medida).
        coh_medida: Array de 31 valores (coherencia de la medición).
        mag_objetivo: Array de 31 valores con la EQ ideal (Ej: el FOH). 
                      Si es None, se asume un objetivo de 0 dB (respuesta plana).
        umbral_coh: Valor mínimo de coherencia (0.0 a 1.0) para aplicar corrección.
        max_cut: Límite máximo de atenuación en dB (seguridad).
        max_boost: Límite máximo de realce en dB (seguridad para los amplificadores).

    Retorna:
        correccion_final: Array de 31 valores con los dB a aplicar en el ecualizador gráfico.
    """
    
    # Definicion del objetivo (Si no se le pasa uno, asume 0 dB para todas las bandas)
    if mag_objetivo is None:
        mag_objetivo = np.zeros_like(mag_medida)
        
    # Calculo diferencia
    diferencia_bruta = mag_objetivo - mag_medida
    
    # Límites de Seguridad (Clamping)
    correccion_limitada = np.clip(diferencia_bruta, max_cut, max_boost)
    
    # Correccion en banda con coherencia suficiente, 0 dB en bandas con baja coherencia
    correccion_final = np.where(coh_medida >= umbral_coh, correccion_limitada, 0.0)
    
    return correccion_final

def convolucionar_ir(senal, respuesta_impulso):
    """
    Convoluciona una señal fuente con la Respuesta al Impulso (IR) de la sala.
    Ambas señales deben tener la misma frecuencia de muestreo.
    """
    # fftconvolve es el estándar de la industria para señales de audio largas
    senal_convolucionada = signal.fftconvolve(senal, respuesta_impulso, mode='full')
    
    # Es vital normalizar el resultado para evitar saturación (clipping)
    # al guardarlo o analizarlo
    pico_maximo = np.max(np.abs(senal_convolucionada))
    if pico_maximo > 0:
        senal_convolucionada = senal_convolucionada / pico_maximo
        
    return senal_convolucionada.astype(np.float32)

def alinear_senales(ref, med):
    """
    Encuentra el retraso entre la medición y la referencia usando 
    correlación cruzada y recorta las señales para que estén 
    perfectamente alineadas en el tiempo y tengan la misma longitud.
    """
    # 1. Encontrar el retraso mediante correlación cruzada (por FFT por velocidad)
    correlacion = signal.correlate(med, ref, mode='full', method='fft')
    
    # El pico máximo de la correlación indica el retraso en muestras
    delay_muestras = np.argmax(np.abs(correlacion)) - (len(ref) - 1)
    
    if delay_muestras > 0:
        # La medición llega tarde (lo normal, el sonido tarda en viajar)
        med_alineada = med[delay_muestras:]
        ref_alineada = ref[:len(med_alineada)]
    elif delay_muestras < 0:
        # La referencia va por detrás (raro, pero cubre errores de ruteo)
        delay_abs = abs(delay_muestras)
        ref_alineada = ref[delay_abs:]
        med_alineada = med[:len(ref_alineada)]
    else:
        # Ya estaban alineadas
        ref_alineada = ref
        med_alineada = med
        
    # 2. Igualar las longitudes exactas cortando la "cola" sobrante
    min_len = min(len(ref_alineada), len(med_alineada))
    
    return ref_alineada[:min_len], med_alineada[:min_len]



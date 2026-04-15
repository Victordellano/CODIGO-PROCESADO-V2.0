import numpy as np
import soundfile as sf
from scipy.io import wavfile
import scipy.signal as signal

def wav_to_code(ruta_archivo):
    """
    Lee CUALQUIER archivo de audio (16, 24, 32 bits, float) usando 'soundfile'.
    Lo devuelve siempre normalizado entre -1.0 y 1.0 en formato float32.
    Lo convierte a Mono si es estéreo. 

    Parámetros:
        ruta_archivo: Ruta del archivo de audio a leer.

    Retorna:
        sample_rate: Frecuencia de muestreo del archivo.
        data: Array NumPy con la señal normalizada a 32 bits.
    """
    # Soundfile lee el audio y ya lo entrega normalizado en flotante
    data, sample_rate = sf.read(ruta_archivo)
    
    # Comprobacion de si es stereo (soundfile devuelve [muestras, canales])
    if len(data.shape) > 1 and data.shape[1] > 1:
        # Suma L y R y divides para hacer mono
        data = np.mean(data, axis=1)
        
    # Paso a float32 para que ocupe la mitad de memoria RAM que el float64
    return sample_rate, data.astype(np.float32)

def generador_ruido_rosa(longitud):
    """
    Genera ruido rosa aplicando un filtro de 1/sqrt(f) a ruido blanco puro
    en el dominio de la frecuencia.
    """
    # Genera ruido blanco
    ruido_blanco = np.random.randn(longitud)
    
    # Paso al dominio de la frecuencia (FFT real)
    X = np.fft.rfft(ruido_blanco)
    
    # Vector de frecuencias
    frecuencias = np.fft.rfftfreq(longitud)
    frecuencias[0] = 1.0  # Evita la división por cero en la componente continua (DC)
    
    # Aplica el filtro: la amplitud cae proporcionalmente a 1/sqrt(f)
    X_rosa = X / np.sqrt(frecuencias)
    
    # Vuelta al dominio del tiempo
    ruido_rosa = np.fft.irfft(X_rosa, n=longitud)
    
    # Normalizacion por pico absoluto (como con la referencia)
    pico_ruido = np.max(np.abs(ruido_rosa))
    if pico_ruido > 0:
        ruido_rosa = ruido_rosa / pico_ruido
        
    return ruido_rosa

def simulador_audio_sala(audio_referencia, sample_rate, eq_curva_db, nivel_ruido=0.0, tipo_ruido="rosa", archivo_ruido=None):
    """
    Genera un archivo de audio en base a una referencia, simulando lo que captaría un micrófono de medición.
    Aplica una curva de EQ de 31 bandas e inyecta ruido para 
    poner a prueba el cálculo de coherencia.

    Parámetros:
        audio_referencia: Array NumPy con el audio de la mesa (FOH).
        sample_rate: Frecuencia de muestreo.
        eq_curva_db: Array o lista de 31 valores (dB) simulando la respuesta de la sala.
        filename: Nombre del archivo de salida.
        nivel_ruido: Cantidad de ruido del público (0.0 a 1.0). Destruye la coherencia.
        tipo_ruido: "rosa" para ruido rosa generado, "publico" para ruido de archivo.
        archivo_ruido: Ruta del archivo de audio a usar como ruido (si tipo_ruido es "publico").

    Retorna:
        sample_rate: Frecuencia de muestreo del audio generado.
        audio_filtrado: Array NumPy con el audio simulado de la sala, normalizado a float32.
    """
    fc = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 
                   400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 
                   4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000])
    
    # Diseño del filtro de la sala
    ganancia_lineal = 10 ** (np.array(eq_curva_db) / 20.0)
    freqs_fir = np.concatenate(([0.0], fc, [sample_rate / 2.0]))
    gains_fir = np.concatenate(([ganancia_lineal[0]], ganancia_lineal, [ganancia_lineal[-1]]))
    filtro_fir = signal.firwin2(4097, freqs_fir, gains_fir, fs=sample_rate)
    
    # Pasa el MISMO audio de la mesa por la sala
    audio_filtrado = signal.fftconvolve(audio_referencia, filtro_fir, mode='same')

    # Inyeccion de ruido
    if nivel_ruido > 0:
        longitud_necesaria = len(audio_referencia)
        
        if tipo_ruido == "rosa":
            ruido_base = generador_ruido_rosa(longitud_necesaria)
            
        elif tipo_ruido == "publico" and archivo_ruido is not None:
            # Llamada interna al lector de WAVs
            fs_ruido, ruido_base = wav_to_code(archivo_ruido)
            
            # Comprobacion de sample rate para evitar errores de alineación temporal
            if fs_ruido != sample_rate:
                print(f"Aviso: El sample rate del ruido ({fs_ruido}Hz) no coincide con la música ({sample_rate}Hz).")
            
            # Ajuste explícito de duraciones
            if len(ruido_base) > longitud_necesaria:
                # Si el ruido es MÁS LARGO, se corta al tamaño necesario
                ruido_base = ruido_base[:longitud_necesaria]
                
            elif len(ruido_base) < longitud_necesaria:
                # Si el ruido es MÁS CORTO, se repite en bucle hasta alcanzar la longitud necesaria
                repeticiones = int(np.ceil(longitud_necesaria / len(ruido_base)))
                ruido_base = np.tile(ruido_base, repeticiones)[:longitud_necesaria]
                
            # 4. Normalizamos el ruido por su pico absoluto
            pico_ruido = np.max(np.abs(ruido_base))
            if pico_ruido > 0:
                ruido_base = ruido_base / pico_ruido
        else:
            ruido_base = np.zeros(longitud_necesaria) 

        # Multiplicamos por el fader de nivel y lo sumamos a la sala
        audio_filtrado += ruido_base * nivel_ruido

        return sample_rate, audio_filtrado.astype(np.float32)
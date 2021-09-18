import numpy as np
import librosa 
from scipy.signal import fftconvolve
from scipy import stats


def ploteo(ax, t, signal, label):
    ax.plot(t, signal, label=label)
    ax.legend()

def envelope(signal, fs, time_interval=0.0050):
    #hacer time interval variable por banda
    #time_interval = 0.0050 #10 - 50 ms
    interval = int(time_interval * fs) 
    
    n_windows = len(signal) // interval 
    remainder = len(signal) % interval
    env = np.empty(n_windows)
    for i in range(n_windows):
        env[i] = signal[i*interval:(i+1)*interval].sum()/interval
    env = env / np.max(abs(env))
    t_env = np.arange(0,n_windows*interval, interval)
    return t_env, librosa.amplitude_to_db(env)


def estim_slope(t_env, env, init, end):
    # Solo me interesa el valor de la slope
    init_idx = np.where(env < init)[0][0]
    try:
        end_idx = np.where(env < end)[0][0]
    except:
        end_idx = len(env)-1
    # regresion lineal 
    x = t_env[init_idx:end_idx+1]
    y = env[init_idx:end_idx+1]
    slope, intercept = stats.linregress(x,y)[0:2]
    return intercept, slope , x, y

def lundeby(signal, fs):
    # CONSTANTES DE DISEÑO
    TIME_INTERVAL = 0.005 # [s] from 0.005 to 0.001 
    NOISE_FLOOR_DISTANCE = 5 # [dB] from 5 to 10. Level above the noise
    INTERVALS = 3 # Intervals per 10 dB of decay. From 3 to 10 for low - high freqs
    MARGIN = 5 # Safety margin from cross point. From 5 to 10 dB of decay 
    DINAMIC_ABOVE, DINAMIC_BELOW = 10, 5 # Dinamic range of 10-20 dB referred to the noise floor
    
    # standarization
    onset = np.argmax(abs(signal))
    signal = signal[onset:]
    signal = signal / np.max(abs(signal))

    # squared response
    #signal_sqr = np.power(signal, 2)
    signal_sqr = abs(signal)
    t = np.arange(0,len(signal_sqr))

    # average smoothing
    t_env, env = envelope(signal_sqr, fs, time_interval=TIME_INTERVAL)
    
    # First estimation of noise floor using the tail (last 10%)
    tail = int(len(t_env) * 0.1)
    noise_level = env[-tail:].sum() / tail
    #print('First estimation of noise floor: {:.2f} dB'.format(noise_level))

    intercept, slope, x_line, y_line = estim_slope(t_env, env, 0, noise_level+NOISE_FLOOR_DISTANCE)
    cross_point = (noise_level - intercept) / slope

    # Find new time interval
    intervals_per_10dB = INTERVALS #3 - 10 [low - high]
    interval_dB = 10 / intervals_per_10dB
    interval = int(-interval_dB / slope)
    time_interval = interval / fs
    #print('New time interval: {:.4f} seconds'.format(time_interval))

    t_env, env= envelope(signal_sqr, fs, time_interval=time_interval)
    
    for i in range(5):
        margin_cross = MARGIN #5-10dB
        safe_cross_point = int(-margin_cross/slope) + int(cross_point)
        tail = int(len(t_env) * 0.1)
        if (safe_cross_point < t_env[-tail]):
            #print('uso el intervalo')
            index_cross = np.where(t_env > safe_cross_point)[0][0]
            noise_level = env[index_cross:].sum() / len(env[index_cross:])
        else:
            #print('uso la tail')
            noise_level = env[-tail:].sum() / tail
        #print('Nueva estimacion del piso de ruido de {:.2f} dB'.format(noise_level))


        def estim_slope_f(t_env, env, init, end):
            x = t_env[init:end+1]
            y = env[init:end+1]
            slope, intercept = stats.linregress(x,y)[0:2]
            return intercept, slope , x, y


        # Estimar la pendiente 5 dB [5-10] encima del pisode ruido para un rango de 10 dB [10-20]

        init = (noise_level+DINAMIC_ABOVE - intercept) / slope
        init = int(init / (time_interval * fs))
        end = (noise_level-DINAMIC_BELOW - intercept) / slope
        end = int(end / (time_interval * fs))

        intercept_f, slope_f, x_line_f, y_line_f = estim_slope_f(t_env, env, init, end)
        cross_point = (noise_level - intercept_f) / slope_f
    # insert delay samples
    cross_point = cross_point + onset
    return int(cross_point)
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from filterbank import Filterbank

    params = {'fs' : 16000,
              'bands' : [125, 250, 500, 1000, 2000, 4000],
              'bandsize' : 1,
              'order' : 4,
              'f_length': 16384,'power' : True}

    signal, fs = librosa.load('files/rir.wav', sr=16000, dtype='float64')

    filterbank = Filterbank(**params)
    bands = filterbank.apply(signal)

    plt.rcParams['axes.titley'] = 1.0    
    plt.rcParams['axes.titlepad'] = -14 

    fig, axs = plt.subplots(6, figsize=(10,25), constrained_layout=True)

    for idx in range(len(bands)):
        band = bands[idx]
        noise_onset = lundeby(band, fs)
        noiseless = band[:noise_onset]
        axs[idx].set_title('{} Hz'.format(params['bands'][idx]))
        axs[idx].plot(librosa.amplitude_to_db(band), label = 'signal')
        axs[idx].plot(librosa.amplitude_to_db(noiseless), label = 'noiseless')
        axs[idx].legend()
    plt.subplots_adjust(hspace = 0.1)
    plt.show()

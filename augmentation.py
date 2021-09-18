import numpy as np
import scipy
import glob
import librosa
import matplotlib.pyplot as plt 
from acoustic_params import Acoustic_params
from filterbank import Filterbank



def get_audio_list(path, file_types = ('.wav', '.WAV', '.flac', '.FLAC')):
    search_path = path + '/**/*'
    audio_list = []
    for file_type in file_types:
        audio_list.extend(glob.glob(search_path+file_type, recursive=True))
    return audio_list


def band_plot(**bands_dict):
    fig, axs = plt.subplots(6, figsize=(12,25)) # !harcoded!
    ticks_labels=['125 Hz', '250 Hz', '500 Hz', '1000 Hz', '2000 Hz', '4000 Hz']
    for label, bands in bands_dict.items():
        for band in range(len(bands)):
            axs[band].plot(bands[band,:], label=label, alpha = 0.8)
            axs[band].set_title(ticks_labels[band], fontsize=14)
            axs[band].legend()
        plt.subplots_adjust(hspace=0.4)


def preprocess_rir(rir, fs, tau=0.0025):
    # normalization
    rir = rir / np.max(abs(rir))
    
    # temporal decompose
    t_d = np.argmax(rir) # direct path
    t_o = int((tau) * fs) #tolerance window in samples (2.5 ms)
    init_idx = t_d - t_o
    final_idx = t_d + t_o + 1

    if init_idx < 0:
        init_idx = 0
    if final_idx > len(rir)-1:
        final_idx = len(rir)-1

    early= rir[init_idx:final_idx]
    late = rir[final_idx:]
    delay = rir[:init_idx]
    
    return delay, early, late


def envelope(signal, win_length=100):
    window = np.ones(win_length)/win_length
    env = np.convolve(abs(signal), window, mode='same')
    return env


def curve_model(t, Am, decay_rate, noise_floor):
    # late field reverb enevelope model
    ones = np.ones(len(t))
    model = Am * np.exp(-t/decay_rate) * ones + (noise_floor*ones)
    model = librosa.amplitude_to_db(model)
    return model

def curve_model_estim(t, Am, decay_rate, noise_floor):
    # funcion que uso para la estimacion
    ones = np.ones(len(t))
    model = Am * np.exp(-t/decay_rate) * ones + (noise_floor*ones)
    #model = librosa.amplitude_to_db(model)
    model = model ** 0.5
    return model



def curve_fit(t, rir):
    # get the envelope
    #rir_env = librosa.amplitude_to_db(envelope(rir))
    rir_env = envelope(rir) ** 0.5
    
    # optimization
    popt, pcov = scipy.optimize.curve_fit(curve_model_estim, t, rir_env, bounds=(0,1))
    return popt


#### mejorar ####
def noiseless_rir(t, Am, decay_rate):
    noise = np.random.normal(0,1,len(t))
    modelo = Am * np.exp(-t/decay_rate) * noise
    return modelo


def cross_fade(señal_1, señal_2, cross_point):
    """
    señal 1 se atenua luego del cross point
    señal 2 se amplifica luego del cross point
    """
    largo = int(50 * 0.001 * 16000) # 800 muestras
    if 2*largo > len(señal_1)-cross_point:
        return señal_1
    ventana = scipy.signal.hann(largo)
    fade_in, fade_out = ventana[:int(largo/2)], ventana[int(largo/2):]
    
    ventana_atenuante = np.concatenate((np.ones(cross_point-int(fade_out.size/2)),
                                        fade_out,
                                        np.zeros(len(señal_1)-cross_point-int(fade_out.size/2))))

    ventana_amplificadora = np.concatenate((np.zeros(cross_point-int(fade_out.size/2)),
                                            fade_in, 
                                            np.ones(len(señal_2)-cross_point-int(fade_out.size/2))))
    return (señal_1*ventana_atenuante) + (señal_2*ventana_amplificadora)


def noise_crossfade(t, rir, params, cross_point):
    rir_noiseless = noiseless_rir(t, params[0], params[1])
    rir_denoised = cross_fade(rir, rir_noiseless, cross_point)
    return rir_denoised
#### ------ ####


def augmentation(t, rir, params, fullband_decay ,TR60_desired):

    decay_rate_d = TR60_desired / (np.log(1000))
    ratio = decay_rate_d / fullband_decay
    t_md = ratio * params[1]

    #Augmentation
    rir_aug = rir * np.exp(-t*((params[1]-t_md)/(params[1]*t_md)))
    return rir_aug


def tr_augmentation(impulse, fs, TR60_desired):
    params = {'fs' : 16000,
          'bands' : [125, 250, 500, 1000, 2000, 4000],
          'bandsize' : 1,
          'order' : 4,
          'f_length': 16384,'power' : True}
    filterbank = Filterbank(**params)
    
    delay, early, rir = preprocess_rir(impulse, fs) # rir = late
    t = np.linspace(0, len(rir)/fs, len(rir)) # vector temporal

    fullband_params = curve_fit(t, rir) # Am, decay, noise_floor

    bands = filterbank.apply(rir)
    bands_aug = np.empty(bands.shape)

    for band in range(len(bands)):
        # obtain curve fit params
        band_params = curve_fit(t, bands[band,:])

        # generate estimated curves
        curve = curve_model(t, *band_params)

        # get noise floor onset
        noise_floor = 20 * np.log10(band_params[-1])
        try:
            noise_onset = np.where(curve < noise_floor + 5)[0][0]
        except:
            noise_onset = len(curve)-2000

        # merge to avoid noise floor 
        denoised = noise_crossfade(t, bands[band,:], band_params, noise_onset)

        # decay augmentation
        band_aug = augmentation(t, denoised, band_params, fullband_params[1] ,TR60_desired)
        bands_aug[band,:] = band_aug

    rir_aug = np.sum(bands_aug, axis=0)
    rir_aug = np.concatenate((delay, early, rir_aug))
    return rir_aug

############## DRR AUGMENTATION ############################

def get_DRR(rir, fs, window_length = 0.0025):
    """Dada una respuesta al impulso de entrada se calcula la
    relacion directo-reverberado que se define como el cociente
    en dB entre la parte temprana y la parte tardia de la respuesta
    al impulso. La separacion de parte temprana y tardia se realiza
    con un ventaneo temporal de 5 ms centrado en el punto de maxima
    amplitud del impulso.
    """
    t_d = np.argmax(rir) # direct path                                                     
    t_o = int((window_length) * fs) #tolerance window in samples                      
    init_idx = t_d - t_o
    final_idx = t_d + t_o + 1

    if init_idx < 0:
        init_idx = 0
    if final_idx > len(rir)-1:
        final_idx = len(rir)-1

    early= rir[init_idx:final_idx]
    late = rir[final_idx:]

    DRR = 10*np.log10((early**2).sum()/(late**2).sum())
    return DRR

def drr_aug(rir, fs, DRR_buscado, window_lenght=0.0025):
    """Realiza la generacion de una nueva respuesta al impulso con diferente
    valor de relacion directo-reverberado. El limite inferior queda determinado por 
    el valor maximo de la parte tardia del impulso. La parte tardia y temprana se dividen 
    con tolerancias temporales de 2.5 ms hacia ambos lados del valor maximo global, por
    convencion. Para aplicar la amplificacion o atenuacion de la parte temprana se utilizan
    ventanas de hamming, evitando la generacion de discontinuidades o artefactos.

    Parametros
    ------------------------------------------------------------------------
    path : string : path del audio a transformar
    DRR_buscado : float : valor de DRR resultante esperado

    Salidas
    ------------------------------------------------------------------------
    rir_aug : numpy array : secuencia numerica del nuevo impulso generado. corresponde
    a una fs de 16000.
    """
    t_d = np.argmax(rir) # direct path                                                     
    t_o = int((window_lenght) * fs) #tolerance window in samples                      
    init_idx = t_d - t_o
    final_idx = t_d + t_o + 1

    if init_idx < 0:
        init_idx = 0
    if final_idx > len(rir)-1:
        final_idx = len(rir)-1

    delay = rir[:init_idx]
    early= rir[init_idx:final_idx]
    late = rir[final_idx:]

    #Busco el coeficiente para llegar a la DRR deseada
    w = np.hamming((t_o*2)+1) #ventana de hamming de 5ms
    a = np.sum((w**2) * (early**2))
    b = 2 * np.sum((1-w)*w*(early**2))
    c = np.sum(((1-w)**2)*(early**2))-(np.power(10,DRR_buscado/10)*np.sum(late**2))
    alpha = bhaskara(a, b, c)
    #import pdb; pdb.set_trace()
    #Defino la nueva parte early
    new_early = (alpha * w * early) + ((1 - w)*early) 
    if np.max(abs(new_early))<np.max(abs(late)):
        #print("El nivel deseado es demasiado bajo")
        new_early = early

    #formo el nuevo impulso
    rir_aug = np.concatenate((delay, new_early, late), dtype='float32')

    DRR = 10*np.log10((new_early**2).sum()/(late**2).sum())
    #print("DRR buscado: {:0.2f}, DRR obtenido: {:0.2f}".format(DRR_buscado, DRR))
    return rir_aug/np.max(abs(rir_aug))

def bhaskara(a, b, c):
    r = b**2 - 4*a*c
    if r > 0:
        num_roots = 2
        x1 = (((-b) + np.sqrt(r))/(2*a))
        x2 = (((-b) - np.sqrt(r))/(2*a))
        return np.max((x1, x2))
    elif r == 0:
        num_roots = 1
        x = (-b) / 2*a
        return x
    else:
        num_roots = 0
        return

#Para rirs con ruido
def impulse_info(filterbank, rir, fs):
    # tiempo de reverberacion
    acoustics_rir = Acoustic_params(rir, fs)                                                                                     
    tr = acoustics_rir.reverberation_time(filterbank, plot = False)
    tr_mid = (tr[2]+tr[3])/2
    
    #Relacion directo-reverberado
    DRR = get_DRR(rir, fs)
    return tr_mid, DRR
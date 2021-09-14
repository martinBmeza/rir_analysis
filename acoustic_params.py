import matplotlib.pyplot as plt
import numpy as np
import librosa
from lundeby import lundeby
from scipy import stats

class Acoustic_params():
    ''' '''
    def __init__(self, impulse, fs):
        self.impulse = impulse
        self.fs = fs
        self.temporal_decompose()

    def temporal_decompose(self, tau=0.0025):
        rir = self.impulse / max(abs(self.impulse))

        t_d = np.argmax(abs(rir)) # direct path
        t_o = int(tau * self.fs) # tolerance window
        
        self.init_idx = t_d - t_o
        self.final_idx = t_d + t_o + 1
        
        if self.init_idx < 0:
            self.init_idx = 0
        if self.final_idx > len(rir)-1:
            self.final_idx = len(rir)-1

        # split response in delay - early - late 
        self.delay = rir[:self.init_idx]
        self.early = rir[self.init_idx:self.final_idx]
        self.late = rir[self.final_idx:]
        
        self.time = np.linspace(0, len(rir)/self.fs, len(rir))
        return


    def reverberation_time(self, filterbank):
        bands = filterbank.apply(self.late)
       
        init = 0
        end = -30
        factor = 2.0
        tr = []

        def plotear_banda(axs, signal, label):
                axs.plot(signal, label=label)
                axs.legend()

        fig, axs = plt.subplots(len(bands), figsize=(15,25))

        for i,band in enumerate(bands):
            noise_onset = lundeby(band, self.fs)
            filtered_signal = band[:noise_onset+1]
            abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))
            abs_completa = np.abs(band) / np.max(np.abs(band))
            
            # Schroeder integration
            sch = np.cumsum(abs_signal[::-1]**2)[::-1]
            sch_db = 10.0 * np.log10(sch / np.max(sch))
            
            #ploteos
            plotear_banda(axs[i], sch_db, 'Schroeder')
            plotear_banda(axs[i], librosa.amplitude_to_db(abs_completa), 'Completa')
            plotear_banda(axs[i], librosa.amplitude_to_db(abs_signal), 'Filtrada')
                    

            # Linear regression
            sch_init = sch_db[np.abs(sch_db - init).argmin()]
            sch_end = sch_db[np.abs(sch_db - end).argmin()]
            init_sample = np.where(sch_db == sch_init)[0][0]
            end_sample = np.where(sch_db == sch_end)[0][0]
            x = np.arange(init_sample, end_sample + 1) / self.fs
            y = sch_db[init_sample:end_sample + 1]
            slope, intercept = stats.linregress(x, y)[0:2]

            # Reverberation time (T30, T20, T10 or EDT)
            db_regress_init = (init - intercept) / slope
            db_regress_end = (end - intercept) / slope
            tr.append(factor * (db_regress_end - db_regress_init))
            #import pdb; pdb.set_trace()
        return tr  

import librosa 
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


class rir():
    """ Main class for the room impulse responses """

    def __init__(self,path):
        self.path = path
        self._load_impulso()


    def _load_impulso(self, tau=0.0025):
        rir, self.fs = librosa.load(self.path, sr=None)
        rir_norm = rir / max(abs(rir))

        t_d = np.argmax(abs(rir)) # direct path
        t_o = int(tau * self.fs) # tolerance window
        
        self.init_idx = t_d - t_o
        self.final_idx = t_d + t_o + 1
        
        if self.init_idx < 0:
            self.init_idx = 0
        if self.final_idx > len(rir)-1:
            self.final_idx = len(rir)-1

        # split response in delay - early - late 
        self.delay = rir_norm[:self.init_idx]
        self.early = rir_norm[self.init_idx:self.final_idx]
        self.late = rir_norm[self.final_idx:]
        
        self.time = np.linspace(0, len(rir_norm)/self.fs, len(rir_norm))
        return


    def plot_rir(self):
        fig, ax = plt.subplots(1, figsize=(10,5))
        ax.plot(self.time[:self.init_idx], self.delay, label='Delay')
        ax.plot(self.time[self.init_idx:self.final_idx], self.early, label='Esarly response')
        ax.plot(self.time[self.final_idx:], self.late, label='Late response')
        ax.legend() 
        plt.show()





if __name__ == '__main__':
    impulso = rir('files/rir.wav')

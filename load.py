import librosa 
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


class RIR():
    """ Main class for the room impulse responses """

    def __init__(self,path, fs=None):
        self.path = path
        self.fs = fs
        self._load_impulso()

    def _load_impulso(self, tau=0.0025):
        rir, self.fs = librosa.load(self.path, sr=self.fs, dtype='float64')
        self.rir = rir / max(abs(rir))
        



if __name__ == '__main__':
    impulso = RIR('files/rir.wav')

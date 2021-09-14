import matplotlib.pyplot as plt
from load import RIR
from acoustic_params import Acoustic_params
from filterbank import Filterbank

#[62.5, 78.745, 99.213, 125.0, 157.490, 198.425, 250.0, 314.980, 396.850, 500, 629.961, 793.701, 1000, 1259.921, 1587.401, 2000, 2519.842, 3174.802, 4000]

impulso = RIR('files/rir.wav', fs = 16000)
params = {'fs' : 16000,
              'bands' :[125, 250, 500, 1000, 2000, 4000],
              'bandsize' : 1,
              'order' : 4,
              'f_length': 16384,
              'power' : True}

filterbank = Filterbank(**params)
acoustics = Acoustic_params(impulso.rir, impulso.fs)
tr = acoustics.reverberation_time(filterbank)
plt.figure()
plt.semilogx(params['bands'],tr)
plt.xticks(params['bands'], ['125 Hz', '250 Hz', '500 Hz', '1 kHz', '2 kHz', '4 kHz'])
plt.ylim(0,5)
plt.show()

import numpy as np
from scipy.signal import butter, lfilter, fftconvolve

class Filterbank():
    """
    Complementary filterbank to split in octave/third bands

    Parameters:
    -----------
        fs : int
            Sample rate of the signal to be filtered
        bands : list
            list of central frequencies of the bands
        bandsize : int
            octave = 1, third octave = 3
        order : int
            order of the butterworth filters
        length : int
            length in samples of the filters
        power : bool
            To compute or not the power-complementary filter bank
    """

    def __init__(self, fs, bands, bandsize, order, f_length, power):
        self._valid_bands(fs, bands, bandsize)
        self._gen_filters(order, f_length, power)


    def _valid_bands(self, fs, bands, bandsize):
        bands = list(bands)
        bands.sort()
        fc = [band*np.power(2, bandsize/2) for band in bands] # upper cut freq for each band
        assert fc[-1] < (fs/2)
        self.fc = fc
        self.fs = fs


    def _gen_filters(self, order, f_length, power):
        numFilts = len(self.fc)
        nbins = f_length

        # impulse signal to ping the IIR filters
        signal_z1 = np.zeros(2 * nbins)
        signal_z1[0] = 1

        # allocate matrix for the filters
        irBands = np.zeros((2 * nbins, numFilts))

        for i in range(numFilts - 1):
            wc = self.fc[i] / (self.fs/2.0) # relative cut freq

            if wc >= 1:
                wc = .999999

            # split the spectrum in upper and lower bands. 
            # Butterworth prototypes for low pass and high pass filters
            B_low, A_low = butter(order, wc, btype='low')
            B_high, A_high = butter(order, wc, btype='high')

            # Ping with the impulse
            # Store the low band
            irBands[:, i] = lfilter(B_low, A_low, signal_z1)
            # Store the high. It will be further splitted next
            signal_z1 = lfilter(B_high, A_high, signal_z1)

            # Repeat for the last band of the filter bank

        irBands[:, -1] = signal_z1

        # Compute power complementary filters
        if power:
            ir2Bands = np.real(np.fft.ifft(np.square(np.abs(np.fft.fft(irBands, axis=0))), axis=0))
        else:
            ir2Bands = np.real(np.fft.ifft(np.abs(np.abs(np.fft.fft(irBands, axis=0))), axis=0))

        ir2Bands = np.concatenate((ir2Bands[nbins:(2 * nbins), :], ir2Bands[0:nbins, :]), axis=0)
        self.filters = ir2Bands


    def apply(self, signal):
        N = self.filters.shape[0] // 2 # filter final length
        filtered_bands = np.empty((len(self.fc), len(signal)))
        # apply each band
        for band, _ in enumerate(self.fc):
            cur_filt = self.filters[:,band]
            filtered = fftconvolve(signal, cur_filt)
            filtered = filtered[N:N+len(signal)]
            filtered_bands[band, :] = filtered
        
        return filtered_bands

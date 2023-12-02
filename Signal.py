import numpy as np


class Signal:
    def __init__(self, name):
        self.name = name
        self.data = []
        self.time = []
        self.sr = None

        self.data_after = []

        # mag and phase
        self.phase = None

        self.smoothing_window_name = 'Rectangle'

        # list of 2 lists freq, mag
        self.fft_data = None  # [[freq],[mag]]

        # list of tubles each one (start,end)
        self.slice_indices = []

        self.smooth_seg = []
        self.smooth_seg_amp = []
        self.each_slider_reference = np.repeat(100, 10)  # initially

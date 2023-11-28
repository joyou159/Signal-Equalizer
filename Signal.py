class Signal:
    def __init__(self, name):
        self.name = name
        self.data = []
        self.time = []
        self.sr = None

        # mag and phase
        self.phase = None

        self.smoothing_window_name = None

        # list of 2 lists freq, mag
        self.fft_data = None
        
        # list of tubles each one (start,end)
        self.slice_indices = []

        self.smooth_seg = []
        self.smooth_seg_amp = []

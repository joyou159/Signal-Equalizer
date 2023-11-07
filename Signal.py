class Signal:
    def __init__(self, name):
        self.name = name
        self.freq_components = []
        self.data = []
        self.time = []
        self.maxFreq = None
        self.components = []
        self.smoothing_window_name = None
        self.smoothing_window_data = None

    def add_component(self, component):
        self.components.append(component)

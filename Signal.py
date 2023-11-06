class Signal:
    def __init__(self, name):
        self.name = name
        self.freq_components = []
        self.data = []
        self.time = []
        self.maxFreq = None

    def add_component(self, component):
        self.components.append(component)

    def change_sample_rate(self, new_sample_rate):
        self.sample_rate = new_sample_rate
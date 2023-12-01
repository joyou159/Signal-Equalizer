from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import librosa.display
from scipy.signal import spectrogram


class MplWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = FigureCanvas(Figure(facecolor='none'))
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        self.canvas.axes = self.canvas.figure.add_subplot(111)

        # Set the color of the axes to white
        self.canvas.axes.tick_params(axis='both', colors='white')
        self.setLayout(vertical_layout)


    def plot_data(self, x_data, y_data, title='Plot', x_label='X-axis', y_label='Y-axis'):
        self.canvas.axes.clear()
        self.canvas.axes.plot(x_data, y_data, label='Data')
        self.canvas.axes.set_title(title)
        self.canvas.axes.set_xlabel(x_label)
        self.canvas.axes.set_ylabel(y_label)
        self.canvas.axes.legend()
        self.canvas.draw()


    def plot_audio_spectrogram(self, audio_data, sample_rate, title='Spectrogram', x_label='Time', y_label='Frequency'):
        self.canvas.axes.clear()
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='log', ax=self.canvas.axes)
        self.canvas.draw()


    def plot_ecg_spectrogram(self, ecg_data, time, title='ECG Spectrogram', x_label='Time', y_label='Frequency'):
        self.canvas.axes.clear()

        sample_rate = 1 / np.mean(np.diff(time))
        frequencies, times, Sxx = spectrogram(ecg_data, fs=sample_rate)

        # Plot the contour plot instead of pcolormesh
        contour = self.canvas.axes.contourf(times, frequencies, 10 * np.log10(Sxx), cmap='viridis', levels=100)
        
        # Add a colorbar for reference
        colorbar = self.canvas.figure.colorbar(contour, ax=self.canvas.axes, label='Power/Frequency [dB/Hz]')
        self.canvas.draw()


    def clear(self):
        self.canvas.axes.clear()
        self.canvas.draw()

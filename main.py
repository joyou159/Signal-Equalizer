from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFileDialog, QMessageBox, QColorDialog, QListWidgetItem, QPushButton, QSlider
from PyQt6.QtCore import Qt
import numpy as np
import sys
from PyQt6 import QtWidgets, uic
import pyqtgraph as pg
import qdarkstyle
import os
import csv
from pydub import AudioSegment
import math
from Signal import Signal
from scipy import signal
from scipy.fft import fft, fftshift
import librosa
from IPython.display import display, Audio

# pip install pyqtgraph pydub
# https://www.ffmpeg.org/download.html
# https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/
# setx PATH "C:\ffmpeg\bin;%PATH%"
# restart


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.init_ui()
        self.selected_function = None
        self.smooth_time = []
        self.smooth_data = []
        self.our_signal = None

    def rectangle_window(self, amplitude, freq, t):
        return amplitude * signal.square(2 * np.pi * freq * t)

    def hamming_window(self, N, amplitude):
        return amplitude * signal.windows.hamming(N)

    def hanning_window(self, N, amplitude):
        return amplitude * signal.windows.hann(N)

    def gaussian_window(self, N, amplitude, std):
        return amplitude * signal.windows.gaussian(N, std)

    def openNewWindow(self):
        self.setEnabled(False)

        self.new_window = QtWidgets.QMainWindow()
        uic.loadUi('SmoothingWindow.ui', self.new_window)
        self.new_window.functionList.addItem('Rectangle')
        self.new_window.functionList.addItem('Hamming')
        self.new_window.functionList.addItem('Hanning')
        self.new_window.functionList.addItem('Gaussian')
        self.new_window.freqSpinBox.setValue(1)
        self.new_window.ampSpinBox.setValue(1)
        self.new_window.stdSpinBox.setValue(1)
        self.new_window.samplesSpinBox.setValue(1)
        self.new_window.functionList.setCurrentIndex(0)
        self.handle_selected_function()

        self.new_window.freqSpinBox.valueChanged.connect(
            self.handle_selected_function)
        self.new_window.ampSpinBox.valueChanged.connect(
            self.handle_selected_function)
        self.new_window.stdSpinBox.valueChanged.connect(
            self.handle_selected_function)

        self.new_window.functionList.currentIndexChanged.connect(
            self.handle_selected_function)

        self.new_window.save.clicked.connect(self.save)

        self.new_window.show()
        self.new_window.destroyed.connect(self.onNewWindowClosed)

    def handle_selected_function(self):
        self.selected_function = self.new_window.functionList.currentText()

        if self.selected_function == 'Gaussian':
            self.new_window.samplesSpinBox.show()
            self.new_window.samples.show()
            self.new_window.stdSpinBox.show()
            self.new_window.std.show()
        elif self.selected_function == 'Rectangle':
            self.new_window.freqSpinBox.show()
            self.new_window.freq.show()
            self.new_window.stdSpinBox.hide()
            self.new_window.std.hide()
            self.new_window.samplesSpinBox.hide()
            self.new_window.samples.hide()
        else:
            self.new_window.samplesSpinBox.show()
            self.new_window.samples.show()
            self.new_window.freqSpinBox.hide()
            self.new_window.freq.hide()
            self.new_window.stdSpinBox.hide()
            self.new_window.std.hide()

        samples = int(self.new_window.samplesSpinBox.text())
        freq = int(self.new_window.freqSpinBox.text())
        amplitude = int(self.new_window.ampSpinBox.text())
        std = int(self.new_window.stdSpinBox.text())
        self.smooth_time = np.linspace(0, 1, 500, endpoint=False)

        if self.selected_function == 'Rectangle':
            self.smooth_data = self.rectangle_window(
                amplitude, freq, self.smooth_time)

        elif self.selected_function == 'Hamming':
            self.smooth_data = self.hamming_window(samples, amplitude)

        elif self.selected_function == 'Hanning':
            self.smooth_data = self.hanning_window(samples, amplitude)

        elif self.selected_function == 'Gaussian':
            self.smooth_data = self.gaussian_window(samples, amplitude, std)

        A = fft(self.smooth_data, 2048) / (len(self.smooth_data)/2.0)
        self.freq = np.linspace(-0.5, 0.5, len(A))
        self.response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))

        self.smoothing_real_time()

    def smoothing_real_time(self):
        self.new_window.smoothingGraph1.clear()

        self.new_window.smoothingGraph2.clear()

        self.new_window.smoothingGraph1.plot(
            self.smooth_data)
        self.new_window.smoothingGraph2.plot(
            self.freq, self.response)

    def save(self):
        self.our_signal.smoothing_window = self.selected_function
        self.our_signal.smoothing_window_data = self.smooth_data
        self.onNewWindowClosed()

    def onNewWindowClosed(self):
        self.new_window.close()
        self.setEnabled(True)

    def init_ui(self):
        # Load the UI Page
        self.ui = uic.loadUi('Mainwindow.ui', self)
        self.setWindowTitle("Signal Equlizer")
        self.graph_style_ui()

        self.ui.smoothingWindow.clicked.connect(self.openNewWindow)
        self.ui.browseFile.clicked.connect(self.browse)

        self.ui.modeList.setCurrentIndex(0)
        self.handle_combobox_selection()

        self.ui.modeList.currentIndexChanged.connect(
            self.handle_combobox_selection)
        self.ui.spectogramCheck.stateChanged.connect(self.show_spectrogram)

    def graph_style_ui(self):
        # Set the background of graph1 and graph2 to transparent
        self.ui.graph1.setBackground("transparent")
        self.ui.graph2.setBackground("transparent")

        # Set the background of spectogram1 and spectogram2 to transparent
        self.ui.spectogram1.setBackground("transparent")
        self.ui.spectogram2.setBackground("transparent")

        # Add items to the modeList widget
        self.ui.modeList.addItem("Uniform Range")
        self.ui.modeList.addItem("Musical Instruments")
        self.ui.modeList.addItem("Animal Sounds")
        self.ui.modeList.addItem("ECG Abnormalities")

    def show_spectrogram(self, state):
        if state == 2:  # Checked state
            self.ui.spectogram1.hide()  # Hide spectrogram1
            self.ui.spectogram2.hide()  # Hide spectrogram2

            # Remove the spectrogram graphs from the layout
            self.ui.spectrogramLayout.removeWidget(self.ui.spectogram1)
            self.ui.spectrogramLayout.removeWidget(self.ui.spectogram2)

        else:
            self.ui.spectogram1.show()  # Show spectrogram1
            self.ui.spectogram2.show()  # Show spectrogram2

            # Adjust the layout to make space for the spectrogram graphs
            self.ui.spectrogramLayout.addWidget(self.ui.spectogram1)
            self.ui.spectrogramLayout.addWidget(self.ui.spectogram2)


    def browse(self):
        file_filter = "Raw Data (*.csv *.wav *mp3)"
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, 'Open Signal File', './', filter=file_filter)

        if file_path:
            file_name = os.path.basename(file_path)
            self.open_file(file_path, file_name)

    def open_file(self, path: str, file_name: str):
        # Lists to store time and data
        time = []  # List to store time values
        data = []  # List to store data values

        # Extract the file extension (last 3 characters) from the path
        filetype = path[-3:]

        if filetype in ["wav", "mp3"]:
            # Read the audio file using librosa
            data, sample_rate = librosa.load(path)

            # Calculate the time data
            duration = librosa.get_duration(y=data, sr=sample_rate)
            time = np.linspace(0, duration, len(data))

        # Check if the file type is CSV
        if filetype == "csv":
            # Open the data file for reading ('r' mode)
            with open(path, 'r') as data_file:
                # Create a CSV reader object with a comma as the delimiter
                data_reader = csv.reader(data_file, delimiter=',')

                # Skip the first row
                next(data_reader)

                # Iterate through each row (line) in the data file
                for row in data_reader:

                    # Extract the time value from the first column (index 0)
                    time_value = float(row[0])

                    # Extract the amplitude value from the second column (index 1)
                    amplitude_value = float(row[1])

                    # Append the time and amplitude values to respective lists
                    time.append(time_value)
                    data.append(amplitude_value)

        # Create a Signal object with the file name without the extension
        self.our_signal = Signal(file_name[:-4])

        self.our_signal.data = data
        self.our_signal.time = time
        self.our_signal.sr = sample_rate
        self.data_split(self.our_signal)

        self.plot_temp(self.our_signal)
        self.show_audio(self.our_signal)

    def plot_temp(self, signal):

        if signal:
            self.ui.graph1.clear()

            # Create a plot item
            self.ui.graph1.setLabel('left', "Amplitude")
            self.ui.graph1.setLabel('bottom', "Time")

            # Initialize the time axis (assuming all signals have the same time axis)
            x_data = signal.time
            y_data = signal.data

            # Plot the mixed waveform
            pen = pg.mkPen(color=(64, 92, 245), width=2)
            plot_item = self.ui.graph1.plot(
                x_data, y_data, name=signal.name, pen=pen)

            # Check if there is already a legend and remove it
            if self.ui.graph1.plotItem.legend is not None:
                self.ui.graph1.plotItem.legend.clear()

            # Add a legend to the plot
            legend = self.ui.graph1.addLegend()
            legend.addItem(plot_item, name=signal.name)


    def show_audio(self, signal):
        audio, sample_rate = signal.data, signal.sr

        # Create an Audio widget to play the audio
        audio_widget = Audio(data=audio, rate=sample_rate)

        # Create a layout for the "beforeWidget" if it doesn't have one
        if not self.ui.beforeWidget.layout():
            layout = QVBoxLayout()
            self.ui.beforeWidget.setLayout(layout)
        else:
            layout = self.ui.beforeWidget.layout()

        # Add the audio_widget to the layout of "beforeWidget"
        layout.addWidget(audio_widget)

        # Display the Audio widget
        display(audio_widget)


    def add_sliders(self, num_sliders):
        # Clear any existing sliders from the widget
        for i in reversed(range(self.ui.slidersWidget.layout().count())):
            self.ui.slidersWidget.layout().itemAt(i).widget().setParent(None)

        # Create and add new sliders
        for _ in range(num_sliders):
            slider = QSlider()
            slider.setOrientation(Qt.Orientation.Vertical)  # Vertical orientation (1)
            self.ui.slidersWidget.layout().addWidget(slider)


    def handle_combobox_selection(self):
        current_index = self.ui.modeList.currentIndex()
        if current_index == 0:
            self.add_sliders(10)
        else:
            self.add_sliders(4)

    def data_split(self, signal):
        current_index = self.ui.modeList.currentIndex()
        if current_index == 0:
            num_slices = 10
            data_length = len(signal.data)
            slice_size = data_length // num_slices

            for i in range(num_slices):
                start = i * slice_size
                end = start + slice_size
                signal.add_component(signal.data[start:end])
        else:
            num_slices = 4
            data_length = len(signal.data)
            slice_size = data_length // num_slices

            for i in range(num_slices):
                start = i * slice_size
                end = start + slice_size
                signal.add_component(signal.data[start:end])


# @lru_cache(maxsize=128)
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
    main = MainWindow()
    main.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

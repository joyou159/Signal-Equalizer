from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QMessageBox, QColorDialog, QListWidgetItem, QPushButton, QSlider
from PyQt6.QtCore import Qt, QTimer
import numpy as np
import pandas as pd
import sys
from PyQt6 import QtWidgets, uic
import pyqtgraph as pg
import qdarkstyle
import os
from pydub import AudioSegment
import math
from Signal import Signal
from scipy import signal
from scipy.fft import fft, fftshift
import librosa
from IPython.display import display, Audio

# pyinstrument
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

        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.update_plot)
        # self.timer.start(1000)  # Update plot every 1 second (adjust this as needed)

    def rectangle_window(self, amplitude, N):
        return amplitude * signal.windows.boxcar(N)

    def hamming_window(self, N, amplitude):
        return amplitude * signal.windows.hamming(N)

    def hanning_window(self, N, amplitude):
        return amplitude * signal.windows.hann(N)

    def gaussian_window(self, N, amplitude, std):
        return amplitude * signal.windows.gaussian(N, std)

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec()

    def openNewWindow(self):
        if self.our_signal == None:
            self.show_error_message("Please select a signal first!")

        else:
            self.setEnabled(False)

            self.new_window = QtWidgets.QMainWindow()
            uic.loadUi('SmoothingWindow.ui', self.new_window)
            self.new_window.functionList.addItem('Rectangle')
            self.new_window.functionList.addItem('Hamming')
            self.new_window.functionList.addItem('Hanning')
            self.new_window.functionList.addItem('Gaussian')
            self.new_window.stdSpinBox.setValue(5)
            self.new_window.stdSpinBox.setRange(0, 1000)
            self.new_window.functionList.setCurrentIndex(0)
            self.handle_selected_function()

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
            self.new_window.stdSpinBox.show()
            self.new_window.std.show()
        else:
            self.new_window.stdSpinBox.hide()
            self.new_window.std.hide()

        self.smoothing_real_time()

    def custom_window(self, samples, amplitude, std):
        if self.selected_function == 'Rectangle':
            smooth_data = self.rectangle_window(
                amplitude, samples)

        elif self.selected_function == 'Hamming':
            smooth_data = self.hamming_window(samples, amplitude)

        elif self.selected_function == 'Hanning':
            smooth_data = self.hanning_window(samples, amplitude)

        elif self.selected_function == 'Gaussian':
            smooth_data = self.gaussian_window(samples, amplitude, std)
        return smooth_data

    def smoothing_real_time(self):
        self.new_window.smoothingGraph1.clear()

        self.new_window.smoothingGraph1.plot(
            self.our_signal.fft_data[0], self.our_signal.fft_data[1])

        for i in range(self.our_signal.frequency_range_splits.shape[1] + 1):
            std = int(self.new_window.stdSpinBox.text())

            if i != 10:
                pos = self.our_signal.frequency_range_splits[0][i]

                current_segment_smooth_window = self.custom_window(
                    len(self.our_signal.frequency_range_splits), max(self.our_signal.amplitude_splits[:, i]), std)

                # Assuming self.new_window.smoothingGraph1 is a PlotItem
                self.new_window.smoothingGraph1.plot(
                    self.our_signal.frequency_range_splits[:, i],
                    current_segment_smooth_window,
                    pen={'color': 'b', 'width': 2}  # 'b' stands for blue color
                )

            else:
                pos = self.our_signal.frequency_range_splits[-1][i-1]

            v_line = pg.InfiniteLine(pos=pos, angle=90, movable=False)
            self.new_window.smoothingGraph1.addItem(v_line)

    def save(self):
        self.our_signal.smoothing_window_name = self.selected_function
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
        file_filter = "Raw Data (*.csv *.wav *.mp3)"
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, 'Open Signal File', './', filter=file_filter)

        if file_path:
            file_name = os.path.basename(file_path)
            self.open_file(file_path, file_name)

    def open_file(self, path: str, file_name: str):
        data = []
        time = []
        sample_rate = 0

        # Extract the file extension
        filetype = path.split('.')[-1]

        if filetype in ["wav", "mp3"]:
            data, sample_rate = librosa.load(path)
            duration = librosa.get_duration(y=data, sr=sample_rate)
            time = np.linspace(0, duration, len(data))

        elif filetype == "csv":
            data_reader = pd.read_csv(
                path, delimiter=',', skiprows=1)  # Skip header row
            time = data_reader.iloc[:, 0].astype(float).tolist()
            data = data_reader.iloc[:, 1].astype(float).tolist()

        signal = Signal(file_name[:-4])
        signal.data = data
        signal.time = time
        signal.sr = sample_rate
        sample_interval = 1 / signal.sr
        x, y = self.get_fft_values(signal, sample_interval, len(signal.data))
        signal.fft_data = np.array([x, y])

        self.process_signal(signal)

        self.our_signal = signal

    def get_fft_values(self, signal, T, N):

        f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
        # we have considered half of the sampled data points due to symmetry nature of FFT
        fft_values = np.fft.fft(signal.data, N)  # complex coefficients of fft
        fft_values = (2/N) * np.abs(fft_values[:N//2])
        return f_values, fft_values

    # def freq_comp(self, signal, sample_rate):
    #     fft_result = fft(signal.data)
    #     # Calculate the frequencies corresponding to the FFT result
    #     frequencies = np.fft.fftfreq(len(fft_result), 1 / sample_rate)
    #     return frequencies, fft_result

    def process_signal(self, signal):
        # self.display_audio(signal)
        self.split_data(signal)
        self.plot_signal(signal)

    def plot_signal(self, signal):
        if signal:
            self.ui.graph1.clear()
            self.ui.graph1.setLabel('left', "Amplitude")
            self.ui.graph1.setLabel('bottom', "Time")

            # Accumulate data for all columns
            x_values, y_values = [], []
            for i in range(signal.frequency_range_splits.shape[1]):
                x_values.extend(signal.frequency_range_splits[:, i])
                y_values.extend(signal.amplitude_splits[:, i])

            # Plot all columns together
            plot_item = self.ui.graph1.plot(
                x_values, y_values, name=f"{signal.name}", pen=(64, 92, 245))

            # Check if there is already a legend and remove it
            if self.ui.graph1.plotItem.legend is not None:
                self.ui.graph1.plotItem.legend.clear()

            # Add a legend to the plot
            legend = self.ui.graph1.addLegend()
            legend.addItem(plot_item, name=f"{signal.name}")

    def split_data(self, signal):
        num_slices = 10 if self.ui.modeList.currentIndex() == 0 else 4
        x = signal.fft_data[0]
        y = signal.fft_data[1]
        x_slices = np.array_split(x, num_slices)
        y_slices = np.array_split(y, num_slices)

        # Trim the values to make sure all columns have the same size
        min_length = min(len(slice_) for slice_ in x_slices)
        x_slices = [slice_[:min_length] for slice_ in x_slices]
        y_slices = [slice_[:min_length] for slice_ in y_slices]

        # Convert lists of arrays to 2D arrays
        signal.frequency_range_splits = np.array(
            x_slices).T  # Transpose to have columns
        signal.amplitude_splits = np.array(y_slices).T

        print(signal.frequency_range_splits.shape)

    def display_audio(self, signal):
        audio_widget = QWidget()
        layout = self.ui.beforeWidget.layout() or QVBoxLayout()
        audio = signal.data
        sample_rate = signal.sr
        audio_widget.setLayout(layout)
        audio_widget.layout().addWidget(Audio(data=audio, rate=sample_rate))
        layout.addWidget(audio_widget)
        display(audio_widget)

    def add_sliders(self, num_sliders):
        layout = self.ui.slidersWidget.layout()
        if layout:
            for i in reversed(range(layout.count())):
                layout.itemAt(i).widget().setParent(None)
        for _ in range(num_sliders):
            slider = QSlider(Qt.Orientation.Vertical)
            layout.addWidget(slider)

    def handle_combobox_selection(self):
        current_index = self.ui.modeList.currentIndex()
        num_sliders = 10 if current_index == 0 else 4
        self.add_sliders(num_sliders)


def main():
    app = QtWidgets.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

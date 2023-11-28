from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QMessageBox, QColorDialog, QListWidgetItem, QPushButton, QSlider
from PyQt6.QtCore import Qt ,QTimer
from PyQt6.QtGui import  QIcon
import numpy as np
import pandas as pd
import sys
from PyQt6 import QtWidgets, uic
import pyqtgraph as pg
import qdarkstyle
import os
from Signal import Signal
from scipy import signal
from scipy.fft import fft, fftshift
import librosa
from IPython.display import display, Audio as IPyAudio
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from mplwidget import MplWidget

matplotlib.use("QtAgg")

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
        self.graph_load_ui()

        # Other Attributes
        self.speed = 50
        self.end_ind = 50
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(lambda: self.updating_graphs(self.signal))

        self.ui.playPause.clicked.connect(self.play_pause)
        self.ui.zoomIn.clicked.connect(self.zoom_in)
        self.ui.zoomOut.clicked.connect(self.zoom_out)
        self.ui.resetButton.clicked.connect(self.reset)

        # Set speed slider properties
        self.speedSlider.setMinimum(0)
        self.speedSlider.setMaximum(200)
        self.speedSlider.setSingleStep(5)
        self.speedSlider.setValue(self.speed)
        self.speedSlider.valueChanged.connect(self.change_speed)


        self.ui.smoothingWindow.clicked.connect(self.openNewWindow)
        self.ui.browseFile.clicked.connect(self.browse)

        self.ui.modeList.setCurrentIndex(0)
        self.handle_combobox_selection()

        self.ui.modeList.currentIndexChanged.connect(
            self.handle_combobox_selection)
        
        self.ui.spectogramCheck.stateChanged.connect(self.show_spectrogram)

    def set_icon(self, icon_path):
        # Load an icon
        icon = QIcon(icon_path)
        # Set the icon for the button
        self.playPause.setIcon(icon)

    def graph_load_ui(self):

        # Create an instance of MplWidget for spectrogram
        self.spectrogram_widget1 = MplWidget()
        self.spectrogram_widget2 = MplWidget()

        # Create a QVBoxLayout for the QWidget
        spectrogram_layout1 = QVBoxLayout(self.ui.spectogram1)
        spectrogram_layout1.addWidget(self.spectrogram_widget1)

        spectrogram_layout2 = QVBoxLayout(self.ui.spectogram2)
        spectrogram_layout2.addWidget(self.spectrogram_widget2)

        # Set the background of graph1 and graph2 to transparent
        self.ui.graph1.setBackground("transparent")
        self.ui.graph2.setBackground("transparent")

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
            data_reader = pd.read_csv(path, delimiter=',', skiprows=1)  # Skip header row
            time = data_reader.iloc[:, 0].astype(float).tolist()  
            data = data_reader.iloc[:, 1].astype(float).tolist()  

        self.signal = Signal(file_name[:-4])
        self.signal.data = data
        self.signal.time = time
        self.signal.sr = sample_rate
        self.data_x = []
        self.data_y = []
        self.process_signal(self.signal)


    def process_signal(self, signal):
        self.plot_signal(signal)
        self.plot_spectrogram(signal)
        # self.display_audio(signal)
        self.split_data(signal)


    def plot_signal(self, signal):
        if signal:
            self.ui.graph1.clear()
            self.ui.graph1.setLabel('left', "Amplitude")
            self.ui.graph1.setLabel('bottom', "Time")
            self.data_x = signal.time[:self.end_ind]
            self.data_y = signal.data[:self.end_ind]
            self.plot_item = self.ui.graph1.plot(self.data_x, self.data_y, name=signal.name, pen=(64, 92, 245))

            # Check if there is already a legend and remove it
            if self.ui.graph1.plotItem.legend is not None:
                self.ui.graph1.plotItem.legend.clear()

            # Add a legend to the plot
            legend = self.ui.graph1.addLegend()
            legend.addItem(self.plot_item, name=signal.name)
            self.set_icon("icons/pause-square.png")
            self.ui.playPause.setText("Pause")
        
        if not self.timer.isActive():
            self.timer.start(50)


    def updating_graphs(self, signal):
            data , time = signal.data, signal.time

            data_X = time[:self.end_ind  + self.speed]
            data_Y = data[:self.end_ind  + self.speed]
            self.end_ind += self.speed

            if (data_X[-1] < 1):
                self.ui.graph1.setXRange(0 , 1)
            else:
                self.ui.graph1.setXRange(data_X[-1] - 1, data_X[-1])

            self.plot_item.setData(data_X, data_Y, visible=True)
            # self.ui.graph1.setLimits(
            #     xMin=0, xMax=self.end_ind, yMin=y_min-0.3, yMax=y_max+0.3)
            # self.ui.graph1.autoRange()


    def play_pause(self):
            if self.timer.isActive():
                self.timer.stop()
                self.set_icon("icons/play-square-svgrepo-com.png")
                self.ui.playPause.setText("Play")
            else:
                self.set_icon("icons/pause-square.png")
                self.ui.playPause.setText("Pause")
                self.timer.start()

    def zoom_in(self):
            view_box = self.graph1.plotItem.getViewBox()
            view_box.scaleBy((0.5, 1))

    def zoom_out(self):
            view_box = self.graph1.plotItem.getViewBox()
            view_box.scaleBy((1.5, 1))

    def change_speed(self):
        self.speed = self.speedSlider.value()

    def reset(self):
        msg_box = QMessageBox()

        msg_box.setText("Do you want to clear the graph?")
        msg_box.setWindowTitle("Clear Graph")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

        result = msg_box.exec()

        if result == QMessageBox.StandardButton.Ok:
            self.ui.graph1.clear()


    def plot_spectrogram(self, signal):
        if signal:
            self.spectrogram_widget1.plot_spectrogram(
                signal.data, signal.sr, title='Spectrogram', x_label='Time', y_label='Frequency')


    def split_data(self, signal):
        num_slices = 10 if self.ui.modeList.currentIndex() == 0 else 4
        data_length = len(signal.data)
        slice_size = data_length // num_slices

        signal.components = [signal.data[i * slice_size:(i + 1) * slice_size] for i in range(num_slices)]

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
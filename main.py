import bisect
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QMessageBox, QColorDialog, QListWidgetItem, QPushButton, QSlider
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon
import numpy as np
import pandas as pd
import sys
from PyQt6 import QtWidgets, uic
import pyqtgraph as pg
import qdarkstyle
import os
from Signal import Signal
from scipy import signal
from scipy.fft import fft, fftshift, ifft
import librosa
from IPython.display import display, Audio as IPyAudio
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from mplwidget import MplWidget
from audioWidget import AudioWidget
import sounddevice as sd
from functools import partial


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
        self.selected_function = "Rectangle"  # window function
        self.our_signal = None  # The current signal
        self.activation = 'uniform'  # the mode of operation (default)
        self.current_slider = None
        self.ecg_flag = False
        self.excess = None

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
            # disable the interactivity in the main window
            self.setEnabled(False)
            self.new_window = QtWidgets.QMainWindow()

            uic.loadUi('SmoothingWindow.ui', self.new_window)
            self.new_window.functionList.addItem(
                'Rectangle')  # the default window function
            self.new_window.functionList.addItem('Hamming')
            self.new_window.functionList.addItem('Hanning')
            self.new_window.functionList.addItem('Gaussian')

            self.new_window.functionList.setCurrentText(
                self.our_signal.smoothing_window_name)
            print(self.our_signal.smoothing_window_name)

            # spine box of the standard deviation of the gaussian window function
            self.new_window.stdSpinBox.setValue(5)
            self.new_window.stdSpinBox.setRange(0, 1000)
            self.handle_selected_function()  # ??

            self.new_window.stdSpinBox.valueChanged.connect(
                self.handle_selected_function)

            self.new_window.functionList.currentIndexChanged.connect(
                self.handle_selected_function)

            self.new_window.save.clicked.connect(self.save)

            self.new_window.show()
            self.new_window.destroyed.connect(self.onNewWindowClosed)

    def handle_selected_mode(self):
        mode = self.ui.modeList.currentIndex()
        if mode == 0:
            self.activation = 'uniform'
        elif mode == 1:
            self.activation = 'music'
        elif mode == 2:
            self.activation = 'animal'
        else:
            self.activation = 'ecg'

    def handle_selected_function(self):
        # why ?? (what about smoothing_window_name in the signal class)

        self.selected_function = self.new_window.functionList.currentText()

        # Due to possible truncation

        if self.selected_function == 'Gaussian':  # layout settings
            self.new_window.stdSpinBox.show()
            self.new_window.std.show()
        else:
            self.new_window.stdSpinBox.hide()
            self.new_window.std.hide()

        self.smoothing_real_time()

    def custom_window(self, samples, amplitude):
        if self.selected_function == 'Rectangle':
            smooth_data = self.rectangle_window(
                amplitude, samples)

        elif self.selected_function == 'Hamming':
            smooth_data = self.hamming_window(samples, amplitude)

        elif self.selected_function == 'Hanning':
            smooth_data = self.hanning_window(samples, amplitude)

        elif self.selected_function == 'Gaussian':
            std = int(self.new_window.stdSpinBox.text())
            smooth_data = self.gaussian_window(samples, amplitude, std)
        return smooth_data

    def smoothing_real_time(self):
        _, last_item = self.our_signal.slice_indices[-1]
        self.new_window.smoothingGraph1.clear()
        self.new_window.smoothingGraph1.plot(
            self.our_signal.fft_data[0][:last_item], self.our_signal.fft_data[1][:last_item])

        for i in range(len(self.our_signal.slice_indices)):
            start, end = self.our_signal.slice_indices[i]
            if self.activation == "uniform":
                if i == len(self.our_signal.slice_indices) - 1:
                    self.segment(start, end-1, sparse=False)
                else:
                    self.segment(start, end, sparse=False)
            else:

                if i == len(self.our_signal.slice_indices) - 1:
                    self.segment(start, end-1, sparse=True)
                else:
                    self.segment(start, end, sparse=True)

            current_segment_smooth_window = self.fill_smooth_segments(i)

            self.new_window.smoothingGraph1.plot(
                self.our_signal.fft_data[0][start:end],
                current_segment_smooth_window, pen={'color': 'b', 'width': 2})

    def segment(self, start, end, sparse=False):
        if sparse:
            start_freq = self.our_signal.fft_data[0][start]
            v_line = pg.InfiniteLine(pos=start_freq, angle=90, movable=False)
            self.new_window.smoothingGraph1.addItem(v_line)
        end_freq = self.our_signal.fft_data[0][end]
        v_line = pg.InfiniteLine(pos=end_freq, angle=90, movable=False)
        self.new_window.smoothingGraph1.addItem(v_line)


    def fill_smooth_segments(self, i):
        start, end = self.our_signal.slice_indices[i]
        if self.our_signal.smooth_seg_amp == [] or self.current_slider == None:
            amp = max(self.our_signal.fft_data[1][start:end])
            self.our_signal.smooth_seg_amp.append(amp)
            current_segment_smooth_window = self.custom_window(
                len(self.our_signal.fft_data[0][start:end]), amp)
            self.our_signal.smooth_seg.append(current_segment_smooth_window)
            print("initialize")

        else:
            if i == self.current_slider:
                amp = (max(self.our_signal.fft_data[1][start:end]) /
                       self.our_signal.smooth_seg_amp[i])
            else:
                amp = max(self.our_signal.fft_data[1][start:end])
            current_segment_smooth_window = self.custom_window(
                len(self.our_signal.fft_data[0][start:end]), amp)
            print("update")
        return current_segment_smooth_window


    def save(self):
        self.our_signal.smoothing_window_name = self.selected_function

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
        self.speed = 200
        self.end_ind = 50
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.updating_graphs)
        # self.timer.timeout.connect(self.plot_spectrogram)

        self.ui.playPause.clicked.connect(self.play_pause)
        self.ui.zoomIn.clicked.connect(self.zoom_in)
        self.ui.zoomOut.clicked.connect(self.zoom_out)
        self.ui.resetButton.clicked.connect(self.reset)
        self.ui.playAudio1.clicked.connect(partial(self.toggle_audio, self.audio_widget1, self.ui.playAudio1, "icons/pause-square.png"))
        self.ui.playAudio2.clicked.connect(partial(self.toggle_audio, self.audio_widget2, self.ui.playAudio2, "icons/pause-square.png"))


        # Set speed slider properties
        self.speedSlider.setMinimum(0)
        self.speedSlider.setMaximum(500)
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

    def set_icon(self, widget_name, icon_path):
        # Load an icon
        icon = QIcon(icon_path)
        # Set the icon for the button
        widget_name.setIcon(icon)

    def graph_load_ui(self):

        # Create an instance of MplWidget for spectrogram
        self.spectrogram_widget1 = MplWidget()
        self.spectrogram_widget2 = MplWidget()

        # Create a QVBoxLayout for the QWidget
        spectrogram_layout1 = QVBoxLayout(self.ui.spectogram1)
        spectrogram_layout1.addWidget(self.spectrogram_widget1)

        spectrogram_layout2 = QVBoxLayout(self.ui.spectogram2)
        spectrogram_layout2.addWidget(self.spectrogram_widget2)

        # Create an instance of the AudioWidget
        self.audio_widget1 = AudioWidget()

        # Set up layout for the audio1 widget
        audio_layout1 = QVBoxLayout(self.audio1)
        audio_layout1.addWidget(self.audio_widget1)
    
        # Create an instance of the AudioWidget
        self.audio_widget2 = AudioWidget()

        # Set up layout for the audio1 widget
        audio_layout2 = QVBoxLayout(self.audio2)
        audio_layout2.addWidget(self.audio_widget2)

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
            self.ecg_flag = True
            data_reader = pd.read_csv(
                path, delimiter=',', skiprows=1)  # Skip header row
            time = np.array(data_reader.iloc[:, 0].astype(float).tolist())
            data = np.array(data_reader.iloc[:, 1].astype(float).tolist())
            sample_rate = 1 / np.mean(np.diff(time))
            self.speed = 3

        self.our_signal = Signal(file_name[:-4])
        self.our_signal.data = data
        self.our_signal.time = time
        self.our_signal.sr = sample_rate
        sample_interval = 1 / self.our_signal.sr
        self.our_signal.smoothing_window_name = self.selected_function
        self.our_signal.data_x = []
        self.our_signal.data_y = []

        x, y = self.get_fft_values(sample_interval, len(self.our_signal.data))
        self.our_signal.fft_data = [x, y]
        self.our_signal.data_after = data

        self.process_signal()

    def get_fft_values(self, T, N):
        f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
        # we have considered half of the sampled data points due to symmetry nature of FFT
        # complex coefficients of fft
        fft_values = np.fft.fft(self.our_signal.data, N)

        self.our_signal.phase = np.angle(fft_values[:N//2])

        fft_values = (2/N) * np.abs(fft_values[:N//2])

        return f_values, fft_values

    def get_inverse_fft_values(self):
        modified_fft_values = np.array(self.our_signal.fft_data[1]) * len(self.our_signal.data)/2 * \
            np.exp(1j * self.our_signal.phase)

        reconstructed_signal = np.fft.irfft(modified_fft_values)

        return reconstructed_signal

    def process_signal(self):
        self.handle_selected_mode()
        self.split_data()
        self.plot_signal()
        self.plot_spectrogram()
        # self.display_audio()

    def plot_signal(self):
        if self.our_signal:
            self.ui.graph1.clear()
            # self.ui.graph1.setLabel('left', "Amplitude")
            # self.ui.graph1.setLabel('bottom', "Time")

            data_x = self.our_signal.time[:self.end_ind]
            data_y_before = self.our_signal.data[:self.end_ind]
            self.plot_item_before = self.ui.graph1.plot(
                data_x, data_y_before, name=self.our_signal.name, pen=(64, 92, 245))

            # Check if there is already a legend and remove it
            if self.ui.graph1.plotItem.legend is not None:
                self.ui.graph1.plotItem.legend.clear()

            # Add a legend to the plot
            legend = self.ui.graph1.addLegend()
            legend.addItem(self.plot_item_before, name=self.our_signal.name)
################################################################################################

            data_y_after = self.our_signal.data_after[:self.end_ind]
            self.plot_item_after = self.ui.graph2.plot(
                data_x, data_y_after, name=self.our_signal.name, pen=(64, 92, 245))

            # Check if there is already a legend and remove it
            if self.ui.graph2.plotItem.legend is not None:
                self.ui.graph2.plotItem.legend.clear()

            # Add a legend to the plot
            legend = self.ui.graph2.addLegend()
            legend.addItem(self.plot_item_after, name=self.our_signal.name)

            self.set_icon(self.ui.playPause, "icons/pause-square.png")
            self.ui.playPause.setText("Pause")

        if not self.timer.isActive():
            self.timer.start(50)

    def updating_graphs(self):
        data_before, data_after, time = self.our_signal.data, self.our_signal.data_after, self.our_signal.time

        data_X = time[:self.end_ind + self.speed]
        data_Y_before = data_before[:self.end_ind + self.speed]
        data_Y_after = data_after[:self.end_ind + self.speed]

        self.end_ind += self.speed

        if (data_X[-1] < 1):
            self.ui.graph1.setXRange(0, 1)
            self.ui.graph2.setXRange(0, 1)
        else:
            self.ui.graph1.setXRange(data_X[-1] - 1, data_X[-1])
            self.ui.graph2.setXRange(data_X[-1] - 1, data_X[-1])

        self.plot_item_before.setData(data_X, data_Y_before, visible=True)
        if len(data_X) == len(data_Y_after):
            self.plot_item_after.setData(
                data_X, data_Y_after, visible=True)
        else:
            self.excess = len(data_X) - len(data_Y_after)
            self.plot_item_after.setData(
                data_X[:-self.excess], data_Y_after, visible=True)

    def play_pause(self):
        if self.our_signal:
            if self.timer.isActive():
                self.timer.stop()
                self.set_icon(self.ui.playPause,
                              "icons/play-square-svgrepo-com.png")
                self.ui.playPause.setText("Play")
                # self.ui.graph1.autoRange()
            else:
                self.set_icon(self.ui.playPause, "icons/pause-square.png")
                self.ui.playPause.setText("Pause")
                self.timer.start()

    def zoom_in(self):
        view_box = self.graph1.plotItem.getViewBox()
        view_box.scaleBy((0.5, 1))

    def zoom_out(self):
        view_box = self.graph1.plotItem.getViewBox()
        view_box.scaleBy((1.5, 1))

    def change_speed(self):
        if self.our_signal:
            self.speed = self.speedSlider.value()

    def reset(self):
        if self.our_signal:
            msg_box = QMessageBox()

            msg_box.setText("Do you want to clear the graph?")
            msg_box.setWindowTitle("Clear Graph")
            msg_box.setStandardButtons(
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

            result = msg_box.exec()

        if result == QMessageBox.StandardButton.Ok:
            self.ui.graph1.clear()
            self.spectrogram_widget1.clear()

    def plot_spectrogram(self):
        if self.our_signal:
            if self.ecg_flag:
                # plot spectrogram of ecg data
                self.spectrogram_widget1.plot_ecg_spectrogram(
                    self.our_signal.data, self.our_signal.time)

            # plot spectrogram of audio data
            self.spectrogram_widget1.plot_audio_spectrogram(
                self.our_signal.data, self.our_signal.sr)
            
            self.spectrogram_widget2.plot_audio_spectrogram(
                self.our_signal.data_after, self.our_signal.sr)
            

    def toggle_audio(self, audio_widget, play_button, icon_path):
        if self.our_signal:
            if not audio_widget.playing:
                audio_widget.play_audio(self.our_signal.data, self.our_signal.sr)
                play_button.setText("Pause Audio")
                self.set_icon(play_button, icon_path)
            else:
                sd.stop()
                play_button.setText("Play Audio")
                self.set_icon(play_button, "icons/play-square-svgrepo-com.png")

    # searching , dont forget it

    def find_closest_index(self, array, target):
        """Find the index of the closest value in the array to the target."""
        index = bisect.bisect_left(array, target)
        if index == 0:
            return 0
        if index == len(array):
            return len(array) - 1
        before = array[index - 1]
        after = array[index]
        if after - target < target - before:
            return index
        else:
            return index - 1

    def split_data(self):
        # round the frequencies
        # self.our_signal.fft_data[0] = [round(
        #     self.our_signal.fft_data[0][i]) for i in range(len(self.our_signal.fft_data[0]))]
        if self.activation == 'uniform':
            num_slices = 10
            excess_elements = len(self.our_signal.fft_data[0]) % num_slices
            if excess_elements:
                self.our_signal.data = self.our_signal.data[:-excess_elements]
                self.our_signal.time = self.our_signal.time[:-excess_elements]
                self.our_signal.fft_data[0] = self.our_signal.fft_data[0][:-excess_elements]
                self.our_signal.fft_data[1] = self.our_signal.fft_data[1][:-excess_elements]
            slice_size = int(len(self.our_signal.fft_data[0])/num_slices)
            self.our_signal.slice_indices = [
                (i * slice_size, (i + 1) * slice_size) for i in range(num_slices)]
        elif self.activation == 'music':

            ranges = [(0, 150), (150, 600), (600, 800), (800, 1200)]

            # Assuming self.our_signal.fft_data[0] contains the frequency values
            frequencies = self.our_signal.fft_data[0]
            for start, end in ranges:
                start_index = self.find_closest_index(frequencies, start)
                end_index = self.find_closest_index(frequencies, end)
                self.our_signal.slice_indices.append((start_index, end_index))

        elif self.activation == 'animal':
            ranges = [(400, 420), (600, 700), (1300, 1600), (3000, 4000)]

            # Assuming self.our_signal.fft_data[0] contains the frequency values
            frequencies = self.our_signal.fft_data[0]
            for start, end in ranges:
                start_index = self.find_closest_index(frequencies, start)
                end_index = self.find_closest_index(frequencies, end)
                self.our_signal.slice_indices.append((start_index, end_index))

    def add_sliders(self, num_sliders):
        layout = self.ui.slidersWidget.layout()
        if layout:
            for i in reversed(range(layout.count())):
                layout.itemAt(i).widget().setParent(None)
        for i in range(num_sliders):
            slider = QSlider(Qt.Orientation.Vertical)
            slider.setValue(50)
            slider.setSingleStep(1)
            slider.setRange(1, 200)
            layout.addWidget(slider)

    def handle_combobox_selection(self):
        current_index = self.ui.modeList.currentIndex()
        num_sliders = 10 if current_index == 0 else 4
        self.add_sliders(num_sliders)
        sliders = self.ui.slidersWidget.findChildren(QSlider)
        for slider in sliders:
            slider.valueChanged.connect(
                lambda slider_value=(slider.value()), slidernum=sliders.index(slider): self.editing(slider_value, slidernum))

    def editing(self, slider_value, slidernum):
        start, end = self.our_signal.slice_indices[slidernum]
        mag_fft = np.array(self.our_signal.fft_data[1][start:end])
        freq_values = np.array(self.our_signal.fft_data[0][start:end])
        self.current_slider = slidernum
        factor_of_multiplication = slider_value / \
            self.our_signal.each_slider_reference[slidernum]
        smooth_data = (factor_of_multiplication) * np.array(
            self.our_signal.smooth_seg[slidernum] / self.our_signal.smooth_seg_amp[slidernum])
        result = mag_fft * smooth_data
        self.our_signal.smooth_seg[slidernum] = smooth_data
        self.our_signal.fft_data[1][start:end] = result
        self.our_signal.smooth_seg_amp[slidernum] = (factor_of_multiplication)
        self.our_signal.each_slider_reference[slidernum] = slider_value

        self.our_signal.data_after = self.get_inverse_fft_values()

        self.spectrogram_widget2.plot_audio_spectrogram(
            self.our_signal.data_after, self.our_signal.sr)


def main():
    app = QtWidgets.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

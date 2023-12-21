from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QVBoxLayout,  QMessageBox,  QSlider, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon
from PyQt6 import QtWidgets, uic
import numpy as np
import pandas as pd
import sys
import pyqtgraph as pg
import qdarkstyle
import os
from scipy import signal
import librosa
import matplotlib
import sounddevice as sd
from functools import partial
import bisect

from Signal import Signal
from mplwidget import MplWidget
from audioWidget import AudioWidget

matplotlib.use("QtAgg")

# pyinstrument
# pip install pyqtgraph pydub
# https://www.ffmpeg.org/download.html
# https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/
# setx PATH "C:\ffmpeg\bin;%PATH%"
# restart


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        """
        Initializes the MainWindow class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        super(MainWindow, self).__init__(*args, **kwargs)

        # Initialize instance variables
        self.init_ui()
        self.selected_function = None  # Window function
        self.our_signal = None  # The current signal
        self.activation = 'uniform'  # The mode of operation (default)
        self.current_slider = None
        self.pause_flag = False
        self.excess = None
        self.ranges = [list(np.repeat(100, 10)),
                       #    [(20, 70), (60, 80), (270, 400),(200, 290)] for another file music2
                       # Guitar  ,  Flute  ,  Harmonica  ,   Xylophone
                       [(0, 170), (170, 250), (250, 400), (400, 1000)],
                       # Dogs   ,    Wolves    ,   Crow    ,     Bat
                       [(0, 450), (450, 1100), (1100, 3000), (3000, 9000)],
                       # (ِ[/]) , (Ventricular tachycardia AND Ventricular couplets) , Ventricular couplets (only),
                       [(0, 6.5), (0, 5), (0, 8)]
                       ]
        self.sparse_state = [False, True, True, True]

####################################### Helper Functions ################################## ######
    def show_error_message(self, message):
        """
        Displays an error message to the user.

        Args:
            message (str): The error message to be displayed.

        Returns:
            None
        """
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec()

    def set_icon(self, widget_name, icon_path):
        """
        Set the icon for a widget.

        Args:
            widget_name (str): The name of the widget.
            icon_path (str): The path to the icon file.

        Returns:
            None
        """
        # Load an icon
        icon = QIcon(icon_path)
        # Set the icon for the button
        widget_name.setIcon(icon)

####################################### Smoothing Window ########################################
    def rectangle_window(self, N, amplitude, std=0):
        """
        Generates a rectangular window function.

        Parameters:
            amplitude (float): The amplitude of the window function.
            N (int): The length of the window.

        Returns:
            numpy.ndarray: The rectangular window function of length N, scaled by the amplitude.
        """
        return amplitude * signal.windows.boxcar(N)

    def hamming_window(self, N, amplitude, std=0):
        """
        Generate a Hamming window of length N.

        Args:
            N (int): The length of the window.
            amplitude (float): The amplitude of the window.

        Returns:
            ndarray: The Hamming window of length N, scaled by the given amplitude.
        """
        return amplitude * signal.windows.hamming(N)

    def hanning_window(self, N, amplitude, std=0):
        """
        Generate a Hanning window of length N.

        Parameters:
            N (int): The length of the window.
            amplitude (float): The amplitude of the window.

        Returns:
            numpy.ndarray: The Hanning window with length N, scaled by the amplitude.
        """
        return amplitude * signal.windows.hann(N)

    def gaussian_window(self, N, amplitude, std):
        """
        Generate a Gaussian window.

        Args:
            N (int): The length of the window.
            amplitude (float): The amplitude of the window.
            std (float): The standard deviation of the Gaussian distribution.

        Returns:
            numpy.ndarray: The generated Gaussian window.

        """
        std = int(self.new_window.stdSpinBox.text())
        return amplitude * signal.windows.gaussian(N, std)

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

            # Connect the close event to a custom function
            self.new_window.closeEvent = self.on_close_event

            # spine box of the standard deviation of the gaussian window function
            self.new_window.stdSpinBox.setValue(50)
            self.new_window.stdSpinBox.setRange(0, 1000)
            self.handle_selected_function()  # ??

            self.new_window.stdSpinBox.valueChanged.connect(
                self.handle_selected_function)

            self.new_window.functionList.currentIndexChanged.connect(
                self.handle_selected_function)

            self.new_window.save.clicked.connect(self.save)

            self.new_window.setWindowTitle("Smoothing window")
            self.new_window.show()
            self.new_window.destroyed.connect(self.onNewWindowClosed)

    def handle_selected_mode(self):
        mode = self.ui.modeList.currentIndex()
        self.our_signal.each_slider_reference = np.repeat(
            100, len(self.ranges[mode]))

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

    def smoothing_real_time(self):
        _, last_item = self.our_signal.slice_indices[-1]
        self.new_window.smoothingGraph1.clear()
        self.new_window.smoothingGraph1.plot(
            self.our_signal.fft_data[0][:last_item], self.our_signal.fft_data[1][:last_item])

        for i in range(len(self.our_signal.slice_indices)):
            start, end = self.our_signal.slice_indices[i]
            mode = self.ui.modeList.currentIndex()
            if i == len(self.our_signal.slice_indices) - 1:
                self.segment(
                    start, end-1, sparse=self.sparse_state[mode])
            else:
                self.segment(start, end, sparse=self.sparse_state[mode])

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

        else:
            amp = max(self.our_signal.fft_data[1][start:end])
            current_segment_smooth_window = self.custom_window(
                len(self.our_signal.fft_data[0][start:end]), amp)

        return current_segment_smooth_window

    def custom_window(self, samples, amplitude):
        self.window_function = {'Rectangle': self.rectangle_window,
                                'Hamming': self.hamming_window, 'Hanning': self.hanning_window, 'Gaussian': self.gaussian_window}

        std = 0
        smooth_data = self.window_function[self.selected_function](
            samples, amplitude, std)
        return smooth_data

    def save(self):
        self.our_signal.smoothing_window_name = self.selected_function
        self.ui.slidersWidget.setEnabled(True)

        self.onNewWindowClosed()

    def on_close_event(self, event):
        self.setEnabled(True)

    def onNewWindowClosed(self):
        self.new_window.close()
        self.setEnabled(True)


####################################### Main Window ########################################


    def init_ui(self):
        """
        Initializes the user interface.
        """
        # Load the UI Page
        self.ui = uic.loadUi('Mainwindow.ui', self)
        self.setWindowTitle("Signal Equlizer")
        self.graph_load_ui()

        # Other Attributes
        self.end_ind = 50
        self.ecg_flag = False
        # self.timer = QTimer()
        # self.timer.setInterval(50)
        # self.timer.timeout.connect(self.updating_graphs)

        self.ui.playPause.clicked.connect(self.play_pause)
        self.ui.zoomIn.clicked.connect(self.zoom_in)
        self.ui.zoomOut.clicked.connect(self.zoom_out)
        self.ui.resetButton.clicked.connect(self.reset)
        self.ui.playAudio1.clicked.connect(partial(
            self.toggle_audio, self.audio_widget1, self.ui.playAudio1, "icons/pause-square.png"))
        self.ui.playAudio2.clicked.connect(partial(
            self.toggle_audio, self.audio_widget2, self.ui.playAudio2, "icons/pause-square.png"))

        # Set speed slider properties

        self.ui.smoothingWindow.clicked.connect(self.openNewWindow)
        self.ui.browseFile.clicked.connect(self.browse)

        self.ui.modeList.setCurrentIndex(0)
        self.handle_combobox_selection()

        self.ui.modeList.currentIndexChanged.connect(
            self.handle_combobox_selection)
        self.ui.spectogramCheck.stateChanged.connect(self.show_spectrogram)

    def setup_widget_layout(self, widget, layout_parent):
        """
        Set up the layout for a widget within a layout parent.

        Args:
            widget: The widget to be added to the layout.
            layout_parent: The parent layout that the widget will be added to.
        """
        layout = QVBoxLayout(layout_parent)
        layout.addWidget(widget)

    def graph_load_ui(self):
        """
        Initializes the user interface for the graph loading functionality.

        This function creates instances of the MplWidget class to display spectrograms and
        instances of the AudioWidget class to play audio. It sets up the layouts for the
        spectrogram and audio widgets. Additionally, it sets the background of graph1 and
        graph2 to transparent and adds items to the modeList widget.

        Parameters:
            self (object): The GraphLoaderUI object.

        Returns:
            None
        """
        # Create instances of MplWidget for spectrogram
        self.spectrogram_widget1 = MplWidget()
        self.spectrogram_widget2 = MplWidget()

        # Set up layouts for spectrogram widgets
        self.setup_widget_layout(self.spectrogram_widget1, self.ui.spectogram1)
        self.setup_widget_layout(self.spectrogram_widget2, self.ui.spectogram2)

        # Create instances of AudioWidget
        self.audio_widget1 = AudioWidget()
        self.audio_widget2 = AudioWidget()

        # Set up layouts for audio widgets
        self.setup_widget_layout(self.audio_widget1, self.audio1)
        self.setup_widget_layout(self.audio_widget2, self.audio2)

        # Set the background of graph1 and graph2 to transparent
        self.ui.graph1.setBackground("transparent")
        self.ui.graph2.setBackground("transparent")

        # Add items to the modeList widget
        modes = ["Uniform Range", "Musical Instruments",
                 "Animal Sounds", "ECG Abnormalities"]
        self.ui.modeList.addItems(modes)

    def show_spectrogram(self, state):
        """
        Shows or hides the spectrogram graphs based on the state.

        Parameters:
            state (int): The state of the spectrogram. 
                         - 2: The spectrogram graphs are hidden.
                         - Other values: The spectrogram graphs are shown.

        Returns:
            None
        """
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

    def initialize_sig_attr(self):
        """
        Initializes the attributes of the class.

        This function sets the default values for the attributes of the class. 
        It sets the value of the `speedSlider` to 5 if the `activation` is "ecg", otherwise it sets the value to 200. 
        It also sets the value of `end_ind` to 50. 

        Parameters:
            self: The instance of the class.

        Returns:
            None
        # """
        # if self.activation == "ecg":
        #     self.speedSlider.setValue(5)

        # else:
        #     self.speedSlider.setValue(200)
        self.end_ind = 50
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.updating_graphs)

####################################### Data Processing ########################################

    def browse(self):
        """
        Opens a file dialog to browse and select a signal file.

        Returns:
            None
        """
        self.selected_function = None  # Window function
        self.activation = 'uniform'  # The mode of operation (default)
        self.current_slider = None
        self.ecg_flag = False
        self.pause_flag = False
        self.excess = None
        self.sparse_state = [False, True, True, True]

        self.file_filter = "Raw Data (*.csv *.wav *.mp3)"
        self.file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, 'Open Signal File', './', filter=self.file_filter)

        if self.file_path:
            self.file_name = os.path.basename(self.file_path)
            self.open_file(self.file_path, self.file_name)

    def open_file(self, path: str, file_name: str):
        """
        Open a file and process its data.

        Args:
            path (str): The path to the file.
            file_name (str): The name of the file.
        """
        data = []
        time = []
        sample_rate = 0

        # Extract the file extension
        filetype = path.split('.')[-1]

        if filetype in ["wav", "mp3"]:
            # Load audio file and extract data, sample rate, and duration
            data, sample_rate = librosa.load(path)
            duration = librosa.get_duration(y=data, sr=sample_rate)
            time = np.linspace(0, duration, len(data))

        elif filetype == "csv":
            self.ecg_flag = True
            # Read CSV file and extract time and data columns
            data_reader = pd.read_csv(
                path, delimiter=',', skiprows=1)  # Skip header row
            time = np.array(data_reader.iloc[:, 0].astype(float).tolist())
            data = np.array(data_reader.iloc[:, 1].astype(float).tolist())
            sample_rate = 1 / np.mean(np.diff(time))
            self.speed = 3

        # Process the data
        self.process_data(data, time, sample_rate, file_name)

    def process_data(self, data: list[float],
                     time: float,
                     sample_rate: int,
                     file_name: str) -> None:
        """
        Process the data by setting the necessary attributes of the Signal class
        and calling the process_signal method.

        Args:
            data (List[float]): The input data.
            time (float): The time duration of the data.
            sample_rate (int): The sample rate of the data.
            file_name (str): The name of the file.

        Returns:
            None
        """

        # Set the selected function
        self.selected_function = "Rectangle"

        # Create an instance of the Signal class
        self.our_signal = Signal(file_name[:-4])

        # Set the attributes of the Signal class
        self.our_signal.data = data
        self.our_signal.time = time
        self.our_signal.sr = sample_rate

        # Calculate the sample interval
        sample_interval = 1 / self.our_signal.sr

        # Set the smoothing window name
        self.our_signal.smoothing_window_name = self.selected_function

        # Initialize the data_x and data_y lists
        self.our_signal.data_x = []
        self.our_signal.data_y = []

        # Get the FFT values
        x, y = self.get_fft_values(sample_interval, len(self.our_signal.data))

        # Set the FFT data
        self.our_signal.fft_data = [x, y]

        # Set the data_after attribute
        self.our_signal.data_after = data

        # Call the process_signal method
        self.process_signal()

    def process_signal(self):
        """
        This function processes a signal by performing several operations. 

        Parameters:
            None

        Returns:
            None
        """
        self.initialize_speed()
        self.ui.slidersWidget.setEnabled(False)
        self.handle_selected_mode()
        self.initialize_sig_attr()
        self.reset_sliders()
        self.split_data()
        self.plot_signal()
        self.plot_spectrogram()

    def initialize_speed(self):
        if self.ecg_flag:
            self.speed = 8
            self.speedSlider.setMinimum(0)
            self.speedSlider.setMaximum(50)
            self.speedSlider.setSingleStep(5)
            self.speedSlider.setValue(self.speed)
            print(self.speed)
        else:
            self.speed = 200
            self.speedSlider.setMinimum(0)
            self.speedSlider.setMaximum(500)
            self.speedSlider.setSingleStep(5)
            self.speedSlider.setValue(self.speed)
        self.speedSlider.valueChanged.connect(self.change_speed)

####################################### Plot Data and Controllers ########################################
    def plot_signal(self):
        """ 
        Plot the signal on two graph objects.
        """
        if not self.our_signal:
            return

        self.ui.graph1.clear()
        self.ui.graph2.clear()

        data_x = self.our_signal.time[:self.end_ind]
        data_y_before = self.our_signal.data[:self.end_ind]
        self.plot_item_before = self.plot_on_graph(
            self.ui.graph1, data_x, data_y_before)

        data_y_after = self.our_signal.data_after[:self.end_ind]
        if len(data_x) == len(data_y_after):
            self.plot_item_after = self.plot_on_graph(
                self.ui.graph2, data_x, data_y_after)
        else:
            excess = len(data_x) - len(data_y_after)
            self.plot_item_after = self.plot_on_graph(
                self.ui.graph2, data_x[:-excess], data_y_after)

        self.set_icon(self.ui.playPause, "icons/pause-square.png")
        self.ui.playPause.setText("Pause")

        if not self.pause_flag and not self.timer.isActive():
            self.timer.start(50)

    def plot_on_graph(self, graph, data_x, data_y):
        """
        Plot the given data on the provided graph.

        Args:
            graph (GraphItem): The graph on which to plot the data.
            data_x (array-like): The x-axis values of the data.
            data_y (array-like): The y-axis values of the data.

        Returns:
            PlotItem: The item representing the plotted data on the graph.
        """
        plot_item = graph.plot(
            data_x, data_y, name=self.our_signal.name, pen=(64, 92, 245))

        # Check if there is already a legend and remove it
        if graph.plotItem.legend is not None:
            graph.plotItem.legend.clear()

        # Add a legend to the plot
        legend = graph.addLegend()
        legend.setParentItem(graph.plotItem)
        legend.addItem(plot_item, name=self.our_signal.name)

        return plot_item

    def updating_graphs(self):
        """
        Update the graphs with new data points.

        This function updates the graphs with the latest data points from the signal.
        It sets the x-axis range of the graphs based on the latest data point.
        It also updates the data for the before and after graphs.

        Returns:
            None
        """
        # Get the data from the signal
        data_before = self.our_signal.data
        data_after = self.our_signal.data_after
        time = self.our_signal.time

        # Get the data points to be plotted
        data_X = time[:self.end_ind + self.speed]
        data_Y_before = data_before[:self.end_ind + self.speed]
        data_Y_after = data_after[:self.end_ind + self.speed]

        # Update the end index
        self.end_ind += self.speed

        # Set the x-axis range of the graphs
        if (data_X[-1] < 1):
            self.ui.graph1.setXRange(0, 1)
            self.ui.graph2.setXRange(0, 1)
        else:
            self.ui.graph1.setXRange(data_X[-1] - 1, data_X[-1])
            self.ui.graph2.setXRange(data_X[-1] - 1, data_X[-1])

        # Update the data for the before graph
        self.plot_item_before.setData(data_X, data_Y_before, visible=True)

        # Update the data for the after graph
        if len(data_X) == len(data_Y_after):
            self.plot_item_after.setData(data_X, data_Y_after, visible=True)
        else:
            self.excess = len(data_X) - len(data_Y_after)
            self.plot_item_after.setData(
                data_X[:-self.excess], data_Y_after, visible=True)

    def plot_spectrogram(self):
        """
        Plot the spectrogram of the signal.

        This function plots the spectrogram of the signal using two different widgets:
        spectrogram_widget1 and spectrogram_widget2. It takes no parameters and has no return value.

        Parameters:
        - None

        Return:
        - None
        """
        if self.our_signal:
            self.spectrogram_widget1.plot_spectrogram(
                self.our_signal.data, self.our_signal.sr)

            self.spectrogram_widget2.plot_spectrogram(
                self.our_signal.data_after, self.our_signal.sr)

    def toggle_audio(self, audio_widget, play_button, icon_path):
        """
        Toggle the audio playback and update the play button text and icon accordingly.

        Args:
            audio_widget (AudioWidget): The audio widget to control.
            play_button (QPushButton): The button used to play/pause the audio.
            icon_path (str): The path to the icon image file.

        Returns:
            None
        """
        if self.our_signal:
            if not audio_widget.playing:
                # Play audio before modifications
                if audio_widget == self.audio_widget1:
                    audio_widget.play_audio(
                        self.our_signal.data, self.our_signal.sr)

                # Play audio after modifications
                else:
                    audio_widget.play_audio(
                        self.our_signal.data_after, self.our_signal.sr)

                play_button.setText("Pause Audio")
                self.set_icon(play_button, icon_path)
            else:
                sd.stop()
                play_button.setText("Play Audio")
                self.set_icon(play_button, "icons/play-square-svgrepo-com.png")

    def play_pause(self):
        """
        Toggles the play/pause functionality of the timer.

        If the timer is active, it stops the timer, changes the icon to play, and updates the button text to "Play".
        If the timer is not active, it starts the timer, changes the icon to pause, and updates the button text to "Pause".
        """

        if self.our_signal:
            if self.timer.isActive():
                self.timer.stop()
                self.set_icon(self.ui.playPause,
                              "icons/play-square-svgrepo-com.png")
                self.ui.playPause.setText("Play")
            else:
                self.set_icon(self.ui.playPause, "icons/pause-square.png")
                self.ui.playPause.setText("Pause")
                self.timer.start()

    def zoom_in(self):
        """
        Zooms in on the graph by scaling the view boxes.
        """
        # Get the view box of graph1 and scale it by (0.5, 1)
        view_box1 = self.graph1.plotItem.getViewBox()
        view_box1.scaleBy((0.5, 1))

        # Get the view box of graph2 and scale it by (0.5, 1)
        view_box2 = self.graph2.plotItem.getViewBox()
        view_box2.scaleBy((0.5, 1))

    def zoom_out(self):
        """
        Zoom out the view boxes of graph1 and graph2 by scaling them horizontally.

        Args:
            self: The current instance of the class.
        """
        # Get the view box of graph1 and scale it horizontally by a factor of 1.5
        view_box1 = self.graph1.plotItem.getViewBox()
        view_box1.scaleBy((1.5, 1))

        # Get the view box of graph2 and scale it horizontally by a factor of 1.5
        view_box2 = self.graph2.plotItem.getViewBox()
        view_box2.scaleBy((1.5, 1))

    def change_speed(self):
        """
        Change the speed of the object.

        """
        if self.our_signal:
            self.speed = self.speedSlider.value()

    def reset_data(self):
        """
        Reset the data in the signal object.
        """
        if self.our_signal:
            # Counteract the effect of the smoothing window
            self.our_signal.data_after = self.our_signal.data
            self.our_signal.fft_data = self.get_fft_values(
                1/self.our_signal.sr, len(self.our_signal.data))

            for i, segment in enumerate(self.our_signal.smooth_seg):
                start, end = self.our_signal.slice_indices[i]

                # Normalize the windowed segment
                normalized_window = segment / \
                    (self.our_signal.smooth_seg_amp[i])

                # Update the amplitude of the smoothing segment
                self.our_signal.smooth_seg_amp[i] = max(
                    self.our_signal.fft_data[1][start:end])

                # Apply the normalized window to the segment
                self.our_signal.smooth_seg[i] = normalized_window * \
                    self.our_signal.smooth_seg_amp[i]

    def reset(self):
        """
        Resets the state of the application, clearing the graph and restarting the signal processing.
        """
        if self.our_signal:
            msg_box = QMessageBox()
            msg_box.setText("Do you want to clear the graph?")
            msg_box.setWindowTitle("Clear Graph")
            msg_box.setStandardButtons(
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            result = msg_box.exec()

        if result == QMessageBox.StandardButton.Ok:
            # Clear the graphs
            self.ui.graph1.clear()
            self.ui.graph2.clear()

            # Clear the spectrogram widgets
            self.spectrogram_widget1.clear()
            self.spectrogram_widget2.clear()

            # Stop the timer
            self.timer.stop()

            # Reset the data
            self.reset_data()

            # Initialize signal attributes
            # self.initialize_sig_attr()

            # Process the signal
            self.handle_selected_mode()
            self.initialize_sig_attr()
            self.reset_sliders()
            self.plot_signal()
            self.plot_spectrogram()

            # Start the timer
            self.timer.start()

    def reset_sliders(self):
        if self.our_signal:
            sliders = self.ui.slidersWidget.findChildren(QSlider)
            for i, slider in enumerate(sliders):
                slider.setValue(self.our_signal.each_slider_reference[i])

####################################### Splitting Data ########################################

    def find_closest_index(self, array, target):
        """
        Find the index of the closest value in the array to the target.

        Args:
            array (list): The array of values.
            target (int or float): The target value.

        Returns:
            int: The index of the closest value in the array to the target.
        """
        index = bisect.bisect_left(array, target)

        # If the target is less than or equal to the first element in the array,
        # return 0 as the index.
        if index == 0:
            return 0

        # If the target is greater than or equal to the last element in the array,
        # return the index of the last element.
        if index == len(array):
            return len(array) - 1

        # Get the values before and after the target index.
        before = array[index - 1]
        after = array[index]

        # If the difference between the value after the target index and the target
        # is smaller than the difference between the value before the target index
        # and the target, return the target index.
        if after - target < target - before:
            return index
        else:
            return index - 1

    def split_data(self):
        """
        Split the data based on the mode selection.

        Returns:
            None
        """
        # Get the mode selection from the UI
        mode_selection = self.ui.modeList.currentIndex()

        # Get the frequencies from the fft_data
        frequencies = self.our_signal.fft_data[0]

        # Clear existing slice indices
        self.our_signal.slice_indices = []

        if mode_selection == 0:
            # Split the frequencies into equal slices
            num_slices = 10
            excess_elements = len(frequencies) % num_slices
            slice_size = len(frequencies) // num_slices
            self.our_signal.slice_indices = [
                (i * slice_size, (i + 1) * slice_size) for i in range(num_slices)]

            # Adjust the last slice to include excess elements
            start, end = self.our_signal.slice_indices[-1]
            last_slice = (start, end + excess_elements)
            self.our_signal.slice_indices[-1] = last_slice

        else:
            # Calculate the slice indices based on the mode selection
            self.calc_boundaries_indices(frequencies, mode_selection)

    def calc_boundaries_indices(self, data, mode):
        """
        Calculate the indices of the start and end boundaries of the selection ranges in the data.

        Args:
            data: The data to search for indices.
            mode: The mode indicating which selection ranges to use.

        Returns:
            None
        """
        # Get the selection ranges based on the mode
        selection_ranges = self.ranges[mode]

        # Iterate over each start and end boundary in the selection ranges
        for start, end in selection_ranges:
            # Find the index closest to the start boundary in the data
            start_index = self.find_closest_index(data, start)

            # Find the index closest to the end boundary in the data
            end_index = self.find_closest_index(data, end)

            # Append the start and end indices to the slice_indices list
            self.our_signal.slice_indices.append((start_index, end_index))

    def get_fft_values(self, T, N):
        """
        Calculate the FFT values of the signal.

        Args:
            T (float): Time period of the signal.
            N (int): Number of data points.

        Returns:
            tuple: A tuple containing the frequency values and the FFT values.

        """

        # Calculate the frequency values
        f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)

        # Calculate the FFT values
        fft_values = np.fft.fft(self.our_signal.data, N)

        # Calculate the phase values
        self.our_signal.phase = np.angle(fft_values[:N//2])

        # Calculate the absolute values of the FFT coefficients
        fft_values = (2/N) * np.abs(fft_values[:N//2])

        return f_values, fft_values

    def get_inverse_fft_values(self):
        """
        Returns the reconstructed signal after applying inverse FFT to the modified FFT values.

        Returns:
            reconstructed_signal (numpy.ndarray): The reconstructed signal.
        """
        # Calculate modified FFT values by multiplying the second element of fft_data by (len(data)/2) and
        # exponentiating it by the phase of the signal
        modified_fft_values = np.array(self.our_signal.fft_data[1]) * (len(self.our_signal.data)/2) * \
            np.exp(1j * self.our_signal.phase)

        # Apply inverse FFT to the modified FFT values to reconstruct the signal
        reconstructed_signal = np.fft.irfft(modified_fft_values)

        return reconstructed_signal

    def editing(self, slider_value, slidernum):
        """
        Edit the signal based on the slider value and slider number.

        Args:
            slider_value (float): The value of the slider.
            slidernum (int): The number of the slider.

        Returns:
            None
        """

        # Check if our_signal exists
        if self.our_signal is not None:
            # Check if smooth_seg is True
            if self.our_signal.smooth_seg:
                # Enable the sliders widget
                self.ui.slidersWidget.setEnabled(True)
                self.pause_flag = True

                # Get the start and end indices for the slice
                start, end = self.our_signal.slice_indices[slidernum]

                # Get the magnitude FFT data and frequency values
                mag_fft = np.array(self.our_signal.fft_data[1][start:end])
                freq_values = np.array(self.our_signal.fft_data[0][start:end])

                # Set the current slider
                self.current_slider = slidernum

                # Calculate the factor of multiplication
                factor_of_multiplication = slider_value / \
                    self.our_signal.each_slider_reference[slidernum]

                # Calculate the smooth data
                smooth_data = (factor_of_multiplication) * np.array(
                    self.our_signal.smooth_seg[slidernum] / self.our_signal.smooth_seg_amp[slidernum])

                # Calculate the result
                result = mag_fft * smooth_data

                # Update the smooth_seg, fft_data, smooth_seg_amp, and each_slider_reference
                self.our_signal.smooth_seg[slidernum] = smooth_data
                self.our_signal.fft_data[1][start:end] = result
                self.our_signal.smooth_seg_amp[slidernum] = factor_of_multiplication
                self.our_signal.each_slider_reference[slidernum] = slider_value

                # Update the data after inverse FFT
                self.our_signal.data_after = self.get_inverse_fft_values()

                # Plot the spectrogram
                self.spectrogram_widget2.plot_spectrogram(
                    self.our_signal.data_after, self.our_signal.sr)

                # Plot the signal
                self.plot_signal()

    def add_sliders(self, num_sliders):
        """
        Adds a specified number of sliders to the UI.

        Args:
            num_sliders (int): The number of sliders to add.
        """
        sliders_num = num_sliders
        # Get the layout of the sliders widget
        layout = self.ui.slidersWidget.layout()

        # If the layout already exists, remove all existing sliders
        if layout:
            for i in reversed(range(layout.count())):
                layout.itemAt(i).widget().setParent(None)

        # Add the specified number of sliders to the layout
        if self.ui.modeList.currentIndex() == 3:
            sliders_num = 3
        for i in range(sliders_num):
            slider = QSlider(Qt.Orientation.Vertical)
            slider.setValue(100)
            slider.setSingleStep(1)
            slider.setRange(1, 200)
            layout.addWidget(slider)
            # if self.ui.modeList.currentIndex == 3:
            #     label = QLabel(f"Label {i + 1}")
            #     self.ui.widget.addWidget(label)

    def handle_combobox_selection(self):
        """
        Handle the selection of an item in the combobox.
        """
        # Get the current index of the combobox
        current_index = self.ui.modeList.currentIndex()

        # Determine the number of sliders based on the current index
        num_sliders = 10 if current_index == 0 else 4

        # Add the specified number of sliders
        self.add_sliders(num_sliders)

        # Find all the sliders in the slidersWidget
        sliders = self.ui.slidersWidget.findChildren(QSlider)

        # Connect the valueChanged signal of each slider to the editing method
        for slider in sliders:
            slider.valueChanged.connect(
                lambda slider_value=(slider.value()), slidernum=sliders.index(slider): self.editing(slider_value, slidernum))


def main():
    app = QtWidgets.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

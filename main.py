from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFileDialog, QMessageBox, QColorDialog, QListWidgetItem, QPushButton, QSlider
from PyQt6.QtCore import Qt
import numpy as np
import sys
from PyQt6 import QtWidgets, uic
import pyqtgraph as pg
import qdarkstyle
import os
import csv
from pydub import AudioSegment

from Signal import Signal

# pip install pyqtgraph pydub
# https://www.ffmpeg.org/download.html
# https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/
# setx PATH "C:\ffmpeg\bin;%PATH%"
# restart 

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.init_ui()


    def init_ui(self):
        # Load the UI Page
        self.ui = uic.loadUi('Mainwindow.ui', self)
        self.setWindowTitle("Signal Equlizer")
        self.graph_style_ui()

        self.ui.browseFile.clicked.connect(self.browse)
        self.ui.modeList.currentIndexChanged.connect(self.handle_combobox_selection)


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

            if filetype in ["wav","mp3"]:
                # Read the WAV file
                audio = AudioSegment.from_file(path)
                data = np.array(audio.get_array_of_samples())
                # Create the time data
                time = np.arange(len(data))


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
            signal = Signal(file_name[:-4])

            signal.data = data
            signal.time = time
            self.plot_temp(signal)
            

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
            plot_item = self.ui.graph1.plot(x_data, y_data, name=signal.name, pen=pen)

            # Check if there is already a legend and remove it
            if self.ui.graph1.plotItem.legend is not None:
                self.ui.graph1.plotItem.legend.clear()

            # Add a legend to the plot
            legend = self.ui.graph1.addLegend()
            legend.addItem(plot_item, name=signal.name)

            

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
        if current_index ==0:
            self.add_sliders(10)
        else: self.add_sliders(4)
   


# @lru_cache(maxsize=128)
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
    main = MainWindow()
    main.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
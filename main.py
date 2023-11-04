from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFileDialog, QMessageBox, QColorDialog, QListWidgetItem, QPushButton
import numpy as np
import sys
from PyQt6 import QtWidgets, uic
import pyqtgraph as pg
import qdarkstyle

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.init_ui()


    def init_ui(self):
        # Load the UI Page
        self.ui = uic.loadUi('Mainwindow.ui', self)
        self.setWindowTitle("Signal Equlizer")
        print("Loading")
        self.ui.graph1.setBackground("transparent")
        self.ui.graph2.setBackground("transparent")
        self.ui.spectogram1.setBackground("transparent")
        self.ui.spectogram2.setBackground("transparent")
        self.ui.modeList.addItem("number1")
        self.ui.modeList.addItem("number2")
        self.ui.modeList.addItem("number3")
        self.ui.modeList.addItem("number4")


# @lru_cache(maxsize=128)
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
    main = MainWindow()
    main.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
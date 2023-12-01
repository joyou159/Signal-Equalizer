from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import QTimer
import sounddevice as sd


class AudioWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.duration_label = QLabel("0:00 / 0:00", self)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)

        # Set up layout
        layout = QHBoxLayout(self)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.duration_label)

        self.style_progress_bar()

        # Initialize timer for updating the duration label and progress bar
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_duration_and_progress)

        # Flag to track whether audio is currently playing
        self.playing = False

    def play_audio(self, audio_data, sample_rate):
        if not self.playing:
            # Calculate total playback time
            self.total_time = len(audio_data) / sample_rate

            # Play audio using sounddevice
            sd.play(audio_data, sample_rate)

            # Set up the timer for updating the duration label and progress bar
            self.timer.start(100)  # Update every 100 milliseconds
            self.start_time = sd.get_stream().time

            # Set the playing flag to True
            self.playing = True

    def update_duration_and_progress(self):
        if sd.get_stream().active:
            elapsed_time = sd.get_stream().time - self.start_time

            # Format the elapsed time and total duration as minutes and seconds
            elapsed_minutes, elapsed_seconds = divmod(int(elapsed_time), 60)
            total_minutes, total_seconds = divmod(int(self.total_time), 60)

            duration_text = f"{elapsed_minutes}:{elapsed_seconds:02d} / {total_minutes}:{total_seconds:02d}"
            self.duration_label.setText(duration_text)

            # Update the progress bar value
            progress = int((elapsed_time / self.total_time) * 100)
            self.progress_bar.setValue(progress)
        else:
            # Stop the timer and reset the playing flag when the stream is not active
            self.timer.stop()
            self.playing = False


    def style_progress_bar(self):
                # Apply the specified style sheet to the progress bar
        progress_bar_style = """
            QScrollBar:horizontal {
                border: 2px solid grey;
                background: #405cf5;
                height: 10px;
                margin: 0px 20px 0 20px;
            }

            QScrollBar::handle:horizontal {
                background: #405cf5;
                min-width: 20px;
            }

            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                background: #405cf5;
                border: none;
            }
        """

        self.progress_bar.setStyleSheet(progress_bar_style)

        # Hide the text on the progress bar
        self.progress_bar.setTextVisible(False)

# Signal Equalizer Desktop Application

## Overview

This desktop application allows users to open a signal, manipulate its frequency components, and reconstruct the modified signal. It supports different modes such as Uniform Range, Musical Instruments, Animal Sounds, and ECG Abnormalities. Users can customize the magnitude of frequency components using sliders and choose from various smoothing windows. The application provides real-time visualization, including signal viewers, spectrograms, and interactive graphs.

## Features

### 1. Modes

- **Uniform Range Mode:**
  - Divides the frequency range of the input signal into 10 equal ranges.
  - Allows users to control the magnitude of each range using sliders.

- **Musical Instruments Mode:**
  - Enables users to manipulate specific musical instrument frequencies.
  - Four sliders correspond to different instrument ranges.

- **Animal Sounds Mode:**
  - Similar to Musical Instruments Mode but for animal sounds.

- **ECG Abnormalities Mode:**
  - Allows users to modify ECG signals with specific arrhythmia types.
  - Four sliders control the magnitude of arrhythmia components.

### 2. Smoothing Windows

- Four smoothing windows are available: Rectangle, Hamming, Hanning, and Gaussian.
- Users can visually customize window parameters.
- Real-time visualization of the smoothing effect on the equalizer.

### 3. Signal Visualization

- Two linked cine signal viewers for input and output signals.
- Functionality panel for play/stop/pause/speed-control/zoom/pan/reset.
- Synchronous viewing of signals.
- Toggle show/hide of spectrograms.

### 4. Spectrograms

- Real-time update of spectrograms upon slider adjustments.
- Option to toggle show/hide of spectrograms.

### 5. Signal Loading and Playback

- Load synthetic or user-provided signals (in formats: CSV, WAV, MP3).
- Playback functionality with play/pause, speed control, zoom, and reset options.

### 6. Customization

- Users can customize window functions, frequency ranges, and other parameters.
- Save and load custom equalizer settings.

## How to Use

1. **Select Mode:**
   - Use the mode dropdown to choose between Uniform Range, Musical Instruments, Animal Sounds, or ECG Abnormalities.

2. **Open Signal:**
   - Click the "Browse" button to select a signal file (CSV, WAV, MP3).
   - The application loads the signal and displays it in the main window.
3. **Choose Smoothing Window:**
   - Select a smoothing window type from the dropdown menu.
   - Customize window parameters if necessary.

4. **Adjust Sliders:**
   - Depending on the mode, sliders will appear for adjusting frequency components.
   - Drag the sliders to change the magnitude of specific frequency ranges.

5. **Visualize and Play:**
   - The application provides real-time visual feedback on signal changes.
   - Use play/pause, speed control, and other playback options for interactive exploration.


## Dependencies

- Python 3.x
- PyQt6
- NumPy
- Pandas
- Matplotlib
- PyQtGraph
- Librosa
- Sounddevice

## How to Run

1. Install the required dependencies using 
```bash
pip install -r requirements.txt
```
2. Run the application using 
```bash
python main.py
```




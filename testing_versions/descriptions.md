# Heartbeat Monitoring using Acoustic Signals

This project implements heartbeat monitoring by using **Frequency Modulated Continuous Wave (FMCW) sonar** techniques. It leverages a speaker to play an inaudible chirp signal and a microphone to capture the reflected signals from the user's chest. By analyzing phase shifts in the reflected signals, the heartbeat rate is estimated in beats per minute (BPM).

## Key Concepts

### 1. **FMCW Chirp Signal**
The system uses an inaudible **FMCW chirp signal** sweeping from **17 kHz to 19 kHz**. This chirp is played through a speaker and reflected off the user's chest. The reflections carry subtle phase shifts due to chest movements caused by heartbeats. These phase shifts are captured for further analysis.

### 2. **Audio Playback and Recording**
The chirp signal is played through the speaker while simultaneously recording the reflected signal through the microphone. This process is managed using the **PyAudio** library, which allows the program to handle real-time audio streams. The reflections contain information about heartbeat-induced chest movements.

### 3. **Phase Extraction**
The recorded reflections are processed to extract the **phase** of the signal using a **Hilbert transform**. This phase signal is crucial, as changes in phase correspond to minute chest movements related to the heartbeat. Smoothing techniques (e.g., the **Savitzky-Golay filter**) are applied to reduce noise in the phase signal.

### 4. **Band-Pass Filtering**
To focus on the frequency range where heartbeats typically occur, a **band-pass filter** is applied to the smoothed phase signal. The filter isolates frequencies between **1 Hz and 2 Hz**, corresponding to heart rates between **60 BPM and 120 BPM**.

### 5. **Heartbeat Rate Estimation**
The filtered phase signal is analyzed using the **Fast Fourier Transform (FFT)** to detect dominant frequencies in the heartbeat range. The dominant frequency is then converted into beats per minute (BPM), providing an estimate of the user's heart rate. The program also computes and displays the average heart rate across multiple time windows.

### 6. **Plotting Results**
The project includes plotting capabilities to visualize:
- The original chirp signal.
- The reflected signal captured by the microphone.
- The extracted phase signal.
- The estimated heart rate over time.

## How to Execute the Code

### 1. **Prerequisites**
Before running the program, make sure you have the required Python libraries installed:
```bash
pip install numpy matplotlib pyaudio scipy
```
### 2. **Running the Code**
To run the code:

 1. Open a terminal or command prompt in the directory containing the Python scripts.
 2. Run either of the scripts by executing:
```bash
python version2.py
```
  or
```bash
python version3.py
```
## Expected Outputs

Once the script is running:

 1. The system will play an inaudible chirp through the speakers and record the reflections.
 2. It will process the reflected signal to estimate the heart rate.
 3. The estimated heart rate will be displayed on the console.
 4. Plots will be generated showing the chirp signal, the reflected signal, the extracted phase signal, and the heart rate over time.

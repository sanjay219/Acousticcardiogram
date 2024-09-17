import numpy as np
import matplotlib.pyplot as plt
import pyaudio
from scipy.fft import fft, fftfreq

# Audio stream parameters
SAMPLE_RATE = 48000  # 48 kHz
DURATION = 60  # Duration in seconds to capture audio
CHANNELS = 1  # Single microphone input
CHUNK = 2048  # Number of audio samples per frame

# Initialize PyAudio
p = pyaudio.PyAudio()

# Function to capture audio from the microphone
def capture_audio(duration, sample_rate):
    print("Starting audio capture...")
    stream = p.open(format=pyaudio.paFloat32,
                    channels=CHANNELS,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    for _ in range(0, int(sample_rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.float32))

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    print("Audio capture finished.")
    return np.concatenate(frames)

# Capture real audio data
audio_signal = capture_audio(DURATION, SAMPLE_RATE)

# Down-convert to complex baseband signal for processing
def down_convert(signal, fc, sampling_rate):
    print("Down-converting signal to baseband...")
    t = np.arange(len(signal)) / sampling_rate
    return signal * np.exp(-1j * 2 * np.pi * fc * t)

# Baseband processing with a center frequency of 18 kHz (example)
fc = 19000
baseband_signal = down_convert(audio_signal, fc, SAMPLE_RATE)

# Extract phase information
print("Extracting phase information...")
signal_phase = np.angle(baseband_signal)

# Apply smoothing to the phase signal to filter out high-frequency noise
def smooth_signal(signal, window_size=100):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

signal_phase_smoothed = smooth_signal(signal_phase)

# Function to perform windowed vital sign extraction
def extract_breath_heart_rates_windowed(phase_signal, sampling_rate, window_duration=5):
    print("Extracting breath and heart rates over time...")
    window_size = int(window_duration * sampling_rate)  # Window size in samples
    num_windows = len(phase_signal) // window_size
    heart_rates = []
    
    for i in range(num_windows):
        window_signal = phase_signal[i*window_size : (i+1)*window_size]
        n = len(window_signal)
        yf = fft(window_signal)
        xf = fftfreq(n, 1 / sampling_rate)

        # Identify heart rate (0.8 - 2 Hz)
        heart_range = (xf > 0.8) & (xf < 2)
        if np.any(heart_range):
            heart_freq = xf[np.argmax(np.abs(yf)[heart_range])]
            heart_rate_bpm = heart_freq * 60
            heart_rates.append(heart_rate_bpm)
        else:
            heart_rates.append(np.nan)  # No valid frequency found in this window

    return heart_rates

# Perform windowed vital sign extraction
heart_rates = extract_breath_heart_rates_windowed(signal_phase_smoothed, SAMPLE_RATE)

# Average the heart rates over all windows, ignoring NaN values
valid_heart_rates = [hr for hr in heart_rates if not np.isnan(hr)]
if valid_heart_rates:
    avg_heart_rate = np.nanmean(valid_heart_rates)
    max_heart_rate = np.nanmax(valid_heart_rates)
    min_heart_rate = np.nanmin(valid_heart_rates)
    print(f"Estimated Average Heart Rate: {avg_heart_rate:.2f} BPM")
    print(f"Estimated Maximum Heart Rate: {max_heart_rate:.2f} BPM")
    print(f"Estimated Minimum Heart Rate: {min_heart_rate:.2f} BPM")
else:
    print("No valid heart rate data found.")

# Plot heart rate over time
time_axis = np.arange(len(heart_rates)) * 5  # Time in seconds (5-second windows)
plt.figure(figsize=(12, 4))
plt.plot(time_axis, heart_rates, marker='o', label='Heart Rate (BPM)')
plt.title('Heart Rate Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Heart Rate (BPM)')
plt.ylim(00, 150)
plt.legend()
plt.show()

# Terminate PyAudio
p.terminate()

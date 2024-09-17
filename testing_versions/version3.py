import numpy as np
import matplotlib.pyplot as plt
import pyaudio
from scipy.signal import chirp, hilbert, butter, lfilter, savgol_filter
from scipy.fft import fft, fftfreq

# Parameters
SAMPLE_RATE = 48000  # 48 kHz sampling rate
DURATION = 20  # Total duration for heart rate monitoring (seconds)
CHIRP_SWEEP_TIME = 0.05  # Sweep time (50 ms)
FREQ_START = 17000  # Starting frequency of the chirp (17 kHz)
FREQ_END = 19000  # Ending frequency of the chirp (19 kHz)
CHUNK = 1024  # Buffer size for audio stream
HEART_RATE_MIN = 1.0  # Minimum heart rate frequency in Hz (~60 BPM)
HEART_RATE_MAX = 2.0  # Maximum heart rate frequency in Hz (~120 BPM)

# PyAudio Initialization
p = pyaudio.PyAudio()

# Generate FMCW Chirp Signal
def generate_fmcw_signal(duration, sample_rate, f_start, f_end, sweep_time):
    t = np.arange(0, duration, 1/sample_rate)
    chirp_signal = chirp(t, f0=f_start, f1=f_end, t1=sweep_time, method='linear')
    return t, chirp_signal

# Stream audio using PyAudio
def play_and_record_audio(chirp_signal, duration, sample_rate):
    # Open audio stream
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    # Play chirp and capture simultaneously
    captured_frames = []
    print("Playing chirp and capturing reflections...")

    num_chunks = int(sample_rate / CHUNK * duration)
    chirp_repeated = np.tile(chirp_signal, (num_chunks // len(chirp_signal) + 1))[:num_chunks * CHUNK]

    for i in range(0, num_chunks):
        start_idx = i * CHUNK
        end_idx = start_idx + CHUNK
        chirp_segment = chirp_repeated[start_idx:end_idx].astype(np.float32)

        stream.write(chirp_segment)
        data = stream.read(CHUNK)
        captured_frames.append(np.frombuffer(data, dtype=np.float32))

    # Stop and close stream
    stream.stop_stream()
    stream.close()

    return np.concatenate(captured_frames)

# Band-pass filter to isolate heartbeat frequency range
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# Extract heartbeat phase using Hilbert transform
def extract_phase(signal):
    analytic_signal = hilbert(signal)
    phase_signal = np.angle(analytic_signal)
    return phase_signal

# Clean the signal (replace NaN, Inf values)
def clean_signal(signal):
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

# Perform FFT and estimate heart rate for a window of data
def estimate_heart_rate(phase_signal, sample_rate):
    phase_signal = clean_signal(phase_signal)
    
    n = len(phase_signal)
    yf = fft(phase_signal)
    xf = fftfreq(n, 1/sample_rate)
    
    pos_indices = np.where(xf > 0)
    xf = xf[pos_indices]
    yf = np.abs(yf[pos_indices])
    
    heart_rate_range = (xf >= HEART_RATE_MIN) & (xf <= HEART_RATE_MAX)
    xf_heart = xf[heart_rate_range]
    yf_heart = yf[heart_rate_range]
    
    if len(yf_heart) == 0 or np.all(np.isnan(yf_heart)):
        return None
    
    dominant_freq = xf_heart[np.argmax(yf_heart)]
    heart_rate_bpm = dominant_freq * 60
    return heart_rate_bpm

# Main execution
if __name__ == "__main__":
    # Generate FMCW chirp signal
    t, chirp_signal = generate_fmcw_signal(DURATION, SAMPLE_RATE, FREQ_START, FREQ_END, CHIRP_SWEEP_TIME)
    
    # Play chirp and record the reflected signal
    reflected_signal = play_and_record_audio(chirp_signal, DURATION, SAMPLE_RATE)
    
    # Extract phase from the reflected signal
    heartbeat_phase = extract_phase(reflected_signal)
    
    # Apply smoothing to the phase signal using Savitzky-Golay filter
    smoothed_phase = savgol_filter(heartbeat_phase, window_length=51, polyorder=2)  # Adjusted window size
    
    # Apply band-pass filter to isolate the heart rate frequency range
    filtered_phase = bandpass_filter(smoothed_phase, HEART_RATE_MIN, HEART_RATE_MAX, SAMPLE_RATE)
    
    # Estimate heart rate in smaller windows (e.g., 2-second windows)
    window_size = SAMPLE_RATE * 2  # 2-second window
    heart_rates = []

    for i in range(0, len(filtered_phase), window_size):
        segment = filtered_phase[i:i + window_size]
        if len(segment) > 0:
            heart_rate_bpm = estimate_heart_rate(segment, SAMPLE_RATE)
            if heart_rate_bpm:
                heart_rates.append(heart_rate_bpm)
            else:
                heart_rates.append(np.nan)  # No valid heart rate for this segment

    # Display the average or last valid heart rate
    valid_heart_rates = [hr for hr in heart_rates if not np.isnan(hr)]
    
    if valid_heart_rates:
        avg_heart_rate_bpm = np.nanmean(valid_heart_rates)  # Average heart rate
        print(f"Estimated Heart Rate: {avg_heart_rate_bpm:.2f} BPM")
    else:
        print("No valid heart rate data found.")

    # Plot the results
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(np.arange(len(chirp_signal)) / SAMPLE_RATE, chirp_signal)
    plt.title("FMCW Chirp Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    
    plt.subplot(4, 1, 2)
    plt.plot(np.arange(len(reflected_signal)) / SAMPLE_RATE, reflected_signal)
    plt.title("Captured Reflected Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    
    plt.subplot(4, 1, 3)
    plt.plot(np.arange(len(heartbeat_phase)) / SAMPLE_RATE, heartbeat_phase)
    plt.title("Extracted Heartbeat Phase Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Phase [radians]")

    # New subplot for Estimated Heart Rate
    plt.subplot(4, 1, 4)
    plt.plot(np.arange(len(heart_rates)) * 2, heart_rates, marker='o')  # Time in seconds (2-second windows)
    plt.title("Estimated Heart Rate Over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Heart Rate [BPM]")
    plt.ylim(0, HEART_RATE_MAX * 60)
    
    plt.tight_layout()
    plt.show()

    # Terminate PyAudio
    p.terminate()

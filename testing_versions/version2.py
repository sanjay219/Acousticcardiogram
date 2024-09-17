import numpy as np
import matplotlib.pyplot as plt
import pyaudio
from scipy.signal import chirp, hilbert, butter, lfilter, savgol_filter
from scipy.fft import fft, fftfreq

# Parameters
SAMPLE_RATE = 48000  # 48 kHz sampling rate
DURATION = 20  # Chirp signal duration (seconds)
CHIRP_SWEEP_TIME = 0.05  # Increased sweep time (50 ms)
FREQ_START = 17000  # Starting frequency of the chirp (17 kHz, inaudible)
FREQ_END = 19000  # Ending frequency of the chirp (19 kHz, inaudible)
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

    for i in range(0, int(sample_rate / CHUNK * duration)):
        # Send chirp segment
        start_idx = i * CHUNK
        end_idx = start_idx + CHUNK
        chirp_segment = chirp_signal[start_idx:end_idx].astype(np.float32)

        # Play chirp and record simultaneously
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
    # Replace NaN, positive and negative infinite values with zero
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

# Perform FFT and estimate heart rate
def estimate_heart_rate(phase_signal, sample_rate):
    # Ensure the phase signal is cleaned of NaNs/Infs
    phase_signal = clean_signal(phase_signal)

    # Apply FFT to the phase signal
    n = len(phase_signal)
    yf = fft(phase_signal)
    xf = fftfreq(n, 1/sample_rate)

    # Focus on the positive frequencies
    pos_indices = np.where(xf > 0)
    xf = xf[pos_indices]
    yf = np.abs(yf[pos_indices])

    # Filter frequencies to focus on the expected heart rate range (1.0 Hz to 2.0 Hz)
    heart_rate_range = (xf >= HEART_RATE_MIN) & (xf <= HEART_RATE_MAX)
    xf_heart = xf[heart_rate_range]
    yf_heart = yf[heart_rate_range]

    # Debugging: Print out the dominant frequencies and amplitudes
    print("Dominant frequencies in heart rate range:", xf_heart)
    print("Corresponding amplitudes:", yf_heart)

    # Check if there are no valid frequencies in the heart rate range
    if len(yf_heart) == 0 or np.all(np.isnan(yf_heart)):
        print("No heart rate found within the expected range or invalid amplitudes.")
        return None

    # Find the dominant frequency in the heart rate range
    dominant_freq = xf_heart[np.argmax(yf_heart)]

    # Convert frequency to BPM (beats per minute)
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
    smoothed_phase = savgol_filter(heartbeat_phase, window_length=101, polyorder=2)
    
    # Apply band-pass filter to isolate the heart rate frequency range
    filtered_phase = bandpass_filter(smoothed_phase, HEART_RATE_MIN, HEART_RATE_MAX, SAMPLE_RATE)
    
    # Estimate heart rate in beats per minute (BPM)
    heart_rate_bpm = estimate_heart_rate(filtered_phase, SAMPLE_RATE)

    if heart_rate_bpm:
        print(f"Estimated Heart Rate: {heart_rate_bpm:.2f} BPM")
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(3, 1, 1)
    plt.plot(t[:len(chirp_signal)], chirp_signal)
    plt.title("FMCW Chirp Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    
    plt.subplot(3, 1, 2)
    plt.plot(t[:len(reflected_signal)], reflected_signal)
    plt.title("Captured Reflected Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    
    plt.subplot(3, 1, 3)
    plt.plot(t[:len(heartbeat_phase)], heartbeat_phase)
    plt.title("Extracted Heartbeat Phase Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Phase [radians]")
    
    plt.tight_layout()
    plt.show()

# Terminate PyAudio
p.terminate()

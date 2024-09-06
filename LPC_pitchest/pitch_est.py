import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt


def lpc_analysis(signal, order):
    """ Perform LPC analysis using autocorrelation method. """
    # Ensure signal is 1-D
    if len(signal.shape) > 1:
        signal = signal.flatten()

    # Compute autocorrelation
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # Keep only positive lags

    # Set up the Yule-Walker equations
    R = np.zeros((order + 1, order + 1))
    r = np.zeros(order + 1)

    for i in range(order + 1):
        r[i] = autocorr[i]
        for j in range(order + 1):
            if i >= j:
                R[i, j] = autocorr[i - j]
                R[j, i] = autocorr[i - j]

    # Solve for LPC coefficients
    lpc_coefficients = np.linalg.solve(R, r)
    return lpc_coefficients


# Specify the path to your audio file
filename = r"C:\Users\ROG ZEPHYRUS\Documents\Seme8\Audio_eng\Audio_eng\LPC_pitchest\aud2.wav"

# Read the audio file
sample_rate, audio = wav.read(filename)

# Check if audio is stereo and convert to mono if needed
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

# Slice the audio
start_sample = int(sample_rate * 0.4)
end_sample = int(sample_rate * 1.5)
audio_sliced = audio[start_sample:end_sample]

# Define parameters
frame_size = 1024
overlap = 512
num_frames = (len(audio_sliced) - overlap) // (frame_size - overlap)
pitch_lpc = np.zeros(num_frames)

# LPC-based pitch estimation
for i in range(num_frames):
    start_idx = i * (frame_size - overlap)
    end_idx = start_idx + frame_size
    frame = audio_sliced[start_idx:end_idx]

    # Apply LPC analysis to obtain coefficients
    lpc_order = 12  # LPC order, adjust as needed
    lpc_coefficients = lpc_analysis(frame, lpc_order)

    # Filter the input signal with LPC coefficients to obtain residual signal
    residual_signal = signal.lfilter(np.concatenate(([1], -lpc_coefficients[1:])), 1, frame)

    # Find peaks in the autocorrelation of the residual signal
    corr = np.correlate(residual_signal, residual_signal, mode='full')
    corr = corr[len(corr) // 2:]  # Keep only positive lags
    peaks, _ = signal.find_peaks(corr)

    # Extract pitch (fundamental frequency) from the first peak
    if len(peaks) > 0:
        pitch_lpc[i] = sample_rate / peaks[0]
    else:
        pitch_lpc[i] = 0  # If no peak is found

# Create time vector for plotting
time = np.arange(num_frames) * (frame_size - overlap) / sample_rate

# Plot pitch over time for LPC
plt.figure()
plt.plot(time, pitch_lpc, linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Pitch (Hz)')
plt.title('Pitch Estimation using LPC')
plt.grid(True)
plt.show()

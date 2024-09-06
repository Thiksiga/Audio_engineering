import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt

# Specify the path to your audio file
filename = r"D:\ACADEMIC E17 DEEE UOP\Sem_8_UOP\EE599 AUDIO ENGINEERING AND ACOUSTICS (TE)(3)\ACTIVITIES\ACTIVITY 1\Dataset\sarigamapa.wav"

# Read the audio file
sample_rate, audio = wav.read(filename)

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
    lpc_coefficients = signal.lpc(frame, lpc_order)

    # Filter the input signal with LPC coefficients to obtain residual signal
    residual_signal = signal.lfilter(lpc_coefficients, 1, frame)

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



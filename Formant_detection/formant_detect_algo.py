import numpy as np
import scipy.signal as signal
import scipy.linalg  # Import for the toeplitz function
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks

# Load the audio file (replace 'audio.wav' with your file path)
sample_rate, audio_data = wavfile.read('C:\\Users\\ROG ZEPHYRUS\\Documents\\Seme8\\Audio_eng\\Audio_eng\\Audio_engineering\\LPC_pitchest\\aud2.wav')

# If stereo, convert to mono by averaging the channels
if audio_data.ndim > 1:
    audio_data = np.mean(audio_data, axis=1)

# Normalize the audio data
audio_data = audio_data / np.max(np.abs(audio_data))

# Apply a Hamming window to the audio data
windowed_signal = audio_data * np.hamming(len(audio_data))


# Frame the signal
def frame_signal(signal, frame_size, hop_size):
    num_frames = int((len(signal) - frame_size) / hop_size) + 1
    frames = np.array([signal[i * hop_size:i * hop_size + frame_size] for i in range(num_frames)])
    return frames


frame_size = 2048
hop_size = 1024
frames = frame_signal(windowed_signal, frame_size, hop_size)


def lpc(frame, order):
    autocorr = np.correlate(frame, frame, mode='full')
    mid = len(autocorr) // 2
    autocorr = autocorr[mid:]

    if len(autocorr) > order + 1:
        autocorr = autocorr[:order + 1]

    R = scipy.linalg.toeplitz(autocorr[:-1])
    LPC_coeffs = np.linalg.solve(R, autocorr[1:])
    return np.concatenate(([1], -LPC_coeffs))


order = 12
lpc_coeffs_list = [lpc(frame, order) for frame in frames]

# You may want to average or select one LPC_coeffs for analysis
lpc_coeffs = lpc_coeffs_list[0]

# Compute the frequency response of the LPC filter
frequencies, response = signal.freqz(1, lpc_coeffs, worN=512, fs=sample_rate)

# Calculate the magnitude of the response
magnitude_response = np.abs(response)

# Find the peaks in the magnitude response
peaks, _ = find_peaks(magnitude_response, distance=20)

# Convert peak indices to frequencies
formant_frequencies = frequencies[peaks]

# Filter out irrelevant peaks (e.g., below 300 Hz or above 4000 Hz for human speech)
formant_frequencies = formant_frequencies[(formant_frequencies > 300) & (formant_frequencies < 4000)]

# Calculate the magnitudes of the detected formant frequencies
formant_magnitudes = magnitude_response[np.isin(frequencies, formant_frequencies)]

# Check lengths for debugging
print(f"Formant Frequencies Length: {len(formant_frequencies)}")
print(f"Formant Magnitudes Length: {len(formant_magnitudes)}")

# Plot the frequency response and formants
plt.figure(figsize=(10, 6))
plt.plot(frequencies, magnitude_response, label='Frequency Response')
plt.plot(formant_frequencies, formant_magnitudes, 'ro', label='Detected Formants')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Formant Detection using LPC')
plt.legend()
plt.show()


# Print the detected formants
print("Detected Formant Frequencies:", formant_frequencies)



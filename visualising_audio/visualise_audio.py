import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pywt

# Load an audio file
filename = "C:\\Users\\ROG ZEPHYRUS\\Documents\\Seme8\\Audio_eng\\Audio_eng\\Audio_engineering\\LPC_pitchest\\aud2.wav"
y, sr = librosa.load(filename, sr=None)

# Time vector for waveform
time = np.linspace(0, len(y) / sr, len(y))

# Plot Waveform
plt.figure(figsize=(14, 20))

plt.subplot(5, 1, 1)
plt.plot(time, y, color='blue')
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot Spectrogram
plt.subplot(5, 1, 2)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')

# Plot Chromagram
plt.subplot(5, 1, 3)
chromagram = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
plt.colorbar()
plt.title('Chromagram')

# Plot MFCC
plt.subplot(5, 1, 4)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
librosa.display.specshow(mfccs, x_axis='time', sr=sr, cmap='inferno')
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')

# Plot Scalogram (Wavelet Transform)
plt.subplot(5, 1, 5)
widths = np.arange(1, 128)
cwtmatr, freqs = pywt.cwt(y, widths, 'morl')

# Take the magnitude for visualization
cwt_abs = np.abs(cwtmatr)
plt.imshow(cwt_abs, extent=[0, len(y) / sr, 1, 128], cmap='PRGn', aspect='auto', origin='lower')
plt.title('Scalogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency Scale')

plt.tight_layout()
plt.show()
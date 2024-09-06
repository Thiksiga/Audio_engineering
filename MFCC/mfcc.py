import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.io import wavfile
import scipy.signal as signal
import IPython.display as ipd

def compute_mfcc(audio, sr, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=40, fmin=0, fmax=None):
    # Step 1: Pre-emphasis filter
    pre_emphasis = 0.97
    emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # Step 2: Framing
    frame_length = int(0.025 * sr)  # 25 ms
    frame_step = int(0.01 * sr)     # 10 ms
    signal_length = len(emphasized_audio)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_audio, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Step 3: Windowing
    frames *= np.hamming(frame_length)

    # Step 4: Fourier Transform and Power Spectrum
    NFFT = n_fft
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    # Step 5: Filter Banks
    fmax = fmax or sr / 2
    mel_points = np.linspace(1125 * np.log(1 + fmin / 700), 1125 * np.log(1 + fmax / 700), n_mels + 2)
    hz_points = 700 * (np.exp(mel_points / 1125) - 1)
    bin_points = np.floor((NFFT + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]   # left
        f_m = bin_points[m]             # center
        f_m_plus = bin_points[m + 1]    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    log_filter_banks = 20 * np.log10(filter_banks)

    # Step 6: Discrete Cosine Transform (DCT)
    mfccs = scipy.fftpack.dct(log_filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    return mfccs

# Load audio file
audio_file = 'C:\\Users\\ROG ZEPHYRUS\\Documents\\Seme8\\Audio_eng\\Audio_eng\LPC_pitchest\\aud2.wav'  # Replace with your audio file path
sr, audio = wavfile.read(audio_file)

# If stereo, take only one channel
if len(audio.shape) > 1:
    audio = audio[:, 0]

# Normalize audio
audio = audio / np.max(np.abs(audio))

# Compute MFCCs
mfccs = compute_mfcc(audio, sr)

# Visualize MFCCs
plt.figure(figsize=(10, 4))
plt.imshow(mfccs.T, aspect='auto', origin='lower', cmap='viridis')
plt.title('MFCC')
plt.xlabel('Frames')
plt.ylabel('MFCC Coefficients')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# Play the audio
ipd.display(ipd.Audio(audio, rate=sr))

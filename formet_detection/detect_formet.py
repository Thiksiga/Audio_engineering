import numpy as np
import librosa
import matplotlib.pyplot as plt


def extract_features(file_path):
    """
    Extract features from the audio file to determine its content format.
    """
    # Load audio file
    y, sr = librosa.load(file_path, duration=30)

    # Extract tempo and spectral centroid
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    return tempo, spectral_centroid


def classify_content(tempo, spectral_centroid):
    """
    Classify the content based on extracted features.
    """
    # Define thresholds for classification
    tempo_threshold = 120  # Example threshold for tempo (beats per minute)
    centroid_threshold = 2000  # Example threshold for spectral centroid (Hz)

    if tempo > tempo_threshold and spectral_centroid > centroid_threshold:
        return 'Music'
    else:
        return 'Speech'


def plot_features(tempo, spectral_centroid):
    """
    Plot extracted features for visualization.
    """
    plt.figure(figsize=(10, 5))

    # Plot tempo
    plt.subplot(1, 2, 1)
    plt.bar(['Tempo'], [tempo], color='blue')
    plt.ylim(0, 200)
    plt.ylabel('Tempo (BPM)')
    plt.title('Tempo')

    # Ensure spectral centroid value is not None
    if spectral_centroid is not None:
        plt.subplot(1, 2, 2)
        plt.bar(['Spectral Centroid'], [spectral_centroid], color='red')
        plt.ylim(0, 5000)
        plt.ylabel('Spectral Centroid (Hz)')
        plt.title('Spectral Centroid')
    else:
        print("Error: Spectral centroid is None")

    plt.tight_layout()
    plt.show()


def main(file_path):
    tempo, spectral_centroid = extract_features(file_path)
    content_type = classify_content(tempo, spectral_centroid)

    print(f'Content type: {content_type}')
    plot_features(tempo, spectral_centroid)



# Example usage
file_path = 'C:\\Users\\ROG ZEPHYRUS\\Documents\\Seme8\\Audio_eng\\Audio_eng\\LPC_pitchest\\aud2.wav'  # Replace with your audio file path
main(file_path)

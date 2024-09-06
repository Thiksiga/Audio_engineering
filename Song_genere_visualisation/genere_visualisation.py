# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import seaborn as sns

# Function to generate synthetic data with a specified number of clusters
def generate_synthetic_data(num_clusters=3, num_samples_per_cluster=100):
    np.random.seed(25)
    data = pd.DataFrame({
        'Tempo': np.concatenate([np.random.uniform(80, 160, num_samples_per_cluster) for _ in range(num_clusters)]),
        'SpectralCentroid': np.concatenate([np.random.uniform(1000, 8000, num_samples_per_cluster) for _ in range(num_clusters)]),
        'Rhythm': np.concatenate([np.random.uniform(0, 1, num_samples_per_cluster) for _ in range(num_clusters)]),
    })
    return data

# Step 1: Generate Synthetic Data with a Variable Number of Clusters
num_clusters = 4  # Change this number as needed
num_samples_per_cluster = 200
data = generate_synthetic_data(num_clusters=num_clusters, num_samples_per_cluster=num_samples_per_cluster)

# Step 2: Feature Extraction
features = data.values

# Step 3: Normalization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 4: Clustering (K-Means) with a Variable Number of Clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Step 5: Visualization of Clusters (PCA)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)

plt.figure(figsize=(12, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=clusters, palette='viridis')
plt.title('PCA - Clusters Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# Step 6: Visualization of Clusters (t-SNE)
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(features_scaled)

plt.figure(figsize=(12, 6))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=clusters, palette='viridis')
plt.title('t-SNE - Clusters Visualization')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Cluster')
plt.show()

# Step 7: Evaluation (Silhouette Score)
silhouette_avg = silhouette_score(features_scaled, clusters)
print(f'Silhouette Score: {silhouette_avg}')

# Step 8: Exploration of Clusters
clustered_data = data.copy()
clustered_data['Cluster'] = clusters
cluster_stats = clustered_data.groupby('Cluster').mean()

print('\nCluster Statistics:')
print(cluster_stats)

# Step 9: Visualization of Feature Distributions within Clusters
plt.figure(figsize=(15, 6))

for i, feature in enumerate(data.columns):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x='Cluster', y=feature, data=clustered_data)
    plt.title(f'Distribution of {feature} by Cluster')

plt.tight_layout()
plt.show()

# Function to classify a sample into a cluster
def classify_sample(tempo, spectral_centroid, rhythm):
    # Manual input of sample features
    new_sample = pd.DataFrame({'Tempo': [tempo], 'SpectralCentroid': [spectral_centroid], 'Rhythm': [rhythm]})

    # Feature extraction
    new_sample_features = new_sample.values

    # Normalization (if used during training)
    new_sample_features = scaler.transform(new_sample_features)

    # Prediction
    predicted_cluster = kmeans.predict(new_sample_features)

    return predicted_cluster[0]

# Example usage 1
tempo_value = 120  # replace with the desired tempo value
spectral_centroid_value = 5000  # replace with the desired spectral centroid value
rhythm_value = 0.8  # replace with the desired rhythm value

predicted_cluster = classify_sample(tempo_value, spectral_centroid_value, rhythm_value)
print(f'The given sample is classified into Cluster {predicted_cluster}')

# Example usage 2
tempo_value = 200  # replace with the desired tempo value
spectral_centroid_value = 4500  # replace with the desired spectral centroid value
rhythm_value = 0.9  # replace with the desired rhythm value

predicted_cluster = classify_sample(tempo_value, spectral_centroid_value, rhythm_value)
print(f'The given sample is classified into Cluster {predicted_cluster}')



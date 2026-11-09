
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 1: Load dataset
df = pd.read_csv("dataset/mushrooms.csv")

# Step 2: Remove class column
# Clustering should not use the actual class labels
X = df.drop("class", axis=1)

# Step 3: Convert categorical data into numerical data
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Step 4: Apply K-Means clustering
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_encoded)

# Step 5: Add cluster labels to dataset
df["Cluster"] = clusters

# Step 6: Display cluster counts
print("\nK-MEANS CLUSTERING RESULTS")

print("\nNumber of records in each cluster:")
print(df["Cluster"].value_counts().sort_index())

# Step 7: Compare clusters with actual classes
# Class is used only for interpretation, not clustering
print("\nCluster and Actual Class Comparison:")
print(pd.crosstab(df["Cluster"], df["class"]))

# Step 8: Calculate silhouette score
score = silhouette_score(
    X_encoded,
    clusters,
    sample_size=2000,
    random_state=42
)

print("\nSilhouette Score:", score)

# Step 9: Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_encoded)

# Step 10: Plot clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    s=10
)

plt.title("K-Means Mushroom Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.savefig("clustering/kmeans_clusters.png")
plt.show()

# Step 11: Save results
df.to_csv("clustering/kmeans_results.csv", index=False)

print("\nK-Means algorithm completed successfully!")

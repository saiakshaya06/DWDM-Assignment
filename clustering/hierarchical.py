
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 1: Load dataset
df = pd.read_csv("dataset/mushrooms.csv")

# Step 2: Select a sample of 1500 records
df_sample = df.sample(n=1500, random_state=42).reset_index(drop=True)

# Step 3: Remove class column
X = df_sample.drop("class", axis=1)

# Step 4: Convert categorical data into numerical data
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Step 5: Apply Agglomerative Hierarchical Clustering
model = AgglomerativeClustering(
    n_clusters=2,
    linkage="ward"
)

clusters = model.fit_predict(X_encoded)

# Step 6: Add cluster labels
df_sample["Cluster"] = clusters

# Step 7: Display cluster counts
print("\nHIERARCHICAL CLUSTERING RESULTS")

print("\nNumber of records in each cluster:")
print(df_sample["Cluster"].value_counts().sort_index())

# Step 8: Compare clusters with actual classes
print("\nCluster and Actual Class Comparison:")
print(pd.crosstab(df_sample["Cluster"], df_sample["class"]))

# Step 9: Calculate silhouette score
score = silhouette_score(
    X_encoded,
    clusters,
    sample_size=1000,
    random_state=42
)

print("\nSilhouette Score:", score)

# Step 10: Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_encoded)

# Step 11: Plot clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    s=15
)

plt.title("Hierarchical Clustering of Mushrooms")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.savefig("clustering/hierarchical_clusters.png")
plt.show()

# Step 12: Save results
df_sample.to_csv(
    "clustering/hierarchical_results.csv",
    index=False
)

print("\nHierarchical Clustering completed successfully!")

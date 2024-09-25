import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the df_models_corrected.csv file
df_models_corrected = pd.read_csv('/home/rafieiva/MyDataBase/codes/PostProcessing/overal_best_performance/df_models_corrected.csv')

# Drop non-numeric columns before imputing and scaling
df_for_imputation = df_models_corrected.drop(columns=['NAME', 'MODEL_NAME'])

# Handle missing values for numeric columns
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df_for_imputation), columns=df_for_imputation.columns)

# Normalize the data
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
import os
os.makedirs("clustering", exist_ok=True)
plt.savefig("clustering/elbow_method.png", dpi=300)


# Fit the KMeans model with the optimal number of clusters (choosing 3 based on visual inspection)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_models_corrected['Cluster'] = kmeans.fit_predict(df_scaled)

# Log the main characteristics of each cluster
cluster_summary = df_models_corrected.groupby('Cluster').mean(numeric_only=True)
cluster_summary['Count'] = df_models_corrected['Cluster'].value_counts()

# Save the cluster summary to a CSV file
cluster_summary.to_csv("clustering/cluster_summary.csv")

# Display the cluster summary
print(cluster_summary)

# Optional: Save the clustered data to a CSV file
df_models_corrected.to_csv("clustering/clustered_data.csv", index=False)

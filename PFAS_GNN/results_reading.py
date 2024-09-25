import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def cluster_hyperparameters():
    # Load your data
    df = pd.read_csv('results/GridSearchResults.csv').dropna()

    # Select relevant features for clustering
    features = ['train_loss', 'val_loss', 'test_loss', 'out_channels', 'epochs', 'lr', 'weight_decay', 'distance_options_km', 'gw_gw_distance_threshold_km']
    X = df[features]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)  # You can change the number of clusters
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Perform PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Add PCA components to the DataFrame for visualization
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

    # Plot the clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=df, palette='viridis')
    plt.title('Clusters of Hyperparameter Combinations')
    plt.tight_layout()
    plt.savefig('results/cluster_hyperparameters.png')  

    # Examine cluster centers
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
    print("Cluster Centers:")
    print(cluster_centers)

    # Save the DataFrame with cluster labels
    df.to_csv('results/clusters_best_models.csv', index=False)


def plot_loss_distributions():
    # Load the data with cluster information
    df = pd.read_csv('results/clusters_best_models.csv').dropna()

    # Filter out specific models and loss values
    df = df[df.gnn_model != 'DeepPReLUModel'].reset_index()
    df = df[(df.train_loss < 10) & (df.val_loss < 10) & (df.test_loss < 10)]
    
    # Melt the dataframe to have a long format for easier plotting with seaborn
    df_melted = df.melt(id_vars=['gnn_model'], value_vars=['train_loss', 'val_loss', 'test_loss'],
                        var_name='Loss_Type', value_name='Loss_Value')

    # Get the number of models for each gnn_model
    model_counts = df_melted['gnn_model'].value_counts()

    # Plot histograms of loss distributions for each gnn_model
    g = sns.FacetGrid(df_melted, col="gnn_model", hue="Loss_Type", col_wrap=3, sharex=False, sharey=False)
    g.map(sns.histplot, "Loss_Value", kde=False, bins=20, alpha=0.7)

    # Add grid, titles, and adjust the legend
    for ax in g.axes.flat:
        gnn_model = ax.get_title().split('=')[-1].strip()
        count = model_counts.get(gnn_model, 0)
        ax.text(0.5, 0.95, f'N={count}', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='black')
        ax.grid(True, alpha=0.3)  # Add grid with low transparency
    
    g.add_legend(loc='upper right', title='Loss Type')
    g.set_titles("{col_name}")
    g.set_axis_labels("Loss Value", "Count")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Loss Distributions for Each GNN Model')

    # Set x-axis and y-axis limits
    g.set(xlim=(0, 10))
    g.set(ylim=(0, 50))

    # Save the plot
    plt.tight_layout()
    plt.savefig('results/loss_distributions_by_gnn_model.png', dpi=600)
    
    plt.show()


if __name__ == '__main__':
    cluster_hyperparameters()
    plot_loss_distributions()



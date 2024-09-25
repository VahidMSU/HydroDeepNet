import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_taylor_diagram(NAME, BASE_PATH, VPUID):
    """
    Plots a Taylor diagram for model comparison based on RMSE, Pearson correlation, and standard deviation.
    The diagram is limited to the angle range from 0° to 180° (positive correlations) and labels the closest model to the origin.
    Different colors are used for different 'cc_model' values.
    
    Parameters:
    model_comparisons (DataFrame): DataFrame containing RMSE, correlation, and standard deviation metrics for each model.
                                   It must include columns: 'NAME', 'std', 'correlation', 'cc_model'.
    """
        # Load the model comparison data
    model_comparisons = pd.read_csv(f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/climate_change_analysis/model_comparisions.csv")
    
    # Extract the required columns for plotting the Taylor diagram
    std_devs = model_comparisons['std'].values
    correlations = model_comparisons['correlation'].values
    model_names = model_comparisons['NAME'].values
    cc_model_names = model_comparisons['cc_model'].values

    # Create a unique color map based on cc_model
    unique_cc_models = np.unique(cc_model_names)
    color_map = {cc_model: plt.cm.tab20(i % 20) for i, cc_model in enumerate(unique_cc_models)}

    # Reference standard deviation (assumed as the mean of the model standard deviations for normalization)
    prism_std = np.mean(std_devs)

    # Setup the polar plot for the Taylor diagram
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    # Initialize a variable to track the closest point to the origin
    min_distance = float('inf')
    closest_model = None
    closest_coords = (0, 0)

    # Plot each model on the Taylor diagram with colors based on 'cc_model'
    for i, model in enumerate(model_names):
        theta = np.arccos(correlations[i])  # Angle based on Pearson correlation
        r = std_devs[i] / prism_std         # Normalized standard deviation

        if not np.isnan(theta) and not np.isnan(r):
            color = color_map[cc_model_names[i]]  # Assign color based on cc_model
            ax.plot(theta, r, 'o', label=model, markersize=6, color=color)
            
            # Calculate the distance from the origin
            distance = np.sqrt((r * np.cos(theta))**2 + (r * np.sin(theta))**2)

            # Check if this model is the closest to the origin
            if distance < min_distance:
                min_distance = distance
                closest_model = model
                closest_coords = (theta, r)

    # Add a label for the closest model to the origin
    if closest_model:
        ax.annotate(f'Closest: {closest_model}', xy=closest_coords, xytext=(10, 10),
                    textcoords='offset points', arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=10, color='red')

    # Customize the plot: limit the angular range from 0° to 180°
    ax.set_thetamin(0)   # Set the minimum angle to 0°
    ax.set_thetamax(180)  # Set the maximum angle to 180°

    # Customize the plot appearance
    ax.set_title('Taylor Diagram: Model Performance (0° to 180°)', fontsize=15)
    ax.set_xlabel('Correlation', fontsize=12)
    ax.set_ylabel('Normalized Standard Deviation', fontsize=12)

    # Create a legend for the 'cc_model' color map
    handles = [plt.Line2D([0], [0], marker='o', color=color_map[cc_model], linestyle='', markersize=6, label=cc_model)
               for cc_model in unique_cc_models]
    ax.legend(handles=handles, title='Climate Change Models', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/climate_change_analysis/taylor_diagram.jpeg", dpi=300)


    

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
class ClimateChangeAnalysis:
    def __init__(self, base_path, vpuid, name):
        self.base_path = base_path
        self.vpuid = vpuid
        self.name = name
        self.target_path = f"/data/MyDataBase/SWATplus_by_VPUID/{vpuid}/huc12/{name}/climate_change_models/"
        self.swatplus_model_path = f"/data/MyDataBase/SWATplus_by_VPUID/{vpuid}/huc12/{name}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut"
        self.save_path = f"/data/MyDataBase/SWATplus_by_VPUID/{vpuid}/huc12/{name}/climate_change_analysis"
        os.makedirs(self.save_path, exist_ok=True)

    def process_model(self, cc_model):
        scenario = cc_model.split('_')[1]
        ensemble = cc_model.split('_')[2]

        print(f"Processing {cc_model} for scenario {scenario}, ensemble {ensemble}")
        cc_model_path = os.path.join(self.target_path, cc_model)
        pcp_files = [file for file in os.listdir(cc_model_path) if file.endswith('.pcp')]
        original_pcp_files = [file for file in os.listdir(self.swatplus_model_path) if file.endswith('.pcp')]
        if not pcp_files:
            return None, None

        all_pcps = []
        prism_pcp = []
        for i, pcp_file in enumerate(pcp_files):
            if i == 0:
                pcp_data = pd.read_csv(os.path.join(self.swatplus_model_path, original_pcp_files[0]), header=None, sep='\s+', skiprows=3, names=['year', 'day', 'pcp'])
                pcp_data['station'] = pcp_file.split('.')[0]
                prism_pcp.append(pcp_data)
            pcp_data = pd.read_csv(os.path.join(cc_model_path, pcp_file), header=None, sep='\s+', skiprows=3, names=['year', 'day', 'pcp'])
            pcp_data['station'] = pcp_file.split('.')[0]
            pcp_data['cc_model'] = cc_model
            pcp_data['scenario'] = scenario
            pcp_data['ensemble'] = ensemble
            all_pcps.append(pcp_data)

        if not all_pcps:
            return None, None

        all_pcps = pd.concat(all_pcps)
        all_pcps['year'] = all_pcps['year'].astype(int)
        all_pcps['day'] = all_pcps['day'].astype(int)
        all_pcps['pcp'] = all_pcps['pcp'].astype(float)
        
        prism_pcp = pd.concat(prism_pcp)
        prism_pcp['year'] = prism_pcp['year'].astype(int)
        prism_pcp['day'] = prism_pcp['day'].astype(int)
        prism_pcp['pcp'] = prism_pcp['pcp'].astype(float)
        prism_pcp = prism_pcp.groupby(['year', 'station'])['pcp'].sum().reset_index()
        prism_pcp = prism_pcp.groupby('year')['pcp'].mean().reset_index()

        all_pcps = all_pcps.groupby(['year','station' ,'cc_model', 'scenario', 'ensemble'])['pcp'].sum().reset_index()
        all_pcps = all_pcps.groupby(['year', 'cc_model', 'scenario', 'ensemble'])['pcp'].mean().reset_index()
        return all_pcps, prism_pcp

    def run_analysis(self):
        cc_models = [d for d in os.listdir(self.target_path) if os.path.isdir(os.path.join(self.target_path, d))]
        all_pcps_models = []
        all_prism_data = None
        
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.process_model, cc_models))
        
        for result in results:
            if result[0] is not None:
                all_pcps_models.append(result[0])
            if result[1] is not None:
                all_prism_data = result[1]
        
        # Concatenate the model data
        if all_pcps_models:
            all_pcps_models = pd.concat(all_pcps_models)
        else:
            print("No model data processed.")
            return

        # Calculate the percentiles for each scenario
        scenarios = ['historical', 'ssp585', 'ssp245', 'ssp370']
        for scenario in scenarios:
            self.plot_percentiles(all_pcps_models, scenario)

        # Plot individual models and ensemble members for historical scenario
        self.plot_individual_models(all_pcps_models, all_prism_data)

    def plot_percentiles(self, all_pcps_models, scenario):
        scenario_data = all_pcps_models[all_pcps_models['scenario'] == scenario]

        percentiles = scenario_data.groupby('year')['pcp'].agg([
            ('2.5th', lambda x: np.percentile(x, 2.5)),
            ('25th', lambda x: np.percentile(x, 25)),
            ('50th', lambda x: np.percentile(x, 50)),
            ('75th', lambda x: np.percentile(x, 75)),
            ('97.5th', lambda x: np.percentile(x, 97.5))
        ])

        plt.figure(figsize=(14, 7))
        plt.fill_between(percentiles.index, percentiles['2.5th'], percentiles['97.5th'], color='skyblue', alpha=0.5)
        plt.plot(percentiles.index, percentiles['2.5th'], color='red', alpha=0.7, linestyle='--')
        plt.plot(percentiles.index, percentiles['97.5th'], color='red', alpha=0.7, linestyle='--')
        plt.plot(percentiles.index, percentiles['25th'], color='green', alpha=0.7, linestyle='-.')
        plt.plot(percentiles.index, percentiles['75th'], color='green', alpha=0.7, linestyle='-.')
        plt.plot(percentiles.index, percentiles['50th'], color='blue', alpha=0.7, linestyle='-')
        self._extracted_from_plot_individual_models_19(
            'Total Annual Precipitation for Scenario ', scenario
        )
        plt.tight_layout()
        plt.legend(['2.5th percentile', '97.5th percentile', '25th percentile', '75th percentile', '50th percentile'])
        plt.savefig(os.path.join(self.save_path, f'{scenario}.jpeg'), dpi=300)
        plt.close()

    def nse(self, obs, sim):
        return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    def mse(self, obs, sim):
        return np.sum((obs - sim) ** 2) / len(obs)
    def pbias(self, obs, sim):
        return np.sum(sim - obs) / np.sum(obs) * 100
    def rmse(self, obs, sim):
        return np.sqrt(np.sum((obs - sim) ** 2) / len(obs))
    def std(self, obs):
        return np.std(obs)
    def correlation(self, obs, sim):
        return np.corrcoef(obs, sim)[0, 1]
    def plot_individual_models(self, all_pcps_models, prism_pcp):
        historical_data = all_pcps_models[all_pcps_models['scenario'] == 'historical']
        unique_models = historical_data['cc_model'].unique()
        historical_data = historical_data[(historical_data['year'] >= 2000) & (historical_data['year'] <= 2014)]
        prism_pcp = prism_pcp[(prism_pcp['year'] >= 2000) & (prism_pcp['year'] <= 2014)]
        model_coparisions = {"NAME": [], "NSE": [], "MSE": [], "PBIAS": [], "RMSE": [], "std": [], "correlation": [], "ensemble": [], "scenario": [], "cc_model": []}
        for model in unique_models:
            model_data = historical_data[historical_data['cc_model'] == model]
            plt.figure(figsize=(10, 6))
            plt.plot(model_data['year'], model_data['pcp'], color='blue', alpha=0.7)
            plt.plot(prism_pcp['year'], prism_pcp['pcp'], color='red', alpha=0.7)
            plt.legend(['LOCA2', 'PRISM'], loc='upper right')
            plt.ylim(600,1400)
            plt.grid(axis='both', linestyle='--', alpha=0.5)
            self._extracted_from_plot_individual_models_19(
                'Total Annual Precipitation for ', model
            )
            ### nse of model_data vs prism_pcp
            nse_val = self.nse(prism_pcp['pcp'].values, model_data.groupby('year')['pcp'].mean().values)
            mse_val = self.mse(prism_pcp['pcp'].values, model_data.groupby('year')['pcp'].mean().values)
            pbias_val = self.pbias(prism_pcp['pcp'].values, model_data.groupby('year')['pcp'].mean().values)
            rmse_val = self.rmse(prism_pcp['pcp'].values, model_data.groupby('year')['pcp'].mean().values)
            std_val = self.std(model_data.groupby('year')['pcp'].mean().values)
            correlation_val = self.correlation(prism_pcp['pcp'].values, model_data.groupby('year')['pcp'].mean().values)

            model_coparisions["NAME"].append(model)
            model_coparisions["NSE"].append(nse_val)
            model_coparisions["MSE"].append(mse_val)
            model_coparisions["PBIAS"].append(pbias_val)
            model_coparisions["RMSE"].append(rmse_val)
            model_coparisions["std"].append(std_val)
            model_coparisions["correlation"].append(correlation_val)
            model_coparisions["ensemble"].append(model.split('_')[2])
            model_coparisions["scenario"].append(model.split('_')[1])
            model_coparisions["cc_model"].append(model.split('_')[0])

            plt.annotate(f'NSE: {nse_val:.2f}\nMSE: {mse_val:.2f}\nPBIAS: {pbias_val:.2f}\nRMSE: {rmse_val:.2f}\nSTD: {std_val:.2f}\nCorrelation: {correlation_val:.2f}', xy=(0.05, 0.85), xycoords='axes fraction')

            plt.tight_layout()
            os.makedirs(os.path.join(self.save_path, "annual_figs"), exist_ok=True)
            plt.savefig(os.path.join(self.save_path, "annual_figs", f'{model}.jpeg'), dpi=300)
            plt.close()

        self.model_coparisions = pd.DataFrame(model_coparisions)
        self.model_coparisions.to_csv(os.path.join(self.save_path, "model_comparisions.csv"), index=False)
        ### rank the models based on NSE
        self.model_coparisions = self.model_coparisions.sort_values('RMSE', ascending=True)
        plt.figure(figsize=(10, 15))
        sns.barplot(data=self.model_coparisions, x='RMSE', y='NAME', color='blue', alpha=0.7)
        plt.xlabel('RMSE')
        plt.ylabel('Model')
        plt.title('Model Comparison based on RSME')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "model_comparisions.jpeg"), dpi=300)
        plt.close()

    # TODO Rename this here and in `plot_percentiles` and `plot_individual_models`
    def _extracted_from_plot_individual_models_19(self, arg0, arg1):
        plt.xlabel('Year')
        plt.ylabel('Total Annual Precipitation (mm)')
        plt.title(f'{arg0}{arg1}')


    def clustering_analysis(self):

        """ NOTE: We are not using this method yet. We need to futher understand the PCA and TSNE plots before using this method. """

        
        ### Perform clustering analysis considering cc_models, ensemble, scenario and NSE, MSE, PBIAS, RMSE as features
        print("Performing clustering analysis")

        # Prepare data
        scaler = StandardScaler()
        features = self.model_coparisions.drop(['NAME', 'ensemble', 'scenario', 'cc_model'], axis=1)
        features = scaler.fit_transform(features)
        additional_features = self.model_coparisions[['cc_model', 'ensemble', 'scenario']]

        # Encode categorical features
        additional_features = pd.get_dummies(additional_features)

        # Combine features
        features = np.hstack([features, additional_features.values])
        input_dim = features.shape[1]

        # Autoencoder parameters
        encoding_dim = 2
        autoencoder = Autoencoder(input_dim, encoding_dim).cuda()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
        num_epochs = 100
        batch_size = 32

        # Convert data to PyTorch tensors
        dataset = torch.tensor(features, dtype=torch.float32).cuda()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train the autoencoder
        for epoch in range(num_epochs):
            for data in dataloader:
                optimizer.zero_grad()
                encoded, decoded = autoencoder(data)
                loss = criterion(decoded, data)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Get encoded features
        autoencoder.eval()
        with torch.no_grad():
            encoded_features, _ = autoencoder(torch.tensor(features, dtype=torch.float32).cuda())
            encoded_features = encoded_features.cpu().numpy()

        # Clustering on encoded features
        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(encoded_features)

        # Add cluster information to the original DataFrame
        self.model_coparisions['cluster'] = clusters
        self.model_coparisions.to_csv(os.path.join(self.save_path, "model_comparisions.csv"), index=False)

        # Plot PCA
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(encoded_features)
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=clusters, palette='viridis', legend='full')
        for i, (index, row) in enumerate(self.model_coparisions.iterrows()):
            plt.text(pca_features[i, 0], pca_features[i, 1], row['cc_model'], fontsize=9)
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.title('PCA Analysis')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "PCA.jpeg"), dpi=300)
        plt.close()

        # Plot TSNE
        tsne = TSNE(n_components=2)
        tsne_features = tsne.fit_transform(encoded_features)
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=clusters, palette='viridis', legend='full')
        for i, (index, row) in enumerate(self.model_coparisions.iterrows()):
            plt.text(tsne_features[i, 0], tsne_features[i, 1], row['cc_model'], fontsize=9)
        plt.xlabel('TSNE1')
        plt.ylabel('TSNE2')
        plt.title('TSNE Analysis')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "TSNE.jpeg"), dpi=300)
        plt.close()

    
if __name__ == "__main__":
    BASE_PATH = "/data/MyDataBase/CIWRE-BAE/SWAT_input/huc12"

    VPUIDS = os.listdir("/data/MyDataBase/SWATplus_by_VPUID")
    for vpuid in VPUIDS:
        if vpuid in ["0000", "0405"]:
            continue
        NAMES = os.listdir(f"/data/MyDataBase/SWATplus_by_VPUID/{vpuid}/huc12")
        NAMES.remove('log.txt')
        for name in NAMES:
            if len(name) <10:
                continue
            analysis = ClimateChangeAnalysis(BASE_PATH, vpuid, name)
            analysis.run_analysis()
            # Plot the Taylor diagram with limited angles, color-coded by 'cc_model', and label the closest point
            plot_taylor_diagram(name, BASE_PATH, vpuid) 

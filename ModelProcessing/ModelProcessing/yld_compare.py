import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import time 
class evaluate_yield:

    def __init__(self, swat_annual_yld):
        self.observed_annual_yld = "/home/rafieiva/MyDataBase/codebase/ModelProcessing/example_inputs/Yield_Michigan_t_ha.csv"
        self.swat_annual_yld = swat_annual_yld

    def calculate_metrics(self):   
        # Calculate metrics
        ## first drop rows with nan values
        self.merged_df = self.merged_df.dropna()
        self.merged_df['yld(t/ha)_obs'] = self.merged_df['yld(t/ha)_obs'].astype(float)
        self.merged_df['yld(t/ha)_sim'] = self.merged_df['yld(t/ha)_sim'].astype(float)
        print(self.merged_df['yld(t/ha)_obs'].values)
        assert not np.all(self.merged_df["yld(t/ha)_obs"].values == 0), "observed values are"
        assert not np.all(self.merged_df["yld(t/ha)_sim"].values == 0), "simulated values are"
        ## now drop rows with zero values
        self.merged_df = self.merged_df[(self.merged_df["yld(t/ha)_obs"] != 0) & (self.merged_df["yld(t/ha)_sim"] != 0)]
        self.nse_val = self.nse(self.merged_df["yld(t/ha)_obs"], self.merged_df["yld(t/ha)_sim"])
        self.rmse_val = self.rmse(self.merged_df["yld(t/ha)_obs"], self.merged_df["yld(t/ha)_sim"])
        self.kge_val = self.kge(self.merged_df["yld(t/ha)_obs"], self.merged_df["yld(t/ha)_sim"])
        self.mape_val = self.mape(self.merged_df["yld(t/ha)_obs"], self.merged_df["yld(t/ha)_sim"])
        self.pbias_val = self.pbias(self.merged_df["yld(t/ha)_obs"], self.merged_df["yld(t/ha)_sim"])
        # Print performance metrics
        print(f"NSE: {self.nse_val:.2f}, RMSE: {self.rmse_val:.2f}, KGE: {self.kge_val:.2f}, MAPE: {self.mape_val:.2f}, PBIAS: {self.pbias_val:.2f}")

        return self.nse_val, self.rmse_val, self.kge_val, self.mape_val, self.pbias_val

    def plot_yield_comparison(self, output_dir=os.getcwd()):
        # Filter data starting from 2001
        self.merged_df = self.merged_df[self.merged_df["year"] >= 2001]

        # Scatter plot for comparing observed vs. simulated yields
        plt.scatter(self.merged_df["yld(t/ha)_obs"], self.merged_df["yld(t/ha)_sim"], color="blue", label="Observed vs Simulated")

        plt.xlabel("Observed Yield (t/ha)")
        plt.ylabel("Simulated Yield (t/ha)")

        # Add a line of equality for reference
        max_yield = max(self.merged_df["yld(t/ha)_obs"].max(), self.merged_df["yld(t/ha)_sim"].max())
        plt.plot([0, max_yield], [0, max_yield], color='red', linestyle='--', label='1:1 Line')

        # Legend and grid
        plt.legend(loc='upper left')
        plt.grid(linestyle='--', linewidth=0.5)

        # Add score metrics as annotation on the plot
        plt.text(0.7, 0.3, f"NSE: {self.nse_val:.2f}\nRMSE: {self.rmse_val:.2f}\nKGE: {self.kge_val:.2f}\nMAPE: {self.mape_val:.2f}\nPBIAS: {self.pbias_val:.2f}",
                fontsize=12, transform=plt.gca().transAxes)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(output_dir, "YLD"), exist_ok=True)

        # Save the plot as a PNG file
        plt.savefig(os.path.join(output_dir, "YLD", f"{self.nse_val:.2f}_yield_comparison_{time.time()}.png"), dpi=300)

        # Close the plot to free up memory
        plt.close()


    # Define performance metrics
    @staticmethod
    def nse(obs, sim):
        ### assert obs is not all zeros
        assert not np.all(obs == 0), "observed values are"
        return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

    @staticmethod
    def rmse(obs, sim):
        return np.sqrt(np.mean((obs - sim) ** 2))

    @staticmethod
    def kge(obs, sim):
        obs_mean = np.mean(obs)
        sim_mean = np.mean(sim)
        obs_std = np.std(obs)
        sim_std = np.std(sim)
        r = np.corrcoef(obs, sim)[0, 1]
        beta = sim_mean / obs_mean
        gamma = (sim_std / sim_mean) / (obs_std / obs_mean)
        kge_value = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
        return kge_value

    @staticmethod
    def mape(obs, sim):
        return np.mean(np.abs((obs - sim) / obs)) * 100

    @staticmethod
    def pbias(obs, sim):
        assert not np.all(obs == 0), "observed values are"
        return np.sum(sim - obs) / np.sum(obs) * 100

    def read_obs_swat_yld(self):
        # Load observed annual yield data
        observed_annual_yld = pd.read_csv(self.observed_annual_yld)
        # Average yield among all counties
        observed_annual_yld = observed_annual_yld.drop(columns=['County']).groupby(["State"]).mean().reset_index()
        # Sort column names by name
        observed_annual_yld = observed_annual_yld.reindex(sorted(observed_annual_yld.columns), axis=1)
        # Transpose and reset index to get the correct format
        obs_df = observed_annual_yld.T.reset_index()
       
        obs_df.columns = ["year", "yld(t/ha)"]
        obs_df = obs_df.drop(index=0)
        ## save to csv
        obs_df["yld(t/ha)"] = obs_df["yld(t/ha)"].astype(float)
    
        ### assert not all obs_df is zeros
        assert not np.all(obs_df["yld(t/ha)"].values == 0), "observed values are"
        # Change the values in column 'year' from "Yield2004" to just the year (2004, 2005, etc.)
        obs_df["year"] = obs_df["year"].str.replace("Yield", "").astype(int)

        swat_bsn_yld = pd.read_csv(self.swat_annual_yld, skiprows=1, engine='python', sep='\s+')
        assert "plant_name" in swat_bsn_yld.columns, "plant_name column not found in swat_bsn_yld"
        # Filter for corn
        swat_bsn_yld = swat_bsn_yld[swat_bsn_yld.plant_name == "corn"]
        if swat_bsn_yld.empty:
            raise ValueError("No corn yield data found in the SWAT output file")
        # Merge observed and simulated data by year
        self.merged_df = pd.merge(obs_df, swat_bsn_yld[['year', "yld(t/ha)"]], on="year", how="inner")

        # Rename columns for clarity
        self.merged_df.columns = ["year", "yld(t/ha)_obs", "yld(t/ha)_sim"]

        ## convert to float
        self.merged_df["yld(t/ha)_obs"] = self.merged_df["yld(t/ha)_obs"].astype(float)
        self.merged_df["yld(t/ha)_sim"] = self.merged_df["yld(t/ha)_sim"].astype(float)

        # Print merged data for validation
        #print(self.merged_df)


    def process_and_evaluate(self):
        try:
            self.read_obs_swat_yld()
            # Calculate and print performance metrics
            nse_value, rmse_value, kge_value, mape_value, pbias_value = self.calculate_metrics()
        
            return nse_value, rmse_value, kge_value, mape_value, pbias_value
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None, None, None

if __name__ == "__main__":
    path = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/04166000/SWAT_gwflow_MODEL/Scenarios/verification_stage_0/basin_crop_yld_yr.txt"
    eval = evaluate_yield(path)
    yld_nse_value, yld_rmse_value, yld_kge_value, yld_mape_value, yld_pbias_value = eval.process_and_evaluate()
    eval.plot_yield_comparison(output_dir=os.getcwd())
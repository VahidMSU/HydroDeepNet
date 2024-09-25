import os
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

class PrecipitationAnalyzer:
    def __init__(self, base_path):
        self.base_path = base_path

    def convert_to_date(self, year, doy):
        """
        Convert year and day of year (DOY) to a datetime object.
        """
        return pd.to_datetime(year * 1000 + doy, format='%Y%j')

    def read_prism_pcp(self, vpuid, level, name):
        """
        Reads and processes PRISM precipitation data.
        """
        prism_path = f"{self.base_path}/SWATplus_by_VPUID/{vpuid}/{level}/{name}/PRISM/"
        prism_files = [f for f in os.listdir(prism_path) if 'pcp' in f]
        all_prism_pcps = []

        for pcp in prism_files:
            pcp_path = os.path.join(prism_path, pcp)
            pcp_df = pd.read_csv(pcp_path, skiprows=3, names=['year', 'doy', 'pcp'], sep='\s+')
            pcp_df['date'] = self.convert_to_date(pcp_df['year'], pcp_df['doy'])
            pcp_df = pcp_df[(pcp_df['date'] >= '2003-01-01') & (pcp_df['date'] <= '2014-12-31')]
            pcp_df.set_index('date', inplace=True)
            all_prism_pcps.append(pcp_df)

        if all_prism_pcps:
            prism_pcp = pd.concat(all_prism_pcps)
            prism_pcp = prism_pcp.resample('ME').sum().reset_index()
            return prism_pcp
        else:
            return None

    def read_cc_model_pcp(self, vpuid, level, name, cc_model):
        """
        Reads and processes climate change model precipitation data.
        """
        sim_path = f"{self.base_path}/SWATplus_by_VPUID/{vpuid}/{level}/{name}/climate_change_models/{cc_model}/"
        sim_files = [f for f in os.listdir(sim_path) if f.endswith('.pcp')]
        all_cc_pcps = []

        for sim_pcp in sim_files:
            sim_pcp_path = os.path.join(sim_path, sim_pcp)
            sim_pcp_df = pd.read_csv(sim_pcp_path, skiprows=3, names=['year', 'doy', 'pcp'], sep='\s+')
            sim_pcp_df['date'] = self.convert_to_date(sim_pcp_df['year'], sim_pcp_df['doy'])
            sim_pcp_df = sim_pcp_df[(sim_pcp_df['date'] >= '2003-01-01') & (sim_pcp_df['date'] <= '2014-12-31')]
            sim_pcp_df.set_index('date', inplace=True)
            all_cc_pcps.append(sim_pcp_df)

        if all_cc_pcps:
            cc_pcp = pd.concat(all_cc_pcps)
            cc_pcp = cc_pcp.resample('ME').sum().reset_index()
            return cc_pcp
        else:
            return None

    def plot_precipitation(self, vpuid, name, cc_model, prism_pcp, cc_pcp):
        """
        Plots the precipitation data from PRISM and the climate change model.
        """
        sim_path = f"{self.base_path}/SWATplus_by_VPUID/{vpuid}/huc12/{name}/climate_change_models/{cc_model}/"

        plt.figure(figsize=(12, 6))
        plt.plot(prism_pcp['date'], prism_pcp['pcp'], label='PRISM')
        plt.plot(cc_pcp['date'], cc_pcp['pcp'], label=cc_model)
        plt.xlabel('Month')
        plt.ylabel('Cumulative Precipitation (mm)')
        plt.title(f'PRISM vs. {cc_model} Precipitation')
        plt.legend()
        plt.grid()
        plt.savefig(f"{sim_path}/{cc_model}_vs_PRISM.png")
        plt.close()

    def analyze_precipitation(self, task):
        vpuid, name, level, cc_model = task
        logging.info(f"Processing {vpuid} {name} {cc_model}")

        prism_pcp = self.read_prism_pcp(vpuid, level, name)
        if prism_pcp is None:
            logging.warning(f"No PRISM data found for {vpuid} {name}")
            return

        cc_pcp = self.read_cc_model_pcp(vpuid, level, name, cc_model)
        if cc_pcp is None:
            logging.warning(f"No climate change model data found for {vpuid} {name} {cc_model}")
            return

        self.plot_precipitation(vpuid, name, cc_model, prism_pcp, cc_pcp)

    def run_analysis(self, level='huc12', parallel=True):
        tasks = []
        vpuid_list = os.listdir(f"{self.base_path}/SWATplus_by_VPUID/")

        for vpuid in vpuid_list:
            name_list = os.listdir(f"{self.base_path}/SWATplus_by_VPUID/{vpuid}/{level}/")
            if 'log.txt' in name_list:
                name_list.remove('log.txt')

            for name in name_list:
                cc_models = os.listdir(f"{self.base_path}/SWATplus_by_VPUID/{vpuid}/{level}/{name}/climate_change_models/")
                for cc_model in cc_models:
                    if 'historical' not in cc_model:
                        continue
                    tasks.append((vpuid, name, level, cc_model))

        if parallel:
            with Pool(processes=20) as pool:  # Adjust the number of processes as needed
                pool.map(self.analyze_precipitation, tasks)
        else:
            for task in tasks:
                self.analyze_precipitation(task)


if __name__ == "__main__":
    base_path = "E:/MyDataBase"
    analyzer = PrecipitationAnalyzer(base_path)
    analyzer.run_analysis(parallel=True)

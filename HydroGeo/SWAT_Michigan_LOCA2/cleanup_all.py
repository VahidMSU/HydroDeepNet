import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def clean_up(path):
    try:
        files = os.listdir(path)
        for file in files:
            os.remove(os.path.join(path, file))
        logging.info(f"Cleaned up {path}")
    except Exception as e:
        logging.error(f"Failed to clean up {path}: {e}")

def process_cc_model(path, cc_model):
    clean_up(os.path.join(path, cc_model))

def process_name(path, NAME, VPUID):
    try:
        # Delete historical figures
        historical_figs_daily = os.path.join(path, VPUID, 'huc12', NAME, 'figures_SWAT_gwflow_MODEL_historical_daily')
        if os.path.exists(historical_figs_daily):
            logging.info(f"Deleting {historical_figs_daily}")
            shutil.rmtree(historical_figs_daily)

        historical_figs_monthly = os.path.join(path, VPUID, 'huc12', NAME, 'figures_SWAT_gwflow_MODEL_historical_monthly')
        if os.path.exists(historical_figs_monthly):
            logging.info(f"Deleting {historical_figs_monthly}")
            shutil.rmtree(historical_figs_monthly)

        # Delete historical performance scores
        logfile = os.path.join(path, VPUID, 'huc12', NAME, 'historical_performance_scores.txt')
        if os.path.exists(logfile):
            logging.info(f"Deleting {logfile}")
            os.remove(logfile)

        # Clean up climate change models
        cc_models_path = os.path.join(path, VPUID, 'huc12', NAME, 'climate_change_models')
        cc_models = os.listdir(cc_models_path)
        with ThreadPoolExecutor() as executor:
            executor.map(process_cc_model, [cc_models_path] * len(cc_models), cc_models)
    except Exception as e:
        logging.error(f"Failed to process {NAME} in {VPUID}: {e}")

def process_vpuid(path, VPUID):
    try:
        huc12_path = os.path.join(path, VPUID, 'huc12')
        NAMES = os.listdir(huc12_path)
        if "log.txt" in NAMES:
            NAMES.remove("log.txt")

        with ThreadPoolExecutor() as executor:
            executor.map(process_name, [path] * len(NAMES), NAMES, [VPUID] * len(NAMES))
    except Exception as e:
        logging.error(f"Failed to process VPUID {VPUID}: {e}")

def main():
    base_path = "/data/MyDataBase/SWATplus_by_VPUID"
    VPUIDs = os.listdir(base_path)
    with ThreadPoolExecutor(max_workers=50) as executor:
        executor.map(process_vpuid, [base_path] * len(VPUIDs), VPUIDs)

if __name__ == "__main__":
    main()

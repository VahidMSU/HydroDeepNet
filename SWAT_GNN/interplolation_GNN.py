import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
def setup_logger(log_file_path):    
    # Clear the log file at the start
    with open(log_file_path, 'w') as f:
        f.write("")

    # Create a custom logger
    logger = logging.getLogger(log_file_path)

    # Set the logging level
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(message)s')  # Only log the message itself
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

path = "/data/MyDataBase/HuronRiverPFAS/Huron_River_Grid_250m_with_features.pkl"
logger = setup_logger("log.txt")
df = pd.read_pickle(path)
logger.info(f"Columns: {list(df.columns)}")
logger.info(f"Shape: {df.shape}")
logger.info(f"Head: {df.head()}")
logger.info(f"dtypes: {df.dtypes}")

##plot
fig, ax = plt.subplots()
print(df.shape)
df = df[df['obs_AQ_THK_1_250m'] > 0]
print(df.shape)
df.plot(ax=ax, color='blue', linewidth=0.5, column='obs_AQ_THK_1_250m', legend=True)
## add colorbar
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=100))
sm._A = []
cbar = plt.colorbar(sm)
cbar.set_label('obs_AQ_THK_1_250m')
plt.title("Huron River PFAS")

plt.savefig("observed_aq_thk_1.png")


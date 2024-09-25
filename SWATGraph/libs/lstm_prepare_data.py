import logging
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

def prepare_data(temporal_data, graph, target_channel_id, streamflow_data_path):
    data = []
    targets = []
    start_year=2000
    end_year=2002
    logging.info(f"Preparing data for target channel ID: {target_channel_id}")

    if target_channel_id not in temporal_data:
        logging.error(f"Target channel ID {target_channel_id} not found in temporal data.")
        return np.array(data), np.array(targets)

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    # Define start and end indices based on the year range
    start_idx = (start_date - datetime(2000, 1, 1)).days
    end_idx = (end_date - datetime(2000, 1, 1)).days + 1  # end_date inclusive
    num_days = (end_date - start_date).days + 1  # Corrected calculation for number of days
    logging.info(f"Start index: {start_idx}, End index: {end_idx}, Number of days: {num_days}")
    
    # Load and filter the streamflow data
    streamflow_df = pd.read_csv(streamflow_data_path)
    logging.info(f"Streamflow data shape: {streamflow_df.shape}")

    # Convert date column to datetime
    streamflow_df['date'] = pd.to_datetime(streamflow_df['date'])

    # Filter by start and end year
    streamflow_df = streamflow_df[(streamflow_df['date'] >= start_date) & (streamflow_df['date'] <= end_date)]
    logging.info(f"Streamflow data shape after filtering: {streamflow_df.shape}")

    # Drop the date column and prepare targets
    targets_array = streamflow_df['streamflow'].values

    # Ensure the targets length matches the data length after num_days offset
    if len(targets_array) < num_days:
        logging.error("The length of the target streamflow data is less than the number of days in the range.")
        return np.array(data), np.array(targets)
    
    for node in graph.nodes:
        # Only if the node has node_role and is the target channel
        if graph.nodes[node].get('node_role') == 'channel' and node == target_channel_id:
            for i in range(start_idx, min(end_idx, len(temporal_data[node]))):
                data.append(temporal_data[node][i : i + 1])
                targets.append(targets_array[i])

    if not data:
        logging.warning(f"No data prepared for target channel ID {target_channel_id}.")

    logging.info(f"Prepared {len(data)} data samples and {len(targets)} target samples.")

    return np.array(data), np.array(targets)

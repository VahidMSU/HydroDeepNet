import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import datetime
from matplotlib.dates import DateFormatter, MonthLocator
# Import the built-in statistics module with an alias to avoid conflict
import statistics as py_statistics
# Import seaborn after the statistics import to avoid the conflict
import seaborn as sns
from scipy.signal import savgol_filter

base_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/"

NAMES = os.listdir(base_path)

NAMES = [x for x in NAMES if os.path.isdir(os.path.join(base_path, x))] 

def load_streamflow_data(file_path):
    """
    Load streamflow data from a CSV file and convert date to datetime format.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing streamflow data
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with streamflow data and properly formatted date
    """
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

def separate_baseflow(df, method='lyne_hollick', alpha=0.925, passes=3):
    """
    Separate baseflow from total streamflow using different methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with streamflow data
    method : str, optional
        Method to use for baseflow separation. Options:
        - 'lyne_hollick': Digital filter method (Lyne and Hollick, 1979)
        - 'local_minimum': Local minimum method
        - 'sliding_interval': Sliding interval method
        - 'chapman': Chapman algorithm
    alpha : float, optional
        Filter parameter for the digital filter method (0.9-0.95 recommended)
    passes : int, optional
        Number of filter passes for the digital filter method
        
    Returns:
    --------
    df_result : pandas.DataFrame
        DataFrame with columns for total streamflow, baseflow, and direct runoff
    """
    if 'streamflow' not in df.columns:
        raise ValueError("DataFrame must contain a 'streamflow' column")
    
    # Create a copy of the dataframe to avoid modifying the original
    df_result = df.copy()
    
    # Ensure no missing values
    df_result['streamflow'] = df_result['streamflow'].interpolate(method='linear')
    
    # Digital filter method (Lyne and Hollick, 1979)
    if method == 'lyne_hollick':
        # Initialize baseflow array
        baseflow = np.zeros(len(df_result))
        streamflow = df_result['streamflow'].values
        
        # Apply the filter multiple times as specified
        for _ in range(passes):
            # First value of baseflow is equal to first value of streamflow
            baseflow[0] = streamflow[0]
            
            # Apply the filter formula
            for i in range(1, len(streamflow)):
                # The Lyne and Hollick filter formula
                baseflow[i] = alpha * baseflow[i-1] + (1-alpha)/2 * (streamflow[i] + streamflow[i-1])
                
                # Constrain baseflow to be less than or equal to streamflow
                if baseflow[i] > streamflow[i]:
                    baseflow[i] = streamflow[i]
                
                # Baseflow cannot be negative
                if baseflow[i] < 0:
                    baseflow[i] = 0
        
        df_result['baseflow'] = baseflow
        
    # Local minimum method
    elif method == 'local_minimum':
        # Set window size for finding local minima (e.g., 5-day window)
        window_size = 5
        
        streamflow = df_result['streamflow'].values
        baseflow = np.zeros(len(streamflow))
        
        # Find local minima
        for i in range(len(streamflow)):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(streamflow), i + window_size//2 + 1)
            baseflow[i] = min(streamflow[start_idx:end_idx])
        
        # Smooth the baseflow using a moving average
        baseflow = pd.Series(baseflow).rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        
        df_result['baseflow'] = baseflow
        
    # Sliding interval method
    elif method == 'sliding_interval':
        # Set interval length in days
        interval = 5
        
        streamflow = df_result['streamflow'].values
        baseflow = np.zeros(len(streamflow))
        
        # Calculate the minimum flow in each interval
        for i in range(len(streamflow)):
            start_idx = max(0, i - interval)
            end_idx = min(len(streamflow), i + interval + 1)
            baseflow[i] = min(streamflow[start_idx:end_idx])
        
        # Smooth the baseflow using a moving average
        baseflow = pd.Series(baseflow).rolling(window=interval*2, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        
        df_result['baseflow'] = baseflow
        
    # Chapman algorithm
    elif method == 'chapman':
        # Set recession constant (typically 0.925-0.98)
        k = 0.98
        
        streamflow = df_result['streamflow'].values
        baseflow = np.zeros(len(streamflow))
        
        # Initialize baseflow
        baseflow[0] = streamflow[0] * 0.5  # Assumption: 50% of initial streamflow is baseflow
        
        # Apply Chapman algorithm
        for i in range(1, len(streamflow)):
            baseflow[i] = k * baseflow[i-1] + (1-k) * streamflow[i]
            
            # Ensure baseflow is not greater than streamflow
            if baseflow[i] > streamflow[i]:
                baseflow[i] = streamflow[i]
        
        df_result['baseflow'] = baseflow
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'lyne_hollick', 'local_minimum', 'sliding_interval', or 'chapman'.")
    
    # Calculate direct runoff
    df_result['direct_runoff'] = df_result['streamflow'] - df_result['baseflow']
    
    # Calculate baseflow index (BFI)
    total_streamflow = df_result['streamflow'].sum()
    total_baseflow = df_result['baseflow'].sum()
    
    if total_streamflow > 0:
        bfi = total_baseflow / total_streamflow
    else:
        bfi = np.nan
    
    # Add BFI as an attribute to the dataframe
    df_result.attrs['bfi'] = bfi
    df_result.attrs['separation_method'] = method
    
    return df_result

for name in NAMES:
    path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{name}/streamflow_data/"

    daily_streamflow_files = [f for f in os.listdir(path) if f.endswith('_daily.csv')]

    for file in daily_streamflow_files:
        file_path = os.path.join(path, file)

        # Read the CSV file into a DataFrame
        df = load_streamflow_data(file_path)

        break

print(df.head())



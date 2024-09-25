import pandas as pd

# Load the CSV file
path = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/04115000/Graphs/cell_hru_riv_wst.csv"
df = pd.read_csv(path)

# Group by hru_id and count unique channel_ids
hru_channel_counts = df.groupby('hru_id')['channel_id'].nunique()

# Filter HRUs that are connected to more than one unique channel
hru_with_multiple_channels = hru_channel_counts[hru_channel_counts > 1]

# Print the results
if not hru_with_multiple_channels.empty:
    print(f"Number of HRUs connected to more than one unique channel: {len(hru_with_multiple_channels)}")
    print("HRUs connected to more than one unique channel:")
    print(hru_with_multiple_channels)
else:
    print("No HRU is connected to more than one unique channel.")

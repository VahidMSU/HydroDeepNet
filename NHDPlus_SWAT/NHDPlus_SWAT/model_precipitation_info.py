import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_annual_precipitation(BASE_PATH, VPUID, LEVEL, NAME):
	pcps = []

	PRISM_path = os.path.join(BASE_PATH, "SWATplus_by_VPUID", VPUID, LEVEL, NAME, 'PRISM')
	pcps_path = glob.glob(PRISM_path + '\\*.pcp')

	for pcp_path in pcps_path:
		pcp = pd.read_csv(pcp_path, skiprows=3, sep='\s+', names=['year', 'day', 'pcp'])
		pcp['NAME'] = NAME
		pcp['LEVEL'] = LEVEL
		pcp['date'] = pd.date_range(start='1990-01-01', end='2022-12-31')
		pcp['date'] = pd.to_datetime(pcp['date'])
		pcp['station'] = os.path.basename(pcp_path).split('.')[0]
		pcp['nstation'] = len(pcps_path)
		pcps.append(pcp)

	# Concatenate all dataframes
	pcps = pd.concat(pcps)
	pcps.set_index('date', inplace=True)

	# Calculate the average sum of annual precipitation
	nstation = pcps[pcps['NAME'] == NAME]['nstation'].unique()[0]
	pcps_annual = pcps[pcps['NAME'] == NAME].resample('YE').sum()['pcp'] / nstation
	pcps_annual.index = pcps_annual.index.to_period('Y')

	# Calculate the average sum of monthly precipitation
	pcps_monthly = pcps[pcps['NAME'] == NAME].resample('ME').sum()['pcp'] / nstation
	pcps_monthly.index = pcps_monthly.index.to_period('M')

	# Plot annual precipitation
	plt.figure(figsize=(12, 4))
	ax1 = plt.subplot(1, 2, 1)
	pcps_annual.plot(kind='bar', ax=ax1)
	_extracted_from_plot_annual_precipitation_34(
		'Annual Precipitation ', NAME, 'Year'
	)
	# Calculate percentiles and add horizontal lines with labels on the lines
	percentiles = [75, 50, 25]
	colors = ['r', 'g', 'b']
	for percentile, color in zip(percentiles, colors):
		y = np.percentile(pcps_annual.values, percentile)
		plt.axhline(y=y, color=color, linestyle='-', linewidth=1)
		plt.text(ax1.get_xlim()[1], y, f' {percentile}th: {y:.0f} mm', va='center', ha='right', color='black')

	# Set x-tick labels (assuming you want to label each year)
	ax1.set_xticklabels([str(year) for year in range(pcps_annual.index[0].start_time.year, 
													pcps_annual.index[-1].start_time.year + 1)], rotation=90)

	# Plot average monthly precipitation
	ax2 = plt.subplot(1, 2, 2)
	pcps_monthly.plot(ax=ax2)
	_extracted_from_plot_annual_precipitation_34(
		'Average Monthly Precipitation ', NAME, 'Month'
	)
	plt.tight_layout()
	plt.savefig(os.path.join(BASE_PATH,'SWATplus_by_VPUID',VPUID ,LEVEL, NAME,'PRISM','Annual_Precipitation.jpeg'), dpi=300)
	plt.savefig(os.path.join(BASE_PATH,'Documentations','Annual_Precipitation',f'Annual_Precipitation_{NAME}.jpeg'), dpi=300)

# TODO Rename this here and in `plot_annual_precipitation`
def _extracted_from_plot_annual_precipitation_34(arg0, NAME, arg2):
	plt.title(f'{arg0}{NAME}')
	plt.xlabel(arg2)
	plt.ylabel('mm')

if __name__ == "__main__":
	BASE_PATH = r'/data/MyDataBase/SWATGenXAppData/'
	VPUID = "0407"
	LEVEL = "huc12"
	NAME = "40700040303"
	plot_annual_precipitation(BASE_PATH, VPUID, LEVEL, NAME)
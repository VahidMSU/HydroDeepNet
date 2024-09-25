import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from parallel_loading import loading_dataset

if __name__ == '__main__':
	BASE_PATH = 'D:/MyDataBase'
	variables, longitudes, latitudes , SNODAS_stations, datasets = loading_dataset(BASE_PATH)


	variable_names = [ 'average_temperature',             
			'melt_rate',               
			'snow_layer_thickness',           
				'snow_water_equivalent',          
			'snowpack_sublimation_rate',      
			'non_snow_accumulation',       
			'snow_accumulation']

	converters = [1, 1/100, 1, 1, 1/ 100, 1/10, 1/10]

	units = ['Kelvin',
	'mm',
	'mm',
	'mm',
	'mm',
	'mm',
	'kg/sqm',
	'kg/sqm']

	# Define the start year and the total number of days
	start_year = 2004  # Example start year
	for variable_name,converter, unit in zip(variable_names, converters, units): 
		# Assuming 'melt_rate' is a 3D array with shape (days, height, width)
		melt_rate = converter*variables[variable_name][:]  # Example array

		num_days = melt_rate.shape[0]

		# Create a date range
		dates = pd.date_range(start=datetime(start_year, 1, 1), periods=num_days)

		# Calculate the number of years
		num_years = len(np.unique(dates.year))

		# Reshape melt_rate to have years, months, and days
		annual_melt_rate = np.empty((num_years, 12, *melt_rate.shape[1:]))

		# Calculate monthly mean melt rate
		for year, month in itertools.product(range(num_years), range(1, 13)):
			# Select data for the current month and year
			mask = (dates.year == start_year + year) & (dates.month == month)
			monthly_data = melt_rate[mask, :, :]

				# Check if there are any non-NaN values before calculating the mean
			monthly_mean = (
				np.nanmean(monthly_data, axis=0)
				if np.any(~np.isnan(monthly_data))
				else np.full(melt_rate.shape[1:], np.nan)
			)
			annual_melt_rate[year, month-1, :, :] = monthly_mean

		# Calculate annual mean melt rate
		with np.errstate(invalid='ignore'):
			annual_mean_melt_rate = np.nanmean(annual_melt_rate, axis=(0, 1))

		# Plotting the monthly mean for each month across all years
		fig, axs = plt.subplots(3, 4, figsize=(15, 10))  # Adjust the size as needed
		axs = axs.flatten()
		month_label = ['Jan','Feb','March','April','May','June','July','Agu','Sep','Oct','Nov','Dec']

		for month in range(12):
			# Only plot if the monthly mean contains non-NaN values
			if not np.all(np.isnan(annual_melt_rate[:, month, :, :])):
				avm=np.nanmean(annual_melt_rate[:, month, :, :], axis=0)
				im = axs[month].imshow(avm, cmap='winter_r', vmax=np.nanpercentile(avm, 97.5), vmin=np.nanpercentile(avm, 2.5))
				axs[month].set_title(f'Mean. {variable_name}({unit}):{month_label[month]}')
				fig.colorbar(im, ax=axs[month])
			else:
				# If the entire month has no data, display a message
				axs[month].text(0.5, 0.5, 'No data for this month', horizontalalignment='center', verticalalignment='center', transform=axs[month].transAxes)
				axs[month].set_title(f'Mean. Monthly: {month_label[month]}')

		plt.tight_layout()
		plt.savefig(f'D:\MyDataBase\Documentations\SNODAS_annual_monthly_figures\Average_Monthly_{variable_name}_Michigan_SNODAS.jpeg',dpi=300)
		plt.title(f'Montly {variable_name}({unit})')
		plt.show()

		# Plotting the annual mean across all months and years
		plt.figure(figsize=(10, 10))  # Adjust the size as needed
		im = plt.imshow(annual_mean_melt_rate, cmap='winter_r', vmax=np.nanpercentile(annual_mean_melt_rate,97.25), vmin=np.nanpercentile(annual_mean_melt_rate,2.5))
		plt.title(f'Mean Annual {variable_name}({unit})')
		plt.colorbar(im)
		plt.savefig(f'/data/MyDataBase/SWATGenXAppData/Documentations/SNODAS_annual_monthly_figures/Average_Annual_{variable_name}_michigan_SNODAS.jpeg',dpi=300)
		plt.show()

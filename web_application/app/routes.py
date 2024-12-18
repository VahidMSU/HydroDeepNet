from flask import render_template, redirect, url_for, request, flash, jsonify, current_app, session
from flask_login import login_user, logout_user, login_required
from app.models import User, ContactMessage
from app.forms import RegistrationForm, LoginForm, ContactForm  # Import the forms
from app.extensions import db
import logging
from functools import partial
from multiprocessing import Process
from app.utils import find_station
from scipy.spatial import cKDTree
from SWATGenX.integrate_streamflow_data import integrate_streamflow_data
from app.utils import get_huc12_geometries, single_model_creation
from app.forms import ModelSettingsForm
from flask import render_template
import h5py
import numpy as np
import os
import json
import pandas as pd

def hydrogeo_dataset_dict(path=None):
	path = "/data/MyDataBase/HydroGeoDataset/HydroGeoDataset_ML_250.h5"
	with h5py.File(path,'r') as f:
		groups = f.keys()
		hydrogeo_dict = {}
		for group in groups:	
			hydrogeo_dict[group] = list(f[group].keys())
	return hydrogeo_dict

def CDL_lookup(code):
	path = "/data/SWATGenXApp/GenXAppData/CDL/CDL_CODES.csv"
	df = pd.read_csv(path)
	df = df[df['CODE'] == code]
	return df.NAME.values[0]    
def read_h5_file(address, lat=None, lon=None, lat_range=None, lon_range=None):
	path = "/data/MyDataBase/HydroGeoDataset/HydroGeoDataset_ML_250.h5"
	
	if lat is not None and lon is not None:
		print(f"requested lat and lon: {lat}, {lon}")
		lat_index, lon_index = get_rowcol_index_by_latlon(lat, lon)	
		print(f"lat_index, lon_index: {lat_index}, {lon_index}")
	else:
		lat_index = lon_index = None
	
	if lat_range is not None and lon_range is not None:
		print(f"requested lat_range and lon_range: {lat_range}, {lon_range}")
		min_lat, max_lat = lat_range
		min_lon, max_lon = lon_range
		min_row, max_row, min_col, max_col = get_rowcol_range_by_latlon(min_lat, max_lat, min_lon, max_lon)
		print(f"min_row, max_row, min_col, max_col: {min_row}, {max_row}, {min_col}, {max_col}")
		lat_range = (min_row, max_row)
		lon_range = (min_col, max_col)
		print(f"lat_range, lon_range: {lat_range}, {lon_range}")




	
	assert os.path.exists(path), f"File not found: {path}"
	assert os.access(path, os.R_OK), f"File not readable: {path}"
	print(f"Reading data from {path} at address: {address}")
	try:
		with h5py.File(path, 'r') as f:
			print(f"{path} opened successfully")
			if lat_index and lon_index:
				data = f[address][lat_index, lon_index]
				if "CDL" in address:
					data = CDL_lookup(data)
					dict_data = {"value": data}
				else:
					dict_data = {"value": data}

			elif lat_range and lon_range:
				data = f[address][lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]]
				data = np.where(data == -999, np.nan, data)  # Replace invalid values with NaN
				### get the median value of the data
				data_median = np.nanmedian(data)
				data_max = np.nanmax(data)
				data_min = np.nanmin(data)
				data_mean = np.nanmean(data)
				data_std = np.nanstd(data)
				if "CDL" in address:
					# Assuming 'data' contains the land-use codes from CDL
					unique, counts = np.unique(data, return_counts=True)

					# Each cell represents an area of 6.25 hectares
					cell_area_ha = 6.25

					# Calculate the total area for each land-use type
					dict_data = {
						CDL_lookup(key): value * cell_area_ha  # Map key to land-use description and calculate area
						for key, value in zip(unique, counts)
					}

					dict_data.update({"Total Area": np.nansum(list(dict_data.values()))})
					dict_data.update({"unit": "hectares"})
								
				else:
				
					dict_data = {
						"number of cells": data.size,
						"median": data_median.round(2),
						"max": data_max.round(2),
						"min": data_min.round(2),
						"mean": data_mean.round(2),
						"std": data_std.round(2)
					}
			else:
				data = f[address][:]
	except Exception as e:	
		print(f"Error reading data: {e}")
		return None
	print(f"Data read successfully: {data}")
	return dict_data	


def get_rowcol_range_by_latlon(desired_min_lat, desired_max_lat, desired_min_lon, desired_max_lon):
	path = "/data/MyDataBase/HydroGeoDataset/HydroGeoDataset_ML_250.h5"
	with h5py.File(path, 'r') as f:
		# Read latitude and longitude arrays
		lat_ = f["geospatial/lat_250m"][:]
		lon_ = f["geospatial/lon_250m"][:]
		
		# Replace missing values (-999) with NaN for better handling
		lat_ = np.where(lat_ == -999, np.nan, lat_)
		lon_ = np.where(lon_ == -999, np.nan, lon_)

		# Create masks for latitude and longitude ranges
		lat_mask = (lat_ >= desired_min_lat) & (lat_ <= desired_max_lat)
		lon_mask = (lon_ >= desired_min_lon) & (lon_ <= desired_max_lon)

		# Combine the masks to identify the valid rows and columns
		combined_mask = lat_mask & lon_mask

		# Check if any valid points are found
		if np.any(combined_mask):
			# Get row and column indices where the combined mask is True
			row_indices, col_indices = np.where(combined_mask)
		else:
			print("No valid points found for the given latitude and longitude range.")

		min_row_number = np.min(row_indices)
		max_row_number = np.max(row_indices)
		min_col_number = np.min(col_indices)
		max_col_number = np.max(col_indices)
		

		print(f"Min row number: {min_row_number}, Max row number: {max_row_number}, Min column number: {min_col_number}, Max column number: {max_col_number}")

		return min_row_number, max_row_number, min_col_number, max_col_number


def get_rowcol_index_by_latlon(desired_lat, desired_lon):

	path = f"/data/MyDataBase/HydroGeoDataset/HydroGeoDataset_ML_250.h5"
	
	with h5py.File(path, 'r') as f:
		lat_ = f["geospatial/lat_250m"][:]
		lat_ = np.where(lat_ == -999, np.nan, lat_)  # Replace invalid values with NaN
		lon_ = f["geospatial/lon_250m"][:]
		lon_ = np.where(lon_ == -999, np.nan, lon_)  # Replace invalid values with NaN

		valid_mask = ~np.isnan(lat_) & ~np.isnan(lon_)
		valid_lat = lat_[valid_mask]
		valid_lon = lon_[valid_mask]

		# Stack valid coordinates into KDTree
		coordinates = np.column_stack((valid_lat, valid_lon))
		tree = cKDTree(coordinates)

		# Query the closest point
		distance, idx = tree.query([desired_lat, desired_lon])

		# Retrieve the original indices of the closest point
		valid_indices = np.where(valid_mask)
		lat_idx = valid_indices[0][idx]
		lon_idx = valid_indices[1][idx]

		print(f"Closest row: {lat_idx}, Closest column: {lon_idx}")

		# Check the latitude and longitude values for the generated row and column
		lat_val = lat_[lat_idx, lon_idx]
		lon_val = lon_[lat_idx, lon_idx]

		print(f"Latitude: {lat_val}, Longitude: {lon_val}")

		return lat_idx, lon_idx





def init_routes(app):
	@app.route('/')
	#@login_required
	def index():
		print("Index route called")	
		logging.info("Index route called. Redirecting to /home.")
		return redirect(url_for('home'))
	@app.route('/dashboard')
	##@login_required
	def dashboard():
		return render_template('dashboard.html')
			
	@app.route('/get_options', methods=['GET'])
	##@login_required 
	def get_options():
		try:
			names_path = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/"
			variables = ['et', 'perc', 'precip', 'snofall', 'snomlt', 'surq_gen', 'wateryld']

			# List all NAMES
			if os.path.exists(names_path):
				names = os.listdir(names_path)
				if "log.txt" in names:
					names.remove("log.txt")
			else:
				names = []

			return jsonify({'names': names, 'variables': variables})
		except Exception as e:
			logging.error(f"Error fetching options: {e}")
			return jsonify({"error": "Failed to fetch options"}), 500

	@app.route('/visualizations', methods=['GET'])
	def visualizations():
		print("Visualizations route called")
		name = request.args.get('NAME', default=None)
		ver = request.args.get('ver', default=None)
		variable = request.args.get('variable', default=None)  # Multiple variables come as a comma-separated string

		print(f"Parameters received - NAME: {name}, Version: {ver}, Variable: {variable}")

		if not all([name, ver, variable]):
			# Check if it's an AJAX request
			if request.headers.get("X-Requested-With") == "XMLHttpRequest":
				return jsonify({"error": "Please provide NAME, Version, and Variable."}), 400
			else:
				return render_template('visualizations.html', error="Please provide NAME, Version, and Variable.")

		base_path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{name}/figures_SWAT_gwflow_MODEL"
		static_plots_path = os.path.join(base_path, "watershed_static_plots")
		video_path = os.path.join(base_path, "verifications_videos")

		# Process variables
		variables = variable.split(",")  # Split by commas for multiple variables
		gif_urls = []
		static_plot_files = []

		for var in variables:
			gif_file = os.path.join(video_path, f"{ver}_{var}_animation.gif")
			static_plot_dir = os.path.join(static_plots_path, var.capitalize())

			if os.path.exists(gif_file):
				gif_urls.append(f"/static/SWATplus_by_VPUID/0000/huc12/{name}/figures_SWAT_gwflow_MODEL/verifications_videos/{ver}_{var}_animation.gif")

			if os.path.exists(static_plot_dir):
				static_plot_files += [
					f"/static/SWATplus_by_VPUID/0000/huc12/{name}/figures_SWAT_gwflow_MODEL/watershed_static_plots/{var.capitalize()}/{file}"
					for file in os.listdir(static_plot_dir) if file.endswith('.png')
				]

		if not gif_urls and not static_plot_files:
			if request.headers.get("X-Requested-With") == "XMLHttpRequest":
				return jsonify({"error": f"No visualizations found for NAME: {name}, Version: {ver}, Variables: {variables}."}), 404
			else:
				return render_template('visualizations.html', error=f"No visualizations found for NAME: {name}, Version: {ver}, Variables: {variables}.")

		print(f"Static plot files: {static_plot_files}")
		print(f"GIF URLs: {gif_urls}")

		# For AJAX requests, return JSON
		if request.headers.get("X-Requested-With") == "XMLHttpRequest":
			return jsonify({
				"gif_files": gif_urls,
				"png_files": static_plot_files
			})

		# For direct access via browser, render the template
		return render_template(
			'visualizations.html',
			name=name,
			ver=ver,
			variables=variables,
			gif_files=gif_urls,
			png_files=static_plot_files
		)


	@app.route('/login', methods=['GET', 'POST'])
	def login():
		print("Login route called")
		logging.info("Login route called")
		form = LoginForm()

		if form.validate_on_submit():
			username = form.username.data
			password = form.password.data
			print(f"Attempting login for: {username}")

			user = User.query.filter_by(username=username).first()
			if user and user.password == password:  # Replace with hashed password checking in production
				login_user(user)
				session.permanent = True
				print("Login successful")
				return redirect(url_for('home'))
			else:
				print("Invalid username or password")
				flash('Invalid username or password', 'danger')

		return render_template('login.html', form=form)


	@app.route('/signup', methods=['GET', 'POST'])
	def signup():
		print("Sign Up route called")
		logging.info("Sign Up route called")
		form = RegistrationForm()
		if form.validate_on_submit():
			logging.info("Form validated successfully")
			user = User(username=form.username.data, email=form.email.data, password=form.password.data)
			try:
				db.session.add(user)
				db.session.commit()
				logging.info("User added to the database successfully")
				flash('Account created successfully! You can now log in.')
				return redirect(url_for('login'))
			except Exception as e:
				logging.error(f"Error adding user to the database: {e}")
				db.session.rollback()
				flash('An error occurred while creating the account. Please try again.')
		else:
			logging.info("Form validation failed")
			for field, errors in form.errors.items():
				for error in errors:
					logging.error(f"Error in {field}: {error}")
		return render_template('register.html', form=form)

	@app.route('/logout')
	##@login_required
	def logout():
		print("Logout route called")
		logging.info("Logout route called")
		logout_user()
		session.clear()  # Clear the session
		return redirect(url_for('login'))

	@app.route('/home')
	##@login_required
	def home():
		logging.info("Home route called, user is authenticated.")
		return render_template('home.html')

	@app.route('/model-settings', methods=['GET', 'POST'])
	##@login_required
	def model_settings():
		print("Model Settings route called")
		logging.info("Model Settings route called")
		form = ModelSettingsForm()
		output = None
		if form.validate_on_submit():
			site_no = form.user_input.data
			try:
				ls_resolution = min(int(form.ls_resolution.data), 500)
				dem_resolution = min(int(form.dem_resolution.data), 250)
				calibration_flag = form.calibration_flag.data
				validation_flag = form.validation_flag.data
				sensitivity_flag = form.sensitivity_flag.data
				cal_pool_size = min(int(form.cal_pool_size.data), 100)
				sen_pool_size = min(int(form.sen_pool_size.data), 500)
				sen_total_evaluations = min(int(form.sen_total_evaluations.data), 5000)
				num_levels = min(int(form.num_levels.data), 20)
				max_cal_iterations = min(int(form.max_iterations.data), 50)
				verification_samples = min(int(form.verification_samples.data), 50)
			except ValueError:
				logging.error("Invalid input received for model settings")
			
			logging.info(f"Model settings received: {site_no}, {ls_resolution}, {dem_resolution}, {calibration_flag}, {validation_flag}, {sensitivity_flag}, {cal_pool_size}, {sen_pool_size}, {sen_total_evaluations}, {num_levels}, {max_cal_iterations}, {verification_samples}")
			wrapped_single_model_creation = partial(single_model_creation, site_no, ls_resolution, dem_resolution, calibration_flag, validation_flag, sensitivity_flag, cal_pool_size, sen_pool_size, sen_total_evaluations, num_levels, max_cal_iterations, verification_samples)
			process = Process(target=wrapped_single_model_creation)
			process.start()
			
			# Redirect to confirmation page after form submission
			return redirect(url_for('model_confirmation'))

		station_data = integrate_streamflow_data(current_app.config['USGS_PATH'])
		station_list = station_data.SiteNumber.unique()
		return render_template('model_settings.html', form=form, output=output, station_list=station_list)

	@app.route('/get_station_characteristics', methods=['GET'])
	##@login_required
	def get_station_characteristics():
		logging.info("Get Station Characteristics route called")
		station_no = request.args.get('station')
		station_data = integrate_streamflow_data(current_app.config['USGS_PATH'])
		logging.info("Calling get_station_characteristics")

		station_row = station_data[station_data.SiteNumber == station_no]
		characteristics_list = station_row.to_dict(orient='records')
		characteristics = characteristics_list[0] if characteristics_list else {}
		#pring the characteristics
		logging.info(f"Station {station_no} found")
		logging.info(f"characteristics: {characteristics}")
		
		if characteristics:
			huc12_list = characteristics.get('HUC12 ids of the watershed', [])
			huc12_list = [str(x.split("'")[1]) for x in huc12_list[1:-1].split(',')]

			geometries = get_huc12_geometries(huc12_list)
			characteristics['geometries'] = geometries
			#logging.info(f"Station {station_no} found: {characteristics}")
			return jsonify(characteristics)
		else:
			return jsonify({"error": "Station not found"}), 404
	@app.route('/about')
	##@login_required
	def about():
		logging.info("About route called")
		return render_template('about.html')


	@app.route('/model-confirmation')
	##@login_required
	def model_confirmation():
		logging.info("Model Confirmation route called")
		return render_template('model_confirmation.html')


	@app.route('/contact', methods=['GET', 'POST'])
	##@login_required
	def contact():
		logging.info("Contact route called")
		form = ContactForm()
		if form.validate_on_submit():
			name = form.name.data
			email = form.email.data
			message = form.message.data
			contact_message = ContactMessage(name=name, email=email, message=message)
			try:
				db.session.add(contact_message)
				db.session.commit()
				logging.info(f"Message from {name} added to the database")
				flash('Your message has been sent successfully!')
			except Exception as e:
				logging.error(f"Error adding message to the database: {e}")
				db.session.rollback()
				flash('An error occurred while sending the message. Please try again.')
			return redirect(url_for('contact'))
		return render_template('contact.html', form=form)

	@app.route('/infrastructure')
	##@login_required
	def infrastructure():
		logging.info("Infrastructure route called")
		return render_template('infrastructure.html')
	
	from app.forms import HydroGeoDatasetForm
	@app.route('/hydro_geo_dataset', methods=['GET', 'POST'])
	def hydro_geo_dataset():
		form = HydroGeoDatasetForm()
		hydrodict = hydrogeo_dataset_dict()

		# Populate variable choices
		available_groups = [
			'CDL', 'EBK', 'LANDFIRE', 'MODIS', 'NHDPlus', 'PRISM', 'SNODAS_monthly',
			'Wellogic', 'climate_pattern', 'geospatial', 'gssurgo', 'population'
		]
		form.variable.choices = [(group, group) for group in available_groups]

		# Repopulate subvariable choices based on the selected variable
		if request.method == 'POST':
			selected_variable = request.form.get('variable')
			if selected_variable in hydrodict:
				form.subvariable.choices = [(item, item) for item in hydrodict[selected_variable]]

			print("Form validated successfully")
			variable = request.form.get('variable')
			subvariable = request.form.get('subvariable')

			# Handle single point and range inputs
			latitude = request.form.get('latitude', None)
			longitude = request.form.get('longitude', None)
			min_latitude = request.form.get('min_latitude', None)
			max_latitude = request.form.get('max_latitude', None)
			min_longitude = request.form.get('min_longitude', None)
			max_longitude = request.form.get('max_longitude', None)

			# Handle single point inputs
			if min_latitude and max_latitude and min_longitude and max_longitude:
				latitude = longitude = None  # Reset single point coordinates

			# Handle polygon inputs
			polygon_coordinates = request.form.get('polygon_coordinates')
			if polygon_coordinates:
				try:
					# Parse polygon JSON
					vertices = json.loads(polygon_coordinates)
					print("Received polygon vertices:", vertices)
					# Calculate polygon bounds
					max_latitudes = [vertex['latitude'] for vertex in vertices]
					max_longitudes = [vertex['longitude'] for vertex in vertices]
					min_latitude = min(max_latitudes)
					max_latitude = max(max_latitudes)
					min_longitude = min(max_longitudes)
					max_longitude = max(max_longitudes)

					print(f"Polygon bounds: ({min_latitude}, {max_latitude}), ({min_longitude}, {max_longitude})")
				except Exception as e:

					print(f"Error parsing polygon coordinates: {e}")
					flash("Invalid polygon coordinates.", "danger")
					return render_template('HydroGeoDataset.html', form=form)
			
			elif not any([latitude, longitude, min_latitude, max_latitude, min_longitude, max_longitude]):
				flash("Please provide either a point or a range for data retrieval.", "danger")
				return render_template('HydroGeoDataset.html', form=form)
			

			if not variable or not subvariable:
				flash("Variable and Subvariable are required.", "danger")
				return render_template('HydroGeoDataset.html', form=form)

			try:
				if latitude and longitude:
					# Fetch data for single point
					print(f"Fetching data for {variable}/{subvariable} at {latitude}, {longitude}")
					data = read_h5_file(
						lat=float(latitude), lon=float(longitude), address=f"{variable}/{subvariable}"
					)
					print(f"Data fetched: {data}")
				elif all([min_latitude, max_latitude, min_longitude, max_longitude]):
					# Fetch data for range
					print(f"Fetching data for {variable}/{subvariable} in range ({min_latitude}, {max_latitude}), ({min_longitude}, {max_longitude})")
					data = read_h5_file(
						lat_range=(float(min_latitude), float(max_latitude)),
						lon_range=(float(min_longitude), float(max_longitude)),
						address=f"{variable}/{subvariable}"
					)
					print(f"Data fetched for range: {data}")
				else:
					flash("Please provide either a point or a range for data retrieval.", "danger")
					return render_template('HydroGeoDataset.html', form=form)

				return render_template(
					'HydroGeoDataset.html',
					form=form,
					variable=variable,
					subvariable=subvariable,
					data=data
				)
			except Exception as e:
				print(f"Error fetching data: {e}")
				flash(f"Error fetching data: {e}", "danger")
		return render_template('HydroGeoDataset.html', form=form)


	@app.route('/get_subvariables', methods=['POST'])
	def get_subvariables():
		print("Get Subvariables route called")
		variable = request.form.get('variable')
		if not variable:
			return jsonify({"error": "Variable is required"}), 400

		# Fetch subvariables for the selected variable
		hydrodict = hydrogeo_dataset_dict()
		subvariables = hydrodict.get(variable, [])
		print(f"Subvariables for {variable}: {subvariables}")

		return jsonify({"subvariables": subvariables})

	@app.route('/ftp-access')
	@login_required
	def ftp_access():
		logging.info("FTP Access route called")
		#return render_template('ftp_access.html')
		return render_template('redirect.html')

	@app.route('/deeplearning_models')
	@login_required
	def deeplearning_models():
		#return render_template('DeepLearning.html')
		return render_template('redirect.html')
	@app.route('/michigan')
	##@login_required
	def michigan():
		logging.info("Michigan route called")
		return render_template('michigan.html')


	@app.route('/search_site', methods=['GET', 'POST'])
	##@login_required
	def search_site():
		logging.info("Search site route called")
		search_term = request.args.get('search_term', '').lower()
		
		if not search_term:
			return jsonify({"error": "Search term is required"}), 400

		try:
			# Paths for the data files
			# Call the utility function to search for station regions
			results = find_station(search_term)

			if results.empty:
				return jsonify({"error": "No matching sites found"}), 404

			return jsonify(results.to_dict(orient='records'))

		except Exception as e:
			logging.error(f"Error while searching for site: {e}")
			return jsonify({"error": "An error occurred during the search"}), 500

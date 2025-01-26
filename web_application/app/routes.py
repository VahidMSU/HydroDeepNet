from flask import render_template, redirect, url_for, request, flash, jsonify, current_app, session
from flask_login import login_user, logout_user, login_required
from app.models import User, ContactMessage
from app.forms import RegistrationForm, LoginForm, ContactForm, ModelSettingsForm, HydroGeoDatasetForm
from app.extensions import db
from functools import partial
from multiprocessing import Process
from app.utils import find_station, get_huc12_geometries, get_huc12_streams_geometries
from SWATGenX.integrate_streamflow_data import integrate_streamflow_data
import os
import json
from app.utils import LoggerSetup, single_model_creation, hydrogeo_dataset_dict, read_h5_file
import numpy as np
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
import pandas as pd
class AppManager:
	def __init__(self, app):
		self.app = app
		self.init_routes()
		self.logger = LoggerSetup(report_path="/data/SWATGenXApp/codes/web_application/logs", verbose=True, rewrite=True)
		self.logger = self.logger.setup_logger("WebAppLogger")

	def init_routes(self):
		@self.app.route('/')
		def index():
			self.logger.info("Index route called. Redirecting to /home.")	
			return redirect(url_for('home'))

		@self.app.route('/dashboard')
		def dashboard():
			return render_template('dashboard.html')

		@self.app.route('/get_options', methods=['GET'])
		def get_options():
			try:
				names_path = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/"
				variables = ['et', 'perc', 'precip', 'snofall', 'snomlt', 'surq_gen', 'wateryld']

				if os.path.exists(names_path):
					names = os.listdir(names_path)
					if "log.txt" in names:
						names.remove("log.txt")
				else:
					names = []
					
				return jsonify({'names': names, 'variables': variables})
			except Exception as e:
				self.logger.error(f"Error fetching options: {e}")
				return jsonify({"error": "Failed to fetch options"}), 500

		@self.app.route('/visualizations', methods=['GET'])
		def visualizations():
			self.logger.info("Visualizations route called")
			name = request.args.get('NAME', default=None)
			ver = request.args.get('ver', default=None)
			variable = request.args.get('variable', default=None)

			if not all([name, ver, variable]):
				if request.headers.get("X-Requested-With") == "XMLHttpRequest":
					return jsonify({"error": "Please provide NAME, Version, and Variable."}), 400
				else:
					return render_template('visualizations.html', error="Please provide NAME, Version, and Variable.")

			base_path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{name}/figures_SWAT_gwflow_MODEL"
			static_plots_path = os.path.join(base_path, "watershed_static_plots")
			video_path = os.path.join(base_path, "verifications_videos")

			variables = variable.split(",")
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

			if request.headers.get("X-Requested-With") == "XMLHttpRequest":
				return jsonify({"gif_files": gif_urls, "png_files": static_plot_files})

			return render_template('visualizations.html', name=name, ver=ver, variables=variables, gif_files=gif_urls, png_files=static_plot_files)

		@self.app.route('/login', methods=['GET', 'POST'])
		def login():
			self.logger.info("Login route called")	
			form = LoginForm()

			if form.validate_on_submit():
				username = form.username.data
				password = form.password.data
				self.logger.info(f"Attempting login for: {username}")	

				user = User.query.filter_by(username=username).first()
				if user and user.password == password:
					login_user(user)
					session.permanent = True
					self.logger.info(f"Login successful for: {username}")
					return redirect(url_for('home'))
				else:
					self.logger.error("Invalid username or password")
					flash('Invalid username or password', 'danger')

			return render_template('login.html', form=form)

		@self.app.route('/signup', methods=['GET', 'POST'])
		def signup():
			self.logger.info("Sign Up route called")	
			form = RegistrationForm()
			if form.validate_on_submit():
				self.logger.info("Form validated successfully")	
				user = User(username=form.username.data, email=form.email.data, password=form.password.data)
				try:
					db.session.add(user)
					db.session.commit()
					self.logger.info("User added to the database successfully")
					flash('Account created successfully! You can now log in.')
					return redirect(url_for('login'))
				except Exception as e:
					self.logger.error(f"Error adding user to the database: {e}")
					db.session.rollback()
					flash('An error occurred while creating the account. Please try again.')
			else:
				self.logger.error("Form validation failed")
				for field, errors in form.errors.items():
					for error in errors:
						self.logger.error(f"Error in {field}: {error}")
			return render_template('register.html', form=form)

		@self.app.route('/logout')
		def logout():
			self.logger.info("Logout route called")
			logout_user()
			session.clear()
			return redirect(url_for('login'))

		@self.app.route('/home')
		def home():
			self.logger.info("Home route called, user is authenticated.")	
			return render_template('home.html')

		@self.app.route('/model-settings', methods=['GET', 'POST'])
		def model_settings():
			self.logger.info("Model Settings route called")	
			form = ModelSettingsForm()
			output = None
			if form.validate_on_submit():
				site_no = form.user_input.data
				try:
					self.logger.info("Form validated successfully")
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
					self.logger.error("Invalid input received for model settings")
				
				self.logger.info(f"Model settings received: {site_no}, {ls_resolution}, {dem_resolution}, {calibration_flag}, {validation_flag}, {sensitivity_flag}, {cal_pool_size}, {sen_pool_size}, {sen_total_evaluations}, {num_levels}, {max_cal_iterations}, {verification_samples}")	
				wrapped_single_model_creation = partial(single_model_creation, site_no, ls_resolution, dem_resolution, calibration_flag, validation_flag, sensitivity_flag, cal_pool_size, sen_pool_size, sen_total_evaluations, num_levels, max_cal_iterations, verification_samples)
				process = Process(target=wrapped_single_model_creation)
				process.start()
				self.logger.info("Model creation process started")

				return redirect(url_for('model_confirmation'))
			
			#station_data = integrate_streamflow_data()
			station_data = pd.read_csv(SWATGenXPaths.FPS_all_stations, dtype={'SiteNumber': str})
			station_list = station_data.SiteNumber.unique()
			return render_template('model_settings.html', form=form, output=output, station_list=station_list)

		@self.app.route('/get_station_characteristics', methods=['GET'])
		def get_station_characteristics():
			self.logger.info("Get Station Characteristics route called")
			station_no = request.args.get('station')
			#station_data = integrate_streamflow_data()
			station_data = pd.read_csv(SWATGenXPaths.FPS_all_stations, dtype={'SiteNumber': str})
			self.logger.info(f"Station number: {station_no}")
			
			station_row = station_data[station_data.SiteNumber == station_no]
			characteristics_list = station_row.to_dict(orient='records')
			characteristics = characteristics_list[0] if characteristics_list else {}
			self.logger.info(f"Station {station_no} found")
			self.logger.info(f"characteristics: {characteristics}")

			if characteristics:
				huc12_list = characteristics.get('HUC12 ids of the watershed', [])
				huc12_list = [str(x.split("'")[1]) for x in huc12_list[1:-1].split(',')]
				geometries = get_huc12_geometries(huc12_list)
				streams_geometries = get_huc12_streams_geometries(huc12_list)
				### check if the geometries are empty
				if not geometries:
					self.logger.error(f"No geometries found for HUC12s: {huc12_list}")
				if not streams_geometries:
					self.logger.error(f"No streams geometries found for HUC12s: {huc12_list}")
					
				characteristics['geometries'] = geometries
				characteristics['streams_geometries'] = streams_geometries
				return jsonify(characteristics)
			else:
				return jsonify({"error": "Station not found"}), 404

		@self.app.route('/about')
		def about():
			self.logger.info("About route called")
			return render_template('about.html')

		@self.app.route('/model-confirmation')
		def model_confirmation():
			self.logger.info("Model Confirmation route called")
			return render_template('model_confirmation.html')

		@self.app.route('/contact', methods=['GET', 'POST'])
		def contact():
			self.logger.info("Contact route called")
			form = ContactForm()
			if form.validate_on_submit():
				name = form.name.data
				email = form.email.data
				message = form.message.data
				contact_message = ContactMessage(name=name, email=email, message=message)
				try:
					db.session.add(contact_message)
					db.session.commit()
					self.logger.info(f"Message from {name} added to the database")	
					flash('Your message has been sent successfully!')
				except Exception as e:
					self.logger.error(f"Error adding message to the database: {e}")	
					db.session.rollback()
					flash('An error occurred while sending the message. Please try again.')
				return redirect(url_for('contact'))
			return render_template('contact.html', form=form)

		@self.app.route('/infrastructure')
		def infrastructure():
			self.logger.info("Infrastructure route called")
			return render_template('infrastructure.html')

		@self.app.route('/hydro_geo_dataset', methods=['GET', 'POST'])
		def hydro_geo_dataset():
			self.logger.info("HydroGeoDataset route called")
			form = HydroGeoDatasetForm()
			hydrodict = hydrogeo_dataset_dict()

			available_groups = [
				'CDL', 'EBK', 'LANDFIRE', 'MODIS', 'NHDPlus', 'PRISM', 'SNODAS_monthly',
				'Wellogic', 'climate_pattern', 'geospatial', 'gssurgo', 'population'
			]
			form.variable.choices = [(group, group) for group in available_groups]
			self.logger.info(f"Available groups: {available_groups}")	
			if request.method == 'POST':
				self.logger.info("Form submitted")
				selected_variable = request.form.get('variable')
				if selected_variable in hydrodict:
					form.subvariable.choices = [(item, item) for item in hydrodict[selected_variable]]

				self.logger.info("Form validated successfully")	
				variable = request.form.get('variable')
				subvariable = request.form.get('subvariable')

				latitude = request.form.get('latitude', None)
				longitude = request.form.get('longitude', None)
				min_latitude = request.form.get('min_latitude', None)
				max_latitude = request.form.get('max_latitude', None)
				min_longitude = request.form.get('min_longitude', None)
				max_longitude = request.form.get('max_longitude', None)

				if min_latitude and max_latitude and min_longitude and max_longitude:
					self.logger.info(f"Received range: ({min_latitude}, {max_latitude}), ({min_longitude}, {max_longitude})")
					latitude = longitude = None

				polygon_coordinates = request.form.get('polygon_coordinates')
				if polygon_coordinates:
					try:
						vertices = json.loads(polygon_coordinates)
						self.logger.info(f"Received polygon vertices: {vertices}")
						max_latitudes = [vertex['latitude'] for vertex in vertices]
						max_longitudes = [vertex['longitude'] for vertex in vertices]
						min_latitude = min(max_latitudes)
						max_latitude = max(max_latitudes)
						min_longitude = min(max_longitudes)
						max_longitude = max(max_longitudes)
						self.logger.info(f"Polygon bounds: ({min_latitude}, {max_latitude}), ({min_longitude}, {max_longitude})")	
					except Exception as e:
						self.logger.error(f"Error parsing polygon coordinates: {e}")
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
						self.logger.info(f"Fetching data for {variable}/{subvariable} at {latitude}, {longitude}")
						raw_data = read_h5_file(
							lat=float(latitude), lon=float(longitude), address=f"{variable}/{subvariable}"
						)
						# Convert np.float32 to float
						data = {key: float(value) if isinstance(value, np.float32) else value for key, value in raw_data.items()}
						self.logger.info(f"Data fetched: {data}")
					elif all([min_latitude, max_latitude, min_longitude, max_longitude]):
						self.logger.info(f"Fetching data for {variable}/{subvariable} in range ({min_latitude}, {max_latitude}), ({min_longitude}, {max_longitude})")
						raw_data = read_h5_file(
							lat_range=(float(min_latitude), float(max_latitude)),
							lon_range=(float(min_longitude), float(max_longitude)),
							address=f"{variable}/{subvariable}"
						)
						# Convert np.float32 to float
						data = {key: float(value) if isinstance(value, np.float32) else value for key, value in raw_data.items()}
						self.logger.info(f"Data fetched for range: {data}")
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
					self.logger.error(f"Error fetching data: {e}")	
					flash(f"Error fetching data: {e}", "danger")
			return render_template('HydroGeoDataset.html', form=form)

		@self.app.route('/get_subvariables', methods=['POST'])
		def get_subvariables():
			self.logger.info("Get Subvariables route called")
			variable = request.form.get('variable')
			self.logger.info(f"Received variable: {variable}")

			if not variable:
				return jsonify({"error": "Variable is required"}), 400

			hydrodict = hydrogeo_dataset_dict()
			subvariables = hydrodict.get(variable, [])
			self.logger.info(f"Subvariables for {variable}: {subvariables}")

			return jsonify({"subvariables": subvariables})

		@self.app.route('/ftp-access')
		#@login_required
		def ftp_access():
			self.logger.info("FTP Access route called")
			#return render_template('redirect.html')
			return render_template("ftp_access.html")

		@self.app.route('/deeplearning_models')
		@login_required
		def deeplearning_models():
			return render_template('DeepLearning.html')
		
		@self.app.route('/vision_system')
		#@login_required
		def vision_system():
			return render_template('VisionSystem.html')

		@self.app.route('/michigan')
		def michigan():
			self.logger.info("Michigan route called")
			return render_template('michigan.html')

		@self.app.route('/search_site', methods=['GET', 'POST'])
		def search_site():
			self.logger.info("Search site route called")	
			search_term = request.args.get('search_term', '').lower()
			if not search_term:
				return jsonify({"error": "Search term is required"}), 400
			
			try:
				results = find_station(search_term)
				if results.empty:
					self.logger.info("No matching sites found")
					return jsonify({"error": "No matching sites found"}), 404
				return jsonify(results.to_dict(orient='records'))
			except Exception as e:
				self.logger.error(f"Error searching for site: {e}")
				return jsonify({"error": "An error occurred during the search"}), 500
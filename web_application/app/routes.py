## /data/SWATGenXApp/codes/web_application/app/routes.py

from flask import (url_for, request,
				jsonify, current_app, session,send_file,
				send_from_directory, redirect)
from app.sftp_manager import create_sftp_user  # Import the SFTP user creation function

from flask_login import (login_user, logout_user,
						login_required, current_user)
from app.extensions import csrf
from app.models import User, ContactMessage

from app.forms import (RegistrationForm, LoginForm, 
						ContactForm, ModelSettingsForm, 
						HydroGeoDatasetForm)

from app.extensions import db
from functools import partial, wraps
from multiprocessing import Process
from app.utils import (find_station, get_huc12_geometries, get_huc12_streams_geometries,
						get_huc12_lakes_geometries, send_verification_email, 
						single_model_creation, hydrogeo_dataset_dict, read_h5_file) 
import os
import json
import numpy as np
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
import pandas as pd
from app.forms import VerificationForm
import requests
from werkzeug.utils import secure_filename
import shutil
import tempfile

def verified_required(f):
	@wraps(f)
	def decorated_function(*args, **kwargs):
		if current_user.is_authenticated and not current_user.is_verified:
			#flash("You must verify your email first!")
			message = "You must verify your email first!"
			#return redirect(url_for('verify'))
			return jsonify({"status": "error", "message": message}), 403
		return f(*args, **kwargs)
	return decorated_function


def check_existing_models(station_name):
	swatgenx_output = SWATGenXPaths.swatgenx_outlet_path
	VPUIDs = os.listdir(swatgenx_output)
	existing_models = []
	for VPUID in VPUIDs:
		# now find model inside huc12 directory
		huc12_path = os.path.join(swatgenx_output, VPUID, "huc12")
		models = os.listdir(huc12_path)
		existing_models.extend(os.path.join(huc12_path, model) for model in models)
	existance_flag = False
	for model in existing_models:
		if station_name in model:
			print(f"Model found for station {station_name} at {model}")
			existance_flag = True
			break
	return existance_flag

class AppManager:
	def __init__(self, app):
		self.app = app
		self.init_routes()
		self.app.logger.info("AppManager initialized!")

	def init_routes(self):
		@self.app.route('/')
		@login_required
		@verified_required
		def index():
			self.app.logger.info("Index route called. Redirecting to /home.")	
			return jsonify({"status": "success", "redirect": "/home"})

		@self.app.route('/verify', methods=['GET', 'POST'])
		@login_required
		def verify():
			"""Email verification route."""
			self.app.logger.info(f"Verification requested by user: {current_user.username}.")
			form = VerificationForm()
			if form.validate_on_submit():
				code_entered = form.verification_code.data.strip()
				if current_user.verification_code == code_entered:
					current_user.is_verified = True
					current_user.verification_code = None
					db.session.commit()
					self.app.logger.info(f"User `{current_user.username}` verified successfully.")
					return jsonify({"status": "success", "redirect": "/home"})
				self.app.logger.warning(f"Invalid verification code entered by `{current_user.username}`.")
				return jsonify({"status": "error", "message": "Invalid verification code."}), 400
			return jsonify({"title": "Verify", "message": "Enter your verification code."})


		@self.app.route('/flask-static/<path:filename>')
		def serve_flask_static(filename):
			return send_from_directory('/data/SWATGenXApp/GenXAppData/', filename)


		@self.app.route('/privacy')
		def privacy():
			self.app.logger.info("Privacy route called")
			#return render_template('privacy.html')
			return jsonify({"title": "Privacy", "message": "privacy page"})
		
		@self.app.route('/terms')
		def terms():
			self.app.logger.info("Terms route called")
			#return render_template('terms.html')
			return jsonify({"title": "Terms", "message": "terms page"})

		@self.app.route('/js/<path:filename>')
		def js_static(filename):
			js_dir = os.path.join(current_app.root_path,  'js')
			return send_from_directory(js_dir, filename)

		@self.app.route('/css/<path:filename>')
		def css_static(filename):
			css_dir = os.path.join(current_app.root_path, 'css')
			return send_from_directory(css_dir, filename)
	

		@self.app.route('/user_dashboard', methods=['GET'])
		@login_required
		@verified_required
		def user_dashboard():
			"""User dashboard route."""
			self.app.logger.info(f"User Dashboard accessed by `{current_user.username}`.")
			return jsonify({"title": "Dashboard", "message": "Your user dashboard."})

		@self.app.route('/get_options', methods=['GET'])
		@login_required
		@verified_required
		def get_options():
			try:
				names_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/"
				variables = ['et', 'perc', 'precip', 'snofall', 'snomlt', 'surq_gen', 'wateryld']

				if os.path.exists(names_path):
					names = os.listdir(names_path)
					if "log.txt" in names:
						names.remove("log.txt")
				else:
					names = []
				return jsonify({'names': names, 'variables': variables})
			except Exception as e:
				self.app.logger.error(f"Error fetching options: {e}")
				return jsonify({"error": "Failed to fetch options"}), 500

		@self.app.route('/visualizations', methods=['GET'])
		@login_required
		@verified_required
		def visualizations():
			self.app.logger.info("Visualizations route called")
			name = request.args.get('NAME', default=None)
			ver = request.args.get('ver', default=None)
			variable = request.args.get('variable', default=None)

			if not all([name, ver, variable]):
				if request.headers.get("X-Requested-With") == "XMLHttpRequest":
					return jsonify({"error": "Please provide NAME, Version, and Variable."}), 400
				else:
					return jsonify({"title": "Visualizations", "message": "Please provide NAME, Version, and Variable."})

			base_path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{name}/figures_SWAT_gwflow_MODEL"
			video_path = os.path.join(base_path, "verifications_videos")

			variables = variable.split(",")
			gif_urls = []

			for var in variables:
				gif_file = os.path.join(video_path, f"{ver}_{var}_animation.gif")
				if os.path.exists(gif_file):
					gif_urls.append(f"/static/SWATplus_by_VPUID/0000/huc12/{name}/figures_SWAT_gwflow_MODEL/verifications_videos/{ver}_{var}_animation.gif")

			if not gif_urls:
				if request.headers.get("X-Requested-With") == "XMLHttpRequest":
					return jsonify({"error": f"No visualizations found for NAME: {name}, Version: {ver}, Variables: {variables}."}), 404
				else:
					return jsonify({"title": "Visualizations", "message": f"No visualizations found for NAME: {name}, Version: {ver}, Variables: {variables}."})

			if request.headers.get("X-Requested-With") == "XMLHttpRequest":
				return jsonify({"gif_files": gif_urls})

			return jsonify({"title": "Visualizations", "message": "Visualizations page", "gif_files": gif_urls})

		@self.app.route('/oauth_callback')
		def oauth_callback():
			code = request.args.get('code')
			if not code:
				#flash('Authorization failed. Please try again.', 'danger')
				#return redirect(url_for('login'))
				message = "Authorization failed. Please try again."
				return jsonify({"status": "error", "message": message, "redirect": "/login"})
			
			# Exchange the code for an access token
			token_url = "https://oauth.msu.edu/token"
			payload = {
				"code": code,
				"client_id": "dummy",# CLIENT_ID,
				"client_secret": "dummy",#CLIENT_SECRET,
				"redirect_uri": url_for('oauth_callback', _external=True),
				"grant_type": "authorization_code"
			}
			response = requests.post(token_url, data=payload)
			if response.status_code != 200:
				#flash('Failed to authenticate with MSU. Please try again.', 'danger')
				#return redirect(url_for('login'))
				message = "Failed to authenticate with MSU. Please try again."
				return jsonify({"status": "error", "message": message, "redirect": "/login"})

			token = response.json().get("access_token")

			# Retrieve user info
			user_info_url = "https://oauth.msu.edu/userinfo"
			headers = {"Authorization": f"Bearer {token}"}
			user_info_response = requests.get(user_info_url, headers=headers)
			if user_info_response.status_code != 200:
				#flash('Failed to retrieve user information.', 'danger')
				#return redirect(url_for('login'))
				message = "Failed to retrieve user information."
				return jsonify({"status": "error", "message": message, "redirect": "/login"})

			user_info = user_info_response.json()
			msu_netid = user_info.get('netid')
			email = user_info.get('email')

			# Check if the user already exists
			user = User.query.filter_by(username=msu_netid).first()
			if not user:
				# Create a new user
				user = User(username=msu_netid, email=email, password="")  # No password needed for MSU login
				db.session.add(user)
				db.session.commit()

			# Log the user in
			login_user(user)
			self.app.logger.info(f"MSU login successful for: {msu_netid}")
			#return redirect(url_for('home'))
			return jsonify({"status": "success", "redirect": "/home"})
		
		
		@csrf.exempt
		@self.app.route('/api/login', methods=['POST'])
		def api_login():
			self.app.logger.info(f"{request.method} request received for /api/login")
			data = request.json
			if not data:
				return jsonify({"error": "No data received"}), 400

			username = data.get('username')
			password = data.get('password')
			if not all([username, password]):
				return jsonify({"error": "Missing username or password"}), 400

			user = User.query.filter_by(username=username).first()
			if user and user.check_password(password):
				login_user(user)
				session.permanent = True
				return jsonify({"success": True, "token": "someJWT"}), 200

			else:
				return jsonify({"error": "Invalid username or password"}), 401
			

				
		@self.app.route('/signup', methods=['GET', 'POST'])
		def signup():
			self.app.logger.info("Sign Up route called")
			form = RegistrationForm()

			if form.validate_on_submit():
				verification_code = send_verification_email(form.email.data)
				self.app.logger.info("Form validated successfully")

				new_user = User(
					username=form.username.data,
					email=form.email.data,
					password=form.password.data,
					verification_code=verification_code
				)

				try:
					db.session.add(new_user)
					db.session.commit()
					self.app.logger.info("User added to DB but not verified yet.")

					# ✅ Automatically create SFTP account
					sftp_result = create_sftp_user(new_user.username)
					if sftp_result.get("status") != "success":
						self.app.logger.error(f"Failed to create SFTP for {new_user.username}: {sftp_result.get('error')}")
						return jsonify({"status": "error", "message": "SFTP account creation failed. Contact support."})

					# ✅ Log user in and redirect
					login_user(new_user)
					return jsonify({"status": "success", "message": "Check your email for verification code.", "redirect": "/verify"})

				except Exception as e:
					self.app.logger.error(f"Error adding user: {e}")
					db.session.rollback()
					return jsonify({"status": "error", "message": "An error occurred while creating the account."})

			# ✅ If form is invalid, return errors
			self.app.logger.error("Form validation failed")
			return jsonify({
				"title": "Sign Up",
				"message": "Registration failed.",
				"errors": form.errors
			}), 400


		@self.app.route('/home', methods=['GET'])
		@login_required
		@verified_required
		def home():
			"""User's home page."""
			self.app.logger.info(f"Home route accessed by user: {current_user.username}.")
			return jsonify({"title": "Home", "message": "Welcome to the app!"})

		@self.app.route('/model-settings', methods=['POST'])
		@login_required
		@verified_required
		def model_settings():
			"""Model settings submission route."""
			data = request.json
			if not data:
				return jsonify({"error": "No data received"}), 400

			# Extract form data
			site_no = data.get("site_no")
			ls_resolution = data.get("ls_resolution", 500)
			dem_resolution = data.get("dem_resolution", 250)

			self.app.logger.info(
				f"Model settings received for Station `{site_no}`: "
				f"LS Resolution: {ls_resolution}, DEM Resolution: {dem_resolution}"
			)
			# Perform model creation
			try:
				wrapped_model_creation = partial(
					single_model_creation,
					current_user.username, site_no, ls_resolution, dem_resolution
				)
				process = Process(target=wrapped_model_creation)
				process.start()
				self.app.logger.info("Model creation process started successfully.")
				return jsonify({"status": "success", "message": "Model creation started!"})
			except Exception as e:
				self.app.logger.error(f"Error starting model creation: {e}")
				return jsonify({"error": "Failed to start model creation"}), 500

		@self.app.route('/api/user_files', methods=['GET'])
		@login_required
		def api_user_files():
			"""
			Lists directories and files for the logged-in user.
			Supports navigation within subdirectories and provides download links.
			"""
			self.app.logger.info("API User Files route called")
			base_user_dir = os.path.join('/data/SWATGenXApp/Users', current_user.username, "SWATplus_by_VPUID")
			subdir = request.args.get('subdir', '')  # Get subdirectory from query params (default: root)
			target_dir = os.path.join(base_user_dir, subdir)
			self.app.logger.info(f"Listing contents for {current_user.username} in: {target_dir}")
			# Security check: Ensure the requested path stays within the user's directory
			if not target_dir.startswith(base_user_dir) or not os.path.exists(target_dir):
				return jsonify({'error': 'Unauthorized or invalid path'}), 403

			self.app.logger.info(f"Listing contents for {current_user.username} in: {target_dir}")

			# Initialize response structure
			contents = {
				'current_path': subdir,
				'parent_path': os.path.dirname(subdir) if subdir else '',  # Parent directory for navigation
				'directories': [],
				'files': []
			}

			# Ensure the directory exists
			if os.path.isdir(target_dir):
				for item in os.listdir(target_dir):
					safe_item = secure_filename(item)  # Prevent path traversal
					item_path = os.path.join(target_dir, safe_item)

					if os.path.isdir(item_path):
						contents['directories'].append({
							'name': safe_item,
							'path': os.path.join(subdir, safe_item).lstrip('/'),
							'download_zip_url': url_for('download_directory', dirpath=f"{subdir}/{safe_item}".lstrip('/'))
						})
					elif os.path.isfile(item_path):
						contents['files'].append({
							'name': safe_item,
							'download_url': url_for('download_user_file', filename=f"{subdir}/{safe_item}".lstrip('/'))
						})

			return jsonify(contents)

		@self.app.route('/download/<path:filename>', methods=['GET'])
		@login_required
		def download_user_file(filename):
			"""
			Securely serves individual files from the user's directory.
			"""
			user_dir = os.path.join('/data/SWATGenXApp/Users', current_user.username, "SWATplus_by_VPUID")
			full_path = os.path.join(user_dir, filename)

			# Security check: Ensure the requested file is within the user's directory
			if not full_path.startswith(user_dir) or not os.path.isfile(full_path):
				return jsonify({'error': 'File not found or access denied'}), 404

			directory, file = os.path.split(full_path)
			return send_from_directory(directory, file, as_attachment=True)

		@self.app.route('/download-directory/<path:dirpath>', methods=['GET'])
		@login_required
		def download_directory(dirpath):
			"""
			Compresses the requested directory into a ZIP file and serves it for download.
			"""
			user_dir = os.path.join('/data/SWATGenXApp/Users', current_user.username, "SWATplus_by_VPUID")
			full_dir_path = os.path.join(user_dir, dirpath)

			# Ensure the requested directory is within the user's allowed directory
			if not full_dir_path.startswith(user_dir) or not os.path.isdir(full_dir_path):
				self.app.logger.error(f"Unauthorized directory access or not found: {full_dir_path}")
				return jsonify({'error': 'Directory not found or access denied'}), 404

			self.app.logger.info(f"Creating ZIP for directory: {full_dir_path}")

			# Create a temporary ZIP file
			with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
				zip_path = tmp_zip.name  # Temporary file path

			# Create ZIP archive
			try:
				shutil.make_archive(zip_path[:-4], 'zip', full_dir_path)  # Strip .zip from tempfile path
			except Exception as e:
				self.app.logger.error(f"Failed to create ZIP: {e}")
				return jsonify({'error': 'Failed to create ZIP file'}), 500

			zip_file_name = f"{os.path.basename(dirpath)}.zip"
			final_zip_path = zip_path[:-4] + ".zip"  # Ensure proper file name

			# Ensure ZIP exists before sending
			if not os.path.exists(final_zip_path):
				self.app.logger.error(f"ZIP file missing: {final_zip_path}")
				return jsonify({'error': 'ZIP file not found'}), 500

			return send_file(final_zip_path, as_attachment=True, download_name=zip_file_name)

		@self.app.route('/api/logout', methods=['POST'])  # Ensure method is 'POST'
		@login_required
		def logout():
			"""Logout route."""
			username = current_user.username if current_user.is_authenticated else "Anonymous"
			current_app.logger.info(f"Logging out user: {username}.")
			logout_user()
			session.clear()
			return jsonify({
				"status": "success",
				"message": "You have been logged out successfully.",
				"redirect": "/login"
			}), 200

		
		@self.app.route('/get_station_characteristics', methods=['GET'])
		def get_station_characteristics():
			self.app.logger.info("Get Station Characteristics route called")
			station_no = request.args.get('station', None)
			
			# Load station CSV
			station_data = pd.read_csv(SWATGenXPaths.FPS_all_stations, dtype={'SiteNumber': str})
			self.app.logger.info(f"Station number: {station_no}")

			# Find the row with that station
			station_row = station_data[station_data.SiteNumber == station_no]
			if station_row.empty:
				self.app.logger.error(f"Station {station_no} not found in CSV.")
				return jsonify({"error": "Station not found"}), 404

			# Convert row to dict
			characteristics = station_row.iloc[0].to_dict()
			self.app.logger.info(f"Found station row for {station_no}")

			# Check if a model already exists
			existance_flag = check_existing_models(station_no)
			characteristics['model_exists'] = str(existance_flag).capitalize()

			# Safely parse the HUC12 list from the CSV field
			# The CSV field looks like: "['040500040703','040500040508', ... ]"
			huc12_str = characteristics.get('HUC12 ids of the watershed')
			if not huc12_str or pd.isna(huc12_str):
				# If missing/empty, just return characteristics without geometry
				self.app.logger.warning(f"No HUC12 data for station {station_no}")
				return jsonify(characteristics)

			# Parse the string as a Python list
			import ast
			try:
				# This safely evaluates the string as a list:
				# e.g. "['040500040703','040500040508']" -> ["040500040703", "040500040508"]
				huc12_list = ast.literal_eval(huc12_str)
			except Exception as e:
				self.app.logger.error(f"Error parsing HUC12 list for {station_no}: {e}")
				return jsonify({"error": "Failed to parse HUC12 data"}), 500

			# Now call geometry functions safely
			geometries = get_huc12_geometries(huc12_list)
			streams_geometries, lake_identifier = get_huc12_streams_geometries(huc12_list)
			lakes_geometries = get_huc12_lakes_geometries(huc12_list, lake_identifier)

			if not geometries:
				self.app.logger.error(f"No geometries found for HUC12s: {huc12_list}")
			if not streams_geometries:
				self.app.logger.error(f"No streams geometries found for HUC12s: {huc12_list}")
			if not lakes_geometries:
				self.app.logger.warning(f"No lakes geometries found for HUC12s: {huc12_list}")

			# Add geometry data to the dictionary
			characteristics['Num HUC12 subbasins'] = len(huc12_list)
			characteristics['geometries'] = geometries
			characteristics['streams_geometries'] = streams_geometries
			characteristics['lakes_geometries'] = lakes_geometries

			# Clean up if you don’t want that field in your final JSON
			characteristics.pop('HUC12 ids of the watershed', None)

			# Return as JSON
			return jsonify(characteristics)


		@self.app.route('/about')
		@login_required
		@verified_required
		def about():
			self.app.logger.info("About route called")
			#return render_template('about.html')
			return jsonify({"title": "About", "message": "about page"})

		@self.app.route('/model-confirmation')
		@login_required
		@verified_required
		def model_confirmation():
			self.app.logger.info("Model Confirmation route called")
			#return render_template('model_confirmation.html')
			return jsonify({"title": "Model Confirmation", "message": "Model confirmation page"})

		@self.app.route('/contact', methods=['GET', 'POST'])
		@login_required
		@verified_required
		def contact():
			self.app.logger.info("Contact route called")
			form = ContactForm()
			if form.validate_on_submit():
				name = form.name.data
				email = form.email.data
				message = form.message.data
				contact_message = ContactMessage(name=name, email=email, message=message)
				try:
					db.session.add(contact_message)
					db.session.commit()
					self.app.logger.info(f"Message from {name} added to the database")	
					#flash('Your message has been sent successfully!')
					message = "Your message has been sent successfully!"
				except Exception as e:
					self.app.logger.error(f"Error adding message to the database: {e}")	
					db.session.rollback()
					#flash('An error occurred while sending the message. Please try again.')
					message = "An error occurred while sending the message. Please try again."
				#return redirect(url_for('contact'))
				return jsonify({"status": "success", "message": message, "redirect": "/contact"})
			#return render_template('contact.html', form=form)
			return jsonify({"title": "Contact", "message": "contact page", "form": form})


		@self.app.route('/infrastructure')
		@login_required
		@verified_required
		def infrastructure():
			self.app.logger.info("Infrastructure route called")
			#return render_template('infrastructure.html')
			return jsonify({"title": "Infrastructure", "message": "infrastructure page"})

		@self.app.route('/hydro_geo_dataset', methods=['GET', 'POST'])
		@login_required
		@verified_required
		def hydro_geo_dataset():
			self.app.logger.info("HydroGeoDataset route called")
			form = HydroGeoDatasetForm()
			hydrodict = hydrogeo_dataset_dict()

			available_groups = [
				'CDL', 'EBK', 'LANDFIRE', 'MODIS', 'NHDPlus', 'PRISM', 'SNODAS_monthly',
				'Wellogic', 'climate_pattern', 'geospatial', 'gssurgo', 'population'
			]
			form.variable.choices = [(group, group) for group in available_groups]
			self.app.logger.info(f"Available groups: {available_groups}")	
			if request.method == 'POST':
				self.app.logger.info("Form submitted")
				selected_variable = request.form.get('variable')
				if selected_variable in hydrodict:
					form.subvariable.choices = [(item, item) for item in hydrodict[selected_variable]]

				self.app.logger.info("Form validated successfully")	
				variable = request.form.get('variable')
				subvariable = request.form.get('subvariable')

				latitude = request.form.get('latitude', None)
				longitude = request.form.get('longitude', None)
				min_latitude = request.form.get('min_latitude', None)
				max_latitude = request.form.get('max_latitude', None)
				min_longitude = request.form.get('min_longitude', None)
				max_longitude = request.form.get('max_longitude', None)

				if min_latitude and max_latitude and min_longitude and max_longitude:
					self.app.logger.info(f"Received range: ({min_latitude}, {max_latitude}), ({min_longitude}, {max_longitude})")
					latitude = longitude = None

				polygon_coordinates = request.form.get('polygon_coordinates')
				if polygon_coordinates:
					try:
						vertices = json.loads(polygon_coordinates)
						self.app.logger.info(f"Received polygon vertices: {vertices}")
						max_latitudes = [vertex['latitude'] for vertex in vertices]
						max_longitudes = [vertex['longitude'] for vertex in vertices]
						min_latitude = min(max_latitudes)
						max_latitude = max(max_latitudes)
						min_longitude = min(max_longitudes)
						max_longitude = max(max_longitudes)
						self.app.logger.info(f"Polygon bounds: ({min_latitude}, {max_latitude}), ({min_longitude}, {max_longitude})")	
					except Exception as e:
						self.app.logger.error(f"Error parsing polygon coordinates: {e}")
						#flash("Invalid polygon coordinates.", "danger")
						message = "Invalid polygon coordinates."
						#return render_template('HydroGeoDataset.html', form=form)
						return jsonify({"title": "HydroGeoDataset", "message": message, "form": form})

				elif not any([latitude, longitude, min_latitude, max_latitude, min_longitude, max_longitude]):
					#flash("Please provide either a point or a range for data retrieval.", "danger")
					#return render_template('HydroGeoDataset.html', form=form)
					message = "Please provide either a point or a range for data retrieval."
					return jsonify({"title": "HydroGeoDataset", "message": message})

				if not variable or not subvariable:
					#flash("Variable and Subvariable are required.", "danger")
					#return render_template('HydroGeoDataset.html', form=form)
					message = "Variable and Subvariable are required."
					return jsonify({"title": "HydroGeoDataset", "message": message, "form": form})
				try:
					if latitude and longitude:
						self.app.logger.info(f"Fetching data for {variable}/{subvariable} at {latitude}, {longitude}")
						raw_data = read_h5_file(
							lat=float(latitude), lon=float(longitude), address=f"{variable}/{subvariable}"
						)
						# Convert np.float32 to float
						data = {key: float(value) if isinstance(value, np.float32) else value for key, value in raw_data.items()}
						self.app.logger.info(f"Data fetched: {data}")
					elif all([min_latitude, max_latitude, min_longitude, max_longitude]):
						self.app.logger.info(f"Fetching data for {variable}/{subvariable} in range ({min_latitude}, {max_latitude}), ({min_longitude}, {max_longitude})")
						raw_data = read_h5_file(
							lat_range=(float(min_latitude), float(max_latitude)),
							lon_range=(float(min_longitude), float(max_longitude)),
							address=f"{variable}/{subvariable}"
						)
						# Convert np.float32 to float
						data = {key: float(value) if isinstance(value, np.float32) else value for key, value in raw_data.items()}
						self.app.logger.info(f"Data fetched for range: {data}")
					else:
						#flash("Please provide either a point or a range for data retrieval.", "danger")
						#return render_template('HydroGeoDataset.html', form=form)
						message = "Please provide either a point or a range for data retrieval."
						return jsonify({"title": "HydroGeoDataset", "message": message, "form": form})

					#return render_template(
					#	'HydroGeoDataset.html',
					#	form=form,
					#	variable=variable,
					#	subvariable=subvariable,
					#	data=data
					#)
					message = "Data fetched successfully"
					return jsonify({"title": "HydroGeoDataset", "message": message, "data": data})
				
				except Exception as e:
					self.app.logger.error(f"Error fetching data: {e}")	
					#flash(f"Error fetching data: {e}", "danger")
					message = f"Error fetching data: {e}"
			#return render_template('HydroGeoDataset.html', form=form)
			return jsonify({"title": "HydroGeoDataset", "message": message, "form": form})	

		@self.app.route('/get_subvariables', methods=['POST'])
		@login_required
		@verified_required
		def get_subvariables():
			self.app.logger.info("Get Subvariables route called")
			variable = request.form.get('variable')
			self.app.logger.info(f"Received variable: {variable}")

			if not variable:
				return jsonify({"error": "Variable is required"}), 400

			hydrodict = hydrogeo_dataset_dict()
			subvariables = hydrodict.get(variable, [])
			self.app.logger.info(f"Subvariables for {variable}: {subvariables}")

			return jsonify({"subvariables": subvariables})

		@self.app.route('/deeplearning_models')
		@login_required
		@verified_required
		def deeplearning_models():
			#return render_template('DeepLearning.html')
			return jsonify({"title": "Deep Learning Models", "message": "Deep Learning Models page"})
		
		@self.app.route('/vision_system')
		@login_required
		@verified_required
		def vision_system():
			self.app.logger.info("Vision System route called")
			#return render_template('VisionSystem.html')
			return jsonify({"title": "Vision System", "message": "Vision System page"})

		@self.app.route('/michigan')
		@login_required
		@verified_required
		def michigan():
			self.app.logger.info("Michigan route called")  
			return jsonify({
				"title": "Michigan",
				"message": "Michigan page"
			})


		@self.app.route('/search_site', methods=['GET', 'POST'])
		def search_site():
			self.app.logger.info("Search site route called")	
			search_term = request.args.get('search_term', '').lower()
			if not search_term:
				return jsonify({"error": "Search term is required"}), 400
				
			try:
				results = find_station(search_term)
				if results.empty:
					self.app.logger.info("No matching sites found")
					return jsonify({"error": "No matching sites found"}), 404
				return jsonify(results.to_dict(orient='records'))
			except Exception as e:
				self.app.logger.error(f"Error searching for site: {e}")
				return jsonify({"error": "An error occurred during the search"}), 500
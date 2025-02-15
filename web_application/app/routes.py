##/data/SWATGenXApp/codes/web_application/app/routes.py
from flask import (url_for, request,
				jsonify, current_app, session,send_file,
				send_from_directory)
from app.sftp_manager import create_sftp_user  # Import the SFTP user creation function
from flask_login import (login_user, logout_user,
						login_required, current_user)
from app.extensions import csrf
from app.models import User, ContactMessage
from app.forms import (SignUpForm,
						ContactForm)
from multiprocessing import Process
from app.utils import (find_station, get_huc12_geometries, get_huc12_streams_geometries,
						get_huc12_lakes_geometries, send_verification_email, check_existing_models,
						single_model_creation, hydrogeo_dataset_dict, read_h5_file) 
import os
import json
import numpy as np
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
import pandas as pd
from app.forms import VerificationForm
from werkzeug.utils import secure_filename
import shutil
import tempfile
from app.decorators import conditional_login_required, conditional_verified_required
from app.utils import LoggerSetup
from app.extensions import db
from functools import partial
import ast
class AppManager:
	def __init__(self, app):
		self.app = app
		self.init_routes()
		log_dir = "/data/SWATGenXApp/codes/web_application/logs"
		self.app.logger = LoggerSetup(log_dir, rewrite=False).setup_logger("FlaskApp")
		self.app.logger.info("AppManager initialized!")
    
	def init_routes(self):
		@self.app.route('/')
		@conditional_login_required
		@conditional_verified_required
		def index():
			self.app.logger.info("Index route called. Redirecting to /home.")	
			return jsonify({"status": "success", "redirect": "/home"})


		@self.app.route('/api/visualization/<name>/<ver>/<variable>', methods=['GET'])
		@conditional_login_required
		@conditional_verified_required
		def serve_visualization(name, ver, variable):
			video_path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{name}/figures_SWAT_gwflow_MODEL/verifications_videos"
			gif_file = os.path.join(video_path, f"{ver}_{variable}_animation.gif")

			if not os.path.exists(gif_file):
				return jsonify({"error": "Visualization not found"}), 404

			return send_file(gif_file, mimetype='image/gif')



		@self.app.route('/api/verify', methods=['POST'])
		def verify():
			self.app.logger.info("Verification attempt received.")

			data = request.get_json()
			email = data.get('email', '').strip()
			code_entered = data.get('verification_code', '').strip()

			if not email or not code_entered:
				return jsonify({"status": "error", "message": "Email and verification code are required."}), 400

			user = User.query.filter_by(email=email).first()

			if not user:
				self.app.logger.warning(f"Verification failed: User with email `{email}` not found.")
				return jsonify({"status": "error", "message": "User not found."}), 404

			if user.is_verified:
				self.app.logger.warning(f"Verification failed: User `{user.username}` is already verified.")
				return jsonify({"status": "error", "message": "User is already verified."}), 400

			if user.verification_code == code_entered:
				user.is_verified = True
				user.verification_code = None
				db.session.commit()

				self.app.logger.info(f"User `{user.username}` verified successfully. Creating SFTP account...")

				sftp_result = create_sftp_user(user.username)
				if sftp_result.get("status") != "success":
					self.app.logger.error(f"SFTP creation failed for {user.username}: {sftp_result.get('error')}")
					return jsonify({"status": "error", "message": "SFTP account creation failed. Contact support."}), 500

				return jsonify({
					"status": "success",
					"message": "Verification successful. Please log in.",
					"redirect": "/login"
				})

			self.app.logger.warning(f"Verification failed: Invalid code for user `{user.username}`.")
			return jsonify({"status": "error", "message": "Invalid verification code."}), 400



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

		@self.app.route('/user_dashboard', methods=['GET'])
		@conditional_login_required
		@conditional_verified_required
		def user_dashboard():
			"""User dashboard route."""
			self.app.logger.info(f"User Dashboard accessed by `{current_user.username}`.")
			return jsonify({"title": "Dashboard", "message": "Your user dashboard."})

		@self.app.route('/get_options', methods=['GET'])
		@conditional_login_required
		@conditional_verified_required
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
		@conditional_login_required
		@conditional_verified_required
		def visualizations():
			self.app.logger.info("Visualizations route called")
			name = request.args.get('NAME', default=None)
			ver = request.args.get('ver', default=None)
			variable = request.args.get('variable', default=None)

			if not all([name, ver, variable]):
				if request.headers.get("X-Requested-With") == "XMLHttpRequest":
					self.app.logger.error("Please provide NAME, Version, and Variable.")
					return jsonify({"error": "Please provide NAME, Version, and Variable."}), 400
				else:
					self.app.logger.error("Please provide NAME, Version, and Variable.")
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
				if not user.is_verified:
					return jsonify({"error": "Email not verified. Please check your email."}), 403

				login_user(user)
				session.permanent = True
				return jsonify({"success": True, "token": "someJWT"}), 200

			return jsonify({"error": "Invalid username or password"}), 401


								
		@self.app.route('/api/signup', methods=['POST'])
		def signup():
			self.app.logger.info("Sign Up route called via API")
			data = request.get_json()

			username = data.get('username')
			email = data.get('email')
			password = data.get('password')
			confirm_password = data.get('confirmPassword')

			errors = {}

			if not username:
				errors['username'] = 'Username is required'
			elif User.query.filter_by(username=username).first():
				errors['username'] = 'That username is taken. Please choose a different one.'

			if not email:
				errors['email'] = 'Email is required'
			elif User.query.filter_by(email=email).first():
				errors['email'] = 'That email is already in use. Please choose a different one.'

			if not password:
				errors['password'] = 'Password is required'
			elif len(password) < 8:
				errors['password'] = 'Password must be at least 8 characters long.'
			elif not any(c.isupper() for c in password):
				errors['password'] = 'Password must contain at least one uppercase letter.'
			elif not any(c.islower() for c in password):
				errors['password'] = 'Password must contain at least one lowercase letter.'
			elif not any(c.isdigit() for c in password):
				errors['password'] = 'Password must contain at least one number.'
			elif not any(c in '@#$^&*()_+={}\[\]|\\:;"\'<>,.?/~`-' for c in password):
				errors['password'] = 'Password must contain at least one special character.'

			if password != confirm_password:
				errors['confirmPassword'] = 'Passwords do not match.'

			if errors:
				return jsonify({"status": "error", "message": "Validation failed", "errors": errors}), 400

			try:
				verification_code = send_verification_email(email)
				new_user = User(username=username, email=email, password=password, verification_code=verification_code, is_verified=False)

				db.session.add(new_user)
				db.session.commit()
				self.app.logger.info(f"User `{new_user.username}` created in unverified state. Verification email sent.")

				# Do NOT create SFTP here!

				return jsonify({"status": "success", "message": "Check your email for verification code.", "redirect": "/verify"})


			except Exception as e:
				db.session.rollback()
				self.app.logger.error(f"Error creating user: {e}")
				return jsonify({"status": "error", "message": "An error occurred while creating the account."}), 500



		@self.app.route('/home', methods=['GET'])
		@conditional_login_required
		@conditional_verified_required
		def home():
			"""User's home page."""
			self.app.logger.info(f"Home route accessed by user: {current_user.username}.")
			return jsonify({"title": "Home", "message": "Welcome to the app!"})

		@self.app.route('/model-settings', methods=['POST'])
		@conditional_login_required
		@conditional_verified_required
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
		@conditional_login_required
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
		@conditional_login_required
		def download_user_file(filename):
			"""
			Securely serves individual files from the user's directory.
			"""
			user_dir = os.path.join('/data/SWATGenXApp/Users', current_user.username, "SWATplus_by_VPUID")
			full_path = os.path.join(user_dir, filename)

			# Security check: Ensure the requested file is within the user's directory
			if not full_path.startswith(user_dir) or not os.path.isfile(full_path):
				self.app.logger.error(f"File not found or access denied: {full_path}")
				return jsonify({'error': 'File not found or access denied'}), 404

			self.app.logger.info(f"Serving file for download: {full_path}")
			directory, file = os.path.split(full_path)
			return send_from_directory(directory, file, as_attachment=True)

		@self.app.route('/download-directory/<path:dirpath>', methods=['GET'])
		@conditional_login_required
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

			self.app.logger.info(f"Serving ZIP file for download: {final_zip_path}")
			return send_file(final_zip_path, as_attachment=True, download_name=zip_file_name)

		@self.app.route('/api/logout', methods=['POST'])  # Ensure method is 'POST'
		@conditional_login_required
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
			try:
				#NOTE:This safely evaluates the string as a list:
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
		@conditional_login_required
		@conditional_verified_required
		def about():
			self.app.logger.info("About route called")
			#return render_template('about.html')
			return jsonify({"title": "About", "message": "about page"})

		@self.app.route('/model-confirmation')
		@conditional_login_required
		@conditional_verified_required
		def model_confirmation():
			self.app.logger.info("Model Confirmation route called")
			#return render_template('model_confirmation.html')
			return jsonify({"title": "Model Confirmation", "message": "Model confirmation page"})

		@self.app.route('/contact', methods=['GET', 'POST'])
		@conditional_login_required
		@conditional_verified_required
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

		@self.app.route('/hydro_geo_dataset', methods=['GET', 'POST'])
		@conditional_login_required
		@conditional_verified_required
		def hydro_geo_dataset():
			"""Handles HydroGeoDataset requests for fetching environmental data."""
			
			self.app.logger.info("HydroGeoDataset route called")

			# Define available dataset groups
			available_groups = [
				'CDL', 'EBK', 'LANDFIRE', 'MODIS', 'NHDPlus', 'PRISM', 'SNODAS_monthly',
				'Wellogic', 'climate_pattern', 'geospatial', 'gssurgo', 'population'
			]
			
			# Load dataset dictionary
			hydrodict = hydrogeo_dataset_dict()

			if request.method == 'GET':
				"""Handle GET request to fetch available variables or subvariables."""
				self.app.logger.info("GET request received")

				variable = request.args.get('variable')
				if variable:
					self.app.logger.info(f"Fetching subvariables for variable: {variable}")
					subvariables = hydrodict.get(variable, [])
					return jsonify({"subvariables": subvariables})
				else:
					return jsonify({"variables": available_groups})

			elif request.method == 'POST':
				"""Handle POST request to fetch data based on user input."""
				self.app.logger.info("Form submitted")
				data_payload = request.get_json()
				self.app.logger.info(f"Received JSON data: {json.dumps(data_payload, indent=2)}")

				# Extract variable and subvariable
				variable = data_payload.get('variable')
				subvariable = data_payload.get('subvariable')

				if not variable or not subvariable:
					message = "Variable and Subvariable are required."
					self.app.logger.error(message)
					return jsonify({"title": "HydroGeoDataset", "message": message}), 400

				# Extract coordinates
				latitude = data_payload.get('latitude')
				longitude = data_payload.get('longitude')
				min_latitude = data_payload.get('min_latitude')
				max_latitude = data_payload.get('max_latitude')
				min_longitude = data_payload.get('min_longitude')
				max_longitude = data_payload.get('max_longitude')

				# Extract polygon if provided
				polygon_coordinates = data_payload.get('polygon_coordinates')

				if polygon_coordinates:
					try:
						vertices = json.loads(polygon_coordinates)
						self.app.logger.info(f"Received polygon vertices: {vertices}")

						# Convert polygon to bounding box
						latitudes = [vertex['latitude'] for vertex in vertices]
						longitudes = [vertex['longitude'] for vertex in vertices]
						
						min_latitude, max_latitude = min(latitudes), max(latitudes)
						min_longitude, max_longitude = min(longitudes), max(longitudes)
						
						self.app.logger.info(f"Polygon bounds: ({min_latitude}, {max_latitude}), ({min_longitude}, {max_longitude})")
						latitude = longitude = None  # Reset single point

					except Exception as e:
						self.app.logger.error(f"Error parsing polygon coordinates: {e}")
						return jsonify({"title": "HydroGeoDataset", "message": "Invalid polygon coordinates."}), 400

				# Validate input (must have either a point or a bounding box)
				if latitude and longitude:
					self.app.logger.info(f"Fetching data for {variable}/{subvariable} at ({latitude}, {longitude})")
					try:
						raw_data = read_h5_file(
							lat=float(latitude),
							lon=float(longitude),
							address=f"{variable}/{subvariable}"
						)
						data = {key: float(value) if isinstance(value, np.float32) else value for key, value in raw_data.items()}
						return jsonify({"title": "HydroGeoDataset", "message": "Data fetched successfully", "data": data})
					except Exception as e:
						self.app.logger.error(f"Error fetching data: {e}")
						return jsonify({"title": "HydroGeoDataset", "message": f"Error fetching data: {e}"}), 500

				elif all([min_latitude, max_latitude, min_longitude, max_longitude]):
					self.app.logger.info(f"Fetching data for {variable}/{subvariable} in range ({min_latitude}, {max_latitude}), ({min_longitude}, {max_longitude})")
					try:
						raw_data = read_h5_file(
							lat_range=(float(min_latitude), float(max_latitude)),
							lon_range=(float(min_longitude), float(max_longitude)),
							address=f"{variable}/{subvariable}"
						)
						data = {key: float(value) if isinstance(value, np.float32) else value for key, value in raw_data.items()}
						return jsonify({"title": "HydroGeoDataset", "message": "Data fetched successfully", "data": data})
					except Exception as e:
						self.app.logger.error(f"Error fetching data: {e}")
						return jsonify({"title": "HydroGeoDataset", "message": f"Error fetching data: {e}"}), 500

				else:
					message = f"Please provide either a point (latitude/longitude) or a range (min/max lat/lon)."
					self.app.logger.error(message)
					return jsonify({"title": "HydroGeoDataset", "message": message}), 400


		@self.app.route('/get_subvariables', methods=['POST'])
		@conditional_login_required
		@conditional_verified_required
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
		
		@self.app.route('/vision_system')
		@conditional_login_required
		@conditional_verified_required
		def vision_system():
			self.app.logger.info("Vision System route called")
			return jsonify({"title": "Vision System", "message": "Vision System page"})

		@self.app.route('/michigan')
		@conditional_login_required
		@conditional_verified_required
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
from flask import (render_template, url_for, request, flash,
				jsonify, current_app, session,send_file,
				send_from_directory, redirect, flash)

from flask_login import (login_user, logout_user,
						login_required, current_user)

from app.models import User, ContactMessage

from app.forms import (RegistrationForm, LoginForm, 
						ContactForm, ModelSettingsForm, 
						HydroGeoDatasetForm)

from app.extensions import db
from functools import partial, wraps
from multiprocessing import Process
from app.utils import (find_station, get_huc12_geometries, get_huc12_streams_geometries,
						 get_huc12_lakes_geometries, send_verification_email, LoggerSetup, 
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
            flash("You must verify your email first!")
            return redirect(url_for('verify'))
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
		self.logger = LoggerSetup(report_path="/data/SWATGenXApp/codes/web_application/logs", verbose=True, rewrite=True)
		self.logger = self.logger.setup_logger("WebAppLogger")

	def init_routes(self):
		@self.app.route('/')
		@login_required
		@verified_required
		def index():
			self.logger.info("Index route called. Redirecting to /home.")	
			return redirect(url_for('home'))

		@self.app.route('/verify', methods=['GET', 'POST'])
		@login_required
		def verify():
			self.logger.info("Verify route called")
			form = VerificationForm()

			if form.validate_on_submit():
				code_entered = form.verification_code.data.strip()
				if current_user.verification_code == code_entered:
					current_user.is_verified = True
					current_user.verification_code = None
					db.session.commit()
					flash("Email verified successfully!")
					self.logger.info("User verified successfully.")
					return redirect(url_for('home'))
				else:
					flash("Invalid code.")
					self.logger.error("Invalid verification code entered.")

			# If GET or invalid code, just show the verify template
			return render_template('verify.html', form=form)

		@self.app.route('/privacy')
		def privacy():
			self.logger.info("Privacy route called")
			return render_template('privacy.html')
		
		@self.app.route('/terms')
		def terms():
			self.logger.info("Terms route called")
			return render_template('terms.html')

		@self.app.route('/js/<path:filename>')
		def js_static(filename):
			js_dir = os.path.join(current_app.root_path,  'js')
			return send_from_directory(js_dir, filename)

		@self.app.route('/css/<path:filename>')
		def css_static(filename):
			css_dir = os.path.join(current_app.root_path, 'css')
			return send_from_directory(css_dir, filename)

		@self.app.route('/user_dashboard')
		@login_required
		@verified_required
		def user_dashboard():
			self.logger.info("User Dashboard route called")
			return render_template('user_dashboard.html')

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
				self.logger.error(f"Error fetching options: {e}")
				return jsonify({"error": "Failed to fetch options"}), 500

		@self.app.route('/visualizations', methods=['GET'])
		@login_required
		@verified_required
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

			base_path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{name}/figures_SWAT_gwflow_MODEL"
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
		
		@self.app.route('/oauth_callback')
		def oauth_callback():
			code = request.args.get('code')
			if not code:
				flash('Authorization failed. Please try again.', 'danger')
				return redirect(url_for('login'))

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
				flash('Failed to authenticate with MSU. Please try again.', 'danger')
				return redirect(url_for('login'))

			token = response.json().get("access_token")

			# Retrieve user info
			user_info_url = "https://oauth.msu.edu/userinfo"
			headers = {"Authorization": f"Bearer {token}"}
			user_info_response = requests.get(user_info_url, headers=headers)
			if user_info_response.status_code != 200:
				flash('Failed to retrieve user information.', 'danger')
				return redirect(url_for('login'))

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
			self.logger.info(f"MSU login successful for: {msu_netid}")
			return redirect(url_for('home'))

		@self.app.route('/login', methods=['GET', 'POST'])
		def login():
			self.logger.info("Login route called")
			form = LoginForm()

			if form.validate_on_submit():
				username = form.username.data
				password = form.password.data
				self.logger.warning(f"Login attempt for: {username}")

				user = User.query.filter_by(username=username).first()
				if user and user.check_password(password):
					# ✅ Log the user in
					login_user(user)
					session.permanent = True

					# ✅ Check SFTP directory access
					sftp_home_dir = f"/data/SWATGenXApp/Users/{username}"
					if not os.path.exists(sftp_home_dir):
						self.logger.error(f"SFTP home directory missing for {username}. Attempting to create...")
						os.makedirs(sftp_home_dir, mode=0o770, exist_ok=True)
						os.chown(sftp_home_dir, user.id, os.getgid())  # Ensure correct ownership

					# ✅ Check if user has correct access
					if not os.access(sftp_home_dir, os.W_OK):
						self.logger.error(f"User {username} does not have write access to SFTP directory.")
						flash("SFTP directory access issue. Contact support.", "danger")
						return redirect(url_for('logout'))

					self.logger.info(f"SFTP directory verified for {username}: {sftp_home_dir}")

					flash("Login successful!", "success")
					return redirect(url_for('home'))
				else:
					self.logger.error("Invalid username or password")
					flash('Invalid username or password', 'danger')

			return render_template('login.html', form=form)

			# ✅ Allow MSU NetID login as an **OPTION**, but do not require it
			#msu_login_url = (
			#	"https://oauth.msu.edu/authorize?"
			#	f"response_type=code&client_id={CLIENT_ID}&"
			#	f"redirect_uri={url_for('oauth_callback', _external=True)}&"
			#	"scope=profile"
			#)

			#return render_template('login.html', form=form)#, msu_login_url=msu_login_url)

		from app.sftp_manager import create_sftp_user  # Import the SFTP user creation function

		@self.app.route('/signup', methods=['GET', 'POST'])
		def signup():
			"""
			- Use at least one uppercase, one lowercase, and one number.
			- Use special characters (@ # $ ^ & * - _ ! + = [ ] { } | \ : ' , . ? / ` ~ ” ( ) ;).
			- Do not include your first, middle, or last name.
			- Do not reuse old passwords or share them.
			"""
			self.logger.info("Sign Up route called")
			form = RegistrationForm()

			if form.validate_on_submit():
				# Generate verification code
				verification_code = send_verification_email(form.email.data)
				self.logger.info("Form validated successfully")

				# ✅ Create the new user
				new_user = User(
					username=form.username.data,
					email=form.email.data,
					password=form.password.data,  # Hashed internally
					verification_code=verification_code
				)

				db.session.add(new_user)
				db.session.commit()

				try:
					# 🔹 **Ensure user is added before creating SFTP**
					db.session.add(new_user)
					db.session.commit()
					self.logger.info("User added to DB but not verified yet.")

					# ✅ **Automatically Create an SFTP Account**
					sftp_result = create_sftp_user(new_user.username)

					if sftp_result.get("status") != "success":
						self.logger.error(f"Failed to create SFTP account for {new_user.username}: {sftp_result.get('error')}")
						flash("SFTP account creation failed. Contact support.", "danger")

					# ✅ **Log the user in immediately**
					login_user(new_user)

					# ✅ **Redirect to verification page**
					flash("Please check your email and enter the verification code below.")
					return redirect(url_for('verify'))

				except Exception as e:
					self.logger.error(f"Error adding user to the database: {e}")
					db.session.rollback()
					flash("An error occurred while creating the account. Please try again.", "danger")

			else:
				self.logger.error("Form validation failed")
				for field, errors in form.errors.items():
					for error in errors:
						self.logger.error(f"Error in {field}: {error}")

			return render_template('register.html', form=form)

		@self.app.route('/home')
		@login_required
		@verified_required
		def home():
			self.logger.info("Home route called, user is authenticated.")	
			return render_template('home.html')

		@self.app.route('/model-settings', methods=['GET', 'POST'])
		@login_required
		@verified_required
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
				wrapped_single_model_creation = partial(single_model_creation, current_user.username, site_no, ls_resolution, dem_resolution, calibration_flag, validation_flag, sensitivity_flag, cal_pool_size, sen_pool_size, sen_total_evaluations, num_levels, max_cal_iterations, verification_samples)
				process = Process(target=wrapped_single_model_creation)
				process.start()
				self.logger.info("Model creation process started")

				return redirect(url_for('model_confirmation'))

			#station_data = integrate_streamflow_data()
			station_data = pd.read_csv(SWATGenXPaths.FPS_all_stations, dtype={'SiteNumber': str})
			station_list = station_data.SiteNumber.unique()
			return render_template('model_settings.html', form=form, output=output, station_list=station_list)

		@self.app.route('/api/user_files', methods=['GET'])
		@login_required
		def api_user_files():
			"""
			Lists directories and files for the logged-in user.
			Supports navigation within subdirectories and provides download links.
			"""
			self.logger.info("API User Files route called")
			base_user_dir = os.path.join('/data/SWATGenXApp/Users', current_user.username, "SWATplus_by_VPUID")
			subdir = request.args.get('subdir', '')  # Get subdirectory from query params (default: root)
			target_dir = os.path.join(base_user_dir, subdir)
			self.logger.info(f"Listing contents for {current_user.username} in: {target_dir}")
			# Security check: Ensure the requested path stays within the user's directory
			if not target_dir.startswith(base_user_dir) or not os.path.exists(target_dir):
				return jsonify({'error': 'Unauthorized or invalid path'}), 403

			self.logger.info(f"Listing contents for {current_user.username} in: {target_dir}")

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
				self.logger.error(f"Unauthorized directory access or not found: {full_dir_path}")
				return jsonify({'error': 'Directory not found or access denied'}), 404

			self.logger.info(f"Creating ZIP for directory: {full_dir_path}")

			# Create a temporary ZIP file
			with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
				zip_path = tmp_zip.name  # Temporary file path

			# Create ZIP archive
			try:
				shutil.make_archive(zip_path[:-4], 'zip', full_dir_path)  # Strip .zip from tempfile path
			except Exception as e:
				self.logger.error(f"Failed to create ZIP: {e}")
				return jsonify({'error': 'Failed to create ZIP file'}), 500

			zip_file_name = f"{os.path.basename(dirpath)}.zip"
			final_zip_path = zip_path[:-4] + ".zip"  # Ensure proper file name

			# Ensure ZIP exists before sending
			if not os.path.exists(final_zip_path):
				self.logger.error(f"ZIP file missing: {final_zip_path}")
				return jsonify({'error': 'ZIP file not found'}), 500

			return send_file(final_zip_path, as_attachment=True, download_name=zip_file_name)

		@self.app.route('/logout', methods=['GET'])
		@login_required
		def logout():
			"""
			Logs the user out and redirects to the login page.
			"""
			self.logger.info(f"User {current_user.username} logged out.")
			logout_user()
			session.clear()  # Clear all session data
			flash("You have been logged out successfully.", "info")
			return redirect(url_for('login'))

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
			existance_flag = check_existing_models(station_no)
			characteristics['model_exists'] = str(existance_flag).capitalize()
			self.logger.info(f"Existance flag: {existance_flag}")
			self.logger.info(f"characteristics: {characteristics}")

			if characteristics:
				huc12_list = characteristics.get('HUC12 ids of the watershed', [])
				huc12_list = [str(x.split("'")[1]) for x in huc12_list[1:-1].split(',')]
				geometries = get_huc12_geometries(huc12_list)
				streams_geometries, lake_identifier = get_huc12_streams_geometries(huc12_list)
				lakes_geometries = get_huc12_lakes_geometries(huc12_list, lake_identifier)
				### check if the geometries are empty
				if not geometries:
					self.logger.error(f"No geometries found for HUC12s: {huc12_list}")
				if not streams_geometries:
					self.logger.error(f"No streams geometries found for HUC12s: {huc12_list}")
				if not lakes_geometries:
					self.logger.warning(f"No lakes geometries found for HUC12s: {huc12_list}")
	
				characteristics.pop('HUC12 ids of the watershed')
				characteristics['Num HUC12 subbasins'] = len(huc12_list)
				characteristics['geometries'] = geometries
				characteristics['streams_geometries'] = streams_geometries
				characteristics['lakes_geometries'] = lakes_geometries
				return jsonify(characteristics)
			else:
				return jsonify({"error": "Station not found"}), 404

		@self.app.route('/about')
		@login_required
		@verified_required
		def about():
			self.logger.info("About route called")
			return render_template('about.html')

		@self.app.route('/model-confirmation')
		@login_required
		@verified_required
		def model_confirmation():
			self.logger.info("Model Confirmation route called")
			return render_template('model_confirmation.html')

		@self.app.route('/contact', methods=['GET', 'POST'])
		@login_required
		@verified_required
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
		@login_required
		@verified_required
		def infrastructure():
			self.logger.info("Infrastructure route called")
			return render_template('infrastructure.html')

		@self.app.route('/hydro_geo_dataset', methods=['GET', 'POST'])
		@login_required
		@verified_required
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
		@login_required
		@verified_required
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

		@self.app.route('/deeplearning_models')
		@login_required
		@verified_required
		def deeplearning_models():
			return render_template('DeepLearning.html')
		
		@self.app.route('/vision_system')
		@login_required
		@verified_required
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
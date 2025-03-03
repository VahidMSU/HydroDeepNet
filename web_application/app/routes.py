##/data/SWATGenXApp/codes/web_application/app/routes.py
from flask import (url_for, request, jsonify, current_app, session, send_file,
                   send_from_directory, redirect)
from app.sftp_manager import create_sftp_user  # Import the SFTP user creation function
from flask_login import (login_user, logout_user,
						current_user)
from app.extensions import csrf
from app.models import User, ContactMessage
from multiprocessing import Process
from app.utils import (find_station, get_huc12_geometries, get_huc12_streams_geometries,
						get_huc12_lakes_geometries, send_verification_email, check_existing_models,
						single_model_creation, hydrogeo_dataset_dict, read_h5_file) 
import os
import json
import numpy as np
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
import pandas as pd
from werkzeug.utils import secure_filename
import shutil
import tempfile
from app.decorators import conditional_login_required, conditional_verified_required
from app.utils import LoggerSetup
from app.extensions import db
from functools import partial
import ast
from flask import send_from_directory
import requests
from flask import Flask, request, jsonify
import sys	
from AI_agent.interactive_agent import interactive_session
from AI_agent.report_generator import generate_reports
import os
from app.tasks import create_model_task
from datetime import datetime
import threading
import zipfile



class AppManager:
	def __init__(self, app):
		self.app = app
		self.init_routes()
		log_dir = "/data/SWATGenXApp/codes/web_application/logs"
		self.app.logger = LoggerSetup(log_dir, rewrite=False).setup_logger("FlaskApp")
		self.app.logger.info("AppManager initialized!")

	def init_routes(self):
		@self.app.route('/api/index')
		@conditional_login_required
		@conditional_verified_required
		def index():
			return jsonify({"status": "success", "message": "Welcome to the API!"})

		@self.app.route('/static/images/<path:filename>')
		def serve_images(filename):
			return send_from_directory('/data/SWATGenXApp/GenXAppData/images', filename)

		@self.app.route('/static/videos/<path:filename>')
		def serve_videos(filename):
			return send_from_directory('/data/SWATGenXApp/GenXAppData/videos', filename)

		
		@self.app.route('/static/visualizations/<name>/<ver>/<variable>.gif', methods=['GET'])
		@conditional_login_required
		@conditional_verified_required
		def serve_visualization(name, ver, variable):
			"""
			Serve visualization GIFs with proper error handling
			"""
			video_path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{name}/figures_SWAT_gwflow_MODEL/verifications_videos"
			gif_file = f"{ver}_{variable}_animation.gif"
			full_path = os.path.join(video_path, gif_file)

			if not os.path.exists(full_path):
				self.app.logger.error(f"Visualization not found: {full_path}")
				return jsonify({"error": "Visualization not found"}), 404

			try:
				return send_file(full_path, mimetype='image/gif')
			except Exception as e:
				self.app.logger.error(f"Error serving visualization: {e}")
				return jsonify({"error": "Error serving visualization"}), 500


		@self.app.route('/', defaults={'path': ''})
		@self.app.route('/<path:path>')
		def serve_frontend(path):
			frontend_build_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'build')
			
			# First try to serve static files
			if path.startswith('static/'):
				return send_from_directory(os.path.join(frontend_build_dir), path)
			
			# Then try to serve the file directly if it exists
			if path and os.path.exists(os.path.join(frontend_build_dir, path)):
				return send_from_directory(frontend_build_dir, path)
			
			# Default to serving index.html
			return send_from_directory(frontend_build_dir, 'index.html')


		@self.app.route('/api/chatbot/initialize', methods=['POST'])
		def chatbot_initialize():
			"""Initialize the AI agent with specific context."""
			try:
				data = request.get_json()
				context = data.get('context', 'general') if data else 'general'
				
				# You can customize the welcome message based on context
				welcome_messages = {
					'hydrogeo_dataset': "Hello! I'm your HydroGeo Assistant. I can help you understand and query the environmental and hydrological datasets available in our system. You can ask about data formats, specific variables, or how to interpret results. What would you like to know?",
					'general': "Hello! I'm ready to assist you with questions about our environmental data platform. How can I help you today?"
				}
				
				welcome_message = welcome_messages.get(context, welcome_messages['general'])
				
				# You could initialize any session state here in your actual implementation
				return jsonify({
					"status": "success", 
					"welcome_message": welcome_message
				})
			except Exception as e:
				self.app.logger.error(f"Error initializing chatbot: {e}")
				return jsonify({"error": "Failed to initialize chatbot"}), 500

		@self.app.route('/api/chatbot', methods=['POST'])
		def chatbot_proxy():
			"""Invoke the AI agent to interact with the user."""
			try:
				data = request.get_json()
				message = data.get('message') if data else None
				
				if not message:
					return jsonify({"error": "Message is required"}), 400
				
				try:
					# Import the function directly, not the module
					
					self.app.logger.info(f"Calling interactive_agent with message: {message}")
					# Call the function
					response = interactive_agent(message)
				except ImportError as e:
					self.app.logger.error(f"Failed to import interactive_agent: {e}")
					response = "I'm sorry, but I'm having trouble accessing my knowledge base right now. Please try again later or contact support."
				except Exception as e:
					self.app.logger.error(f"Error calling interactive_agent: {e}")
					response = "I'm sorry, I encountered an error while processing your request. Please try again with a different question."
				
				return jsonify({"response": response})
			except Exception as e:
				self.app.logger.error(f"Error in chatbot proxy: {e}")
				return jsonify({"error": "Failed to process request", "response": "I'm sorry, I encountered an error while processing your request."}), 500

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
				error_msg = "Please provide NAME, Version, and Variable."
				self.app.logger.error(error_msg)
				return jsonify({"error": error_msg}), 400

			base_path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{name}/figures_SWAT_gwflow_MODEL"
			if not os.path.exists(base_path):
				error_msg = f"No visualization data found for watershed: {name}"
				self.app.logger.error(error_msg)
				return jsonify({"error": error_msg}), 404

			video_path = os.path.join(base_path, "verifications_videos")
			if not os.path.exists(video_path):
				error_msg = f"No visualization videos found for watershed: {name}"
				self.app.logger.error(error_msg)
				return jsonify({"error": error_msg}), 404

			variables = variable.split(",")
			gif_urls = []
			missing_vars = []

			for var in variables:
				gif_file = os.path.join(video_path, f"{ver}_{var}_animation.gif")
				if os.path.exists(gif_file):
					gif_urls.append(f"/static/visualizations/{name}/{ver}/{var}.gif")
				else:
					missing_vars.append(var)
					self.app.logger.warning(f"Missing visualization for {var} in {gif_file}")

			if not gif_urls:
				error_msg = f"No visualizations found for NAME: {name}, Version: {ver}, Variables: {variables}."
				if missing_vars:
					error_msg += f" Missing variables: {', '.join(missing_vars)}"
				self.app.logger.error(error_msg)
				return jsonify({"error": error_msg}), 404

			response_data = {
				"gif_files": gif_urls
			}
			
			if missing_vars:
				response_data["warnings"] = f"Some variables were not found: {', '.join(missing_vars)}"

			return jsonify(response_data)
		
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
			ls_resolution = data.get("ls_resolution", 250)
			dem_resolution = data.get("dem_resolution", 30)

			self.app.logger.info(
				f"Model settings received for Station `{site_no}`: "
				"LS Resolution: {ls_resolution}, DEM Resolution: {dem_resolution}"
			)
			# Perform model creation
			try:
				### if user is anonymous, do not proceed
				if current_user.is_anonymous:
					self.app.logger.warning("User is not logged in. Using 'None' as username.")
					import time
					time.sleep(5)
					return jsonify({"error": "User is not logged in"}), 403
					
				# Submit task to Celery - proper import
				task = create_model_task.delay(
					current_user.username, 
					site_no, 
					ls_resolution, 
					dem_resolution
				)
				
				self.app.logger.info(f"Model creation task {task.id} scheduled successfully.")
				return jsonify({
					"status": "success", 
					"message": "Model creation started!",
					"task_id": task.id
				})
			except Exception as e:
				self.app.logger.error(f"Error scheduling model creation: {e}")
				return jsonify({"error": f"Failed to start model creation: {str(e)}"}), 500


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
			user_dir = os.path.join('/data/SWATGenXApp/Users', current_user.username, "SWATplus_by_VPUID")
			full_path = os.path.join(user_dir, filename)

			if not full_path.startswith(user_dir) or not os.path.isfile(full_path):
				self.app.logger.error(f"File not found or access denied: {full_path}")
				return jsonify({'error': 'File not found or access denied'}), 404

			self.app.logger.info(f"Serving file for download: {full_path}")
			directory, file = os.path.split(full_path)
			return send_from_directory(directory, file, as_attachment=True)

		@self.app.route('/download-directory/<path:dirpath>', methods=['GET'])
		@conditional_login_required
		def download_directory(dirpath):
			user_dir = os.path.join('/data/SWATGenXApp/Users', current_user.username, "SWATplus_by_VPUID")
			full_dir_path = os.path.join(user_dir, dirpath)

			if not full_dir_path.startswith(user_dir) or not os.path.isdir(full_dir_path):
				self.app.logger.error(f"Unauthorized directory access or not found: {full_dir_path}")
				return jsonify({'error': 'Directory not found or access denied'}), 404

			self.app.logger.info(f"Creating ZIP for directory: {full_dir_path}")

			with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
				zip_path = tmp_zip.name

			try:
				shutil.make_archive(zip_path[:-4], 'zip', full_dir_path)
			except Exception as e:
				self.app.logger.error(f"Failed to create ZIP: {e}")
				return jsonify({'error': 'Failed to create ZIP file'}), 500

			zip_file_name = f"{os.path.basename(dirpath)}.zip"
			final_zip_path = zip_path[:-4] + ".zip"

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
		
		@self.app.route('/contact', methods=['POST'])
		@conditional_login_required
		@conditional_verified_required
		def contact():
			self.app.logger.info("Contact route called")

			data = request.get_json()
			if not data:
				return jsonify({"status": "error", "message": "Invalid JSON data"}), 400

			name = data.get('name')
			email = data.get('email')
			message = data.get('message')

			if not all([name, email, message]):
				return jsonify({"status": "error", "message": "All fields are required"}), 400

			contact_message = ContactMessage(name=name, email=email, message=message)
			try:
				db.session.add(contact_message)
				db.session.commit()
				self.app.logger.info(f"Message from {name} added to the database")
				return jsonify({"status": "success", "message": "Your message has been sent successfully!"})
			except Exception as e:
				self.app.logger.error(f"Error adding message to the database: {e}")
				db.session.rollback()
				return jsonify({"status": "error", "message": "An error occurred while sending the message."}), 500

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
					return jsonify({"error": message}), 400

				# Extract coordinates
				latitude = data_payload.get('latitude')
				longitude = data_payload.get('longitude')
				min_latitude = data_payload.get('min_latitude')
				max_latitude = data_payload.get('max_latitude')
				min_longitude = data_payload.get('min_longitude')
				max_longitude = data_payload.get('max_longitude')

				# Extract polygon if provided and process it
				polygon_coordinates = data_payload.get('polygon_coordinates')

				if polygon_coordinates:
					try:
						# Parse the polygon coordinates - handle both string and direct JSON
						if isinstance(polygon_coordinates, str):
							vertices = json.loads(polygon_coordinates)
						else:
							vertices = polygon_coordinates
							
						self.app.logger.info(f"Received polygon vertices: {vertices}")

						# Convert polygon to bounding box if needed
						if isinstance(vertices, list) and vertices:
							# Check if vertices is a list of coordinate objects or a list of coordinate pairs
							if isinstance(vertices[0], dict) and 'latitude' in vertices[0] and 'longitude' in vertices[0]:
								# Format: [{'latitude': x, 'longitude': y}, ...]
								latitudes = [float(vertex['latitude']) for vertex in vertices]
								longitudes = [float(vertex['longitude']) for vertex in vertices]
							elif isinstance(vertices[0], (list, tuple)) and len(vertices[0]) >= 2:
								# Format: [[lon, lat], ...]
								latitudes = [float(vertex[1]) for vertex in vertices]
								longitudes = [float(vertex[0]) for vertex in vertices]
							else:
								raise ValueError(f"Unrecognized vertices format: {vertices[0]}")
								
							min_latitude, max_latitude = min(latitudes), max(latitudes)
							min_longitude, max_longitude = min(longitudes), max(longitudes)
							
							self.app.logger.info(f"Polygon bounds: ({min_latitude}, {max_latitude}), ({min_longitude}, {max_longitude})")

					except Exception as e:
						self.app.logger.error(f"Error parsing polygon coordinates: {e}")
						return jsonify({"error": f"Invalid polygon coordinates: {e}"}), 400

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
						return jsonify({"message": "Data fetched successfully", "data": data})
					except Exception as e:
						self.app.logger.error(f"Error fetching data: {e}")
						return jsonify({"error": f"Error fetching data: {e}"}), 500

				elif all([min_latitude, max_latitude, min_longitude, max_longitude]):
					self.app.logger.info(f"Fetching data for {variable}/{subvariable} in range ({min_latitude}, {max_latitude}), ({min_longitude}, {max_longitude})")
					try:
						raw_data = read_h5_file(
							lat_range=(float(min_latitude), float(max_latitude)),
							lon_range=(float(min_longitude), float(max_longitude)),
							address=f"{variable}/{subvariable}"
						)
						data = {key: float(value) if isinstance(value, np.float32) else value for key, value in raw_data.items()}
						return jsonify({"message": "Data fetched successfully", "data": data})
					except Exception as e:
						self.app.logger.error(f"Error fetching data: {e}")
						return jsonify({"error": f"Error fetching data: {e}"}), 500

				else:
					message = f"Please provide either a point (latitude/longitude) or a range (min/max lat/lon)."
					self.app.logger.error(message)
					return jsonify({"error": message}), 400

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

		@self.app.route('/api/generate_report', methods=['POST'])
		@conditional_login_required
		@conditional_verified_required
		def generate_report():
			"""Generate a report based on user-selected area and parameters."""
			self.app.logger.info("Report generation request received")
			
			try:
				data = request.get_json()
				if not data:
					return jsonify({"error": "No data received"}), 400
				
				# Extract and validate bounding box coordinates
				min_latitude = float(data.get('min_latitude'))
				max_latitude = float(data.get('max_latitude'))
				min_longitude = float(data.get('min_longitude'))
				max_longitude = float(data.get('max_longitude'))
				
				if not all([min_latitude, max_latitude, min_longitude, max_longitude]):
					return jsonify({"error": "Bounding box coordinates are required"}), 400
				
				# Log the coordinates for debugging
				self.app.logger.info(f"Coordinates: {min_longitude}, {min_latitude}, {max_longitude}, {max_latitude}")
				
				# Verify coordinate validity
				if not (-90 <= min_latitude <= 90 and -90 <= max_latitude <= 90 and 
						-180 <= min_longitude <= 180 and -180 <= max_longitude <= 180):
					return jsonify({"error": "Invalid coordinate values"}), 400
				
				# Check if we're dealing with polygon coordinates and process them properly
				polygon_coordinates = data.get('polygon_coordinates')
				geometry_type = data.get('geometry_type', 'extent')
				
				if polygon_coordinates:
					try:
						# Try to parse the polygon coordinates if it's a string
						if isinstance(polygon_coordinates, str):
							coords = json.loads(polygon_coordinates)
						else:
							coords = polygon_coordinates
							
						self.app.logger.info(f"Using polygon with {len(coords)} vertices")
						# Store the processed coordinates for later use if needed
						processed_polygon = coords
					except Exception as e:
						self.app.logger.error(f"Error parsing polygon coordinates: {e}")
						return jsonify({"error": f"Invalid polygon format: {e}"}), 400
				
				# Extract report parameters
				report_type = data.get('report_type', 'all')
				start_year = int(data.get('start_year', 2010))
				end_year = int(data.get('end_year', 2020))
				resolution = int(data.get('resolution', 250))
				aggregation = data.get('aggregation', 'monthly')
				include_climate_change = data.get('include_climate_change', False)
				
				# Create output directory for the report
				username = current_user.username
				timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
				output_dir = os.path.join('/data/SWATGenXApp/Users', username, "Reports", timestamp)
				os.makedirs(output_dir, exist_ok=True)
				
				# Create configuration for report generator
				config = {
					'RESOLUTION': resolution,
					'resolution': resolution,
					'start_year': start_year,
					'end_year': end_year,
					'bounding_box': [min_longitude, min_latitude, max_longitude, max_latitude],
					'aggregation': aggregation,
					'include_climate_change': include_climate_change,
					'geometry_type': geometry_type
				}
				
				# Add polygon if available
				if polygon_coordinates:
					config['polygon_coordinates'] = processed_polygon
				
				# Log the configuration
				self.app.logger.info(f"Report configuration: {config}")
				
				# Generate the report in a background thread to prevent blocking
				def generate_report_task():
					try:
						# Import the report generator function
						from AI_agent.report_generator import run_report_generation
						
						reports = run_report_generation(report_type, config, output_dir, parallel=True)
						# Save report metadata to database or file for later retrieval
						report_metadata = {
							'username': username,
							'timestamp': timestamp,
							'report_type': report_type,
							'bounding_box': [min_longitude, min_latitude, max_longitude, max_latitude],
							'output_dir': output_dir,
							'reports': reports,
							'status': 'completed' if reports else 'failed'
						}
						
						# Save metadata to file
						with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
							json.dump(report_metadata, f, indent=2)
						
						self.app.logger.info(f"Report generation completed: {output_dir}")
					except Exception as e:
						self.app.logger.error(f"Error generating report: {e}")
						# Save error information
						error_info = {
							'username': username,
							'timestamp': timestamp,
							'error': str(e),
							'status': 'failed'
						}
						with open(os.path.join(output_dir, 'error.json'), 'w') as f:
							json.dump(error_info, f, indent=2)
				
				# Start the background task
				thread = threading.Thread(target=generate_report_task)
				thread.daemon = True
				thread.start()

					
				return jsonify({
					'status': 'success',
					'message': 'Report generation started',
					'report_id': timestamp,
					'output_dir': output_dir
				})
			
		
			except Exception as e:
				self.app.logger.error(f"Error initiating report generation: {e}")
				return jsonify({"error": f"Failed to start report generation: {str(e)}"}), 500

		@self.app.route('/api/get_reports', methods=['GET'])
		@conditional_login_required
		@conditional_verified_required
		def get_reports():
			"""Get a list of reports generated by the user."""
			username = current_user.username
			reports_dir = os.path.join('/data/SWATGenXApp/Users', username, "Reports")
			
			if not os.path.exists(reports_dir):
				return jsonify({"reports": []})
			
			reports = []
			for report_id in os.listdir(reports_dir):
				report_path = os.path.join(reports_dir, report_id)
				if os.path.isdir(report_path):
					metadata_path = os.path.join(report_path, 'metadata.json')
					error_path = os.path.join(report_path, 'error.json')
					
					if os.path.exists(metadata_path):
						with open(metadata_path, 'r') as f:
							metadata = json.load(f)
						reports.append(metadata)
					elif os.path.exists(error_path):
						with open(error_path, 'r') as f:
							error_info = json.load(f)
						reports.append(error_info)
					else:
						# Report is still processing or was interrupted
						reports.append({
							'report_id': report_id,
							'status': 'processing',
							'timestamp': report_id
						})
			
			# Sort by timestamp (newest first)
			reports.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
			return jsonify({"reports": reports})

		@self.app.route('/api/reports/<report_id>/download', methods=['GET'])
		@conditional_login_required
		@conditional_verified_required
		def download_report(report_id):
			"""Download a generated report."""
			if not report_id:
				self.app.logger.error("Missing report ID in download request")
				return jsonify({'error': 'Report ID is required'}), 400
				
			username = current_user.username
			report_dir = os.path.join('/data/SWATGenXApp/Users', username, "Reports", report_id)
			
			self.app.logger.info(f"Attempting to download report: {report_dir}")
			
			if not os.path.exists(report_dir) or not os.path.isdir(report_dir):
				self.app.logger.error(f"Report directory not found: {report_dir}")
				return jsonify({'error': f'Report with ID {report_id} not found'}), 404
			
			# Check if the metadata file exists
			metadata_path = os.path.join(report_dir, 'metadata.json')
			if not os.path.exists(metadata_path):
				self.app.logger.error(f"Report metadata not found: {metadata_path}")
				return jsonify({'error': f'Report metadata for ID {report_id} not found'}), 404
			
			try:
				# Read metadata to get report paths
				with open(metadata_path, 'r') as f:
					metadata = json.load(f)
				
				# Create a ZIP file with all report files
				zip_filename = f"{report_id}_reports.zip"
				zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
				
				# List of files included in the ZIP
				included_files = []
				
				# Create a new ZIP file with all the reports
				with zipfile.ZipFile(zip_path, 'w') as report_zip:
					# Add metadata file
					report_zip.write(metadata_path, os.path.basename(metadata_path))
					included_files.append(metadata_path)
					
					# Add all report files
					for report_path in metadata.get('reports', []):
						if os.path.exists(report_path):
							# Add file to ZIP with relative path from report directory
							arcname = os.path.relpath(report_path, report_dir)
							report_zip.write(report_path, arcname)
							included_files.append(report_path)
				
				self.app.logger.info(f"Generated report ZIP file: {zip_path} with {len(included_files)} files")
				self.app.logger.debug(f"Files included: {included_files}")
				
				return send_file(
					zip_path, 
					mimetype='application/zip', 
					download_name=zip_filename, 
					as_attachment=True
				)
			
			except Exception as e:
				self.app.logger.error(f"Error creating report ZIP: {e}")
				return jsonify({
					'error': f'Failed to create report package: {str(e)}',
					'report_id': report_id
				}), 500

		@self.app.route('/api/reports/<report_id>/view', methods=['GET'])
		@self.app.route('/api/reports/<report_id>/view/<path:subpath>', methods=['GET'])
		@conditional_login_required
		@conditional_verified_required
		def view_report(report_id, subpath=None):
			"""View a specific report file or the default report page."""
			# Add extensive debugging
			self.app.logger.info(f"VIEW REPORT CALLED - ID: {report_id}, Subpath: {subpath}")
			self.app.logger.info(f"Request URL: {request.url}")
			self.app.logger.info(f"Request Path: {request.path}")
			
			if not report_id:
				self.app.logger.error("Missing report ID in view request")
				return jsonify({'error': 'Report ID is required'}), 400
					
			username = current_user.username
			report_dir = os.path.join('/data/SWATGenXApp/Users', username, "Reports", report_id)
			
			self.app.logger.info(f"Report directory: {report_dir}")
			
			if not os.path.exists(report_dir) or not os.path.isdir(report_dir):
				self.app.logger.error(f"Report directory not found: {report_dir}")
				return jsonify({'error': f'Report with ID {report_id} not found'}), 404
			
				# Debug listing of directory contents to help troubleshoot
			try:
				self.app.logger.info(f"Listing report directory contents:")
				for root, dirs, files in os.walk(report_dir):
					rel_root = os.path.relpath(root, report_dir)
					self.app.logger.info(f"Directory: {rel_root}")
					for file in files:
						self.app.logger.info(f"  File: {os.path.join(rel_root, file)}")
			except Exception as e:
				self.app.logger.error(f"Error listing directory contents: {e}")
			
			# If a specific subpath is requested, serve that file directly
			if subpath:
				# Improve path normalization to handle all cases
				# Remove any leading slashes and normalize path
				clean_subpath = subpath.lstrip('/')
				clean_subpath = os.path.normpath(clean_subpath)
				file_path = os.path.join(report_dir, clean_subpath)
				
				self.app.logger.info(f"Requested subpath: {subpath}")
				self.app.logger.info(f"Normalized subpath: {clean_subpath}")
				self.app.logger.info(f"Full file path: {file_path}")
				
				# Verify the path is still within the report directory to prevent directory traversal
				if not os.path.abspath(file_path).startswith(os.path.abspath(report_dir)):
					self.app.logger.error(f"Attempted directory traversal: {file_path}")
					return jsonify({'error': 'Access denied'}), 403
				
				# Special handling for HTML files - we need to check if they exist
				# and properly set MIME type for browser rendering
				if os.path.exists(file_path) and os.path.isfile(file_path):
					self.app.logger.info(f"File exists, serving: {file_path}")
					file_dir, file_name = os.path.split(file_path)
					
					# Determine content type based on file extension
					_, ext = os.path.splitext(file_name)
					content_type = None
					if ext.lower() == '.html':
						content_type = 'text/html'
					elif ext.lower() == '.css':
						content_type = 'text/css'
					elif ext.lower() == '.js':
						content_type = 'application/javascript'
					elif ext.lower() == '.png':
						content_type = 'image/png'
					elif ext.lower() == '.jpg' or ext.lower() == '.jpeg':
						content_type = 'image/jpeg'
					
					if content_type:
						self.app.logger.info(f"Serving with content type: {content_type}")
						return send_from_directory(file_dir, file_name, mimetype=content_type)
					else:
						self.app.logger.info(f"Serving with auto-detected content type")
						return send_from_directory(file_dir, file_name)
				else:
					self.app.logger.error(f"Subpath file not found: {file_path}")
					return jsonify({'error': 'File not found'}), 404
			
			# No subpath specified - find an appropriate default file to display
			# First check for index.html in the root directory
			index_path = os.path.join(report_dir, 'index.html')
			if os.path.exists(index_path):
				self.app.logger.info(f"Serving main index.html: {index_path}")
				return send_file(index_path, mimetype='text/html')
			
			# ... existing fallback code ...
			
		@self.app.route('/api/reports/<report_id>/status', methods=['GET'])
		@conditional_login_required
		@conditional_verified_required
		def get_report_status(report_id):
			"""Check the status of a report."""
			if not report_id:
				self.app.logger.error("Missing report ID in status check request")
				return jsonify({'error': 'Report ID is required'}), 400
				
			username = current_user.username
			report_dir = os.path.join('/data/SWATGenXApp/Users', username, "Reports", report_id)
			
			self.app.logger.info(f"Checking status of report: {report_dir}")
			
			# Check if report directory exists
			if not os.path.exists(report_dir) or not os.path.isdir(report_dir):
				self.app.logger.error(f"Report directory not found: {report_dir}")
				return jsonify({
					'status': 'not_found',
					'error': f'Report with ID {report_id} not found'
				}), 404
			
			# Look for metadata or error files
			metadata_path = os.path.join(report_dir, 'metadata.json')
			error_path = os.path.join(report_dir, 'error.json')
			
			# Return appropriate response based on which files exist
			if os.path.exists(metadata_path):
				try:
					with open(metadata_path, 'r') as f:
						metadata = json.load(f)
					
					# Add the report ID if not included
					if 'report_id' not in metadata:
						metadata['report_id'] = report_id
						
					# Make sure status is included
					if 'status' not in metadata:
						metadata['status'] = 'completed'
						
					return jsonify(metadata)
				except Exception as e:
					self.app.logger.error(f"Error reading report metadata: {e}")
					return jsonify({
						'status': 'error',
						'report_id': report_id,
						'error': f'Error reading report metadata: {str(e)}'
					}), 500
			
			elif os.path.exists(error_path):
				try:
					with open(error_path, 'r') as f:
						error_info = json.load(f)
					
					# Add the report ID if not included
					if 'report_id' not in error_info:
						error_info['report_id'] = report_id
						
					# Make sure status is included
					if 'status' not in error_info:
						error_info['status'] = 'failed'
						
					return jsonify(error_info)
				except Exception as e:
					self.app.logger.error(f"Error reading report error info: {e}")
					return jsonify({
						'status': 'error',
						'report_id': report_id,
						'error': f'Error reading report error info: {str(e)}'
					}), 500
			
			else:
				# Report is still processing
				return jsonify({
					'status': 'processing',
					'report_id': report_id,
					'message': 'Report is still being generated'
				})
		@self.app.route('/api/reports/<report_id>', defaults={'path': ''})
		@self.app.route('/api/reports/<report_id>/<path:path>')
		@conditional_login_required
		@conditional_verified_required
		def report_redirect(report_id, path=''):
			"""
			Redirect incorrect report URLs to the correct ones with the '/view/' segment.
			This handles cases when a link in a report points to an incorrect URL.
			"""
			self.app.logger.info(f"Redirect handler called for: {report_id}/{path}")
			
			# If this is not already a view URL, redirect to the proper view URL
			if not path.startswith('view/') and path != 'view':
				redirect_url = f"/api/reports/{report_id}/view/{path}"
				self.app.logger.info(f"Redirecting to: {redirect_url}")
				return redirect(redirect_url)
			
			return jsonify({'error': 'Invalid report URL'}), 404
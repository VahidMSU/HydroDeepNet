from flask import render_template, redirect, url_for, request, flash, jsonify, current_app, session
from flask_login import login_user, logout_user, login_required
from app.models import User, ContactMessage
from app.forms import RegistrationForm, LoginForm, ContactForm  # Import the forms
from app.extensions import db
import logging
from functools import partial
from multiprocessing import Process
import sys
from app.utils import find_station
sys.path.append(r'/data/SWATGenXApp/codes/SWATGenX')
from SWATGenX.integrate_streamflow_data import integrate_streamflow_data
from app.utils import get_huc12_geometries, single_model_creation
from app.forms import ModelSettingsForm
from flask import render_template
import os

def init_routes(app):
	@app.route('/')
	def index():
		print("Index route called")	
		logging.info("Index route called. Redirecting to /home.")
		return redirect(url_for('home'))
	@app.route('/dashboard')
	#@login_required
	def dashboard():
		return render_template('dashboard.html')
			
	@app.route('/get_options', methods=['GET'])
	#@login_required
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
	#@login_required
	def visualizations():
		print("Visualizations route called")
		name = request.args.get('NAME', default=None)
		ver = request.args.get('ver', default=None)
		variable = request.args.get('variable', default=None)

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

		video_file = os.path.join(video_path, f"{ver}_{variable}_animation.gif")
		static_plot_dir = os.path.join(static_plots_path, variable.capitalize())

		gif_url = f"/static/SWATplus_by_VPUID/0000/huc12/{name}/figures_SWAT_gwflow_MODEL/verifications_videos/{ver}_{variable}_animation.gif"
		static_plot_files = []

		if os.path.exists(static_plot_dir):
			static_plot_files = [
				f"/static/SWATplus_by_VPUID/0000/huc12/{name}/figures_SWAT_gwflow_MODEL/watershed_static_plots/{variable.capitalize()}/{file}"
				for file in os.listdir(static_plot_dir) if file.endswith('.png')
			]

		if not os.path.exists(video_file) and not static_plot_files:
			if request.headers.get("X-Requested-With") == "XMLHttpRequest":
				return jsonify({"error": f"No visualizations found for NAME: {name}, Version: {ver}, Variable: {variable}."}), 404
			else:
				return render_template('visualizations.html', error=f"No visualizations found for NAME: {name}, Version: {ver}, Variable: {variable}.")

		print(f"Static plot files: {static_plot_files}")
		print(f"GIF URL: {gif_url}")

		# For AJAX requests, return JSON
		if request.headers.get("X-Requested-With") == "XMLHttpRequest":
			return jsonify({
				"gif_file": gif_url if os.path.exists(video_file) else None,
				"png_files": static_plot_files
			})

		# For direct access via browser, render the template
		return render_template(
			'visualizations.html',
			name=name,
			ver=ver,
			variable=variable,
			gif_file=gif_url,
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
			user = User.query.filter_by(username=username).first()
			if user and user.password == password:
				login_user(user)
				session.permanent = True
				logging.info(f"User {username} successfully logged in.")
				return redirect(url_for('home'))
			else:
				logging.warning(f"Invalid login attempt for username: {username}")
				flash('Invalid username or password')
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
	#@login_required
	def logout():
		print("Logout route called")
		logging.info("Logout route called")
		logout_user()
		session.clear()  # Clear the session
		return redirect(url_for('login'))

	@app.route('/home')
	#@login_required
	def home():
		logging.info("Home route called, user is authenticated.")
		return render_template('home.html')

	@app.route('/model-settings', methods=['GET', 'POST'])
	#@login_required
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
	#@login_required
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
	#@login_required
	def about():
		logging.info("About route called")
		return render_template('about.html')


	@app.route('/model-confirmation')
	#@login_required
	def model_confirmation():
		logging.info("Model Confirmation route called")
		return render_template('model_confirmation.html')


	@app.route('/contact', methods=['GET', 'POST'])
	#@login_required
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
	#@login_required
	def infrastructure():
		logging.info("Infrastructure route called")
		return render_template('infrastructure.html')

	@app.route('/ftp-access')
	#@login_required
	def ftp_access():
		return render_template('ftp_access.html')

	@app.route('/deeplearning_models')
	#@login_required
	def deeplearning_models():
		return render_template('DeepLearning.html')
	@app.route('/michigan')
	#@login_required
	def michigan():
		logging.info("Michigan route called")
		return render_template('michigan.html')


	@app.route('/search_site', methods=['GET', 'POST'])
	#@login_required
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

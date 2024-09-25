from flask import render_template, redirect, url_for, request, flash, jsonify, current_app, session
from flask_login import login_user, logout_user, login_required
from app.models import User, ContactMessage
from app.forms import RegistrationForm, LoginForm, ContactForm  # Import the forms
from app.extensions import db
import logging
from functools import partial
from multiprocessing import Process
import sys
sys.path.append(r'/data/MyDataBase/SWATGenXAppData/codes/NHDPlus_SWAT')
sys.path.append(r'/data/MyDataBase/SWATGenXAppData/codes/ModelProcessing')
from NHDPlus_SWAT.SWATGenXCommand import SWATGenXCommand
from NHDPlus_SWAT.integrate_streamflow_data import integrate_streamflow_data
from ModelProcessing.core import process_SCV_SWATGenXModel
from app.utils import get_huc12_geometries, single_model_creation
from app.forms import ModelSettingsForm

def init_routes(app):
	@app.route('/')
	def index():
		return redirect(url_for('home'))

	@app.route('/login', methods=['GET', 'POST'])
	def login():
		logging.info("Login route called")
		form = LoginForm()
		if form.validate_on_submit():
			username = form.username.data
			password = form.password.data
			user = User.query.filter_by(username=username).first()
			if user and user.password == password:
				login_user(user)
				session.permanent = True  # Set session as permanent
				return redirect(url_for('home'))
			else:
				flash('Invalid username or password')
		return render_template('login.html', form=form)

	@app.route('/signup', methods=['GET', 'POST'])
	def signup():
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
	@login_required
	def logout():
		logging.info("Logout route called")
		logout_user()
		session.clear()  # Clear the session
		return redirect(url_for('login'))

	@app.route('/home')
	@login_required
	def home():
		logging.info("Home route called")
		return render_template('home.html')

	@app.route('/model-settings', methods=['GET', 'POST'])
	@login_required
	def model_settings():
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

		station_data = integrate_streamflow_data(current_app.config['USGS_PATH'])
		station_list = station_data.SiteNumber.unique()
		return render_template('model_settings.html', form=form, output=output, station_list=station_list)

	@app.route('/get_station_characteristics', methods=['GET'])
	@login_required
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
	@login_required
	def about():
		logging.info("About route called")
		return render_template('about.html')

	@app.route('/contact', methods=['GET', 'POST'])
	@login_required
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
	@login_required
	def infrastructure():
		logging.info("Infrastructure route called")
		return render_template('infrastructure.html')

	@app.route('/ftp-access')
	@login_required
	def ftp_access():
		return render_template('ftp_access.html')

	@app.route('/deeplearning_models')
	@login_required
	def deeplearning_models():
		return render_template('DeepLearning.html')
	@app.route('/michigan')
	@login_required
	def michigan():
		logging.info("Michigan route called")
		return render_template('michigan.html')

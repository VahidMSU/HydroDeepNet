#/data/SWATGenXApp/codes/web_application/app/forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Email, EqualTo, ValidationError
from app.models import User
from wtforms import SelectField, BooleanField, IntegerField
from wtforms.validators import NumberRange
from wtforms.validators import Optional


from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class VerificationForm(FlaskForm):
    verification_code = StringField('Verification Code', validators=[DataRequired()])
    submit = SubmitField('Verify')

import re
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, ValidationError
from flask_wtf import FlaskForm
class SignUpForm(FlaskForm):
    """
    SignUp orm is a Flask-WTF form used for user SignUp 

    This form collects the username, email, password, and confirmation of the password from the user.
    It includes validation methods to ensure that the username and email are unique and that the password meets the complexity requirements.

    Attributes:
        username (StringField): The username of the user.
        email (StringField): The email address of the user.
        password (PasswordField): The password for the user account.
        confirm_password (PasswordField): A field to confirm the user's password.
        submit (SubmitField): A button to submit the SignUp form.

    Methods:
        validate_username(username): Validates that the username is unique.
        validate_password(password): Validates that the password meets the complexity requirements.
        validate_email(email): Validates that the email is unique.
    """

    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField(
        'Confirm Password', 
        validators=[DataRequired(), EqualTo('password', message="Passwords must match.")]
    )
    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        """Validates that the username is unique."""
        if User.query.filter_by(username=username.data).first():
            raise ValidationError('That username is taken. Please choose a different one.')

    def validate_password(self, password):
        """
        Validates that the password meets the complexity requirements:
        - At least 8 characters long
        - Contains at least one uppercase letter
        - Contains at least one lowercase letter
        - Contains at least one number
        - Contains at least one special character
        """
        password_data = password.data
        if len(password_data) < 8:
            raise ValidationError('Password must be at least 8 characters long.')
        if not re.search(r'[A-Z]', password_data):
            raise ValidationError('Password must contain at least one uppercase letter.')
        if not re.search(r'[a-z]', password_data):
            raise ValidationError('Password must contain at least one lowercase letter.')
        if not re.search(r'\d', password_data):
            raise ValidationError('Password must contain at least one number.')
        if not re.search(r'[@#$^&*()_+={}\[\]|\\:;"\'<>,.?/~`-]', password_data):
            raise ValidationError('Password must contain at least one special character.')

    def validate_email(self, email):
        """Validates that the email is unique."""
        if User.query.filter_by(email=email.data).first():
            raise ValidationError('That email is already in use. Please choose a different one.')


class LoginForm(FlaskForm):

    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Login')

class HydroGeoDatasetForm(FlaskForm):
    # Fields for single point selection
    latitude = StringField('Latitude', validators=[Optional()])
    longitude = StringField('Longitude', validators=[Optional()])
    
    # Fields for range selection
    min_latitude = StringField('Min Latitude', validators=[Optional()])
    max_latitude = StringField('Max Latitude', validators=[Optional()])
    min_longitude = StringField('Min Longitude', validators=[Optional()])
    max_longitude = StringField('Max Longitude', validators=[Optional()])
    
    # Variable and subvariable selection
    variable = SelectField(
        'Variable',
        validators=[DataRequired()],
        choices=[]  # Dynamically populated in the view
    )
    subvariable = SelectField(
        'Subvariable',
        validators=[DataRequired()],
        choices=[]  # Dynamically populated in the view
    )
    submit = SubmitField('Get Variable Value')

class ContactForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    message = TextAreaField('Message', validators=[DataRequired()])
    submit = SubmitField('Submit')

class ModelSettingsForm(FlaskForm):
    user_input = StringField('USGS Station Number', validators=[DataRequired()])
    search_input = StringField('Search Site Name')  # New search field
    ls_resolution = SelectField('Landuse/Soil Resolution', choices=[('30', '30'), ('100', '100'), ('250', '250'), ('500', '500'), ('1000', '1000'), ('2000', '2000')], validators=[DataRequired()])
    dem_resolution = SelectField('DEM Resolution', choices=[('30', '30'), ('100', '100'), ('250', '250')], validators=[DataRequired()])
    calibration_flag = BooleanField('Calibration')
    max_iterations = IntegerField('Max Iterations', validators=[NumberRange(min=1, max=50)], default=25)
    cal_pool_size = IntegerField('Population Size', validators=[NumberRange(min=1, max=100)], default=50)
    sensitivity_flag = BooleanField('Sensitivity')
    sen_pool_size = IntegerField('Pool Size', validators=[NumberRange(min=1, max=500)], default=180)
    sen_total_evaluations = IntegerField('Total Evaluations', validators=[NumberRange(min=1, max=5000)], default=1000)
    num_levels = IntegerField('Number of Levels', validators=[NumberRange(min=1, max=20)], default=10)
    validation_flag = BooleanField('Validation')
    verification_samples = IntegerField('Samples', validators=[NumberRange(min=1, max=50)], default=25)
    submit = SubmitField('Run')
    search_submit = SubmitField('Search')  # New submit button for search

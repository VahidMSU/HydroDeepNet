from flask import (Blueprint, request, jsonify, current_app, session, redirect)
from flask_login import (login_user, logout_user, current_user)
from app.models import User
from app.extensions import db, csrf
from app.utils import send_verification_email
from app.sftp_manager import create_sftp_user

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/api/login', methods=['POST'])
@csrf.exempt
def api_login():
    current_app.logger.info(f"{request.method} request received for /api/login")
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
            # User exists and password is correct, but email is not verified
            current_app.logger.info(f"User {username} attempted login but is not verified. Redirecting to verify page.")
            return jsonify({
                "success": True, 
                "verified": False, 
                "email": user.email,
                "message": "Your email has not been verified. Please verify your email.",
                "redirect": "/verify"
            }), 200

        login_user(user)
        session.permanent = True
        return jsonify({"success": True, "verified": True, "token": "someJWT"}), 200

    return jsonify({"error": "Invalid username or password"}), 401

@auth_bp.route('/api/logout', methods=['POST'])
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

@auth_bp.route('/api/signup', methods=['POST'])
def signup():
    current_app.logger.info("Sign Up route called via API")
    data = request.get_json()
    if not data:
        current_app.logger.error("No JSON data received in signup request")
        return jsonify({"status": "error", "message": "Invalid request format", "details": "No JSON data received"}), 400
    
    current_app.logger.debug(f"Signup data received: {data}")
    
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
        current_app.logger.info(f"User `{new_user.username}` created in unverified state. Verification email sent.")
        return jsonify({"status": "success", "message": "Check your email for verification code.", "redirect": "/verify"})
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error creating user: {e}")
        return jsonify({"status": "error", "message": "An error occurred while creating the account."}), 500

@auth_bp.route('/api/verify', methods=['POST'])
def verify():
    current_app.logger.info("Verification attempt received.")

    data = request.get_json()
    email = data.get('email', '').strip()
    code_entered = data.get('verification_code', '').strip()

    if not email or not code_entered:
        return jsonify({"status": "error", "message": "Email and verification code are required."}), 400

    user = User.query.filter_by(email=email).first()

    if not user:
        current_app.logger.warning(f"Verification failed: User with email `{email}` not found.")
        return jsonify({"status": "error", "message": "User not found."}), 404

    if user.is_verified:
        current_app.logger.warning(f"Verification failed: User `{user.username}` is already verified.")
        return jsonify({"status": "error", "message": "User is already verified."}), 400

    if user.verification_code == code_entered:
        user.is_verified = True
        user.verification_code = None
        db.session.commit()

        current_app.logger.info(f"User `{user.username}` verified successfully. Creating SFTP account...")

        sftp_result = create_sftp_user(user.username)
        if sftp_result.get("status") != "success":
            current_app.logger.error(f"SFTP creation failed for {user.username}: {sftp_result.get('error')}")
            return jsonify({"status": "error", "message": "SFTP account creation failed. Contact support."}), 500

        return jsonify({
            "status": "success",
            "message": "Verification successful. Please log in.",
            "redirect": "/login"
        })

    current_app.logger.warning(f"Verification failed: Invalid code for user `{user.username}`.")
    return jsonify({"status": "error", "message": "Invalid verification code."}), 400

@auth_bp.route('/sign_up', methods=['GET', 'POST'])
def sign_up_redirect():
    """Redirect old sign_up route to the Single Page App"""
    current_app.logger.info("Redirecting /sign_up to frontend")
    if request.method == 'POST':
        # If it's a POST request, it might be trying to submit data directly
        current_app.logger.warning("POST to /sign_up received - redirecting to /api/signup")
        return redirect('/api/signup')
    return redirect('/#/signup')  # Redirect to the React router path

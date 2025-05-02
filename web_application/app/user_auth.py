from flask import Blueprint, jsonify, current_app, url_for, request, send_from_directory, send_file
from flask import redirect, session
from flask_login import current_user, login_user, logout_user, login_required
from app.models import ContactMessage, User
from app.extensions import db, csrf
from app.decorators import conditional_login_required, conditional_verified_required
from app.comm_utils import send_verification_email
from app.sftp_manager import create_sftp_user, delete_sftp_user
from app.oauth_utils import get_google_auth_url, process_google_callback, login_oauth_user
import os
import tempfile
from werkzeug.utils import secure_filename
import shutil

# Combined user and auth blueprint
user_auth_bp = Blueprint('user_auth', __name__)

# Authentication routes (from auth.py)
@user_auth_bp.route('/api/login', methods=['POST'])
@csrf.exempt
def api_login():
    """API Login route."""
    current_app.logger.info("Login attempt received")
    try:
        data = request.get_json()
        username = data.get('username', '')

        # Don't log or store the password in variables
        if not username or 'password' not in data:
            current_app.logger.warning("Login failed: Missing username or password")
            return jsonify({"status": "error", "message": "Username and password are required"}), 400

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(data.get('password', '')):
            # Clear password from memory as soon as possible
            data['password'] = None

            login_user(user, remember=data.get('remember_me', False))
            current_app.logger.info(f"User {username} logged in successfully")

            return jsonify({
                "status": "success",
                "message": "Login successful",
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "is_verified": user.is_verified
                }
            })
        else:
            current_app.logger.warning(f"Login failed for user {username}: Invalid credentials")
            return jsonify({"status": "error", "message": "Invalid username or password"}), 401

    except Exception as e:
        current_app.logger.error(f"Login error: {e}")
        return jsonify({"status": "error", "message": "An error occurred during login"}), 500

@user_auth_bp.route('/api/logout', methods=['POST'])
def logout():
    """Logout route."""
    current_app.logger.info(f"Logout request for user: {current_user.username if not current_user.is_anonymous else 'anonymous'}")
    logout_user()
    return jsonify({"status": "success", "message": "Logged out successfully"})

@user_auth_bp.route('/api/signup', methods=['POST'])
@csrf.exempt
def signup():
    """Signup route."""
    current_app.logger.info("Signup attempt received")
    try:
        data = request.get_json()
        if not data:
            current_app.logger.warning("Signup failed: no data provided")
            return jsonify({"status": "error", "message": "No data provided"}), 400

        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password')

        # Don't log or store password in variables
        if not username or not email or not password:
            current_app.logger.warning("Signup failed: missing required fields")
            return jsonify({"status": "error", "message": "Username, email and password are required"}), 400

        # Validate username format - only allow letters and numbers
        import re
        if not re.match(r'^[a-zA-Z0-9]+$', username):
            current_app.logger.warning(f"Signup failed: invalid username format: {username}")
            return jsonify({"status": "error", "message": "Username must contain only letters and numbers"}), 400

        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            current_app.logger.warning(f"Signup failed: username '{username}' already exists")
            return jsonify({"status": "error", "message": f"Username '{username}' is already taken"}), 400

        if User.query.filter_by(email=email).first():
            current_app.logger.warning(f"Signup failed: email '{email}' already exists")
            return jsonify({"status": "error", "message": f"Email '{email}' is already registered"}), 400

        # Create new user directly with password from request data
        user = User(username=username, email=email)
        user.set_password(password)

        # Clear password from memory immediately
        password = None
        data['password'] = None

        # Generate verification token
        verification_token = user.get_verification_token()

        db.session.add(user)
        db.session.commit()
        current_app.logger.info(f"User '{username}' created successfully")

        # Send verification email
        send_verification_email(user, verification_token)
        current_app.logger.info(f"Verification email sent to {email}")

        # Create user directory structure
        user_dir = os.path.join(current_app.config['USER_PATH'], username)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir, exist_ok=True)
            os.makedirs(os.path.join(user_dir, 'SWATplus_by_VPUID'), exist_ok=True)
            os.makedirs(os.path.join(user_dir, 'Reports'), exist_ok=True)
            current_app.logger.info(f"Created directory structure for user '{username}'")

        # Create SFTP account automatically - but pass None for password, as create_sftp_user
        # should instead get password via more secure means like directly from the database
        sftp_result = create_sftp_user(username)
        if sftp_result.get('success'):
            current_app.logger.info(f"SFTP account created for user '{username}'")
        else:
            current_app.logger.warning(f"Failed to create SFTP account for user '{username}': {sftp_result.get('message')}")

        return jsonify({
            "status": "success",
            "message": "User registered successfully. Please check your email to verify your account."
        })

    except Exception as e:
        current_app.logger.error(f"Signup error: {e}")
        db.session.rollback()  # Roll back the transaction in case of error
        return jsonify({"status": "error", "message": "An error occurred during registration"}), 500

@user_auth_bp.route('/api/verify', methods=['POST'])
def verify():
    """Email verification route."""
    current_app.logger.info("Verification attempt received")
    try:
        data = request.get_json()
        token = data.get('token', '')
        verification_code = data.get('verification_code', '')
        email = data.get('email', '')

        if not token and not verification_code:
            current_app.logger.warning("Verification failed: no token or code provided")
            return jsonify({"status": "error", "message": "Verification token or code is required"}), 400

        # First try to find a user by verification code if provided
        user = None
        if verification_code:
            user = User.query.filter_by(verification_code=verification_code).first()
            if email:
                # If email is provided, make sure it matches
                if user and user.email != email:
                    user = None

        # Fall back to token verification if no user found by code
        if not user and token:
            user = User.verify_token(token)

        if not user:
            current_app.logger.warning("Verification failed: invalid or expired token/code")
            return jsonify({"status": "error", "message": "Invalid or expired verification token/code"}), 400

        user.is_verified = True
        user.verification_code = None  # Clear the verification code after use
        db.session.commit()
        current_app.logger.info(f"User '{user.username}' verified successfully")

        return jsonify({
            "status": "success",
            "message": "Email verified successfully. You can now log in.",
            "username": user.username
        })

    except Exception as e:
        current_app.logger.error(f"Verification error: {e}")
        return jsonify({"status": "error", "message": "An error occurred during verification"}), 500

@user_auth_bp.route('/sign_up', methods=['GET', 'POST'])
def sign_up_redirect():
    """Redirect old sign_up route to the Single Page App"""
    return redirect('/signup')

@user_auth_bp.route('/api/validate-session', methods=['GET'])
@conditional_login_required
def validate_session():
    """Validate user session and return user details."""
    current_app.logger.info(f"Session validation for user: {current_user.username}")

    return jsonify({
        "status": "success",
        "message": "Session is valid",
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "is_verified": current_user.is_verified
        }
    })

# User management routes (from user.py)
@user_auth_bp.route('/api/index')
@conditional_login_required
@conditional_verified_required
def index():
    return jsonify({"status": "success", "message": "Welcome to the API!"})

@user_auth_bp.route('/user_dashboard', methods=['GET'])
@conditional_login_required
@conditional_verified_required
def user_dashboard():
    """User dashboard route."""
    current_app.logger.info(f"User Dashboard accessed by `{current_user.username}`.")
    return jsonify({"title": "Dashboard", "message": "Your user dashboard."})

@user_auth_bp.route('/home', methods=['GET'])
@conditional_login_required
@conditional_verified_required
def home():
    """User's home page."""
    current_app.logger.info(f"Home route accessed by user: {current_user.username}.")
    return jsonify({"title": "Home", "message": "Welcome to the app!"})

@user_auth_bp.route('/about')
@conditional_login_required
@conditional_verified_required
def about():
    current_app.logger.info("About route called")
    return jsonify({"title": "About", "message": "about page"})

@user_auth_bp.route('/privacy')
def privacy():
    current_app.logger.info("Privacy route called")
    return jsonify({"title": "Privacy", "message": "privacy page"})

@user_auth_bp.route('/terms')
def terms():
    current_app.logger.info("Terms route called")
    return jsonify({"title": "Terms", "message": "terms page"})

@user_auth_bp.route('/contact', methods=['POST'])
@conditional_login_required
@conditional_verified_required
def contact():
    current_app.logger.info("Contact route called")

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
        current_app.logger.info(f"Message from {name} added to the database")
        return jsonify({"status": "success", "message": "Your message has been sent successfully!"})
    except Exception as e:
        current_app.logger.error(f"Error adding message to the database: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": "An error occurred while sending the message."}), 500

@user_auth_bp.route('/api/user_files', methods=['GET'])
@conditional_login_required
def api_user_files():
    """
    Lists directories and files for the logged-in user.
    Supports navigation within subdirectories and provides download links.
    """
    current_app.logger.info("API User Files route called")
    base_user_dir = os.path.join(current_app.config['USER_PATH'], current_user.username, "SWATplus_by_VPUID")
    subdir = request.args.get('subdir', '')  # Get subdirectory from query params (default: root)
    target_dir = os.path.join(base_user_dir, subdir)
    current_app.logger.info(f"Listing contents for {current_user.username} in: {target_dir}")
    # Security check: Ensure the requested path stays within the user's directory
    if not target_dir.startswith(base_user_dir) or not os.path.exists(target_dir):
        return jsonify({'error': 'Unauthorized or invalid path'}), 403

    current_app.logger.info(f"Listing contents for {current_user.username} in: {target_dir}")

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
            rel_path = os.path.join(subdir, safe_item).lstrip('/')

            if os.path.isdir(item_path):
                # Generate both path-based and direct download URLs
                download_url = url_for('user_auth.download_directory', dirpath=rel_path)
                current_app.logger.info(f"Generated directory download URL: {download_url}")

                contents['directories'].append({
                    'name': safe_item,
                    'path': rel_path,
                    'download_zip_url': download_url
                })
            elif os.path.isfile(item_path):
                download_url = url_for('user_auth.download_user_file', filename=rel_path)
                current_app.logger.info(f"Generated file download URL: {download_url}")
                contents['files'].append({
                    'name': safe_item,
                    'download_url': download_url
                })

    return jsonify(contents)

@user_auth_bp.route('/download/<path:filename>', methods=['GET'])
@conditional_login_required
def download_user_file(filename):
    current_app.logger.info(f"download_user_file called with filename: {filename}")  # Log the filename

    user_dir = os.path.join(current_app.config['USER_PATH'], current_user.username, "SWATplus_by_VPUID")
    filename = filename.lstrip("/")
    full_path = os.path.join(user_dir, filename)

    current_app.logger.info(f"Constructed file path: {full_path}")  # Log the constructed path

    if not os.path.exists(full_path):
        current_app.logger.error(f"File not found: {full_path}")  # Log if the file does not exist
        return jsonify({'error': 'File not found or access denied'}), 404

    current_app.logger.info(f"Serving file for download: {full_path}")
    directory, file = os.path.split(full_path)
    return send_from_directory(directory, file, as_attachment=True)

@user_auth_bp.route('/download_directory/<path:dirpath>', methods=['GET'])
@conditional_login_required
def download_directory(dirpath):
    """
    Download a directory as a ZIP file.
    """
    current_app.logger.info(f"download_directory called with dirpath: {dirpath}")

    user_dir = os.path.join(current_app.config['USER_PATH'], current_user.username, "SWATplus_by_VPUID")
    dirpath = dirpath.lstrip("/")
    full_dir_path = os.path.join(user_dir, dirpath)

    current_app.logger.info(f"Constructed directory path: {full_dir_path}")

    if not full_dir_path.startswith(user_dir) or not os.path.isdir(full_dir_path):
        current_app.logger.error(f"Directory not found or access denied: {full_dir_path}")
        return jsonify({'error': 'Directory not found or access denied'}), 404

    try:
        # Create a temporary file for the zip
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            zip_path = tmp_zip.name

        # Create the zip archive (remove .zip extension for make_archive)
        base_zip_path = zip_path[:-4]
        current_app.logger.info(f"Creating ZIP archive at: {base_zip_path}")

        shutil.make_archive(base_zip_path, 'zip', full_dir_path)

        # Recreate the full path with .zip extension
        final_zip_path = f"{base_zip_path}.zip"

        if not os.path.exists(final_zip_path):
            current_app.logger.error(f"Failed to create ZIP file at: {final_zip_path}")
            return jsonify({'error': 'Failed to create ZIP file'}), 500

        # Use the last directory name for the download filename
        zip_file_name = f"{os.path.basename(dirpath)}.zip"
        current_app.logger.info(f"Serving ZIP file for download: {final_zip_path} as {zip_file_name}")

        return send_file(
            final_zip_path,
            as_attachment=True,
            download_name=zip_file_name,
            mimetype='application/zip'
        )

    except Exception as e:
        current_app.logger.error(f"Error creating ZIP file: {str(e)}")
        return jsonify({'error': f'Failed to create ZIP file: {str(e)}'}), 500

# Also add the original route to maintain compatibility
@user_auth_bp.route('/download/<path:dirpath>', methods=['GET'])
@conditional_login_required
def download_dir_compat(dirpath):
    """Compatibility route that redirects to the new download_directory endpoint"""
    current_app.logger.info(f"Redirecting old directory download URL format: {dirpath}")
    return download_directory(dirpath)

@user_auth_bp.route('/michigan')
@conditional_login_required
@conditional_verified_required
def michigan():
    current_app.logger.info("Michigan route called")
    return jsonify({
        "title": "Michigan",
        "message": "Michigan page"
    })

# Google OAuth routes
@user_auth_bp.route('/api/login/google', methods=['GET'])
@csrf.exempt
def login_google():
    """Initiate Google OAuth flow"""
    current_app.logger.info("Google login initiated")

    # Check if user is already logged in
    if current_user.is_authenticated:
        return redirect('/')

    # Get Google authorization URL
    google_auth_url = get_google_auth_url()
    if not google_auth_url:
        current_app.logger.error("Failed to generate Google authorization URL")
        return jsonify({"status": "error", "message": "Unable to connect to Google authentication service"}), 500

    return redirect(google_auth_url)

@user_auth_bp.route('/api/login/google/callback', methods=['GET'])
@csrf.exempt
def google_callback():
    """Handle the Google OAuth callback"""
    current_app.logger.info("Google callback received")

    # Check if user is already logged in
    if current_user.is_authenticated:
        current_app.logger.info(f"User already authenticated as {current_user.username}, redirecting to home")
        return redirect('/')

    # Get error from request if any
    error = request.args.get('error')
    if error:
        current_app.logger.error(f"Google OAuth error: {error}")
        return redirect('/login?error=google_oauth_error')

    # Process callback and get user info
    user_data = process_google_callback()
    if not user_data:
        current_app.logger.error("Google authentication failed - could not get user data")
        return redirect('/login?error=google_auth_failed')

    # Log the user data we received (except sensitive info)
    current_app.logger.info(f"Received Google user data for: {user_data.get('email')}")

    # Find or create user and log them in
    try:
        user = login_oauth_user(user_data)
        if not user:
            current_app.logger.error(f"Failed to login/create user with Google OAuth: {user_data.get('email')}")
            return redirect('/login?error=user_creation_failed')

        # Set a success flag in the session
        session['oauth_login_success'] = True
        session['oauth_provider'] = 'google'
        session['oauth_email'] = user_data.get('email')

        # Log the user in with Flask-Login
        login_user(user, remember=True)
        current_app.logger.info(f"User {user.username} logged in with Google OAuth")

        # Create SFTP user if it doesn't exist
        if not os.path.exists(f"{current_app.config['USER_PATH']}/{user.username}"):
            try:
                # Create user directory structure
                user_dir = os.path.join(current_app.config['USER_PATH'], user.username)
                os.makedirs(user_dir, exist_ok=True)
                os.makedirs(os.path.join(user_dir, 'SWATplus_by_VPUID'), exist_ok=True)
                os.makedirs(os.path.join(user_dir, 'Reports'), exist_ok=True)
                current_app.logger.info(f"Created directory structure for OAuth user '{user.username}'")

                # Create SFTP account
                sftp_result = create_sftp_user(user.username)
                if sftp_result.get('success'):
                    current_app.logger.info(f"SFTP account created for OAuth user '{user.username}'")
                else:
                    current_app.logger.warning(f"Failed to create SFTP account for OAuth user '{user.username}': {sftp_result.get('message')}")
            except Exception as e:
                current_app.logger.error(f"Error setting up user directory for OAuth user: {e}")
                # Continue anyway since this shouldn't block login

        # Get the frontend URL from config or use a default
        frontend_url = current_app.config.get('SITE_URL', 'http://localhost:3000')
        redirect_url = f"{frontend_url}/?google_login=success&username={user.username}"
        current_app.logger.info(f"Redirecting OAuth user to: {redirect_url}")

        # Redirect to the home page with success parameter
        return redirect(redirect_url)

    except Exception as e:
        current_app.logger.error(f"Unexpected error in Google OAuth callback: {str(e)}")
        return redirect('/login?error=internal_error')

# Add a new route to check Google OAuth login status
@user_auth_bp.route('/api/check-oauth-login', methods=['GET'])
def check_oauth_login():
    """Check if a user has logged in with OAuth"""
    if current_user.is_authenticated:
        # Check if this was an OAuth login
        oauth_success = session.get('oauth_login_success', False)
        provider = session.get('oauth_provider', '')

        return jsonify({
            "status": "success",
            "authenticated": True,
            "oauth_login": oauth_success,
            "provider": provider,
            "user": {
                "id": current_user.id,
                "username": current_user.username,
                "email": current_user.email,
                "is_verified": current_user.is_verified
            }
        })
    else:
        return jsonify({
            "status": "success",
            "authenticated": False,
            "oauth_login": False
        })

# SFTP Routes (from sftp_routes.py)
@user_auth_bp.route("/api/sftp/create", methods=["POST"])
@login_required
@csrf.exempt
def create_sftp():
    """Create an SFTP user when requested from the frontend."""
    try:
        data = request.get_json()
        username = current_user.username

        if 'password' not in data:
            current_app.logger.warning(f"SFTP creation failed: missing password for user {username}")
            return jsonify({"error": "Password is required"}), 400

        # Call the function that actually creates the SFTP user
        result = create_sftp_user(username, data.get('password'))

        # Clear password from memory immediately
        data['password'] = None

        if result.get("success"):
            current_app.logger.info(f"SFTP user created successfully for {username}")
            return jsonify({"status": "success", "message": result.get("message", "SFTP user created successfully")})
        else:
            current_app.logger.error(f"SFTP creation failed: {result.get('message')}")
            return jsonify({"status": "error", "message": result.get("message", "Failed to create SFTP user")}), 500

    except Exception as e:
        current_app.logger.error(f"Error in SFTP create route: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@user_auth_bp.route("/api/sftp/delete", methods=["POST"])
@login_required
@csrf.exempt
def delete_sftp():
    """Delete an SFTP user."""
    try:
        username = current_user.username

        # Call the function that actually deletes the SFTP user
        result = delete_sftp_user(username)

        if result.get("success"):
            current_app.logger.info(f"SFTP user deleted successfully for {username}")
            return jsonify({"status": "success", "message": result.get("message", "SFTP user deleted successfully")})
        else:
            current_app.logger.error(f"SFTP deletion failed: {result.get('message')}")
            return jsonify({"status": "error", "message": result.get("message", "Failed to delete SFTP user")}), 500

    except Exception as e:
        current_app.logger.error(f"Error in SFTP delete route: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
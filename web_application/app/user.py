from flask import Blueprint, jsonify, current_app, url_for, request, send_from_directory, send_file
from flask_login import current_user
from app.models import ContactMessage
from app.extensions import db
from app.decorators import conditional_login_required, conditional_verified_required
import os
import tempfile
from werkzeug.utils import secure_filename
import shutil

user_bp = Blueprint('user', __name__)

@user_bp.route('/api/index')
@conditional_login_required
@conditional_verified_required
def index():
    return jsonify({"status": "success", "message": "Welcome to the API!"})

@user_bp.route('/user_dashboard', methods=['GET'])
@conditional_login_required
@conditional_verified_required
def user_dashboard():
    """User dashboard route."""
    current_app.logger.info(f"User Dashboard accessed by `{current_user.username}`.")
    return jsonify({"title": "Dashboard", "message": "Your user dashboard."})

@user_bp.route('/home', methods=['GET'])
@conditional_login_required
@conditional_verified_required
def home():
    """User's home page."""
    current_app.logger.info(f"Home route accessed by user: {current_user.username}.")
    return jsonify({"title": "Home", "message": "Welcome to the app!"})

@user_bp.route('/about')
@conditional_login_required
@conditional_verified_required
def about():
    current_app.logger.info("About route called")
    return jsonify({"title": "About", "message": "about page"})

@user_bp.route('/privacy')
def privacy():
    current_app.logger.info("Privacy route called")
    return jsonify({"title": "Privacy", "message": "privacy page"})

@user_bp.route('/terms')
def terms():
    current_app.logger.info("Terms route called")
    return jsonify({"title": "Terms", "message": "terms page"})

@user_bp.route('/contact', methods=['POST'])
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

@user_bp.route('/api/user_files', methods=['GET'])
@conditional_login_required
def api_user_files():
    """
    Lists directories and files for the logged-in user.
    Supports navigation within subdirectories and provides download links.
    """
    current_app.logger.info("API User Files route called")
    base_user_dir = os.path.join('/data/SWATGenXApp/Users', current_user.username, "SWATplus_by_VPUID")
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
            safe_item = secure_filename(item) # Prevent path traversal
            item_path = os.path.join(target_dir, safe_item)

            if os.path.isdir(item_path):
                contents['directories'].append({
                    'name': safe_item,
                    'path': os.path.join(subdir, safe_item).lstrip('/'),
                    'download_zip_url': url_for('user.download_directory', dirpath=f"{subdir}/{safe_item}".lstrip('/'))
                })
            elif os.path.isfile(item_path):
                contents['files'].append({
                    'name': safe_item,
                    'download_url': url_for('user.download_user_file', filename=f"{subdir}/{safe_item}".lstrip('/'))
                })

    return jsonify(contents)

@user_bp.route('/download/<path:filename>', methods=['GET'])
@conditional_login_required
def download_user_file(filename):
    user_dir = os.path.join('/data/SWATGenXApp/Users', current_user.username, "SWATplus_by_VPUID")
    full_path = os.path.join(user_dir, filename)

    if not full_path.startswith(user_dir) or not os.path.isfile(full_path):
        current_app.logger.error(f"File not found or access denied: {full_path}")
        return jsonify({'error': 'File not found or access denied'}), 404

    current_app.logger.info(f"Serving file for download: {full_path}")
    directory, file = os.path.split(full_path)
    return send_from_directory(directory, file, as_attachment=True)

@user_bp.route('/download-directory/<path:dirpath>', methods=['GET'])
@conditional_login_required
def download_directory(dirpath):
    user_dir = os.path.join('/data/SWATGenXApp/Users', current_user.username, "SWATplus_by_VPUID")
    full_dir_path = os.path.join(user_dir, dirpath)

    if not full_dir_path.startswith(user_dir) or not os.path.isdir(full_dir_path):
        current_app.logger.error(f"Unauthorized directory access or not found: {full_dir_path}")
        return jsonify({'error': 'Directory not found or access denied'}), 404

    current_app.logger.info(f"Creating ZIP for directory: {full_dir_path}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        zip_path = tmp_zip.name

    try:
        shutil.make_archive(zip_path[:-4], 'zip', full_dir_path)
    except Exception as e:
        current_app.logger.error(f"Failed to create ZIP: {e}")
        return jsonify({'error': 'Failed to create ZIP file'}), 500

    zip_file_name = f"{os.path.basename(dirpath)}.zip"
    final_zip_path = zip_path[:-4] + ".zip"

    if not os.path.exists(final_zip_path):
        current_app.logger.error(f"ZIP file missing: {final_zip_path}")
        return jsonify({'error': 'ZIP file not found'}), 500

    current_app.logger.info(f"Serving ZIP file for download: {final_zip_path}")
    return send_file(final_zip_path, as_attachment=True, download_name=zip_file_name)

@user_bp.route('/michigan')
@conditional_login_required
@conditional_verified_required
def michigan():
    current_app.logger.info("Michigan route called")  
    return jsonify({
        "title": "Michigan",
        "message": "Michigan page"
    })

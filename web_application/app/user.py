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
            safe_item = secure_filename(item)  # Prevent path traversal
            item_path = os.path.join(target_dir, safe_item)
            rel_path = os.path.join(subdir, safe_item).lstrip('/')

            if os.path.isdir(item_path):
                # Generate both path-based and direct download URLs
                download_url = url_for('user.download_directory', dirpath=rel_path)
                current_app.logger.info(f"Generated directory download URL: {download_url}")
                
                contents['directories'].append({
                    'name': safe_item,
                    'path': rel_path,
                    'download_zip_url': download_url
                })
            elif os.path.isfile(item_path):
                download_url = url_for('user.download_user_file', filename=rel_path)
                current_app.logger.info(f"Generated file download URL: {download_url}")
                contents['files'].append({
                    'name': safe_item,
                    'download_url': download_url
                })

    return jsonify(contents)

@user_bp.route('/download/<path:filename>', methods=['GET'])
@conditional_login_required
def download_user_file(filename):
    current_app.logger.info(f"download_user_file called with filename: {filename}")  # Log the filename

    user_dir = os.path.join('/data/SWATGenXApp/Users', current_user.username, "SWATplus_by_VPUID")
    filename = filename.lstrip("/")
    full_path = os.path.join(user_dir, filename)

    current_app.logger.info(f"Constructed file path: {full_path}")  # Log the constructed path

    if not os.path.exists(full_path):
        current_app.logger.error(f"File not found: {full_path}")  # Log if the file does not exist
        return jsonify({'error': 'File not found or access denied'}), 404

    current_app.logger.info(f"Serving file for download: {full_path}")
    directory, file = os.path.split(full_path)
    return send_from_directory(directory, file, as_attachment=True)

@user_bp.route('/download_directory/<path:dirpath>', methods=['GET'])
@conditional_login_required
def download_directory(dirpath):
    """
    Download a directory as a ZIP file.
    """
    current_app.logger.info(f"download_directory called with dirpath: {dirpath}")
    
    user_dir = os.path.join('/data/SWATGenXApp/Users', current_user.username, "SWATplus_by_VPUID")
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
@user_bp.route('/download/<path:dirpath>', methods=['GET'])
@conditional_login_required
def download_dir_compat(dirpath):
    """Compatibility route that redirects to the new download_directory endpoint"""
    current_app.logger.info(f"Redirecting old directory download URL format: {dirpath}")
    return download_directory(dirpath)

@user_bp.route('/michigan')
@conditional_login_required
@conditional_verified_required
def michigan():
    current_app.logger.info("Michigan route called")  
    return jsonify({
        "title": "Michigan",
        "message": "Michigan page"
    })

from flask import Blueprint, send_from_directory, send_file, current_app
import os

static_bp = Blueprint('static', __name__)

@static_bp.route('/static/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('/data/SWATGenXApp/GenXAppData/images', filename)

@static_bp.route('/static/videos/<path:filename>')
def serve_videos(filename):
    return send_from_directory('/data/SWATGenXApp/GenXAppData/videos', filename)

@static_bp.route('/static/visualizations/<path:filename>')
def serve_visualizations(filename):
    return send_from_directory('/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12', filename)

@static_bp.route('/', defaults={'path': ''})
@static_bp.route('/<path:path>')
def serve_frontend(path):
    frontend_build_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'build')
    
    # First try to serve static files
    if path.startswith('static/'):
        return send_from_directory(os.path.join(frontend_build_dir), path)
    
    # Then try to serve the file directly if it exists
    if path and os.path.exists(os.path.join(frontend_build_dir, path)):
        return send_from_directory(frontend_build_dir, path)
    
    # Default to serving index.html
    return send_from_directory(frontend_build_dir, 'index.html')

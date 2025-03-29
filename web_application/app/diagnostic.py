from flask import Blueprint, jsonify, request, current_app, send_from_directory, url_for
import os
import platform
import sys
import datetime
import json
from app.extensions import csrf

# Create the blueprint with a specific url_prefix to avoid path conflicts
diagnostic_bp = Blueprint('diagnostic', __name__, url_prefix='/api/diagnostic')

# Apply CSRF exemption to the entire blueprint
csrf.exempt(diagnostic_bp)

@diagnostic_bp.route('/echo', methods=['POST', 'OPTIONS'])
def echo():
    """Echo endpoint to verify frontend-to-backend communication"""
    current_app.logger.info(f"Echo endpoint called: {request.method}")
    
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({"message": "CORS preflight handled"})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-CSRF-Token'
        return response
    
    # Log request information
    current_app.logger.info(f"Request headers: {dict(request.headers)}")
    current_app.logger.info(f"Request content type: {request.content_type}")
    
    # Handle JSON data
    try:
        if request.is_json:
            data = request.json
        else:
            data = {"non_json": True, "data": request.data.decode('utf-8', errors='replace')}
    except Exception as e:
        data = {"error": str(e)}
    
    # Return all request information
    return jsonify({
        "success": True,
        "received_data": data,
        "request_info": {
            "method": request.method,
            "url": request.url,
            "path": request.path,
            "headers": dict(request.headers),
            "remote_addr": request.remote_addr,
            "content_type": request.content_type,
            "is_json": request.is_json
        },
        "server_info": {
            "time": datetime.datetime.now().isoformat(),
            "environment": os.environ.get('FLASK_ENV', 'not set'),
            "host": request.host
        }
    })

@diagnostic_bp.route('/status', methods=['GET'])
def status():
    """Simple status endpoint to verify API access"""
    current_app.logger.info(f"Status endpoint called by {request.remote_addr}")
    return jsonify({
        "status": "ok",
        "time": datetime.datetime.now().isoformat(),
        "environment": os.environ.get('FLASK_ENV', 'not set')
    })

# Add route to test model settings specifically
@diagnostic_bp.route('/test-model-settings', methods=['POST', 'OPTIONS'])
def test_model_settings():
    """Test endpoint that simulates the model-settings route behavior"""
    current_app.logger.info(f"Test model-settings endpoint called: {request.method}")
    
    # Handle OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({"message": "CORS preflight handled"})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-CSRF-Token'
        return response
    
    try:
        # Log request details
        current_app.logger.info(f"Request headers: {dict(request.headers)}")
        current_app.logger.info(f"Request content type: {request.content_type}")
        
        # Extract data
        if request.is_json:
            data = request.json
        else:
            try:
                data = json.loads(request.data.decode('utf-8'))
            except:
                data = {"error": "Could not parse request as JSON"}
        
        # Return success response that mimics model-settings
        response = jsonify({
            "status": "success",
            "message": "Model settings test successful",
            "received_data": data,
            "task_id": "test-task-id"
        })
        
        current_app.logger.info("Returning success response from test-model-settings")
        return response
    except Exception as e:
        current_app.logger.error(f"Error in test-model-settings: {e}")
        return jsonify({
            "status": "error",
            "message": "Error processing request",
            "details": str(e)
        }), 500

# File system access diagnostics

@diagnostic_bp.route('/file-access', methods=['GET'])
def file_access_diagnostics():
    """
    Checks and reports file system access across important directories
    """
    results = {
        "static_paths": {},
        "download_endpoints": {},
        "environment": os.environ.get('FLASK_ENV', 'not set')
    }
    
    # Check static file paths
    static_paths = {
        "images": "/data/SWATGenXApp/GenXAppData/images",
        "videos": "/data/SWATGenXApp/GenXAppData/videos",
        "visualizations": "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12",
        "user_files": "/data/SWATGenXApp/Users"
    }
    
    for name, path in static_paths.items():
        results["static_paths"][name] = {
            "path": path,
            "exists": os.path.exists(path),
            "is_dir": os.path.isdir(path) if os.path.exists(path) else False,
            "readable": os.access(path, os.R_OK) if os.path.exists(path) else False,
            "sample_files": []
        }
        
        # Try to list some sample files
        if os.path.isdir(path) and os.access(path, os.R_OK):
            try:
                files = os.listdir(path)[:5]  # Just list first 5 files
                results["static_paths"][name]["sample_files"] = files
            except Exception as e:
                results["static_paths"][name]["error"] = str(e)
    
    # Check endpoints for downloads
    endpoints = {
        "static_images": url_for('static.serve_images', filename='placeholder.jpg', _external=True),
        "static_videos": url_for('static.serve_videos', filename='placeholder.mp4', _external=True),
        "static_visualizations": url_for('static.serve_visualizations', filename='placeholder.png', _external=True),
        "api_visualizations": url_for('api.api_serve_visualizations', filename='placeholder.png', _external=True),
    }
    
    results["download_endpoints"] = endpoints
    
    return jsonify(results)

@diagnostic_bp.route('/serve-test-file', methods=['GET'])
def serve_test_file():
    """
    Generate and serve a test file to verify download capabilities
    """
    # Create test file content
    test_content = f"""
    SWATGenX File Access Test
    Generated: {datetime.datetime.now().isoformat()}
    Environment: {os.environ.get('FLASK_ENV', 'not set')}
    Host: {request.host}
    Platform: {platform.platform()}
    Python: {sys.version}
    """
    
    # Create a temporary directory to store the test file
    test_dir = os.path.join(current_app.instance_path, 'test_files')
    os.makedirs(test_dir, exist_ok=True)
    
    # Create the test file
    test_file_path = os.path.join(test_dir, 'access_test.txt')
    with open(test_file_path, 'w') as f:
        f.write(test_content)
    
    return send_from_directory(
        test_dir, 
        'access_test.txt',
        as_attachment=True, 
        download_name='swatgenx_access_test.txt'
    )

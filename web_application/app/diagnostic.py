from flask import Blueprint, jsonify, request, current_app
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

from flask import Blueprint, jsonify, request, current_app, send_file, redirect, send_from_directory
from flask_login import current_user
import os
import json
import threading
import zipfile
import tempfile
from datetime import datetime
from app.decorators import conditional_login_required, conditional_verified_required

report_bp = Blueprint('report', __name__)

# Handle both API and non-API versions of reports route
@report_bp.route('/api/reports', methods=['GET'])
@report_bp.route('/reports', methods=['GET'])
@conditional_login_required
@conditional_verified_required
def get_reports_redirect():
    """Redirect to the get_reports API endpoint"""
    return redirect('/api/get_reports')

# Handle report paths with the appropriate normalization
@report_bp.route('/api/reports/<report_id>', methods=['GET'])
@report_bp.route('/reports/<report_id>', methods=['GET'])
@conditional_login_required
@conditional_verified_required
def report_view_redirect(report_id):
    """Redirect to the report view API endpoint"""
    return redirect(f"/api/reports/{report_id}/view")

@report_bp.route('/api/generate_report', methods=['POST'])
@conditional_login_required
@conditional_verified_required
def generate_report():
    """Generate a report based on user-selected area and parameters."""
    current_app.logger.info("Report generation request received")
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400
        
        # Extract and validate bounding box coordinates
        min_latitude = float(data.get('min_latitude'))
        max_latitude = float(data.get('max_latitude'))
        min_longitude = float(data.get('min_longitude'))
        max_longitude = float(data.get('max_longitude'))
        
        if not all([min_latitude, max_latitude, min_longitude, max_longitude]):
            return jsonify({"error": "Bounding box coordinates are required"}), 400
        
        # Log the coordinates for debugging
        current_app.logger.info(f"Coordinates: {min_longitude}, {min_latitude}, {max_longitude}, {max_latitude}")
        
        # Verify coordinate validity
        if not (-90 <= min_latitude <= 90 and -90 <= max_latitude <= 90 and 
                -180 <= min_longitude <= 180 and -180 <= max_longitude <= 180):
            return jsonify({"error": "Invalid coordinate values"}), 400
        
        # Check if we're dealing with polygon coordinates and process them properly
        polygon_coordinates = data.get('polygon_coordinates')
        geometry_type = data.get('geometry_type', 'extent')
        
        if polygon_coordinates:
            try:
                # Try to parse the polygon coordinates if it's a string
                if isinstance(polygon_coordinates, str):
                    coords = json.loads(polygon_coordinates)
                else:
                    coords = polygon_coordinates
                    
                current_app.logger.info(f"Using polygon with {len(coords)} vertices")
                # Store the processed coordinates for later use if needed
                processed_polygon = coords
            except Exception as e:
                current_app.logger.error(f"Error parsing polygon coordinates: {e}")
                return jsonify({"error": f"Invalid polygon format: {e}"}), 400
        
        # Extract report parameters
        report_type = data.get('report_type', 'all')
        start_year = int(data.get('start_year', 2010))
        end_year = int(data.get('end_year', 2020))
        resolution = int(data.get('resolution', 250))
        aggregation = data.get('aggregation', 'monthly')
        include_climate_change = data.get('include_climate_change', False)
        
        # Create output directory for the report
        username = current_user.username
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('/data/SWATGenXApp/Users', username, "Reports", timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create configuration for report generator
        config = {
            'RESOLUTION': resolution,
            'resolution': resolution,
            'start_year': start_year,
            'end_year': end_year,
            'bounding_box': [min_longitude, min_latitude, max_longitude, max_latitude],
            'aggregation': aggregation,
            'include_climate_change': include_climate_change,
            'geometry_type': geometry_type
        }
        
        # Add polygon if available
        if polygon_coordinates:
            config['polygon_coordinates'] = processed_polygon
        
        # Log the configuration
        current_app.logger.info(f"Report configuration: {config}")
        
        # Generate the report in a background thread to prevent blocking
        def generate_report_task():
            try:
                # Import the report generator function
                from AI_agent.report_generator import run_report_generation
                
                reports = run_report_generation(report_type, config, output_dir, parallel=True)
                # Save report metadata to database or file for later retrieval
                report_metadata = {
                    'username': username,
                    'timestamp': timestamp,
                    'report_type': report_type,
                    'bounding_box': [min_longitude, min_latitude, max_longitude, max_latitude],
                    'output_dir': output_dir,
                    'reports': reports,
                    'status': 'completed' if reports else 'failed'
                }
                
                # Save metadata to file
                with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
                    json.dump(report_metadata, f, indent=2)
                
                current_app.logger.info(f"Report generation completed: {output_dir}")
            except Exception as e:
                current_app.logger.error(f"Error generating report: {e}")
                # Save error information
                error_info = {
                    'username': username,
                    'timestamp': timestamp,
                    'error': str(e),
                    'status': 'failed'
                }
                with open(os.path.join(output_dir, 'error.json'), 'w') as f:
                    json.dump(error_info, f, indent=2)
        
        # Start the background task
        thread = threading.Thread(target=generate_report_task)
        thread.daemon = True
        thread.start()
            
        return jsonify({
            'status': 'success',
            'message': 'Report generation started',
            'report_id': timestamp,
            'output_dir': output_dir
        })
    
    except Exception as e:
        current_app.logger.error(f"Error initiating report generation: {e}")
        return jsonify({"error": f"Failed to start report generation: {str(e)}"}), 500

@report_bp.route('/api/get_reports', methods=['GET'])
@conditional_login_required
@conditional_verified_required
def get_reports():
    """Get a list of reports generated by the user."""
    username = current_user.username
    reports_dir = os.path.join('/data/SWATGenXApp/Users', username, "Reports")
    
    if not os.path.exists(reports_dir):
        return jsonify({"reports": []})
    
    reports = []
    for report_id in os.listdir(reports_dir):
        report_path = os.path.join(reports_dir, report_id)
        if os.path.isdir(report_path):
            metadata_path = os.path.join(report_path, 'metadata.json')
            error_path = os.path.join(report_path, 'error.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                reports.append(metadata)
            elif os.path.exists(error_path):
                with open(error_path, 'r') as f:
                    error_info = json.load(f)
                reports.append(error_info)
            else:
                # Report is still processing or was interrupted
                reports.append({
                    'report_id': report_id,
                    'status': 'processing',
                    'timestamp': report_id
                })
    
    # Sort by timestamp (newest first)
    reports.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return jsonify({"reports": reports})

@report_bp.route('/api/reports/<report_id>/download', methods=['GET'])
@conditional_login_required
@conditional_verified_required
def download_report(report_id):
    """Download a generated report."""
    if not report_id:
        current_app.logger.error("Missing report ID in download request")
        return jsonify({'error': 'Report ID is required'}), 400
        
    username = current_user.username
    report_dir = os.path.join('/data/SWATGenXApp/Users', username, "Reports", report_id)
    
    current_app.logger.info(f"Attempting to download report: {report_dir}")
    
    if not os.path.exists(report_dir) or not os.path.isdir(report_dir):
        current_app.logger.error(f"Report directory not found: {report_dir}")
        return jsonify({'error': f'Report with ID {report_id} not found'}), 404
    
    # Check if the metadata file exists
    metadata_path = os.path.join(report_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        current_app.logger.error(f"Report metadata not found: {metadata_path}")
        return jsonify({'error': f'Report metadata for ID {report_id} not found'}), 404
    
    try:
        # Read metadata to get report paths
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create a ZIP file with all report files
        zip_filename = f"{report_id}_reports.zip"
        zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
        
        # List of files included in the ZIP
        included_files = []
        
        # Create a new ZIP file with all the reports
        with zipfile.ZipFile(zip_path, 'w') as report_zip:
            # Add metadata file
            report_zip.write(metadata_path, os.path.basename(metadata_path))
            included_files.append(metadata_path)
            
            # Add all report files
            for report_path in metadata.get('reports', []):
                if os.path.exists(report_path):
                    # Add file to ZIP with relative path from report directory
                    arcname = os.path.relpath(report_path, report_dir)
                    report_zip.write(report_path, arcname)
                    included_files.append(report_path)
        
        current_app.logger.info(f"Generated report ZIP file: {zip_path} with {len(included_files)} files")
        current_app.logger.debug(f"Files included: {included_files}")
        
        return send_file(
            zip_path, 
            mimetype='application/zip', 
            download_name=zip_filename, 
            as_attachment=True
        )
    
    except Exception as e:
        current_app.logger.error(f"Error creating report ZIP: {e}")
        return jsonify({
            'error': f'Failed to create report package: {str(e)}',
            'report_id': report_id
        }), 500

@report_bp.route('/api/reports/<report_id>/view', methods=['GET'])
@report_bp.route('/api/reports/<report_id>/view/<path:subpath>', methods=['GET'])
@conditional_login_required
@conditional_verified_required
def view_report(report_id, subpath=None):
    """View a specific report file or the default report page."""
    # Add extensive debugging
    current_app.logger.info(f"VIEW REPORT CALLED - ID: {report_id}, Subpath: {subpath}")
    current_app.logger.info(f"Request URL: {request.url}")
    current_app.logger.info(f"Request Path: {request.path}")
    
    if not report_id:
        current_app.logger.error("Missing report ID in view request")
        return jsonify({'error': 'Report ID is required'}), 400
            
    username = current_user.username
    report_dir = os.path.join('/data/SWATGenXApp/Users', username, "Reports", report_id)
    
    current_app.logger.info(f"Report directory: {report_dir}")
    
    if not os.path.exists(report_dir) or not os.path.isdir(report_dir):
        current_app.logger.error(f"Report directory not found: {report_dir}")
        return jsonify({'error': f'Report with ID {report_id} not found'}), 404
    
    # Debug listing of directory contents to help troubleshoot
    try:
        current_app.logger.info(f"Listing report directory contents:")
        for root, dirs, files in os.walk(report_dir):
            rel_root = os.path.relpath(root, report_dir)
            current_app.logger.info(f"Directory: {rel_root}")
            for file in files:
                current_app.logger.info(f"  File: {os.path.join(rel_root, file)}")
    except Exception as e:
        current_app.logger.error(f"Error listing directory contents: {e}")
    
    # If a specific subpath is requested, serve that file directly
    if subpath:
        # Improve path normalization to handle all cases
        # Remove any leading slashes and normalize path
        clean_subpath = subpath.lstrip('/')
        clean_subpath = os.path.normpath(clean_subpath)
        file_path = os.path.join(report_dir, clean_subpath)
        
        current_app.logger.info(f"Requested subpath: {subpath}")
        current_app.logger.info(f"Normalized subpath: {clean_subpath}")
        current_app.logger.info(f"Full file path: {file_path}")
        
        # Verify the path is still within the report directory to prevent directory traversal
        if not os.path.abspath(file_path).startswith(os.path.abspath(report_dir)):
            current_app.logger.error(f"Attempted directory traversal: {file_path}")
            return jsonify({'error': 'Access denied'}), 403
        
        # Special handling for HTML files - we need to check if they exist
        # and properly set MIME type for browser rendering
        if os.path.exists(file_path) and os.path.isfile(file_path):
            current_app.logger.info(f"File exists, serving: {file_path}")
            file_dir, file_name = os.path.split(file_path)
            
            # Determine content type based on file extension
            _, ext = os.path.splitext(file_name)
            content_type = None
            if ext.lower() == '.html':
                content_type = 'text/html'
            elif ext.lower() == '.css':
                content_type = 'text/css'
            elif ext.lower() == '.js':
                content_type = 'application/javascript'
            elif ext.lower() == '.png':
                content_type = 'image/png'
            elif ext.lower() == '.jpg' or ext.lower() == '.jpeg':
                content_type = 'image/jpeg'
            
            if content_type:
                current_app.logger.info(f"Serving with content type: {content_type}")
                return send_from_directory(file_dir, file_name, mimetype=content_type)
            else:
                current_app.logger.info(f"Serving with auto-detected content type")
                return send_from_directory(file_dir, file_name)
        else:
            current_app.logger.error(f"Subpath file not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
    
    # No subpath specified - find an appropriate default file to display
    # First check for index.html in the root directory
    index_path = os.path.join(report_dir, 'index.html')
    if os.path.exists(index_path):
        current_app.logger.info(f"Serving main index.html: {index_path}")
        return send_file(index_path, mimetype='text/html')
    
    # Return report metadata if no index file
    metadata_path = os.path.join(report_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return jsonify(metadata)
    
    return jsonify({'error': 'No viewable content found'}), 404

@report_bp.route('/api/reports/<report_id>/status', methods=['GET'])
@conditional_login_required
@conditional_verified_required
def get_report_status(report_id):
    """Check the status of a report."""
    if not report_id:
        current_app.logger.error("Missing report ID in status check request")
        return jsonify({'error': 'Report ID is required'}), 400
        
    username = current_user.username
    report_dir = os.path.join('/data/SWATGenXApp/Users', username, "Reports", report_id)
    
    current_app.logger.info(f"Checking status of report: {report_dir}")
    
    # Check if report directory exists
    if not os.path.exists(report_dir) or not os.path.isdir(report_dir):
        current_app.logger.error(f"Report directory not found: {report_dir}")
        return jsonify({
            'status': 'not_found',
            'error': f'Report with ID {report_id} not found'
        }), 404
    
    # Look for metadata or error files
    metadata_path = os.path.join(report_dir, 'metadata.json')
    error_path = os.path.join(report_dir, 'error.json')
    
    # Return appropriate response based on which files exist
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Add the report ID if not included
            if 'report_id' not in metadata:
                metadata['report_id'] = report_id
                
            # Make sure status is included
            if 'status' not in metadata:
                metadata['status'] = 'completed'
                
            return jsonify(metadata)
        except Exception as e:
            current_app.logger.error(f"Error reading report metadata: {e}")
            return jsonify({
                'status': 'error',
                'report_id': report_id,
                'error': f'Error reading report metadata: {str(e)}'
            }), 500
    
    elif os.path.exists(error_path):
        try:
            with open(error_path, 'r') as f:
                error_info = json.load(f)
            
            # Add the report ID if not included
            if 'report_id' not in error_info:
                error_info['report_id'] = report_id
                
            # Make sure status is included
            if 'status' not in error_info:
                error_info['status'] = 'failed'
                
            return jsonify(error_info)
        except Exception as e:
            current_app.logger.error(f"Error reading report error info: {e}")
            return jsonify({
                'status': 'error',
                'report_id': report_id,
                'error': f'Error reading report error info: {str(e)}'
            }), 500
    
    else:
        # Report is still processing
        return jsonify({
            'status': 'processing',
            'report_id': report_id,
            'message': 'Report is still being generated'
        })

@report_bp.route('/api/reports/<report_id>', defaults={'path': ''})
@report_bp.route('/api/reports/<report_id>/<path:path>')
@conditional_login_required
@conditional_verified_required
def report_redirect(report_id, path=''):
    """
    Redirect incorrect report URLs to the correct ones with the '/view/' segment.
    This handles cases when a link in a report points to an incorrect URL.
    """
    current_app.logger.info(f"Redirect handler called for: {report_id}/{path}")
    
    # If this is not already a view URL, redirect to the proper view URL
    if not path.startswith('view/') and path != 'view':
        redirect_url = f"/api/reports/{report_id}/view/{path}"
        current_app.logger.info(f"Redirecting to: {redirect_url}")
        return redirect(redirect_url)
    
    return jsonify({'error': 'Invalid report URL'}), 404

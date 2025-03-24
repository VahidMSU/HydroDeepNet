from flask import Blueprint, jsonify, request, current_app
from flask_login import current_user
import pandas as pd
import ast
from app.decorators import conditional_login_required, conditional_verified_required
from app.utils import (find_station, check_existing_models, get_huc12_geometries,
                      get_huc12_streams_geometries, get_huc12_lakes_geometries)
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
from app.extensions import csrf

model_bp = Blueprint('model', __name__)

@model_bp.route('/api/search_site', methods=['GET', 'POST'])
def search_site():
    current_app.logger.info("Search site route called")
    search_term = request.args.get('search_term', '').lower()
    if not search_term:
        return jsonify({"error": "Search term is required"}), 400
        
    try:
        results = find_station(search_term)
        if results.empty:
            current_app.logger.info("No matching sites found")
            return jsonify({"error": "No matching sites found"}), 404
        return jsonify(results.to_dict(orient='records'))
    except Exception as e:
        current_app.logger.error(f"Error searching for site: {e}")
        return jsonify({"error": "An error occurred during the search"}), 500

@model_bp.route('/api/get_station_characteristics', methods=['GET'])
def get_station_characteristics():
    current_app.logger.info("Get Station Characteristics route called")
    station_no = request.args.get('station', None)
    
    # Load station CSV
    station_data = pd.read_csv(SWATGenXPaths.FPS_all_stations, dtype={'SiteNumber': str})
    current_app.logger.info(f"Station number: {station_no}")

    # Find the row with that station
    station_row = station_data[station_data.SiteNumber == station_no]
    if station_row.empty:
        current_app.logger.error(f"Station {station_no} not found in CSV.")
        return jsonify({"error": "Station not found"}), 404

    # Convert row to dict
    characteristics = station_row.iloc[0].to_dict()
    current_app.logger.info(f"Found station row for {station_no}")

    # Check if a model already exists
    #existance_flag = check_existing_models(station_no, config=SWATGenXPaths)
    #characteristics['model_exists'] = str(existance_flag).capitalize()

    # Safely parse the HUC12 list from the CSV field
    # The CSV field looks like: "['040500040703','040500040508', ... ]"
    huc12_str = characteristics.get('HUC12 ids of the watershed')
    if not huc12_str or pd.isna(huc12_str):
        # If missing/empty, just return characteristics without geometry
        current_app.logger.warning(f"No HUC12 data for station {station_no}")
        return jsonify(characteristics)
    # Parse the string as a Python list
    try:
        #NOTE:This safely evaluates the string as a list:
        # e.g. "['040500040703','040500040508']" -> ["040500040703", "040500040508"]
        huc12_list = ast.literal_eval(huc12_str)
    except Exception as e:
        current_app.logger.error(f"Error parsing HUC12 list for {station_no}: {e}")
        return jsonify({"error": "Failed to parse HUC12 data"}), 500
    # Now call geometry functions safely
    geometries = get_huc12_geometries(huc12_list)
    streams_geometries, lake_identifier = get_huc12_streams_geometries(huc12_list)
    lakes_geometries = get_huc12_lakes_geometries(huc12_list, lake_identifier)
    if not geometries:
        current_app.logger.error(f"No geometries found for HUC12s: {huc12_list}")
    if not streams_geometries:
        current_app.logger.error(f"No streams geometries found for HUC12s: {huc12_list}")
    if not lakes_geometries:
        current_app.logger.warning(f"No lakes geometries found for HUC12s: {huc12_list}")
    # Add geometry data to the dictionary
    characteristics['Num HUC12 subbasins'] = len(huc12_list)
    characteristics['geometries'] = geometries
    characteristics['streams_geometries'] = streams_geometries
    characteristics['lakes_geometries'] = lakes_geometries
    # Clean up if you don't want that field in your final JSON
    characteristics.pop('HUC12 ids of the watershed', None)
    # Return as JSON
    return jsonify(characteristics)

# Support both API and non-API versions for model settings
@model_bp.route('/api/model-settings', methods=['POST'])
@conditional_login_required
@conditional_verified_required
@csrf.exempt  # Explicitly exempt from CSRF to prevent issues
def model_settings():
    """Model settings submission route."""
    current_app.logger.info(f"Model settings route called via {request.url} - starting processing")
    current_app.logger.info(f"Request headers: {dict(request.headers)}")
    current_app.logger.info(f"Request content type: {request.content_type}")
    
    try:
        data = request.json
        if not data:
            current_app.logger.error(f"No JSON data received in request to {request.url}")
            # Try to get raw data if JSON fails
            raw_data = request.get_data(as_text=True)
            current_app.logger.error(f"Raw request data: {raw_data[:200]}")
            return jsonify({
                "status": "error", 
                "message": "No data received or invalid JSON"
            }), 400

        # Extract form data
        site_no = data.get("site_no")
        ls_resolution = data.get("ls_resolution", 250)
        dem_resolution = data.get("dem_resolution", 30)

        current_app.logger.info(
            f"Model settings received for Station `{site_no}`: "
            f"LS Resolution: {ls_resolution}, DEM Resolution: {dem_resolution}"
        )
        
        # Check if user is authenticated
        if current_user.is_anonymous:
            current_app.logger.warning("User is not logged in. Unable to proceed.")
            return jsonify({
                "status": "error", 
                "message": "You must be logged in to create a model"
            }), 403
        
        # Check Redis connection first
        redis_working = False
        redis_error = None
        
        try:
            from redis import Redis
            # Try multiple Redis URLs
            redis_urls = [
                current_app.config.get('REDIS_URL', 'redis://localhost:6379/0'),
                'redis://127.0.0.1:6379/0',
                'redis://redis:6379/0'
            ]
            
            for url in redis_urls:
                try:
                    current_app.logger.info(f"Testing Redis connection at {url}")
                    redis_client = Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)
                    if redis_client.ping():
                        current_app.logger.info(f"Redis connection successful at {url}")
                        redis_working = True
                        # Update the app's Redis URL if needed
                        if url != current_app.config.get('REDIS_URL'):
                            current_app.logger.info(f"Updating Redis URL from {current_app.config.get('REDIS_URL')} to {url}")
                            current_app.config['REDIS_URL'] = url
                        break
                except Exception as e:
                    current_app.logger.warning(f"Redis connection failed at {url}: {e}")
                    redis_error = str(e)
            
            if not redis_working:
                current_app.logger.error("All Redis connection attempts failed")
                return jsonify({
                    "status": "error",
                    "message": "The model creation service is currently unavailable",
                    "details": "Could not connect to Redis server"
                }), 503
                
        except ImportError as e:
            current_app.logger.error(f"Redis module not available: {e}")
            return jsonify({
                "status": "error",
                "message": "Model creation service configuration issue",
                "details": "Required modules not available"
            }), 500
            
        # Celery task creation with retry logic
        from app.swatgenx_tasks import create_model_task
        
        max_retries = 3
        retry_delay = 1
        last_error = None
        
        for attempt in range(max_retries):
            try:
                current_app.logger.info(f"Creating Celery task (attempt {attempt+1})")
                task = create_model_task.delay(
                    current_user.username, 
                    site_no, 
                    ls_resolution, 
                    dem_resolution
                )
                
                current_app.logger.info(f"Model creation task {task.id} scheduled successfully")
                response = jsonify({
                    "status": "success", 
                    "message": "Model creation started",
                    "task_id": task.id
                })
                current_app.logger.info(f"Returning JSON response: {response.data}")
                return response
            except Exception as e:
                last_error = e
                current_app.logger.error(f"Error scheduling task on attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        # All retries failed
        current_app.logger.error(f"All task creation attempts failed: {last_error}")
        return jsonify({
            "status": "error",
            "message": "Failed to schedule model creation task",
            "details": str(last_error)
        }), 500
            
    except Exception as e:
        current_app.logger.error(f"Unexpected error in model_settings: {e}")
        import traceback
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Full error traceback: {error_trace}")
        
        # Ensure we return JSON even when errors occur
        return jsonify({
            "status": "error", 
            "message": "Server error while processing your request",
            "details": str(e)
        }), 500

# Support both API and non-API versions for model confirmation
@model_bp.route('/api/model-confirmation')
@model_bp.route('/model-confirmation')
@conditional_login_required
@conditional_verified_required
def model_confirmation():
    current_app.logger.info("Model Confirmation route called")
    return jsonify({"title": "Model Confirmation", "message": "Model confirmation page"})

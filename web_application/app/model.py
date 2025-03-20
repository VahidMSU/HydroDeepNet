from flask import Blueprint, jsonify, request, current_app
from flask_login import current_user
import pandas as pd
import ast
from app.decorators import conditional_login_required, conditional_verified_required
from app.utils import (find_station, check_existing_models, get_huc12_geometries,
                      get_huc12_streams_geometries, get_huc12_lakes_geometries)
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
from app.tasks import create_model_task

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
    existance_flag = check_existing_models(station_no)
    characteristics['model_exists'] = str(existance_flag).capitalize()

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
@model_bp.route('/model-settings', methods=['POST'])
@conditional_login_required
@conditional_verified_required
def model_settings():
    """Model settings submission route."""
    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400

    # Extract form data
    site_no = data.get("site_no")
    ls_resolution = data.get("ls_resolution", 250)
    dem_resolution = data.get("dem_resolution", 30)

    current_app.logger.info(
        f"Model settings received for Station `{site_no}`: "
        f"LS Resolution: {ls_resolution}, DEM Resolution: {dem_resolution}"
    )
    
    # Perform model creation with improved error handling
    try:
        if current_user.is_anonymous:
            current_app.logger.warning("User is not logged in. Using 'None' as username.")
            return jsonify({"error": "User is not logged in"}), 403
            
        # Try to import redis_utils, but handle the case if the module doesn't exist
        try:
            from app.redis_utils import check_redis_health
            
            # Check Redis health before submitting task
            redis_status = check_redis_health()
            if not redis_status['healthy']:
                current_app.logger.error(f"Redis health check failed: {redis_status['message']}")
                return jsonify({
                    "status": "error", 
                    "message": "The model creation service is currently unavailable. Please try again later or contact support.",
                    "details": redis_status['message']
                }), 503
        except ImportError:
            # If redis_utils is not available, proceed anyway and let Celery handle errors
            current_app.logger.warning("Redis health check not available - proceeding anyway")
        
        # Direct Redis check using basic Redis client
        try:
            from redis import Redis
            redis_client = Redis.from_url(current_app.config.get('REDIS_URL', 'redis://localhost:6379/0'), 
                                          socket_timeout=2, 
                                          socket_connect_timeout=2)
            if not redis_client.ping():
                raise ConnectionError("Redis ping failed")
            current_app.logger.info("Redis connection successful with direct ping")
        except Exception as e:
            current_app.logger.error(f"Direct Redis connection check failed: {e}")
            # Try fallback to 127.0.0.1 explicit address
            try:
                redis_client = Redis.from_url('redis://127.0.0.1:6379/0', 
                                              socket_timeout=2, 
                                              socket_connect_timeout=2)
                if redis_client.ping():
                    current_app.logger.info("Redis connection successful with fallback to 127.0.0.1")
                    # Update the Redis URL
                    current_app.config['REDIS_URL'] = 'redis://127.0.0.1:6379/0'
                else:
                    raise ConnectionError("Redis ping failed on fallback")
            except Exception as e2:
                current_app.logger.error(f"Fallback Redis connection check failed: {e2}")
                return jsonify({
                    "status": "error",
                    "message": "Could not connect to the model creation service. Please try again later.",
                    "details": "Redis connection failed"
                }), 503
        
        # Submit task to Celery with explicit retry logic
        max_retries = 3
        retry_delay = 1
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Import the task directly to avoid circular imports
                from app.tasks import create_model_task
                
                task = create_model_task.delay(
                    current_user.username, 
                    site_no, 
                    ls_resolution, 
                    dem_resolution
                )
                
                current_app.logger.info(f"Model creation task {task.id} scheduled successfully.")
                return jsonify({
                    "status": "success", 
                    "message": "Model creation started!",
                    "task_id": task.id
                })
            except ConnectionError as ce:
                last_error = ce
                current_app.logger.error(f"Redis connection error on attempt {attempt+1}: {ce}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
            except Exception as e:
                last_error = e
                current_app.logger.error(f"Error scheduling task on attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        # If we got here, all retries failed
        current_app.logger.error(f"All retry attempts failed: {last_error}")
        return jsonify({
            "status": "error",
            "message": "Could not connect to the model creation service after multiple attempts. Please try again later.",
            "details": str(last_error)
        }), 503
            
    except Exception as e:
        current_app.logger.error(f"Error scheduling model creation: {e}")
        return jsonify({"error": f"Failed to start model creation: {str(e)}"}), 500

# Support both API and non-API versions for model confirmation
@model_bp.route('/api/model-confirmation')
@model_bp.route('/model-confirmation')
@conditional_login_required
@conditional_verified_required
def model_confirmation():
    current_app.logger.info("Model Confirmation route called")
    return jsonify({"title": "Model Confirmation", "message": "Model confirmation page"})

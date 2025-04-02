from flask import Blueprint, jsonify, request, current_app
from flask_login import current_user
import pandas as pd
import ast
from datetime import datetime
from app.decorators import conditional_login_required, conditional_verified_required
from app.utils import (find_station, check_existing_models, get_huc12_geometries,
                      get_huc12_streams_geometries, get_huc12_lakes_geometries)
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
from app.extensions import csrf
import geopandas as gpd
from app.comm_utils import send_model_start_email, check_redis_health

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
@csrf.exempt
def model_settings():
        
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
        
        try:
            # Use the consolidated Redis health check
            redis_health = check_redis_health()
            
            if redis_health['healthy']:
                current_app.logger.info(f"Redis connection successful: {redis_health['message']}")
                redis_working = True
                # Redis URL is already updated in the health check function
            else:
                current_app.logger.error(f"Redis health check failed: {redis_health['message']}")
                return jsonify({
                    "status": "error",
                    "message": "The model creation service is currently unavailable",
                    "details": redis_health['message']
                }), 503
                
        except ImportError as e:
            current_app.logger.error(f"Redis module not available: {e}")
            return jsonify({
                "status": "error",
                "message": "Model creation service configuration issue",
                "details": "Required modules not available"
            }), 500
            
        # Check active tasks for this user, limit to 5 concurrent tasks
        try:
            from app.task_tracker import task_tracker
            user_tasks = task_tracker.get_user_tasks(current_user.username)
            active_tasks = [t for t in user_tasks if t.get('status') not in 
                           [task_tracker.STATUS_SUCCESS, task_tracker.STATUS_FAILURE, task_tracker.STATUS_REVOKED]]
            
            if len(active_tasks) >= 5:
                current_app.logger.warning(f"User {current_user.username} has {len(active_tasks)} active tasks, limit is 5")
                return jsonify({
                    "status": "error",
                    "message": "You have reached the maximum limit of 5 concurrent model creation tasks. Please wait for some of your existing tasks to complete."
                }), 429
            
            current_app.logger.info(f"User {current_user.username} has {len(active_tasks)} active tasks (under limit)")
        except Exception as e:
            current_app.logger.error(f"Error checking active tasks: {e}")
            # Continue processing - we don't want to block model creation if task tracking fails
        
        # Celery task creation with retry logic
        from app.swatgenx_tasks import create_model_task
        import time
        from datetime import datetime
        
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
                
                # Add task to the task tracker if not already registered by the task
                try:
                    from app.task_tracker import task_tracker
                    task_info = task_tracker.get_task_status(task.id)
                    
                    if not task_info:
                        # Register task if not already registered by the task itself
                        task_tracker.register_task(
                            task.id,
                            current_user.username,
                            site_no,
                            {
                                "ls_resolution": ls_resolution,
                                "dem_resolution": dem_resolution,
                                "start_time": time.time(),
                                "source": "api_submission",
                                "api_time": datetime.now().isoformat()
                            }
                        )
                except Exception as track_error:
                    current_app.logger.error(f"Error registering task with tracker: {track_error}")
                    # Continue processing - we don't want to block model creation if task tracking fails
                
                # Send email notification for task start
                try:
                    # Prepare model info for email
                    model_info = {
                        "Site Number": site_no,
                        "LS Resolution": ls_resolution,
                        "DEM Resolution": dem_resolution,
                        "Request Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Send the email notification using our consolidated email utility
                    if current_user.email:
                        email_sent = send_model_start_email(
                            current_user.username,
                            current_user.email,
                            site_no,
                            model_info,
                            task.id
                        )
                        if email_sent:
                            current_app.logger.info(f"Model start email sent to {current_user.email}")
                        else:
                            current_app.logger.warning(f"Failed to send model start email to {current_user.email}")
                    else:
                        current_app.logger.warning(f"No email address found for user {current_user.username}")
                except Exception as email_error:
                    current_app.logger.error(f"Error sending model start email: {str(email_error)}")
                    # Continue with the response even if email fails
                
                response = jsonify({
                    "status": "success", 
                    "message": "Model creation started",
                    "task_id": task.id,
                    "queue_position": len(active_tasks) + 1
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

@model_bp.route('/api/get_station_geometries', methods=['GET'])
def get_station_geometries():
    """Fetch all station geometries for map-based selection."""
    current_app.logger.info("Fetching station geometries for map display")
    
    try:
        import geopandas as gpd
        import pandas as pd
        import os
        import json
        from time import time
        
        # Create a cache key based on the file's modification time
        FPS_geometry_name_shp_path = SWATGenXPaths.FPS_CONUS_stations
        
        # Check if file exists
        if not os.path.exists(FPS_geometry_name_shp_path):
            current_app.logger.error(f"Station shapefile not found: {FPS_geometry_name_shp_path}")
            return jsonify({"error": "Station data file not found"}), 404
            
        # Add cache headers to improve performance
        cache_timeout = 3600  # 1 hour cache
        response = None
        
        try:
            # Read station geometries from shapefile with optimization
            start_time = time()
            gdf = gpd.read_file(FPS_geometry_name_shp_path)
            current_app.logger.info(f"Read shapefile in {time() - start_time:.2f} seconds")
            
            # Convert to a simplified GeoJSON structure with only necessary fields
            stations_geojson = {
                "type": "FeatureCollection",
                "features": []
            }
            
            # Process each station with optimized data handling
            start_time = time()
            for idx, row in gdf.iterrows():
                try:
                    # Extract geometry and properties
                    geometry = row.geometry.__geo_interface__ if row.geometry else None
                    
                    # Only include points with valid geometry
                    if geometry:
                        feature = {
                            "type": "Feature",
                            "geometry": geometry,
                            "properties": {
                                "SiteNumber": str(row.get("SiteNumber", "")),  # Ensure it's a string
                                "SiteName": str(row.get("SiteName", "")),
                                "id": idx
                            }
                        }
                        stations_geojson["features"].append(feature)
                except Exception as e:
                    current_app.logger.warning(f"Error processing station at index {idx}: {e}")
                    continue
            
            current_app.logger.info(f"Processed {len(stations_geojson['features'])} stations in {time() - start_time:.2f} seconds")
            
            # Create response with caching headers
            response = jsonify(stations_geojson)
            response.cache_control.max_age = cache_timeout
            response.cache_control.public = True
            
            current_app.logger.info(f"Returning {len(stations_geojson['features'])} station geometries with cache for {cache_timeout} seconds")
            return response
            
        except Exception as e:
            current_app.logger.error(f"Error reading station shapefile: {e}")
            current_app.logger.error(f"Shapefile path: {FPS_geometry_name_shp_path}")
            raise
        
    except ImportError as e:
        current_app.logger.error(f"Missing library: {e}")
        return jsonify({"error": f"Server configuration error: {str(e)}"}), 500
    except Exception as e:
        import traceback
        current_app.logger.error(f"Error fetching station geometries: {e}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to fetch station geometries", "details": str(e)}), 500

@model_bp.route('/api/task_status/<task_id>', methods=['GET'])
@conditional_login_required
def get_task_status(task_id):
    """Get the status of a specific task."""
    current_app.logger.info(f"Task status check for task ID: {task_id}")
    
    try:
        from app.task_tracker import task_tracker
        task_info = task_tracker.get_task_status(task_id)
        
        if not task_info:
            current_app.logger.warning(f"Task {task_id} not found in tracker")
            
            # Fall back to Celery's status check if not in our tracker
            from app.swatgenx_tasks import create_model_task
            celery_status = create_model_task.AsyncResult(task_id)
            
            if celery_status.state:
                return jsonify({
                    "status": "success",
                    "task_status": celery_status.state,
                    "task_id": task_id,
                    "source": "celery_direct",
                    "result": str(celery_status.result) if celery_status.result else None
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Task not found"
                }), 404
        
        # Check if the user is authorized to view this task
        if task_info.get('username') != current_user.username and not current_user.is_admin:
            current_app.logger.warning(f"Unauthorized access to task {task_id} by user {current_user.username}")
            return jsonify({
                "status": "error",
                "message": "You are not authorized to view this task"
            }), 403
            
        return jsonify({
            "status": "success",
            "task_info": task_info
        })
        
    except Exception as e:
        current_app.logger.error(f"Error retrieving task status: {e}")
        return jsonify({
            "status": "error",
            "message": "Error retrieving task status",
            "details": str(e)
        }), 500

@model_bp.route('/api/user_tasks', methods=['GET'])
@conditional_login_required
def get_user_tasks():
    """Get all tasks for the current user."""
    current_app.logger.info(f"Fetching tasks for user: {current_user.username}")
    
    try:
        from app.task_tracker import task_tracker
        
        # Get task limit from query parameters (default: 50)
        limit = request.args.get('limit', 50, type=int)
        
        # Get status filter from query parameters (optional)
        status_filter = request.args.get('status', None)
        
        # Get tasks for the user
        user_tasks = task_tracker.get_user_tasks(current_user.username, limit=limit)
        
        # Apply status filter if provided
        if status_filter:
            user_tasks = [t for t in user_tasks if t.get('status') == status_filter]
        
        # Sort tasks by creation time (newest first)
        user_tasks.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jsonify({
            "status": "success",
            "task_count": len(user_tasks),
            "tasks": user_tasks
        })
        
    except Exception as e:
        current_app.logger.error(f"Error retrieving user tasks: {e}")
        return jsonify({
            "status": "error",
            "message": "Error retrieving user tasks",
            "details": str(e)
        }), 500

@model_bp.route('/api/active_tasks', methods=['GET'])
@conditional_login_required
def get_active_tasks():
    """Get all active tasks in the system (admin only)."""
    current_app.logger.info(f"Active tasks check by user: {current_user.username}")
    
    # Check if user is admin (implement your admin check here)
    is_admin = current_user.username in ['admin', 'swatgenx'] 
    if not is_admin:
        current_app.logger.warning(f"Non-admin user {current_user.username} attempted to access active tasks")
        return jsonify({
            "status": "error",
            "message": "Admin access required"
        }), 403
    
    try:
        from app.task_tracker import task_tracker
        
        # Get limit from query parameters (default: 50)
        limit = request.args.get('limit', 50, type=int)
        
        # Get all active tasks
        active_tasks = task_tracker.get_active_tasks(limit=limit)
        
        return jsonify({
            "status": "success",
            "task_count": len(active_tasks),
            "tasks": active_tasks
        })
        
    except Exception as e:
        current_app.logger.error(f"Error retrieving active tasks: {e}")
        return jsonify({
            "status": "error",
            "message": "Error retrieving active tasks",
            "details": str(e)
        }), 500

@model_bp.route('/api/task_dashboard', methods=['GET'])
@conditional_login_required
def task_dashboard():
    """Dashboard showing task status and statistics."""
    current_app.logger.info(f"Task dashboard accessed by user: {current_user.username}")
    
    return jsonify({
        "status": "success",
        "title": "Task Dashboard",
        "message": "Dashboard showing all your model creation tasks"
    })

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
import numpy as np
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

    # Find the row with that station - fix the boolean comparison issue
    station_row = station_data[station_data.SiteNumber.astype(str) == str(station_no)]
    if station_row.empty:
        current_app.logger.error(f"Station {station_no} not found in CSV.")
        return jsonify({"error": "Station not found"}), 404

    # Convert row to dict - Including all available columns
    characteristics = station_row.iloc[0].to_dict()
    
    # Rename some fields for better readability in the frontend
    field_mappings = {
        'Drainage area (sqkm)': 'DrainageArea',
        'Number of expected records (1999-2022)': 'ExpectedRecords',
        'Streamflow records gap (1999-2022) (%)': 'StreamflowGapPercent',
        'HUC12 id of the station': 'StationHUC12',
    }
    
    # Apply field renaming for better frontend usage
    for old_name, new_name in field_mappings.items():
        if old_name in characteristics:
            characteristics[new_name] = characteristics.pop(old_name)
    
    current_app.logger.info(f"Found station row for {station_no}")

    # Safely parse the HUC12 list from the CSV field
    # The CSV field looks like: "['040500040703','040500040508', ... ]"
    huc12_str = characteristics.get('HUC12 ids of the watershed')
    
    # Explicitly check if the value is NaN or None
    if huc12_str is None or (isinstance(huc12_str, float) and pd.isna(huc12_str)):
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
    
    # Clean up NaN values for JSON serialization
    for key, value in characteristics.items():
        if isinstance(value, (float, int, bool)) and pd.isna(value):
            characteristics[key] = None
        elif isinstance(value, np.ndarray):
            # Convert numpy arrays to lists for JSON serialization
            characteristics[key] = value.tolist()
        elif pd.api.types.is_scalar(value) and pd.isna(value):
            # Handle other scalar NaN values
            characteristics[key] = None
            
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
        import sys
        
        # Create a cache key based on the file's modification time
        FPS_geometry_name_shp_path = SWATGenXPaths.FPS_CONUS_stations
        
        # Check if file exists
        if not os.path.exists(FPS_geometry_name_shp_path):
            current_app.logger.error(f"Station shapefile not found: {FPS_geometry_name_shp_path}")
            return jsonify({"error": "Station data file not found"}), 404
            
        # Get file modification time for ETag generation
        try:
            mtime = os.path.getmtime(FPS_geometry_name_shp_path)
            etag = f'"{int(mtime)}"'
            
            # Check If-None-Match header for conditional request
            if_none_match = request.headers.get('If-None-Match')
            if if_none_match and if_none_match == etag:
                current_app.logger.info("Returning 304 Not Modified - client cache is valid")
                return "", 304
                
        except (OSError, IOError) as e:
            current_app.logger.warning(f"Error getting file modification time: {e}")
            # Continue without ETag if we can't get mtime
            etag = None
        
        # Add cache headers to improve performance
        cache_timeout = 86400  # 24 hour cache (increased from 1 hour)
        response = None
        
        try:
            # Read station geometries from shapefile with optimization and memory management
            start_time = time()
            # Set low_memory=False to avoid mixed type warnings
            gpd.options.io_engine = "pyogrio"  # Use faster engine if available
            
            # Try to limit memory usage by only reading necessary columns
            try:
                gdf = gpd.read_file(
                    FPS_geometry_name_shp_path, 
                    columns=["SiteNumber", "SiteName", "geometry"]
                )
            except Exception as col_error:
                current_app.logger.warning(f"Couldn't read with column filter, falling back to full read: {col_error}")
                gdf = gpd.read_file(FPS_geometry_name_shp_path)
                
            current_app.logger.info(f"Read shapefile in {time() - start_time:.2f} seconds, found {len(gdf)} stations")
            
            # Convert to a simplified GeoJSON structure with only necessary fields
            stations_geojson = {
                "type": "FeatureCollection",
                "features": []
            }
            
            # Process each station with optimized data handling
            start_time = time()
            feature_count = 0
            error_count = 0
            
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
                                "id": int(idx)
                            }
                        }
                        stations_geojson["features"].append(feature)
                        feature_count += 1
                except Exception as e:
                    error_count += 1
                    if error_count < 10:  # Only log first few errors to avoid filling logs
                        current_app.logger.warning(f"Error processing station at index {idx}: {e}")
                    
                # Periodically force garbage collection to avoid memory issues
                if idx % 5000 == 0 and idx > 0:
                    import gc
                    gc.collect()
            
            current_app.logger.info(
                f"Processed {feature_count} stations in {time() - start_time:.2f} seconds "
                f"with {error_count} errors"
            )
            
            # Verify we have features to return
            if len(stations_geojson["features"]) == 0:
                current_app.logger.error("No valid station features found in the shapefile")
                return jsonify({"error": "No valid station features found"}), 500
            
            # Create response with caching headers
            response = jsonify(stations_geojson)
            
            # Set aggressive caching headers to prevent frequent reloads
            if etag:
                response.headers["ETag"] = etag
            
            # Set Cache-Control and Expires headers 
            response.cache_control.max_age = cache_timeout
            response.cache_control.public = True
            
            # Set Expires header for older browsers/proxies
            from datetime import datetime, timedelta
            from email.utils import formatdate
            expires_time = datetime.now() + timedelta(seconds=cache_timeout)
            response.headers["Expires"] = formatdate(expires_time.timestamp(), localtime=False, usegmt=True)
            
            # Set Last-Modified header
            if mtime:
                from datetime import datetime  
                last_modified = datetime.fromtimestamp(mtime)
                response.headers["Last-Modified"] = formatdate(last_modified.timestamp(), localtime=False, usegmt=True)
            
            current_app.logger.info(
                f"Returning {len(stations_geojson['features'])} station geometries "
                f"with cache for {cache_timeout} seconds"
            )
            return response
            
        except Exception as e:
            current_app.logger.error(f"Error processing station shapefile: {e}")
            current_app.logger.error(f"Shapefile path: {FPS_geometry_name_shp_path}")
            import traceback
            current_app.logger.error(traceback.format_exc())
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

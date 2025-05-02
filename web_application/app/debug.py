from flask import Blueprint, jsonify, current_app, request, send_from_directory, url_for
import os
import platform
import sys
import datetime
import inspect
import json
from app.extensions import csrf
from flask_login import current_user
import traceback
import time

# Combined diagnostic and debug blueprint
debug_bp = Blueprint('debug', __name__, url_prefix='/api/debug')

# Apply CSRF exemption to the entire blueprint
csrf.exempt(debug_bp)

# ======= DIAGNOSTIC ROUTES =======
@debug_bp.route('/echo', methods=['POST', 'OPTIONS'])
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

@debug_bp.route('/status', methods=['GET'])
def status():
    """Simple status endpoint to verify API access"""
    current_app.logger.info(f"Status endpoint called by {request.remote_addr}")
    return jsonify({
        "status": "ok",
        "time": datetime.datetime.now().isoformat(),
        "environment": os.environ.get('FLASK_ENV', 'not set')
    })

# Add route to test model settings specifically
@debug_bp.route('/test-model-settings', methods=['POST', 'OPTIONS'])
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
@debug_bp.route('/file-access', methods=['GET'])
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
        "images": current_app.config['STATIC_IMAGES_PATH'],
        "videos": current_app.config['STATIC_VIDEOS_PATH'],
        "visualizations": current_app.config['VISUALIZATION_PATH'],
        "user_files": current_app.config['USER_PATH']
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

@debug_bp.route('/serve-test-file', methods=['GET'])
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

# ======= DEBUG ROUTES =======
@debug_bp.route('/routes', methods=['GET'])
def list_routes():
    """
    Debug endpoint to list all registered routes in the application.
    This helps identify routing issues and mismatches.
    """
    app = current_app._get_current_object()
    routes = []
    
    # Get all registered routes from the Flask app
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': sorted([method for method in rule.methods if method not in ('HEAD', 'OPTIONS')]),
            'path': str(rule),
            'arguments': sorted([arg for arg in rule.arguments]),
        })
    
    # Sort by path for easier reading
    routes.sort(key=lambda r: r['path'])
    
    return jsonify({
        'total_routes': len(routes),
        'routes': routes
    })

@debug_bp.route('/request', methods=['GET', 'POST', 'PUT', 'DELETE'])
def debug_request():
    """
    Debug endpoint that echoes request information.
    Useful for troubleshooting client requests.
    """
    # Basic request info
    info = {
        'method': request.method,
        'url': request.url,
        'path': request.path,
        'remote_addr': request.remote_addr,
        'headers': dict(request.headers),
    }
    
    # Get query parameters
    info['query_params'] = dict(request.args)
    
    # Get form data if present
    if request.form:
        info['form_data'] = dict(request.form)
    
    # Get JSON data if present
    if request.is_json:
        info['json_data'] = request.get_json()
    
    # Get files if present
    if request.files:
        info['files'] = [f for f in request.files]
    
    # Add this to logs for server-side debugging
    current_app.logger.info(f"Debug request: {info}")
    
    return jsonify(info)

@debug_bp.route('/reports', methods=['GET'])
def debug_reports():
    """Debug endpoint to inspect the structure of reports directory"""
    reports_dir = current_app.config['USER_PATH']
    users = {}
    
    try:
        # Get list of users with reports
        for user in os.listdir(reports_dir):
            user_report_dir = os.path.join(reports_dir, user, "Reports")
            if os.path.exists(user_report_dir) and os.path.isdir(user_report_dir):
                users[user] = []
                try:
                    # List reports for this user
                    for report_id in os.listdir(user_report_dir):
                        report_path = os.path.join(user_report_dir, report_id)
                        if os.path.isdir(report_path):
                            report_info = {
                                "report_id": report_id,
                                "path": report_path,
                                "files": [],
                                "has_metadata": os.path.exists(os.path.join(report_path, "metadata.json")),
                                "has_error": os.path.exists(os.path.join(report_path, "error.json"))
                            }
                            
                            # Get first 5 files in report directory as sample
                            try:
                                files = [f for f in os.listdir(report_path) if os.path.isfile(os.path.join(report_path, f))][:5]
                                report_info["files"] = files
                            except Exception as e:
                                report_info["file_error"] = str(e)
                                
                            users[user].append(report_info)
                except Exception as e:
                    users[user] = {"error": str(e)}
        
        return jsonify({
            "status": "success",
            "reports_dir": reports_dir,
            "users": users
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@debug_bp.route('/check_report/<report_id>', methods=['GET'])
def debug_check_report(report_id):
    """Debug endpoint to check a specific report"""
    # Get username from query parameters or use a default
    username = request.args.get('username', 'admin')
    report_dir = os.path.join(current_app.config['USER_PATH'], username, "Reports", report_id)
    
    if not os.path.exists(report_dir) or not os.path.isdir(report_dir):
        return jsonify({
            "status": "error",
            "message": f"Report directory not found: {report_dir}"
        }), 404
    
    # Collect information about the report
    report_info = {
        "report_id": report_id,
        "path": report_dir,
        "exists": os.path.exists(report_dir),
        "is_dir": os.path.isdir(report_dir),
        "readable": os.access(report_dir, os.R_OK),
        "files": [],
        "metadata": None,
        "error_info": None
    }
    
    # Check for metadata and error files
    metadata_path = os.path.join(report_dir, 'metadata.json')
    error_path = os.path.join(report_dir, 'error.json')
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                report_info["metadata"] = json.load(f)
        except Exception as e:
            report_info["metadata_error"] = str(e)
    
    if os.path.exists(error_path):
        try:
            with open(error_path, 'r') as f:
                report_info["error_info"] = json.load(f)
        except Exception as e:
            report_info["error_load_error"] = str(e)
    
    # List files in the report directory
    try:
        for item in os.listdir(report_dir):
            item_path = os.path.join(report_dir, item)
            if os.path.isfile(item_path):
                file_info = {
                    "name": item,
                    "size": os.path.getsize(item_path),
                    "last_modified": datetime.datetime.fromtimestamp(os.path.getmtime(item_path)).isoformat()
                }
                report_info["files"].append(file_info)
    except Exception as e:
        report_info["file_listing_error"] = str(e)
    
    return jsonify(report_info)

@debug_bp.route('/report_contents/<report_id>', methods=['GET'])
def debug_report_contents(report_id):
    """List all files in a report directory with their paths"""
    # Get username from query parameters or use a default
    username = request.args.get('username', 'admin')
    report_dir = os.path.join(current_app.config['USER_PATH'], username, "Reports", report_id)
    
    if not os.path.exists(report_dir) or not os.path.isdir(report_dir):
        return jsonify({
            "status": "error",
            "message": f"Report directory not found: {report_dir}"
        }), 404
    
    # Collect file information recursively
    file_list = []
    try:
        for root, dirs, files in os.walk(report_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, report_dir)
                
                file_info = {
                    "name": file,
                    "path": rel_path,
                    "abs_path": file_path,
                    "size": os.path.getsize(file_path),
                    "last_modified": datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                }
                
                # For small text files, include content for debugging
                if os.path.getsize(file_path) < 10240 and file.endswith(('.txt', '.json', '.html', '.csv')):
                    try:
                        with open(file_path, 'r', errors='replace') as f:
                            file_info["content"] = f.read()
                    except Exception as e:
                        file_info["content_error"] = str(e)
                
                file_list.append(file_info)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
    return jsonify({
        "report_id": report_id,
        "path": report_dir,
        "file_count": len(file_list),
        "files": file_list
    })

@debug_bp.route('/system-info', methods=['GET'])
def system_info():
    """Return system information for debugging"""
    info = {
        "time": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "current_directory": os.getcwd(),
        "environment": {
            "FLASK_ENV": os.environ.get('FLASK_ENV', 'Not set'),
            "FLASK_APP": os.environ.get('FLASK_APP', 'Not set'),
            "PYTHONPATH": os.environ.get('PYTHONPATH', 'Not set'),
            "PATH": os.environ.get('PATH', 'Not set')[:100] + '...',  # Truncate for readability
        },
        "user": {
            "authenticated": current_user.is_authenticated,
            "username": current_user.username if current_user.is_authenticated else None
        }
    }
    return jsonify(info)

@debug_bp.route('/celery-config', methods=['GET'])
def celery_config():
    """Return Celery configuration for debugging"""
    try:
        # Import Celery instance
        from celery_app import celery
        
        # Get important configuration settings
        config = {
            "broker_url": celery.conf.broker_url,
            "result_backend": celery.conf.result_backend,
            "broker_connection_retry": celery.conf.broker_connection_retry,
            "broker_connection_retry_on_startup": celery.conf.broker_connection_retry_on_startup,
            "worker_prefetch_multiplier": celery.conf.worker_prefetch_multiplier,
            "task_default_rate_limit": celery.conf.task_default_rate_limit,
            "worker_disable_rate_limits": celery.conf.worker_disable_rate_limits,
            "task_time_limit": celery.conf.task_time_limit,
            "task_soft_time_limit": celery.conf.task_soft_time_limit,
            "worker_max_tasks_per_child": celery.conf.worker_max_tasks_per_child,
            "worker_max_memory_per_child": celery.conf.worker_max_memory_per_child,
        }
        
        # Get environment variables related to Celery
        celery_env_vars = {}
        for key, value in os.environ.items():
            if key.startswith('CELERY_'):
                celery_env_vars[key] = value
        
        return jsonify({
            "status": "success",
            "celery_config": config,
            "celery_environment_variables": celery_env_vars
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error accessing Celery configuration: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@debug_bp.route('/redis-check', methods=['GET'])
def redis_check():
    """Test Redis connection and report status"""
    try:
        from redis import Redis
        
        redis_url = current_app.config.get('REDIS_URL', 'redis://localhost:6379/0')
        redis_client = Redis.from_url(redis_url, socket_timeout=3)
        
        # Test ping
        start_time = time.time()
        ping_result = redis_client.ping()
        ping_time = time.time() - start_time
        
        # Test set/get
        test_key = f"debug_test_{time.time()}"
        start_time = time.time()
        redis_client.set(test_key, "test_value", ex=60)
        get_result = redis_client.get(test_key)
        operation_time = time.time() - start_time
        
        # Clean up
        redis_client.delete(test_key)
        
        return jsonify({
            "status": "success",
            "redis_url": redis_url,
            "ping_success": ping_result,
            "ping_time_ms": round(ping_time * 1000, 2),
            "get_set_success": get_result == b"test_value",
            "get_set_time_ms": round(operation_time * 1000, 2),
            "message": "Redis is working correctly"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Redis connection error: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@debug_bp.route('/test-task', methods=['POST'])
def test_task():
    """Launch a test task to verify Celery configuration"""
    try:
        # Import task directly from swatgenx_tasks
        from app.swatgenx_tasks import create_model_task
        
        # Get parameters or use defaults
        data = request.get_json() or {}
        username = data.get('username', current_user.username if current_user.is_authenticated else 'test_user')
        site_no = data.get('site_no', '06853800')  # Default to a known test site
        ls_resolution = data.get('ls_resolution', 250)
        dem_resolution = data.get('dem_resolution', 30)
        
        # Launch task with apply_async to test rate limiting
        task = create_model_task.apply_async(
            args=[username, site_no, ls_resolution, dem_resolution],
            queue='model_creation'
        )
        
        return jsonify({
            "status": "success",
            "message": "Test task launched successfully",
            "task_id": task.id,
            "parameters": {
                "username": username,
                "site_no": site_no,
                "ls_resolution": ls_resolution,
                "dem_resolution": dem_resolution
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error launching test task: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@debug_bp.route('/workers', methods=['GET'])
def get_workers():
    """Get active Celery workers and their status"""
    try:
        from celery_app import celery
        
        # Get active workers using Celery's inspect API
        inspect = celery.control.inspect()
        
        # Get stats for each worker
        stats = inspect.stats() or {}
        active = inspect.active() or {}
        registered = inspect.registered() or {}
        
        # Create structured response
        workers_info = {}
        for worker_name in set(list(stats.keys()) + list(active.keys()) + list(registered.keys())):
            workers_info[worker_name] = {
                "status": "online" if worker_name in stats else "offline",
                "stats": stats.get(worker_name, {}),
                "active_tasks": len(active.get(worker_name, [])),
                "registered_tasks": len(registered.get(worker_name, []))
            }
        
        return jsonify({
            "status": "success",
            "active_workers_count": len(stats),
            "workers": workers_info
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error getting worker information: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@debug_bp.route('/api/debug/email-check', methods=['GET'])
def email_check():
    """Check email configuration"""
    try:
        import smtplib
        from email.mime.text import MIMEText
        
        # Define SMTP settings
        smtp_server = "express.mail.msu.edu"
        smtp_port = 25
        
        # Test connection only (no sending)
        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
            # Try an EHLO command to check connection
            ehlo_response = server.ehlo()
            
            # Get server features
            smtp_features = {}
            if hasattr(server, 'esmtp_features'):
                smtp_features = server.esmtp_features
                
            return jsonify({
                "status": "success",
                "message": "Successfully connected to SMTP server",
                "smtp_server": smtp_server,
                "smtp_port": smtp_port,
                "ehlo_response": str(ehlo_response),
                "smtp_features": {k.decode() if isinstance(k, bytes) else k: 
                                 v.decode() if isinstance(v, bytes) else v 
                                 for k, v in smtp_features.items()}
            })
    except Exception as e:
        current_app.logger.error(f"Email configuration error: {e}")
        import traceback
        return jsonify({
            "status": "error",
            "message": f"Email configuration error: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@debug_bp.route('/api/debug/email-logs', methods=['GET'])
def email_logs():
    """Check recent email logs"""
    try:
        # Try to read the last few lines of the mail log
        log_file = "/var/log/mail.log"
        mail_log_content = []
        
        if os.path.exists(log_file):
            try:
                # Get last 50 lines
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    mail_log_content = lines[-50:] if len(lines) > 50 else lines
            except Exception as e:
                mail_log_content = [f"Error reading mail.log: {str(e)}"]
        else:
            mail_log_content = ["Mail log file not found"]
            
        # Try alternative location if first wasn't found
        if not os.path.exists(log_file):
            alternative_logs = ["/var/log/maillog", "/var/log/exim4/mainlog"]
            for alt_log in alternative_logs:
                if os.path.exists(alt_log):
                    try:
                        with open(alt_log, 'r') as f:
                            lines = f.readlines()
                            mail_log_content = lines[-50:] if len(lines) > 50 else lines
                        log_file = alt_log
                        break
                    except Exception as e:
                        mail_log_content.append(f"Error reading {alt_log}: {str(e)}")
        
        # Try application email logs
        app_email_log = os.path.join(current_app.config['LOG_PATH'], "email.log")
        app_email_content = []
        if os.path.exists(app_email_log):
            try:
                with open(app_email_log, 'r') as f:
                    lines = f.readlines()
                    app_email_content = lines[-50:] if len(lines) > 50 else lines
            except Exception as e:
                app_email_content = [f"Error reading app email log: {str(e)}"]
        
        return jsonify({
            "status": "success",
            "message": "Email logs retrieved",
            "system_mail_log": {
                "path": log_file,
                "exists": os.path.exists(log_file),
                "content": mail_log_content
            },
            "app_email_log": {
                "path": app_email_log,
                "exists": os.path.exists(app_email_log),
                "content": app_email_content
            }
        })
    except Exception as e:
        current_app.logger.error(f"Error checking email logs: {e}")
        import traceback
        return jsonify({
            "status": "error",
            "message": f"Error checking email logs: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

def register_debug_routes(app):
    """Register the debug routes with the Flask app"""
    debug_enabled = os.environ.get('FLASK_ENV') != 'production'
    if debug_enabled:
        app.register_blueprint(debug_bp)
        app.logger.info("Debug routes enabled")
    else:
        app.logger.info("Debug routes disabled in production")
from flask import Blueprint, jsonify, request, current_app
import inspect
import os

routes_debug = Blueprint('routes_debug', __name__)

@routes_debug.route('/api/debug/routes', methods=['GET'])
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

@routes_debug.route('/api/debug/request', methods=['GET', 'POST', 'PUT', 'DELETE'])
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

def register_debug_routes(app):
    """Register the debug routes with the Flask app"""
    if os.environ.get('FLASK_ENV') == 'development' or os.environ.get('DEBUG') == '1':
        app.register_blueprint(routes_debug)
        app.logger.info("Debug routes registered")
    else:
        app.logger.info("Debug routes not registered (not in development mode)")

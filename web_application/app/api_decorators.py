import functools
import logging
import os
from flask import request, jsonify, current_app

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler(os.path.join('/data/SWATGenXApp/codes/web_application/logs', 'ApiDecorators.log'))
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.info("Logger initialized: /data/SWATGenXApp/codes/web_application/logs/ApiDecorators.log")

def require_api_key(view_function):
    """
    Decorator that checks for a valid API key in the request.
    API key can be provided in headers as X-API-Key or as a query parameter 'api_key'.
    """
    @functools.wraps(view_function)
    def decorated_function(*args, **kwargs):
        # Get API key from header or query parameter
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        # Check if API key is provided
        if not api_key:
            logger.warning("API request missing API key")
            return jsonify({'status': 'error', 'message': 'API key is required'}), 401
            
        # Validate the API key (could be enhanced with database lookup)
        valid_api_keys = current_app.config.get('API_KEYS', [])
        master_key = current_app.config.get('MASTER_API_KEY')
        
        if api_key == master_key or api_key in valid_api_keys:
            logger.info(f"Valid API key used for request to {request.path}")
            return view_function(*args, **kwargs)
        else:
            logger.warning(f"Invalid API key attempted: {api_key[:5]}...")
            return jsonify({'status': 'error', 'message': 'Invalid API key'}), 401
            
    return decorated_function

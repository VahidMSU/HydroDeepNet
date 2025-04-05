import os
import json
from datetime import timedelta
import sys
sys.path.append('/data/SWATGenXApp/codes/SWATGenX')
sys.path.append('/data/SWATGenXApp/codes/AI_agent')
sys.path.append('/data/SWATGenXApp/codes/MODFLOW')

from SWATGenX.SWATGenXLogging import LoggerSetup

class Config:
    logger = LoggerSetup("/data/SWATGenXApp/codes/web_application/logs", rewrite=False)
    logger = logger.setup_logger("Config")
    
    # Shibboleth SSO Configuration
    SAML_METADATA_URL = "https://login.msu.edu/idp/shibboleth"
    SAML_SP_ENTITY_ID = "https://ciwre-bae.campusad.msu.edu/shibboleth"
    SAML_SP_METADATA = "/etc/shibboleth/sp-metadata.xml"
    SAML_SP_PRIVATE_KEY = "/etc/shibboleth/sp-key.pem"
    SAML_SP_CERTIFICATE = "/etc/shibboleth/sp-cert.pem"
    SAML_LOGIN_REDIRECT = "/sso/login"
    SAML_LOGOUT_REDIRECT = "/sso/logout"

    # Determine the site URL based on environment
    ENV = os.getenv('FLASK_ENV', 'production')
    if ENV == 'development':
        SITE_URL = 'http://localhost:3000'
    else:
        SITE_URL = 'https://ciwre-bae.campusad.msu.edu'
    
    logger.info(f"Using SITE_URL: {SITE_URL}")
    
    BASE_PATH = os.getenv('BASE_PATH', '/data/SWATGenXApp/GenXAppData/')
    USGS_PATH = os.getenv('USGS_PATH', '/data/SWATGenXApp/GenXAppData/USGS')
    try:
        # Read the secret key from a file or environment variable
        with open('/data/SWATGenXApp/codes/ciwre-bae-crs/ciwre-bae.campusad.msu.edu.key') as f:
            SECRET_KEY = f.read().strip()
            logger.info("Secret key file found")
    except FileNotFoundError:
        logger.error("Secret key file not found")
        SECRET_KEY = os.getenv
    
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///site.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)  # Extend session lifetime as needed
    
    # Ensure secure session management
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Redis configuration
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # Celery configuration with environment variables for better maintainability
    # Worker configuration
    CELERY_WORKER_COUNT = int(os.getenv('CELERY_WORKER_COUNT', '30'))
    CELERY_WORKER_CONCURRENCY = int(os.getenv('CELERY_WORKER_CONCURRENCY', '8'))
    CELERY_MAX_TASKS_PER_CHILD = int(os.getenv('CELERY_MAX_TASKS_PER_CHILD', '100'))
    CELERY_MAX_MEMORY_PER_CHILD_MB = int(os.getenv('CELERY_MAX_MEMORY_PER_CHILD_MB', '8192'))  # 8GB
    
    # Task execution settings
    CELERY_WORKER_PREFETCH_MULTIPLIER = int(os.getenv('CELERY_WORKER_PREFETCH_MULTIPLIER', '8'))
    CELERY_TASK_SOFT_TIME_LIMIT = int(os.getenv('CELERY_TASK_SOFT_TIME_LIMIT', '43200'))  # 12 hours
    CELERY_TASK_TIME_LIMIT = int(os.getenv('CELERY_TASK_TIME_LIMIT', '86400'))  # 24 hours
    
    # Rate limiting configuration
    CELERY_DISABLE_RATE_LIMITS = os.getenv('CELERY_DISABLE_RATE_LIMITS', 'true').lower() == 'true'
    CELERY_TASK_DEFAULT_RATE_LIMIT = os.getenv('CELERY_TASK_DEFAULT_RATE_LIMIT', None)
    
    # Connection settings
    CELERY_BROKER_CONNECTION_RETRY = os.getenv('CELERY_BROKER_CONNECTION_RETRY', 'true').lower() == 'true'
    CELERY_BROKER_CONNECTION_MAX_RETRIES = int(os.getenv('CELERY_BROKER_CONNECTION_MAX_RETRIES', '20'))
    CELERY_REDIS_MAX_CONNECTIONS = int(os.getenv('CELERY_REDIS_MAX_CONNECTIONS', '500'))
    
    # Model creation queue configuration
    CELERY_MODEL_CREATION_WORKER_PERCENT = int(os.getenv('CELERY_MODEL_CREATION_WORKER_PERCENT', '70'))
    
    try:
        import redis
        redis_client = redis.StrictRedis.from_url(REDIS_URL)
        redis_client.ping()
        logger.info("Connected to Redis successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")

    # For debugging/production control
    PRIVATE_MODE = os.getenv('PRIVATE_MODE', 'true').lower() == 'true'

    if ENV == 'production':
        DEBUG = False
        TESTING = False
        # Make sure we're using the production URL
        SITE_URL = 'https://ciwre-bae.campusad.msu.edu'
    else:
        DEBUG = True
        TESTING = True
        SESSION_COOKIE_SECURE = False
        REMEMBER_COOKIE_SECURE = False
        SESSION_COOKIE_SAMESITE = None
        # Use localhost for development
        SITE_URL = 'http://localhost:3000'

    logger.info(f"ENV: {ENV}, PRIVATE_MODE: {PRIVATE_MODE}, SITE_URL: {SITE_URL}")

    # Google OAuth Configuration
    # Try to load from client_secrets file if available, otherwise use environment variables
    GOOGLE_CLIENT_ID = ''
    GOOGLE_CLIENT_SECRET = ''
    
    try:
        with open('/data/SWATGenXApp/codes/scripts/client_secret_952114684215-6gfiic7kq9i873vtdbp1hd8i7mp758uh.apps.googleusercontent.com.json') as f:
            client_info = json.load(f)
            if 'web' in client_info:
                GOOGLE_CLIENT_ID = client_info['web']['client_id']
                GOOGLE_CLIENT_SECRET = client_info['web']['client_secret']
                logger.info("Google OAuth credentials loaded from client secret file")
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Could not load Google client secrets from file: {e}")
        # Fall back to environment variables
        GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')
        GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET', '')
    
    # If we still don't have the credentials, log a warning
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        logger.warning("Google OAuth credentials not found in client secret file or environment variables")
    
    GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
    OAUTHLIB_INSECURE_TRANSPORT = os.getenv('FLASK_ENV', 'production') == 'development'

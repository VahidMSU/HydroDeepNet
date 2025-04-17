import os
import json
from datetime import timedelta
import sys
# Extend system paths
sys.path += [
    '/data/SWATGenXApp/codes/SWATGenX',
    '/data/SWATGenXApp/codes/GeoReporter',
    '/data/SWATGenXApp/codes/MODFLOW'
]
from SWATGenX.SWATGenXLogging import LoggerSetup



def load_secret_key(path, logger):
    try:
        with open(path) as f:
            logger.info("Secret key file loaded successfully")
            return f.read().strip()
    except FileNotFoundError:
        logger.error("Secret key file not found")
        return os.getenv('SECRET_KEY', 'fallback-secret')

def load_google_credentials(path, logger):
    try:
        with open(path) as f:
            data = json.load(f)
            creds = data.get('web', {})
            logger.info("Google OAuth credentials loaded")
            return creds.get('client_id', ''), creds.get('client_secret', '')
    except Exception as e:
        logger.warning(f"Failed to load Google credentials: {e}")
        return os.getenv('GOOGLE_CLIENT_ID', ''), os.getenv('GOOGLE_CLIENT_SECRET', '')

class Config:
    # Paths and Logging
    WEB_APP_PATH = os.getenv('WEB_APP_PATH', '/data/SWATGenXApp/codes/web_application')
    BASE_PATH = os.getenv('BASE_PATH', '/data/SWATGenXApp/GenXAppData/')
    USGS_PATH = os.getenv('USGS_PATH', '/data/SWATGenXApp/GenXAppData/USGS')
    USER_PATH = os.getenv('USER_PATH', '/data/SWATGenXApp/Users')
    DB_PATH = os.getenv('DB_PATH', '/data/SWATGenXApp/codes/web_application/instance/site.db')
    VISUALIZATION_PATH = os.getenv('VISUALIZATION_PATH', '/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/')
    LOG_PATH = os.getenv('LOG_PATH', '/data/SWATGenXApp/codes/web_application/logs')
    HYDROGEODATASET_PATH = os.getenv('HYDROGEODATASET_PATH', '/data/SWATGenXApp/GenXAppData/HydroGeoDataset/HydroGeoDataset_ML_250.h5')
    STATIC_IMAGES_PATH = os.getenv('STATIC_IMAGES_PATH', '/data/SWATGenXApp/GenXAppData/images')
    STATIC_VIDEOS_PATH = os.getenv('STATIC_VIDEOS_PATH', '/data/SWATGenXApp/GenXAppData/videos')

    logger = LoggerSetup(f"{WEB_APP_PATH}/logs", rewrite=False).setup_logger("Config")

    # Certificates and Secret Key
    SSL_CERT_FILE = "/data/SWATGenXApp/codes/ciwre-bae-crs/ciwre-bae.campusad.msu.edu.key"
    assert os.path.exists(SSL_CERT_FILE), f"SSL certificate not found: {SSL_CERT_FILE}"
    SECRET_KEY = load_secret_key(SSL_CERT_FILE, logger)

    # Shibboleth SSO Configuration
    SAML_METADATA_URL = "https://login.msu.edu/idp/shibboleth"
    SAML_SP_ENTITY_ID = "https://ciwre-bae.campusad.msu.edu/shibboleth"
    SAML_SP_METADATA = "/etc/shibboleth/sp-metadata.xml"
    SAML_SP_PRIVATE_KEY = "/etc/shibboleth/sp-key.pem"
    SAML_SP_CERTIFICATE = "/etc/shibboleth/sp-cert.pem"
    SAML_LOGIN_REDIRECT = "/sso/login"
    SAML_LOGOUT_REDIRECT = "/sso/logout"

    # Environment & Deployment Settings
    ENV = os.getenv('FLASK_ENV', 'production')
    DEBUG = ENV != 'production'
    TESTING = ENV != 'production'
    PRIVATE_MODE = os.getenv('PRIVATE_MODE', 'true').lower() == 'true'

    SITE_URL = {
        'production': 'https://ciwre-bae.campusad.msu.edu',
        'development': 'http://localhost:3000'
    }.get(ENV, 'http://localhost:3000')

    logger.info(f"ENV: {ENV}, PRIVATE_MODE: {PRIVATE_MODE}, SITE_URL: {SITE_URL}")

    # Session & Cookie Settings
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
    SESSION_COOKIE_SECURE = not DEBUG
    REMEMBER_COOKIE_SECURE = not DEBUG
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = None if DEBUG else 'Lax'

    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///site.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Redis
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    try:
        import redis
        redis.StrictRedis.from_url(REDIS_URL).ping()
        logger.info("Connected to Redis successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")

    # Celery
    CELERY_WORKER_COUNT = int(os.getenv('CELERY_WORKER_COUNT', 30))
    CELERY_WORKER_CONCURRENCY = int(os.getenv('CELERY_WORKER_CONCURRENCY', 8))
    CELERY_MAX_TASKS_PER_CHILD = int(os.getenv('CELERY_MAX_TASKS_PER_CHILD', 100))
    CELERY_MAX_MEMORY_PER_CHILD_MB = int(os.getenv('CELERY_MAX_MEMORY_PER_CHILD_MB', 8192))
    CELERY_WORKER_PREFETCH_MULTIPLIER = int(os.getenv('CELERY_WORKER_PREFETCH_MULTIPLIER', 8))
    CELERY_TASK_SOFT_TIME_LIMIT = int(os.getenv('CELERY_TASK_SOFT_TIME_LIMIT', 43200))
    CELERY_TASK_TIME_LIMIT = int(os.getenv('CELERY_TASK_TIME_LIMIT', 86400))
    CELERY_DISABLE_RATE_LIMITS = os.getenv('CELERY_DISABLE_RATE_LIMITS', 'true').lower() == 'true'
    CELERY_TASK_DEFAULT_RATE_LIMIT = os.getenv('CELERY_TASK_DEFAULT_RATE_LIMIT')
    CELERY_BROKER_CONNECTION_RETRY = os.getenv('CELERY_BROKER_CONNECTION_RETRY', 'true').lower() == 'true'
    CELERY_BROKER_CONNECTION_MAX_RETRIES = int(os.getenv('CELERY_BROKER_CONNECTION_MAX_RETRIES', 20))
    CELERY_REDIS_MAX_CONNECTIONS = int(os.getenv('CELERY_REDIS_MAX_CONNECTIONS', 500))
    CELERY_MODEL_CREATION_WORKER_PERCENT = int(os.getenv('CELERY_MODEL_CREATION_WORKER_PERCENT', 70))

    # Google OAuth
    GOOGLE_CLIENT_SECRET_FILE = '/data/SWATGenXApp/codes/scripts/google-services/client_secret_952114684215-6gfiic7kq9i873vtdbp1hd8i7mp758uh.apps.googleusercontent.com.json'
    assert os.path.exists(GOOGLE_CLIENT_SECRET_FILE), f"Google client secret file not found: {GOOGLE_CLIENT_SECRET_FILE}"

    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET = load_google_credentials(GOOGLE_CLIENT_SECRET_FILE, logger)
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        logger.warning("Google OAuth credentials are missing")

    GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
    OAUTHLIB_INSECURE_TRANSPORT = ENV == 'development'

    # API key (Optional Google Maps or other services)
    GOOGLE_API_KEY = 'AIzaSyCNvHkGEo33oOYffvAkBql6kC9ty7ijklM'

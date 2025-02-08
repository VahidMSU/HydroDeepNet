import os
from datetime import timedelta
import os
import sys
sys.path.append('/data/SWATGenXApp/codes/SWATGenX')

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
    
    SESSION_COOKIE_SECURE = True  # Ensure cookies are only sent over HTTPS
    REMEMBER_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True  # Mitigate XSS attacks
    
    # For debugging/production control
    DEBUG = True
    TESTING = True
    logger.info(f"Debug: {DEBUG}, Testing: {TESTING}")

from app import create_app
import logging
import os
import sys
from werkzeug.middleware.proxy_fix import ProxyFix

# Set up logging
apptype = "Flask-app"
sys.path.append('/data/SWATGenXApp/codes/SWATGenX')
from SWATGenX.SWATGenXLogging import LoggerSetup

logger = LoggerSetup("/data/SWATGenXApp/codes/web_application/logs", rewrite=False)
logger = logger.setup_logger(apptype)

# Ensure Matplotlib cache directory exists
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# Create Flask app
app = create_app(apptype)
logger.info("App created")


from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.middleware.proxy_fix import ProxyFix

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# **Expose `app` globally for Waitress and Apache**
application = app  # For Apache WSGI

if __name__ == '__main__':
    from waitress import serve
    logger.info("Starting server")
    serve(app, host='0.0.0.0', port=5050)

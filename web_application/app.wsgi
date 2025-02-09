from app import create_app
from werkzeug.middleware.proxy_fix import ProxyFix

# Create the Flask app
application = create_app()

# Apply ProxyFix middleware to handle proxy headers (e.g., X-Forwarded-For)
application.wsgi_app = ProxyFix(application.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)

# Log that the WSGI application has started
application.logger.info("WSGI application initialized via Apache.")

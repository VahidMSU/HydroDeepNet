from app import create_app
import logging
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'
logging.basicConfig(level=logging.INFO)

app = create_app()


if __name__ == '__main__':
    from waitress import serve
    logging.info("Starting server")
    serve(app, host='0.0.0.0', port=5500)

# curl -I https://ciwre-bae.campusad.msu.edu/get_options
# curl -I https://ciwre-bae.campusad.msu.edu:5500/
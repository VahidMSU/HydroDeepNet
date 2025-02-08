### SWATGenXApp Web Application Deployment Overview

#### Application Structure & Directories

- **Application Root Directory:** `/data/SWATGenXApp/`
- **Static Files Directory:** `/data/SWATGenXApp/GenXAppData/`
- **User Data Directory:** `/data/SWATGenXApp/Users/`

#### Web Server Management

- **Apache Server Status:** Run the command to check the status of the Apache server.
- **Flask Application Service (Development Mode):** Run the command to check the status of the Flask application service.
- **Accessing Flask Development Environment:**
  - Open a web browser and navigate to `http://127.0.0.1:5050/home`.
  - Ensure the page is opened using HTTP, not HTTPS.

#### Modifying Flask Service Configuration

- The Flask systemd service configuration file is located at:
  `/etc/systemd/system/flask-app.service`.
  Modify this file if changes to the Flask systemd settings are required.

#### Web Application Database

- **SQLite Database Path:**
  `/data/SWATGenXApp/codes/web_application/instance/site.db`

#### Apache Configuration Files

- **Primary Virtual Host Configuration:**
  File: `/etc/apache2/sites-available/ciwre-bae.conf`
- **Default Apache Configuration File:**
  File: `/etc/apache2/sites-available/000-default.conf`

#### Running Flask & Apache Manually

- **Flask Application Execution:**
  The Flask application can be started manually from the following location:
  `/data/SWATGenXApp/codes/web_application/run.py`
- **Apache WSGI Entry Point:**
  Apache runs the application using the WSGI entry point located at:
  `/data/SWATGenXApp/codes/web_application/app.wsgi`

This document provides an overview of the deployment structure, configuration files, and management of the SWATGenXApp web application.


Check out these:
 
curl -I https://ciwre-bae.campusad.msu.edu/static/js/
curl -I https://ciwre-bae.campusad.msu.edu/static/css/
curl -I https://ciwre-bae.campusad.msu.edu/static/images/
curl -I http://ciwre-bae.campusad.msu.edu/api/
curl -I http://ciwre-bae.campusad.msu.edu/api/status
 
###########
React development env: localhost:3000
 
 
##############
The only thing remain is to wrapping and moving template htmls to JXS in:
legacy: /data/SWATGenXApp/codes/web_application/app/templates
react: /data/SWATGenXApp/codes/web_application/frontend/src/pages
 
I have done it for the home page: /data/SWATGenXApp/codes/web_application/frontend/src/pages/Home.js
 
###############
Directories of css and js that moved from the legacy template rendering:
/data/SWATGenXApp/codes/web_application/frontend/build/static/js
/data/SWATGenXApp/codes/web_application/frontend/build/static/css
curl -I https://ciwre-bae.campusad.msu.edu/static/js/home.js
curl -I https://ciwre-bae.campusad.msu.edu/static/css/home.css
 
#####
run React: 
cd /data/SWATGenXApp/codes/web_application/frontend
npm start
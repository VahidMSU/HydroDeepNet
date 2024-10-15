# Stop and disable Apache service
sudo systemctl stop apache2
sudo systemctl disable apache2

# Create log file and set permissions
sudo touch /var/log/myapp.log
sudo chown www-data:www-data /var/log/myapp.log
sudo chmod 664 /var/log/myapp.log


# Remove only existing files (without throwing errors for missing files)
sudo rm -f /etc/ssl/private/ciwre-bae.campusad.msu.edu.key 
sudo rm -f /etc/ssl/certs/ciwre-bae_campusad_msu_edu_cert.cer   #as Certificate only, PEM encoded
sudo rm -f /etc/ssl/certs/ciwre-bae_campusad_msu_edu.pem   #as Certificate (w/ issuer after), PEM encoded
sudo rm -f /etc/ssl/certs/ciwre-bae_campusad_msu_edu.cer  #as Certificate (w/ chain), PEM encoded
sudo rm -f /etc/ssl/certs/ciwre-bae_campusad_msu_edu_interm.cer #as Root/Intermediate(s) only, PEM encoded 
sudo rm -f /etc/ssl/certs/ciwre-bae_campusad_msu_edu_interm2.cer #as Intermediate(s)/Root only, PEM encoded
sudo rm -f /etc/ssl/certs/ciwre-bae_campusad_msu_edu_fullchain.cer #as Full chain, PEM encoded

sudo rm -f /var/log/apache2/swatgenx_ssl_error.log
sudo rm -f /var/log/apache2/swatgenx_ssl_access.log
sudo rm -f /var/log/apache2/error.log





# Verify the full chain certificate was created
sudo ls -l /etc/ssl/certs/ciwre-bae_campusad_msu_edu.cer




# Copy new certificate and key files
cd /data/SWATGenXApp/codes/web_application/ciwre-bae-crs/
sudo cp ciwre-bae_campusad_msu_edu_cert.cer /etc/ssl/certs/
sudo cp ciwre-bae_campusad_msu_edu.cer /etc/ssl/certs/

## check the certificates
openssl s_client -connect ciwre-bae.campusad.msu.edu:443 -CAfile /etc/ssl/certs/ciwre-bae_campusad_msu_edu.cer
openssl s_client -connect ciwre-bae.campusad.msu.edu:443 -CAfile /etc/ssl/certs/ciwre-bae_campusad_msu_edu_cert.cer


## copy the private key
sudo cp ciwre-bae.campusad.msu.edu.key /etc/ssl/private/


# Ensure the files are owned by root and have the correct permissions
sudo chown root:root /etc/ssl/certs/ciwre-bae_campusad_msu_edu.cer  /etc/ssl/certs/ciwre-bae_campusad_msu_edu_cert.cer
sudo chmod 644 /etc/ssl/certs/ciwre-bae_campusad_msu_edu.cer /etc/ssl/certs/ciwre-bae_campusad_msu_edu_cert.cer
sudo chown root:root /etc/ssl/private/ciwre-bae.campusad.msu.edu.key
sudo chmod 600 /etc/ssl/private/ciwre-bae.campusad.msu.edu.key

# Verify the files in /etc/ssl/certs/ and /etc/ssl/private/ directories
sudo ls -l /etc/ssl/certs/ciwre-bae_campusad_msu_edu.cer
sudo ls -l /etc/ssl/private/ciwre-bae.campusad.msu.edu.key

# Create the SSL Apache configuration for HTTPS (port 443)
# Create the SSL Apache configuration for HTTPS (port 443)
sudo bash -c 'cat > /etc/apache2/sites-available/SWATGenX-ssl.conf << EOL
<IfModule mod_ssl.c>
<VirtualHost *:443>
    ServerName ciwre-bae.campusad.msu.edu
    DocumentRoot /data/SWATGenXApp/

    SSLEngine on
    
    SSLCertificateFile /etc/ssl/certs/ciwre-bae_campusad_msu_edu_cert.cer
    SSLCertificateChainFile /etc/ssl/certs/ciwre-bae_campusad_msu_edu.cer
    SSLCertificateKeyFile /etc/ssl/private/ciwre-bae.campusad.msu.edu.key
    # Set allowed SSL/TLS protocols
    SSLProtocol all -SSLv3 -TLSv1 -TLSv1.1

    # Updated WSGI daemon process name
    WSGIDaemonProcess swatgenx_app_ssl python-home=/data/SWATGenXApp/codes/.venv python-path=/data/SWATGenXApp/codes processes=10 threads=20 maximum-requests=1000
    WSGIProcessGroup swatgenx_app_ssl
    WSGIScriptAlias / /data/SWATGenXApp/codes/web_application/app.wsgi
    WSGIApplicationGroup %{GLOBAL}

    Timeout 600
    KeepAlive On
    KeepAliveTimeout 15
    MaxKeepAliveRequests 100

    Alias /static /data/SWATGenXApp/codes/web_application/static
    <Directory /data/SWATGenXApp/codes/web_application/static>
        Require all granted
    </Directory>

    <Directory /data/SWATGenXApp/codes/web_application>
        Options -Indexes +FollowSymLinks
        AllowOverride All
        Require all granted
        LogLevel info ssl:warn authz_core:debug
    </Directory>

    ErrorLog ${APACHE_LOG_DIR}/swatgenx_ssl_error.log
    CustomLog ${APACHE_LOG_DIR}/swatgenx_ssl_access.log combined
</VirtualHost>
</IfModule>
EOL'


# Create the SWATGenX Apache configuration for HTTP (port 80)
sudo bash -c 'cat > /etc/apache2/sites-available/SWATGenX.conf << EOL
<VirtualHost *:80>
    ServerName ciwre-bae.campusad.msu.edu
    DocumentRoot /data/SWATGenXApp/

    <Directory /data/SWATGenXApp/codes/web_application>
        Options -Indexes +FollowSymLinks
        AllowOverride All
        Require all granted
        LogLevel info ssl:warn authz_core:debug

    </Directory>
</VirtualHost>
EOL'

sudo bash -c 'cat > /etc/apache2/ports.conf << EOL
# Listening only on IPv4 for HTTP
Listen 0.0.0.0:80

<IfModule ssl_module>
    # Listening only on IPv4 for HTTPS
    Listen 0.0.0.0:443
</IfModule>
EOL'

sudo chown -R www-data:www-data /var/log/apache2/swatgenx_ssl_error.log /var/log/apache2/swatgenx_ssl_access.log
sudo chown -R www-data:www-data /data/SWATGenXApp/



# Enable necessary Apache modules and sites
sudo a2enmod ssl rewrite headers wsgi
sudo a2ensite SWATGenX.conf
sudo a2ensite SWATGenX-ssl.conf

# Test the Apache configuration
sudo apachectl configtest

# Restart Apache to apply the new configuration
sudo systemctl restart apache2

# Tail the log to check for errors
tail -f /data/SWATGenXApp/logs/myapp.log

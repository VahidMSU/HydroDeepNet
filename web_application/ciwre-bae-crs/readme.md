# The csr is created by openssl:

https://phoenixnap.com/kb/generate-openssl-certificate-signing-request

# Use the following configuration file for apache setup:

/data/SWATGenXApp/codes/web_application/ciwre-bae-crs/ciwre-bae.conf


# For our apache server, we need to use /etc/ssl/certs/ciwre-bae_campusad_msu_edu_cert.cer as certificate and /etc/ssl/certs/ciwre-bae_campusad_msu_edu.cer as full chain. 

    Make sure the following setting are present tin the ciwre-bae.conf file:
 
    SSLCertificateFile /etc/ssl/certs/ciwre-bae_campusad_msu_edu_cert.cer
    SSLCertificateChainFile /etc/ssl/certs/ciwre-bae_campusad_msu_edu.cer
    
    SSLCertificateKeyFile /etc/ssl/private/ciwre-bae.campusad.msu.edu.key



# The web application for apache is here:

/data/SWATGenXApp/codes/web_application/app.wsgi



To setup port:

# Update ports.conf to listen only on IPv4
sudo bash -c 'cat > /etc/apache2/ports.conf << EOL
Listen 0.0.0.0:80
<IfModule ssl_module>
    Listen 0.0.0.0:443
</IfModule>
EOL'


# Set permissions for SWATGenX application

sudo chown -R www-data:www-data /data/SWATGenXApp/
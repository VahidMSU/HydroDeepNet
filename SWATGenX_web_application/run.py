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


# sudo cat /etc/apache2/sites-available/default-ssl.conf

#/etc/apache2/sites-available/SWATGenX.conf
#sudo a2ensite SWATGenX.conf
#sudo a2dissite default-ssl.conf
#sudo systemctl reload apache2
#sudo chmod 644 /home/rafieiva/MyDataBase/codebase/SWATGenX_web_application/ciwre-bae-crs/ciwre-bae_campusad_msu_edu_cert.cer
#sudo chmod 600 /home/rafieiva/MyDataBase/codebase/SWATGenX_web_application/ciwre-bae-crs/ciwre-bae.campusad.msu.edu.key
##sudo netstat -tuln | grep 443
# sudo nano /etc/apache2/sites-available/default-ssl.conf
#sudo chown www-data:www-data /home/rafieiva/MyDataBase/codebase/SWATGenX_web_application/ciwre-bae-crs/ciwre-bae.campusad.msu.edu.key
#sudo chmod 640 /home/rafieiva/MyDataBase/codebase/SWATGenX_web_application/ciwre-bae-crs/ciwre-bae.campusad.msu.edu.key
#  sudo a2enmod wsgi

# curl -I https://ciwre-bae.campusad.msu.edu/


# Set correct permissions for the .ssh directory


# Set correct permissions for the private key (id_rsa)


# Set correct permissions for the public key (id_rsa.pub)


# Set correct permissions for authorized_keys

for user in $(ls /data/SWATGenXApp/Users); do
    USER_HOME="/data/SWATGenXApp/Users/$user"

    # Grant access to Apache and specific SFTP users
    sudo setfacl -R -m u:www-data:rwx "$USER_HOME"
    sudo setfacl -R -m g:www-data:rwx "$USER_HOME"
    sudo setfacl -R -m u:$user:rwx "$USER_HOME"

    # Ensure default permissions for new files
    sudo setfacl -R -d -m u:www-data:rwx "$USER_HOME"
    sudo setfacl -R -d -m g:www-data:rwx "$USER_HOME"
    sudo setfacl -R -d -m u:$user:rwx "$USER_HOME"
    
    # Restrict access to others
    sudo setfacl -R -m o::--- "$USER_HOME"
    sudo setfacl -R -d -m o::--- "$USER_HOME"
done

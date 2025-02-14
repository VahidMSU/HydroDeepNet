import os 
import sqlite3

path =  "/data/SWATGenXApp/codes/web_application/instance/site.db"

usename = 'menly42'

## remove username for the database

def remove_user(path, username):
    try:
        conn = sqlite3.connect(path)
        c = conn.cursor()
        c.execute("DELETE FROM user WHERE username = ?", (username,))
        conn.commit()
        conn.close()
        return {"message": f"User {username} removed successfully."}
    except sqlite3.Error as e:
        return {"error": f"Failed to remove user {username}: {e}"}
    
remove_user(path, usename)
# Output: {'message': 'User guess removed successfully.'}
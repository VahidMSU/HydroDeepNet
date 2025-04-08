import sqlite3

database = '/data/SWATGenXApp/codes/web_application/instance/site.db'
username = "guest"

##remove user from database

def delete_user(username):
    try:
        conn = sqlite3.connect(database)
        cur = conn.cursor()
        cur.execute("DELETE FROM user WHERE username=?", (username,))
        conn.commit()
        conn.close()
        return True     

    except Exception as e:
        print(e)
        return False    
    

delete_user(username)
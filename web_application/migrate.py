# manage.py
from app import create_app, db  # or however you import your app and db
from flask_migrate import Migrate

app = create_app()  # create_app() should return your Flask app
migrate = Migrate(app, db)

# If your code is not using a create_app pattern, just do:
# from app import app, db
# migrate = Migrate(app, db)

if __name__ == "__main__":
    app.run()

from app import create_app
from app.extensions import db
from app.models import User

app = create_app()

with app.app_context():
    # Find user
    user = User.query.filter_by(username="vahidr32").first()

    # If user exists, delete it
    if user:
        db.session.delete(user)
        db.session.commit()
        print("User vahidr32 deleted successfully")
    else:
        print("User not found")
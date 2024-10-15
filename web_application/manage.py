from app import create_app, db
from app.models import User
from flask_migrate import Migrate
from flask.cli import FlaskGroup

app = create_app()
migrate = Migrate(app, db)

cli = FlaskGroup(create_app=create_app)

@cli.command('create_admin')
def create_admin():
    with app.app_context():
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', email='admin@example.com', password='admin')
            db.session.add(admin)
            db.session.commit()
            print('Admin user created.')

if __name__ == '__main__':
    cli()

from celery import Celery

def make_celery(app=None):
    """Create a Celery instance"""
    celery = Celery(
        'web_application',
        broker='redis://localhost:6379/0',
        backend='redis://localhost:6379/0',
        include=['app.tasks']
    )
    
    # Optional: Configure Celery from Flask app config
    if app:
        celery.conf.update(app.config)
        
        class ContextTask(celery.Task):
            def __call__(self, *args, **kwargs):
                with app.app_context():
                    return self.run(*args, **kwargs)
                    
        celery.Task = ContextTask
        
    return celery

# Create the celery instance
celery = make_celery()
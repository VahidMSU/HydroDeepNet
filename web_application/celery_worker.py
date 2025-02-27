import os
import sys

# Add the project directory to Python path (one level up from this file)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celery_app import celery

if __name__ == '__main__':
    celery.start()
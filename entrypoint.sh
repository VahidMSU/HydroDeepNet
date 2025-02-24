#!/bin/bash

# Start services
nginx -g 'daemon off;' & 
gunicorn -b 0.0.0.0:5000 run:app

# Wait for all background processes
wait

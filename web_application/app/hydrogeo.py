from flask import Blueprint, jsonify, request, current_app
from flask_login import current_user
import json
import pandas as pd
import numpy as np
from app.decorators import conditional_login_required, conditional_verified_required
from app.utils import hydrogeo_dataset_dict, read_h5_file
from app.extensions import csrf

hydrogeo_bp = Blueprint('hydrogeo', __name__)

@hydrogeo_bp.route('/hydro_geo_dataset', methods=['GET', 'POST'])
@conditional_login_required
@conditional_verified_required
def hydro_geo_dataset():
    """Handles HydroGeoDataset requests for fetching environmental data."""
    
    current_app.logger.info("HydroGeoDataset route called")

    # Define available dataset groups
    available_groups = [
        'CDL', 'EBK', 'LANDFIRE', 'MODIS', 'NHDPlus', 'PRISM', 'SNODAS_monthly',
        'Wellogic', 'climate_pattern', 'geospatial', 'gssurgo', 'population'
    ]
    
    # Load dataset dictionary
    hydrodict = hydrogeo_dataset_dict()

    if request.method == 'GET':
        """Handle GET request to fetch available variables or subvariables."""
        current_app.logger.info("GET request received")

        variable = request.args.get('variable')
        if variable:
            current_app.logger.info(f"Fetching subvariables for variable: {variable}")
            subvariables = hydrodict.get(variable, [])
            return jsonify({"subvariables": subvariables})
        else:
            return jsonify({"variables": available_groups})

    elif request.method == 'POST':
        """Handle POST request to fetch data based on user input."""
        current_app.logger.info("Form submitted")
        data_payload = request.get_json()
        current_app.logger.info(f"Received JSON data: {json.dumps(data_payload, indent=2)}")

        # Extract variable and subvariable
        variable = data_payload.get('variable')
        subvariable = data_payload.get('subvariable')

        if not variable or not subvariable:
            message = "Variable and Subvariable are required."
            current_app.logger.error(message)
            return jsonify({"error": message}), 400

        # Extract coordinates
        latitude = data_payload.get('latitude')
        longitude = data_payload.get('longitude')
        min_latitude = data_payload.get('min_latitude')
        max_latitude = data_payload.get('max_latitude')
        min_longitude = data_payload.get('min_longitude')
        max_longitude = data_payload.get('max_longitude')

        # Extract polygon if provided and process it
        polygon_coordinates = data_payload.get('polygon_coordinates')

        if polygon_coordinates:
            try:
                # Parse the polygon coordinates - handle both string and direct JSON
                if isinstance(polygon_coordinates, str):
                    vertices = json.loads(polygon_coordinates)
                else:
                    vertices = polygon_coordinates
                    
                current_app.logger.info(f"Received polygon vertices: {vertices}")

                # Convert polygon to bounding box if needed
                if isinstance(vertices, list) and vertices:
                    # Check if vertices is a list of coordinate objects or a list of coordinate pairs
                    if isinstance(vertices[0], dict) and 'latitude' in vertices[0] and 'longitude' in vertices[0]:
                        # Format: [{'latitude': x, 'longitude': y}, ...]
                        latitudes = [float(vertex['latitude']) for vertex in vertices]
                        longitudes = [float(vertex['longitude']) for vertex in vertices]
                    elif isinstance(vertices[0], (list, tuple)) and len(vertices[0]) >= 2:
                        # Format: [[lon, lat], ...]
                        latitudes = [float(vertex[1]) for vertex in vertices]
                        longitudes = [float(vertex[0]) for vertex in vertices]
                    else:
                        raise ValueError(f"Unrecognized vertices format: {vertices[0]}")
                        
                    min_latitude, max_latitude = min(latitudes), max(latitudes)
                    min_longitude, max_longitude = min(longitudes), max(longitudes)
                    
                    current_app.logger.info(f"Polygon bounds: ({min_latitude}, {max_latitude}), ({min_longitude}, {max_longitude})")

            except Exception as e:
                current_app.logger.error(f"Error parsing polygon coordinates: {e}")
                return jsonify({"error": f"Invalid polygon coordinates: {e}"}), 400

        # Validate input (must have either a point or a bounding box)
        if latitude and longitude:
            current_app.logger.info(f"Fetching data for {variable}/{subvariable} at ({latitude}, {longitude})")
            try:
                raw_data = read_h5_file(
                    lat=float(latitude),
                    lon=float(longitude),
                    address=f"{variable}/{subvariable}"
                )
                data = {key: float(value) if isinstance(value, np.float32) else value for key, value in raw_data.items()}
                return jsonify({"message": "Data fetched successfully", "data": data})
            except Exception as e:
                current_app.logger.error(f"Error fetching data: {e}")
                return jsonify({"error": f"Error fetching data: {e}"}), 500

        elif all([min_latitude, max_latitude, min_longitude, max_longitude]):
            current_app.logger.info(f"Fetching data for {variable}/{subvariable} in range ({min_latitude}, {max_latitude}), ({min_longitude}, {max_longitude})")
            try:
                raw_data = read_h5_file(
                    lat_range=(float(min_latitude), float(max_latitude)),
                    lon_range=(float(min_longitude), float(max_longitude)),
                    address=f"{variable}/{subvariable}"
                )
                data = {key: float(value) if isinstance(value, np.float32) else value for key, value in raw_data.items()}
                return jsonify({"message": "Data fetched successfully", "data": data})
            except Exception as e:
                current_app.logger.error(f"Error fetching data: {e}")
                return jsonify({"error": f"Error fetching data: {e}"}), 500

        else:
            message = f"Please provide either a point (latitude/longitude) or a range (min/max lat/lon)."
            current_app.logger.error(message)
            return jsonify({"error": message}), 400

@hydrogeo_bp.route('/get_subvariables', methods=['POST'])
@conditional_login_required
@conditional_verified_required
def get_subvariables():
    current_app.logger.info("Get Subvariables route called")
    variable = request.form.get('variable')
    current_app.logger.info(f"Received variable: {variable}")

    if not variable:
        return jsonify({"error": "Variable is required"}), 400

    hydrodict = hydrogeo_dataset_dict()
    subvariables = hydrodict.get(variable, [])
    current_app.logger.info(f"Subvariables for {variable}: {subvariables}")

    return jsonify({"subvariables": subvariables})

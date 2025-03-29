from flask import Blueprint, jsonify, send_from_directory
import os

api_bp = Blueprint("api", __name__)

@api_bp.route("/", methods=["GET"])
def api_root():
    return jsonify({"message": "API Root"})

@api_bp.route("/status", methods=["GET"])
def api_status():
    return jsonify({"status": "OK"})

@api_bp.route("/static/visualizations/<path:filename>", methods=["GET"])
def api_serve_visualizations(filename):
    """
    Direct API route to serve visualization files.
    This ensures visualization files can be accessed through the API in development.
    """
    visualizations_dir = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12"
    return send_from_directory(visualizations_dir, filename)

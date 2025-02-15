from flask import Blueprint, jsonify

api_bp = Blueprint("api", __name__)

@api_bp.route("/", methods=["GET"])
def api_root():
    return jsonify({"message": "API Root"})
@api_bp.route("/status", methods=["GET"])
def api_status():
    return jsonify({"status": "OK"})

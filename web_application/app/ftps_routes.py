from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from app.extensions import csrf  # Import CSRF instance
from app.ftps_manager import create_ftps_user, delete_ftps_user, list_ftps_users

ftps_bp = Blueprint("ftps", __name__)

@ftps_bp.route("/ftps/create", methods=["POST"])
@login_required
@csrf.exempt  # Explicitly exempt CSRF here
def create_ftps():
    """Create an FTPS user when requested from the frontend."""
    if not current_user.is_admin:
        return jsonify({"error": "Unauthorized"}), 403

    username = request.json.get("username")
    if not username:
        return jsonify({"error": "Username required"}), 400

    result = create_ftps_user(username)
    return jsonify(result)

@ftps_bp.route("/ftps/delete", methods=["POST"])
@login_required
@csrf.exempt  # Explicitly exempt CSRF here
def delete_ftps():
    """Delete an FTPS user."""
    if not current_user.is_admin:
        return jsonify({"error": "Unauthorized"}), 403

    username = request.json.get("username")
    if not username:
        return jsonify({"error": "Username required"}), 400

    result = delete_ftps_user(username)
    return jsonify(result)

@ftps_bp.route("/ftps/list", methods=["GET"])
@login_required
def list_ftps():
    """List all FTPS users."""
    if not current_user.is_admin:
        return jsonify({"error": "Unauthorized"}), 403

    result = list_ftps_users()
    return jsonify(result)
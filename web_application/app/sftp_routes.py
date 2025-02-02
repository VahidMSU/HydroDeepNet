from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from app.sftp_manager import create_sftp_user, delete_sftp_user

sftp_bp = Blueprint("sftp", __name__)

@sftp_bp.route("/sftp/create", methods=["POST"])
@login_required
def create_sftp():
    """Create an SFTP user when requested from the frontend."""
    if not current_user.is_admin:  # Ensure only admin can create users
        return jsonify({"error": "Unauthorized"}), 403

    username = request.json.get("username")
    if not username:
        return jsonify({"error": "Username required"}), 400

    result = create_sftp_user(username)
    return jsonify(result)

@sftp_bp.route("/sftp/delete", methods=["POST"])
@login_required
def delete_sftp():
    """Delete an SFTP user."""
    if not current_user.is_admin:
        return jsonify({"error": "Unauthorized"}), 403

    username = request.json.get("username")
    if not username:
        return jsonify({"error": "Username required"}), 400

    result = delete_sftp_user(username)
    return jsonify(result)

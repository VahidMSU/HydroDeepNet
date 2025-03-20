from flask import Blueprint, jsonify, request, current_app, redirect
from flask_login import current_user
import os
from app.decorators import conditional_login_required, conditional_verified_required

visualization_bp = Blueprint('visualization', __name__)

@visualization_bp.route('/api/get_options', methods=['GET'])
@conditional_login_required
@conditional_verified_required
def get_options():
    try:
        names_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/"
        variables = ['et', 'perc', 'precip', 'snofall', 'snomlt', 'surq_gen', 'wateryld']

        if os.path.exists(names_path):
            names = os.listdir(names_path)
            if "log.txt" in names:
                names.remove("log.txt")
        else:
            names = []
        return jsonify({'names': names, 'variables': variables})
    except Exception as e:
        current_app.logger.error(f"Error fetching options: {e}")
        return jsonify({"error": "Failed to fetch options"}), 500

@visualization_bp.route('/api/visualizations', methods=['GET'])
@conditional_login_required
@conditional_verified_required
def visualizations():
    current_app.logger.info("Visualizations route called")
    name = request.args.get('NAME', default=None)
    ver = request.args.get('ver', default=None)
    variable = request.args.get('variable', default=None)

    if not all([name, ver, variable]):
        error_msg = "Please provide NAME, Version, and Variable."
        current_app.logger.error(error_msg)
        return jsonify({"error": error_msg}), 400

    # Use filesystem path only to check if the directory exists
    base_path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{name}/figures_SWAT_gwflow_MODEL"
    if not os.path.exists(base_path):
        error_msg = f"No visualization data found for watershed: {name}"
        current_app.logger.error(error_msg)
        return jsonify({"error": error_msg}), 404

    video_path = os.path.join(base_path, "verifications_videos")
    if not os.path.exists(video_path):
        error_msg = f"No visualization videos found for watershed: {name}"
        current_app.logger.error(error_msg)
        return jsonify({"error": error_msg}), 404

    variables = variable.split(",")
    gif_urls = []
    missing_vars = []

    for var in variables:
        # Check if file exists in filesystem
        gif_file = os.path.join(video_path, f"{ver}_{var}_animation.gif")
        if os.path.exists(gif_file):
            # Return URL path, not filesystem path
            gif_urls.append(f"/static/visualizations/{name}/figures_SWAT_gwflow_MODEL/verifications_videos/{ver}_{var}_animation.gif")
        else:
            missing_vars.append(var)
            current_app.logger.warning(f"Missing visualization for {var} in {gif_file}")

    if not gif_urls:
        error_msg = f"No visualizations found for NAME: {name}, Version: {ver}, Variables: {variables}."
        if missing_vars:
            error_msg += f" Missing variables: {', '.join(missing_vars)}"
        current_app.logger.error(error_msg)
        return jsonify({"error": error_msg}), 404

    response_data = {
        "gif_files": gif_urls
    }
    
    if missing_vars:
        response_data["warnings"] = f"Some variables were not found: {', '.join(missing_vars)}"

    return jsonify(response_data)

# Support both API and non-API versions of visualization routes
@visualization_bp.route('/api/static/visualizations/<name>/<ver>/<variable>.gif', methods=['GET'])
@visualization_bp.route('/static/visualizations/<name>/<ver>/<variable>.gif', methods=['GET'])
@conditional_login_required
@conditional_verified_required
def serve_visualization(name, ver, variable):
    """
    Serve visualization GIFs with proper error handling
    """
    # Construct the path to the visualization file using the static URL
    # This will be handled by Apache's Alias directive
    static_url = f"/static/visualizations/{name}/figures_SWAT_gwflow_MODEL/verifications_videos/{ver}_{variable}_animation.gif"
    
    # Check if the file exists on the filesystem before redirecting
    file_path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{name}/figures_SWAT_gwflow_MODEL/verifications_videos/{ver}_{variable}_animation.gif"
    if not os.path.exists(file_path):
        current_app.logger.error(f"Visualization not found: {file_path}")
        return jsonify({"error": "Visualization not found"}), 404
        
    # Redirect to the static URL that will be handled by Apache
    return redirect(static_url)

# Support both API and non-API versions of vision system
@visualization_bp.route('/api/vision_system')
@visualization_bp.route('/vision_system')
@conditional_login_required
@conditional_verified_required
def vision_system():
    current_app.logger.info("Vision System route called")
    return jsonify({"title": "Vision System", "message": "Vision System page"})

from flask import Blueprint, jsonify, current_app, request
import os
import json

debug_bp = Blueprint('debug', __name__)

@debug_bp.route('/api/debug/reports', methods=['GET'])
def debug_reports():
    """Debug endpoint to inspect the structure of reports directory"""
    try:
        reports_root = '/data/SWATGenXApp/Users'
        users_with_reports = []
        
        # Scan users directory
        if os.path.exists(reports_root):
            for user in os.listdir(reports_root):
                user_reports_dir = os.path.join(reports_root, user, 'Reports')
                if os.path.exists(user_reports_dir):
                    reports = []
                    
                    # Scan reports for this user
                    for report_id in os.listdir(user_reports_dir):
                        report_dir = os.path.join(user_reports_dir, report_id)
                        if not os.path.isdir(report_dir):
                            continue
                            
                        # Check for metadata
                        metadata_file = os.path.join(report_dir, 'metadata.json')
                        error_file = os.path.join(report_dir, 'error.json')
                        
                        report_info = {
                            'report_id': report_id,
                            'path': report_dir,
                            'has_metadata': os.path.exists(metadata_file),
                            'has_error': os.path.exists(error_file),
                            'files': []
                        }
                        
                        # List all files in the directory
                        for root, dirs, files in os.walk(report_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                rel_path = os.path.relpath(file_path, report_dir)
                                size = os.path.getsize(file_path)
                                report_info['files'].append({
                                    'name': file,
                                    'path': rel_path,
                                    'size': size
                                })
                        
                        # Add metadata content if available
                        if os.path.exists(metadata_file):
                            try:
                                with open(metadata_file, 'r') as f:
                                    report_info['metadata'] = json.load(f)
                            except Exception as e:
                                report_info['metadata_error'] = str(e)
                                
                        reports.append(report_info)
                    
                    users_with_reports.append({
                        'username': user,
                        'reports': reports
                    })
        
        return jsonify({
            'status': 'success',
            'users': users_with_reports
        })
    except Exception as e:
        current_app.logger.error(f"Error in debug reports: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@debug_bp.route('/api/debug/check_report/<report_id>', methods=['GET'])
def debug_check_report(report_id):
    """Debug endpoint to check a specific report"""
    try:
        if not report_id:
            return jsonify({'error': 'Report ID is required'}), 400
            
        reports_found = []
        
        # Search for the report in all users' directories
        reports_root = '/data/SWATGenXApp/Users'
        for user in os.listdir(reports_root):
            user_report_dir = os.path.join(reports_root, user, 'Reports', report_id)
            if os.path.isdir(user_report_dir):
                report_info = {
                    'user': user,
                    'report_id': report_id,
                    'path': user_report_dir,
                    'files': []
                }
                
                # Check for metadata
                metadata_file = os.path.join(user_report_dir, 'metadata.json')
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            report_info['metadata'] = json.load(f)
                    except Exception as e:
                        report_info['metadata_error'] = str(e)
                
                # List all files in the directory
                for root, dirs, files in os.walk(user_report_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, user_report_dir)
                        size = os.path.getsize(file_path)
                        report_info['files'].append({
                            'name': file,
                            'path': rel_path,
                            'size': size
                        })
                
                reports_found.append(report_info)
        
        if not reports_found:
            return jsonify({
                'status': 'error',
                'message': f'Report {report_id} not found'
            }), 404
            
        return jsonify({
            'status': 'success',
            'reports': reports_found
        })
    except Exception as e:
        current_app.logger.error(f"Error checking report {report_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@debug_bp.route('/api/debug/report_contents/<report_id>', methods=['GET'])
def debug_report_contents(report_id):
    """List all files in a report directory with their paths"""
    try:
        if not report_id:
            return jsonify({'error': 'Report ID is required'}), 400
        
        username = request.args.get('username')
        if not username:
            return jsonify({'error': 'Username is required'}), 400
        
        report_dir = os.path.join('/data/SWATGenXApp/Users', username, 'Reports', report_id)
        if not os.path.isdir(report_dir):
            return jsonify({
                'status': 'error',
                'message': f'Report directory not found: {report_dir}'
            }), 404
        
        # List all files recursively
        all_files = []
        for root, dirs, files in os.walk(report_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, report_dir)
                file_size = os.path.getsize(file_path)
                file_ext = os.path.splitext(file)[1].lower()
                all_files.append({
                    'path': rel_path,
                    'full_path': file_path,
                    'size': file_size,
                    'extension': file_ext,
                    'is_html': file_ext == '.html',
                })
        
        # Sort files by path for easier reading
        all_files.sort(key=lambda x: x['path'])
        
        # Count files by type
        file_types = {}
        for file in all_files:
            ext = file['extension']
            if ext in file_types:
                file_types[ext] += 1
            else:
                file_types[ext] = 1
        
        return jsonify({
            'status': 'success',
            'report_dir': report_dir,
            'file_count': len(all_files),
            'file_types': file_types,
            'files': all_files
        })
    
    except Exception as e:
        current_app.logger.error(f"Error in debug_report_contents: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

"""
Simple HTTP server to view 3D MODFLOW models in a web browser.
This allows viewing the models without having to copy files.
"""

import http.server
import socketserver
import argparse
import os
import webbrowser
import sys
import tempfile
from urllib.parse import quote
import threading
import shutil

def create_index_html(base_dir):
    """Create an index.html file that lists all available 3D visualizations."""
    # Find all subdirectories that might contain visualizations
    watersheds = []
    for root, dirs, files in os.walk(base_dir):
        if "visualization" in dirs:
            vis_dir = os.path.join(root, "visualization")
            html_files = [f for f in os.listdir(vis_dir) if f.endswith('.html')]
            if html_files:
                rel_path = os.path.relpath(vis_dir, base_dir)
                name = os.path.basename(os.path.dirname(os.path.dirname(vis_dir)))
                watersheds.append({
                    'name': name,
                    'path': rel_path,
                    'html_files': html_files
                })
    
    # Create the HTML content
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MODFLOW 3D Model Viewer</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1, h2 {
                color: #2c3e50;
            }
            .watershed {
                margin-bottom: 30px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
                border-left: 5px solid #3498db;
            }
            .visualization {
                margin: 10px 0;
                padding: 10px;
                background-color: #fff;
                border-radius: 4px;
                border: 1px solid #ddd;
            }
            a {
                color: #3498db;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <h1>MODFLOW 3D Model Viewer</h1>
        <p>Select a watershed and visualization to view:</p>
    """
    
    if not watersheds:
        html += "<p>No visualizations found. Run <code>visualize_model.py</code> to create them.</p>"
    else:
        for watershed in watersheds:
            html += f"""
            <div class="watershed">
                <h2>Watershed: {watershed['name']}</h2>
                <div class="visualizations">
            """
            
            for html_file in watershed['html_files']:
                file_path = os.path.join(watershed['path'], html_file)
                html += f"""
                <div class="visualization">
                    <a href="/{file_path}" target="_blank">{html_file}</a>
                </div>
                """
            
            html += """
                </div>
            </div>
            """
    
    html += """
    </body>
    </html>
    """
    
    # Try to write the HTML to the base directory, but if it fails, use a temp directory
    try:
        index_path = os.path.join(base_dir, "index.html")
        with open(index_path, 'w') as f:
            f.write(html)
        print(f"Created index file at: {index_path}")
    except PermissionError:
        # Create a temporary file instead
        temp_dir = tempfile.mkdtemp(prefix="modflow_viewer_")
        index_path = os.path.join(temp_dir, "index.html")
        with open(index_path, 'w') as f:
            f.write(html)
        print(f"Created index file in temporary directory: {index_path}")
    
    return index_path, os.path.dirname(index_path)

def main():
    parser = argparse.ArgumentParser(description='Start a web server to view 3D MODFLOW models')
    parser.add_argument('--port', type=int, default=8000, help='Port number (default: 8000)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open web browser automatically')
    parser.add_argument('--base-dir', default=None, 
                        help='Base directory to serve (default: current user directory)')
    args = parser.parse_args()
    
    # Set default base directory to a user-accessible location if not specified
    if args.base_dir is None:
        args.base_dir = os.path.expanduser("~")
        print(f"Using default base directory: {args.base_dir}")
    
    # Check if path exists
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory does not exist: {args.base_dir}")
        return 1
    
    # Create index.html
    index_path, serve_dir = create_index_html(args.base_dir)
    
    # Set up a simple HTTP server
    os.chdir(serve_dir)
    handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        url = f"http://localhost:{args.port}"
        print(f"Serving at: {url}")
        
        # Open web browser if requested
        if not args.no_browser:
            def open_browser():
                # Open the index.html file directly
                index_url = f"{url}/index.html"
                print(f"Opening browser to: {index_url}")
                webbrowser.open(index_url)
            
            threading.Timer(1.0, open_browser).start()
        
        try:
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            # Clean up temporary directory if one was created
            if serve_dir != args.base_dir:
                try:
                    shutil.rmtree(serve_dir)
                    print(f"Cleaned up temporary directory: {serve_dir}")
                except Exception as e:
                    print(f"Failed to clean up temporary directory: {e}")
            return 0

if __name__ == "__main__":
    sys.exit(main())

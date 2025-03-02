"""
Module for converting Markdown reports to HTML with proper formatting and structure.
"""
import os
import logging
import markdown
from pathlib import Path
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_markdown_to_html(markdown_path: str) -> Optional[str]:
    """
    Convert a Markdown file to HTML.
    
    Args:
        markdown_path: Path to the Markdown file
        
    Returns:
        Path to the generated HTML file or None if conversion failed
    """
    try:
        # Read Markdown content
        with open(markdown_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=[
                'markdown.extensions.tables',
                'markdown.extensions.fenced_code',
                'markdown.extensions.toc'
            ]
        )
        
        # Create HTML wrapper
        html_output = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Environmental Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .report-navigation {{
            background-color: #f8f9fa;
            padding: 10px 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            border: 1px solid #e9ecef;
        }}
        .report-navigation a {{
            margin-right: 15px;
            color: #007bff;
            text-decoration: none;
        }}
        .report-navigation a:hover {{
            text-decoration: underline;
        }}
        code {{
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="report-navigation">
        <a href="../index.html">← Back to Reports Index</a>
    </div>
    {html_content}
</body>
</html>"""
        
        # Save HTML file
        html_path = os.path.splitext(markdown_path)[0] + ".html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_output)
        
        logger.info(f"Converted {markdown_path} to {html_path}")
        return html_path
    
    except Exception as e:
        logger.error(f"Error converting {markdown_path} to HTML: {e}")
        return None

def create_report_index(output_dir: str) -> Optional[str]:
    """
    Create an index.html file that links to all reports in subdirectories.
    
    Args:
        output_dir: Directory containing report subdirectories
        
    Returns:
        Path to the created index.html file or None if creation failed
    """
    try:
        # Find all HTML files
        html_files = []
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.html') and file != 'index.html':
                    rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                    report_type = os.path.basename(os.path.dirname(rel_path))
                    html_files.append({
                        'path': rel_path,
                        'name': os.path.splitext(file)[0],
                        'type': report_type
                    })
        
        # Group by report type
        report_groups = {}
        for file in html_files:
            if file['type'] not in report_groups:
                report_groups[file['type']] = []
            report_groups[file['type']].append(file)
        
        # Create HTML content
        content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Environmental Reports</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .report-card {{
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 20px;
            background-color: #f8f9fa;
            transition: transform 0.3s ease;
        }}
        .report-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .report-section {{
            margin-bottom: 40px;
        }}
        .report-links {{
            list-style-type: none;
            padding-left: 0;
        }}
        .report-links li {{
            margin-bottom: 10px;
        }}
        .report-links a {{
            display: block;
            padding: 10px 15px;
            background: #e9ecef;
            color: #495057;
            text-decoration: none;
            border-radius: 5px;
            transition: background 0.2s;
        }}
        .report-links a:hover {{
            background: #dee2e6;
            color: #212529;
        }}
        .report-type-header {{
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }}
        .report-icon {{
            font-size: 24px;
            margin-right: 10px;
        }}
    </style>
</head>
<body>
    <h1>Environmental Report Index</h1>
    <p>This page contains links to all generated environmental reports for your selected area.</p>
"""
        
        # Add report sections
        icons = {
            'prism': '🌦️',
            'modis': '🛰️',
            'cdl': '🌾',
            'groundwater': '💧',
            'gov_units': '🏛️',
            'climate_change': '🌡️'
        }
        
        for report_type, files in report_groups.items():
            icon = icons.get(report_type, '📊')
            content += f"""
    <div class="report-section report-card">
        <div class="report-type-header">
            <span class="report-icon">{icon}</span>
            <h2>{report_type.title()} Reports</h2>
        </div>
        <ul class="report-links">
"""
            
            for file in files:
                content += f"""            <li><a href="{file['path']}">{file['name'].title().replace('_', ' ')}</a></li>\n"""
            
            content += """        </ul>
    </div>
"""
        
        content += """</body>
</html>
"""
        
        # Write the index file
        index_path = os.path.join(output_dir, 'index.html')
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Created report index at {index_path}")
        return index_path
    
    except Exception as e:
        logger.error(f"Error creating report index: {e}")
        return None

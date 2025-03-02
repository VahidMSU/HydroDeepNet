"""
HTML Report Converter - Convert Markdown reports to HTML with embedded images.

This module provides functionality to convert Markdown reports to HTML format
with properly embedded images, suitable for web display or React integration.
"""
import os
import re
import base64
import markdown
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from bs4 import BeautifulSoup
import mimetypes

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def get_image_mime_type(file_path: str) -> str:
    """
    Get the MIME type for an image file.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        MIME type string (default: image/png if cannot be determined)
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'image/png'

def encode_image_base64(image_path: str) -> Optional[str]:
    """
    Encode an image file as base64.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded image with data URI prefix, or None if file not found
    """
    try:
        if not os.path.exists(image_path):
            logger.warning(f"Image file not found: {image_path}")
            return None
            
        mime_type = get_image_mime_type(image_path)
        
        with open(image_path, 'rb') as img_file:
            encoded = base64.b64encode(img_file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def convert_markdown_to_html(markdown_path: str, 
                           embed_images: bool = True, 
                           output_path: Optional[str] = None,
                           css_path: Optional[str] = None) -> str:
    """
    Convert a Markdown report to HTML with optional image embedding.
    
    Args:
        markdown_path: Path to the Markdown file
        embed_images: Whether to embed images as base64 (True) or keep as links (False)
        output_path: Path to save the HTML output (if None, derives from markdown_path)
        css_path: Optional path to a CSS file to include
        
    Returns:
        Path to the generated HTML file
    """
    try:
        if not os.path.exists(markdown_path):
            logger.error(f"Markdown file not found: {markdown_path}")
            return ""
        
        # Set output path if not provided
        if output_path is None:
            output_path = os.path.splitext(markdown_path)[0] + '.html'
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Read markdown content
        with open(markdown_path, 'r') as md_file:
            md_content = md_file.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(md_content, extensions=['tables'])
        
        # Parse with BeautifulSoup for easier manipulation
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Process images if embedding is requested
        if embed_images:
            img_tags = soup.find_all('img')
            md_dir = os.path.dirname(markdown_path)
            
            for img in img_tags:
                if 'src' in img.attrs:
                    img_src = img['src']
                    
                    # Handle relative paths
                    if not os.path.isabs(img_src):
                        img_path = os.path.join(md_dir, img_src)
                    else:
                        img_path = img_src
                    
                    # Encode image to base64
                    base64_img = encode_image_base64(img_path)
                    
                    if base64_img:
                        img['src'] = base64_img
                    else:
                        logger.warning(f"Could not embed image: {img_src}")
        
        # Add CSS if provided
        head = soup.new_tag('head')
        
        # Add metadata
        meta_charset = soup.new_tag('meta')
        meta_charset['charset'] = 'utf-8'
        head.append(meta_charset)
        
        meta_viewport = soup.new_tag('meta')
        meta_viewport['name'] = 'viewport'
        meta_viewport['content'] = 'width=device-width, initial-scale=1'
        head.append(meta_viewport)
        
        # Add title from the first h1 tag if available
        h1_tag = soup.find('h1')
        title_tag = soup.new_tag('title')
        title_tag.string = h1_tag.text if h1_tag else 'Report'
        head.append(title_tag)
        
        # Add custom CSS
        style_tag = soup.new_tag('style')
        if css_path and os.path.exists(css_path):
            with open(css_path, 'r') as css_file:
                style_tag.string = css_file.read()
        else:
            # Default CSS for better readability
            style_tag.string = """
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #2c3e50;
                margin-top: 24px;
                margin-bottom: 16px;
            }
            h1 { font-size: 2em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
            h2 { font-size: 1.6em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
            h3 { font-size: 1.3em; }
            h4 { font-size: 1.1em; }
            img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin: 15px 0;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
                overflow-x: auto;
                display: block;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px 12px;
                text-align: left;
            }
            th {
                background-color: #f6f8fa;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            code {
                background-color: #f6f8fa;
                padding: 0.2em 0.4em;
                border-radius: 3px;
                font-family: monospace;
            }
            a {
                color: #0366d6;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
            """
        head.append(style_tag)
        
        # Create HTML structure
        html_tag = soup.new_tag('html')
        html_tag.append(head)
        
        # Wrap existing content in body tag
        body = soup.new_tag('body')
        for content in list(soup.contents):
            body.append(content)
        
        html_tag.append(body)
        
        # Create the final HTML
        final_html = f"<!DOCTYPE html>\n{str(html_tag)}"
        
        # Write HTML to file
        with open(output_path, 'w', encoding='utf-8') as html_file:
            html_file.write(final_html)
        
        logger.info(f"HTML report generated: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error converting Markdown to HTML: {e}", exc_info=True)
        return ""

def convert_all_markdown_reports(report_dir: str, recursive: bool = True) -> Dict[str, str]:
    """
    Convert all Markdown reports in a directory to HTML.
    
    Args:
        report_dir: Directory containing reports
        recursive: Whether to search subdirectories recursively
        
    Returns:
        Dictionary mapping original Markdown paths to generated HTML paths
    """
    converted_files = {}
    
    try:
        # Find all markdown files
        search_pattern = '**/*.md' if recursive else '*.md'
        markdown_files = list(Path(report_dir).glob(search_pattern))
        
        if not markdown_files:
            logger.warning(f"No Markdown files found in {report_dir}")
            return {}
        
        logger.info(f"Found {len(markdown_files)} Markdown files to convert")
        
        # Convert each file
        for md_file in markdown_files:
            md_path = str(md_file)
            html_path = os.path.splitext(md_path)[0] + '.html'
            
            result = convert_markdown_to_html(md_path, output_path=html_path)
            
            if result:
                converted_files[md_path] = html_path
        
        logger.info(f"Successfully converted {len(converted_files)} files to HTML")
        return converted_files
        
    except Exception as e:
        logger.error(f"Error converting Markdown reports: {e}", exc_info=True)
        return converted_files

def create_report_index(report_dir: str, output_path: Optional[str] = None) -> str:
    """
    Create an HTML index of all reports in a directory.
    
    Args:
        report_dir: Directory containing reports
        output_path: Path to save the index HTML (default: index.html in report_dir)
        
    Returns:
        Path to the generated index file
    """
    try:
        if not os.path.exists(report_dir):
            logger.error(f"Report directory not found: {report_dir}")
            return ""
        
        # Set default output path
        if output_path is None:
            output_path = os.path.join(report_dir, "index.html")
        
        # Find all HTML reports
        html_files = list(Path(report_dir).glob("**/*.html"))
        html_files = [f for f in html_files if f.name != os.path.basename(output_path)]
        
        if not html_files:
            logger.warning(f"No HTML files found in {report_dir}")
            return ""
        
        # Create index HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("<!DOCTYPE html>\n")
            f.write("<html>\n<head>\n")
            f.write("  <meta charset=\"utf-8\">\n")
            f.write("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
            f.write("  <title>Report Index</title>\n")
            f.write("  <style>\n")
            f.write("    body {\n")
            f.write("      font-family: Arial, sans-serif;\n")
            f.write("      line-height: 1.6;\n")
            f.write("      max-width: 1200px;\n")
            f.write("      margin: 0 auto;\n")
            f.write("      padding: 20px;\n")
            f.write("      color: #333;\n")
            f.write("    }\n")
            f.write("    h1 { color: #2c3e50; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }\n")
            f.write("    .report-list { list-style-type: none; padding: 0; }\n")
            f.write("    .report-item { margin-bottom: 15px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }\n")
            f.write("    .report-title { font-size: 1.2em; color: #0366d6; text-decoration: none; }\n")
            f.write("    .report-title:hover { text-decoration: underline; }\n")
            f.write("    .report-path { color: #666; font-size: 0.9em; margin-top: 5px; font-style: italic; }\n")
            f.write("  </style>\n")
            f.write("</head>\n<body>\n")
            f.write("  <h1>Available Reports</h1>\n")
            f.write("  <ul class=\"report-list\">\n")
            
            # Group reports by type
            report_types = {}
            for html_file in html_files:
                # Get report type from parent directory name
                report_type = html_file.parent.name
                
                if report_type not in report_types:
                    report_types[report_type] = []
                
                # Try to extract title from HTML
                title = html_file.stem
                try:
                    with open(html_file, 'r', encoding='utf-8') as html_content:
                        soup = BeautifulSoup(html_content.read(), 'html.parser')
                        title_tag = soup.find('title')
                        if title_tag and title_tag.string:
                            title = title_tag.string
                except Exception as e:
                    logger.warning(f"Could not extract title from {html_file}: {e}")
                
                # Add to report types
                relative_path = os.path.relpath(html_file, report_dir)
                report_types[report_type].append((str(html_file), title, relative_path))
            
            # Write report groups
            for report_type, reports in sorted(report_types.items()):
                if reports:
                    f.write(f"    <h2>{report_type.replace('_', ' ').title()}</h2>\n")
                    
                    for _, title, rel_path in sorted(reports):
                        f.write("    <li class=\"report-item\">\n")
                        f.write(f"      <a href=\"{rel_path.replace(os.sep, '/')}\" class=\"report-title\">{title}</a>\n")
                        f.write(f"      <div class=\"report-path\">{rel_path}</div>\n")
                        f.write("    </li>\n")
            
            f.write("  </ul>\n")
            f.write("</body>\n</html>")
        
        logger.info(f"Report index created: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating report index: {e}", exc_info=True)
        return ""

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        
        if os.path.isdir(input_path):
            # Convert all reports in directory
            converted = convert_all_markdown_reports(input_path)
            
            # Create index
            index_path = create_report_index(input_path)
            
            if converted:
                print(f"Converted {len(converted)} reports to HTML")
                
                if index_path:
                    print(f"Created index: {index_path}")
            else:
                print("No reports were converted")
        elif os.path.isfile(input_path) and input_path.lower().endswith('.md'):
            # Convert single file
            output = convert_markdown_to_html(input_path)
            
            if output:
                print(f"Converted {input_path} to {output}")
            else:
                print(f"Failed to convert {input_path}")
        else:
            print("Invalid input path. Please provide a Markdown file or directory.")
    else:
        print("Usage: python html_report_converter.py <markdown_file_or_directory>")

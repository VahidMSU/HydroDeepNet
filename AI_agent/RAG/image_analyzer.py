import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from agno import Agent
from agno.models.openai import OpenAIChat
from datetime import datetime

# Configure logging
from Logger import LoggerSetup
logger_setup = LoggerSetup()
logger = logger_setup.setup_logger()

class ImageAnalyzer:
    """
    A specialized module for analyzing and interpreting images in scientific contexts.
    Works with visualization_manager.py to provide advanced image analysis capabilities.
    """
    
    def __init__(self, base_path=None, model="gpt-4o"):
        """
        Initialize the ImageAnalyzer with a base path and model.
        
        Args:
            base_path: Base directory for image discovery
            model: Model to use for image analysis (default: gpt-4o)
        """
        self.base_path = base_path
        self.model_name = model
        self.discovered_images = []
        self.analysis_cache = {}
        self.image_metadata = {}
        self.image_agent = None
        
        # Initialize vision model
        self._initialize_vision_agent()
        
        # Image categorization schemas
        self.image_categories = {
            "climate": ["climate", "temperature", "precipitation", "seasonal"],
            "vegetation": ["NDVI", "EVI", "LAI", "ET", "vegetation"],
            "soil": ["soil", "texture", "distribution", "map"],
            "groundwater": ["groundwater", "aquifer", "water", "H_COND"],
            "land_use": ["cdl", "crop", "land", "diversity"]
        }
    
    def _initialize_vision_agent(self):
        """Initialize the vision model for image analysis."""
        try:
            self.image_agent = Agent(
                model=OpenAIChat(id=self.model_name),
                agent_id="image-analyst",
                name="Scientific Image Analyzer",
                markdown=True,
                debug_mode=True,
                instructions=[
                    "You are a Scientific Image Analyzer specialized in environmental and hydrological data.",
                    "You examine scientific visualizations, maps, charts, and graphs with expert precision.",
                    "You can identify visualization types, extract data points, and interpret trends.",
                    "You understand spatial patterns in maps and can correlate visual features with data.",
                    "You can detect outliers, anomalies, and significant features in scientific charts.",
                    "You provide structured, detailed analyses focusing on scientific relevance."
                ]
            )
            logger.info("Vision agent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing vision agent: {str(e)}")
            self.image_agent = None
            return False
    
    def discover_images(self, custom_path=None):
        """
        Discover image files in the specified directory.
        
        Args:
            custom_path: Optional custom path to scan for images
        
        Returns:
            Dictionary with counts of discovered images by type
        """
        search_path = custom_path or self.base_path
        if not search_path:
            logger.error("No path specified for image discovery")
            return {"error": "No path specified"}
        
        self.discovered_images = []
        image_counts = {"png": 0, "jpg": 0, "jpeg": 0, "tiff": 0, "bmp": 0}
        
        try:
            # Walk through directory structure
            for root, _, files in os.walk(search_path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()[1:]
                    if ext in ["png", "jpg", "jpeg", "tiff", "bmp"]:
                        full_path = os.path.join(root, file)
                        self.discovered_images.append(full_path)
                        image_counts[ext] = image_counts.get(ext, 0) + 1
                        
                        # Generate basic metadata
                        self._extract_image_metadata(full_path)
            
            logger.info(f"Discovered {len(self.discovered_images)} images: {image_counts}")
            return image_counts
        except Exception as e:
            logger.error(f"Error discovering images: {str(e)}")
            return {"error": str(e)}
    
    def _extract_image_metadata(self, image_path):
        """
        Extract basic metadata from an image file.
        
        Args:
            image_path: Path to the image file
        """
        try:
            img = Image.open(image_path)
            filename = os.path.basename(image_path)
            
            metadata = {
                "filename": filename,
                "path": image_path,
                "format": img.format,
                "size": img.size,
                "mode": img.mode,
                "created": datetime.fromtimestamp(os.path.getctime(image_path)),
                "modified": datetime.fromtimestamp(os.path.getmtime(image_path)),
                "category": self._categorize_image(filename)
            }
            
            self.image_metadata[image_path] = metadata
        except Exception as e:
            logger.error(f"Error extracting metadata for {image_path}: {str(e)}")
    
    def _categorize_image(self, filename):
        """
        Categorize an image based on its filename.
        
        Args:
            filename: Image filename
            
        Returns:
            List of categories that match the image
        """
        filename_lower = filename.lower()
        categories = []
        
        for category, keywords in self.image_categories.items():
            if any(keyword.lower() in filename_lower for keyword in keywords):
                categories.append(category)
        
        return categories if categories else ["uncategorized"]
    
    def analyze_image(self, image_path, context_info=None, force_reanalysis=False):
        """
        Analyze an image using the vision agent.
        
        Args:
            image_path: Path to the image to analyze
            context_info: Optional additional context to aid analysis
            force_reanalysis: Force reanalysis even if cached result exists
            
        Returns:
            Analysis result as text
        """
        # Check if we have a cached analysis
        if not force_reanalysis and image_path in self.analysis_cache:
            logger.info(f"Using cached analysis for {os.path.basename(image_path)}")
            return self.analysis_cache[image_path]
        
        # Verify image exists
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"
        
        # Check if vision agent is available
        if self.image_agent is None:
            return "Error: Vision analysis agent is not initialized"
        
        try:
            # Prepare image for analysis
            img_obj = Image.open(image_path)
            
            # Create analysis prompt with context
            filename = os.path.basename(image_path)
            prompt = f"""
            Please analyze this scientific visualization/image: {filename}
            
            Focus on these aspects:
            1. What type of visualization is this (map, chart, graph, etc.)?
            2. What variables or data are being represented?
            3. What are the key patterns, trends, or features visible?
            4. What scientific insights can be drawn from this visualization?
            5. How might this relate to environmental or hydrological analysis?
            
            {context_info or ""}
            
            Provide a comprehensive, structured analysis with your key observations.
            """
            
            # Use the image agent to analyze
            logger.info(f"Analyzing image: {filename}")
            analysis = self.image_agent.print_response(prompt, images=[img_obj], stream=False)
            
            # Cache the result
            self.analysis_cache[image_path] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            return f"Error during image analysis: {str(e)}"
    
    def find_images_by_name(self, image_name):
        """
        Find images by name (with or without extension).
        
        Args:
            image_name: Full or partial image name to search for
            
        Returns:
            List of matching image paths
        """
        # Remove extension if present to improve matching
        base_image_name = os.path.splitext(image_name)[0]
        
        matching_images = []
        for path in self.discovered_images:
            path_basename = os.path.basename(path)
            path_basename_noext = os.path.splitext(path_basename)[0]
            
            # Match with or without extension
            if (base_image_name.lower() in path_basename_noext.lower() or 
                image_name.lower() in path_basename.lower()):
                matching_images.append(path)
        
        return matching_images
    
    def get_related_data_context(self, image_path, data_files=None):
        """
        Find related data files that may provide context for an image.
        
        Args:
            image_path: Path to the image
            data_files: Optional list of data files to search through
            
        Returns:
            Context information as text
        """
        image_name = os.path.basename(image_path).lower()
        context_parts = []
        
        # Extract potential keywords from the filename
        parts = image_name.replace('.png', '').replace('.jpg', '').split('_')
        
        if not data_files:
            return "No data files provided to search for context."
        
        # Look for related CSV files
        for csv_path in [f for f in data_files if f.lower().endswith('.csv')]:
            csv_name = os.path.basename(csv_path).lower()
            
            # Check if any part of the image name matches the CSV name
            if any(part in csv_name for part in parts if len(part) > 2):
                context_parts.append(f"Related data file: {os.path.basename(csv_path)}")
                
                # Try to get column names from the CSV
                try:
                    df = pd.read_csv(csv_path, nrows=1)
                    columns = df.columns.tolist()
                    context_parts.append(f"Data columns: {', '.join(columns)}")
                except Exception as e:
                    logger.error(f"Error reading CSV {csv_path}: {str(e)}")
        
        # Look for related markdown files that might have descriptions
        for md_path in [f for f in data_files if f.lower().endswith('.md')]:
            md_name = os.path.basename(md_path).lower()
            
            # Check if any part of the image name matches the markdown name
            if any(part in md_name for part in parts if len(part) > 2):
                context_parts.append(f"Related documentation: {os.path.basename(md_path)}")
                
                # Try to extract relevant sections from the markdown
                try:
                    with open(md_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Look for sections that mention the image name
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if image_name in line.lower():
                            # Extract the surrounding context
                            start = max(0, i-5)
                            end = min(len(lines), i+5)
                            relevant_lines = lines[start:end]
                            context_parts.append("Related documentation excerpt:")
                            context_parts.append("\n".join(relevant_lines))
                            break
                except Exception as e:
                    logger.error(f"Error reading markdown {md_path}: {str(e)}")
        
        # If we have context, format it nicely
        if context_parts:
            return "Context information:\n" + "\n".join(context_parts)
        else:
            return "No additional context information found for this image."
    
    def compare_images(self, image_paths, labels=None):
        """
        Compare multiple images and provide analysis of similarities/differences.
        
        Args:
            image_paths: List of paths to images to compare
            labels: Optional list of labels for the images
            
        Returns:
            Comparison analysis
        """
        if len(image_paths) < 2:
            return "Need at least two images to compare"
        
        if not self.image_agent:
            return "Image analysis agent not initialized"
        
        try:
            # Prepare images
            images = [Image.open(path) for path in image_paths]
            image_names = [os.path.basename(path) for path in image_paths]
            
            # Create labels if not provided
            if not labels:
                labels = [f"Image {i+1}: {name}" for i, name in enumerate(image_names)]
            
            # Create comparison prompt
            prompt = f"""
            Please compare these {len(images)} scientific visualizations:
            {', '.join(f"{label} ({name})" for label, name in zip(labels, image_names))}
            
            Focus on:
            1. How these visualizations relate to each other
            2. Key similarities and differences between them
            3. What temporal, spatial, or thematic patterns emerge when viewed together
            4. What scientific insights can be gained by comparing these visualizations
            
            Provide a comprehensive comparison that highlights the most important relationships.
            """
            
            # Use the image agent for comparison
            comparison = self.image_agent.print_response(prompt, images=images, stream=False)
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing images: {str(e)}")
            return f"Error during image comparison: {str(e)}"
    
    def generate_image_report(self, image_path, output_format="markdown"):
        """
        Generate a comprehensive report for an image.
        
        Args:
            image_path: Path to the image
            output_format: Format for the report ("markdown" or "html")
            
        Returns:
            Report in the specified format
        """
        if not os.path.exists(image_path):
            return f"Error: Image not found at {image_path}"
        
        try:
            # Get basic metadata
            if image_path not in self.image_metadata:
                self._extract_image_metadata(image_path)
            
            metadata = self.image_metadata.get(image_path, {})
            
            # Get or generate analysis
            analysis = self.analyze_image(image_path)
            
            # Generate report based on format
            if output_format.lower() == "markdown":
                return self._generate_markdown_report(image_path, metadata, analysis)
            elif output_format.lower() == "html":
                return self._generate_html_report(image_path, metadata, analysis)
            else:
                return f"Unsupported output format: {output_format}"
                
        except Exception as e:
            logger.error(f"Error generating report for {image_path}: {str(e)}")
            return f"Error generating image report: {str(e)}"
    
    def _generate_markdown_report(self, image_path, metadata, analysis):
        """Generate a markdown report for an image."""
        filename = metadata.get("filename", os.path.basename(image_path))
        
        # Format creation/modification dates
        created = metadata.get("created", "Unknown")
        if isinstance(created, datetime):
            created = created.strftime("%Y-%m-%d %H:%M:%S")
            
        modified = metadata.get("modified", "Unknown")
        if isinstance(modified, datetime):
            modified = modified.strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate the report
        report = f"""# Image Analysis Report: {filename}

## Metadata
- **Filename**: {filename}
- **Format**: {metadata.get("format", "Unknown")}
- **Dimensions**: {metadata.get("size", "Unknown")}
- **Color Mode**: {metadata.get("mode", "Unknown")}
- **Created**: {created}
- **Modified**: {modified}
- **Categories**: {", ".join(metadata.get("category", ["Uncategorized"]))}

## Analysis
{analysis}

## Recommendations
Based on this analysis, consider exploring:
- Related datasets that may provide context for these observations
- Time series data to understand temporal patterns
- Spatial analysis to further investigate geographic distributions
- Additional visualization methods that might reveal other patterns

---
*Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        return report
    
    def _generate_html_report(self, image_path, metadata, analysis):
        """Generate an HTML report for an image."""
        filename = metadata.get("filename", os.path.basename(image_path))
        
        # Format creation/modification dates
        created = metadata.get("created", "Unknown")
        if isinstance(created, datetime):
            created = created.strftime("%Y-%m-%d %H:%M:%S")
            
        modified = metadata.get("modified", "Unknown")
        if isinstance(modified, datetime):
            modified = modified.strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate HTML report
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Image Analysis: {filename}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2 {{ color: #2c3e50; }}
        .metadata {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .analysis {{ margin: 20px 0; }}
        .image-container {{ text-align: center; margin: 20px 0; }}
        img {{ max-width: 100%; border: 1px solid #ddd; }}
        .footer {{ font-size: 0.8em; color: #7f8c8d; margin-top: 30px; border-top: 1px solid #eee; padding-top: 10px; }}
    </style>
</head>
<body>
    <h1>Image Analysis Report: {filename}</h1>
    
    <div class="image-container">
        <img src="file://{image_path.replace(os.sep, '/')}" alt="{filename}">
    </div>
    
    <div class="metadata">
        <h2>Metadata</h2>
        <ul>
            <li><strong>Filename:</strong> {filename}</li>
            <li><strong>Format:</strong> {metadata.get("format", "Unknown")}</li>
            <li><strong>Dimensions:</strong> {metadata.get("size", "Unknown")}</li>
            <li><strong>Color Mode:</strong> {metadata.get("mode", "Unknown")}</li>
            <li><strong>Created:</strong> {created}</li>
            <li><strong>Modified:</strong> {modified}</li>
            <li><strong>Categories:</strong> {", ".join(metadata.get("category", ["Uncategorized"]))}</li>
        </ul>
    </div>
    
    <div class="analysis">
        <h2>Analysis</h2>
        {analysis.replace('\n', '<br>')}
    </div>
    
    <div class="recommendations">
        <h2>Recommendations</h2>
        <p>Based on this analysis, consider exploring:</p>
        <ul>
            <li>Related datasets that may provide context for these observations</li>
            <li>Time series data to understand temporal patterns</li>
            <li>Spatial analysis to further investigate geographic distributions</li>
            <li>Additional visualization methods that might reveal other patterns</li>
        </ul>
    </div>
    
    <div class="footer">
        Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
</body>
</html>
"""
        return html 
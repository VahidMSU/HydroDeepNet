import os
import re
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
import base64
from PIL import Image
import io

from Logger import LoggerSetup
# Initialize logger using setup_logger method
logger_setup = LoggerSetup()
logger = logger_setup.setup_logger()

class ImageAnalyzer:
    """Handles image analysis for the document reader."""
    
    def __init__(self, document_reader):
        """Initialize the image analyzer with reference to the document reader."""
        self.document_reader = document_reader
        self.image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        self.image_name_pattern = re.compile(r'\b(?:image|figure|fig\.?|chart|graph|plot|diagram|photo|picture)\s*(?:of|showing|displaying|titled|named|called|labeled)?\s*["\']?([^"\']+)["\']?', re.IGNORECASE)
    
    def extract_image_names_from_message(self, message: str) -> List[str]:
        """Extract potential image names from a user message using regex patterns."""
        try:
            # First, look for direct file references
            potential_names = []
            
            # Look for common image-related terms followed by names
            matches = self.image_name_pattern.findall(message)
            for match in matches:
                potential_names.append(match.strip())
            
            # Look for any words with image extensions
            for ext in self.image_extensions:
                pattern = re.compile(r'\b\w+' + re.escape(ext) + r'\b', re.IGNORECASE)
                file_matches = pattern.findall(message)
                potential_names.extend(file_matches)
            
            # Remove duplicates and return
            return list(set(potential_names))
            
        except Exception as e:
            logger.error(f"Error extracting image names: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def find_matching_images(self, search_term: str) -> List[str]:
        """Find images that match the given search term."""
        matching_images = []
        
        # Get all image files
        all_images = self.document_reader.discovered_files.get('image', [])
        
        # If it's a number, treat it as an index
        if search_term.isdigit():
            index = int(search_term) - 1
            if 0 <= index < len(all_images):
                return [all_images[index]]
            else:
                return []
        
        # Otherwise, match by filename
        for image_path in all_images:
            filename = os.path.basename(image_path)
            if search_term.lower() in filename.lower():
                matching_images.append(image_path)
        
        return matching_images
    
    def encode_image(self, image_path: str) -> Optional[str]:
        """Encode an image file to base64 for analysis."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """Get metadata about an image (dimensions, format, etc)."""
        try:
            with Image.open(image_path) as img:
                return {
                    'filename': os.path.basename(image_path),
                    'path': image_path,
                    'format': img.format,
                    'mode': img.mode,
                    'width': img.width,
                    'height': img.height,
                    'size': os.path.getsize(image_path) / 1024  # size in KB
                }
        except Exception as e:
            logger.error(f"Error getting image metadata for {image_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'filename': os.path.basename(image_path),
                'path': image_path,
                'error': str(e)
            }
    
    def analyze_image(self, image_path: str) -> str:
        """Analyze an image using the visual analyst agent."""
        try:
            # Check if the file exists
            if not os.path.exists(image_path):
                return f"Error: Image file not found: {image_path}"
            
            # Get image metadata
            metadata = self.get_image_metadata(image_path)
            
            # Encode the image
            encoded_image = self.encode_image(image_path)
            if not encoded_image:
                return f"Error: Failed to encode image {os.path.basename(image_path)}"
            
            # Check if the visual analyst agent exists
            if not hasattr(self.document_reader, 'visual_analyst_agent') or not self.document_reader.visual_analyst_agent:
                return "Visual analyst agent not initialized. Cannot analyze image."
            
            # Prepare the analysis prompt
            prompt = f"""Analyze the provided image in detail. The image is named '{os.path.basename(image_path)}' 
            with dimensions {metadata['width']}x{metadata['height']}.
            
            Provide a complete description of what you see in the image, including:
            1. The main subject or content of the image
            2. Any text content visible in the image
            3. Any charts, graphs, or data visualizations (describe the axes, trends, and key data points)
            4. Any technical diagrams or schematics (describe components and their relationships)
            5. The overall context or purpose of this image
            
            Be precise, technical, and detailed in your analysis. Focus on extracting actionable information 
            and insights from the image rather than just describing what it looks like.
            """
            
            # Get the analysis from the agent
            analysis = self.document_reader.visual_analyst_agent.run(prompt, image=encoded_image)
            
            # Clean up the response
            if hasattr(self.document_reader, 'response_handler'):
                analysis = self.document_reader.response_handler.clean_response(analysis)
            
            # Add a header
            result = f"## Analysis of {os.path.basename(image_path)}\n\n{analysis}"
            
            # Add the image to the context for future reference
            self._add_image_to_context(image_path, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error analyzing image: {str(e)}"
    
    def _add_image_to_context(self, image_path: str, analysis: str) -> None:
        """Add the analyzed image to the document reader context."""
        try:
            # Ensure the analyzed_images dict exists in context
            if 'analyzed_images' not in self.document_reader.context:
                self.document_reader.context['analyzed_images'] = {}
            
            # Add the image analysis to the context
            self.document_reader.context['analyzed_images'][os.path.basename(image_path)] = {
                'path': image_path,
                'analysis': analysis,
                'timestamp': self.document_reader.context.get('timestamp', 0)
            }
            
        except Exception as e:
            logger.error(f"Error adding image to context: {str(e)}")
            logger.error(traceback.format_exc())
    
    def analyze_specific_image(self, message: str, pending_actions: Dict) -> Tuple[str, Dict]:
        """Analyze a specific image based on user message or pending actions."""
        try:
            # Check if there's a pending image analysis action
            if 'image_analysis' in pending_actions:
                image_content = pending_actions['image_analysis'].get('content', '').strip()
                
                # Find images that match the specified content
                matching_images = []
                
                if image_content:
                    # Search for image by the specified name
                    matching_images = self.find_matching_images(image_content)
                else:
                    # Try to extract image names from the message
                    potential_names = self.extract_image_names_from_message(message)
                    for name in potential_names:
                        found_images = self.find_matching_images(name)
                        matching_images.extend(found_images)
                
                # Check if any images were found
                if not matching_images:
                    # No matching images found, check if we have any images at all
                    all_images = self.document_reader.discovered_files.get('image', [])
                    
                    if not all_images:
                        return "I don't see any images in the provided documents.", pending_actions
                    
                    if len(all_images) == 1:
                        # Only one image exists, use it
                        image_path = all_images[0]
                        analysis = self.analyze_image(image_path)
                        pending_actions['image_analysis']['result'] = analysis
                        return analysis, pending_actions
                    else:
                        # Multiple images exist but none match, provide options
                        options = "I found multiple images. Which one would you like me to analyze?\n\n"
                        for i, img_path in enumerate(all_images):
                            options += f"{i+1}. {os.path.basename(img_path)}\n"
                        pending_actions['image_analysis']['options'] = all_images
                        return options, pending_actions
                
                elif len(matching_images) == 1:
                    # One matching image found, analyze it
                    image_path = matching_images[0]
                    analysis = self.analyze_image(image_path)
                    pending_actions['image_analysis']['result'] = analysis
                    return analysis, pending_actions
                
                else:
                    # Multiple matching images found, provide options
                    options = f"I found multiple images matching '{image_content}'. Which one would you like me to analyze?\n\n"
                    for i, img_path in enumerate(matching_images):
                        options += f"{i+1}. {os.path.basename(img_path)}\n"
                    pending_actions['image_analysis']['options'] = matching_images
                    return options, pending_actions
                
            # No pending image analysis action
            return message, pending_actions
            
        except Exception as e:
            logger.error(f"Error in analyze_specific_image: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error analyzing image: {str(e)}", pending_actions 
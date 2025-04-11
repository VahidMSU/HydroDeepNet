import os
import re
import pandas as pd
from context_manager import context

import json 
        
def discover_files(base_path, config, logger):
    """Auto-discover and categorize files in the given directory."""
    logger.info(f"Auto-discovering files in {base_path}")
    
    # Initialize discovered files by category
    discovered_files = {
        'pdf': [],
        'csv': [],
        'json': [],
        'docx': [],
        'md': [],
        'txt': [],
        'png': [],
        'jpg': [],
        'html': [],
        'other': []
    }
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()[1:]
            
            # Categorize file by extension
            if file_ext in discovered_files:
                discovered_files[file_ext].append(file_path)
            else:
                discovered_files['other'].append(file_path)
    
    # Log information about discovered files
    discovered_count = {k: len(v) for k, v in discovered_files.items() if v}
    logger.info(f"Discovered file counts: {discovered_count}")
    
    # Update config with discovered files
    for file_type, file_paths in discovered_files.items():
        if file_paths:
            if file_type == 'csv':
                for i, path in enumerate(file_paths):
                    table_name = f"{file_type}_{i}_docs"
                    if 'csv' not in config:
                        config['csv'] = []
                    
                    # Check if path is already in config
                    if not any(cfg.get('path') == path for cfg in config['csv']):
                        config['csv'].append({
                            'path': path,
                            'table_name': table_name
                        })
            elif file_type == 'json':
                for i, path in enumerate(file_paths):
                    table_name = f"{file_type}_{i}_docs"
                    if 'json' not in config:
                        config['json'] = []
                    
                    # Check if path is already in config
                    if not any(cfg.get('path') == path for cfg in config['json']):
                        config['json'].append({
                            'path': path,
                            'table_name': table_name
                        })
            elif file_type in ['png', 'jpg']:
                for i, path in enumerate(file_paths):
                    table_name = f"image_{i}_analysis"
                    if 'image' not in config:
                        config['image'] = []
                    
                    # Check if path is already in config
                    if not any(cfg.get('path') == path for cfg in config['image']):
                        config['image'].append({
                            'path': path,
                            'table_name': table_name
                        })
                    # Create a basic image summary
                    create_image_summary(path)
            elif file_type in ['pdf', 'docx', 'md', 'txt']:
                for i, path in enumerate(file_paths):
                    table_name = f"{file_type}_{i}_docs"
                    if file_type not in config:
                        config[file_type] = []
                    
                    # Check if path is already in config
                    if not any(cfg.get('path') == path for cfg in config[file_type]):
                        config[file_type].append({
                            'path': path,
                            'table_name': table_name
                        })
    
    logger.info(f"Discovered files: {json.dumps({k: len(v) for k, v in discovered_files.items()})}")
    return discovered_files
    


def extract_image_name(message, discovered_files=None):
    """Extract the image name from a user message."""
    patterns = [
        r'analyze\s+(\S+(?:\.png|\.jpg)?)',
        r'analyse\s+(\S+(?:\.png|\.jpg)?)',
        r'analysis\s+(\S+(?:\.png|\.jpg)?)',
        r'show\s+(\S+(?:\.png|\.jpg)?)',
        r'image\s+(\S+(?:\.png|\.jpg)?)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message.lower())
        if match:
            # Get the image name without extension if specified
            image_name = match.group(1)
            # Remove extension if present
            image_name = os.path.splitext(image_name)[0]
            return image_name
    
    # If discovered_files is provided, check for mentions of files that match image files
    if discovered_files is None:
        discovered_files = context.discovered_files
        
    for img_type in ['png', 'jpg']:
        for file_path in discovered_files.get(img_type, []):
            file_name = os.path.basename(file_path)
            # Remove extension for comparison
            base_name = os.path.splitext(file_name)[0]
            
            # Check if the base name appears in the message
            if base_name.lower() in message.lower():
                return base_name
    
    return None

def create_image_summary(path):
    """Create a summary of an image file."""
    try:
        from PIL import Image
        image = Image.open(path)
        summary = f"""
        Image file: {path}
        Size: {image.size}
        Format: {image.format}
        Mode: {image.mode}
        Is animated: {getattr(image, 'is_animated', False)}
        Frames: {getattr(image, 'n_frames', 1)}
        """
        
        # Save summary to context
        context.save_analysis_result(path + "_metadata", summary.strip())
        
        return summary.strip()
    except Exception as e:
        return f"Error creating image summary: {str(e)}"

def calculate_fuzzy_match(str1, str2):
    """Calculate fuzzy match score between two strings."""
    # Convert both strings to lowercase for comparison
    str1 = str1.lower()
    str2 = str2.lower()
    
    # If either string is empty, return 0
    if not str1 or not str2:
        return 0
    
    # If strings are identical, return 1
    if str1 == str2:
        return 1
    
    # Split into words and find common words
    words1 = set(str1.split())
    words2 = set(str2.split())
    
    # Calculate Jaccard similarity for words
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0
    
    # Calculate word-based similarity
    word_similarity = intersection / union
    
    # Calculate character-based similarity for non-exact word matches
    char_similarity = len(set(str1).intersection(set(str2))) / len(set(str1).union(set(str2)))
    
    # Return weighted average of word and character similarity
    return 0.7 * word_similarity + 0.3 * char_similarity

def extract_main_topic(query):
    """Extract the main topic from a query."""
    topic_indicators = {
        'solar': ['solar', 'pv', 'photovoltaic', 'sun', 'energy', 'power', 'renewable'],
        'groundwater': ['water', 'groundwater', 'aquifer', 'well', 'hydrology'],
        'precipitation': ['rain', 'precipitation', 'rainfall', 'weather', 'storm', 'climate'],
        'temperature': ['temperature', 'heat', 'warm', 'cold', 'climate'],
        'statistics': ['statistics', 'stats', 'average', 'mean', 'median', 'trend', 'analysis']
    }
    
    # Count indicators for each topic
    topic_scores = {topic: 0 for topic in topic_indicators}
    for topic, indicators in topic_indicators.items():
        for indicator in indicators:
            if indicator in query.lower():
                topic_scores[topic] += 1
    
    # Find topic with highest score
    if any(score > 0 for score in topic_scores.values()):
        main_topic = max(topic_scores.items(), key=lambda x: x[1])[0]
        return main_topic
    else:
        return "general_inquiry"

import os
import re
import pandas as pd

def calculate_similarity(str1, str2):
    """Calculate simple similarity between two strings using Jaccard similarity."""
    if not str1 or not str2:
        return 0
        
    # Convert to sets of words for a simple Jaccard similarity
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    
    if not set1 or not set2:
        return 0
        
    # Calculate Jaccard similarity: intersection over union
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0

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
    if discovered_files:
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
        return summary.strip()
    except Exception as e:
        return f"Error creating image summary: {str(e)}"

def is_response_complete(text):
    """Check if a response appears to be complete."""
    if not text:
        return False
        
    # Check for cut-off indicators
    incomplete_indicators = [
        '...', '…',  # Ellipsis at the end
        lambda t: t.endswith((':', ',', '-', '(', '{')),  # Ends with punctuation that expects more content
        lambda t: t.count('(') > t.count(')'),  # Unbalanced parentheses
        lambda t: t.count('{') > t.count('}'),  # Unbalanced braces
        lambda t: t.count('[') > t.count(']'),  # Unbalanced brackets
    ]
    
    for indicator in incomplete_indicators:
        if callable(indicator):
            if indicator(text):
                return False
        elif text.strip().endswith(indicator):
            return False
            
    return True

def deduplicate_content(text, similarity_threshold=0.8):
    """Remove duplicated content from a response."""
    if not text:
        return text
        
    # Split text into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) <= 1:
        return text
        
    # Filter out duplicate paragraphs while preserving order
    unique_paragraphs = []
    seen_paragraphs = set()
    
    for paragraph in paragraphs:
        # Check if this is a duplicate or very similar to an existing paragraph
        is_duplicate = paragraph in seen_paragraphs
        is_similar = any(calculate_similarity(paragraph, p) > similarity_threshold for p in unique_paragraphs)
        
        if not is_duplicate and not is_similar:
            unique_paragraphs.append(paragraph)
            seen_paragraphs.add(paragraph)
            
    # Reconstruct the text
    return '\n\n'.join(unique_paragraphs)

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

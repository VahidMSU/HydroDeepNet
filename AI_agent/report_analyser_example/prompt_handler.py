"""
Prompt handling utilities for natural language processing of user queries.
This module provides functions to extract semantic information from text input.
"""
import re
import logging
import pandas as pd

def extract_keywords(text):
    """
    Extract important keywords from a message.
    
    Args:
        text (str): The input text to extract keywords from
        
    Returns:
        list: List of extracted keywords
    """
    # Common English stopwords to filter out
    common_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
        'for', 'with', 'about', 'is', 'are', 'was', 'were', 'be', 'been', 
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could',
        'will', 'would', 'should', 'may', 'might', 'must', 'shall'
    }
    
    # Split text into words and convert to lowercase
    words = text.lower().split()
    
    # Remove punctuation from words
    cleaned_words = []
    for word in words:
        # Remove common punctuation
        word = word.strip('.,;:!?()[]{}"\'-')
        if word and len(word) > 2:  # Only keep words with 3+ characters
            cleaned_words.append(word)
    
    # Filter out common words
    keywords = [word for word in cleaned_words if word not in common_words]
    
    return keywords


def rank_keywords(keywords, text):
    """
    Rank extracted keywords by importance based on frequency and position.
    
    Args:
        keywords (list): List of extracted keywords
        text (str): The original text
        
    Returns:
        list: Ranked list of keywords with scores
    """
    # Count occurrences of each keyword
    keyword_counts = {}
    for keyword in keywords:
        if keyword not in keyword_counts:
            keyword_counts[keyword] = text.lower().count(keyword)
    
    # Score keywords based on frequency and other factors
    keyword_scores = []
    for keyword, count in keyword_counts.items():
        # Calculate position score (earlier = more important)
        position = text.lower().find(keyword)
        position_score = 1.0 if position < len(text) // 3 else 0.5
        
        # Calculate final score
        score = count * position_score
        
        keyword_scores.append((keyword, score))
    
    # Sort by score in descending order
    return sorted(keyword_scores, key=lambda x: x[1], reverse=True)


def clean_response_output(text, logger=None):
    """
    Clean output text by removing debug information and formatting.
    
    Args:
        text (str): The text to clean
        logger (logging.Logger, optional): Logger to use for error reporting
        
    Returns:
        str: The cleaned text
    """
    if not text:
        return ""
    
    try:
        # Remove ANSI color codes
        text = re.sub(r'\x1b\[[0-9;]*m', '', text)
        
        # Remove debug headers and info
        text = re.sub(r'DEBUG.*?METRICS.*?\n', '', text, flags=re.DOTALL)
        text = re.sub(r'DEBUG .*?OpenAI Response.*?\n', '', text, flags=re.DOTALL)
        text = re.sub(r'DEBUG Added .*?AgentMemory.*?\n', '', text, flags=re.DOTALL)
        text = re.sub(r'DEBUG Logging Agent Run.*?\n', '', text, flags=re.DOTALL)
        text = re.sub(r'DEBUG \*{15} Agent Run.*?\n', '', text, flags=re.DOTALL)
        
        # Remove box characters and metadata surrounding the message and response
        text = re.sub(r'┏━+\s*Message\s*━+┓.*?┗━+┛', '', text, flags=re.DOTALL)
        text = re.sub(r'┏━+\s*Response.*?┓.*?┗━+┛', '', text, flags=re.DOTALL)
        
        # Remove individual box characters that may remain
        text = re.sub(r'[┏┓┗┛━┃]', '', text)
        
        # Remove header lines
        text = re.sub(r'Message.*?\n', '', text, flags=re.DOTALL)
        text = re.sub(r'Response \(\d+\.\d+s\).*?\n', '', text, flags=re.DOTALL)
        text = re.sub(r'Response.*?\n', '', text, flags=re.DOTALL)
        
        # Remove line numbers and file paths from logs
        text = re.sub(r'\b\d{4}-\d{2}-\d{2}.*?- ', '', text)
        
        # Clean up whitespace
        lines = text.split('\n')
        
        # Remove empty lines at the beginning and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
            
        # Join lines back together
        cleaned_text = '\n'.join(lines)
        
        # Remove AI: prefix that might appear
        cleaned_text = re.sub(r'^AI:\s*', '', cleaned_text)
        
        return cleaned_text.strip()
    except Exception as e:
        if logger:
            logger.error(f"Error cleaning response output: {str(e)}")
        return text  # Return original text if cleaning fails


def should_visualize(query, csv_path, logger=None):
    """
    Determine if we should create a visualization based on the query and data.
    
    Args:
        query (str): The user query text
        csv_path (str): Path to the CSV file to analyze
        logger (logging.Logger, optional): Logger to use for error reporting
        
    Returns:
        bool: True if a visualization would be helpful, False otherwise
    """
    visualization_indicators = ['show', 'plot', 'graph', 'chart', 'visualize', 'trend', 
                            'compare', 'relationship', 'correlation', 'pattern']
    
    # Check if query explicitly asks for visualization
    if any(indicator in query.lower() for indicator in visualization_indicators):
        return True
    
    try:
        # Check if data is suitable for visualization
        df = pd.read_csv(csv_path)
        
        # Need at least one numeric column
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            return False
        
        # Check for time-series data (date columns or sequential values)
        has_date_col = any('date' in col.lower() or 'time' in col.lower() or 'year' in col.lower() 
                        for col in df.columns)
        
        # If data has multiple numeric columns or time dimension, it's good for visualization
        return has_date_col or len(numeric_cols) >= 2
        
    except Exception as e:
        if logger:
            logger.error(f"Error in should_visualize: {str(e)}")
        return False

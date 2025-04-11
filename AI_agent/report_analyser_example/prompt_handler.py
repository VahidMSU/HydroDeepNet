"""
Prompt handling utilities for natural language processing of user queries.
This module provides functions to extract semantic information from text input.
"""
import re
import logging
import pandas as pd
import os
from context_manager import context

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

    
def enhance_relevance(response, query):
    """Try to enhance the relevance of a response to the query."""
    # If response already seems relevant, leave it as is
    if calculate_query_relevance(response, query) >= 0.7:
        return response
        
    # Extract key information from the query
    query_terms = extract_keywords(query)
    
    # Append a note addressing the query more directly
    enhanced = response.strip()
    enhanced += "\n\nRegarding your specific question about " + ", ".join(query_terms[:3]) + ": "
    enhanced += f"I've provided the available information above. If you need more specific details about {' or '.join(query_terms[:2])}, please let me know."
    
    return enhanced


def validate_response(response, query, logger):
    """
    Validate and improve the agent's response.
    
    Args:
        response: The raw response from the agent
        query: The user's query that prompted this response
        
    Returns:
        The validated and potentially enhanced response
    """
    logger.debug(f"Validating response: length={len(response) if response else 0}")
    
    # Skip validation if response is None or empty
    if not response or not response.strip():
        logger.warning("Empty response received during validation")
        return "I apologize, but I couldn't generate a complete response."
        
    # If the response is very short, check if it's an error or apology
    if len(response.strip()) < 20:
        logger.warning(f"Very short response: '{response}'")
        if "error" in response.lower() or "apologize" in response.lower() or "sorry" in response.lower():
            return response
        else:
            # For very short non-error responses, we assume they're valid
            return response
        
    # Check for incomplete or truncated responses
    if not is_response_complete(response):
        logger.debug("Response appears incomplete, completing it")
        response = complete_response(response)
        
    # Check for repeated content
    if has_repeated_sections(response):
        logger.debug("Response has repeated sections, deduplicating")
        response = deduplicate_content(response)
        
    # Only add this context clarification for long responses that seem to be
    # generic or not directly addressing the query
    relevance_score = calculate_query_relevance(response, query)
    logger.debug(f"Response relevance score: {relevance_score:.2f}")
    
    if len(response) > 500 and relevance_score < 0.3:
        query_terms = ", ".join(extract_keywords(query)[:3])
        if query_terms:
            logger.debug(f"Adding clarification for low-relevance response about: {query_terms}")
            # Only add clarification text if the response seems generic and unrelated
            if not response.endswith("\n"):
                response += "\n\n"
            else:
                response += "\n"
            response += f"Regarding your specific question about {query_terms}: I've provided the available information above. If you need more specific details, please let me know."
    
    # Remove any empty list items or bullet points
    response = re.sub(r'\n\s*[-*]\s*\n', '\n\n', response)
    
    # Remove any remaining debug information
    cleaned_response = clean_response_output(response)
    
    return cleaned_response

def has_repeated_sections(text):
    """Check if a response has repeated sections."""
    if not text:
        return False
        
    # Split text into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) <= 1:
        return False
        
    # Check for duplicate paragraphs
    seen_paragraphs = set()
    for paragraph in paragraphs:
        if paragraph in seen_paragraphs:
            return True
        seen_paragraphs.add(paragraph)
        
    # Check for similar consecutive sections
    for i in range(len(paragraphs) - 1):
        similarity = calculate_similarity(paragraphs[i], paragraphs[i+1])
        if similarity > 0.7:  # 70% similarity threshold
            return True
            
    return False

def complete_response(text):
    """Try to make an incomplete response complete."""
    if not text:
        return "I apologize, but I couldn't generate a complete response."
        
    # If cut off with ellipsis, add a proper ending
    if text.strip().endswith(('...', '…')):
        return text.strip() + "\n\nI apologize, but the response was truncated. Please let me know if you'd like more information on this topic."
        
    # Balance unmatched parentheses, brackets, braces
    for open_char, close_char in [('(', ')'), ('[', ']'), ('{', '}')]:
        open_count = text.count(open_char)
        close_count = text.count(close_char)
        if open_count > close_count:
            # Add missing closing characters
            text = text + (close_char * (open_count - close_count))
            
    # If ends with a colon, add a concluding sentence
    if text.strip().endswith(':'):
        text = text + " I'll provide more details if you'd like additional information."
        
    return text
    
def calculate_query_relevance(response, query):
    """Calculate how relevant a response is to the original query."""
    if not response or not query:
        return 0.0
        
    # Extract key terms from the query
    query_terms = set(extract_keywords(query))
    if not query_terms:
        return 1.0  # No meaningful terms to match
        
    # Check how many query terms appear in the response
    response_lower = response.lower()
    matched_terms = sum(1 for term in query_terms if term.lower() in response_lower)
    
    # Calculate relevance score
    relevance_score = matched_terms / len(query_terms) if query_terms else 0.0
    
    return relevance_score


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




def handle_markdown_command(args, discovered_files, logger):
    """Handle /markdown command to view markdown files."""
    # If no args are provided, list all markdown files
    if not args.strip():
        md_files = context.discovered_files.get('md', [])
        if not md_files:
            return "No markdown files have been discovered yet. Use `/discover path` to find files."
        
        md_list = "## Available Markdown Files:\n\n"
        for i, path in enumerate(md_files, 1):
            md_list += f"{i}. {os.path.basename(path)}\n"
        
        md_list += "\nTo view a markdown file, use `/markdown [filename]` or `/md [filename]`"
        return md_list
    
    # If a filename is provided, find and read the file
    file_path = args.strip()
    
    # Remove extension if present to improve matching
    base_file_name = os.path.splitext(file_path)[0]
    
    # Find matching files if path is partial
    matching_files = []
    for path in context.discovered_files.get('md', []):
        path_basename = os.path.basename(path)
        path_basename_noext = os.path.splitext(path_basename)[0]
        
        # Match with or without extension
        if (base_file_name.lower() in path_basename_noext.lower() or
            file_path.lower() in path_basename.lower()):
            matching_files.append(path)
    
    if not matching_files:
        return f"No markdown files found matching '{file_path}'. Use `/files` to see available files."
    
    if len(matching_files) > 1:
        file_list = "\n".join([f"- {os.path.basename(path)}" for path in matching_files])
        return f"Multiple markdown files found matching '{file_path}'. Please be more specific:\n\n{file_list}"
    
    # We found exactly one matching file
    target_md = matching_files[0]
    
    # Update context
    context.set_current_topic('markdown_view', [target_md])
    
    try:
        # Check if we have a cached analysis
        cached_analysis = context.get_file_analysis(target_md)
        if cached_analysis:
            return cached_analysis
            
        # Read the markdown file
        with open(target_md, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Prepare response
        response = f"## Markdown File: {os.path.basename(target_md)}\n\n"
        response += content
        
        # Save to context
        context.save_analysis_result(target_md, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Error reading markdown file: {str(e)}")
        return f"Error reading markdown file '{os.path.basename(target_md)}': {str(e)}"

def handle_csv_command(args, discovered_files, logger, limit=5):
    """Handle /csv command to view and analyze CSV files."""
    from csv_utils import analyze_csv, perform_column_stats
    
    # If no args are provided, list all CSV files
    if not args.strip():
        csv_files = context.discovered_files.get('csv', [])
        if not csv_files:
            return "No CSV files have been discovered yet. Use `/discover path` to find files."
        
        csv_list = "## Available CSV Files:\n\n"
        for i, path in enumerate(csv_files, 1):
            csv_list += f"{i}. {os.path.basename(path)}\n"
        
        csv_list += "\nTo view a CSV file, use `/csv [filename]` or `/csv [filename] [column_name]` to analyze a specific column."
        return csv_list
    
    # Parse arguments - could be filename only or filename + column
    args_list = args.split()
    file_path = args_list[0]
    column_name = args_list[1] if len(args_list) > 1 else None
    
    # Remove extension if present to improve matching
    base_file_name = os.path.splitext(file_path)[0]
    
    # Find matching files if path is partial
    matching_files = []
    for path in context.discovered_files.get('csv', []):
        path_basename = os.path.basename(path)
        path_basename_noext = os.path.splitext(path_basename)[0]
        
        # Match with or without extension
        if (base_file_name.lower() in path_basename_noext.lower() or
            file_path.lower() in path_basename.lower()):
            matching_files.append(path)
    
    if not matching_files:
        return f"No CSV files found matching '{file_path}'. Use `/files` to see available files."
    
    if len(matching_files) > 1:
        file_list = "\n".join([f"- {os.path.basename(path)}" for path in matching_files])
        return f"Multiple CSV files found matching '{file_path}'. Please be more specific:\n\n{file_list}"
    
    # We found exactly one matching file
    target_csv = matching_files[0]
    
    # Update context
    context.set_current_topic('csv_analysis', [target_csv])
    
    try:
        # If a column name is provided, perform column analysis
        if column_name:
            # Check if we already have analysis results for this column
            cache_key = f"{target_csv}:{column_name}"
            cached_results = context.get_file_analysis(cache_key)
            if cached_results:
                return cached_results
            
            # Otherwise perform the analysis
            column_analysis = perform_column_stats(target_csv, column_name, logger)
            
            # Save the results to context
            context.save_analysis_result(cache_key, column_analysis)
            
            return column_analysis
        
        # Otherwise show general CSV info
        cached_results = context.get_file_analysis(target_csv)
        if cached_results:
            return cached_results
        
        # If not in cache, analyze the CSV
        csv_analysis = analyze_csv(target_csv, limit, logger)
        
        # Save the results
        context.save_analysis_result(target_csv, csv_analysis)
        
        return csv_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing CSV file: {str(e)}")
        return f"Error analyzing CSV file '{os.path.basename(target_csv)}': {str(e)}"

def handle_json_command(args, discovered_files, logger):
    """Handle /json command to view and analyze JSON files."""
    from json_queries import analyze_json_file
    
    # If no args are provided, list all JSON files
    if not args.strip():
        json_files = context.discovered_files.get('json', [])
        if not json_files:
            return "No JSON files have been discovered yet. Use `/discover path` to find files."
        
        json_list = "## Available JSON Files:\n\n"
        for i, path in enumerate(json_files, 1):
            json_list += f"{i}. {os.path.basename(path)}\n"
        
        json_list += "\nTo view a JSON file, use `/json [filename]`"
        return json_list
    
    # Parse arguments
    file_path = args.split()[0]
    
    # Remove extension if present to improve matching
    base_file_name = os.path.splitext(file_path)[0]
    
    # Find matching files if path is partial
    matching_files = []
    for path in context.discovered_files.get('json', []):
        path_basename = os.path.basename(path)
        path_basename_noext = os.path.splitext(path_basename)[0]
        
        # Match with or without extension
        if (base_file_name.lower() in path_basename_noext.lower() or
            file_path.lower() in path_basename.lower()):
            matching_files.append(path)
    
    if not matching_files:
        return f"No JSON files found matching '{file_path}'. Use `/files` to see available files."
    
    if len(matching_files) > 1:
        file_list = "\n".join([f"- {os.path.basename(path)}" for path in matching_files])
        return f"Multiple JSON files found matching '{file_path}'. Please be more specific:\n\n{file_list}"
    
    # We found exactly one matching file
    target_json = matching_files[0]
    
    # Update context
    context.set_current_topic('json_analysis', [target_json])
    
    try:
        # Check if we already have analysis results for this file
        cached_results = context.get_file_analysis(target_json)
        if cached_results:
            return cached_results
        
        # Otherwise analyze the JSON
        json_analysis = analyze_json_file(target_json, logger)
        
        # Save the results
        context.save_analysis_result(target_json, json_analysis)
        
        return json_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing JSON file: {str(e)}")
        return f"Error analyzing JSON file '{os.path.basename(target_json)}': {str(e)}"
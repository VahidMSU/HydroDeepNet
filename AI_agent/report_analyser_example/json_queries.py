import json
import os
from context_manager import context
from prompt_handler import extract_keywords



def analyze_json_for_query(query, json_path)->str:
    """Analyze a JSON file for a given query."""

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract query keywords
    keywords = extract_keywords(query)
    
    # Simple summary
    summary = f"The JSON file '{os.path.basename(json_path)}' "
    
    if isinstance(data, dict):
        summary += f"contains {len(data)} top-level keys.\n"
        
        # Find keys that match keywords
        matching_keys = []
        for key in data.keys():
            if any(kw in key.lower() for kw in keywords):
                matching_keys.append(key)
        
        if matching_keys:
            summary += "Here are the sections that seem relevant to your question:\n\n"
            for key in matching_keys[:3]:  # Limit to first 3 matches
                value = data[key]
                if isinstance(value, (dict, list)) and len(str(value)) > 500:
                    # Summarize large objects/arrays
                    if isinstance(value, dict):
                        summary += f"- {key}: A dictionary with {len(value)} keys\n"
                    else:
                        summary += f"- {key}: An array with {len(value)} items\n"
                else:
                    # Show the actual value for smaller data
                    summary += f"- {key}: {json.dumps(value)}\n"
        else:
            # No direct matches, show top-level structure
            summary += "Top-level keys: " + ", ".join(list(data.keys())[:10])
            if len(data) > 10:
                summary += f" and {len(data) - 10} more."
    elif isinstance(data, list):
        summary += f"contains a list with {len(data)} items.\n"
        if data and isinstance(data[0], dict):
            # Show structure of list items
            keys = data[0].keys()
            summary += f"Each item has these fields: {', '.join(keys)}\n"
            
            # Try to find items matching the query
            matching_items = []
            for item in data[:20]:  # Only check first 20 items
                if any(kw in str(item).lower() for kw in keywords):
                    matching_items.append(item)
            
            if matching_items:
                summary += f"\nFound {len(matching_items)} items relevant to your query. First match:\n"
                summary += json.dumps(matching_items[0], indent=2)
    
    # Save analysis result in context
    context.save_analysis_result(json_path, summary)
    
    return summary


def analyze_json_file(json_path, logger):
    """Analyze a JSON file and provide a comprehensive summary."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        file_name = os.path.basename(json_path)
        summary = f"## Analysis of {file_name}\n\n"
        
        # Determine JSON structure
        if isinstance(data, dict):
            summary += f"This JSON file contains an object with {len(data)} top-level keys.\n\n"
            
            # List the top-level keys
            summary += "### Top-level keys:\n"
            for key in data.keys():
                value = data[key]
                if isinstance(value, dict):
                    summary += f"- {key}: Object with {len(value)} keys\n"
                elif isinstance(value, list):
                    summary += f"- {key}: Array with {len(value)} items\n"
                else:
                    # For simple values, show them directly
                    summary += f"- {key}: {str(value)[:50]}" + ("..." if len(str(value)) > 50 else "") + "\n"
            
            # Provide a sample of nested data if available
            complex_keys = [k for k, v in data.items() if isinstance(v, (dict, list))]
            if complex_keys:
                sample_key = complex_keys[0]
                summary += f"\n### Sample from '{sample_key}':\n"
                sample = data[sample_key]
                if isinstance(sample, dict):
                    summary += json.dumps(dict(list(sample.items())[:3]), indent=2)
                    if len(sample) > 3:
                        summary += f"\n... and {len(sample) - 3} more keys"
                elif isinstance(sample, list) and sample:
                    summary += json.dumps(sample[0], indent=2)
                    if len(sample) > 1:
                        summary += f"\n... and {len(sample) - 1} more items"
                        
        elif isinstance(data, list):
            summary += f"This JSON file contains an array with {len(data)} items.\n\n"
            
            # Show structure of the first item if it's a dictionary
            if data and isinstance(data[0], dict):
                keys = data[0].keys()
                summary += f"### Each item has these fields:\n"
                for key in keys:
                    summary += f"- {key}\n"
                
                # Show a sample item
                summary += "\n### Sample item:\n"
                summary += json.dumps(data[0], indent=2)
                
            # Show statistics if all items are numeric
            elif data and all(isinstance(item, (int, float)) for item in data):
                import statistics
                summary += "### Numeric array statistics:\n"
                summary += f"- Min: {min(data)}\n"
                summary += f"- Max: {max(data)}\n"
                summary += f"- Mean: {statistics.mean(data):.2f}\n"
                summary += f"- Median: {statistics.median(data):.2f}\n"
                
            # Just show a few examples otherwise
            else:
                summary += "### First few items:\n"
                for i, item in enumerate(data[:5]):
                    summary += f"{i+1}. {str(item)[:50]}" + ("..." if len(str(item)) > 50 else "") + "\n"
                if len(data) > 5:
                    summary += f"... and {len(data) - 5} more items"
        
        # Save analysis in context
        context.save_analysis_result(json_path, summary)
        
        return summary
    except Exception as e:
        logger.error(f"Error analyzing JSON file: {str(e)}")
        return f"Error analyzing JSON file '{os.path.basename(json_path)}': {str(e)}"
import os
import pandas as pd
from context_manager import context


def quick_csv_analysis(csv_path):
    """Quick analysis of CSV data focused on answering the user's query."""

    # Check file size first
    row_count = 0
    with open(csv_path, 'r') as f:
        for i, _ in enumerate(f):
            row_count = i + 1
            if row_count > 1000:
                return f"CSV file {os.path.basename(csv_path)} has {row_count} rows (showing summary only due to size)."
    
    # Read the CSV data
    df = pd.read_csv(csv_path)
    
    # Basic statistics
    summary = f"The file contains {len(df)} rows and {len(df.columns)} columns.\n"
    summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
    
    # Find numeric columns for statistical analysis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        summary += "Key statistics:\n"
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            summary += f"- {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, avg={df[col].mean():.2f}\n"
    
    # Check for interesting patterns
    if len(numeric_cols) >= 2:
        # Find highest correlation
        corr_matrix = df[numeric_cols].corr()
        highest_corr = 0
        pair = (None, None)
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if abs(corr_matrix.loc[col1, col2]) > abs(highest_corr):
                    highest_corr = corr_matrix.loc[col1, col2]
                    pair = (col1, col2)
        
        if pair[0] and abs(highest_corr) > 0.5:
            corr_type = "positive" if highest_corr > 0 else "negative"
            summary += f"\nI noticed a strong {corr_type} correlation ({highest_corr:.2f}) between {pair[0]} and {pair[1]}."
    
    return summary
    


def handle_csv_command(args, discovered_files, logger):
    """Handle /csv command to view and analyze CSV files."""
    
    # If no args are provided, list all CSV files
    if not args.strip():
        csv_files = discovered_files.get('csv', [])
        if not csv_files:
            return "No CSV files have been discovered yet. Use /discover path to find files."
        
        csv_list = "## Available CSV Files:\n\n"
        for i, path in enumerate(csv_files, 1):
            csv_list += f"{i}. {os.path.basename(path)}\n"
        
        csv_list += "\nTo analyze a CSV file, use `/csv [filename]` or `/analyze [filename]`"
        return csv_list
    
    # If a filename is provided, find and analyze the file
    file_path = args.strip()
    
    # Remove extension if present to improve matching
    base_file_name = os.path.splitext(file_path)[0]
    
    # Find matching files if path is partial
    matching_files = []
    for path in discovered_files.get('csv', []):
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
        # Read the CSV data
        df = pd.read_csv(target_csv)
        
        # Prepare response
        filename = os.path.basename(target_csv)
        response = f"## Analysis of {filename}\n\n"
        response += f"This CSV file contains {len(df)} rows and {len(df.columns)} columns.\n\n"
        
        # List all columns
        response += "### Columns:\n"
        for col in df.columns:
            response += f"- {col}\n"
        
        # Show a preview
        response += "\n### Preview (first 5 rows):\n"
        response += df.head(5).to_string()
        
        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            response += "\n\n### Basic Statistics:\n"
            # Calculate statistics for numeric columns
            stats_df = df[numeric_cols].describe().round(2)
            response += stats_df.to_string()
            
            # Suggest visualization if there are numeric columns
            response += "\n\nYou can create visualizations with `/visualize " + filename + " [x_column] [y_column]`"
        
        # Save the analysis result in context
        context.save_analysis_result(target_csv, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing CSV file: {str(e)}")
        return f"Error analyzing CSV file '{os.path.basename(target_csv)}': {str(e)}"

"""
Data visualization utilities for creating charts and plots from various data sources.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
from prompt_handler import extract_keywords, should_visualize

class DataVisualizer:
    """
    A class for creating and managing visualizations from data sources.
    
    This class provides methods for:
    - Creating contextual visualizations based on user queries
    - Determining when visualization would be helpful
    - Supporting various chart types (line, bar, scatter, histogram)
    """
    
    def __init__(self, logger=None):
        """
        Initialize the DataVisualizer.
        
        Args:
            logger: Optional logger instance for logging
        """
        self.logger = logger
    
    def visualize_data(self, file_path, x_column=None, y_column=None, chart_type='auto'):
        """
        Generate a visualization for numerical data in CSV files.
        
        Args:
            file_path: Path to the CSV file
            x_column: Column to use for x-axis
            y_column: Column to use for y-axis
            chart_type: Type of chart ('auto', 'line', 'bar', 'scatter')
            
        Returns:
            Path to generated visualization or error message
        """
        try:
            if not file_path.endswith('.csv'):
                return f"Visualization is currently only supported for CSV files. The file {file_path} is not a CSV."
            
            # Check if file is too large
            row_count = 0
            with open(file_path, 'r') as f:
                for i, _ in enumerate(f):
                    row_count = i + 1
                    if row_count > 1000:
                        return f"Warning: CSV file {os.path.basename(file_path)} has more than 1000 rows ({row_count}). Visualization skipped to prevent memory issues. Consider visualizing a subset of the data instead."
            
            df = pd.read_csv(file_path)
            
            # If columns aren't specified, try to guess appropriate ones
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if not numeric_cols:
                return "No numeric columns found for visualization."
            
            if not x_column:
                # Try to find a good x-axis - prefer date-like or first column
                if any(col.lower().find('date') >= 0 for col in df.columns):
                    date_cols = [col for col in df.columns if col.lower().find('date') >= 0]
                    x_column = date_cols[0]
                else:
                    x_column = df.columns[0]
            
            if not y_column:
                # Use first numeric column that's not the x column
                for col in numeric_cols:
                    if col != x_column:
                        y_column = col
                        break
                else:
                    y_column = numeric_cols[0]
            
            # Determine chart type if auto
            if chart_type == 'auto':
                if len(df) > 50:
                    chart_type = 'line'
                else:
                    chart_type = 'bar'
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            
            if chart_type == 'line':
                plt.plot(df[x_column], df[y_column])
            elif chart_type == 'bar':
                plt.bar(df[x_column], df[y_column])
            elif chart_type == 'scatter':
                plt.scatter(df[x_column], df[y_column])
            
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f"{y_column} vs {x_column} from {os.path.basename(file_path)}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the plot to a file
            output_dir = os.path.join(os.path.dirname(file_path), "visualizations")
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_{x_column}_{y_column}_{chart_type}.png"
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path)
            plt.close()
            
            return output_path
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"visualize_data Error visualizing data: {str(e)}")
            return f"Error creating visualization: {str(e)}"
    
    def should_visualize(self, query, csv_path):
        """
        Determine if we should create a visualization based on the query and data.
        
        Args:
            query: User query text
            csv_path: Path to the CSV file
            
        Returns:
            bool: True if visualization would be helpful
        """
        return should_visualize(query, csv_path, self.logger)
    
    def create_contextual_visualization(self, query, csv_path):
        """
        Create a visualization customized to answer the specific query.
        
        Args:
            query: User query text
            csv_path: Path to the CSV file
            
        Returns:
            Path to the generated visualization or None
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Determine which columns to visualize based on the query
            keywords = extract_keywords(query)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if not numeric_cols:
                return None
            
            # Try to find columns mentioned in the query
            mentioned_cols = []
            for col in df.columns:
                if any(kw in col.lower() for kw in keywords):
                    mentioned_cols.append(col)
            
            # Determine x and y columns
            x_col = None
            y_col = None
            
            # Look for date/time columns first for x-axis
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower():
                    x_col = col
                    break
            
            # If no date column, use the first column
            if not x_col:
                x_col = df.columns[0]
            
            # For y-axis, prioritize columns mentioned in the query
            for col in mentioned_cols:
                if col in numeric_cols:
                    y_col = col
                    break
            
            # If no mentioned column is numeric, use the first numeric column
            if not y_col and numeric_cols:
                y_col = numeric_cols[0]
            
            # Determine appropriate chart type
            chart_type = 'line'  # Default
            if 'distribution' in query.lower() or 'histogram' in query.lower():
                chart_type = 'histogram'
            elif 'scatter' in query.lower() or 'correlation' in query.lower() or 'relationship' in query.lower():
                chart_type = 'scatter'
            elif len(df) < 30:  # Small datasets look better as bar charts
                chart_type = 'bar'
            
            # Create the visualization
            plt.figure(figsize=(10, 6))
            
            if chart_type == 'line':
                plt.plot(df[x_col], df[y_col])
                plt.title(f"{y_col} over {x_col}")
            elif chart_type == 'bar':
                plt.bar(df[x_col], df[y_col])
                plt.title(f"{y_col} by {x_col}")
            elif chart_type == 'scatter':
                plt.scatter(df[x_col], df[y_col])
                plt.title(f"Relationship between {x_col} and {y_col}")
            elif chart_type == 'histogram':
                plt.hist(df[y_col], bins=15)
                plt.title(f"Distribution of {y_col}")
            
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the plot
            output_dir = os.path.join(os.path.dirname(csv_path), "visualizations")
            os.makedirs(output_dir, exist_ok=True)
            
            # Use query keywords in the filename
            query_slug = "_".join(keywords[:2]).replace(' ', '_')
            filename = f"auto_viz_{query_slug}_{y_col}_{chart_type}.png"
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path)
            plt.close()
            
            return output_path
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"create_contextual_visualization Error creating visualization: {str(e)}")
            return None
            
    def analyze_csv_data(self, file_path):
        """
        Generate a basic statistical analysis for CSV files.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            dict: Analysis results or error message
        """
        try:
            if not file_path.endswith('.csv'):
                return f"Statistical analysis is currently only supported for CSV files. The file {file_path} is not a CSV."
            
            # Check if file is too large
            row_count = 0
            with open(file_path, 'r') as f:
                for i, _ in enumerate(f):
                    row_count = i + 1
                    if row_count > 1000:
                        return f"Warning: CSV file {os.path.basename(file_path)} has more than 1000 rows ({row_count}). Analysis skipped to prevent memory issues. Consider analyzing a subset of the data instead."
            
            df = pd.read_csv(file_path)
            
            # Get numeric columns for analysis
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if not numeric_cols:
                return "No numeric columns found for analysis."
            
            # Generate basic statistics
            stats = df[numeric_cols].describe().to_dict()
            
            # Calculate correlations if there are multiple numeric columns
            correlations = None
            if len(numeric_cols) > 1:
                correlations = df[numeric_cols].corr().to_dict()
            
            # Prepare the analysis results
            analysis = {
                'file_name': os.path.basename(file_path),
                'path': file_path,
                'rows': len(df),
                'columns': len(df.columns),
                'statistics': stats,
                'correlations': correlations,
                'missing_values_count': df.isnull().sum().to_dict()
            }
            
            return analysis
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"analyze_csv_data Error analyzing data: {str(e)}")
            return f"Error analyzing data: {str(e)}" 
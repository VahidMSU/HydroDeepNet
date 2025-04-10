import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import traceback
import json
import re

# Configure matplotlib to use a non-interactive backend
matplotlib.use('Agg')

from Logger import LoggerSetup
from document_reader_image import ImageAnalyzer

# Initialize logger using setup_logger method
logger_setup = LoggerSetup()
logger = logger_setup.setup_logger()

class VisualizationManager:
    """Handles visualization creation and management for the document reader."""
    
    def __init__(self, document_reader=None):
        """Initialize the visualization handler with reference to the document reader."""
        self.document_reader = document_reader
        self.output_dir = None
        self.visualization_cache = {}
        self.style = 'default'  # Default visualization style
        
        # Set default visualization parameters
        self.default_params = {
            'figsize': (10, 6),
            'dpi': 100,
            'cmap': 'viridis',
            'title_fontsize': 14,
            'axis_fontsize': 12,
            'grid': True,
            'legend': True
        }
        
        # Set default Seaborn style
        sns.set_style("whitegrid")
        
    def set_output_directory(self, directory: str) -> bool:
        """Set the output directory for saving visualizations."""
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            self.output_dir = directory
            logger.info(f"Set visualization output directory to {directory}")
            return True
        except Exception as e:
            logger.error(f"Error setting output directory: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def visualize_data(self, 
                       file_path: str, 
                       x_column: Optional[str] = None, 
                       y_column: Optional[str] = None, 
                       chart_type: str = 'auto',
                       **kwargs) -> Dict[str, Any]:
        """
        Create a visualization from tabular data (CSV).
        
        Args:
            file_path: Path to the CSV file
            x_column: Column to use for x-axis (optional)
            y_column: Column to use for y-axis (optional)
            chart_type: Type of chart to create (auto, line, bar, scatter, hist, box, heatmap, etc.)
            **kwargs: Additional parameters for the visualization
            
        Returns:
            Dictionary with visualization metadata including the file path
        """
        try:
            # Determine output directory
            if not self.output_dir:
                # Use the same directory as the input file by default
                self.output_dir = os.path.dirname(file_path)
            
            # Generate a unique filename for the visualization
            filename = f"viz_{os.path.basename(file_path).replace('.csv', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            output_path = os.path.join(self.output_dir, filename)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Get column names
            columns = df.columns.tolist()
            
            # If x_column and y_column not specified, try to determine them
            if not x_column and not y_column:
                x_column, y_column = self._infer_columns(df)
            elif not y_column and x_column:
                # If only x_column is specified, find numeric columns for y_column
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols and x_column in numeric_cols:
                    # If x_column is numeric, use the first different numeric column
                    other_numeric = [col for col in numeric_cols if col != x_column]
                    if other_numeric:
                        y_column = other_numeric[0]
                elif numeric_cols:
                    # Otherwise use the first numeric column
                    y_column = numeric_cols[0]
            
            # Determine the best chart type if set to auto
            if chart_type == 'auto':
                chart_type = self._infer_chart_type(df, x_column, y_column)
            
            # Create the visualization
            plt.figure(figsize=kwargs.get('figsize', self.default_params['figsize']))
            
            # Generate the appropriate chart
            if chart_type == 'line':
                self._create_line_chart(df, x_column, y_column, **kwargs)
            elif chart_type == 'bar':
                self._create_bar_chart(df, x_column, y_column, **kwargs)
            elif chart_type == 'scatter':
                self._create_scatter_chart(df, x_column, y_column, **kwargs)
            elif chart_type == 'hist':
                self._create_histogram(df, x_column, **kwargs)
            elif chart_type == 'box':
                self._create_box_plot(df, x_column, y_column, **kwargs)
            elif chart_type == 'heatmap':
                self._create_heatmap(df, **kwargs)
            elif chart_type == 'pie':
                self._create_pie_chart(df, x_column, y_column, **kwargs)
            else:
                # Default to line chart if specified type is not recognized
                chart_type = 'line'
                self._create_line_chart(df, x_column, y_column, **kwargs)
            
            # Add a grid if specified
            if kwargs.get('grid', self.default_params['grid']):
                plt.grid(True, alpha=0.3)
            
            # Add a title if specified
            title = kwargs.get('title', f"{chart_type.capitalize()} Chart: {y_column} vs {x_column}")
            plt.title(title, fontsize=kwargs.get('title_fontsize', self.default_params['title_fontsize']))
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(output_path, dpi=kwargs.get('dpi', self.default_params['dpi']))
            plt.close()
            
            # Create metadata about the visualization
            metadata = {
                'file_path': output_path,
                'source_data': file_path,
                'chart_type': chart_type,
                'x_column': x_column,
                'y_column': y_column,
                'created_at': datetime.now().isoformat(),
                'parameters': {**self.default_params, **kwargs}
            }
            
            # Cache the visualization metadata
            self.visualization_cache[output_path] = metadata
            
            logger.info(f"Created {chart_type} visualization: {output_path}")
            return metadata
        
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'source_data': file_path,
                'chart_type': chart_type
            }
    
    def _infer_columns(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Infer the best columns to use for x and y axes."""
        columns = df.columns.tolist()
        
        # Check for date/time columns for x-axis
        date_columns = [col for col in columns if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower()]
        
        # Check for numeric columns for y-axis
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        # If we have date columns and numeric columns, use the first of each
        if date_columns and numeric_columns:
            return date_columns[0], numeric_columns[0]
        
        # If we have multiple numeric columns, use the first two
        if len(numeric_columns) >= 2:
            return numeric_columns[0], numeric_columns[1]
        
        # Otherwise use the first two columns
        if len(columns) >= 2:
            return columns[0], columns[1]
        
        # Fallback to using the same column for both if only one exists
        return columns[0], columns[0]
    
    def _infer_chart_type(self, df: pd.DataFrame, x_column: str, y_column: str) -> str:
        """Infer the best chart type based on the data."""
        # Check if x column contains date/time values
        is_date_x = 'date' in x_column.lower() or 'time' in x_column.lower() or 'year' in x_column.lower()
        
        # Check number of unique values in x
        try:
            x_unique = df[x_column].nunique()
        except:
            x_unique = 0
        
        # Check if columns are numeric
        try:
            x_numeric = pd.api.types.is_numeric_dtype(df[x_column])
            y_numeric = pd.api.types.is_numeric_dtype(df[y_column])
        except:
            x_numeric = False
            y_numeric = False
        
        # Logic for determining chart type
        if is_date_x and y_numeric:
            # Time series data is best with line charts
            return 'line'
        elif x_unique <= 20 and y_numeric:
            # Categorical data with few categories is good for bar charts
            return 'bar'
        elif x_numeric and y_numeric:
            # Two numeric columns can use scatter or line
            # Check for correlation to decide
            try:
                correlation = df[x_column].corr(df[y_column])
                if abs(correlation) > 0.7:
                    # Strong correlation suggests a trend line
                    return 'line'
                else:
                    # Weak correlation is better shown as scatter
                    return 'scatter'
            except:
                # Default to scatter if correlation can't be calculated
                return 'scatter'
        elif x_unique <= 10 and y_column == x_column:
            # Single column with few categories can use a pie chart
            return 'pie'
        else:
            # Default to a line chart
            return 'line'
    
    def _create_line_chart(self, df: pd.DataFrame, x_column: str, y_column: str, **kwargs):
        """Create a line chart."""
        plt.plot(df[x_column], df[y_column], marker=kwargs.get('marker', 'o'), 
                 linestyle=kwargs.get('linestyle', '-'), 
                 color=kwargs.get('color', 'blue'),
                 linewidth=kwargs.get('linewidth', 2),
                 alpha=kwargs.get('alpha', 0.8))
        
        plt.xlabel(kwargs.get('xlabel', x_column), fontsize=kwargs.get('axis_fontsize', self.default_params['axis_fontsize']))
        plt.ylabel(kwargs.get('ylabel', y_column), fontsize=kwargs.get('axis_fontsize', self.default_params['axis_fontsize']))
        
        # Rotate x-axis labels if there are many or they're long
        if df[x_column].nunique() > 10 or df[x_column].astype(str).str.len().max() > 10:
            plt.xticks(rotation=45, ha='right')
    
    def _create_bar_chart(self, df: pd.DataFrame, x_column: str, y_column: str, **kwargs):
        """Create a bar chart."""
        plt.bar(df[x_column], df[y_column], 
                color=kwargs.get('color', 'blue'),
                alpha=kwargs.get('alpha', 0.8),
                width=kwargs.get('width', 0.8))
        
        plt.xlabel(kwargs.get('xlabel', x_column), fontsize=kwargs.get('axis_fontsize', self.default_params['axis_fontsize']))
        plt.ylabel(kwargs.get('ylabel', y_column), fontsize=kwargs.get('axis_fontsize', self.default_params['axis_fontsize']))
        
        # Rotate x-axis labels if there are many or they're long
        if df[x_column].nunique() > 7 or df[x_column].astype(str).str.len().max() > 7:
            plt.xticks(rotation=45, ha='right')
    
    def _create_scatter_chart(self, df: pd.DataFrame, x_column: str, y_column: str, **kwargs):
        """Create a scatter plot."""
        plt.scatter(df[x_column], df[y_column],
                   color=kwargs.get('color', 'blue'),
                   alpha=kwargs.get('alpha', 0.6),
                   s=kwargs.get('marker_size', 50))
        
        plt.xlabel(kwargs.get('xlabel', x_column), fontsize=kwargs.get('axis_fontsize', self.default_params['axis_fontsize']))
        plt.ylabel(kwargs.get('ylabel', y_column), fontsize=kwargs.get('axis_fontsize', self.default_params['axis_fontsize']))
        
        # Add a trend line if specified
        if kwargs.get('trendline', False):
            z = np.polyfit(df[x_column], df[y_column], 1)
            p = np.poly1d(z)
            plt.plot(df[x_column], p(df[x_column]), 
                     linestyle='--', 
                     color=kwargs.get('trendline_color', 'red'),
                     alpha=0.8)
    
    def _create_histogram(self, df: pd.DataFrame, column: str, **kwargs):
        """Create a histogram."""
        plt.hist(df[column], 
                bins=kwargs.get('bins', 10),
                color=kwargs.get('color', 'blue'),
                alpha=kwargs.get('alpha', 0.7),
                edgecolor='black')
        
        plt.xlabel(kwargs.get('xlabel', column), fontsize=kwargs.get('axis_fontsize', self.default_params['axis_fontsize']))
        plt.ylabel(kwargs.get('ylabel', 'Frequency'), fontsize=kwargs.get('axis_fontsize', self.default_params['axis_fontsize']))
    
    def _create_box_plot(self, df: pd.DataFrame, x_column: str, y_column: str, **kwargs):
        """Create a box plot."""
        if x_column == y_column:
            # Single column box plot
            plt.boxplot(df[x_column], vert=kwargs.get('vertical', True))
            plt.xlabel(kwargs.get('xlabel', ''), fontsize=kwargs.get('axis_fontsize', self.default_params['axis_fontsize']))
            plt.ylabel(kwargs.get('ylabel', x_column), fontsize=kwargs.get('axis_fontsize', self.default_params['axis_fontsize']))
        else:
            # Grouped box plot
            sns.boxplot(x=x_column, y=y_column, data=df)
            plt.xlabel(kwargs.get('xlabel', x_column), fontsize=kwargs.get('axis_fontsize', self.default_params['axis_fontsize']))
            plt.ylabel(kwargs.get('ylabel', y_column), fontsize=kwargs.get('axis_fontsize', self.default_params['axis_fontsize']))
    
    def _create_heatmap(self, df: pd.DataFrame, **kwargs):
        """Create a correlation heatmap."""
        # Get correlation matrix for numeric columns
        corr_matrix = df.select_dtypes(include=['number']).corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                   annot=kwargs.get('annotate', True),
                   cmap=kwargs.get('cmap', self.default_params['cmap']),
                   linewidths=0.5,
                   fmt='.2f')
        
        plt.title(kwargs.get('title', 'Correlation Heatmap'), 
                 fontsize=kwargs.get('title_fontsize', self.default_params['title_fontsize']))
        
        # Adjust subplot parameters
        plt.tight_layout()
    
    def _create_pie_chart(self, df: pd.DataFrame, x_column: str, y_column: str = None, **kwargs):
        """Create a pie chart."""
        if y_column and y_column != x_column:
            # Use x_column for labels and y_column for values
            data = df.groupby(x_column)[y_column].sum()
            labels = data.index
            values = data.values
        else:
            # Use value counts of x_column
            data = df[x_column].value_counts()
            labels = data.index
            values = data.values
        
        # Create pie chart
        plt.pie(values, 
               labels=labels if len(labels) <= 7 else None,  # Only show labels if there aren't too many
               autopct='%1.1f%%',
               shadow=kwargs.get('shadow', False),
               startangle=kwargs.get('startangle', 90),
               explode=kwargs.get('explode', None))
        
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Add legend if there are many categories or if specified
        if len(labels) > 7 or kwargs.get('legend', True):
            plt.legend(labels, title=x_column, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    def create_multi_chart(self, 
                         file_path: str,
                         chart_configs: List[Dict[str, Any]],
                         title: str = "Multi-Chart Analysis",
                         **kwargs) -> Dict[str, Any]:
        """
        Create multiple charts in a single figure.
        
        Args:
            file_path: Path to the CSV file
            chart_configs: List of dictionaries with chart configurations
            title: Overall figure title
            **kwargs: Additional parameters for the figure
            
        Returns:
            Dictionary with visualization metadata
        """
        try:
            # Determine output directory
            if not self.output_dir:
                # Use the same directory as the input file by default
                self.output_dir = os.path.dirname(file_path)
            
            # Generate a unique filename for the visualization
            filename = f"multichart_{os.path.basename(file_path).replace('.csv', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            output_path = os.path.join(self.output_dir, filename)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Calculate grid dimensions
            n_charts = len(chart_configs)
            if n_charts <= 2:
                n_rows, n_cols = 1, n_charts
            else:
                n_cols = min(3, n_charts)
                n_rows = (n_charts + n_cols - 1) // n_cols  # Ceiling division
            
            # Create figure and subplots
            fig_width = kwargs.get('fig_width', 5 * n_cols)
            fig_height = kwargs.get('fig_height', 4 * n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
            
            # Make axes iterable even for a single subplot
            if n_charts == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            # Create each chart in its subplot
            for i, config in enumerate(chart_configs):
                row = i // n_cols
                col = i % n_cols
                
                ax = axes[row, col] if n_rows > 1 else axes[col]
                plt.sca(ax)
                
                chart_type = config.get('chart_type', 'line')
                x_column = config.get('x_column')
                y_column = config.get('y_column')
                
                if x_column is None or y_column is None:
                    x_column, y_column = self._infer_columns(df)
                    config['x_column'] = x_column
                    config['y_column'] = y_column
                
                # Create the specific chart type
                if chart_type == 'line':
                    self._create_line_chart(df, x_column, y_column, **config)
                elif chart_type == 'bar':
                    self._create_bar_chart(df, x_column, y_column, **config)
                elif chart_type == 'scatter':
                    self._create_scatter_chart(df, x_column, y_column, **config)
                elif chart_type == 'hist':
                    self._create_histogram(df, x_column, **config)
                elif chart_type == 'box':
                    self._create_box_plot(df, x_column, y_column, **config)
                
                # Set subplot title
                subplot_title = config.get('title', f"{chart_type.capitalize()}: {y_column} vs {x_column}")
                ax.set_title(subplot_title)
            
            # Hide any unused subplots
            for i in range(n_charts, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.axis('off')
            
            # Set overall title
            fig.suptitle(title, fontsize=kwargs.get('title_fontsize', 16))
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
            plt.savefig(output_path, dpi=kwargs.get('dpi', self.default_params['dpi']))
            plt.close()
            
            # Create metadata
            metadata = {
                'file_path': output_path,
                'source_data': file_path,
                'chart_configs': chart_configs,
                'created_at': datetime.now().isoformat()
            }
            
            # Cache the visualization metadata
            self.visualization_cache[output_path] = metadata
            
            logger.info(f"Created multi-chart visualization: {output_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error creating multi-chart visualization: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'source_data': file_path
            }
    
    def create_visualization_from_query(self, 
                                      query: str, 
                                      file_path: str) -> Dict[str, Any]:
        """
        Create a visualization based on a natural language query.
        
        Args:
            query: Natural language query describing the desired visualization
            file_path: Path to the CSV file
            
        Returns:
            Dictionary with visualization metadata
        """
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Get column names
            columns = df.columns.tolist()
            
            # Look for column names in the query
            mentioned_columns = [col for col in columns if col.lower() in query.lower()]
            
            # Determine x and y columns
            x_column, y_column = None, None
            
            # Look for visualization type hints in the query
            viz_types = {
                'line': ['line', 'trend', 'time', 'series', 'change', 'over time'],
                'bar': ['bar', 'comparison', 'compare', 'histogram', 'distribution'],
                'scatter': ['scatter', 'relationship', 'correlation', 'between'],
                'pie': ['pie', 'portion', 'percentage', 'share', 'breakdown'],
                'box': ['box', 'range', 'dispersion', 'distribution', 'quartile'],
                'heatmap': ['heat', 'matrix', 'correlation', 'between all']
            }
            
            chart_type = 'auto'
            for vtype, keywords in viz_types.items():
                if any(kw in query.lower() for kw in keywords):
                    chart_type = vtype
                    break
            
            # Extract time/date related columns for x-axis if this appears to be a time series
            if 'time' in query.lower() or 'trend' in query.lower() or 'over' in query.lower():
                date_cols = [col for col in columns if 'date' in col.lower() or 'time' in col.lower() 
                           or 'year' in col.lower() or 'month' in col.lower()]
                if date_cols:
                    x_column = date_cols[0]
            
            # Extract specific column reference patterns
            x_patterns = [
                r'x[- ]axis\s*[is:]*\s*(\w+)', 
                r'plot\s+(\w+)\s+on\s+x', 
                r'(\w+)\s+versus',
                r'(\w+)\s+vs\.?'
            ]
            
            y_patterns = [
                r'y[- ]axis\s*[is:]*\s*(\w+)', 
                r'plot\s+(\w+)\s+on\s+y', 
                r'versus\s+(\w+)',
                r'vs\.?\s+(\w+)'
            ]
            
            # Check for column extraction from patterns
            for pattern in x_patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    potential_x = matches[0]
                    matching_col = next((col for col in columns if potential_x.lower() in col.lower()), None)
                    if matching_col:
                        x_column = matching_col
                        break
            
            for pattern in y_patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    potential_y = matches[0]
                    matching_col = next((col for col in columns if potential_y.lower() in col.lower()), None)
                    if matching_col:
                        y_column = matching_col
                        break
            
            # If we have mentioned columns but couldn't extract x or y specifically
            if mentioned_columns and (not x_column or not y_column):
                if len(mentioned_columns) >= 2:
                    # Use the first two mentioned columns
                    x_column = x_column or mentioned_columns[0]
                    y_column = y_column or mentioned_columns[1]
                elif len(mentioned_columns) == 1:
                    # Use the mentioned column as y and find a suitable x
                    y_column = y_column or mentioned_columns[0]
                    
                    # Look for a suitable x_column
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    date_cols = [col for col in columns if 'date' in col.lower() or 'time' in col.lower() 
                               or 'year' in col.lower() or 'month' in col.lower()]
                    
                    if date_cols:
                        x_column = date_cols[0]
                    elif numeric_cols and y_column in numeric_cols:
                        # Find another numeric column different from y_column
                        other_numeric = [col for col in numeric_cols if col != y_column]
                        if other_numeric:
                            x_column = other_numeric[0]
            
            # If we still don't have columns, infer them
            if not x_column or not y_column:
                x_column, y_column = self._infer_columns(df)
            
            # Extract additional parameters from the query
            title = None
            title_patterns = [
                r'title[d:]?\s+["\'](.*?)["\']',
                r'call(?:ed|ing) it\s+["\'](.*?)["\']',
                r'name(?:d|ing) it\s+["\'](.*?)["\']'
            ]
            
            for pattern in title_patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    title = matches[0]
                    break
            
            if not title:
                # Generate a title based on the query and columns
                if 'time' in query.lower() or 'trend' in query.lower():
                    title = f"{y_column} Over Time"
                elif 'comparison' in query.lower() or 'compare' in query.lower():
                    title = f"Comparison of {y_column} by {x_column}"
                elif 'relationship' in query.lower() or 'correlation' in query.lower():
                    title = f"Relationship Between {x_column} and {y_column}"
                else:
                    title = f"{y_column} vs {x_column}"
            
            # Create the visualization with the detected parameters
            return self.visualize_data(
                file_path=file_path,
                x_column=x_column,
                y_column=y_column,
                chart_type=chart_type,
                title=title
            )
            
        except Exception as e:
            logger.error(f"Error creating visualization from query: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'source_data': file_path,
                'query': query
            }
    
    def get_visualization_data(self, file_path: str) -> Dict[str, Any]:
        """Get metadata for a visualization by file path."""
        if file_path in self.visualization_cache:
            return self.visualization_cache[file_path]
        
        # If not in cache but file exists, create basic metadata
        if os.path.exists(file_path) and file_path.lower().endswith(('.png', '.jpg')):
            metadata = {
                'file_path': file_path,
                'created_at': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                'size': os.path.getsize(file_path)
            }
            return metadata
        
        return {'error': 'Visualization not found or not in cache'}
    
    def list_visualizations(self) -> List[Dict[str, Any]]:
        """List all created visualizations with metadata."""
        visualizations = []
        
        # First add all cached visualizations
        for file_path, metadata in self.visualization_cache.items():
            if os.path.exists(file_path):
                visualizations.append(metadata)
        
        # Then look for any visualization files in the output directory
        if self.output_dir and os.path.exists(self.output_dir):
            for filename in os.listdir(self.output_dir):
                if filename.lower().endswith(('.png', '.jpg')):
                    file_path = os.path.join(self.output_dir, filename)
                    
                    # Skip files already in cache
                    if file_path in self.visualization_cache:
                        continue
                    
                    # Create basic metadata
                    metadata = {
                        'file_path': file_path,
                        'filename': filename,
                        'created_at': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                        'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                        'size': os.path.getsize(file_path)
                    }
                    visualizations.append(metadata)
        
        return visualizations 
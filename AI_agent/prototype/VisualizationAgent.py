from agno.vectordb.pgvector import PgVector
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.media import Image
from agno.models.openai import OpenAIChat
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import io
import base64
import uuid

from Logger import Logger

class VisualizationAgent:
    """Agent for creating various data visualizations and charts."""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 model_name: str = "gpt-4",
                 output_dir: str = "visualizations",
                 vector_db_url: str = "postgresql+psycopg://ai:ai@localhost:5432/ai"):
        """
        Initialize the visualization agent.
        
        Args:
            log_dir: Directory for log files
            model_name: Name of the model to use
            output_dir: Directory for saving visualizations
            vector_db_url: URL for vector database connection
        """
        # Set up logging
        self.logger = Logger(
            log_dir=log_dir,
            app_name="visualization_agent"
        )
        self.logger.info("Initializing VisualizationAgent")
        
        # Set up vector database
        self.vector_db = PgVector(
            table_name="visualization_knowledge",
            db_url=vector_db_url
        )
        
        # Set up embedder
        self.embedder = OpenAIEmbedder()
        
        # Set up agent model
        self.model = OpenAIChat(name=model_name)
        
        # Create output directory if it doesn't exist
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define supported visualization types
        self.chart_types = {
            "line": self._create_line_chart,
            "bar": self._create_bar_chart,
            "scatter": self._create_scatter_plot,
            "pie": self._create_pie_chart,
            "histogram": self._create_histogram,
            "heatmap": self._create_heatmap,
            "box": self._create_box_plot,
            "violin": self._create_violin_plot,
            "correlation": self._create_correlation_matrix
        }
        
        # Style configuration
        plt.style.use('ggplot')
        self.color_palettes = {
            "default": "viridis",
            "categorical": "Set2",
            "sequential": "Blues",
            "diverging": "RdBu"
        }
        
        self.logger.info("VisualizationAgent initialization completed")
    
    def process_request(self, 
                       request: str, 
                       data: Optional[Union[pd.DataFrame, str]] = None,
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a visualization request.
        
        Args:
            request: The visualization request (e.g., "Plot temperature over time")
            data: DataFrame or path to data file
            context: Optional context information
            
        Returns:
            Dict containing visualization information and results
        """
        self.logger.info(f"Processing visualization request: {request[:50]}...")
        
        try:
            # Load data if needed
            if data is None:
                return {
                    "success": False,
                    "error": "No data provided for visualization.",
                    "request": request
                }
            
            if isinstance(data, str):
                # Data is a file path
                data_path = Path(data)
                if not data_path.exists():
                    return {
                        "success": False, 
                        "error": f"Data file not found: {data}",
                        "request": request
                    }
                
                # Load based on file type
                if data_path.suffix.lower() == '.csv':
                    df = pd.read_csv(data_path)
                elif data_path.suffix.lower() == '.json':
                    df = pd.read_json(data_path)
                elif data_path.suffix.lower() in ['.xls', '.xlsx']:
                    df = pd.read_excel(data_path)
                else:
                    return {
                        "success": False,
                        "error": f"Unsupported data file format: {data_path.suffix}",
                        "request": request
                    }
            else:
                # Data is already a DataFrame
                df = data
            
            # Analyze the request to determine chart type and parameters
            chart_info = self._analyze_request(request, df, context)
            
            if not chart_info["success"]:
                return chart_info
            
            # Create the visualization
            result = self._create_visualization(chart_info, df)
            
            # Log and return results
            self.logger.info(f"Visualization created: {chart_info['chart_type']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing visualization request: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Error creating visualization: {str(e)}",
                "request": request
            }
    
    def _analyze_request(self, 
                        request: str, 
                        data: pd.DataFrame,
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze the visualization request using an LLM to determine chart parameters.
        
        Args:
            request: User's visualization request
            data: DataFrame to visualize
            context: Optional context information
            
        Returns:
            Dictionary with chart parameters and metadata
        """
        try:
            # Prepare prompt with data info
            column_info = {}
            for col in data.columns:
                dtype = str(data[col].dtype)
                sample = str(data[col].iloc[0])
                column_info[col] = {"type": dtype, "sample": sample}
            
            # Build prompt
            prompt = f"""
            I need to create a visualization based on this request: "{request}"
            
            The data has the following columns:
            {json.dumps(column_info, indent=2)}
            
            Data shape: {data.shape[0]} rows x {data.shape[1]} columns
            
            Analyze the request and determine the most appropriate visualization parameters.
            Return a JSON object with the following fields:
            - chart_type: Type of chart to create (one of: line, bar, scatter, pie, histogram, heatmap, box, violin, correlation)
            - title: Chart title
            - x_axis: Column name for the x-axis (if applicable)
            - y_axis: Column name for the y-axis (if applicable)
            - color_by: Column name to use for coloring (optional)
            - group_by: Column name to use for grouping (optional)
            - filters: Dictionary of filters to apply to the data (optional)
            - palette: Color palette to use (default, categorical, sequential, diverging)
            
            Respond with only the JSON object.
            """
            
            # Call LLM
            response = self.model.chat(prompt)
            
            # Extract JSON from response
            try:
                chart_info = json.loads(response)
                
                # Validate chart_type
                if chart_info.get("chart_type") not in self.chart_types:
                    return {
                        "success": False,
                        "error": f"Unsupported chart type: {chart_info.get('chart_type')}. Supported types: {list(self.chart_types.keys())}"
                    }
                
                # Validate axes columns exist in data
                for axis in ["x_axis", "y_axis"]:
                    if axis in chart_info and chart_info[axis] is not None and chart_info[axis] not in data.columns:
                        return {
                            "success": False,
                            "error": f"Column '{chart_info[axis]}' not found in data for {axis}."
                        }
                
                # Add success flag
                chart_info["success"] = True
                return chart_info
                
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "Could not parse chart parameters from LLM response."
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing visualization request: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Analysis error: {str(e)}"
            }
    
    def _create_visualization(self, 
                             chart_info: Dict[str, Any],
                             data: pd.DataFrame) -> Dict[str, Any]:
        """
        Create the visualization based on chart info.
        
        Args:
            chart_info: Dict with chart parameters
            data: DataFrame to visualize
            
        Returns:
            Dict with visualization results
        """
        chart_type = chart_info["chart_type"]
        
        if chart_type not in self.chart_types:
            return {
                "success": False,
                "error": f"Unsupported chart type: {chart_type}"
            }
        
        # Apply filters if provided
        if "filters" in chart_info and chart_info["filters"]:
            try:
                for col, filter_val in chart_info["filters"].items():
                    if col in data.columns:
                        # Handle different filter types
                        if isinstance(filter_val, dict) and "min" in filter_val and "max" in filter_val:
                            # Range filter
                            data = data[(data[col] >= filter_val["min"]) & (data[col] <= filter_val["max"])]
                        elif isinstance(filter_val, list):
                            # List of values
                            data = data[data[col].isin(filter_val)]
                        else:
                            # Single value
                            data = data[data[col] == filter_val]
            except Exception as e:
                self.logger.warning(f"Error applying filters: {str(e)}")
        
        # Create the visualization
        try:
            # Call the appropriate chart function
            chart_function = self.chart_types[chart_type]
            fig, ax = chart_function(chart_info, data)
            
            # Add title if provided
            if "title" in chart_info and chart_info["title"]:
                plt.title(chart_info["title"])
            
            # Save the visualization
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{chart_type}_{timestamp}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
            # Convert to base64 for embedding
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close(fig)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return {
                "success": True,
                "chart_type": chart_type,
                "filepath": str(filepath),
                "image_base64": image_base64,
                "parameters": chart_info
            }
            
        except Exception as e:
            self.logger.error(f"Error creating {chart_type} chart: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Error creating {chart_type} chart: {str(e)}",
                "chart_type": chart_type
            }
    
    def _create_line_chart(self, 
                          chart_info: Dict[str, Any],
                          data: pd.DataFrame):
        """Create a line chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = chart_info["x_axis"]
        y = chart_info["y_axis"]
        
        if "group_by" in chart_info and chart_info["group_by"] in data.columns:
            # Create grouped line chart
            grouped = data.groupby([x, chart_info["group_by"]])[y].mean().unstack()
            grouped.plot(ax=ax, marker='o')
        else:
            # Create simple line chart
            data.plot(x=x, y=y, kind='line', ax=ax, marker='o')
        
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(True, alpha=0.3)
        
        return fig, ax
    
    def _create_bar_chart(self,
                         chart_info: Dict[str, Any],
                         data: pd.DataFrame):
        """Create a bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = chart_info["x_axis"]
        y = chart_info["y_axis"]
        
        palette = self.color_palettes.get(chart_info.get("palette", "default"), "viridis")
        
        if "group_by" in chart_info and chart_info["group_by"] in data.columns:
            # Create grouped bar chart
            grouped_data = data.groupby([x, chart_info["group_by"]])[y].mean().unstack()
            grouped_data.plot(kind='bar', ax=ax, colormap=palette)
        else:
            # Create simple bar chart
            data.plot(x=x, y=y, kind='bar', ax=ax, colormap=palette)
        
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=45)
        
        return fig, ax
    
    def _create_scatter_plot(self,
                            chart_info: Dict[str, Any],
                            data: pd.DataFrame):
        """Create a scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x = chart_info["x_axis"]
        y = chart_info["y_axis"]
        
        # Color by a third variable if specified
        if "color_by" in chart_info and chart_info["color_by"] in data.columns:
            color_col = chart_info["color_by"]
            palette = self.color_palettes.get(chart_info.get("palette", "default"), "viridis")
            
            # Check if the color column is numeric or categorical
            if data[color_col].dtype.kind in 'ifc':  # integer, float, complex
                scatter = ax.scatter(data[x], data[y], c=data[color_col], cmap=palette, alpha=0.7)
                plt.colorbar(scatter, ax=ax, label=color_col)
            else:
                # For categorical data, use seaborn
                sns.scatterplot(x=x, y=y, hue=color_col, data=data, ax=ax, palette=palette)
                plt.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Simple scatter plot
            ax.scatter(data[x], data[y], alpha=0.7)
        
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(True, alpha=0.3)
        
        return fig, ax
    
    def _create_pie_chart(self,
                         chart_info: Dict[str, Any],
                         data: pd.DataFrame):
        """Create a pie chart."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        categories = chart_info["x_axis"]
        
        # If y_axis is specified, use it as values, otherwise count occurrences
        if "y_axis" in chart_info and chart_info["y_axis"] in data.columns:
            values = chart_info["y_axis"]
            # Group by category and sum values
            pie_data = data.groupby(categories)[values].sum()
        else:
            # Count occurrences of each category
            pie_data = data[categories].value_counts()
        
        # Filter small wedges
        threshold = 0.03  # 3% threshold
        small_wedges = pie_data[pie_data / pie_data.sum() < threshold]
        if not small_wedges.empty:
            other = small_wedges.sum()
            pie_data = pie_data[pie_data / pie_data.sum() >= threshold]
            pie_data["Other"] = other
        
        palette = self.color_palettes.get(chart_info.get("palette", "categorical"), "Set2")
        pie_data.plot(kind='pie', ax=ax, autopct='%1.1f%%', colormap=palette)
        ax.set_ylabel('')  # Remove y-label
        
        return fig, ax
    
    def _create_histogram(self,
                         chart_info: Dict[str, Any],
                         data: pd.DataFrame):
        """Create a histogram."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        column = chart_info["x_axis"]
        bins = chart_info.get("bins", 20)  # Default to 20 bins if not specified
        
        palette = self.color_palettes.get(chart_info.get("palette", "default"), "viridis")
        
        if "group_by" in chart_info and chart_info["group_by"] in data.columns:
            # Create grouped histograms
            for name, group in data.groupby(chart_info["group_by"]):
                group[column].plot.hist(alpha=0.5, label=name, bins=bins, ax=ax)
            plt.legend()
        else:
            # Create simple histogram
            data[column].plot.hist(bins=bins, ax=ax, color=plt.cm.get_cmap(palette)(0.6))
        
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        return fig, ax
    
    def _create_heatmap(self,
                       chart_info: Dict[str, Any],
                       data: pd.DataFrame):
        """Create a heatmap."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # If specific columns are provided, use them, otherwise use all numeric columns
        if "x_axis" in chart_info and "y_axis" in chart_info:
            x_col = chart_info["x_axis"]
            y_col = chart_info["y_axis"]
            
            # Create pivot table
            if "color_by" in chart_info and chart_info["color_by"] in data.columns:
                pivot_data = data.pivot_table(index=y_col, columns=x_col, values=chart_info["color_by"], aggfunc='mean')
            else:
                # Without value column, create frequency heatmap
                pivot_data = pd.crosstab(data[y_col], data[x_col])
            
        else:
            # Use correlation matrix of numeric columns
            numeric_data = data.select_dtypes(include=['number'])
            pivot_data = numeric_data.corr()
        
        palette = self.color_palettes.get(chart_info.get("palette", "diverging"), "RdBu_r")
        sns.heatmap(pivot_data, annot=True, cmap=palette, linewidths=0.5, ax=ax)
        
        plt.tight_layout()
        
        return fig, ax
    
    def _create_box_plot(self,
                        chart_info: Dict[str, Any],
                        data: pd.DataFrame):
        """Create a box plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = chart_info["x_axis"]
        y = chart_info["y_axis"]
        
        palette = self.color_palettes.get(chart_info.get("palette", "categorical"), "Set2")
        
        sns.boxplot(x=x, y=y, data=data, ax=ax, palette=palette)
        
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=45)
        
        return fig, ax
    
    def _create_violin_plot(self,
                           chart_info: Dict[str, Any],
                           data: pd.DataFrame):
        """Create a violin plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = chart_info["x_axis"]
        y = chart_info["y_axis"]
        
        palette = self.color_palettes.get(chart_info.get("palette", "categorical"), "Set2")
        
        sns.violinplot(x=x, y=y, data=data, ax=ax, palette=palette)
        
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=45)
        
        return fig, ax
    
    def _create_correlation_matrix(self,
                                  chart_info: Dict[str, Any],
                                  data: pd.DataFrame):
        """Create a correlation matrix."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get numeric columns for correlation
        numeric_data = data.select_dtypes(include=['number'])
        
        if numeric_data.empty:
            raise ValueError("No numeric columns found for correlation matrix")
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Select specific columns if provided
        if "columns" in chart_info and isinstance(chart_info["columns"], list):
            columns = [col for col in chart_info["columns"] if col in corr_matrix.columns]
            if columns:
                corr_matrix = corr_matrix.loc[columns, columns]
        
        # Create heatmap
        palette = self.color_palettes.get(chart_info.get("palette", "diverging"), "RdBu_r")
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(corr_matrix, mask=mask, cmap=palette, vmax=1, vmin=-1, center=0,
                   annot=True, fmt=".2f", square=True, linewidths=0.5, ax=ax)
        
        plt.tight_layout()
        
        return fig, ax

# --- Unit Tests ---
import unittest
from unittest.mock import patch, MagicMock

class TestVisualizationAgent(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.test_log_dir = "test_logs"
        self.test_output_dir = "test_visualizations"
        # Clean up potential leftover dirs from previous runs
        if os.path.exists(self.test_log_dir):
            import shutil
            shutil.rmtree(self.test_log_dir)
        if os.path.exists(self.test_output_dir):
            import shutil
            shutil.rmtree(self.test_output_dir)
            
        # Create data dir for tests
        os.makedirs("test_data", exist_ok=True)
            
        # Patch PgVector and OpenAIEmbedder as they are not the focus here
        with patch('agno.vectordb.pgvector.PgVector', MagicMock()), \
             patch('agno.embedder.openai.OpenAIEmbedder', MagicMock()):
            self.agent = VisualizationAgent(
                log_dir=self.test_log_dir,
                output_dir=self.test_output_dir
            )
            
        # Add data_dir attribute to agent for testing
        self.agent.data_dir = Path("test_data")
            
        # Create dummy data
        self.dummy_data = pd.DataFrame({
            'time': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'value': [10, 15, 12],
            'category': ['A', 'B', 'A']
        })

    def tearDown(self):
        """Clean up after tests."""
        # Remove test directories
        import shutil
        if os.path.exists(self.test_log_dir):
            shutil.rmtree(self.test_log_dir)
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
        if os.path.exists("test_data"):
            shutil.rmtree("test_data")

    def test_initialization(self):
        """Test agent initialization and directory creation."""
        # Check if output_dir exists (log_dir is managed by Logger)
        self.assertTrue(os.path.exists(self.test_output_dir))
        self.assertIsInstance(self.agent.logger, Logger)
        self.assertTrue(len(self.agent.chart_types) > 0)

    def test_analyze_request_success(self):
        """Test successful analysis of a visualization request."""
        # Patch the internal _analyze_request method directly
        with patch.object(self.agent, '_analyze_request') as mock_analyze:
            mock_analyze.return_value = {
                "success": True,
                "chart_type": "line",
                "title": "Value over Time",
                "x_axis": "time",
                "y_axis": "value"
            }
            
            request = "Plot value over time"
            result = self.agent.process_request(request, data=self.dummy_data)
            
            self.assertTrue(result["success"])
            mock_analyze.assert_called_once()

    def test_analyze_request_failure_parsing(self):
        """Test request analysis failure due to LLM response parsing error."""
        # Patch the internal _analyze_request method directly
        with patch.object(self.agent, '_analyze_request') as mock_analyze:
            mock_analyze.return_value = {
                "success": False,
                "error": "Could not parse chart parameters"
            }
            
            request = "Show me a chart"
            result = self.agent.process_request(request, data=self.dummy_data)
            
            self.assertFalse(result["success"])
            mock_analyze.assert_called_once()

    def test_process_request_dataframe_success(self):
        """Test successful processing of a request with DataFrame input."""
        # Patch the internal _analyze_request method directly
        with patch.object(self.agent, '_analyze_request') as mock_analyze:
            # Configure the analyze mock
            mock_analyze.return_value = {
                "success": True,
                "chart_type": "line",
                "title": "Test Chart",
                "x_axis": "time",
                "y_axis": "value"
            }
            
            # Patch the _create_visualization method
            with patch.object(self.agent, '_create_visualization') as mock_create_viz:
                # Configure the visualization mock
                mock_create_viz.return_value = {
                    "success": True,
                    "chart_type": "line",
                    "filepath": str(Path(self.test_output_dir) / "test_chart.png"),
                    "image_base64": "base64_image_data"
                }
                
                # Test process_request with mocked dependencies
                request = "Make a line chart"
                result = self.agent.process_request(request, data=self.dummy_data)
                
                self.assertTrue(result["success"])
                self.assertEqual(result["chart_type"], "line")
                # Output path should be as provided by the mock
                self.assertTrue(result["filepath"].endswith(".png"))
                mock_analyze.assert_called_once_with(request, self.dummy_data, None)
                mock_create_viz.assert_called_once()

    def test_process_request_no_data(self):
        """Test processing request when no data is provided."""
        request = "Create a plot"
        result = self.agent.process_request(request, data=None)
        
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "No data provided for visualization.")

    def test_process_request_file_not_found(self):
        """Test processing request with a non-existent file path."""
        request = "Plot from file"
        result = self.agent.process_request(request, data="non_existent_file.csv")
        
        self.assertFalse(result["success"])
        self.assertTrue(result["error"].startswith("Data file not found:"))
        
    def test_process_request_unsupported_file(self):
        """Test processing request with an unsupported file type."""
        # Create a dummy unsupported file
        dummy_file = Path("test_data") / "dummy.txt"
        with open(dummy_file, "w") as f:
            f.write("dummy data")
            
        request = "Plot from txt file"
        result = self.agent.process_request(request, data=str(dummy_file))
        
        self.assertFalse(result["success"])
        self.assertTrue(result["error"].startswith("Unsupported data file format:"))

# --- Main execution block for tests ---
if __name__ == '__main__':
    unittest.main() 
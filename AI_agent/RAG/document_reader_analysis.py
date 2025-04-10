import os
import re
import json
import pandas as pd
import numpy as np
import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from document_reader_visualization import VisualizationManager, ImageAnalyzer

from Logger import LoggerSetup
# Initialize logger using setup_logger method
logger_setup = LoggerSetup()
logger = logger_setup.setup_logger()

class AnalysisHandler:
    """Handles analysis of different file types and documents."""
    
    def __init__(self, document_reader):
        """Initialize the analysis handler with a reference to the document reader.
        
        Args:
            document_reader: The InteractiveDocumentReader instance
        """
        self.document_reader = document_reader
        self.viz_manager = VisualizationManager()
        self.image_analyzer = ImageAnalyzer(document_reader)
        self.analysis_cache = {}  # Cache analysis results
    
    def analyze_file(self, file_path: str, analysis_type: str = "auto") -> Dict[str, Any]:
        """Analyze a file based on its type.
        
        Args:
            file_path: Path to the file to analyze
            analysis_type: Type of analysis to perform (auto, basic, detailed, etc.)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            # Check cache
            cache_key = f"{file_path}_{analysis_type}"
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]
            
            # Determine file type and analyze accordingly
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.csv', '.xlsx', '.xls']:
                result = self.analyze_tabular_data(file_path, analysis_type)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                result = self.analyze_image(file_path, analysis_type)
            elif file_ext in ['.md', '.txt']:
                result = self.analyze_text_document(file_path, analysis_type)
            elif file_ext == '.json':
                result = self.analyze_json_document(file_path, analysis_type)
            else:
                result = {"error": f"Unsupported file type: {file_ext}"}
            
            # Cache results
            self.analysis_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def analyze_tabular_data(self, file_path: str, analysis_type: str = "basic") -> Dict[str, Any]:
        """Analyze tabular data files (CSV, Excel).
        
        Args:
            file_path: Path to the tabular data file
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Read the file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:  # Excel
                df = pd.read_excel(file_path)
            
            # Basic analysis (always performed)
            result = {
                "file_info": {
                    "name": os.path.basename(file_path),
                    "path": file_path,
                    "size_bytes": os.path.getsize(file_path),
                    "last_modified": os.path.getmtime(file_path)
                },
                "structure": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                    "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                    "datetime_columns": df.select_dtypes(include=['datetime']).columns.tolist(),
                    "missing_values": df.isnull().sum().to_dict(),
                    "total_missing": int(df.isnull().sum().sum()),
                    "missing_percentage": round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2)
                },
                "preview": df.head(5).to_dict(orient='records')
            }
            
            # Basic summary statistics
            if analysis_type in ["basic", "detailed", "auto"]:
                result["summary_stats"] = {}
                numeric_df = df.select_dtypes(include=['number'])
                
                if not numeric_df.empty:
                    stats = numeric_df.describe().to_dict()
                    result["summary_stats"]["numeric"] = stats
                
                categorical_df = df.select_dtypes(include=['object', 'category'])
                if not categorical_df.empty:
                    cat_stats = {}
                    for col in categorical_df.columns:
                        value_counts = categorical_df[col].value_counts().head(10).to_dict()
                        unique_count = categorical_df[col].nunique()
                        cat_stats[col] = {
                            "unique_values": unique_count,
                            "top_values": value_counts,
                            "has_more": unique_count > 10
                        }
                    result["summary_stats"]["categorical"] = cat_stats
            
            # Detailed analysis
            if analysis_type in ["detailed", "auto"] and len(df) > 0:
                # Correlation analysis for numeric columns
                numeric_df = df.select_dtypes(include=['number'])
                if len(numeric_df.columns) >= 2:
                    corr_matrix = numeric_df.corr().round(2).fillna(0).to_dict()
                    
                    # Find highest correlations
                    high_corrs = []
                    for i, col1 in enumerate(numeric_df.columns):
                        for j, col2 in enumerate(numeric_df.columns):
                            if i < j:  # Only get unique pairs
                                corr = round(numeric_df[col1].corr(numeric_df[col2]), 2)
                                if abs(corr) >= 0.5:  # Only strong correlations
                                    high_corrs.append({
                                        "columns": [col1, col2],
                                        "correlation": corr,
                                        "strength": "strong positive" if corr >= 0.7 else 
                                                   "moderate positive" if corr >= 0.5 else
                                                   "strong negative" if corr <= -0.7 else
                                                   "moderate negative"
                                    })
                    
                    result["correlations"] = {
                        "matrix": corr_matrix,
                        "strong_correlations": sorted(high_corrs, key=lambda x: abs(x["correlation"]), reverse=True)
                    }
                
                # Additional stats for numeric columns
                if not numeric_df.empty:
                    advanced_stats = {}
                    for col in numeric_df.columns:
                        col_data = numeric_df[col].dropna()
                        if len(col_data) > 0:
                            # Calculate skewness and kurtosis
                            skewness = round(col_data.skew(), 2)
                            kurtosis = round(col_data.kurtosis(), 2)
                            
                            # Calculate outliers using IQR
                            q1 = col_data.quantile(0.25)
                            q3 = col_data.quantile(0.75)
                            iqr = q3 - q1
                            outlier_count = ((col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))).sum()
                            
                            advanced_stats[col] = {
                                "skewness": skewness,
                                "kurtosis": kurtosis,
                                "outliers": int(outlier_count),
                                "distribution": "normal" if abs(skewness) < 0.5 and abs(kurtosis) < 0.5 else "skewed"
                            }
                    
                    result["advanced_stats"] = advanced_stats
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing tabular data {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def analyze_image(self, file_path: str, analysis_type: str = "basic") -> Dict[str, Any]:
        """Analyze image files.
        
        Args:
            file_path: Path to the image file
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary with image analysis results
        """
        try:
            # Use the ImageAnalyzer for image analysis
            analysis = self.image_analyzer.analyze_image(file_path)
            
            # Generate a text report
            if analysis_type in ["detailed", "auto"]:
                report = self.image_analyzer.generate_image_report(file_path)
                analysis["text_report"] = report
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing image {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def analyze_text_document(self, file_path: str, analysis_type: str = "basic") -> Dict[str, Any]:
        """Analyze text documents like Markdown or plain text files.
        
        Args:
            file_path: Path to the text document
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary with text analysis results
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic analysis
            word_count = len(re.findall(r'\b\w+\b', content))
            line_count = len(content.splitlines())
            char_count = len(content)
            
            result = {
                "file_info": {
                    "name": os.path.basename(file_path),
                    "path": file_path,
                    "size_bytes": os.path.getsize(file_path),
                    "last_modified": os.path.getmtime(file_path),
                    "extension": os.path.splitext(file_path)[1]
                },
                "statistics": {
                    "word_count": word_count,
                    "line_count": line_count,
                    "character_count": char_count,
                    "average_line_length": round(char_count / max(line_count, 1), 1)
                },
                "preview": content[:500] + ("..." if len(content) > 500 else "")
            }
            
            # More detailed analysis for longer documents
            if analysis_type in ["detailed", "auto"] and word_count > 100:
                # Split content into paragraphs and identify sections
                paragraphs = [p for p in re.split(r'\n\s*\n', content) if p.strip()]
                
                # Identify headings in markdown
                if file_path.endswith('.md'):
                    headings = re.findall(r'^(#+)\s+(.+)$', content, re.MULTILINE)
                    heading_structure = [{"level": len(h[0]), "text": h[1].strip()} for h in headings]
                    result["structure"] = {
                        "paragraphs": len(paragraphs),
                        "headings": heading_structure
                    }
                else:
                    result["structure"] = {
                        "paragraphs": len(paragraphs)
                    }
                
                # Extract potential keywords 
                words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
                word_freq = {}
                for word in words:
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
                
                # Get top words by frequency
                common_words = {"the", "and", "for", "that", "this", "with", "from", "have", "are", "not", "been"}
                filtered_words = {k: v for k, v in word_freq.items() if k not in common_words}
                top_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:15]
                
                result["content_analysis"] = {
                    "top_words": dict(top_words),
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text document {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def analyze_json_document(self, file_path: str, analysis_type: str = "basic") -> Dict[str, Any]:
        """Analyze JSON documents.
        
        Args:
            file_path: Path to the JSON file
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary with JSON analysis results
        """
        try:
            # Read JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Basic analysis
            result = {
                "file_info": {
                    "name": os.path.basename(file_path),
                    "path": file_path,
                    "size_bytes": os.path.getsize(file_path),
                    "last_modified": os.path.getmtime(file_path)
                },
                "structure": self._analyze_json_structure(json_data)
            }
            
            # Generate preview but handle potentially large JSON objects
            if isinstance(json_data, dict):
                preview = {k: json_data[k] for k in list(json_data.keys())[:5]}
            elif isinstance(json_data, list):
                preview = json_data[:3]
            else:
                preview = json_data
            
            result["preview"] = preview
            
            # Add sample data or detailed analysis
            if analysis_type in ["detailed", "auto"]:
                result["detailed"] = self._get_detailed_json_analysis(json_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing JSON document {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _analyze_json_structure(self, json_data, max_depth=3, current_depth=0) -> Dict[str, Any]:
        """Analyze the structure of a JSON object.
        
        Args:
            json_data: The JSON data to analyze
            max_depth: Maximum depth to analyze
            current_depth: Current depth in the analysis
            
        Returns:
            Dictionary with structure information
        """
        if current_depth >= max_depth:
            return {"type": type(json_data).__name__, "max_depth_reached": True}
        
        if isinstance(json_data, dict):
            keys = list(json_data.keys())
            result = {
                "type": "object",
                "keys_count": len(keys),
                "keys": keys[:10] + (["..."] if len(keys) > 10 else [])
            }
            
            # Analyze a sample of keys if there are too many
            if len(keys) > 0:
                sample_keys = keys[:3]
                properties = {}
                for key in sample_keys:
                    properties[key] = self._analyze_json_structure(json_data[key], max_depth, current_depth + 1)
                result["properties_sample"] = properties
            
            return result
            
        elif isinstance(json_data, list):
            result = {
                "type": "array",
                "length": len(json_data)
            }
            
            # Analyze array items if not empty
            if len(json_data) > 0:
                # Check if all items are of the same type
                item_types = set(type(item).__name__ for item in json_data[:10])
                result["item_types"] = list(item_types)
                
                # Sample item analysis
                if len(json_data) > 0:
                    result["sample_item"] = self._analyze_json_structure(json_data[0], max_depth, current_depth + 1)
            
            return result
            
        else:
            # For primitive types
            return {
                "type": type(json_data).__name__,
                "value": str(json_data) if not isinstance(json_data, (int, float, bool, type(None))) else json_data
            }
    
    def _get_detailed_json_analysis(self, json_data) -> Dict[str, Any]:
        """Get more detailed analysis of JSON data.
        
        Args:
            json_data: The JSON data to analyze
            
        Returns:
            Dictionary with detailed analysis
        """
        result = {}
        
        # Extract nested data structure types
        if isinstance(json_data, dict):
            value_types = {}
            for key in json_data:
                value_type = type(json_data[key]).__name__
                if value_type in value_types:
                    value_types[value_type] += 1
                else:
                    value_types[value_type] = 1
                    
            result["value_types"] = value_types
            
            # Check for potential schema inconsistency
            if "list" in value_types and "dict" in value_types:
                result["potential_issues"] = ["Mixed data structures (lists and dictionaries)"]
                
        elif isinstance(json_data, list) and len(json_data) > 0:
            # Check if list items have consistent structure
            if all(isinstance(item, dict) for item in json_data):
                # For list of dictionaries, check key consistency
                all_keys = set()
                for item in json_data:
                    all_keys.update(item.keys())
                
                # Check if all items have the same keys
                common_keys = set.intersection(*[set(item.keys()) for item in json_data])
                missing_keys = all_keys - common_keys
                
                result["array_analysis"] = {
                    "total_keys": len(all_keys),
                    "common_keys": len(common_keys),
                    "inconsistent_keys": len(missing_keys),
                    "key_consistency_percentage": round(len(common_keys) / max(1, len(all_keys)) * 100, 1)
                }
                
                if missing_keys:
                    result["inconsistent_keys"] = list(missing_keys)
            else:
                # For mixed array types
                item_types = [type(item).__name__ for item in json_data]
                type_counts = {}
                for item_type in item_types:
                    if item_type in type_counts:
                        type_counts[item_type] += 1
                    else:
                        type_counts[item_type] = 1
                
                result["array_analysis"] = {
                    "item_type_counts": type_counts,
                    "mixed_types": len(type_counts) > 1
                }
        
        return result
    
    def suggest_visualizations(self, file_path: str) -> List[Dict[str, Any]]:
        """Suggest appropriate visualizations for a data file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            List of suggested visualization options
        """
        try:
            if not file_path.endswith(('.csv', '.xlsx', '.xls')):
                return [{"error": "Visualization suggestions only available for tabular data"}]
            
            # Read the file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:  # Excel
                df = pd.read_excel(file_path)
            
            suggestions = []
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            # Distribution visualizations for numeric columns
            for col in numeric_cols[:3]:  # Limit to first 3 to avoid too many suggestions
                suggestions.append({
                    "chart_type": "histogram",
                    "title": f"Distribution of {col}",
                    "columns": [col],
                    "description": f"Visualize the distribution of values in {col}"
                })
            
            # Categorical data visualizations
            for col in categorical_cols[:3]:
                if df[col].nunique() <= 10:
                    suggestions.append({
                        "chart_type": "bar",
                        "title": f"Count by {col}",
                        "columns": [col],
                        "description": f"Compare counts across different {col} categories"
                    })
                    
                    suggestions.append({
                        "chart_type": "pie",
                        "title": f"Percentage by {col}",
                        "columns": [col],
                        "description": f"Show the percentage breakdown of {col} categories"
                    })
            
            # Correlation visualizations
            if len(numeric_cols) >= 2:
                # Scatter plots for pairs of numeric columns
                for i, col1 in enumerate(numeric_cols[:2]):
                    for col2 in numeric_cols[i+1:3]:  # Limit combinations
                        suggestions.append({
                            "chart_type": "scatter",
                            "title": f"{col1} vs {col2}",
                            "columns": [col1, col2],
                            "description": f"Explore relationship between {col1} and {col2}"
                        })
                
                # Correlation heatmap
                if len(numeric_cols) >= 3:
                    suggestions.append({
                        "chart_type": "heatmap",
                        "title": "Correlation Matrix",
                        "columns": numeric_cols,
                        "description": "Visualize correlations between numeric variables"
                    })
            
            # Time series visualizations
            if datetime_cols and numeric_cols:
                for date_col in datetime_cols[:1]:
                    for numeric_col in numeric_cols[:2]:
                        suggestions.append({
                            "chart_type": "line",
                            "title": f"{numeric_col} over {date_col}",
                            "columns": [date_col, numeric_col],
                            "description": f"Track changes in {numeric_col} over time"
                        })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error suggesting visualizations for {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return [{"error": str(e)}]
    
    def extract_insights(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Extract key insights from analysis results.
        
        Args:
            analysis_result: Analysis results dictionary
            
        Returns:
            List of insight statements
        """
        insights = []
        
        try:
            # Check if there's an error in the analysis
            if "error" in analysis_result:
                return [f"Analysis error: {analysis_result['error']}"]
            
            # Insights for tabular data
            if "structure" in analysis_result and "rows" in analysis_result["structure"]:
                # Basic data size insights
                rows = analysis_result["structure"]["rows"]
                cols = analysis_result["structure"]["columns"]
                insights.append(f"The dataset contains {rows} rows and {cols} columns.")
                
                # Missing data insights
                if "missing_percentage" in analysis_result["structure"]:
                    missing_pct = analysis_result["structure"]["missing_percentage"]
                    if missing_pct > 0:
                        severity = "significant" if missing_pct > 10 else "some"
                        insights.append(f"There is {severity} missing data ({missing_pct}% of all values).")
                
                # Correlation insights
                if "correlations" in analysis_result and "strong_correlations" in analysis_result["correlations"]:
                    strong_corrs = analysis_result["correlations"]["strong_correlations"]
                    if strong_corrs:
                        top_corr = strong_corrs[0]
                        col1, col2 = top_corr["columns"]
                        corr_val = top_corr["correlation"]
                        direction = "positive" if corr_val > 0 else "negative"
                        insights.append(f"Strong {direction} correlation ({corr_val}) found between {col1} and {col2}.")
                
                # Distribution insights
                if "advanced_stats" in analysis_result:
                    skewed_cols = [col for col, stats in analysis_result["advanced_stats"].items() 
                                if stats.get("distribution") == "skewed"]
                    if skewed_cols:
                        insights.append(f"The following columns have skewed distributions: {', '.join(skewed_cols[:3])}" + 
                                     (f" and {len(skewed_cols)-3} more" if len(skewed_cols) > 3 else ""))
                
                # Outlier insights
                outlier_cols = []
                if "advanced_stats" in analysis_result:
                    for col, stats in analysis_result["advanced_stats"].items():
                        if stats.get("outliers", 0) > 0:
                            outlier_pct = round((stats["outliers"] / rows) * 100, 1)
                            if outlier_pct > 1:  # Only report significant outliers
                                outlier_cols.append((col, stats["outliers"], outlier_pct))
                
                if outlier_cols:
                    top_outlier = max(outlier_cols, key=lambda x: x[1])
                    insights.append(f"Column '{top_outlier[0]}' has the most outliers ({top_outlier[1]} values, {top_outlier[2]}% of data).")
            
            # Insights for images
            elif "metadata" in analysis_result and "analysis" in analysis_result:
                metadata = analysis_result.get("metadata", {})
                img_analysis = analysis_result.get("analysis", {})
                
                # Basic image insights
                if "filename" in metadata and "size" in metadata:
                    w, h = metadata["size"] if isinstance(metadata["size"], tuple) else (metadata["width"], metadata["height"])
                    insights.append(f"Image '{metadata['filename']}' has dimensions {w}x{h} pixels.")
                
                # Color insights
                if "avg_color_rgb" in img_analysis:
                    avg_color = img_analysis["avg_color_rgb"]
                    brightness = img_analysis.get("brightness", 0)
                    brightness_desc = "dark" if brightness < 85 else "bright" if brightness > 170 else "medium brightness"
                    insights.append(f"The image has an average {brightness_desc} color profile.")
                
                # Edge detection insights
                if "edge_percentage" in img_analysis:
                    edge_pct = img_analysis["edge_percentage"] * 100
                    complexity = "high detail/complexity" if edge_pct > 15 else "moderate detail" if edge_pct > 5 else "low detail/simplicity"
                    insights.append(f"The image shows {complexity} (edge density: {edge_pct:.1f}%).")
            
            # Insights for text documents
            elif "statistics" in analysis_result and "word_count" in analysis_result["statistics"]:
                stats = analysis_result["statistics"]
                
                # Basic text insights
                word_count = stats["word_count"]
                size_desc = "very long" if word_count > 2000 else "long" if word_count > 1000 else "medium-length" if word_count > 500 else "short"
                insights.append(f"This is a {size_desc} document with {word_count} words across {stats['line_count']} lines.")
                
                # Content analysis
                if "content_analysis" in analysis_result and "top_words" in analysis_result["content_analysis"]:
                    top_words = list(analysis_result["content_analysis"]["top_words"].items())
                    if top_words:
                        top_3_words = ", ".join([f"{word} ({count})" for word, count in top_words[:3]])
                        insights.append(f"Most frequent significant words: {top_3_words}.")
                
                # Structure insights
                if "structure" in analysis_result and "headings" in analysis_result["structure"]:
                    heading_count = len(analysis_result["structure"]["headings"])
                    if heading_count > 0:
                        insights.append(f"The document contains {heading_count} section headings.")
            
            # Insights for JSON data
            elif "structure" in analysis_result and "type" in analysis_result["structure"]:
                structure = analysis_result["structure"]
                
                # Basic JSON insights
                if structure["type"] == "object":
                    insights.append(f"JSON object with {structure['keys_count']} top-level keys.")
                elif structure["type"] == "array":
                    item_desc = ""
                    if "item_types" in structure:
                        if len(structure["item_types"]) == 1:
                            item_desc = f" of {structure['item_types'][0]}s"
                        else:
                            item_desc = " of mixed types"
                    insights.append(f"JSON array containing {structure['length']} items{item_desc}.")
                
                # Schema consistency insights
                if "detailed" in analysis_result and "array_analysis" in analysis_result["detailed"]:
                    array_analysis = analysis_result["detailed"]["array_analysis"]
                    if "key_consistency_percentage" in array_analysis:
                        consistency = array_analysis["key_consistency_percentage"]
                        desc = "consistent" if consistency == 100 else "mostly consistent" if consistency > 80 else "inconsistent"
                        insights.append(f"The JSON array has a {desc} schema structure ({consistency}% key consistency).")
                    elif "mixed_types" in array_analysis and array_analysis["mixed_types"]:
                        insights.append("The JSON array contains items of different types.")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
            logger.error(traceback.format_exc())
            return [f"Error generating insights: {str(e)}"] 
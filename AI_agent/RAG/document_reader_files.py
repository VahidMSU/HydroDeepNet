import os
import re
import json
import logging
import traceback
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
import glob
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from collections import defaultdict

from Logger import LoggerSetup
# Initialize logger using setup_logger method
logger_setup = LoggerSetup()
logger = logger_setup.setup_logger()

class FileManager:
    """Manages file operations for the document reader."""
    
    def __init__(self, base_directory: str = None):
        """Initialize the file manager with a base directory."""
        self.base_directory = base_directory or os.getcwd()
        self.file_cache = {}  # Cache file contents to avoid redundant file reads
        self.file_metadata = {}  # Store metadata about files
        self.known_extensions = {
            'text': ['.txt', '.md'],
            'data': ['.csv', '.json', '.xlsx'],
            'image': ['.png', '.jpg', '.jpeg', '.gif'],
            'code': ['.py', '.js', '.html', '.css']
        }
    
    def set_base_directory(self, directory: str) -> None:
        """Set the base directory for file operations."""
        if os.path.exists(directory) and os.path.isdir(directory):
            self.base_directory = directory
        else:
            raise ValueError(f"Directory '{directory}' does not exist or is not a directory")
    
    def discover_files(self, directory: str = None, file_types: List[str] = None) -> Dict[str, List[str]]:
        """
        Discover files in the specified directory.
        
        Args:
            directory: Directory to scan for files, defaults to base_directory
            file_types: List of file extensions to look for, defaults to all known extensions
            
        Returns:
            Dictionary mapping file types to lists of file paths
        """
        try:
            directory = directory or self.base_directory
            
            if not os.path.exists(directory):
                logger.error(f"Directory '{directory}' does not exist")
                return {}
            
            # Get all extensions if not specified
            if not file_types:
                file_types = [ext for exts in self.known_extensions.values() for ext in exts]
            
            # Convert extensions to lowercase and ensure they start with a dot
            file_types = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in file_types]
            
            # Group files by type
            files_by_type = defaultdict(list)
            
            for root, _, files in os.walk(directory):
                for file in files:
                    _, ext = os.path.splitext(file)
                    ext = ext.lower()
                    
                    if not file_types or ext in file_types:
                        rel_path = os.path.relpath(os.path.join(root, file), directory)
                        
                        # Group by category
                        if ext in self.known_extensions['text']:
                            files_by_type['text'].append(rel_path)
                        elif ext in self.known_extensions['data']:
                            files_by_type['data'].append(rel_path)
                        elif ext in self.known_extensions['image']:
                            files_by_type['image'].append(rel_path)
                        elif ext in self.known_extensions['code']:
                            files_by_type['code'].append(rel_path)
                        else:
                            files_by_type['other'].append(rel_path)
            
            # Sort files in each category
            for file_type in files_by_type:
                files_by_type[file_type].sort()
            
            return dict(files_by_type)
            
        except Exception as e:
            logger.error(f"Error discovering files: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def get_file_path(self, file_name: str) -> str:
        """
        Get the full path for a file.
        
        Args:
            file_name: Name of the file (can be a relative path)
            
        Returns:
            Full path to the file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        try:
            # Try the file name exactly as provided
            full_path = os.path.join(self.base_directory, file_name)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                return full_path
            
            # Try to search for the file (case-insensitive)
            for root, _, files in os.walk(self.base_directory):
                for file in files:
                    if file.lower() == os.path.basename(file_name).lower():
                        return os.path.join(root, file)
            
            # Try to find a partial match (useful when users provide incomplete file names)
            file_name_lower = os.path.basename(file_name).lower()
            for root, _, files in os.walk(self.base_directory):
                for file in files:
                    if file_name_lower in file.lower():
                        return os.path.join(root, file)
            
            # If we get here, the file wasn't found
            raise FileNotFoundError(f"File '{file_name}' not found in {self.base_directory}")
            
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise
            
            logger.error(f"Error getting file path: {str(e)}")
            logger.error(traceback.format_exc())
            raise FileNotFoundError(f"File '{file_name}' not found or error accessing it")
    
    def get_file_type(self, file_path: str) -> str:
        """
        Get the type of a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type: 'text', 'data', 'image', 'code', or 'other'
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        for file_type, extensions in self.known_extensions.items():
            if ext in extensions:
                return file_type
        
        # Get the extension without the dot
        return ext[1:] if ext.startswith('.') else ext
    
    def read_file(self, file_path: str, file_type: str = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Read a file and return its contents and metadata.
        
        Args:
            file_path: Path to the file
            file_type: Optional file type override
            
        Returns:
            Tuple of (file_contents, metadata)
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File '{file_path}' not found")
            
            # Check cache first
            file_hash = self._get_file_hash(file_path)
            if file_hash in self.file_cache:
                return self.file_cache[file_hash], self.file_metadata.get(file_hash, {})
            
            # Determine file type if not provided
            if not file_type:
                file_type = self.get_file_type(file_path)
            
            # Read file based on type
            if file_type == 'text' or file_path.lower().endswith(('.txt', '.md')):
                content, metadata = self._read_text_file(file_path)
            
            elif file_type == 'data' or file_path.lower().endswith('.csv'):
                content, metadata = self._read_csv_file(file_path)
            
            elif file_type == 'data' or file_path.lower().endswith('.json'):
                content, metadata = self._read_json_file(file_path)
            
            elif file_type == 'image' or file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                content, metadata = self._read_image_file(file_path)
            
            else:
                # Default to reading as text
                content, metadata = self._read_text_file(file_path)
            
            # Cache the results
            self.file_cache[file_hash] = content
            self.file_metadata[file_hash] = metadata
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Error reading file '{file_path}': {str(e)}")
            logger.error(traceback.format_exc())
            if isinstance(e, FileNotFoundError):
                raise
            raise ValueError(f"Error reading file '{file_path}': {str(e)}")
    
    def _read_text_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Read a text file and return its contents and metadata."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        metadata = {
            'size': os.path.getsize(file_path),
            'last_modified': os.path.getmtime(file_path),
            'line_count': len(content.split('\n')),
            'word_count': len(content.split()),
            'char_count': len(content)
        }
        
        # If it's a markdown file, get headers
        if file_path.lower().endswith('.md'):
            metadata['headers'] = self._extract_markdown_headers(content)
        
        return content, metadata
    
    def _read_csv_file(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Read a CSV file and return its contents as a DataFrame and metadata."""
        try:
            df = pd.read_csv(file_path)
            
            metadata = {
                'size': os.path.getsize(file_path),
                'last_modified': os.path.getmtime(file_path),
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'column_types': {col: str(df[col].dtype) for col in df.columns},
                'missing_values': df.isnull().sum().sum()
            }
            
            return df, metadata
            
        except Exception as e:
            logger.error(f"Error reading CSV file '{file_path}': {str(e)}")
            raise ValueError(f"Error reading CSV file: {str(e)}")
    
    def _read_json_file(self, file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Read a JSON file and return its contents and metadata."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = json.load(f)
        
        metadata = {
            'size': os.path.getsize(file_path),
            'last_modified': os.path.getmtime(file_path)
        }
        
        # Get structure information
        if isinstance(content, dict):
            metadata['keys'] = list(content.keys())
            metadata['structure'] = 'object'
        elif isinstance(content, list):
            metadata['length'] = len(content)
            metadata['structure'] = 'array'
            if content and isinstance(content[0], dict):
                metadata['sample_keys'] = list(content[0].keys())
        
        return content, metadata
    
    def _read_image_file(self, file_path: str) -> Tuple[Image.Image, Dict[str, Any]]:
        """Read an image file and return its contents and metadata."""
        try:
            image = Image.open(file_path)
            
            metadata = {
                'size': os.path.getsize(file_path),
                'last_modified': os.path.getmtime(file_path),
                'width': image.width,
                'height': image.height,
                'format': image.format,
                'mode': image.mode
            }
            
            return image, metadata
            
        except Exception as e:
            logger.error(f"Error reading image file '{file_path}': {str(e)}")
            raise ValueError(f"Error reading image file: {str(e)}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get a hash of the file's path and modification time."""
        mtime = str(os.path.getmtime(file_path))
        return hashlib.md5(f"{file_path}:{mtime}".encode()).hexdigest()
    
    def _extract_markdown_headers(self, content: str) -> Dict[str, List[str]]:
        """Extract headers from markdown content."""
        headers = {
            'h1': [],
            'h2': [],
            'h3': [],
            'h4': [],
            'h5': [],
            'h6': []
        }
        
        # Match headers like "# Header" or "## Subheader"
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        
        for match in header_pattern.finditer(content):
            level = len(match.group(1))
            text = match.group(2).strip()
            headers[f'h{level}'].append(text)
        
        return headers
    
    def search_files(self, search_term: str, file_types: List[str] = None) -> Dict[str, List[str]]:
        """
        Search for files containing the search term.
        
        Args:
            search_term: Term to search for
            file_types: Optional list of file types to search in
            
        Returns:
            Dictionary mapping file types to lists of matching file paths
        """
        try:
            search_term = search_term.lower()
            results_by_type = defaultdict(list)
            
            # Discover all files if needed
            all_files = self.discover_files(file_types=file_types)
            
            # Loop through each file type and search within files
            for file_type, files in all_files.items():
                for file_rel_path in files:
                    file_path = os.path.join(self.base_directory, file_rel_path)
                    
                    # Skip binary files for content searching
                    if file_type == 'image':
                        # For images, just check the filename
                        if search_term in os.path.basename(file_path).lower():
                            results_by_type[file_type].append(file_rel_path)
                        continue
                    
                    # Read the file and search for the term
                    try:
                        content, _ = self.read_file(file_path)
                        
                        # For DataFrames, convert to string for searching
                        if isinstance(content, pd.DataFrame):
                            content_str = str(content.to_string())
                            if search_term in content_str.lower():
                                results_by_type[file_type].append(file_rel_path)
                                continue
                            
                            # Also search column names
                            if any(search_term in col.lower() for col in content.columns):
                                results_by_type[file_type].append(file_rel_path)
                                continue
                        
                        # For dictionaries, convert to string for searching
                        elif isinstance(content, dict):
                            content_str = json.dumps(content)
                            if search_term in content_str.lower():
                                results_by_type[file_type].append(file_rel_path)
                                continue
                        
                        # For other content types (primarily text), search directly
                        elif isinstance(content, str):
                            if search_term in content.lower():
                                results_by_type[file_type].append(file_rel_path)
                                continue
                    
                    except Exception as e:
                        logger.warning(f"Error searching in file '{file_path}': {str(e)}")
                        continue
            
            # Convert defaultdict to regular dict
            return dict(results_by_type)
            
        except Exception as e:
            logger.error(f"Error searching files: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def find_images(self) -> List[str]:
        """Find all image files in the base directory."""
        try:
            files_by_type = self.discover_files()
            return files_by_type.get('image', [])
            
        except Exception as e:
            logger.error(f"Error finding images: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def get_file_preview(self, file_path: str, max_lines: int = 10) -> str:
        """
        Get a preview of a file's contents.
        
        Args:
            file_path: Path to the file
            max_lines: Maximum number of lines to include in the preview
            
        Returns:
            Preview of the file's contents
        """
        try:
            # Get file extension
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            # For text-based files
            if ext in self.known_extensions['text'] or ext in self.known_extensions['code']:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            lines.append('...')
                            break
                        lines.append(line.rstrip())
                return '\n'.join(lines)
            
            # For CSV files
            elif ext == '.csv':
                df = pd.read_csv(file_path)
                return df.head(max_lines).to_string()
            
            # For JSON files
            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    data = json.load(f)
                return json.dumps(data, indent=2)[:500] + '...' if len(json.dumps(data)) > 500 else json.dumps(data, indent=2)
            
            # For image files
            elif ext in self.known_extensions['image']:
                img = Image.open(file_path)
                return f"Image: {img.format} {img.size[0]}x{img.size[1]} {img.mode}"
            
            # For other file types
            else:
                return f"File: {os.path.basename(file_path)} ({os.path.getsize(file_path)} bytes)"
                
        except Exception as e:
            logger.error(f"Error getting file preview: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error previewing file: {str(e)}"
    
    def get_file_stats(self, directory: str = None) -> Dict[str, Any]:
        """
        Get statistics about files in the directory.
        
        Args:
            directory: Directory to get stats for, defaults to base_directory
            
        Returns:
            Dictionary with file statistics
        """
        try:
            directory = directory or self.base_directory
            
            if not os.path.exists(directory):
                logger.error(f"Directory '{directory}' does not exist")
                return {}
            
            # Discover all files
            files_by_type = self.discover_files(directory)
            
            # Initialize stats
            stats = {
                'total_files': sum(len(files) for files in files_by_type.values()),
                'by_type': {file_type: len(files) for file_type, files in files_by_type.items()},
                'details': {}
            }
            
            # Get details for CSV files
            if 'data' in files_by_type and files_by_type['data']:
                csv_files = [f for f in files_by_type['data'] if f.lower().endswith('.csv')]
                stats['details']['csv'] = {}
                
                for file_rel_path in csv_files:
                    file_path = os.path.join(directory, file_rel_path)
                    try:
                        df = pd.read_csv(file_path)
                        stats['details']['csv'][file_rel_path] = {
                            'rows': len(df),
                            'columns': len(df.columns)
                        }
                    except Exception as e:
                        logger.warning(f"Error reading CSV file '{file_path}': {str(e)}")
            
            # Get details for image files
            if 'image' in files_by_type and files_by_type['image']:
                stats['details']['image'] = {}
                
                for file_rel_path in files_by_type['image']:
                    file_path = os.path.join(directory, file_rel_path)
                    try:
                        img = Image.open(file_path)
                        stats['details']['image'][file_rel_path] = {
                            'width': img.width,
                            'height': img.height,
                            'format': img.format
                        }
                    except Exception as e:
                        logger.warning(f"Error reading image file '{file_path}': {str(e)}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting file stats: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def encode_image(self, image_path: str) -> str:
        """Encode an image to base64 for analysis."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
                
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def visualize_data(self, file_path: str, x_column: str, y_column: Optional[str] = None, 
                       plot_type: str = 'auto', **kwargs) -> str:
        """Create a visualization of data from a CSV file."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check if file is a CSV
            if not file_path.lower().endswith('.csv'):
                raise ValueError("Visualization is only supported for CSV files.")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if columns exist
            if x_column not in df.columns:
                raise ValueError(f"Column '{x_column}' not found in the CSV file.")
            
            if y_column and y_column not in df.columns:
                raise ValueError(f"Column '{y_column}' not found in the CSV file.")
            
            # Determine plot type if auto
            if plot_type == 'auto':
                if y_column:
                    # If both x and y are specified, default to scatter plot
                    plot_type = 'scatter'
                else:
                    # If only x is specified, check data type
                    if pd.api.types.is_numeric_dtype(df[x_column]):
                        plot_type = 'histogram'
                    else:
                        plot_type = 'bar'
            
            # Create figure and axes
            plt.figure(figsize=(10, 6))
            
            # Create the plot based on type
            if plot_type == 'histogram':
                sns.histplot(df[x_column], kde=True)
                plt.xlabel(x_column)
                plt.ylabel('Frequency')
                plt.title(f'Histogram of {x_column}')
                
            elif plot_type == 'bar':
                value_counts = df[x_column].value_counts().sort_index()
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.xlabel(x_column)
                plt.ylabel('Count')
                plt.title(f'Bar plot of {x_column}')
                plt.xticks(rotation=45)
                
            elif plot_type == 'scatter':
                if not y_column:
                    raise ValueError("Y column must be specified for scatter plot.")
                sns.scatterplot(x=df[x_column], y=df[y_column])
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title(f'{y_column} vs {x_column}')
                
            elif plot_type == 'line':
                if not y_column:
                    raise ValueError("Y column must be specified for line plot.")
                sns.lineplot(x=df[x_column], y=df[y_column])
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title(f'{y_column} vs {x_column}')
                
            elif plot_type == 'box':
                sns.boxplot(y=df[x_column])
                plt.ylabel(x_column)
                plt.title(f'Box plot of {x_column}')
                
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            # Save the plot to a buffer
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Encode the plot as base64
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Close the plot to free memory
            plt.close()
            
            # Add the plot to the context
            plot_info = {
                'type': plot_type,
                'x_column': x_column,
                'y_column': y_column,
                'source_file': os.path.basename(file_path),
                'image_base64': image_base64
            }
            
            # Store the plot in the context
            if 'visualizations' not in self.document_reader.context:
                self.document_reader.context['visualizations'] = []
            
            self.document_reader.context['visualizations'].append(plot_info)
            
            logger.info(f"Created {plot_type} visualization of {x_column}" + 
                       (f" vs {y_column}" if y_column else ""))
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error visualizing data: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_file_summary(self, file_data: Dict[str, Any]) -> str:
        """Generate a summary of a file based on its type and content."""
        print(f"File data: {file_data}")    
        try:
            file_type = file_data.get('type', '').lower()
            
            if file_type in ['md', 'txt']:
                # For text files, provide a summary of the content
                content = file_data.get('content', '')
                total_lines = file_data.get('metadata', {}).get('lines', 0)
                
                return f"Text file with {total_lines} lines. First few lines:\n\n" + \
                        content[:500] + ("..." if len(content) > 500 else "")
                
            elif file_type == 'csv':
                # For CSV files, provide summary statistics
                df = file_data.get('content')
                if df is None:
                    return "Empty CSV file or failed to load."
                
                metadata = file_data.get('metadata', {})
                rows = metadata.get('rows', 0)
                columns = metadata.get('columns', 0)
                column_names = metadata.get('column_names', [])
                
                summary = f"CSV file with {rows} rows and {columns} columns.\n"
                summary += f"Columns: {', '.join(column_names)}\n\n"
                
                # Add a sample of the data
                summary += "Sample data:\n"
                summary += df.head(5).to_string()
                
                return summary
                
            elif file_type in ['png', 'jpg', 'jpeg', 'gif']:
                # For images, provide metadata
                metadata = file_data.get('metadata', {})
                width = metadata.get('width', 0)
                height = metadata.get('height', 0)
                img_format = metadata.get('format', 'Unknown')
                
                return f"Image file ({img_format}): {width}x{height} pixels."
                
            else:
                return f"Unsupported file type: {file_type}"
            
        except Exception as e:
            logger.error(f"Error getting file summary: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error getting file summary: {str(e)}"

    
    def find_matching_files(self, search_term: str) -> List[Dict[str, str]]:
        """Find files that match a search term in their name or path."""
        try:
            if not search_term or not self.document_reader.context.get('available_files'):
                return []
            
            matching_files = []
            search_term_lower = search_term.lower()
            
            # Search in each file type
            for ext, file_list in self.document_reader.context.get('available_files', {}).items():
                for file_path in file_list:
                    if search_term_lower in file_path.lower():
                        matching_files.append({
                            'path': file_path,
                            'name': os.path.basename(file_path),
                            'type': ext
                        })
            
            return matching_files
            
        except Exception as e:
            logger.error(f"Error finding matching files: {str(e)}")
            logger.error(traceback.format_exc())
            return [] 
        

if __name__ == "__main__":
    file_manager = FileManager()
    file_manager.set_base_directory("/data/SWATGenXApp/Users/admin/Reports/")
    print(file_manager.discover_files())

    # Get file stats
    stats = file_manager.get_file_stats()
    #print(stats)

    # Read the file to get content and metadata
    file_path = "20250408_154141/prism/prism_report.md"
    # Get the full system path
    full_path = file_manager.get_file_path(file_path)
    # Read the file to get content and metadata
    content, metadata = file_manager.read_file(full_path)
    # Create file data dictionary for the summary function
    file_data = {
        'type': 'md',
        'content': content,
        'metadata': metadata
    }
    # Then get file summary using the file data
    summary = file_manager.get_file_summary(file_data)
    print(summary)
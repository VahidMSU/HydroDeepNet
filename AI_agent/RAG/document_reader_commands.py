import os
import re
import logging
import traceback
import json
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from Logger import LoggerSetup
# Initialize logger using setup_logger method
logger_setup = LoggerSetup()
logger = logger_setup.setup_logger()

class CommandHandler:
    """Handles commands for the document reader."""
    
    def __init__(self, document_reader):
        """Initialize the command handler with reference to the document reader."""
        self.document_reader = document_reader
        self.commands = {
            "help": self.help_command,
            "list": self.list_command,
            "read": self.read_command,
            "analyze": self.analyze_command,
            "visualize": self.visualize_command,
            "summary": self.summary_command,
            "search": self.search_command,
            "image": self.image_command,
            "csv": self.csv_command,
            "discover": self.discover_command,
            "clear": self.clear_command
        }
    
    def process_command(self, message: str) -> Optional[str]:
        """Process a command from the user message."""
        try:
            # Check if the message is a command
            command_match = re.match(r'^/(\w+)(.*)', message.strip())
            if not command_match:
                return None
            
            command = command_match.group(1).lower()
            arguments = command_match.group(2).strip()
            
            # Check if the command exists
            if command not in self.commands:
                return f"Unknown command: /{command}. Type /help for available commands."
            
            # Execute the command
            return self.commands[command](arguments)
            
        except Exception as e:
            logger.error(f"Error processing command: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error processing command: {str(e)}"
    
    def help_command(self, arguments: str) -> str:
        """Display help information about available commands."""
        help_text = """
## Available Commands

- **/help**: Display this help message
- **/list [type]**: List available files (optional: specify type like csv, md, txt, png)
- **/read [filename]**: Read and display the content of a file
- **/analyze [filename]**: Analyze a file and provide insights
- **/visualize [filename] [x_column] [y_column] [type]**: Create a visualization from CSV data
- **/summary**: Show a summary of available data sources
- **/search [term]**: Search for files containing the specified term
- **/image [filename]**: Display and analyze an image
- **/csv [filename] [query]**: Query a CSV file (e.g., "/csv data.csv show first 5 rows")
- **/discover [directory]**: Auto-discover files in a directory
- **/clear**: Clear the conversation history

You can also ask questions about your data in natural language.
        """
        return help_text
    
    def list_command(self, arguments: str) -> str:
        """List available files, optionally filtered by type."""
        try:
            file_type = arguments.strip().lower() if arguments else None
            
            if 'available_files' not in self.document_reader.context:
                return "No files discovered yet. Use /discover [directory] first."
            
            result = "## Available Files\n\n"
            
            if file_type:
                if file_type not in self.document_reader.context['available_files']:
                    return f"No files of type '{file_type}' found."
                
                files = self.document_reader.context['available_files'][file_type]
                if not files:
                    return f"No {file_type} files found."
                
                result += f"### {file_type.upper()} Files ({len(files)})\n"
                for file in files:
                    result += f"- {file}\n"
            else:
                for ext, files in self.document_reader.context['available_files'].items():
                    if files:
                        result += f"### {ext.upper()} Files ({len(files)})\n"
                        for file in files:
                            result += f"- {file}\n"
                        result += "\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error listing files: {str(e)}"
    
    def read_command(self, arguments: str) -> str:
        """Read and display the content of a file."""
        try:
            file_name = arguments.strip()
            if not file_name:
                return "Usage: /read [filename]"
            
            # Get the full path of the file
            file_path = self.document_reader.file_manager.get_file_path(file_name)
            
            # Read the file
            file_data = self.document_reader.file_manager.read_file(file_path)
            
            # Process based on file type
            if file_data['type'] in ['md', 'txt']:
                content = file_data['content']
                return f"## {file_data['name']}\n\n{content}"
            
            elif file_data['type'] == 'csv':
                df = file_data['content']
                preview = df.head(10).to_string()
                return f"## {file_data['name']} (CSV)\n\nShowing first 10 rows (out of {len(df)}):\n\n```\n{preview}\n```"
            
            elif file_data['type'] in ['png', 'jpg', 'jpeg', 'gif']:
                metadata = file_data['metadata']
                image_info = f"Image: {file_data['name']} ({metadata['width']}x{metadata['height']} pixels, {metadata['format']})"
                
                # Update pending action to analyze this image
                self.document_reader.context['pending_action'] = {
                    'type': 'analyze_image',
                    'file_path': file_path
                }
                
                return f"{image_info}\n\nWould you like me to analyze this image for you?"
            
            else:
                return f"Unsupported file type: {file_data['type']}"
            
        except FileNotFoundError:
            return f"File not found: {arguments}. Use /list to see available files."
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error reading file: {str(e)}"
    
    def analyze_command(self, arguments: str) -> str:
        """Analyze a file and provide insights."""
        try:
            file_name = arguments.strip()
            if not file_name:
                return "Usage: /analyze [filename]"
            
            # Get the full path of the file
            file_path = self.document_reader.file_manager.get_file_path(file_name)
            
            # Check file extension
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            if ext in ['.png', '.jpg', '.jpeg', '.gif']:
                # Update pending action to analyze this image
                self.document_reader.context['pending_action'] = {
                    'type': 'analyze_image',
                    'file_path': file_path
                }
                
                return f"I'll analyze the image {os.path.basename(file_path)} for you. One moment..."
            
            elif ext == '.csv':
                # Update pending action to analyze this CSV
                self.document_reader.context['pending_action'] = {
                    'type': 'analyze_csv',
                    'file_path': file_path
                }
                
                return f"I'll analyze the CSV data in {os.path.basename(file_path)} for you. One moment..."
            
            elif ext in ['.md', '.txt']:
                # Update pending action to analyze this text file
                self.document_reader.context['pending_action'] = {
                    'type': 'analyze_text',
                    'file_path': file_path
                }
                
                return f"I'll analyze the content of {os.path.basename(file_path)} for you. One moment..."
            
            else:
                return f"Unsupported file type for analysis: {ext}"
            
        except FileNotFoundError:
            return f"File not found: {arguments}. Use /list to see available files."
        except Exception as e:
            logger.error(f"Error setting up analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error setting up analysis: {str(e)}"
    
    def visualize_command(self, arguments: str) -> str:
        """Create a visualization from CSV data."""
        try:
            # Parse arguments
            args = arguments.split()
            
            if not args:
                return "Usage: /visualize [filename] [x_column] [y_column] [type]"
            
            file_name = args[0]
            
            # Get the full path of the file
            try:
                file_path = self.document_reader.file_manager.get_file_path(file_name)
            except FileNotFoundError:
                return f"File not found: {file_name}. Use /list to see available files."
            
            # Check if it's a CSV file
            if not file_path.lower().endswith('.csv'):
                return f"Visualization is only supported for CSV files. '{file_name}' is not a CSV file."
            
            # Read the CSV to get column names
            df = pd.read_csv(file_path)
            
            # Parse additional arguments
            x_column = None
            y_column = None
            plot_type = 'auto'
            
            if len(args) >= 2:
                x_column = args[1]
                if x_column not in df.columns:
                    return f"Column '{x_column}' not found in the CSV file. Available columns: {', '.join(df.columns)}"
            
            if len(args) >= 3:
                y_column = args[2]
                if y_column not in df.columns:
                    return f"Column '{y_column}' not found in the CSV file. Available columns: {', '.join(df.columns)}"
            
            if len(args) >= 4:
                plot_type = args[3].lower()
            
            # If x_column is not specified, ask for more information
            if not x_column:
                column_list = '\n'.join([f"- {col}" for col in df.columns])
                return f"Please specify at least one column to visualize.\n\nAvailable columns in {file_name}:\n{column_list}\n\nExample: /visualize {file_name} [column_name]"
            
            # Create the visualization
            try:
                image_base64 = self.document_reader.file_manager.visualize_data(
                    file_path, x_column, y_column, plot_type
                )
                
                # Update context with the visualization
                plot_info = {
                    'type': plot_type,
                    'x_column': x_column,
                    'y_column': y_column,
                    'source_file': os.path.basename(file_path)
                }
                
                # Return success message
                viz_type = plot_type if plot_type != 'auto' else 'appropriate'
                columns_desc = f"{x_column}" if not y_column else f"{y_column} vs {x_column}"
                
                return f"Created a {viz_type} visualization of {columns_desc} from {os.path.basename(file_path)}."
                
            except Exception as e:
                logger.error(f"Error creating visualization: {str(e)}")
                logger.error(traceback.format_exc())
                return f"Error creating visualization: {str(e)}"
            
        except Exception as e:
            logger.error(f"Error processing visualization command: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error processing visualization command: {str(e)}"
    
    def summary_command(self, arguments: str) -> str:
        """Show a summary of available data sources."""
        try:
            if 'available_files' not in self.document_reader.context:
                return "No files discovered yet. Use /discover [directory] first."
            
            # Count files by type
            file_counts = {ext: len(files) for ext, files in self.document_reader.context['available_files'].items() if files}
            
            # Prepare summary
            summary = "## Data Source Summary\n\n"
            
            # File counts
            summary += "### File Counts\n"
            for ext, count in file_counts.items():
                summary += f"- {ext.upper()} Files: {count}\n"
            
            # Notable files
            summary += "\n### Notable Files\n"
            
            # For CSV files, show data dimensions
            csv_files = self.document_reader.context['available_files'].get('csv', [])
            if csv_files:
                summary += "#### CSV Data\n"
                for file_path in csv_files[:5]:  # Show first 5 CSV files
                    full_path = os.path.join(self.document_reader.file_manager.base_directory, file_path)
                    try:
                        df = pd.read_csv(full_path)
                        summary += f"- {file_path}: {df.shape[0]} rows, {df.shape[1]} columns\n"
                    except Exception as e:
                        summary += f"- {file_path}: Error reading file\n"
                
                if len(csv_files) > 5:
                    summary += f"- ... and {len(csv_files) - 5} more CSV files\n"
            
            # For images, show a count
            image_files = []
            for img_type in ['png', 'jpg', 'jpeg', 'gif']:
                image_files.extend(self.document_reader.context['available_files'].get(img_type, []))
            
            if image_files:
                summary += "\n#### Images\n"
                summary += f"Found {len(image_files)} image files.\n"
                
                # Show a sample of images (max 5)
                for i, img_path in enumerate(image_files[:5]):
                    summary += f"- {img_path}\n"
                
                if len(image_files) > 5:
                    summary += f"- ... and {len(image_files) - 5} more images\n"
            
            # For text/markdown files, show a count and list
            text_files = []
            for txt_type in ['txt', 'md']:
                text_files.extend(self.document_reader.context['available_files'].get(txt_type, []))
            
            if text_files:
                summary += "\n#### Text Documents\n"
                summary += f"Found {len(text_files)} text/markdown files.\n"
                
                # Show a sample of text files (max 5)
                for i, txt_path in enumerate(text_files[:5]):
                    summary += f"- {txt_path}\n"
                
                if len(text_files) > 5:
                    summary += f"- ... and {len(text_files) - 5} more text files\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error generating summary: {str(e)}"
    
    def search_command(self, arguments: str) -> str:
        """Search for files containing the specified term."""
        try:
            search_term = arguments.strip()
            if not search_term:
                return "Usage: /search [term]"
            
            if 'available_files' not in self.document_reader.context:
                return "No files discovered yet. Use /discover [directory] first."
            
            # Find matching files
            matching_files = self.document_reader.file_manager.find_matching_files(search_term)
            
            if not matching_files:
                return f"No files found matching '{search_term}'."
            
            # Group results by file type
            results_by_type = {}
            for file_info in matching_files:
                file_type = file_info['type']
                if file_type not in results_by_type:
                    results_by_type[file_type] = []
                results_by_type[file_type].append(file_info)
            
            # Format the results
            result = f"## Search Results for '{search_term}'\n\n"
            result += f"Found {len(matching_files)} matching files:\n\n"
            
            for file_type, files in results_by_type.items():
                result += f"### {file_type.upper()} Files ({len(files)})\n"
                for file_info in files:
                    result += f"- {file_info['path']}\n"
                result += "\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching files: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error searching files: {str(e)}"
    
    def image_command(self, arguments: str) -> str:
        """Display and analyze an image."""
        try:
            image_name = arguments.strip()
            if not image_name:
                # No specific image, check for available images
                image_files = self.document_reader.file_manager.find_images()
                
                if not image_files:
                    return "No image files found. Use /discover [directory] first to find images."
                
                if len(image_files) == 1:
                    # Only one image, use it
                    image_path = os.path.join(self.document_reader.file_manager.base_directory, image_files[0])
                    
                    # Update pending action to analyze this image
                    self.document_reader.context['pending_action'] = {
                        'type': 'analyze_image',
                        'file_path': image_path
                    }
                    
                    return f"I'll analyze the image {os.path.basename(image_path)} for you. One moment..."
                
                else:
                    # Multiple images, list them
                    result = "Please specify which image to analyze:\n\n"
                    for i, img_path in enumerate(image_files[:10]):
                        result += f"{i+1}. {img_path}\n"
                    
                    if len(image_files) > 10:
                        result += f"... and {len(image_files) - 10} more images\n"
                    
                    result += "\nUse /image [filename] to analyze a specific image."
                    return result
            
            # Specific image requested
            try:
                image_path = self.document_reader.file_manager.get_file_path(image_name)
                
                # Check if it's an image file
                file_type = self.document_reader.file_manager.get_file_type(image_path)
                if file_type not in ['png', 'jpg', 'jpeg', 'gif']:
                    return f"The file '{image_name}' is not an image file."
                
                # Update pending action to analyze this image
                self.document_reader.context['pending_action'] = {
                    'type': 'analyze_image',
                    'file_path': image_path
                }
                
                return f"I'll analyze the image {os.path.basename(image_path)} for you. One moment..."
                
            except FileNotFoundError:
                return f"Image not found: {image_name}. Use /list to see available files."
            
        except Exception as e:
            logger.error(f"Error processing image command: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error processing image command: {str(e)}"
    
    def csv_command(self, arguments: str) -> str:
        """Query a CSV file."""
        try:
            args = arguments.strip().split(maxsplit=1)
            
            if not args:
                return "Usage: /csv [filename] [query]"
            
            file_name = args[0]
            query = args[1] if len(args) > 1 else None
            
            # Get the full path of the file
            try:
                file_path = self.document_reader.file_manager.get_file_path(file_name)
            except FileNotFoundError:
                return f"File not found: {file_name}. Use /list to see available files."
            
            # Check if it's a CSV file
            if not file_path.lower().endswith('.csv'):
                return f"'{file_name}' is not a CSV file."
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            if not query:
                # No query provided, show basic info
                info = f"## {os.path.basename(file_path)}\n\n"
                info += f"This CSV file has {df.shape[0]} rows and {df.shape[1]} columns.\n\n"
                info += "### Columns\n"
                
                # Show column names and data types
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    info += f"- **{col}** ({dtype})\n"
                
                info += "\n### Sample Data\n"
                info += "```\n" + df.head(5).to_string() + "\n```"
                
                return info
            
            # Process the query
            query_lower = query.lower()
            
            if "show" in query_lower and "first" in query_lower:
                # Extract number of rows to show
                match = re.search(r'first\s+(\d+)', query_lower)
                if match:
                    n_rows = int(match.group(1))
                    result = f"First {n_rows} rows of {os.path.basename(file_path)}:\n\n"
                    result += "```\n" + df.head(n_rows).to_string() + "\n```"
                    return result
                else:
                    result = f"First 5 rows of {os.path.basename(file_path)}:\n\n"
                    result += "```\n" + df.head(5).to_string() + "\n```"
                    return result
            
            elif "show" in query_lower and "last" in query_lower:
                # Extract number of rows to show
                match = re.search(r'last\s+(\d+)', query_lower)
                if match:
                    n_rows = int(match.group(1))
                    result = f"Last {n_rows} rows of {os.path.basename(file_path)}:\n\n"
                    result += "```\n" + df.tail(n_rows).to_string() + "\n```"
                    return result
                else:
                    result = f"Last 5 rows of {os.path.basename(file_path)}:\n\n"
                    result += "```\n" + df.tail(5).to_string() + "\n```"
                    return result
            
            elif "describe" in query_lower or "statistics" in query_lower or "stats" in query_lower:
                # Generate descriptive statistics
                result = f"Descriptive statistics for {os.path.basename(file_path)}:\n\n"
                result += "```\n" + df.describe().to_string() + "\n```"
                return result
            
            elif "count" in query_lower or "null" in query_lower or "missing" in query_lower:
                # Count null/missing values
                null_counts = df.isnull().sum()
                result = f"Null value counts for {os.path.basename(file_path)}:\n\n"
                result += "```\n" + null_counts.to_string() + "\n```"
                return result
            
            elif "filter" in query_lower or "where" in query_lower or "condition" in query_lower:
                # Extract the filter condition
                filter_text = query.split("filter", 1)[1].strip() if "filter" in query_lower else \
                            query.split("where", 1)[1].strip() if "where" in query_lower else \
                            query.split("condition", 1)[1].strip()
                
                # Update pending action to filter CSV data
                self.document_reader.context['pending_action'] = {
                    'type': 'filter_csv',
                    'file_path': file_path,
                    'filter_text': filter_text
                }
                
                return f"I'll filter the data in {os.path.basename(file_path)} using the condition: {filter_text}"
            
            else:
                # For more complex queries, use the agent to interpret
                self.document_reader.context['pending_action'] = {
                    'type': 'query_csv',
                    'file_path': file_path,
                    'query': query
                }
                
                return f"I'll analyze '{query}' on the data in {os.path.basename(file_path)}. One moment..."
            
        except Exception as e:
            logger.error(f"Error processing CSV command: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error processing CSV command: {str(e)}"
    
    def discover_command(self, arguments: str) -> str:
        """Auto-discover files in a directory."""
        try:
            directory = arguments.strip()
            if not directory:
                # Check if we have a base directory set
                if self.document_reader.file_manager.base_directory:
                    directory = self.document_reader.file_manager.base_directory
                else:
                    return "Usage: /discover [directory]"
            
            # Set the base directory
            self.document_reader.file_manager.set_base_directory(directory)
            
            # Discover files
            files = self.document_reader.file_manager.discover_files()
            
            # Count files by type
            file_counts = {ext: len(file_list) for ext, file_list in files.items() if file_list}
            total_files = sum(len(file_list) for file_list in files.values())
            
            # Create summary
            result = f"## File Discovery Results\n\n"
            result += f"Discovered {total_files} files in '{directory}':\n\n"
            
            for ext, count in file_counts.items():
                result += f"- {ext.upper()} Files: {count}\n"
            
            if total_files > 0:
                result += "\nUse /list to see all files or /list [type] to see files of a specific type."
            else:
                result += "\nNo files found with supported extensions."
            
            return result
            
        except Exception as e:
            logger.error(f"Error discovering files: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error discovering files: {str(e)}"
    
    def clear_command(self, arguments: str) -> str:
        """Clear the conversation history."""
        try:
            # Clear messages but keep context
            self.document_reader.messages = []
            
            return "Conversation history cleared."
            
        except Exception as e:
            logger.error(f"Error clearing history: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error clearing history: {str(e)}"


class ResponseHandler:
    """Handles formatting and validation of responses for the document reader."""
    
    def __init__(self, document_reader):
        """Initialize the response handler with a reference to the document reader."""
        self.document_reader = document_reader
    
    def clean_response(self, response):
        """Clean and format the response."""
        try:
            # Remove any leading/trailing whitespace
            response = response.strip()
            
            # Check if the response is empty
            if not response:
                return "I'm sorry, I wasn't able to generate a response."
            
            # Remove apology patterns
            apology_patterns = [
                r"I apologize, but I don't have enough information.*",
                r"I'm sorry, I don't have access to.*",
                r"I apologize, but I cannot.*",
                r"I'm sorry, but I'm not able to.*",
                r"As an AI language model, I don't have the ability to.*",
                r"As an AI assistant, I don't have the capability to.*"
            ]
            
            for pattern in apology_patterns:
                response = re.sub(pattern, "", response, flags=re.IGNORECASE)
            
            # Remove AI self-references
            ai_references = [
                r"As an AI language model,",
                r"As an AI assistant,",
                r"As a language model,"
            ]
            
            for reference in ai_references:
                response = response.replace(reference, "")
            
            # Clean up multiple newlines
            response = re.sub(r'\n{3,}', '\n\n', response)
            
            # Make sure the response isn't empty after cleaning
            response = response.strip()
            if not response:
                return "I'm analyzing your request."
            
            return response
            
        except Exception as e:
            logger.error(f"Error cleaning response: {str(e)}")
            logger.error(traceback.format_exc())
            return response  # Return the original response if cleaning fails
    
    def validate_response(self, response):
        """Validate the response for quality and completeness."""
        try:
            # Check response length
            if len(response) < 10:
                logger.warning(f"Response too short: {response}")
                return False, "Response too short"
            
            # Check for incomplete sentences (ending without proper punctuation)
            last_sentence = response.strip().split('.')[-1].strip()
            if last_sentence and len(last_sentence) > 30 and not any(last_sentence.endswith(p) for p in ['.', '!', '?', ':', ';']):
                logger.warning(f"Response may be cut off: {last_sentence}")
                return False, "Response may be cut off"
            
            # Check for common error patterns
            error_patterns = [
                r"Error processing",
                r"I encountered an error",
                r"An error occurred",
                r"Unable to process",
                r"Failed to"
            ]
            
            for pattern in error_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    error_context = re.search(f".{{0,50}}{pattern}.{{0,50}}", response, re.IGNORECASE)
                    logger.warning(f"Response contains error message: {error_context.group(0) if error_context else pattern}")
                    # Don't return false here, as we want to show the user the error
            
            # Response looks good
            return True, "Valid response"
            
        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            logger.error(traceback.format_exc())
            return False, f"Error validating response: {str(e)}" 
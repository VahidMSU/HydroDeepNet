import os
import re
import json
import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict

from Logger import LoggerSetup
# Initialize logger using setup_logger method
logger_setup = LoggerSetup()
logger = logger_setup.setup_logger()

class QuestionHandler:
    """Handles natural language questions and maintains a knowledge graph for the document reader."""
    
    def __init__(self, document_reader):
        """Initialize the question handler with reference to the document reader."""
        self.document_reader = document_reader
        self.knowledge_graph = {}
        self.question_patterns = self._compile_question_patterns()
        self.recent_questions = []  # Keep track of recent questions
        self.question_context = {}  # Context for follow-up questions
    
    def _compile_question_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for common question types."""
        patterns = {
            'file_content': re.compile(r'what(?:.+?)(?:in|inside|content of)(?:.+?)(?P<file_name>[a-zA-Z0-9_\-\.]+)', re.IGNORECASE),
            'file_list': re.compile(r'what(?:.+?)(?:files|documents)(?:.+?)(?:available|have|are there)', re.IGNORECASE),
            'csv_stats': re.compile(r'(?:stats|statistics|summary)(?:.+?)(?P<file_name>[a-zA-Z0-9_\-\.]+\.csv)', re.IGNORECASE),
            'visualization': re.compile(r'(?:visualize|plot|chart|graph)(?:.+?)(?P<file_name>[a-zA-Z0-9_\-\.]+\.csv)', re.IGNORECASE),
            'image_analysis': re.compile(r'(?:analyze|explain|describe)(?:.+?)(?:image|picture|figure|chart)(?:.+?)(?P<file_name>[a-zA-Z0-9_\-\.]+\.(png|jpg|jpeg|gif))', re.IGNORECASE),
            'data_question': re.compile(r'(?:what|how|find|calculate|compute)(?:.+?)(?P<file_name>[a-zA-Z0-9_\-\.]+\.csv)', re.IGNORECASE),
        }
        return patterns
    
    def process_question(self, message: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Process a natural language question.
        Returns a tuple: (is_question, pending_action_or_none)
        """
        try:
            # Add to recent questions for context
            self.recent_questions.append(message)
            if len(self.recent_questions) > 5:
                self.recent_questions.pop(0)
            
            # Check if it's a follow-up question
            if self._is_followup_question(message):
                return self._handle_followup_question(message)
            
            # Check for different question patterns
            for question_type, pattern in self.question_patterns.items():
                match = pattern.search(message)
                if match:
                    if question_type == 'file_content':
                        file_name = match.group('file_name')
                        return self._handle_file_content_question(file_name)
                    
                    elif question_type == 'file_list':
                        return self._handle_file_list_question()
                    
                    elif question_type == 'csv_stats':
                        file_name = match.group('file_name')
                        return self._handle_csv_stats_question(file_name)
                    
                    elif question_type == 'visualization':
                        file_name = match.group('file_name')
                        return self._handle_visualization_question(file_name, message)
                    
                    elif question_type == 'image_analysis':
                        file_name = match.group('file_name')
                        return self._handle_image_analysis_question(file_name)
                    
                    elif question_type == 'data_question':
                        file_name = match.group('file_name')
                        return self._handle_data_question(file_name, message)
            
            # General data-related question - send to agent
            if any(term in message.lower() for term in ['data', 'csv', 'spreadsheet', 'rows', 'columns']):
                return True, {
                    'type': 'general_data_question',
                    'message': message
                }
            
            # Image-related question
            if any(term in message.lower() for term in ['image', 'picture', 'figure', 'chart', 'graph', 'plot']):
                return self._handle_general_image_question(message)
            
            # If we couldn't identify a specific question type
            return False, None
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None
    
    def _is_followup_question(self, message: str) -> bool:
        """Check if a message is a follow-up question based on pronouns and context."""
        # Check for pronouns or context indicators
        followup_indicators = [
            r'\bit\b',  # "it"
            r'\bthat\b',  # "that"
            r'\bthis\b',  # "this"
            r'\bthose\b',  # "those"
            r'\bthey\b',  # "they"
            r'\bthe\b',  # "the" followed by known context
            r'\bthe (file|data|image|chart|plot|graph|table|values|results)\b',
            r'^(what|how|why|when|where|who|which)\b',  # Questions starting with wh-words without context
            r'^(can|could) (you|I)\b',  # Questions starting with "can you" or "could you"
            r'^(show|tell|give) me\b',  # Directives without clear context
        ]
        
        # If there are no recent questions, this can't be a follow-up
        if not self.recent_questions or len(self.recent_questions) <= 1:
            return False
        
        # Check if this message matches any of the follow-up patterns
        return any(re.search(pattern, message.lower()) for pattern in followup_indicators)
    
    def _handle_followup_question(self, message: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Handle a follow-up question using context from previous questions."""
        # Get the last non-followup question context
        if not self.question_context:
            return False, None
        
        last_context = self.question_context.get('last_context')
        if not last_context:
            return False, None
        
        context_type = last_context.get('type')
        
        if context_type == 'file_content':
            file_name = last_context.get('file_name')
            return self._handle_file_content_question(file_name)
        
        elif context_type == 'csv_analysis':
            file_name = last_context.get('file_name')
            return True, {
                'type': 'query_csv',
                'file_path': file_name,
                'query': message
            }
        
        elif context_type == 'image_analysis':
            file_name = last_context.get('file_name')
            return self._handle_image_analysis_question(file_name)
        
        # If we can't determine the context, treat as a new question
        return False, None
    
    def _handle_file_content_question(self, file_name: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Handle a question about file content."""
        try:
            # Save context for follow-up questions
            self.question_context['last_context'] = {
                'type': 'file_content',
                'file_name': file_name
            }
            
            # Try to find the file
            try:
                file_path = self.document_reader.file_manager.get_file_path(file_name)
            except FileNotFoundError:
                return False, None
            
            # Update pending action based on file type
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            if ext in ['.csv']:
                return True, {
                    'type': 'analyze_csv',
                    'file_path': file_path
                }
            
            elif ext in ['.md', '.txt']:
                return True, {
                    'type': 'analyze_text',
                    'file_path': file_path
                }
            
            elif ext in ['.png', '.jpg', '.jpeg', '.gif']:
                return True, {
                    'type': 'analyze_image',
                    'file_path': file_path
                }
            
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"Error handling file content question: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None
    
    def _handle_file_list_question(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Handle a question about available files."""
        try:
            # No file information needed for listing
            self.question_context['last_context'] = {
                'type': 'file_list'
            }
            
            # Check if we have discovered files
            if 'available_files' not in self.document_reader.context:
                return False, None
            
            # Just return True to indicate we recognized this as a question
            # The agent will handle displaying the file list
            return True, {
                'type': 'list_files'
            }
                
        except Exception as e:
            logger.error(f"Error handling file list question: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None
    
    def _handle_csv_stats_question(self, file_name: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Handle a question about CSV statistics."""
        try:
            # Save context for follow-up questions
            self.question_context['last_context'] = {
                'type': 'csv_analysis',
                'file_name': file_name
            }
            
            # Try to find the file
            try:
                file_path = self.document_reader.file_manager.get_file_path(file_name)
            except FileNotFoundError:
                return False, None
            
            # Update pending action
            return True, {
                'type': 'analyze_csv',
                'file_path': file_path,
                'analysis_type': 'statistics'
            }
                
        except Exception as e:
            logger.error(f"Error handling CSV stats question: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None
    
    def _handle_visualization_question(self, file_name: str, message: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Handle a question about creating visualizations."""
        try:
            # Try to find the file
            try:
                file_path = self.document_reader.file_manager.get_file_path(file_name)
            except FileNotFoundError:
                return False, None
            
            # Check if it's a CSV file
            if not file_path.lower().endswith('.csv'):
                return False, None
            
            # Extract information about columns to visualize
            df = pd.read_csv(file_path)
            columns = list(df.columns)
            
            # Try to find columns mentioned in the message
            mentioned_columns = []
            for col in columns:
                if col.lower() in message.lower():
                    mentioned_columns.append(col)
            
            # Extract visualization type
            viz_type = 'auto'
            viz_types = {
                'bar': ['bar', 'histogram'],
                'line': ['line', 'trend'],
                'scatter': ['scatter', 'point'],
                'pie': ['pie'],
                'box': ['box', 'boxplot'],
                'heatmap': ['heat', 'heatmap', 'correlation']
            }
            
            for vtype, keywords in viz_types.items():
                if any(keyword in message.lower() for keyword in keywords):
                    viz_type = vtype
                    break
            
            # Save context for follow-up questions
            self.question_context['last_context'] = {
                'type': 'visualization',
                'file_name': file_path,
                'columns': mentioned_columns,
                'viz_type': viz_type
            }
            
            # Update pending action
            return True, {
                'type': 'create_visualization',
                'file_path': file_path,
                'columns': mentioned_columns,
                'viz_type': viz_type,
                'message': message
            }
                
        except Exception as e:
            logger.error(f"Error handling visualization question: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None
    
    def _handle_image_analysis_question(self, file_name: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Handle a question about analyzing an image."""
        try:
            # Save context for follow-up questions
            self.question_context['last_context'] = {
                'type': 'image_analysis',
                'file_name': file_name
            }
            
            # Try to find the file
            try:
                file_path = self.document_reader.file_manager.get_file_path(file_name)
            except FileNotFoundError:
                return False, None
            
            # Check if it's an image file
            file_type = self.document_reader.file_manager.get_file_type(file_path)
            if file_type not in ['png', 'jpg', 'jpeg', 'gif']:
                return False, None
            
            # Update pending action
            return True, {
                'type': 'analyze_image',
                'file_path': file_path
            }
                
        except Exception as e:
            logger.error(f"Error handling image analysis question: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None
    
    def _handle_data_question(self, file_name: str, message: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Handle a question about data in a CSV file."""
        try:
            # Try to find the file
            try:
                file_path = self.document_reader.file_manager.get_file_path(file_name)
            except FileNotFoundError:
                return False, None
            
            # Save context for follow-up questions
            self.question_context['last_context'] = {
                'type': 'csv_analysis',
                'file_name': file_path
            }
            
            # Update pending action
            return True, {
                'type': 'query_csv',
                'file_path': file_path,
                'query': message
            }
                
        except Exception as e:
            logger.error(f"Error handling data question: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None
    
    def _handle_general_image_question(self, message: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Handle a general question about images when no specific image is mentioned."""
        try:
            # Check if we have images available
            if 'available_files' not in self.document_reader.context:
                return False, None
            
            # Find all image files
            image_files = self.document_reader.file_manager.find_images()
            
            if not image_files:
                return False, None
            
            if len(image_files) == 1:
                # Only one image available, use it
                image_path = os.path.join(self.document_reader.file_manager.base_directory, image_files[0])
                
                # Save context for follow-up questions
                self.question_context['last_context'] = {
                    'type': 'image_analysis',
                    'file_name': image_path
                }
                
                # Update pending action
                return True, {
                    'type': 'analyze_image',
                    'file_path': image_path,
                    'query': message
                }
            
            else:
                # Multiple images, need to list them
                return True, {
                    'type': 'list_images',
                    'message': message
                }
                
        except Exception as e:
            logger.error(f"Error handling general image question: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None
    
    def update_knowledge_graph(self, file_data: Dict[str, Any]) -> None:
        """Update the knowledge graph with information from a file."""
        try:
            file_type = file_data.get('type')
            file_name = file_data.get('name')
            file_path = file_data.get('path')
            
            if not file_type or not file_name or not file_path:
                return
            
            # Initialize if needed
            if file_name not in self.knowledge_graph:
                self.knowledge_graph[file_name] = {
                    'type': file_type,
                    'path': file_path,
                    'metadata': {},
                    'content_summary': {}
                }
            
            # Update metadata
            if 'metadata' in file_data:
                self.knowledge_graph[file_name]['metadata'].update(file_data['metadata'])
            
            # Update content summary based on file type
            if file_type == 'csv':
                self._update_csv_knowledge(file_name, file_data)
            elif file_type in ['png', 'jpg', 'jpeg', 'gif']:
                self._update_image_knowledge(file_name, file_data)
            elif file_type in ['md', 'txt']:
                self._update_text_knowledge(file_name, file_data)
            
        except Exception as e:
            logger.error(f"Error updating knowledge graph: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _update_csv_knowledge(self, file_name: str, file_data: Dict[str, Any]) -> None:
        """Update knowledge graph with CSV file information."""
        try:
            if 'content' not in file_data:
                return
            
            df = file_data['content']
            if not isinstance(df, pd.DataFrame):
                return
            
            # Get basic statistics
            stats = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'column_types': {col: str(df[col].dtype) for col in df.columns},
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_columns': [col for col, dtype in df.dtypes.items() if pd.api.types.is_numeric_dtype(dtype)]
            }
            
            # For numeric columns, get more statistics
            if stats['numeric_columns']:
                stats['numeric_stats'] = {}
                for col in stats['numeric_columns']:
                    col_stats = df[col].describe().to_dict()
                    # Convert numpy types to native Python types for JSON serialization
                    col_stats = {k: float(v) if isinstance(v, (np.int64, np.float64)) else v 
                               for k, v in col_stats.items()}
                    stats['numeric_stats'][col] = col_stats
            
            # Update the knowledge graph
            self.knowledge_graph[file_name]['content_summary'].update(stats)
            
        except Exception as e:
            logger.error(f"Error updating CSV knowledge: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _update_image_knowledge(self, file_name: str, file_data: Dict[str, Any]) -> None:
        """Update knowledge graph with image file information."""
        try:
            if 'metadata' not in file_data:
                return
            
            metadata = file_data['metadata']
            
            # Update the knowledge graph
            self.knowledge_graph[file_name]['content_summary'] = {
                'width': metadata.get('width'),
                'height': metadata.get('height'),
                'format': metadata.get('format'),
                'mode': metadata.get('mode')
            }
            
            # Add analysis results if available
            if 'analysis' in file_data:
                self.knowledge_graph[file_name]['analysis'] = file_data['analysis']
            
        except Exception as e:
            logger.error(f"Error updating image knowledge: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _update_text_knowledge(self, file_name: str, file_data: Dict[str, Any]) -> None:
        """Update knowledge graph with text file information."""
        try:
            if 'content' not in file_data:
                return
            
            content = file_data['content']
            
            # Get basic statistics
            lines = content.split('\n')
            words = content.split()
            chars = len(content)
            
            # Check for markdown headers
            headers = []
            if file_data['type'] == 'md':
                header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
                headers = [m.group(2) for m in header_pattern.finditer(content)]
            
            # Update the knowledge graph
            self.knowledge_graph[file_name]['content_summary'] = {
                'lines': len(lines),
                'words': len(words),
                'chars': chars,
                'headers': headers
            }
            
        except Exception as e:
            logger.error(f"Error updating text knowledge: {str(e)}")
            logger.error(traceback.format_exc())
    
    def get_knowledge_for_entity(self, entity_name: str) -> Dict[str, Any]:
        """Get all knowledge about a specific entity."""
        return self.knowledge_graph.get(entity_name, {})
    
    def get_all_knowledge(self) -> Dict[str, Any]:
        """Get the entire knowledge graph."""
        return self.knowledge_graph
    
    def get_entities_by_type(self, entity_type: str) -> List[str]:
        """Get all entities of a specific type."""
        return [name for name, data in self.knowledge_graph.items() 
                if data.get('type') == entity_type]
    
    def search_knowledge_graph(self, search_term: str) -> Dict[str, Any]:
        """Search the knowledge graph for entities matching the search term."""
        results = {}
        search_term = search_term.lower()
        
        for name, data in self.knowledge_graph.items():
            # Search in entity name
            if search_term in name.lower():
                results[name] = data
                continue
            
            # Search in metadata and content summary
            if any(search_term in str(v).lower() for v in data.get('metadata', {}).values()):
                results[name] = data
                continue
                
            if any(search_term in str(v).lower() for v in data.get('content_summary', {}).values()):
                results[name] = data
                continue
        
        return results 
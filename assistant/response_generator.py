from typing import Dict, List, Any, Optional
import os
import json
import re
from Logger import LoggerSetup

class ResponseGenerator:
    """
    Component responsible for generating responses to user queries
    based on query understanding and relevant file information.
    """
    
    def __init__(self, llm_service=None):
        """
        Initialize the response generator
        
        Args:
            llm_service: Optional service for LLM-based response generation
        """
        self.logger = LoggerSetup(rewrite=False, verbose=False)
        self.logger.info("Initializing Response Generator")
        self.llm_service = llm_service
        
    def generate_response(self, 
                          query: str, 
                          query_info: Dict[str, Any],
                          relevant_files: List[str],
                          memory_system) -> Dict[str, Any]:
        """
        Generate a response to the user query
        
        Args:
            query (str): The original user query
            query_info (dict): Information extracted from the query
            relevant_files (list): Paths to relevant files
            memory_system: Memory system for accessing file content
            
        Returns:
            dict: Response data with answer and file references
        """
        self.logger.debug(f"Generating response for query: {query}")
        
        # Check if we have any relevant files first
        if not relevant_files:
            # If no relevant files were found, try to find them directly from query
            self.logger.info("No relevant files provided, attempting to extract filenames from query")
            extracted_files = self._extract_file_references(query, memory_system)
            if extracted_files:
                self.logger.info(f"Found potentially referenced files in query: {extracted_files}")
                relevant_files = extracted_files
        
        # Get file content for relevant files
        file_contents = self._get_file_contents(relevant_files, memory_system)
        
        # Determine response strategy based on query intent
        intent = query_info.get("intent", "general")
        
        if self.llm_service:
            # If we have an LLM service, use it for response generation
            return self._generate_llm_response(query, query_info, file_contents)
        else:
            # Otherwise use rule-based response generation
            return self._generate_rule_based_response(query, query_info, file_contents)
    
    def _extract_file_references(self, query: str, memory_system) -> List[str]:
        """
        Extract potential file references from the query
        
        Args:
            query (str): The query to analyze
            memory_system: Memory system for looking up files
            
        Returns:
            list: List of file paths that may be referenced in the query
        """
        # Find patterns like "analyze [filename]" or "show [filename]"
        referenced_files = []
        
        # Common file extensions to look for
        extensions = ['.csv', '.png', '.jpg', '.jpeg', '.md', '.txt']
        
        # Look for filenames with extensions in the query
        for ext in extensions:
            pattern = rf'[\w\-_\s]+{ext}'
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                # Clean up the match
                clean_match = match.strip()
                
                # Try to find this file in memory
                file_records = memory_system.get_related_files(clean_match, [clean_match.split('.')[0]], limit=3)
                for record in file_records:
                    if 'original_path' in record:
                        referenced_files.append(record['original_path'])
        
        # Also look for specific filename references without extension
        words = query.split()
        for word in words:
            if len(word) > 3 and not word.lower() in ['show', 'analyze', 'find', 'about', 'what', 'where', 'when', 'file']:
                # Try to find this file in memory by partial name match
                file_records = memory_system.get_related_files(word, [word], limit=3)
                for record in file_records:
                    if 'original_path' in record:
                        referenced_files.append(record['original_path'])
        
        return list(set(referenced_files))  # Remove duplicates
    
    def _get_file_contents(self, file_paths: List[str], memory_system) -> Dict[str, Any]:
        """
        Get content and metadata for relevant files
        
        Args:
            file_paths: List of file paths
            memory_system: Memory system for accessing file content
            
        Returns:
            dict: Map of file paths to content and metadata
        """
        self.logger.debug(f"Getting file contents for {len(file_paths)} files")
        file_contents = {}
        
        for file_path in file_paths:
            # Get file record from memory
            file_record = self._get_file_record(file_path, memory_system)
            
            if file_record:
                self.logger.debug(f"Found file record for {file_path}")
                file_contents[file_path] = file_record
            else:
                self.logger.debug(f"No file record found for {file_path}, trying basename lookup")
                # If file not found by path, try looking by filename
                basename = os.path.basename(file_path)
                file_records = memory_system.get_related_files(basename, [basename.split('.')[0]], limit=1)
                if file_records:
                    self.logger.debug(f"Found file by basename: {basename}")
                    file_contents[file_path] = file_records[0]
                
        return file_contents
    
    def _get_file_record(self, file_path: str, memory_system) -> Optional[Dict[str, Any]]:
        """
        Get file record from memory system
        
        Args:
            file_path: Path to the file
            memory_system: Memory system instance
            
        Returns:
            dict or None: File record if found
        """
        # Method 1: Direct lookup in file_memory
        if hasattr(memory_system, "file_memory"):
            for path, record in memory_system.file_memory.items():
                if path == file_path:
                    return record
                
                # Also check basename matching if full path doesn't match
                if os.path.basename(path) == os.path.basename(file_path):
                    return record
        
        # Method 2: Use get_related_files method
        basename = os.path.basename(file_path)
        file_records = memory_system.get_related_files(basename, [basename.split('.')[0]], limit=1)
        if file_records and len(file_records) > 0:
            return file_records[0]
        
        # Method 3: For backward compatibility with HierarchicalMemory
        if hasattr(memory_system, "document_memory"):
            for doc_id, doc_data in memory_system.document_memory.items():
                if doc_id == file_path:
                    return doc_data
                
                # Also check basename matching
                if os.path.basename(doc_id) == os.path.basename(file_path):
                    return doc_data
        
        # Method 4: Search in files directory directly
        if hasattr(memory_system, "files_dir"):
            basename = os.path.basename(file_path)
            # Look for files with similar name
            for json_file_path in memory_system.files_dir.glob("*.json"):
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    file_record = json.load(f)
                    if file_record.get("file_name") == basename:
                        return file_record
        
        # File not found in memory
        self.logger.warning(f"File not found in memory: {file_path}")
        return None
    
    def _generate_rule_based_response(self, 
                                     query: str, 
                                     query_info: Dict[str, Any],
                                     file_contents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response using rule-based approach
        
        Args:
            query: User query
            query_info: Information about the query
            file_contents: Relevant file contents
            
        Returns:
            dict: Response data
        """
        intent = query_info.get("intent", "general")
        keywords = query_info.get("keywords", [])
        
        # Initialize response components
        answer_parts = []
        metadata = {
            "intent": intent,
            "relevant_files": list(file_contents.keys()),
            "keywords": keywords
        }
        
        # Generate response based on intent
        if intent == "search":
            answer = self._generate_search_response(query, query_info, file_contents)
        elif intent == "analyze":
            answer = self._generate_analysis_response(query, query_info, file_contents)
        elif intent == "geographic":
            answer = self._generate_geographic_response(query, query_info, file_contents)
        elif intent == "help":
            answer = self._generate_help_response(query, query_info)
        else:
            answer = self._generate_general_response(query, query_info, file_contents)
            
        return {
            "answer": answer,
            "relevant_files": list(file_contents.keys()),
            "metadata": metadata
        }
    
    def _generate_llm_response(self, 
                              query: str, 
                              query_info: Dict[str, Any],
                              file_contents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response using an LLM service
        
        Args:
            query: User query
            query_info: Information about the query
            file_contents: Relevant file contents
            
        Returns:
            dict: Response data
        """
        # This is a placeholder for LLM-based response generation
        # In a real implementation, this would format a prompt and call an LLM
        
        # For now, fall back to rule-based response
        return self._generate_rule_based_response(query, query_info, file_contents)
    
    def _generate_search_response(self, 
                                 query: str, 
                                 query_info: Dict[str, Any],
                                 file_contents: Dict[str, Any]) -> str:
        """Generate response for search intent"""
        if not file_contents:
            return "I couldn't find any relevant files that match your search criteria."
            
        # Format basic search results
        answer_parts = [f"I found {len(file_contents)} relevant files:"]
        
        # Add file information
        for i, (file_path, file_data) in enumerate(file_contents.items(), 1):
            file_name = os.path.basename(file_path)
            file_type = file_data.get("file_type", "document")
            
            # Extract a snippet from content if available
            content = file_data.get("content", "")
            snippet = content[:100] + "..." if content and len(content) > 100 else content
            
            answer_parts.append(f"{i}. {file_name} ({file_type}): {snippet}")
            
        return "\n".join(answer_parts)
    
    def _generate_analysis_response(self, 
                                   query: str, 
                                   query_info: Dict[str, Any],
                                   file_contents: Dict[str, Any]) -> str:
        """Generate response for analysis intent"""
        if not file_contents:
            return "I don't have enough information to perform this analysis. Please provide more details or specify which files to analyze."
            
        # For analysis, we'd typically use an LLM to generate insights
        # This is a simplified placeholder
        answer_parts = ["Based on my analysis of the relevant files:"]
        
        # For each file, extract key information based on file type
        for file_path, file_data in file_contents.items():
            file_name = os.path.basename(file_path)
            file_type = file_data.get("file_type", "unknown")
            
            if file_type == "csv":
                # For CSV files, include statistical information if available
                if "metadata" in file_data and "statistics" in file_data["metadata"]:
                    stats = file_data["metadata"]["statistics"]
                    answer_parts.append(f"- {file_name} contains numerical data with statistics: {stats}")
                else:
                    answer_parts.append(f"- {file_name} contains tabular data that may be relevant.")
            
            elif file_type in ["image", "png", "jpg", "jpeg"]:
                # For images, include content description
                content = file_data.get("content", "")
                if content:
                    answer_parts.append(f"- {file_name} shows: {content[:150]}...")
            
            else:
                # For other file types, include a general reference
                answer_parts.append(f"- {file_name} contains information that may help with your analysis.")
                
        return "\n".join(answer_parts)
    
    def _generate_geographic_response(self, 
                                     query: str, 
                                     query_info: Dict[str, Any],
                                     file_contents: Dict[str, Any]) -> str:
        """Generate response for geographic intent"""
        # Get geographic entities from query
        geo_entities = query_info.get("geographic_entities", [])
        
        if not geo_entities:
            return "I detected that you're asking about a geographic location, but I couldn't identify which specific location. Could you please clarify?"
            
        # Format the response with geographic information
        answer_parts = [f"Regarding the location(s): {', '.join(geo_entities)}"]
        
        if not file_contents:
            answer_parts.append("I don't have specific information about these locations in my knowledge base.")
        else:
            # Include information from relevant files
            answer_parts.append(f"I found information in {len(file_contents)} relevant files:")
            
            for file_path, file_data in file_contents.items():
                file_name = os.path.basename(file_path)
                # Extract geographic information if available
                if "geographic_info" in file_data:
                    geo_info = file_data["geographic_info"]
                    answer_parts.append(f"- {file_name} contains geographic information: {geo_info}")
                else:
                    # Look for mentions in general content
                    content = file_data.get("content", "")
                    if content:
                        # Look for mentions of the geographic entities
                        matches = []
                        for entity in geo_entities:
                            if entity.lower() in content.lower():
                                # Find context around the mention
                                idx = content.lower().find(entity.lower())
                                if idx >= 0:
                                    start = max(0, idx - 50)
                                    end = min(len(content), idx + len(entity) + 50)
                                    context = content[start:end].strip()
                                    matches.append(context)
                        
                        if matches:
                            answer_parts.append(f"- {file_name} mentions: {' ... '.join(matches[:2])}")
                        else:
                            answer_parts.append(f"- {file_name} may contain relevant geographic information.")
                            
        return "\n".join(answer_parts)
    
    def _generate_help_response(self, 
                               query: str, 
                               query_info: Dict[str, Any]) -> str:
        """Generate response for help intent"""
        # Provide a helpful overview of capabilities
        return (
            "I can help you with the following:\n"
            "1. Searching for specific files or information in the knowledge base\n"
            "2. Analyzing data from CSV files, images, and text documents\n"
            "3. Answering questions about geographic locations and their relationships\n"
            "4. Summarizing content from various file types\n\n"
            "Try asking me specific questions about the files, or request information about a particular topic or location."
        )
    
    def _generate_general_response(self, 
                                  query: str, 
                                  query_info: Dict[str, Any],
                                  file_contents: Dict[str, Any]) -> str:
        """Generate general response when intent is unclear"""
        if not file_contents:
            return (
                "I don't have specific information to answer your question. "
                "Try asking about specific files, topics, or locations in the knowledge base."
            )
            
        # Provide a general response with available information
        answer_parts = ["Here's what I found that might help answer your question:"]
        
        # Add information from relevant files
        for file_path, file_data in file_contents.items():
            file_name = os.path.basename(file_path)
            content = file_data.get("content", "")
            
            if content:
                # Look for relevant sections that might contain answers
                keywords = query_info.get("keywords", [])
                matches = []
                
                for keyword in keywords:
                    if keyword.lower() in content.lower():
                        # Find context around the keyword
                        idx = content.lower().find(keyword.lower())
                        if idx >= 0:
                            start = max(0, idx - 50)
                            end = min(len(content), idx + len(keyword) + 50)
                            context = content[start:end].strip()
                            matches.append(context)
                
                if matches:
                    answer_parts.append(f"- {file_name} contains: {' ... '.join(matches[:2])}")
                else:
                    answer_parts.append(f"- {file_name} may contain relevant information.")
                    
        return "\n".join(answer_parts) 
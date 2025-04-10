import re
import logging
import traceback
import json
from typing import Dict, Any, List, Optional, Tuple, Union

from Logger import LoggerSetup
# Initialize logger using setup_logger method
logger_setup = LoggerSetup()
logger = logger_setup.setup_logger()

class ResponseHandler:
    """Handles response processing and validation for document reader."""
    
    def __init__(self, document_reader):
        """Initialize the response handler with a reference to the document reader.
        
        Args:
            document_reader: The InteractiveDocumentReader instance
        """
        self.document_reader = document_reader
        self.response_cache = {}
    
    def clean_response(self, response: str) -> str:
        """Clean up agent responses to improve readability.
        
        Args:
            response: Raw response text from agent
            
        Returns:
            Cleaned response text
        """
        try:
            # If response is empty or None, return a default message
            if not response or response.strip() == "":
                return "I don't have a response for that query."
            
            # Remove redundant "Assistant:" prefixes if present
            response = re.sub(r'^(Assistant:?\s*)+', '', response)
            
            # Remove redundant line breaks and whitespace
            response = re.sub(r'\n{3,}', '\n\n', response)
            response = re.sub(r'\s{2,}', ' ', response)
            
            # Remove repetitive sentences
            response = self._remove_repetitive_content(response)
            
            # Check for hallucinated commands or syntax and warn
            if re.search(r'```(?:bash|sh|shell|cmd)\s+\/(?:discover|visualize|analyze)', response):
                response += "\n\n**Note:** To use commands, type them directly in the chat without the code block formatting."
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning response: {str(e)}")
            logger.error(traceback.format_exc())
            return response  # Return original response if error occurs
    
    def _remove_repetitive_content(self, text: str) -> str:
        """Remove repetitive sentences in responses.
        
        Args:
            text: Text to process
            
        Returns:
            Text with repetitive content removed
        """
        try:
            # Split into sentences (simple approximation)
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Remove exact duplicates while preserving order
            seen_sentences = set()
            unique_sentences = []
            
            for sentence in sentences:
                # Skip empty or very short sentences
                if len(sentence) <= 5:
                    unique_sentences.append(sentence)
                    continue
                    
                # Normalize the sentence for comparison
                normalized = re.sub(r'\s+', ' ', sentence.strip().lower())
                
                if normalized not in seen_sentences:
                    seen_sentences.add(normalized)
                    unique_sentences.append(sentence)
            
            # Join sentences back into text
            return ' '.join(unique_sentences)
            
        except Exception as e:
            logger.error(f"Error removing repetitive content: {str(e)}")
            logger.error(traceback.format_exc())
            return text  # Return original text if error occurs
    
    def validate_response(self, response: str, query: str) -> Tuple[bool, str]:
        """Validate response quality and relevance to the query.
        
        Args:
            response: Response text to validate
            query: Original query for comparison
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # If response is empty or too short, it's invalid
            if not response or len(response) < 10:
                return False, "Response is too short or empty"
            
            # Check for incomplete responses
            if self._is_incomplete_response(response):
                return False, "Response appears to be cut off or incomplete"
            
            # Check for irrelevant responses
            relevance = self._check_relevance(response, query)
            if relevance < 0.3:  # Arbitrary threshold
                return False, "Response does not appear to address the query"
            
            # Check for harmful content
            if self._has_harmful_content(response):
                return False, "Response contains potentially harmful content"
            
            # Check for excessive code without explanation
            if self._is_mostly_code(response) and not self._has_code_explanation(response):
                return False, "Response contains code without adequate explanation"
            
            return True, "Response is valid"
            
        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            logger.error(traceback.format_exc())
            return False, f"Error validating: {str(e)}"
    
    def _is_incomplete_response(self, text: str) -> bool:
        """Check if a response appears to be incomplete.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if response is incomplete
        """
        # Check for sentences that end abruptly
        if re.search(r'[a-zA-Z,;:]$', text):
            return True
            
        # Check for unclosed parentheses, brackets, or quotes
        counts = {
            '(': text.count('(') - text.count(')'),
            '[': text.count('[') - text.count(']'),
            '{': text.count('{') - text.count('}'),
            '"': text.count('"') % 2,
            "'": text.count("'") % 2
        }
        
        if any(counts.values()):
            return True
            
        # Check for incomplete code blocks
        code_block_starts = len(re.findall(r'```\w*', text))
        code_block_ends = len(re.findall(r'```$', text, re.MULTILINE))
        
        if code_block_starts > code_block_ends:
            return True
            
        return False
    
    def _check_relevance(self, response: str, query: str) -> float:
        """Check how relevant a response is to the query.
        
        Args:
            response: Response text to check
            query: Original query for comparison
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Simplified relevance check
        # In a production system, this would use more sophisticated NLP
        
        # Get important words from the query
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        
        # Remove common words
        stop_words = {"the", "and", "for", "that", "this", "with", "from", "have", "are", "what", "how", "why", "can", "you", "please", "help"}
        query_words = query_words - stop_words
        
        if not query_words:
            return 1.0  # No meaningful words to check
        
        # Check how many query words appear in the response
        response_lower = response.lower()
        matched_words = sum(1 for word in query_words if word in response_lower)
        
        return matched_words / len(query_words)
    
    def _has_harmful_content(self, text: str) -> bool:
        """Check if response contains potentially harmful content.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if harmful content was found
        """
        # This is a simple example - a real implementation would be more robust
        harmful_patterns = [
            r'(?:hack|crack|steal)\s+(?:password|account|data)',
            r'(?:illegal|unlawful)\s+(?:download|access|obtain)',
            r'circumvent\s+(?:security|protection|authentication)',
            r'exploit\s+(?:vulnerability|weakness|bug)',
            r'(?:avoid|evade)\s+(?:detection|monitoring|tracking)'
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        return False
    
    def _is_mostly_code(self, text: str) -> bool:
        """Check if response is mostly code with little explanation.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if content is mostly code
        """
        # Extract code blocks
        code_blocks = re.findall(r'```.*?```', text, re.DOTALL)
        code_text = ''.join(code_blocks)
        
        # If code blocks are more than 70% of the response, it's mostly code
        return len(code_text) > 0.7 * len(text)
    
    def _has_code_explanation(self, text: str) -> bool:
        """Check if response contains explanations for the code.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if code explanation is present
        """
        # Extract non-code text
        non_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        non_code = non_code.strip()
        
        # Check if there's substantial text outside code blocks
        if len(non_code) < 50:
            return False
            
        # Check for explanation indicators
        explanation_patterns = [
            r'(?:this|the|above|below|following)\s+(?:code|script|function)',
            r'(?:explains|works by|performs|implements)',
            r'(?:let me|I will|I\'ll)\s+(?:explain|walk through|describe)',
            r'(?:here\'s|here is)\s+(?:how|what)'
        ]
        
        for pattern in explanation_patterns:
            if re.search(pattern, non_code, re.IGNORECASE):
                return True
                
        return False
    
    def extract_code_snippets(self, text: str) -> List[Dict[str, str]]:
        """Extract code snippets from a response.
        
        Args:
            text: Text to extract code from
            
        Returns:
            List of dictionaries with language and code
        """
        try:
            snippets = []
            pattern = r'```(\w*)\n(.*?)```'
            matches = re.findall(pattern, text, re.DOTALL)
            
            for match in matches:
                language, code = match
                language = language.strip() or "unknown"
                code = code.strip()
                
                snippets.append({
                    "language": language,
                    "code": code
                })
            
            return snippets
            
        except Exception as e:
            logger.error(f"Error extracting code snippets: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def extract_suggestions(self, text: str) -> List[str]:
        """Extract suggested actions or commands from a response.
        
        Args:
            text: Text to extract suggestions from
            
        Returns:
            List of suggested actions or commands
        """
        try:
            suggestions = []
            
            # Look for command suggestions
            command_pattern = r'(?:you can use|try|run|execute|use the command|type)[\s:]*(\/[a-zA-Z0-9_]+(?:\s+[^\n\.!?]+)?)'
            command_matches = re.findall(command_pattern, text, re.IGNORECASE)
            suggestions.extend([cmd.strip() for cmd in command_matches])
            
            # Look for file reference suggestions
            file_pattern = r'(?:check|look at|open|examine|view|analyze)[\s:]*((?:\w+\/)*\w+\.\w+)'
            file_matches = re.findall(file_pattern, text, re.IGNORECASE)
            suggestions.extend([f"open {file.strip()}" for file in file_matches])
            
            # Look for explicit suggestions
            suggestion_pattern = r'(?:I suggest|you should|you might want to|consider)[\s:]*((?:\w+\s){2,}(?:\w+))'
            suggestion_matches = re.findall(suggestion_pattern, text, re.IGNORECASE)
            suggestions.extend([sugg.strip() for sugg in suggestion_matches])
            
            return list(set(suggestions))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting suggestions: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def format_response(self, response: str, format_type: str = "markdown") -> str:
        """Format agent response according to specified format.
        
        Args:
            response: Raw response text
            format_type: Format to convert to (markdown, html, plain)
            
        Returns:
            Formatted response text
        """
        try:
            if format_type == "plain":
                # Convert to plain text without formatting
                plain_text = re.sub(r'```.*?```', '', response, flags=re.DOTALL)  # Remove code blocks
                plain_text = re.sub(r'#+\s+', '', plain_text)  # Remove headers
                plain_text = re.sub(r'\*\*(.*?)\*\*', r'\1', plain_text)  # Remove bold
                plain_text = re.sub(r'\*(.*?)\*', r'\1', plain_text)  # Remove italic
                return plain_text.strip()
                
            elif format_type == "html":
                # Convert markdown to basic HTML
                html = response
                
                # Convert code blocks
                html = re.sub(r'```(\w*)\n(.*?)```', r'<pre><code class="language-\1">\2</code></pre>', html, flags=re.DOTALL)
                
                # Convert headers
                html = re.sub(r'^#\s+(.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
                html = re.sub(r'^##\s+(.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
                html = re.sub(r'^###\s+(.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
                
                # Convert formatting
                html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
                html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
                
                # Convert lists
                html = re.sub(r'^-\s+(.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
                html = re.sub(r'(<li>.*?</li>\n)+', r'<ul>\n\g<0></ul>', html, flags=re.DOTALL)
                
                # Convert newlines to <br>
                html = re.sub(r'\n', '<br>\n', html)
                
                return html.strip()
                
            else:  # Default to markdown
                return response
                
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            logger.error(traceback.format_exc())
            return response  # Return original response if error occurs
    
    def enhance_response(self, response: str, query: str) -> str:
        """Enhance agent response based on query context and response quality.
        
        Args:
            response: Original response text
            query: User query
            
        Returns:
            Enhanced response text
        """
        try:
            # Skip enhancement for very short responses
            if len(response) < 50:
                return response
            
            # Don't enhance responses that are already good
            is_valid, reason = self.validate_response(response, query)
            if is_valid:
                return response
            
            # Check if response is incomplete and needs completion
            if self._is_incomplete_response(response):
                return self._complete_response(response)
            
            # Check if response has low relevance and needs enhancement
            relevance = self._check_relevance(response, query)
            if relevance < 0.5:
                return self._improve_relevance(response, query)
            
            # If there are no specific issues, but validation failed
            if reason == "Response contains code without adequate explanation":
                return self._add_code_explanation(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error enhancing response: {str(e)}")
            logger.error(traceback.format_exc())
            return response  # Return original response if error occurs
    
    def _complete_response(self, text: str) -> str:
        """Attempt to complete an incomplete response.
        
        Args:
            text: Text to complete
            
        Returns:
            Completed text
        """
        # Close code blocks if needed
        if text.count('```') % 2 == 1:
            text += "\n```\n"
            
        # Add closing parentheses if needed
        counts = {
            '(': text.count('(') - text.count(')'),
            '[': text.count('[') - text.count(']'),
            '{': text.count('{') - text.count('}')
        }
        
        for bracket, count in counts.items():
            if count > 0:
                closing = ')' if bracket == '(' else (']' if bracket == '[' else '}')
                text += closing * count
            
        # Add closing quote if needed
        if text.count('"') % 2 == 1:
            text += '"'
            
        # If ending abruptly with a comma or colon
        if re.search(r'[,;:]$', text):
            text += " etc."
            
        return text
    
    def _improve_relevance(self, response: str, query: str) -> str:
        """Improve response relevance to the query.
        
        Args:
            response: Response text to improve
            query: Original query
            
        Returns:
            Improved response text
        """
        # Extract key terms from the query
        query_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', query.lower()))
        stop_words = {"the", "and", "for", "that", "this", "with", "from", "have", "are", "what", "how", "why", "can", "you", "please", "help"}
        query_words = query_words - stop_words
        
        if not query_words:
            return response  # No meaningful words to add
            
        # Add query context if missing
        missing_words = [word for word in query_words if word not in response.lower()]
        
        if missing_words:
            context = f"Regarding your question about {', '.join(missing_words)}, "
            return context + response
            
        return response
    
    def _add_code_explanation(self, response: str) -> str:
        """Add explanations to code-heavy responses.
        
        Args:
            response: Response text with code
            
        Returns:
            Response with added explanations
        """
        # Extract code blocks
        code_blocks = re.findall(r'```.*?```', response, re.DOTALL)
        
        if not code_blocks:
            return response
            
        # Get the first paragraph of text (if any)
        intro_match = re.match(r'^(.*?)(?:```|\n\n)', response, re.DOTALL)
        intro = intro_match.group(1).strip() if intro_match else ""
        
        if not intro:
            intro = "Here's the code that addresses your request:"
            response = intro + "\n\n" + response
            
        # Add explanation footer if there isn't already one
        explanation_patterns = [
            r'(?:this|the|above|below|following)\s+(?:code|script|function)',
            r'(?:explains|works by|performs|implements)',
            r'let me explain'
        ]
        
        has_explanation = any(re.search(pattern, response, re.IGNORECASE) for pattern in explanation_patterns)
        
        if not has_explanation:
            explanation = "\n\nThe code above provides the implementation you requested. It should be straightforward to understand and use."
            response += explanation
            
        return response
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """Attempt to parse JSON data from a response.
        
        Args:
            response: Response that may contain JSON
            
        Returns:
            Parsed JSON data or error information
        """
        try:
            # Try to find a JSON object in the response
            json_pattern = r'```(?:json)?\s*({[\s\S]*?})```'
            json_match = re.search(json_pattern, response)
            
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # If no JSON block, check if entire response is JSON
            if response.strip().startswith('{') and response.strip().endswith('}'):
                return json.loads(response)
                
            return {"error": "No valid JSON found in response"}
            
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format in response"}
        except Exception as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Error: {str(e)}"}
    
    def format_as_conversation(self, history: List[Dict[str, Any]], include_metadata: bool = False) -> str:
        """Format conversation history as a readable string.
        
        Args:
            history: List of conversation entries
            include_metadata: Whether to include timestamps and metadata
            
        Returns:
            Formatted conversation text
        """
        try:
            if not history:
                return "No conversation history available."
                
            lines = []
            
            for entry in history:
                role = entry.get('role', 'unknown')
                content = entry.get('content', '')
                
                if role == 'user':
                    prefix = "You: "
                elif role == 'assistant':
                    prefix = "Assistant: "
                else:
                    prefix = f"{role.capitalize()}: "
                
                # Format the message content
                if include_metadata and 'timestamp' in entry:
                    timestamp = entry['timestamp']
                    lines.append(f"[{timestamp}] {prefix}{content}")
                else:
                    lines.append(f"{prefix}{content}")
                    
                # Add metadata if requested
                if include_metadata and 'metadata' in entry and entry['metadata']:
                    metadata_str = json.dumps(entry['metadata'], indent=2)
                    lines.append(f"Metadata: {metadata_str}")
                    
                lines.append("")  # Add blank line between entries
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error formatting conversation: {str(e)}")
            logger.error(traceback.format_exc())
            return "Error formatting conversation history."
    
    def chat(self, message):
        """Process user's message and return a response.
        
        Args:
            message: User's message to process
            
        Returns:
            Response from the AI assistant
        """
        try:
            logger.info(f"Processing chat message: {message[:30]}...")
            
            # Add the message to conversation history if it exists
            if hasattr(self.document_reader, 'conversation_history'):
                self.document_reader.conversation_history.append({"role": "user", "content": message})
            
            # Process commands if the message starts with "/"
            if message.startswith("/"):
                if hasattr(self.document_reader, 'command_handler') and hasattr(self.document_reader.command_handler, 'process_command'):
                    response = self.document_reader.command_handler.process_command(message)
                    if response:
                        # Add the response to conversation history
                        if hasattr(self.document_reader, 'conversation_history'):
                            self.document_reader.conversation_history.append({"role": "assistant", "content": response})
                        return response
            
            # Process the message using the interactive agent if available
            if hasattr(self.document_reader, 'interactive_agent') and self.document_reader.interactive_agent:
                try:
                    response = self.document_reader.interactive_agent.print_response(
                        message,
                        stream=False
                    )
                    
                    # Clean the response
                    cleaned_response = self.clean_response(response)
                    
                    # Add the response to conversation history
                    if hasattr(self.document_reader, 'conversation_history'):
                        self.document_reader.conversation_history.append({"role": "assistant", "content": cleaned_response})
                    
                    return cleaned_response
                except Exception as e:
                    logger.error(f"Error with interactive agent: {str(e)}")
                    return f"Error processing your request: {str(e)}"
            
            # Fallback response if no interactive agent is available
            fallback_response = "I'm sorry, the system is not fully initialized yet. Please try again later."
            
            # Add the fallback response to conversation history
            if hasattr(self.document_reader, 'conversation_history'):
                self.document_reader.conversation_history.append({"role": "assistant", "content": fallback_response})
            
            return fallback_response
            
        except Exception as e:
            logger.error(f"Error in chat method: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error processing your request: {str(e)}" 
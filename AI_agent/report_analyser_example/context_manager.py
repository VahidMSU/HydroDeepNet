from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os
import uuid


@dataclass
class AppContext:
    """
    Shared context for the document reader application.
    This dataclass manages state that needs to be available across multiple modules.
    """
    # Session information
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    
    # File discovery and tracking
    discovered_files: Dict[str, List[str]] = field(default_factory=dict)
    current_files: List[str] = field(default_factory=list)
    base_path: Optional[str] = None
    
    # Conversation tracking
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_topic: str = "general"
    
    # Analysis state
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    visualizations: Dict[str, str] = field(default_factory=dict)
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    verbose: bool = False
    
    def __post_init__(self):
        """Initialize derived properties after the instance has been created."""
        # Ensure discovered_files has the expected structure
        if not self.discovered_files:
            self.discovered_files = {
                'csv': [],
                'md': [],
                'txt': [],
                'json': [],
                'png': [],
                'jpg': []
            }
            
        # Set up base path if not provided
        if not self.base_path:
            self.base_path = os.getcwd()
            
    def update_files(self, file_type: str, file_paths: List[str]):
        """Update the discovered files of a specific type."""
        if file_type not in self.discovered_files:
            self.discovered_files[file_type] = []
            
        # Add new files without duplicates
        for path in file_paths:
            if path not in self.discovered_files[file_type]:
                self.discovered_files[file_type].append(path)
    
    def add_to_conversation(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": __import__('time').time()
        })
        
    def set_current_topic(self, topic: str, files: Optional[List[str]] = None):
        """Update the current topic and relevant files."""
        self.current_topic = topic
        if files:
            self.current_files = files
            
    def save_analysis_result(self, file_path: str, result: Any):
        """Save analysis results for a specific file."""
        self.analysis_results[file_path] = result
        
    def save_visualization(self, query: str, file_path: str, vis_path: str):
        """Save path to a generated visualization."""
        key = f"{query}:{file_path}"
        self.visualizations[key] = vis_path
        
    def get_file_analysis(self, file_path: str) -> Optional[Any]:
        """Get analysis results for a specific file if available."""
        return self.analysis_results.get(file_path)


# Create a singleton instance to be imported by other modules
context = AppContext()

def reset_context():
    """Reset the context to initial state (useful for testing)."""
    global context
    context = AppContext() 
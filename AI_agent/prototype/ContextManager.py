import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import uuid
from collections import deque

@dataclass
class ContextEntry:
    """A single context entry containing information about an interaction or state."""
    id: str
    timestamp: datetime
    type: str  # conversation, state_change, action, etc.
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    references: List[str] = field(default_factory=list)

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    id: str
    timestamp: datetime
    user_input: str
    agent_response: str
    context_used: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContextManager:
    """Manages context information for AI agents including history, state, and retrieval."""
    
    def __init__(self,
                 max_history_size: int = 100,
                 storage_dir: str = "context_storage",
                 log_dir: str = "logs",
                 log_level: int = logging.INFO):
        """
        Initialize the context manager.
        
        Args:
            max_history_size: Maximum number of context entries to keep in memory
            storage_dir: Directory for persisting context data
            log_dir: Directory for log files
            log_level: Logging level
        """
        # Set up logging
        self._setup_logging(log_dir, log_level)
        
        self.logger.info("Initializing ContextManager")
        
        # Initialize session ID
        self.session_id = str(uuid.uuid4())
        
        # Set up storage
        self.storage_dir = Path(storage_dir)
        self.context_dir = self.storage_dir / "contexts"
        self.conversation_dir = self.storage_dir / "conversations"
        self.state_dir = self.storage_dir / "states"
        
        # Create directories
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.context_dir.mkdir(exist_ok=True)
        self.conversation_dir.mkdir(exist_ok=True)
        self.state_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Storage initialized at: {self.storage_dir}")
        
        # Initialize context storage
        self.max_history_size = max_history_size
        self.context_history: deque[ContextEntry] = deque(maxlen=max_history_size)
        self.conversation_history: deque[ConversationTurn] = deque(maxlen=max_history_size)
        
        # Current state storage
        self.current_state: Dict[str, Any] = {}
        self.active_contexts: Dict[str, ContextEntry] = {}
        
        self.logger.info("ContextManager initialization completed")

    def _setup_logging(self, log_dir: str, log_level: int):
        """Set up logging configuration."""
        # Create logger
        self.logger = logging.getLogger("ContextManager")
        self.logger.setLevel(log_level)
        
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # File handler for operations
        ops_handler = logging.FileHandler(
            log_path / "context_manager_operations.log"
        )
        ops_handler.setLevel(log_level)
        ops_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(ops_handler)
        
        # File handler for errors
        error_handler = logging.FileHandler(
            log_path / "context_manager_errors.log"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
            'Exception: %(exc_info)s'
        ))
        self.logger.addHandler(error_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)
        
        self.logger.info("Logging setup completed")

    def add_context(self,
                   type: str,
                   content: Dict[str, Any],
                   metadata: Optional[Dict[str, Any]] = None,
                   source: Optional[str] = None,
                   references: Optional[List[str]] = None) -> ContextEntry:
        """
        Add a new context entry.
        
        Args:
            type: Type of context entry
            content: Content of the context
            metadata: Optional metadata
            source: Optional source of the context
            references: Optional list of related context IDs
            
        Returns:
            Created ContextEntry
        """
        self.logger.info(f"Adding new context entry of type: {type}")
        try:
            entry = ContextEntry(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                type=type,
                content=content,
                metadata=metadata or {},
                source=source,
                references=references or []
            )
            
            # Add to history
            self.context_history.append(entry)
            
            # Persist context
            self._persist_context(entry)
            
            # Update active contexts if needed
            if type in ['state_change', 'active_context']:
                self.active_contexts[entry.id] = entry
            
            self.logger.debug(f"Context entry added with ID: {entry.id}")
            return entry
            
        except Exception as e:
            self.logger.error(f"Error adding context: {str(e)}", exc_info=True)
            raise

    def add_conversation_turn(self,
                            user_input: str,
                            agent_response: str,
                            context_used: Dict[str, Any],
                            metadata: Optional[Dict[str, Any]] = None) -> ConversationTurn:
        """
        Add a conversation turn to the history.
        
        Args:
            user_input: User's input
            agent_response: Agent's response
            context_used: Context information used for this turn
            metadata: Optional metadata
            
        Returns:
            Created ConversationTurn
        """
        self.logger.info("Adding new conversation turn")
        try:
            turn = ConversationTurn(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                user_input=user_input,
                agent_response=agent_response,
                context_used=context_used,
                metadata=metadata or {}
            )
            
            # Add to history
            self.conversation_history.append(turn)
            
            # Persist conversation
            self._persist_conversation(turn)
            
            self.logger.debug(f"Conversation turn added with ID: {turn.id}")
            return turn
            
        except Exception as e:
            self.logger.error(f"Error adding conversation turn: {str(e)}", exc_info=True)
            raise

    def update_state(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the current state with new information.
        
        Args:
            updates: Dictionary of state updates
            
        Returns:
            Updated state
        """
        self.logger.info("Updating current state")
        try:
            # Create state change context
            self.add_context(
                type="state_change",
                content={"previous": self.current_state.copy(), "updates": updates}
            )
            
            # Update state
            self.current_state.update(updates)
            
            # Persist state
            self._persist_state()
            
            self.logger.debug("State updated successfully")
            return self.current_state
            
        except Exception as e:
            self.logger.error(f"Error updating state: {str(e)}", exc_info=True)
            raise

    def get_relevant_context(self,
                           query: str,
                           context_types: Optional[List[str]] = None,
                           max_entries: int = 5) -> List[ContextEntry]:
        """
        Retrieve relevant context entries based on a query.
        
        Args:
            query: Search query
            context_types: Optional list of context types to filter
            max_entries: Maximum number of entries to return
            
        Returns:
            List of relevant context entries
        """
        self.logger.info(f"Retrieving relevant context for query: {query}")
        try:
            # Filter by type if specified
            entries = list(self.context_history)
            if context_types:
                entries = [e for e in entries if e.type in context_types]
            
            # Sort by relevance and recency
            # This is a simple implementation - could be enhanced with embedding-based search
            relevant = sorted(
                entries,
                key=lambda x: (
                    self._calculate_relevance(query, x),
                    x.timestamp
                ),
                reverse=True
            )
            
            self.logger.debug(f"Found {len(relevant[:max_entries])} relevant entries")
            return relevant[:max_entries]
            
        except Exception as e:
            self.logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
            return []

    def get_conversation_history(self,
                               limit: Optional[int] = None,
                               start_time: Optional[datetime] = None) -> List[ConversationTurn]:
        """
        Retrieve conversation history.
        
        Args:
            limit: Optional maximum number of turns to return
            start_time: Optional start time filter
            
        Returns:
            List of conversation turns
        """
        self.logger.info("Retrieving conversation history")
        try:
            history = list(self.conversation_history)
            
            # Apply filters
            if start_time:
                history = [t for t in history if t.timestamp >= start_time]
            
            if limit:
                history = history[-limit:]
            
            self.logger.debug(f"Retrieved {len(history)} conversation turns")
            return history
            
        except Exception as e:
            self.logger.error(f"Error retrieving conversation history: {str(e)}", exc_info=True)
            return []

    def get_state(self) -> Dict[str, Any]:
        """Get the current state."""
        return self.current_state.copy()

    def get_active_contexts(self) -> Dict[str, ContextEntry]:
        """Get currently active contexts."""
        return self.active_contexts.copy()

    def clear_context(self, context_id: str):
        """
        Clear a specific context entry.
        
        Args:
            context_id: ID of the context to clear
        """
        self.logger.info(f"Clearing context: {context_id}")
        try:
            # Remove from active contexts
            if context_id in self.active_contexts:
                del self.active_contexts[context_id]
            
            # Mark as cleared in storage
            context_file = self.context_dir / f"{context_id}.json"
            if context_file.exists():
                with open(context_file, 'r') as f:
                    data = json.load(f)
                data['cleared'] = True
                with open(context_file, 'w') as f:
                    json.dump(data, f)
            
            self.logger.debug(f"Context {context_id} cleared")
            
        except Exception as e:
            self.logger.error(f"Error clearing context: {str(e)}", exc_info=True)
            raise

    def _persist_context(self, entry: ContextEntry):
        """Persist a context entry to storage."""
        try:
            file_path = self.context_dir / f"{entry.id}.json"
            with open(file_path, 'w') as f:
                json.dump({
                    'id': entry.id,
                    'timestamp': entry.timestamp.isoformat(),
                    'type': entry.type,
                    'content': entry.content,
                    'metadata': entry.metadata,
                    'source': entry.source,
                    'references': entry.references
                }, f)
        except Exception as e:
            self.logger.error(f"Error persisting context: {str(e)}", exc_info=True)

    def _persist_conversation(self, turn: ConversationTurn):
        """Persist a conversation turn to storage."""
        try:
            file_path = self.conversation_dir / f"{turn.id}.json"
            with open(file_path, 'w') as f:
                json.dump({
                    'id': turn.id,
                    'timestamp': turn.timestamp.isoformat(),
                    'user_input': turn.user_input,
                    'agent_response': turn.agent_response,
                    'context_used': turn.context_used,
                    'metadata': turn.metadata
                }, f)
        except Exception as e:
            self.logger.error(f"Error persisting conversation: {str(e)}", exc_info=True)

    def _persist_state(self):
        """Persist current state to storage."""
        try:
            file_path = self.state_dir / "current_state.json"
            with open(file_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'state': self.current_state
                }, f)
        except Exception as e:
            self.logger.error(f"Error persisting state: {str(e)}", exc_info=True)

    def _calculate_relevance(self, query: str, entry: ContextEntry) -> float:
        """
        Calculate relevance score between query and context entry.
        This is a simple implementation that could be enhanced with more sophisticated matching.
        """
        try:
            # Convert query and content to lowercase for comparison
            query = query.lower()
            content_str = json.dumps(entry.content).lower()
            
            # Calculate simple word overlap
            query_words = set(query.split())
            content_words = set(content_str.split())
            
            overlap = len(query_words.intersection(content_words))
            total = len(query_words.union(content_words))
            
            return overlap / total if total > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating relevance: {str(e)}", exc_info=True)
            return 0.0

    def load_state(self):
        """Load persisted state from storage."""
        self.logger.info("Loading persisted state")
        try:
            state_file = self.state_dir / "current_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                self.current_state = data['state']
                self.logger.debug("State loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}", exc_info=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the context manager's state."""
        self.logger.info("Gathering context manager statistics")
        try:
            stats = {
                "context_entries": {
                    "total": len(self.context_history),
                    "by_type": {},
                    "active": len(self.active_contexts)
                },
                "conversation": {
                    "total_turns": len(self.conversation_history)
                },
                "state": {
                    "keys": len(self.current_state),
                    "last_updated": None
                }
            }
            
            # Context type statistics
            for entry in self.context_history:
                if entry.type not in stats["context_entries"]["by_type"]:
                    stats["context_entries"]["by_type"][entry.type] = 0
                stats["context_entries"]["by_type"][entry.type] += 1
            
            # State last updated
            state_file = self.state_dir / "current_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                stats["state"]["last_updated"] = data['timestamp']
            
            self.logger.info("Statistics gathered successfully")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {str(e)}", exc_info=True)
            return {}


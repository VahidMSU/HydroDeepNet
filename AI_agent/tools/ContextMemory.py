import json
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
import sqlite3
from pathlib import Path

# Import the config loader
from config_loader import get_config

class ContextMemory:
    """
    Provides persistent storage for agent context, including:
    - Conversation history
    - User preferences
    - Insights from data analysis
    - Session metadata
    - Conversation state for stateful interactions
    
    This class combines the functionality of the previous ConversationContext and
    ContextMemory classes into a single unified interface.
    """
    
    def __init__(self, storage_path: Optional[str] = None, max_history: int = 10, logger=None):
        """
        Initialize context memory with storage path.
        
        Args:
            storage_path: Path to the SQLite database file
            max_history: Maximum number of conversation entries to keep in memory
            logger: Logger instance to use
        """
        # Get storage path from config, fallback to provided path or default
        config = get_config()
        db_path_config = config.get('context_memory_db', 'memory_store.db')
        self.storage_path = storage_path or db_path_config
        self.logger = logger or logging.getLogger(__name__)
        self.max_history = max_history
        
        # Ensure the directory for the DB exists if it's not in the current dir
        db_dir = Path(self.storage_path).parent
        if db_dir != Path('.'):
            os.makedirs(db_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Current session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # In-memory cache
        self.cache = {
            "conversation": [],
            "insights": [],
            "preferences": {},
            "viewed_files": []
        }
        
        # Session data (compatibility with ConversationContext)
        self.session_data = {
            "viewed_files": [],
            "viewed_groups": [],
            "viewed_reports": [],
            "last_query_time": None,
            "insights": []
        }
        
        # Conversation state tracking
        self.conversation_state = {
            "awaiting_clarification": False,
            "clarification_type": None,
            "clarification_group": None,
            "awaiting_file_selection": False,
            "selected_file_type": None,
            "current_report": None,
            "current_group": None
        }
        
        # For compatibility with ConversationContext
        self.conversation_history = []
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            # Create sessions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                metadata TEXT
            )
            ''')
            
            # Create conversation table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                role TEXT,
                content TEXT,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
            ''')
            
            # Create insights table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                content TEXT,
                source TEXT,
                relevance REAL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
            ''')
            
            # Create file views table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_views (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                file_name TEXT,
                file_type TEXT,
                group_name TEXT,
                report_id TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
            ''')
            
            # Create user preferences table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                preference_key TEXT,
                preference_value TEXT,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
            ''')
            
            # Create conversation state table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                state_key TEXT,
                state_value TEXT,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            # Fall back to in-memory only
    
    def set_conversation_state(self, key: str, value: Any) -> bool:
        """
        Set a conversation state variable.
        
        Args:
            key: State variable name
            value: State variable value
            
        Returns:
            Success status
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Update in-memory state
            self.conversation_state[key] = value
            
            # Convert value to string if needed
            if not isinstance(value, str) and value is not None:
                str_value = json.dumps(value)
            else:
                str_value = value if value is not None else "null"
            
            # Update database
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            # Check if state exists
            cursor.execute(
                "SELECT id FROM conversation_state WHERE session_id = ? AND state_key = ?",
                (self.session_id, key)
            )
            
            existing_state = cursor.fetchone()
            
            if existing_state:
                # Update existing state
                cursor.execute(
                    "UPDATE conversation_state SET state_value = ?, timestamp = ? WHERE session_id = ? AND state_key = ?",
                    (str_value, timestamp, self.session_id, key)
                )
            else:
                # Insert new state
                cursor.execute(
                    "INSERT INTO conversation_state (session_id, state_key, state_value, timestamp) VALUES (?, ?, ?, ?)",
                    (self.session_id, key, str_value, timestamp)
                )
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting conversation state: {str(e)}")
            return False
    
    def get_conversation_state(self, key: str, default: Any = None) -> Any:
        """
        Get a conversation state variable.
        
        Args:
            key: State variable name
            default: Default value if state not found
            
        Returns:
            State variable value
        """
        # Check in-memory state first
        if key in self.conversation_state:
            return self.conversation_state[key]
        
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT state_value FROM conversation_state WHERE session_id = ? AND state_key = ?",
                (self.session_id, key)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                value = row[0]
                # Try to parse as JSON in case it's a complex type
                if value == "null":
                    parsed_value = None
                else:
                    try:
                        parsed_value = json.loads(value)
                        self.conversation_state[key] = parsed_value
                    except:
                        self.conversation_state[key] = value
                        return value
                return parsed_value
            
            return default
            
        except Exception as e:
            self.logger.error(f"Error getting conversation state: {str(e)}")
            return default
    
    def clear_conversation_state(self):
        """
        Reset all conversation state variables.
        
        Returns:
            Success status
        """
        try:
            # Reset in-memory state
            self.conversation_state = {
                "awaiting_clarification": False,
                "clarification_type": None,
                "clarification_group": None,
                "awaiting_file_selection": False,
                "selected_file_type": None,
                "current_report": None,
                "current_group": None
            }
            
            # Clear database state
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "DELETE FROM conversation_state WHERE session_id = ?",
                (self.session_id,)
            )
            
            conn.commit()
            conn.close()
            
            self.logger.info("Conversation state cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing conversation state: {str(e)}")
            return False
    
    def load_conversation_state(self):
        """
        Load all conversation state variables for the current session.
        
        Returns:
            Dictionary of state variables
        """
        try:
            conn = sqlite3.connect(self.storage_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT state_key, state_value FROM conversation_state WHERE session_id = ?",
                (self.session_id,)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            # Update in-memory state
            for row in rows:
                key = row["state_key"]
                value = row["state_value"]
                
                # Parse value
                if value == "null":
                    self.conversation_state[key] = None
                else:
                    try:
                        self.conversation_state[key] = json.loads(value)
                    except:
                        self.conversation_state[key] = value
            
            return self.conversation_state
            
        except Exception as e:
            self.logger.error(f"Error loading conversation state: {str(e)}")
            return self.conversation_state
    
    def start_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new session and record it in the database.
        
        Args:
            metadata: Optional metadata about the session
            
        Returns:
            Session ID string
        """
        try:
            # Create a new session ID
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Store session in database
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO sessions (session_id, start_time, metadata) VALUES (?, ?, ?)",
                (
                    self.session_id, 
                    datetime.now().isoformat(),
                    json.dumps(metadata or {})
                )
            )
            
            conn.commit()
            conn.close()
            
            # Reset conversation state
            self.clear_conversation_state()
            
            self.logger.info(f"Started new session: {self.session_id}")
            return self.session_id
            
        except Exception as e:
            self.logger.error(f"Error starting session: {str(e)}")
            return self.session_id
    
    def end_session(self) -> bool:
        """
        End the current session and update the database.
        
        Returns:
            Success status
        """
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE sessions SET end_time = ? WHERE session_id = ?",
                (datetime.now().isoformat(), self.session_id)
            )
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Ended session: {self.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error ending session: {str(e)}")
            return False
    
    # ConversationContext compatibility methods
    def add_user_message(self, message: str):
        """Add a user message to conversation history (ConversationContext compatibility)."""
        entry = {
            "role": "user", 
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_history.append(entry)
        self._trim_history()
        
        # Also add to the persistent storage
        self.add_message("user", message)
    
    def add_assistant_message(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Add an assistant message to conversation history (ConversationContext compatibility)."""
        entry = {
            "role": "assistant", 
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.conversation_history.append(entry)
        self._trim_history()
        
        # Also add to the persistent storage
        self.add_message("assistant", message, metadata)
        
    def _trim_history(self):
        """Trim history to max size (ConversationContext compatibility)."""
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_recent_history(self, num_entries: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent conversation history (ConversationContext compatibility)."""
        return self.conversation_history[-num_entries:]
    
    def get_formatted_history(self, num_entries: int = 5) -> str:
        """
        Get formatted conversation history for context.
        
        Args:
            num_entries: Maximum number of entries to retrieve
            
        Returns:
            Formatted conversation history as a string
        """
        recent = self.get_conversation_history(num_entries)
        formatted = []
        
        for entry in recent:
            role = "User" if entry.get("role") == "user" else "Assistant"
            content = entry.get("content", "")
            formatted.append(f"{role}: {content}")
            
        return "\n".join(formatted)
    
    # Original ContextMemory methods
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a message to the conversation history.
        
        Args:
            role: Role of the message sender (user/assistant)
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            Success status
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Add to in-memory cache
            self.cache["conversation"].append({
                "role": role,
                "content": content,
                "timestamp": timestamp,
                "metadata": metadata or {}
            })
            
            # Trim in-memory cache if needed
            if len(self.cache["conversation"]) > 100:
                self.cache["conversation"] = self.cache["conversation"][-100:]
            
            # Add to database
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO conversations (session_id, timestamp, role, content, metadata) VALUES (?, ?, ?, ?, ?)",
                (
                    self.session_id,
                    timestamp,
                    role,
                    content,
                    json.dumps(metadata or {})
                )
            )
            
            conn.commit()
            conn.close()
            
            # Also update conversation_history for compatibility
            entry = {
                "role": role, 
                "content": content,
                "timestamp": timestamp,
                "metadata": metadata or {}
            }
            self.conversation_history.append(entry)
            self._trim_history()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding message: {str(e)}")
            return False
    
    def add_insight(self, content: str, source: str, relevance: float = 1.0) -> bool:
        """
        Add an insight from analysis to the memory.
        
        Args:
            content: Insight content
            source: Source of the insight (file, group, etc.)
            relevance: Relevance score (0.0-1.0)
            
        Returns:
            Success status
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Add to in-memory cache
            self.cache["insights"].append({
                "content": content,
                "source": source,
                "timestamp": timestamp,
                "relevance": relevance
            })
            
            # Add to session data for compatibility
            self.session_data["insights"].append({
                "content": content,
                "source": source,
                "timestamp": timestamp
            })
            
            # Add to database
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO insights (session_id, timestamp, content, source, relevance) VALUES (?, ?, ?, ?, ?)",
                (
                    self.session_id,
                    timestamp,
                    content,
                    source,
                    relevance
                )
            )
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding insight: {str(e)}")
            return False
    
    # Compatible with both implementations
    def record_insight(self, insight: str, source: str):
        """Record a key insight from analysis (ConversationContext compatibility)."""
        return self.add_insight(insight, source)
    
    def record_file_view(self, file_name: str, file_type: str, group_name: str, report_id: str) -> bool:
        """
        Record that a file was viewed.
        
        Args:
            file_name: Name of the file
            file_type: Type of the file
            group_name: Group the file belongs to
            report_id: Report ID
            
        Returns:
            Success status
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Add to in-memory cache
            self.cache["viewed_files"].append({
                "file_name": file_name,
                "file_type": file_type,
                "group_name": group_name,
                "report_id": report_id,
                "timestamp": timestamp
            })
            
            # Add to session data for compatibility
            self.session_data["viewed_files"].append({
                "file_name": file_name,
                "file_type": file_type,
                "group": group_name,
                "report": report_id,
                "timestamp": timestamp
            })
            
            # Add to viewed groups and reports if not already there
            if group_name not in self.session_data["viewed_groups"]:
                self.session_data["viewed_groups"].append(group_name)
            
            if report_id not in self.session_data["viewed_reports"]:
                self.session_data["viewed_reports"].append(report_id)
            
            # Add to database
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO file_views (session_id, timestamp, file_name, file_type, group_name, report_id) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.session_id,
                    timestamp,
                    file_name,
                    file_type,
                    group_name,
                    report_id
                )
            )
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording file view: {str(e)}")
            return False
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation history.
        
        Args:
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of conversation messages
        """
        # If we have enough in cache, use that
        if len(self.cache["conversation"]) >= limit:
            return self.cache["conversation"][-limit:]
        
        try:
            conn = sqlite3.connect(self.storage_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT timestamp, role, content, metadata FROM conversations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (self.session_id, limit)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dictionaries
            result = []
            for row in reversed(rows):  # Reverse to get chronological order
                result.append({
                    "timestamp": row["timestamp"],
                    "role": row["role"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"])
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting conversation history: {str(e)}")
            return self.cache["conversation"][-limit:]
    
    def get_insights(self, limit: int = 10, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get insights from memory.
        
        Args:
            limit: Maximum number of insights to retrieve
            source: Optional source filter
            
        Returns:
            List of insights
        """
        try:
            conn = sqlite3.connect(self.storage_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if source:
                cursor.execute(
                    "SELECT timestamp, content, source, relevance FROM insights WHERE session_id = ? AND source LIKE ? ORDER BY relevance DESC, timestamp DESC LIMIT ?",
                    (self.session_id, f"%{source}%", limit)
                )
            else:
                cursor.execute(
                    "SELECT timestamp, content, source, relevance FROM insights WHERE session_id = ? ORDER BY relevance DESC, timestamp DESC LIMIT ?",
                    (self.session_id, limit)
                )
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dictionaries
            result = []
            for row in rows:
                result.append({
                    "timestamp": row["timestamp"],
                    "content": row["content"],
                    "source": row["source"],
                    "relevance": row["relevance"]
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting insights: {str(e)}")
            
            # Fall back to in-memory cache
            if source:
                filtered = [i for i in self.cache["insights"] if source in i["source"]]
                return sorted(filtered, key=lambda x: x["relevance"], reverse=True)[:limit]
            else:
                return sorted(self.cache["insights"], key=lambda x: x["relevance"], reverse=True)[:limit]
    
    def get_recent_file_views(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recently viewed files.
        
        Args:
            limit: Maximum number of file views to retrieve
            
        Returns:
            List of file views
        """
        try:
            conn = sqlite3.connect(self.storage_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT timestamp, file_name, file_type, group_name, report_id FROM file_views WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (self.session_id, limit)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dictionaries
            result = []
            for row in rows:
                result.append({
                    "timestamp": row["timestamp"],
                    "file_name": row["file_name"],
                    "file_type": row["file_type"],
                    "group_name": row["group_name"],
                    "report_id": row["report_id"]
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting recent file views: {str(e)}")
            return self.cache["viewed_files"][-limit:]
    
    def set_preference(self, key: str, value: Any) -> bool:
        """
        Set a user preference.
        
        Args:
            key: Preference key
            value: Preference value
            
        Returns:
            Success status
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Convert value to string if needed
            if not isinstance(value, str):
                value = json.dumps(value)
            
            # Update in-memory cache
            self.cache["preferences"][key] = value
            
            # Update database
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            # Check if preference exists
            cursor.execute(
                "SELECT id FROM user_preferences WHERE session_id = ? AND preference_key = ?",
                (self.session_id, key)
            )
            
            existing_pref = cursor.fetchone()
            
            if existing_pref:
                # Update existing preference
                cursor.execute(
                    "UPDATE user_preferences SET preference_value = ?, timestamp = ? WHERE session_id = ? AND preference_key = ?",
                    (value, timestamp, self.session_id, key)
                )
            else:
                # Insert new preference
                cursor.execute(
                    "INSERT INTO user_preferences (session_id, preference_key, preference_value, timestamp) VALUES (?, ?, ?, ?)",
                    (self.session_id, key, value, timestamp)
                )
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting preference: {str(e)}")
            return False
    
    def get_preference(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a user preference.
        
        Args:
            key: Preference key
            default: Default value if preference not found
            
        Returns:
            Preference value
        """
        # Check in-memory cache first
        if key in self.cache["preferences"]:
            value = self.cache["preferences"][key]
            try:
                return json.loads(value)
            except:
                return value
        
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT preference_value FROM user_preferences WHERE session_id = ? AND preference_key = ?",
                (self.session_id, key)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                value = row[0]
                # Try to parse as JSON in case it's a complex type
                try:
                    parsed_value = json.loads(value)
                    self.cache["preferences"][key] = value
                    return parsed_value
                except:
                    self.cache["preferences"][key] = value
                    return value
            
            return default
            
        except Exception as e:
            self.logger.error(f"Error getting preference: {str(e)}")
            return default
    
    def export_session_data(self, include_all_sessions: bool = False) -> Dict[str, Any]:
        """
        Export all session data as a JSON-serializable dictionary.
        
        Args:
            include_all_sessions: Whether to include data from all sessions
            
        Returns:
            Dictionary of session data
        """
        try:
            conn = sqlite3.connect(self.storage_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            session_filter = "" if include_all_sessions else f"WHERE session_id = '{self.session_id}'"
            
            # Get sessions
            cursor.execute(f"SELECT * FROM sessions {session_filter}")
            sessions = [dict(row) for row in cursor.fetchall()]
            
            result = {"sessions": sessions}
            
            # Add conversations
            for session in sessions:
                session_id = session["session_id"]
                
                cursor.execute("SELECT * FROM conversations WHERE session_id = ?", (session_id,))
                session["conversations"] = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute("SELECT * FROM insights WHERE session_id = ?", (session_id,))
                session["insights"] = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute("SELECT * FROM file_views WHERE session_id = ?", (session_id,))
                session["file_views"] = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute("SELECT * FROM user_preferences WHERE session_id = ?", (session_id,))
                session["preferences"] = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error exporting session data: {str(e)}")
            
            # Return in-memory cache as fallback
            return {
                "sessions": [{
                    "session_id": self.session_id,
                    "conversations": self.cache["conversation"],
                    "insights": self.cache["insights"],
                    "file_views": self.cache["viewed_files"],
                    "preferences": [{"key": k, "value": v} for k, v in self.cache["preferences"].items()]
                }]
            }

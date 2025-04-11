"""
Memory management module for the document reader.
This module provides functions for storing and retrieving conversation history and analysis results.
"""
import logging
from context_manager import context
from typing import List, Dict, Any, Optional
import time

logger = logging.getLogger("AI_AgentLogger")

def save_conversation(message: str, role: str = "user") -> None:
    """
    Save a message to the conversation history.
    
    Args:
        message: The message to save
        role: The role of the message sender (user or assistant)
    """
    try:
        context.add_to_conversation(role, message)
        logger.debug(f"Added {role} message to conversation history: {message[:30]}...")
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}")


def get_conversation_history(max_messages: int = 10) -> List[Dict[str, Any]]:
    """
    Get the recent conversation history.
    
    Args:
        max_messages: Maximum number of messages to return
        
    Returns:
        List of conversation messages
    """
    try:
        # Return the most recent messages
        if context.conversation_history:
            return context.conversation_history[-max_messages:]
        return []
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}")
        return []


def save_file_analysis(file_path: str, analysis_result: Any) -> None:
    """
    Save analysis results for a file.
    
    Args:
        file_path: Path to the analyzed file
        analysis_result: The analysis result to save
    """
    try:
        context.save_analysis_result(file_path, analysis_result)
        logger.debug(f"Saved analysis result for {file_path}")
    except Exception as e:
        logger.error(f"Error saving file analysis: {str(e)}")


def get_file_analysis(file_path: str) -> Optional[Any]:
    """
    Get analysis results for a file if available.
    
    Args:
        file_path: Path to the file
        
    Returns:
        The analysis result or None if not found
    """
    try:
        return context.get_file_analysis(file_path)
    except Exception as e:
        logger.error(f"Error retrieving file analysis: {str(e)}")
        return None


def set_current_topic(topic: str, relevant_files: List[str] = None) -> None:
    """
    Set the current conversation topic and relevant files.
    
    Args:
        topic: The current topic
        relevant_files: List of files relevant to the topic
    """
    try:
        context.set_current_topic(topic, relevant_files)
        logger.debug(f"Set current topic to '{topic}' with {len(relevant_files or [])} relevant files")
    except Exception as e:
        logger.error(f"Error setting current topic: {str(e)}")


def get_current_topic() -> Dict[str, Any]:
    """
    Get the current conversation topic and relevant files.
    
    Returns:
        Dictionary with current topic and files
    """
    try:
        return {
            "topic": context.current_topic,
            "files": context.current_files
        }
    except Exception as e:
        logger.error(f"Error retrieving current topic: {str(e)}")
        return {"topic": None, "files": []}


def save_visualization(query: str, file_path: str, visualization_path: str) -> None:
    """
    Save the path to a generated visualization.
    
    Args:
        query: The query that prompted the visualization
        file_path: The data file used for the visualization
        visualization_path: Path to the generated visualization
    """
    try:
        context.save_visualization(query, file_path, visualization_path)
        logger.debug(f"Saved visualization for query '{query[:30]}...' using {file_path}")
    except Exception as e:
        logger.error(f"Error saving visualization: {str(e)}")


def get_visualizations() -> Dict[str, str]:
    """
    Get all saved visualizations.
    
    Returns:
        Dictionary of query:file_path keys to visualization paths
    """
    try:
        return context.visualizations
    except Exception as e:
        logger.error(f"Error retrieving visualizations: {str(e)}")
        return {}

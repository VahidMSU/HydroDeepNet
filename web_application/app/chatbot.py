from flask import Blueprint, jsonify, request, current_app
import os
import logging
import sys
import traceback
import subprocess
import tempfile
import json
import threading
import time
import signal
import atexit

# Initialize the AI assistant only once
from assistant.interactive_agent import interactive_agent
username = "admin"
# Global variables to hold the initialized agent
initialized_user_agent = None
initialized_query_engine = None

def initialize_ai_assistant():
    global initialized_user_agent, initialized_query_engine
    try:
        current_app.logger.info("Initializing AI assistant...")
        initialized_user_agent, initialized_query_engine = interactive_agent(username)
        if initialized_user_agent and initialized_query_engine:
            current_app.logger.info("AI assistant initialized successfully")
            return True
        else:
            current_app.logger.error("Failed to initialize AI assistant")
            return False
    except Exception as e:
        current_app.logger.error(f"Error initializing AI assistant: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return False

# Blueprint for chatbot routes
chatbot_bp = Blueprint('chatbot', __name__)

# Global variable to track the subprocess
ai_process = None
ai_process_lock = threading.Lock()

# Function to clean up subprocess on exit
def cleanup_ai_process():
    global ai_process
    with ai_process_lock:
        if ai_process:
            current_app.logger.info("Terminating AI process on application exit")
            try:
                # Send SIGTERM to process group to kill any child processes too
                os.killpg(os.getpgid(ai_process.pid), signal.SIGTERM)
            except Exception as e:
                current_app.logger.error(f"Error terminating AI process: {e}")

# Register cleanup function to run on Flask app exit
atexit.register(cleanup_ai_process)

@chatbot_bp.route('/api/chatbot/initialize', methods=['POST'])
def chatbot_initialize():
    """Initialize the AI agent with specific context."""
    try:
        data = request.get_json()
        context = data.get('context', 'hydrogeo_dataset') if data else 'hydrogeo_dataset'
        model_id = data.get('model', 'gpt-4o') if data else 'gpt-4o'
        session_id = data.get('session_id')
        
        # Initialize the AI assistant if not already done
        global initialized_user_agent, initialized_query_engine
        if initialized_user_agent is None or initialized_query_engine is None:
            success = initialize_ai_assistant()
            if not success:
                return jsonify({
                    "status": "error",
                    "message": "Failed to initialize AI assistant"
                }), 500
        
        # Create a welcome message that doesn't mention a specific model
        welcome_message = "Hello! I'm HydroInsight, your AI assistant for environmental and hydrological data analysis. I can help you explore and understand water resources data. What would you like to know?"
        
        return jsonify({
            "status": "success",
            "message": f"Chatbot initialized with context: {context}, model: {model_id}",
            "session_id": session_id or "new_session_" + str(int(time.time())),
            "welcome_message": welcome_message
        })
    except Exception as e:
        current_app.logger.error(f"Error initializing chatbot: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Failed to initialize chatbot: {str(e)}"
        }), 500

@chatbot_bp.route('/api/chatbot', methods=['POST'])
def chatbot_proxy():
    """Invoke the AI agent to interact with the user."""
    try:
        data = request.get_json()
        message = data.get('message') if data else None
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
                
        # Get response directly from the AI agent
        response = call_run_ai_script(message)
        
        # Return exactly what was returned from the AI agent
        return jsonify({
            "response": response,
            "status": "success" if response is not None else "error",
            "model": "dummy_model"
        })
        
    except Exception as e:
        current_app.logger.error(f"Error processing chatbot request: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        # Return error status but no custom message
        return jsonify({
            "response": None,
            "status": "error"
        }), 500

def call_run_ai_script(message):
    """Call the run_ai.py script and pass the message to it."""
    global initialized_user_agent, initialized_query_engine
    
    # Make sure the AI assistant is initialized
    if initialized_user_agent is None or initialized_query_engine is None:
        success = initialize_ai_assistant()
        if not success:
            # Log the error but don't generate a custom message
            current_app.logger.error("AI assistant not initialized, cannot process message")
            return None
    
    # Process the message and return exactly what comes from the interactive agent
    response = initialized_user_agent.process_query(message, initialized_query_engine)
    
    # Return the response as-is, even if None
    return response
    
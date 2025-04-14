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
        
        # Initialize any required resources
        # This could be expanded based on specific initialization needs
        
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
        model_id = data.get('model', 'gpt-4o') if data else 'gpt-4o'
        session_id = data.get('session_id')
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        current_app.logger.info(f"Processing message with model: {model_id}")
        
        # Use run_ai.py to process the message
        response = call_run_ai_script(message, current_app.logger, model_id, session_id)
        
        return jsonify({
            "response": response,
            "status": "success",
            "model": model_id
        })
        
    except Exception as e:
        current_app.logger.error(f"Error processing chatbot request: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "response": f"Error processing your request: {str(e)}",
            "status": "error"
        }), 500

def call_run_ai_script(message, logger, model_id='gpt-4o', session_id=None):
    """Call the run_ai.py script and pass the message to it."""
    global ai_process
    
    # Path to the run_ai.py script (Python version instead of bash)
    script_path = "/data/SWATGenXApp/codes/AI_agent/run_ai.py"
    
    # Use the Python interpreter from the virtual environment
    python_path = "/data/SWATGenXApp/codes/.venv/bin/python"
    
    logger.info(f"Starting AI request for message: {message[:50]}...")
    logger.info(f"Using model: {model_id}, session_id: {session_id}")
    
    # Create a temporary file to store the message and response
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as input_file, \
         tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as output_file:
        
        # Write the message to the input file with additional metadata
        input_data = {
            "message": message,
            "model": model_id,
            "session_id": session_id
        }
        json.dump(input_data, input_file)
        input_file.flush()
        
        logger.info(f"Input file created at: {input_file.name}")
        
        try:
            # Terminate any existing process to ensure a clean start
            with ai_process_lock:
                if ai_process is not None:
                    try:
                        if ai_process.poll() is None:  # Only if it's still running
                            logger.info(f"Terminating existing AI process before starting new one")
                            os.killpg(os.getpgid(ai_process.pid), signal.SIGTERM)
                            ai_process.wait(timeout=2)  # Wait up to 2 seconds for termination
                    except Exception as e:
                        logger.warning(f"Error terminating AI process: {e}")
                        logger.warning(traceback.format_exc())
                
                # Always start a new process for each request
                logger.info(f"Starting new AI process with script: {script_path}")
                command = [python_path, script_path, "--input", input_file.name, "--output", output_file.name, "--model", model_id, "--debug"]
                logger.info(f"Command: {' '.join(command)}")
                
                ai_process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    preexec_fn=os.setsid  # Use this to create a new process group
                )
                logger.info(f"AI process started with PID: {ai_process.pid}")
            
            # Wait for the output file to be populated (with timeout)
            max_wait_time = 120  # seconds
            wait_interval = 0.5  # seconds
            elapsed_time = 0
            
            logger.info(f"Waiting for AI response (max {max_wait_time} seconds)")
            
            # Loop until we get a response or timeout
            while elapsed_time < max_wait_time:
                # Check if the process has exited
                if ai_process.poll() is not None:
                    exit_code = ai_process.returncode
                    stdout, stderr = ai_process.communicate()
                    
                    # Log stdout and stderr for debugging
                    logger.info(f"AI process stdout: {stdout}")
                    if stderr:
                        logger.error(f"AI process stderr: {stderr}")
                    
                    # If exit code is 0, this is normal completion (not an error)
                    if exit_code == 0:
                        logger.info(f"AI process completed successfully with exit code 0")
                        # The process exited cleanly, just read the response
                        if os.path.exists(output_file.name) and os.path.getsize(output_file.name) > 0:
                            try:
                                with open(output_file.name, 'r') as f:
                                    response_data = json.load(f)
                                    # Reset the process
                                    ai_process = None
                                    logger.info(f"AI response received successfully after process exit")
                                    return response_data.get("response", "No response received from AI")
                            except Exception as e:
                                logger.error(f"Error reading response after process exit: {e}")
                                logger.error(traceback.format_exc())
                                break
                    else:
                        # Non-zero exit code indicates an error
                        logger.warning(f"AI process exited with code {exit_code}")
                        logger.warning(f"AI process stdout: {stdout}")
                        logger.warning(f"AI process stderr: {stderr}")
                        
                        # Check if stderr contains model-related errors to provide better error messages
                        if "model" in stderr.lower() and "OpenAI" in stderr:
                            logger.warning("Detected potential model availability issue")
                            return "I'm having trouble connecting to my language model. The system might be temporarily unavailable or the requested model isn't accessible. Please try again or try a different model like 'gpt-4' or 'gpt-3.5-turbo'."
                        
                        break
                
                # Check if output file has content
                if os.path.exists(output_file.name) and os.path.getsize(output_file.name) > 0:
                    try:
                        with open(output_file.name, 'r') as f:
                            temp_data = json.load(f)
                            logger.info(f"Output file content: {json.dumps(temp_data)[:200]}...")
                            if temp_data.get("status") != "processing":
                                logger.info(f"Output received after {elapsed_time} seconds")
                                # Terminate the process since we got our response
                                try:
                                    if ai_process.poll() is None:  # Only if it's still running
                                        os.killpg(os.getpgid(ai_process.pid), signal.SIGTERM)
                                except Exception as e:
                                    logger.warning(f"Error terminating AI process after response: {e}")
                                ai_process = None
                                return temp_data.get("response", "No response received from AI")
                    except json.JSONDecodeError:
                        # File might still be being written
                        pass
                    
                time.sleep(wait_interval)
                elapsed_time += wait_interval
                
                # Every 10 seconds, log that we're still waiting
                if elapsed_time % 10 < 0.5:
                    logger.info(f"Still waiting for AI response after {elapsed_time} seconds")
                    
                    # Also check if we can get any output from the process
                    if ai_process.poll() is None:  # Process is still running
                        try:
                            # Try to read output without blocking in case there's partial data
                            stdout_data, stderr_data = "", ""
                            if ai_process.stdout:
                                stdout_data = ai_process.stdout.read()
                            if ai_process.stderr:
                                stderr_data = ai_process.stderr.read()
                                
                            if stdout_data:
                                logger.info(f"Partial stdout: {stdout_data[:200]}...")
                            if stderr_data:
                                logger.error(f"Partial stderr: {stderr_data[:200]}...")
                        except:
                            # Ignore errors from non-blocking reads
                            pass
            
            # We've timed out or the process exited with an error
            logger.warning("Timed out waiting for AI response or process exited with error")
            
            # Try to capture any final output
            if ai_process and ai_process.poll() is None:
                try:
                    # Non-blocking read attempt
                    stdout, stderr = "", ""
                    try:
                        stdout, stderr = ai_process.communicate(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                        
                    if stdout:
                        logger.info(f"Final stdout before termination: {stdout[:200]}...")
                    if stderr:
                        logger.error(f"Final stderr before termination: {stderr[:200]}...")
                except:
                    pass
            
            # Clean up the process if it's still running
            with ai_process_lock:
                if ai_process is not None and ai_process.poll() is None:
                    try:
                        os.killpg(os.getpgid(ai_process.pid), signal.SIGTERM)
                    except Exception as e:
                        logger.warning(f"Error terminating AI process: {e}")
                ai_process = None
            
            return "The AI process took too long to respond. This might be due to high demand or processing a complex query. Please try again with a simpler question or try later."
            
        except Exception as e:
            logger.error(f"Error calling run_ai.py: {e}")
            logger.error(traceback.format_exc())
            
            # Clean up the process if it's still running
            if ai_process:
                try:
                    if ai_process.poll() is None:
                        os.killpg(os.getpgid(ai_process.pid), signal.SIGTERM)
                except Exception as term_err:
                    logger.error(f"Error terminating AI process: {term_err}")
                ai_process = None
            
            # Provide user-friendly error messages based on error type
            error_str = str(e)
            if "model" in error_str.lower() or "openai" in error_str.lower():
                return "I'm having trouble connecting to my language model service. The issue might be related to model availability or API access. Please try again later or try a different model."
            elif "connection" in error_str.lower() or "timeout" in error_str.lower():
                return "I'm having trouble establishing a connection. This might be due to network issues or service unavailability. Please try again later."
            else:
                return f"Sorry, I encountered an error while processing your request. Technical details: {str(e)}"
        finally:
            # Clean up the temporary files
            try:
                os.unlink(input_file.name)
                os.unlink(output_file.name)
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {e}")
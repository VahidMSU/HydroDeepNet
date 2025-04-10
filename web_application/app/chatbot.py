from flask import Blueprint, jsonify, request, current_app
import os
import logging
import sys
import traceback

# Set up logging outside of application context
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Blueprint for chatbot routes
chatbot_bp = Blueprint('chatbot', __name__)

# Try to import Agno modules, but handle gracefully if not available
try:
    from report_analyser_example.agno_hydrogeo import get_agno_agent, agno_respond
    AGNO_AVAILABLE = True
    logger.info("Successfully imported Agno modules")
except ImportError as e:
    AGNO_AVAILABLE = False
    logger.warning(f"Agno modules not available: {str(e)}. Will use fallback responses.")
except Exception as e:
    AGNO_AVAILABLE = False
    logger.error(f"Error importing Agno modules: {str(e)}")
    logger.error(traceback.format_exc())

# Check if Agno should be used
def should_use_agno(client_preference=None):
    """Determine if Agno should be used based on environment and client preference"""
    env_preference = os.environ.get('USE_AGNO', 'true').lower() == 'true'
    
    # If client explicitly requests a preference, honor it
    if client_preference is not None:
        return env_preference and client_preference and AGNO_AVAILABLE
    
    # Otherwise use environment setting
    return env_preference and AGNO_AVAILABLE

@chatbot_bp.route('/api/chatbot/initialize', methods=['POST'])
def chatbot_initialize():
    """Initialize the AI agent with specific context."""
    try:
        data = request.get_json()
        context = data.get('context', 'general') if data else 'general'
        model_id = data.get('model', 'gemini-1.5-flash') if data else 'gemini-1.5-flash'
        client_use_agno = data.get('use_agno', True) if data else True
        
        app_logger = current_app.logger
        app_logger.info(f"Initializing chatbot with context: {context}, model: {model_id}, client_use_agno: {client_use_agno}")
        
        # Determine if we should use Agno
        use_agno = should_use_agno(client_use_agno)
        
        # Default welcome message if Agno is not available
        welcome_messages = {
            'hydrogeo_dataset': "Hello! I'm your HydroGeo Assistant. I can help you understand and query the environmental and hydrological datasets available in our system. You can ask about data formats, specific variables, or how to interpret results. What would you like to know?",
            'general': "Hello! I'm ready to assist you with questions about our environmental data platform. How can I help you today?"
        }
        
        if use_agno:
            try:
                app_logger.info(f"Using Agno with model {model_id}")
                
                # Set API key in environment if not already there
                if not os.environ.get('GOOGLE_API_KEY') and current_app.config.get('GOOGLE_API_KEY'):
                    os.environ['GOOGLE_API_KEY'] = current_app.config.get('GOOGLE_API_KEY')
                    app_logger.info("Set GOOGLE_API_KEY in environment")
                
                # Get an Agno agent with the specified model
                agent = get_agno_agent(model_id)
                
                # Generate welcome message with Agno
                prompt = "Introduce yourself as the HydroGeo Assistant, an expert in environmental and hydrological data. Keep it brief, friendly, and mention that you can answer questions about data sources, interpretation, and analysis."
                welcome_message = agno_respond(agent, prompt)
                
                return jsonify({
                    "status": "success", 
                    "welcome_message": welcome_message,
                    "using_agno": True,
                    "model": model_id
                })
            except Exception as e:
                app_logger.error(f"Error initializing Agno agent: {str(e)}")
                app_logger.error(traceback.format_exc())
                # Fall back to default message if Agno fails
                welcome_message = welcome_messages.get(context, welcome_messages['general'])
                return jsonify({
                    "status": "warning", 
                    "welcome_message": welcome_message,
                    "using_agno": False,
                    "error": str(e)
                })
        else:
            # Use default welcome message if Agno is not available
            welcome_message = welcome_messages.get(context, welcome_messages['general'])
            return jsonify({
                "status": "success", 
                "welcome_message": welcome_message,
                "using_agno": False
            })
    except Exception as e:
        if current_app:
            current_app.logger.error(f"Error initializing chatbot: {str(e)}")
            current_app.logger.error(traceback.format_exc())
        else:
            logger.error(f"Error initializing chatbot: {str(e)}")
            logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "error": "Failed to initialize chatbot",
            "using_agno": False
        }), 500

@chatbot_bp.route('/api/chatbot', methods=['POST'])
def chatbot_proxy():
    """Invoke the AI agent to interact with the user."""
    try:
        data = request.get_json()
        message = data.get('message') if data else None
        context = data.get('context', 'hydrogeo_dataset') if data else 'hydrogeo_dataset'
        model_id = data.get('model', 'gemini-1.5-flash') if data else 'gemini-1.5-flash'
        client_use_agno = data.get('use_agno', True) if data else True
        
        app_logger = current_app.logger
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Determine if we should use Agno
        use_agno = should_use_agno(client_use_agno)
        
        app_logger.info(f"Processing message with context: {context}, model: {model_id}, use_agno: {use_agno}")
        
        if use_agno:
            try:
                # Set API key in environment if not already there
                if not os.environ.get('GOOGLE_API_KEY') and current_app.config.get('GOOGLE_API_KEY'):
                    os.environ['GOOGLE_API_KEY'] = current_app.config.get('GOOGLE_API_KEY')
                    app_logger.info("Set GOOGLE_API_KEY in environment")
                
                # Get an Agno agent with the specified model
                agent = get_agno_agent(model_id)
                
                # Generate response with Agno
                response = agno_respond(agent, message)
                
                return jsonify({
                    "response": response,
                    "status": "success",
                    "using_agno": True,
                    "model": model_id
                })
            except Exception as e:
                app_logger.error(f"Error with Agno response: {str(e)}")
                app_logger.error(traceback.format_exc())
                # Fall back to legacy response
                fallback_response = "I'm sorry, I encountered an error processing your request with Agno. Please try again with a different question or contact support."
                return jsonify({
                    "response": fallback_response,
                    "status": "error",
                    "using_agno": False,
                    "error": str(e)
                })
        else:
            try:
                # Try to import the interactive_agent function if it exists
                try:
                    # Dynamic import for backward compatibility
                    sys.path.append('/data/SWATGenXApp/codes/AI_agent')
                    from interactive_AI import interactive_agent
                    has_interactive_agent = True
                except ImportError:
                    has_interactive_agent = False
                
                if has_interactive_agent:
                    # Use legacy interactive_agent function
                    app_logger.info(f"Calling interactive_agent with message: {message}")
                    response = interactive_agent(message)
                else:
                    # Fallback response if no AI service is available
                    response = "I'm sorry, but the AI assistant is currently unavailable. Please try again later or contact support."
            except Exception as e:
                app_logger.error(f"Error in legacy response: {str(e)}")
                app_logger.error(traceback.format_exc())
                response = "I'm sorry, I encountered an error while processing your request. Please try again with a different question."
            
            return jsonify({
                "response": response,
                "status": "success",
                "using_agno": False
            })
    except Exception as e:
        if current_app:
            current_app.logger.error(f"Error in chatbot proxy: {str(e)}")
            current_app.logger.error(traceback.format_exc())
        else:
            logger.error(f"Error in chatbot proxy: {str(e)}")
            logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to process request", 
            "response": "I'm sorry, I encountered an error while processing your request.",
            "using_agno": False
        }), 500
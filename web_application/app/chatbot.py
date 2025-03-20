from flask import Blueprint, jsonify, request, current_app

chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route('/api/chatbot/initialize', methods=['POST'])
def chatbot_initialize():
    """Initialize the AI agent with specific context."""
    try:
        data = request.get_json()
        context = data.get('context', 'general') if data else 'general'
        
        # You can customize the welcome message based on context
        welcome_messages = {
            'hydrogeo_dataset': "Hello! I'm your HydroGeo Assistant. I can help you understand and query the environmental and hydrological datasets available in our system. You can ask about data formats, specific variables, or how to interpret results. What would you like to know?",
            'general': "Hello! I'm ready to assist you with questions about our environmental data platform. How can I help you today?"
        }
        
        welcome_message = welcome_messages.get(context, welcome_messages['general'])
        
        # You could initialize any session state here in your actual implementation
        return jsonify({
            "status": "success", 
            "welcome_message": welcome_message
        })
    except Exception as e:
        current_app.logger.error(f"Error initializing chatbot: {e}")
        return jsonify({"error": "Failed to initialize chatbot"}), 500

@chatbot_bp.route('/api/chatbot', methods=['POST'])
def chatbot_proxy():
    """Invoke the AI agent to interact with the user."""
    try:
        data = request.get_json()
        message = data.get('message') if data else None
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        try:
            # Import the function directly, not the module
            
            current_app.logger.info(f"Calling interactive_agent with message: {message}")
            # Call the function
            response = interactive_agent(message)
        except ImportError as e:
            current_app.logger.error(f"Failed to import interactive_agent: {e}")
            response = "I'm sorry, but I'm having trouble accessing my knowledge base right now. Please try again later or contact support."
        except Exception as e:
            current_app.logger.error(f"Error calling interactive_agent: {e}")
            response = "I'm sorry, I encountered an error while processing your request. Please try again with a different question."
        
        return jsonify({"response": response})
    except Exception as e:
        current_app.logger.error(f"Error in chatbot proxy: {e}")
        return jsonify({"error": "Failed to process request", "response": "I'm sorry, I encountered an error while processing your request."}), 500

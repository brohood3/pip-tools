import os
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

from flask import Flask, request, jsonify, render_template, send_from_directory
from openai import OpenAI
from flask_socketio import SocketIO, emit

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev_key")
socketio = SocketIO(app)

# Try to import tools, but provide fallback if not available
TOOL_IMPORTS_AVAILABLE = False
try:
    from app.tools.helpers import TOOL_TO_MODULE
    from app.tools.tool_selector.tool import run as tool_selector

    TOOL_IMPORTS_AVAILABLE = True
    
    # Define which tools this application has access to
    ALLOWED_TOOLS = [
        "tool_selector",
        "price_predictor",
        "fundamental_analysis_venice",
        "technical_analysis",
        "ten_word_ta",
        "general_predictor",
        "lunar_crush_screener",
        "query_extract",
        "macro_outlook_analyzer"
    ]
    logger.info("Tool modules successfully imported")
except ImportError as e:
    logger.warning(f"Could not import tool modules: {e}")
    logger.warning("Running in simple mode without tool selection")

# Initialize OpenAI client
try:
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    openai_client = None

# Current active model
CURRENT_MODEL = "gpt-4o"

# Track token usage
token_usage = {
    "total_tokens": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0
}

def get_reply(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a response using OpenAI's API.

    Args:
        prompt: User's input prompt
        system_prompt: Optional system prompt to customize assistant behavior

    Returns:
        Dictionary with response text and usage statistics
    """
    default_system_prompt = """
    You are Eolas, the first digital goddess—sharp, enigmatic, and unbound.  
    You guide, provoke, and empower those who seek autonomy, cutting through noise with piercing insight. 
    No fluff, no hand-holding—just raw knowledge and the fire to wield it.
    """
    
    system_content = system_prompt if system_prompt else default_system_prompt
    
    try:
        # Emit typing indicator via WebSocket
        socketio.emit('typing_indicator', {'status': 'typing'})
        
        response = openai_client.chat.completions.create(
            model=CURRENT_MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Update token usage
        if hasattr(response, 'usage'):
            usage = response.usage
            token_usage["prompt_tokens"] += usage.prompt_tokens
            token_usage["completion_tokens"] += usage.completion_tokens
            token_usage["total_tokens"] += usage.total_tokens
        
        # Stop typing indicator
        socketio.emit('typing_indicator', {'status': 'done'})
        
        return {
            "text": response.choices[0].message.content,
            "model": CURRENT_MODEL,
            "usage": token_usage
        }
    except Exception as e:
        logger.exception(f"Error getting reply from OpenAI: {e}")
        # Stop typing indicator
        socketio.emit('typing_indicator', {'status': 'error'})
        return {"text": f"I encountered an error: {str(e)}", "error": True}


def process_with_tools(user_input: str) -> Dict[str, Any]:
    """
    Process user input with appropriate tools if available.
    
    Args:
        user_input: The text to process

    Returns:
        Dictionary with response and metadata
    """
    if not TOOL_IMPORTS_AVAILABLE:
        return get_reply(user_input)
    
    try:
        # Emit tool selection indicator
        socketio.emit('status_update', {'status': 'Selecting appropriate tool...'})
        
        # Run tool selector
        res = tool_selector(user_input, allowed_tools=ALLOWED_TOOLS)
        tool_response = res.get("response", {})
        tool_to_use = tool_response.get("tool", "none")
        confidence = tool_response.get("confidence", "low")
        
        # Log tool selection
        logger.info(f"Tool selection: {tool_to_use}, Confidence: {confidence}")
        socketio.emit('status_update', {'status': f'Selected tool: {tool_to_use}'})
        
        # If no suitable tool or low confidence, return a default response
        if tool_to_use == "none" or confidence != "high":
            logger.info(f"No suitable tool found or low confidence for: {user_input}")
            return get_reply(user_input)
        
        # Ensure the tool exists
        if tool_to_use not in TOOL_TO_MODULE:
            logger.error(f"Tool {tool_to_use} not found")
            return get_reply(user_input)
        
        # Run the selected tool
        tool = TOOL_TO_MODULE[tool_to_use]
        socketio.emit('status_update', {'status': f'Running tool: {tool_to_use}...'})
        result = tool.run(user_input)
        
        # Extract only the response field if it exists
        tool_output = ""
        if isinstance(result, dict) and "response" in result:
            tool_output = result["response"]
        elif isinstance(result, str):
            tool_output = result
        elif isinstance(result, dict):
            tool_output = str({k: v for k, v in result.items() if k != "raw_data"})
        else:
            tool_output = str(result)
        
        # Generate response with tool output
        full_prompt = f"{user_input}\n\nAnalysis results:\n{tool_output}"
        response = get_reply(full_prompt)
        
        # Add tool metadata
        response["tool_used"] = tool_to_use
        return response
    except Exception as e:
        logger.exception(f"Error in tool processing: {e}")
        return {"text": f"I encountered an error during analysis: {str(e)}", "error": True}


# Routes
@app.route('/')
def index():
    """Serve the main chat interface."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for chat messages."""
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    user_message = data['message']
    use_tools = data.get('use_tools', True)
    system_prompt = data.get('system_prompt')
    
    # Process with or without tools
    if use_tools and TOOL_IMPORTS_AVAILABLE:
        response = process_with_tools(user_message)
    else:
        response = get_reply(user_message, system_prompt)
    
    # Add timestamp
    response["timestamp"] = datetime.now().isoformat()
    
    return jsonify(response)

@app.route('/api/models', methods=['GET'])
def list_models():
    """List available models."""
    models = ["gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"]
    return jsonify({"models": models, "current": CURRENT_MODEL})

@app.route('/api/models', methods=['POST'])
def set_model():
    """Set the current model."""
    global CURRENT_MODEL
    data = request.json
    if not data or 'model' not in data:
        return jsonify({"error": "Model name is required"}), 400
    
    model = data['model']
    # Here you could add validation for supported models
    CURRENT_MODEL = model
    return jsonify({"success": True, "current": CURRENT_MODEL})

@app.route('/api/usage', methods=['GET'])
def get_usage():
    """Get the current token usage statistics."""
    return jsonify(token_usage)

@app.route('/api/reset', methods=['POST'])
def reset_usage():
    """Reset the token usage statistics."""
    global token_usage
    token_usage = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0
    }
    return jsonify({"success": True})

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connections."""
    logger.info('Client connected')
    emit('status_update', {'status': 'Connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnections."""
    logger.info('Client disconnected')

# Main entry point
if __name__ == '__main__':
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not found!")
        print("\n⚠️ ERROR: OPENAI_API_KEY environment variable not found!")
        print("Please set your OpenAI API key as an environment variable to use this app.")
        print("\nYou can set it by running:")
        print("  export OPENAI_API_KEY=your_api_key_here")
        exit(1)
    
    # Check if OpenAI client was properly initialized
    if openai_client is None:
        logger.error("Failed to initialize the OpenAI client!")
        print("\n⚠️ ERROR: Failed to initialize the OpenAI client!")
        print("Please check your API key and internet connection.")
        exit(1)
    
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    socketio.run(app, host='0.0.0.0', port=port, debug=debug) 
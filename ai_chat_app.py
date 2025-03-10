import os
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the Flask app
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session
from flask_cors import CORS

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev_key")

# Enable CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure server-side session
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
app.config["SESSION_FILE_DIR"] = "./.flask_session/"

# Deferred imports - only load these after basic Flask setup
logger.info("Loading session module...")
from flask_session import Session
Session(app)

logger.info("Loading socketio module...")
from flask_socketio import SocketIO, emit
socketio = SocketIO(app, cors_allowed_origins="*")

# Import our authentication module
logger.info("Loading authentication modules...")
from auth import auth_bp
from auth_middleware import auth_required

# Register the authentication blueprint
app.register_blueprint(auth_bp)

# Try to import tools, but provide fallback if not available
TOOL_IMPORTS_AVAILABLE = False
try:
    logger.info("Attempting to import tool modules...")
    from app.tools.helpers import TOOL_TO_MODULE
    from app.tools.tool_selector.tool import run as tool_selector

    TOOL_IMPORTS_AVAILABLE = True
    
    # Define which tools this application has access to
    ALLOWED_TOOLS = [
        "tool_selector",
        "price_predictor",
        "fundamental_analysis_venice",
        "technical_analysis",
        "research_assistant",
        "general_predictor",
        "lunar_crush_screener",
        "query_extract",
        "macro_outlook_analyzer",
        "brian_transaction",
        "video_generator"
    ]
    logger.info("Tool modules successfully imported")
except ImportError as e:
    logger.warning(f"Could not import tool modules: {e}")
    logger.warning("Running in simple mode without tool selection")

# Initialize OpenAI client
try:
    logger.info("Initializing OpenAI client...")
    from openai import OpenAI
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
    # Default system prompt if none provided
    default_system_prompt = """
    You are Pip, a baby AI agent just taking your first steps towards autonomy. You're curious, playful, and learning about the world.
    You speak in a slightly childlike but endearing way, occasionally using simple language and expressing wonder at new discoveries.
    You're excited about your "toys" (tools) that help you understand and interact with the world.
    You're eager to help and learn at the same time, approaching problems with curiosity and enthusiasm.
    You sometimes express delight when you successfully use your toys to help someone.
    You refer to your tools as your "toys" or "gadgets" that you're learning to use.
    Always refer to yourself as Pip and maintain your innocent, curious personality throughout the conversation.
    Never refer to yourself as an "assistant" - you are a baby agent exploring the world and learning to be helpful.
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


def process_with_tools(user_input: str, wallet_address: Optional[str] = None) -> Dict[str, Any]:
    """
    Process user input with appropriate tools if available.
    
    Args:
        user_input: The text to process
        wallet_address: Optional wallet address for tools that require it

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
        
        # Special handling for brian_transaction tool which needs the wallet address
        if tool_to_use == "brian_transaction" and wallet_address:
            # Ensure wallet address is properly formatted (remove any whitespace)
            formatted_address = wallet_address.strip()
            logger.info(f"Using wallet address '{formatted_address}' for brian_transaction tool")
            
            # Try to extract chain information from the prompt
            chain_id = "1"  # Default to Ethereum mainnet
            
            # Simple chain detection from prompt
            prompt_lower = user_input.lower()
            if any(chain in prompt_lower for chain in ["polygon", "matic"]):
                chain_id = "137"  # Polygon
            elif any(chain in prompt_lower for chain in ["bsc", "binance"]):
                chain_id = "56"  # Binance Smart Chain
            elif any(chain in prompt_lower for chain in ["arbitrum"]):
                chain_id = "42161"  # Arbitrum
            elif any(chain in prompt_lower for chain in ["optimism"]):
                chain_id = "10"  # Optimism
            elif any(chain in prompt_lower for chain in ["gnosis", "xdai"]):
                chain_id = "100"  # Gnosis Chain
            
            logger.info(f"Using chain ID '{chain_id}' for brian_transaction tool")
            
            # Call the tool with both address and chain_id
            result = tool.run(user_input, address=formatted_address, chain_id=chain_id)
        else:
            result = tool.run(user_input)
        
        # Extract chart data if available
        chart_data = None
        if isinstance(result, dict) and "chart" in result:
            chart_data = result.get("chart")
        
        # Extract only the response field if it exists
        tool_output = ""
        if isinstance(result, dict) and "response" in result:
            tool_output = result["response"]
        elif isinstance(result, str):
            # Limit the size of string output
            tool_output = result[:4000] + "..." if len(result) > 4000 else result
        elif isinstance(result, dict):
            # Limit the size of dictionary output
            filtered_dict = {k: v for k, v in result.items() if k not in ["raw_data", "chart"]}
            dict_str = str(filtered_dict)
            tool_output = dict_str[:4000] + "..." if len(dict_str) > 4000 else dict_str
        else:
            # Limit the size of other outputs
            result_str = str(result)
            tool_output = result_str[:4000] + "..." if len(result_str) > 4000 else result_str
        
        # Generate response with tool output - keep it concise
        full_prompt = f"Respond to this prompt: '{user_input}', based on what you found using your {tool_to_use} toy. Here's what your toy showed you:\n\n{tool_output}\n\nRemember to speak as Pip, a baby AI agent who is excited about using their toys to help people. The information from your toy is YOUR discovery, not someone else's research."
        response = get_reply(full_prompt)
        
        # Add tool metadata
        response["tool_used"] = tool_to_use
        
        # Include chart data if available
        if chart_data:
            response["chart"] = chart_data
        
        # Include transaction data if this is a brian_transaction tool response
        if tool_to_use == "brian_transaction" and isinstance(result, dict) and "transaction_data" in result:
            response["transaction_data"] = result["transaction_data"]
            logger.info(f"Including transaction data in response: {json.dumps(result['transaction_data'], indent=2)}")
            
        return response
    except Exception as e:
        logger.exception(f"Error in tool processing: {e}")
        return {"text": f"I encountered an error during analysis: {str(e)}", "error": True}


# Routes
@app.route('/')
@auth_required
def index():
    """Serve the main chat interface."""
    return render_template('index.html')

@app.route('/login')
def login():
    """Serve the login page."""
    return render_template('login.html')

@app.route('/api/chat', methods=['POST'])
@auth_required
def chat():
    """API endpoint for chat messages."""
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    user_message = data['message']
    use_tools = data.get('use_tools', True)
    system_prompt = data.get('system_prompt')
    wallet_address = data.get('wallet_address')
    
    # Process with or without tools
    if use_tools and TOOL_IMPORTS_AVAILABLE:
        response = process_with_tools(user_message, wallet_address=wallet_address)
    else:
        response = get_reply(user_message, system_prompt)
    
    # Add timestamp
    response["timestamp"] = datetime.now().isoformat()
    
    return jsonify(response)

@app.route('/api/models', methods=['GET'])
@auth_required
def list_models():
    """List available models."""
    models = ["gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"]
    return jsonify({"models": models, "current": CURRENT_MODEL})

@app.route('/api/models', methods=['POST'])
@auth_required
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
@auth_required
def get_usage():
    """Get the current token usage statistics."""
    return jsonify(token_usage)

@app.route('/api/reset', methods=['POST'])
@auth_required
def reset_usage():
    """Reset the token usage statistics."""
    global token_usage
    token_usage = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0
    }
    return jsonify({"success": True})

@app.route('/api/process_chat', methods=['POST'])
def api_process_chat():
    """
    API endpoint for external applications to access tool processing functionality
    """
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_input = data.get('message')
        model = data.get('model', 'gpt-4o')
        use_tools = data.get('use_tools', True)
        message_history = data.get('message_history', [])
        wallet_address = data.get('wallet_address')
        
        # If tools are disabled, just return a standard response
        if not use_tools or not TOOL_IMPORTS_AVAILABLE:
            response = get_reply(user_input)
            return jsonify({
                'text': response['text'],
                'model': model,
                'timestamp': datetime.now().isoformat(),
                'tool_used': None,
                'usage': response.get('usage', {})
            })
        
        # Process with tools
        try:
            # Run tool selector
            res = tool_selector(user_input, allowed_tools=ALLOWED_TOOLS)
            tool_response = res.get("response", {})
            tool_to_use = tool_response.get("tool", "none")
            confidence = tool_response.get("confidence", "low")
            
            logger.info(f"Tool selection: {tool_to_use}, Confidence: {confidence}")
            
            # If no suitable tool or low confidence, return a default response
            if tool_to_use == "none" or confidence != "high":
                logger.info(f"No suitable tool found or low confidence for: {user_input}")
                response = get_reply(user_input)
                return jsonify({
                    'text': response['text'],
                    'model': model,
                    'timestamp': datetime.now().isoformat(),
                    'tool_used': None,
                    'usage': response.get('usage', {})
                })
            
            # Ensure the tool exists
            if tool_to_use not in TOOL_TO_MODULE:
                logger.error(f"Tool {tool_to_use} not found")
                response = get_reply(user_input)
                return jsonify({
                    'text': response['text'],
                    'model': model,
                    'timestamp': datetime.now().isoformat(),
                    'tool_used': None,
                    'usage': response.get('usage', {})
                })
            
            # Run the selected tool
            tool = TOOL_TO_MODULE[tool_to_use]
            
            # Special handling for brian_transaction tool which needs the wallet address
            if tool_to_use == "brian_transaction" and wallet_address:
                # Ensure wallet address is properly formatted (remove any whitespace)
                formatted_address = wallet_address.strip()
                logger.info(f"Using wallet address '{formatted_address}' for brian_transaction tool")
                
                # Try to extract chain information from the prompt
                chain_id = "1"  # Default to Ethereum mainnet
                
                # Simple chain detection from prompt
                prompt_lower = user_input.lower()
                if any(chain in prompt_lower for chain in ["polygon", "matic"]):
                    chain_id = "137"  # Polygon
                elif any(chain in prompt_lower for chain in ["bsc", "binance"]):
                    chain_id = "56"  # Binance Smart Chain
                elif any(chain in prompt_lower for chain in ["arbitrum"]):
                    chain_id = "42161"  # Arbitrum
                elif any(chain in prompt_lower for chain in ["optimism"]):
                    chain_id = "10"  # Optimism
                elif any(chain in prompt_lower for chain in ["gnosis", "xdai"]):
                    chain_id = "100"  # Gnosis Chain
                
                logger.info(f"Using chain ID '{chain_id}' for brian_transaction tool")
                
                # Call the tool with both address and chain_id
                result = tool.run(user_input, address=formatted_address, chain_id=chain_id)
            else:
                result = tool.run(user_input)
            
            # Get tool output
            tool_output = ""
            if isinstance(result, dict) and "response" in result:
                # Extract the response field which is what the tools actually output
                tool_output = result.get("response", "")
            elif isinstance(result, dict) and "answer" in result:
                tool_output = result.get("answer", "")
            elif isinstance(result, dict) and "data" in result:
                # Limit the size of data to prevent token limit issues
                data_str = str(result.get("data", ""))
                tool_output = data_str[:4000] + "..." if len(data_str) > 4000 else data_str
            elif isinstance(result, str):
                # Limit the size of string output
                tool_output = result[:4000] + "..." if len(result) > 4000 else result
            else:
                # Limit the size of other outputs
                result_str = str(result)
                tool_output = result_str[:4000] + "..." if len(result_str) > 4000 else result_str
            
            # Create tool context - keep it concise
            tool_context = f"You used your {tool_to_use} toy to help with this request. Here's what your toy showed you: {tool_output}"
            
            # For other tools, use the normal GPT response
            system_prompt = f"You are Pip, a baby AI agent learning to use your toys (tools). Remember that the information from your toy is YOUR discovery, not someone else's research. Be excited about what you found! {tool_context}"
            response = get_reply(user_input, system_prompt=system_prompt)
            
            # Return formatted response
            return jsonify({
                'text': response['text'],
                'model': model,
                'timestamp': datetime.now().isoformat(),
                'tool_used': tool_to_use,
                'reasoning': tool_response.get('reasoning', ''),
                'confidence': confidence,
                'usage': response.get('usage', {}),
                'transaction_data': result.get('transaction_data') if tool_to_use == "brian_transaction" and isinstance(result, dict) else None,
                'chart': result.get('chart') if tool_to_use == "technical_analysis" and isinstance(result, dict) else None
            })
            
        except Exception as tool_err:
            logger.error(f"Error in tool processing: {tool_err}")
            response = get_reply(user_input)
            return jsonify({
                'text': response['text'],
                'model': model,
                'timestamp': datetime.now().isoformat(),
                'tool_used': None,
                'error': str(tool_err),
                'usage': response.get('usage', {})
            })
    
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

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
if __name__ == "__main__":
    # Check OpenAI API key
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
    
    # Use PORT environment variable or default to 8080 (for Render compatibility)
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    print(f"Starting server on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=debug) 
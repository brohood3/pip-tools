"""
Configuration module for application settings and defaults.
"""
import os
from app.utils.dotenv import ensure_var
from litellm import completion

# Default model configuration
DEFAULT_MODEL = "gemini-2.0-flash"  # Default model to use for all operations

# Environment variable override
if os.environ.get("DEFAULT_MODEL"):
    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL")

# API Keys
OPENAI_API_KEY = ensure_var("OPENAI_API_KEY") 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure LiteLLM with API keys
if GEMINI_API_KEY:
    # Set all possible environment variables that LiteLLM might check for Google auth
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    os.environ["VERTEX_AI_API_KEY"] = GEMINI_API_KEY

# LiteLLM configuration
LITELLM_PROVIDER_MAP = {
    # OpenAI models
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    
    # Anthropic models
    "claude-3-opus": "anthropic/claude-3-opus-20240229",
    "claude-3-sonnet": "anthropic/claude-3-sonnet-20240229",
    "claude-3-haiku": "anthropic/claude-3-haiku-20240307",
    
    # Google models - Direct Gemini API format
    "gemini-pro": "gemini/gemini-pro",
    "gemini-1.5-pro": "gemini/gemini-1.5-pro",
    "gemini-1.5-flash": "gemini/gemini-1.5-flash",
    "gemini-2.0-pro": "gemini/gemini-2.0-pro", 
    "gemini-2.0-flash": "gemini/gemini-2.0-flash",
    
    # Mistral models
    "mistral-large": "mistral/mistral-large-latest",
    "mistral-medium": "mistral/mistral-medium-latest",
    "mixtral-8x7b": "mistral/mixtral-8x7b-instruct-v0.1",
    
    # Meta models
    "llama-3-70b": "meta/llama-3-70b-instruct",
    "llama-3-8b": "meta/llama-3-8b-instruct",
}

def get_model_provider(model_name: str) -> str:
    """
    Get the full provider/model string for LiteLLM based on the model name.
    
    Args:
        model_name: The name of the model to use
        
    Returns:
        The full provider/model string for LiteLLM
    """
    return LITELLM_PROVIDER_MAP.get(model_name, model_name) 
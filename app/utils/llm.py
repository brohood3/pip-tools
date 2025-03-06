"""
LLM utility functions for interacting with various language models through LiteLLM.
"""

from typing import List, Dict, Any, Optional, Union
import os
from litellm import completion
from app.utils.config import get_model_provider, DEFAULT_MODEL
from app.utils.logging import logger
import json

def generate_completion(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    json_mode: bool = False,
    json_schema: Optional[Dict[str, Any]] = None,
) -> Union[str, Dict[str, Any]]:
    """
    Generate a completion using LiteLLM with the specified model.
    
    Args:
        prompt: The user prompt to send to the model
        system_prompt: Optional system prompt to set context
        model: The model to use (defaults to DEFAULT_MODEL)
        temperature: Controls randomness (0-1)
        max_tokens: Maximum number of tokens to generate
        stream: Whether to stream the response
        json_mode: Whether to request JSON output
        json_schema: Optional schema for structured JSON output (for Gemini models)
        
    Returns:
        The generated completion text or streaming response
    """
    try:
        # Use the provided model or fall back to default
        model_name = model or DEFAULT_MODEL
        
        # Get the full provider/model string
        provider_model = get_model_provider(model_name)
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Set up parameters
        params = {
            "model": provider_model,
            "messages": messages,
            "temperature": temperature,
        }
        
        # Add optional parameters
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        # Special handling for Gemini models
        if model_name.startswith("gemini-"):
            # For Gemini models, explicitly pass the API key
            gemini_api_key = os.getenv("GEMINI_API_KEY_2")
            if gemini_api_key:
                params["api_key"] = gemini_api_key
            
            # For Gemini models with JSON mode, use the appropriate config
            if json_mode:
                extra_body = {
                    "response_mime_type": "application/json"
                }
                
                # If a schema is provided, add it to the configuration
                if json_schema:
                    extra_body["response_schema"] = json_schema
                
                params["extra_body"] = extra_body
                
                # Remove standard OpenAI response_format if it was added
                if "response_format" in params:
                    del params["response_format"]
        else:
            # For OpenAI and other models, use the standard response_format
            if json_mode:
                params["response_format"] = {"type": "json_object"}
        
        # Log the request
        logger.info(f"Generating completion with model: {provider_model}")
        logger.info(f"Request parameters: {params}")
        
        # Make the API call
        response = completion(**params)
        
        if stream:
            return response
        else:
            # For Gemini models with JSON mode, we might need to clean up the response
            content = response.choices[0].message.content
            if json_mode and model_name.startswith("gemini-") and isinstance(content, str):
                # Check if the response is wrapped in markdown code blocks
                if content.startswith("```") and "```" in content:
                    # Extract the JSON content from the markdown code block
                    content = content.split("```")[1]
                    if content.startswith("json\n"):
                        content = content[5:]  # Remove "json\n" prefix
                    
                    # Try to parse the JSON
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        # If parsing fails, return the original content
                        return content
            
            return content
            
    except Exception as e:
        logger.error(f"Error generating completion: {str(e)}")
        raise 
"""
Research Assistant Tool

A simple tool that uses Perplexity API to fetch up-to-date information
for general knowledge queries.
"""

import os
from typing import Dict, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import HTTPException

# Load environment variables
load_dotenv()


class ResearchAssistant:
    """Simple research tool using Perplexity API."""

    def __init__(self):
        """Initialize with Perplexity API client."""
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

        if not self.perplexity_api_key:
            raise ValueError("Missing required PERPLEXITY_API_KEY")

        self.perplexity_client = OpenAI(
            api_key=self.perplexity_api_key, 
            base_url="https://api.perplexity.ai"
        )

    def run(
        self, 
        query: str, 
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make a single research query using Perplexity API.

        Args:
            query: The research query to answer
            system_prompt: Optional custom system prompt
            model: Optional model parameter (ignored for Perplexity API calls)

        Returns:
            Dict containing response and metadata
        """
        try:
            # Default system prompt for research
            default_system_prompt = (
                "You are a helpful research assistant that provides accurate, "
                "up-to-date information. Provide comprehensive answers with factual "
                "information. When relevant, include sources, dates, and context. "
                "If information might be outdated or uncertain, acknowledge this clearly."
            )
            
            # Always use sonar-reasoning for Perplexity
            perplexity_model = "sonar-reasoning"
            
            response = self.perplexity_client.chat.completions.create(
                model=perplexity_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt or default_system_prompt,
                    },
                    {"role": "user", "content": query},
                ],
            )
            
            # Prepare response data
            result = response.choices[0].message.content
            
            # Return response with metadata
            return {
                "response": result,
                "metadata": {
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "model": perplexity_model,
                    "requested_model": model  # Store the originally requested model for reference
                }
            }
            
        except Exception as e:
            return {"error": str(e)}


# Function to be called from the API
def run(
    query: str, 
    system_prompt: Optional[str] = None, 
    model: Optional[str] = None
) -> Dict[str, Any]:
    return ResearchAssistant().run(query, system_prompt, model) 
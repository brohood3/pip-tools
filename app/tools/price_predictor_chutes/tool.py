"""
Price Predictor Tool (Chutes version)

AI-powered price prediction tool that combines research from multiple sources
to generate detailed price predictions for cryptocurrency and market-related questions.
"""

import os
from typing import Dict, Optional, Any, TypedDict
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import HTTPException
import httpx

# Load environment variables
load_dotenv()


class ResearchResults(TypedDict):
    """TypedDict defining the structure of research results."""

    context: str
    factors: str
    dates: str
    alternatives: str
    existing_predictions: str


class PricePredictor:
    """Price prediction tool using OpenAI and Perplexity APIs."""

    def __init__(self):
        """Initialize with API clients and configuration."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.chutes_api_key = os.getenv("CHUTES_API_KEY")

        if not all([self.openai_api_key, self.perplexity_api_key, self.chutes_api_key]):
            raise ValueError("Missing required API keys")

        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.perplexity_client = OpenAI(
            api_key=self.perplexity_api_key, base_url="https://api.perplexity.ai"
        )
        self.chutes_client = OpenAI(
            api_key=self.chutes_api_key,
            base_url="https://chutes-qwen-qwen2-5-72b-instruct.chutes.ai/v1"
        )

    def _get_time_context(self) -> str:
        """Generate current time context for research queries."""
        now = datetime.now()
        return f"""Current time context:
- Current date: {now.strftime('%Y-%m-%d')}
- Current year: {now.year}
- Current month: {now.strftime('%B')}"""

    def run(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Main entry point for the price predictor tool.

        Args:
            prompt: User's prediction request
            system_prompt: Optional custom system prompt for the final prediction

        Returns:
            Dict containing prediction results and metadata
        """
        try:
            # Get research from Perplexity
            research = self._get_research(prompt)
            if not research:
                return {"error": "Failed to gather research data"}

            # Generate prediction with Chutes
            prediction = self._generate_prediction(prompt, research, system_prompt)
            if not prediction:
                return {"error": "Failed to generate prediction"}

            # Store context in metadata
            metadata = {
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "research": research
            }

            return {"response": prediction, "metadata": metadata}

        except Exception as e:
            print(f"Error in run method: {str(e)}")
            return {"error": str(e)}

    def _get_research(self, prompt: str) -> Optional[str]:
        """Get comprehensive research from Perplexity."""
        try:
            print("Making Perplexity API call for research...")
            time_context = self._get_time_context()
            research_prompt = f"""Analyze the price prediction request for: {prompt}

Provide a comprehensive but concise analysis covering:
1. Current price, market cap, and recent performance
2. Key factors influencing the price (technical, fundamental, sentiment)
3. Important dates and events that could impact price
4. Different price scenarios (bull, base, bear cases)
5. Expert predictions and technical levels

{time_context}

Format the response in clear sections with bullet points. Focus on specific numbers and data. Be concise."""

            response = self.perplexity_client.chat.completions.create(
                model="sonar-reasoning",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cryptocurrency research analyst. Provide accurate, data-driven analysis with specific numbers and clear insights."
                    },
                    {"role": "user", "content": research_prompt}
                ],
                max_tokens=1000,  # Limit research output
                temperature=0.7
            )
            print(f"Perplexity response tokens: {response.usage.completion_tokens}")
            print("Perplexity API call successful")
            research_content = response.choices[0].message.content
            print(f"Research length: {len(research_content)} characters")
            return research_content
        except Exception as e:
            print(f"Error in research: {str(e)}")
            return None

    def _generate_prediction(
        self, question: str, research: str, system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """Generate the final prediction using Chutes."""
        try:
            print("Making Chutes API call for prediction...")
            time_context = self._get_time_context()
            prompt = f"""Based on this research, provide a detailed price prediction for: {question}

RESEARCH:
{research}

{time_context}

Provide your analysis in this format:

MAIN PREDICTION:
- Price range with timeframe
- Confidence level (1-10)
- Key drivers

SCENARIOS:
1. Bull Case
2. Base Case
3. Base Case

KEY LEVELS:
- Support
- Resistance"""

            default_system_prompt = """You are a precise analytical engine specializing in price predictions. 
Your goal is to make decisive predictions with clear price targets and probabilities. 
Always provide specific numbers and ranges."""

            print(f"Prompt length: {len(prompt)} characters")
            completion = self.chutes_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt if system_prompt else default_system_prompt,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                stream=False,
                max_tokens=1000  # Limit prediction output
            )
            print(f"Chutes response tokens: {completion.usage.completion_tokens}")
            print("Chutes API call successful")
            prediction_content = completion.choices[0].message.content
            print(f"Prediction length: {len(prediction_content)} characters")
            return prediction_content

        except Exception as e:
            print(f"Error generating prediction: {str(e)}")
            return None


# added the following to have uniformity in the way we call tools
def run(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    return PricePredictor().run(prompt, system_prompt)

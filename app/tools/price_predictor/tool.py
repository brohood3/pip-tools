"""
Price Predictor Tool

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

        if not all([self.openai_api_key, self.perplexity_api_key]):
            raise ValueError("Missing required API keys")

        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.perplexity_client = OpenAI(
            api_key=self.perplexity_api_key, base_url="https://api.perplexity.ai"
        )

    def run(self, prompt: str) -> Dict[str, Any]:
        """Main entry point for the price predictor tool.

        Args:
            prompt: User's prediction request

        Returns:
            Dict containing prediction results and metadata
        """
        try:
            # Gather research
            research_results = self._gather_research(prompt)
            if not research_results:
                return {"error": "Failed to gather research data"}

            # Generate prediction
            prediction = self._generate_prediction(prompt, research_results)
            if not prediction:
                return {"error": "Failed to generate prediction"}

            # Store all context in metadata
            metadata = {
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "research_results": research_results,
            }

            return {"response": prediction, "metadata": metadata}

        except Exception as e:
            return {"error": str(e)}

    def _get_research(self, prompt: str) -> str:
        """Make a research query using Perplexity."""
        try:
            response = self.perplexity_client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research assistant focused on providing accurate, well-sourced information. For token price questions, always include current price, recent price action, and market sentiment. Be direct and concise.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in research query: {e}")
            return f"Research failed: {str(e)}"

    def _gather_research(self, question: str) -> Optional[ResearchResults]:
        """Gather all research components."""
        try:
            return {
                "context": self._research_context(question),
                "factors": self._research_factors(question),
                "dates": self._research_dates(question),
                "alternatives": self._research_alternatives(question),
                "existing_predictions": self._research_existing_predictions(question),
            }
        except Exception as e:
            print(f"Error gathering research: {e}")
            return None

    def _research_context(self, question: str) -> str:
        """Research the general context and background."""
        prompt = f"""Analyze the current price context for: {question}

Focus on:
- Current price and market cap
- Recent price movements (7d, 30d, YTD)
- Trading volume trends and patterns
- Market sentiment indicators
- Price correlation with major assets
- Technical analysis indicators
- Market structure and liquidity"""
        return self._get_research(prompt)

    def _research_factors(self, question: str) -> str:
        """Research key influencing factors."""
        prompt = f"""What are the main factors that will influence the price of: {question}

Analyze:
- Token utility and adoption metrics
- Project development activity
- Competition and market positioning
- Upcoming token events (unlocks, burns, etc.)
- Market correlation factors (BTC, ETH)
- Trading volume and liquidity metrics
- Institutional interest
- Market sentiment indicators
- Technical analysis patterns"""
        return self._get_research(prompt)

    def _research_dates(self, question: str) -> str:
        """Research relevant dates and timelines."""
        prompt = f"""What are the critical dates that could impact the price of: {question}

Focus on:
- Upcoming protocol updates
- Token unlock schedules
- Partnership announcements
- Market events that could impact price
- Historical price action dates
- Technical analysis timeframes
- Network upgrade schedules
- Major ecosystem events"""
        return self._get_research(prompt)

    def _research_alternatives(self, question: str) -> str:
        """Research alternative price scenarios."""
        prompt = f"""What are the most likely price scenarios for: {question}

Analyze:
- Bull case: What could drive significant price upside?
  * Target price levels
  * Required conditions
  * Probability assessment
  * Timeline expectations

- Base case: Most likely price trajectory
  * Expected trading range
  * Key support/resistance levels
  * Volume expectations
  * Market conditions

- Bear case: Major risks and potential downsides
  * Price floor estimates
  * Risk factors
  * Warning signals
  * Market conditions"""
        return self._get_research(prompt)

    def _research_existing_predictions(self, question: str) -> str:
        """Research existing price predictions and expert opinions."""
        prompt = f"""What specific price predictions exist for: {question}

Gather:
- Analyst price targets
- Technical analysis forecasts
- On-chain metric projections
- Market maker positioning
- Options market implications
- Community sentiment levels
- Institutional forecasts
- Historical price patterns"""
        return self._get_research(prompt)

    def _create_prediction_prompt(
        self, question: str, research: ResearchResults
    ) -> str:
        """Create the final prediction prompt."""
        return f"""Based on the following research, provide a detailed price prediction. You MUST provide specific price ranges with probabilities, even if confidence is low.

QUESTION: {question}

RESEARCH FINDINGS:

1. Price Context and Background:
{research['context']}

2. Key Price Influencing Factors:
{research['factors']}

3. Relevant Dates and Timelines:
{research['dates']}

4. Alternative Price Scenarios:
{research['alternatives']}

5. Expert Price Predictions:
{research['existing_predictions']}

Provide your analysis in this format:

MAIN PRICE PREDICTION:
[Specific price range with timeframe, e.g., "$X to $Y by [date]" with probability]

KEY FACTORS DRIVING THIS PREDICTION:
- [Most important factor]
- [Second most important factor]
- [Third most important factor]

PRICE SCENARIOS:
1. Bull Case: $[price] (probability: X%)
   - Key drivers
   - Required conditions
   - Technical levels
   - Timeline

2. Base Case: $[price] (probability: Y%)
   - Key drivers
   - Required conditions
   - Technical levels
   - Timeline

3. Bear Case: $[price] (probability: Z%)
   - Key drivers
   - Required conditions
   - Technical levels
   - Timeline

CONFIDENCE LEVEL: [1-10]
[If confidence is low (1-4), explain why but still provide specific price targets]

TIME HORIZON: [Specific timeframe for the prediction]

CRITICAL PRICE RISKS:
- [Key risk factor 1]
- [Key risk factor 2]
- [Key risk factor 3]

KEY PRICE LEVELS TO WATCH:
- Resistance: $[level 1], $[level 2]
- Support: $[level 1], $[level 2]
- Volume zones: $[level 1], $[level 2]"""

    def _generate_prediction(
        self, question: str, research: ResearchResults
    ) -> Optional[str]:
        """Generate the final prediction using OpenAI."""
        try:
            prompt = self._create_prediction_prompt(question, research)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a precise analytical engine specializing in price predictions. Your goal is to make decisive predictions with clear, differentiated probabilities.

PROBABILITY GUIDELINES:
- Never use 50% as it indicates complete uncertainty
- Base case should be your highest probability scenario (35-65%)
- Bull and bear cases should have differentiated probabilities that reflect your analysis
- Total probabilities must sum to 100%
- Use narrower probability ranges (e.g., 42% not 40-45%)
- Back up probability assignments with specific factors from your research

Always provide specific numerical predictions with ranges. If uncertain, explain the uncertainties but still provide your best assessment based on the available data.""",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating prediction: {e}")
            return None


# added the following to have uniformity in the way we call tools
def run(prompt: str) -> Dict[str, Any]:
    return PricePredictor().run(prompt)

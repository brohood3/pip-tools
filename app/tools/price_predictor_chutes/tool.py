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
        self.chutes_api_key = os.getenv("CHUTES_API_KEY")

        if not all([self.openai_api_key, self.perplexity_api_key, self.chutes_api_key]):
            raise ValueError("Missing required API keys")

        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.perplexity_client = OpenAI(
            api_key=self.perplexity_api_key, base_url="https://api.perplexity.ai"
        )
        self.chutes_client = OpenAI(
            api_key=self.chutes_api_key,
            base_url="https://chutes-unsloth-llama-3-3-70b-instruct.chutes.ai/v1"
        )

    def _get_time_context(self) -> str:
        """Generate current time context for research queries."""
        now = datetime.now()
        return f"""Current time context:
- Current date: {now.strftime('%Y-%m-%d')}
- Current year: {now.year}
- Current month: {now.strftime('%B')}
Please ensure all analysis and predictions are made with this current time context in mind."""

    def run(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Main entry point for the price predictor tool.

        Args:
            prompt: User's prediction request
            system_prompt: Optional custom system prompt for the final prediction

        Returns:
            Dict containing prediction results and metadata
        """
        try:
            # Gather research
            research_results = self._gather_research(prompt)
            if not research_results:
                return {"error": "Failed to gather research data"}

            # Generate prediction
            prediction = self._generate_prediction(prompt, research_results, system_prompt)
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
            time_context = self._get_time_context()
            response = self.perplexity_client.chat.completions.create(
                model="sonar-reasoning",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a research assistant focused on providing accurate, well-sourced information. For token price questions, always include current price, recent price action, and market sentiment. Be direct and concise.

{time_context}

Always ensure your research and analysis is current and relevant to the present date.""",
                    },
                    {"role": "user", "content": f"{prompt}\n\nNote: Please consider the current date and time context in your analysis."},
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
[Specific price range with timeframe, e.g., "$X to $Y by [date]"]

KEY FACTORS DRIVING THIS PREDICTION:
- [Most important factor]
- [Second most important factor]
- [Third most important factor]

PRICE SCENARIOS:
1. Bull Case: $[price]
   - Key drivers
   - Required conditions
   - Technical levels
   - Timeline

2. Base Case: $[price]
   - Key drivers
   - Required conditions
   - Technical levels
   - Timeline

3. Bear Case: $[price]
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
        self, question: str, research: ResearchResults, system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """Generate the final prediction using Chutes."""
        try:
            prompt = self._create_prediction_prompt(question, research)
            time_context = self._get_time_context()

            default_system_prompt = f"""You are a precise analytical engine specializing in price predictions. Your goal is to make decisive predictions with clear price targets and differentiated probabilities.

{time_context}

Always provide specific price targets with clear ranges. If uncertain, explain the uncertainties but still provide your best price targets based on the available data and research. Ensure all predictions and analysis are made within the current time context.

If your confidence level is 4 or lower, start your response with a note suggesting the user to provide more context about the token/project (e.g., full project name, website, or other identifying information) to get a more accurate prediction."""

            completion = self.chutes_client.chat.completions.create(
                model="unsloth/Llama-3.3-70B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt if system_prompt else default_system_prompt,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                stream=False
            )

            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error generating prediction: {e}")
            return None


# added the following to have uniformity in the way we call tools
def run(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    return PricePredictor().run(prompt, system_prompt)

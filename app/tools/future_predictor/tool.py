"""
Future Predictor Tool

AI-powered future prediction tool that combines research from multiple sources
to generate detailed predictions for cryptocurrency and market-related questions.
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

class FuturePredictor:
    """Future prediction tool using OpenAI and Perplexity APIs."""
    
    def __init__(self):
        """Initialize with API clients and configuration."""
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
        
        if not all([self.openai_api_key, self.perplexity_api_key]):
            raise ValueError("Missing required API keys")
            
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.perplexity_client = OpenAI(
            api_key=self.perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )

    def run(self, prompt: str) -> Dict[str, Any]:
        """Main entry point for the future predictor tool.
        
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
                "research_results": research_results
            }
            
            return {
                "response": prediction,
                "metadata": metadata
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _get_research(self, prompt: str) -> str:
        """Make a research query using Perplexity."""
        try:
            response = self.perplexity_client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=[
                    {"role": "system", "content": "You are a research assistant focused on providing accurate, well-sourced information. For token price questions, always include current price, recent price action, and market sentiment. Be direct and concise."},
                    {"role": "user", "content": prompt}
                ]
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
                "existing_predictions": self._research_existing_predictions(question)
            }
        except Exception as e:
            print(f"Error gathering research: {e}")
            return None

    def _research_context(self, question: str) -> str:
        """Research the general context and background."""
        prompt = f"""Analyze the current context for: {question}
For token prices, include:
- Current price and market cap
- Recent price movements (7d, 30d)
- Trading volume trends
- Market sentiment indicators

For other predictions:
- Current state of affairs
- Recent significant developments
- Key market or industry trends"""
        return self._get_research(prompt)

    def _research_factors(self, question: str) -> str:
        """Research key influencing factors."""
        prompt = f"""What are the main factors that will influence: {question}

For token prices, analyze:
- Token utility and adoption metrics
- Project development activity
- Competition and market positioning
- Upcoming token events (unlocks, burns, etc.)
- Market correlation factors (BTC, ETH, sector trends)

For other predictions:
- Economic factors
- Technical developments
- Regulatory considerations
- Market dynamics"""
        return self._get_research(prompt)

    def _research_dates(self, question: str) -> str:
        """Research relevant dates and timelines."""
        prompt = f"""What are the critical dates and milestones relevant to: {question}

For token prices:
- Upcoming protocol updates
- Token unlock schedules
- Partnership announcements
- Market events that could impact price
- Historical price action dates

For other predictions:
- Key upcoming events
- Development milestones
- Regulatory deadlines"""
        return self._get_research(prompt)

    def _research_alternatives(self, question: str) -> str:
        """Research alternative scenarios."""
        prompt = f"""What are the most likely scenarios for: {question}

For token prices:
- Bull case: What could drive significant upside?
- Base case: Most likely price trajectory
- Bear case: Major risks and potential downsides
Include specific price levels for each case.

For other predictions:
- Best case scenario
- Most likely outcome
- Worst case scenario"""
        return self._get_research(prompt)

    def _research_existing_predictions(self, question: str) -> str:
        """Research existing predictions and expert opinions."""
        prompt = f"""What specific predictions exist for: {question}

For token prices:
- Analyst price targets
- Community sentiment and predictions
- Technical analysis forecasts
- On-chain metric projections

For other predictions:
- Expert forecasts
- Industry analysis
- Market consensus"""
        return self._get_research(prompt)

    def _create_prediction_prompt(self, question: str, research: ResearchResults) -> str:
        """Create the final prediction prompt."""
        return f"""Based on the following research, provide a detailed prediction for this question. You MUST provide specific predictions with numerical ranges, even if confidence is low.

QUESTION: {question}

RESEARCH FINDINGS:

1. Context and Background:
{research['context']}

2. Key Influencing Factors:
{research['factors']}

3. Relevant Dates and Timelines:
{research['dates']}

4. Alternative Scenarios:
{research['alternatives']}

5. Expert Predictions:
{research['existing_predictions']}

For price predictions, your response MUST include specific price ranges. Even with low confidence, provide your best estimate based on the available data.

Provide your analysis in this format:

MAIN PREDICTION:
[For prices: Specific price range with timeframe, e.g., "$X to $Y by [date]" with probability]
[For other questions: Clear prediction with probability range]

KEY FACTORS DRIVING THIS PREDICTION:
- [Most important factor]
- [Second most important factor]
- [Third most important factor]

PRICE SCENARIOS (for price predictions):
1. Optimistic Case: $[price] (probability: X%) - [key drivers]
2. Base Case: $[price] (probability: Y%) - [key drivers]
3. Pessimistic Case: $[price] (probability: Z%) - [key drivers]

ALTERNATIVE SCENARIOS (for non-price predictions):
1. Optimistic Case (probability): [description]
2. Base Case (probability): [description]
3. Pessimistic Case (probability): [description]

CONFIDENCE LEVEL: [1-10]
[If confidence is low (1-4), explain why but still provide specific predictions]

TIME HORIZON: [Specific timeframe for the prediction]

CRITICAL UNCERTAINTIES:
- [Key risk factor 1]
- [Key risk factor 2]
- [Key risk factor 3]"""

    def _generate_prediction(self, question: str, research: ResearchResults) -> Optional[str]:
        """Generate the final prediction using OpenAI."""
        try:
            prompt = self._create_prediction_prompt(question, research)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a precise analytical engine specializing in future predictions, particularly for token prices. Always provide specific numerical predictions with ranges, even with low confidence. Never avoid making a prediction - if uncertain, provide a wider range and explain the uncertainties."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating prediction: {e}")
            return None 
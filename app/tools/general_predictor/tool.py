"""
General Predictor Tool

AI-powered prediction tool that combines research from multiple sources
to generate detailed predictions for market-related questions, excluding price predictions.
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


class GeneralPredictor:
    """General prediction tool using OpenAI and Perplexity APIs."""

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

    def run(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Main entry point for the general predictor tool.

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
            response = self.perplexity_client.chat.completions.create(
                model="sonar-reasoning",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research assistant focused on providing accurate, well-sourced information. Focus on market trends, developments, and industry analysis. Be direct and concise.",
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
        prompt = f"""Analyze the current context for: {question}

Focus on:
- Current state of affairs
- Recent significant developments
- Key market or industry trends
- Relevant stakeholders and their positions
- Historical context and patterns"""
        return self._get_research(prompt)

    def _research_factors(self, question: str) -> str:
        """Research key influencing factors."""
        prompt = f"""What are the main factors that will influence: {question}

Analyze:
- Economic factors
- Technical developments
- Regulatory considerations
- Market dynamics
- Industry-specific trends
- Competitive landscape
- Stakeholder interests
- External influences"""
        return self._get_research(prompt)

    def _research_dates(self, question: str) -> str:
        """Research relevant dates and timelines."""
        prompt = f"""What are the critical dates and milestones relevant to: {question}

Focus on:
- Key upcoming events
- Development milestones
- Regulatory deadlines
- Industry conferences
- Planned announcements
- Historical significant dates
- Seasonal factors"""
        return self._get_research(prompt)

    def _research_alternatives(self, question: str) -> str:
        """Research alternative scenarios."""
        prompt = f"""What are the most likely scenarios for: {question}

Consider:
- Best case scenario: What could drive significant positive outcomes?
- Base case: Most likely trajectory
- Worst case scenario: Major risks and potential negative outcomes

For each scenario:
- Key drivers
- Required conditions
- Impact assessment
- Timeline expectations"""
        return self._get_research(prompt)

    def _research_existing_predictions(self, question: str) -> str:
        """Research existing predictions and expert opinions."""
        prompt = f"""What specific predictions exist for: {question}

Gather:
- Expert forecasts
- Industry analysis
- Market consensus
- Academic research
- Stakeholder expectations
- Historical precedents
- Contrarian views"""
        return self._get_research(prompt)

    def _create_prediction_prompt(
        self, question: str, research: ResearchResults
    ) -> str:
        """Create the final prediction prompt."""
        return f"""Based on the following research, provide a detailed prediction for this question. You MUST provide specific predictions with clear outcomes and probabilities.

PROBABILITY GUIDELINES:
- Never use 50% as it indicates complete uncertainty
- Base case should be your highest probability scenario (35-65%)
- Alternative scenarios should have differentiated probabilities that reflect your analysis
- Total probabilities must sum to 100%
- Use specific probability numbers (e.g., 42% not 40-45%)
- Back up probability assignments with specific factors from your research

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

Provide your analysis in this format:

MAIN PREDICTION:
[Clear, specific prediction with probability range]

KEY FACTORS DRIVING THIS PREDICTION:
- [Most important factor]
- [Second most important factor]
- [Third most important factor]

SCENARIO ANALYSIS:
1. Optimistic Case (probability): [description]
   - Key drivers
   - Required conditions
   - Expected timeline

2. Base Case (probability): [description]
   - Key drivers
   - Required conditions
   - Expected timeline

3. Pessimistic Case (probability): [description]
   - Key drivers
   - Required conditions
   - Expected timeline

CONFIDENCE LEVEL: [1-10]
[If confidence is low (1-4), explain why]

TIME HORIZON: [Specific timeframe for the prediction]

CRITICAL UNCERTAINTIES:
- [Key risk factor 1]
- [Key risk factor 2]
- [Key risk factor 3]

MONITORING METRICS:
- [Key metric 1 to watch]
- [Key metric 2 to watch]
- [Key metric 3 to watch]"""

    def _generate_prediction(
        self, question: str, research: ResearchResults, system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """Generate the final prediction using OpenAI."""
        try:
            prompt = self._create_prediction_prompt(question, research)

            completion = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt if system_prompt else "You are a precise analytical engine specializing in future predictions. Your goal is to make decisive predictions with clear outcomes. If uncertain, explain the uncertainties but still provide your best assessment based on the available data and research.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )

            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error generating prediction: {e}")
            return None


# added the following to have uniformity in the way we call tools
def run(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    return GeneralPredictor().run(prompt, system_prompt) 
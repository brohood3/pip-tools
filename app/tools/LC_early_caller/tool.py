"""
Early Caller Tool

A tool to identify early cryptocurrency opportunities using LunarCrush data:
1) Analyzes snapshot data (galaxy_score & alt_rank changes)
2) Filters for promising daily changes
3) Verifies momentum through time-series analysis
4) Provides AI-powered insights on the opportunities
"""

import os
import requests
import math
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import HTTPException
from pydantic import BaseModel, Field, validator

# Load environment variables
load_dotenv()

class EarlyCallerConfig(BaseModel):
    """Configuration model for Early Caller screening parameters."""
    
    SCORE_DIFF_THRESHOLD: float = Field(
        default=0.0,
        ge=0,
        description="Minimum Galaxy Score improvement required (0 to disable)"
    )
    
    ALTRANK_DIFF_THRESHOLD: float = Field(
        default=0.0,
        ge=0,
        description="Minimum AltRank improvement required (0 to disable)"
    )
    
    MAX_7D_PRICE_CHANGE: float = Field(
        default=999.0,  # Effectively no limit
        ge=0,
        description="Maximum 7-day price change allowed"
    )
    
    MAX_24H_PRICE_CHANGE: float = Field(
        default=999.0,  # Effectively no limit
        ge=0,
        description="Maximum 24-hour price change allowed"
    )
    
    MIN_MARKET_CAP: float = Field(
        default=0.0,
        ge=0,
        description="Minimum market capitalization"
    )
    
    MAX_MARKET_CAP: float = Field(
        default=1e12,  # 1 trillion, effectively no limit
        ge=0,
        description="Maximum market capitalization"
    )
    
    MIN_VOLUME_24H: float = Field(
        default=0.0,
        ge=0,
        description="Minimum 24-hour trading volume"
    )
    
    MIN_GALAXY_SCORE: float = Field(
        default=0.0,
        ge=0,
        description="Minimum Galaxy Score (0 to disable)"
    )
    
    TIME_SERIES_INTERVAL: str = Field(
        default="1w",
        description="Time series interval"
    )
    
    TIME_SERIES_BUCKET: str = Field(
        default="day",
        description="Time series bucket"
    )
    
    TIME_SERIES_LOOKBACK: int = Field(
        default=7,
        ge=0,
        description="Time series lookback period"
    )
    
    MIN_GS_SLOPE: float = Field(
        default=0.0,
        ge=0,
        description="Minimum Galaxy Score slope (0 to disable)"
    )
    
    MIN_SENTIMENT: float = Field(
        default=0.0,
        ge=0,
        description="Minimum sentiment score (0 to disable)"
    )
    
    check_social_dom: bool = Field(
        default=True,
        description="Check social dominance"
    )

    @validator('MAX_MARKET_CAP')
    def max_market_cap_must_exceed_min(cls, v, values):
        if 'MIN_MARKET_CAP' in values and v < values['MIN_MARKET_CAP']:
            raise ValueError('MAX_MARKET_CAP must be greater than MIN_MARKET_CAP')
        return v 

    @classmethod
    def parse_raw(cls, raw_json: str, *args, **kwargs):
        """Handle both clean JSON and markdown-formatted JSON."""
        if '```' in raw_json:
            # Extract just the JSON part
            start = raw_json.find('{')
            end = raw_json.rfind('}') + 1
            if start >= 0 and end > start:
                raw_json = raw_json[start:end]
        return super().parse_raw(raw_json, *args, **kwargs)

    class Config:
        # Allow extra fields to be ignored
        extra = "ignore"

class EarlyCaller:
    """Early opportunity detection tool using LunarCrush data and GPT-4."""

    def __init__(self):
        """Initialize with API clients and configuration."""
        self.lunar_api_key = os.getenv("LUNARCRUSH_API_KEY")
        if not self.lunar_api_key:
            raise ValueError("LUNARCRUSH_API_KEY environment variable is not set")

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.base_url = "https://lunarcrush.com/api4"

        # Configuration
        self.SCORE_DIFF_THRESHOLD = 0
        self.ALTRANK_DIFF_THRESHOLD = 0
        self.MAX_7D_PRICE_CHANGE = 999
        self.MAX_24H_PRICE_CHANGE = 999
        self.MIN_MARKET_CAP = 0
        self.MAX_MARKET_CAP = 1e12
        self.MIN_VOLUME_24H = 0
        self.MIN_GALAXY_SCORE = 0
        self.TIME_SERIES_INTERVAL = "1w"
        self.TIME_SERIES_BUCKET = "day"
        self.TIME_SERIES_LOOKBACK = 7
        self.MIN_GS_SLOPE = 0
        self.MIN_SENTIMENT = 0
        self.check_social_dom = True

    def run(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Main entry point for the early caller tool."""
        try:
            # Extract configuration from prompt
            config = self._extract_config_from_prompt(prompt)
            
            # Update instance configuration
            self.SCORE_DIFF_THRESHOLD = config.SCORE_DIFF_THRESHOLD
            self.ALTRANK_DIFF_THRESHOLD = config.ALTRANK_DIFF_THRESHOLD
            self.MAX_7D_PRICE_CHANGE = config.MAX_7D_PRICE_CHANGE
            self.MAX_24H_PRICE_CHANGE = config.MAX_24H_PRICE_CHANGE
            self.MIN_MARKET_CAP = config.MIN_MARKET_CAP
            self.MAX_MARKET_CAP = config.MAX_MARKET_CAP
            self.MIN_VOLUME_24H = config.MIN_VOLUME_24H
            self.MIN_GALAXY_SCORE = config.MIN_GALAXY_SCORE
            self.TIME_SERIES_LOOKBACK = config.TIME_SERIES_LOOKBACK
            self.MIN_GS_SLOPE = config.MIN_GS_SLOPE
            self.MIN_SENTIMENT = config.MIN_SENTIMENT

            # Get snapshot data
            snapshot_data = self._fetch_coins_list()
            
            # Screen coins
            shortlist = self._screen_coins_snapshot(snapshot_data)
            
            # Sort by galaxy score difference (highest first)
            shortlist.sort(
                key=lambda x: (
                    x.get("galaxy_score", 0) - 
                    x.get("galaxy_score_previous", x.get("galaxy_score", 0))
                ), 
                reverse=True
            )
            shortlist = shortlist[:5]  # Limit to top 5
            
            # Analyze time series for each passing coin
            final_list = []
            for coin in shortlist:
                coin_id = coin.get("id")
                if not coin_id:
                    continue

                ts_data = self._fetch_coin_time_series(coin_id)
                if self._analyze_time_series_advanced(ts_data, coin):
                    final_list.append(coin)

            # Generate summary and analysis
            summary = self._generate_coin_summary(final_list)
            
            # Include original prompt in analysis context
            analysis = self._get_llm_analysis(final_list, system_prompt, original_prompt=prompt)

            return {
                "response": analysis,
                "metadata": {
                    "coins_analyzed": len(snapshot_data.get("data", [])),
                    "passed_initial_screen": len(shortlist),
                    "total_passed_screen": len(self._screen_coins_snapshot(snapshot_data)),
                    "final_opportunities": len(final_list),
                    "opportunities": summary,
                    "timestamp": datetime.now().isoformat(),
                    "applied_config": config.dict()
                }
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def _fetch_coins_list(self) -> dict:
        """Fetch snapshot data from LunarCrush."""
        url = f"{self.base_url}/public/coins/list/v2"
        params = {"key": self.lunar_api_key, "limit": 0}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def _screen_coins_snapshot(self, data: dict) -> List[Dict[str, Any]]:
        """Screen coins based on daily snapshot differences and absolute metrics."""
        coins = data.get("data", [])
        screened_coins = []

        for coin in coins:
            # Get basic metrics
            galaxy_score = coin.get("galaxy_score", 0)
            galaxy_score_prev = coin.get("galaxy_score_previous", galaxy_score)
            alt_rank = coin.get("alt_rank", 99999)
            alt_rank_prev = coin.get("alt_rank_previous", alt_rank)
            pct_change_7d = coin.get("percent_change_7d", 0)
            pct_change_24h = coin.get("percent_change_24h", 0)
            
            # Get additional metrics
            market_cap = coin.get("market_cap", 0)
            volume_24h = coin.get("volume_24h", 0)

            # Skip if key metrics are missing
            if any(x is None for x in [galaxy_score, alt_rank]):
                continue

            # Calculate differences
            gs_diff = galaxy_score - galaxy_score_prev
            ar_diff = alt_rank_prev - alt_rank  # improvement if positive

            # Check all criteria
            if (gs_diff >= self.SCORE_DIFF_THRESHOLD and
                ar_diff >= self.ALTRANK_DIFF_THRESHOLD and
                pct_change_7d < self.MAX_7D_PRICE_CHANGE and
                abs(pct_change_24h) < self.MAX_24H_PRICE_CHANGE and
                market_cap >= self.MIN_MARKET_CAP and
                market_cap <= self.MAX_MARKET_CAP and
                volume_24h >= self.MIN_VOLUME_24H and
                galaxy_score >= self.MIN_GALAXY_SCORE):

                screened_coins.append(coin)

        # Sort by galaxy score difference (highest first)
        screened_coins.sort(
            key=lambda x: (
                x.get("galaxy_score", 0) - 
                x.get("galaxy_score_previous", x.get("galaxy_score", 0))
            ), 
            reverse=True
        )
        
        return screened_coins

    def _fetch_coin_time_series(self, coin_id_or_symbol: str) -> dict:
        """Fetch time-series data for a specific coin from LunarCrush."""
        url = f"{self.base_url}/public/coins/{coin_id_or_symbol}/time-series/v2"
        
        params = {
            "interval": self.TIME_SERIES_INTERVAL,
            "bucket": self.TIME_SERIES_BUCKET
        }
        
        headers = {
            "Authorization": f"Bearer {self.lunar_api_key}"
        }
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"data": []}

    def _analyze_time_series_advanced(self, coin_time_series: Dict[str, Any], coin: Dict[str, Any]) -> bool:
        """Analyze time series and store slopes in coin data."""
        entries = coin_time_series.get("data", [])
        if not entries:
            return False

        relevant_entries = entries[-self.TIME_SERIES_LOOKBACK:] if len(entries) > self.TIME_SERIES_LOOKBACK else entries
        if len(relevant_entries) < 2:
            return False

        # Calculate slopes and store in coin data
        n_points = len(relevant_entries)
        
        # Use .get() for now until we confirm the data structure
        gs_first = relevant_entries[0].get("galaxy_score")
        gs_last = relevant_entries[-1].get("galaxy_score")
        alt_first = relevant_entries[0].get("alt_rank")
        alt_last = relevant_entries[-1].get("alt_rank")
        
        # Get sentiment and social dominance data - with fallbacks
        sentiments = [pt.get("sentiment_score", pt.get("sentiment", 50)) for pt in relevant_entries]  # Try both possible keys
        social_doms = [pt.get("social_dominance", 0) for pt in relevant_entries]
        
        # Store slopes in coin data
        coin["metrics"] = {
            "galaxy_score_slope": (gs_last - gs_first) / (n_points - 1),
            "alt_rank_slope": (alt_first - alt_last) / (n_points - 1),
            "avg_sentiment": sum(sentiments) / len(sentiments),
            "social_dom_change": social_doms[-1] - social_doms[0]
        }

        # Analysis criteria
        is_gs_rising = (coin["metrics"]["galaxy_score_slope"] >= self.MIN_GS_SLOPE)
        is_alt_improving = (coin["metrics"]["alt_rank_slope"] > 0)
        is_sentiment_ok = (coin["metrics"]["avg_sentiment"] >= self.MIN_SENTIMENT)
        social_dom_ok = True if not self.check_social_dom else (coin["metrics"]["social_dom_change"] >= -2.0)

        signals = [is_gs_rising, is_alt_improving, is_sentiment_ok, social_dom_ok]
        return sum(1 for x in signals if x) == 4

    def _generate_coin_summary(self, coins: List[Dict[str, Any]]) -> str:
        """Generate a concise summary including time series metrics."""
        if not coins:
            return "No coins passed the screening criteria."

        summary = "Found the following potential early opportunities:\n\n"
        for coin in coins:
            metrics = coin.get("metrics", {})
            summary += f"• {coin['name']} ({coin['symbol']}):\n"
            summary += f"  - Market Cap: ${coin.get('market_cap', 0):,.0f}\n"
            summary += f"  - 24h Volume: ${coin.get('volume_24h', 0):,.0f}\n"
            summary += f"  - Galaxy Score: {coin.get('galaxy_score', '?')} (prev: {coin.get('galaxy_score_previous', '?')})\n"
            summary += f"  - Galaxy Score 7d Slope: {metrics.get('galaxy_score_slope', 0):+.2f} points/day\n"
            summary += f"  - AltRank: {coin.get('alt_rank', '?')} (prev: {coin.get('alt_rank_previous', '?')})\n"
            summary += f"  - AltRank 7d Slope: {metrics.get('alt_rank_slope', 0):+.2f} points/day\n"
            summary += f"  - Average Sentiment: {metrics.get('avg_sentiment', 0):.1f}%\n"
            summary += f"  - Social Dom Change: {metrics.get('social_dom_change', 0):+.2f}%\n"
            summary += f"  - 24h Change: {coin.get('percent_change_24h', 0):+.2f}%\n"
            summary += f"  - 7d Change: {coin.get('percent_change_7d', 0):+.2f}%\n\n"

        return summary

    def _get_llm_analysis(self, coins: List[Dict[str, Any]], system_prompt: Optional[str] = None, original_prompt: Optional[str] = None) -> str:
        """Get LLM analysis of the screened coins."""
        summary = self._generate_coin_summary(coins)
        
        if coins:
            prompt = f"""
            These cryptocurrencies have shown strong fundamental and sentiment signals in our screening:

            {summary}

            Analyze these opportunities and provide actionable insights. Consider market conditions, 
            technical signals, and potential catalysts. Be specific but natural in your analysis.
            Focus on what makes each opportunity interesting right now.
            """
        else:
            prompt = f"""
            Let the user know that no opportunities were found matching their criteria.

            Your request: "{original_prompt}"

            Current screening parameters:
            - Galaxy Score Improvement: >= {self.SCORE_DIFF_THRESHOLD}
            - AltRank Improvement: >= {self.ALTRANK_DIFF_THRESHOLD}
            - Price Change 7d: < {self.MAX_7D_PRICE_CHANGE}%
            - Price Change 24h: < {self.MAX_24H_PRICE_CHANGE}%
            - Market Cap Range: ${self.MIN_MARKET_CAP:,.0f} - ${self.MAX_MARKET_CAP:,.0f}
            - Minimum Volume 24h: ${self.MIN_VOLUME_24H:,.0f}
            - Minimum Galaxy Score: {self.MIN_GALAXY_SCORE}
            - Minimum GS Slope: {self.MIN_GS_SLOPE} points/day
            - Minimum Sentiment: {self.MIN_SENTIMENT}%

            Quick suggestions:
            1. Which parameters are likely too strict?
            2. What adjustments would find similar opportunities?
            3. Alternative screening approaches?

            Be brief and specific in your recommendations.
            """

        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Use provided system prompt if available, otherwise use default
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                messages.insert(0, {
                    "role": "system", 
                    "content": "You are a cryptocurrency analyst with expertise in market screening and opportunity detection. Provide clear, actionable insights and help users refine their screening approach."
                })

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.8,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error getting LLM analysis: {str(e)}"

    def _extract_config_from_prompt(self, prompt: str) -> EarlyCallerConfig:
        """Extract and validate configuration from prompt."""
        config_prompt = f"""
        Analyze this request and configure appropriate parameters: "{prompt}"

        Available Metrics and Their Meaning:
        - SCORE_DIFF_THRESHOLD: How much Galaxy Score has improved (higher = stronger recent momentum)
        - ALTRANK_DIFF_THRESHOLD: How much AltRank has improved (higher = stronger rank gains)
        - MIN_GALAXY_SCORE: Base Galaxy Score requirement (higher = stronger overall metrics)
        - MIN_SENTIMENT: Community sentiment requirement (higher = more positive sentiment)
        - MIN_GS_SLOPE: Rate of Galaxy Score improvement (higher = faster improvement)
        - MIN_MARKET_CAP & MAX_MARKET_CAP: Market cap range (lower = smaller caps)
        - MIN_VOLUME_24H: Trading volume requirement (higher = more liquid)
        - MAX_24H_PRICE_CHANGE & MAX_7D_PRICE_CHANGE: Price movement limits (lower = less volatile)

        All parameters start at 0 (disabled). Analyze the request and set appropriate values based on:
        1. Explicit requirements (e.g., "market cap between 1M and 100M")
        2. Implied requirements (e.g., "early opportunities" → lower market cap, improving metrics)
        3. Risk preferences (e.g., "safe picks" → higher minimum scores, lower volatility)

        Return a ONLY JSON object with the parameters you want to modify. Example:
        {{
            "MIN_MARKET_CAP": 1000000,
            "MAX_MARKET_CAP": 100000000,
            "MIN_VOLUME_24H": 500000,
            "SCORE_DIFF_THRESHOLD": 5.0
        }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a cryptocurrency screening expert. Analyze requests and configure optimal parameters for finding opportunities. Return only valid JSON."
                    },
                    {"role": "user", "content": config_prompt}
                ],
                temperature=0.5  
            )
            
            config = EarlyCallerConfig.parse_raw(response.choices[0].message.content)
            return config
            
        except Exception as e:
            return EarlyCallerConfig()

# Function to match the interface pattern of other tools
def run(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    return EarlyCaller().run(prompt, system_prompt)

if __name__ == "__main__":
    # Test prompts with more natural language
    test_prompts = [
        # Very specific/stringent criteria
        "find coins with market cap 5M-50M, minimum 500k volume, galaxy score above 70, and altrank improvement of at least 100 positions",
        
        # Moderate criteria with focus on momentum
        "show me coins under 200M market cap with strong galaxy score momentum and good volume",
        
        # Basic criteria (current test case)
        "show me all coins between 1M and 100M market cap with at least 100k volume, ignore all other filters",
        
        # Focus on recent performance
        "find coins that have significantly improved their metrics in the last 24 hours",
        
        # Very loose/ambiguous
        "show me interesting early opportunities",
        
        # Mixed specific and ambiguous
        "find promising coins under 50M market cap with good social signals",
        
        # Focus on stability
        "show stable coins between 10M-500M market cap with low price volatility but improving metrics"
    ]

    tool = EarlyCaller()
    
    for prompt in test_prompts:
        print("\n" + "="*80)
        print(f"Testing prompt: {prompt}")
        print("="*80)
        
        try:
            # Run analysis
            result = tool.run(prompt)
            
            # Print results using the config that was actually applied
            config = result['metadata']['applied_config']
            print("\nCONFIGURATION CHOSEN BY LLM:")
            print("===========================")
            print(f"Market Cap Range: ${config['MIN_MARKET_CAP']:,.0f} - ${config['MAX_MARKET_CAP']:,.0f}")
            print(f"Minimum Volume: ${config['MIN_VOLUME_24H']:,.0f}")
            print(f"Galaxy Score: min {config['MIN_GALAXY_SCORE']}, improvement >= {config['SCORE_DIFF_THRESHOLD']}")
            print(f"AltRank Improvement: >= {config['ALTRANK_DIFF_THRESHOLD']}")
            print(f"Price Change Limits: 24h <= {config['MAX_24H_PRICE_CHANGE']}%, 7d <= {config['MAX_7D_PRICE_CHANGE']}%")
            print(f"Technical Criteria: GS Slope >= {config['MIN_GS_SLOPE']}, Min Sentiment: {config['MIN_SENTIMENT']}%")
            
            print(f"\nSCREENING RESULTS:")
            print("==================")
            print(f"Passed Basic Screen: {result['metadata'].get('total_passed_screen', 0):,}")
            print(f"Top Opportunities Analyzed: {result['metadata'].get('passed_initial_screen', 0)}")
            print(f"Final Opportunities: {result['metadata'].get('final_opportunities', 0)}")
            
            if result['metadata'].get('final_opportunities', 0) > 0:
                print(f"\nDETAILED OPPORTUNITIES:")
                print("=====================")
                print(result['metadata']['opportunities'])
            
            print("\n" + "="*80 + "\n")
            
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")
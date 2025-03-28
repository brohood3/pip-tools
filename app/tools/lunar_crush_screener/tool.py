"""
LunarCrush Screener Tool

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
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import HTTPException
from pydantic import BaseModel, Field, validator
from app.utils.config import DEFAULT_MODEL
from app.utils.llm import generate_completion

# Load environment variables
load_dotenv()

class LunarCrushScreenerConfig(BaseModel):
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

class LunarCrushScreener:
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
        
        # Use .get() with default values to avoid NoneType errors
        gs_first = relevant_entries[0].get("galaxy_score", 0)
        gs_last = relevant_entries[-1].get("galaxy_score", 0)
        alt_first = relevant_entries[0].get("alt_rank", 0)
        alt_last = relevant_entries[-1].get("alt_rank", 0)
        
        # Skip if we don't have valid data
        if gs_first is None or gs_last is None or alt_first is None or alt_last is None:
            return False
            
        # Convert to numeric values with defaults if needed
        gs_first = float(gs_first) if gs_first is not None else 0
        gs_last = float(gs_last) if gs_last is not None else 0
        alt_first = float(alt_first) if alt_first is not None else 0
        alt_last = float(alt_last) if alt_last is not None else 0
        
        # Get sentiment and social dominance data - with fallbacks
        sentiments = [float(pt.get("sentiment_score", pt.get("sentiment", 50)) or 50) for pt in relevant_entries]  # Try both possible keys
        social_doms = [float(pt.get("social_dominance", 0) or 0) for pt in relevant_entries]
        
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
        """Generate LLM analysis of screened coins."""
        if not coins:
            return "No coins matched the screening criteria."
        
        try:
            # Format coin data for LLM
            coins_data = []
            for coin in coins:
                metrics = coin.get("metrics", {})
                coin_data = {
                    "name": coin.get("name", "Unknown"),
                    "symbol": coin.get("symbol", "Unknown"),
                    "price": coin.get("price", 0),
                    "market_cap": coin.get("market_cap", 0),
                    "volume_24h": coin.get("volume_24h", 0),
                    "galaxy_score": coin.get("galaxy_score", 0),
                    "galaxy_score_previous": coin.get("galaxy_score_previous", 0),
                    "galaxy_score_diff": coin.get("galaxy_score", 0) - coin.get("galaxy_score_previous", 0),
                    "alt_rank": coin.get("alt_rank", 0),
                    "alt_rank_previous": coin.get("alt_rank_previous", 0),
                    "alt_rank_diff": coin.get("alt_rank_previous", 0) - coin.get("alt_rank", 0),
                    "price_change_24h": coin.get("percent_change_24h", 0),
                    "price_change_7d": coin.get("percent_change_7d", 0),
                    "social_score": coin.get("social_score", 0),
                    "social_volume": coin.get("social_volume", 0),
                    "social_dominance": coin.get("social_dominance", 0),
                    "galaxy_score_slope": metrics.get("galaxy_score_slope", 0),
                    "alt_rank_slope": metrics.get("alt_rank_slope", 0),
                    "avg_sentiment": metrics.get("avg_sentiment", 0),
                    "social_dom_change": metrics.get("social_dom_change", 0)
                }
                coins_data.append(coin_data)
            
            # Create information string containing all the gathered data
            information = json.dumps(coins_data, indent=2)
            
            # Create prompt using the user's initial prompt
            prompt = f"""The user asked: "{original_prompt if original_prompt else 'Find promising crypto opportunities'}"

I've fetched and analyzed the following cryptocurrency data for them:
{information}

Create a helpful response that analyzes this data to answer their query."""

            default_system_prompt = "You are a cryptocurrency analyst focused on token-specific analysis. When responding, always begin by mentioning that this data was automatically gathered by the LunarCrush Screener tool. Analyze the provided token data including social metrics, price action, and market dynamics. Focus your insights specifically on the individual tokens' metrics and what they indicate about each token's current momentum and potential. Avoid general market commentary and focus on data-driven insights about these specific tokens. Ignore null values, but be specific about the numbers."
            
            return generate_completion(
                prompt=prompt,
                system_prompt=system_prompt if system_prompt else default_system_prompt,
                model=DEFAULT_MODEL,
                temperature=0.7
            )
            
        except Exception as e:
            print(f"Error generating LLM analysis: {e}")
            return f"Error generating analysis: {str(e)}"

    def _extract_config_from_prompt(self, prompt: str) -> LunarCrushScreenerConfig:
        """Extract configuration parameters from the user prompt using LLM."""
        # Default configuration
        default_config = LunarCrushScreenerConfig()
        
        # If prompt is empty or too short, return default config
        if not prompt or len(prompt.strip()) < 10:
            return default_config
            
        try:
            # Create prompt for parameter extraction
            extraction_prompt = f"""Extract screening parameters from this request: "{prompt}"

Valid parameters:
- SCORE_DIFF_THRESHOLD: Minimum Galaxy Score improvement (default: {default_config.SCORE_DIFF_THRESHOLD})
- ALTRANK_DIFF_THRESHOLD: Minimum AltRank improvement (default: {default_config.ALTRANK_DIFF_THRESHOLD})
- MAX_7D_PRICE_CHANGE: Maximum 7-day price change allowed (default: {default_config.MAX_7D_PRICE_CHANGE})
- MAX_24H_PRICE_CHANGE: Maximum 24-hour price change allowed (default: {default_config.MAX_24H_PRICE_CHANGE})
- MIN_MARKET_CAP: Minimum market capitalization (default: {default_config.MIN_MARKET_CAP})
- MAX_MARKET_CAP: Maximum market capitalization (default: {default_config.MAX_MARKET_CAP})
- MIN_VOLUME_24H: Minimum 24-hour trading volume (default: {default_config.MIN_VOLUME_24H})
- MIN_GALAXY_SCORE: Minimum Galaxy Score (default: {default_config.MIN_GALAXY_SCORE})
- MIN_GS_SLOPE: Minimum Galaxy Score slope (default: {default_config.MIN_GS_SLOPE})
- MIN_SENTIMENT: Minimum sentiment score (default: {default_config.MIN_SENTIMENT})

Respond with a valid JSON object containing only the parameters that should be changed from defaults. For example:
{{"SCORE_DIFF_THRESHOLD": 5.0, "MIN_MARKET_CAP": 10000000}}

If no parameters should be changed, respond with an empty JSON object: {{}}.

IMPORTANT: Only include parameters that are explicitly mentioned or clearly implied in the request. Do not make assumptions about parameters that aren't mentioned."""

            system_prompt = "You are a parameter extraction assistant. Extract only the parameters explicitly mentioned or clearly implied in the request. Respond with a valid JSON object containing only the parameters that should be changed from defaults."
            
            # Define JSON schema for the response
            json_schema = {
                "type": "OBJECT",
                "properties": {
                    "SCORE_DIFF_THRESHOLD": {"type": "NUMBER"},
                    "ALTRANK_DIFF_THRESHOLD": {"type": "NUMBER"},
                    "MAX_7D_PRICE_CHANGE": {"type": "NUMBER"},
                    "MAX_24H_PRICE_CHANGE": {"type": "NUMBER"},
                    "MIN_MARKET_CAP": {"type": "NUMBER"},
                    "MAX_MARKET_CAP": {"type": "NUMBER"},
                    "MIN_VOLUME_24H": {"type": "NUMBER"},
                    "MIN_GALAXY_SCORE": {"type": "NUMBER"},
                    "MIN_GS_SLOPE": {"type": "NUMBER"},
                    "MIN_SENTIMENT": {"type": "NUMBER"},
                    "check_social_dom": {"type": "BOOLEAN"}
                }
            }
            
            # Print debug information
            print(f"Extracting config from prompt: {prompt}")
            
            response_text = generate_completion(
                prompt=extraction_prompt,
                system_prompt=system_prompt,
                model=DEFAULT_MODEL,
                temperature=0.1,
                json_mode=True,
                json_schema=json_schema
            )
            
            # Print the raw response for debugging
            print(f"Raw LLM response: {response_text}")
            print(f"Response type: {type(response_text)}")
            
            # Process the response
            try:
                # Check if the response is already a dictionary (some models might return parsed JSON)
                if isinstance(response_text, dict):
                    params = response_text
                else:
                    # Clean up the response text to handle potential formatting issues
                    if response_text and isinstance(response_text, str):
                        # Remove markdown code blocks if present
                        if '```' in response_text:
                            # Extract just the JSON part
                            start = response_text.find('{')
                            end = response_text.rfind('}') + 1
                            if start >= 0 and end > start:
                                response_text = response_text[start:end]
                        
                        # Remove any leading/trailing whitespace
                        response_text = response_text.strip()
                        
                        # Ensure it starts with { and ends with }
                        if not (response_text.startswith('{') and response_text.endswith('}')):
                            print(f"Invalid JSON format: {response_text}")
                            return default_config
                    
                    # Try to parse the response as JSON
                    params = json.loads(response_text)
                
                print(f"Parsed parameters: {params}")
                
                # Validate numeric parameters
                for key, value in list(params.items()):
                    if key in ["SCORE_DIFF_THRESHOLD", "ALTRANK_DIFF_THRESHOLD", "MAX_7D_PRICE_CHANGE", 
                              "MAX_24H_PRICE_CHANGE", "MIN_MARKET_CAP", "MAX_MARKET_CAP", 
                              "MIN_VOLUME_24H", "MIN_GALAXY_SCORE", "MIN_GS_SLOPE", "MIN_SENTIMENT"]:
                        try:
                            # Convert to float if it's a string or int
                            if isinstance(value, (str, int)):
                                params[key] = float(value)
                            # Ensure it's a number
                            elif not isinstance(value, float):
                                print(f"Invalid value for {key}: {value}, removing")
                                del params[key]
                        except (ValueError, TypeError):
                            print(f"Error converting {key}: {value} to float, removing")
                            del params[key]
                
                # Create config with extracted parameters
                config = LunarCrushScreenerConfig.parse_obj({**default_config.dict(), **params})
                print(f"Final config: {config}")
                return config
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from response: {response_text}, Error: {str(e)}")
                # Try to salvage the response by using regex to extract key-value pairs
                try:
                    if isinstance(response_text, str):
                        # Simple regex to extract key-value pairs
                        pattern = r'"([^"]+)":\s*([0-9.]+)'
                        matches = re.findall(pattern, response_text)
                        if matches:
                            params = {key: float(value) for key, value in matches}
                            return LunarCrushScreenerConfig.parse_obj({**default_config.dict(), **params})
                except Exception as regex_error:
                    print(f"Error extracting parameters with regex: {str(regex_error)}")
                
                return default_config
            
        except Exception as e:
            print(f"Error extracting config from prompt: {e}")
            import traceback
            traceback.print_exc()
            return default_config

# Function to match the interface pattern of other tools
def run(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    return LunarCrushScreener().run(prompt, system_prompt)

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

    tool = LunarCrushScreener()
    
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
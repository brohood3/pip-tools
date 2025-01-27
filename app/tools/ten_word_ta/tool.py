"""
Ten Word TA Tool

A concise technical analysis tool that provides a ten-word summary
with price targets based on key technical indicators.
"""

import os
import json
from typing import Dict, Optional, Any, List
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import HTTPException
import requests

# Load environment variables
load_dotenv()


class TenWordTA:
    """Ten Word Technical Analysis tool using TAapi and GPT-4."""

    def __init__(self):
        """Initialize with API clients and configuration."""
        self.taapi_api_key = os.getenv("TAAPI_API_KEY")
        if not self.taapi_api_key:
            raise ValueError("TAAPI_API_KEY environment variable is not set")

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.taapi_base_url = "https://api.taapi.io"

    def run(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Main entry point for the ten word TA tool.

        Args:
            prompt: User's analysis request
            system_prompt: Optional system prompt to override default behavior

        Returns:
            Dict containing ten word analysis and metadata
        """
        try:
            # Extract token and interval from prompt
            token, interval = self.parse_prompt_with_llm(prompt)
            if not token:
                raise HTTPException(
                    status_code=400,
                    detail="Could not determine which token to analyze. Please specify a token."
                )

            # Get available symbols and find best pair
            available_symbols = self.get_available_symbols()
            if not available_symbols:
                raise HTTPException(
                    status_code=500,
                    detail="Could not fetch available trading pairs. Please try again later."
                )

            pair = self.find_best_pair(token, available_symbols)
            if not pair:
                raise HTTPException(
                    status_code=400,
                    detail=f"No trading pair found for {token}. Please verify the token symbol and try again."
                )

            # Fetch indicators
            indicators = self._fetch_indicators(pair, interval=interval)
            if not indicators:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient data for {pair} on {interval} timeframe."
                )

            # Generate ten word analysis
            analysis = self._generate_ten_word_ta(indicators, pair, interval, system_prompt)

            # Store all context in metadata
            metadata = {
                "prompt": prompt,
                "token": token,
                "pair": pair,
                "interval": interval,
                "timestamp": datetime.now().isoformat(),
                "indicators": indicators,
            }

            return {"response": analysis, "metadata": metadata}

        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )

    def parse_prompt_with_llm(self, prompt: str) -> tuple[Optional[str], str]:
        """Extract token and timeframe from prompt using GPT."""
        try:
            context = f"""Extract the cryptocurrency token name and timeframe from the following analysis request.
Valid timeframes are: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d, 1w

Example inputs and outputs:
Input: "give me a technical analysis for Bitcoin"
Output: {{"token": "BTC", "timeframe": "1d"}}

Input: "analyze ETH on 4 hour timeframe"
Output: {{"token": "ETH", "timeframe": "4h"}}

Input: "what's your view on NEAR for the next hour"
Output: {{"token": "NEAR", "timeframe": "1h"}}

Input: "daily analysis of Cardano"
Output: {{"token": "ADA", "timeframe": "1d"}}

Now extract from this request: "{prompt}"

IMPORTANT: Respond with ONLY the raw JSON object. Do not include markdown formatting, code blocks, or any other text. The response should start with {{ and end with }}."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a trading expert that extracts token names and timeframes from analysis requests. Always respond with a valid JSON object.",
                    },
                    {"role": "user", "content": context},
                ],
                temperature=0,
            )

            response_text = response.choices[0].message.content.strip()

            try:
                data = json.loads(response_text)
                return data.get("token"), data.get("timeframe", "1d")
            except json.JSONDecodeError:
                return None, "1d"

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error parsing trading pair: {str(e)}"
            )

    def get_available_symbols(self) -> List[str]:
        """Get list of available trading pairs from TAapi."""
        try:
            # Fetch available symbols directly from Gate.io
            url = f"{self.taapi_base_url}/exchange-symbols"
            response = requests.get(
                url, params={"secret": self.taapi_api_key, "exchange": "gateio"}
            )

            if not response.ok:
                print(f"\nError fetching symbols: {response.status_code}")
                print(f"Response: {response.text}")
                return self._get_fallback_symbols()

            symbols = response.json()
            if not symbols or not isinstance(symbols, list):
                print("\nInvalid response format from symbols endpoint")
                return self._get_fallback_symbols()

            # Filter for USDT pairs and ensure proper formatting
            formatted_pairs = [
                symbol
                for symbol in symbols
                if isinstance(symbol, str) and symbol.endswith("/USDT")
            ]

            if formatted_pairs:
                print(f"\nFetched {len(formatted_pairs)} trading pairs from Gate.io")
                return sorted(formatted_pairs)

            return self._get_fallback_symbols()

        except Exception as e:
            print(f"\nError fetching trading pairs: {str(e)}")
            return self._get_fallback_symbols()

    def _get_fallback_symbols(self) -> List[str]:
        """Return a fallback list of common trading pairs."""
        print("\nUsing fallback symbol list")
        return [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "ADA/USDT",
            "DOGE/USDT",
            "MATIC/USDT",
            "DOT/USDT",
            "LTC/USDT",
            "AVAX/USDT",
            "LINK/USDT",
            "UNI/USDT",
            "ATOM/USDT",
            "ETC/USDT",
            "XLM/USDT",
            "ALGO/USDT",
            "NEAR/USDT",
            "FTM/USDT",
            "SAND/USDT",
        ]

    def find_best_pair(self, token: str, available_symbols: List[str]) -> Optional[str]:
        """Find the best trading pair for a given token."""
        try:
            # Clean and standardize token
            token = token.strip().upper()

            # Remove /USDT if present
            token = token.replace('/USDT', '')
            
            # Try exact USDT pair match
            exact_match = f"{token}/USDT"
            if exact_match in available_symbols:
                print(f"\nFound exact match: {exact_match}")
                return exact_match
            
            print(f"\nNo exact match found for token: {token}")
            return None

        except Exception as e:
            print(f"\nError finding best pair: {str(e)}")
            return None

    def _fetch_indicators(
        self, symbol: str, interval: str = "1d", exchange: str = "gateio"
    ) -> Optional[Dict[str, Any]]:
        """Fetch essential technical indicators using TAapi."""
        try:
            url = f"{self.taapi_base_url}/bulk"

            # Define our essential indicators for ten word analysis
            indicators = [
                # Price and Trend
                {"indicator": "price"},
                {"indicator": "supertrend"},
                {"indicator": "fibonacciretracement"},
                # Moving Averages
                {"indicator": "sma", "period": 20},
                {"indicator": "ema", "period": 20},
                # Momentum
                {"indicator": "rsi"},
                {"indicator": "macd"},
                # Volatility
                {"indicator": "bbands"},
                # Volume
                {"indicator": "volume"},
            ]

            # Initialize the result dictionary and valid indicator counter
            result = {}
            valid_indicators = 0

            # Process indicators
            payload = {
                "secret": self.taapi_api_key,
                "construct": {
                    "exchange": exchange,
                    "symbol": symbol,
                    "interval": interval,
                    "indicators": indicators,
                },
            }

            response = requests.post(url, json=payload)
            if not response.ok:
                print(f"Error Response Status: {response.status_code}")
                print(f"Error Response Content: {response.text}")
                return None

            data = response.json().get("data", [])

            # Process response data
            for indicator_data in data:
                if indicator_data.get("result") and not indicator_data.get("result").get("error"):
                    if indicator_data["indicator"] in ["sma", "ema"]:
                        # Handle moving averages with different periods
                        indicator_type = indicator_data["indicator"]
                        if indicator_type not in result:
                            result[indicator_type] = {}
                        period = indicator_data.get("period", "20")
                        result[indicator_type][f"period_{period}"] = indicator_data["result"].get("value")
                    else:
                        # Handle other indicators
                        valid_indicators += 1
                        result[indicator_data["indicator"]] = indicator_data["result"]

            # Return None if we don't have enough valid indicators
            if valid_indicators < 4:  # We need fewer indicators than TA Essential
                print(f"\nInsufficient valid indicators: {valid_indicators}")
                return None

            return result

        except Exception as e:
            print(f"\nError fetching indicators: {str(e)}")
            return None

    def _generate_ten_word_ta(
        self, indicators: Dict[str, Any], pair: str, interval: str, system_prompt: Optional[str] = None
    ) -> str:
        """Generate ten word technical analysis with price target."""
        try:
            # Create prompt for GPT-4
            prompt = f"""Generate a EXACTLY ten-word technical analysis for {pair} on {interval} timeframe.

Technical Indicators:
{json.dumps(indicators, indent=2)}

Requirements:
1. EXACTLY ten words (count carefully)
2. Include token symbol
3. Include direction (bullish/bearish/sideways)
4. Include specific price target or range
5. Use remaining words for key reasons

Format: "[TOKEN] [DIRECTION] to [TARGET]: [KEY REASONS]"

Example good responses:
"BTC bullish to $45K: MA cross, RSI strong, volume rising"
"ETH range-bound $2200-2400: BBands squeeze, low volume, neutral RSI"
"SOL bearish to $78: MACD negative, support broken, RSI oversold"

IMPORTANT: Count words carefully. Hyphenated terms count as one word. Numbers count as one word."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt or "You are a precise technical analyst who provides exactly ten-word analyses. You always include price targets based on technical indicators.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"\nError generating analysis: {str(e)}")
            return "Error generating ten word analysis"


# added the following to have uniformity in the way we call tools
def run(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    return TenWordTA().run(prompt, system_prompt) 
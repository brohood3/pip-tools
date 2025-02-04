"""
Technical Analysis Tool

A comprehensive technical analysis tool using multiple indicator categories:
- Moving Averages (SMA, EMA, DEMA, TEMA, VWMA)
- Trend Indicators (Supertrend, ADX, DMI, Vortex)
- Volume Analysis and Money Flow
- Volatility Measures (Standard Deviation, ATR)
- Momentum Oscillators (MACD, RSI)
- Price Action (Bollinger Bands, Support/Resistance)
- Fibonacci Retracements
- Complex Indicators (Ichimoku Cloud)

Provides detailed market analysis with clear trade setups, entry/exit points,
and risk management rules across different timeframes and trading styles.
"""

import os
import requests
import aiohttp
import asyncio
from typing import Dict, Optional, List, TypedDict, Union, Any, Set
from datetime import datetime
from dotenv import load_dotenv
import json
from openai import OpenAI
from fastapi import HTTPException

from app.tools.technical_analysis.indicators import IndicatorRegistry, IndicatorCategory

# Load environment variables
load_dotenv()

# API Configuration
TAAPI_BASE_URL = "https://api.taapi.io"
MAX_INDICATORS = 20  # Maximum number of indicators allowed by the API
MIN_REQUIRED_INDICATORS = 2  # Minimum required indicators for valid analysis

# Time Intervals and Horizons
INTERVAL_HORIZONS = {
    "1m": "very short-term (minutes to hours)",
    "5m": "very short-term (hours)",
    "15m": "short-term (hours)",
    "30m": "short-term (hours to a day)",
    "1h": "intraday (1-2 days)",
    "2h": "intraday (2-3 days)",
    "4h": "short-term (3-5 days)",
    "12h": "medium-term (1-2 weeks)",
    "1d": "medium-term (2-4 weeks)",
    "1w": "long-term (1-3 months)",
}

# Default Analysis Parameters
DEFAULT_TIMEFRAME = "1d"
DEFAULT_EXCHANGE = "gateio"
DEFAULT_STRATEGY = "trend_following"

# Module-level instance for the run function
_ta_instance = None

def get_ta_instance():
    """Get or create the TechnicalAnalysis instance."""
    global _ta_instance
    if _ta_instance is None:
        _ta_instance = TechnicalAnalysis()
    return _ta_instance

def run(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Run technical analysis on the given prompt.
    
    This is the main entry point for the technical analysis tool when used as a module.
    It maintains a single instance of TechnicalAnalysis for efficiency.

    Args:
        prompt (str): User's analysis request in natural language
        system_prompt (Optional[str]): Optional system prompt to customize GPT's behavior

    Returns:
        Dict[str, Any]: Analysis results containing response and metadata
    """
    ta = get_ta_instance()
    return ta.run(prompt, system_prompt)

class PromptAnalysis(TypedDict):
    """TypedDict defining the structure of prompt analysis results.
    
    Attributes:
        token (Optional[str]): The cryptocurrency token/symbol to analyze
        timeframe (str): The analysis timeframe (e.g., "1m", "5m", "1h", "1d")
        indicators (Set[str]): Set of technical indicators to use in the analysis
        strategy_type (str): Type of trading strategy (e.g., "trend_following", "momentum")
        analysis_focus (List[str]): List of analysis focus areas (e.g., ["price_action", "volume"])
    """
    token: Optional[str]
    timeframe: str
    indicators: Set[str]
    strategy_type: str
    analysis_focus: List[str]


class TechnicalIndicators(TypedDict):
    """TypedDict defining the structure of technical indicators data.
    
    Attributes:
        sma (Dict[str, float]): Simple Moving Averages with different periods
        ema (Dict[str, float]): Exponential Moving Averages with different periods
        supertrend (Dict[str, Union[float, str]]): Supertrend values and signals
        adx (Dict[str, float]): Average Directional Index values
        dmi (Dict[str, float]): Directional Movement Index values
        volume (Dict[str, float]): Volume metrics
        stddev (Dict[str, float]): Standard deviation values
        macd (Dict[str, float]): MACD components (MACD line, signal, histogram)
        rsi (Dict[str, float]): Relative Strength Index values
        bbands (Dict[str, float]): Bollinger Bands values
        fibonacciretracement (Dict[str, Any]): Fibonacci retracement levels
        ichimoku (Dict[str, float]): Ichimoku Cloud components
    """
    sma: Dict[str, float]
    ema: Dict[str, float]
    supertrend: Dict[str, Union[float, str]]
    adx: Dict[str, float]
    dmi: Dict[str, float]
    volume: Dict[str, float]
    stddev: Dict[str, float]
    macd: Dict[str, float]
    rsi: Dict[str, float]
    bbands: Dict[str, float]
    fibonacciretracement: Dict[str, Any]
    ichimoku: Dict[str, float]


class TechnicalAnalysis:
    """A comprehensive technical analysis tool for cryptocurrency markets.
    
    This class provides methods for analyzing cryptocurrency markets using various
    technical indicators, generating trading setups, and providing detailed market analysis.
    It integrates with TAapi for indicator calculations and OpenAI for analysis generation.
    
    Attributes:
        taapi_api_key (str): API key for TAapi service
        openai_api_key (str): API key for OpenAI service
        openai_client (OpenAI): OpenAI client instance
        taapi_base_url (str): Base URL for TAapi endpoints
        indicator_registry (IndicatorRegistry): Registry of available technical indicators
    """

    def __init__(self):
        """Initialize the TechnicalAnalysis tool with API clients and configuration.
        
        Raises:
            ValueError: If required API keys are not set in environment variables
        """
        # Load API keys from environment
        self.taapi_api_key = os.getenv("TAAPI_API_KEY")
        if not self.taapi_api_key:
            raise ValueError("TAAPI_API_KEY environment variable is not set")

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.taapi_base_url = TAAPI_BASE_URL
        
        # Initialize indicator registry
        self.indicator_registry = IndicatorRegistry()

    def run(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Main entry point for the technical analysis tool.

        This method orchestrates the entire analysis process:
        1. Parses the user's prompt to extract analysis parameters
        2. Identifies the appropriate trading pair
        3. Fetches relevant technical indicators
        4. Generates a comprehensive analysis using GPT-4

        Args:
            prompt (str): User's analysis request in natural language
            system_prompt (Optional[str]): Optional system prompt to customize GPT's behavior

        Returns:
            Dict[str, Any]: Analysis results containing:
                - response: Generated analysis text
                - metadata: Analysis context and technical indicators used
                - error: Error message if analysis fails

        Raises:
            HTTPException: If there's an error during analysis generation
        """
        try:
            print(f"\nReceived system prompt in run: {system_prompt}")  # Debug log
            
            # Extract analysis parameters from prompt
            analysis_params = self.parse_prompt_with_llm(prompt)
            if not analysis_params["token"]:
                return {
                    "error": "Could not determine which token to analyze. Please specify a token."
                }

            # Get available symbols and find best pair
            available_symbols = self.get_available_symbols_sync()
            if not available_symbols:
                return {
                    "error": "Could not fetch available trading pairs. Please try again later."
                }

            pair = self.find_best_pair(analysis_params["token"], available_symbols)
            if not pair:
                return {
                    "error": f"No trading pair found for {analysis_params['token']}. Please verify the token symbol and try again."
                }

            # Fetch indicators
            indicators = self._fetch_indicators(
                pair, 
                interval=analysis_params["timeframe"],
                selected_indicators=analysis_params["indicators"],
                analysis_focus=analysis_params["analysis_focus"]
            )
            if not indicators:
                return {
                    "error": f"Insufficient data for {pair} on {analysis_params['timeframe']} timeframe."
                }

            # Generate analysis
            analysis = self.generate_analysis(
                indicators, 
                pair, 
                analysis_params["timeframe"], 
                prompt,
                system_prompt,
                strategy_type=analysis_params["strategy_type"],
                analysis_focus=analysis_params["analysis_focus"]
            )

            # Store all context in metadata
            metadata = {
                "prompt": prompt,
                "token": analysis_params["token"],
                "pair": pair,
                "interval": analysis_params["timeframe"],
                "strategy_type": analysis_params["strategy_type"],
                "analysis_focus": list(analysis_params["analysis_focus"]),
                "selected_indicators": list(analysis_params["indicators"]),
                "timestamp": datetime.now().isoformat(),
                "data_quality": "partial" if len(indicators) < len(analysis_params["indicators"]) else "full",
                "technical_indicators": self.format_indicators_json(indicators),
            }

            return {"response": analysis, "metadata": metadata}

        except Exception as e:
            return {"error": str(e)}

    def parse_prompt_with_llm(self, prompt: str) -> PromptAnalysis:
        """Extract analysis parameters from the user's prompt using GPT.

        Uses GPT to intelligently parse the user's natural language request and extract
        key parameters needed for technical analysis, including:
        - Token/symbol to analyze
        - Timeframe for analysis
        - Relevant technical indicators
        - Trading strategy type
        - Analysis focus areas

        Args:
            prompt (str): User's analysis request in natural language

        Returns:
            PromptAnalysis: Structured analysis parameters including token, timeframe,
                          indicators, strategy type, and analysis focus

        Raises:
            HTTPException: If there's an error parsing the trading request
        """
        try:
            context = f"""Analyze the following trading analysis request and extract key parameters.
Consider the following aspects:
1. Token/Symbol to analyze
2. Timeframe for analysis
3. Relevant technical indicators based on the analysis needs
4. Type of trading strategy suggested by the request
5. Main focus areas for the analysis

Available Technical Indicators by Category:

Trend Indicators:
- Moving Averages: SMA, EMA, DEMA, TEMA, VWMA (periods: 20, 50, 200)
- Trend Direction: Supertrend, ADX, DMI, Vortex
- Complex: Ichimoku Cloud, Williams Alligator

Momentum Indicators:
- Oscillators: RSI, Stochastic, StochRSI
- MACD, CCI, MFI
- Awesome Oscillator (AO)
- Ultimate Oscillator
- Williams %R

Volume/Money Flow:
- Volume
- On-Balance Volume (OBV)
- Chaikin Money Flow (CMF)
- Volume Weighted Average Price (VWAP)
- Ease of Movement (EOM)

Volatility Indicators:
- Bollinger Bands
- ATR (Average True Range)
- Standard Deviation
- Keltner Channels
- Donchian Channels

Price Action:
- Support/Resistance
- Fibonacci Retracements
- Pivot Points
- Price Channels
- Candlestick Patterns

Valid timeframes are: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d, 1w

Strategy Types:
- trend_following
- momentum
- mean_reversion
- breakout
- scalping
- swing_trading
- pattern_trading
- volatility_trading

Example inputs and outputs:
Input: "give me a trend analysis for Bitcoin"
Output: {{
    "token": "BTC",
    "timeframe": "1d",
    "indicators": ["sma", "ema", "supertrend", "adx", "dmi", "vortex"],
    "strategy_type": "trend_following",
    "analysis_focus": ["trend", "price_action"]
}}

Input: "analyze ETH momentum on 4 hour timeframe with volume analysis"
Output: {{
    "token": "ETH",
    "timeframe": "4h",
    "indicators": ["macd", "rsi", "ao", "cmf", "vwap", "volume"],
    "strategy_type": "momentum",
    "analysis_focus": ["momentum", "volume"]
}}

Input: "what's your view on NEAR for scalping with volatility focus"
Output: {{
    "token": "NEAR",
    "timeframe": "5m",
    "indicators": ["bbands", "atr", "stoch", "volume", "mfi"],
    "strategy_type": "scalping",
    "analysis_focus": ["volatility", "momentum", "price_action"]
}}

Input: "need breakout setup for SOL/USDT with volume confirmation"
Output: {{
    "token": "SOL",
    "timeframe": "15m",
    "indicators": ["donchian", "volume", "obv", "vwap", "atr"],
    "strategy_type": "breakout",
    "analysis_focus": ["price_action", "volume", "volatility"]
}}

Now analyze this request: "{prompt}"

IMPORTANT: Respond with ONLY the raw JSON object. Do not include markdown formatting, code blocks, or any other text. The response should start with {{ and end with }}."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a trading expert that analyzes trading requests and extracts key parameters. Always respond with a valid JSON object containing all required fields.",
                    },
                    {"role": "user", "content": context},
                ],
                temperature=0,
            )

            response_text = response.choices[0].message.content.strip()

            try:
                data = json.loads(response_text)
                return PromptAnalysis(
                    token=data.get("token"),
                    timeframe=data.get("timeframe", "1d"),
                    indicators=set(data.get("indicators", [])),
                    strategy_type=data.get("strategy_type", "trend_following"),
                    analysis_focus=data.get("analysis_focus", ["price_action"])
                )
            except json.JSONDecodeError:
                return PromptAnalysis(
                    token=None,
                    timeframe="1d",
                    indicators=set(["sma", "ema", "macd", "rsi"]),  # Default indicators
                    strategy_type="trend_following",
                    analysis_focus=["price_action"]
                )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error parsing trading request: {str(e)}"
            )

    def _fetch_exchange_symbols_sync(self, exchange: str) -> List[str]:
        """Synchronously fetch available trading pairs from a specific exchange."""
        try:
            url = f"{self.taapi_base_url}/exchange-symbols"
            response = requests.get(url, params={"secret": self.taapi_api_key, "exchange": exchange})
            if response.status_code == 200:
                symbols = response.json()
                if isinstance(symbols, list):
                    filtered_symbols = [
                        symbol for symbol in symbols 
                        if isinstance(symbol, str) and symbol.endswith("/USDT")
                    ]
                    print(f"\nFetched {len(symbols)} trading pairs from {exchange.capitalize()}")
                    return filtered_symbols
        except Exception as e:
            print(f"\nError fetching {exchange} symbols: {str(e)}")
        return []

    def get_available_symbols_sync(self) -> List[str]:
        """Get available trading pairs synchronously."""
        try:
            # Fetch from multiple exchanges
            binance_symbols = self._fetch_exchange_symbols_sync("binance")
            gateio_symbols = self._fetch_exchange_symbols_sync("gateio")
            
            # Combine and deduplicate symbols
            all_symbols = set(binance_symbols + gateio_symbols)
            return sorted(list(all_symbols)) if all_symbols else self._get_fallback_symbols()
        except Exception as e:
            print(f"\nError fetching trading pairs: {str(e)}")
            return self._get_fallback_symbols()

    def _get_fallback_symbols(self) -> List[str]:
        """Provide a fallback list of common trading pairs.

        Used when unable to fetch current trading pairs from exchanges.
        Contains major cryptocurrencies paired with USDT.

        Returns:
            List[str]: List of common cryptocurrency trading pairs
        """
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
        """Find the best matching trading pair for a given token.

        This method performs an exact match search for the token against available
        trading pairs. It only returns USDT pairs to ensure consistency in analysis.

        Args:
            token (str): Token symbol to find a pair for (e.g., "BTC", "ETH")
            available_symbols (List[str]): List of available trading pairs

        Returns:
            Optional[str]: The matching trading pair if found (e.g., "BTC/USDT"),
                          None if no exact match is found

        Example:
            >>> find_best_pair("BTC", ["BTC/USDT", "ETH/USDT"])
            "BTC/USDT"
        """
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
        self, symbol: str, interval: str = DEFAULT_TIMEFRAME, 
        exchange: str = DEFAULT_EXCHANGE, 
        selected_indicators: Set[str] = None,
        analysis_focus: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch and calculate technical indicators for a given trading pair.

        This method fetches multiple technical indicators in a single batch request,
        optimizing for both coverage and API usage. It combines:
        - Base indicators (always included)
        - User-selected indicators
        - Category-specific indicators based on analysis focus

        Args:
            symbol (str): Trading pair to analyze (e.g., "BTC/USDT")
            interval (str, optional): Timeframe for analysis. Defaults to DEFAULT_TIMEFRAME
            exchange (str, optional): Exchange to fetch data from. Defaults to DEFAULT_EXCHANGE
            selected_indicators (Set[str], optional): Specific indicators requested by user
            analysis_focus (List[str], optional): Categories of analysis to focus on

        Returns:
            Optional[Dict[str, Any]]: Dictionary of calculated indicators and their values,
                                    None if insufficient valid indicators are available

        Raises:
            HTTPException: If there's an error fetching indicators from the API
        """
        try:
            url = f"{self.taapi_base_url}/bulk"
            
            # Start with base indicators
            indicators = self.indicator_registry.get_base_indicators()
            
            # Add selected indicators if specified
            if selected_indicators:
                selected = self.indicator_registry.format_for_taapi(
                    list(selected_indicators),
                    MAX_INDICATORS - len(indicators)
                )
                indicators.extend(selected)
            
            # Add strategy-specific indicators if needed
            if analysis_focus:
                for focus in analysis_focus:
                    if len(indicators) >= MAX_INDICATORS:
                        break
                    try:
                        category = IndicatorCategory(focus)
                        focus_indicators = self.indicator_registry.get_indicators_by_category(category)
                        remaining_slots = MAX_INDICATORS - len(indicators)
                        indicators.extend(focus_indicators[:remaining_slots])
                    except ValueError:
                        # Invalid category, skip it
                        continue

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
            result = {}
            valid_indicators = 0
            
            for indicator_data in data:
                if indicator_data.get("result") and not indicator_data.get("result").get("error"):
                    if indicator_data["indicator"] in ["sma", "ema", "dema", "tema", "vwma"]:
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
            if valid_indicators < MIN_REQUIRED_INDICATORS:
                print(f"\nInsufficient valid indicators: {valid_indicators}")
                return None

            return result

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error fetching indicators: {str(e)}"
            )

    def format_indicators_json(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Format raw indicator data into a structured JSON format.

        Organizes indicators into logical categories for better readability and analysis:
        - Moving Averages (SMA, EMA)
        - Trend Indicators (Supertrend, ADX, DMI)
        - Volume and Volatility metrics
        - Momentum Indicators (MACD, RSI)
        - Price Bands and Support/Resistance levels
        - Complex Indicators (Ichimoku)

        Args:
            indicators (Dict[str, Any]): Raw indicator data from API

        Returns:
            Dict[str, Any]: Structured indicator data organized by categories

        Example:
            >>> format_indicators_json({"sma": {"period_20": 45000}, "rsi": {"value": 65}})
            {
                "moving_averages": {"sma": {"period_20": 45000}},
                "momentum": {"rsi": {"value": 65}}
            }
        """
        try:
            formatted = {}

            # Moving Averages
            moving_averages = {}
            for ma_type in ["sma", "ema"]:
                if ma_type in indicators:
                    moving_averages[ma_type] = indicators[ma_type]
            if moving_averages:
                formatted["moving_averages"] = moving_averages

            # Trend Indicators
            trend = {}
            if "supertrend" in indicators:
                trend["supertrend"] = indicators["supertrend"]
            if "adx" in indicators:
                trend["adx"] = indicators["adx"]
            if "dmi" in indicators:
                trend["dmi"] = indicators["dmi"]
            if trend:
                formatted["trend"] = trend

            # Volume and Volatility
            volume_volatility = {}
            if "volume" in indicators:
                volume_volatility["volume"] = indicators["volume"]
            if "stddev" in indicators:
                volume_volatility["stddev"] = indicators["stddev"]
            if volume_volatility:
                formatted["volume_volatility"] = volume_volatility

            # Momentum and Oscillators
            momentum = {}
            if "macd" in indicators:
                momentum["macd"] = indicators["macd"]
            if "rsi" in indicators:
                momentum["rsi"] = indicators["rsi"]
            if momentum:
                formatted["momentum"] = momentum

            # Price Bands and Support/Resistance
            price = {}
            if "bbands" in indicators:
                price["bollinger"] = indicators["bbands"]
            if "fibonacciretracement" in indicators:
                price["fibonacci"] = indicators["fibonacciretracement"]
            if price:
                formatted["price"] = price

            # Complex Indicators
            if "ichimoku" in indicators:
                formatted["ichimoku"] = indicators["ichimoku"]

            return formatted

        except Exception as e:
            print(f"Error formatting indicators: {str(e)}")
            return {}

    def generate_analysis(
        self, indicators: Dict[str, Any], symbol: str, timeframe: str, 
        user_prompt: str, system_prompt: Optional[str] = None,
        strategy_type: str = DEFAULT_STRATEGY, analysis_focus: List[str] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive technical analysis using GPT-4.

        This method combines technical indicator data with natural language processing
        to generate detailed market analysis. It considers:
        - Multiple timeframe perspectives
        - Trading strategy context
        - User's specific analysis requirements
        - Risk management considerations

        Args:
            indicators (Dict[str, Any]): Technical indicator data
            symbol (str): Trading pair being analyzed
            timeframe (str): Analysis timeframe
            user_prompt (str): Original user's analysis request
            system_prompt (Optional[str]): Custom system prompt for GPT
            strategy_type (str, optional): Trading strategy focus. Defaults to DEFAULT_STRATEGY
            analysis_focus (List[str], optional): Specific areas to analyze

        Returns:
            Dict[str, Any]: Analysis results containing:
                - response: Generated analysis text
                - metadata: Analysis context and parameters used

        Raises:
            HTTPException: If there's an error generating the analysis

        Example:
            >>> generate_analysis(indicators_data, "BTC/USDT", "4h", "Analyze BTC trend")
            {
                "response": "Detailed market analysis...",
                "metadata": {
                    "token": "BTC",
                    "timeframe": "4h",
                    ...
                }
            }
        """
        try:
            time_horizon = INTERVAL_HORIZONS.get(timeframe, "medium-term")
            
            if not system_prompt:
                system_prompt = """You are an expert technical analyst. Your goal is to provide clear, actionable insights that directly answer the user's question while explaining what the technical indicators reveal about the market. Focus on being clear and natural in your explanations, avoiding rigid structures unless they serve the analysis."""

            # Format indicators for better readability
            formatted_indicators = json.dumps(indicators, indent=2)
            
            analysis_request = f"""ANALYSIS CONTEXT:
- Asset: {symbol}
- Timeframe: {timeframe} ({time_horizon})
- Strategy Focus: {strategy_type if strategy_type else 'Not specified'}
- Analysis Areas: {', '.join(analysis_focus) if analysis_focus else 'All available indicators'}
- Original Question: "{user_prompt}"

Guidelines for this analysis:
- Start by directly addressing the original question using the most relevant indicators
- Explain what the indicators are telling us about the market in clear, actionable terms
- Highlight important signals or patterns you observe
- Include specific price levels and clear rules when relevant
- If you spot potential trade setups, describe them naturally including:
  * Entry conditions and price levels
  * Stop loss placement and reasoning
  * Take profit targets
  * Position sizing suggestions
  * Risk management considerations
- Feel free to point out any limitations or additional context needed
- You don't need to follow a rigid structure - let the analysis flow based on what's most important for answering the original question

Consider the time horizon of {time_horizon} when providing your analysis.

Here are the technical indicators for {symbol} on the {timeframe} timeframe:

{formatted_indicators}

Please analyze these indicators in the context of the original question."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_request}
            ]

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )

            # Return analysis with metadata
            return {
                "response": response.choices[0].message.content,
                "metadata": {
                    "prompt": user_prompt,
                    "token": symbol.split('/')[0],
                    "pair": symbol,
                    "interval": timeframe,
                    "strategy_type": strategy_type,
                    "analysis_focus": analysis_focus,
                    "selected_indicators": list(self._get_selected_indicators(indicators)),
                    "timestamp": datetime.now().isoformat(),
                    "data_quality": "full",
                    "technical_indicators": indicators
                }
            }

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating analysis: {str(e)}"
            )

    def _get_selected_indicators(self, indicators: Dict[str, Any]) -> Set[str]:
        """Extract the list of indicators used in the analysis."""
        selected = set()
        
        # Check main indicators
        for key in indicators.keys():
            if key in ["moving_averages", "trend", "volume_volatility", "momentum", "price"]:
                # These are categories, check their contents
                for subkey in indicators[key].keys():
                    selected.add(subkey)
            else:
                # Direct indicator
                selected.add(key)
        
        return selected


def main():
    ta_tool = TechnicalAnalysis()
    
    test_cases = [
        # Test Case 1: Basic trend analysis
        "Analyze BTC trend on 4h timeframe",
        
        # Test Case 2: Volume analysis with specific indicator
        "Check SOL's volume profile on 15min chart using OBV",
        
        # Test Case 3: Complex multi-indicator analysis
        "Find momentum setups for ETH on 1h timeframe using RSI and MACD",
        
        # Test Case 4: Breakout analysis
        "Look for breakout opportunities in NEAR using Bollinger Bands and volume",
        
        # Test Case 5: Scalping setup
        "Give me scalping setups for BNB/USDT on 5min chart"
    ]
    
    print("\nRunning Technical Analysis Tests\n")
    
    for prompt in test_cases:
        print("=" * 80)
        print(f"Test Case: {prompt}")
        print("=" * 80 + "\n")
        
        result = ta_tool.run(prompt)
        print(json.dumps(result, indent=2))
        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()

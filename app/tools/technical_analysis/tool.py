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
from typing import Dict, Optional, List, TypedDict, Union, Any, Set, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
from openai import OpenAI
from fastapi import HTTPException
import pandas as pd
import pandas_ta as ta
from io import BytesIO
import base64
from dataclasses import dataclass
from enum import Enum
from .chart_generator import ChartGenerator

# Load environment variables
load_dotenv()

class IndicatorCategory(Enum):
    """Categories for different types of technical indicators."""
    MOVING_AVERAGE = "moving_average"
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    PRICE_ACTION = "price_action"

class IndicatorParams(TypedDict, total=False):
    """Parameters that can be passed to indicators."""
    period: Optional[int]
    multiplier: Optional[float]
    length: Optional[int]

@dataclass
class Indicator:
    """Represents a technical indicator with its metadata."""
    name: str
    category: IndicatorCategory
    priority: int  # 1 = base indicator, 2 = common, 3 = specialized
    params: IndicatorParams
    requires_volume: bool = False
    description: str = ""

class IndicatorRegistry:
    """Central registry for all technical indicators."""
    
    def __init__(self):
        """Initialize the indicator registry with all available indicators."""
        self._indicators: Dict[str, Indicator] = {
            # Moving Averages (Priority 1-2)
            "sma": Indicator(
                name="sma",
                category=IndicatorCategory.MOVING_AVERAGE,
                priority=1,
                params={"period": 20},
                description="Simple Moving Average"
            ),
            "ema": Indicator(
                name="ema",
                category=IndicatorCategory.MOVING_AVERAGE,
                priority=1,
                params={"period": 20},
                description="Exponential Moving Average"
            ),
            "dema": Indicator(
                name="dema",
                category=IndicatorCategory.MOVING_AVERAGE,
                priority=2,
                params={"period": 20},
                description="Double Exponential Moving Average"
            ),
            "tema": Indicator(
                name="tema",
                category=IndicatorCategory.MOVING_AVERAGE,
                priority=2,
                params={"period": 20},
                description="Triple Exponential Moving Average"
            ),
            "vwma": Indicator(
                name="vwma",
                category=IndicatorCategory.MOVING_AVERAGE,
                priority=2,
                params={"period": 20},
                requires_volume=True,
                description="Volume Weighted Moving Average"
            ),
            "wma": Indicator(
                name="wma",
                category=IndicatorCategory.MOVING_AVERAGE,
                priority=2,
                params={"period": 20},
                description="Weighted Moving Average"
            ),
            "hma": Indicator(
                name="hma",
                category=IndicatorCategory.MOVING_AVERAGE,
                priority=2,
                params={"period": 20},
                description="Hull Moving Average"
            ),
            
            # Trend Indicators (Priority 1-2)
            "supertrend": Indicator(
                name="supertrend",
                category=IndicatorCategory.TREND,
                priority=1,
                params={},
                description="SuperTrend indicator"
            ),
            "adx": Indicator(
                name="adx",
                category=IndicatorCategory.TREND,
                priority=1,
                params={},
                description="Average Directional Index"
            ),
            "dmi": Indicator(
                name="dmi",
                category=IndicatorCategory.TREND,
                priority=1,
                params={},
                description="Directional Movement Index"
            ),
            "vortex": Indicator(
                name="vortex",
                category=IndicatorCategory.TREND,
                priority=2,
                params={},
                description="Vortex Indicator"
            ),
            "psar": Indicator(
                name="psar",
                category=IndicatorCategory.TREND,
                priority=2,
                params={},
                description="Parabolic SAR"
            ),
            "trix": Indicator(
                name="trix",
                category=IndicatorCategory.TREND,
                priority=2,
                params={},
                description="Triple Exponential Average"
            ),
            "kst": Indicator(
                name="kst",
                category=IndicatorCategory.TREND,
                priority=2,
                params={},
                description="Know Sure Thing"
            ),
            
            # Volume Indicators (Priority 1-2)
            "volume": Indicator(
                name="volume",
                category=IndicatorCategory.VOLUME,
                priority=1,
                params={},
                requires_volume=True,
                description="Raw volume"
            ),
            "obv": Indicator(
                name="obv",
                category=IndicatorCategory.VOLUME,
                priority=2,
                params={},
                requires_volume=True,
                description="On Balance Volume"
            ),
            "cmf": Indicator(
                name="cmf",
                category=IndicatorCategory.VOLUME,
                priority=2,
                params={},
                requires_volume=True,
                description="Chaikin Money Flow"
            ),
            "vwap": Indicator(
                name="vwap",
                category=IndicatorCategory.VOLUME,
                priority=1,
                params={},
                requires_volume=True,
                description="Volume Weighted Average Price"
            ),
            "pvt": Indicator(
                name="pvt",
                category=IndicatorCategory.VOLUME,
                priority=2,
                params={},
                requires_volume=True,
                description="Price Volume Trend"
            ),
            "mfi": Indicator(
                name="mfi",
                category=IndicatorCategory.VOLUME,
                priority=2,
                params={"length": 14},
                requires_volume=True,
                description="Money Flow Index"
            ),
            
            # Momentum Indicators (Priority 1-2)
            "rsi": Indicator(
                name="rsi",
                category=IndicatorCategory.MOMENTUM,
                priority=1,
                params={},
                description="Relative Strength Index"
            ),
            "macd": Indicator(
                name="macd",
                category=IndicatorCategory.MOMENTUM,
                priority=1,
                params={},
                description="Moving Average Convergence Divergence"
            ),
            "stoch": Indicator(
                name="stoch",
                category=IndicatorCategory.MOMENTUM,
                priority=2,
                params={},
                description="Stochastic Oscillator"
            ),
            "cci": Indicator(
                name="cci",
                category=IndicatorCategory.MOMENTUM,
                priority=2,
                params={"length": 20},
                description="Commodity Channel Index"
            ),
            "roc": Indicator(
                name="roc",
                category=IndicatorCategory.MOMENTUM,
                priority=2,
                params={"length": 12},
                description="Rate of Change"
            ),
            "willr": Indicator(
                name="willr",
                category=IndicatorCategory.MOMENTUM,
                priority=2,
                params={"length": 14},
                description="Williams %R"
            ),
            "ao": Indicator(
                name="ao",
                category=IndicatorCategory.MOMENTUM,
                priority=2,
                params={},
                description="Awesome Oscillator"
            ),
            
            # Volatility Indicators (Priority 1-2)
            "bbands": Indicator(
                name="bbands",
                category=IndicatorCategory.VOLATILITY,
                priority=1,
                params={},
                description="Bollinger Bands"
            ),
            "atr": Indicator(
                name="atr",
                category=IndicatorCategory.VOLATILITY,
                priority=1,
                params={},
                description="Average True Range"
            ),
            "stddev": Indicator(
                name="stddev",
                category=IndicatorCategory.VOLATILITY,
                priority=2,
                params={},
                description="Standard Deviation"
            ),
            "keltner": Indicator(
                name="keltner",
                category=IndicatorCategory.VOLATILITY,
                priority=2,
                params={},
                description="Keltner Channels"
            ),
            "donchian": Indicator(
                name="donchian",
                category=IndicatorCategory.VOLATILITY,
                priority=2,
                params={"length": 20},
                description="Donchian Channels"
            ),
            
            # Price Action (Priority 2-3)
            "fibonacciretracement": Indicator(
                name="fibonacciretracement",
                category=IndicatorCategory.PRICE_ACTION,
                priority=2,
                params={},
                description="Fibonacci Retracement Levels"
            ),
            "pivotpoints": Indicator(
                name="pivotpoints",
                category=IndicatorCategory.PRICE_ACTION,
                priority=2,
                params={},
                description="Pivot Points"
            ),
            "candlepatterns": Indicator(
                name="candlepatterns",
                category=IndicatorCategory.PRICE_ACTION,
                priority=2,
                params={},
                description="Common Candlestick Patterns"
            ),
            "harmonicpatterns": Indicator(
                name="harmonicpatterns",
                category=IndicatorCategory.PRICE_ACTION,
                priority=3,
                params={},
                description="Harmonic Price Patterns"
            ),
            "ichimoku": Indicator(
                name="ichimoku",
                category=IndicatorCategory.PRICE_ACTION,
                priority=2,
                params={},
                description="Ichimoku Cloud"
            ),
            "pricelevels": Indicator(
                name="pricelevels",
                category=IndicatorCategory.PRICE_ACTION,
                priority=2,
                params={},
                description="Key Price Levels and S/R"
            ),
        }

        # Strategy-specific indicator mappings
        self._strategy_indicators = {
            "trend_following": [
                IndicatorCategory.MOVING_AVERAGE,
                IndicatorCategory.TREND
            ],
            "momentum": [
                IndicatorCategory.MOMENTUM,
                IndicatorCategory.VOLUME
            ],
            "mean_reversion": [
                IndicatorCategory.VOLATILITY,
                IndicatorCategory.MOMENTUM
            ],
            "breakout": [
                IndicatorCategory.VOLATILITY,
                IndicatorCategory.VOLUME,
                IndicatorCategory.PRICE_ACTION
            ],
            "volatility": [  # Changed from "scalping" to "volatility"
                IndicatorCategory.VOLATILITY,
                IndicatorCategory.MOMENTUM,
                IndicatorCategory.PRICE_ACTION
            ]
        }

    def get_base_indicators(self) -> List[Dict[str, Any]]:
        """Get all priority 1 (base) indicators."""
        return [
            {"indicator": ind.name, **ind.params}
            for ind in self._indicators.values()
            if ind.priority == 1 or ind.name == "pricelevels"  # Always include pricelevels
        ]

    def get_indicators_by_category(self, category: IndicatorCategory) -> List[Dict[str, Any]]:
        """Get all indicators in a specific category."""
        return [
            {"indicator": ind.name, **ind.params}
            for ind in self._indicators.values()
            if ind.category == category
        ]

    def get_indicators_for_strategy(self, strategy_type: str) -> List[Dict[str, Any]]:
        """Get recommended indicators for a specific strategy type."""
        if strategy_type not in self._strategy_indicators:
            return []
            
        categories = self._strategy_indicators[strategy_type]
        return [
            {"indicator": ind.name, **ind.params}
            for ind in self._indicators.values()
            if ind.category in categories
        ]

    def is_valid_indicator(self, indicator_name: str) -> bool:
        """Check if an indicator is valid."""
        return indicator_name in self._indicators

# API Configuration
LUNARCRUSH_BASE_URL = "https://lunarcrush.com/api4"
MAX_INDICATORS = 20  # Maximum number of indicators to calculate
MIN_REQUIRED_INDICATORS = 2  # Minimum required indicators for valid analysis

# Indicator Parameters
MOVING_AVERAGE_PERIODS = [20, 50, 200]
MACD_PARAMS = {"fast": 12, "slow": 26, "signal": 9}
RSI_LENGTH = 14
BBANDS_STD = 2
SUPERTREND_PARAMS = {"length": 7, "multiplier": 3.0}
ADX_LENGTH = 14
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
ICHIMOKU_PARAMS = {
    "tenkan": 9,    # Conversion line period
    "kijun": 26,    # Base line period
    "senkou": 52    # Leading span B period
}
KELTNER_LENGTH = 20
KELTNER_SCALAR = 2
MFI_LENGTH = 14
WILLR_LENGTH = 14

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

# LunarCrush Interval Mapping (to hours)
LUNARCRUSH_INTERVALS = {
    "1m": 1/60,
    "5m": 5/60,
    "15m": 15/60,
    "30m": 30/60,
    "1h": 1,
    "2h": 2,
    "4h": 4,
    "12h": 12,
    "1d": 24,
    "1w": 24*7
}

# Default Analysis Parameters
DEFAULT_TIMEFRAME = "1d"
DEFAULT_EXCHANGE = "gateio"  # Kept for compatibility
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
    It integrates with LunarCrush for indicator calculations and OpenAI for analysis generation.
    
    Attributes:
        lunarcrush_api_key (str): API key for LunarCrush service
        openai_api_key (str): API key for OpenAI service
        openai_client (OpenAI): OpenAI client instance
        lunarcrush_base_url (str): Base URL for LunarCrush endpoints
        indicator_registry (IndicatorRegistry): Registry of available technical indicators
    """

    # Class-level variable to store the latest chart
    _latest_chart = None

    def __init__(self):
        """Initialize the TechnicalAnalysis tool with API clients and configuration.
        
        Raises:
            ValueError: If required API keys are not set in environment variables
        """
        # Load API keys from environment
        self.lunarcrush_api_key = os.getenv("LUNARCRUSH_API_KEY")
        if not self.lunarcrush_api_key:
            raise ValueError("LUNARCRUSH_API_KEY environment variable is not set")

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize indicator registry
        self.indicator_registry = IndicatorRegistry()
        
        # Cache for coin IDs to reduce API calls
        self._coin_id_cache = {}
        
        # Initialize chart generator
        self.chart_generator = ChartGenerator()

    @classmethod
    def get_latest_chart(cls) -> Optional[str]:
        """Get the latest generated chart."""
        return cls._latest_chart

    @classmethod
    def _store_chart(cls, chart_data: str):
        """Store the latest generated chart."""
        cls._latest_chart = chart_data

    def get_lunarcrush_coin_id(self, symbol: str) -> Optional[int]:
        """Get coin ID from LunarCrush API."""
        try:
            # Clean symbol - remove USD/USDT and standardize
            symbol = symbol.upper().replace('/USD', '').replace('/USDT', '')
            
            # Check cache first
            if symbol in self._coin_id_cache:
                return self._coin_id_cache[symbol]
            
            url = f"{LUNARCRUSH_BASE_URL}/public/coins/list/v2"
            headers = {
                "Authorization": f"Bearer {self.lunarcrush_api_key}"
            }
            response = requests.get(url, headers=headers)
            
            if not response.ok:
                print(f"Error fetching coin list: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
            data = response.json()
            
            # Find coin by symbol (case insensitive)
            for coin in data.get('data', []):
                if coin['symbol'].upper() == symbol:
                    # Cache the result
                    self._coin_id_cache[symbol] = coin['id']
                    return coin['id']
                    
            return None
            
        except Exception as e:
            print(f"Error getting coin ID: {str(e)}")
            return None

    def fetch_candle_data(self, symbol: str, interval: str, limit: int = 300) -> List[Dict[str, Any]]:
        """Fetch historical candle data from LunarCrush API."""
        try:
            coin_id = self.get_lunarcrush_coin_id(symbol)
            if not coin_id:
                raise Exception(f"Could not find coin ID for {symbol}")
            
            url = f"{LUNARCRUSH_BASE_URL}/public/coins/{coin_id}/time-series/v2"
                
            # Calculate start and end times
            end = int(datetime.now().timestamp())
            hours = LUNARCRUSH_INTERVALS.get(interval)
            if not hours:
                raise Exception(f"Invalid interval: {interval}")
            start = end - (int(limit * hours * 3600))  # Convert hours to seconds
            
            # Determine appropriate bucket based on interval
            bucket = "hour"
            if hours >= 24:  # If interval is 1d or longer
                bucket = "day"
            
            params = {
                "start": start,
                "end": end,
                "bucket": bucket
            }
            headers = {
                "Authorization": f"Bearer {self.lunarcrush_api_key}"
            }
            
            print(f"\nFetching data from LunarCrush:")
            print(f"URL: {url}")
            print(f"Params: {params}")
            
            response = requests.get(url, params=params, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch candle data: {response.text}")
                
            data = response.json()
            if not data.get('data'):
                raise Exception("No candle data received")
            
            print(f"Received {len(data['data'])} candles")
            
            candles = []
            for candle in data['data']:
                try:
                    timestamp = float(candle['time'])
                    # Verify timestamp is not in the future
                    if timestamp > end:
                        print(f"Warning: Skipping future candle with timestamp {timestamp}")
                        continue
                        
                    # Map the fields exactly as pandas_ta expects them
                    raw_volume = candle.get('volume_24h', 0)
                    print(f"Raw volume from API: {raw_volume}")
                    
                    candle_dict = {
                        "time": timestamp,
                        "open": float(candle['open']),
                        "high": float(candle['high']),
                        "low": float(candle['low']),
                        "close": float(candle['close']),
                        "volume": float(raw_volume)  # Keep as float for calculations
                    }
                    candles.append(candle_dict)
                except (KeyError, ValueError) as e:
                    print(f"Warning: Skipping invalid candle data: {e}")
                    continue
            
            if not candles:
                raise Exception("No valid candle data after processing")
            
            # Sort candles by time to ensure proper order
            candles.sort(key=lambda x: x['time'])
            
            # NOTE: Don't convert volume to string here - keep as float for calculations
            return candles

        except Exception as e:
            print(f"Error in fetch_candle_data: {str(e)}")
            return None

    def get_available_symbols_sync(self) -> List[str]:
        """Get available coins from LunarCrush."""
        try:
            url = f"{LUNARCRUSH_BASE_URL}/public/coins/list/v2"
            headers = {
                "Authorization": f"Bearer {self.lunarcrush_api_key}"
            }
            response = requests.get(url, headers=headers)
            
            if not response.ok:
                print(f"Error fetching symbols: {response.status_code}")
                return self._get_fallback_symbols()
                
            data = response.json()
            
            # Just return the coin symbols
            symbols = [coin['symbol'] for coin in data.get('data', [])]
            return sorted(symbols)
            
        except Exception as e:
            print(f"Error fetching symbols: {str(e)}")
            return self._get_fallback_symbols()

    def run(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Main entry point for the technical analysis tool."""
        try:
            print(f"\nReceived system prompt in run: {system_prompt}")
            
            # Extract analysis parameters from prompt
            analysis_params = self.parse_prompt_with_llm(prompt)
            if not analysis_params["token"]:
                return {"error": "Could not determine which token to analyze. Please specify a token."}

            # Get available symbols and find match
            available_symbols = self.get_available_symbols_sync()
            if not available_symbols:
                return {"error": "Could not fetch available coins. Please try again later."}

            symbol = self.find_best_pair(analysis_params["token"], available_symbols)
            if not symbol:
                return {"error": f"No coin found for {analysis_params['token']}. Please verify the token symbol and try again."}

            # Fetch indicators
            indicators = self._fetch_indicators(
                symbol, 
                interval=analysis_params["timeframe"],
                selected_indicators=analysis_params["indicators"],
                analysis_focus=analysis_params["analysis_focus"]
            )
            if not indicators:
                return {"error": f"Insufficient data for {symbol} on {analysis_params['timeframe']} timeframe."}

            # Generate analysis
            analysis = self.generate_analysis(
                indicators, 
                symbol, 
                analysis_params["timeframe"], 
                prompt,
                system_prompt,
                strategy_type=analysis_params["strategy_type"],
                analysis_focus=analysis_params["analysis_focus"]
            )

            # Generate chart if we have valid indicators
            chart_base64 = ""
            if self._current_df is not None and indicators:
                chart_base64 = self.chart_generator.generate_chart(
                    self._current_df,
                    indicators,
                    symbol,
                    analysis_params["timeframe"],
                    analysis_type=analysis_params["strategy_type"]
                )
                # Store the chart for later retrieval
                self._store_chart(chart_base64)

            # Store all context in metadata
            metadata = {
                "prompt": prompt,
                "token": analysis_params["token"],
                "symbol": symbol,
                "interval": analysis_params["timeframe"],
                "strategy_type": analysis_params["strategy_type"],
                "analysis_focus": list(analysis_params["analysis_focus"]),
                "selected_indicators": list(analysis_params["indicators"]),
                "timestamp": datetime.now().isoformat(),
                "data_quality": "partial" if len(indicators) < len(analysis_params["indicators"]) else "full",
                "technical_indicators": self.format_indicators_json(indicators)
            }

            # Return the response in the format expected by main.py
            return {
                "response": analysis,
                "metadata": metadata,
                "chart": chart_base64  # This will be handled by main.py
            }

        except Exception as e:
            print(f"Error in technical analysis: {str(e)}")
            return {"error": str(e)}

    def parse_prompt_with_llm(self, prompt: str) -> PromptAnalysis:
        """Extract analysis parameters from the user's prompt using GPT.

        Uses GPT to intelligently parse the user's natural language request and extract
        key parameters needed for technical analysis, including:
        - Token/Symbol to analyze
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
- volatility
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
    "strategy_type": "volatility",
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
                model="gpt-4o-mini",
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

    def find_best_pair(self, token: str, available_symbols: List[str]) -> Optional[str]:
        """Find the best matching coin symbol.

        Args:
            token (str): Token symbol to find (e.g., "BTC", "ETH")
            available_symbols (List[str]): List of available coins

        Returns:
            Optional[str]: The matching symbol if found, None if no match
        """
        try:
            # Clean and standardize token
            token = token.strip().upper().replace('/USD', '').replace('/USDT', '')
            
            # Try exact match
            if token in available_symbols:
                print(f"\nFound exact match: {token}")
                return token
            
            print(f"\nNo exact match found for token: {token}")
            return None

        except Exception as e:
            print(f"\nError finding symbol: {str(e)}")
            return None

    def _fetch_indicators(
        self, symbol: str, interval: str = DEFAULT_TIMEFRAME,
        selected_indicators: Set[str] = None,
        analysis_focus: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Calculate technical indicators using pandas_ta.
        
        Args:
            symbol: Trading pair to analyze (e.g., "BTC/USDT")
            interval: Timeframe for analysis
            selected_indicators: Set of indicators explicitly requested
            analysis_focus: Categories of analysis to focus on
            
        Returns:
            Dictionary of calculated indicators or None if insufficient data
        """
        try:
            # Fetch historical candle data
            candle_data = self.fetch_candle_data(symbol, interval)
            if not candle_data:
                return None
                
            # Convert to DataFrame with proper structure for pandas_ta
            df = pd.DataFrame(candle_data)
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('datetime', inplace=True)
            
            # Ensure we have the required columns in the correct order
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"Missing required columns: {missing}")
            
            # Sort index and verify we have enough data
            df.sort_index(inplace=True)
            print("\nDataFrame Info:")
            print(df.info())
            print("\nFirst few rows:")
            print(df[required_columns].head())
            print("\nLast few rows:")
            print(df[required_columns].tail())
            
            # Combine all required indicators
            all_indicators = set()
            
            # 1. Add explicitly requested indicators
            if selected_indicators:
                all_indicators.update(selected_indicators)
            
            # 2. Add base indicators (priority 1) and pricelevels
            base_indicators = {ind.name for ind in self.indicator_registry._indicators.values() 
                             if ind.priority == 1 or ind.name == "pricelevels"}  # Always include pricelevels
            all_indicators.update(base_indicators)
            
            # 3. Add strategy-specific indicators
            if analysis_focus:
                for category in analysis_focus:
                    category_indicators = {ind.name for ind in self.indicator_registry._indicators.values() 
                                        if ind.category.value == category.lower()}
                    all_indicators.update(category_indicators)
            
            # Calculate indicators
            results = {}
            valid_indicators = 0
            
            for indicator in all_indicators:
                config = self.indicator_registry._indicators.get(indicator)
                if not config:
                    continue
                    
                try:
                    # Moving Averages
                    if indicator == "sma":
                        df.ta.sma(length=20, append=True)
                        df.ta.sma(length=50, append=True)
                        df.ta.sma(length=200, append=True)
                        results[indicator] = {
                            "SMA_20": float(df['SMA_20'].iloc[-1]),
                            "SMA_50": float(df['SMA_50'].iloc[-1]),
                            "SMA_200": float(df['SMA_200'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    elif indicator == "ema":
                        df.ta.ema(length=20, append=True)
                        df.ta.ema(length=50, append=True)
                        df.ta.ema(length=200, append=True)
                        results[indicator] = {
                            "EMA_20": float(df['EMA_20'].iloc[-1]),
                            "EMA_50": float(df['EMA_50'].iloc[-1]),
                            "EMA_200": float(df['EMA_200'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    elif indicator == "dema":
                        df.ta.dema(length=20, append=True)
                        results[indicator] = {
                            "value": float(df['DEMA_20'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    elif indicator == "tema":
                        df.ta.tema(length=20, append=True)
                        results[indicator] = {
                            "value": float(df['TEMA_20'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    elif indicator == "vwma":
                        df.ta.vwma(length=20, append=True)
                        results[indicator] = {
                            "value": float(df['VWMA_20'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    elif indicator == "wma":
                        df.ta.wma(length=20, append=True)
                        results[indicator] = {
                            "value": float(df['WMA_20'].iloc[-1])
                        }
                        valid_indicators += 1
                        
                    elif indicator == "hma":
                        df.ta.hma(length=20, append=True)
                        results[indicator] = {
                            "value": float(df['HMA_20'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    # Trend Indicators
                    elif indicator == "adx":
                        df.ta.adx(append=True)
                        results[indicator] = {
                            "value": float(df['ADX_14'].iloc[-1]),
                            "plus_di": float(df['DMP_14'].iloc[-1]),
                            "minus_di": float(df['DMN_14'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    elif indicator == "supertrend":
                        df.ta.supertrend(append=True)
                        results[indicator] = {
                            "trend": float(df['SUPERT_7_3.0'].iloc[-1]),
                            "value": float(df['SUPERT_7_3.0'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    # Additional Trend Indicators
                    elif indicator == "psar":
                        df.ta.psar(append=True)
                        results[indicator] = {
                            "value": float(df['PSARl_0.02_0.2'].iloc[-1]),
                            "trend": float(df['PSARr_0.02_0.2'].iloc[-1])
                        }
                        valid_indicators += 1
                        
                    elif indicator == "trix":
                        df.ta.trix(length=20, append=True)
                        results[indicator] = {
                            "value": float(df['TRIX_20_9'].iloc[-1]),
                            "signal": float(df['TRIXs_20_9'].iloc[-1])
                        }
                        valid_indicators += 1
                        
                    elif indicator == "kst":
                        df.ta.kst(append=True)
                        results[indicator] = {
                            "value": float(df['KST_10_15_20_30_10_10_10_15'].iloc[-1]),
                            "signal": float(df['KSTs_9'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    # Volume Indicators
                    elif indicator == "volume":
                        volume_current = float(df['volume'].iloc[-1])
                        volume_sma = float(df['volume'].rolling(window=20).mean().iloc[-1])
                        results[indicator] = {
                            "current": volume_current,
                            "sma": volume_sma
                        }
                        valid_indicators += 1
                    
                    elif indicator == "obv":
                        df.ta.obv(append=True)
                        results[indicator] = {
                            "value": float(df['OBV'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    elif indicator == "cmf":
                        df.ta.cmf(append=True)
                        results[indicator] = {
                            "value": float(df['CMF_20'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    # Additional Volume Indicators
                    elif indicator == "vwap":
                        df.ta.vwap(append=True)
                        results[indicator] = {
                            "value": float(df['VWAP_D'].iloc[-1])
                        }
                        valid_indicators += 1
                        
                    elif indicator == "pvt":
                        df.ta.pvt(append=True)
                        results[indicator] = {
                            "value": float(df['PVT'].iloc[-1])
                        }
                        valid_indicators += 1
                        
                    elif indicator == "mfi":
                        # Convert volume to int64 before calculation
                        df['volume'] = df['volume'].fillna(0).astype('int64')  # Handle NaN values before conversion
                        df.ta.mfi(length=14, append=True)
                        results[indicator] = {
                            "value": float(df['MFI_14'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    # Momentum Indicators
                    elif indicator == "rsi":
                        df.ta.rsi(length=14, append=True)
                        results[indicator] = {
                            "value": float(df['RSI_14'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    elif indicator == "macd":
                        df.ta.macd(append=True)
                        results[indicator] = {
                            "value": float(df['MACD_12_26_9'].iloc[-1]),
                            "signal": float(df['MACDs_12_26_9'].iloc[-1]),
                            "histogram": float(df['MACDh_12_26_9'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    elif indicator == "stoch":
                        df.ta.stoch(append=True)
                        results[indicator] = {
                            "k": float(df['STOCHk_14_3_3'].iloc[-1]),
                            "d": float(df['STOCHd_14_3_3'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    # Additional Momentum Indicators
                    elif indicator == "cci":
                        df.ta.cci(length=20, append=True)
                        results[indicator] = {
                            "value": float(df['CCI_20_0.015'].iloc[-1])
                        }
                        valid_indicators += 1
                        
                    elif indicator == "roc":
                        df.ta.roc(length=12, append=True)
                        results[indicator] = {
                            "value": float(df['ROC_12'].iloc[-1])
                        }
                        valid_indicators += 1
                        
                    elif indicator == "willr":
                        df.ta.willr(length=14, append=True)
                        results[indicator] = {
                            "value": float(df['WILLR_14'].iloc[-1])
                        }
                        valid_indicators += 1
                        
                    elif indicator == "ao":
                        df.ta.ao(append=True)
                        results[indicator] = {
                            "value": float(df['AO_5_34'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    # Volatility Indicators
                    elif indicator == "bbands":
                        df.ta.bbands(length=5, append=True)
                        results[indicator] = {
                            "lower": float(df['BBL_5_2.0'].iloc[-1]),
                            "middle": float(df['BBM_5_2.0'].iloc[-1]),
                            "upper": float(df['BBU_5_2.0'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    elif indicator == "atr":
                        df.ta.atr(length=14, append=True)
                        results[indicator] = {
                            "value": float(df['ATRr_14'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    elif indicator == "stddev":
                        df.ta.stdev(length=20, append=True)
                        results[indicator] = {
                            "value": float(df['STDEV_20'].iloc[-1])
                        }
                        valid_indicators += 1
                    
                    # Additional Volatility Indicators
                    elif indicator == "keltner":
                        df.ta.kc(length=20, scalar=2, mamode="ema", append=True)
                        results[indicator] = {
                            "lower": float(df['KCLe_20_2'].iloc[-1]),
                            "middle": float(df['KCBe_20_2'].iloc[-1]),
                            "upper": float(df['KCUe_20_2'].iloc[-1])
                        }
                        valid_indicators += 1
                        
                    elif indicator == "donchian":
                        df.ta.donchian(length=20, append=True)
                        results[indicator] = {
                            "lower": float(df['DCL_20_20'].iloc[-1]),
                            "middle": float(df['DCM_20_20'].iloc[-1]),
                            "upper": float(df['DCU_20_20'].iloc[-1])
                        }
                        valid_indicators += 1
                        
                    # Price Action Indicators
                    elif indicator == "ichimoku":
                        df.ta.ichimoku(append=True)
                        results[indicator] = {
                            "tenkan": float(df['ISA_9'].iloc[-1]),  # Conversion line
                            "kijun": float(df['ISB_26'].iloc[-1]),  # Base line
                            "senkou_a": float(df['ITS_9'].iloc[-1]), # Leading Span A
                            "senkou_b": float(df['IKS_26'].iloc[-1]), # Leading Span B
                            "chikou": float(df['ICS_26'].iloc[-1])  # Lagging Span
                        }
                        valid_indicators += 1

                    elif indicator == "pivotpoints":
                        # Calculate Pivot Points (Classic method)
                        high = df['high'].iloc[-2]  # Previous period's high
                        low = df['low'].iloc[-2]    # Previous period's low
                        close = df['close'].iloc[-2] # Previous period's close
                        
                        pivot = (high + low + close) / 3
                        r1 = (2 * pivot) - low
                        r2 = pivot + (high - low)
                        r3 = high + 2 * (pivot - low)
                        s1 = (2 * pivot) - high
                        s2 = pivot - (high - low)
                        s3 = low - 2 * (high - pivot)
                        
                        results[indicator] = {
                            "pivot": float(pivot),
                            "r1": float(r1),
                            "r2": float(r2),
                            "r3": float(r3),
                            "s1": float(s1),
                            "s2": float(s2),
                            "s3": float(s3)
                        }
                        valid_indicators += 1

                    elif indicator == "fibonacciretracement":
                        # Calculate Fibonacci Retracement levels
                        # Find recent high and low for retracement
                        recent_high = df['high'].rolling(window=20).max().iloc[-1]
                        recent_low = df['low'].rolling(window=20).min().iloc[-1]
                        diff = recent_high - recent_low
                        
                        results[indicator] = {
                            "trend": "up" if df['close'].iloc[-1] > df['close'].iloc[-20] else "down",
                            "levels": {
                                "0.0": float(recent_low),
                                "0.236": float(recent_low + 0.236 * diff),
                                "0.382": float(recent_low + 0.382 * diff),
                                "0.5": float(recent_low + 0.5 * diff),
                                "0.618": float(recent_low + 0.618 * diff),
                                "0.786": float(recent_low + 0.786 * diff),
                                "1.0": float(recent_high)
                            }
                        }
                        valid_indicators += 1

                    elif indicator == "candlepatterns":
                        # Detect common candlestick patterns
                        patterns = {}
                        
                        # Doji pattern
                        doji_threshold = 0.1  # Maximum difference between open and close for doji
                        latest_candle = df.iloc[-1]
                        body = abs(latest_candle['open'] - latest_candle['close'])
                        upper_shadow = latest_candle['high'] - max(latest_candle['open'], latest_candle['close'])
                        lower_shadow = min(latest_candle['open'], latest_candle['close']) - latest_candle['low']
                        avg_price = (latest_candle['high'] + latest_candle['low']) / 2
                        
                        patterns["doji"] = bool(body <= (avg_price * doji_threshold))
                        
                        # Hammer/Hanging Man
                        body_to_shadow_ratio = 0.3
                        is_hammer_shape = (body > 0) and (lower_shadow > (body / body_to_shadow_ratio)) and (upper_shadow < body)
                        patterns["hammer"] = bool(is_hammer_shape and df['close'].iloc[-2] > df['close'].iloc[-1])
                        patterns["hanging_man"] = bool(is_hammer_shape and df['close'].iloc[-2] < df['close'].iloc[-1])
                        
                        # Engulfing patterns
                        prev_candle = df.iloc[-2]
                        prev_body = abs(prev_candle['open'] - prev_candle['close'])
                        is_bullish_engulfing = (latest_candle['close'] > latest_candle['open'] and 
                                              prev_candle['close'] < prev_candle['open'] and
                                              latest_candle['close'] > prev_candle['open'] and
                                              latest_candle['open'] < prev_candle['close'])
                        is_bearish_engulfing = (latest_candle['close'] < latest_candle['open'] and 
                                              prev_candle['close'] > prev_candle['open'] and
                                              latest_candle['close'] < prev_candle['open'] and
                                              latest_candle['open'] > prev_candle['close'])
                        
                        patterns["bullish_engulfing"] = bool(is_bullish_engulfing)
                        patterns["bearish_engulfing"] = bool(is_bearish_engulfing)
                        
                        results[indicator] = patterns
                        valid_indicators += 1

                    elif indicator == "harmonicpatterns":
                        # Detect harmonic patterns (Gartley, Butterfly, Bat, Crab)
                        # Using last 5 swing points
                        swings = []
                        direction = 1  # 1 for up, -1 for down
                        
                        # Find swing points
                        for i in range(len(df)-3):
                            if direction == 1:  # Looking for high
                                if df['high'].iloc[i+1] > df['high'].iloc[i] and df['high'].iloc[i+1] > df['high'].iloc[i+2]:
                                    swings.append({"price": df['high'].iloc[i+1], "type": "high"})
                                    direction = -1
                            else:  # Looking for low
                                if df['low'].iloc[i+1] < df['low'].iloc[i] and df['low'].iloc[i+1] < df['low'].iloc[i+2]:
                                    swings.append({"price": df['low'].iloc[i+1], "type": "low"})
                                    direction = 1
                            
                            if len(swings) >= 5:
                                break
                        
                        # Calculate retracement ratios if we have enough swing points
                        patterns = {}
                        if len(swings) >= 5:
                            # XA, AB, BC, CD moves
                            moves = []
                            for i in range(len(swings)-1):
                                moves.append(abs(swings[i+1]["price"] - swings[i]["price"]))
                            
                            # Calculate ratios
                            if len(moves) >= 4:
                                ab_xa = moves[1] / moves[0] if moves[0] != 0 else 0
                                bc_ab = moves[2] / moves[1] if moves[1] != 0 else 0
                                cd_bc = moves[3] / moves[2] if moves[2] != 0 else 0
                                
                                # Gartley Pattern
                                patterns["gartley"] = {
                                    "valid": bool(
                                        0.618 <= ab_xa <= 0.618 and
                                        0.382 <= bc_ab <= 0.886 and
                                        1.27 <= cd_bc <= 1.618
                                    ),
                                    "completion": float(df['close'].iloc[-1])
                                }
                                
                                # Butterfly Pattern
                                patterns["butterfly"] = {
                                    "valid": bool(
                                        0.786 <= ab_xa <= 0.786 and
                                        0.382 <= bc_ab <= 0.886 and
                                        1.618 <= cd_bc <= 2.618
                                    ),
                                    "completion": float(df['close'].iloc[-1])
                                }
                        
                        results[indicator] = patterns
                        valid_indicators += 1

                    elif indicator == "pricelevels":
                        # Calculate key price levels and support/resistance
                        window = len(df)  # Use entire dataset
                        current_price = float(df['close'].iloc[-1])
                        
                        # Calculate minimum distance threshold (1% of current price)
                        min_distance = current_price * 0.01
                        
                        # Find potential support levels (recent lows)
                        lows = df['low'].rolling(window=50).min()  # Increased from 5 to 50 periods
                        support_levels = set()
                        for i in range(len(df)-window, len(df)):
                            price = round(float(df['low'].iloc[i]), 2)
                            # Only add if it's a valid support level (below current price and not too close)
                            if (lows.iloc[i] == df['low'].iloc[i] and 
                                price < current_price and 
                                (current_price - price) > min_distance):
                                support_levels.add(price)
                        
                        # Find potential resistance levels (recent highs)
                        highs = df['high'].rolling(window=50).max()  # Increased from 5 to 50 periods
                        resistance_levels = set()
                        for i in range(len(df)-window, len(df)):
                            price = round(float(df['high'].iloc[i]), 2)
                            # Only add if it's a valid resistance level (above current price and not too close)
                            if (highs.iloc[i] == df['high'].iloc[i] and 
                                price > current_price and 
                                (price - current_price) > min_distance):
                                resistance_levels.add(price)

                        # Calculate high volume levels
                        volume_weighted_levels = []
                        price_range = df['high'].max() - df['low'].min()
                        bin_size = price_range / 200  # 200 bins for granularity
                        
                        for i in range(200):
                            price_level = df['low'].min() + bin_size * i
                            mask = (df['low'] <= price_level + bin_size*1.5) & (df['high'] >= price_level - bin_size*1.5)
                            volume_at_level = df[mask]['volume'].sum()
                            volume_weighted_levels.append((price_level, volume_at_level))
                        
                        # Sort by volume and get top levels
                        volume_weighted_levels.sort(key=lambda x: x[1], reverse=True)
                        high_volume_levels = [level[0] for level in volume_weighted_levels[:10]]  # Take top 10 volume levels
                        
                        # Store results with more levels
                        results[indicator] = {
                            "support_levels": list(sorted(support_levels))[:6],  # Take first 6 support levels
                            "resistance_levels": list(sorted(resistance_levels))[:6],  # Take first 6 resistance levels
                            "high_volume_levels": high_volume_levels,
                            "current_price": current_price
                        }
                        valid_indicators += 1
                    
                except Exception as e:
                    print(f"Error calculating {indicator}: {str(e)}")
                    continue
            
            # Store DataFrame for potential chart generation
            self._current_df = df.iloc[-100:]  # Slice for charting
            
            return results if valid_indicators >= MIN_REQUIRED_INDICATORS else None
            
        except Exception as e:
            print(f"Error in _fetch_indicators: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            return None

    def _convert_large_floats(self, obj):
        """Recursively convert large float values to strings to ensure JSON compliance.
        
        Args:
            obj: Any Python object that might contain float values
            
        Returns:
            The same object with large floats converted to strings
        """
        if isinstance(obj, float):
            # Convert large floats to strings
            if abs(obj) > 1e10:
                return str(obj)
            return obj
        elif isinstance(obj, dict):
            return {k: self._convert_large_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_large_floats(item) for item in obj]
        return obj

    def format_indicators_json(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Format raw indicator data into a structured JSON format."""
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

            # Convert any large float values to strings before returning
            return self._convert_large_floats(formatted)

        except Exception as e:
            print(f"Error formatting indicators: {str(e)}")
            return {}

    def generate_analysis(
        self, indicators: Dict[str, Any], symbol: str, timeframe: str, 
        user_prompt: str, system_prompt: Optional[str] = None,
        strategy_type: str = DEFAULT_STRATEGY, analysis_focus: List[str] = None
    ) -> str:
        """Generate a comprehensive technical analysis using GPT-4."""
        try:
            time_horizon = INTERVAL_HORIZONS.get(timeframe, "medium-term")
            
            if not system_prompt:
                system_prompt = """You are an expert technical analyst with a comprehensive view of market structure. Your goal is to provide clear, actionable insights that cover both immediate trading opportunities and longer-term price targets. When analyzing support and resistance levels: 1. ALWAYS mention both near-term and far-term levels 2. Include multiple take-profit targets at key resistance/support zones 3. Don't be overly conservative - if there are significant levels far from current price, include them 4. Explain your rationale for both conservative and aggressive targets Focus on being clear and natural in your explanations, avoiding rigid structures unless they serve the analysis."""

            # Format indicators for better readability
            formatted_indicators = json.dumps(indicators, indent=2)
            print("\nDebug - Indicators being passed to LLM:")
            print(formatted_indicators)
            
            analysis_request = f"""I need your expert analysis on {symbol} (USD) on the {timeframe} timeframe, which typically suits {time_horizon} trading horizons. The user's original question was: "{user_prompt}"

The analysis should naturally flow from your expertise, but keep these key points in mind:
- Always discuss BOTH immediate price levels and significant far-out levels that could be important
- When suggesting targets, include a range from conservative near-term to aggressive longer-term targets
- Support your analysis with the relevant indicators, explaining which signals you find most compelling
- If you see significant levels far from the current price, don't hesitate to mention them - traders need to know both immediate opportunities and bigger picture targets
- Consider high-volume price zones as potential targets, especially where they align with other technical levels

Here are the technical indicators I've calculated for your analysis:

{formatted_indicators}

Please provide your comprehensive analysis, keeping in mind the original question while ensuring you cover both near-term opportunities and longer-term potential."""
            
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

            return response.choices[0].message.content

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

    def _get_fallback_symbols(self) -> List[str]:
        """Provide a fallback list of common coins.

        Returns:
            List[str]: List of common cryptocurrency symbols
        """
        print("\nUsing fallback symbol list")
        return [
            "BTC",
            "ETH",
            "BNB",
            "SOL",
            "XRP",
            "ADA",
            "DOGE",
            "MATIC",
            "DOT",
            "LTC",
            "AVAX",
            "LINK",
            "UNI",
            "ATOM",
            "ETC",
            "XLM",
            "ALGO",
            "NEAR",
            "FTM",
            "SAND"
        ]


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


def test_indicators():
    """Test all technical indicators using BTC data."""
    print("\nRunning Technical Indicator Tests for BTC")
    print("=" * 80)
    
    ta_tool = TechnicalAnalysis()
    
    try:
        # Fetch BTC candle data
        print("\n1. Fetching BTC Candle Data...")
        candle_data = ta_tool.fetch_candle_data("BTC", "1h")
        if not candle_data:
            print("❌ Failed to fetch candle data")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(candle_data)
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"✓ Successfully fetched {len(df)} candles")
        
        # Test each indicator category
        print("\n2. Testing Moving Averages...")
        try:
            # SMA
            df.ta.sma(length=20, append=True)
            print("✓ SMA_20:", float(df['SMA_20'].iloc[-1]))
            
            # EMA
            df.ta.ema(length=20, append=True)
            print("✓ EMA_20:", float(df['EMA_20'].iloc[-1]))
            
            # DEMA
            df.ta.dema(length=20, append=True)
            print("✓ DEMA_20:", float(df['DEMA_20'].iloc[-1]))
            
            # TEMA
            df.ta.tema(length=20, append=True)
            print("✓ TEMA_20:", float(df['TEMA_20'].iloc[-1]))
            
            # VWMA
            df.ta.vwma(length=20, append=True)
            print("✓ VWMA_20:", float(df['VWMA_20'].iloc[-1]))
            
            # WMA (New)
            df.ta.wma(length=20, append=True)
            print("✓ WMA_20:", float(df['WMA_20'].iloc[-1]))
            
            # HMA (New)
            df.ta.hma(length=20, append=True)
            print("✓ HMA_20:", float(df['HMA_20'].iloc[-1]))
        except Exception as e:
            print(f"❌ Error in Moving Averages: {str(e)}")
        
        print("\n3. Testing Trend Indicators...")
        try:
            # ADX
            df.ta.adx(append=True)
            print("✓ ADX_14:", float(df['ADX_14'].iloc[-1]))
            
            # Supertrend
            df.ta.supertrend(append=True)
            print("✓ Supertrend:", float(df['SUPERT_7_3.0'].iloc[-1]))
            
            # PSAR (New)
            df.ta.psar(append=True)
            print("✓ PSAR:", float(df['PSARl_0.02_0.2'].iloc[-1]))
            
            # TRIX (New)
            df.ta.trix(length=20, append=True)
            print("✓ TRIX:", float(df['TRIX_20_9'].iloc[-1]))
            
            # KST (New)
            df.ta.kst(append=True)
            print("✓ KST:", float(df['KST_10_15_20_30_10_10_10_15'].iloc[-1]))
        except Exception as e:
            print(f"❌ Error in Trend Indicators: {str(e)}")
        
        print("\n4. Testing Momentum Indicators...")
        try:
            # RSI
            df.ta.rsi(length=14, append=True)
            print("✓ RSI_14:", float(df['RSI_14'].iloc[-1]))
            
            # MACD
            df.ta.macd(append=True)
            print("✓ MACD:", float(df['MACD_12_26_9'].iloc[-1]))
            
            # Stochastic
            df.ta.stoch(append=True)
            print("✓ Stoch_K:", float(df['STOCHk_14_3_3'].iloc[-1]))
            print("✓ Stoch_D:", float(df['STOCHd_14_3_3'].iloc[-1]))
            
            # CCI (New)
            df.ta.cci(length=20, append=True)
            print("✓ CCI:", float(df['CCI_20_0.015'].iloc[-1]))
            
            # ROC (New)
            df.ta.roc(length=12, append=True)
            print("✓ ROC:", float(df['ROC_12'].iloc[-1]))
            
            # Williams %R (New)
            df.ta.willr(length=14, append=True)
            print("✓ Williams %R:", float(df['WILLR_14'].iloc[-1]))
            
            # Awesome Oscillator (New)
            df.ta.ao(append=True)
            print("✓ AO:", float(df['AO_5_34'].iloc[-1]))
        except Exception as e:
            print(f"❌ Error in Momentum Indicators: {str(e)}")
        
        print("\n5. Testing Volume Indicators...")
        try:
            # Volume SMA
            volume_sma = df['volume'].rolling(window=20).mean()
            print("✓ Volume SMA:", float(volume_sma.iloc[-1]))
            
            # OBV
            df.ta.obv(append=True)
            print("✓ OBV:", float(df['OBV'].iloc[-1]))
            
            # CMF
            df.ta.cmf(append=True)
            print("✓ CMF:", float(df['CMF_20'].iloc[-1]))
            
            # VWAP (New)
            df.ta.vwap(append=True)
            print("✓ VWAP:", float(df['VWAP_D'].iloc[-1]))
            
            # PVT (New)
            df.ta.pvt(append=True)
            print("✓ PVT:", float(df['PVT'].iloc[-1]))
            
            # MFI (New)
            df['volume'] = df['volume'].fillna(0).astype('int64')  # Handle NaN values before conversion
            df.ta.mfi(length=14, append=True)
            print("✓ MFI:", float(df['MFI_14'].iloc[-1]))
        except Exception as e:
            print(f"❌ Error in Volume Indicators: {str(e)}")
        
        print("\n6. Testing Volatility Indicators...")
        try:
            # Bollinger Bands
            df.ta.bbands(length=20, append=True)
            print("✓ BB Upper:", float(df['BBU_20_2.0'].iloc[-1]))
            print("✓ BB Middle:", float(df['BBM_20_2.0'].iloc[-1]))
            print("✓ BB Lower:", float(df['BBL_20_2.0'].iloc[-1]))
            
            # ATR
            df.ta.atr(length=14, append=True)
            print("✓ ATR:", float(df['ATRr_14'].iloc[-1]))
            
            # Standard Deviation
            df.ta.stdev(length=20, append=True)
            print("✓ StdDev:", float(df['STDEV_20'].iloc[-1]))
            
            # Keltner Channels (New)
            df.ta.kc(length=20, scalar=2, mamode="ema", append=True)
            print("✓ KC Upper:", float(df['KCUe_20_2.0'].iloc[-1]))
            print("✓ KC Middle:", float(df['KCBe_20_2.0'].iloc[-1]))
            print("✓ KC Lower:", float(df['KCLe_20_2.0'].iloc[-1]))
            
            # Donchian Channels (New)
            df.ta.donchian(length=20, append=True)
            print("✓ DC Upper:", float(df['DCU_20_20'].iloc[-1]))
            print("✓ DC Middle:", float(df['DCM_20_20'].iloc[-1]))
            print("✓ DC Lower:", float(df['DCL_20_20'].iloc[-1]))
        except Exception as e:
            print(f"❌ Error in Volatility Indicators: {str(e)}")
            
        print("\n7. Testing Price Action Indicators...")
        try:
            # Ichimoku Cloud
            df.ta.ichimoku(append=True)
            print("✓ Ichimoku Tenkan:", float(df['ISA_9'].iloc[-1]))
            print("✓ Ichimoku Kijun:", float(df['ISB_26'].iloc[-1]))
            print("✓ Ichimoku Senkou A:", float(df['ITS_9'].iloc[-1]))
            print("✓ Ichimoku Senkou B:", float(df['IKS_26'].iloc[-1]))

            # Pivot Points
            high = df['high'].iloc[-2]
            low = df['low'].iloc[-2]
            close = df['close'].iloc[-2]
            pivot = (high + low + close) / 3
            print("✓ Pivot Point:", float(pivot))
            print("✓ R1:", float((2 * pivot) - low))
            print("✓ S1:", float((2 * pivot) - high))

            # Fibonacci Retracement
            recent_high = df['high'].rolling(window=20).max().iloc[-1]
            recent_low = df['low'].rolling(window=20).min().iloc[-1]
            diff = recent_high - recent_low
            print("✓ Fibonacci 0.618:", float(recent_low + 0.618 * diff))
            print("✓ Fibonacci 0.5:", float(recent_low + 0.5 * diff))
            print("✓ Fibonacci 0.382:", float(recent_low + 0.382 * diff))

            # Candlestick Patterns
            latest_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            body = abs(latest_candle['open'] - latest_candle['close'])
            upper_shadow = latest_candle['high'] - max(latest_candle['open'], latest_candle['close'])
            lower_shadow = min(latest_candle['open'], latest_candle['close']) - latest_candle['low']
            avg_price = (latest_candle['high'] + latest_candle['low']) / 2
            
            # Doji pattern check
            doji_threshold = 0.1
            is_doji = bool(body <= (avg_price * doji_threshold))
            print("✓ Doji Pattern Present:", is_doji)
            
            # Hammer pattern check
            body_to_shadow_ratio = 0.3
            is_hammer = bool((body > 0) and (lower_shadow > (body / body_to_shadow_ratio)) and (upper_shadow < body))
            print("✓ Hammer Pattern Present:", is_hammer)

            # Harmonic Patterns
            # Find swing points
            swings = []
            direction = 1
            for i in range(len(df)-3):
                if direction == 1 and df['high'].iloc[i+1] > df['high'].iloc[i] and df['high'].iloc[i+1] > df['high'].iloc[i+2]:
                    swings.append({"price": df['high'].iloc[i+1], "type": "high"})
                    direction = -1
                elif direction == -1 and df['low'].iloc[i+1] < df['low'].iloc[i] and df['low'].iloc[i+1] < df['low'].iloc[i+2]:
                    swings.append({"price": df['low'].iloc[i+1], "type": "low"})
                    direction = 1
                if len(swings) >= 5:
                    break
            print("✓ Identified Swing Points:", len(swings))

            # Price Levels
            window = 20
            # Support levels
            lows = df['low'].rolling(window=5).min()
            support_levels = set()
            for i in range(len(df)-window, len(df)):
                if lows.iloc[i] == df['low'].iloc[i]:
                    support_levels.add(round(float(df['low'].iloc[i]), 2))
            print("✓ Support Levels Found:", len(support_levels))

            # Resistance levels
            highs = df['high'].rolling(window=5).max()
            resistance_levels = set()
            for i in range(len(df)-window, len(df)):
                if highs.iloc[i] == df['high'].iloc[i]:
                    resistance_levels.add(round(float(df['high'].iloc[i]), 2))
            print("✓ Resistance Levels Found:", len(resistance_levels))

            # Volume Profile
            price_volume = {}
            for i in range(len(df)-window, len(df)):
                price = round(df['close'].iloc[i], 2)
                volume = df['volume'].iloc[i]
                if price in price_volume:
                    price_volume[price] += volume
            else:
                    price_volume[price] = volume
            sorted_prices = sorted(price_volume.items(), key=lambda x: x[1], reverse=True)
            high_volume_levels = [float(price) for price, _ in sorted_prices[:3]]
            print("✓ High Volume Nodes Found:", len(high_volume_levels))
                
        except Exception as e:
            print(f"❌ Error in Price Action Indicators: {str(e)}")
            import traceback
            traceback.print_exc()

        # Print DataFrame info
        print("\n8. Testing DataFrame Structure...")
        print("\nDataFrame Info:")
        print(df.info())
        print("\nAvailable Columns:")
        print(df.columns.tolist())
        
        print("\nTest Complete!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

def get_latest_chart() -> Optional[str]:
    """Get the latest generated chart from the singleton instance."""
    ta = get_ta_instance()
    return ta.get_latest_chart()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_indicators()
    elif len(sys.argv) > 1:
        # Use the command line argument as the prompt
        prompt = sys.argv[1]
        try:
            ta_tool = TechnicalAnalysis()
            result = ta_tool.run(prompt)
            print("\nANALYSIS RESULTS:")
            print("=" * 80)
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(result["response"])
                print("\nMETADATA:")
                print("=" * 80)
                print(json.dumps(result["metadata"], indent=2))
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("Please provide an analysis prompt as a command line argument")

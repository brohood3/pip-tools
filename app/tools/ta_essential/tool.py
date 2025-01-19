"""
Essential Technical Analysis Tool

A focused technical analysis tool using key indicators:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Trend Indicators (Supertrend, ADX, DMI)
- Volume Analysis
- Standard Deviation
- MACD
- Bollinger Bands
- RSI
- Fibonacci Retracements
- Ichimoku Cloud
"""

import os
import requests
from typing import Dict, Optional, List, TypedDict, Union, Any
from datetime import datetime
from dotenv import load_dotenv
import json
from openai import OpenAI
from fastapi import HTTPException

# Load environment variables
load_dotenv()


class TechnicalIndicators(TypedDict):
    """TypedDict defining the structure of technical indicators."""

    # Moving Averages
    sma: Dict[str, float]  # Multiple SMAs with different periods
    ema: Dict[str, float]  # Multiple EMAs with different periods

    # Trend Indicators
    supertrend: Dict[str, Union[float, str]]  # value and valueAdvice ("long"/"short")
    adx: Dict[str, float]  # Single value
    dmi: Dict[str, float]  # pdi, mdi values

    # Volume and Volatility
    volume: Dict[str, float]  # Current volume
    stddev: Dict[str, float]  # Standard deviation

    # Momentum and Oscillators
    macd: Dict[str, float]  # valueMACD, valueMACDSignal, valueMACDHist
    rsi: Dict[str, float]  # Single value

    # Price Bands and Support/Resistance
    bbands: Dict[str, float]  # Upper, middle, lower bands
    fibonacciretracement: Dict[str, Any]  # Multiple Fibonacci levels

    # Complex Indicators
    ichimoku: Dict[str, float]  # Multiple Ichimoku components


class TechnicalAnalysis:
    """Essential Technical Analysis tool using TAapi and GPT-4."""

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

    def run(self, prompt: str) -> Dict[str, Any]:
        """Main entry point for the technical analysis tool.

        Args:
            prompt: User's analysis request

        Returns:
            Dict containing analysis results and metadata
        """
        try:
            # Extract token and interval from prompt
            token, interval = self.parse_prompt_with_llm(prompt)
            if not token:
                return {
                    "error": "Could not determine which token to analyze. Please specify a token."
                }

            # Get available symbols and find best pair
            available_symbols = self.get_available_symbols()
            if not available_symbols:
                return {
                    "error": "Could not fetch available trading pairs. Please try again later."
                }

            pair = self.find_best_pair(token, available_symbols)
            if not pair:
                return {
                    "error": f"No trading pair found for {token}. Please verify the token symbol and try again."
                }

            # Fetch indicators
            indicators = self._fetch_indicators(pair, interval=interval)
            if not indicators:
                return {
                    "error": f"Insufficient data for {pair} on {interval} timeframe."
                }

            # Generate analysis
            analysis = self.generate_analysis(indicators, pair, interval, prompt)

            # Store all context in metadata
            metadata = {
                "prompt": prompt,
                "token": token,
                "pair": pair,
                "interval": interval,
                "timestamp": datetime.now().isoformat(),
                "data_quality": "partial" if len(indicators) < 8 else "full",
                "technical_indicators": self.format_indicators_json(indicators),
            }

            return {"response": analysis, "metadata": metadata}

        except Exception as e:
            return {"error": str(e)}

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
            token = token.replace("/USDT", "")

            # Common token name mappings
            token_mappings = {
                "BITCOIN": "BTC",
                "ETHEREUM": "ETH",
                "CARDANO": "ADA",
                "SOLANA": "SOL",
                "POLYGON": "MATIC",
                "POLKADOT": "DOT",
                "CHAINLINK": "LINK",
                "AVALANCHE": "AVAX",
                "DOGECOIN": "DOGE",
                "RIPPLE": "XRP",
                "LITECOIN": "LTC",
                "COSMOS": "ATOM",
                "BINANCE COIN": "BNB",
                "BNB COIN": "BNB",
            }

            # Try to map full name to symbol
            if token in token_mappings:
                token = token_mappings[token]

            # First try exact USDT pair match
            exact_match = f"{token}/USDT"
            if exact_match in available_symbols:
                print(f"\nFound exact match: {exact_match}")
                return exact_match

            # Then try case-insensitive match
            for symbol in available_symbols:
                base = symbol.split("/")[0]  # Get base token
                if base.upper() == token:
                    print(f"\nFound case-insensitive match: {symbol}")
                    return symbol

            # Finally try partial matches
            partial_matches = []
            for symbol in available_symbols:
                base = symbol.split("/")[0]  # Get base token
                # Check if token is contained in base or base in token
                if token in base.upper() or base.upper() in token:
                    partial_matches.append(symbol)

            if partial_matches:
                # Prioritize shorter matches as they're likely more accurate
                partial_matches.sort(key=lambda x: len(x.split("/")[0]))
                best_match = partial_matches[0]
                print(f"\nFound partial match: {best_match}")
                return best_match

            print(f"\nNo matching pair found for token: {token}")
            return None

        except Exception as e:
            print(f"\nError finding best pair: {str(e)}")
            return None

    def _fetch_indicators(
        self, symbol: str, interval: str = "1d", exchange: str = "gateio"
    ) -> Optional[Dict[str, Any]]:
        """Fetch technical indicators using TAapi."""
        try:
            url = f"{self.taapi_base_url}/bulk"

            # Define our essential indicators
            indicators = [
                # Moving Averages
                {"indicator": "sma", "period": 20},
                {"indicator": "sma", "period": 50},
                {"indicator": "sma", "period": 200},
                {"indicator": "ema", "period": 20},
                {"indicator": "ema", "period": 50},
                {"indicator": "ema", "period": 200},
                # Trend Indicators
                {"indicator": "supertrend"},
                {"indicator": "adx"},
                {"indicator": "dmi"},
                # Volume and Volatility
                {"indicator": "volume"},
                {"indicator": "stddev"},
                # Momentum and Oscillators
                {"indicator": "macd"},
                {"indicator": "rsi"},
                # Price Bands and Support/Resistance
                {"indicator": "bbands"},
                {"indicator": "fibonacciretracement"},
                # Complex Indicators
                {"indicator": "ichimoku"},
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
                if indicator_data.get("result") and not indicator_data.get(
                    "result"
                ).get("error"):
                    if indicator_data["indicator"] in ["sma", "ema"]:
                        # Handle moving averages with different periods
                        indicator_type = indicator_data["indicator"]
                        if indicator_type not in result:
                            result[indicator_type] = {}
                        period = indicator_data.get("period", "20")
                        result[indicator_type][f"period_{period}"] = indicator_data[
                            "result"
                        ].get("value")
                    else:
                        # Handle other indicators
                        valid_indicators += 1
                        result[indicator_data["indicator"]] = indicator_data["result"]

            # Return None if we don't have enough valid indicators
            if valid_indicators < 5:
                print(f"\nInsufficient valid indicators: {valid_indicators}")
                return None

            return result

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error fetching indicators: {str(e)}"
            )

    def format_indicators_json(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Format indicators into a clean JSON structure."""
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
        self,
        indicators: Dict[str, Any],
        symbol: str,
        interval: str,
        original_prompt: str,
    ) -> str:
        """Generate an opinionated technical analysis report using GPT-4."""
        # Map intervals to human-readable time horizons
        interval_horizons = {
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

        time_horizon = interval_horizons.get(interval, "medium-term")

        # Build indicator sections based on available data
        indicator_sections = []

        # Moving Averages
        moving_averages = []
        for ma_type in ["sma", "ema"]:
            if ma_type in indicators:
                ma_values = []
                for period in ["20", "50", "200"]:
                    if f"period_{period}" in indicators[ma_type]:
                        ma_values.append(
                            f"{period} [{indicators[ma_type][f'period_{period}']:.2f}]"
                        )
                if ma_values:
                    moving_averages.append(
                        f"• {ma_type.upper()}: {', '.join(ma_values)}"
                    )

        if moving_averages:
            indicator_sections.append("Moving Averages:\n" + "\n".join(moving_averages))

        # Trend Indicators
        trend_indicators = []
        if "supertrend" in indicators and "value" in indicators["supertrend"]:
            trend_indicators.append(
                f"• Supertrend: {indicators['supertrend']['value']:.2f} (Signal: {indicators['supertrend'].get('valueAdvice', 'N/A')})"
            )

        if "adx" in indicators and "value" in indicators["adx"]:
            trend_indicators.append(f"• ADX: {indicators['adx']['value']:.2f}")

        if "dmi" in indicators and all(k in indicators["dmi"] for k in ["pdi", "mdi"]):
            trend_indicators.append(
                f"• DMI: +DI {indicators['dmi']['pdi']:.2f}, -DI {indicators['dmi']['mdi']:.2f}"
            )

        if trend_indicators:
            indicator_sections.append(
                "Trend Indicators:\n" + "\n".join(trend_indicators)
            )

        # Volume and Volatility
        volume_indicators = []
        if "volume" in indicators and "value" in indicators["volume"]:
            volume_indicators.append(f"• Volume: {indicators['volume']['value']:.2f}")

        if "stddev" in indicators and "value" in indicators["stddev"]:
            volume_indicators.append(
                f"• Standard Deviation: {indicators['stddev']['value']:.2f}"
            )

        if volume_indicators:
            indicator_sections.append(
                "Volume & Volatility:\n" + "\n".join(volume_indicators)
            )

        # Momentum Indicators
        momentum_indicators = []
        if "macd" in indicators and all(
            k in indicators["macd"]
            for k in ["valueMACD", "valueMACDSignal", "valueMACDHist"]
        ):
            macd = indicators["macd"]
            momentum_indicators.append(
                f"• MACD: Line [{macd['valueMACD']:.2f}], Signal [{macd['valueMACDSignal']:.2f}], Hist [{macd['valueMACDHist']:.2f}]"
            )

        if "rsi" in indicators and "value" in indicators["rsi"]:
            momentum_indicators.append(f"• RSI: {indicators['rsi']['value']:.2f}")

        if momentum_indicators:
            indicator_sections.append(
                "Momentum Indicators:\n" + "\n".join(momentum_indicators)
            )

        # Price Bands and Support/Resistance
        price_indicators = []
        if "bbands" in indicators and all(
            k in indicators["bbands"]
            for k in ["valueUpperBand", "valueMiddleBand", "valueLowerBand"]
        ):
            bb = indicators["bbands"]
            price_indicators.append(
                f"• Bollinger Bands: Upper[{bb['valueUpperBand']:.2f}], Mid[{bb['valueMiddleBand']:.2f}], Lower[{bb['valueLowerBand']:.2f}]"
            )

        if "fibonacciretracement" in indicators:
            fib = indicators["fibonacciretracement"]
            if all(k in fib for k in ["value", "trend", "startPrice", "endPrice"]):
                price_indicators.append(
                    f"• Fibonacci: {fib['value']:.2f} ({fib['trend']})"
                )
                price_indicators.append(
                    f"  Range: {fib['startPrice']} → {fib['endPrice']}"
                )

        if price_indicators:
            indicator_sections.append(
                "Price Structure:\n" + "\n".join(price_indicators)
            )

        # Ichimoku Cloud
        if "ichimoku" in indicators:
            cloud_indicators = []
            ichimoku = indicators["ichimoku"]

            if "conversion" in ichimoku:
                cloud_indicators.append(
                    f"• Conversion Line (Tenkan-sen): {ichimoku['conversion']:.2f}"
                )
            if "base" in ichimoku:
                cloud_indicators.append(
                    f"• Base Line (Kijun-sen): {ichimoku['base']:.2f}"
                )
            if "spanA" in ichimoku:
                cloud_indicators.append(
                    f"• Leading Span A (Senkou Span A): {ichimoku['spanA']:.2f}"
                )
            if "spanB" in ichimoku:
                cloud_indicators.append(
                    f"• Leading Span B (Senkou Span B): {ichimoku['spanB']:.2f}"
                )
            if "lagging" in ichimoku:
                cloud_indicators.append(
                    f"• Lagging Span (Chikou Span): {ichimoku['lagging']:.2f}"
                )

            if cloud_indicators:
                indicator_sections.append(
                    "Ichimoku Cloud:\n" + "\n".join(cloud_indicators)
                )

        # Build the context for GPT-4
        context = f"""You are a professional technical trader specializing in cryptocurrency markets.
Your analysis focuses on essential indicators to identify high-probability trading setups.

ANALYSIS REQUEST:
Original Query: "{original_prompt}"
Asset: {symbol}
Timeframe: {interval} candles
Trading Horizon: {time_horizon}
Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
Data Quality: {'Partial (Limited Historical Data)' if len(indicators) < 8 else 'Full'}

TECHNICAL INDICATORS:

{chr(10).join(indicator_sections)}

ANALYSIS REQUIREMENTS:

1. MARKET CONTEXT (20%)
- Current market phase (ranging/trending)
- Volume profile analysis
- Key support/resistance zones
- Market structure (Higher Highs/Lower Lows)

2. TRADE SETUP (35%)
- Primary trade direction
- Entry trigger conditions
- Stop loss placement
- Take profit levels
- Position sizing recommendation
- Risk:Reward calculation

3. INDICATOR SIGNALS (25%)
- Trend indicator alignment
- Momentum confirmation
- Price action patterns
- Volume confirmation
- Divergence analysis

4. EXECUTION PLAN (20%)
- Entry order types and placement
- Stop loss management rules
- Take profit strategy (partial/full exits)
- Position management rules
- Setup invalidation criteria

IMPORTANT GUIDELINES:
- Focus on {time_horizon} trading opportunities
- Prioritize high-probability setups
- Use specific price levels for entries/exits
- Emphasize risk management rules
- Consider multiple timeframe alignment
- Highlight setup expiration conditions

Your analysis should be precise and actionable, focusing on clear trade setups with defined entry, exit, and risk management rules."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a technical trader focusing on essential indicators and clear trade setups. Your analysis emphasizes precise entry/exit levels, risk management, and {time_horizon} trading opportunities. Use direct, actionable language.",
                    },
                    {"role": "user", "content": context},
                ],
                temperature=0.7,
                max_tokens=2000,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating analysis: {str(e)}"
            )


# added the following to have uniformity in the way we call tools
def run(prompt: str) -> Dict[str, Any]:
    return TechnicalAnalysis().run(prompt)

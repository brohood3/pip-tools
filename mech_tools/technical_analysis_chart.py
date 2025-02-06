"""
Technical Analysis Chart Generator

A tool for generating technical analysis charts for cryptocurrency pairs using pandas and pandas_ta.
Supports multiple technical indicators including trend, momentum, volume and volatility indicators.
Generates interactive charts with price data and selected technical indicators.
"""

import os
import requests
from typing import Dict, Optional, List, Any
from datetime import datetime
from dotenv import load_dotenv
import json
from openai import OpenAI
from fastapi import HTTPException

import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Load environment variables
load_dotenv()

valid_indicators = [
    # Trend Indicators (3)
    "ema",
    "supertrend", 
    "psar",
    # Momentum Indicators (2)
    "rsi",
    "mfi",
    # Volume & Volatility (2)
    "bbands",
    "vwap",
    # Support/Resistance & Additional Indicators (3)
    "roc",
    "mom",
    "wma",
]

class ChartGenerator:
    """Chart generation tool for technical analysis using market data and indicators."""

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
        """Main entry point for the technical analysis tool.

        Args:
            prompt: User's analysis request
            system_prompt: Optional custom system prompt for the analysis

        Returns:
            Dict containing analysis results and metadata
        """
        try:
            # Extract token and interval from prompt
            token, interval, indicators = self.parse_prompt_with_llm(prompt)
            if not token:
                return {
                    "error": "Could not determine which token to analyze. Please specify a token."
                }
            
            print(f"Found the following indicators: {indicators}")

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

            # Fetch Candle data
            candle_data = self.fetch_candle_data(pair, interval=interval)

            print(f"Candle data: {candle_data}")
            if not candle_data:
                return {
                    "error": f"Could not fetch Candle data for {pair} on {interval} timeframe."
                }
            
            chart_images = []
            for indicator in indicators:
                chart_image = self.generate_chart(indicator, pair, interval, candle_data)
                chart_images.append(chart_image)

            # Store all context in metadata
            metadata = {
                "prompt": prompt,
                "token": token,
                "pair": pair,
                "interval": interval,
                "timestamp": datetime.now().isoformat(),
                "data_quality": "partial" if len(indicators) < 20 else "full",
                "technical_indicators": indicators,
            }

            return {"response": chart_images, "metadata": metadata}

        except Exception as e:
            return {"error": str(e)}

    def parse_prompt_with_llm(self, prompt: str) -> tuple[Optional[str], str, List[str]]:
        """Extract token, timeframe and indicators from prompt using GPT."""
        try:
            context = f"""Extract the cryptocurrency token name, timeframe and technical indicators from the following analysis request.
Valid timeframes are: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d, 1w
Valid indicators are: {", ".join(valid_indicators)}

Example inputs and outputs:
Input: "give me a technical analysis for Bitcoin using RSI and MACD"
Output: {{"token": "BTC", "timeframe": "1d", "indicators": ["rsi", "macd"]}}

Input: "analyze ETH on 4 hour timeframe with bollinger bands, RSI, MACD and stochastic"
Output: {{"token": "ETH", "timeframe": "4h", "indicators": ["bbands", "rsi", "macd"]}}

Input: "what's your view on NEAR for the next hour using moving averages"
Output: {{"token": "NEAR", "timeframe": "1h", "indicators": ["ema"]}}

Input: "daily analysis of Cardano with all indicators"
Output: {{"token": "ADA", "timeframe": "1d", "indicators": ["rsi", "macd", "bbands", "mfi", "supertrend", "psar", "vwap", "roc", "mom", "wma"]}}

Now extract from this request: "{prompt}"

IMPORTANT: Respond with ONLY the raw JSON object. Do not include markdown formatting, code blocks, or any other text. The response should start with {{ and end with }}."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a trading expert that extracts token names, timeframes and technical indicators from analysis requests. Always respond with a valid JSON object.",
                    },
                    {"role": "user", "content": context},
                ],
                temperature=0,
            )

            response_text = response.choices[0].message.content.strip()

            try:
                data = json.loads(response_text)
                return data.get("token"), data.get("timeframe", "1d"), data.get("indicators", [])
            except json.JSONDecodeError:
                return None, "1d", []

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
        """Find the best trading pair for a given token. Only accepts exact matches."""
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


    def fetch_candle_data(
        self, symbol: str, interval: str = "1d", exchange: str = "gateio", candles: int = 200
    ) -> Optional[Dict[str, Any]]:
        """Fetch symbol pair candle price data using TAapi."""
        try:
            payload = {
                "secret": self.taapi_api_key,
                "exchange": exchange,
                "symbol": symbol,
                "interval": interval,
                "results": candles,
            }

            url = f"{self.taapi_base_url}/candle"
            response = requests.get(url = url, params = payload)

            if not response.ok:
                print(f"Error Response Status: {response.status_code}")
                print(f"Error Response Content: {response.text}")
                return None

            return response.json()

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error fetching price data: {str(e)}"
            )


    def generate_chart(self, indicator: str, pair: str, interval: str, price_data: Dict[str, List[Any]]) -> str:
        """Generate a price chart with technical indicators using pandas and pandas_ta.
        
        Args:
            indicators: List of technical indicators to include
            pair: Trading pair symbol
            interval: Time interval for the chart
            price_data: List of dictionaries containing candle data
            
        Returns:
            Base64 encoded chart image with data URI prefix for direct PNG viewing
        """
        print(f"Generating chart for {pair} with indicator: {indicator}")

        # Extract data from dictionaries into lists
        datetimes = price_data['timestamp'][:]
        opens = price_data['open'][:]
        highs = price_data['high'][:]
        lows = price_data['low'][:]
        closes = price_data['close'][:]
        volumes = [int(float(v)) for v in price_data['volume']]
        
        # Create DataFrame
        df = pd.DataFrame({
            'datetime': datetimes,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        # Convert datetime strings to datetime objects
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # Plot price
    
        print(f"Calculating {indicator}...")
        if indicator == 'rsi':
            # Calculate RSI
            df['RSI'] = ta.rsi(df.close)
            fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 8))

            # Price subplot
            axs[0].plot(df.close)
            axs[0].set_title(f'{pair} Price Chart ({interval}) - RSI')
            axs[0].set_ylabel('Price')
            axs[0].grid()

            # RSI subplot
            axs[1].axhline(y=70, color='r', linestyle='--')
            axs[1].axhline(y=30, color='g', linestyle='--')
            axs[1].plot(df['RSI'], color='orange')
            axs[1].grid(True)
            axs[1].set_ylabel('RSI')

        elif indicator == 'mfi':
            # Calculate MFI
            df['MFI'] = ta.mfi(high=df.high, low=df.low, close=df.close, volume=df.volume)
            fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 8))

            # Price subplot
            axs[0].plot(df.close)
            axs[0].set_title(f'{pair} Price Chart ({interval}) - MFI')
            axs[0].set_ylabel('Price')
            axs[0].grid()

            # MFI subplot
            axs[1].axhline(y=80, color='r', linestyle='--')
            axs[1].axhline(y=20, color='g', linestyle='--')
            axs[1].plot(df['MFI'], color='orange')
            axs[1].grid(True)
            axs[1].set_ylabel('MFI')

        elif indicator == 'bbands':
            # Calculate Bollinger Bands
            bbands = ta.bbands(df.close, length=20, std=2)
            df['BB_UPPER'] = bbands['BBU_20_2.0']
            df['BB_MIDDLE'] = bbands['BBM_20_2.0'] 
            df['BB_LOWER'] = bbands['BBL_20_2.0']
            
            # Plot price and Bollinger Bands
            plt.plot(df.close, label='Price', color='blue')
            plt.plot(df['BB_UPPER'], label='Upper Band', color='red', linestyle='--')
            plt.plot(df['BB_MIDDLE'], label='Middle Band', color='orange', linestyle='-')
            plt.plot(df['BB_LOWER'], label='Lower Band', color='green', linestyle='--')
            
            # Set title and labels
            plt.title(f'{pair} Price Chart ({interval}) - Bollinger Bands')
            plt.ylabel('Price')
            plt.grid(True)

        elif indicator == 'ema':
            # Plot price and EMA
            plt.plot(df.close)
            df['EMA'] = ta.ema(df.close)
            plt.plot(df['EMA'], color='orange')

            # Set title and labels
            plt.title(f'{pair} Price Chart ({interval}) - EMA')
            plt.ylabel('Price')
            plt.grid(True)

        elif indicator == 'supertrend':
            # Calculate Supertrend
            supertrend = ta.supertrend(high=df.high, low=df.low, close=df.close, length=7, multiplier=3.0)
            df['SUPERTREND'] = supertrend['SUPERT_7_3.0']
            df['SUPERTREND_DIRECTION'] = supertrend['SUPERTd_7_3.0']
            
            # Remove garbage datapoints where supertrend is 0
            df['SUPERTREND'] = df['SUPERTREND'].replace(0, float('nan'))
            
            # Plot price
            plt.plot(df.close, color='blue')
            
            # Plot Supertrend with different colors based on direction
            for i in range(len(df)-1):
                if df['SUPERTREND_DIRECTION'].iloc[i] == 1:  # Uptrend
                    plt.plot(df.index[i:i+2], df['SUPERTREND'].iloc[i:i+2], 
                            color='green', label='Supertrend' if i == 0 else "")
                else:  # Downtrend
                    plt.plot(df.index[i:i+2], df['SUPERTREND'].iloc[i:i+2], 
                            color='red', label='Supertrend' if i == 0 else "")

            # Set title and labels
            plt.title(f'{pair} Price Chart ({interval}) - Supertrend')
            plt.ylabel('Price')
            plt.grid(True)

        elif indicator == 'psar':
            # Calculate Parabolic SAR
            psar = ta.psar(high=df.high, low=df.low, close=df.close)
            df['PSARl_0.02_0.2'] = psar['PSARl_0.02_0.2']  # Long position SAR
            df['PSARs_0.02_0.2'] = psar['PSARs_0.02_0.2']  # Short position SAR
            
            # Plot price
            plt.plot(df.close, label='Price', color='blue')
            
            # Plot PSAR dots, filtering out 0 values
            bullish_psar = df['PSARl_0.02_0.2'].replace(0, float('nan'))
            bearish_psar = df['PSARs_0.02_0.2'].replace(0, float('nan'))
            
            plt.scatter(df.index, bullish_psar,
                       color='green', label='PSAR Bullish', marker='^', s=20)
            plt.scatter(df.index, bearish_psar,
                       color='red', label='PSAR Bearish', marker='v', s=20)

            # Set title and labels
            plt.title(f'{pair} Price Chart ({interval}) - Parabolic SAR')
            plt.ylabel('Price')
            plt.grid(True)

        elif indicator == 'vwap':
            # Calculate VWAP
            df['VWAP'] = ta.vwap(high=df.high, low=df.low, close=df.close, volume=df.volume)
            
            # Plot price and VWAP
            plt.plot(df.close, label='Price', color='blue')
            plt.plot(df['VWAP'], color='purple', label='VWAP')

            # Set title and labels
            plt.title(f'{pair} Price Chart ({interval}) - VWAP')
            plt.ylabel('Price')
            plt.grid(True)

        elif indicator == 'roc':
            # Calculate Rate of Change
            df['ROC'] = ta.roc(df.close)
            fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 8))

            # Price subplot
            axs[0].plot(df.close)
            axs[0].set_title(f'{pair} Price Chart ({interval}) - ROC')
            axs[0].set_ylabel('Price')
            axs[0].grid(True)

            # ROC subplot
            axs[1].axhline(y=0, color='k', linestyle='--')
            axs[1].plot(df['ROC'], color='purple')
            axs[1].grid(True)
            axs[1].set_ylabel('Rate of Change (%)')

        elif indicator == 'mom':
            # Calculate Momentum
            df['MOM'] = ta.mom(df.close)
            fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 8))

            # Price subplot
            axs[0].plot(df.close)
            axs[0].set_title(f'{pair} Price Chart ({interval}) - Momentum')
            axs[0].set_ylabel('Price')
            axs[0].grid(True)

            # Momentum subplot
            axs[1].axhline(y=0, color='k', linestyle='--')
            axs[1].plot(df['MOM'], color='purple')
            axs[1].grid(True)
            axs[1].set_ylabel('Momentum')

        elif indicator == 'wma':
            # Plot price and WMA
            plt.plot(df.close, label='Price')
            df['WMA'] = ta.wma(df.close, length=20)
            plt.plot(df['WMA'], color='orange', label='WMA')

            # Set title and labels
            plt.title(f'{pair} Price Chart ({interval}) - WMA')
            plt.ylabel('Price')
            plt.grid(True)
            
        # Save chart to bytes
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        
        # Encode as base64 string with data URI prefix for PNG
        chart_b64 = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"
        plt.close()
        
        print("Chart generation completed successfully")
        return chart_b64

# added the following to have uniformity in the way we call tools
def run(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    return ChartGenerator().run(prompt, system_prompt)
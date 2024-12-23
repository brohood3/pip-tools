"""
Technical Analysis Script using TAapi and OpenAI GPT.
Fetches technical indicators and generates AI-powered analysis for cryptocurrency pairs.
"""

import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, TypedDict, Union
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import time
import openai

# --- Environment Setup ---
load_dotenv()

TAAPI_KEY = os.getenv('TAAPI_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not TAAPI_KEY:
    raise ValueError("TAAPI_KEY environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

openai.api_key = OPENAI_API_KEY
TAAPI_BASE_URL = "https://api.taapi.io"

# --- Data Models ---
class TechnicalIndicators(TypedDict):
    """TypedDict defining the structure of technical indicators."""
    # Trend Indicators
    ema: Dict[str, float]  # Multiple EMAs with different periods (20, 50, 200)
    supertrend: Dict[str, Union[float, str]]  # value and valueAdvice ("long"/"short")
    adx: Dict[str, float]  # Single value
    dmi: Dict[str, float]  # adx, pdi, mdi values
    psar: Dict[str, float]  # Single value
    
    # Momentum Indicators
    rsi: Dict[str, float]  # Single value
    macd: Dict[str, float]  # valueMACD, valueMACDSignal, valueMACDHist
    stoch: Dict[str, float]  # valueK, valueD
    mfi: Dict[str, float]  # Single value
    cci: Dict[str, float]  # Single value
    
    # Pattern Recognition
    doji: Dict[str, float]  # value (0, 100, or -100)
    engulfing: Dict[str, float]  # value (0, 100, or -100)
    hammer: Dict[str, float]  # value (0, 100, or -100)
    shootingstar: Dict[str, float]  # value (0, 100, or -100)
    
    # Support/Resistance & Volatility
    fibonacciretracement: Dict[str, Union[float, str, int]]  # value, trend, startPrice, endPrice, timestamps
    bbands: Dict[str, float]  # valueUpperBand, valueMiddleBand, valueLowerBand
    atr: Dict[str, float]  # Single value
    volume: Dict[str, float]  # Single value
    vwap: Dict[str, float]  # Single value
    
    # Volume Indicators
    ad: Dict[str, float]  # Chaikin A/D Line
    adosc: Dict[str, float]  # Chaikin A/D Oscillator
    cmf: Dict[str, float]  # Chaikin Money Flow
    obv: Dict[str, float]  # On Balance Volume
    vosc: Dict[str, float]  # Volume Oscillator

# --- API Interaction Functions ---
def get_available_symbols() -> List[str]:
    """Fetch available trading pairs from TAapi for Gate.io."""
    try:
        url = f"{TAAPI_BASE_URL}/exchange-symbols"
        response = requests.get(url, params={
            "secret": TAAPI_KEY,
            "exchange": "gateio"
        })
        
        if not response.ok:
            print(f"Warning: Failed to fetch Gate.io symbols ({response.status_code}). Using BTC analysis.")
            return []
            
        symbols = response.json()
        if not symbols or not isinstance(symbols, list):
            print("Warning: Invalid response format from symbols endpoint. Using BTC analysis.")
            return []
            
        print(f"\nFetched {len(symbols)} trading pairs from Gate.io")
        return symbols
        
    except Exception as e:
        print(f"Warning: Could not fetch Gate.io symbols ({str(e)}). Using BTC analysis.")
        return []

def fetch_indicators(symbol: str, exchange: str = "gateio", interval: str = "1d") -> Optional[TechnicalIndicators]:
    """
    Fetch technical indicators using TAapi's bulk endpoint.
    Indicators are split into batches to comply with API limits.
    """
    try:
        url = f"{TAAPI_BASE_URL}/bulk"
        
        print(f"\nFetching indicators for {symbol} on {interval} timeframe from {exchange}...")
        
        # Split indicators into batches of 20 calculations
        batch1 = [
            # Trend Indicators (7)
            {"indicator": "ema", "period": 20},
            {"indicator": "ema", "period": 50},
            {"indicator": "ema", "period": 200},
            {"indicator": "supertrend"},
            {"indicator": "adx"},
            {"indicator": "dmi"},
            {"indicator": "psar"},
            
            # Momentum Indicators (5)
            {"indicator": "rsi"},
            {"indicator": "macd"},
            {"indicator": "stoch"},
            {"indicator": "mfi"},
            {"indicator": "cci"},
            
            # Core Pattern Recognition (4)
            {"indicator": "doji"},
            {"indicator": "engulfing"},
            {"indicator": "hammer"},
            {"indicator": "shootingstar"},
            
            # Volume & Volatility (4)
            {"indicator": "bbands"},
            {"indicator": "atr"},
            {"indicator": "volume"},
            {"indicator": "vwap"}
        ]
        
        batch2 = [
            # Pattern Recognition (9)
            {"indicator": "doji"},
            {"indicator": "stalledpattern"},
            {"indicator": "engulfing"},
            {"indicator": "hammer"},
            {"indicator": "morningstar"},
            {"indicator": "eveningstar"},
            {"indicator": "shootingstar"},
            {"indicator": "dragonflydoji"},
            {"indicator": "gravestonedoji"},
            
            # Support/Resistance & Additional Indicators (11)
            {"indicator": "fibonacciretracement"},
            {"indicator": "roc"},
            {"indicator": "willr"},
            {"indicator": "mom"},
            {"indicator": "trix"},
            {"indicator": "stochrsi"},
            {"indicator": "wma"},
            {"indicator": "tema"},
            {"indicator": "ad"},  # Chaikin A/D Line
            {"indicator": "adosc"},  # Chaikin A/D Oscillator
            {"indicator": "cmf"}  # Chaikin Money Flow
        ]
        
        batch3 = [
            # Additional Volume Indicators
            {"indicator": "obv"},  # On Balance Volume
            {"indicator": "vosc"},  # Volume Oscillator
            {"indicator": "volume"}  # Current volume for comparison
        ]
        
        # Initialize the result dictionary
        result = {}
        
        # Function to process a batch
        def process_batch(indicators):
            payload = {
                "secret": TAAPI_KEY,
                "construct": {
                    "exchange": exchange,
                    "symbol": symbol,
                    "interval": interval,
                    "indicators": indicators
                }
            }
            
            print("\nSending request with payload:")
            print(json.dumps(payload, indent=2))
            
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            
            if not response.ok:
                print(f"\nError Response Status: {response.status_code}")
                print("Error Response Headers:")
                print(json.dumps(dict(response.headers), indent=2))
                print("\nError Response Content:")
                print(response.text)
                return None
            
            response_data = response.json()
            print("\nReceived Response:")
            print(json.dumps(response_data, indent=2))
            
            return response_data.get("data", [])
        
        # Process first batch
        print("\nProcessing first batch of indicators...")
        batch1_data = process_batch(batch1)
        if not batch1_data:
            return None
        
        # Process second batch
        print("\nProcessing second batch of indicators...")
        batch2_data = process_batch(batch2)
        if not batch2_data:
            return None
        
        # Process third batch
        print("\nProcessing third batch of indicators...")
        batch3_data = process_batch(batch3)
        if not batch3_data:
            return None
        
        # Combine results from all batches
        all_data = batch1_data + batch2_data + batch3_data
        
        # Map the responses to our TechnicalIndicators structure
        for indicator_data in all_data:
            indicator_id = indicator_data["id"]
            indicator_result = indicator_data["result"]
            
            # Extract indicator name from the ID
            # Format: exchange_symbol_interval_indicator_params
            id_parts = indicator_id.split("_")
            indicator_name = id_parts[3]
            
            if "ema" in indicator_id:
                period = id_parts[4]
                if "ema" not in result:
                    result["ema"] = {}
                result["ema"][f"period_{period}"] = indicator_result["value"]
            else:
                result[indicator_name] = indicator_result
        
        return result
        
    except Exception as e:
        print(f"\nError fetching indicators:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        if hasattr(e, 'response'):
            print("\nResponse Details:")
            print(f"Status Code: {e.response.status_code}")
            print(f"Response Text: {e.response.text}")
        return None

# --- Data Processing Functions ---
def format_indicators_json(indicators: TechnicalIndicators) -> dict:
    """Format the indicators into a structured JSON by category."""
    return {
        "trend_indicators": {
            "ema": indicators.get('ema', {}),
            "supertrend": indicators.get('supertrend', {}),
            "adx": indicators.get('adx', {}),
            "dmi": indicators.get('dmi', {}),
            "psar": indicators.get('psar', {})
        },
        "momentum_indicators": {
            "rsi": indicators.get('rsi', {}),
            "macd": indicators.get('macd', {}),
            "stochastic": indicators.get('stoch', {}),
            "mfi": indicators.get('mfi', {}),
            "cci": indicators.get('cci', {})
        },
        "pattern_signals": {
            "doji": indicators.get('doji', {}),
            "engulfing": indicators.get('engulfing', {}),
            "hammer": indicators.get('hammer', {}),
            "shooting_star": indicators.get('shootingstar', {})
        },
        "price_and_volume": {
            "fibonacci": indicators.get('fibonacciretracement', {}),
            "bollinger_bands": indicators.get('bbands', {}),
            "atr": indicators.get('atr', {}),
            "volume": indicators.get('volume', {}),
            "vwap": indicators.get('vwap', {})
        },
        "volume_analysis": {
            "chaikin_money_flow": indicators.get('cmf', {}),
            "accumulation_distribution": indicators.get('ad', {}),
            "chaikin_oscillator": indicators.get('adosc', {}),
            "on_balance_volume": indicators.get('obv', {}),
            "volume_oscillator": indicators.get('vosc', {})
        }
    }

def find_best_pair(token: str, available_symbols: List[str]) -> Optional[str]:
    """
    Find the best trading pair for a given token.
    Prioritizes USDT pairs, then ETH, then BTC pairs.
    """
    token = token.upper()
    quote_priorities = ["USDT", "ETH", "BTC"]
    
    for quote in quote_priorities:
        pair = f"{token}/{quote}"
        if pair in available_symbols:
            return pair
    return None

# --- Natural Language Processing Functions ---
def parse_analysis_request(prompt: str) -> tuple[str, str]:
    """Parse a user's analysis request to extract token and timeframe."""
    # Convert to lowercase for easier matching
    prompt = prompt.lower()
    
    # Default values
    interval = "1d"  # Default to daily timeframe
    
    # Extract timeframe if specified
    timeframes = {
        "minute": "1m", "minutes": "1m", "1m": "1m",
        "hourly": "1h", "hour": "1h", "1h": "1h", "1 hour": "1h",
        "4h": "4h", "4 hour": "4h", "4 hours": "4h",
        "daily": "1d", "day": "1d", "1d": "1d",
        "weekly": "1w", "week": "1w", "1w": "1w"
    }
    
    # Words to ignore when looking for the token
    ignore_words = {
        "analysis", "ta", "technical", "for", "on", "me", "give", "make", "of",
        "timeframe", "chart", "charts", "price", "market", "markets", "trading",
        "trend", "trends", "view", "outlook", "analysis", "analyze", "with", "focus",
        "volume", "momentum", "trend", "patterns"
    }
    ignore_words.update(timeframes.keys())
    
    # First, find the timeframe
    for timeframe, value in timeframes.items():
        if timeframe in prompt:
            interval = value
            break
    
    # Then find the token - look for the last word that's not in ignore_words
    words = prompt.split()
    token = None
    for word in reversed(words):
        if word not in ignore_words:
            token = word.upper()
            break
    
    return token, interval

def parse_prompt_with_llm(prompt: str) -> tuple[str, str]:
    """Use GPT-4o-mini to extract token and timeframe from natural language prompt."""
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

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using the new mini model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts cryptocurrency analysis parameters from natural language requests. You respond with raw JSON only."},
                {"role": "user", "content": context}
            ],
            temperature=0.1,  # Low temperature for consistent outputs
            max_tokens=100
        )
        
        # Print raw LLM output
        raw_output = response.choices[0].message.content
        print(f"\nRaw LLM Output: {raw_output}")
        
        # Clean up the response - remove any markdown code block syntax
        cleaned_output = raw_output.strip()
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output.split("\n", 1)[1]  # Remove first line
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output.rsplit("\n", 1)[0]  # Remove last line
        cleaned_output = cleaned_output.strip()
        if cleaned_output.startswith("json"):
            cleaned_output = cleaned_output[4:].strip()  # Remove "json" language specifier
            
        print(f"\nCleaned Output: {cleaned_output}")
        
        # Parse the response
        result = json.loads(cleaned_output)
        return result["token"], result["timeframe"]
        
    except Exception as e:
        print(f"Error parsing prompt with LLM: {str(e)}")
        # Fall back to basic parsing if LLM fails
        return parse_analysis_request(prompt)

# --- Analysis Generation Functions ---
def generate_analysis(indicators: TechnicalIndicators, symbol: str, interval: str) -> str:
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
        "1w": "long-term (1-3 months)"
    }
    
    time_horizon = interval_horizons.get(interval, "medium-term")
    
    # Format the context for GPT-4
    context = f"""You are an experienced crypto technical analyst known for providing clear, opinionated market analysis.
Given the following technical indicators for {symbol} on the {interval} timeframe, provide your expert interpretation.

ANALYSIS PARAMETERS:
• Timeframe: {interval} candles
• Trading Horizon: {time_horizon}
• Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC

CURRENT MARKET DATA:

Trend Indicators:
• EMAs: 20 [{indicators['ema']['period_20']:.2f}], 50 [{indicators['ema']['period_50']:.2f}], 200 [{indicators['ema']['period_200']:.2f}]
• Supertrend: {indicators['supertrend']['value']:.2f} (Signal: {indicators['supertrend']['valueAdvice']})
• ADX: {indicators['adx']['value']:.2f} | DMI: +DI {indicators['dmi']['pdi']:.2f}, -DI {indicators['dmi']['mdi']:.2f}
• PSAR: {indicators['psar']['value']:.2f}

Momentum & Oscillators:
• RSI: {indicators['rsi']['value']:.2f}
• MACD: Line [{indicators['macd']['valueMACD']:.2f}], Signal [{indicators['macd']['valueMACDSignal']:.2f}], Hist [{indicators['macd']['valueMACDHist']:.2f}]
• Stochastic: K[{indicators['stoch']['valueK']:.2f}], D[{indicators['stoch']['valueD']:.2f}]
• MFI: {indicators['mfi']['value']:.2f}
• CCI: {indicators['cci']['value']:.2f}

Pattern Signals:
• Doji: {indicators['doji']['value']} | Engulfing: {indicators['engulfing']['value']}
• Hammer: {indicators['hammer']['value']} | Shooting Star: {indicators['shootingstar']['value']}

Price Structure & Volume:
• Fibonacci: {indicators['fibonacciretracement']['value']:.2f} ({indicators['fibonacciretracement']['trend']})
  Range: {indicators['fibonacciretracement']['startPrice']} → {indicators['fibonacciretracement']['endPrice']}
• Bollinger Bands: Upper[{indicators['bbands']['valueUpperBand']:.2f}], Mid[{indicators['bbands']['valueMiddleBand']:.2f}], Lower[{indicators['bbands']['valueLowerBand']:.2f}]
• ATR: {indicators['atr']['value']:.2f}

Volume Analysis:
• Current Volume: {indicators['volume']['value']:.2f}
• Chaikin Money Flow: {indicators['cmf']['value']:.2f}
• A/D Line: {indicators['ad']['value']:.2f}
• A/D Oscillator: {indicators['adosc']['value']:.2f}
• On Balance Volume: {indicators['obv']['value']:.2f}
• Volume Oscillator: {indicators['vosc']['value']:.2f}
• VWAP: {indicators['vwap']['value']:.2f}

Volume Interpretation:
• Money Flow: {'Positive' if indicators['cmf']['value'] > 0 else 'Negative'} (CMF)
• Volume Trend: {'Increasing' if indicators['vosc']['value'] > 0 else 'Decreasing'} (VOSC)
• Price/Volume Alignment: {'Confirming' if (indicators['obv']['value'] > 0) == (indicators['adosc']['value'] > 0) else 'Diverging'}

Based on these indicators, provide a concise but thorough analysis for the {time_horizon} horizon that:
1. States your CLEAR DIRECTIONAL BIAS (bullish/bearish/neutral) with confidence level
2. Identifies the MOST SIGNIFICANT signals that form your bias
3. Points out any CONFLICTING signals that need attention
4. Highlights KEY PRICE LEVELS for {time_horizon} trading:
   - Entry opportunities (with specific trigger conditions)
   - Stop loss placement (with rationale)
   - Take profit targets (with timeframes)
5. Notes any SPECIFIC SETUPS or patterns forming
6. Provides ACTIONABLE INSIGHTS rather than just describing the indicators

Focus on how these indicators INTERACT with each other to form a complete picture. If you see conflicting signals, explain which ones you're giving more weight to and why.

Remember: 
- Be opinionated and clear in your analysis
- Point out risks to your thesis
- All price targets and analysis should align with the {time_horizon} trading horizon
- Specify whether setups are for swing trading or position trading given the timeframe"""

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are a seasoned technical analyst specializing in {time_horizon} {symbol} analysis. You focus on meaningful interpretation of indicators rather than just describing them."},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating analysis: {str(e)}")
        return "Error generating analysis. Please try again."

# --- Main Execution Functions ---
def run_analysis(prompt: str) -> str:
    """Run technical analysis and return JSON output."""
    available_symbols = get_available_symbols()
    if not available_symbols:
        return run_default_analysis()
    
    try:
        token, interval = parse_prompt_with_llm(prompt)
        print(f"\nExtracted Parameters:")
        print(f"Token: {token}")
        print(f"Timeframe: {interval}")
    except Exception as e:
        print(f"\nFalling back to basic prompt parsing due to error: {str(e)}")
        token, interval = parse_analysis_request(prompt)
    
    if not token:
        return run_default_analysis()
    
    pair = find_best_pair(token, available_symbols)
    
    if not pair:
        return run_default_analysis(requested_token=token)
    
    indicators = fetch_indicators(pair, interval=interval)
    if not indicators:
        return run_default_analysis(requested_token=token)
    
    analysis = generate_analysis(indicators, pair, interval)
    
    # Create the final JSON output
    output = {
        "metadata": {
            "token": token,
            "pair": pair,
            "interval": interval,
            "timestamp": datetime.now().isoformat(),
        },
        "technical_indicators": format_indicators_json(indicators),
        "ai_analysis": analysis
    }
    
    return json.dumps(output, indent=2)

def run_default_analysis(requested_token: str = None) -> str:
    """Run default analysis using BTC/USDT and return JSON output."""
    indicators = fetch_indicators("BTC/USDT", interval="1d")
    if not indicators:
        return json.dumps({"error": "Could not fetch market data. Please try again later."})
    
    analysis = generate_analysis(indicators, "BTC/USDT", "1d")
    
    output = {
        "metadata": {
            "note": f"No trading pair found for {requested_token}. Using BTC as market proxy." if requested_token else "Using default BTC analysis",
            "token": "BTC",
            "pair": "BTC/USDT",
            "interval": "1d",
            "timestamp": datetime.now().isoformat(),
        },
        "technical_indicators": format_indicators_json(indicators),
        "ai_analysis": analysis
    }
    
    return json.dumps(output, indent=2)

# --- Main Entry Point ---
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("Enter your analysis request (e.g., 'ta analysis for NEAR'): ")
    
    print(f"\n{'='*50}")
    print("Technical Analysis Request")
    print(f"Prompt: {prompt}")
    print(f"{'='*50}")
    
    analysis_json = run_analysis(prompt)
    print(analysis_json)
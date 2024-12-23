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

# Load environment variables
load_dotenv()

# Constants
TAAPI_KEY = os.getenv('TAAPI_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not TAAPI_KEY:
    raise ValueError("TAAPI_KEY environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

openai.api_key = OPENAI_API_KEY
TAAPI_BASE_URL = "https://api.taapi.io"

class TechnicalIndicators(TypedDict):
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

def fetch_indicators(symbol: str, exchange: str = "binance", interval: str = "1d") -> Optional[TechnicalIndicators]:
    """
    Fetch technical indicators using TAapi's bulk endpoint.
    
    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        exchange: Exchange name (default: "binance")
        interval: Time interval (1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d, 1w)
                 Defaults to 1d (daily) for standard technical analysis
    """
    try:
        url = f"{TAAPI_BASE_URL}/bulk"
        
        print(f"\nFetching indicators for {symbol} on {interval} timeframe...")
        
        # Single batch of 20 indicators (maximum allowed)
        indicators = [
            # Trend Indicators (7)
            {"indicator": "ema", "period": 20},  # Base trend
            {"indicator": "ema", "period": 50},  # Medium trend
            {"indicator": "ema", "period": 200},  # Long-term trend
            {"indicator": "supertrend"},  # Trend direction with stop loss
            {"indicator": "adx"},  # Trend strength
            {"indicator": "dmi"},  # Trend direction
            {"indicator": "psar"},  # Trend reversal points
            
            # Momentum Indicators (5)
            {"indicator": "rsi"},  # Base momentum
            {"indicator": "macd"},  # Trend momentum and crossovers
            {"indicator": "stoch"},  # Overbought/Oversold with crossovers
            {"indicator": "mfi"},  # Volume-weighted RSI
            {"indicator": "cci"},  # Trend deviations
            
            # Pattern Recognition (4)
            {"indicator": "doji"},  # Indecision/reversal
            {"indicator": "engulfing"},  # Strong reversal
            {"indicator": "hammer"},  # Bottom reversal
            {"indicator": "shootingstar"},  # Top reversal
            
            # Support/Resistance & Volatility (4)
            {"indicator": "fibonacciretracement"},  # Key S/R levels
            {"indicator": "bbands"},  # Volatility channels
            {"indicator": "atr"},  # Volatility measure
            {"indicator": "volume"}  # Volume confirmation
        ]
        
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
            
            # If we hit rate limit (429), wait for the reset time
            if response.status_code == 429:
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                if reset_time:
                    wait_time = max(reset_time - int(time.time()), 15)
                    print(f"\nRate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    # Retry the request
                    return fetch_indicators(symbol, exchange, interval)
            return None
        
        # Process the response
        response_data = response.json()
        print("\nReceived Response:")
        print(json.dumps(response_data, indent=2))
        
        data = response_data.get("data", [])
        
        # Initialize the result dictionary
        result = {}
        
        # Map the responses to our TechnicalIndicators structure
        for indicator_data in data:
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

def generate_analysis(indicators: TechnicalIndicators, symbol: str, interval: str) -> str:
    """
    Generate an opinionated technical analysis report using GPT-4 based on the indicators.
    Focuses on meaningful insights and actionable interpretation of the indicators.
    """
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

Price Structure:
• Fibonacci: {indicators['fibonacciretracement']['value']:.2f} ({indicators['fibonacciretracement']['trend']})
  Range: {indicators['fibonacciretracement']['startPrice']} → {indicators['fibonacciretracement']['endPrice']}
• Bollinger Bands: Upper[{indicators['bbands']['valueUpperBand']:.2f}], Mid[{indicators['bbands']['valueMiddleBand']:.2f}], Lower[{indicators['bbands']['valueLowerBand']:.2f}]
• ATR: {indicators['atr']['value']:.2f}
• Volume: {indicators['volume']['value']:.2f}

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
            model="gpt-4o",  # Using GPT-4 for more sophisticated analysis
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

if __name__ == "__main__":
    import sys
    from datetime import datetime
    
    # Available intervals (ordered from smallest to largest)
    VALID_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "12h", "1d", "1w"]
    DEFAULT_INTERVAL = "1d"  # Default to daily timeframe
    
    # Use command line arguments if provided
    token_id = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
    interval = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in VALID_INTERVALS else DEFAULT_INTERVAL
    
    print(f"\n{'='*50}")
    print(f"Technical Analysis - {token_id}")
    print(f"Timeframe: {interval}")
    print(f"{'='*50}")
    
    # Fetch indicators
    indicators = fetch_indicators(token_id, interval=interval)
    if indicators:
        # Get current time for reference
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\nTechnical Analysis Results:")
        print(f"Analysis Time: {current_time}")
        print(f"Symbol: {token_id}")
        print(f"Timeframe: {interval.upper()}")
        print("-" * 50)
        print(json.dumps(indicators, indent=2))
        
        # Generate and display AI analysis
        print("\nGenerating AI Analysis...")
        print("=" * 50)
        analysis = generate_analysis(indicators, token_id, interval)
        print(analysis)
"""
Technical Analysis Tool

Technical Analysis Script using TAapi and OpenAI GPT.
Fetches technical indicators and generates AI-powered analysis for cryptocurrency pairs.
"""

# --- Imports ---
import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, TypedDict, Union, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import time
from openai import OpenAI
from fastapi import HTTPException

# --- Type Definitions ---
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

class TechnicalAnalysis:
    def __init__(self):
        """Initialize the Technical Analysis tool with required API clients"""
        load_dotenv()
        
        # API Keys
        self.taapi_api_key = os.getenv('TAAPI_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.taapi_api_key:
            raise HTTPException(status_code=500, detail="Missing TAAPI_API_KEY environment variable")
        if not self.openai_api_key:
            raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY environment variable")
            
        # Initialize API clients
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.taapi_base_url = "https://api.taapi.io"

    def run(self, prompt: str) -> Dict[str, Any]:
        """Main entry point for the tool"""
        # Extract trading pair and timeframe
        symbol, timeframe = self._parse_trading_pair(prompt)
        if not symbol:
            raise HTTPException(
                status_code=400,
                detail="Could not determine the trading pair from your request. Please provide a clearer trading pair (e.g., BTC/USDT)."
            )
        
        # Get technical indicators
        indicators = self._fetch_indicators(symbol, timeframe)
        if not indicators:
            raise HTTPException(
                status_code=404,
                detail=f"Could not fetch technical indicators for {symbol}"
            )
        
        # Get analysis
        analysis = self._get_technical_analysis(indicators, prompt, symbol, timeframe)
        if not analysis:
            raise HTTPException(
                status_code=500,
                detail="Error generating analysis"
            )
        
        # Return structured response
        return {
            "trading_pair": {
                "symbol": symbol,
                "timeframe": timeframe
            },
            "indicators": indicators,
            "analysis": analysis
        }

    def _parse_trading_pair(self, prompt: str) -> tuple[Optional[str], str]:
        """Extract trading pair and timeframe from prompt using GPT."""
        try:
            analysis_prompt = f"""Given this analysis request: "{prompt}"

Extract the trading pair and timeframe.
Rules for trading pairs:
- Must be in format BASE/QUOTE (e.g., BTC/USDT)
- Common quote currencies: USDT, USD, BTC, ETH
- Always uppercase
- If timeframe not specified, use "1d" (daily)
- Valid timeframes: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d, 1w
- If unsure about either, respond with "Cannot determine"

Examples:
Input: "Analyze BTC technical indicators"
Output:
Trading Pair: BTC/USDT
Timeframe: 1d

Input: "Check ETH/BTC 4h indicators"
Output:
Trading Pair: ETH/BTC
Timeframe: 4h

Format your response exactly like the examples above."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "system",
                    "content": "You are a trading expert. Extract trading pairs and timeframes from analysis requests. Be precise and conservative - if unsure, respond with 'Cannot determine'."
                }, {
                    "role": "user",
                    "content": analysis_prompt
                }]
            )
            
            response_text = response.choices[0].message.content
            
            # Check for explicit uncertainty
            if "Cannot determine" in response_text:
                return None, "1d"
            
            # Parse response
            trading_pair = None
            timeframe = "1d"  # default
            
            for line in response_text.split('\n'):
                if line.startswith('Trading Pair:'):
                    trading_pair = line.replace('Trading Pair:', '').strip()
                elif line.startswith('Timeframe:'):
                    timeframe = line.replace('Timeframe:', '').strip()
            
            if not trading_pair:
                return None, timeframe
                
            # Verify trading pair format
            if '/' not in trading_pair:
                trading_pair = f"{trading_pair}/USDT"
            
            return trading_pair, timeframe
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error parsing trading pair: {str(e)}")

    def _fetch_indicators(self, symbol: str, interval: str = "1d", exchange: str = "gateio") -> Optional[Dict[str, Any]]:
        """Fetch technical indicators using TAapi."""
        try:
            url = f"{self.taapi_base_url}/bulk"
            
            # Define core indicators batch
            indicators = [
                # Trend Indicators
                {"indicator": "ema", "period": 20},
                {"indicator": "ema", "period": 50},
                {"indicator": "ema", "period": 200},
                {"indicator": "supertrend"},
                {"indicator": "adx"},
                {"indicator": "dmi"},
                {"indicator": "psar"},
                
                # Momentum Indicators
                {"indicator": "rsi"},
                {"indicator": "macd"},
                {"indicator": "stoch"},
                {"indicator": "mfi"},
                {"indicator": "cci"},
                
                # Pattern Recognition
                {"indicator": "doji"},
                {"indicator": "engulfing"},
                {"indicator": "hammer"},
                {"indicator": "shootingstar"},
                
                # Volume & Volatility
                {"indicator": "bbands"},
                {"indicator": "atr"},
                {"indicator": "volume"},
                {"indicator": "vwap"}
            ]
            
            payload = {
                "secret": self.taapi_api_key,
                "construct": {
                    "exchange": exchange,
                    "symbol": symbol,
                    "interval": interval,
                    "indicators": indicators
                }
            }
            
            response = requests.post(url, json=payload)
            if not response.ok:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"TAapi error: {response.text}"
                )
            
            data = response.json().get('data', [])
            
            # Process indicators into structured format
            result = {}
            for indicator in data:
                if indicator.get('result') and not indicator.get('result').get('error'):
                    name = indicator['indicator']
                    if name == 'ema':
                        if 'ema' not in result:
                            result['ema'] = {}
                        period = indicator['id'].split('_')[-2]
                        result['ema'][f'period_{period}'] = indicator['result'].get('value')
                    else:
                        result[name] = indicator['result']
            
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching indicators: {str(e)}")

    def _get_technical_analysis(self, indicators: Dict[str, Any], original_prompt: str, symbol: str, timeframe: str) -> Optional[str]:
        """Generate technical analysis using GPT."""
        try:
            # Format indicators for prompt
            formatted_indicators = []
            
            # Trend Indicators
            if 'ema' in indicators:
                ema = indicators['ema']
                formatted_indicators.append(f"EMA:")
                for period, value in ema.items():
                    formatted_indicators.append(f"- {period}: {value:.2f}")
            
            if 'supertrend' in indicators:
                st = indicators['supertrend']
                formatted_indicators.append(f"Supertrend: {st.get('value', 'N/A')} ({st.get('valueAdvice', 'N/A')})")
            
            if 'adx' in indicators:
                formatted_indicators.append(f"ADX: {indicators['adx'].get('value', 'N/A')}")
            
            # Momentum Indicators
            if 'rsi' in indicators:
                formatted_indicators.append(f"RSI: {indicators['rsi'].get('value', 'N/A')}")
            
            if 'macd' in indicators:
                macd = indicators['macd']
                formatted_indicators.append(f"MACD:")
                formatted_indicators.append(f"- MACD Line: {macd.get('valueMACD', 'N/A')}")
                formatted_indicators.append(f"- Signal Line: {macd.get('valueMACDSignal', 'N/A')}")
                formatted_indicators.append(f"- Histogram: {macd.get('valueMACDHist', 'N/A')}")
            
            # Pattern Recognition
            patterns = []
            for pattern in ['doji', 'engulfing', 'hammer', 'shootingstar']:
                if pattern in indicators and indicators[pattern].get('value') != 0:
                    patterns.append(f"- {pattern.title()}: {'Bullish' if indicators[pattern]['value'] > 0 else 'Bearish'}")
            
            if patterns:
                formatted_indicators.append("Patterns Detected:")
                formatted_indicators.extend(patterns)

            indicators_text = "\n".join(formatted_indicators)

            prompt = f"""As a professional technical analyst, provide a comprehensive analysis for {symbol} on the {timeframe} timeframe:

Original Request: "{original_prompt}"

Technical Indicators:
{indicators_text}

Focus on:
1. Trend Analysis:
   - Current trend direction and strength
   - Key moving averages and their implications
   - Support/resistance levels

2. Momentum Analysis:
   - RSI and MACD signals
   - Potential overbought/oversold conditions
   - Momentum divergences

3. Pattern Recognition:
   - Significant chart patterns
   - Candlestick patterns
   - Pattern reliability assessment

4. Trading Implications:
   - Key levels to watch
   - Potential entry/exit points
   - Risk management considerations

Let the indicators guide your analysis. Highlight both bullish and bearish signals."""

            completion = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional technical analyst at a prestigious trading firm. Your analyses are known for being thorough and objective, letting the technical indicators guide your conclusions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating analysis: {str(e)}") 
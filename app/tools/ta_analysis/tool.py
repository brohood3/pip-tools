"""
Technical Analysis Tool

Technical Analysis Script using TAapi and OpenAI GPT.
Fetches technical indicators and generates AI-powered analysis for cryptocurrency pairs.
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
    """Technical Analysis tool using TAapi and GPT-4."""
    
    def __init__(self):
        """Initialize with API clients and configuration."""
        self.taapi_api_key = os.getenv('TAAPI_API_KEY')
        if not self.taapi_api_key:
            raise ValueError("TAAPI_API_KEY environment variable is not set")
            
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
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
                return {"error": "Could not determine which token to analyze. Please specify a token."}
            
            # Get available symbols and find best pair
            available_symbols = self.get_available_symbols()
            if not available_symbols:
                return {"error": "Could not fetch available trading pairs. Please try again later."}
            
            pair = self.find_best_pair(token, available_symbols)
            if not pair:
                return {"error": f"No trading pair found for {token}. Please verify the token symbol and try again."}
            
            # Fetch indicators
            indicators = self._fetch_indicators(pair, interval=interval)
            if not indicators:
                return {"error": f"Insufficient data for {pair} on {interval} timeframe."}
            
            # Generate analysis
            analysis = self.generate_analysis(indicators, pair, interval, prompt)
            
            # Store all context in metadata
            metadata = {
                "prompt": prompt,
                "token": token,
                "pair": pair,
                "interval": interval,
                "timestamp": datetime.now().isoformat(),
                "data_quality": "partial" if len(indicators) < 20 else "full",
                "technical_indicators": self.format_indicators_json(indicators)
            }
            
            return {
                "analysis": analysis,
                "metadata": metadata
            }
            
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
                messages=[{
                    "role": "system",
                    "content": "You are a trading expert that extracts token names and timeframes from analysis requests. Always respond with a valid JSON object."
                }, {
                    "role": "user",
                    "content": context
                }],
                temperature=0
            )
            
            response_text = response.choices[0].message.content.strip()
            
            try:
                data = json.loads(response_text)
                return data.get("token"), data.get("timeframe", "1d")
            except json.JSONDecodeError:
                return None, "1d"
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error parsing trading pair: {str(e)}")

    def get_available_symbols(self) -> List[str]:
        """Get list of available trading pairs from TAapi."""
        try:
            # Fetch available symbols directly from Gate.io
            url = f"{self.taapi_base_url}/exchange-symbols"
            response = requests.get(url, params={
                "secret": self.taapi_api_key,
                "exchange": "gateio"
            })
            
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
                symbol for symbol in symbols 
                if isinstance(symbol, str) and symbol.endswith('/USDT')
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
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
            "ADA/USDT", "DOGE/USDT", "MATIC/USDT", "DOT/USDT", "LTC/USDT",
            "AVAX/USDT", "LINK/USDT", "UNI/USDT", "ATOM/USDT", "ETC/USDT",
            "XLM/USDT", "ALGO/USDT", "NEAR/USDT", "FTM/USDT", "SAND/USDT"
        ]

    def find_best_pair(self, token: str, available_symbols: List[str]) -> Optional[str]:
        """Find the best trading pair for a given token."""
        try:
            # Clean and standardize token
            token = token.strip().upper()
            
            # Remove /USDT if present
            token = token.replace('/USDT', '')
            
            # Common token name mappings
            token_mappings = {
                'BITCOIN': 'BTC',
                'ETHEREUM': 'ETH',
                'CARDANO': 'ADA',
                'SOLANA': 'SOL',
                'POLYGON': 'MATIC',
                'POLKADOT': 'DOT',
                'CHAINLINK': 'LINK',
                'AVALANCHE': 'AVAX',
                'DOGECOIN': 'DOGE',
                'RIPPLE': 'XRP',
                'LITECOIN': 'LTC',
                'COSMOS': 'ATOM',
                'BINANCE COIN': 'BNB',
                'BNB COIN': 'BNB'
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
                base = symbol.split('/')[0]  # Get base token
                if base.upper() == token:
                    print(f"\nFound case-insensitive match: {symbol}")
                    return symbol
            
            # Finally try partial matches
            partial_matches = []
            for symbol in available_symbols:
                base = symbol.split('/')[0]  # Get base token
                # Check if token is contained in base or base in token
                if token in base.upper() or base.upper() in token:
                    partial_matches.append(symbol)
            
            if partial_matches:
                # Prioritize shorter matches as they're likely more accurate
                partial_matches.sort(key=lambda x: len(x.split('/')[0]))
                best_match = partial_matches[0]
                print(f"\nFound partial match: {best_match}")
                return best_match
            
            print(f"\nNo matching pair found for token: {token}")
            return None
            
        except Exception as e:
            print(f"\nError finding best pair: {str(e)}")
            return None

    def format_indicators_json(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Format indicators into a clean JSON structure."""
        try:
            formatted = {}
            
            # Trend Indicators
            trend = {}
            if 'ema' in indicators:
                trend['ema'] = indicators['ema']
            if 'supertrend' in indicators:
                trend['supertrend'] = indicators['supertrend']
            if 'adx' in indicators:
                trend['adx'] = indicators['adx']
            if 'dmi' in indicators:
                trend['dmi'] = indicators['dmi']
            if 'psar' in indicators:
                trend['psar'] = indicators['psar']
            if trend:
                formatted['trend'] = trend
            
            # Momentum Indicators
            momentum = {}
            if 'rsi' in indicators:
                momentum['rsi'] = indicators['rsi']
            if 'macd' in indicators:
                momentum['macd'] = indicators['macd']
            if 'stoch' in indicators:
                momentum['stoch'] = indicators['stoch']
            if 'mfi' in indicators:
                momentum['mfi'] = indicators['mfi']
            if 'cci' in indicators:
                momentum['cci'] = indicators['cci']
            if momentum:
                formatted['momentum'] = momentum
            
            # Pattern Recognition
            patterns = {}
            for pattern in ['doji', 'engulfing', 'hammer', 'shootingstar']:
                if pattern in indicators:
                    patterns[pattern] = indicators[pattern]
            if patterns:
                formatted['patterns'] = patterns
            
            # Price Structure & Volume
            price = {}
            if 'fibonacciretracement' in indicators:
                price['fibonacci'] = indicators['fibonacciretracement']
            if 'bbands' in indicators:
                price['bollinger'] = indicators['bbands']
            if 'atr' in indicators:
                price['atr'] = indicators['atr']
            if price:
                formatted['price'] = price
            
            # Volume Analysis
            volume = {}
            if 'volume' in indicators:
                volume['current'] = indicators['volume']
            if 'vwap' in indicators:
                volume['vwap'] = indicators['vwap']
            if 'cmf' in indicators:
                volume['chaikin_mf'] = indicators['cmf']
            if 'ad' in indicators:
                volume['accumulation_dist'] = indicators['ad']
            if 'adosc' in indicators:
                volume['chaikin_osc'] = indicators['adosc']
            if 'obv' in indicators:
                volume['on_balance'] = indicators['obv']
            if volume:
                formatted['volume'] = volume
            
            return formatted
            
        except Exception as e:
            print(f"Error formatting indicators: {str(e)}")
            return {}

    def _fetch_indicators(self, symbol: str, interval: str = "1d", exchange: str = "gateio") -> Optional[Dict[str, Any]]:
        """Fetch technical indicators using TAapi."""
        try:
            url = f"{self.taapi_base_url}/bulk"
            
            # Split indicators into batches to comply with API limits
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
                {"indicator": "stalledpattern"},
                {"indicator": "morningstar"},
                {"indicator": "eveningstar"},
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
            
            # Initialize the result dictionary and valid indicator counter
            result = {}
            valid_indicators = 0
            
            # Function to process a batch
            def process_batch(indicators):
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
                    print(f"Error Response Status: {response.status_code}")
                    print(f"Error Response Content: {response.text}")
                    return None
                
                return response.json().get("data", [])
            
            # Process first batch
            print("\nProcessing first batch of indicators...")
            batch1_data = process_batch(batch1)
            if batch1_data:
                for indicator_data in batch1_data:
                    if indicator_data.get("result") and not indicator_data.get("result").get("error"):
                        if indicator_data["indicator"] == "ema":
                            if "ema" not in result:
                                result["ema"] = {}
                            period = indicator_data["id"].split("_")[-2]
                            result["ema"][f"period_{period}"] = indicator_data["result"].get("value")
                        else:
                            valid_indicators += 1
                            result[indicator_data["indicator"]] = indicator_data["result"]
            
            # Process second batch
            print("\nProcessing second batch of indicators...")
            batch2_data = process_batch(batch2)
            if batch2_data:
                for indicator_data in batch2_data:
                    if indicator_data.get("result") and not indicator_data.get("result").get("error"):
                        valid_indicators += 1
                        result[indicator_data["indicator"]] = indicator_data["result"]
            
            # Process third batch
            print("\nProcessing third batch of indicators...")
            batch3_data = process_batch(batch3)
            if batch3_data:
                for indicator_data in batch3_data:
                    if indicator_data.get("result") and not indicator_data.get("result").get("error"):
                        valid_indicators += 1
                        result[indicator_data["indicator"]] = indicator_data["result"]
            
            # Return None if we don't have enough valid indicators
            if valid_indicators < 5:
                print(f"\nInsufficient valid indicators: {valid_indicators}")
                return None
            
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching indicators: {str(e)}")

    def generate_analysis(self, indicators: Dict[str, Any], symbol: str, interval: str, original_prompt: str) -> str:
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
        
        # Build indicator sections based on available data
        indicator_sections = []
        
        # Trend Indicators
        trend_indicators = []
        if 'ema' in indicators:
            ema_values = []
            for period in ['20', '50', '200']:
                if f'period_{period}' in indicators['ema']:
                    ema_values.append(f"{period} [{indicators['ema'][f'period_{period}']:.2f}]")
            if ema_values:
                trend_indicators.append(f"• EMAs: {', '.join(ema_values)}")
        
        if 'supertrend' in indicators and 'value' in indicators['supertrend']:
            trend_indicators.append(f"• Supertrend: {indicators['supertrend']['value']:.2f} (Signal: {indicators['supertrend'].get('valueAdvice', 'N/A')})")
        
        if 'adx' in indicators and 'value' in indicators['adx']:
            trend_indicators.append(f"• ADX: {indicators['adx']['value']:.2f}")
        
        if 'dmi' in indicators and all(k in indicators['dmi'] for k in ['pdi', 'mdi']):
            trend_indicators.append(f"• DMI: +DI {indicators['dmi']['pdi']:.2f}, -DI {indicators['dmi']['mdi']:.2f}")
        
        if 'psar' in indicators and 'value' in indicators['psar']:
            trend_indicators.append(f"• PSAR: {indicators['psar']['value']:.2f}")
        
        if trend_indicators:
            indicator_sections.append("Trend Indicators:\n" + "\n".join(trend_indicators))
        
        # Momentum & Oscillators
        momentum_indicators = []
        if 'rsi' in indicators and 'value' in indicators['rsi']:
            momentum_indicators.append(f"• RSI: {indicators['rsi']['value']:.2f}")
        
        if 'macd' in indicators and all(k in indicators['macd'] for k in ['valueMACD', 'valueMACDSignal', 'valueMACDHist']):
            macd = indicators['macd']
            momentum_indicators.append(f"• MACD: Line [{macd['valueMACD']:.2f}], Signal [{macd['valueMACDSignal']:.2f}], Hist [{macd['valueMACDHist']:.2f}]")
        
        if 'stoch' in indicators and all(k in indicators['stoch'] for k in ['valueK', 'valueD']):
            stoch = indicators['stoch']
            momentum_indicators.append(f"• Stochastic: K[{stoch['valueK']:.2f}], D[{stoch['valueD']:.2f}]")
        
        if 'mfi' in indicators and 'value' in indicators['mfi']:
            momentum_indicators.append(f"• MFI: {indicators['mfi']['value']:.2f}")
        
        if 'cci' in indicators and 'value' in indicators['cci']:
            momentum_indicators.append(f"• CCI: {indicators['cci']['value']:.2f}")
        
        # Additional momentum indicators
        if 'roc' in indicators and 'value' in indicators['roc']:
            momentum_indicators.append(f"• ROC: {indicators['roc']['value']:.2f}")
        
        if 'willr' in indicators and 'value' in indicators['willr']:
            momentum_indicators.append(f"• Williams %R: {indicators['willr']['value']:.2f}")
        
        if 'mom' in indicators and 'value' in indicators['mom']:
            momentum_indicators.append(f"• Momentum: {indicators['mom']['value']:.2f}")
        
        if 'trix' in indicators and 'value' in indicators['trix']:
            momentum_indicators.append(f"• TRIX: {indicators['trix']['value']:.2f}")
        
        if 'stochrsi' in indicators and 'value' in indicators['stochrsi']:
            momentum_indicators.append(f"• Stoch RSI: {indicators['stochrsi']['value']:.2f}")
        
        if momentum_indicators:
            indicator_sections.append("Momentum & Oscillators:\n" + "\n".join(momentum_indicators))
        
        # Pattern Signals
        pattern_indicators = []
        for pattern in ['doji', 'engulfing', 'hammer', 'shootingstar', 'morningstar', 'eveningstar', 'dragonflydoji', 'gravestonedoji', 'stalledpattern']:
            if pattern in indicators and 'value' in indicators[pattern]:
                pattern_indicators.append(f"• {pattern.title()}: {indicators[pattern]['value']}")
        
        if pattern_indicators:
            indicator_sections.append("Pattern Signals:\n" + "\n".join(pattern_indicators))
        
        # Price Structure & Volume
        price_indicators = []
        if 'fibonacciretracement' in indicators:
            fib = indicators['fibonacciretracement']
            if all(k in fib for k in ['value', 'trend', 'startPrice', 'endPrice']):
                price_indicators.append(f"• Fibonacci: {fib['value']:.2f} ({fib['trend']})")
                price_indicators.append(f"  Range: {fib['startPrice']} → {fib['endPrice']}")
        
        if 'bbands' in indicators and all(k in indicators['bbands'] for k in ['valueUpperBand', 'valueMiddleBand', 'valueLowerBand']):
            bb = indicators['bbands']
            price_indicators.append(f"• Bollinger Bands: Upper[{bb['valueUpperBand']:.2f}], Mid[{bb['valueMiddleBand']:.2f}], Lower[{bb['valueLowerBand']:.2f}]")
        
        if 'atr' in indicators and 'value' in indicators['atr']:
            price_indicators.append(f"• ATR: {indicators['atr']['value']:.2f}")
        
        if price_indicators:
            indicator_sections.append("Price Structure & Volume:\n" + "\n".join(price_indicators))
        
        # Volume Analysis
        volume_indicators = []
        if 'volume' in indicators and 'value' in indicators['volume']:
            volume_indicators.append(f"• Volume: {indicators['volume']['value']:.2f}")
        
        if 'vwap' in indicators and 'value' in indicators['vwap']:
            volume_indicators.append(f"• VWAP: {indicators['vwap']['value']:.2f}")
        
        if 'ad' in indicators and 'value' in indicators['ad']:
            volume_indicators.append(f"• Chaikin A/D: {indicators['ad']['value']:.2f}")
        
        if 'adosc' in indicators and 'value' in indicators['adosc']:
            volume_indicators.append(f"• A/D Oscillator: {indicators['adosc']['value']:.2f}")
        
        if 'cmf' in indicators and 'value' in indicators['cmf']:
            volume_indicators.append(f"• Chaikin MF: {indicators['cmf']['value']:.2f}")
        
        if 'obv' in indicators and 'value' in indicators['obv']:
            volume_indicators.append(f"• On Balance Volume: {indicators['obv']['value']:.2f}")
        
        if 'vosc' in indicators and 'value' in indicators['vosc']:
            volume_indicators.append(f"• Volume Oscillator: {indicators['vosc']['value']:.2f}")
        
        if volume_indicators:
            indicator_sections.append("Volume Analysis:\n" + "\n".join(volume_indicators))
        
        # Build the context for GPT-4
        context = f"""You are a seasoned technical analyst specializing in {time_horizon} cryptocurrency analysis.
Your analysis is known for being clear, opinionated, and actionable.

ANALYSIS REQUEST:
Original Query: "{original_prompt}"
Asset: {symbol}
Timeframe: {interval} candles
Trading Horizon: {time_horizon}
Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
Data Quality: {'Partial (Limited Historical Data)' if len(indicators) < 20 else 'Full'}

TECHNICAL INDICATORS:

{chr(10).join(indicator_sections)}

ANALYSIS REQUIREMENTS:

1. DIRECTIONAL BIAS & CONFIDENCE (25% of analysis)
- State a clear directional bias (bullish/bearish/neutral)
- Provide a confidence level (high/medium/low) with rationale
- Identify the 2-3 most significant signals forming your bias
- Note any conflicting signals and explain which you're prioritizing

2. KEY PRICE LEVELS & SETUPS (35% of analysis)
- Current price structure and important levels
- Specific entry zones with trigger conditions
- Stop loss levels with clear rationale
- Take profit targets aligned with {time_horizon} horizon
- Risk:Reward ratios for suggested setups

3. PATTERN ANALYSIS (20% of analysis)
- Highlight any significant chart patterns forming
- Evaluate pattern reliability in current context
- Note any relevant candlestick signals
- Discuss volume confirmation/divergence

4. RISK ASSESSMENT (20% of analysis)
- Key risks to your thesis
- Market conditions affecting reliability
- Volatility considerations
- Alternative scenarios to watch for

IMPORTANT GUIDELINES:
- Focus on {time_horizon} setups and signals
- Provide specific, actionable levels and conditions
- Explain indicator interactions and confluences
- Adjust confidence based on data quality
- Be clear about assumptions and limitations
- Highlight timing considerations for setups

Your analysis should be thorough but concise, focusing on actionable insights rather than just describing indicators. Emphasize how different signals interact to form a complete picture."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a professional technical analyst specializing in {time_horizon} cryptocurrency trading. Your analysis combines multiple timeframes and indicators to form actionable insights, always considering risk management and market context."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                temperature=0.7,
                max_tokens=2000  # Increased for more detailed analysis
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating analysis: {str(e)}") 
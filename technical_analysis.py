import pandas as pd
import numpy as np
from typing import Dict, Optional, List, TypedDict
import requests
import ta
from datetime import datetime, timedelta
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TechnicalIndicators(TypedDict):
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    sma_20: float
    sma_50: float
    sma_200: float
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    volume_sma: float
    volume_ratio: float
    mfi: float
    obv: float
    volume_trend: str
    bullish_divergence: bool
    bearish_divergence: bool
    current_price: float
    current_volume: float
    support_levels: List[float]
    resistance_levels: List[float]
    
    # New trend indicators
    adx: float
    di_plus: float
    di_minus: float
    ichimoku_a: float
    ichimoku_b: float
    ichimoku_base: float
    ichimoku_conv: float
    psar: float
    
    # New momentum indicators
    stoch_k: float
    stoch_d: float
    williams_r: float
    roc: float
    ao: float
    
    # New volume indicators
    acc_dist: float
    cmf: float
    eom: float
    force_index: float
    vpt: float
    
    # New volatility indicators
    atr: float
    keltner_high: float
    keltner_mid: float
    keltner_low: float
    donchian_high: float
    donchian_mid: float
    donchian_low: float
    
    # Candlestick patterns
    doji: bool
    hammer: bool
    shooting_star: bool
    bullish_engulfing: bool
    bearish_engulfing: bool
    
    # Price patterns
    double_top: bool
    double_bottom: bool
    price_breakout: bool
    price_breakdown: bool
    volume_breakout: bool

PATTERN_REQUIREMENTS = {
    'double_top': {
        'min_time': 20,    # Minimum bars between peaks
        'max_time': 60,    # Maximum bars between peaks
        'height_ratio': 0.02,  # Maximum difference in peak heights (2%)
        'depth_ratio': 0.05    # Minimum pullback depth (5%)
    },
    'head_shoulders': {
        'symmetry_tolerance': 0.1,  # 10% tolerance in shoulder heights
        'neckline_slope_max': 0.02  # Maximum neckline slope
    }
}

def fetch_historical_data(token_id: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    Fetch historical price and volume data from CoinGecko
    Returns DataFrame with OHLCV data
    Default to 365 days to ensure enough data for 200-day SMA
    """
    try:
        # First get OHLC data
        ohlc_url = f"https://api.coingecko.com/api/v3/coins/{token_id}/ohlc"
        ohlc_params = {
            "vs_currency": "usd",
            "days": str(days)
        }
        headers = {
            "accept": "application/json",
            "x-cg-demo-api-key": os.getenv('COINGECKO_API_KEY')
        }
        
        ohlc_response = requests.get(ohlc_url, params=ohlc_params, headers=headers)
        ohlc_response.raise_for_status()
        ohlc_data = ohlc_response.json()
        
        # Create DataFrame from OHLC data
        df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Get volume data separately
        volume_url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart"
        volume_params = {
            "vs_currency": "usd",
            "days": str(days),
            "interval": "daily"
        }
        
        volume_response = requests.get(volume_url, params=volume_params, headers=headers)
        volume_response.raise_for_status()
        volume_data = volume_response.json()
        
        # Create volume DataFrame
        volume_df = pd.DataFrame(volume_data['total_volumes'], columns=['timestamp', 'volume'])
        volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
        volume_df.set_index('timestamp', inplace=True)
        
        # Resample both DataFrames to daily frequency to ensure alignment
        df = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        
        volume_df = volume_df.resample('D').last()
        
        # Merge OHLC and volume data
        df = df.join(volume_df, how='left')
        
        # Fill any missing values
        df = df.ffill()
        
        # Ensure we have all required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            print("Missing required columns in data")
            return None
            
        print(f"Fetched {len(df)} days of OHLCV data")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First date: {df.index[0]}")
        print(f"Last date: {df.index[-1]}")
        print("\nSample of data:")
        print(df.tail().to_string())
        
        return df
        
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

def find_support_resistance_levels(df: pd.DataFrame, lookback_periods: int = 90, num_levels: int = 6) -> tuple[List[float], List[float]]:
    """
    Calculate key support and resistance levels using price action and volume
    Returns (support_levels, resistance_levels)
    """
    try:
        def find_levels_for_period(period_df: pd.DataFrame) -> tuple[List[float], List[float]]:
            current_price = period_df['close'].iloc[-1]
            
            # Find pivot points using rolling windows
            window = 5  # Look at 5 periods on each side
            
            # Calculate rolling max and min with smaller window for more points
            period_df['rolling_high'] = period_df['high'].rolling(window=window*2+1, center=True).max()
            period_df['rolling_low'] = period_df['low'].rolling(window=window*2+1, center=True).min()
            
            # Identify pivot points with less strict conditions
            pivot_highs = period_df[period_df['high'] >= period_df['rolling_high'] * 0.998]['high']  # Allow 0.2% variation
            pivot_lows = period_df[period_df['low'] <= period_df['rolling_low'] * 1.002]['low']      # Allow 0.2% variation
            
            # Add volume weight with more granular bins
            volume_weighted_levels = []
            
            # Create price bins (200 bins for more granularity)
            price_range = period_df['high'].max() - period_df['low'].min()
            bin_size = price_range / 200  # Increased from 100 to 200 bins
            
            for i in range(200):  # Increased range to match bin count
                price_level = period_df['low'].min() + bin_size * i
                # Find volumes near this price level with wider range
                mask = (period_df['low'] <= price_level + bin_size*1.5) & (period_df['high'] >= price_level - bin_size*1.5)
                volume_at_level = period_df[mask]['volume'].sum()
                volume_weighted_levels.append((price_level, volume_at_level))
            
            # Sort by volume
            volume_weighted_levels.sort(key=lambda x: x[1], reverse=True)
            
            # Get more high volume levels
            high_volume_levels = [level[0] for level in volume_weighted_levels[:20]]  # Increased from 10 to 20
            
            # Combine all potential levels
            all_levels = list(pivot_highs) + list(pivot_lows) + high_volume_levels
            
            # Remove duplicates and sort
            all_levels = sorted(list(set(all_levels)))
            
            # Separate into support and resistance based on current price
            support_levels = sorted([level for level in all_levels if level < current_price], reverse=True)
            resistance_levels = sorted([level for level in all_levels if level > current_price])
            
            return support_levels, resistance_levels
        
        # Get data for different timeframes
        near_term_df = df.tail(30).copy()
        medium_term_df = df.tail(90).copy()
        long_term_df = df.tail(200).copy()
        
        # Get levels for each timeframe
        near_term_supports, near_term_resistances = find_levels_for_period(near_term_df)
        medium_term_supports, medium_term_resistances = find_levels_for_period(medium_term_df)
        long_term_supports, long_term_resistances = find_levels_for_period(long_term_df)
        
        # Combine all levels
        all_supports = near_term_supports + medium_term_supports + long_term_supports
        all_resistances = near_term_resistances + medium_term_resistances + long_term_resistances
        
        # Remove duplicates and sort
        all_supports = sorted(list(set(all_supports)), reverse=True)
        all_resistances = sorted(list(set(all_resistances)))
        
        # Add Fibonacci levels from the long-term range
        high = long_term_df['high'].max()
        low = long_term_df['low'].min()
        price_range = high - low
        current_price = df['close'].iloc[-1]
        
        # Add more Fibonacci levels
        fib_levels = [
            low + price_range * 0.236,
            low + price_range * 0.382,
            low + price_range * 0.5,
            low + price_range * 0.618,
            low + price_range * 0.786,
            low + price_range * 1.0,    # 100% retracement
            low + price_range * 1.236,   # Extended Fibonacci
            low + price_range * 1.618,   # Golden ratio extension
            high - price_range * 0.236,
            high - price_range * 0.382,
            high - price_range * 0.5,
            high - price_range * 0.618,
            high - price_range * 0.786
        ]
        
        for fib in fib_levels:
            if fib < current_price:
                all_supports.append(fib)
            else:
                all_resistances.append(fib)
        
        # Remove duplicates and sort again
        all_supports = sorted(list(set(all_supports)), reverse=True)
        all_resistances = sorted(list(set(all_resistances)))
        
        # Group levels into clusters with smaller threshold
        def cluster_levels(levels: List[float], threshold: float = 0.01) -> List[float]:  # Reduced threshold from 0.02 to 0.01
            if not levels:
                return []
            
            clustered = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - current_cluster[0]) / current_cluster[0] <= threshold:
                    current_cluster.append(level)
                else:
                    # Use weighted average based on volume for cluster center
                    clustered.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [level]
            
            clustered.append(sum(current_cluster) / len(current_cluster))
            return clustered
        
        # Cluster the levels and take top levels
        support_levels = cluster_levels(all_supports)[:num_levels]
        resistance_levels = cluster_levels(all_resistances)[:num_levels]
        
        return support_levels, resistance_levels
        
    except Exception as e:
        print(f"Error calculating support/resistance levels: {e}")
        return [], []

def calculate_technical_indicators(df: pd.DataFrame) -> Optional[TechnicalIndicators]:
    """
    Calculate various technical indicators
    Returns dictionary of indicator values
    """
    try:
        # Ensure we have enough data
        min_periods = 200  # Minimum periods needed for longest indicator (200 SMA)
        if len(df) < min_periods:
            print(f"Warning: Not enough data. Need at least {min_periods} periods.")
            return None
            
        # Verify required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            print("Missing required columns for technical analysis")
            return None
            
        # Initialize indicators
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(
            df['close'],
            window_slow=26,
            window_fast=12,
            window_sign=9
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Moving averages
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20)
        df['bollinger_upper'] = bollinger.bollinger_hband()
        df['bollinger_middle'] = bollinger.bollinger_mavg()
        df['bollinger_lower'] = bollinger.bollinger_lband()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ema'] = ta.trend.EMAIndicator(df['volume'], window=20).ema_indicator()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Money Flow Index
        df['mfi'] = ta.volume.MFIIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            window=14
        ).money_flow_index()
        
        # On-Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['close'],
            volume=df['volume']
        ).on_balance_volume()
        
        # New Trend Indicators
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['di_plus'] = adx.adx_pos()
        df['di_minus'] = adx.adx_neg()
        
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        
        df['psar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
        
        # New Momentum Indicators
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
        df['ao'] = ta.momentum.AwesomeOscillatorIndicator(df['high'], df['low']).awesome_oscillator()
        
        # New Volume Indicators
        df['acc_dist'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        df['eom'] = ta.volume.EaseOfMovementIndicator(df['high'], df['low'], df['volume']).ease_of_movement()
        df['force_index'] = ta.volume.ForceIndexIndicator(df['close'], df['volume']).force_index()
        df['vpt'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
        
        # New Volatility Indicators
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['keltner_high'] = keltner.keltner_channel_hband()
        df['keltner_mid'] = keltner.keltner_channel_mband()
        df['keltner_low'] = keltner.keltner_channel_lband()
        
        donchian = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
        df['donchian_high'] = donchian.donchian_channel_hband()
        df['donchian_mid'] = donchian.donchian_channel_mband()
        df['donchian_low'] = donchian.donchian_channel_lband()
        
        # Enhanced trend analysis
        trend_context = analyze_trend_context(df)
        
        # Enhanced divergence detection
        divergences = detect_divergences(df)
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Get recent volume trend
        volume_trend = 'increasing' if df['volume'].tail(5).is_monotonic_increasing else 'decreasing' if df['volume'].tail(5).is_monotonic_decreasing else 'mixed'
        
        # Calculate support and resistance levels
        support_levels, resistance_levels = find_support_resistance_levels(df)
        
        # Detect patterns using enhanced peak/trough detection
        peaks, troughs = find_peaks_troughs(df)
        candlestick_patterns = detect_candlestick_patterns(df)
        price_patterns = detect_price_patterns(df)
        
        # Create indicators dictionary with all values
        return TechnicalIndicators(
            # Existing indicators
            rsi=latest['rsi'],
            macd=latest['macd'],
            macd_signal=latest['macd_signal'],
            macd_hist=latest['macd_hist'],
            sma_20=latest['sma_20'],
            sma_50=latest['sma_50'],
            sma_200=latest['sma_200'],
            bollinger_upper=latest['bollinger_upper'],
            bollinger_middle=latest['bollinger_middle'],
            bollinger_lower=latest['bollinger_lower'],
            volume_sma=latest['volume_sma'],
            volume_ratio=latest['volume_ratio'],
            mfi=latest['mfi'],
            obv=latest['obv'],
            volume_trend=volume_trend,
            bullish_divergence=divergences['bullish']['regular'] or divergences['bullish']['hidden'],
            bearish_divergence=divergences['bearish']['regular'] or divergences['bearish']['hidden'],
            current_price=latest['close'],
            current_volume=latest['volume'],
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            
            # New trend indicators
            adx=latest['adx'],
            di_plus=latest['di_plus'],
            di_minus=latest['di_minus'],
            ichimoku_a=latest['ichimoku_a'],
            ichimoku_b=latest['ichimoku_b'],
            ichimoku_base=latest['ichimoku_base'],
            ichimoku_conv=latest['ichimoku_conv'],
            psar=latest['psar'],
            
            # New momentum indicators
            stoch_k=latest['stoch_k'],
            stoch_d=latest['stoch_d'],
            williams_r=latest['williams_r'],
            roc=latest['roc'],
            ao=latest['ao'],
            
            # New volume indicators
            acc_dist=latest['acc_dist'],
            cmf=latest['cmf'],
            eom=latest['eom'],
            force_index=latest['force_index'],
            vpt=latest['vpt'],
            
            # New volatility indicators
            atr=latest['atr'],
            keltner_high=latest['keltner_high'],
            keltner_mid=latest['keltner_mid'],
            keltner_low=latest['keltner_low'],
            donchian_high=latest['donchian_high'],
            donchian_mid=latest['donchian_mid'],
            donchian_low=latest['donchian_low'],
            
            # Candlestick patterns
            doji=candlestick_patterns.get('doji', False),
            hammer=candlestick_patterns.get('hammer', False),
            shooting_star=candlestick_patterns.get('shooting_star', False),
            bullish_engulfing=candlestick_patterns.get('bullish_engulfing', False),
            bearish_engulfing=candlestick_patterns.get('bearish_engulfing', False),
            
            # Price patterns
            double_top=price_patterns.get('double_top', False),
            double_bottom=price_patterns.get('double_bottom', False),
            price_breakout=price_patterns.get('price_breakout', False),
            price_breakdown=price_patterns.get('price_breakdown', False),
            volume_breakout=price_patterns.get('volume_breakout', False)
        )
        
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return None

def run_analysis(token_id: str):
    print(f"\nStarting Technical Analysis for {token_id}...")
    
    # Fetch historical data
    print("Fetching historical data...")
    df = fetch_historical_data(token_id)
    
    if df is not None:
        # Calculate indicators
        print("Calculating technical indicators...")
        indicators = calculate_technical_indicators(df)
        
        if indicators:
            # Get enhanced trend analysis
            trend_context = analyze_trend_context(df)
            
            # Get enhanced divergence analysis
            divergences = detect_divergences(df)
            
            print("\n=== TECHNICAL ANALYSIS DATA ===")
            print(f"\nAsset: {token_id.upper()}")
            
            print("\nPrice Action:")
            print(f"- Current Price: ${indicators['current_price']:,.2f}")
            print(f"- 20 SMA: ${indicators['sma_20']:,.2f}")
            print(f"- 50 SMA: ${indicators['sma_50']:,.2f}")
            print(f"- 200 SMA: ${indicators['sma_200']:,.2f}")
            
            print("\nTrend Analysis:")
            print(f"- MA Slopes:")
            print(f"  * 20 MA: {trend_context['ma_20_slope']:.2f}%")
            print(f"  * 50 MA: {trend_context['ma_50_slope']:.2f}%")
            print(f"- Price Structure:")
            print(f"  * Higher Highs: {'Yes' if trend_context['higher_highs'] else 'No'}")
            print(f"  * Lower Lows: {'Yes' if trend_context['lower_lows'] else 'No'}")
            print(f"- Volume Confirms Trend: {'Yes' if trend_context['volume_confirms_trend'] else 'No'}")
            print(f"- Trend Strength: {trend_context['trend_strength']}")
            print(f"- Trend Direction: {trend_context['trend_direction']}")
            print("- Timeframe Alignment:")
            print(f"  * Short-term (20): {'Up' if trend_context['short_term_trend'] else 'Down'}")
            print(f"  * Medium-term (50): {'Up' if trend_context['medium_term_trend'] else 'Down'}")
            print(f"  * Long-term (200): {'Up' if trend_context['long_term_trend'] else 'Down'}")
            print(f"  * All Aligned: {'Yes' if trend_context['alignment_desc'] else 'No'}")
            
            print("\nSupport & Resistance:")
            print("Resistance Levels:")
            for i, level in enumerate(indicators['resistance_levels'], 1):
                print(f"- R{i}: ${level:,.2f}")
            print("Support Levels:")
            for i, level in enumerate(indicators['support_levels'], 1):
                print(f"- S{i}: ${level:,.2f}")
            
            print("\nTrend Indicators:")
            print(f"- ADX (Trend Strength): {indicators['adx']:.2f}")
            print(f"- DI+ (Bullish Pressure): {indicators['di_plus']:.2f}")
            print(f"- DI- (Bearish Pressure): {indicators['di_minus']:.2f}")
            print(f"- Parabolic SAR: ${indicators['psar']:,.2f}")
            print("\nIchimoku Cloud:")
            print(f"- Conversion Line: ${indicators['ichimoku_conv']:,.2f}")
            print(f"- Base Line: ${indicators['ichimoku_base']:,.2f}")
            print(f"- Leading Span A: ${indicators['ichimoku_a']:,.2f}")
            print(f"- Leading Span B: ${indicators['ichimoku_b']:,.2f}")
            
            print("\nMomentum Indicators:")
            print(f"- RSI (14): {indicators['rsi']:.2f}")
            print(f"- Stochastic K: {indicators['stoch_k']:.2f}")
            print(f"- Stochastic D: {indicators['stoch_d']:.2f}")
            print(f"- Williams %R: {indicators['williams_r']:.2f}")
            print(f"- Rate of Change: {indicators['roc']:.2f}")
            print(f"- Awesome Oscillator: {indicators['ao']:.2f}")
            print(f"- MACD: {indicators['macd']:.2f}")
            print(f"- MACD Signal: {indicators['macd_signal']:.2f}")
            print(f"- MACD Histogram: {indicators['macd_hist']:.2f}")
            print(f"- Money Flow Index: {indicators['mfi']:.2f}")
            
            print("\nVolatility Indicators:")
            print(f"- ATR: ${indicators['atr']:,.2f}")
            print("\nBollinger Bands:")
            print(f"- Upper: ${indicators['bollinger_upper']:,.2f}")
            print(f"- Middle: ${indicators['bollinger_middle']:,.2f}")
            print(f"- Lower: ${indicators['bollinger_lower']:,.2f}")
            print("\nKeltner Channel:")
            print(f"- Upper: ${indicators['keltner_high']:,.2f}")
            print(f"- Middle: ${indicators['keltner_mid']:,.2f}")
            print(f"- Lower: ${indicators['keltner_low']:,.2f}")
            print("\nDonchian Channel:")
            print(f"- Upper: ${indicators['donchian_high']:,.2f}")
            print(f"- Middle: ${indicators['donchian_mid']:,.2f}")
            print(f"- Lower: ${indicators['donchian_low']:,.2f}")
            
            print("\nVolume Analysis:")
            print(f"- Current Volume: {indicators['current_volume']:,.2f}")
            print(f"- Volume SMA (20): {indicators['volume_sma']:,.2f}")
            print(f"- Volume Ratio: {indicators['volume_ratio']:.2f}")
            print(f"- Volume Trend (5 periods): {indicators['volume_trend']}")
            print(f"- On-Balance Volume: {indicators['obv']:,.2f}")
            print(f"- Accumulation/Distribution: {indicators['acc_dist']:,.2f}")
            print(f"- Chaikin Money Flow: {indicators['cmf']:.2f}")
            print(f"- Ease of Movement: {indicators['eom']:.2f}")
            print(f"- Force Index: {indicators['force_index']:,.2f}")
            print(f"- Volume-Price Trend: {indicators['vpt']:,.2f}")
            
            print("\nCandlestick Patterns:")
            if indicators['doji']: print("- Doji")
            if indicators['hammer']: print("- Hammer")
            if indicators['shooting_star']: print("- Shooting Star")
            if indicators['bullish_engulfing']: print("- Bullish Engulfing")
            if indicators['bearish_engulfing']: print("- Bearish Engulfing")
            
            print("\nPrice Patterns:")
            if indicators['double_top']: print("- Double Top")
            if indicators['double_bottom']: print("- Double Bottom")
            if indicators['price_breakout']: print("- Price Breakout")
            if indicators['price_breakdown']: print("- Price Breakdown")
            if indicators['volume_breakout']: print("- Volume-Confirmed Breakout")
            
            print("\nDivergence Analysis:")
            print("Bullish Divergences:")
            if divergences['bullish']['regular']:
                print(f"- Regular (Strength: {divergences['bullish']['strength']:.2f})")
            if divergences['bullish']['hidden']:
                print("- Hidden")
            if not divergences['bullish']['regular'] and not divergences['bullish']['hidden']:
                print("- None detected")
                
            print("Bearish Divergences:")
            if divergences['bearish']['regular']:
                print(f"- Regular (Strength: {divergences['bearish']['strength']:.2f})")
            if divergences['bearish']['hidden']:
                print("- Hidden")
            if not divergences['bearish']['regular'] and not divergences['bearish']['hidden']:
                print("- None detected")
            
            print("\nOverall Market Context:")
            print(f"- Trend Phase: {trend_context['trend_strength']} {trend_context['trend_direction']}")
            print(f"- Price Structure: {'Bullish' if trend_context['higher_highs'] and not trend_context['lower_lows'] else 'Bearish' if trend_context['lower_lows'] and not trend_context['higher_highs'] else 'Mixed'}")
            print(f"- Volume Context: {'Confirming' if trend_context['volume_confirms_trend'] else 'Non-confirming'}")
            print(f"- Timeframe Alignment: {'Strong' if trend_context['alignment_desc'] else 'Mixed'}")
            print(f"- RSI Condition: {'Overbought' if indicators['rsi'] > 70 else 'Oversold' if indicators['rsi'] < 30 else 'Neutral'}")
            print(f"- MACD Signal: {'Bullish' if indicators['macd'] > indicators['macd_signal'] else 'Bearish'}")
            print(f"- Volume vs SMA: {'Above Average' if indicators['volume_ratio'] > 1 else 'Below Average'}")
    else:
        print("Failed to fetch historical data.")

def detect_candlestick_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Detect common candlestick patterns using traditional TA rules
    Returns dictionary of pattern signals
    """
    try:
        # Ensure we have enough data
        if len(df) < 15:  # Increased minimum required periods
            print("Not enough data for candlestick pattern detection")
            return {}
            
        # Remove any rows with NaN values
        df = df.dropna()
        if len(df) < 15:
            print("Not enough valid data after removing NaN values")
            return {}
        
        # Get latest candles
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate basic measures for latest candle
        body = latest['close'] - latest['open']
        upper_shadow = latest['high'] - max(latest['open'], latest['close'])
        lower_shadow = min(latest['open'], latest['close']) - latest['low']
        total_range = latest['high'] - latest['low']
        
        print("\nCandlestick Measurements:")
        print(f"Body: {body:.2f}")
        print(f"Upper Shadow: {upper_shadow:.2f}")
        print(f"Lower Shadow: {lower_shadow:.2f}")
        print(f"Total Range: {total_range:.2f}")
        
        patterns = {}
        
        # Calculate average range and volatility using 10 periods
        avg_range = df['high'].tail(10).mean() - df['low'].tail(10).mean()
        if avg_range == 0:
            print("Invalid average range")
            return {}
            
        print(f"Average Range: {avg_range:.2f}")
        
        # Doji - Traditional rules: very small body (<=10% of range) with shadows
        doji_conditions = [
            abs(body) <= (total_range * 0.1),  # Body is 10% or less of total range
            total_range > (avg_range * 0.5),    # Significant enough range
            (upper_shadow + lower_shadow) > abs(body) * 2  # Shadows dominate
        ]
        patterns['doji'] = all(doji_conditions)
        print(f"Doji Conditions: {doji_conditions}")
        
        # Hammer - Traditional rules: long lower shadow, small upper shadow, appears after decline
        # Check for downward pressure in recent periods (15 periods for trend confirmation)
        price_declining = sum(df['close'].tail(15) < df['open'].tail(15)) >= 8  # More than half bearish
        hammer_conditions = [
            lower_shadow > (total_range * 0.6),  # Lower shadow is at least 60% of range
            upper_shadow <= (total_range * 0.1), # Minimal upper shadow
            abs(body) <= (total_range * 0.3),    # Body is not too large
            price_declining                       # Recent downward pressure
        ]
        patterns['hammer'] = all(hammer_conditions)
        print(f"Hammer Conditions: {hammer_conditions}")
        
        # Shooting Star - Traditional rules: long upper shadow, small lower shadow, appears after advance
        price_advancing = sum(df['close'].tail(15) > df['open'].tail(15)) >= 8  # More than half bullish
        star_conditions = [
            upper_shadow > (total_range * 0.6),  # Upper shadow is at least 60% of range
            lower_shadow <= (total_range * 0.1), # Minimal lower shadow
            abs(body) <= (total_range * 0.3),    # Body is not too large
            price_advancing                       # Recent upward pressure
        ]
        patterns['shooting_star'] = all(star_conditions)
        print(f"Shooting Star Conditions: {star_conditions}")
        
        # Engulfing patterns - Traditional rules: second candle completely engulfs first
        # Also considering relative size and market context
        prev_body = prev['close'] - prev['open']
        prev_size = abs(prev_body)
        curr_size = abs(body)
        
        # Check recent trend for context (10 periods)
        recent_trend_bearish = sum(df['close'].tail(10) < df['open'].tail(10)) >= 6
        recent_trend_bullish = sum(df['close'].tail(10) > df['open'].tail(10)) >= 6
        
        bullish_engulfing_conditions = [
            body > 0,                            # Current candle is bullish
            prev_body < 0,                       # Previous candle is bearish
            latest['close'] > prev['open'],      # Closes above previous open
            latest['open'] < prev['close'],      # Opens below previous close
            curr_size > prev_size * 1.5,         # Significantly larger
            recent_trend_bearish                 # Appears in downtrend
        ]
        patterns['bullish_engulfing'] = all(bullish_engulfing_conditions)
        print(f"Bullish Engulfing Conditions: {bullish_engulfing_conditions}")
        
        bearish_engulfing_conditions = [
            body < 0,                            # Current candle is bearish
            prev_body > 0,                       # Previous candle is bullish
            latest['open'] > prev['close'],      # Opens above previous close
            latest['close'] < prev['open'],      # Closes below previous open
            curr_size > prev_size * 1.5,         # Significantly larger
            recent_trend_bullish                 # Appears in uptrend
        ]
        patterns['bearish_engulfing'] = all(bearish_engulfing_conditions)
        print(f"Bearish Engulfing Conditions: {bearish_engulfing_conditions}")
        
        return patterns
        
    except Exception as e:
        print(f"Error detecting candlestick patterns: {e}")
        return {}

def detect_price_patterns(df: pd.DataFrame, window: int = 60) -> Dict[str, bool]:  # Increased default window
    """
    Detect technical price patterns using traditional TA rules
    Returns dictionary of pattern signals
    """
    try:
        # Ensure we have enough data
        if len(df) < window:
            print(f"Not enough data for price pattern detection (need at least {window} periods)")
            return {}
            
        # Remove any rows with NaN values
        df = df.dropna()
        if len(df) < window:
            print("Not enough valid data after removing NaN values")
            return {}
        
        patterns = {}
        
        # Get latest values
        latest_close = df['close'].iloc[-1]
        latest_volume = df['volume'].iloc[-1]
        
        # Calculate ATR for volatility-based thresholds (using standard 14 periods)
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().iloc[-1]
        if pd.isna(atr) or atr == 0:
            print("Invalid ATR value")
            return {}
            
        print(f"\nPrice Pattern Measurements:")
        print(f"ATR: {atr:.2f}")
        print(f"Latest Close: {latest_close:.2f}")
        print(f"Latest Volume: {latest_volume:.2f}")
        
        # Find significant peaks and troughs with wider window
        def find_peaks_troughs(data: pd.Series, window: int = 10) -> tuple[list, list]:  # Increased from 5 to 10
            peaks = []
            troughs = []
            
            for i in range(window, len(data) - window):
                # Look for more significant peaks/troughs
                if all(data.iloc[i] > data.iloc[i-j] for j in range(1, window+1)) and \
                   all(data.iloc[i] > data.iloc[i+j] for j in range(1, window+1)):
                    peaks.append((i, data.iloc[i]))
                    
                if all(data.iloc[i] < data.iloc[i-j] for j in range(1, window+1)) and \
                   all(data.iloc[i] < data.iloc[i+j] for j in range(1, window+1)):
                    troughs.append((i, data.iloc[i]))
            
            return peaks, troughs
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_troughs(df['high'], window=10)
        
        print(f"Found {len(peaks)} peaks and {len(troughs)} troughs")
        
        # Double Top Pattern using PATTERN_REQUIREMENTS
        if len(peaks) >= 2:
            last_two_peaks = peaks[-2:]
            peak_diff = abs(last_two_peaks[0][1] - last_two_peaks[1][1])
            time_between = last_two_peaks[1][0] - last_two_peaks[0][0]
            min_between = min(df['low'].iloc[last_two_peaks[0][0]:last_two_peaks[1][0]+1])
            avg_peak_height = (last_two_peaks[0][1] + last_two_peaks[1][1]) / 2
            
            double_top_conditions = [
                peak_diff <= (avg_peak_height * PATTERN_REQUIREMENTS['double_top']['height_ratio']),  # Height difference within tolerance
                time_between >= PATTERN_REQUIREMENTS['double_top']['min_time'],  # Minimum time between peaks
                time_between <= PATTERN_REQUIREMENTS['double_top']['max_time'],  # Maximum time between peaks
                min_between < min(last_two_peaks[0][1], last_two_peaks[1][1]) * (1 - PATTERN_REQUIREMENTS['double_top']['depth_ratio'])  # Significant pullback
            ]
            patterns['double_top'] = all(double_top_conditions)
            print(f"Double Top Conditions: {double_top_conditions}")
        
        # Double Bottom Pattern using similar requirements
        if len(troughs) >= 2:
            last_two_troughs = troughs[-2:]
            trough_diff = abs(last_two_troughs[0][1] - last_two_troughs[1][1])
            time_between = last_two_troughs[1][0] - last_two_troughs[0][0]
            max_between = max(df['high'].iloc[last_two_troughs[0][0]:last_two_troughs[1][0]+1])
            avg_trough_height = (last_two_troughs[0][1] + last_two_troughs[1][1]) / 2
            
            double_bottom_conditions = [
                trough_diff <= (avg_trough_height * PATTERN_REQUIREMENTS['double_top']['height_ratio']),  # Using same ratio as double top
                time_between >= PATTERN_REQUIREMENTS['double_top']['min_time'],
                time_between <= PATTERN_REQUIREMENTS['double_top']['max_time'],
                max_between > max(last_two_troughs[0][1], last_two_troughs[1][1]) * (1 + PATTERN_REQUIREMENTS['double_top']['depth_ratio'])
            ]
            patterns['double_bottom'] = all(double_bottom_conditions)
            print(f"Double Bottom Conditions: {double_bottom_conditions}")
        
        # Head and Shoulders Pattern using PATTERN_REQUIREMENTS
        if len(peaks) >= 3:
            last_three_peaks = peaks[-3:]
            left_shoulder = last_three_peaks[0][1]
            head = last_three_peaks[1][1]
            right_shoulder = last_three_peaks[2][1]
            
            # Check shoulder symmetry
            shoulder_height_diff = abs(left_shoulder - right_shoulder)
            avg_shoulder_height = (left_shoulder + right_shoulder) / 2
            shoulder_symmetry = shoulder_height_diff <= (avg_shoulder_height * PATTERN_REQUIREMENTS['head_shoulders']['symmetry_tolerance'])
            
            # Calculate neckline slope
            left_trough = min(df['low'].iloc[last_three_peaks[0][0]:last_three_peaks[1][0]])
            right_trough = min(df['low'].iloc[last_three_peaks[1][0]:last_three_peaks[2][0]])
            time_diff = last_three_peaks[2][0] - last_three_peaks[0][0]
            neckline_slope = abs((right_trough - left_trough) / time_diff)
            
            head_shoulders_conditions = [
                shoulder_symmetry,
                head > max(left_shoulder, right_shoulder),
                neckline_slope <= PATTERN_REQUIREMENTS['head_shoulders']['neckline_slope_max']
            ]
            patterns['head_shoulders'] = all(head_shoulders_conditions)
            print(f"Head and Shoulders Conditions: {head_shoulders_conditions}")
        
        # Calculate dynamic support and resistance using longer periods
        def calculate_pivots(df: pd.DataFrame, window: int = 30) -> tuple[float, float]:  # Increased from 20 to 30
            recent_df = df.tail(window)
            pivot = (recent_df['high'].mean() + recent_df['low'].mean() + recent_df['close'].mean()) / 3
            resistance = pivot + (pivot - recent_df['low'].mean())
            support = pivot - (recent_df['high'].mean() - pivot)
            return support, resistance
        
        support, resistance = calculate_pivots(df)
        avg_volume = df['volume'].rolling(window=30).mean().iloc[-1]  # Increased from 20 to 30
        
        # Breakout patterns with volume confirmation (using 5 periods for momentum)
        breakout_conditions = [
            latest_close > resistance,                    # Price above resistance
            latest_close > (resistance + atr * 0.5),      # Significant break
            latest_volume > avg_volume * 1.5,             # Above average volume
            df['close'].tail(5).is_monotonic_increasing  # 5 periods of momentum
        ]
        patterns['price_breakout'] = all(breakout_conditions)
        print(f"Breakout Conditions: {breakout_conditions}")
        
        # Breakdown patterns with volume confirmation
        breakdown_conditions = [
            latest_close < support,                      # Price below support
            latest_close < (support - atr * 0.5),        # Significant break
            latest_volume > avg_volume * 1.5,            # Above average volume
            df['close'].tail(5).is_monotonic_decreasing # 5 periods of momentum
        ]
        patterns['price_breakdown'] = all(breakdown_conditions)
        print(f"Breakdown Conditions: {breakdown_conditions}")
        
        # Volume-confirmed breakout (strong breakout with sustained volume)
        if patterns['price_breakout']:
            volume_confirmation = all(df['volume'].tail(5) > avg_volume * 1.5)  # 5 periods of high volume
            patterns['volume_breakout'] = volume_confirmation
        
        return patterns
        
    except Exception as e:
        print(f"Error detecting price patterns: {e}")
        return {}

def analyze_trend_context(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Enhanced trend analysis using multiple confirmations
    Returns dictionary of trend characteristics
    """
    try:
        # Calculate slopes of multiple MAs
        ma_20_slope = (df['sma_20'].diff(5) / df['sma_20'].shift(5)) * 100
        ma_50_slope = (df['sma_50'].diff(10) / df['sma_50'].shift(10)) * 100
        ma_200_slope = (df['sma_200'].diff(20) / df['sma_200'].shift(20)) * 100
        
        # Get current values
        current_price = df['close'].iloc[-1]
        current_ma20 = df['sma_20'].iloc[-1]
        current_ma50 = df['sma_50'].iloc[-1]
        current_ma200 = df['sma_200'].iloc[-1]
        
        # Define trend determination for each timeframe
        def determine_trend(prices: pd.Series, ma_series: pd.Series, slope: float) -> str:
            """
            Determine trend based on:
            1. Price relative to MA
            2. MA slope
            3. Recent price action
            """
            price = prices.iloc[-1]
            ma = ma_series.iloc[-1]
            
            # Calculate recent highs and lows
            recent_high = prices.tail(10).max()
            recent_low = prices.tail(10).min()
            prev_high = prices.iloc[-11:-1].max()
            prev_low = prices.iloc[-11:-1].min()
            
            # Bullish conditions
            if price > ma and slope > 0 and recent_high > prev_high:
                return "Strong Up"
            elif price > ma and (slope > 0 or recent_high > prev_high):
                return "Moderate Up"
            elif price > ma or slope > 0:
                return "Weak Up"
            
            # Bearish conditions
            elif price < ma and slope < 0 and recent_low < prev_low:
                return "Strong Down"
            elif price < ma and (slope < 0 or recent_low < prev_low):
                return "Moderate Down"
            elif price < ma or slope < 0:
                return "Weak Down"
            
            return "Neutral"
        
        # Analyze each timeframe
        short_trend = determine_trend(
            df['close'].tail(20),
            df['sma_20'].tail(20),
            ma_20_slope.iloc[-1]
        )
        
        medium_trend = determine_trend(
            df['close'].tail(50),
            df['sma_50'].tail(50),
            ma_50_slope.iloc[-1]
        )
        
        long_trend = determine_trend(
            df['close'].tail(200),
            df['sma_200'].tail(200),
            ma_200_slope.iloc[-1]
        )
        
        # Determine trend alignment strength
        def get_trend_direction(trend: str) -> int:
            """Convert trend to numeric direction: 1 (up), 0 (neutral), -1 (down)"""
            if "Up" in trend:
                return 1
            elif "Down" in trend:
                return -1
            return 0
        
        trend_directions = [
            get_trend_direction(t) for t in [short_trend, medium_trend, long_trend]
        ]
        
        # Calculate alignment score (-3 to 3, where 3 is perfectly aligned up, -3 perfectly aligned down)
        alignment_score = sum(trend_directions)
        
        # Analyze price structure (higher highs/lower lows)
        def analyze_swing_points(data: pd.Series, window: int = 5) -> tuple[bool, bool]:
            highs = []
            lows = []
            for i in range(window, len(data) - window):
                if all(data.iloc[i] > data.iloc[i-j] for j in range(1, window+1)) and \
                   all(data.iloc[i] > data.iloc[i+j] for j in range(1, window+1)):
                    highs.append(data.iloc[i])
                if all(data.iloc[i] < data.iloc[i-j] for j in range(1, window+1)) and \
                   all(data.iloc[i] < data.iloc[i+j] for j in range(1, window+1)):
                    lows.append(data.iloc[i])
            
            higher_highs = len(highs) >= 2 and all(highs[i] > highs[i-1] for i in range(1, len(highs)))
            lower_lows = len(lows) >= 2 and all(lows[i] < lows[i-1] for i in range(1, len(lows)))
            return higher_highs, lower_lows
        
        higher_highs, lower_lows = analyze_swing_points(df['close'].tail(lookback))
        
        # Analyze volume trend relative to price
        price_up_days = df['close'] > df['open']
        volume_on_up_days = df.loc[price_up_days, 'volume'].mean()
        volume_on_down_days = df.loc[~price_up_days, 'volume'].mean()
        volume_trend = volume_on_up_days > volume_on_down_days
        
        # Calculate ADX trend strength
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        current_adx = adx.adx().iloc[-1]
        di_plus = adx.adx_pos().iloc[-1]
        di_minus = adx.adx_neg().iloc[-1]
        
        # Determine trend characteristics
        trend_strength = 'Strong' if current_adx > 25 else 'Moderate' if current_adx > 15 else 'Weak'
        trend_direction = 'Bullish' if di_plus > di_minus else 'Bearish'
        
        # Determine alignment description
        if alignment_score >= 2:
            alignment_desc = "Strongly Aligned Bullish"
        elif alignment_score == 1:
            alignment_desc = "Moderately Aligned Bullish"
        elif alignment_score == 0:
            alignment_desc = "Mixed"
        elif alignment_score == -1:
            alignment_desc = "Moderately Aligned Bearish"
        else:
            alignment_desc = "Strongly Aligned Bearish"
        
        return {
            'ma_20_slope': ma_20_slope.iloc[-1],
            'ma_50_slope': ma_50_slope.iloc[-1],
            'ma_200_slope': ma_200_slope.iloc[-1],
            'higher_highs': higher_highs,
            'lower_lows': lower_lows,
            'volume_confirms_trend': volume_trend,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'short_term_trend': short_trend,
            'medium_term_trend': medium_trend,
            'long_term_trend': long_trend,
            'alignment_score': alignment_score,
            'alignment_desc': alignment_desc,
            'price_above_ma20': current_price > current_ma20,
            'price_above_ma50': current_price > current_ma50,
            'price_above_ma200': current_price > current_ma200
        }
    except Exception as e:
        print(f"Error analyzing trend context: {e}")
        return {}

def find_peaks_troughs(df: pd.DataFrame, window: int = 10, volume_threshold: float = 1.5) -> tuple[list, list]:
    """
    Enhanced peak/trough detection with volume confirmation and significance filtering
    """
    try:
        peaks = []
        troughs = []
        
        # Calculate average range for significance threshold
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        avg_volume = df['volume'].rolling(window=20).mean()
        
        for i in range(window, len(df) - window):
            # Price conditions for peaks
            if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window+1)):
                
                # Calculate peak significance
                height_significance = (df['high'].iloc[i] - df['low'].iloc[i-window:i+window].min()) / atr.iloc[i]
                volume_significance = df['volume'].iloc[i] / avg_volume.iloc[i]
                
                # Only add significant peaks
                if height_significance > 2 and volume_significance > volume_threshold:
                    peaks.append((i, df['high'].iloc[i], height_significance, volume_significance))
            
            # Price conditions for troughs
            if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window+1)):
                
                # Calculate trough significance
                depth_significance = (df['high'].iloc[i-window:i+window].max() - df['low'].iloc[i]) / atr.iloc[i]
                volume_significance = df['volume'].iloc[i] / avg_volume.iloc[i]
                
                # Only add significant troughs
                if depth_significance > 2 and volume_significance > volume_threshold:
                    troughs.append((i, df['low'].iloc[i], depth_significance, volume_significance))
        
        # Sort by significance
        peaks.sort(key=lambda x: x[2] * x[3], reverse=True)
        troughs.sort(key=lambda x: x[2] * x[3], reverse=True)
        
        return peaks, troughs
        
    except Exception as e:
        print(f"Error finding peaks and troughs: {e}")
        return [], []

def detect_divergences(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Enhanced divergence detection with multiple confirmations
    """
    try:
        divergences = {
            'bullish': {'regular': False, 'hidden': False, 'strength': 0},
            'bearish': {'regular': False, 'hidden': False, 'strength': 0}
        }
        
        # Calculate multiple oscillators
        rsi = ta.momentum.RSIIndicator(df['close']).rsi()
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        stoch_k = stoch.stoch()
        macd = ta.trend.MACD(df['close'])
        macd_line = macd.macd()
        
        # Get recent price swings
        price_peaks, price_troughs = find_peaks_troughs(df[['high', 'low', 'close', 'volume']])
        
        if len(price_peaks) >= 2 and len(price_troughs) >= 2:
            # Regular Bullish Divergence (lower lows in price, higher lows in oscillator)
            last_two_price_troughs = price_troughs[-2:]
            
            # Check RSI
            rsi_at_troughs = [rsi.iloc[t[0]] for t in last_two_price_troughs]
            rsi_bullish_div = last_two_price_troughs[1][1] < last_two_price_troughs[0][1] and \
                             rsi_at_troughs[1] > rsi_at_troughs[0]
            
            # Check Stochastic
            stoch_at_troughs = [stoch_k.iloc[t[0]] for t in last_two_price_troughs]
            stoch_bullish_div = last_two_price_troughs[1][1] < last_two_price_troughs[0][1] and \
                               stoch_at_troughs[1] > stoch_at_troughs[0]
            
            # Check MACD
            macd_at_troughs = [macd_line.iloc[t[0]] for t in last_two_price_troughs]
            macd_bullish_div = last_two_price_troughs[1][1] < last_two_price_troughs[0][1] and \
                              macd_at_troughs[1] > macd_at_troughs[0]
            
            # Calculate divergence strength
            bullish_div_count = sum([rsi_bullish_div, stoch_bullish_div, macd_bullish_div])
            if bullish_div_count >= 2:  # Require at least 2 oscillators to confirm
                divergences['bullish']['regular'] = True
                divergences['bullish']['strength'] = bullish_div_count / 3  # Normalize to 0-1
            
            # Hidden Bullish Divergence (higher lows in price, lower lows in oscillator)
            hidden_bullish_div = last_two_price_troughs[1][1] > last_two_price_troughs[0][1] and \
                                any([
                                    rsi_at_troughs[1] < rsi_at_troughs[0],
                                    stoch_at_troughs[1] < stoch_at_troughs[0],
                                    macd_at_troughs[1] < macd_at_troughs[0]
                                ])
            divergences['bullish']['hidden'] = hidden_bullish_div
            
            # Regular Bearish Divergence (higher highs in price, lower highs in oscillator)
            last_two_price_peaks = price_peaks[-2:]
            
            # Check RSI
            rsi_at_peaks = [rsi.iloc[p[0]] for p in last_two_price_peaks]
            rsi_bearish_div = last_two_price_peaks[1][1] > last_two_price_peaks[0][1] and \
                             rsi_at_peaks[1] < rsi_at_peaks[0]
            
            # Check Stochastic
            stoch_at_peaks = [stoch_k.iloc[p[0]] for p in last_two_price_peaks]
            stoch_bearish_div = last_two_price_peaks[1][1] > last_two_price_peaks[0][1] and \
                               stoch_at_peaks[1] < stoch_at_peaks[0]
            
            # Check MACD
            macd_at_peaks = [macd_line.iloc[p[0]] for p in last_two_price_peaks]
            macd_bearish_div = last_two_price_peaks[1][1] > last_two_price_peaks[0][1] and \
                              macd_at_peaks[1] < macd_at_peaks[0]
            
            # Calculate divergence strength
            bearish_div_count = sum([rsi_bearish_div, stoch_bearish_div, macd_bearish_div])
            if bearish_div_count >= 2:  # Require at least 2 oscillators to confirm
                divergences['bearish']['regular'] = True
                divergences['bearish']['strength'] = bearish_div_count / 3  # Normalize to 0-1
            
            # Hidden Bearish Divergence (lower highs in price, higher highs in oscillator)
            hidden_bearish_div = last_two_price_peaks[1][1] < last_two_price_peaks[0][1] and \
                                any([
                                    rsi_at_peaks[1] > rsi_at_peaks[0],
                                    stoch_at_peaks[1] > stoch_at_peaks[0],
                                    macd_at_peaks[1] > macd_at_peaks[0]
                                ])
            divergences['bearish']['hidden'] = hidden_bearish_div
        
        return divergences
        
    except Exception as e:
        print(f"Error detecting divergences: {e}")
        return {
            'bullish': {'regular': False, 'hidden': False, 'strength': 0},
            'bearish': {'regular': False, 'hidden': False, 'strength': 0}
        }

if __name__ == "__main__":
    import sys
    
    # Use command line argument if provided, otherwise default to bitcoin
    token_id = sys.argv[1] if len(sys.argv) > 1 else "bitcoin"
    run_analysis(token_id) 
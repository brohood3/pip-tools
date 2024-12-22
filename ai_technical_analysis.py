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

def fetch_and_validate_data(token_id: str) -> Optional[pd.DataFrame]:
    """
    Fetch and validate OHLC and volume data from CoinGecko
    Returns DataFrame with OHLC data at 4-hour intervals for the last 30 days
    """
    try:
        api_key = os.getenv('COINGECKO_API_KEY')
        if not api_key:
            raise ValueError("COINGECKO_API_KEY environment variable is not set")

        # Fetch OHLC data
        print(f"\nFetching OHLC data for {token_id}...")
        ohlc_df = _fetch_ohlc_data(token_id, api_key)
        if ohlc_df is None:
            return None

        # Fetch volume data
        print(f"\nFetching volume data for {token_id}...")
        volume_df = _fetch_volume_data(token_id, api_key)
        if volume_df is None:
            return None

        # Align and validate the data
        df = _align_and_validate_data(ohlc_df, volume_df)
        if df is not None:
            _analyze_data(df)
        return df

    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

def _fetch_ohlc_data(token_id: str, api_key: str) -> Optional[pd.DataFrame]:
    """Fetch OHLC data from the /coins/{id}/ohlc endpoint"""
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/ohlc"
    params = {
        'vs_currency': 'usd',
        'days': '30'  # 30 days for 4-hour intervals
    }
    headers = {
        'accept': 'application/json',
        'x-cg-demo-api-key': api_key
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            print("No OHLC data received")
            return None

        # Create DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"Received {len(df)} OHLC data points")
        return df

    except Exception as e:
        print(f"Error fetching OHLC data: {str(e)}")
        return None

def _fetch_volume_data(token_id: str, api_key: str) -> Optional[pd.DataFrame]:
    """Fetch volume data from the /coins/{id}/market_chart endpoint"""
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': '30',
        'interval': None  # Get highest granularity for 30 days
    }
    headers = {
        'accept': 'application/json',
        'x-cg-demo-api-key': api_key
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        if not data or 'total_volumes' not in data:
            print("No volume data received")
            return None

        # Create DataFrame
        df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"Received {len(df)} volume data points")
        return df

    except Exception as e:
        print(f"Error fetching volume data: {str(e)}")
        return None

def _align_and_validate_data(ohlc_df: pd.DataFrame, volume_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Align and validate OHLC and volume data"""
    try:
        if ohlc_df is None or volume_df is None:
            return None

        # Expected values for 30 days of 4-hour data
        expected_interval = timedelta(hours=4)
        expected_periods = 180  # 30 days * 6 periods per day

        # Check for expected number of periods
        if len(ohlc_df) < expected_periods * 0.9:  # Allow 10% missing data
            print(f"Warning: Insufficient data points. Expected {expected_periods}, got {len(ohlc_df)}")

        # Resample volume data to match OHLC intervals
        volume_df = volume_df.resample('4h').sum()

        # Ensure both dataframes are sorted
        ohlc_df.sort_index(inplace=True)
        volume_df.sort_index(inplace=True)

        # Align the data
        df = ohlc_df.join(volume_df, how='inner')

        # Remove any duplicate indices
        df = df[~df.index.duplicated(keep='last')]

        # Validate time intervals
        time_diffs = df.index.to_series().diff()
        modal_interval = time_diffs.mode()[0]
        if abs(modal_interval - expected_interval) > expected_interval * 0.1:  # Allow 10% deviation
            print(f"Warning: Unexpected time interval. Expected {expected_interval}, got {modal_interval}")

        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print("\nWarning: Missing values detected:")
            print(missing_values[missing_values > 0])
            # Forward fill missing values
            df.ffill(inplace=True)

        # Validate data ranges
        _validate_data_ranges(df)

        return df

    except Exception as e:
        print(f"Error aligning and validating data: {str(e)}")
        return None

def _validate_data_ranges(df: pd.DataFrame):
    """Validate data ranges for anomalies"""
    # Check for price anomalies
    price_std = df['close'].std()
    price_mean = df['close'].mean()
    price_outliers = df[abs(df['close'] - price_mean) > 3 * price_std]
    if not price_outliers.empty:
        print("\nWarning: Potential price outliers detected:")
        print(price_outliers)

    # Check for volume anomalies
    volume_std = df['volume'].std()
    volume_mean = df['volume'].mean()
    volume_outliers = df[abs(df['volume'] - volume_mean) > 3 * volume_std]
    if not volume_outliers.empty:
        print("\nWarning: Potential volume outliers detected:")
        print(volume_outliers)

    # Check for zero or negative values
    if (df[['open', 'high', 'low', 'close', 'volume']] <= 0).any().any():
        print("\nWarning: Zero or negative values detected in the data")

    # Check for high-low relationship
    invalid_hl = df[df['high'] < df['low']]
    if not invalid_hl.empty:
        print("\nWarning: Invalid high-low relationship detected:")
        print(invalid_hl)

    # Check for OHLC relationship
    invalid_ohlc = df[
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ]
    if not invalid_ohlc.empty:
        print("\nWarning: Invalid OHLC relationship detected:")
        print(invalid_ohlc)

def _analyze_data(df: pd.DataFrame):
    """Analyze the fetched data for quality and completeness"""
    print("\nData Analysis Summary:")
    print(f"Total periods: {len(df)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Calculate time intervals
    time_diffs = df.index.to_series().diff()
    print("\nTime interval distribution:")
    print(time_diffs.value_counts().head())
    
    # Print data statistics
    print("\nData statistics:")
    print(df.describe())
    
    # Show sample data
    print("\nFirst 5 periods:")
    print(df.head().to_string())
    print("\nLast 5 periods:")
    print(df.tail().to_string())

def find_support_resistance_levels(df: pd.DataFrame, lookback_periods: int = 180, num_levels: int = 6) -> tuple[List[float], List[float]]:
    """
    Calculate key support and resistance levels using price action and volume
    Adjusted for 4-hour timeframe
    Returns (support_levels, resistance_levels)
    """
    try:
        def find_levels_for_period(period_df: pd.DataFrame) -> tuple[List[float], List[float]]:
            current_price = period_df['close'].iloc[-1]
            
            # Find pivot points using rolling windows (adjusted for 4-hour)
            window = 15  # Look at 2.5 days on each side
            
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
    Calculate various technical indicators for 4-hour data
    Returns dictionary of indicator values
    """
    try:
        # Verify we have enough data (30 days = 180 periods)
        if len(df) < 180:
            print(f"Warning: Not enough data. Need at least 180 periods, got {len(df)}")
            return None
            
        # Verify required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            print("Missing required columns for technical analysis")
            return None
            
        # Create a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Initialize indicators with adjusted periods for 4-hour timeframe
        # RSI (14 periods = 2.3 days)
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD (12/26/9 periods = 2/4.3/1.5 days)
        macd = ta.trend.MACD(
            df['close'],
            window_slow=26,
            window_fast=12,
            window_sign=9
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Moving averages (adjusted for 4-hour timeframe)
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=30).sma_indicator()  # 5 days
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=72).sma_indicator()  # 12 days
        df['sma_200'] = ta.trend.SMAIndicator(df['close'], window=180).sma_indicator()  # 30 days
        
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
        
        # Forward fill any NaN values that might occur at the start of some indicators
        df = df.ffill()
        
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

def get_current_price(token_id: str) -> float:
    """
    Get real-time price from CoinGecko's simple price endpoint
    """
    try:
        api_key = os.getenv('COINGECKO_API_KEY')
        if not api_key:
            raise ValueError("COINGECKO_API_KEY environment variable is not set")
            
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": token_id,
            "vs_currencies": "usd"
        }
        headers = {
            "accept": "application/json",
            "x-cg-demo-api-key": api_key
        }
        
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data[token_id]['usd']
    except Exception as e:
        print(f"Error fetching current price: {e}")
        return None

def run_analysis(token_id):
    print(f"\nStarting Technical Analysis for {token_id}...")
    
    print("Fetching historical data...")
    df = fetch_and_validate_data(token_id)
    
    if df is None or len(df) < 30:  # Minimum required periods
        print("Error: Insufficient data for analysis")
        return
    
    # Calculate technical indicators
    print("Calculating technical indicators...")
    indicators = calculate_technical_indicators(df)
    
    if indicators is None:
        print("Error: Failed to calculate technical indicators")
        return
    
    # Detect patterns
    patterns = detect_price_patterns(df)
    
    # Analyze trend context
    trend_context = analyze_trend_context(df)
    
    # Detect divergences
    divergences = detect_divergences(df)
    
    print("\n=== TECHNICAL ANALYSIS DATA ===\n")
    print(f"Asset: {token_id.upper()}")
    print(f"Timeframe: 30 days with 4-hour intervals")
    print(f"Total Periods: {len(df)} 4-hour candles\n")
    
    # Price Action
    current_price = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    price_diff = current_price - prev_close
    price_diff_pct = (price_diff / prev_close) * 100
    
    print("Price Action:")
    print(f"- Real-time Price: ${current_price:.2f}")
    print(f"- Latest 4h Close: ${prev_close:.2f}")
    print(f"- Price Difference: ${price_diff:.2f} ({price_diff_pct:.2f}%)")
    print(f"- 20 Period MA (5 days): ${indicators['sma_20']:.2f}")
    print(f"- 50 Period MA (12.5 days): ${indicators['sma_50']:.2f}")
    
    print("\nTrend Analysis (4h timeframe):")
    print("- MA Slopes:")
    print(f"  * 20 Period MA: {trend_context['ma_20_slope']:.2f}%")
    print(f"  * 50 Period MA: {trend_context['ma_50_slope']:.2f}%")
    print(f"- Current Trend: {trend_context['trend_direction']} ({trend_context['trend_strength']})")
    print(f"- ADX (Trend Strength): {indicators['adx']:.2f}")
    print(f"- DI+ (Bullish Pressure): {indicators['di_plus']:.2f}")
    print(f"- DI- (Bearish Pressure): {indicators['di_minus']:.2f}")
    print(f"- Parabolic SAR: ${indicators['psar']:.2f}")
    
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
    
    print("\nVolume Analysis:")
    avg_volume = df['volume'].mean()
    current_volume = df['volume'].iloc[-1]
    volume_ratio = (current_volume / avg_volume) * 100
    print(f"- Average 4h Volume: ${avg_volume:,.2f}")
    print(f"- Current 4h Volume: ${current_volume:,.2f}")
    print(f"- Volume Ratio: {volume_ratio:.2f}% of average")
    print(f"- Money Flow Index: {indicators['mfi']:.2f}")
    print(f"- On-Balance Volume: {indicators['obv']:.2f}")
    print(f"- Accumulation/Distribution: {indicators['acc_dist']:.2f}")
    print(f"- Chaikin Money Flow: {indicators['cmf']:.2f}")
    print(f"- Ease of Movement: {indicators['eom']:.2f}")
    print(f"- Force Index: {indicators['force_index']:.2f}")
    print(f"- Volume-Price Trend: {indicators['vpt']:.2f}")
    
    print("\nVolatility Indicators:")
    print(f"- ATR: ${indicators['atr']:.2f}")
    print("\nBollinger Bands:")
    print(f"- Upper: ${indicators['bollinger_upper']:.2f}")
    print(f"- Middle: ${indicators['bollinger_middle']:.2f}")
    print(f"- Lower: ${indicators['bollinger_lower']:.2f}")
    print("\nKeltner Channel:")
    print(f"- Upper: ${indicators['keltner_high']:.2f}")
    print(f"- Middle: ${indicators['keltner_mid']:.2f}")
    print(f"- Lower: ${indicators['keltner_low']:.2f}")
    print("\nDonchian Channel:")
    print(f"- Upper: ${indicators['donchian_high']:.2f}")
    print(f"- Middle: ${indicators['donchian_mid']:.2f}")
    print(f"- Lower: ${indicators['donchian_low']:.2f}")
    
    print("\nIchimoku Cloud:")
    print(f"- Conversion Line: ${indicators['ichimoku_conv']:.2f}")
    print(f"- Base Line: ${indicators['ichimoku_base']:.2f}")
    print(f"- Leading Span A: ${indicators['ichimoku_a']:.2f}")
    print(f"- Leading Span B: ${indicators['ichimoku_b']:.2f}")
    
    # Pattern Analysis
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
    
    # Support and Resistance
    print("\nKey Price Levels:")
    support_levels, resistance_levels = find_support_resistance_levels(df)
    
    print("Support Levels:")
    for level in sorted(support_levels, reverse=True)[:3]:
        print(f"- ${level:.2f}")
    
    print("Resistance Levels:")
    for level in sorted(resistance_levels)[:3]:
        print(f"- ${level:.2f}")
    
    # Generate AI Summary
    print("\nGenerating AI analysis summary...")
    summary = generate_ai_summary(indicators, trend_context, divergences)
    print("\nAI Analysis Summary:")
    print(summary)

def detect_candlestick_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Detect common candlestick patterns using traditional TA rules with more sensitive thresholds
    """
    try:
        if len(df) < 2:
            print("Not enough data for candlestick pattern detection")
            return {}
        
        # Get latest candles
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate basic measures for latest candle
        body = latest['close'] - latest['open']
        upper_shadow = latest['high'] - max(latest['open'], latest['close'])
        lower_shadow = min(latest['open'], latest['close']) - latest['low']
        total_range = latest['high'] - latest['low']
        
        if total_range == 0:
            return {}
        
        patterns = {}
        
        # Calculate average range and volatility using 3 periods (more sensitive)
        avg_range = df['high'].tail(3).mean() - df['low'].tail(3).mean()
        if avg_range == 0:
            return {}
            
        # Doji - More sensitive criteria
        doji_conditions = [
            abs(body) <= (total_range * 0.2),  # Increased from 0.15 to 0.2
            total_range > (avg_range * 0.3),    # Reduced from 0.4 to 0.3
            (upper_shadow + lower_shadow) > abs(body) * 1.2  # Reduced from 1.5 to 1.2
        ]
        patterns['doji'] = all(doji_conditions)
        
        # Hammer - More sensitive criteria
        price_declining = sum(df['close'].tail(5) < df['open'].tail(5)) >= 3  # Reduced from 8 to 5 periods
        hammer_conditions = [
            lower_shadow > (total_range * 0.4),  # Reduced from 0.5 to 0.4
            upper_shadow <= (total_range * 0.2), # Increased from 0.15 to 0.2
            abs(body) <= (total_range * 0.5),    # Increased from 0.4 to 0.5
            price_declining                       # Recent downward pressure
        ]
        patterns['hammer'] = all(hammer_conditions)
        
        # Shooting Star - More sensitive criteria
        price_advancing = sum(df['close'].tail(5) > df['open'].tail(5)) >= 3  # Reduced from 8 to 5 periods
        star_conditions = [
            upper_shadow > (total_range * 0.4),  # Reduced from 0.5 to 0.4
            lower_shadow <= (total_range * 0.2), # Increased from 0.15 to 0.2
            abs(body) <= (total_range * 0.5),    # Increased from 0.4 to 0.5
            price_advancing                       # Recent upward pressure
        ]
        patterns['shooting_star'] = all(star_conditions)
        
        # Engulfing patterns with more sensitive criteria
        prev_body = prev['close'] - prev['open']
        curr_size = abs(body)
        prev_size = abs(prev_body)
        
        # Check recent trend for context (reduced from 5 to 3 periods)
        recent_trend_bearish = sum(df['close'].tail(3) < df['open'].tail(3)) >= 2
        recent_trend_bullish = sum(df['close'].tail(3) > df['open'].tail(3)) >= 2
        
        bullish_engulfing_conditions = [
            body > 0,                            # Current candle is bullish
            prev_body < 0,                       # Previous candle is bearish
            latest['close'] > prev['open'] * 0.999,  # Closes near/above previous open (0.1% tolerance)
            latest['open'] < prev['close'] * 1.001,  # Opens near/below previous close (0.1% tolerance)
            curr_size > prev_size * 1.1,         # Reduced from 1.2 to 1.1
            recent_trend_bearish                 # Appears in downtrend
        ]
        patterns['bullish_engulfing'] = all(bullish_engulfing_conditions)
        
        bearish_engulfing_conditions = [
            body < 0,                            # Current candle is bearish
            prev_body > 0,                       # Previous candle is bullish
            latest['open'] > prev['close'] * 0.999,  # Opens near/above previous close (0.1% tolerance)
            latest['close'] < prev['open'] * 1.001,  # Closes near/below previous open (0.1% tolerance)
            curr_size > prev_size * 1.1,         # Reduced from 1.2 to 1.1
            recent_trend_bullish                 # Appears in uptrend
        ]
        patterns['bearish_engulfing'] = all(bearish_engulfing_conditions)
        
        return patterns
        
    except Exception as e:
        print(f"Error detecting candlestick patterns: {e}")
        return {}

def detect_price_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Detect technical price patterns using more sensitive thresholds
    """
    try:
        patterns = {}
        
        if len(df) < 20:  # Reduced from 30 to 20 periods minimum
            print("Not enough data for price pattern detection")
            return patterns
        
        # Get latest values
        latest_close = df['close'].iloc[-1]
        latest_volume = df['volume'].iloc[-1]
        
        # Calculate ATR with shorter period (reduced from 10 to 5 periods)
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=5).average_true_range().iloc[-1]
        if pd.isna(atr) or atr == 0:
            print("Invalid ATR value")
            return patterns
            
        # Find peaks and troughs with smaller window
        def find_peaks_troughs(data: pd.Series, window: int = 3) -> tuple[list, list]:  # Reduced from 5 to 3
            peaks = []
            troughs = []
            
            for i in range(window, len(data) - window):
                # More sensitive peak/trough detection
                if all(data.iloc[i] > data.iloc[i-j] * 0.999 for j in range(1, window+1)) and \
                   all(data.iloc[i] > data.iloc[i+j] * 0.999 for j in range(1, window+1)):
                    peaks.append((i, data.iloc[i]))
                    
                if all(data.iloc[i] < data.iloc[i-j] * 1.001 for j in range(1, window+1)) and \
                   all(data.iloc[i] < data.iloc[i+j] * 1.001 for j in range(1, window+1)):
                    troughs.append((i, data.iloc[i]))
            
            return peaks, troughs
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_troughs(df['high'])
        
        # Double Top Pattern with more sensitive requirements
        if len(peaks) >= 2:
            last_two_peaks = peaks[-2:]
            peak_diff = abs(last_two_peaks[0][1] - last_two_peaks[1][1])
            time_between = last_two_peaks[1][0] - last_two_peaks[0][0]
            min_between = min(df['low'].iloc[last_two_peaks[0][0]:last_two_peaks[1][0]+1])
            avg_peak_height = (last_two_peaks[0][1] + last_two_peaks[1][1]) / 2
            
            double_top_conditions = [
                peak_diff <= (avg_peak_height * 0.04),  # Increased from 0.03 to 0.04
                time_between >= 5,   # Reduced from 10 to 5
                time_between <= 40,  # Increased from 30 to 40
                min_between < min(last_two_peaks[0][1], last_two_peaks[1][1]) * 0.98  # Changed from 0.97 to 0.98
            ]
            patterns['double_top'] = all(double_top_conditions)
        
        # Double Bottom Pattern with more sensitive requirements
        if len(troughs) >= 2:
            last_two_troughs = troughs[-2:]
            trough_diff = abs(last_two_troughs[0][1] - last_two_troughs[1][1])
            time_between = last_two_troughs[1][0] - last_two_troughs[0][0]
            max_between = max(df['high'].iloc[last_two_troughs[0][0]:last_two_troughs[1][0]+1])
            avg_trough_height = (last_two_troughs[0][1] + last_two_troughs[1][1]) / 2
            
            double_bottom_conditions = [
                trough_diff <= (avg_trough_height * 0.04),  # Increased from 0.03 to 0.04
                time_between >= 5,   # Reduced from 10 to 5
                time_between <= 40,  # Increased from 30 to 40
                max_between > max(last_two_troughs[0][1], last_two_troughs[1][1]) * 1.02  # Changed from 1.03 to 1.02
            ]
            patterns['double_bottom'] = all(double_bottom_conditions)
        
        # Calculate dynamic support and resistance using shorter periods
        def calculate_pivots(df: pd.DataFrame, window: int = 10) -> tuple[float, float]:  # Reduced from 15 to 10
            recent_df = df.tail(window)
            pivot = (recent_df['high'].mean() + recent_df['low'].mean() + recent_df['close'].mean()) / 3
            resistance = pivot + (pivot - recent_df['low'].mean())
            support = pivot - (recent_df['high'].mean() - pivot)
            return support, resistance
        
        support, resistance = calculate_pivots(df)
        avg_volume = df['volume'].rolling(window=10).mean().iloc[-1]  # Reduced from 15 to 10
        
        # Breakout patterns with more sensitive volume confirmation
        breakout_conditions = [
            latest_close > resistance,                    # Price above resistance
            latest_close > (resistance + atr * 0.2),      # Reduced from 0.3 to 0.2
            latest_volume > avg_volume * 1.2,             # Reduced from 1.3 to 1.2
            df['close'].tail(2).is_monotonic_increasing  # Reduced from 3 to 2 periods
        ]
        patterns['price_breakout'] = all(breakout_conditions)
        
        # Breakdown patterns with more sensitive confirmation
        breakdown_conditions = [
            latest_close < support,                      # Price below support
            latest_close < (support - atr * 0.2),        # Reduced from 0.3 to 0.2
            latest_volume > avg_volume * 1.2,            # Reduced from 1.3 to 1.2
            df['close'].tail(2).is_monotonic_decreasing # Reduced from 3 to 2 periods
        ]
        patterns['price_breakdown'] = all(breakdown_conditions)
        
        # Volume-confirmed breakout with more sensitive criteria
        if patterns['price_breakout']:
            volume_confirmation = all(df['volume'].tail(2) > avg_volume * 1.2)  # Reduced from 3 to 2 periods and 1.3 to 1.2
            patterns['volume_breakout'] = volume_confirmation
        
        return patterns
        
    except Exception as e:
        print(f"Error detecting price patterns: {e}")
        return {}

def analyze_trend_context(df: pd.DataFrame) -> dict:
    """
    Analyze trend context using multiple timeframes and indicators
    """
    try:
        context = {}
        
        # Calculate moving averages
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        
        # Calculate slopes
        context['ma_20_slope'] = ((df['sma_20'].iloc[-1] - df['sma_20'].iloc[-20]) / df['sma_20'].iloc[-20]) * 100
        context['ma_50_slope'] = ((df['sma_50'].iloc[-1] - df['sma_50'].iloc[-50]) / df['sma_50'].iloc[-50]) * 100
        
        # Determine trend strength and direction
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        current_adx = adx.adx().iloc[-1]
        di_plus = adx.adx_pos().iloc[-1]
        di_minus = adx.adx_neg().iloc[-1]
        
        context['trend_strength'] = 'Strong' if current_adx > 25 else 'Moderate' if current_adx > 15 else 'Weak'
        context['trend_direction'] = 'Bullish' if di_plus > di_minus else 'Bearish'
        
        # Price structure analysis
        recent_highs = df['high'].rolling(window=20).max()
        recent_lows = df['low'].rolling(window=20).min()
        context['higher_highs'] = recent_highs.iloc[-1] > recent_highs.iloc[-20]
        context['lower_lows'] = recent_lows.iloc[-1] < recent_lows.iloc[-20]
        
        # Volume trend analysis
        avg_volume = df['volume'].rolling(window=20).mean()
        context['volume_confirms_trend'] = df['volume'].iloc[-1] > avg_volume.iloc[-1]
        
        return context
        
    except Exception as e:
        print(f"Error analyzing trend context: {e}")
        return {
            'trend_strength': 'Unknown',
            'trend_direction': 'Unknown',
            'ma_20_slope': 0,
            'ma_50_slope': 0,
            'higher_highs': False,
            'lower_lows': False,
            'volume_confirms_trend': False
        }

def generate_ai_summary(indicators: TechnicalIndicators, trend_context: dict, divergences: dict) -> str:
    """
    Generate an insightful, opinionated technical analysis summary using OpenAI
    """
    try:
        client = OpenAI()
        
        # Format the data for the prompt
        prompt = f"""As a seasoned technical analyst, provide an insightful and opinionated analysis of the current market structure and potential opportunities/risks based on 4-hour interval data over the last 30 days. Here's the technical data:

Timeframe Context:
- Analysis Period: 30 days
- Interval: 4-hour candles
- Total Periods: 180 (6 candles per day)

Market Context:
Current Price: ${indicators['current_price']:,.2f} (4h close)
Trend: {trend_context['trend_strength']} {trend_context['trend_direction']}
MA Slopes (4h timeframe): 
- 20 Period MA (5 days): {trend_context['ma_20_slope']:.2f}%
- 50 Period MA (12.5 days): {trend_context['ma_50_slope']:.2f}%
Price vs MAs: 
- 20 Period MA: ${indicators['sma_20']:,.2f}
- 50 Period MA: ${indicators['sma_50']:,.2f}

Trend Strength & Direction:
ADX: {indicators['adx']:.2f} ({trend_context['trend_strength']})
DI+/DI-: {indicators['di_plus']:.2f}/{indicators['di_minus']:.2f}
Parabolic SAR: ${indicators['psar']:.2f}

Price Structure:
- Higher Highs: {'Yes' if trend_context['higher_highs'] else 'No'}
- Lower Lows: {'Yes' if trend_context['lower_lows'] else 'No'}
- Volume Confirms: {'Yes' if trend_context['volume_confirms_trend'] else 'No'}

Momentum Indicators:
RSI (14): {indicators['rsi']:.2f}
Stochastic: K={indicators['stoch_k']:.2f}, D={indicators['stoch_d']:.2f}
Williams %R: {indicators['williams_r']:.2f}
Rate of Change: {indicators['roc']:.2f}
Awesome Oscillator: {indicators['ao']:.2f}
MACD: {indicators['macd']:.2f} (Signal: {indicators['macd_signal']:.2f}, Hist: {indicators['macd_hist']:.2f})

Volume Analysis:
Current Volume: {indicators['current_volume']:.2f}
Volume vs Average: {indicators['volume_ratio']:.2f}x
Money Flow Index: {indicators['mfi']:.2f}
On-Balance Volume: {indicators['obv']:.2f}
Accumulation/Distribution: {indicators['acc_dist']:.2f}
Chaikin Money Flow: {indicators['cmf']:.2f}
Ease of Movement: {indicators['eom']:.2f}
Force Index: {indicators['force_index']:.2f}
Volume-Price Trend: {indicators['vpt']:.2f}

Volatility & Channels:
ATR: ${indicators['atr']:.2f}
Bollinger Bands:
- Upper: ${indicators['bollinger_upper']:.2f}
- Middle: ${indicators['bollinger_middle']:.2f}
- Lower: ${indicators['bollinger_lower']:.2f}
Keltner Channel:
- Upper: ${indicators['keltner_high']:.2f}
- Middle: ${indicators['keltner_mid']:.2f}
- Lower: ${indicators['keltner_low']:.2f}
Donchian Channel:
- Upper: ${indicators['donchian_high']:.2f}
- Middle: ${indicators['donchian_mid']:.2f}
- Lower: ${indicators['donchian_low']:.2f}

Ichimoku Cloud:
- Conversion Line: ${indicators['ichimoku_conv']:.2f}
- Base Line: ${indicators['ichimoku_base']:.2f}
- Leading Span A: ${indicators['ichimoku_a']:.2f}
- Leading Span B: ${indicators['ichimoku_b']:.2f}

Candlestick Patterns:
{'🔔 Doji' if indicators['doji'] else ''}
{'🔔 Hammer' if indicators['hammer'] else ''}
{'🔔 Shooting Star' if indicators['shooting_star'] else ''}
{'🔔 Bullish Engulfing' if indicators['bullish_engulfing'] else ''}
{'🔔 Bearish Engulfing' if indicators['bearish_engulfing'] else ''}

Price Patterns:
{'🔔 Double Top' if indicators['double_top'] else ''}
{'🔔 Double Bottom' if indicators['double_bottom'] else ''}
{'🔔 Price Breakout' if indicators['price_breakout'] else ''}
{'🔔 Price Breakdown' if indicators['price_breakdown'] else ''}
{'🔔 Volume-Confirmed Breakout' if indicators['volume_breakout'] else ''}

Divergences:
{'🔔 Regular Bullish Divergence' if divergences['bullish']['regular'] else ''}
{'🔔 Hidden Bullish Divergence' if divergences['bullish']['hidden'] else ''}
{'🔔 Regular Bearish Divergence' if divergences['bearish']['regular'] else ''}
{'🔔 Hidden Bearish Divergence' if divergences['bearish']['hidden'] else ''}

Based on this comprehensive technical data, provide an opinionated analysis that:
1. Interprets the current market structure and what it suggests about market psychology
2. Highlights the most significant patterns/signals and explains WHY they matter now
3. Identifies potential trade setups or areas of interest, with clear reasoning
4. Discusses potential risks and invalidation points
5. Suggests specific price levels or conditions to watch for confirmation/invalidation

Be direct and opinionated - if you see a compelling setup or concerning pattern, say so clearly. Focus on the most meaningful signals rather than listing everything. Explain WHY certain patterns or indicators matter in the current context, and how they interact with each other to form a complete picture.

Keep the analysis as concise as possible, focusing on actionable intelligence rather than just describing indicators. If there are conflicting signals, explain which ones you believe are more reliable in the current context and why."""

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a veteran technical analyst known for providing bold, insightful market analysis. You focus on identifying meaningful patterns and market psychology rather than just reciting indicators. Your analysis should be opinionated and actionable, backed by clear technical reasoning. When you see multiple confirming signals, you should emphasize their significance. When signals conflict, you should explain which ones you trust more and why."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating AI summary: {e}")
        return "Error generating analysis summary."

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

if __name__ == "__main__":
    import sys
    
    # Use command line argument if provided, otherwise default to bitcoin
    token_id = sys.argv[1] if len(sys.argv) > 1 else "bitcoin"
    run_analysis(token_id) 
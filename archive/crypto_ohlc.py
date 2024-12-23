import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CryptoOHLCFetcher:
    def __init__(self):
        self.api_key = os.getenv('COINGECKO_API_KEY')
        self.base_url = 'https://api.coingecko.com/api/v3'
        self.headers = {'x-cg-demo-api-key': self.api_key}

    def fetch_ohlc(self, coin_id, days, vs_currency='usd'):
        """
        Fetch OHLC data from CoinGecko API.
        
        Supported timeframes:
        - 1 day: 30-minute interval data
        - 7 days: 4-hour interval data
        - 14 days: 4-hour interval data
        - 30 days: 4-hour interval data
        - 90 days: 1-day interval data
        - 180 days: 1-day interval data
        - 365 days: 1-day interval data
        - max: 1-day interval data
        """
        # Validate days parameter
        valid_days = ['1', '7', '14', '30', '90', '180', '365', 'max']
        if str(days) not in valid_days:
            raise ValueError(f"Days parameter must be one of {valid_days}")

        # Determine expected interval based on timeframe
        if days == '1':
            expected_interval = timedelta(minutes=30)
            expected_periods = 48
        elif days in ['7', '14', '30']:
            expected_interval = timedelta(hours=4)
            expected_periods = int(int(days) * 6)  # 6 4-hour periods per day
        else:
            expected_interval = timedelta(days=1)
            expected_periods = int(days) if days != 'max' else None

        try:
            # Fetch OHLC data
            print(f"\nFetching {days} days of OHLC data for {coin_id}...")
            ohlc_df = self._fetch_ohlc_data(coin_id, days, vs_currency)
            if ohlc_df is None:
                return None

            # Fetch volume data
            print(f"\nFetching volume data for {coin_id}...")
            volume_df = self._fetch_volume_data(coin_id, days, vs_currency)
            if volume_df is None:
                return None

            # Align and validate the data
            df = self._align_and_validate_data(ohlc_df, volume_df, expected_interval, expected_periods)
            if df is not None:
                self._analyze_data(df, days)
            return df

        except Exception as e:
            print(f"Error in fetch_ohlc: {str(e)}")
            return None

    def _fetch_ohlc_data(self, coin_id, days, vs_currency):
        """Fetch OHLC data from the /coins/{id}/ohlc endpoint"""
        url = f"{self.base_url}/coins/{coin_id}/ohlc"
        params = {
            'vs_currency': vs_currency,
            'days': str(days)
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
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

    def _fetch_volume_data(self, coin_id, days, vs_currency):
        """Fetch volume data from the /coins/{id}/market_chart endpoint"""
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': str(days),
            'interval': 'daily' if days not in ['1', '7', '14', '30'] else None
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
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

    def _align_and_validate_data(self, ohlc_df, volume_df, expected_interval, expected_periods):
        """Align and validate OHLC and volume data"""
        try:
            if ohlc_df is None or volume_df is None:
                return None

            # Check for expected number of periods
            if expected_periods and len(ohlc_df) < expected_periods * 0.9:  # Allow 10% missing data
                print(f"Warning: Insufficient data points. Expected {expected_periods}, got {len(ohlc_df)}")

            # Resample volume data to match OHLC intervals
            if expected_interval == timedelta(minutes=30):
                volume_df = volume_df.resample('30T').sum()
            elif expected_interval == timedelta(hours=4):
                volume_df = volume_df.resample('4H').sum()
            else:
                volume_df = volume_df.resample('D').sum()

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
            self._validate_data_ranges(df)

            return df

        except Exception as e:
            print(f"Error aligning and validating data: {str(e)}")
            return None

    def _validate_data_ranges(self, df):
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

    def _analyze_data(self, df, days):
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

def main():
    fetcher = CryptoOHLCFetcher()
    
    # Test different timeframes
    timeframes = ['1', '7', '30']
    for days in timeframes:
        print(f"\n=== Testing {days} day{'s' if days != '1' else ''} of data ===")
        df = fetcher.fetch_ohlc('bitcoin', days)
        if df is not None:
            print(f"\nSuccessfully fetched and validated data for {days} day(s)")

if __name__ == "__main__":
    main() 
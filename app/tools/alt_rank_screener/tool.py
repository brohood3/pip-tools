"""
Alt Rank Screener Tool

A specialized cryptocurrency screener that focuses on alt rank movements and price action:
1) Identifies coins with significant alt rank improvements (decrease of 100+ positions)
2) Filters for specific price action (5-15% increase in 24h)
3) Requires minimum market cap ($5M) and 24h volume ($500K) for liquidity
4) Validates against 30-day moving average for top 5 coins by Galaxy Score
5) Filters out coins with high alt rank volatility (std dev > 1000)
6) Sorts final results by Galaxy Score
"""

import os
import requests
import sqlite3
from typing import List, Dict, Any
from datetime import datetime, timedelta
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AltRankScreener:
    """Cryptocurrency screener focused on alt rank movements."""

    def __init__(self, use_historical: bool = False, historical_date: str = None, db_path: str = "historical_data.db"):
        """Initialize the screener with API configuration or database settings.
        
        Args:
            use_historical (bool): If True, use historical data from database instead of API
            historical_date (str): Date to analyze in YYYY-MM-DD format (required if use_historical is True)
            db_path (str): Path to the SQLite database file
        """
        self.use_historical = use_historical
        self.db_path = db_path
        
        if use_historical:
            if not historical_date:
                raise ValueError("historical_date is required when use_historical is True")
            try:
                self.historical_date = datetime.strptime(historical_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("historical_date must be in YYYY-MM-DD format")
        else:
            self.api_key = os.getenv("LUNARCRUSH_API_KEY")
            if not self.api_key:
                raise ValueError("LUNARCRUSH_API_KEY environment variable is not set")
            self.base_url = "https://lunarcrush.com/api4"

    def _fetch_coins_list(self) -> dict:
        """Fetch current snapshot data for all coins."""
        if self.use_historical:
            print(f"\n🔍 Fetching historical coin data for {self.historical_date.date()}...")
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the Unix timestamp for the target date
            target_timestamp = int(self.historical_date.timestamp())
            
            # Query to get all coins data for the specific date
            query = """
            WITH RankedData AS (
                -- Get data for each coin on the target date
                SELECT 
                    coin_id as id,  -- Alias coin_id as id to match API format
                    symbol,
                    symbol as name,  -- Use symbol as name since we don't store names
                    time,
                    open,
                    close,
                    high,
                    low,
                    volume_24h,
                    market_cap,
                    market_dominance,
                    circulating_supply,
                    sentiment,
                    spam,
                    galaxy_score,
                    volatility,
                    alt_rank,
                    contributors_active,
                    contributors_created,
                    posts_active,
                    posts_created,
                    interactions,
                    social_dominance,
                    LAG(alt_rank) OVER (PARTITION BY symbol ORDER BY time) as alt_rank_previous,
                    -- Calculate 24h percent change using previous day's close
                    100 * (close - LAG(close) OVER (PARTITION BY symbol ORDER BY time)) / 
                          NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY time), 0) as percent_change_24h,
                    -- Calculate 7d percent change using 7-day previous close
                    100 * (close - LAG(close, 7) OVER (PARTITION BY symbol ORDER BY time)) / 
                          NULLIF(LAG(close, 7) OVER (PARTITION BY symbol ORDER BY time), 0) as percent_change_7d
                FROM coin_history
                WHERE time <= ?
                ORDER BY time DESC
            )
            SELECT *
            FROM RankedData
            WHERE time = (
                SELECT MAX(time)
                FROM coin_history
                WHERE time <= ?
            )
            """
            
            cursor.execute(query, (target_timestamp, target_timestamp))
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            data = []
            for row in rows:
                coin_data = dict(zip(columns, row))
                data.append(coin_data)
            
            conn.close()
            
            print(f"✅ Successfully fetched historical data for {len(data)} coins")
            return {"data": data}
            
        else:
            print("\n🔍 Fetching coin list from LunarCrush API...")
            url = f"{self.base_url}/public/coins/list/v2"
            params = {
                "sort": "market_cap_rank",
                "limit": 100,
                "page": 0,
                "desc": "true"
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            print(f"✅ Successfully fetched data for {len(data.get('data', []))} coins")
            return data

    def _initial_filter(self, coins_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply initial filtering conditions:
        - Alt rank decrease >= 100
        - Price increase between 5-15% in 24h
        - Market cap > $5M
        - 24h volume > $500K
        """
        print("\n🔍 Applying initial filters...")
        print(f"Starting with {len(coins_data)} coins")
        
        filtered_coins = []
        alt_rank_passed = 0
        price_passed = 0
        market_cap_passed = 0
        volume_passed = 0
        
        # Define minimum thresholds
        MIN_MARKET_CAP = 5_000_000  # $5M
        MIN_VOLUME_24H = 500_000    # $500K
        
        for coin in coins_data:
            # Get current and previous alt ranks
            current_alt_rank = coin.get('alt_rank')
            previous_alt_rank = coin.get('alt_rank_previous')
            
            # Get price and market metrics
            price_change_24h = coin.get('percent_change_24h', 0)
            market_cap = coin.get('market_cap', 0)
            volume_24h = coin.get('volume_24h', 0)
            
            # Skip if we're missing any required data
            if any(x is None for x in [current_alt_rank, previous_alt_rank, market_cap, volume_24h]):
                continue
            
            # Convert to appropriate types (database might return strings)
            try:
                current_alt_rank = float(current_alt_rank)
                previous_alt_rank = float(previous_alt_rank)
                price_change_24h = float(price_change_24h) if price_change_24h is not None else 0
                market_cap = float(market_cap)
                volume_24h = float(volume_24h)
            except (ValueError, TypeError):
                continue
            
            # Calculate alt rank improvement (decrease in rank number)
            alt_rank_improvement = previous_alt_rank - current_alt_rank
            
            # Check conditions separately for better logging
            alt_rank_condition = alt_rank_improvement >= 100
            price_condition = 5 <= price_change_24h <= 15
            market_cap_condition = market_cap >= MIN_MARKET_CAP
            volume_condition = volume_24h >= MIN_VOLUME_24H
            
            # Update counters for statistics
            if alt_rank_condition:
                alt_rank_passed += 1
            if price_condition:
                price_passed += 1
            if market_cap_condition:
                market_cap_passed += 1
            if volume_condition:
                volume_passed += 1
            
            # Check if all conditions are met
            if all([alt_rank_condition, price_condition, market_cap_condition, volume_condition]):
                filtered_coins.append(coin)
                print(f"\n✨ Found matching coin: {coin['symbol']}")
                print(f"   Alt Rank: {current_alt_rank} (improved by {alt_rank_improvement} positions)")
                print(f"   24h Price Change: {price_change_24h:+.2f}%")
                print(f"   Market Cap: ${market_cap:,.0f}")
                print(f"   24h Volume: ${volume_24h:,.0f}")
                print(f"   Galaxy Score: {coin.get('galaxy_score', '?')}")
        
        # Sort by Galaxy Score and take top 5
        filtered_coins.sort(key=lambda x: float(x.get('galaxy_score', 0) or 0), reverse=True)
        top_coins = filtered_coins[:5]
        
        print(f"\n📊 Filtering Statistics:")
        print(f"   - Total coins analyzed: {len(coins_data)}")
        print(f"   - Passed alt rank filter (≥100 improvement): {alt_rank_passed}")
        print(f"   - Passed price filter (5-15% increase): {price_passed}")
        print(f"   - Passed market cap filter (>${MIN_MARKET_CAP:,}): {market_cap_passed}")
        print(f"   - Passed volume filter (>${MIN_VOLUME_24H:,}): {volume_passed}")
        print(f"   - Passed all filters: {len(filtered_coins)}")
        print(f"   - Selected for analysis (top 5 by Galaxy Score): {len(top_coins)}")
        
        if top_coins:
            print("\n🌟 Top 5 Coins Selected for Analysis:")
            for idx, coin in enumerate(top_coins, 1):
                print(f"   {idx}. {coin['symbol']}")
                print(f"      Galaxy Score: {coin.get('galaxy_score', '?')}")
                print(f"      Alt Rank: {coin.get('alt_rank', '?')} (improved by {float(coin.get('alt_rank_previous', 0)) - float(coin.get('alt_rank', 0))} positions)")
                print(f"      Market Cap: ${float(coin.get('market_cap', 0)):,.0f}")
                print(f"      24h Volume: ${float(coin.get('volume_24h', 0)):,.0f}")
        
        return top_coins

    def _fetch_coin_time_series(self, coin_id: str) -> dict:
        """Fetch 30-day time series data for a specific coin."""
        if self.use_historical:
            # Calculate date range for historical analysis
            end_date = self.historical_date
            start_date = end_date - timedelta(days=30)
            
            # Convert dates to Unix timestamps
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            # Query database for historical time series
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
            SELECT 
                time,
                open,
                close,
                high,
                low,
                volume_24h,
                market_cap,
                market_dominance,
                circulating_supply,
                sentiment,
                spam,
                galaxy_score,
                volatility,
                alt_rank,
                contributors_active,
                contributors_created,
                posts_active,
                posts_created,
                interactions,
                social_dominance
            FROM coin_history
            WHERE coin_id = ?
            AND time BETWEEN ? AND ?
            ORDER BY time ASC
            """
            
            cursor.execute(query, (coin_id, start_timestamp, end_timestamp))
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            data = []
            for row in rows:
                entry = dict(zip(columns, row))
                data.append(entry)
            
            conn.close()
            return {"data": data}
            
        else:
            url = f"{self.base_url}/public/coins/{coin_id}/time-series/v2"
            params = {
                "interval": "1m",
                "bucket": "day"
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()

    def _analyze_moving_average(self, coin: Dict[str, Any]) -> bool:
        """
        Analyze if coin's current alt rank is better than its 30-day moving average.
        Also checks for alt rank volatility using standard deviation.
        Returns True if coin should be kept, False if it should be filtered out.
        """
        try:
            print(f"\n📈 Analyzing {coin['symbol']} moving average...")
            
            # Fetch time series data
            time_series = self._fetch_coin_time_series(coin['id'])
            data = time_series.get('data', [])
            
            if len(data) < 30:
                print(f"❌ Insufficient data: only {len(data)} days available (need 30)")
                return False
            
            # Extract alt ranks for the last 30 days, filtering out None and invalid values
            alt_ranks = []
            print("\n   Historical Alt Ranks:")
            for i, entry in enumerate(data[-30:]):
                rank = entry.get('alt_rank')
                if rank is not None and isinstance(rank, (int, float)) and rank > 0:
                    alt_ranks.append(rank)
                    print(f"   Day {i+1}: {rank}")
                else:
                    print(f"   Day {i+1}: Invalid data")
            
            if len(alt_ranks) < 15:  # Require at least 15 days of valid data for a 30-day MA
                print(f"❌ Insufficient valid data points: only {len(alt_ranks)} valid days (need at least 15)")
                return False
            
            # Calculate volatility (standard deviation of Alt Rank)
            alt_rank_std_dev = np.std(alt_ranks)
            print(f"\n   Alt Rank Volatility (StdDev): {alt_rank_std_dev:.2f}")
            
            # Check volatility threshold
            if alt_rank_std_dev > 1000:
                print(f"❌ Skipping due to high Alt Rank volatility (threshold: 1000)")
                return False
            
            # Calculate moving average
            moving_avg = np.mean(alt_ranks)
            current_alt_rank = coin.get('alt_rank', 99999)
            
            print(f"\n   Current Alt Rank: {current_alt_rank}")
            print(f"   Valid data points: {len(alt_ranks)} days")
            print(f"   30-day Moving Average: {moving_avg:.2f}")
            
            # Current alt rank should be better (numerically lower) than moving average
            result = current_alt_rank < moving_avg
            print(f"   {'✅ Passed' if result else '❌ Failed'} moving average check")
            
            # Calculate and display additional statistics
            min_rank = min(alt_ranks)
            max_rank = max(alt_ranks)
            median_rank = np.median(alt_ranks)
            print(f"\n   Additional Statistics:")
            print(f"   - Best Rank: {min_rank}")
            print(f"   - Worst Rank: {max_rank}")
            print(f"   - Median Rank: {median_rank:.0f}")
            print(f"   - Current vs Best: {'+' if current_alt_rank > min_rank else ''}{current_alt_rank - min_rank:.0f} positions")
            print(f"   - Current vs Worst: {'+' if current_alt_rank > max_rank else ''}{current_alt_rank - max_rank:.0f} positions")
            print(f"   - Rank Volatility: {alt_rank_std_dev:.2f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error analyzing {coin['symbol']}: {str(e)}")
            return False

    def _format_opportunities(self, coins: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format the opportunities in a structured way for the order creator.
        
        Returns:
            List of opportunities with standardized fields for order creation.
        """
        opportunities = []
        
        for coin in coins:
            # Ensure all numeric values are properly converted
            try:
                opportunity = {
                    "symbol": coin["symbol"],
                    "entry": {
                        "price": float(coin["close"]),
                        "timestamp": int(coin["time"])
                    },
                    "metrics": {
                        "galaxy_score": float(coin.get("galaxy_score", 0)),
                        "alt_rank": {
                            "current": float(coin.get("alt_rank", 0)),
                            "previous": float(coin.get("alt_rank_previous", 0)),
                            "improvement": float(coin.get("alt_rank_previous", 0)) - float(coin.get("alt_rank", 0))
                        },
                        "market_cap": float(coin.get("market_cap", 0)),
                        "volume_24h": float(coin.get("volume_24h", 0)),
                        "volatility": float(coin.get("volatility", 0)),
                        "price_changes": {
                            "24h": float(coin.get("percent_change_24h", 0)),
                            "7d": float(coin.get("percent_change_7d", 0))
                        }
                    },
                    "technical": {
                        "high_24h": float(coin.get("high", 0)),
                        "low_24h": float(coin.get("low", 0)),
                        "open": float(coin.get("open", 0)),
                        "close": float(coin.get("close", 0))
                    },
                    "social": {
                        "sentiment": float(coin.get("sentiment", 0)),
                        "social_dominance": float(coin.get("social_dominance", 0)),
                        "social_contributors": {
                            "active": int(coin.get("contributors_active", 0)),
                            "new": int(coin.get("contributors_created", 0))
                        },
                        "social_posts": {
                            "active": int(coin.get("posts_active", 0)),
                            "new": int(coin.get("posts_created", 0))
                        },
                        "interactions": int(coin.get("interactions", 0))
                    }
                }
                opportunities.append(opportunity)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not process {coin.get('symbol', 'UNKNOWN')}: {str(e)}")
                continue
        
        return opportunities

    def run(self) -> Dict[str, Any]:
        """Execute the screening process and return structured results."""
        try:
            print("\n🚀 Starting Alt Rank Screener...")
            print("=" * 50)
            
            # Step 1: Get initial coin data
            coins_data = self._fetch_coins_list()
            all_coins = coins_data.get('data', [])
            
            # Step 2: Apply initial filtering and get top 5 by Galaxy Score
            filtered_coins = self._initial_filter(all_coins)
            
            # Step 3: Apply moving average analysis to filtered coins
            print("\n📊 Analyzing moving averages...")
            print(f"Checking {len(filtered_coins)} top coins for moving average criteria")
            
            final_coins = []
            for coin in filtered_coins:
                if self._analyze_moving_average(coin):
                    final_coins.append(coin)
            
            # Step 4: Sort by Galaxy Score
            final_coins.sort(key=lambda x: float(x.get('galaxy_score', 0) or 0), reverse=True)
            
            # Step 5: Format opportunities for order creator
            opportunities = self._format_opportunities(final_coins)
            
            # Generate summary for display
            summary = self._generate_summary(final_coins)
            
            print("\n📈 Final Results:")
            print("=" * 50)
            print(f"Initial coins analyzed: {len(all_coins)}")
            print(f"Passed initial filters: {len(filtered_coins)}")
            print(f"Passed all criteria: {len(final_coins)}")
            print("=" * 50)
            
            return {
                "opportunities": opportunities,
                "summary": summary,
                "metadata": {
                    "coins_analyzed": len(all_coins),
                    "passed_initial_filter": len(filtered_coins),
                    "final_opportunities": len(final_coins),
                    "timestamp": datetime.now().isoformat(),
                    "source": "historical" if self.use_historical else "live",
                    "date": self.historical_date.date().isoformat() if self.use_historical else datetime.now().date().isoformat()
                }
            }

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            return {
                "opportunities": [],
                "summary": f"Error: {str(e)}",
                "metadata": {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }

    def _generate_summary(self, coins: List[Dict[str, Any]]) -> str:
        """Generate a readable summary of the screening results."""
        if not coins:
            return "No coins matched the screening criteria."

        summary = "Found the following opportunities:\n\n"
        for idx, coin in enumerate(coins, 1):
            summary += f"{idx}. {coin['name']} ({coin['symbol']}):\n"
            summary += f"   - Alt Rank: {coin.get('alt_rank', '?')} (prev: {coin.get('alt_rank_previous', '?')})\n"
            summary += f"   - Galaxy Score: {coin.get('galaxy_score', '?')}\n"
            summary += f"   - 24h Price Change: {coin.get('percent_change_24h', 0):+.2f}%\n"
            summary += f"   - Market Cap: ${coin.get('market_cap', 0):,.0f}\n"
            summary += f"   - 24h Volume: ${coin.get('volume_24h', 0):,.0f}\n"
            summary += f"   - 7d Price Change: {coin.get('percent_change_7d', 0):+.2f}%\n\n"

        return summary

def run(use_historical: bool = False, historical_date: str = None, db_path: str = "historical_data.db") -> Dict[str, Any]:
    """Entry point for the screener.
    
    Args:
        use_historical (bool): If True, use historical data from database
        historical_date (str): Date to analyze in YYYY-MM-DD format (required if use_historical is True)
        db_path (str): Path to the SQLite database file
    """
    return AltRankScreener(use_historical=use_historical, 
                          historical_date=historical_date, 
                          db_path=db_path).run()

if __name__ == "__main__":
    # Run with historical data for Feb 1st 2025
    result = run(use_historical=True, historical_date="2025-02-01")
    
    # Print summary
    print("\n🎯 Historical Opportunities (2025-02-01):")
    print("=" * 50)
    print(result["summary"])
    
    # Print structured opportunities data
    print("\n📊 Structured Opportunities Data:")
    print("=" * 50)
    import json
    print(json.dumps(result["opportunities"], indent=2)) 
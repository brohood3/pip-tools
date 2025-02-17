import os
import time
import sqlite3
import requests
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables (API key, etc.)
load_dotenv()

API_KEY = os.getenv("LUNARCRUSH_API_KEY")  # Replace with your actual key if not using .env
BASE_URL = "https://lunarcrush.com/api4"

def create_db_and_table(db_name: str = "historical_data.db") -> sqlite3.Connection:
    """
    Creates (or opens) a SQLite database and a table to store detailed time-series data.
    Returns the database connection object.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS coin_history (
            coin_id TEXT,
            symbol TEXT,
            time INTEGER,               -- We'll store Unix time (in seconds) as INTEGER
            open REAL,
            close REAL,
            high REAL,
            low REAL,
            volume_24h REAL,
            market_cap REAL,
            market_dominance REAL,
            circulating_supply REAL,
            sentiment REAL,
            spam INTEGER,
            galaxy_score REAL,
            volatility REAL,
            alt_rank INTEGER,
            contributors_active INTEGER,
            contributors_created INTEGER,
            posts_active INTEGER,
            posts_created INTEGER,
            interactions INTEGER,
            social_dominance REAL,
            PRIMARY KEY (coin_id, time)
        )
    """)

    conn.commit()
    return conn

def fetch_historical_data(coin_id: str, days: int = 90) -> List[Dict]:
    """
    Fetch daily historical data for the given coin_id (over 'days' days),
    returning a list of data points from LunarCrush's time-series endpoint.
    Each element in the list corresponds to one day's data.
    """
    url = f"{BASE_URL}/public/coins/{coin_id}/time-series/v2"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    # For 90 days of daily data, we use 3m interval
    params = {
        "interval": "3m",  # This gets us enough data to cover 90 days
        "bucket": "day"    # This ensures each data point represents a day
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json().get("data", [])
        
        # Filter to get exactly the last 90 days
        if data:
            data = data[-days:]
            
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for coin_id={coin_id}: {e}")
        return []

def insert_time_series_data(
    conn: sqlite3.Connection, 
    coin_id: str, 
    symbol: str, 
    time_series: List[Dict]
):
    """
    Insert or replace the time-series data for a single coin into the SQLite database.
    """
    cursor = conn.cursor()
    
    for entry in time_series:
        timestamp = entry.get("time", None)                # Integer seconds
        open_price = entry.get("open", None)
        close_price = entry.get("close", None)
        high_price = entry.get("high", None)
        low_price = entry.get("low", None)
        volume_24h = entry.get("volume_24h", None)
        market_cap = entry.get("market_cap", None)
        market_dominance = entry.get("market_dominance", None)
        circulating_supply = entry.get("circulating_supply", None)
        sentiment = entry.get("sentiment", None)
        spam = entry.get("spam", None)
        galaxy_score = entry.get("galaxy_score", None)
        volatility = entry.get("volatility", None)
        alt_rank = entry.get("alt_rank", None)
        contributors_active = entry.get("contributors_active", None)
        contributors_created = entry.get("contributors_created", None)
        posts_active = entry.get("posts_active", None)
        posts_created = entry.get("posts_created", None)
        interactions = entry.get("interactions", None)
        social_dominance = entry.get("social_dominance", None)

        cursor.execute("""
            INSERT OR REPLACE INTO coin_history (
                coin_id, symbol, time,
                open, close, high, low,
                volume_24h, market_cap,
                market_dominance, circulating_supply,
                sentiment, spam,
                galaxy_score, volatility, alt_rank,
                contributors_active, contributors_created,
                posts_active, posts_created,
                interactions, social_dominance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            coin_id, symbol, timestamp,
            open_price, close_price, high_price, low_price,
            volume_24h, market_cap,
            market_dominance, circulating_supply,
            sentiment, spam,
            galaxy_score, volatility, alt_rank,
            contributors_active, contributors_created,
            posts_active, posts_created,
            interactions, social_dominance
        ))
    
    conn.commit()

def get_top_coins(limit: int = 5) -> List[Dict]:
    """
    Fetch the top 'limit' coins by market cap from LunarCrush.
    Handles pagination to get the full requested number of coins.
    """
    print(f"Fetching top {limit} coins by market cap...")
    
    all_coins = []
    page = 0  # API uses 0-based pagination
    page_size = 100  # LunarCrush's max page size
    
    while len(all_coins) < limit:
        # Construct URL with pagination
        url = f"{BASE_URL}/public/coins/list/v2"
        params = {
            "sort": "market_cap_rank",
            "limit": page_size,
            "page": page # Get highest market cap first
        }
        
        # Set up headers
        headers = {
            "Authorization": f"Bearer {API_KEY}"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            # Parse JSON response
            result = response.json()
            data = result.get("data", [])
            
            if not data:  # No more data available
                break
                
            all_coins.extend(data)
            page += 1
            
            print(f"Fetched page {page}, total coins so far: {len(all_coins)}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {str(e)}")
            break
    
    # Trim to requested limit
    all_coins = all_coins[:limit]
    
    # Print verification info
    if all_coins:
        print("\nVerifying top coins (showing first 5):")
        for coin in all_coins[:5]:  # Show first 5 for verification
            print(f"- {coin.get('symbol')}: #{coin.get('market_cap_rank')} | Market Cap ${coin.get('market_cap', 0):,.2f}")
        print(f"... and {len(all_coins)-5} more coins")
    else:
        print("No coins returned from API")
        
    return all_coins

def build_historical_database(
    db_name: str = "historical_data.db", 
    coin_limit: int = 1000,  # Changed default to 1000
    days: int = 90
):
    """
    Build a local SQLite database containing all of the fields from the time-series endpoint
    for 'coin_limit' coins, each with 'days' worth of daily data.
    """
    # 1) Create / open SQLite database
    conn = create_db_and_table(db_name)
    
    # 2) Get top 'coin_limit' coins
    coins = get_top_coins(limit=coin_limit)
    total_coins = len(coins)
    
    if not coins:
        print("❌ No coins fetched. Exiting.")
        return
    
    print(f"\n🚀 Starting data collection for {total_coins} coins...")
    requests_this_minute = 0
    successful_fetches = 0
    failed_fetches = 0
    
    # 3) For each coin, fetch daily data and store
    for i, coin in enumerate(coins, start=1):
        coin_id = coin.get("id")
        symbol = coin.get("symbol", "UNKNOWN")
        market_cap = coin.get("market_cap", 0)

        print(f"\n[{i}/{total_coins}] Processing {symbol} (#{coin.get('market_cap_rank')})")
        print(f"   Market Cap: ${market_cap:,.2f}")
        
        try:
            time_series = fetch_historical_data(coin_id, days=days)
            if time_series:
                insert_time_series_data(conn, coin_id, symbol, time_series)
                successful_fetches += 1
                print(f"   ✅ Successfully fetched {len(time_series)} days of data")
            else:
                failed_fetches += 1
                print(f"   ❌ Failed to fetch data")
        
            # Rate limit: after every 10 requests, sleep for 61 seconds
            requests_this_minute += 1
            if requests_this_minute >= 10:
                print(f"\n⏳ Rate limit reached. Sleeping for 61 seconds...")
                print(f"   Progress: {i}/{total_coins} coins processed")
                print(f"   Success: {successful_fetches}, Failed: {failed_fetches}")
                time.sleep(61)
                requests_this_minute = 0
                
        except Exception as e:
            failed_fetches += 1
            print(f"   ❌ Error processing {symbol}: {str(e)}")
    
    conn.close()
    print("\n✅ Finished building historical database")
    print(f"Total coins processed: {total_coins}")
    print(f"Successful fetches: {successful_fetches}")
    print(f"Failed fetches: {failed_fetches}")

def verify_database(db_name: str = "historical_data.db"):
    """
    Verify the contents of the database by printing summary statistics
    for each coin's historical data.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    print("\n📊 Database Verification:")
    print("=" * 50)
    
    # Get unique coins
    cursor.execute("SELECT DISTINCT symbol FROM coin_history ORDER BY symbol")
    coins = cursor.fetchall()
    
    for coin_symbol in coins:
        symbol = coin_symbol[0]
        # Get data points count and date range for each coin
        cursor.execute("""
            SELECT 
                COUNT(*) as count,
                MIN(datetime(time, 'unixepoch')) as start_date,
                MAX(datetime(time, 'unixepoch')) as end_date,
                ROUND(AVG(CASE WHEN galaxy_score IS NOT NULL THEN galaxy_score END), 2) as avg_galaxy_score,
                ROUND(AVG(CASE WHEN alt_rank IS NOT NULL THEN alt_rank END), 2) as avg_alt_rank,
                ROUND(AVG(CASE WHEN market_cap IS NOT NULL THEN market_cap END), 2) as avg_market_cap
            FROM coin_history 
            WHERE symbol = ?
        """, (symbol,))
        
        stats = cursor.fetchone()
        count, start_date, end_date, avg_galaxy_score, avg_alt_rank, avg_market_cap = stats
        
        print(f"\n🪙 {symbol}:")
        print(f"   Data points: {count}")
        print(f"   Date range: {start_date} to {end_date}")
        print(f"   Avg Galaxy Score: {avg_galaxy_score or 'N/A'}")
        print(f"   Avg Alt Rank: {avg_alt_rank or 'N/A'}")
        print(f"   Avg Market Cap: ${avg_market_cap:,.2f}" if avg_market_cap else "   Avg Market Cap: N/A")
    
    # Print overall statistics
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT symbol) as num_coins,
            COUNT(*) as total_records,
            MIN(datetime(time, 'unixepoch')) as earliest_date,
            MAX(datetime(time, 'unixepoch')) as latest_date
        FROM coin_history
    """)
    
    overall_stats = cursor.fetchone()
    num_coins, total_records, earliest_date, latest_date = overall_stats
    
    print("\n📈 Overall Statistics:")
    print("=" * 50)
    print(f"Total coins in database: {num_coins}")
    print(f"Total data points: {total_records}")
    print(f"Date range: {earliest_date} to {latest_date}")
    
    conn.close()

if __name__ == "__main__":
    # Build database with 1000 coins
    build_historical_database(
        db_name="historical_data.db",
        coin_limit=1000,  # Changed to 1000 coins
        days=90
    )
    
    # Verify the database contents
    verify_database("historical_data.db") 
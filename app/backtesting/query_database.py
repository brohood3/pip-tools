import sqlite3
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
from datetime import datetime

def query_coin_data(db_name: str, symbol: str) -> pd.DataFrame:
    """
    Query all data for a specific coin and return as a pandas DataFrame.
    """
    conn = sqlite3.connect(db_name)
    
    # Query all columns
    query = """
    SELECT 
        datetime(time, 'unixepoch') as date,
        symbol,
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
    WHERE symbol = ?
    ORDER BY time ASC
    """
    
    # Load into pandas DataFrame
    df = pd.read_sql_query(query, conn, params=(symbol,))
    
    # Convert date string to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Set date as index
    df.set_index('date', inplace=True)
    
    conn.close()
    return df

def print_coin_summary(df: pd.DataFrame):
    """Print a summary of the coin's data."""
    print("\n📊 Data Summary:")
    print("=" * 50)
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    print(f"Total Days: {len(df)}")
    
    print("\n📈 Latest Values:")
    latest = df.iloc[-1]
    print(f"Price: ${latest['price']:,.2f}")
    print(f"Market Cap: ${latest['market_cap']:,.2f}")
    print(f"Galaxy Score: {latest['galaxy_score']}")
    print(f"Alt Rank: {latest['alt_rank']}")
    print(f"Sentiment: {latest['sentiment']}")
    
    print("\n📊 Statistics:")
    print(f"Average Galaxy Score: {df['galaxy_score'].mean():.2f}")
    print(f"Average Alt Rank: {df['alt_rank'].mean():.2f}")
    print(f"Price Range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
    print(f"Volume Range: ${df['volume_24h'].min():,.2f} - ${df['volume_24h'].max():,.2f}")

def plot_metrics(df: pd.DataFrame, symbol: str):
    """Create plots for key metrics."""
    plt.style.use('seaborn')
    
    # Create a 2x2 subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Price
    ax1.plot(df.index, df['price'])
    ax1.set_title(f'{symbol} Price')
    ax1.set_ylabel('Price ($)')
    
    # Galaxy Score
    ax2.plot(df.index, df['galaxy_score'])
    ax2.set_title('Galaxy Score')
    ax2.set_ylabel('Score')
    
    # Alt Rank
    ax3.plot(df.index, df['alt_rank'])
    ax3.set_title('Alt Rank')
    ax3.set_ylabel('Rank')
    
    # Social Dominance
    ax4.plot(df.index, df['social_dominance'])
    ax4.set_title('Social Dominance')
    ax4.set_ylabel('Dominance (%)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Query SOL data
    df = query_coin_data("historical_data.db", "SOL")
    
    # Show all columns and their non-null counts
    print("\n📋 Available Columns:")
    print("=" * 50)
    print(df.info())
    
    print("\n📊 Sample Data (last 2 days, all columns):")
    print("=" * 50)
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)        # Don't wrap
    pd.set_option('display.max_rows', None)     # Show all rows
    print(df.tail(2)) 
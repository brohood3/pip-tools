"""
Backtesting Orchestrator

Coordinates the execution of:
1. Alt Rank Screener (historical mode)
2. Order Generator
3. Portfolio Manager

Using historical data to simulate trading performance.
"""

from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Any, Optional
import json
import os
import sys

# Add the app directory to the Python path
app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if app_dir not in sys.path:
    sys.path.append(app_dir)

from app.tools.alt_rank_screener.tool import run as run_screener
from app.tools.limit_order_generator.tool import run as generate_orders
from .portfolio import Portfolio, DateTimeEncoder

class Backtester:
    def __init__(self, 
                 start_date: str,
                 end_date: str,
                 initial_balance: float = 10000,
                 db_path: str = "app/backtesting/historical_data.db"):
        """
        Initialize backtester with date range and configuration.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            initial_balance: Starting portfolio balance
            db_path: Path to historical database
        """
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.db_path = db_path
        self.portfolio = Portfolio(initial_balance=initial_balance)
        self.trade_details = []  # Store detailed trade information
        
        # Validate dates
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
            
        # Connect to database
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Verify we have data for the date range
        self.cursor.execute("""
            SELECT MIN(time), MAX(time)
            FROM coin_history
        """)
        db_start, db_end = self.cursor.fetchone()
        db_start_date = datetime.fromtimestamp(db_start)
        db_end_date = datetime.fromtimestamp(db_end)
        
        if self.start_date < db_start_date or self.end_date > db_end_date:
            raise ValueError(f"Date range {start_date} to {end_date} exceeds available data range "
                           f"{db_start_date.date()} to {db_end_date.date()}")

    def _get_prices_for_date(self, date: datetime) -> Dict[str, Dict[str, float]]:
        """Get OHLC prices and other metrics for all coins on a specific date."""
        timestamp = int(date.timestamp())
        
        # Query to get all price data for the specific date
        query = """
        WITH RankedData AS (
            SELECT 
                symbol,
                open,
                close,
                high,
                low,
                volume_24h,
                market_cap,
                time,
                LAG(close) OVER (PARTITION BY symbol ORDER BY time) as prev_close,
                LAG(close, 7) OVER (PARTITION BY symbol ORDER BY time) as prev_7d_close
            FROM coin_history
            WHERE time <= ?
            ORDER BY time DESC
        )
        SELECT 
            symbol,
            open,
            close,
            high,
            low,
            volume_24h,
            market_cap,
            -- Calculate 24h percent change
            100 * (close - prev_close) / NULLIF(prev_close, 0) as percent_change_24h,
            -- Calculate 7d percent change
            100 * (close - prev_7d_close) / NULLIF(prev_7d_close, 0) as percent_change_7d
        FROM RankedData
        WHERE time = (
            SELECT MAX(time)
            FROM coin_history
            WHERE time <= ?
        )
        """
        
        self.cursor.execute(query, (timestamp, timestamp))
        columns = [description[0] for description in self.cursor.description]
        
        prices = {}
        for row in self.cursor.fetchall():
            data = dict(zip(columns, row))
            symbol = data.pop('symbol')  # Remove symbol from data dict
            prices[symbol] = data
        
        # Debug: Print prices for active positions
        active_positions = list(self.portfolio.positions.items())  # Get (trade_id, Position) pairs
        if active_positions:
            print("\n💰 Current Prices:")
            for trade_id, position in active_positions:
                symbol = position.symbol  # Get the actual symbol from the position
                if symbol in prices:
                    data = prices[symbol]
                    print(f"   {symbol}:")
                    print(f"      Entry: ${position.entry['price']:.8f}")
                    print(f"      Current: ${data['close']:.8f}")
                    print(f"      High: ${data['high']:.8f}")
                    print(f"      Low: ${data['low']:.8f}")
                    print(f"      24h Change: {data['percent_change_24h']:+.2f}%")
                    print(f"      Size Remaining: {position.state.size_remaining:.2%}")
                else:
                    print(f"   ⚠️ No price data for symbol {symbol}")
        
        return prices

    def run(self) -> Dict[str, Any]:
        """
        Execute the backtest over the specified date range.
        
        Returns:
            Dict containing backtest results and statistics
        """
        print(f"\n🚀 Starting backtest from {self.start_date.date()} to {self.end_date.date()}")
        print("=" * 50)
        
        current_date = self.start_date
        daily_stats = []
        
        while current_date <= self.end_date:
            print(f"\n📅 Processing {current_date.date()}")
            
            # 1. Run screener on historical data
            screener_results = run_screener(
                use_historical=True,
                historical_date=current_date.strftime("%Y-%m-%d"),
                db_path=self.db_path
            )
            
            opportunities = screener_results.get("opportunities", [])
            if opportunities:
                print(f"✨ Found {len(opportunities)} opportunities")
                
                # 2. Generate orders for opportunities
                orders_result = generate_orders(opportunities)
                orders = orders_result.get("orders", [])
                
                # Debug: Print order details
                for order in orders:
                    print(f"\n📋 New Order for {order['symbol']}:")
                    print(f"   Entry: ${order['entry']['price']:.8f}")
                    print(f"   Stop Loss: ${order['orders']['stop_loss']['price']:.8f} ({order['orders']['stop_loss']['percentage']}%)")
                    for idx, tp in enumerate(order['orders']['take_profit'], 1):
                        print(f"   TP{idx}: ${tp['price']:.8f} ({tp['percentage']}%, size: {tp['size']:.1%})")
                    
                    # Open position and get trade ID from portfolio
                    returned_trade_id = self.portfolio.open_position(order, current_date)
                    if returned_trade_id:
                        print(f"✅ Opened position for {order['symbol']} with trade_id {returned_trade_id}")
                        
                        # Calculate position size and amount
                        position_size = 0.01  # 1% position size
                        position_amount = self.portfolio.current_balance * position_size
                        
                        # Store trade details with the portfolio-assigned trade ID
                        self.trade_details.append({
                            "trade_id": returned_trade_id,
                            "symbol": order['symbol'],
                            "entry": {
                                "date": current_date.strftime("%Y-%m-%d"),
                                "price": order['entry']['price'],
                                "amount": position_amount,
                                "size": position_size
                            },
                            "stop_loss": {
                                "price": order['orders']['stop_loss']['price'],
                                "percentage": order['orders']['stop_loss']['percentage']
                            },
                            "take_profits": [
                                {
                                    "level": idx + 1,
                                    "price": tp['price'],
                                    "percentage": tp['percentage'],
                                    "size": tp['size']
                                }
                                for idx, tp in enumerate(order['orders']['take_profit'])
                            ],
                            "metrics": order.get('metrics', {}),
                            "exits": [],
                            "exit_sequence": [],
                            "final_pnl": 0.0,
                            "status": "open"
                        })
                    else:
                        print(f"❌ Failed to open position for {order['symbol']}")
            
            # 4. Update existing positions with current prices
            prices = self._get_prices_for_date(current_date)
            executed_orders = self.portfolio.update_positions(prices, current_date)
            
            # Update trade details with exit information
            for order in executed_orders:
                for trade in self.trade_details:
                    if trade['trade_id'] == order['trade_id']:  # Match by trade_id instead of symbol
                        # Calculate exit amount based on the original entry amount and exit size
                        exit_amount = trade['entry']['amount'] * order['size']
                        
                        # Record exit details
                        exit_info = {
                            "date": current_date.strftime("%Y-%m-%d"),
                            "type": order['exit_type'],
                            "price": order['price'],
                            "size": order['size'],
                            "amount": exit_amount,
                            "pnl": order['pnl']
                        }
                        trade['exits'].append(exit_info)
                        trade['exit_sequence'].append(order['exit_type'])
                        trade['final_pnl'] += order['pnl']
                        
                        # Update trade status
                        if order['exit_type'] == 'stop_loss':
                            trade['status'] = 'stopped_out'
                            break
                        elif trade['status'] == 'open' and sum(exit['size'] for exit in trade['exits']) >= 0.99:  # Account for floating point
                            trade['status'] = 'closed_by_tp'
                
                if order['exit_type'] == 'stop_loss':
                    break
            
            # Debug: Print executed orders
            if executed_orders:
                print("\n🔄 Executed Orders:")
                for order in executed_orders:
                    print(f"   {order['symbol']}:")
                    print(f"      Type: {order['exit_type']}")
                    print(f"      Price: ${order['price']:.8f}")
                    print(f"      Size: {order['size']:.1%}")
                    print(f"      PnL: ${order['pnl']:.2f}")
            
            # Log daily statistics
            portfolio_stats = self.portfolio.get_portfolio_summary()
            daily_stats.append({
                "date": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                "balance": portfolio_stats["current_balance"],
                "open_positions": portfolio_stats["open_positions"],
                "total_trades": portfolio_stats["total_trades"],
                "executed_orders": executed_orders if executed_orders else []  # Add executed orders to daily stats
            })
            
            # Print daily summary with more detail
            print(f"\n📊 Daily Summary:")
            print(f"   Balance: ${portfolio_stats['current_balance']:,.2f}")
            print(f"   Open Positions: {portfolio_stats['open_positions']}")
            print(f"   Total Trades: {portfolio_stats['total_trades']}")
            if executed_orders:
                print(f"   Orders Executed: {len(executed_orders)}")
                print(f"   Daily PnL: ${sum(order['pnl'] for order in executed_orders):,.2f}")
            
            # Debug: Print unrealized PnL for open positions
            if self.portfolio.positions:
                print("\n📈 Unrealized PnL:")
                total_unrealized = 0
                for symbol, position in self.portfolio.positions.items():
                    if symbol in prices:
                        unrealized = position.get_unrealized_pnl(prices[symbol]['close'])
                        total_unrealized += unrealized
                        print(f"   {symbol}: ${unrealized:,.2f}")
                print(f"   Total Unrealized: ${total_unrealized:,.2f}")
            
            current_date += timedelta(days=1)
        
        # Get final statistics
        final_stats = self.portfolio.get_detailed_stats()
        
        # Get final prices for unrealized PnL calculation
        final_prices = self._get_prices_for_date(current_date)
        
        # Calculate trade performance metrics including open positions
        closed_trades = [t for t in self.trade_details if t['status'] != 'open']
        open_trades = [t for t in self.trade_details if t['status'] == 'open']
        
        # Calculate unrealized PnL for open trades
        for trade in open_trades:
            if trade['symbol'] in final_prices:
                current_price = final_prices[trade['symbol']]['close']
                entry_price = trade['entry']['price']
                remaining_size = 1.0 - sum(exit['size'] for exit in trade['exits'])
                unrealized_pnl = (current_price - entry_price) * remaining_size * (trade['entry']['amount'] / entry_price)
                trade['unrealized_pnl'] = unrealized_pnl
            else:
                trade['unrealized_pnl'] = 0.0
        
        # Calculate statistics including unrealized PnL
        won_trades = [t for t in closed_trades if t['final_pnl'] > 0]
        lost_trades = [t for t in closed_trades if t['final_pnl'] <= 0]
        
        total_realized_pnl = sum(t['final_pnl'] for t in closed_trades)
        total_unrealized_pnl = sum(t.get('unrealized_pnl', 0) for t in open_trades)
        
        results = {
            "statistics": {
                "initial_balance": self.portfolio.initial_balance,
                "final_balance": self.portfolio.current_balance,
                "total_profit": self.portfolio.current_balance - self.portfolio.initial_balance,
                "profit_percent": ((self.portfolio.current_balance / self.portfolio.initial_balance) - 1) * 100,
                "total_trades": len(self.trade_details),
                "won_trades": len(won_trades),
                "lost_trades": len(lost_trades),
                "open_trades": len(open_trades),
                "win_rate": len(won_trades) / len(closed_trades) if closed_trades else 0,
                "average_win": sum(t['final_pnl'] for t in won_trades) / len(won_trades) if won_trades else 0,
                "average_loss": sum(t['final_pnl'] for t in lost_trades) / len(lost_trades) if lost_trades else 0,
                "largest_win": max((t['final_pnl'] for t in won_trades), default=0),
                "largest_loss": min((t['final_pnl'] for t in lost_trades), default=0),
                "profit_factor": abs(sum(t['final_pnl'] for t in won_trades) / 
                                   sum(t['final_pnl'] for t in lost_trades))
                                   if lost_trades and sum(t['final_pnl'] for t in lost_trades) != 0 
                                   else float('inf'),
                "realized_pnl": total_realized_pnl,
                "unrealized_pnl": total_unrealized_pnl,
                "total_pnl": total_realized_pnl + total_unrealized_pnl
            },
            "trade_history": [
                {
                    "trade_id": trade['trade_id'],
                    "symbol": trade['symbol'],
                    "entry": trade['entry'],
                    "entry_amount": trade['entry']['amount'],
                    "exit_sequence": trade['exit_sequence'],
                    "exits": trade['exits'],
                    "final_pnl": trade['final_pnl'],
                    "status": trade['status'],
                    "metrics": {
                        "galaxy_score": trade['metrics'].get('galaxy_score', 0),
                        "alt_rank": trade['metrics'].get('alt_rank', {})
                    }
                }
                for trade in sorted(self.trade_details, key=lambda x: x['entry']['date'])
            ],
            "daily_balance": [
                {
                    "date": day["date"],
                    "balance": day["balance"],
                    "open_positions": day["open_positions"],
                    "total_trades": day["total_trades"],
                    "executed_orders": [
                        {
                            "trade_id": order["trade_id"],  # Add trade ID to executed orders
                            "symbol": order["symbol"],
                            "type": order["exit_type"],
                            "price": order["price"],
                            "size": order["size"],
                            "pnl": order["pnl"]
                        }
                        for order in day.get("executed_orders", [])
                    ] if "executed_orders" in day else []
                }
                for day in daily_stats
            ]
        }
        
        # Save results to file
        with open("simulation_results.json", "w") as f:
            json.dump(results, f, indent=2, cls=DateTimeEncoder)
        
        print("\n✅ Backtest completed!")
        print("=" * 50)
        print(f"Initial Balance: ${results['statistics']['initial_balance']:,.2f}")
        print(f"Final Balance: ${results['statistics']['final_balance']:,.2f}")
        print(f"Total Profit: ${results['statistics']['total_profit']:,.2f} ({results['statistics']['profit_percent']:,.2f}%)")
        print(f"Total Trades: {results['statistics']['total_trades']}")
        print(f"Won Trades: {results['statistics']['won_trades']}")
        print(f"Lost Trades: {results['statistics']['lost_trades']}")
        print(f"Open Trades: {results['statistics']['open_trades']}")
        print(f"Win Rate: {results['statistics']['win_rate']*100:.1f}%")
        print(f"Average Win: ${results['statistics']['average_win']:,.2f}")
        print(f"Average Loss: ${results['statistics']['average_loss']:,.2f}")
        print(f"Largest Win: ${results['statistics']['largest_win']:,.2f}")
        print(f"Largest Loss: ${results['statistics']['largest_loss']:,.2f}")
        print(f"Profit Factor: {results['statistics']['profit_factor']:.2f}")
        print(f"Realized PnL: ${results['statistics']['realized_pnl']:,.2f}")
        print(f"Unrealized PnL: ${results['statistics']['unrealized_pnl']:,.2f}")
        print(f"Total PnL: ${results['statistics']['total_pnl']:,.2f}")
        print("\nDetailed results saved to simulation_results.json")
        
        return results

def run_backtest(
    start_date: str,
    end_date: str,
    initial_balance: float = 10000,
    db_path: str = "app/backtesting/historical_data.db"
) -> Dict[str, Any]:
    """
    Run a backtest over the specified date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        initial_balance: Starting portfolio balance
        db_path: Path to historical database
    
    Returns:
        Dict containing backtest results and statistics
    """
    backtester = Backtester(
        start_date=start_date,
        end_date=end_date,
        initial_balance=initial_balance,
        db_path=db_path
    )
    return backtester.run()

if __name__ == "__main__":
    # Run a sample backtest for February 2025
    results = run_backtest(
        start_date="2024-12-01",
        end_date="2025-02-15",
        initial_balance=10000
    ) 
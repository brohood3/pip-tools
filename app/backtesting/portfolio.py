"""
Portfolio Management Module for Backtesting

Handles:
1. Position tracking (entry, exits, partial fills)
2. Portfolio value calculations
3. Basic performance metrics
"""

from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass
import json

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@dataclass
class PositionState:
    """Tracks the current state of a position."""
    size_remaining: float = 1.0  # Changed from 1.0 to 0.01 (1% of portfolio)
    realized_pnl: float = 0.0    # Realized profit/loss in quote currency
    cost_basis: float = 0.0      # Total cost of position
    partial_exits: List[Dict] = None  # Track each partial exit
    
    def __post_init__(self):
        self.partial_exits = [] if self.partial_exits is None else self.partial_exits

class Position:
    """Tracks a single position from entry to exit."""
    
    def __init__(self, order: Dict[str, Any], entry_date: datetime, portfolio_balance: float, trade_id: str):
        self.symbol = order["symbol"]
        self.trade_id = trade_id  # Add unique trade ID
        self.entry = order["entry"]
        self.entry_date = entry_date
        self.position_type = order["position_type"]
        self.take_profits = order["orders"]["take_profit"]
        self.stop_loss = order["orders"]["stop_loss"]
        self.initial_risk_reward = order["risk_reward_ratio"]
        self.metrics = order["metrics"]  # Store original screening metrics
        
        # Calculate position amount (1% of current portfolio balance)
        self.position_size = 0.01  # 1% position size
        self.initial_position_value = portfolio_balance * self.position_size
        self.units = self.initial_position_value / self.entry["price"]
        
        # Track position state
        self.state = PositionState()
        self.state.cost_basis = self.entry["price"]
        self.state.size_remaining = 1.0  # Start with 100% of position
        
        # Position status
        self.is_open = True
        self.exit_date = None
        self.exit_reason = None  # "tp1", "tp2", "tp3", "sl", "manual"
        
    def update(self, price_data: Dict[str, float], current_date: datetime) -> List[Dict]:
        """
        Update position state based on current price data.
        Returns list of executed orders (if any TP/SL hit).
        """
        if not self.is_open:
            return []
        
        executed_orders = []
        
        # Check if low price hit stop loss
        if price_data["low"] <= self.stop_loss["price"]:
            exit_order = self._execute_stop_loss(price_data["low"], current_date)
            executed_orders.append(exit_order)
            return executed_orders
        
        # Check if high price hit any take profits (in order)
        for idx, tp in enumerate(self.take_profits):
            if price_data["high"] >= tp["price"] and tp["size"] > 0:
                exit_order = self._execute_take_profit(tp["price"], current_date, idx)
                executed_orders.append(exit_order)
                # Don't break here - multiple TPs could be hit at once
        
        return executed_orders
    
    def _execute_stop_loss(self, price: float, date: datetime) -> Dict:
        """Execute stop loss order for remaining position."""
        # Calculate units to sell
        units_to_sell = self.units * self.state.size_remaining
        
        # Calculate actual PnL in quote currency
        entry_value = units_to_sell * self.state.cost_basis
        exit_value = units_to_sell * price
        pnl = exit_value - entry_value
        
        exit_order = {
            "symbol": self.symbol,
            "trade_id": self.trade_id,  # Include trade ID in exit order
            "exit_type": "stop_loss",
            "price": price,
            "size": self.state.size_remaining,
            "pnl": pnl,
            "date": date
        }
        
        self.state.realized_pnl += pnl
        self.state.size_remaining = 0
        self.is_open = False
        self.exit_date = date
        self.exit_reason = "sl"
        
        return exit_order
    
    def _execute_take_profit(self, price: float, date: datetime, tp_index: int) -> Dict:
        """Execute take profit order."""
        tp_level = self.take_profits[tp_index]
        
        # Calculate units to sell
        units_to_sell = self.units * tp_level["size"]
        
        # Calculate actual PnL in quote currency
        entry_value = units_to_sell * self.state.cost_basis
        exit_value = units_to_sell * price
        pnl = exit_value - entry_value
        
        exit_order = {
            "symbol": self.symbol,
            "trade_id": self.trade_id,  # Include trade ID in exit order
            "exit_type": f"tp{tp_index + 1}",
            "price": price,
            "size": tp_level["size"],
            "pnl": pnl,
            "date": date
        }
        
        self.state.realized_pnl += pnl
        self.state.size_remaining -= tp_level["size"]
        self.state.partial_exits.append(exit_order)
        
        # Check if position is fully closed
        if self.state.size_remaining <= 0:
            self.is_open = False
            self.exit_date = date
            self.exit_reason = f"tp{tp_index + 1}"
        
        # Clear this TP level
        tp_level["size"] = 0
        
        return exit_order
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL for remaining position size."""
        if not self.is_open or self.state.size_remaining <= 0:
            return 0.0
        
        # Calculate unrealized PnL in quote currency
        units_remaining = self.units * self.state.size_remaining
        entry_value = units_remaining * self.state.cost_basis
        current_value = units_remaining * current_price
        return current_value - entry_value
    
    def get_position_summary(self) -> Dict:
        """Get current position summary."""
        return {
            "symbol": self.symbol,
            "entry_date": self.entry_date.isoformat(),
            "entry_price": self.entry["price"],
            "position_type": self.position_type,
            "size_remaining": self.state.size_remaining,
            "realized_pnl": self.state.realized_pnl,
            "is_open": self.is_open,
            "exit_date": self.exit_date.isoformat() if self.exit_date else None,
            "exit_reason": self.exit_reason,
            "partial_exits": self.state.partial_exits,
            "original_metrics": self.metrics
        }

class Portfolio:
    """Manages multiple positions and tracks overall performance."""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, Position] = {}  # Active positions by trade_id
        self.position_by_symbol: Dict[str, str] = {}  # Maps symbol to trade_id
        self.closed_positions: List[Position] = []
        self.trade_history: List[Dict] = []
        
        # Performance tracking
        self.equity_curve = []
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        self.next_trade_id = 1  # Counter for generating trade IDs
    
    def open_position(self, order: Dict[str, Any], current_date: datetime) -> str:
        """
        Open a new position from an order.
        Returns the trade_id if position was opened successfully, empty string otherwise.
        """
        symbol = order["symbol"]
        
        # Check if position already exists for this symbol
        if symbol in self.position_by_symbol:
            return ""
        
        # Generate unique trade ID
        trade_id = str(self.next_trade_id)
        self.next_trade_id += 1
        
        # Create new position
        position = Position(order, current_date, self.current_balance, trade_id)
        self.positions[trade_id] = position
        self.position_by_symbol[symbol] = trade_id
        
        # Log position opening
        self.trade_history.append({
            "type": "entry",
            "trade_id": trade_id,
            "symbol": symbol,
            "date": current_date.isoformat(),
            "price": order["entry"]["price"],
            "size": 0.01
        })
        
        return trade_id
    
    def update_positions(self, current_prices: Dict[str, Dict[str, float]], current_date: datetime) -> List[Dict]:
        """
        Update all open positions with current prices.
        Returns list of executed orders.
        """
        executed_orders = []
        
        # Update each position
        for trade_id, position in list(self.positions.items()):
            if position.symbol not in current_prices:
                continue
            
            price_data = current_prices[position.symbol]
            orders = position.update(price_data, current_date)
            
            # Process executed orders
            for order in orders:
                executed_orders.append(order)
                self.trade_history.append({
                    "type": "exit",
                    "trade_id": trade_id,
                    "symbol": position.symbol,
                    "date": current_date.isoformat(),
                    **order
                })
                
                # Update portfolio balance
                self.current_balance += order["pnl"]
                
                # Update max drawdown
                self.peak_balance = max(self.peak_balance, self.current_balance)
                current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Move closed positions to history and remove from active positions
            if not position.is_open:
                self.closed_positions.append(position)
                del self.positions[trade_id]
                del self.position_by_symbol[position.symbol]
        
        # Update equity curve
        equity = self.calculate_total_equity(current_prices)
        self.equity_curve.append((current_date.isoformat(), equity))
        
        return executed_orders
    
    def calculate_total_equity(self, current_prices: Dict[str, Dict[str, float]]) -> float:
        """Calculate total portfolio value including unrealized PnL."""
        equity = self.current_balance
        
        # Add unrealized PnL from open positions
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                equity += position.get_unrealized_pnl(current_prices[symbol]["close"])
        
        return equity
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        return {
            "current_balance": self.current_balance,
            "initial_balance": self.initial_balance,
            "total_return_percent": ((self.current_balance / self.initial_balance) - 1) * 100,
            "max_drawdown_percent": self.max_drawdown * 100,
            "open_positions": len(self.positions),
            "closed_positions": len(self.closed_positions),
            "total_trades": len(self.trade_history)
        }
    
    def get_detailed_stats(self) -> Dict:
        """Get detailed portfolio statistics."""
        winning_trades = [p for p in self.closed_positions if p.state.realized_pnl > 0]
        losing_trades = [p for p in self.closed_positions if p.state.realized_pnl <= 0]
        
        return {
            "summary": self.get_portfolio_summary(),
            "trade_stats": {
                "total_trades": len(self.closed_positions),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": len(winning_trades) / len(self.closed_positions) if self.closed_positions else 0,
                "avg_win": sum(p.state.realized_pnl for p in winning_trades) / len(winning_trades) if winning_trades else 0,
                "avg_loss": sum(p.state.realized_pnl for p in losing_trades) / len(losing_trades) if losing_trades else 0,
                "largest_win": max((p.state.realized_pnl for p in winning_trades), default=0),
                "largest_loss": min((p.state.realized_pnl for p in losing_trades), default=0)
            },
            "current_positions": [
                {
                    "symbol": symbol,
                    "summary": position.get_position_summary()
                }
                for symbol, position in self.positions.items()
            ]
        }

if __name__ == "__main__":
    # Example usage
    portfolio = Portfolio(initial_balance=10000)
    
    # Sample order from order generator
    sample_order = {
        "symbol": "PEPE",
        "entry": {
            "price": 0.00001386,
            "timestamp": 1738281600
        },
        "position_type": "LONG",
        "orders": {
            "take_profit": [
                {"price": 0.00001524, "size": 0.3, "percentage": 10.0},
                {"price": 0.00001663, "size": 0.4, "percentage": 20.0},
                {"price": 0.00001871, "size": 0.3, "percentage": 35.0}
            ],
            "stop_loss": {
                "price": 0.00001289,
                "size": 1.0,
                "percentage": -7.0
            }
        },
        "risk_reward_ratio": 3.07,
        "metrics": {
            "galaxy_score": 50.0,
            "alt_rank": {"current": 4.0, "previous": 188.0}
        }
    }
    
    # Open position
    current_date = datetime.now()
    trade_id = portfolio.open_position(sample_order, current_date)
    
    # Simulate some price updates
    prices = {"PEPE": {"low": 0.00001289, "high": 0.00001524, "close": 0.00001524}}  # Hit first TP
    executed = portfolio.update_positions(prices, current_date)
    
    # Print results
    print("\n📊 Portfolio Status:")
    print("=" * 50)
    print(json.dumps(portfolio.get_detailed_stats(), indent=2, cls=DateTimeEncoder)) 
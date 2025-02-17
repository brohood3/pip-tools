"""
Limit Order Generator Tool

Generates limit orders with:
1. Entry price
2. Stop loss (10% below entry)
3. Take profit levels (10%, 20%, 35%)
"""

from typing import Dict, List, Any

def run(opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate limit orders for the given opportunities.
    
    Args:
        opportunities: List of opportunities from the screener
        
    Returns:
        Dict containing generated orders
    """
    orders = []
    
    for opp in opportunities:
        entry_price = opp['entry']['price']
        
        # Calculate stop loss (10% below entry)
        stop_loss_price = entry_price * 0.9
        
        # Calculate take profit levels
        tp1_price = entry_price * 1.1  # 10% profit
        tp2_price = entry_price * 1.2  # 20% profit
        tp3_price = entry_price * 1.35  # 35% profit
        
        order = {
            "symbol": opp['symbol'],
            "entry": {
                "price": entry_price,
                "timestamp": opp['entry']['timestamp']
            },
            "position_type": "long",
            "risk_reward_ratio": 3.5,
            "orders": {
                "stop_loss": {
                    "price": stop_loss_price,
                    "percentage": -10.0
                },
                "take_profit": [
                    {
                        "price": tp1_price,
                        "percentage": 10.0,
                        "size": 0.3  # 30% of position
                    },
                    {
                        "price": tp2_price,
                        "percentage": 20.0,
                        "size": 0.4  # 40% of position
                    },
                    {
                        "price": tp3_price,
                        "percentage": 35.0,
                        "size": 0.3  # 30% of position
                    }
                ]
            },
            "metrics": opp.get('metrics', {})
        }
        
        orders.append(order)
    
    return {
        "orders": orders,
        "metadata": {
            "total_opportunities": len(opportunities),
            "orders_generated": len(orders)
        }
    } 
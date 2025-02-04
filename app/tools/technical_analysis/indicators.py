"""
Indicator Registry for Technical Analysis Tool

This module provides a centralized registry for all technical indicators
used in the technical analysis tool. It includes metadata and configuration
for each indicator.
"""

from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class IndicatorCategory(Enum):
    """Categories for different types of technical indicators."""
    MOVING_AVERAGE = "moving_average"
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    PRICE_ACTION = "price_action"


class IndicatorParams(TypedDict, total=False):
    """Parameters that can be passed to indicators."""
    period: Optional[int]
    multiplier: Optional[float]
    length: Optional[int]


@dataclass
class Indicator:
    """Represents a technical indicator with its metadata."""
    name: str
    category: IndicatorCategory
    priority: int  # 1 = base indicator, 2 = common, 3 = specialized
    params: IndicatorParams
    requires_volume: bool = False
    description: str = ""


class IndicatorRegistry:
    """Central registry for all technical indicators."""
    
    def __init__(self):
        """Initialize the indicator registry with all available indicators."""
        self._indicators: Dict[str, Indicator] = {
            # Moving Averages (Priority 1-2)
            "sma": Indicator(
                name="sma",
                category=IndicatorCategory.MOVING_AVERAGE,
                priority=1,
                params={"period": 20},
                description="Simple Moving Average"
            ),
            "ema": Indicator(
                name="ema",
                category=IndicatorCategory.MOVING_AVERAGE,
                priority=1,
                params={"period": 20},
                description="Exponential Moving Average"
            ),
            "dema": Indicator(
                name="dema",
                category=IndicatorCategory.MOVING_AVERAGE,
                priority=2,
                params={"period": 20},
                description="Double Exponential Moving Average"
            ),
            "tema": Indicator(
                name="tema",
                category=IndicatorCategory.MOVING_AVERAGE,
                priority=2,
                params={"period": 20},
                description="Triple Exponential Moving Average"
            ),
            "vwma": Indicator(
                name="vwma",
                category=IndicatorCategory.MOVING_AVERAGE,
                priority=2,
                params={"period": 20},
                requires_volume=True,
                description="Volume Weighted Moving Average"
            ),
            
            # Trend Indicators (Priority 1-2)
            "supertrend": Indicator(
                name="supertrend",
                category=IndicatorCategory.TREND,
                priority=1,
                params={},
                description="SuperTrend indicator"
            ),
            "adx": Indicator(
                name="adx",
                category=IndicatorCategory.TREND,
                priority=1,
                params={},
                description="Average Directional Index"
            ),
            "dmi": Indicator(
                name="dmi",
                category=IndicatorCategory.TREND,
                priority=1,
                params={},
                description="Directional Movement Index"
            ),
            "vortex": Indicator(
                name="vortex",
                category=IndicatorCategory.TREND,
                priority=2,
                params={},
                description="Vortex Indicator"
            ),
            
            # Volume Indicators (Priority 1-2)
            "volume": Indicator(
                name="volume",
                category=IndicatorCategory.VOLUME,
                priority=1,
                params={},
                requires_volume=True,
                description="Raw volume"
            ),
            "obv": Indicator(
                name="obv",
                category=IndicatorCategory.VOLUME,
                priority=2,
                params={},
                requires_volume=True,
                description="On Balance Volume"
            ),
            "cmf": Indicator(
                name="cmf",
                category=IndicatorCategory.VOLUME,
                priority=2,
                params={},
                requires_volume=True,
                description="Chaikin Money Flow"
            ),
            
            # Momentum Indicators (Priority 1-2)
            "rsi": Indicator(
                name="rsi",
                category=IndicatorCategory.MOMENTUM,
                priority=1,
                params={},
                description="Relative Strength Index"
            ),
            "macd": Indicator(
                name="macd",
                category=IndicatorCategory.MOMENTUM,
                priority=1,
                params={},
                description="Moving Average Convergence Divergence"
            ),
            "stoch": Indicator(
                name="stoch",
                category=IndicatorCategory.MOMENTUM,
                priority=2,
                params={},
                description="Stochastic Oscillator"
            ),
            
            # Volatility Indicators (Priority 1-2)
            "bbands": Indicator(
                name="bbands",
                category=IndicatorCategory.VOLATILITY,
                priority=1,
                params={},
                description="Bollinger Bands"
            ),
            "atr": Indicator(
                name="atr",
                category=IndicatorCategory.VOLATILITY,
                priority=1,
                params={},
                description="Average True Range"
            ),
            "stddev": Indicator(
                name="stddev",
                category=IndicatorCategory.VOLATILITY,
                priority=2,
                params={},
                description="Standard Deviation"
            ),
            
            # Price Action (Priority 2-3)
            "fibonacciretracement": Indicator(
                name="fibonacciretracement",
                category=IndicatorCategory.PRICE_ACTION,
                priority=2,
                params={},
                description="Fibonacci Retracement Levels"
            ),
            "pivotpoints": Indicator(
                name="pivotpoints",
                category=IndicatorCategory.PRICE_ACTION,
                priority=2,
                params={},
                description="Pivot Points"
            ),
        }

        # Strategy-specific indicator mappings
        self._strategy_indicators = {
            "trend_following": [
                IndicatorCategory.MOVING_AVERAGE,
                IndicatorCategory.TREND
            ],
            "momentum": [
                IndicatorCategory.MOMENTUM,
                IndicatorCategory.VOLUME
            ],
            "mean_reversion": [
                IndicatorCategory.VOLATILITY,
                IndicatorCategory.MOMENTUM
            ],
            "breakout": [
                IndicatorCategory.VOLATILITY,
                IndicatorCategory.VOLUME,
                IndicatorCategory.PRICE_ACTION
            ],
            "scalping": [
                IndicatorCategory.MOMENTUM,
                IndicatorCategory.VOLATILITY
            ]
        }

    def get_base_indicators(self) -> List[Dict[str, Any]]:
        """Get all priority 1 (base) indicators."""
        return [
            {"indicator": ind.name, **ind.params}
            for ind in self._indicators.values()
            if ind.priority == 1
        ]

    def get_indicators_by_category(self, category: IndicatorCategory) -> List[Dict[str, Any]]:
        """Get all indicators in a specific category."""
        return [
            {"indicator": ind.name, **ind.params}
            for ind in self._indicators.values()
            if ind.category == category
        ]

    def get_indicators_for_strategy(self, strategy_type: str) -> List[Dict[str, Any]]:
        """Get recommended indicators for a specific strategy type."""
        if strategy_type not in self._strategy_indicators:
            return []
            
        categories = self._strategy_indicators[strategy_type]
        indicators = []
        for category in categories:
            indicators.extend(self.get_indicators_by_category(category))
        return indicators

    def format_for_taapi(self, indicators: List[str], max_indicators: int = 20) -> List[Dict[str, Any]]:
        """Format indicators for TAAPI request."""
        formatted = []
        for ind_name in indicators:
            if len(formatted) >= max_indicators:
                break
            if ind_name in self._indicators:
                indicator = self._indicators[ind_name]
                formatted.append({"indicator": indicator.name, **indicator.params})
        return formatted

    def is_valid_indicator(self, indicator_name: str) -> bool:
        """Check if an indicator is valid."""
        return indicator_name in self._indicators 
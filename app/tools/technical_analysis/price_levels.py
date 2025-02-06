from typing import Dict, Any, List, Optional

class PriceLevels:
    def __init__(self, indicators: Dict[str, Any], timeframe: str):
        self.timeframe = timeframe
        self.indicators = indicators
        
    def get_relevant_levels(self) -> Dict[str, Any]:
        """Extract relevant price levels based on timeframe and indicators"""
        return {
            "support": self._get_support_levels(),
            "resistance": self._get_resistance_levels(),
            "moving_averages": self._get_ma_levels(),
            "volume_nodes": self._get_volume_levels(),
        }
    
    def _get_support_levels(self) -> List[float]:
        """Extract support levels from indicators"""
        levels = []
        
        # Get support levels from price_levels indicator
        if 'pricelevels' in self.indicators:
            price_data = self.indicators['pricelevels']
            levels.extend(price_data.get('support_levels', []))
            
        # Add Bollinger Band lower as support
        if 'bbands' in self.indicators:
            bb_data = self.indicators['bbands']
            if 'lower' in bb_data:
                levels.append(bb_data['lower'])
                
        return sorted(list(set(levels)))  # Remove duplicates and sort
    
    def _get_resistance_levels(self) -> List[float]:
        """Extract resistance levels from indicators"""
        levels = []
        
        # Get resistance levels from price_levels indicator
        if 'pricelevels' in self.indicators:
            price_data = self.indicators['pricelevels']
            levels.extend(price_data.get('resistance_levels', []))
            
        # Add Bollinger Band upper as resistance
        if 'bbands' in self.indicators:
            bb_data = self.indicators['bbands']
            if 'upper' in bb_data:
                levels.append(bb_data['upper'])
                
        return sorted(list(set(levels)))  # Remove duplicates and sort
    
    def _get_ma_levels(self) -> Dict[str, float]:
        """Extract moving average levels"""
        levels = {}
        
        if 'sma' in self.indicators:
            sma_data = self.indicators['sma']
            for period in ['SMA_20', 'SMA_50', 'SMA_200']:
                if period in sma_data:
                    levels[period] = sma_data[period]
                    
        return levels
    
    def _get_volume_levels(self) -> List[float]:
        """Extract high volume price levels"""
        levels = []
        
        if 'pricelevels' in self.indicators:
            price_data = self.indicators['pricelevels']
            levels.extend(price_data.get('high_volume_levels', []))
            
        return sorted(levels) 
import mplfinance as mpf
import pandas as pd
from io import BytesIO
import base64
from typing import Dict, Any, List, Optional
from .price_levels import PriceLevels

class ChartGenerator:
    def __init__(self):
        self.style = self._create_default_style()
        # Define analysis type configurations
        self.analysis_configs = {
            "trend_following": {
                "indicators": ["sma", "supertrend", "adx"],
                "overlays": ["SMA_20", "SMA_50", "SMA_200"],
                "show_volume": True,
                "show_support_resistance": True,
                "show_volume_nodes": False,
                "show_bbands": False
            },
            "momentum": {
                "indicators": ["sma", "macd", "rsi", "stoch", "ao"],
                "overlays": ["SMA_20"],
                "show_volume": True,
                "show_support_resistance": False,
                "show_volume_nodes": False,
                "show_bbands": False
            },
            "mean_reversion": {
                "indicators": ["bbands", "rsi", "stoch", "sma", "ema"],
                "overlays": ["SMA_20", "EMA_50"],
                "show_volume": True,
                "show_support_resistance": True,
                "show_volume_nodes": False,
                "show_bbands": True
            },
            "breakout": {
                "indicators": ["atr", "volume", "donchian", "supertrend", "pivotpoints", "candlepatterns"],
                "overlays": [],
                "show_volume": True,
                "show_support_resistance": True,
                "show_volume_nodes": True,
                "show_bbands": False
            },
            "volatility": {
                "indicators": ["bbands", "atr", "stddev", "donchian", "psar", "ema", "adx"],
                "overlays": [],
                "show_volume": True,
                "show_support_resistance": False,
                "show_volume_nodes": False,
                "show_bbands": True
            },
            "default": {  # Fallback configuration
                "indicators": ["sma", "bbands"],
                "overlays": ["SMA_20", "SMA_50"],
                "show_volume": True,
                "show_support_resistance": True,
                "show_volume_nodes": False,
                "show_bbands": True
            }
        }
        
    def _create_default_style(self):
        """Create a professional, social-media friendly style"""
        return mpf.make_mpf_style(
            base_mpf_style='charles',
            gridstyle='',
            facecolor='white',
            edgecolor='black',
            figcolor='white',
            marketcolors=mpf.make_marketcolors(
                up='#26a69a',          # Green for up candles
                down='#ef5350',        # Red for down candles
                edge='inherit',        # Use same colors for edges
                wick='inherit',        # Use same colors for wicks
                volume='#1f77b4',      # Blue for volume
                ohlc='#000000',        # Black for OHLC
                alpha=1.0              # Full opacity for main colors
            ),
            rc={
                'font.size': 10,
                'font.weight': 'bold',
                'axes.labelsize': 10,
                'axes.labelweight': 'bold',
                'axes.facecolor': 'white',
                'axes.edgecolor': 'black',
                'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black',
                'legend.facecolor': 'white',
                'legend.edgecolor': 'black',
                'legend.fontsize': 10,
                'legend.labelcolor': 'black',
                'savefig.facecolor': 'white'
            }
        )
    
    def generate_chart(self, df: pd.DataFrame, indicators: Dict[str, Any], 
                      symbol: str, timeframe: str, analysis_type: str = "default") -> str:
        """Generate high-quality chart for social media
        
        Args:
            df: DataFrame with OHLCV data
            indicators: Dictionary of technical indicators
            symbol: Trading pair symbol
            timeframe: Chart timeframe
            analysis_type: Type of analysis to determine which indicators to show
            
        Returns:
            Base64 encoded PNG image
        """
        try:
            config = self.analysis_configs.get(analysis_type, self.analysis_configs["default"])
            print(f"\nUsing configuration for {analysis_type} analysis:")
            print(config)
            
            # Ensure DataFrame is properly formatted
            df = self._prepare_dataframe(df)
            
            # Extract price levels if needed
            levels = {}
            if config["show_support_resistance"] or config["show_volume_nodes"]:
                price_levels = PriceLevels(indicators, timeframe)
                levels = price_levels.get_relevant_levels()
                print("\nDebug - Price Levels:")
                print(f"Support: {levels.get('support', [])}")
                print(f"Resistance: {levels.get('resistance', [])}")
                print(f"Volume Nodes: {levels.get('volume_nodes', [])}")
            
            # Create indicator overlays based on configuration
            addplot = []
            
            # ------------------------------------------------
            # 1) Set the initial 'next_panel' index
            # ------------------------------------------------
            next_panel = 1
            if config["show_volume"]:
                # If volume is displayed, it sits in panel=1 => the next available panel is 2.
                next_panel = 2

            # ------------------------------------------------
            # 2) Overlays on the main panel=0
            # ------------------------------------------------
            if 'sma' in config['indicators']:
                sma_data = indicators.get('sma', {})
                for overlay in config["overlays"]:
                    if overlay in sma_data and overlay in df.columns:
                        color = {
                            'SMA_20': '#2196f3',
                            'SMA_50': '#ff9800',
                            'SMA_200': '#f44336'
                        }.get(overlay, '#2196f3')
                        addplot.append(
                            mpf.make_addplot(
                                df[overlay],
                                panel=0,
                                color=color,
                                width=1.5,
                                label=overlay.replace('_', ' ')
                            )
                        )

            # Supertrend overlay
            if 'supertrend' in config['indicators']:
                if 'SUPERT_7_3.0' in df.columns:
                    addplot.append(
                        mpf.make_addplot(
                            df['SUPERT_7_3.0'],
                            panel=0,
                            color='#9c27b0',
                            width=1.5,
                            label='Supertrend'
                        )
                    )

            # Bollinger Bands overlay if specified
            if config["show_bbands"] and 'bbands' in config['indicators']:
                if 'BBU_5_2.0' in df.columns and 'BBL_5_2.0' in df.columns:
                    addplot.extend([
                        mpf.make_addplot(
                            df['BBU_5_2.0'],
                            panel=0,
                            color='#00bcd4',
                            alpha=0.3,
                            width=1.0,
                            label='BB Upper',
                            fill_between={'y1': df['BBL_5_2.0'].values, 'y2': df['BBU_5_2.0'].values, 'alpha': 0.1, 'color': '#00bcd4'}
                        ),
                        mpf.make_addplot(
                            df['BBL_5_2.0'],
                            panel=0,
                            color='#00bcd4',
                            alpha=0.3,
                            width=1.0,
                            label='BB Lower'
                        )
                    ])

            # Add PSAR if configured
            if 'psar' in config['indicators']:
                if 'PSARl_0.02_0.2' in df.columns:
                    addplot.append(
                        mpf.make_addplot(
                            df['PSARl_0.02_0.2'],
                            panel=0,
                            type='scatter',
                            marker='.',
                            color='#673ab7',
                            alpha=0.6,
                            label='PSAR'
                        )
                    )

            # Add Donchian Channels if configured
            if 'donchian' in config['indicators']:
                if 'DCU_20_20' in df.columns and 'DCL_20_20' in df.columns:
                    addplot.extend([
                        mpf.make_addplot(
                            df['DCU_20_20'],
                            panel=0,
                            color='#ff4081',  # Pink
                            alpha=0.3,
                            width=1.0,
                            label='DC Upper',
                            fill_between=dict(y1=df['DCL_20_20'].values, y2=df['DCU_20_20'].values, alpha=0.1, color='#ff4081')
                        ),
                        mpf.make_addplot(
                            df['DCL_20_20'],
                            panel=0,
                            color='#ff4081',  # Pink
                            alpha=0.3,
                            width=1.0,
                            label='DC Lower'
                        )
                    ])

            # Add EMA if configured
            if 'ema' in config['indicators']:
                if 'EMA_20' in df.columns:
                    addplot.append(
                        mpf.make_addplot(
                            df['EMA_20'],
                            panel=0,
                            color='#ffd700',
                            width=1.0,
                            label='EMA 20'
                        )
                    )

            # ------------------------------------------------
            # 3) ADX in its own panel (if columns exist)
            # ------------------------------------------------
            if 'adx' in config['indicators']:
                if 'ADX_14' in df.columns:
                    addplot.append(
                        mpf.make_addplot(
                            df['ADX_14'],
                            panel=next_panel,
                            color='#4caf50',
                            width=1.0,
                            label='ADX'
                        )
                    )
                    # increment the panel index because we actually added
                    next_panel += 1

            # ------------------------------------------------
            # 4) MACD in its own panel
            # ------------------------------------------------
            if 'macd' in config['indicators']:
                macd_created = False
                panel_for_macd = next_panel

                if 'MACD_12_26_9' in df.columns:
                    addplot.append(
                        mpf.make_addplot(
                            df['MACD_12_26_9'],
                            panel=panel_for_macd,
                            color='#2196f3',
                            width=1.0,
                            label='MACD'
                        )
                    )
                    macd_created = True

                if 'MACDs_12_26_9' in df.columns:
                    addplot.append(
                        mpf.make_addplot(
                            df['MACDs_12_26_9'],
                            panel=panel_for_macd,
                            color='#ff9800',
                            width=1.0,
                            label='Signal'
                        )
                    )
                    macd_created = True

                if 'MACDh_12_26_9' in df.columns:
                    addplot.append(
                        mpf.make_addplot(
                            df['MACDh_12_26_9'],
                            panel=panel_for_macd,
                            type='bar',
                            color='#4caf50',
                            alpha=0.5,
                            label='MACD Hist'
                        )
                    )
                    macd_created = True

                if macd_created:
                    next_panel += 1

            # ------------------------------------------------
            # 5) AO in its own panel
            # ------------------------------------------------
            if 'ao' in config['indicators']:
                if 'AO_5_34' in df.columns:
                    addplot.append(
                        mpf.make_addplot(
                            df['AO_5_34'],
                            panel=next_panel,
                            type='bar',
                            color='#4caf50',
                            alpha=0.5,
                            label='AO'
                        )
                    )
                    next_panel += 1

            # ------------------------------------------------
            # 6) RSI and Stoch share one new panel
            # ------------------------------------------------
            # Check if either is in config => if so, they share next_panel
            rsi_stoch_panel = None
            if 'rsi' in config['indicators'] or 'stoch' in config['indicators']:
                rsi_stoch_panel = next_panel

            rsi_stoch_plots_added = False

            # RSI
            if 'rsi' in config['indicators'] and 'RSI_14' in df.columns:
                addplot.append(
                    mpf.make_addplot(
                        df['RSI_14'],
                        panel=rsi_stoch_panel,
                        color='#f44336',
                        width=1.0,
                        label='RSI',
                        secondary_y=False
                    )
                )
                rsi_stoch_plots_added = True
                # Optional: Overbought(70)/Oversold(30)

            # Stoch
            if 'stoch' in config['indicators']:
                if 'STOCHk_14_3_3' in df.columns:
                    addplot.append(
                        mpf.make_addplot(
                            df['STOCHk_14_3_3'],
                            panel=rsi_stoch_panel,
                            color='#2196f3',
                            width=1.0,
                            label='Stoch %K',
                            secondary_y=False
                        )
                    )
                    rsi_stoch_plots_added = True

                if 'STOCHd_14_3_3' in df.columns:
                    addplot.append(
                        mpf.make_addplot(
                            df['STOCHd_14_3_3'],
                            panel=rsi_stoch_panel,
                            color='#ff9800',
                            width=1.0,
                            label='Stoch %D',
                            secondary_y=False
                        )
                    )
                    rsi_stoch_plots_added = True

            # Only increment next_panel if we actually added RSI or Stoch
            if rsi_stoch_panel is not None and rsi_stoch_plots_added:
                next_panel += 1

            # ------------------------------------------------
            # 7) Separate panels for other volatility indicators
            # ------------------------------------------------
            
            # ATR in its own panel if configured
            if 'atr' in config['indicators']:
                if 'ATRr_14' in df.columns:
                    addplot.append(
                        mpf.make_addplot(
                            df['ATRr_14'],
                            panel=next_panel,
                            color='#e91e63',
                            width=1.0,
                            label='ATR'
                        )
                    )
                    next_panel += 1

            # Standard Deviation in its own panel if configured
            if 'stddev' in config['indicators']:
                if 'STDEV_20' in df.columns:
                    addplot.append(
                        mpf.make_addplot(
                            df['STDEV_20'],
                            panel=next_panel,
                            color='#009688',
                            width=1.0,
                            label='StdDev'
                        )
                    )
                    next_panel += 1

            # ------------------------------------------------
            # Plot Support and Resistance Levels
            # ------------------------------------------------
            if config["show_support_resistance"] and levels:
                # Plot support levels
                for i, level in enumerate(levels.get('support', [])):
                    # Create a series with the support level value repeated for the entire DataFrame
                    support_line = pd.Series(level, index=df.index)
                    plot_args = {
                        'panel': 0,
                        'color': '#4caf50',  # Green
                        'width': 1.0,
                        'linestyle': '--',
                        'alpha': 0.6
                    }
                    if i == 0:  # Only add label for first line
                        plot_args['label'] = 'Support'
                    addplot.append(mpf.make_addplot(support_line, **plot_args))

                # Plot resistance levels
                for i, level in enumerate(levels.get('resistance', [])):
                    # Create a series with the resistance level value repeated for the entire DataFrame
                    resistance_line = pd.Series(level, index=df.index)
                    plot_args = {
                        'panel': 0,
                        'color': '#f44336',  # Red
                        'width': 1.0,
                        'linestyle': '--',
                        'alpha': 0.6
                    }
                    if i == 0:  # Only add label for first line
                        plot_args['label'] = 'Resistance'
                    addplot.append(mpf.make_addplot(resistance_line, **plot_args))

            # Plot Volume Profile Nodes if configured
            if config["show_volume_nodes"] and levels:
                for i, node in enumerate(levels.get('volume_nodes', [])):
                    # Create a series with the volume node level value repeated
                    node_line = pd.Series(node, index=df.index)
                    plot_args = {
                        'panel': 0,
                        'color': '#9c27b0',  # Purple
                        'width': 1.0,
                        'linestyle': ':',
                        'alpha': 0.4
                    }
                    if i == 0:  # Only add label for first line
                        plot_args['label'] = 'Volume Node'
                    addplot.append(mpf.make_addplot(node_line, **plot_args))

            # ------------------------------------------------
            # 8) Build panel_ratios for all used panels
            # ------------------------------------------------
            num_panels = next_panel
            panel_ratios = [3]  # main price panel up front
            if config["show_volume"]:
                panel_ratios.append(1)  # volume panel
            used_so_far = len(panel_ratios)
            for _ in range(num_panels - used_so_far):
                panel_ratios.append(1)

            # Now call mpf.plot() using these dynamic panel assignments
            fig, axes = mpf.plot(
                df,
                type='line',
                style=self.style,
                volume=config["show_volume"],
                figsize=(12, 8),
                panel_ratios=panel_ratios,
                addplot=addplot,
                returnfig=True,
                tight_layout=False,
                volume_panel=1 if config["show_volume"] else None,
                update_width_config=dict(
                    line_width=2.5,  # Increased line width for price
                    volume_width=0.6,
                    volume_linewidth=0.8
                ),
                ylabel_lower='Volume (Millions)',
                linecolor='black'  # Set price line color to black
            )
            
            # Add title above the chart
            fig.suptitle(f'{symbol} {timeframe} Technical Analysis', 
                        y=0.95,  # Position above the chart
                        fontsize=12,
                        fontweight='bold',
                        color='black')
            
            # Adjust layout to prevent title overlap
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for title
            
            # Add annotations to the chart
            print("\nDebug - Adding annotations to chart:")
            for ann in self._create_annotations(df, indicators):
                print(f"Adding annotation: {ann}")
                try:
                    bbox_props = ann.get('bbox', dict(
                        facecolor='white',
                        edgecolor=ann['color'],
                        alpha=0.7,
                        boxstyle='round,pad=0.5'
                    )) if ann.get('bbox') is not None else None
                    
                    arrow_props = dict(
                        arrowstyle='->',
                        color=ann['color'],
                        alpha=0.8,
                        connectionstyle='arc3,rad=0.2'
                    ) if ann.get('use_arrow', True) else None
                    
                    # Use the specified panel or default to panel 0 (price panel)
                    target_panel = ann.get('panel', 0)
                    axes[target_panel].annotate(
                        ann['text'],
                        xy=(ann['x'], ann['y']),
                        xytext=(20, 10) if ann.get('bbox') is not None else (0, 0),  # No offset for arrows
                        textcoords='offset points',
                        color=ann['color'],
                        fontsize=10,
                        fontweight='bold',
                        alpha=0.8,
                        bbox=bbox_props,
                        arrowprops=arrow_props
                    )
                    print(f"Successfully added annotation: {ann['text']}")
                except Exception as e:
                    print(f"Error adding annotation {ann['text']}: {str(e)}")
                    traceback.print_exc()
            
            # Add legend with black text color and white background
            if addplot:
                # Add legend to price panel for price indicators only
                price_panel_plots = [p for p in addplot if not hasattr(p, 'panel') or p.panel == 0]
                if price_panel_plots:
                    legend = axes[0].legend(
                        loc='upper left',
                        facecolor='white',
                        edgecolor='black',
                        labelcolor='black',
                        framealpha=0.9
                    )
                    # Ensure legend text is black
                    for text in legend.get_texts():
                        text.set_color('black')
            
            # Save to bytes buffer with high DPI
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=300)
            buf.seek(0)
            
            # Convert to base64
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Clean up
            fig.clear()
            
            return image_base64
            
        except Exception as e:
            print(f"Error generating chart: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""
        
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for mplfinance"""
        try:
            # Create a copy of the DataFrame to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Ensure we have required columns
            required_columns = ['close', 'volume']  # Only need close price for line plot
            if not all(col in df.columns for col in required_columns):
                raise ValueError("DataFrame missing required columns")
                
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'datetime' in df.columns:
                    df.set_index('datetime', inplace=True)
                else:
                    raise ValueError("DataFrame must have datetime index")
            
            # For line plot, we use close price for all OHLC values
            df.loc[:, 'open'] = df['close']
            df.loc[:, 'high'] = df['close']
            df.loc[:, 'low'] = df['close']
            
            # Ensure close price and volume are float type
            df.loc[:, 'close'] = df['close'].astype(float)
            
            # Scale down volume to prevent integer overflow
            df.loc[:, 'volume'] = (df['volume'] / 1e6).astype(float)  # Convert to millions
                    
            return df
            
        except Exception as e:
            print(f"Error preparing DataFrame: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_annotations(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create annotations for key price levels and patterns"""
        annotations = []
        
        try:
            print("\nDebug - Creating Annotations:")
            print(f"Available indicators: {list(indicators.keys())}")
            
            # Use integer index for x-coordinate
            last_idx = len(df) - 1
            price_range = df['high'].max() - df['low'].min()
            
            # Add key price levels from indicators first (so it appears at the bottom)
            if 'pricelevels' in indicators:
                print("Found pricelevels in indicators")
                price_data = indicators['pricelevels']
                print(f"Price data: {price_data}")
                current_price = price_data.get('current_price', df['close'].iloc[-1])
                
                # Add current price marker at bottom left
                annotations.append({
                    'x': 2,  # Further left
                    'y': df['low'].min() - price_range * 0.15,  # Adjusted for more spacing
                    'text': f'Current Price: ${current_price:,.2f}',
                    'color': '#000000',
                    'bbox': dict(
                        facecolor='white',
                        edgecolor='#666666',
                        alpha=0.9,
                        boxstyle='round,pad=0.5',
                        mutation_aspect=0.5
                    ),
                    'use_arrow': False
                })
            else:
                print("No pricelevels found in indicators")
            
            # Add trend direction below the price with more spacing
            if 'supertrend' in indicators:
                print("Found supertrend in indicators")
                trend = indicators['supertrend'].get('trend', 0)
                print(f"Supertrend value: {trend}")
                if trend > 0:
                    annotations.append({
                        'x': 2,  # Further left
                        'y': df['low'].min() - price_range *0.25,  # More spacing below price
                        'text': 'TREND: BULLISH ▲',
                        'color': '#2e7d32',
                        'bbox': dict(
                            facecolor='#e8f5e9',
                            edgecolor='#2e7d32',
                            alpha=0.9,
                            boxstyle='round,pad=0.5'
                        ),
                        'use_arrow': False
                    })
                else:
                    annotations.append({
                        'x': 2,  # Further left
                        'y': df['low'].min() - price_range *0.25,  # More spacing below price
                        'text': 'TREND: BEARISH ▼',
                        'color': '#c62828',
                        'bbox': dict(
                            facecolor='#ffebee',
                            edgecolor='#c62828',
                            alpha=0.9,
                            boxstyle='round,pad=0.5'
                        ),
                        'use_arrow': False
                    })

            # Add MACD crossover arrows (simplified)
            if 'macd' in indicators:
                macd_data = indicators['macd']
                macd_value = macd_data.get('value')
                signal_value = macd_data.get('signal')
                
                if macd_value is not None and signal_value is not None:
                    # Find the crossover point in the data
                    macd_series = df['MACD_12_26_9']
                    signal_series = df['MACDs_12_26_9']
                    
                    # Look for crossover in last 10 periods
                    for i in range(len(df)-10, len(df)-1):
                        if (macd_series.iloc[i] <= signal_series.iloc[i] and 
                            macd_series.iloc[i+1] > signal_series.iloc[i+1]):
                            # Bullish crossover
                            annotations.append({
                                'x': i+1,  # Point of crossover
                                'y': 0,  # Center of MACD panel
                                'text': '▲',  # Up arrow
                                'color': '#4caf50',  # Green
                                'bbox': None,
                                'use_arrow': False,
                                'panel': 3  # Specify MACD panel
                            })
                        elif (macd_series.iloc[i] >= signal_series.iloc[i] and 
                              macd_series.iloc[i+1] < signal_series.iloc[i+1]):
                            # Bearish crossover
                            annotations.append({
                                'x': i+1,  # Point of crossover
                                'y': 0,  # Center of MACD panel
                                'text': '▼',  # Down arrow
                                'color': '#f44336',  # Red
                                'bbox': None,
                                'use_arrow': False,
                                'panel': 3  # Specify MACD panel
                            })
            
            return annotations
            
        except Exception as e:
            print(f"Error creating annotations: {str(e)}")
            import traceback
            traceback.print_exc()
            return [] 
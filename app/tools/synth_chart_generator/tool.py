"""
Synth Chart Generator Tool

A tool that generates price prediction visualizations using the Synth API:
1) Fetches price predictions from Synth API for a given asset
2) Processes multiple simulations to find the most likely price trajectory
3) Generates an interactive visualization showing the median prediction and individual simulations
4) Returns both the chart and prediction data for analysis
"""

import os
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import requests
from typing import Dict, Optional, List, Any
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import yaml

# Global variable to store the latest chart
_latest_chart: Optional[str] = None

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml file."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_latest_chart() -> Optional[str]:
    """Get the latest generated chart.
    
    Returns:
        Optional[str]: Base64 encoded chart data if available, None otherwise
    """
    global _latest_chart
    return _latest_chart


def store_latest_chart(chart_data: str) -> None:
    """Store the latest generated chart.
    
    Args:
        chart_data: Base64 encoded chart data
    """
    global _latest_chart
    _latest_chart = chart_data


class APIClients:
    def __init__(self):
        """Initialize API clients with required keys."""
        self.synth_api_key = os.getenv("SYNTH_API_KEY")
        if not self.synth_api_key:
            raise ValueError("Missing SYNTH_API_KEY environment variable")


def get_synth_predictions(clients: APIClients, asset: str) -> Optional[List[Dict[str, Any]]]:
    """Fetches price predictions from the Synth API for a given asset.
    
    Makes a GET request to the Synth API endpoint to retrieve price predictions for a given asset.
    The predictions are for the next 24 hours with 5-minute intervals.
    
    Args:
        clients: APIClients instance containing the required Synth API key
        asset: The cryptocurrency asset to analyze (e.g., "BTC")
        
    Returns:
        Optional[List[Dict[str, Any]]]: List of prediction data if successful, None if the API request fails
        or no predictions are available
    """
    synth_api_key = clients.synth_api_key
    endpoint = "https://synth.mode.network/prediction/best"

    # Set up parameters
    params = {
        "asset": asset,
        "time_increment": 300,  # 5 minutes in seconds
        "time_length": 86400    # 24 hours in seconds
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Apikey {synth_api_key}"
    }

    response = requests.get(endpoint, params=params, headers=headers)
    if response.status_code != 200:
        print(f"API request failed with status code: {response.status_code}")
        print(f"Response content: {response.text}")
        return None

    data = response.json()
    if not data:
        print(f"No predictions available for parameters: {params}")
        return None

    print(f"Successfully fetched {asset} price predictions")
    return data


def get_most_likely_prediction(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find the most likely price prediction by analyzing all simulations.
    
    Args:
        data: List of prediction data containing simulations
        
    Returns:
        List of 24 hourly price points representing the most likely price trajectory
    """
    # Dictionary to store all prices for each timestamp
    time_to_prices = {}

    simulations = data[0]["prediction"] 

    for simulation in simulations:
        for point in simulation:
            time = point["time"]
            price = point["price"]

            if time not in time_to_prices:
                time_to_prices[time] = []
            time_to_prices[time].append(price)

    # Get most likely price for each timestamp
    most_likely_prediction = []
    sorted_times = sorted(time_to_prices.keys())

    for time in sorted_times:
        prices = time_to_prices[time]
        # Use median price as most likely price level
        most_likely_price = sorted(prices)[len(prices)//2]

        most_likely_prediction.append({
            "time": time,
            "price": most_likely_price
        })

    print("Generated most likely prediction")
    return most_likely_prediction


def get_price_prediction_chart(best_prediction_data: List[Dict[str, Any]], asset_predictions: List[Dict[str, Any]], asset: str) -> str:
    """Generate a price prediction chart showing the median prediction and individual simulations.
    
    Args:
        best_prediction_data: List of dictionaries containing the median price prediction data points
        asset_predictions: List of dictionaries containing all simulation data
        asset: The cryptocurrency asset being analyzed
        
    Returns:
        Base64 encoded chart image with data URI prefix for direct PNG viewing
    """
    print(f"Generating price prediction chart for {asset}")

    # Extract data from dictionaries into lists
    datetimes = [pd.to_datetime(item['time']).timestamp() for item in best_prediction_data]
    prices = [item['price'] for item in best_prediction_data]

    # Create DataFrame
    df = pd.DataFrame({
        'datetime': datetimes,
        'open': prices,
        'high': prices,
        'low': prices,
        'close': prices,
        'volume': prices
    })
    # Convert datetime strings to datetime objects
    df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    # Set style for dark theme
    plt.style.use('dark_background')

    # Create figure with dark background
    fig, price_ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('black')
    price_ax.set_facecolor('black')

    # Plot median prediction with specified color and increased line width
    median_line = price_ax.plot(df.close, label=f'Median Price Prediction({asset})', color='#d7f416', linewidth=2.5)[0]
    price_ax.set_title(f'{asset} Price Simulations', color='white', fontsize=12)
    price_ax.set_ylabel('Price', color='white')
    price_ax.grid(True, color='gray', alpha=0.2)

    # Set only the left x-axis limit to match data start
    left_limit = df.index[0]
    price_ax.set_xlim(left=left_limit)

    # Plot all prediction simulations
    for prediction in asset_predictions[0]["prediction"]:
        datetimes = [pd.to_datetime(item['time']).timestamp() for item in prediction]
        prices = [item['price'] for item in prediction]

        # Convert timestamps to datetime and create a Series
        dates = pd.to_datetime(datetimes, unit='s')
        prediction_series = pd.Series(prices, index=dates)

        # Plot with lower alpha for visibility
        price_ax.plot(prediction_series, alpha=0.2, color='grey', label='_nolegend_')

    # Add final price annotation
    final_price = df.close.iloc[-1]
    final_time = df.index[-1]
    price_ax.annotate(
        f'${final_price:,.2f}',
        xy=(final_time, final_price),
        xytext=(10, 0),
        textcoords='offset points',
        color='white',
        fontweight='bold',
        bbox=dict(facecolor='black', edgecolor='#d7f416', alpha=0.7)
    )

    # Style the axes
    price_ax.tick_params(colors='white')
    for spine in price_ax.spines.values():
        spine.set_color('white')

    # Add legend with white text
    legend = price_ax.legend(loc='upper right')
    plt.setp(legend.get_texts(), color='white')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save chart to bytes
    img = BytesIO()
    plt.savefig(img, format='png', facecolor='black', edgecolor='none')
    img.seek(0)

    # Encode as base64 string with data URI prefix for PNG
    chart_b64 = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"
    plt.close()

    print("Chart generation completed successfully")
    return chart_b64


def _run_internal(asset: str) -> Dict[str, Any]:
    """Internal function to run the synth chart generation.
    
    Args:
        asset: The cryptocurrency asset to analyze (e.g., "BTC")
        
    Returns:
        Dictionary containing:
        - response: Most likely prediction price
        - metadata: Information about the prediction including all price points
        - chart: Base64 encoded chart data
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize API clients
        clients = APIClients()

        # Get predictions from Synth API
        asset_predictions = get_synth_predictions(clients=clients, asset=asset)
        if not asset_predictions:
            return {
                "error": "Failed to get price predictions from Synth subnet.",
                "metadata": {
                    "asset": asset,
                    "timestamp": datetime.now().isoformat()
                }
            }

        # Process predictions to get most likely trajectory
        most_likely_prediction = get_most_likely_prediction(asset_predictions)
        
        # Get the final predicted price
        final_prediction = most_likely_prediction[-1]["price"]

        # Generate visualization
        chart = get_price_prediction_chart(most_likely_prediction, asset_predictions, asset)
        
        # Store the latest chart
        store_latest_chart(chart)

        # Return results in new format
        return {
            "response": f"${final_prediction:,.2f}",
            "metadata": {
                "asset": asset,
                "timestamp": datetime.now().isoformat(),
                "prediction_timeframe": "24h",
                "predictions": most_likely_prediction
            },
            "chart": chart  # Return chart data directly
        }

    except Exception as e:
        print(f"Error in Synth chart generation: {str(e)}")
        return {
            "error": str(e),
            "metadata": {
                "asset": asset,
                "timestamp": datetime.now().isoformat()
            }
        }

class SynthChartGenerator:
    """Synth chart generation tool for Bitcoin price predictions."""
    
    def __init__(self):
        """Initialize the Synth chart generator."""
        pass
        
    def run(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Run Synth chart generation and price prediction analysis.
        
        Args:
            prompt: Ignored - tool always generates BTC predictions
            system_prompt: Optional system prompt (unused)
            
        Returns:
            Dictionary containing:
            - response: Final predicted price with formatting
            - metadata: Information about the prediction including all price points
            - chart: Base64 encoded chart data
        """
        return _run_internal("BTC")

# added the following to have uniformity in the way we call tools
def run(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Main entry point for the Synth chart generator tool.
    
    Always generates Bitcoin price predictions regardless of prompt.
    """
    return SynthChartGenerator().run(prompt, system_prompt) 
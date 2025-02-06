#!/usr/bin/env python3
import requests
import json
import base64
import sys

def save_chart(prompt: str, output_file: str = "chart.png"):
    """Save the chart from the technical analysis API response."""
    
    # Make API request
    response = requests.post(
        "http://localhost:8000/technical_analysis",
        json={
            "prompt": prompt,
            "system_prompt": "You are an expert technical analyst."
        }
    )
    
    # Check if request was successful
    if response.status_code != 200:
        print(f"Error: API request failed with status {response.status_code}")
        print(response.text)
        return
        
    # Parse response
    data = response.json()
    if "result" not in data or "chart" not in data["result"]:
        print("Error: No chart found in response")
        return
        
    # Get chart data
    chart_base64 = data["result"]["chart"]
    if not chart_base64:
        print("Error: Chart data is empty")
        return
        
    # Decode and save chart
    try:
        chart_data = base64.b64decode(chart_base64)
        with open(output_file, "wb") as f:
            f.write(chart_data)
        print(f"Chart saved to {output_file}")
        
        # Print analysis
        print("\nAnalysis:")
        print("=" * 80)
        print(data["result"]["response"])
        
    except Exception as e:
        print(f"Error saving chart: {str(e)}")

if __name__ == "__main__":
    # Get prompt from command line arguments or use default
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Analyze BTC trend on 4h timeframe with moving averages and bollinger bands"
    save_chart(prompt) 
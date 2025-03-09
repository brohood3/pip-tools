"""
Run a real test of the video generator tool with the Replicate API.

This script will generate a short video using the wan-video/wan-2.1-1.3b model.
"""

import os
import sys
from dotenv import load_dotenv
import time

# Add the parent directory to the path so we can import the tool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the tool
from video_generator.tool import run

# Load environment variables from .env file
load_dotenv()

# Set the API token
api_token = os.getenv("REPLICATE_API_TOKEN")
if not api_token:
    # Use the provided token if not in environment
    api_token = "r8_BetWyE5rihhv0s5eLIrE3zHtBc6Rwpz0Mwg19"
    os.environ["REPLICATE_API_TOKEN"] = api_token
    print(f"Using provided API token: {api_token[:5]}...{api_token[-5:]}")
else:
    print(f"Using API token from environment: {api_token[:5]}...{api_token[-5:]}")

def main():
    """Run a test of the video generator tool."""
    # Simple prompt for testing
    prompt = "a cat playing with a ball of yarn, cute, high quality"
    
    print(f"Generating video for prompt: '{prompt}'")
    print("This may take a minute or two...")
    
    # Start timer
    start_time = time.time()
    
    # Generate the video
    try:
        result = run(prompt=prompt)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Get the path to the generated video
        video_path = result["response"]["video"]
        
        print(f"\nVideo generated successfully in {elapsed_time:.2f} seconds!")
        print(f"Video saved to: {video_path}")
        print(f"Format: {result['response']['format']}")
        print(f"Model used: {result['metadata']['model']}")
        print(f"Timestamp: {result['metadata']['timestamp']}")
        
        # Print file size
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # Convert to MB
        print(f"File size: {file_size:.2f} MB")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during video generation: {str(e)}")
        elapsed_time = time.time() - start_time
        print(f"Test failed after {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main() 
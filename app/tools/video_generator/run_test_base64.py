"""
Run a real test of the video generator tool with base64 encoding.

This script will generate a short video and return it as a base64 encoded string.
"""

import os
import sys
from dotenv import load_dotenv
import time
import base64

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
    """Run a test of the video generator tool with base64 encoding."""
    # Simple prompt for testing
    prompt = "a robot dancing in a futuristic city, high quality"
    
    print(f"Generating video for prompt: '{prompt}' with base64 encoding")
    print("This may take a minute or two...")
    
    # Start timer
    start_time = time.time()
    
    # Generate the video with base64 encoding
    try:
        result = run(prompt=prompt, return_base64=True)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Get the base64 encoded video
        video_base64 = result["response"]["video"]
        
        print(f"\nVideo generated successfully in {elapsed_time:.2f} seconds!")
        print(f"Format: {result['response']['format']}")
        print(f"Base64 string length: {len(video_base64)} characters")
        print(f"Model used: {result['metadata']['model']}")
        print(f"Timestamp: {result['metadata']['timestamp']}")
        
        # Save the base64 encoded video to a file for demonstration
        with open("base64_video.txt", "w") as f:
            f.write(video_base64)
        print(f"Base64 encoded video saved to: base64_video.txt")
        
        # Decode the base64 string and save as a video file
        video_data = base64.b64decode(video_base64)
        with open("decoded_video.mp4", "wb") as f:
            f.write(video_data)
        
        # Print file size
        file_size = os.path.getsize("decoded_video.mp4") / (1024 * 1024)  # Convert to MB
        print(f"Decoded video saved to: decoded_video.mp4")
        print(f"File size: {file_size:.2f} MB")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during video generation: {str(e)}")
        elapsed_time = time.time() - start_time
        print(f"Test failed after {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main() 
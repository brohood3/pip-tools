"""
Video Generator Tool

A tool that generates videos based on text prompts using the Replicate API
with the wan-video/wan-2.1-1.3b model.
"""

import os
import io
import base64
from typing import Dict, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
import replicate

# Load environment variables
load_dotenv()


class VideoGenerator:
    """Video generation tool using Replicate API with wan-video/wan-2.1-1.3b model."""

    def __init__(self):
        """Initialize with Replicate API token."""
        self.replicate_api_token = os.getenv("REPLICATE_API_TOKEN")

        if not self.replicate_api_token:
            raise ValueError("Missing required REPLICATE_API_TOKEN")
        
        # Set the API token for the replicate client
        os.environ["REPLICATE_API_TOKEN"] = self.replicate_api_token

    def run(
        self, 
        prompt: str,
        model: Optional[str] = None,
        return_base64: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a video based on a text prompt using Replicate API.

        Args:
            prompt: The text prompt describing the video to generate
            model: Optional model parameter (defaults to wan-video/wan-2.1-1.3b)
            return_base64: Whether to return the video as base64 encoded string

        Returns:
            Dict containing response and metadata
        """
        try:
            # Default model
            replicate_model = model or "wan-video/wan-2.1-1.3b"
            
            # Run the model
            output = replicate.run(
                replicate_model,
                input={"prompt": prompt}
            )
            
            # If return_base64 is True, convert the video to base64
            video_data = None
            if return_base64:
                buffer = io.BytesIO()
                buffer.write(output.read())
                buffer.seek(0)
                video_data = base64.b64encode(buffer.read()).decode('utf-8')
            else:
                # Save the video to a temporary file and return the path
                temp_file_path = f"temp_video_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
                with open(temp_file_path, "wb") as file:
                    file.write(output.read())
                video_data = temp_file_path
            
            # Format for consistency with other tools
            format_type = "base64" if return_base64 else "file_path"
            
            # Return response with metadata
            return {
                "response": {
                    "video": video_data,
                    "format": format_type
                },
                "metadata": {
                    "prompt": prompt,
                    "timestamp": datetime.now().isoformat(),
                    "model": replicate_model,
                    "requested_model": model,  # Store the originally requested model for reference
                    "format_type": format_type
                }
            }
            
        except Exception as e:
            return {"error": str(e)}


# Function to be called from the API
def run(
    prompt: str,
    model: Optional[str] = None,
    return_base64: bool = False
) -> Dict[str, Any]:
    return VideoGenerator().run(prompt, model, return_base64) 
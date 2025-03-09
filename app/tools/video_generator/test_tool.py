"""
Test script for the video generator tool.

This script tests the basic functionality of the video generator tool.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO

# Add the parent directory to the path so we can import the tool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the tool
from video_generator.tool import VideoGenerator, run


class TestVideoGenerator(unittest.TestCase):
    """Test cases for the VideoGenerator tool."""

    @patch.dict(os.environ, {"REPLICATE_API_TOKEN": "test_token"})
    @patch("replicate.run")
    def test_video_generation_file_output(self, mock_replicate_run):
        """Test video generation with file output."""
        # Create a mock response from replicate
        mock_output = MagicMock()
        mock_output.read.return_value = b"fake video data"
        mock_replicate_run.return_value = mock_output

        # Run the tool
        result = run(prompt="test prompt")

        # Check that replicate.run was called with the correct arguments
        mock_replicate_run.assert_called_once_with(
            "wan-video/wan-2.1-1.3b",
            input={"prompt": "test prompt"}
        )

        # Check the structure of the result
        self.assertIn("response", result)
        self.assertIn("metadata", result)
        
        # Check response fields
        self.assertIn("video", result["response"])
        self.assertIn("format", result["response"])
        self.assertEqual(result["response"]["format"], "file_path")
        
        # Check metadata fields
        self.assertIn("prompt", result["metadata"])
        self.assertEqual(result["metadata"]["prompt"], "test prompt")
        self.assertIn("model", result["metadata"])
        self.assertEqual(result["metadata"]["model"], "wan-video/wan-2.1-1.3b")
        self.assertIn("timestamp", result["metadata"])
        self.assertIn("format_type", result["metadata"])
        self.assertEqual(result["metadata"]["format_type"], "file_path")

        # Check that the file exists
        self.assertTrue(os.path.exists(result["response"]["video"]))
        
        # Clean up the file
        os.remove(result["response"]["video"])

    @patch.dict(os.environ, {"REPLICATE_API_TOKEN": "test_token"})
    @patch("replicate.run")
    def test_video_generation_base64_output(self, mock_replicate_run):
        """Test video generation with base64 output."""
        # Create a mock response from replicate
        mock_output = MagicMock()
        mock_output.read.return_value = b"fake video data"
        mock_replicate_run.return_value = mock_output

        # Run the tool
        result = run(prompt="test prompt", return_base64=True)

        # Check the structure of the result
        self.assertIn("response", result)
        self.assertIn("metadata", result)
        
        # Check response fields
        self.assertIn("video", result["response"])
        self.assertIn("format", result["response"])
        self.assertEqual(result["response"]["format"], "base64")
        
        # Check that the video is a base64 string
        self.assertIsInstance(result["response"]["video"], str)
        
        # Check metadata fields
        self.assertIn("format_type", result["metadata"])
        self.assertEqual(result["metadata"]["format_type"], "base64")

    @patch.dict(os.environ, {"REPLICATE_API_TOKEN": "test_token"})
    @patch("replicate.run")
    def test_custom_model(self, mock_replicate_run):
        """Test using a custom model."""
        # Create a mock response from replicate
        mock_output = MagicMock()
        mock_output.read.return_value = b"fake video data"
        mock_replicate_run.return_value = mock_output

        # Run the tool with a custom model
        custom_model = "custom/model-name"
        result = run(prompt="test prompt", model=custom_model)

        # Check that replicate.run was called with the correct model
        mock_replicate_run.assert_called_once_with(
            custom_model,
            input={"prompt": "test prompt"}
        )
        
        # Check that the metadata contains the correct model information
        self.assertEqual(result["metadata"]["model"], custom_model)
        self.assertEqual(result["metadata"]["requested_model"], custom_model)

    def test_missing_api_token(self):
        """Test behavior when API token is missing."""
        # Clear the environment variable
        original_token = os.environ.get("REPLICATE_API_TOKEN")
        if "REPLICATE_API_TOKEN" in os.environ:
            del os.environ["REPLICATE_API_TOKEN"]
        
        try:
            # Print environment variables for debugging
            print("Environment variables:", [key for key in os.environ.keys() if key.startswith("REPLICATE")])
            
            # Create a VideoGenerator instance without an API token
            with self.assertRaises(ValueError) as context:
                VideoGenerator()
            
            # Check the error message
            self.assertIn("Missing required REPLICATE_API_TOKEN", str(context.exception))
        finally:
            # Restore the original token if it existed
            if original_token:
                os.environ["REPLICATE_API_TOKEN"] = original_token


if __name__ == "__main__":
    unittest.main() 
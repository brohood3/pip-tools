"""
WSGI entry point for Gunicorn
"""

import os
import sys
import logging

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("wsgi")

logger.info("Initializing application...")

try:
    # Import the Flask app and socketio
    from ai_chat_app import app, socketio
    
    # For Gunicorn + eventlet worker
    # When using gunicorn with the eventlet worker and socketio, this format is required
    # for proper WebSocket support
    application = socketio.wsgi_app
    
    logger.info("Application initialized successfully")
except Exception as e:
    logger.error(f"Error initializing application: {e}")
    # Create a simple WSGI application that returns an error
    def application(environ, start_response):
        status = '500 Internal Server Error'
        response_headers = [('Content-type', 'text/plain')]
        start_response(status, response_headers)
        return [b'Application failed to initialize. Check logs for details.']
    
    # Re-raise the exception to ensure it's logged properly
    raise

# This is the entry point for direct Flask run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    socketio.run(app, host='0.0.0.0', port=port) 
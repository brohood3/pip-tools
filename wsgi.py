"""
WSGI entry point for Gunicorn
"""

from ai_chat_app import app

if __name__ == "__main__":
    app.run() 
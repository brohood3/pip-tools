#!/bin/bash
# Start script for Render deployment

echo "Starting the Pip AI Chat application..."

# Verify the application files exist
if [ ! -f "ai_chat_app.py" ]; then
  echo "Error: ai_chat_app.py not found!"
  ls -la
  exit 1
fi

if [ ! -f "wsgi.py" ]; then
  echo "Error: wsgi.py not found!"
  ls -la
  exit 1
fi

# Check Python version
python --version

# List current directory contents
echo "Current directory contents:"
ls -la

# Start the application using Gunicorn with the correct module
echo "Starting Gunicorn with wsgi:app..."
exec gunicorn -k eventlet -w 1 wsgi:app --bind 0.0.0.0:$PORT --log-level debug 
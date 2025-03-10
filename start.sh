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
echo "Python version:"
python --version

# Check memory available
echo "Memory information:"
free -m

# Print disk usage
echo "Disk usage:"
df -h

# Check if critical modules are installed
echo "Checking for critical modules..."
python -c "import flask; print(f'Flask version: {flask.__version__}')"
python -c "import openai; print(f'OpenAI version: {openai.__version__}')"
python -c "import gunicorn; print(f'Gunicorn version: {gunicorn.__version__}')"
python -c "import eventlet; print(f'Eventlet version: {eventlet.__version__}')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"
python -c "import siwe; print(f'SIWE version: {siwe.__version__}')"

# Check if PORT is set, default to 8080 if not
if [ -z "$PORT" ]; then
  echo "PORT environment variable not set, defaulting to 8080"
  export PORT=8080
else
  echo "Using PORT: $PORT"
fi

# Set environment variable to disable matplotlib font cache building at startup
export MPLCONFIGDIR=/tmp/matplotlib

# Disable unnecessary warning messages
export PYTHONWARNINGS="ignore::DeprecationWarning"

# Start the application using Gunicorn with the correct module
echo "Starting Gunicorn with SocketIO on port $PORT..."
echo "Using extended timeout of 120 seconds for worker initialization..."

# Use preload to load app once before forking, increase timeout, reduce worker connections
exec gunicorn -k eventlet -w 1 "wsgi:application" \
  --bind 0.0.0.0:$PORT \
  --log-level debug \
  --timeout 120 \
  --graceful-timeout 60 \
  --keep-alive 5 \
  --preload \
  --worker-connections 100 \
  --access-logfile - \
  --error-logfile - 
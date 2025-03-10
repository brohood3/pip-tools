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

# Check if critical modules are installed
echo "Checking for critical modules..."
python -c "import flask; print(f'Flask version: {flask.__version__}')"
python -c "import openai; print(f'OpenAI version: {openai.__version__}')"
python -c "import gunicorn; print(f'Gunicorn version: {gunicorn.__version__}')"
python -c "import eventlet; print(f'Eventlet version: {eventlet.__version__}')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"
python -c "import siwe; print(f'SIWE version: {siwe.__version__}')"

# Start the application using Gunicorn with the correct module
echo "Starting Gunicorn with wsgi:app..."
exec gunicorn -k eventlet -w 1 wsgi:app --bind 0.0.0.0:$PORT --log-level debug 
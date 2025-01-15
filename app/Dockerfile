FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 8000

# Set Python environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Start the application with increased timeout
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout", "300", "--timeout-keep-alive", "300"] 
# Use a base image with Python
FROM python:3.10-slim
# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive
# Install system-level dependencies including ALL Playwright requirements
RUN apt-get update && \
    apt-get install -y build-essential curl \
    libglib2.0-0 libnss3 libnspr4 libdbus-1-3 \
    libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libatspi2.0-0 libxcomposite1 libxdamage1 libxext6 \
    libxfixes3 libxrandr2 libgbm1 libxkbcommon0 \
    libpango-1.0-0 libcairo2 libasound2 \
    libxcb1 libxcb-shm0 libxcb-render0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-shape0 \
    libxcb-sync1 libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 \
    libx11-xcb1 libxkbcommon-x11-0 libgtk-3-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* \
# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
# Add Poetry to the PATH
ENV PATH="/root/.local/bin:${PATH}"
# Set the working directory in the container
WORKDIR /app
# Copy only the essential files to avoid cache invalidation
COPY pyproject.toml poetry.lock ./
# Install project dependencies with Poetry
RUN poetry install --no-dev
# Copy the rest of the project files
COPY . .
# Expose the port your Flask app listens on (e.g., 5000)
EXPOSE 5000
ENV FLASK_APP="app.py"
# Set the entry point for running the Flask app
CMD ["poetry", "run", "flask", "run", "--host=0.0.0.0"]
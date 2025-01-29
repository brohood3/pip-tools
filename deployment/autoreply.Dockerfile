# Use a base image with Python
FROM python:3.10-slim
# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive
# Install system-level dependencies including ALL Playwright requirements
RUN apt-get update && apt-get install -y build-essential curl

RUN curl -sSL https://install.python-poetry.org | python3 - && ls -al /root/.local/bin && \
    /root/.local/bin/poetry --version
# Add Poetry to the PATH
ENV PATH="/root/.local/bin:${PATH}"
# Set the working directory in the container
WORKDIR /app
# Copy only the essential files to avoid cache invalidation
COPY pyproject.toml poetry.lock ./
# Install project dependencies with Poetry
RUN /root/.local/bin/poetry install --no-root
# Copy the rest of the project files
COPY . .
# Set the entry point for running the Flask app
CMD ["poetry", "run", "creator_bid_autoreply.py"]
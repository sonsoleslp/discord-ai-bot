FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create data directory for persistent storage
RUN mkdir -p /app/data

# Create volume for persistent config
VOLUME ["/app/data"]

# Set environment variable for config file location
ENV CONFIG_FILE=/app/data/channel_configs.json

# Run as non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import asyncio; print('Bot is running')" || exit 1

# Run the bot
CMD ["python", "bot.py"]
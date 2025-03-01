FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Create a script to fix imports and run the application
RUN echo '#!/bin/bash\n\
# Create any missing __init__.py files\n\
find /app -type d -exec touch {}/__init__.py \;\n\
\n\
# Run the application while explicitly ignoring __pycache__\n\
cd /app && PYTHONPATH=/app uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir /app/src --reload-exclude __pycache__\n\
' > /app/run.sh && chmod +x /app/run.sh

# Command to run the application
CMD ["/app/run.sh"]

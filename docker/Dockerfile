FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 5000

# Set working directory to API folder
WORKDIR /app/src/api

# Run the application
CMD ["python", "app.py"]

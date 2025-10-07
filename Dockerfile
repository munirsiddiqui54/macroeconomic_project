# Use official Python 3.9 slim image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file
COPY api_requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r api_requirements.txt

# Copy the whole project (src and models) into /app/
COPY src/ ./src/
COPY models/ ./models/

# Expose port 5000
EXPOSE 5000

# Set working directory to where app.py lives
WORKDIR /app/src/api

# Run Flask app
CMD ["python", "app.py"]

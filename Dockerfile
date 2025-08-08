# Use an official slim Python image
FROM python:3.11-slim

# Author label (optional)
LABEL authors="USER"

# Set working directory
WORKDIR /app

# Copy app code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (adjust if needed)
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]

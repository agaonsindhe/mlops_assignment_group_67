# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy application files to the container
COPY src/app.py .
COPY model.pkl .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 6000

# Run the Flask app
CMD ["python", "app.py"]

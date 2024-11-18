# Use the official Python 3.10 slim image as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt (or poetry.lock and pyproject.toml if using Poetry) into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files into the container
COPY . /app/

# Set the environment variable to indicate that the app is running in production
ENV UVICORN_CMD="uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

# Expose the port the app will run on
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["sh", "-c", "$UVICORN_CMD"]

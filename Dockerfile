# Use a Python base image
FROM python:3.9-alpine

# Set working directory
WORKDIR /app

# Copy algorithm files into the container
COPY algorithm.py cities.csv ./

# Install dependencies
RUN pip install numpy

# Command to run the algorithm
CMD ["python", "algorithm.py"]

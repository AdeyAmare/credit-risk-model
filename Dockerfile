# ============================================================
# Dockerfile - Containerize the Credit Risk API
# ============================================================
# This file tells Docker how to build an image of our API.
# An image is like a snapshot of our application that can run anywhere.

# STEP 1: Start with a Python base image
# We use python:3.11-slim because it's small but has everything we need
FROM python:3.11-slim

# STEP 2: Set the working directory inside the container
# All commands after this will run from /app
WORKDIR /app

# STEP 3: Copy the requirements file first
# Docker caches each step, so if requirements.txt doesn't change,
# it won't reinstall packages (faster rebuilds)
COPY requirements.txt .

# STEP 4: Install Python dependencies
# --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# STEP 5: Copy the source code and model
# We copy only what we need to keep the image small
COPY src/ ./src/
COPY models/ ./models/

# STEP 6: Expose port 8000
# This tells Docker that our app listens on port 8000
EXPOSE 8000

# STEP 7: Set the command to run when the container starts
# uvicorn runs our FastAPI app
# --host 0.0.0.0 allows connections from outside the container
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

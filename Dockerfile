# Use Python base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Install dependencies
# COPY Pipfile Pipfile.lock ./ 
# RUN pip install --no-cache-dir pipenv && pipenv install --system --deploy
# RUN pip install exceptiongroup
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install transformers==4.47.0 pillow bitsandbytes accelerate fastapi uvicorn python-multipart

# Copy the application code
COPY ./api /app/api

# Expose the app port
EXPOSE 8000

# Command to start the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

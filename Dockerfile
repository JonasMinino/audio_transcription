FROM python:3.11-slim
WORKDIR /app

# Install necessary packages and Python dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY ./requirements.txt .
COPY ./.asoundrc /root/.asoundrc

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application files
COPY ./app.py .

# Expose the port
EXPOSE 7860

# Define entrypoint and command
ENTRYPOINT ["python"]
CMD ["app.py"]

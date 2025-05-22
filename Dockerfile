FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git build-essential ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]

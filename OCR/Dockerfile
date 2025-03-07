FROM python:3.9-slim
WORKDIR /app
# Install python/pip
USER root
ENV PYTHONUNBUFFERED=1
RUN apt-get update
RUN apt-get -y install zbar-tools build-essential tesseract-ocr python3 python3-dev ffmpeg libsm6 libxext6
RUN python3 -m ensurepip
RUN pip3 install --no-cache --upgrade pip setuptools wheel gunicorn dash_auth
COPY . /app
RUN pip install -r requirements.txt
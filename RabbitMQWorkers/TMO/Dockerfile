FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY TMORabbitMQWorker.py TMORabbitMQWorker.py
EXPOSE 8090 5672 80
CMD [ "python3", "TMORabbitMQWorker.py"]

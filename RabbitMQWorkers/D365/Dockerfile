FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY D365RabbitMQWorker.py D365RabbitMQWorker.py
EXPOSE 5672 80 443
CMD [ "python3", "D365RabbitMQWorker.py"]

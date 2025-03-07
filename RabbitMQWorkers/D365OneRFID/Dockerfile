FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY D365OneRFIDWorker.py D365OneRFIDWorker.py
EXPOSE 5672 80 443
CMD [ "python3", "D365OneRFIDWorker.py"]

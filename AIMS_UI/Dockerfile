FROM python:3.8-slim-buster
WORKDIR /
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY ./ ./
EXPOSE 5672 80 443
CMD [ "python3", "AIMS.py"]

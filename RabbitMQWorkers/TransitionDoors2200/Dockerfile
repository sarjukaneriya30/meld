FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY TransitionDoors2200Worker.py TransitionDoors2200Worker.py
EXPOSE 5672 80 443
CMD [ "python3", "TransitionDoors2200Worker.py"]

FROM python:3.10.7
WORKDIR /app

# Ignore warning about runing pip as the root user as this is an isolated environment.
RUN pip install --upgrade pip

# First round of installations
RUN apt-get update \
    && apt-get install ffmpeg libsm6 libxext6 curl coreutils -y

# Second round: install the Microsoft ODBC driver for SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18

# Copy the library, data, and configuration
COPY . ./

# Install the requirements
RUN pip install -r requirements.txt

# Download the YOLO weights and verify the checksum is correct
RUN curl -o ./data/yolov3-tiny-416/yolo.weights https://pjreddie.com/media/files/yolov3-tiny.weights \
    && sha256sum -b data/yolov3-tiny-416/yolo.weights > yolo.weights.sha256sum \
    && diff yolo.weights.sha256sum.ref yolo.weights.sha256sum \
    && echo "Downloaded YOLO weights successfully"

# Run Physical Instrusion
CMD [ "python3", "startup.py"]

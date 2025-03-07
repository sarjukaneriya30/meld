FROM ubuntu:18.04
COPY . /data
RUN chmod +x /data/prometheus
CMD [ "/data/prometheus", "--config.file=/data/prometheus.yml"]
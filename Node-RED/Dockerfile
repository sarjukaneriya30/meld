FROM nodered/node-red
# Install python/pip
USER root
RUN npm config set fetch-retry-mintimeout 20000
RUN npm config set fetch-retry-maxtimeout 120000
RUN npm install node-red-contrib-amqp-ack python3
RUN apk add --update --no-cache py3-pip python3-dev

# Copy package.json to the WORKDIR so npm builds all
# of your added nodes modules for Node-RED
#Commented out until I figure out how the json should be structured
#COPY package.json .
#RUN npm install --unsafe-perm --no-update-notifier --no-fund --only=production

# Copy _your_ Node-RED project files into place
# NOTE: This will only work if you DO NOT later mount /data as an external volume.
#       If you need to use an external volume for persistence then
#       copy your settings and flows files to that volume instead.
COPY . /data
# Commented out until I have credentials for it.
#COPY flows_cred.json /data/flows_cred.json

# You should add extra nodes via your package.json file but you can also add them here:
#WORKDIR /usr/src/node-red
#RUN npm install node-red-node-smooth
RUN pip install -r /data/BinPacking/requirements.txt
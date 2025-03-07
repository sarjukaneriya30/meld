# version for deployment
############
import os
import sys
import pika
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import base64
import re
import json
import collections
import datetime

#############
from graphCreator8_Libs import *  # custom classes
from WaypointTMO2_lib import *  # custom classes

######################
# load in BAY2Obj8
data_path = './'

picklefile = open(data_path + 'BAY2Obj8', 'rb')
BAY2Obj_loaded = pickle.load(picklefile)
picklefile.close()

###############################
# creat BAY2_WaypointTMO object
BAY2_WaypointTMO_Obj = BAY2_WaypointTMO(name='BAY2_WaypointTMO',BAYs=BAY2Obj_loaded)

##############################
# Connect to RabbitMQ server.
# Note: BlockingConnection blocks the execution thread until the called function returns.
connection = pika.BlockingConnection(pika.ConnectionParameters(host='10.13.0.34', port=5672))
# Connect to 'Waypoint' queue. D365 will send messages to an HTTP endpoint
channel = connection.channel()
channel.queue_declare(queue='ShortestPathWaypoint', durable=True)

# Bind the queue to the exchange to receive MQTT messages.
channel.queue_bind(exchange='amq.topic', queue='ShortestPathWaypoint', routing_key='ShortestPathWaypoint')

#######################################################
# NOTE: (1) the codes above only need to be initialized (for one time).
#       (2) the codes below should be in event handler.
#######################################################

# Define a callback to print out the message received.
def callback(ch, method, properties, body):
    try:
        print("Enter callback function", file=sys.stdout, flush=True)
        MELD_Locs = json.loads(body)

        def order_levels(output):
            '''
            Function to order ALREADY SORTED locations so there is minimal movement up and down of VNA
            '''
            parsed_output = [(o[5:7], o[3:5], o[7]) for o in output]
            both_levels = [item for item, count in collections.Counter([(o[0], o[1]) for o in parsed_output]).items() if count > 1]

            # get indexes in parsed output of all duplicates
            matches = {}
            for idx, loc in enumerate(parsed_output):
                if (loc[0], loc[1]) in both_levels:
                    if (loc[0], loc[1]) not in matches.keys():
                        matches[(loc[0], loc[1])] = [idx]
                    else:
                        matches[(loc[0], loc[1])].append(idx)

            matches = {k:v for k, v in matches.items() if 0 not in v} # remove first loc
            
            # determine if match needs to be reversed; implement
            previously_swapped = []
            for idx, loc in enumerate(parsed_output):
            #     if idx not in new_order:
        #         if (loc[0], loc[1]) not in matches.keys():
        #             print('Not swapping: {}'.format(idx))
                if (loc[0], loc[1]) in matches.keys():
                    group = matches[(loc[0], loc[1])]
                    if group[0] not in previously_swapped:
        #                 print('Maybe swapping: {}'.format(group[0]))
                        first_level = parsed_output[group[0]][2]
                        previous_level = parsed_output[group[0]-1][2]
        #                 print(first_level, previous_level)
                        if first_level != previous_level:
        #                     print('Definitely swapping {} and {}'.format(group[0], group[1]))
                            idx1 = group[1]
                            idx2 = group[0]
                            loc1 = parsed_output[group[1]]
                            loc2 = parsed_output[group[0]]
                            parsed_output[idx2] = loc1
                            parsed_output[idx1] = loc2
                            previously_swapped.append(group[0])
        #                 else:
        #                     print('Not swapping: {}'.format(group[0], group[1]))
            # unparse output
            output = ['I21'+o[1]+o[0]+o[2]+'A' for o in parsed_output]
            return output

        # read in (D365) list of storage locations
        ret_locs = BAY2Obj_loaded.MELD_to_WH_LOCconverter(MELD_Locs=MELD_Locs['Locations'])

        if MELD_Locs['WorkPoolId'] == 'Automated Count':
            print('This is a cycle count job for the WAYPOINT', flush=True)
            requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

            headers = {
                "Accept": "application/json",
                "Authorization": "Basic bmljaG9sYXNjaHJpc3RlbnNlbjozMzQwYTVkYS0zNGUwLTRiM2EtOTE1OS1mMjQyZDM5N2Q0NTI="
            }

            READER_BASE_URL = "https://10.42.4.230")
            print(str(datetime.datetime.now()) + ": READER_BASE_URL: " + READER_BASE_URL, flush=True)
            READER_API_URL = READER_BASE_URL + "/ZoneManager/rfrainapi.php/"

            C_EMAIL = "customer@kpmg.com"
            C_PASSWORD = C_EMAIL + "!"
            C_NAME = "kpmg"

            def get_sessionkey():
                sessionkey = None

                data = {
                    "email": base64.b64encode(bytes(C_EMAIL, "utf-8")),
                    "password": base64.b64encode(bytes(C_PASSWORD, "utf-8")),
                    "cname": C_NAME
                    }

                response = requests.post(READER_API_URL + "get_sessionkey", data, verify=False)

                if response.status_code == 200 and response.headers["Content-Type"] == "application/json;charset=utf-8":
                    response_body = response.json()

                if isinstance(response_body, dict) and "results" in response_body:
                    results = response_body["results"]
                    if isinstance(results, dict) and "sessionkey" in results:
                        sessionkey = results["sessionkey"]

                return sessionkey

            def destroy_sessionkey(sessionkey):
                result = False

                if sessionkey is not None:

                    data = {
                        "sessionkey": sessionkey
                    }

                response = requests.post(READER_API_URL + "destroy_sessionkey", data, verify=False)

                if response.status_code == 200 and response.headers["Content-Type"] == "application/json;charset=utf-8":
                    response_body = response.json()

                if isinstance(response_body, dict) and "success" in response_body:
                    result = response_body["success"]

                return result

            def get_realtime_engine_status(sessionkey):
                status = []

                if sessionkey is None:
                    return status

                data = {
                    "sessionkey": sessionkey
                }

                response = requests.post(READER_API_URL + "get_realtime_engine_status", data, verify=False)

                if response.status_code == 200 and response.headers["Content-Type"] == "application/json;charset=utf-8":
                    response_body = response.json()

                if isinstance(response_body, dict) and "results" in response_body:
                    status = response_body["results"]

                return status

            def set_wp_route(sessionkey, route):

                result = False

                data = {
                    "sessionkey": sessionkey,
                    "routename": json.dumps(route).replace('"', '\"')
                    }

                response = requests.post(READER_API_URL + "set_wp_route", data, verify=False)

                if response.status_code == 200 and response.headers["Content-Type"] == "application/json;charset=utf-8":
                    response_body = response.json()

                if isinstance(response_body, dict) and "success" in response_body:
                    result = response_body["success"]

                return result

            # get a session key
            sessionkey = get_sessionkey()
            if not sessionkey:
                print('No session key obtained.', flush=True)
                sys.exit(1)
            print("Session key obtained.", flush=True)

            # get the current location of the waypoint amr
            status = get_realtime_engine_status(sessionkey)
            engine_status = "Stopped"
            if status and isinstance(status, dict):
                if "status" in status:
                    engine_status = status["status"]
                    print('Engine status: {}'.format(engine_status), flush=True)

            # The set_wp_route endpoint will not work unless Realtime engine is started
            if engine_status == "Stopped":
                destroy_sessionkey(sessionkey)
                sys.exit(1)

            # Extract the starting location
            start_loc = status['current_location']
            print('Start location: {}'.format(start_loc), flush=True)

            # if at the enzone, use the charging station location
            if start_loc == 'Enzone':
                start_loc = 'I223089AA' # approx location of charging station

            # check the format is a real location in bay 2
            loc_format_check = re.findall('.\w.\d{4}\w{2}', start_loc)
            if not loc_format_check:
                print('Current location is formatted incorrectly ({}); using charging station'.format(start_loc), flush=True)
                start_loc = 'I223089AA' # approx location of charging station

            # format starting location for algorithm
            agents = [('agent1',str((int(start_loc[5:7]),int(start_loc[3:5]))))]

            # calcualte waypoint optimal sequence
            ret_waypoint3 = BAY2_WaypointTMO_Obj.waypointOptimalSeq_multiAgents(agents=agents,Items=ret_locs)

            # convert to barcoded location format
            ret_OptMELDLocs = BAY2_WaypointTMO_Obj.Optimal_MELDLocs(MELDlocs=MELD_Locs['Locations'],OptimalAssignments=ret_waypoint3)

            # reshape route
            output = ret_OptMELDLocs['Task1']['locations']
            print('Shortest path algorithm complete; reordering levels...', flush=True)
            output = order_levels(output)
            print('Level ordering complete.', flush=True)
            route = [{'rn':loc} for loc in output] + [{'rn':'Enzone'}]
            print('Reordered route complete:', flush=True)
            print(' -> '.join(output), flush=True)

            # send instructions to waypoint api
            route_results = set_wp_route(sessionkey, route)
            
            if route_results:
                print('Route successfully received by RFRain.', flush=True)
            else:
                print('Error sending route to RFRain; exiting...', flush=True)
                destroy_sessionkey(sessionkey)
                sys.exit(1)

            destroy_sessionkey(sessionkey)
            print(str(route), file=sys.stdout, flush=True)

        if MELD_Locs['WorkPoolId'] == 'AMR': # TMO
            print('This is a cycle count for the VNA', flush=True)
            
            # start loc is VNA charging station
            start_loc = 'I222690AA'
            print('Starting location is the charging station, approximately at {}'.format(start_loc), flush=True)
            
            # format starting location for algorithm
            agents = [('agent1',str((int(start_loc[5:7]),int(start_loc[3:5]))))]

            # calcualte optimal sequence
            ret_waypoint3 = BAY2_WaypointTMO_Obj.waypointOptimalSeq_multiAgents(agents=agents,Items=ret_locs)

            # convert to barcoded location format
            ret_OptMELDLocs = BAY2_WaypointTMO_Obj.Optimal_MELDLocs(MELDlocs=MELD_Locs['Locations'],OptimalAssignments=ret_waypoint3)

            # reshape route
            output = ret_OptMELDLocs['Task1']['locations']
            print('Shortest path algorithm complete; reordering levels...', flush=True)

            output = order_levels(output)
            print('Level ordering complete.', flush=True)
            
            print('Reordered route complete:', flush=True)
            print(' -> '.join(output), flush=True)
            
            print('Output generated, reshaping for TMO...', flush=True)
            # format results for TMO
            tmo_output = {"TransactionID": MELD_Locs["CountOperationId"], 
            "Type": "Cycle", 
            "VehicleID": 0, 
            "VehicleType": "VNA", 
            "Priority": 0, 
            "LoadContainer": "", 
            "LoadContent": "", 
            "LoadIdentifier": "", 
            "Locations": [{"Operation":"Scan", "Name":loc, "Level":0, "Position":0, "Inventory":"System"} for loc in output]}
            print('TMO output ready.', flush=True)

            print(tmo_output, flush=True)
            
            '''
            tmo_output
            '''
            headers = {'Content-Type': 'application/json'}
            response = requests.post("http://10.13.0.34:1880/TMO", headers=headers, json=tmo_output, verify=False)
            


        #####################################
        # # output: send to MELD
        # # Post to HTTP endpoint
        # try:
        #     requests.post('20.141.84.142:8090/Waypoint', data=go_this_way, verify=False)
        # except Exception as e:
        #     print("Error posting to HTTP endpoint: " + e)
        # Send ACK

        channel.basic_ack(delivery_tag = method.delivery_tag)
    except Exception as e:
        print("Error: " + str(e), flush=True)
        channel.basic_ack(delivery_tag = method.delivery_tag)

print("Before consuming")
# Start consuming messages from the queue.
channel.basic_consume(
    queue='ShortestPathWaypoint', on_message_callback=callback, auto_ack=False)

# Start consuming messages from the queue.
print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()

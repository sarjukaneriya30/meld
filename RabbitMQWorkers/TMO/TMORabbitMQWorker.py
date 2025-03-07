import pika, os, sys
import requests
import json

def main():

    # Connect to RabbitMQ server.
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='10.13.0.34'))

    # Connect to 'TMO' queue.
    channel = connection.channel()
    channel.queue_declare(queue='TMO', durable=True)

    # Bind the queue to the exchange to receive MQTT messages.
    channel.queue_bind(exchange='amq.topic', queue='TMO', routing_key='TMO')

    # Define a callback to print out the message received.
    def callback(ch, method, properties, body):
        try:
            print("Attempting to POST to TMO endpoint with body: ", flush=True)
            try:
                print(str(body), flush=True)
            finally:
                print(json.dumps(body), flush=True)
            print("\n")
            # POST the body to an HTTP endpoint
            headers = {'Content-Type': 'application/json'}
            response = requests.post('http://10.13.0.66:8041/TMOService/v2/Tasks', headers=headers, data=body, verify=False, timeout=10)

            print("TMO responded with code " + str(response.status_code))
            print(response.text, flush=True)
            print("\n\n", flush=True)

        except Exception as e:
            print("Error posting to HTTP endpoint: " + str(e), flush=True)

        print("\n\n")
        
        # Send ACK
        ch.basic_ack(delivery_tag = method.delivery_tag)
        print('Received message, forwarded to TMO HTTP endpoint: {}'.format(body), flush=True)

    print("Before consuming", flush=True)
    # Start consuming messages from the queue.
    channel.basic_consume(
        queue='TMO', on_message_callback=callback, auto_ack=False)
    
    # Start consuming messages from the queue.
    print(' [*] TMO worker waiting for messages. To exit press CTRL+C', flush=True)
    channel.start_consuming()

# Run the main function.
if __name__ == '__main__':
    while True:
        try:
            main()
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        except Exception as e:
            print('Error' + str(e))


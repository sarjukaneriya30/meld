import pika, os, sys
import requests

def main():

    # Connect to RabbitMQ server.
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='10.13.0.34'))

    # Connect to 'Attabotics' queue.
    channel = connection.channel()
    channel.queue_declare(queue='Attabotics', durable=True)

    # Bind the queue to the exchange to receive MQTT messages.
    channel.queue_bind(exchange='amq.topic', queue='Attabotics', routing_key='Attabotics')

    # Define a callback to print out the message received.
    def callback(ch, method, properties, body):
        # POST the body to an HTTP endpoint
        print("Received message, forwarding to Attabotics HTTP endpoint: {}".format(body), flush=True)
        requests.post('http://10.17.1.100:8090/api/control', headers={"Authorization": "Basic Og==", "Content-Type": "application/json", "Host": "10.13.0.34"}, data=body, verify=False)
        print("Sent message to D365 HTTP endpoint: {}".format(body), flush=True)

        
        # Send ACK
        ch.basic_ack(delivery_tag = method.delivery_tag)
        print('Received message, forwarded to Attabotics HTTP endpoint: {}'.format(body), flush=True)

    print("Before consuming")
    # Start consuming messages from the queue.
    channel.basic_consume(
        queue='Attabotics', on_message_callback=callback, auto_ack=False)
    
    # Start consuming messages from the queue.
    print(' [*] Attabotics worker waiting for messages. To exit press CTRL+C', flush=True)
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
            print(e, flush=True)


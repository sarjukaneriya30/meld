`   1import pika, os, uuid, datetime, sys
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

def main():
    # Connect to RabbitMQ server.
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='10.13.0.34'))

    # Connect to 'AutonomousScan' queue.
    channel = connection.channel()
    channel.queue_declare(queue='AutonomousScan', durable=True)

    # Bind the queue to the exchange to receive MQTT messages.
    channel.queue_bind(exchange='amq.topic', queue='AutonomousScan', routing_key='AutonomousScan')

    # Define a callback to print out the message received.
    def callback(ch, method, properties, body):
        try:
            # D365 Blob Storage Connection String:
            account_url = "https://sharedaccount.blob.rack01.usmc5gsmartwarehouse.com/"
            container_name = "d365"
            credential = {
                    "account_name": 'sharedaccount',
                    "account_key": 'ecqdYYx8tlBmKJbmjrBKSroK/oeZZz5WbSiM1NjO56yr8SI8TzfdVd9d32kB+X158lMzwicfN4SEB6RSd8q0ZQ=='
                }
            container_client = ContainerClient(account_url = account_url, container_name = container_name, credential = credential, api_version = '2019-02-02')

            # D365 Blob Storage Container Name:
            blob_client = container_client.get_blob_client(blob='AutonomousScan/AutonomousScan' + str(datetime.datetime.now()) + '.txt')
            blob_client.upload_blob(body)
        except Exception as e:
            print(e)
        
        # Send ACK
        ch.basic_ack(delivery_tag = method.delivery_tag)
        print('Received message, stored in blob storage: {}'.format(body))

    # Start consuming messages from the queue.
    channel.basic_consume(
        queue='AutonomousScan', on_message_callback=callback, auto_ack=False)
    
    # Start consuming messages from the queue.
    print(' [*] AutonomousScan worker waiting for messages. To exit press CTRL+C')
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

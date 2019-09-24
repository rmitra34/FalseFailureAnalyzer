"""
Needs subprocess to spawn new Workers and pika for RabbitMQ Handling
"""
import subprocess
import pika
import time

CAP = 15
scale_factor = 5

def queue_creation():
    """
    Sets up the connection to RabbitMQ
    :return: message count inside broker and consumer count inside broker
    """
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', heartbeat=0))
    channel = connection.channel()
    queue = channel.queue_declare(queue='execution', durable=True, exclusive=False, auto_delete=False, passive=True)
    return queue.method.message_count, queue.method.consumer_count


def scaling():
    """
    Increases the amount of workers based on how long the queue is.
    """
    time.sleep(2)
    message_count, consumer_count = queue_creation()
    # Must be same value as prefetch count for workers, can set it to whatever you want to
    new_workers = message_count / scale_factor
    new_workers = int(new_workers)
    new_workers = new_workers - consumer_count
    if new_workers > 0:
        i = 0
        while consumer_count < CAP and i < new_workers:
            daemon_deploy()
            consumer_count += 1
            i += 1


def daemon_deploy():
    """
    Spawns new worker when called
    """
    print('in deploy method')
    subprocess.call(['python3 ExecIDQueue.py &'], shell=True)

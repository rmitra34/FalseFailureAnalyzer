import pika
import uuid
import DeployDaemon
import SQLConnector
from pymemcache.client import base
import time

MEM_CACHE = base.Client(('/tmp/memcached.sock'))
FETCH_COUNT = 5


def send_token(token):
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    unique_identifier = str(uuid.uuid4())
    if token.isdigit():
        remote_path = SQLConnector.debug(token)
        MEM_CACHE.set(unique_identifier, [remote_path, token])
    else:
        remote_path = token
        MEM_CACHE.set(unique_identifier, [remote_path])

    queue = channel.queue_declare(queue='execution', durable=True, exclusive=False, auto_delete=False)
    channel.basic_publish(exchange='', routing_key='execution', body=unique_identifier,
                            properties=pika.BasicProperties(delivery_mode=2))
    if queue.method.consumer_count <= 0:
        DeployDaemon.daemon_deploy()
    DeployDaemon.scaling()
    connection.close()
    return " [x] Sent %r" % unique_identifier


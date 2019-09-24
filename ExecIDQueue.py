"""
Imports all modules used for parsing logfiles and then feeding into testModel.
"""

import re
import os
from datetime import datetime
from ML_Model import Model
import pika
import warnings
from pymemcache.client import base
import csv
import ast
import data_ingestion_optimized
from collections import OrderedDict

warnings.filterwarnings("ignore")

# paramiko.util.log_to_file('/tmp/paramiko.log')

# timeout = 60

LABEL_MAPPING = {
    0: 'Hardware',
    1: 'Other',
    2: 'Script',
    3: 'Software',
    4: 'Tools',
}

MEMORY_CACHE = base.Client(('/tmp/memcached.sock'))


def on_visit(client):
    """
    Way to solve concurrency issue by using a counter and using a memoryClient
    :param client:
    """
    while True:
        result, cas = client.gets('visitors')
        result = int(result)
        result += 1
        if client.cas('visitors', result, cas):
            break


RE_LIST = [
    r'(Jan?|Feb?|Mar?|Apr?|May|Jun?|Jul?|Aug?|Sep?|Oct?|Nov?|Dec?)\s+\d{1,2}\s+',
    r'\d{2}:\d{2}:\d{2}\s+',
    r'[\[].*?[\]]',
    r'<\/?data>',
    r'<\/?value>',
    r'<\/?valueType>',
    r'<\/?name>',
    r'<\/?valueUnit>',
    r'<\/?object-value>',
    r'<\/?object-value-type>',
    r'<\/?cli>',
    r'<[0-9]{4,5}>',
    r'<\/?DATA>'
]


def output_prediction(text, model_file):
    print('Testing...')
    model = Model()
    model.load_model(model_file)
    prob = model.get_predict_prob(text)
    indices = list(range(5))
    # return LABEL_MAPPING[prob]
    prob = prob.tolist()
    max_value = max(prob)
    max_index = prob.index(max_value)
    prob[max_index] = 0
    indices.remove(max_index)
    second_value = max(prob)
    second_index = prob.index(second_value)
    indices.remove(second_index)
    output_bucket = OrderedDict()
    output_bucket[LABEL_MAPPING[max_index]] = max_value
    output_bucket[LABEL_MAPPING[second_index]] = second_value
    for i in indices:
        output_bucket[LABEL_MAPPING[i]] = prob[i]
    return output_bucket


def callback(ch, method, body):
    """
    Callback function when worker receives a message to be analyzed
    :param ch: Channel through which worker is operating
    :param method: More information and qualities about the message
    :param body: The actual message itself.
    """
    print(" [x] Received %r" % body)
    # time.sleep(body.count(b'.'))
    ch.basic_ack(delivery_tag=method.delivery_tag)
    output = MEMORY_CACHE.get(body.decode())
    output = ast.literal_eval(output.decode())
    remote_path = output[0]
    logpath_set = ast.literal_eval(MEMORY_CACHE.get('past_logpaths').decode())
    output_fieldnames = ['unique_identifier', 'query_message']
    identifier_bucket = dict()
    if remote_path in logpath_set:
        prev_identifer = logpath_set[remote_path]
        prev_output = MEMORY_CACHE.get(prev_identifer)
        MEMORY_CACHE.set(body.decode(), prev_output)
        identifier_bucket['unique_identifier'] = body.decode()
        identifier_bucket['query_message'] = prev_output
    else:
        logpath_set[remote_path] = body.decode()
        MEMORY_CACHE.set('past_logpaths', logpath_set)
        text = data_ingestion_optimized.read_sample(remote_path)
        if text != 'Path Not Accessible' and text is not None:
            model_list = os.listdir('models')
            datetime_list = []
            for model in model_list:
                time_string = model.rfind('_')
                time_str = model[0:time_string]
                date_time = datetime.strptime(time_str,  "%Y_%b_%d_%H_%M")
                datetime_list.append(date_time)
            recent_model = max(datetime_list)
            recent_model = recent_model.strftime("%Y_%b_%d_%H_%M")
            output_bucket = output_prediction(text, recent_model)
            on_visit(MEMORY_CACHE)
            update_list = ast.literal_eval(MEMORY_CACHE.get(body.decode()).decode())
            update_list.append(output_bucket)
        else:
            on_visit(MEMORY_CACHE)
            identifier_bucket = dict()
            update_list = ast.literal_eval(MEMORY_CACHE.get(body.decode()).decode())
            update_list.append('PATH NOT FOUND')
        MEMORY_CACHE.set(body.decode(), update_list)
        identifier_bucket['unique_identifier'] = body.decode()
        identifier_bucket['query_message'] = update_list
    with open('database.csv', 'a') as outfile:
        w = csv.DictWriter(outfile, fieldnames=output_fieldnames)
        w.writeheader()
        w.writerow(identifier_bucket)


CONNECTION = pika.BlockingConnection(pika.ConnectionParameters('localhost', heartbeat=0))
CHANNEL = CONNECTION.channel()

CHANNEL.queue_declare(queue='execution', durable=True, exclusive=False, auto_delete=False, passive=True)
CHANNEL.basic_qos(prefetch_count=5)

for method_, prop_, body_ in CHANNEL.consume('execution'):
    callback(ch=CHANNEL, method=method_, body=body_)
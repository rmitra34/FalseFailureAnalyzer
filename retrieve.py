"""
Imports all modules to retrieve from memory cache or database
"""
import csv
import json
from pymemcache.client import base


MEMORY_CACHE = base.Client(('/tmp/memcached.sock'))


def retrieve_prediction(id):
    """
    Takes arguments from cli and retrieves predictions
    """
    return check_cache(id)


def check_cache(message):
    """
    First check memory cache and see if it contains results, if not check database
    :param message: check if given message in memcache
    """
    result = MEMORY_CACHE.get(str(message))
    if result is None:
        result = check_database(message)
    return result


def check_database(message):
    """
    Check DB for given message
    :param message: check if given message in database
    """
    reader = csv.DictReader(open('database.csv'), fieldnames=['unique_identifier', 'query_message'])
    for row in reader:
        clean_row = json.loads(json.dumps(row))
        if str(message) == str(clean_row['unique_identifier']):
            return 'Database Query:' + clean_row['query_message']
    return 'PREDICTION NOT FOUND'



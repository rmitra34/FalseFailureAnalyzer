"""
Needs requests and json to retrieve and unpack database query.
"""
import requests
import base64
import json


def debug(execid):
    """
    Given a test execution ID goes into EDW and returns logpath of said Execution ID
    Keyword Arguments: execid - test execution id given
    """
    username = 'rishabm'
    password = 'Juniper16$'
    teradata_database_alias = 'CBR_RESTAPI'

    # HTTP
    url = 'https://edwrestapi.juniper.net/tdrest/systems/' + teradata_database_alias + '/queries'
    headers = {}
    headers['Content-Type'] = 'application/json'
    headers['Accept'] = 'application/vnd.com.teradata.rest-v1.0+json'
    headers['Authorization'] = 'Basic %s' % \
                               base64.encodestring(("%s:%s" % (username, password)).encode()).decode().replace('\n', '')

    queryBands = {}
    queryBands['applicationName'] = 'CBR_RESTAPI'  # 'MyApp'
    queryBands['version'] = '1.0'

    data = {}

    data['query'] = 'SELECT * FROM dr_er_debug_exec WHERE test_exec_id = {}'.format(str(execid))
    data['queryBands'] = queryBands
    data['format'] = 'array'

    response = requests.request(method='POST', url=url, headers=headers, data=json.dumps(data), verify=False)
    results = response.json()
    try:
        logpath = results['results'][0]['data'][0]
        return logpath[3]
    except IndexError:
        return 'Not valid input'
    except TypeError:
        return 'Not valid input'

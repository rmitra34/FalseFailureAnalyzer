from flask import Flask, request
from pymemcache.client import base
import retrieve
import send
import flask
from ml_analyzer import Model

MEM_CACHE = base.Client(('/tmp/memcached.sock'))
MEM_CACHE.set('visitors', 0)
MEM_CACHE.set('past_logpaths', dict())
app = Flask(__name__)


@app.route('/')
def starter():
    return flask.render_template('formexecution.html')


@app.route('/<token>', methods=['GET'])
def get_token_execid(token):
    token = str(token)
    return str(retrieve.retrieve_prediction(token))


@app.route('/postprediction', methods=['POST'])
def post_token():
    token = str(request.form['param'])
    message = token
    return send.send_token(token)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

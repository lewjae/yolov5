from functools import wraps
import json
import jwt
import os

from flask.globals import request
from config import Config

config = Config.get_instance()

# middleware - checks for remote connection and handels authentication
def token_check():
    def _token_check(f):
        @wraps(f)
        def __token_check(*args, **kwargs):
            token = None

            # check environment variable for secure gate
            if not config.get('jwt_secure') == 'True':
                return f(*args, **kwargs)

            # allow through any request from the same host (local)
            if(request.remote_addr == "127.0.0.1" or request.remote_addr == "0.0.0.0"):
                return f(*args, **kwargs)

            if 'X-Access-Token' in request.headers:
                token = request.headers['X-Access-Token']

            if not token:
                return json.dumps({ 'message': 'a valid token is missing '}), 401
            
            try:
                jwt.decode(token, config.get('jwt_secret'), algorithms=['HS256'])
            except:
                return json.dumps({ 'message': 'token is invalid' }), 401
            
            return f(*args, **kwargs)
        return __token_check
    return _token_check

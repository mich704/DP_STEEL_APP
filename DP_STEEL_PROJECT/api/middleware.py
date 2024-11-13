# api/middleware.py

import logging

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.method == 'GET':
            pass
            #logger.info(f"GET request to: {request.get_full_path()}")
            #logger.info(f"Request headers: {request.headers}")

        response = self.get_response(request)
        return response
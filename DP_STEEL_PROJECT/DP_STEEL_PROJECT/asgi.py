# asgi.py

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django.urls import path
from api import consumers  # Import your consumers

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')  # Replace 'myproject'

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter([
            path("ws/microstructure_analyser/<str:session_id>/", consumers.MAConsumer.as_asgi()),  # Example WebSocket route
            path("ws/extract_images/<str:session_id>/", consumers.ExtractorConsumer.as_asgi()),  # Example WebSocket route
        ])
    ),
})

import json
from channels.generic.websocket import AsyncWebsocketConsumer

class MyConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        await self.channel_layer.group_add(
            self.session_id,
            self.channel_name
        )
        await self.accept()  # Accept the WebSocket connection

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.session_id,
            self.channel_name
        )

    async def receive(self, text_data):
        # Handle receiving data from WebSocket
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        
        # Echo the message back to the WebSocket
        await self.send(text_data=json.dumps({
            'message': message + ' (echoed back to you)'
        }))

    async def progress_update(self, event):
        message = event['message']
        
        # Send progress update to WebSocket
        await self.send(text_data=json.dumps({
            'message': message
        }))

class MAConsumer(MyConsumer):
    pass

class ExtractorConsumer(MyConsumer):
    async def receive(self, text_data):
        # Handle receiving data from WebSocket
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        
        # Echo the message back to the WebSocket
        await self.send(text_data=json.dumps({
            'message': message + ' EXTRACTOR (echoed back to you)'
        }))
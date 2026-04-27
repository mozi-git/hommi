import base64
import struct
from flask import Flask
from flask_socketio import SocketIO
from argparse import ArgumentParser
import numpy as np

# Create Flask app and SocketIO server
app = Flask(__name__)
socketio = SocketIO(app)

def decode_pose(data, label):
    try:
        payload = base64.b64decode(data)
        matrix = struct.unpack('<16f', payload[:64])
        timestamp = struct.unpack('<d', payload[64:72])[0]
        pose = np.array(matrix).reshape(4, 4).T
        print(f"{label} pose received at {timestamp:.3f}s:")
        print(pose)
    except Exception as e:
        print(f"Failed to decode {label} pose: {e}")

@socketio.on('connect')
def on_connect():
    print('Client connected')

@socketio.on('disconnect')
def on_disconnect():
    print('Client disconnected')

@socketio.on('updateLeft')
def handle_left(data):
    decode_pose(data, "Left")

@socketio.on('updateRight')
def handle_right(data):
    decode_pose(data, "Right")

@socketio.on('updateHead')
def handle_head(data):
    decode_pose(data, "Head")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--host', default='10.130.20.101', help='Host IP to bind the server')
    parser.add_argument('--port', type=int, default=5555, help='Port to bind the server')
    args = parser.parse_args()

    print(f"Starting pose server at {args.host}:{args.port}...")
    socketio.run(app, host=args.host, port=args.port, allow_unsafe_werkzeug=True)

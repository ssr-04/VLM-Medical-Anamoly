from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import base64
import cv2
from inference import main
from flask_cors import CORS
import socket
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def send_ip_address(ip_address):
  if ip_address:
      url = "https://med-vlm.onrender.com/url"
      data = {"url": ip_address}
      
      try:
          response = requests.post(url, json=data)
          if response.status_code == 200:
              print("IP address sent successfully.")
          else:
              print(f"Failed to send IP address. Status code: {response.status_code}")
      except Exception as e:
          print(f"Error sending IP address: {e}")
  else:
      print("Invalid IP address. Cannot send.")

def get_ip_address():
    try:
        # Create a socket connection to retrieve the IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to a remote server (doesn't actually send data)
        s.connect(("8.8.8.8", 80))
        # Get the IP address of the local machine
        ip_address = s.getsockname()[0]
        s.close()
        print("IP address retrieved")
        return ip_address
    except Exception as e:
        print(f"Error getting IP address: {e}")
        return None

send_ip_address(get_ip_address())

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        print("Got Get request at /")
        return jsonify({'message': 'Welcome! Check API at /api'}), 200
    elif request.method == 'POST':
        return jsonify({'error': 'POST method not allowed on this endpoint'}), 405

@app.route('/api', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        print("File error at /api post : No file")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("File error at /api post")
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Open image using Pillow
        file.save(file.filename)
        print("Image received and saved successfully")
        report = main(file.filename)
        img = Image.open("./anomaly_heatmap_overlay.png")
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Encode image in base64
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        print(report)
        # Prepare response data
        response_data = {
            'data': report,
            'image': img_base64
        }

        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("started")
    app.run(host='0.0.0.0', port=8000)

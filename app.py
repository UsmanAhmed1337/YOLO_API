from flask import Flask, request, jsonify
from flask_cors import CORS
from YOLO import load_model, inference_image, inference_video
import json
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

model_path = './best.pt'  
model = load_model(model_path)

@app.route('/')
def hello_world():
    return 'Welcome! The endpoint is up and running.'


@app.route('/inference_image', methods=['POST'])
def api_inference_image():
    try:
        if not request.content_type.startswith('multipart/form-data'):
            return jsonify({'error': 'Invalid content type'}), 400
        
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))
        bounding_boxes = inference_image(model, image)
        return jsonify(bounding_boxes)
    
    except Exception as e:
        print(e)
        return jsonify({'error': f'An error occurred: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)

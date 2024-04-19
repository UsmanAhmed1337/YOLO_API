from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from YOLO import load_model, inference_image, inference_video
import json
from PIL import Image
import io

app = Flask(__name__)
CORS(app)


model_path = './best.pt'
model = load_model(model_path)

@app.route('/', methods=['GET'])
def hello_world():
    return 'Welcome! The endpoint is up and running.'


@app.route('/inference_image', methods=['POST'])
def api_inference_image():

    try:
        if not request.content_type.startswith('multipart/form-data'):
            return jsonify({'error': 'Invalid content type'}), 400
        
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))
        inference_image(model, image)
        return send_file('img.jpg', mimetype='image/jpeg')
    
    except Exception as e:
        print(e)
        return jsonify({'error': f'An error occurred: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True)

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize

import torch
from flask import Flask, request, send_file, Response
import flask_helpers as fh
from io import BytesIO
import base64
from PIL import Image
from json import dumps


MODEL, CLASS_NAMES = fh.get_default_model()

# Flask setting
app = Flask(__name__)
UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello():
    response_string = """Welcome to Mask R-CNN
    Endpoints:
    POST /changemodel - For selecting which model is running
        Json Body with:
	        "modelName": "name of model",
	        "modelUrl": "public endpoint to download",
	        "classNames": array of strings for the class names

    POST /visualize - Returns image with masks drawn on
        Multipart form data with the file
    
    POST /base64 - Accepts base64 encoded string image and returns an array of base 64 encoded strings
        Json Body with:
            "base64Image": "base64 encoded image data"
    """
    return response_string

@app.route("/changemodel", methods=['POST'])
def change_model():
    global MODEL
    global CLASS_NAMES
    data = request.get_json()
    model_name = data.get('modelName')
    model_url = data.get('modelUrl')
    class_names = data.get('classNames')

    MODEL, CLASS_NAMES = fh.set_model(model_name, model_url, class_names)

    return "Successfully updated model to {}".format(model_name)

@app.route("/visualize", methods=['POST'])
def return_visualized_image():
    # Get image from request and change to array
    image = fh.image_from_request(request)
    image = fh.image_to_array(image)

    # Run detection
    results = MODEL.detect([image])
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                             CLASS_NAMES, r['scores'])

    buf = BytesIO()
    plt.savefig(buf, format='jpg')

    response = Response()
    response.set_data(buf.getvalue())
    response.headers['Content-Type'] = 'image/jpeg'
    return response


@app.route('/base64', methods=['POST'])
def mask_base64_objects():
    # Get the image data from json 'base64Image'
    data = request.get_json()
    base64_image = data.get('base64Image')

    # Convert to image
    full_im = Image.open(BytesIO(base64.b64decode(base64_image)))
    img_format = full_im.format

    # Converts to array with only RGB channels
    image = fh.image_to_array(full_im)

    # Run detection
    results = MODEL.detect([image])
    r = results[0]

    # Get the outputs
    outputs = fh.extract_bounding_boxes(image, r)
    #fh.save_images_locally(outputs)
    output_strings = fh.outputs_to_base64(outputs, img_format)

    # Return response in json format
    return Response(dumps({'croppedImageList': output_strings, 'model': MODEL.config.NAME}), mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
import os

# For Mask RCNN
import torch
import coco
import utils
import model as modellib
import visualize
from config import Config

# For processing images
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import wget

# Global vars
ENCODING = 'utf-8'

def extract_bounding_boxes(image, results, apply_mask=True):
    """Slices regions defined by bounding box into separate images
    Inputs:
        image: np.array, original image
        results: results object, results of maskrcnn
        apply_mask: boolean, toggles if the mask should be applied
    Output:
        output_images: list of np.arrays, image segments
    """
    bounding_boxes = results['rois']
    output_images = []
    
    try:
        num_images = bounding_boxes.shape[0]
        for ix in range(num_images):
            cur_image = image.copy()
            if apply_mask:
                cur_mask = np.squeeze(results['masks'][:,:,ix])
                cur_image[cur_mask==0,:] = 255
            
            #bbox array [num_instances, (y1, x1, y2, x2)].
            bb = bounding_boxes[ix,:]
            cur_out = cur_image[bb[0]:bb[2], bb[1]:bb[3]]
            
            output_images.append(cur_out)

    except Exception as e:
        print(e)
    
    return output_images

def save_images_locally(output_images):
    """Saves the results of extract_bounding_boxes locally
    Inputs:
        output_images: list of np.arrays, image segments
    """

    for ix in range(len(output_images)):
        cur_img = Image.fromarray(output_images[ix])
        save_name = "result_{}.jpg".format(ix)
        cur_img.save(save_name)

def outputs_to_base64(output_images, img_format):
    """Converts the output images to a list of base64 strings
    Input:
        output_images: list of np.arrays, image segments
        img_format: string, format of original image
    Output:
        output_strings: list of string, base64 encoded images
    """
    output_strings = []
    # Encode back to base64 to send back to ImageProcessing function
    for ix in range(len(output_images)):
        cur_image = Image.fromarray(output_images[ix])

        cropped_img_bytes = BytesIO()
        cur_image.save(cropped_img_bytes, format=img_format)
        cropped_img_bytes = cropped_img_bytes.getvalue()
        base64_bytes = base64.b64encode(cropped_img_bytes)

        base64_string = base64_bytes.decode(ENCODING)
        output_strings.append(base64_string)
    
    return output_strings

def get_default_model():
    """Loads the coco classifier as the default model
    Input:
        none
    Outputs:
        model: mask rcnn model, the coco dataset trained model for predictions
        class_names: list of strings, the class names for the model prediction classes
    """
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        # GPU_COUNT = 0 for CPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()

    if not torch.cuda.is_available():
        config.GPU_COUNT = 0
    config.display()

    # Create model object.
    model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
    if config.GPU_COUNT:
        model = model.cuda()

    # Load weights trained on MS-COCO
    model.load_state_dict(torch.load(COCO_MODEL_PATH))

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']
    
    return model, class_names

def set_model(model_name, model_url, class_names):
    """Changes that model being used for the endpoint
    Inputs:
        model_name: string, name of the model to use
        model_url: string, url that the model can be downloaded from
        class_names: list of strings, the names for the classes
    Outputs:
        model: Mask RCNN model, model that predictions are made with
        class_names: list of strings, the names for the classes
    """
    # Directory name setting
    root_dir = os.getcwd()
    model_dir = os.path.join(root_dir, "logs")

    if not os.path.exists(model_name):
        model_path = wget.download(model_url)
    else:
        model_path = model_name

    config = CustomInferenceConfig()
    config.COCO_MODEL_PATH = model_path
    config.NUM_CLASSES = len(class_names)
    config.NAME = model_name

    if not torch.cuda.is_available():
        config.GPU_COUNT = 0
    config.display()

    model = modellib.MaskRCNN(config=config, model_dir=model_dir)
    if config.GPU_COUNT:
        print('Has Cuda')
        model = model.cuda()

    # Load weights trained on MS-COCO
    model.load_weights(model_path)

    return model, class_names

class CustomInferenceConfig(coco.CocoConfig):
    """Derives from the base Config class and overrides some values."""
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 4

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.90
    
    # Necessary for docker image to optimize memory usage best
    NUM_WORKERS = 0

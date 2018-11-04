import numpy as np
from PIL import Image

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
    num_images = bounding_boxes.shape[0]
    output_images = []

    for ix in range(num_images):
        cur_image = image.copy()
        if apply_mask:
            cur_mask = np.squeeze(results['masks'][:,:,ix])
            cur_image[cur_mask==0,:] = 255
        
        #bbox array [num_instances, (y1, x1, y2, x2)].
        bb = bounding_boxes[ix,:]
        cur_out = cur_image[bb[0]:bb[2], bb[1]:bb[3]]
        
        output_images.append(cur_out)
    
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
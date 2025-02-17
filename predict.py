from PIL import Image, TiffImagePlugin
import torch
import model
import cv2 
import numpy as np

Image.MAX_IMAGE_PIXELS = None


def model_predict_mask(dapi, model_INPUT_WIDTH, model_INPUT_HEIGHT, remote_model):
    assert type(dapi) == TiffImagePlugin.TiffImageFile or type(dapi) == Image.Image, f'Please use a PIL.TiffImagePlugin.TiffImageFile, not {type(dapi)}'
    # return the model (already on the device, GPU or CPU), plus the device
    mod, device = model.get_model(remote_path=remote_model)
    transformer = model.get_transformer(model_INPUT_WIDTH,model_INPUT_HEIGHT)
    # Apply transformer to dapi Image
    dapi_tensor = transformer(dapi).to(device)
    
    
    with torch.no_grad():
        out = mod(dapi_tensor.unsqueeze(0))
    
    mask = torch.sigmoid(out.squeeze()).cpu().numpy()
    
    # This returns a floating point mask / before it's been thresholded
    return mask

def get_gated_mask_by_confidence(mask, confidence_threshold):
    assert mask.dtype == np.float32, 'The mask must be a numpy array of type float32'
    
    gated_mask = (mask > confidence_threshold) * 255
    
    return gated_mask.astype(np.uint8)
    

def get_bounding_rects(mask):
    assert mask.dtype == np.uint8, 'The mask must be a numpy arrray of type uint8'
    # Thresholding image and finding external contours only
    ret, threshold = cv2.threshold(mask, 127,255,0)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = []
    scores = []
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        rects.append([x,y,x+w,y+h])
        scores.append(w*h)
    
    # Applying Non Maxmimum Supression, to eliminate boxes inside boxes
    new_rect_list = cv2.dnn.NMSBoxes(rects,scores,np.mean(scores) + np.std(scores),0.9)
    return [tuple(rects[nr]) for nr in new_rect_list]



def get_rois_model(img, CONFIDENCE_THRESHOLDS, model_INPUT_WIDTH, model_INPUT_HEIGHT, remote_model):
    assert type(img) == TiffImagePlugin.TiffImageFile or type(img) == Image.Image, f'Please use a PIL.TiffImagePlugin.TiffImageFile, not {type(img)}'
    
    # Predict Mask
    mask = model_predict_mask(img, model_INPUT_WIDTH, model_INPUT_HEIGHT,remote_model)
    # create semantic wrapper for the rois, conf values
    p_rois = []

    # Iterate over the confidence thresholds
    for conf in CONFIDENCE_THRESHOLDS:
        # create semantic wrapper for the roi List
        temp_rois = []
        # Make prediction
        gated_mask = get_gated_mask_by_confidence(mask,conf)

        # Get RegionsOfInterest for each Core
        for rect in get_bounding_rects(gated_mask):
            temp_rois.append(
                dict(
                x1=rect[0],
                y1=rect[1],
                x2=rect[2],
                y2=rect[3],
                img_w=model_INPUT_WIDTH,
                img_h=model_INPUT_HEIGHT
            )
        )

        p_rois.append(dict(
            confidence_value=conf,
            rois=temp_rois
        ))
        

    return p_rois
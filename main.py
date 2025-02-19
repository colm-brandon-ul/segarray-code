import argparse
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import os
Image.MAX_IMAGE_PIXELS = None
from tifffile import TiffFile



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--train', type=bool, default=False)
    args.add_argument('--name', type=str, default='default')
    args.add_argument('--lr', type=float, default=0.01, required=False)
    args.add_argument('--batch_size', type=int, default=32, required=False)
    args.add_argument('--epochs', type=int, default=10, required=False)
    args.add_argument('--model_input_width', type=int, default=1024, required=False)
    args.add_argument('--model_input_height', type=int, default=1024, required=False)

    # Inference only
    args.add_argument('--img', type=str, default='input.png', required=False)
    # preprocess the image with contrast enhancement
    args.add_argument('--preprocess', type=bool, default=False, required=False)
    args.add_argument('--contast_factor', type=float, default=2, required=False)
    args.add_argument('--page_number', type=int, default=0, required=False)
    args.add_argument('--downsample_factor', type=int, default=5, required=False)
    
    args.add_argument('--remote_model', type=str, default="https://drive.google.com/uc?export=download&id=1pfVjit0_EO1w3J41EmISdXYZ549y60e0", required=False)
    args.add_argument('--confidence_thresholds', type=list, default=[0.5,0.7,0.9,0.99,0.999,0.9999,0.99999], required=False)
    args.add_argument('--output_dir', type=str, default='output', required=False)

    args = args.parse_args()

    if args.train:
        ...
    else:
        # inference only
        from predict import get_rois_model

        # check if it's a tiff

        # regex for tiff files (include qptiff, etc..)
        

        if args.img.endswith('tif') or args.img.endswith('tiff'):

            with TiffFile(args.img) as tif:
                is_multipage = len(tif.pages) > 1
                if is_multipage:
                    img = tif.pages[args.page_number]
                    img = img.asarray()
                    img = Image.fromarray(img)
                    img = img.resize((img.width // args.downsample_factor, img.height // args.downsample_factor))
                else:
                    img = Image.open(args.img)
                    img = img.resize((img.width // args.downsample_factor, img.height // args.downsample_factor))

        else:
            # load image
            img = Image.open(args.img)
        if args.preprocess:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(args.contast_factor)

        source_w, source_h = img.size

        p_rois = get_rois_model(img, args.confidence_thresholds, args.model_input_width, args.model_input_height, args.remote_model)


        for p_roi in p_rois:
            # draw bounding box over image
            cval = p_roi.get('confidence_value')
            rois = p_roi.get('rois')

            img_copy = img.copy()

            roi_line_width_multuplier = img.size[0] // args.model_input_width

            # draw the bounding boxes over image
            for roi in rois:
                x1,y1,x2,y2,img_w,img_h = roi.get('x1'),roi.get('y1'),roi.get('x2'),roi.get('y2'),roi.get('img_w'),roi.get('img_h')
                # tranlate the bounding box to the original image size
                x1 = int(x1 * source_w / img_w)
                x2 = int(x2 * source_w / img_w)
                y1 = int(y1 * source_h / img_h)
                y2 = int(y2 * source_h / img_h)

                # convert to cv2 image
                img_copy = cv2.cvtColor(np.array(img_copy), cv2.COLOR_RGB2BGR)
                img_copy = cv2.rectangle(img_copy, (x1,y1), (x2,y2), (255,255,255), 2*roi_line_width_multuplier)
            
            # convert back to PIL image
            img_copy = Image.fromarray(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
            # if output_dir does not exist, create it
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            # strip path from input
            img_name = os.path.basename(args.img)
            img_name = img_name.split(".")[0]

            img_copy.save(f'{args.output_dir}/{img_name}_output_{cval}.png')

        



        



from PIL import Image 
import numpy as np 
import cv2

def crop_bounding_boxes(img,boxes) : 
    images = []
    for i in range(len(boxes)): 
        x0 = boxes[i][0]
        y0 = boxes[i][1]
        x1 = boxes[i][2]
        y1 = boxes[i][3]
        area = (x0,y0,x1,y1) 
        cropped  = img[int(y0.item()):int(y1.item()),int(x0.item()):int(x1.item()),:]

        opencv_image = np.array(cropped) 
        opencv_image = opencv_image[:,:,::-1].copy()
        images.append(opencv_image) 
    return images

import torch
import sys
sys.path.insert(0, './yolo') 
import os 
#yolo
from yolo.detect import detect 
import pathlib
import yolo.models 
from RektNet.keypoint_detect import keypoint_detect 
import argparse
from PIL import Image, ImageDraw, ImageFont
from bounding_box_crop import crop_bounding_boxes 
import cv2 
import numpy as np 
import time

##RektNet weights 
weights_rekt = "loss_0.22.pt" 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_path', type=str, default = "inference/", help='path to target images for processing')
    parser.add_argument('--output_path', type=str, default="outputs/")
    parser.add_argument('--weights_path', type=str, default="best_small.pt", help='path to weights file')
    

    #car contour 
    car_contours = [np.array([[40,1200],[54,1180],[291,1129],[630,1095],[696,983],[728,962],[811,938],[845,937],[843,883],[855,883],[858,942],[947,959],[979,969],[996,982],[1121,1091],[1186,1113],[1292,1113],[1431,1131],[1600,1135],[1600,1200]], dtype=np.int32) ]
    edge_contours = [np.array([[0,300],[0,1200],[1600,1200],[1600,300],[1580,300],[1580,1180],[20,1180],[20,300]], dtype=np.int32) ]

  
    
    #RektNet
    parser.add_argument('--img', help='path to single image', type=str, default="gs://mit-dut-driverless-external/ConeColourLabels/vid_3_frame_22063_0.jpg")
    parser.add_argument('--img_size', help='image size', default=80, type=int)
    parser.add_argument('--output', help='path to upload the detection', default="outputs/visualization/")
    opt = parser.parse_args()
    
    a = os.listdir(opt.target_path) 
    for f in a : 
        original_image = cv2.imread(opt.target_path + f) 
        boxes, img = detect(opt.weights_path,opt.target_path + f)
        h, w = original_image.shape[0], original_image.shape[1]
        box_imgs = crop_bounding_boxes(img,boxes)
             
        count = 0 
        arr_discard = [] #array for discarding elements
        for b in boxes : 
            x0 = int(b[0])
            y0 = int(b[1]) 
            x1 = int(b[2]) 
            y1 = int(b[3])
            c = int(b[4])
            start = time.time()
            #car edges 
            inside_car = []
            pts = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
            inside = 0 
            for pt in pts : 
                   inside1 = cv2.pointPolygonTest(car_contours[0],pt,False)
                   
                   dist1 = cv2.pointPolygonTest(car_contours[0],pt,True)
                   print("dist1",dist1) 
                   if inside1 == 1 and dist1 < 10 : 
                       inside1 = -1.0 

                   inside2 = cv2.pointPolygonTest(edge_contours[0],pt,False)
                   print(inside1,inside2) 
                   if inside1 == 1 or inside2 == 1 or inside1 == 0 or inside2 == 0: 
                       inside = 1
                       arr_discard.append(count) 
                       break
            count += 1 
            print("contour check takes",(time.time() - start) * 1000) 
            color = (255,0,0)
            if inside == 1: 
                continue
            if c == 1: 
                color = (0,0,255) 
            elif c == 2 :
                color = (0,69,255) 
            elif c == 3: 
                color = (0,255,255)
            elif c == 4: 
                color = (255,255,255) 
            cv2.rectangle(original_image,(x0,y0),(x1,y1),color,1)
        print("arr discard",arr_discard) 
        j = 0 
        for box in box_imgs:
            if j in arr_discard: 
                print("break") 
                j+=1
                continue
            img = keypoint_detect(original_image,weights_rekt,box,opt.img_size,opt.output,boxes[j],weights_rekt) 
            j += 1
        for cnt  in car_contours: 

            cv2.drawContours(original_image,[cnt],0,(255,255,255),2)
        
        for cnt  in edge_contours: 
        
            cv2.drawContours(original_image,[cnt],0,(255,0,0),2)
        cv2.imwrite(opt.output_path + f ,original_image)
print("Done")     

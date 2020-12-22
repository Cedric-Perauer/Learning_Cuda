import argparse
import os
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolo.models.experimental import attempt_load
from yolo.utils.datasets import LoadStreams, LoadImages
from yolo.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from yolo.utils.torch_utils import select_device, load_classifier, time_synchronized
import time 


def detect(weights,filename,save_img=False):
    #iou params and such 

    weights = weights
    output =  "inference/output" 
    img_size =640
    device = "cpu" 
    if torch.cuda.is_available() : 
        device = "cuda:0" #use cuda device 0 always  
    iou_threshold = 0.45 
    conf_threshold = 0.25 

    out, source, weights, view_img, save_txt, imgsz = \
        output,filename, weights, False, False, img_size

    # Initialize
    set_logging()
    device = select_device(device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16


    dataset = LoadImages(source, img_size=imgsz)

    # Get nasmes and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    out = [] #use this to store all detections for BB
    for path, img, im0s, vid_cap in dataset:
        start = time.time()
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        t1 = time_synchronized()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_threshold, iou_threshold)
        t2 = time_synchronized()
        end = time.time()
        
       

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
    if det is im0 : 
        return None,None
    out = torch.cat((det[:,:4],det[:,-1].unsqueeze(1)),dim=1)
    return out, im0#bounding boxes output 

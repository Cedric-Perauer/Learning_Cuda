import torch
import cv2
import numpy as np
import argparse
import sys
import os
import sys
import shutil
from RektNet.utils import vis_tensor_and_save, prep_image
import time 

from RektNet.keypoint_net import KeypointNet

def keypoint_detect(large_img,model,image,img_size,output,corner,weights):

    output_path = output

    model_path = model

    model_filepath = model_path

    orig_image = image.copy()
    image_size = (img_size, img_size)

    device = "cpu"
    if torch.cuda.is_available(): 
        device = "cuda:0"
    h, w, _ = image.shape
    
    image = prep_image(image,image_size) 
    image = (image.transpose((2, 0, 1)) / 255.0)[np.newaxis, :]
    image = torch.from_numpy(image).type('torch.FloatTensor')
    img = image.clone()
    for i in range(9): 
        image = torch.cat([image,img],0)
    print("shape is:",image.shape)
    model = KeypointNet().to(device)
    model.load_state_dict(torch.load(weights,map_location=torch.device(device)).get('model'))
    model.eval()
    image = image.to(device)
    output = model(image)
    out = np.empty(shape=(0, output[0][0].shape[2]))
    for o in output[0][0]:
        chan = np.array(o.cpu().data)
        cmin = chan.min()
        cmax = chan.max()
        chan -= cmin
        chan /= cmax - cmin
        out = np.concatenate((out, chan), axis=0)
    #cv2.imwrite(output_path + img_name + "_hm.jpg", out * 255)


    
    h, w, _ = orig_image.shape

    c = 0 
    #get top corner poses 
    x0 = corner[0]
    y0 = corner[1]
    colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (127, 255, 127), (255, 127, 127)]
    
    #midpoints for height calc
    slope = (output[1][0][-1][1] - output[1][0][-2][1])/(output[1][0][-1][0]-output[1][0][-2][0])
    neg = 1
    if slope < 0 : 
        neg = -1
    mid_pt = abs(output[1][0][-1] - output[1][0][-2]) * torch.Tensor([0.5,neg * 0.5]).to("cuda:0") + output[1][0][-2]
    import pdb
    pdb.set_trace()
    f = 1.2e-3 #focal length
    img_h = (mid_pt[1].item() - output[1][0][0][1].item()) * h 
    px_size = 6e-6 
    hbb = px_size * img_h 
    cone_h = 0.30

    distance = f / (hbb * cone_h) 


    for pt in np.array(output[1][0].cpu().data):
                x = int(pt[0] * w) #x pixel value 
                y = int(pt[1] * h) #y pixel value 
                new_x = x + int(x0)  
                new_y = y + int(y0) 
                cv2.circle(large_img, (new_x,new_y), 2, colors[c], -1)
                c+= 1  
    new_x = int(mid_pt[0].item() * w) + int(x0)  
    new_y = int(mid_pt[1].item() * h) + int(y0)  
    cv2.circle(large_img,(new_x,new_y),2,(255,255,255),-1)
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
      
    # org
    x = int(output[1][0][0][0].item() * w) + int(x0) 
    y = int(output[1][0][0][1].item() * h) + int(y0) 
    
    org = (x,y) 
    # fontScale 
    fontScale = 1
       
    # Blue color in BGR 
    color = (255, 0, 0) 
      
    # Line thickness of 2 px 
    thickness = 2
       
    # Using cv2.putText() method 
    image = cv2.putText(large_img,  str(round(distance,2)), org, font,  
                       fontScale, color, thickness, cv2.LINE_AA) 
    return image



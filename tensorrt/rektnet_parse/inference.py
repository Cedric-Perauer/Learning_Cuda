import tensorrt as trt
import pycuda.autoinit 
import pycuda.driver as cuda 
import cv2 
import numpy as np
import torch 
import pdb

device = "cpu"
img_sz = 80 
num_kpt = 7
softmax_internal = False
batch_size = 1

def soft_argmax(inp):
      values_y = torch.linspace(0, (img_sz - 1.) / img_sz, img_sz, dtype=inp.dtype, device=inp.device)
      values_x = torch.linspace(0, (img_sz - 1.) / img_sz, img_sz, dtype=inp.dtype, device=inp.device)
      exp_y = (inp.sum(3) * values_y).sum(-1)
      exp_x = (inp.sum(2) * values_x).sum(-1)
      return torch.stack([exp_x, exp_y], -1)

def flat_softmax(inp):
    flat = inp.view(-1,img_sz*img_sz)
    pdb.set_trace()
    flat = torch.nn.functional.softmax(flat,1)
    #expand out again 
    pdb.set_trace()
    hm  =  flat.view(-1,num_kpt,img_sz,img_sz)
    return hm 


def output_process(inp):
    hm = flat_softmax(inp)
    pdb.set_trace()
    out = soft_argmax(hm)
    return out 


def prep_image(image,target_image_size):
    h,w,_ = image.shape
    image = cv2.resize(image, (80,80))
    image = (image.transpose((2, 0, 1)) / 255.0)[np.newaxis, :]
    
    return image


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)
with open('./rektnet.engine','rb') as f: 
    engine_bytes = f.read()
    engine = runtime.deserialize_cuda_engine(engine_bytes) 

rektnet = engine.create_execution_context()
img = cv2.imread("img.jpg")
image_size = (80,80)

image = prep_image(img,image_size)
prep_img = image.copy()
for i in range(batch_size-1): 
     image = np.vstack((image,prep_img))
image = np.ascontiguousarray(image,dtype=np.float32) 



## allocate GPU memory

d_input_ids = cuda.mem_alloc(image.nbytes)
stream = cuda.Stream() 
cuda.memcpy_htod_async(d_input_ids, image, stream)
##output
rektnet_output1 = torch.zeros((batch_size,7,80,80),device=device).cpu().detach().numpy()

d_output = cuda.mem_alloc(rektnet_output1.nbytes)


bindings = [int(d_input_ids),int(d_output)] 
##forward pass 
import time
for i in range(100):
    start = time.time()
    print("start")
    rektnet.execute_async(batch_size,bindings,stream.handle,None)

    cuda.memcpy_dtoh_async(rektnet_output1, d_output, stream)
    stream.synchronize() 
    print("duration is :",(time.time()-start)*1000, " ms" )

out = None
if not softmax_internal:
  out = output_process(torch.from_numpy(rektnet_output1))

else : 
    out = rektnet_output1

import pdb
pdb.set_trace()

"""
for pt in out[0] : 
    x = pt[0].item() * img_sz
    y = pt[1].item() * img_sz
    cv2.circle(prep_img,(int(x),int(y)),2,(255,0,0),2)
    
cv2.imshow("prep image",prep_img)
"""
cv2.waitKey(0)

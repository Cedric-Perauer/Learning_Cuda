import numpy as np 
import cv2
from utils import vis_tensor_and_save
import os
import torch 
import pandas as pd



base_dir = "/home/cedric/PERCEPTION_CAMERA/Deep_Learning/RektNet"
output_uri =  base_dir + "/outputs/visualization/top_losses/"
intput_uri =  base_dir + "/dataset/RektNet_Dataset/"

class Datatracker(): 
    '''
    this class is used to as a wrapper to keep track of the examples in each set and is used to track the top losses
    '''
    def __init__(self,k= 10,max_loss= 100,set_type="val",overfit_mode=False):     
        self.preds = {}  #predictions
        self.labels = {} #labels
        self.overfit = overfit_mode
        self.image_map = {}
        self.image_map_sort = {}
        self.k = k   #goes for k examples with worst loss values
        self.loss = max_loss #max loss mode needs to be implemented still
        self.top_k = {}
        self.set_type = set_type
        self.output_path = base_dir + "/outputs/visualization/top_losses/" + self.set_type + "/"


    def __len__(self): 
        return len(self.image_map)
    
    def __str__(self): 
        for key,value in self.image_map.items(): 
            print(key,"->",value)
        return "End of Map"

    def add_to_map(self,img,dists,preds,labels):
     
        if img in self.image_map.keys() : 
            assert("Already in the map")

        else : 
            #add total error to the dict 
            dists.append(np.sum(dists))
            assert(dists[-1] == np.sum(dists[:-1]),"Total Sum is incorrect")
            self.image_map.update({img:dists})

            #labels and preds maps
            self.preds.update({img:preds})
            self.labels.update({img:labels})

        assert(self.labels.keys()==self.preds.keys()==self.image_map.keys(), "Keys are not equal") 
    

    #by default sort by total error 
    def sort_map(self,sort_by=-1): 
        for key, val in self.image_map.items(): 
            if key in self.image_map_sort.keys(): 
                raise Exception("Value already in the sorted map")
            else : 
                self.image_map_sort.update({key : self.image_map[key][-1]}) 
                
        #sort by total value 
        self.image_map_sort = sorted(self.image_map_sort.items(), key = lambda x : x[1], reverse= True)

    #save k images with top losses
    def plot_top_losses(self): 
        
        df = pd.read_csv(base_dir + "/dataset/rektnet_label.csv")
        df.columns = ['img_name', 'Unnamed: 1', 'top', 'mid_L_top', 'mid_R_top','mid_L_bot', 'mid_R_bot', 'bot_L', 'bot_R', '\n']

        #delete previous top loss files
        test = os.listdir(self.output_path)
        for item in test:
            os.remove(os.path.join(self.output_path,item))

        for k, arr in enumerate(self.image_map_sort) :
            img_name = arr[0]
            total_error = arr[1]
            #get labels and predictions from hash maps 
            preds = self.preds[img_name]
            labels = self.labels[img_name]
            #get row of cur image from df
            row_img = df.loc[df['img_name']==img_name].iloc[:,2:-1]
            blank_image = np.zeros((500,500,3), np.uint8)
            #store labels here 
            labels_array = []
            for i in range(0,row_img.shape[1]): 
                cur = row_img.iloc[0,i][1:-1]
                a = [int(s) for s in cur.split(',')]
                labels_array.append(a)
            labels_array = np.asarray(labels_array)

            #convert image to opencv format
            img = cv2.imread(intput_uri+img_name)
            h,w,c = img.shape
            #get total error
            error = self.image_map[img_name][-1]
            #just reshape
            preds = preds.reshape(7,2)
            #colors
            i = 0
            colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (127, 255, 127), (255, 127, 127)]
            labels = labels.squeeze()

            for pt in np.array(labels_array):
                #ground truth points
                x = pt[0]
                y = pt[1]
                cv2.circle(img, (x, y), 2,(255,255,255), -1)
            
            for pt in np.array(preds):
                #prediction points
                cv2.circle(img, (int(pt[0] * w), int(pt[1] * h)), 2, colors[i], -1)
                i+=1
            
            img = cv2.resize(img,(10*w,10*h),interpolation=cv2.INTER_CUBIC)
            img = cv2.copyMakeBorder(img,100,0,100,100,cv2.BORDER_CONSTANT,value=(255,255,255))
            h,w,c = img.shape
            cv2.circle(img, (0, 0), 2,(0,0,0), -1)
            
            #image wruze
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontScale              = 1
            fontColor              = (0,0,0)
            lineType               = 1
            cv2.putText(img,'Total Error :' + str(error), 
                    (0,40), 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
            #cv2.imshow("Img",img)
            #cv2.waitKey(0)

            cv2.imwrite(self.output_path + img_name,img)
            

            if k+1 == self.k and not self.overfit : 
                break  

       
    


        
     

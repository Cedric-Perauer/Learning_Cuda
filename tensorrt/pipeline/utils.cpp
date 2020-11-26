#include <iostream> 
#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>


class Metric_Tracker { 
public: 
         
std::vector<std::vector<float>> labels = {}; 
int tp = 0; 
int fp = 0; 
int fn = 0; 
int height = 0;
int width = 0;
std::vector<std::vector<float>> gts = {}; 

std::vector<std::vector<float>> gts_0 = {}; 
std::vector<std::vector<float>> gts_1 = {}; 
std::vector<std::vector<float>> gts_2 = {}; 
std::vector<std::vector<float>> gts_3 = {}; 
std::vector<std::vector<float>> gts_4 = {}; 



std::vector<std::vector<float>> preds_0 = {}; 
std::vector<std::vector<float>> preds_1 = {}; 
std::vector<std::vector<float>> preds_2 = {}; 
std::vector<std::vector<float>> preds_3 = {}; 
std::vector<std::vector<float>> preds_4 = {}; 

std::vector<std::vector<std::vector<float>>> class_preds = {}; 

std::string base_path = "/home/cedric/Learning_Cuda/tensorrt/yolov5/labels_pred/"; 

int i = 0;
std::vector<std::vector<std::vector<float>>> class_gts = {}; 

std::vector<std::vector<float>> preds = {};  


//constructors
Metric_Tracker(){}

~Metric_Tracker(){} 
		
//takes filename and returns vector for bounding box 
void get_gt(const std::string &filename,const cv::Mat &img)
{i = 0; 
 gts = {};  
 width = img.cols; 
 height = img.rows;
 labels = {};
 gts_0 = {};  
 gts_1 = {};  
 gts_2 = {};  
 gts_3 = {};  
 gts_4 = {}; 
 preds = {}; 

 std::size_t pos = filename.find(".");
 std::string f = filename.substr(0,pos); 
  
std::string base = "/home/cedric/Learning_Cuda/tensorrt/yolov5/labels/"; 
 //open correct label file and read data into vector 
 std::ifstream myfile(base + f +".txt"); 
 std::string line;
 if(myfile.is_open()) {
	 while(std::getline(myfile,line,'\r'))
	 {       std::istringstream iss(line); 
		 std::vector<float> label; 
		 for(std::string s; iss >> s; ) 
                 { 
		   label.push_back(std::stof(s));
		   if(label[0] == 0)  //blue
		   {
                    gts_0.push_back(label);  
		   }
		   else if(label[0] == 1) //orange 
		   {

                    gts_1.push_back(label);  
		   }
		   else if(label[0] == 2) //large orange
		   {

                    gts_2.push_back(label);  
		   }
		   else if(label[0] == 3)  //yellow
		   {

                    gts_3.push_back(label);  
		   }
		   else if(label[0] == 4)  //unknown 
	           {
                     gts_4.push_back(label); 
		   }

	         }
		 labels.push_back(label); 
	 }
           }
 myfile.close();
}

//create gt files 
void create_gt_files(const std::string &filename) { 
 std::size_t pos = filename.find("."); 
 std::string f = filename.substr(0,pos); 
 std::ofstream myfile(base_path + f + ".txt");  
 for(auto lab : preds) 
 {   
     for(auto vals : lab) 
     { 
        myfile << vals << " "; 
     } 	     
     myfile << "\n"; 
 } 
 myfile.close();
} 

//add prediction 
void add_pred(const cv::Rect &pred_r,const int &cls){ 

  float x_mid = pred_r.x + (pred_r.width / 2.0); 
  x_mid /= width; 
  float y_mid = pred_r.y + (pred_r.height / 2.0);
  y_mid /= height;
  float box_h = pred_r.height /((float)height);
  float box_w = pred_r.width /((float)width);
     
	  
  std::vector<float> pred = {cls,x_mid,y_mid,box_w,box_h,0};
  preds.push_back(pred);  
} 


//printing function 
void print_labels(const std::vector<std::vector<float>> &labels){
       for(auto label : labels) 
       { 
	       for(auto a : label)  
	       { 
		       std::cout << a << " "; 
	       } 
	       std::cout << "\n";
       } 	       

} 	


void print_label(const std::vector<float> &label)
{
	for(auto a: label) 
	{
		std::cout << a << std::endl;

	}
}


void plot_gt(cv::Mat &img) 
{
         std::cout << "size of gt" << gts.size() << std::endl;
        for(auto label : gts) 
	{
         
           cv::rectangle(img,cv::Point(label[0],label[1]),cv::Point(label[2],label[3]),cv::Scalar(255,0,0),1);
	    
	}
		       	
        
}


//compute IOU 
void iou(const cv::Rect &pred_r)
{
 float max_iou = 0.0;  
 int idx = 0;
 std::vector<std::vector<float>>::iterator cur; 
 for(auto gt = labels.begin(); gt != labels.end(); gt++)  
 {
 
  std::vector<float> pred = {pred_r.x, pred_r.y,pred_r.width + pred_r.x, pred_r.height + pred_r.y};
  float w = gt->at(3) * width;  
  float h = gt->at(4) * height;  
  float xt = gt->at(1) * width- w/2.0;  
  float yt = gt->at(2) * height - h/2.0;  
  float xbot = gt->at(1) * width + w/2.0;  
  float ybot = gt->at(2) * height + h/2.0; 
  
  std::vector<float> ground_truth = {xt,yt,xbot,ybot};
 if (i ==0)
{ 
 gts.push_back(ground_truth);
}
 float xa = std::max(pred[0],ground_truth[0]);
 float ya = std::max(pred[1],ground_truth[1]);
 float xb = std::min(pred[2],ground_truth[2]);
 float yb = std::min(pred[3],ground_truth[3]);

 float intersection = std::max(static_cast<float>(0.0),xb-xa) * std::max(static_cast<float>(0.0),yb-ya);
 int area_a = (pred[2]-pred[0] ) * (pred[3]-pred[1] ); 
 int area_b = (ground_truth[2]-ground_truth[0] ) * (ground_truth[3]-ground_truth[1] ); 
 
 float iou = intersection/static_cast<float>(area_a + area_b - intersection);
 if (iou > max_iou)
 {
	 max_iou = iou;
	 cur = gt; 
 }

}
i++; 

std::cout << "max iou = " << max_iou << std::endl;


if (max_iou < 0.5)
{
 fp++; 
}
else if (max_iou >= 0.5){
 tp++; 
 labels.erase(cur);
}
std::cout << "fp=" << fp << std::endl;
std::cout << "tp=" << tp << std::endl;
std::cout << "labels=" << labels.size() << std::endl;

}

//number of fps 
float fn_calc(){
	std::cout << "label size" << labels.size() << std::endl;
       fn += labels.size();
}



//metrics
void calc_metrics()
{       
	std::cout << "fps : " <<fp << std::endl;
	std::cout << "tps : " << tp << std::endl;
	std::cout << "fns : " << fn << std::endl;
	float prec = static_cast<float>(tp)/ static_cast<float>(tp+fp); 
	float recall = static_cast<float>(tp)/static_cast<float>(tp+fn); 
        float f1 = 2 * (prec *recall) /(prec + recall); 
	std::cout << "Precision :" << prec << " recall :" << recall << std::endl;
}
}; 




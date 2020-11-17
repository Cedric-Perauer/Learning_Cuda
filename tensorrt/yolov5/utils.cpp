#include <iostream> 
#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>


class Metric_Tracker { 
	public: 

	Metric_Tracker(){
	}

	~Metric_Tracker(){} 
		
//takes filename and returns vector for bounding box 
std::vector<std::vector<float>> get_gt(const std::string &filename)
{
 
 std::size_t pos = filename.find(".");
 std::string f = filename.substr(0,pos); 
 std::string base_path = "/home/cedric/tensorrtx/yolov5/labels/"; 
 
 //open correct label file and read data into vector 
 std::vector<std::vector<float>> labels; 
 std::cout << f << std::endl;
 std::ifstream myfile(base_path + f +".txt"); 
 std::string line;
 if(myfile.is_open()) {
	 while(std::getline(myfile,line,'\r'))
	 {
		 std::cout << line << "\n"; 
	 }
           }

 myfile.close();
return labels; 
}


//compute IOU 
float iou(const std::vector<float> &pred, const std::vector<float> & ground_truth)
{
 float xa = std::max(pred[0],ground_truth[0]);
 float ya = std::max(pred[1],ground_truth[1]);
 float xb = std::min(pred[2],ground_truth[2]);
 float yb = std::min(pred[3],ground_truth[3]);

 float intersection = std::max(static_cast<float>(0.0),xa-xb + 1) * std::max(static_cast<float>(0.0),yb-ya + 1);

 int area_a = (pred[2]-pred[0] + 1) * (pred[3]-pred[1] + 1); 
 int area_b = (ground_truth[2]-ground_truth[0] + 1) * (ground_truth[3]-ground_truth[1] + 1); 

 float iou = intersection/(area_a + area_b - intersection);

 return iou; 

}

float calc_metrics(const int &tp, const int &fp, const int &fn)
{
	float prec = (static_cast<float>(tp)/(tp+fp)); 
	float recall = (static_cast<float>(tp)/(tp+fn)); 
        float f1 = 2 * (prec *recall) /(prec + recall); 

}
}; 




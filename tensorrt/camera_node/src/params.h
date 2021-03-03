#ifndef PARAMS_H
#define PARAMS_H

#include <string>
#include <iostream>
#include <fstream>

struct PipelineParams {

  // filtering parameters
  float cone_large_height; 
  float cone_small_height;
  float focal_length;
  float pixel_size;
  float center_x;
  float center_y;
  float width_scale; 
  float dist_scale; 
  

  bool fromFile(const std::string &filePath) {
    std::ifstream file_(filePath);

    if (!file_.is_open()) {
      std::cerr << "Params file not found!" << std::endl;
      return false;
    }

    std::string line_;
    int i = 0;
    while (getline(file_, line_)) {
      if (line_[0] == '#') continue;
      if (line_.empty()) continue;

      std::stringstream check1(line_);
      std::string paramName;

      check1 >> paramName;
      if (paramName == "cone_large_height:") {
        check1 >> cone_large_height;
      } else if (paramName == "cone_small_height:") {
        check1 >> cone_small_height;
      } else if (paramName == "focal_length:") {
        check1 >> focal_length;
      } else if (paramName == "pixel_size:") {
        check1 >> pixel_size;
      } else if (paramName == "center_x:") {
        check1 >> center_x;
      } else if (paramName == "center_y:") {
        check1 >> center_y;
      } else if (paramName == "width_scale:") {
        check1 >> width_scale;
      } else if (paramName == "dist_scale:") {
        check1 >> dist_scale;
      } 
      else {
        std::cerr << "Unrecognized pipeline parameter: " << paramName << std::endl;
        assert(0);
      }
    }
    file_.close();
    return true;
  }

};

#endif

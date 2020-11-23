#include <stdio.h>
#include <iostream>

int main()
{
  /*
   * Assign values to these variables so that the output string below prints the
   * requested properties of the currently active GPU.
   */

  int deviceId;
  std::string name;
  int computeCapabilityMajor;
  int computeCapabilityMinor;
  int multiProcessorCount;
  int warpSize;
  
  cudaGetDevice(&deviceId);
  cudaDeviceProp props; 
  
  cudaGetDeviceProperties(&props,deviceId);
  computeCapabilityMajor = props.major; 
  computeCapabilityMinor = props.minor;
  multiProcessorCount = props.multiProcessorCount; 
  warpSize = props.warpSize; 
  name = props.name; 

  /*
   * There should be no need to modify the output string below.
   */

  printf("Device ID: %d\nNumber of SMs: %d\nCompute Capability Major: %d\nCompute Capability Minor: %d\nWarp Size: %d\n GPU Type : %s \n", deviceId, multiProcessorCount, computeCapabilityMajor, computeCapabilityMinor, warpSize,name);
}

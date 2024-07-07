#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <chrono>
#include <string>

#include "C:\Users\artur\Desktop\QAP_CUDA\include\solution.hpp"
#include "C:\Users\artur\Desktop\QAP_CUDA\include\bee_algorithm.hpp"


void cudaCheckError() {                                                                                    
  cudaError_t e = cudaGetLastError();                                                
  
  if (e != cudaSuccess) {                                                                                  
    printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); 
    exit(EXIT_FAILURE);                                                              
  }                                                                                  
}


int main() {
  auto t1 = std::chrono::steady_clock::now();

  int n = 70;
  int m = 60;
  int e = 20;

  int nep = 30;
  int nsp = 20;
  int ngh = 5;

  int cev_split_trials = 10;
  int lifespan = 0;

  std::string file_name = "wil100";

  int epochs = 150;
  bool use_cuda = true;

  BeeAlgorithm ba = BeeAlgorithm(n, m, e, nep, nsp, ngh, cev_split_trials, lifespan, file_name, epochs, use_cuda);
  ba.run();

  auto t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration = t2 - t1;
  ba.set_time(duration.count());

  cudaCheckError();

  return EXIT_SUCCESS;
}

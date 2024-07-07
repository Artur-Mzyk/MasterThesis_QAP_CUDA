#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <chrono>

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

  int n = 20;
  int m = 80;
  int e = 30;

  int nep = 50;
  int nsp = 30;
  int ngh = 5;

  int cev_split_trials = 10;
  int lifespan = 5;

  char* file_path = "data/els19.dat";
  int epochs = 50;
  bool use_cuda = true;

  BeeAlgorithm ba = BeeAlgorithm(n, m, e, nep, nsp, ngh, cev_split_trials, lifespan, file_path, epochs, use_cuda);
  ba.run();

  auto t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration = t2 - t1;
  std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

  cudaCheckError();

  return EXIT_SUCCESS;
}

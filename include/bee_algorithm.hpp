#ifndef QAP_BEE_ALGORITHM_HPP
#define QAP_BEE_ALGORITHM_HPP

#include <iostream>
#include <curand_kernel.h>
#include <string>

#include "solution.hpp"


class BeeAlgorithm {
    private:
        int n;
        int m;
        int e;
        int nep;
        int nsp;
        int ngh;
        std::string file_name;
        int epochs;
        int cev_split_trials;
        int lifespan;
        int N;
        float **D;
        float **F;
        Solution** population;
        bool use_cuda;
        float target_fitness;

    public:
    	BeeAlgorithm(int n, int m, int e, int nep, int nsp, int ngh, int cev_split_trials, int lifespan, std::string file_name, int epochs, bool use_cuda);
    	~BeeAlgorithm();
        void sort();
        void run();
        void save(float* fitnesses);
        void set_time(double computation_time);
};


__global__ void setup_kernel(curandState * state, unsigned long seed);

__global__ void process_bee_classic(int N, int* d_pi_flatten, float* d_D_F_flatten, int n, int m, int e, int nep, int nsp, int ngh, int cev_split_trials, int epochs, curandState* states, float* d_best_fitnesses);
__global__ void process_bee_ABC(int N, int* d_pi_flatten, float* d_D_F_flatten, int n, int m, int e, int nep, int nsp, int ngh, int cev_split_trials, int epochs, curandState* states, float* d_best_fitnesses, int lifespan);

__device__ float get_fitness(int N, int* pi, float* D_flatten, float* F_flatten);
__device__ void permutate(int N, int* pi, curandState* states, int j);
__device__ int* get_best_neighbour(int N, int* pi, float* D_flatten, float* F_flatten, int n, int size, curandState* states, int j);

__device__ float get_fitness_conditional(int N, int* pi, float* D_flatten, float* F_flatten, int* H, int* LH, int* U, int size);
__device__ void permutate_conditional(int N, int* pi, curandState* states, int j, int* LH, int size);
__device__ int* get_best_neighbour_conditional(int N, int* pi, float* D_flatten, float* F_flatten, int n, int size, curandState* states, int j, int cev_split_trials);

#endif //QAP_BEE_ALGORITHM_HPP

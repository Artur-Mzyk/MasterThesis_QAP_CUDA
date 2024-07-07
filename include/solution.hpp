#ifndef QAP_SOLUTION_HPP
#define QAP_SOLUTION_HPP

#include <fstream>
#include <iostream>


class Solution {
    public:
        int N;
        int* pi;
        float **D;
        float **F;
    	float fitness;
        int lifespan;

    	Solution(int N_, float** D_, float** F_, int lifespan_);
    	~Solution();
        void permutate();
        void permutate_conditional(int* LH, int size);
    	void calculate_fitness();
        float calculate_conditional_fitness(int* H, int* LH, int* U, int size);
        float get_fitness();
        void set_pi(int* pi);
        Solution** generate_neighbours(int n, int size);
        Solution** generate_conditional_neighbours(int n, int size, int fields);
};

#endif //QAP_SOLUTION_HPP

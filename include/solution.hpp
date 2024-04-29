#ifndef QAP_SOLUTION_HPP
#define QAP_SOLUTION_HPP


#include <iostream>
#include <fstream>


#define NULL nullptr


class Solution {
    private:
        std::string file_path;
        int N;
        int* pi;
        float **D;
        float **F;
    	float fitness;
    public:
    	Solution(std::string file_path);
    	~Solution();
        void generate_permutation();
        void permutate();
        void display();
        void show_matrices();
    	void calculate_fitness();
        float get_fitness();
        void set_pi(int* pi);
        int* get_pi();
        Solution** generate_neighbours(int n, int size);
};


#endif //QAP_SOLUTION_HPP

#ifndef QAP_POPULATION_HPP
#define QAP_POPULATION_HPP


#include <iostream>
#include <vector>

#include "solution.hpp"


class Population {
    private:
        int n;
        std::string file_path;
        Solution** solutions;
    public:
    	Population(int N, std::string file_path);
    	~Population();
        int get_size();
        void set_solutions(Solution** solutions);
        Solution** get_solutions();
        Solution** get_best(int k);
        void sort();
        void display();
};


#endif //QAP_POPULATION_HPP

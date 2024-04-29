#ifndef QAP_BEE_ALGORITHM_HPP
#define QAP_BEE_ALGORITHM_HPP


#include <iostream>
#include <vector>

#include "solution.hpp"
#include "population.hpp"


class BeeAlgorithm {
    private:
        int n;
        int m;
        int e;
        int nep;
        int nsp;
        int ngh;
        std::string file_path;
        int epochs;
        Population* population;
    public:
    	BeeAlgorithm(int n, int m, int e, int nep, int nsp, int ngh, std::string file_path, int epochs);
    	~BeeAlgorithm();
        void run();
};


#endif //QAP_BEE_ALGORITHM_HPP

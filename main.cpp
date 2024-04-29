#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>

#include "solution.hpp"
#include "population.hpp"
#include "bee_algorithm.hpp"


int main()
{   
    int n = 30;
    int m = 20;
    int e = 5;
    int nep = 10;
    int nsp = 5;
    int ngh = 2;
    std::string file_path = "data/coral.ise.lehigh.edu_wp-content_uploads_2014_07_data.d_els19.dat";
    int epochs = 100;
    BeeAlgorithm ba = BeeAlgorithm(n, m, e, nep, nsp, ngh, file_path, epochs);
    ba.run();

    return EXIT_SUCCESS;
}

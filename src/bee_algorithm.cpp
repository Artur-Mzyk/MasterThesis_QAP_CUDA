#include "bee_algorithm.hpp"


BeeAlgorithm::BeeAlgorithm(int n_, int m_, int e_, int nep_, int nsp_, int ngh_, std::string file_path_, int epochs_) {
    n = n_;
    m = m_;
    e = e_;
    nep = nep_;
    nsp = nsp_;
    ngh = ngh_;
    file_path = file_path_;
    epochs = epochs_;
    population = new Population(n, file_path);
}


BeeAlgorithm::~BeeAlgorithm() {
}


void BeeAlgorithm::run() {
    Solution** solutions;
    Solution** new_solutions = new Solution*[n];
    Solution** best_solutions = new Solution*[n];
    Population* neighbourhood;
    int neighbourhood_quantity;
    int neighbourhood_size;

    for (int i = 0; i < epochs; i++) {
        std::cout << "Epoch " << i + 1 << std::endl;
        population -> sort();
        solutions = population -> get_best(n);
        best_solutions[i] = solutions[0];
        best_solutions[i] -> display();

        for (int j = 0; j < n; j++) {
            if (j < m) {
                neighbourhood_quantity = (j < e) ? nep : nsp;
                neighbourhood = new Population(neighbourhood_quantity, file_path);
                neighbourhood -> set_solutions(solutions[j] -> generate_neighbours(neighbourhood_quantity, ngh));
                new_solutions[j] = (neighbourhood -> get_best(1))[0];
            } else {
                new_solutions[j] = new Solution(file_path);
            }
        }

        population -> set_solutions(new_solutions);
        std::cout << std::endl;
    }
}

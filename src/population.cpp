#include "population.hpp"


Population::Population(int n_, std::string file_path_) {
    n = n_;
    file_path = file_path_;
    solutions = new Solution*[n];
    srand(time(NULL));

    for (int i = 0; i < n; i++) {
        solutions[i] = new Solution(file_path);
    }
}


Population::~Population() {
    if (solutions != NULL) {
        for (int i = 0; i < n; i++) {
            delete[] solutions[i];
        }

        delete[] solutions;
	}
}


int Population::get_size() {
    return n;
}


void Population::set_solutions(Solution** solutions_) {
    solutions = solutions_;
}


Solution** Population::get_solutions() {
    return solutions;
}


Solution** Population::get_best(int k) {
    sort();
    Solution** best_solutions = new Solution*[k];

    if (k == -1) {
        k = n;
    }

    for (int i = 0; i < k; i++) {
        best_solutions[i] = solutions[i];
    }

    return best_solutions;
}


void Population::sort() {
    bool swapped;
    Solution* temp;

    for (int i = 0; i < n - 1; i++) {
        swapped = false;

        for (int j = 0; j < n - i - 1; j++) {
            if (solutions[j] -> get_fitness() > solutions[j + 1] -> get_fitness()) {
                temp = solutions[j];
                solutions[j] = solutions[j + 1];
                solutions[j + 1] = temp;
                swapped = true;
            }
        }

        if (!swapped)
            break;
    }
}


void Population::display() {
    for (int i = 0; i < n; i++) {
        solutions[i] -> display();
    }
}

#include "solution.hpp"


Solution::Solution(std::string file_path_) {
    file_path = file_path_;

    std::ifstream file;
    float data;
    file.open(file_path);
    file >> data;
    N = int(data);

    pi = new int[N];
    D = new float* [N];
    F = new float* [N];

    for (int i = 0; i < N; i++) {
        pi[i] = i;
        D[i] = new float[N];
        F[i] = new float[N];
    }

    generate_permutation();

    int k = 0;

    while (file >> data) {
        if (k < N * N) {
            D[int(k / N)][k % N] = data;
        } else {
            F[int(k / N - N)][k % N] = data;
        }

        k++;
    }

    file.close();

    fitness = 0.0;
}


Solution::~Solution() {
    if (pi != NULL) {
        delete[] pi;
	}

    if (D != NULL) {
        for (int i = 0; i < N; i++) {
            delete[] D[i];
        }

        delete[] D;
	}

    if (F != NULL) {
        for (int i = 0; i < N; i++) {
            delete[] F[i];
        }

        delete[] F;
	}
}


void Solution::generate_permutation() {
    int j;
    int temp;

    for (int i = 0; i < N; i++) {
        j = rand() % (N - i) + i;
        temp = pi[j];
        pi[j] = pi[i];
        pi[i] = temp;
    }
}


void Solution::permutate() {
    int i = 0;
    int j = 0;
    int temp;

    while (i == j) {
        i = rand() % N;
        j = rand() % N;
    }

    temp = pi[j];
    pi[j] = pi[i];
    pi[i] = temp;
}


void Solution::display() {
    std::cout << "Solution: (";

    for (int i = 0; i < N; i++) {
        std::cout << pi[i];

        if (i < N - 1) {
            std::cout << ", ";
        }
    }

    std::cout << ")\tFitness: " << fitness << std::endl;
}


void Solution::show_matrices() {
    std::cout << "Matrix D:" << std::endl;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << D[i][j] << "\t";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Matrix F:" << std::endl;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << F[i][j] << "\t";
        }

        std::cout << std::endl;
    }
}


void Solution::calculate_fitness() {
    float sum = 0.0;
    int pi_i;
    int pi_j;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            pi_i = pi[i];
            pi_j = pi[j];
            sum = sum + D[pi_i][pi_j] * F[i][j];
        }
    }

	fitness = sum;
}


float Solution::get_fitness() {
    calculate_fitness();

    return fitness;
}


void Solution::set_pi(int* pi_) {
    for (int i = 0; i < N; i++) {
        pi[i] = pi_[i];
    }
}


int* Solution::get_pi() {
    return pi;
}


Solution** Solution::generate_neighbours(int n, int size) {
    Solution** neighbours = new Solution*[n];

    for (int i = 0; i < n; i++) {
        neighbours[i] = new Solution(file_path);
        neighbours[i] -> set_pi(pi);

        for (int j = 0; j < size; j++) {
            neighbours[i] -> permutate();
        }
    }

    return neighbours;
}

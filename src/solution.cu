#include "C:\Users\artur\Desktop\QAP_CUDA\include\solution.hpp"


Solution::Solution(int N_, float** D_, float** F_, int lifespan_) {
    N = N_;
    lifespan = lifespan_;
    pi = new int[N];
    D = new float*[N];
    F = new float*[N];

    for (int i = 0; i < N; i++) {
        pi[i] = i;
        D[i] = new float[N];
        F[i] = new float[N];

        for (int j = 0; j < N; j++) {
            D[i][j] = D_[i][j];
            F[i][j] = F_[i][j];
        }
    }

    int j;
    int temp;

    for (int i = 0; i < N; i++) {
        j = rand() % (N - i) + i;
        temp = pi[j];
        pi[j] = pi[i];
        pi[i] = temp;
    }

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


void Solution::permutate_conditional(int* LH, int size) {
    int* new_LH = new int[size];
    int* pi_copy = new int[N];

    for (int i = 0; i < size; i++) {
        new_LH[i] = LH[i];
    }

    for (int i = 0; i < N; i++) {
        pi_copy[i] = pi[i];
    }

    int j;
    int temp;

    for (int i = 0; i < size; i++) {
        j = rand() % (size - i) + i;
        temp = new_LH[j];
        new_LH[j] = new_LH[i];
        new_LH[i] = temp;
    }

    for (int i = 0; i < size; i++) {
        pi[new_LH[i]] = pi_copy[LH[i]];
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


float Solution::calculate_conditional_fitness(int* H, int* LH, int* U, int size) {
    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;
    float sum4 = 0.0;
    float sum5 = 0.0;
    int pi_i;
    int pi_j;
    int h = N - size;
    int lh = size;
    int u = size;

    for (int a = 0; a < N - size; a++) {
        for (int b = 0; b < N - size; b++) {
            pi_i = pi[H[a]];
            pi_j = pi[H[b]];
            sum1 = sum1 + F[H[a]][H[b]] * D[pi_i][pi_j];
        }
    }

    for (int a = 0; a < N - size; a++) {
        for (int b = 0; b < size; b++) {
            for (int c = 0; c < size; c++) {
                pi_i = pi[H[a]];
                sum2 = sum2 + D[H[a]][LH[b]] * F[pi_i][U[c]];
            }
        }
    }

    for (int a = 0; a < N - size; a++) {
        for (int b = 0; b < size; b++) {
            for (int c = 0; c < size; c++) {
                pi_i = pi[H[a]];
                sum3 = sum3 + D[LH[b]][H[a]] * F[U[c]][pi_i];
            }
        }
    }

    for (int a = 0; a < size; a++) {
        for (int b = 0; b < size; b++) {
            if (LH[a] != LH[b]) {
                for (int c = 0; c < size; c++) {
                    for (int d = 0; d < size; d++) {
                        if (U[c] != U[d]) {
                            sum4 = sum4 + D[LH[a]][LH[b]] * F[U[c]][U[d]];
                        }
                    }
                }
            }
        }
    }

    for (int a = 0; a < size; a++) {
        for (int b = 0; b < size; b++) {
            sum5 = sum5 +  F[LH[a]][LH[a]] * D[U[b]][U[b]];
        }
    }

    float sum = sum1 + sum2 / (N - h) + sum3 / (N - h) + sum4 / ((N - h) * (N - h - 1)) + sum5 / (N - h);
	
    return sum;
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


Solution** Solution::generate_neighbours(int n, int size) {
    Solution** neighbours = new Solution*[n];

    for (int i = 0; i < n; i++) {
        neighbours[i] = new Solution(N, D, F, lifespan);
        neighbours[i] -> set_pi(pi);

        for (int j = 0; j < size; j++) {
            neighbours[i] -> permutate();
        }
    }

    return neighbours;
}


Solution** Solution::generate_conditional_neighbours(int n, int size, int fields) {
    srand(time(NULL));
    int* idx = new int[N];
    int* H = new int[N - size];
    int* LH = new int[size];
    int* U = new int[size];
    int j;
    int temp;
    float min_fitness;
    int* min_fitness_LH;

    for (int f = 0; f < fields; f++) {
        for (int i = 0; i < N; i++) {
            idx[i] = i;
        }

        for (int i = 0; i < N; i++) {
            j = rand() % (N - i) + i;
            temp = idx[j];
            idx[j] = idx[i];
            idx[i] = temp;
        }

        for (int i = 0; i < N; i++) {
            if (i < N - size) {
                H[i] = idx[i];
            } else {
                LH[i - N + size] = idx[i];
                U[i - N + size] = pi[idx[i]];
            }
        }

        float current_fitness = calculate_conditional_fitness(H, LH, U, size);

        if (f == 0 || current_fitness < min_fitness) {
            min_fitness = current_fitness;
            min_fitness_LH = LH;
        }
    }

    Solution** neighbours = new Solution*[n];

    for (int i = 0; i < n; i++) {
        neighbours[i] = new Solution(N, D, F, lifespan);
        neighbours[i] -> set_pi(pi);

        for (int j = 0; j < size; j++) {
            neighbours[i] -> permutate_conditional(min_fitness_LH, size);
        }
    }

    return neighbours;
}

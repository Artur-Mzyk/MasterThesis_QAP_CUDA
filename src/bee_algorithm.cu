#include "C:\Users\artur\Desktop\QAP_CUDA\include\bee_algorithm.hpp"


BeeAlgorithm::BeeAlgorithm(int n_, int m_, int e_, int nep_, int nsp_, int ngh_, int cev_split_trials_, int lifespan_, std::string file_name_, int epochs_, bool use_cuda_) {
    n = n_;
    m = m_;
    e = e_;
    nep = nep_;
    nsp = nsp_;
    ngh = ngh_;
    file_name = file_name_;
    epochs = epochs_;
    cev_split_trials = cev_split_trials_;
    lifespan = lifespan_;
    use_cuda = use_cuda_;

    std::string instance_path = "data/instances/" + file_name + ".dat";
    std::string solution_path = "data/solutions/" + file_name + ".sln";

    std::ifstream instance;
    float data;
    instance.open(instance_path);
    instance >> data;
    N = int(data);

    D = new float*[N];
    F = new float*[N];

    for (int i = 0; i < N; i++) {
        D[i] = new float[N];
        F[i] = new float[N];
    }

    int k = 0;

    while (instance >> data) {
        if (k < N * N) {
            D[int(k / N)][k % N] = data;
        } else {
            F[int(k / N - N)][k % N] = data;
        }

        k++;
    }

    instance.close();

    std::ifstream solution;
    solution.open(solution_path);
    solution >> target_fitness;
    solution >> target_fitness;
    solution.close();

    population = new Solution*[n];

    for (int i = 0; i < n; i++) {
        population[i] = new Solution(N, D, F, 0);
    }

    sort();
}


BeeAlgorithm::~BeeAlgorithm() {
}


void BeeAlgorithm::sort() {
    bool swapped;
    Solution* temp;

    for (int i = 0; i < n - 1; i++) {
        swapped = false;

        for (int j = 0; j < n - i - 1; j++) {
            if (population[j] -> get_fitness() > population[j + 1] -> get_fitness()) {
                temp = population[j];
                population[j] = population[j + 1];
                population[j + 1] = temp;
                swapped = true;
            }
        }

        if (!swapped)
            break;
    }
}


void BeeAlgorithm::run() {
    if (use_cuda) {
        curandState* states;
        cudaMalloc((void**)&states, n * sizeof(curandState));
        setup_kernel<<<1, n>>>(states, time(NULL));

        float* h_D_F_flatten = new float[2 * N * N];
        int* h_pi_flatten = new int[n * N];
        float* h_best_fitnesses = new float[epochs];
        float* d_D_F_flatten;
        int* d_pi_flatten;
        float* d_best_fitnesses;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                h_D_F_flatten[i * N + j] = D[i][j];
                h_D_F_flatten[(i + N) * N + j] = F[i][j];
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < N; j++) {
                h_pi_flatten[i * N + j] = population[i] -> pi[j];
            }
        }

        cudaMalloc((void**) &d_D_F_flatten, 2 * N * N * sizeof(float));
        cudaMalloc((void**) &d_pi_flatten, n * N * sizeof(int));
        cudaMalloc((void**) &d_best_fitnesses, epochs * sizeof(float));

        cudaMemcpy(d_D_F_flatten, h_D_F_flatten, 2 * N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pi_flatten, h_pi_flatten, n * N * sizeof(int), cudaMemcpyHostToDevice);

        if (lifespan > 0) {
            process_bee_ABC<<<1, n, n * sizeof(float)>>>(N, d_pi_flatten, d_D_F_flatten, n, m, e, nep, nsp, ngh, cev_split_trials, epochs, states, d_best_fitnesses, lifespan);
        } else {
            process_bee_classic<<<1, n, n * sizeof(float) + epochs * sizeof(int)>>>(N, d_pi_flatten, d_D_F_flatten, n, m, e, nep, nsp, ngh, cev_split_trials, epochs, states, d_best_fitnesses);
        }

        cudaFree(states);
        cudaFree(d_D_F_flatten);
        cudaFree(d_pi_flatten);
        
        cudaMemcpy(h_best_fitnesses, d_best_fitnesses, epochs * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_best_fitnesses);

        save(h_best_fitnesses);
    } else {
        float* h_best_fitnesses = new float[epochs];

        for (int i = 0; i < epochs; i++) {
            if (lifespan > 0) {
                sort();
                h_best_fitnesses[i] = population[0] -> get_fitness();

                for (int j = 0; j < n; j++) {
                    Solution** neighbours = (cev_split_trials > 0) ? population[j] -> generate_conditional_neighbours(1, ngh, cev_split_trials) : population[j] -> generate_neighbours(1, ngh);

                    if (neighbours[0] -> get_fitness() < population[j] -> get_fitness()) {
                        population[j] = neighbours[0];
                    } else {
                        population[j] -> lifespan = population[j] -> lifespan + 1;
                    }
                }

                float fitness_sum = 0.0;

                for (int j = 0; j < n; j++) {
                    fitness_sum = fitness_sum + population[j] -> get_fitness();
                }

                float* roulette_wheel = new float[n];
                roulette_wheel[0] = 0;

                for (int j = 0; j < n - 1; j++) {
                    roulette_wheel[j + 1] = roulette_wheel[j] + population[j] -> get_fitness() / fitness_sum;
                }

                Solution** population_copy = new Solution*[n];

                for (int j = 0; j < n; j++) {
                    population_copy[j] = population[j];
                }

                for (int j = 0; j < n; j++) {
                    float p = (float)((double)rand()) / RAND_MAX;

                    for (int k = 1; k < n; k++) {
                        if (p < roulette_wheel[k]) {
                            Solution** neighbours = (cev_split_trials > 0) ? population_copy[k - 1] -> generate_conditional_neighbours(1, ngh, cev_split_trials) : population_copy[k - 1] -> generate_neighbours(1, ngh);

                            if (neighbours[0] -> get_fitness() < population[j] -> get_fitness()) {
                                population[j] = neighbours[0];
                            } else {
                                population[j] -> lifespan = population[j] -> lifespan + 1;
                            }
                        }
                    }
                }

                for (int j = 0; j < n; j++) {
                    if (population[j] -> lifespan == lifespan) {
                        population[j] = new Solution(N, D, F, 0);
                    }
                }
            } else {
                sort();
                h_best_fitnesses[i] = population[0] -> get_fitness();

                for (int j = 0; j < n; j++) {
                    if (j < m) {
                        int neighbourhood_quantity = (j < e) ? nep : nsp;
                        Solution* solution = population[j];
                        Solution** neighbourhood = new Solution*[neighbourhood_quantity];

                        if (cev_split_trials > 0) {
                            neighbourhood = solution -> generate_conditional_neighbours(neighbourhood_quantity, ngh, cev_split_trials);
                        } else {
                            neighbourhood = solution -> generate_neighbours(neighbourhood_quantity, ngh);
                        }

                        bool swapped;
                        Solution* temp;

                        for (int i = 0; i < neighbourhood_quantity - 1; i++) {
                            swapped = false;

                            for (int k = 0; k < neighbourhood_quantity - i - 1; k++) {
                                if (neighbourhood[k] -> get_fitness() > neighbourhood[k + 1] -> get_fitness()) {
                                    temp = neighbourhood[k];
                                    neighbourhood[k] = neighbourhood[k + 1];
                                    neighbourhood[k + 1] = temp;
                                    swapped = true;
                                }
                            }

                            if (!swapped)
                                break;
                        }

                        population[j] = neighbourhood[0];
                    } else {
                        population[j] = new Solution(N, D, F, 0);
                    }
                }
            }
        }

        save(h_best_fitnesses);
    }
}


void BeeAlgorithm::save(float* fitnesses) {
    std::ofstream file;
    file.open("data/results/" + file_name + ".txt");

    for (int i = 0; i < epochs; i++) {
        file << fitnesses[i] << std::endl;
    }

    file << target_fitness << std::endl;
    file.close();
}


void BeeAlgorithm::set_time(double computation_time) {
    std::ofstream file;
    file.open("data/results/" + file_name + ".txt", std::ios_base::app);
    file << computation_time;
    file.close();
}


__global__ void setup_kernel(curandState* states, unsigned long seed) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, j, 0, &states[j]);
} 


__global__ void process_bee_classic(int N, int* d_pi_flatten, float* d_D_F_flatten, int n, int m, int e, int nep, int nsp, int ngh, int cev_split_trials, int epochs, curandState* states, float* d_best_fitnesses) {
    int j = threadIdx.x;
    float* D_flatten = new float[N * N];
    float* F_flatten = new float[N * N];
    int* pi = new int[N];

    for (int k = 0; k < N * N; k++) {
        D_flatten[k] = d_D_F_flatten[k];
        F_flatten[k] = d_D_F_flatten[k + N * N];
    }

    for (int k = 0; k < N; k++) {
        pi[k] = d_pi_flatten[j * N + k];
    }

    extern __shared__ float fitnesses[];
    extern __shared__ int best_fitnesses_idx[];

    for (int i = 0; i < epochs; i++) {
        fitnesses[j] = get_fitness(N, pi, D_flatten, F_flatten);

        __syncthreads();

        if (j == 0) {
            float best_fitness = fitnesses[0];
            float best_fitness_idx = 0;

            for (int k = 1; k < n; k++) {
                if (fitnesses[k] < best_fitness) {
                    best_fitness = fitnesses[k];
                    best_fitness_idx = k;
                }
            }

            d_best_fitnesses[i] = best_fitness;
            best_fitnesses_idx[i] = best_fitness_idx;
        }

        int order = 0;

        for (int k = 0; k < n; k++) {
            if (fitnesses[j] > fitnesses[k]) {
                order++;
            }
        }

        __syncthreads();

        if (order < m) {
            int neighbourhood_quantity = (order < e) ? nep : nsp;

            if (cev_split_trials > 0) {
                pi = get_best_neighbour_conditional(N, pi, D_flatten, F_flatten, neighbourhood_quantity, ngh, states, j, cev_split_trials);
            } else {
                pi = get_best_neighbour(N, pi, D_flatten, F_flatten, neighbourhood_quantity, ngh, states, j);
            }
        } else {
            for (int k = 0; k < N; k++) {
                pi[k] = k;
            }

            for (int k = 0; k < N; k++) {
                curandState state = states[j];
                int l = curand(&state) % (N - k) + k;
                states[j] = state;
                int temp = pi[l];
                pi[l] = pi[k];
                pi[k] = temp;
            }
        }
    }
}


__global__ void process_bee_ABC(int N, int* d_pi_flatten, float* d_D_F_flatten, int n, int m, int e, int nep, int nsp, int ngh, int cev_split_trials, int epochs, curandState* states, float* d_best_fitnesses, int lifespan) {
    int j = threadIdx.x;
    float* D_flatten = new float[N * N];
    float* F_flatten = new float[N * N];
    int* pi = new int[N];
    int* pi_temp = new int[N];
    float* roulette_wheel = new float[n];
    int sol_lifespan = 0;

    for (int k = 0; k < N * N; k++) {
        D_flatten[k] = d_D_F_flatten[k];
        F_flatten[k] = d_D_F_flatten[k + N * N];
    }

    for (int k = 0; k < N; k++) {
        pi[k] = d_pi_flatten[j * N + k];
    }

    extern __shared__ float fitnesses[];

    for (int i = 0; i < epochs; i++) {
        fitnesses[j] = get_fitness(N, pi, D_flatten, F_flatten);

        __syncthreads();

        if (j == 0) {
            float best_fitness = fitnesses[0];

            for (int k = 1; k < n; k++) {
                if (fitnesses[k] < best_fitness) {
                    best_fitness = fitnesses[k];
                }
            }

            d_best_fitnesses[i] = best_fitness;
        }

        roulette_wheel[0] = 0.0;
        float fitness_sum = 0.0;

        for (int k = 0; k < n; k++) {
            fitness_sum = fitness_sum + fitnesses[k];
        }

        for (int k = 0; k < n - 1; k++) {
            roulette_wheel[k + 1] = roulette_wheel[k] + fitnesses[k] / fitness_sum;
        }

        for (int k = 0; k < N; k++) {
            d_pi_flatten[j * N + k] = pi[k];
        }

        __syncthreads();

        if (cev_split_trials > 0) {
            pi_temp = get_best_neighbour_conditional(N, pi, D_flatten, F_flatten, 1, ngh, states, j, cev_split_trials);
        } else {
            pi_temp = get_best_neighbour(N, pi, D_flatten, F_flatten, 1, ngh, states, j);
        }

        if (get_fitness(N, pi_temp, D_flatten, F_flatten) < get_fitness(N, pi, D_flatten, F_flatten)) {
            for (int k = 0; k < N; k++) {
                pi[k] = pi_temp[k];
            }
        } else {
            sol_lifespan++;
        }

        curandState state = states[j];
        float p = curand_uniform(&state);
        states[j] = state;

        int k = 0;

        while (p >= roulette_wheel[k + 1] && k < n - 1) {
            k++;
        }

        for (int l = 0; l < N; l++) {
            pi_temp[l] = d_pi_flatten[k * N + l];
        }

        if (cev_split_trials > 0) {
            pi_temp = get_best_neighbour_conditional(N, pi_temp, D_flatten, F_flatten, 1, ngh, states, j, cev_split_trials);
        } else {
            pi_temp = get_best_neighbour(N, pi_temp, D_flatten, F_flatten, 1, ngh, states, j);
        }

        if (get_fitness(N, pi_temp, D_flatten, F_flatten) < get_fitness(N, pi, D_flatten, F_flatten)) {
            for (int l = 0; l < N; l++) {
                pi[l] = pi_temp[l];
            }
        } else {
            sol_lifespan++;
        }

        if (sol_lifespan == lifespan) {
            sol_lifespan = 0;

            for (int k = 0; k < N; k++) {
                pi[k] = k;
            }

            for (int k = 0; k < N; k++) {
                curandState state = states[j];
                int l = curand(&state) % (N - k) + k;
                states[j] = state;
                int temp = pi[l];
                pi[l] = pi[k];
                pi[k] = temp;
            }
        }
    }
}


__device__ float get_fitness(int N, int* pi, float* D_flatten, float* F_flatten) {
    float fitness = 0.0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fitness = fitness + D_flatten[pi[i] * N + pi[j]] * F_flatten[i * N + j];
        }
    }

	return fitness;
}


__device__ void permutate(int N, int* pi, curandState* states, int j) {
    int a = 0;
    int b = 0;
    curandState state;

    while (a == b) {
        state = states[j];
        a = curand(&state) % N;
        states[j] = state;

        state = states[j];
        b = curand(&state) % N;
        states[j] = state;
    }

    int temp = pi[b];
    pi[b] = pi[a];
    pi[a] = temp;
}


__device__ int* get_best_neighbour(int N, int* pi, float* D_flatten, float* F_flatten, int n, int size, curandState* states, int j) {
    int* pi_temp = new int[N];

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < N; k++) {
            pi_temp[k] = pi[k];
        }

        for (int k = 0; k < size; k++) {
            permutate(N, pi_temp, states, j);
        }

        if (get_fitness(N, pi_temp, D_flatten, F_flatten) < get_fitness(N, pi, D_flatten, F_flatten)) {
            for (int k = 0; k < N; k++) {
                pi[k] = pi_temp[k];
            }
        }
    }

    delete[] pi_temp;

    return pi;
}


__device__ float get_fitness_conditional(int N, int* pi, float* D_flatten, float* F_flatten, int* H, int* LH, int* U, int size) {
    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;
    float sum4 = 0.0;
    float sum5 = 0.0;
    int h = N - size;

    for (int a = 0; a < N - size; a++) {
        for (int b = 0; b < N - size; b++) {
            sum1 = sum1 + F_flatten[H[a] * N + H[b]] * D_flatten[pi[H[a]] * N + pi[H[b]]];
        }
    }

    for (int a = 0; a < N - size; a++) {
        for (int b = 0; b < size; b++) {
            for (int c = 0; c < size; c++) {
                sum2 = sum2 + D_flatten[H[a] * N + LH[b]] * F_flatten[pi[H[a]] * N + U[c]];
            }
        }
    }

    for (int a = 0; a < N - size; a++) {
        for (int b = 0; b < size; b++) {
            for (int c = 0; c < size; c++) {
                sum3 = sum3 + D_flatten[LH[b] * N + H[a]] * F_flatten[U[c] * N + pi[H[a]]];
            }
        }
    }

    for (int a = 0; a < size; a++) {
        for (int b = 0; b < size; b++) {
            if (LH[a] != LH[b]) {
                for (int c = 0; c < size; c++) {
                    for (int d = 0; d < size; d++) {
                        if (U[c] != U[d]) {
                            sum4 = sum4 + D_flatten[LH[a] * N + LH[b]] * F_flatten[U[c] * N + U[d]];
                        }
                    }
                }
            }
        }
    }

    for (int a = 0; a < size; a++) {
        for (int b = 0; b < size; b++) {
            sum5 = sum5 +  F_flatten[LH[a] * N + LH[a]] * D_flatten[U[b] * N + U[b]];
        }
    }

    float fitness = sum1 + sum2 / (N - h) + sum3 / (N - h) + sum4 / ((N - h) * (N - h - 1)) + sum5 / (N - h);
	
    return fitness;
}


__device__ void permutate_conditional(int N, int* pi, curandState* states, int j, int* LH, int size) {
    int a = 0;
    int b = 0;
    int c = 0;
    int d = 0;
    curandState state;
    int temp;

    while (c == d) {
        state = states[j];
        a = curand(&state) % size;
        states[j] = state;

        state = states[j];
        b = curand(&state) % size;
        states[j] = state;

        c = LH[a];
        d = LH[b];
    }

    temp = pi[d];
    pi[d] = pi[c];
    pi[c] = temp;
}


__device__ int* get_best_neighbour_conditional(int N, int* pi, float* D_flatten, float* F_flatten, int n, int size, curandState* states, int j, int cev_split_trials) {
    int* idx = new int[N];
    int* H = new int[N - size];
    int* LH = new int[size];
    int* U = new int[size];
    float min_fitness;
    int* min_fitness_LH;

    for (int i = 0; i < cev_split_trials; i++) {
        for (int k = 0; k < N; k++) {
            idx[k] = k;
        }

        for (int k = 0; k < N; k++) {
            curandState state = states[j];
            int l = curand(&state) % (N - k) + k;
            states[j] = state;
            int temp = idx[l];
            idx[l] = idx[k];
            idx[k] = temp;
        }

        for (int k = 0; k < N; k++) {
            if (k < N - size) {
                H[k] = idx[k];
            } else {
                LH[k - N + size] = idx[k];
                U[k - N + size] = pi[idx[k]];
            }
        }

        float current_fitness = get_fitness_conditional(N, pi, D_flatten, F_flatten, H, LH, U, size);

        if (i == 0 || current_fitness < min_fitness) {
            min_fitness = current_fitness;
            min_fitness_LH = LH;
        }
    }

    delete[] idx;
    delete[] H;
    delete[] LH;
    delete[] U;

    int* pi_temp = new int[N];

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < N; k++) {
            pi_temp[k] = pi[k];
        }

        for (int k = 0; k < N; k++) {
            permutate_conditional(N, pi_temp, states, j, min_fitness_LH, size);
        }

        if (get_fitness(N, pi_temp, D_flatten, F_flatten) < get_fitness(N, pi, D_flatten, F_flatten)) {
            for (int k = 0; k < N; k++) {
                pi[k] = pi_temp[k];
            }
        }
    }

    delete[] pi_temp;

    return pi;
}

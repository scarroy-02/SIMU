#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <pthread.h>

// using namespace std;

double H(int N, double a, double x) {
    double num = std::sqrt(N);
    double den = 1 / std::pow(N,9) + std::pow(std::sin(a * x * std::pow(N,7)),2);
    return num / den;
}

double f_1(double x[6]) {
    return H(3, M_PI, x[0]) * H(2, 1.37, x[1]);
}

double f_2(double x[6]) {
    return H(2, 9.33, x[2]) + H(3, 10.2, x[3]);
}

double f_3(double x[6]) {
    return std::min(H(2, M_E, x[4]), std::min(H(2, 1.2, x[1]), H(3, 1.37, x[5])));
}

double f_4(double x[6]) {
    return std::max(H(3, 1 / M_PI, x[0]), std::max(H(2, 0.9734, x[1]), H(2, std::sqrt(7), x[3])));
}

double f_5(double x[6]) {
    return std::abs(H(2, 3.37, x[5]) - H(3, 4.97, x[1]));
}


double f_6(double x[6]) {
    return 1 / (-(H(2, M_PI, x[0]) + H(3, 9.33, x[2]) + H(2, 10.2, x[3])));
}

double f(double x[6]) {
    return (f_1(x) * f_2(x) + f_3(x) * f_5(x) + f_4(x)) * f_6(x);
}

double y(double t, int k) { return (2 * t - 1) / (2 * k); };

double a_k(int k) {
    double sum = 0;
    for (int t_1 = 1; t_1 <= k; t_1++) {
        for (int t_2 = 1; t_2 <= k; t_2++) {
            for (int t_3 = 1; t_3 <= k; t_3++) {
                for (int t_4 = 1; t_4 <= k; t_4++) {
                    for (int t_5 = 1; t_5 <= k; t_5++) {
                        for (int t_6 = 1; t_6 <= k; t_6++) {
                            double values[6] = {y(t_1,k), y(t_2,k), y(t_3,k), y(t_4,k), y(t_5,k), y(t_6,k)};
                            sum += f(values);
                        }
                    }
                }
            }
        }
    }
    return sum / std::pow(k, 6);
}

double b_k_single(int k) {

    double total_sum = 0.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0,1.0);

    for (int i = 0; i < k * k * k; i++) {
        double sample[k*k*k][6];
        for (int j = 0; j < k*k*k; j++) { 
            for (int m = 0; m < 6; m++) {
                sample[j][m] = dis(gen);
            }
            total_sum += f(sample[j]);
        }
    
    }

    // Returning the normalized sum.
    return total_sum / pow(k, 6);
}

int main() {
    int k[7] = {10,12,14,16,18,20,25}; // You can change this to your desired value of k.

    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::ofstream File;
    File.open ("output.txt");

    for (int i = 0; i < 7; i++){
        double result1 = a_k(k[i]);
        double result2 = b_k_single(k[i]);
        File << "k = \t";
        File << k[i];
        File << "\n";
        File << "2^k = \t";
        File << std::pow(2,k[i]);
        File << "\n";
        File << "a_k = \t";
        File << result1;
        File << "\n";
        File << "b_k = \t";
        File << result2;
        File << "\n\n";
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        std::cout << "k: " << k[i] << std::endl;
        std::cout << "Result a_k: " << result1 << std::endl;
        std::cout << "Result b_k: " << result2 << std::endl;
        std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    }
    
    File.close();

    return 0;
}

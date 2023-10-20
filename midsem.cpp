#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

double H(int N, double a, double x) {
    double num = std::sqrt(N);
    double den = 1 / std::pow(N,9) + std::pow(std::sin(a * x * std::pow(N,7)),2);
    return num / den;
}

double f_1(double x[6]) {
    return H(3, M_PI, x[0]) * H(4, 1.37, x[1]);
}

double f_2(double x[6]) {
    return H(2, 9.33, x[2]) + H(3, 10.2, x[3]);
}

double f_3(double x[6]) {
    return std::min(H(4, M_E, x[4]), std::min(H(5, 1.2, x[1]), H(3, 1.37, x[5])));
}

double f_4(double x[6]) {
    return std::max(H(3, 1 / M_PI, x[0]), std::max(H(7, 0.9734, x[1]), H(4, std::sqrt(7), x[3])));
}

double f_5(double x[6]) {
    return std::abs(H(5, 3.37, x[5]) - H(7, 0.97, x[1]));
}


double f_6(double x[6]) {
    return std::exp(-(H(25, M_PI, x[0]) + H(31, 9.33, x[2]) + H(47, 1.2, x[3])));
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


int main() {
    int k = 20; // You can change this to your desired value of k.

    auto start_time = std::chrono::high_resolution_clock::now();

    double result = a_k(k);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "Result: " << result << std::endl;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    return 0;
}
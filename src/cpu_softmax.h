#ifndef CPU_SOFTMAX_H
#define CPU_SOFTMAX_H

#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <cassert>

/**
 * @brief Numerically stable CPU Softmax implementation using double-precision accumulator.
 * 
 * Formula:
 *   1. m = max_j (x_{r, j})
 *   2. s = sum_j exp(x_{r, j} - m)
 *   3. y_{r, i} = exp(x_{r, i} - m) / s
 * 
 * @param input Host pointer to row-major input matrix (rows x cols)
 * @param output Host pointer to row-major output matrix (rows x cols)
 * @param rows Number of rows in matrix
 * @param cols Number of columns in matrix
 * @return Execution time in milliseconds
 */
inline double cpu_softmax(const float* input, float* output, int rows, int cols) {
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < rows; ++r) {
        const float* row_in = input + r * cols;
        float* row_out = output + r * cols;

        // Step 1: Find row maximum for numerical stability
        float row_max = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < cols; ++c) {
            if (row_in[c] > row_max) {
                row_max = row_in[c];
            }
        }

        // Step 2: Compute sum of shifted exponentials in double precision
        double sum_exp = 0.0;
        for (int c = 0; c < cols; ++c) {
            double shifted = static_cast<double>(row_in[c]) - static_cast<double>(row_max);
            double val = std::exp(shifted);
            sum_exp += val;
        }

        // Step 3: Normalize and write output probabilities
        double inv_sum = 1.0 / sum_exp;
        for (int c = 0; c < cols; ++c) {
            double shifted = static_cast<double>(row_in[c]) - static_cast<double>(row_max);
            double prob = std::exp(shifted) * inv_sum;
            row_out[c] = static_cast<float>(prob);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
    return elapsed.count();
}

/**
 * @brief Detailed validation metrics comparing test output against reference.
 */
struct ValidationResult {
    bool passed;
    float max_abs_error;
    float mean_abs_error;
    float max_row_sum_error;
    int mismatched_elements;
    int total_elements;
};

/**
 * @brief Validates Softmax matrix outputs with strict 1e-5f tolerance,
 * non-negativity checks, finite value checks, and row-sum checks.
 */
inline ValidationResult validate_softmax_output(const float* ref, const float* test, int rows, int cols, float tol = 1.0e-5f) {
    ValidationResult res;
    res.passed = true;
    res.max_abs_error = 0.0f;
    res.mean_abs_error = 0.0f;
    res.max_row_sum_error = 0.0f;
    res.mismatched_elements = 0;
    res.total_elements = rows * cols;

    double sum_abs_err = 0.0;

    for (int r = 0; r < rows; ++r) {
        double row_sum_test = 0.0;

        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            float r_val = ref[idx];
            float t_val = test[idx];

            // 1. Check finite (not NaN, not Inf) and non-negative
            if (std::isnan(t_val) || std::isinf(t_val) || t_val < 0.0f) {
                res.passed = false;
                res.mismatched_elements++;
            }

            float diff = std::fabs(r_val - t_val);
            if (diff > res.max_abs_error) {
                res.max_abs_error = diff;
            }
            sum_abs_err += diff;

            if (diff > tol) {
                res.mismatched_elements++;
            }

            row_sum_test += static_cast<double>(t_val);
        }

        // 2. Check row sum deviation from 1.0
        float row_sum_diff = std::fabs(static_cast<float>(row_sum_test - 1.0));
        if (row_sum_diff > res.max_row_sum_error) {
            res.max_row_sum_error = row_sum_diff;
        }
        if (row_sum_diff > tol * 5.0f) { // Allow minor reduction summation variance
            if (row_sum_diff > 1.0e-4f) {
                res.passed = false;
            }
        }
    }

    res.mean_abs_error = static_cast<float>(sum_abs_err / (rows * cols));

    if (res.max_abs_error > tol || res.mismatched_elements > 0) {
        res.passed = false;
    }

    return res;
}

#endif // CPU_SOFTMAX_H

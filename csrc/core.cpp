#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

static inline double dot(int n, const double* a, const double* b) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

void clip(int n, const double* mat, const double* vec, int* result) {
    int* u = new int[n];
    int* step = new int[n];
    int* d = new int[n + 1];
    for (int i = 0; i < n + 1; i++) {
        d[i] = n - 1;
    }

    double* F = new double[n * n];
    double* p = new double[n];
    double* dist = new double[n + 1];
    dist[n] = 0;

    double rho = std::numeric_limits<double>::infinity();
    int k = n;
    memcpy(F + (n - 1) * n, vec, n * sizeof(double));

    while (true) {
        do {
            if (k != 0) {
                k--;
                for (int j = d[k]; j > k; j--) {
                    F[(j - 1) * n + k] = F[j * n + k] - u[j] * mat[j * n + k];
                }
                p[k] = F[k * n + k] / mat[k * n + k];
                u[k] = round(p[k]);
                double g = (p[k] - u[k]) * mat[k * n + k];
                step[k] = g > 0 ? 1 : -1;
                dist[k] = dist[k + 1] + g * g;
            } else {
                memcpy(result, u, n * sizeof(int));
                rho = dist[0];
            }
        } while (dist[k] < rho);
        int m = k;
        do {
            if (k == n - 1) {
                delete[] u;
                delete[] step;
                delete[] d;
                delete[] p;
                delete[] dist;
                delete[] F;
                return;
            }
            k++;
            u[k] += step[k];
            step[k] = -step[k] - (step[k] > 0 ? 1 : -1);
            double g = (p[k] - u[k]) * mat[k * n + k];
            dist[k] = dist[k + 1] + g * g;
        } while (dist[k] >= rho);

        for (int i = m; i < k; i++) {
            d[i] = k;
        }
        for (int i = m - 1; i >= 0; i--) {
            if (d[i] < k) {
                d[i] = k;
            } else {
                break;
            }
        }
    }
}

void reduce(int n, double* mat, double delta) {
    int i;
    double delta_square = delta * delta;

    // Allocate Buffers
    double* Q = new double[n * n];
    double* buffer = new double[n];
    double* q_norm = new double[n];

    while (true) {
        // Gram-Schmidt
        for (int row = 0; row < n; row++) {
            double* dst_line = Q + row * n;
            double* src_line = mat + row * n;
            for (int j = 0; j < row; j++) {
                buffer[j] = dot(n, src_line, Q + j * n) / q_norm[j];
            }
            for (int col = 0; col < n; col++) {
                double val = src_line[col];
                for (int j = 0; j < row; j++) {
                    val -= Q[j * n + col] * buffer[j];
                }
                dst_line[col] = val;
            }
            q_norm[row] = dot(n, dst_line, dst_line);
        }
        // Reduction Step
        for (i = 1; i < n; i++) {
            for (int j = i - 1; j >= 0; j--) {
                double coeff = round(dot(n, mat + i * n, Q + j * n) / q_norm[j]);
                for (int col = 0; col < n; ++col) {
                    mat[i * n + col] -= coeff * mat[j * n + col];
                }
            }
        }
        // Swap Step
        for (i = 0; i < n - 1; i++) {
            // Compute
            double norm_0 = dot(n, Q + i * n, Q + i * n);
            double coeff = dot(n, mat + (i + 1) * n, Q + i * n) / norm_0;
            double norm_1 = 0.0;
            for (int col = 0; col < n; ++col) {
                double elem = coeff * Q[i * n + col] + Q[(i + 1) * n + col];
                norm_1 += elem * elem;
            }
            if (delta_square * norm_0 > norm_1) {
                memcpy(buffer, mat + i * n, n * sizeof(double));
                memcpy(mat + i * n, mat + (i + 1) * n, n * sizeof(double));
                memcpy(mat + (i + 1) * n, buffer, n * sizeof(double));
                break;
            }
        }
        if (i == n - 1) {
            break;
        }
    }
    delete[] Q;
    delete[] buffer;
    delete[] q_norm;
}

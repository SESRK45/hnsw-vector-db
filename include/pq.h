#pragma once
#include <vector>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace hnsw {

// Product Quantizer using Lloyd's K-Means and Asymmetric Distance Computation (ADC)
class ProductQuantizer {
public:
    int M_pq;   // Number of sub-vectors
    int K;      // Centroids per sub-space
    int D;      // Total dimensionality
    int D_sub;  // Dimensions per sub-space (D / M_pq)

    // centroids[m][k][d]
    std::vector<std::vector<std::vector<float>>> centroids;
    bool is_trained = false;

    ProductQuantizer(int d = 128, int m = 16, int k = 256)
        : M_pq(m), K(k), D(d), D_sub(d / m) {
        if (m <= 0 || d % m != 0) {
            throw std::invalid_argument("Dimension D must be divisible by M_pq");
        }
        centroids.resize(M_pq, std::vector<std::vector<float>>(K, std::vector<float>(D_sub, 0.0f)));
    }

    void train(const std::vector<std::vector<float>>& dataset, int iterations = 10) {
        if (dataset.empty()) {
            throw std::invalid_argument("Cannot train PQ on an empty dataset");
        }
        if (static_cast<int>(dataset[0].size()) != D) {
            D = static_cast<int>(dataset[0].size());
            if (D % M_pq != 0) {
                throw std::invalid_argument("Dataset dimension must be divisible by M_pq");
            }
            D_sub = D / M_pq;
            centroids.assign(M_pq, std::vector<std::vector<float>>(K, std::vector<float>(D_sub, 0.0f)));
        }

        std::mt19937 rng(1337);
        std::uniform_int_distribution<size_t> dist(0, dataset.size() - 1);

        // Random centroid initialization
        for (int m = 0; m < M_pq; ++m) {
            for (int k = 0; k < K; ++k) {
                size_t random_idx = dist(rng);
                for (int d = 0; d < D_sub; ++d) {
                    centroids[m][k][d] = dataset[random_idx][m * D_sub + d];
                }
            }
        }

        // Lloyd's K-Means iterations
        for (int iter = 0; iter < iterations; ++iter) {
            std::vector<std::vector<std::vector<float>>> new_centroids(
                M_pq, std::vector<std::vector<float>>(K, std::vector<float>(D_sub, 0.0f)));
            std::vector<std::vector<int>> counts(M_pq, std::vector<int>(K, 0));

            for (const auto& vec : dataset) {
                for (int m = 0; m < M_pq; ++m) {
                    float min_dist = 1e30f;
                    int best_k = 0;
                    for (int k = 0; k < K; ++k) {
                        float sub_dist = 0.0f;
                        for (int d = 0; d < D_sub; ++d) {
                            float diff = vec[m * D_sub + d] - centroids[m][k][d];
                            sub_dist += diff * diff;
                        }
                        if (sub_dist < min_dist) {
                            min_dist = sub_dist;
                            best_k = k;
                        }
                    }
                    counts[m][best_k]++;
                    for (int d = 0; d < D_sub; ++d) {
                        new_centroids[m][best_k][d] += vec[m * D_sub + d];
                    }
                }
            }

            for (int m = 0; m < M_pq; ++m) {
                for (int k = 0; k < K; ++k) {
                    if (counts[m][k] > 0) {
                        for (int d = 0; d < D_sub; ++d) {
                            centroids[m][k][d] = new_centroids[m][k][d] / counts[m][k];
                        }
                    } else {
                        size_t random_idx = dist(rng);
                        for (int d = 0; d < D_sub; ++d) {
                            centroids[m][k][d] = dataset[random_idx][m * D_sub + d];
                        }
                    }
                }
            }
        }
        is_trained = true;
    }

    std::vector<uint8_t> encode(const std::vector<float>& vec) const {
        std::vector<uint8_t> codes(M_pq);
        for (int m = 0; m < M_pq; ++m) {
            float min_dist = 1e30f;
            int best_k = 0;
            for (int k = 0; k < K; ++k) {
                float sub_dist = 0.0f;
                for (int d = 0; d < D_sub; ++d) {
                    float diff = vec[m * D_sub + d] - centroids[m][k][d];
                    sub_dist += diff * diff;
                }
                if (sub_dist < min_dist) {
                    min_dist = sub_dist;
                    best_k = k;
                }
            }
            codes[m] = static_cast<uint8_t>(best_k);
        }
        return codes;
    }

    std::vector<std::vector<float>> compute_lut(const std::vector<float>& query) const {
        std::vector<std::vector<float>> lut(M_pq, std::vector<float>(K, 0.0f));
        for (int m = 0; m < M_pq; ++m) {
            for (int k = 0; k < K; ++k) {
                float sub_dist = 0.0f;
                for (int d = 0; d < D_sub; ++d) {
                    float diff = query[m * D_sub + d] - centroids[m][k][d];
                    sub_dist += diff * diff;
                }
                lut[m][k] = sub_dist;
            }
        }
        return lut;
    }

    inline float adc_distance(const std::vector<std::vector<float>>& lut,
                              const std::vector<uint8_t>& codes) const {
        float dist = 0.0f;
        for (int m = 0; m < M_pq; ++m) {
            dist += lut[m][codes[m]];
        }
        return dist;
    }
};

} // namespace hnsw

#pragma once
#include <vector>
#include <cmath>
#include <immintrin.h>
#include "node.h"

namespace hnsw {

// AVX2-accelerated SQ8 dequantization + L2 or Angular distance
inline float l2_distance_quantized(const float* query, size_t d, const Node& node) {
    float scale = (node.v_max - node.v_min) / 255.0f;
    float min_val = node.v_min;

#ifdef USE_ANGULAR_DISTANCE
    __m256 sum256 = _mm256_setzero_ps();
    __m256 v_scale = _mm256_set1_ps(scale);
    __m256 v_min = _mm256_set1_ps(min_val);

    size_t i = 0;
    for (; i + 7 < d; i += 8) {
        __m256 v_q = _mm256_loadu_ps(&query[i]);
        alignas(32) float dequant[8];
        for (int j = 0; j < 8; ++j) {
            dequant[j] = static_cast<float>(node.quantized_vector[i + j]);
        }
        __m256 v_int = _mm256_load_ps(dequant);
        __m256 v_deq = _mm256_fmadd_ps(v_int, v_scale, v_min);
        sum256 = _mm256_fmadd_ps(v_q, v_deq, sum256);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, sum256);
    float total_dist = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < d; ++i) {
        float val_b = node.quantized_vector[i] * scale + min_val;
        total_dist += query[i] * val_b;
    }
    return 1.0f - total_dist;
#else
    __m256 sum256 = _mm256_setzero_ps();
    __m256 v_scale = _mm256_set1_ps(scale);
    __m256 v_min = _mm256_set1_ps(min_val);

    size_t i = 0;
    for (; i + 7 < d; i += 8) {
        __m256 v_q = _mm256_loadu_ps(&query[i]);
        alignas(32) float dequant[8];
        for (int j = 0; j < 8; ++j) {
            dequant[j] = static_cast<float>(node.quantized_vector[i + j]);
        }
        __m256 v_int = _mm256_load_ps(dequant);
        __m256 v_deq = _mm256_fmadd_ps(v_int, v_scale, v_min);
        __m256 diff = _mm256_sub_ps(v_q, v_deq);
        sum256 = _mm256_fmadd_ps(diff, diff, sum256);
    }

    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, sum256);
    float total_dist = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for (; i < d; ++i) {
        float val_b = node.quantized_vector[i] * scale + min_val;
        float diff = query[i] - val_b;
        total_dist += diff * diff;
    }
    return total_dist;
#endif
}

// Distance between two SQ8-compressed nodes
inline float l2_distance_quantized_nodes(const Node& a, const Node& b) {
    size_t d = a.quantized_vector.size();
    float scale_a = (a.v_max - a.v_min) / 255.0f;
    float min_a = a.v_min;

    float scale_b = (b.v_max - b.v_min) / 255.0f;
    float min_b = b.v_min;

#ifdef USE_ANGULAR_DISTANCE
    float total_dist = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        float val_a = a.quantized_vector[i] * scale_a + min_a;
        float val_b = b.quantized_vector[i] * scale_b + min_b;
        total_dist += val_a * val_b;
    }
    return 1.0f - total_dist;
#else
    float total_dist = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        float val_a = a.quantized_vector[i] * scale_a + min_a;
        float val_b = b.quantized_vector[i] * scale_b + min_b;
        float diff = val_a - val_b;
        total_dist += diff * diff;
    }
    return total_dist;
#endif
}

// AVX2 FMA3 distance kernel for float32 vectors
inline float l2_distance_avx2(const float* a, const float* b, size_t dim) {
#ifdef USE_ANGULAR_DISTANCE
    __m256 sum256 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 v_a = _mm256_loadu_ps(a + i);
        __m256 v_b = _mm256_loadu_ps(b + i);
        sum256 = _mm256_fmadd_ps(v_a, v_b, sum256);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, sum256);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return 1.0f - sum;
#else
    __m256 sum256 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 v_a = _mm256_loadu_ps(a + i);
        __m256 v_b = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(v_a, v_b);
        sum256 = _mm256_fmadd_ps(diff, diff, sum256);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, sum256);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
#endif
}

inline float l2_distance(const std::vector<float>& a, const std::vector<float>& b) {
    return l2_distance_avx2(a.data(), b.data(), a.size());
}

// Exact float32 distance used for candidate re-ranking
inline float l2_distance_exact(const float* query, size_t d, const Node& node) {
    if (node.original_vector.empty()) {
        return l2_distance_quantized(query, d, node);
    }
    return l2_distance_avx2(query, node.original_vector.data(), d);
}

} // namespace hnsw

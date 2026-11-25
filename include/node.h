#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include "arena.h"

namespace hnsw {

using NodeId = int;

struct Node {
    NodeId id;

    // Original float32 vector retained for exact re-ranking
    std::vector<float> original_vector;

    // Asymmetric local SQ8 quantized vector (4x reduction vs float32)
    std::vector<uint8_t> quantized_vector;
    float v_min = 0.0f;
    float v_max = 0.0f;

    bool deleted = false;
    uint32_t metadata_label = 0;

    // Product quantization codes
    std::vector<uint8_t> pq_codes;

    // Contiguous arena-backed adjacency lists per layer
    std::vector<ArenaSpan<NodeId>> edges;

    Node() : id(-1) {}

    Node(NodeId _id, const float* _vec, size_t dim, int max_level, uint32_t _label,
         ArenaAllocator<NodeId>* arena, int m_max, int m_max0)
        : id(_id), metadata_label(_label), pq_codes(0) {

        original_vector.assign(_vec, _vec + dim);

        v_min = std::numeric_limits<float>::max();
        v_max = std::numeric_limits<float>::lowest();

        for (size_t i = 0; i < dim; ++i) {
            if (_vec[i] < v_min) v_min = _vec[i];
            if (_vec[i] > v_max) v_max = _vec[i];
        }

        float scale = (v_max - v_min) / 255.0f;
        if (scale == 0.0f) scale = 1.0f;
        quantized_vector.resize(dim);
        for (size_t i = 0; i < dim; ++i) {
            float val = std::round((_vec[i] - v_min) / scale);
            val = std::max(0.0f, std::min(255.0f, val));
            quantized_vector[i] = static_cast<uint8_t>(val);
        }

        edges.resize(max_level + 1);
        for (int l = 0; l <= max_level; ++l) {
            // Allocate extra headroom on Layer 0 for density-aware degree expansion
            int cap = (l == 0) ? static_cast<int>(m_max0 * 1.5) + 1 : m_max + 1;
            edges[l].data_ptr = arena->allocate(cap);
            edges[l].capacity = cap;
            edges[l].count = 0;
        }
    }
};

} // namespace hnsw

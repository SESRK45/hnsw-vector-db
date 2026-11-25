#pragma once
#include <vector>
#include <random>
#include <string>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <memory>
#include <cmath>
#include "node.h"
#include "priority_queue.h"
#include "pq.h"
#include "arena.h"

namespace hnsw {

class HNSWGraph {
private:
    std::vector<Node> nodes;
    std::vector<std::unique_ptr<std::shared_mutex>> node_locks;

    std::atomic<NodeId> enter_point{-1};
    std::atomic<int> max_layer{0};
    std::mutex ep_lock;

    int M;
    int M_max;
    int M_max0;
    int ef_construction;
    double mult;

    std::mt19937 rng;

    int get_random_level();
    MaxHeap search_layer(const float* query_data, size_t dim, NodeId ep, int ef, int layer);
    std::vector<NodeId> select_neighbors(const float* query_data, size_t dim, MaxHeap& candidates, int M_limit, int layer);

public:
    ProductQuantizer pq;
    ArenaAllocator<NodeId> edge_arena;

    HNSWGraph(int M_val = 16, int ef_const = 200, size_t max_edges_capacity = 50000000)
        : M(M_val), M_max(M_val), M_max0(M_val * 2), ef_construction(ef_const), edge_arena(max_edges_capacity) {
        mult = 1.0 / std::log(1.0 * M);
        rng.seed(1337);
    }

    void reserve_nodes(size_t max_nodes) {
        nodes.resize(max_nodes);
        while (node_locks.size() < max_nodes) {
            node_locks.push_back(std::make_unique<std::shared_mutex>());
        }
    }

    void insert(NodeId id, const float* vec_data, size_t dim, uint32_t label = 0);
    void remove(NodeId id);
    std::vector<NodeId> search(const float* query_data, size_t dim, int k, int ef_search);
    std::vector<NodeId> search_filtered(const float* query_data, size_t dim, int k, int ef_search, uint32_t target_label);

    void build_pq(const std::vector<std::vector<float>>& dataset);
    std::vector<NodeId> search_pq(const float* query_data, size_t dim, int k, int ef_search);

    void save(const std::string& filename) const;
    void load(const std::string& filename);
    void export_to_mmap(const std::string& filename) const;

    size_t size() const { return nodes.size(); }
};

} // namespace hnsw

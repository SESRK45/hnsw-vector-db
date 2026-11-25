#include "hnsw.h"
#include "distance.h"
#include "mmap_hnsw.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_set>

namespace hnsw {

namespace {

std::vector<NodeId> copy_layer_edges(const Node& node, size_t layer,
                                     const std::vector<std::unique_ptr<std::shared_mutex>>& locks,
                                     NodeId node_id) {
    std::vector<NodeId> out;
    auto copy_span = [&](const ArenaSpan<NodeId>& span) {
        if (span.data_ptr != nullptr && span.count > 0) {
            out.assign(span.data_ptr, span.data_ptr + span.count);
        }
    };
    if (locks.size() > static_cast<size_t>(node_id) && locks[node_id]) {
        std::shared_lock<std::shared_mutex> lock(*locks[node_id]);
        copy_span(node.edges[layer]);
    } else {
        copy_span(node.edges[layer]);
    }
    return out;
}

} // namespace

int HNSWGraph::get_random_level() {
    thread_local std::mt19937 tl_rng(1337);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(tl_rng);
    if (r == 0.0) r = 1e-6;
    return static_cast<int>(-std::log(r) * mult);
}

MaxHeap HNSWGraph::search_layer(const float* query_data, size_t dim, NodeId ep, int ef, int layer) {
    MinHeap candidates;
    MaxHeap top_results;
    std::unordered_set<NodeId> visited;

    visited.insert(ep);
    float initial_dist = l2_distance_quantized(query_data, dim, nodes.at(ep));
    candidates.push({initial_dist, ep});
    top_results.push({initial_dist, ep});

    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();

        float dist_c = current.first;
        NodeId c = current.second;

        if (dist_c > top_results.top().first && top_results.size() >= static_cast<size_t>(ef)) {
            break;
        }

        std::vector<NodeId> current_edges = copy_layer_edges(nodes.at(c), layer, node_locks, c);

        for (NodeId neighbor : current_edges) {
            if (visited.find(neighbor) == visited.end()) {
                __builtin_prefetch(nodes.at(neighbor).quantized_vector.data(), 0, 1);
            }
        }

        for (NodeId neighbor : current_edges) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                float dist_n = l2_distance_quantized(query_data, dim, nodes.at(neighbor));

                if (top_results.size() < static_cast<size_t>(ef) || dist_n < top_results.top().first) {
                    candidates.push({dist_n, neighbor});
                    top_results.push({dist_n, neighbor});

                    if (top_results.size() > static_cast<size_t>(ef)) {
                        top_results.pop();
                    }
                }
            }
        }
    }

    return top_results;
}

std::vector<NodeId> HNSWGraph::select_neighbors(const float* query_data, size_t dim, MaxHeap& candidates,
                                                int base_M_limit, int layer) {
    (void)query_data;
    (void)dim;

    std::vector<NodeId> selected;
    std::vector<std::pair<float, NodeId>> candidates_vec;

    while (!candidates.empty()) {
        candidates_vec.push_back(candidates.top());
        candidates.pop();
    }
    std::reverse(candidates_vec.begin(), candidates_vec.end());

    if (candidates_vec.empty()) return selected;

    float density_factor = 0.0f;
    for (const auto& c : candidates_vec) {
        density_factor += c.first;
    }
    density_factor /= candidates_vec.size();

    int dynamic_M = base_M_limit;
    if (layer == 0 && density_factor > 1.5f) {
        dynamic_M = static_cast<int>(base_M_limit * 1.5);
    }

    // Vamana-style alpha-pruning (relaxed alpha = 1.2 on Layer 0 to preserve diverse paths)
    float alpha = (layer == 0) ? 1.2f : 1.0f;

    for (const auto& c : candidates_vec) {
        if (selected.size() >= static_cast<size_t>(dynamic_M)) break;

        NodeId cand_id = c.second;
        float dist_to_query = c.first;
        bool good = true;

        for (NodeId sel_id : selected) {
            float dist_cand_to_sel = l2_distance_quantized_nodes(nodes.at(cand_id), nodes.at(sel_id));
            if (alpha * dist_cand_to_sel < dist_to_query) {
                good = false;
                break;
            }
        }

        if (good) selected.push_back(cand_id);
    }

    // Ensure minimum connectivity in sparse neighborhoods
    if (selected.size() < static_cast<size_t>(base_M_limit / 2)) {
        for (const auto& c : candidates_vec) {
            if (selected.size() >= static_cast<size_t>(base_M_limit / 2)) break;
            if (std::find(selected.begin(), selected.end(), c.second) == selected.end()) {
                selected.push_back(c.second);
            }
        }
    }

    return selected;
}

void HNSWGraph::insert(NodeId id, const float* vec_data, size_t dim, uint32_t label) {
    int level = get_random_level();

    if (id < 0 || id >= static_cast<int>(nodes.size())) {
        throw std::runtime_error("Graph nodes must be reserved via reserve_nodes() before insert()");
    }
    nodes[id] = Node(id, vec_data, dim, level, label, &edge_arena, M_max, M_max0);

    if (enter_point.load(std::memory_order_acquire) == -1) {
        std::unique_lock<std::mutex> lock(ep_lock);
        if (enter_point.load(std::memory_order_relaxed) == -1) {
            max_layer.store(level, std::memory_order_relaxed);
            enter_point.store(id, std::memory_order_release);
            return;
        }
    }

    NodeId curr_ep = enter_point.load(std::memory_order_acquire);
    int curr_max_layer = max_layer.load(std::memory_order_relaxed);

    float min_dist = l2_distance_quantized(vec_data, dim, nodes.at(curr_ep));

    for (int l = curr_max_layer; l > level; --l) {
        bool changed = true;
        while (changed) {
            changed = false;

            std::vector<NodeId> current_edges = copy_layer_edges(nodes.at(curr_ep), l, node_locks, curr_ep);

            for (NodeId neighbor : current_edges) {
                float dist = l2_distance_quantized(vec_data, dim, nodes.at(neighbor));
                if (dist < min_dist) {
                    min_dist = dist;
                    curr_ep = neighbor;
                    changed = true;
                }
            }
        }
    }

    for (int l = std::min(curr_max_layer, level); l >= 0; --l) {
        MaxHeap W = search_layer(vec_data, dim, curr_ep, ef_construction, l);

        int M_limit = (l == 0) ? M_max0 : M_max;
        std::vector<NodeId> neighbors = select_neighbors(vec_data, dim, W, M_limit, l);

        {
            std::unique_lock<std::shared_mutex> lock(*node_locks[id]);
            nodes.at(id).edges[l].assign(neighbors);
        }

        for (NodeId neighbor : neighbors) {
            std::unique_lock<std::shared_mutex> lock(*node_locks[neighbor]);
            nodes.at(neighbor).edges[l].push_back(id);
            if (nodes.at(neighbor).edges[l].size() > static_cast<size_t>(M_limit)) {
                MaxHeap n_candidates;
                for (NodeId n_neighbor : nodes.at(neighbor).edges[l]) {
                    float d = l2_distance_quantized_nodes(nodes.at(neighbor), nodes.at(n_neighbor));
                    n_candidates.push({d, n_neighbor});
                }
                nodes.at(neighbor).edges[l].assign(select_neighbors(vec_data, dim, n_candidates, M_limit, l));
            }
        }

        if (!neighbors.empty()) {
            curr_ep = neighbors.front();
        }
    }

    if (level > max_layer.load(std::memory_order_relaxed)) {
        std::unique_lock<std::mutex> lock(ep_lock);
        if (level > max_layer.load(std::memory_order_relaxed)) {
            max_layer.store(level, std::memory_order_relaxed);
            enter_point.store(id, std::memory_order_release);
        }
    }
}

std::vector<NodeId> HNSWGraph::search(const float* query_data, size_t dim, int k, int ef_search) {
    NodeId curr_ep = enter_point.load(std::memory_order_acquire);
    if (curr_ep == -1) return {};
    int curr_max_layer = max_layer.load(std::memory_order_relaxed);

    float min_dist = l2_distance_quantized(query_data, dim, nodes.at(curr_ep));

    for (int l = curr_max_layer; l > 0; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            std::vector<NodeId> current_edges = copy_layer_edges(nodes.at(curr_ep), l, node_locks, curr_ep);
            for (NodeId neighbor : current_edges) {
                float dist = l2_distance_quantized(query_data, dim, nodes.at(neighbor));
                if (dist < min_dist) {
                    min_dist = dist;
                    curr_ep = neighbor;
                    changed = true;
                }
            }
        }
    }

    int effective_ef = std::max(k, ef_search);
    MaxHeap W = search_layer(query_data, dim, curr_ep, effective_ef, 0);

    std::vector<std::pair<float, NodeId>> rerank_pool;
    while (!W.empty()) {
        NodeId res_id = W.top().second;
        if (!nodes.at(res_id).deleted) {
            float exact_dist = l2_distance_exact(query_data, dim, nodes.at(res_id));
            rerank_pool.push_back({exact_dist, res_id});
        }
        W.pop();
    }

    std::sort(rerank_pool.begin(), rerank_pool.end());

    std::vector<NodeId> results;
    for (size_t i = 0; i < std::min(static_cast<size_t>(k), rerank_pool.size()); ++i) {
        results.push_back(rerank_pool[i].second);
    }

    return results;
}

std::vector<NodeId> HNSWGraph::search_filtered(const float* query_data, size_t dim, int k, int ef_search,
                                               uint32_t target_label) {
    NodeId curr_ep = enter_point.load(std::memory_order_acquire);
    if (curr_ep == -1) return {};
    int curr_max_layer = max_layer.load(std::memory_order_relaxed);

    float min_dist = l2_distance_quantized(query_data, dim, nodes.at(curr_ep));
    for (int l = curr_max_layer; l > 0; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            std::vector<NodeId> current_edges = copy_layer_edges(nodes.at(curr_ep), l, node_locks, curr_ep);
            for (NodeId neighbor : current_edges) {
                float dist = l2_distance_quantized(query_data, dim, nodes.at(neighbor));
                if (dist < min_dist) {
                    min_dist = dist;
                    curr_ep = neighbor;
                    changed = true;
                }
            }
        }
    }

    std::unordered_set<NodeId> visited;
    visited.insert(curr_ep);

    MinHeap candidates;
    MaxHeap filtered_results;

    float initial_dist = l2_distance_quantized(query_data, dim, nodes.at(curr_ep));
    candidates.push({initial_dist, curr_ep});

    if (!nodes.at(curr_ep).deleted && nodes.at(curr_ep).metadata_label == target_label) {
        filtered_results.push({initial_dist, curr_ep});
    }

    int max_exploration = ef_search * 10;
    int explored = 0;

    while (!candidates.empty() && explored < max_exploration) {
        auto current = candidates.top();
        candidates.pop();
        explored++;

        float dist_c = current.first;
        NodeId c = current.second;

        if (filtered_results.size() >= static_cast<size_t>(ef_search) && dist_c > filtered_results.top().first) {
            break;
        }

        std::vector<NodeId> current_edges = copy_layer_edges(nodes.at(c), 0, node_locks, c);

        for (NodeId neighbor : current_edges) {
            if (visited.find(neighbor) == visited.end()) {
                __builtin_prefetch(nodes.at(neighbor).quantized_vector.data(), 0, 1);
            }
        }

        for (NodeId neighbor : current_edges) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);

                float dist_n = l2_distance_quantized(query_data, dim, nodes.at(neighbor));
                candidates.push({dist_n, neighbor});

                if (!nodes.at(neighbor).deleted && nodes.at(neighbor).metadata_label == target_label) {
                    filtered_results.push({dist_n, neighbor});
                    if (filtered_results.size() > static_cast<size_t>(ef_search)) {
                        filtered_results.pop();
                    }
                }
            }
        }
    }

    std::vector<NodeId> results;
    while (!filtered_results.empty()) {
        results.push_back(filtered_results.top().second);
        filtered_results.pop();
    }
    std::reverse(results.begin(), results.end());

    if (results.size() > static_cast<size_t>(k)) results.resize(k);
    return results;
}

void HNSWGraph::build_pq(const std::vector<std::vector<float>>& dataset) {
    pq.train(dataset, 10);

    #pragma omp parallel for
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].id != -1) {
            std::vector<uint8_t> codes = pq.encode(dataset[nodes[i].id]);
            std::unique_lock<std::shared_mutex> lock(*node_locks[nodes[i].id]);
            nodes[i].pq_codes = std::move(codes);
        }
    }
}

std::vector<NodeId> HNSWGraph::search_pq(const float* query_data, size_t dim, int k, int ef_search) {
    NodeId curr_ep = enter_point.load(std::memory_order_acquire);
    if (curr_ep == -1 || !pq.is_trained) return {};
    int curr_max_layer = max_layer.load(std::memory_order_relaxed);

    std::vector<float> query_vec(query_data, query_data + dim);
    std::vector<std::vector<float>> lut = pq.compute_lut(query_vec);
    float min_dist = pq.adc_distance(lut, nodes.at(curr_ep).pq_codes);

    for (int l = curr_max_layer; l > 0; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            std::vector<NodeId> current_edges = copy_layer_edges(nodes.at(curr_ep), l, node_locks, curr_ep);
            for (NodeId neighbor : current_edges) {
                float dist = pq.adc_distance(lut, nodes.at(neighbor).pq_codes);
                if (dist < min_dist) {
                    min_dist = dist;
                    curr_ep = neighbor;
                    changed = true;
                }
            }
        }
    }

    std::unordered_set<NodeId> visited;
    visited.insert(curr_ep);

    MinHeap candidates;
    MaxHeap top_results;

    float initial_dist = pq.adc_distance(lut, nodes.at(curr_ep).pq_codes);
    candidates.push({initial_dist, curr_ep});
    top_results.push({initial_dist, curr_ep});

    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();

        float dist_c = current.first;
        NodeId c = current.second;

        if (dist_c > top_results.top().first && top_results.size() >= static_cast<size_t>(ef_search)) {
            break;
        }

        std::vector<NodeId> current_edges = copy_layer_edges(nodes.at(c), 0, node_locks, c);

        for (NodeId neighbor : current_edges) {
            if (visited.find(neighbor) == visited.end()) {
                __builtin_prefetch(nodes.at(neighbor).pq_codes.data(), 0, 1);
            }
        }

        for (NodeId neighbor : current_edges) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);

                float dist_n = pq.adc_distance(lut, nodes.at(neighbor).pq_codes);

                if (top_results.size() < static_cast<size_t>(ef_search) || dist_n < top_results.top().first) {
                    candidates.push({dist_n, neighbor});
                    top_results.push({dist_n, neighbor});
                    if (top_results.size() > static_cast<size_t>(ef_search)) {
                        top_results.pop();
                    }
                }
            }
        }
    }

    std::vector<NodeId> results;
    while (!top_results.empty()) {
        NodeId res_id = top_results.top().second;
        if (!nodes.at(res_id).deleted) {
            results.push_back(res_id);
        }
        top_results.pop();
    }
    std::reverse(results.begin(), results.end());

    if (results.size() > static_cast<size_t>(k)) results.resize(k);
    return results;
}

void HNSWGraph::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("Cannot open file for writing: " + filename);

    out.write(reinterpret_cast<const char*>(&M), sizeof(M));
    out.write(reinterpret_cast<const char*>(&M_max), sizeof(M_max));
    out.write(reinterpret_cast<const char*>(&M_max0), sizeof(M_max0));
    out.write(reinterpret_cast<const char*>(&ef_construction), sizeof(ef_construction));
    out.write(reinterpret_cast<const char*>(&mult), sizeof(mult));

    NodeId ep = enter_point.load(std::memory_order_acquire);
    out.write(reinterpret_cast<const char*>(&ep), sizeof(ep));
    int ml = max_layer.load(std::memory_order_relaxed);
    out.write(reinterpret_cast<const char*>(&ml), sizeof(ml));

    size_t num_nodes = nodes.size();
    out.write(reinterpret_cast<const char*>(&num_nodes), sizeof(num_nodes));

    for (const auto& node : nodes) {
        out.write(reinterpret_cast<const char*>(&node.id), sizeof(node.id));
        out.write(reinterpret_cast<const char*>(&node.deleted), sizeof(node.deleted));
        out.write(reinterpret_cast<const char*>(&node.v_min), sizeof(node.v_min));
        out.write(reinterpret_cast<const char*>(&node.v_max), sizeof(node.v_max));
        out.write(reinterpret_cast<const char*>(&node.metadata_label), sizeof(node.metadata_label));

        size_t orig_size = node.original_vector.size();
        out.write(reinterpret_cast<const char*>(&orig_size), sizeof(orig_size));
        if (orig_size > 0) {
            out.write(reinterpret_cast<const char*>(node.original_vector.data()), orig_size * sizeof(float));
        }

        size_t vec_size = node.quantized_vector.size();
        out.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
        if (vec_size > 0) {
            out.write(reinterpret_cast<const char*>(node.quantized_vector.data()), vec_size * sizeof(uint8_t));
        }

        size_t num_layers = node.edges.size();
        out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

        for (const auto& layer_edges : node.edges) {
            size_t num_edges = layer_edges.size();
            out.write(reinterpret_cast<const char*>(&num_edges), sizeof(num_edges));
            if (num_edges > 0) {
                out.write(reinterpret_cast<const char*>(layer_edges.data()), num_edges * sizeof(NodeId));
            }
        }
    }
    out.close();
}

void HNSWGraph::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open file for reading: " + filename);

    in.read(reinterpret_cast<char*>(&M), sizeof(M));
    in.read(reinterpret_cast<char*>(&M_max), sizeof(M_max));
    in.read(reinterpret_cast<char*>(&M_max0), sizeof(M_max0));
    in.read(reinterpret_cast<char*>(&ef_construction), sizeof(ef_construction));
    in.read(reinterpret_cast<char*>(&mult), sizeof(mult));

    NodeId ep;
    in.read(reinterpret_cast<char*>(&ep), sizeof(ep));
    enter_point.store(ep, std::memory_order_release);

    int ml;
    in.read(reinterpret_cast<char*>(&ml), sizeof(ml));
    max_layer.store(ml, std::memory_order_relaxed);

    size_t num_nodes;
    in.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));

    edge_arena.reset();
    nodes.assign(num_nodes, Node());
    node_locks.clear();
    node_locks.reserve(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
        node_locks.push_back(std::make_unique<std::shared_mutex>());
    }

    for (size_t i = 0; i < num_nodes; ++i) {
        Node& node = nodes[i];
        in.read(reinterpret_cast<char*>(&node.id), sizeof(node.id));
        in.read(reinterpret_cast<char*>(&node.deleted), sizeof(node.deleted));
        in.read(reinterpret_cast<char*>(&node.v_min), sizeof(node.v_min));
        in.read(reinterpret_cast<char*>(&node.v_max), sizeof(node.v_max));
        in.read(reinterpret_cast<char*>(&node.metadata_label), sizeof(node.metadata_label));

        size_t orig_size;
        in.read(reinterpret_cast<char*>(&orig_size), sizeof(orig_size));
        node.original_vector.resize(orig_size);
        if (orig_size > 0) {
            in.read(reinterpret_cast<char*>(node.original_vector.data()), orig_size * sizeof(float));
        }

        size_t vec_size;
        in.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));
        node.quantized_vector.resize(vec_size);
        if (vec_size > 0) {
            in.read(reinterpret_cast<char*>(node.quantized_vector.data()), vec_size * sizeof(uint8_t));
        }

        size_t num_layers;
        in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
        node.edges.resize(num_layers);

        for (size_t l = 0; l < num_layers; ++l) {
            size_t num_edges;
            in.read(reinterpret_cast<char*>(&num_edges), sizeof(num_edges));
            size_t base_cap = (l == 0) ? static_cast<size_t>(M_max0 * 1.5) + 1 : static_cast<size_t>(M_max + 1);
            size_t cap = std::max(base_cap, num_edges);
            node.edges[l].data_ptr = edge_arena.allocate(cap);
            node.edges[l].capacity = cap;
            node.edges[l].count = num_edges;
            if (num_edges > 0) {
                in.read(reinterpret_cast<char*>(node.edges[l].data()), num_edges * sizeof(NodeId));
            }
        }
    }
    in.close();
}

void HNSWGraph::remove(NodeId id) {
    if (id < 0 || id >= static_cast<int>(nodes.size()) || nodes.at(id).id == -1 || nodes.at(id).deleted) {
        return;
    }

    {
        std::unique_lock<std::shared_mutex> lock(*node_locks[id]);
        nodes[id].deleted = true;
    }

    // Neighbor Bypass Healing Heuristic: remove deleted node and bridge starved neighbors
    for (size_t l = 0; l < nodes[id].edges.size(); ++l) {
        std::vector<NodeId> neighbors = copy_layer_edges(nodes[id], l, node_locks, id);

        int M_limit = (l == 0) ? M_max0 : M_max;

        for (NodeId neighbor : neighbors) {
            if (nodes[neighbor].deleted) continue;

            std::unique_lock<std::shared_mutex> lock(*node_locks[neighbor]);
            auto& n_edges = nodes[neighbor].edges[l];

            size_t write_idx = 0;
            for (size_t read_idx = 0; read_idx < n_edges.count; ++read_idx) {
                if (n_edges.data_ptr[read_idx] != id) {
                    n_edges.data_ptr[write_idx++] = n_edges.data_ptr[read_idx];
                }
            }
            n_edges.count = write_idx;

            if (n_edges.size() < static_cast<size_t>(M_limit / 2)) {
                for (NodeId replacement : neighbors) {
                    if (replacement != neighbor && replacement != id && !nodes[replacement].deleted) {
                        if (std::find(n_edges.begin(), n_edges.end(), replacement) == n_edges.end()) {
                            n_edges.push_back(replacement);
                            if (n_edges.size() >= static_cast<size_t>(M_limit)) break;
                        }
                    }
                }
            }
        }
    }
}

void HNSWGraph::export_to_mmap(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("Cannot create mmap file");

    NodeId ep = enter_point.load(std::memory_order_acquire);
    out.write(reinterpret_cast<const char*>(&ep), sizeof(ep));
    int ml = max_layer.load(std::memory_order_relaxed);
    out.write(reinterpret_cast<const char*>(&ml), sizeof(ml));
    size_t n_nodes = nodes.size();
    out.write(reinterpret_cast<const char*>(&n_nodes), sizeof(n_nodes));

    for (size_t i = 0; i < n_nodes; ++i) {
        DiskNode dnode = {};
        dnode.deleted = nodes[i].deleted;
        dnode.v_min = nodes[i].v_min;
        dnode.v_max = nodes[i].v_max;
        dnode.metadata_label = nodes[i].metadata_label;

        size_t d = std::min(static_cast<size_t>(MMAP_DIM), nodes[i].quantized_vector.size());
        std::copy(nodes[i].quantized_vector.begin(), nodes[i].quantized_vector.begin() + d, dnode.vec);

        size_t layers = std::min(static_cast<size_t>(MMAP_MAX_LAYERS), nodes[i].edges.size());
        for (size_t l = 0; l < layers; ++l) {
            size_t edge_count = std::min(static_cast<size_t>(MMAP_M_MAX), nodes[i].edges[l].size());
            dnode.num_edges[l] = static_cast<int>(edge_count);
            std::copy(nodes[i].edges[l].begin(), nodes[i].edges[l].begin() + edge_count, dnode.edges[l]);
        }

        out.write(reinterpret_cast<const char*>(&dnode), sizeof(DiskNode));
    }
    out.close();
}

} // namespace hnsw

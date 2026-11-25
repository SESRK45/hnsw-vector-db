#include "mmap_hnsw.h"
#include "priority_queue.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdexcept>
#include <unordered_set>
#include <algorithm>
#include <iostream>

namespace hnsw {

inline float l2_distance_disk(const float* query_data, size_t dim, const DiskNode& node) {
    float scale = (node.v_max - node.v_min) / 255.0f;
    float min_val = node.v_min;
    float total_dist = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float val_b = node.vec[i] * scale + min_val;
        float diff = query_data[i] - val_b;
        total_dist += diff * diff;
    }
    return total_dist;
}

MmapHNSWGraph::MmapHNSWGraph(const std::string& filename) {
    fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) throw std::runtime_error("Could not open mmap file");

    struct stat sb;
    if (fstat(fd, &sb) == -1) throw std::runtime_error("Could not stat mmap file");
    file_size = sb.st_size;

    // Map binary index directly into virtual memory and disable sequential readahead
    mapped_data = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped_data == MAP_FAILED) throw std::runtime_error("mmap failed");

    madvise(mapped_data, file_size, MADV_RANDOM);

    char* ptr = static_cast<char*>(mapped_data);

    enter_point = *reinterpret_cast<NodeId*>(ptr); ptr += sizeof(NodeId);
    max_layer = *reinterpret_cast<int*>(ptr); ptr += sizeof(int);
    num_nodes = *reinterpret_cast<size_t*>(ptr); ptr += sizeof(size_t);

    disk_nodes = reinterpret_cast<DiskNode*>(ptr);
}

MmapHNSWGraph::~MmapHNSWGraph() {
    if (mapped_data != MAP_FAILED) {
        munmap(mapped_data, file_size);
    }
    if (fd != -1) {
        close(fd);
    }
}

std::vector<NodeId> MmapHNSWGraph::search(const float* query_data, size_t dim, int k, int ef_search) const {
    if (enter_point == -1) return {};

    NodeId curr_ep = enter_point;
    float min_dist = l2_distance_disk(query_data, dim, disk_nodes[curr_ep]);

    for (int l = max_layer; l > 0; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            if (l >= MMAP_MAX_LAYERS) continue;

            int num_edges = disk_nodes[curr_ep].num_edges[l];
            for (int i = 0; i < num_edges; ++i) {
                NodeId neighbor = disk_nodes[curr_ep].edges[l][i];
                float dist = l2_distance_disk(query_data, dim, disk_nodes[neighbor]);
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

    float initial_dist = l2_distance_disk(query_data, dim, disk_nodes[curr_ep]);
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

        int num_edges = disk_nodes[c].num_edges[0];
        for (int i = 0; i < num_edges; ++i) {
            NodeId neighbor = disk_nodes[c].edges[0][i];

            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                __builtin_prefetch(&disk_nodes[neighbor], 0, 1);

                float dist_n = l2_distance_disk(query_data, dim, disk_nodes[neighbor]);

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
        if (!disk_nodes[res_id].deleted) {
            results.push_back(res_id);
        }
        top_results.pop();
    }
    std::reverse(results.begin(), results.end());

    if (results.size() > static_cast<size_t>(k)) results.resize(k);

    return results;
}

} // namespace hnsw

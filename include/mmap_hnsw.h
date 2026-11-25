#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "node.h"

namespace hnsw {

#define MMAP_DIM 128
#define MMAP_MAX_LAYERS 5
#define MMAP_M_MAX 64

// Fixed-width packed record for zero-copy POSIX mmap out-of-core traversal
#pragma pack(push, 1)
struct DiskNode {
    bool deleted;
    float v_min;
    float v_max;
    uint32_t metadata_label;
    uint8_t vec[MMAP_DIM];

    int num_edges[MMAP_MAX_LAYERS];
    NodeId edges[MMAP_MAX_LAYERS][MMAP_M_MAX];
};
#pragma pack(pop)

class MmapHNSWGraph {
private:
    int fd;
    size_t file_size;
    void* mapped_data;

    NodeId enter_point;
    int max_layer;
    size_t num_nodes;

    DiskNode* disk_nodes;

public:
    MmapHNSWGraph(const std::string& filename);
    ~MmapHNSWGraph();

    std::vector<NodeId> search(const float* query_data, size_t dim, int k, int ef_search) const;
};

} // namespace hnsw

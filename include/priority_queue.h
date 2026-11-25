#pragma once
#include <queue>
#include <vector>
#include "node.h"

namespace hnsw {

using DistNodePair = std::pair<float, NodeId>;

// Max-Heap: top element is the farthest node (used for bounded candidate pruning)
using MaxHeap = std::priority_queue<DistNodePair>;

// Min-Heap: top element is the closest node (used for best-first traversal)
using MinHeap = std::priority_queue<DistNodePair, std::vector<DistNodePair>, std::greater<DistNodePair>>;

} // namespace hnsw

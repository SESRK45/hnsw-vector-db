#include "distributed.h"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "[Cluster] Booting up Distributed Vector DB Cluster..." << std::endl;

    hnsw::HNSWServer shard1(18081);
    hnsw::HNSWServer shard2(18082);
    hnsw::HNSWServer shard3(18083);

    shard1.graph.reserve_nodes(500);
    shard2.graph.reserve_nodes(500);
    shard3.graph.reserve_nodes(500);

    shard1.start();
    shard2.start();
    shard3.start();

    std::cout << "[Cluster] 3 Shards active on ports 18081, 18082, 18083." << std::endl;

    std::cout << "[Cluster] Ingesting vectors into shards..." << std::endl;
    std::vector<float> v1(128, 1.0f);
    std::vector<float> v2(128, 2.0f);
    std::vector<float> v3(128, 3.0f);
    shard1.graph.insert(100, v1.data(), v1.size());
    shard2.graph.insert(200, v2.data(), v2.size());
    shard3.graph.insert(300, v3.data(), v3.size());

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::vector<std::string> ips = {"127.0.0.1", "127.0.0.1", "127.0.0.1"};
    std::vector<int> ports = {18081, 18082, 18083};
    hnsw::HNSWRouter router(ips, ports);

    std::cout << "[Cluster] Sending scatter-gather query to router..." << std::endl;
    std::vector<float> query(128, 2.1f);

    auto results = router.search_cluster(query.data(), query.size(), 3, 10);

    std::cout << "[Cluster] Global Top-K Results: ";
    for (auto id : results) std::cout << id << " ";
    std::cout << std::endl;

    shard1.stop();
    shard2.stop();
    shard3.stop();
    return 0;
}

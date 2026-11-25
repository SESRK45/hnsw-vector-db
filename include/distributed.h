#pragma once
#include "hnsw.h"
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <netinet/in.h>

namespace hnsw {

// TCP shard worker serving local HNSW search queries
class HNSWServer {
private:
    int port;
    int server_fd = -1;
    std::atomic<bool> running{false};
    std::thread listener_thread;

public:
    HNSWGraph graph;

    HNSWServer(int p);
    ~HNSWServer();

    void start();
    void stop();

private:
    void listen_loop();
    void handle_client(int client_socket);
};

// Scatter-gather router that broadcasts queries across shards
class HNSWRouter {
private:
    std::vector<std::string> shard_ips;
    std::vector<int> shard_ports;

public:
    HNSWRouter(const std::vector<std::string>& ips, const std::vector<int>& ports);

    std::vector<NodeId> search_cluster(const float* query_data, size_t dim, int k, int ef_search);
};

} // namespace hnsw

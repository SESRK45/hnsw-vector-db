#include "distributed.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>
#include <future>
#include <algorithm>

namespace hnsw {

HNSWServer::HNSWServer(int p) : port(p) {}

HNSWServer::~HNSWServer() { stop(); }

void HNSWServer::start() {
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) throw std::runtime_error("Socket failed");

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, reinterpret_cast<struct sockaddr*>(&address), sizeof(address)) < 0) {
        throw std::runtime_error("Bind failed on port " + std::to_string(port));
    }

    if (listen(server_fd, 16) < 0) throw std::runtime_error("Listen failed");

    running = true;
    listener_thread = std::thread(&HNSWServer::listen_loop, this);
}

void HNSWServer::stop() {
    if (running) {
        running = false;
        if (server_fd >= 0) {
            shutdown(server_fd, SHUT_RDWR);
            close(server_fd);
            server_fd = -1;
        }
        if (listener_thread.joinable()) listener_thread.join();
    }
}

void HNSWServer::listen_loop() {
    while (running) {
        struct sockaddr_in address;
        socklen_t addrlen = sizeof(address);
        int client_socket = accept(server_fd, reinterpret_cast<struct sockaddr*>(&address), &addrlen);

        if (client_socket < 0) {
            if (running) continue;
            break;
        }

        std::thread(&HNSWServer::handle_client, this, client_socket).detach();
    }
}

void HNSWServer::handle_client(int client_socket) {
    // Wire protocol: [int k] [int ef] [int dim] [float payload...]
    int header[3] = {0, 0, 0};
    ssize_t bytes = recv(client_socket, header, sizeof(header), MSG_WAITALL);
    if (bytes != sizeof(header)) {
        close(client_socket);
        return;
    }

    int k = header[0];
    int ef = header[1];
    int dim = header[2];

    if (k <= 0 || ef <= 0 || dim <= 0 || dim > 65536) {
        close(client_socket);
        return;
    }

    std::vector<float> query(static_cast<size_t>(dim));
    ssize_t payload_bytes = recv(client_socket, query.data(), query.size() * sizeof(float), MSG_WAITALL);
    if (payload_bytes != static_cast<ssize_t>(query.size() * sizeof(float))) {
        close(client_socket);
        return;
    }

    std::vector<NodeId> results = graph.search(query.data(), query.size(), k, ef);

    int res_size = static_cast<int>(results.size());
    send(client_socket, &res_size, sizeof(res_size), 0);
    if (res_size > 0) {
        send(client_socket, results.data(), res_size * sizeof(NodeId), 0);
    }

    close(client_socket);
}

HNSWRouter::HNSWRouter(const std::vector<std::string>& ips, const std::vector<int>& ports)
    : shard_ips(ips), shard_ports(ports) {}

std::vector<NodeId> HNSWRouter::search_cluster(const float* query_data, size_t dim, int k, int ef_search) {
    std::vector<std::future<std::vector<NodeId>>> futures;

    for (size_t i = 0; i < shard_ips.size(); ++i) {
        futures.push_back(std::async(std::launch::async, [this, i, query_data, dim, k, ef_search]() {
            int sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0) return std::vector<NodeId>{};

            struct sockaddr_in serv_addr;
            serv_addr.sin_family = AF_INET;
            serv_addr.sin_port = htons(shard_ports[i]);
            inet_pton(AF_INET, shard_ips[i].c_str(), &serv_addr.sin_addr);

            if (connect(sock, reinterpret_cast<struct sockaddr*>(&serv_addr), sizeof(serv_addr)) < 0) {
                close(sock);
                return std::vector<NodeId>{};
            }

            int header[3] = {k, ef_search, static_cast<int>(dim)};
            send(sock, header, sizeof(header), 0);
            send(sock, query_data, dim * sizeof(float), 0);

            int res_size = 0;
            if (recv(sock, &res_size, sizeof(res_size), MSG_WAITALL) != sizeof(res_size) ||
                res_size <= 0 || res_size > 10000) {
                close(sock);
                return std::vector<NodeId>{};
            }

            std::vector<NodeId> local_res(static_cast<size_t>(res_size));
            recv(sock, local_res.data(), res_size * sizeof(NodeId), MSG_WAITALL);

            close(sock);
            return local_res;
        }));
    }

    std::vector<NodeId> global_results;
    for (auto& f : futures) {
        std::vector<NodeId> res = f.get();
        global_results.insert(global_results.end(), res.begin(), res.end());
    }

    if (global_results.size() > static_cast<size_t>(k)) {
        global_results.resize(k);
    }

    return global_results;
}

} // namespace hnsw

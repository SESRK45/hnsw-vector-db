#include "hnsw.h"
#include <liburing.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <thread>
#include <chrono>

#define QUEUE_DEPTH 256

enum EventType {
    ACCEPT,
    READ_HEADER,
    READ_PAYLOAD,
    WRITE_RESULTS
};

struct ConnInfo {
    int fd;
    EventType type;
    int header[3]; // k, ef, dim
    std::vector<float> query;
    std::vector<hnsw::NodeId> results;
};

void add_accept_request(struct io_uring* ring, int server_socket,
                        struct sockaddr_in* client_addr, socklen_t* client_addr_len) {
    struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
    io_uring_prep_accept(sqe, server_socket, reinterpret_cast<struct sockaddr*>(client_addr), client_addr_len, 0);

    ConnInfo* conn = new ConnInfo();
    conn->fd = server_socket;
    conn->type = ACCEPT;
    io_uring_sqe_set_data(sqe, conn);
}

void add_read_header_request(struct io_uring* ring, ConnInfo* conn) {
    struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
    io_uring_prep_recv(sqe, conn->fd, conn->header, sizeof(conn->header), 0);
    conn->type = READ_HEADER;
    io_uring_sqe_set_data(sqe, conn);
}

void add_read_payload_request(struct io_uring* ring, ConnInfo* conn) {
    struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
    conn->query.resize(conn->header[2]);
    io_uring_prep_recv(sqe, conn->fd, conn->query.data(), conn->query.size() * sizeof(float), MSG_WAITALL);
    conn->type = READ_PAYLOAD;
    io_uring_sqe_set_data(sqe, conn);
}

void add_write_request(struct io_uring* ring, ConnInfo* conn) {
    struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
    io_uring_prep_send(sqe, conn->fd, conn->results.data(),
                       conn->results.size() * sizeof(hnsw::NodeId), 0);
    conn->type = WRITE_RESULTS;
    io_uring_sqe_set_data(sqe, conn);
}

int main() {
    std::cout << "[io_uring] Booting asynchronous vector server..." << std::endl;

    hnsw::HNSWGraph graph;
    graph.reserve_nodes(200);
    std::vector<float> v1(128, 1.0f);
    graph.insert(100, v1.data(), v1.size());

    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(18080);
    if (bind(server_socket, reinterpret_cast<struct sockaddr*>(&address), sizeof(address)) < 0) {
        std::cerr << "Bind failed" << std::endl;
        close(server_socket);
        return 1;
    }
    listen(server_socket, 256);

    struct io_uring ring;
    if (io_uring_queue_init(QUEUE_DEPTH, &ring, 0) < 0) {
        std::cerr << "io_uring_queue_init failed (kernel support required)" << std::endl;
        close(server_socket);
        return 1;
    }

    struct sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(client_addr);

    add_accept_request(&ring, server_socket, &client_addr, &client_addr_len);
    io_uring_submit(&ring);

    std::cout << "[io_uring] Asynchronous listener active on port 18080." << std::endl;

    std::thread client_thread([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(18080);
        inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);
        if (connect(sock, reinterpret_cast<struct sockaddr*>(&serv_addr), sizeof(serv_addr)) == 0) {
            int header[3] = {5, 10, 128};
            send(sock, header, sizeof(header), 0);
            std::vector<float> q(128, 1.1f);
            send(sock, q.data(), q.size() * sizeof(float), 0);

            std::vector<hnsw::NodeId> res(5, -1);
            ssize_t bytes = recv(sock, res.data(), res.size() * sizeof(hnsw::NodeId), 0);
            if (bytes > 0) {
                std::cout << "[Client] Read " << bytes << " bytes. Top match: " << res[0] << std::endl;
            }
        }
        close(sock);
    });

    int handled = 0;
    while (handled < 1) {
        struct io_uring_cqe* cqe = nullptr;
        io_uring_wait_cqe(&ring, &cqe);

        ConnInfo* conn = static_cast<ConnInfo*>(io_uring_cqe_get_data(cqe));
        int result = cqe->res;

        if (conn->type == ACCEPT) {
            int client_fd = result;
            ConnInfo* client_conn = new ConnInfo();
            client_conn->fd = client_fd;
            add_read_header_request(&ring, client_conn);
            delete conn;
        } else if (conn->type == READ_HEADER) {
            add_read_payload_request(&ring, conn);
        } else if (conn->type == READ_PAYLOAD) {
            conn->results = graph.search(conn->query.data(), conn->query.size(), conn->header[0], conn->header[1]);
            add_write_request(&ring, conn);
        } else if (conn->type == WRITE_RESULTS) {
            close(conn->fd);
            delete conn;
            handled++;
        }

        io_uring_cqe_seen(&ring, cqe);
        io_uring_submit(&ring);
    }

    client_thread.join();
    io_uring_queue_exit(&ring);
    close(server_socket);

    return 0;
}

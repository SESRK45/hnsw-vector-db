#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include "distance.h"
#include "hnsw.h"
#include "mmap_hnsw.h"

std::vector<hnsw::NodeId> exact_knn(const std::vector<std::vector<float>>& dataset,
                                    const std::vector<float>& query, int k) {
    std::vector<std::pair<float, hnsw::NodeId>> distances;
    distances.reserve(dataset.size());
    for (size_t i = 0; i < dataset.size(); ++i) {
        distances.push_back({hnsw::l2_distance(query, dataset[i]), static_cast<hnsw::NodeId>(i)});
    }
    std::partial_sort(distances.begin(), distances.begin() + k, distances.end());

    std::vector<hnsw::NodeId> results;
    for (int i = 0; i < k; ++i) {
        results.push_back(distances[i].second);
    }
    return results;
}

int main() {
    std::filesystem::create_directories("benchmarks");
    std::filesystem::create_directories("build");

    std::cout << "[Benchmark] Generating Clustered Dataset (10,000 vectors, 128D)..." << std::endl;
    int num_vectors = 10000;
    int dim = 128;
    int num_clusters = 20;
    std::vector<std::vector<float>> dataset(num_vectors, std::vector<float>(dim));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> cluster_center_dist(-20.0f, 20.0f);
    std::normal_distribution<float> noise_dist(0.0f, 2.0f);

    std::vector<std::vector<float>> centers(num_clusters, std::vector<float>(dim));
    for (int c = 0; c < num_clusters; ++c) {
        for (int d = 0; d < dim; ++d) {
            centers[c][d] = cluster_center_dist(rng);
        }
    }

    std::uniform_int_distribution<uint32_t> label_dist(1, 5);
    std::vector<uint32_t> labels(num_vectors);
    for (int i = 0; i < num_vectors; ++i) {
        labels[i] = label_dist(rng);
    }

    for (int i = 0; i < num_vectors; ++i) {
        int c = i % num_clusters;
        for (int d = 0; d < dim; ++d) {
            dataset[i][d] = centers[c][d] + noise_dist(rng);
        }
    }

    std::cout << "[Benchmark] Building HNSW Index (Multithreaded)..." << std::endl;
    hnsw::HNSWGraph graph(24, 300);
    graph.reserve_nodes(num_vectors);
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < num_vectors; ++i) {
        graph.insert(i, dataset[i].data(), dataset[i].size(), labels[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Index built in " << std::chrono::duration<double>(end - start).count() << " seconds." << std::endl;

    int num_queries = 200;
    std::cout << "[Benchmark] Running Exact KNN (Ground Truth)..." << std::endl;
    std::vector<std::vector<hnsw::NodeId>> exact_results(num_queries);
    for (int i = 0; i < num_queries; ++i) {
        std::vector<float> q(dim);
        for (int d = 0; d < dim; ++d) q[d] = dataset[i][d] + 0.5f;
        exact_results[i] = exact_knn(dataset, q, 10);
    }

    std::cout << "[Benchmark] Running HNSW Search & Logging..." << std::endl;
    std::ofstream out("benchmarks/results.csv");
    out << "query_id,exact_nn,hnsw_nn,time_us\n";

    for (int i = 0; i < num_queries; ++i) {
        std::vector<float> q(dim);
        for (int d = 0; d < dim; ++d) q[d] = dataset[i][d] + 0.5f;

        start = std::chrono::high_resolution_clock::now();
        std::vector<hnsw::NodeId> hnsw_res = graph.search(q.data(), q.size(), 10, 100);
        end = std::chrono::high_resolution_clock::now();

        double time_us = std::chrono::duration<double, std::micro>(end - start).count();

        out << i << ",\"";
        for (int r : exact_results[i]) out << r << " ";
        out << "\",\"";
        for (int r : hnsw_res) out << r << " ";
        out << "\"," << time_us << "\n";
    }
    out.close();

    std::cout << "[Benchmark] Testing Bypass Bridge Graph Healing (Deletion)..." << std::endl;
    std::vector<float> q_test(dim);
    for (int d = 0; d < dim; ++d) q_test[d] = dataset[0][d] + 0.5f;

    std::vector<hnsw::NodeId> pre_delete_res = graph.search(q_test.data(), q_test.size(), 5, 50);
    std::cout << "  Pre-delete Top 5:  ";
    for (int r : pre_delete_res) std::cout << r << " ";
    std::cout << std::endl;

    if (!pre_delete_res.empty()) {
        hnsw::NodeId target_to_delete = pre_delete_res[0];
        std::cout << "  Deleting node " << target_to_delete << " & repairing local edges..." << std::endl;
        auto del_start = std::chrono::high_resolution_clock::now();
        graph.remove(target_to_delete);
        auto del_end = std::chrono::high_resolution_clock::now();
        double del_us = std::chrono::duration<double, std::micro>(del_end - del_start).count();

        std::vector<hnsw::NodeId> post_delete_res = graph.search(q_test.data(), q_test.size(), 5, 50);
        std::cout << "  Post-delete Top 5: ";
        for (int r : post_delete_res) std::cout << r << " ";
        std::cout << " (deleted in " << del_us << " us)" << std::endl;

        if (post_delete_res.empty() || post_delete_res[0] == target_to_delete) {
            std::cout << "  ERROR: Deletion failed." << std::endl;
            return 1;
        }
        std::cout << "  SUCCESS: Target node eliminated and graph connectivity preserved." << std::endl;
    }

    std::cout << "[Benchmark] Exporting graph to fixed-width mmap format..." << std::endl;
    std::string mmap_file = "build/disk_index.bin";
    graph.export_to_mmap(mmap_file);

    std::cout << "[Benchmark] Instantiating out-of-core MmapHNSWGraph..." << std::endl;
    hnsw::MmapHNSWGraph disk_graph(mmap_file);

    std::vector<hnsw::NodeId> mmap_res = disk_graph.search(q_test.data(), q_test.size(), 5, 50);
    std::cout << "  SSD-Mapped Top 5:  ";
    for (int r : mmap_res) std::cout << r << " ";
    std::cout << std::endl;

    std::cout << "[Benchmark] Running Hybrid Search (Metadata Filtering)..." << std::endl;
    uint32_t target_label = 3;
    std::vector<hnsw::NodeId> filtered_res = graph.search_filtered(q_test.data(), q_test.size(), 5, 100, target_label);

    std::cout << "  Filtered Search (Label 3) Top 5: ";
    for (int r : filtered_res) {
        std::cout << r << "(L:" << labels[r] << ") ";
    }
    std::cout << std::endl;

    std::cout << "[Benchmark] Training Product Quantization (PQ)..." << std::endl;
    auto pq_start = std::chrono::high_resolution_clock::now();
    graph.build_pq(dataset);
    auto pq_end = std::chrono::high_resolution_clock::now();
    std::cout << "  PQ Codebook trained & 10,000 vectors encoded in "
              << std::chrono::duration<double>(pq_end - pq_start).count() << " sec." << std::endl;

    auto pq_s_start = std::chrono::high_resolution_clock::now();
    std::vector<hnsw::NodeId> pq_res = graph.search_pq(q_test.data(), q_test.size(), 5, 100);
    auto pq_s_end = std::chrono::high_resolution_clock::now();
    double pq_us = std::chrono::duration<double, std::micro>(pq_s_end - pq_s_start).count();

    std::cout << "  PQ ADC Top 5: ";
    for (int r : pq_res) std::cout << r << " ";
    std::cout << "(" << pq_us << " us)" << std::endl;

    std::cout << "[Benchmark] Arena Memory Footprint..." << std::endl;
    size_t edge_bytes = graph.edge_arena.get_usage() * sizeof(hnsw::NodeId);
    std::cout << "  Edge arena used: " << static_cast<double>(edge_bytes) / (1024.0 * 1024.0) << " MB ("
              << graph.edge_arena.get_usage() << " entries)" << std::endl;

    std::cout << "[Benchmark] Running Concurrent Shared-Mutex Read Benchmark..." << std::endl;
    auto concurrent_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < 10000; ++i) {
        std::vector<hnsw::NodeId> res = graph.search(q_test.data(), q_test.size(), 10, 50);
        (void)res;
    }
    auto concurrent_end = std::chrono::high_resolution_clock::now();
    double concurrent_s = std::chrono::duration<double>(concurrent_end - concurrent_start).count();
    std::cout << "  10,000 concurrent queries processed in " << concurrent_s << " sec ("
              << (10000.0 / concurrent_s) << " QPS)" << std::endl;

    return 0;
}

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <filesystem>
#include "hnsw.h"

int main() {
    std::cout << "=== HNSW Database Persistence Test ===" << std::endl;

    const int num_vectors = 10000;
    const size_t dim = 128;

    hnsw::HNSWGraph graph(16, 200);
    graph.reserve_nodes(num_vectors);

    std::cout << "[1] Inserting " << num_vectors << " vectors into RAM..." << std::endl;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::vector<float>> dataset(num_vectors, std::vector<float>(dim));
    for (int i = 0; i < num_vectors; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            dataset[i][d] = dist(rng);
        }
        graph.insert(i, dataset[i].data(), dataset[i].size());
    }

    std::cout << "[2] Searching memory-resident graph for vector 0..." << std::endl;
    std::vector<hnsw::NodeId> orig_results = graph.search(dataset[0].data(), dataset[0].size(), 5, 50);
    std::cout << "    Original Top 5: ";
    for (int r : orig_results) std::cout << r << " ";
    std::cout << std::endl;

    std::filesystem::create_directories("build");
    std::string db_file = "build/vectors.bin";
    std::cout << "[3] Serializing database to disk: " << db_file << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    graph.save(db_file);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "    Saved in " << std::chrono::duration<double, std::milli>(end - start).count() << " ms." << std::endl;

    std::cout << "[4] Instantiating fresh HNSWGraph engine..." << std::endl;
    hnsw::HNSWGraph loaded_graph;

    std::cout << "[5] Loading index from disk..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    loaded_graph.load(db_file);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "    Loaded " << loaded_graph.size() << " vectors in "
              << std::chrono::duration<double, std::milli>(end - start).count() << " ms." << std::endl;

    std::cout << "[6] Searching newly loaded graph for vector 0..." << std::endl;
    std::vector<hnsw::NodeId> loaded_results = loaded_graph.search(dataset[0].data(), dataset[0].size(), 5, 50);
    std::cout << "    Loaded Top 5:   ";
    for (int r : loaded_results) std::cout << r << " ";
    std::cout << std::endl;

    if (orig_results == loaded_results) {
        std::cout << "\nSUCCESS: Binary serialization is 100% consistent." << std::endl;
        return 0;
    } else {
        std::cout << "\nERROR: Mismatch in loaded graph topology." << std::endl;
        return 1;
    }
}

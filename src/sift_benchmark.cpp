#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <sched.h>
#include "hnsw.h"

using namespace std;
using namespace std::chrono;

vector<vector<float>> read_fvecs(const string& filename, size_t limit = 0) {
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot open " + filename);
    vector<vector<float>> data;
    while (!file.eof()) {
        int dim;
        if (!file.read(reinterpret_cast<char*>(&dim), sizeof(int))) break;
        vector<float> vec(dim);
        if (!file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float))) break;
        data.push_back(move(vec));
        if (limit > 0 && data.size() >= limit) break;
    }
    return data;
}

vector<vector<int>> read_ivecs(const string& filename, size_t limit = 0) {
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot open " + filename);
    vector<vector<int>> data;
    while (!file.eof()) {
        int dim;
        if (!file.read(reinterpret_cast<char*>(&dim), sizeof(int))) break;
        vector<int> vec(dim);
        if (!file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int))) break;
        data.push_back(move(vec));
        if (limit > 0 && data.size() >= limit) break;
    }
    return data;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <base_fvecs> <query_fvecs> <groundtruth_ivecs>" << endl;
        return 1;
    }

    string base_file = argv[1];
    string query_file = argv[2];
    string gt_file = argv[3];

    cout << "Loading Dataset: " << base_file << endl;

    auto dataset = read_fvecs(base_file, 0);
    auto queries = read_fvecs(query_file, 1000);
    auto ground_truth = read_ivecs(gt_file, 1000);

    int num_vectors = static_cast<int>(dataset.size());
    int num_queries = static_cast<int>(queries.size());
    int k = 10;

    cout << "Loaded " << num_vectors << " vectors and " << num_queries << " queries." << endl;

    int M = 16;
    int ef_construction = 200;
    cout << "Building HNSW Graph (M=" << M << ", ef_construction=" << ef_construction << ")..." << endl;
    hnsw::HNSWGraph graph(M, ef_construction);
    graph.reserve_nodes(num_vectors);

    auto start_build = high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < num_vectors; ++i) {
        graph.insert(i, dataset[i].data(), dataset[i].size(), 0);
        if (i % 100000 == 0 && i > 0) {
            #pragma omp critical
            cout << "  Inserted " << i << " vectors..." << endl;
        }
    }
    auto end_build = high_resolution_clock::now();
    cout << "Build time: " << duration<double>(end_build - start_build).count() << "s" << endl;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        cerr << "Warning: could not pin query thread to CPU Core 0" << endl;
    }

    vector<int> ef_search_values = {10, 20, 40, 80, 160, 320};

    cout << "\n===============================================================" << endl;
    cout << left << setw(12) << "ef_search"
         << setw(14) << "Recall@10"
         << setw(14) << "QPS"
         << setw(14) << "Latency(ms)"
         << setw(12) << "StdDev(%)" << endl;
    cout << "===============================================================" << endl;

    for (int ef : ef_search_values) {
        vector<double> run_recalls;
        vector<double> run_qps;
        const int NUM_RUNS = 3;

        for (int run = 0; run < NUM_RUNS; ++run) {
            auto start_search = high_resolution_clock::now();
            double total_recall = 0.0;

            for (int q = 0; q < num_queries; ++q) {
                vector<hnsw::NodeId> res = graph.search(queries[q].data(), queries[q].size(), k, ef);

                int matches = 0;
                unordered_set<int> gt(ground_truth[q].begin(), ground_truth[q].begin() + k);
                for (auto id : res) {
                    if (gt.count(id)) matches++;
                }
                total_recall += static_cast<double>(matches) / k;
            }
            auto end_search = high_resolution_clock::now();

            double avg_recall = total_recall / num_queries;
            double time_sec = duration<double>(end_search - start_search).count();
            double qps = num_queries / time_sec;

            run_recalls.push_back(avg_recall);
            run_qps.push_back(qps);
        }

        double mean_recall = 0.0, mean_qps = 0.0;
        for (int r = 0; r < NUM_RUNS; ++r) {
            mean_recall += run_recalls[r];
            mean_qps += run_qps[r];
        }
        mean_recall /= NUM_RUNS;
        mean_qps /= NUM_RUNS;

        double var_qps = 0.0;
        for (int r = 0; r < NUM_RUNS; ++r) {
            double diff = run_qps[r] - mean_qps;
            var_qps += diff * diff;
        }
        double std_pct = (mean_qps > 0.0) ? (std::sqrt(var_qps / NUM_RUNS) / mean_qps) * 100.0 : 0.0;
        double latency_ms = (mean_qps > 0.0) ? (1000.0 / mean_qps) : 0.0;

        cout << left << setw(12) << ef
             << setw(14) << fixed << setprecision(4) << mean_recall
             << setw(14) << setprecision(1) << mean_qps
             << setw(14) << setprecision(3) << latency_ms
             << setw(12) << setprecision(2) << std_pct << endl;
    }

    return 0;
}

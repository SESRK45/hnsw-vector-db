#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>
#include "hnsw.h"
#include "distance.h"

using namespace std;
using namespace std::chrono;

int main() {
    int num_vectors = 20000;
    int num_queries = 100;
    int dim = 128;
    int k = 10;

    cout << "Generating dataset of " << num_vectors << " vectors..." << endl;
    vector<vector<float>> dataset(num_vectors, vector<float>(dim));
    vector<vector<float>> queries(num_queries, vector<float>(dim));

    mt19937 rng(42);
    int num_clusters = 100;
    uniform_real_distribution<float> center_dist(-10.0f, 10.0f);
    normal_distribution<float> noise_dist(0.0f, 1.5f);

    vector<vector<float>> centers(num_clusters, vector<float>(dim));
    for (int c = 0; c < num_clusters; ++c)
        for (int d = 0; d < dim; ++d)
            centers[c][d] = center_dist(rng);

    for (int i = 0; i < num_vectors; ++i) {
        int c = i % num_clusters;
        for (int d = 0; d < dim; ++d)
            dataset[i][d] = centers[c][d] + noise_dist(rng);
    }

    for (int i = 0; i < num_queries; ++i) {
        int c = i % num_clusters;
        for (int d = 0; d < dim; ++d)
            queries[i][d] = centers[c][d] + noise_dist(rng);
    }

    cout << "Building HNSW Graph (M=16, ef_construction=200)..." << endl;
    hnsw::HNSWGraph graph(16, 200);
    graph.reserve_nodes(num_vectors);
    auto start_build = high_resolution_clock::now();
    for (int i = 0; i < num_vectors; ++i) {
        graph.insert(i, dataset[i].data(), dataset[i].size());
    }
    auto end_build = high_resolution_clock::now();
    cout << "Build time: " << duration<double>(end_build - start_build).count() << "s" << endl;

    cout << "Computing Ground Truth (Exact KNN)..." << endl;
    vector<vector<hnsw::NodeId>> ground_truth(num_queries, vector<hnsw::NodeId>(k));
    for (int q = 0; q < num_queries; ++q) {
        vector<pair<float, hnsw::NodeId>> dists(num_vectors);
        for (int i = 0; i < num_vectors; ++i) {
            dists[i] = {hnsw::l2_distance(queries[q], dataset[i]), i};
        }
        sort(dists.begin(), dists.end());
        for (int i = 0; i < k; ++i) ground_truth[q][i] = dists[i].second;
    }

    vector<int> ef_search_values = {10, 20, 40, 80, 160, 320};

    cout << "\n=============================================" << endl;
    cout << setw(10) << "ef_search" << setw(15) << "Recall@" << k << setw(15) << "QPS" << endl;
    cout << "=============================================" << endl;

    for (int ef : ef_search_values) {
        auto start_search = high_resolution_clock::now();
        double total_recall = 0.0;

        for (int q = 0; q < num_queries; ++q) {
            vector<hnsw::NodeId> res = graph.search(queries[q].data(), queries[q].size(), k, ef);

            int matches = 0;
            for (auto id : res) {
                if (find(ground_truth[q].begin(), ground_truth[q].end(), id) != ground_truth[q].end()) {
                    matches++;
                }
            }
            total_recall += static_cast<double>(matches) / k;
        }
        auto end_search = high_resolution_clock::now();

        double avg_recall = total_recall / num_queries;
        double time_sec = duration<double>(end_search - start_search).count();
        double qps = num_queries / time_sec;

        cout << setw(10) << ef << setw(14) << fixed << setprecision(4) << avg_recall
             << setw(15) << setprecision(1) << qps << endl;
    }

    return 0;
}

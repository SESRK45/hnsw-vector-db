#include <iostream>
#include <vector>
#include <immintrin.h>
#include <cstdint>
#include <chrono>
#include <sched.h>

using namespace std;

int main() {
    int num_vectors = 1000000;
    int M = 64;
    int K = 256;

    // Flat LUT (64 subspaces x 256 centroids x 4 bytes = 65.5 KB)
    vector<float> lut(M * K, 1.0f);

    vector<uint8_t> database(num_vectors * M);
    for (size_t i = 0; i < database.size(); ++i) {
        database[i] = static_cast<uint8_t>(i % 256);
    }

    cout << "Allocated 1M vectors (64 bytes each). Total memory: "
         << database.size() / (1024 * 1024) << " MB" << endl;

    vector<float> results(num_vectors, 0.0f);

    // Warm up caches
    for (int i = 0; i < 100; ++i) results[i] = lut[database[i]];

    cout << "Pinning thread to CPU Core 0 for hardware isolation..." << endl;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        cerr << "Warning: Could not pin thread to CPU 0" << endl;
    }

    int num_queries = 100;

    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t start_cycles = __builtin_ia32_rdtsc();

    float* lut_ptr = lut.data();
    uint8_t* db_ptr = database.data();

    for (int q = 0; q < num_queries; ++q) {
        for (int i = 0; i < num_vectors; ++i) {
            __m256 total_dist = _mm256_setzero_ps();
            uint8_t* vec = db_ptr + i * M;

            for (int m = 0; m < M; m += 8) {
                __m128i ids_8 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(vec + m));
                __m256i ids_32 = _mm256_cvtepu8_epi32(ids_8);

                __m256i chunk_offsets = _mm256_setr_epi32(
                    (m + 0) * 256, (m + 1) * 256, (m + 2) * 256, (m + 3) * 256,
                    (m + 4) * 256, (m + 5) * 256, (m + 6) * 256, (m + 7) * 256
                );

                __m256i final_offsets = _mm256_add_epi32(ids_32, chunk_offsets);
                __m256 dists = _mm256_i32gather_ps(lut_ptr, final_offsets, 4);
                total_dist = _mm256_add_ps(total_dist, dists);
            }

            float temp[8];
            _mm256_storeu_ps(temp, total_dist);
            results[i] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
        }
    }

    uint64_t end_cycles = __builtin_ia32_rdtsc();
    auto end_time = std::chrono::high_resolution_clock::now();

    double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double time_per_query = total_time_ms / num_queries;

    uint64_t total_cycles = end_cycles - start_cycles;
    double cycles_per_query = static_cast<double>(total_cycles) / num_queries;
    double cycles_per_vector = cycles_per_query / num_vectors;

    cout << "\n=== HARDWARE ISOLATED RESULTS (1 CORE) ===" << endl;
    cout << "Wall-Clock Time per Query (1M vectors): " << time_per_query << " ms" << endl;
    cout << "Hardware Cycles per Vector: " << cycles_per_vector << " cycles" << endl;

    if (results[0] == -1.0f) cout << "ignore" << endl;

    return 0;
}

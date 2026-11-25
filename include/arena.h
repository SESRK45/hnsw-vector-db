#pragma once
#include <vector>
#include <atomic>
#include <stdexcept>
#include <string>
#include <cstdint>

namespace hnsw {

// Contiguous pre-allocated arena bump allocator to avoid small-block heap fragmentation.
template <typename T>
class ArenaAllocator {
private:
    std::vector<T> buffer;
    std::atomic<size_t> offset{0};

public:
    ArenaAllocator(size_t max_elements) {
        buffer.resize(max_elements);
    }

    T* allocate(size_t n) {
        size_t start = offset.fetch_add(n, std::memory_order_relaxed);
        if (start + n > buffer.size()) {
            throw std::runtime_error("Arena capacity exceeded. Increase max_edges_capacity.");
        }
        return &buffer[start];
    }

    void reset() {
        offset.store(0, std::memory_order_relaxed);
    }

    size_t get_usage() const { return offset.load(); }
    size_t get_capacity() const { return buffer.size(); }
};

// Lightweight non-owning span into the pre-allocated arena buffer.
template <typename T>
struct ArenaSpan {
    T* data_ptr = nullptr;
    size_t count = 0;
    size_t capacity = 0;

    void push_back(const T& val) {
        if (count < capacity) {
            data_ptr[count++] = val;
        } else {
            throw std::runtime_error("ArenaSpan capacity exceeded (count=" + std::to_string(count) +
                                     ", cap=" + std::to_string(capacity) + ")");
        }
    }

    T& operator[](size_t i) { return data_ptr[i]; }
    const T& operator[](size_t i) const { return data_ptr[i]; }

    T* begin() { return data_ptr; }
    T* end() { return data_ptr + count; }
    const T* begin() const { return data_ptr; }
    const T* end() const { return data_ptr + count; }

    T* data() { return data_ptr; }
    const T* data() const { return data_ptr; }

    size_t size() const { return count; }
    bool empty() const { return count == 0; }

    void erase(T* first, T* last) {
        if (first >= last) return;
        size_t dist = static_cast<size_t>(last - first);
        T* end_ptr = data_ptr + count;
        for (T* it = first; it + dist < end_ptr; ++it) {
            *it = *(it + dist);
        }
        count -= dist;
    }

    template <typename Container>
    void assign(const Container& c) {
        count = 0;
        for (const auto& val : c) {
            push_back(val);
        }
    }
};

} // namespace hnsw

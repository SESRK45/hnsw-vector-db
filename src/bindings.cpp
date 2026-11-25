#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "hnsw.h"

namespace py = pybind11;

PYBIND11_MODULE(custom_hnsw, m) {
    m.doc() = "C++ HNSW Vector Database Engine with AVX2 SIMD, SQ8/PQ, and Graph Healing";

    py::class_<hnsw::HNSWGraph>(m, "HNSWGraph")
        .def(py::init<int, int, size_t>(),
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("max_edges_capacity") = 50000000,
             "Initialize the graph index with M connections per node and ef_construction")

        .def("reserve_nodes", &hnsw::HNSWGraph::reserve_nodes, py::arg("max_nodes"),
             "Pre-allocate node and lock slots prior to insertion")

        .def("insert", [](hnsw::HNSWGraph& self, hnsw::NodeId id,
                          py::array_t<float, py::array::c_style | py::array::forcecast> vec,
                          uint32_t label) {
            py::buffer_info buf = vec.request();
            if (buf.ndim != 1) throw std::runtime_error("Vector must be 1-dimensional");
            self.insert(id, static_cast<const float*>(buf.ptr), buf.shape[0], label);
        }, py::arg("id"), py::arg("vec"), py::arg("label") = 0,
           "Insert a vector from a NumPy buffer without intermediate copies")

        .def("remove", &hnsw::HNSWGraph::remove, py::arg("id"),
             "Delete a vector and repair neighbor connectivity via Bypass Bridge healing")

        .def("build_index", [](hnsw::HNSWGraph& self,
                               py::array_t<float, py::array::c_style | py::array::forcecast> dataset) {
            py::buffer_info buf = dataset.request();
            if (buf.ndim != 2) throw std::runtime_error("Dataset must be 2-dimensional");
            size_t num_vectors = static_cast<size_t>(buf.shape[0]);
            size_t dim = static_cast<size_t>(buf.shape[1]);
            const float* ptr = static_cast<const float*>(buf.ptr);
            self.reserve_nodes(num_vectors);

            py::gil_scoped_release release;
            #pragma omp parallel for
            for (size_t i = 0; i < num_vectors; ++i) {
                self.insert(static_cast<hnsw::NodeId>(i), ptr + i * dim, dim, 0);
            }
        }, py::arg("dataset"), "Build index in parallel from a 2D NumPy array using OpenMP")

        .def("search", [](hnsw::HNSWGraph& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> query,
                          int k, int ef_search) {
            py::buffer_info buf = query.request();
            if (buf.ndim != 1) throw std::runtime_error("Query must be 1-dimensional");
            const float* ptr = static_cast<const float*>(buf.ptr);
            size_t dim = static_cast<size_t>(buf.shape[0]);
            py::gil_scoped_release release;
            return self.search(ptr, dim, k, ef_search);
        }, py::arg("query"), py::arg("k") = 10, py::arg("ef_search") = 50,
           "Search top-k nearest neighbors")

        .def("search_batch", [](hnsw::HNSWGraph& self,
                                py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                                int k, int ef_search) {
            py::buffer_info buf = queries.request();
            if (buf.ndim != 2) throw std::runtime_error("Queries must be 2-dimensional");
            size_t num_queries = static_cast<size_t>(buf.shape[0]);
            size_t dim = static_cast<size_t>(buf.shape[1]);
            const float* ptr = static_cast<const float*>(buf.ptr);

            py::array_t<hnsw::NodeId> results({num_queries, static_cast<size_t>(k)});
            py::buffer_info res_buf = results.request();
            hnsw::NodeId* res_ptr = static_cast<hnsw::NodeId*>(res_buf.ptr);

            py::gil_scoped_release release;
            #pragma omp parallel for
            for (size_t i = 0; i < num_queries; ++i) {
                std::vector<hnsw::NodeId> res = self.search(ptr + i * dim, dim, k, ef_search);
                for (size_t j = 0; j < static_cast<size_t>(k); ++j) {
                    res_ptr[i * k + j] = (j < res.size()) ? res[j] : -1;
                }
            }
            return results;
        }, py::arg("queries"), py::arg("k") = 10, py::arg("ef_search") = 50,
           "Batch search across a 2D NumPy query matrix using OpenMP")

        .def("build_pq", [](hnsw::HNSWGraph& self,
                            py::array_t<float, py::array::c_style | py::array::forcecast> dataset) {
            py::buffer_info buf = dataset.request();
            if (buf.ndim != 2) throw std::runtime_error("Dataset must be 2-dimensional");
            std::vector<std::vector<float>> data(buf.shape[0], std::vector<float>(buf.shape[1]));
            const float* ptr = static_cast<const float*>(buf.ptr);
            for (ssize_t i = 0; i < buf.shape[0]; ++i) {
                for (ssize_t j = 0; j < buf.shape[1]; ++j) {
                    data[i][j] = ptr[i * buf.shape[1] + j];
                }
            }
            self.build_pq(data);
        }, py::arg("dataset"), "Train PQ codebooks and encode indexed vectors")

        .def("search_pq", [](hnsw::HNSWGraph& self,
                             py::array_t<float, py::array::c_style | py::array::forcecast> query,
                             int k, int ef_search) {
            py::buffer_info buf = query.request();
            if (buf.ndim != 1) throw std::runtime_error("Query must be 1-dimensional");
            const float* ptr = static_cast<const float*>(buf.ptr);
            size_t dim = static_cast<size_t>(buf.shape[0]);
            py::gil_scoped_release release;
            return self.search_pq(ptr, dim, k, ef_search);
        }, py::arg("query"), py::arg("k") = 10, py::arg("ef_search") = 50,
           "Search using PQ Asymmetric Distance Computation")

        .def("search_pq_batch", [](hnsw::HNSWGraph& self,
                                   py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                                   int k, int ef_search) {
            py::buffer_info buf = queries.request();
            if (buf.ndim != 2) throw std::runtime_error("Queries must be 2-dimensional");
            size_t num_queries = static_cast<size_t>(buf.shape[0]);
            size_t dim = static_cast<size_t>(buf.shape[1]);
            const float* ptr = static_cast<const float*>(buf.ptr);

            py::array_t<hnsw::NodeId> results({num_queries, static_cast<size_t>(k)});
            py::buffer_info res_buf = results.request();
            hnsw::NodeId* res_ptr = static_cast<hnsw::NodeId*>(res_buf.ptr);

            py::gil_scoped_release release;
            #pragma omp parallel for
            for (size_t i = 0; i < num_queries; ++i) {
                std::vector<hnsw::NodeId> res = self.search_pq(ptr + i * dim, dim, k, ef_search);
                for (size_t j = 0; j < static_cast<size_t>(k); ++j) {
                    res_ptr[i * k + j] = (j < res.size()) ? res[j] : -1;
                }
            }
            return results;
        }, py::arg("queries"), py::arg("k") = 10, py::arg("ef_search") = 50,
           "Batch PQ search across a 2D NumPy query matrix")

        .def("save", &hnsw::HNSWGraph::save, py::arg("filename"))
        .def("load", &hnsw::HNSWGraph::load, py::arg("filename"))
        .def("size", &hnsw::HNSWGraph::size);
}

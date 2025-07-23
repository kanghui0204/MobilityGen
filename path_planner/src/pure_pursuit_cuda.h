#pragma once

#include <cuda_runtime.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


const double max_dist_threshold = 1e9;

struct MinResult {
    double min_dist = max_dist_threshold;
    double min_dist_along_seg = -1;
    double nearest_x = 0.0;
    double nearest_y = 0.0;
    size_t min_index = -1;
};

class PurePursuitCuda {
public:
    PurePursuitCuda(py::array_t<double> path_points, py::array_t<double> distances);
    ~PurePursuitCuda();

    PurePursuitCuda(const PurePursuitCuda&) = delete;
    PurePursuitCuda& operator=(const PurePursuitCuda&) = delete;

    py::tuple find_nearest(py::array_t<double> query_point);
    cudaStream_t getStream() const { return stream_; }

private:
    double* d_path_points_ = nullptr;
    double* d_distances_ = nullptr;
    MinResult* d_reduce_buffer_ = nullptr;

    double* h_out_nearest_point_ = nullptr; // pinned host mem
    double* h_out_dist_along_seg_ = nullptr; // pinned host mem
    double* h_out_dist_to_seg_ = nullptr; // pinned host mem
    size_t* h_out_seg_idx_ = nullptr; // pinned host mem

    int length_ = 0;
    int smCount_ = 0;
    int maxThreadsPerSM_ = 0;
    int maxThreadsPerBlock_ = 0;
    int maxBlocks_ = 0; 
    
    int gridSize_ = 0;   
    int blockSize_ = 0;   
    int device_id_ = 0; //Now we use GPU 0 , and we can accept gpu id in future

    cudaStream_t stream_ = nullptr;

    void allocateAndCopy(py::array_t<double> path_points, py::array_t<double> distances);

   
    void freeDeviceMemory();

    void checkArrays(py::array& path_points, py::array& distances);
};


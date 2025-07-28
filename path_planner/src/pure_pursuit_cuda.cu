#include "pure_pursuit_cuda.h"
#include "cuda_utils.h"
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <memory>   

namespace py = pybind11;

static unsigned int maxPowerOfTwo(unsigned int n) {
    if (n == 0)
        return 1;
    unsigned int p = 1;
    while (p << 1 <= n) {
        p <<= 1;
    }
    return p;
}


__forceinline__ __device__ void nearest_point_on_segment(
    const double2& a,
    const double2& b,
    double pos_x, double pos_y,
    double2& result,
    double& dist)
{

    double a2b_norm_x,a2b_norm_y,a2b_mag;
    {
        double a2b_x = b.x - a.x;
        double a2b_y = b.y - a.y;
        a2b_mag = sqrt(a2b_x * a2b_x + a2b_y * a2b_y);
        double denom = a2b_mag + 1e-6;
        a2b_norm_x = a2b_x / denom;
        a2b_norm_y = a2b_y / denom;
    }

    {
        double a2c_x = pos_x - a.x;
        double a2c_y = pos_y - a.y;
        dist = a2c_x * a2b_norm_x + a2c_y * a2b_norm_y;
    }

    if (dist < 0.0) {
        result.x = a.x; result.y = a.y;
    } else if (dist > a2b_mag) {
        result.x = b.x; result.y = b.y;
    } else {
        result.x = a.x + a2b_norm_x * (dist);
        result.y = a.y + a2b_norm_y * (dist);
    }
}

__forceinline__ __device__ bool less_than(double dist1, int idx1, double dist2, int idx2) {
    if (dist1 < dist2) return true;
    if (dist1 > dist2) return false;
    return idx1 < idx2;
}

__global__ void find_nearest_kernel(
    double pos_x, double pos_y,
    const double* d_points, const size_t length,
    const double* d_distances,MinResult* buffer)
{

    const int32_t tid = threadIdx.x;
    const int32_t bid = blockIdx.x;

    const double2* pts = reinterpret_cast<const double2*>(d_points);

    double min_dist = max_dist_threshold;

    MinResult local_val;
    extern __shared__ double smem[];
    double* sdist = reinterpret_cast<double*>(smem);
    int* sidx = reinterpret_cast<int*>(smem + blockDim.x);

    for (size_t ind = bid*blockDim.x+tid;ind < length-1;ind += (size_t)blockDim.x*gridDim.x){
        double2 a = pts[ind];
        double2 b = pts[ind+1];
        double2 nearest_pt;
        double proj_dist;
        double dist_to_seg;
        {
           nearest_point_on_segment(a, b, pos_x, pos_y, nearest_pt, proj_dist);
           double dx = pos_x - nearest_pt.x;
           double dy = pos_y - nearest_pt.y;
           dist_to_seg = sqrt(dx*dx + dy*dy);
        }

        if (dist_to_seg < min_dist){
             min_dist = dist_to_seg;
             local_val.min_dist = dist_to_seg;
             local_val.min_index = ind;
             local_val.nearest_x = nearest_pt.x;
             local_val.nearest_y = nearest_pt.y;
             local_val.min_dist_along_seg = d_distances[ind] + proj_dist;
        }
    }

    sdist[tid] = min_dist;
    sidx[tid] = tid;
    __syncthreads();

    int blockSize = blockDim.x;
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockSize) {
            double dist_left = sdist[tid];
            int idx_left = sidx[tid];
            double dist_right = sdist[tid + stride];
            int idx_right = sidx[tid + stride];

            if (less_than(dist_right, idx_right, dist_left, idx_left)) {
                sdist[tid] = dist_right;
                sidx[tid] = idx_right;
            }
        }
        __syncthreads();
    }

    if (tid == sidx[0]) {
         buffer[bid] = local_val;
    }
}

 __global__ void find_nearest_final_kernel(const MinResult* buffer, const int length,double* h_out_nearest_point_,double* h_out_dist_along_seg_ , double* h_out_dist_to_seg_, size_t* h_out_seg_idx_){
    const int32_t tid = threadIdx.x;
    extern __shared__ double smem[];
    double* sdist = reinterpret_cast<double*>(smem);
    int* sidx = reinterpret_cast<int*>(smem + blockDim.x);

    MinResult local_val;
    for (int32_t ind = tid;ind < length;ind+=blockDim.x){
        MinResult tmp_val = buffer[ind];
        if (local_val.min_index == -1){
              local_val = tmp_val;
        }
        else{
              if (tmp_val.min_dist<local_val.min_dist){
                    local_val = tmp_val;
              }
        }
    }
    sdist[tid] = local_val.min_dist;
    sidx[tid] =  local_val.min_index;
    __syncthreads();
  
    int blockSize = blockDim.x;
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockSize) {
            double dist_left = sdist[tid];
            int idx_left = sidx[tid];
            double dist_right = sdist[tid + stride];
            int idx_right = sidx[tid + stride];

            if (less_than(dist_right, idx_right, dist_left, idx_left)) {
                sdist[tid] = dist_right;
                sidx[tid] = idx_right;
            }
        }
        __syncthreads();
    }
    if (local_val.min_index == sidx[tid]){
         h_out_nearest_point_[0] = local_val.nearest_x;
         h_out_nearest_point_[1] = local_val.nearest_y;
         h_out_dist_along_seg_[0] = local_val.min_dist_along_seg;
         h_out_dist_to_seg_[0] = local_val.min_dist;
         h_out_seg_idx_[0] = local_val.min_index;
    }


}

PurePursuitCuda::PurePursuitCuda(py::array_t<double> path_points, py::array_t<double> distances) {
    int old_device;
    CHECK_CUDA(cudaGetDevice(&old_device));
    CHECK_CUDA(cudaSetDevice(device_id_));
    checkArrays(path_points, distances);

    length_ = int(path_points.shape(0));

    CHECK_CUDA(cudaStreamCreate(&stream_));

    maxThreadsPerBlock_ = 1024;
    blockSize_ = 1024;
    if (length_ < blockSize_){
        blockSize_ = maxPowerOfTwo(length_);
    }
    
    gridSize_ = (length_-1)/blockSize_+1;

    allocateAndCopy(path_points, distances);

    CHECK_CUDA(cudaSetDevice(old_device));

}

void PurePursuitCuda::checkArrays(py::array& path_points, py::array& distances) {
    if (path_points.dtype().kind() != 'f' || path_points.itemsize() != 8) {
        throw std::runtime_error("path_points must be a numpy array of dtype float64");
    }
    if (distances.dtype().kind() != 'f' || distances.itemsize() != 8) {
        throw std::runtime_error("distances must be a numpy array of dtype float64");
    }
}

void PurePursuitCuda::allocateAndCopy(py::array_t<double> path_points, py::array_t<double> distances) {
    size_t path_bytes = path_points.size() * sizeof(double);
    CHECK_CUDA(cudaMallocAsync(&d_path_points_, path_bytes, stream_));

    size_t dist_bytes = distances.size() * sizeof(double);
    CHECK_CUDA(cudaMallocAsync(&d_distances_, dist_bytes, stream_));

    size_t buffer_bytes = gridSize_ * sizeof(MinResult);
    CHECK_CUDA(cudaMallocAsync(&d_reduce_buffer_, buffer_bytes, stream_));

    CHECK_CUDA(cudaMemcpyAsync(d_path_points_, path_points.data(), path_bytes, cudaMemcpyHostToDevice, stream_));
    CHECK_CUDA(cudaMemcpyAsync(d_distances_, distances.data(), dist_bytes, cudaMemcpyHostToDevice, stream_));

    CHECK_CUDA(cudaStreamSynchronize(stream_));

    CHECK_CUDA(cudaHostAlloc((void**)&h_out_nearest_point_, 2 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA(cudaHostAlloc((void**)&h_out_dist_along_seg_, sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA(cudaHostAlloc((void**)&h_out_dist_to_seg_, sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA(cudaHostAlloc((void**)&h_out_seg_idx_, sizeof(size_t), cudaHostAllocPortable));

}


py::tuple PurePursuitCuda::find_nearest(py::array_t<double> query_point) {
    int old_device;
    CHECK_CUDA(cudaGetDevice(&old_device));
    CHECK_CUDA(cudaSetDevice(device_id_));
    if (query_point.ndim() != 1 || query_point.shape(0) != 2)
        throw std::runtime_error("query_point must be 1D array of size 2");
    auto q = query_point.unchecked<1>();
    double posx = q(0);
    double posy = q(1);
    int smem_bytes = (sizeof(double) + sizeof(int)) * blockSize_;
    find_nearest_kernel<<<gridSize_, blockSize_, smem_bytes, stream_>>>(
        posx,posy,d_path_points_, length_, d_distances_, d_reduce_buffer_);

    int second_block_size = 1024;
    if (gridSize_<second_block_size) second_block_size = maxPowerOfTwo(gridSize_);

    smem_bytes = (sizeof(double) + sizeof(int)) * second_block_size;
    find_nearest_final_kernel<<<1,second_block_size,smem_bytes,stream_>>>(d_reduce_buffer_,gridSize_,h_out_nearest_point_,h_out_dist_along_seg_,h_out_dist_to_seg_,h_out_seg_idx_);

    cudaStreamSynchronize(stream_);

    double nearest_pt[2];
    nearest_pt[0] = h_out_nearest_point_[0];
    nearest_pt[1] = h_out_nearest_point_[1];
    double dist_along_seg = h_out_dist_along_seg_[0];
    double dist_to_seg = h_out_dist_to_seg_[0];
    int seg_idx = h_out_seg_idx_[0];
  
    CHECK_CUDA(cudaSetDevice(old_device));

    return py::make_tuple(
        py::array_t<double>({2}, nearest_pt), dist_along_seg, dist_to_seg, py::make_tuple(seg_idx, seg_idx+1));
}

PurePursuitCuda::~PurePursuitCuda() {
    int old_device;
    CHECK_CUDA(cudaGetDevice(&old_device));
    CHECK_CUDA(cudaSetDevice(device_id_));

    freeDeviceMemory();
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    CHECK_CUDA(cudaSetDevice(old_device));
}

void PurePursuitCuda::freeDeviceMemory() {
    if (d_path_points_ != nullptr) {
        cudaFreeAsync(d_path_points_,stream_);
        d_path_points_ = nullptr;
    }
    if (d_distances_ != nullptr) {
        cudaFreeAsync(d_distances_,stream_);
        d_distances_ = nullptr;
    }
    if (d_reduce_buffer_ != nullptr) {
        cudaFreeAsync(d_reduce_buffer_,stream_);
        d_reduce_buffer_ = nullptr;
    }
    CHECK_CUDA(cudaStreamSynchronize(stream_));


    if (h_out_nearest_point_ != nullptr) {
        cudaFreeHost(h_out_nearest_point_);
        h_out_nearest_point_ = nullptr;
    }
    if (h_out_dist_along_seg_ != nullptr) {
        cudaFreeHost(h_out_dist_along_seg_);
        h_out_dist_along_seg_ = nullptr;
    }
    if (h_out_dist_to_seg_ != nullptr) {
        cudaFreeHost(h_out_dist_to_seg_);
        h_out_dist_to_seg_ = nullptr;
    }
    if (h_out_seg_idx_ != nullptr) {
        cudaFreeHost(h_out_seg_idx_);
        h_out_seg_idx_ = nullptr;
    }

}

PYBIND11_MODULE(_path_helper_gpu_op, m) {
    m.doc() = "Pure Pursuit CUDA GPU Operations";

    py::class_<PurePursuitCuda, std::shared_ptr<PurePursuitCuda>>(m, "PurePursuitCuda")
        .def(py::init<py::array_t<double>, py::array_t<double>>())  
        .def("find_nearest", &PurePursuitCuda::find_nearest);

    m.def("pure_pursuit_cuda", [](py::array_t<double> path_points, py::array_t<double> distances) {
        return std::make_shared<PurePursuitCuda>(path_points, distances);
    }, py::arg("path_points"), py::arg("distances"),
    "Pure Pursuit CUDA using float64 numpy arrays");
}



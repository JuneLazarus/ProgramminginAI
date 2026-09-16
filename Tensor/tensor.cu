#include <stdio.h>
#include <string.h>
#include <cmath>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>

const int kCudaThreadsNum = 512;

inline int CudaGetBlocks(const int N) {
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}

#define CUDA_CHECK(expr) do {                                          \
    cudaError_t _err = (expr);                                         \
    if (_err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(_err));         \
        throw std::runtime_error(cudaGetErrorString(_err));            \
    }                                                                  \
} while (0)

#define CUDA_KERNEL_LOOP(i, n)                                  \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;         \
         i < (n);                                               \
         i += blockDim.x * gridDim.x)

__global__ void gpu_relu_forward(const float* x, float* out, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

__global__ void gpu_relu_backward(const float* grad, const float* x, float* dx, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        dx[i] = x[i] > 0.0f ? grad[i] : 0.0f;
    }
}

__global__ void gpu_sigmoid_forward(const float* x, float* out, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        out[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

__global__ void gpu_sigmoid_backward(const float* grad, const float* x, float* dx, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        float s = 1.0f / (1.0f + expf(-x[i]));
        dx[i] = grad[i] * s * (1.0f - s);
    }
}

enum class Device{
    CPU,
    GPU
};

class Storage{
public:
    Device device_;
    size_t size_;
    std::shared_ptr<float> data_ptr_;

    Storage(size_t size, Device device): device_(device), size_(size) {
        if(device_ == Device::CPU) {
            data_ptr_ = std::shared_ptr<float>(new float[size_], [](float* p) { delete[] p; });
        } else if(device == Device::GPU) {
            float* d_ptr;
            cudaMalloc((void**)&d_ptr, size_ * sizeof(float));
            data_ptr_ = std::shared_ptr<float>(d_ptr, [](float* p) { cudaFree(p); });
        } else {
            throw std::runtime_error("unsupported device type");
        }
    }

};

class Tensor{
private:
    std::shared_ptr<Storage> storage_;
    std::vector<size_t> shape_;
    size_t total_elements_;
    
    size_t calc_total_size() const {
        size_t total = 1;
        for(auto dim: shape_) {
            total *= dim;
        }
        return total;
    }

public:
    Tensor(const std::vector<size_t>& shape, Device device): shape_(shape) {
        total_elements_ = calc_total_size();
        storage_ = std::make_shared<Storage>(total_elements_, device);
    }

    ~Tensor() = default;

    Device device() const {
        return storage_->device_;
    }

    const std::vector<size_t>& shape()const {
        return shape_;
    }

    size_t total_elements() const {
        return total_elements_;
    }

    void zeros() {
        if(total_elements_ == 0){
            throw std::runtime_error("tensor shape invalid");
        }
        if (storage_->device_ == Device::CPU) {
            std::memset(storage_->data_ptr_.get(), 0.0f, total_elements_ * sizeof(float));
        } else if (storage_->device_ == Device::GPU) {
            CUDA_CHECK(cudaMemset(storage_->data_ptr_.get(), 0.0f, total_elements_ * sizeof(float)));
        } else {
            throw std::runtime_error("unsupported device type");
        }
    }

    void ones() {
        if(total_elements_ == 0){
            throw std::runtime_error("tensor shape invalid");
        }
        if (storage_->device_ == Device::CPU) {
            float* p = storage_->data_ptr_.get();
            for (size_t i = 0; i < total_elements_; ++i) {
                p[i] = 1.0f;
            }
        } else if (storage_->device_ == Device::GPU) {
            Tensor t(shape_, Device::CPU);
            float* p = t.storage_->data_ptr_.get();
            for (size_t i = 0; i < total_elements_; ++i) {
                p[i] = 1.0f;
            }
            CUDA_CHECK(cudaMemcpy(storage_->data_ptr_.get(), t.storage_->data_ptr_.get(),
                total_elements_ * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            throw std::runtime_error("unsupported device type");
        }
    }

    float query(const std::vector<size_t>& position) const {
        if(position.size() != shape_.size()) {
            throw std::runtime_error("dimension mismatch");
        }

        if(storage_->device_ != Device::CPU && storage_->device_ != Device::GPU){
            throw std::runtime_error("unsupported device type");
        }

        size_t index = 0;
        size_t mul = 1;
        for(size_t i = position.size(); i-- > 0;) {
            if(position[i] >= shape_[i]) {
                throw std::runtime_error("index out of range");
            }
            index += position[i] * mul;
            mul *= shape_[i];
        }

        if(storage_->device_ == Device::CPU) {
            return storage_->data_ptr_.get()[index];
        } else {
            Tensor t(shape_, Device::CPU);
            CUDA_CHECK(cudaMemcpy(t.storage_->data_ptr_.get(), storage_->data_ptr_.get(),
                        total_elements_ * sizeof(float), cudaMemcpyDeviceToHost));
            float value = t.storage_->data_ptr_.get()[index];
            return value;
        }
    }

    void show() const {
        if(storage_->device_ == Device::CPU){
            const float* print_ptr_ = storage_->data_ptr_.get();
            for(size_t i = 0; i < total_elements_; ++i) {
                printf("%.2f\n", print_ptr_[i]);
            }
        } else if(storage_->device_ == Device::GPU) {
            Tensor t(shape_, Device::CPU);
            CUDA_CHECK(cudaMemcpy(t.storage_->data_ptr_.get(), storage_->data_ptr_.get(),
                        total_elements_ * sizeof(float), cudaMemcpyDeviceToHost));
            const float* print_ptr_ = t.storage_->data_ptr_.get();
            for(size_t i = 0; i < total_elements_; ++i) {
                printf("%.2f\n", print_ptr_[i]);
            }
        } else{
            throw std::runtime_error("unsupported device type");
        }
    }

    Tensor cpu() const {
        if(storage_->device_ == Device::CPU) {
            return *this;
        }
        
        Tensor t(shape_, Device::CPU);
        CUDA_CHECK(cudaMemcpy(t.storage_->data_ptr_.get(), storage_->data_ptr_.get(),
                    total_elements_ * sizeof(float), cudaMemcpyDeviceToHost));
        return t;
    }

    Tensor gpu() const {
        if(storage_->device_ == Device::GPU) {
            return *this;
        }

        Tensor t(shape_, Device::GPU);
        CUDA_CHECK(cudaMemcpy(t.storage_->data_ptr_.get(), storage_->data_ptr_.get(),
                    total_elements_ * sizeof(float), cudaMemcpyHostToDevice));
        return t;
    }

    float* data() {
        if(storage_->device_ != Device::CPU && storage_->device_ != Device::GPU) {
            throw std::runtime_error("unsupported device type");
        }
        return storage_->data_ptr_.get();
    }

    const float* data() const {
        if(storage_->device_ != Device::CPU && storage_->device_ != Device::GPU) {
            throw std::runtime_error("unsupported device type");
        }
        return storage_->data_ptr_.get();
    }

    Tensor relu_forward() const {
        Tensor out_(shape_, storage_->device_);
        const float* x = storage_->data_ptr_.get();
        float* out = out_.storage_->data_ptr_.get();
        int n = int(total_elements_);
        if(storage_->device_ == Device::CPU) {
            for(int i = 0; i < n; ++i) {
                out[i] = x[i] > 0.0f? x[i]: 0.0f;
            }
        } else if(storage_->device_ == Device::GPU) {
            if(n > 0) {
                int size_ = CudaGetBlocks(n);
                gpu_relu_forward<<<size_, kCudaThreadsNum>>>(x, out, n);
                CUDA_CHECK(cudaGetLastError());
            }
        } else{
            throw std::runtime_error("unsupported device type");
        }
        return out_;
    }

    Tensor relu_backward(const Tensor& grad_) const {
        Tensor dx_(shape_, storage_->device_);
        const float* g = grad_.storage_->data_ptr_.get();
        const float* x = storage_->data_ptr_.get();
        float* dx = dx_.storage_->data_ptr_.get();
        int n = int(total_elements_);
        if(storage_->device_ == Device::CPU) {
            for(int i = 0; i < n; ++i) {
                dx[i] = x[i] > 0.0f? g[i]: 0.0f;
            }
        } else if(storage_->device_ == Device::GPU) {
            if(n > 0) {
                int size_ = CudaGetBlocks(n);
                gpu_relu_backward<<<size_, kCudaThreadsNum>>>(g, x, dx, n);
                CUDA_CHECK(cudaGetLastError());
            }
        } else{
            throw std::runtime_error("unsupported device type");
        }
        return dx_;
    }

    Tensor sigmoid_forward() const {
        Tensor out_(shape_, storage_->device_);
        const float* x = storage_->data_ptr_.get();
        float* out = out_.storage_->data_ptr_.get();
        int n = int(total_elements_);
        if(storage_->device_ == Device::CPU) {
            for(int i = 0; i < n; ++i) {
                out[i] = 1.0f / (1 + expf(-x[i]));
            }
        } else if(storage_->device_ == Device::GPU) {
            if(n > 0) {
                int size_ = CudaGetBlocks(n);
                gpu_sigmoid_forward<<<size_, kCudaThreadsNum>>>(x, out, n);
                CUDA_CHECK(cudaGetLastError());
            }
        } else{
            throw std::runtime_error("unsupported device type");
        }
        return out_;
    }

    Tensor sigmoid_backward(const Tensor& grad_) const {
        Tensor dx_(shape_, storage_->device_);
        const float* g = grad_.storage_->data_ptr_.get();
        const float* x = storage_->data_ptr_.get();
        float* dx = dx_.storage_->data_ptr_.get();
        int n = int(total_elements_);
        if(storage_->device_ == Device::CPU) {
            for(size_t i = 0; i < n; ++i) {
                float s = 1.0f / (1.0f + expf(-x[i]));
                dx[i] = g[i] * s * (1.0f - s);
            }
        } else if(storage_->device_ == Device::GPU) {
            if(n > 0) {
                int size_ = CudaGetBlocks(n);
                gpu_sigmoid_backward<<<size_, kCudaThreadsNum>>>(g, x, dx, n);
                CUDA_CHECK(cudaGetLastError());
            }
        } else{
            throw std::runtime_error("unsupported device type");
        }
        return dx_;
    }

};

int main() {
    try { 
        std::vector<size_t> shape = {2, 3, 3};
        Tensor cpu_tensor(shape, Device::CPU);

        size_t size = cpu_tensor.total_elements();
        printf("amount of elements: %zu\n", size);

        float* ptr = cpu_tensor.data();
        for(size_t i = 0; i < size; ++i) {
            ptr[i] = (i * 1.0f - 9.0f) / size;
        }

        Tensor gpu_tensor = cpu_tensor.gpu();
        gpu_tensor.show();

        std::vector<size_t> query_position = {1, 2, 2};
        printf("query: %.2f\n", gpu_tensor.query(query_position));

        // grad 1
        Tensor grad(shape, Device::CPU);
        grad.ones();
        printf("grad:\n");
        grad.show();

        // relu
        Tensor relu_gpu_tensor = gpu_tensor.relu_forward();
        Tensor relu_cpu_tensor = relu_gpu_tensor.cpu();
        printf("relu_cpu_tensor:\n");
        relu_cpu_tensor.show();

        Tensor reluback_gpu_tensor = gpu_tensor.relu_backward(grad.gpu());
        printf("reluback_gpu_tensor:\n");
        reluback_gpu_tensor.show();

        // sigmoid
        Tensor sigmoid_gpu_tensor = gpu_tensor.sigmoid_forward();
        Tensor sigmoid_cpu_tensor = sigmoid_gpu_tensor.cpu();
        printf("sigmoid_cpu_tensor:\n");
        sigmoid_cpu_tensor.show();

        Tensor sigmoidback_gpu_tensor = gpu_tensor.sigmoid_backward(grad.gpu());
        printf("sigmoidback_gpu_tensor:\n");
        sigmoidback_gpu_tensor.show();

    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
    }
    return 0;
}
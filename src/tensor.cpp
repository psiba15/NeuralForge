#include "tensor.h"

// ── Constructors ──────────────────────────────────────────────────

Tensor::Tensor(std::vector<int> shape) 
    : shape_(shape), owns_data_(true) {
    
    compute_strides();
    
    total_size_ = 1;
    for (int dim : shape_) {
        if (dim <= 0) throw std::invalid_argument("Shape dimensions must be positive");
        total_size_ *= dim;
    }
    
    data_ = new float[total_size_]();  // () = zero initialize
}

Tensor::Tensor(std::vector<int> shape, const std::vector<float>& data)
    : shape_(shape), owns_data_(true) {
    
    compute_strides();
    
    total_size_ = 1;
    for (int dim : shape_) total_size_ *= dim;
    
    if ((int)data.size() != total_size_)
        throw std::invalid_argument("Data size does not match shape");
    
    data_ = new float[total_size_];
    for (int i = 0; i < total_size_; i++) data_[i] = data[i];
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), strides_(other.strides_),
      total_size_(other.total_size_), owns_data_(true) {
    
    data_ = new float[total_size_];
    for (int i = 0; i < total_size_; i++) data_[i] = other.data_[i];
}

Tensor::~Tensor() {
    if (owns_data_ && data_) {
        delete[] data_;
        data_ = nullptr;
    }
}

// ── Element access ────────────────────────────────────────────────

float& Tensor::at(std::vector<int> indices) {
    if (indices.size() != shape_.size())
        throw std::invalid_argument("Wrong number of indices");
    
    int flat_index = 0;
    for (int i = 0; i < (int)indices.size(); i++) {
        if (indices[i] < 0 || indices[i] >= shape_[i])
            throw std::out_of_range("Index out of bounds");
        flat_index += indices[i] * strides_[i];
    }
    return data_[flat_index];
}

float Tensor::at(std::vector<int> indices) const {
    if (indices.size() != shape_.size())
        throw std::invalid_argument("Wrong number of indices");
    
    int flat_index = 0;
    for (int i = 0; i < (int)indices.size(); i++) {
        if (indices[i] < 0 || indices[i] >= shape_[i])
            throw std::out_of_range("Index out of bounds");
        flat_index += indices[i] * strides_[i];
    }
    return data_[flat_index];
}

// ── Shape operations ──────────────────────────────────────────────

Tensor Tensor::transpose() const {
    if (shape_.size() != 2)
        throw std::runtime_error("Transpose sirf 2D tensors pe kaam karta hai");
    
    // naya tensor banao — shape palti hui
    Tensor result({shape_[1], shape_[0]});
    
    // data copy karo — transpose karke
    for (int i = 0; i < shape_[0]; i++)
        for (int j = 0; j < shape_[1]; j++)
            result.at({j, i}) = this->at({i, j});
    
    return result;
}

Tensor Tensor::reshape(std::vector<int> new_shape) const {
    int new_total = 1;
    for (int dim : new_shape) new_total *= dim;
    
    if (new_total != total_size_)
        throw std::invalid_argument("Reshape: total elements match nahi kar rahe");
    
    // data copy karo new shape mein
    Tensor result(new_shape);
    for (int i = 0; i < total_size_; i++) result.data_ptr()[i] = data_[i];
    return result;
}

// ── Math operations ───────────────────────────────────────────────

Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_)
        throw std::invalid_argument("Add: shapes match nahi kar rahe");
    
    Tensor result(shape_);
    for (int i = 0; i < total_size_; i++)
        result.data_ptr()[i] = data_[i] + other.data_[i];
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_)
        throw std::invalid_argument("Subtract: shapes match nahi kar rahe");
    
    Tensor result(shape_);
    for (int i = 0; i < total_size_; i++)
        result.data_ptr()[i] = data_[i] - other.data_[i];
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_)
        throw std::invalid_argument("Element-wise multiply: shapes match nahi");
    
    Tensor result(shape_);
    for (int i = 0; i < total_size_; i++)
        result.data_ptr()[i] = data_[i] * other.data_[i];
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape_);
    for (int i = 0; i < total_size_; i++)
        result.data_ptr()[i] = data_[i] * scalar;
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    // shape check — A: {M,K}, B: {K,N} → result: {M,N}
    if (shape_.size() != 2 || other.shape_.size() != 2)
        throw std::invalid_argument("Matmul sirf 2D tensors pe kaam karta hai");
    
    if (shape_[1] != other.shape_[0])
        throw std::invalid_argument("Matmul: A columns != B rows");
    
    int M = shape_[0];
    int K = shape_[1];
    int N = other.shape_[1];
    
    Tensor result({M, N});
    
    // teen nested loops — naive CPU matmul
    // Step 4 mein CUDA se replace karenge
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += this->at({i, k}) * other.at({k, j});
            result.at({i, j}) = sum;
        }
    
    return result;
}

// ── Utility ───────────────────────────────────────────────────────

void Tensor::fill(float value) {
    for (int i = 0; i < total_size_; i++) data_[i] = value;
}

void Tensor::print() const {
    std::cout << "Tensor shape: [";
    for (int i = 0; i < (int)shape_.size(); i++) {
        std::cout << shape_[i];
        if (i < (int)shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    if (shape_.size() == 1) {
        std::cout << "[";
        for (int i = 0; i < total_size_; i++) {
            std::cout << data_[i];
            if (i < total_size_ - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    else if (shape_.size() == 2) {
        for (int i = 0; i < shape_[0]; i++) {
            std::cout << "[";
            for (int j = 0; j < shape_[1]; j++) {
                std::cout << at({i, j});
                if (j < shape_[1] - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    }
}

// ── Static factory functions ──────────────────────────────────────

Tensor Tensor::zeros(std::vector<int> shape) {
    Tensor t(shape);
    t.fill(0.0f);
    return t;
}

Tensor Tensor::ones(std::vector<int> shape) {
    Tensor t(shape);
    t.fill(1.0f);
    return t;
}

Tensor Tensor::random_normal(std::vector<int> shape, float mean, float std) {
    Tensor t(shape);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);
    
    for (int i = 0; i < t.total_size_; i++)
        t.data_[i] = dist(gen);
    
    return t;
}
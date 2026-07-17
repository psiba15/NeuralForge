#pragma once

#include <vector>
#include <iostream>
#include <stdexcept>
#include <random>
#include <cmath>
#include <functional>

class Tensor {
private:
    float* data_;           // actual numbers flat array mein
    std::vector<int> shape_;    // dimensions
    std::vector<int> strides_;  // jump sizes
    int total_size_;            // total elements
    bool owns_data_;            // kya maine khud allocate kiya?

    // strides calculate karna — shape se
    void compute_strides() {
        strides_.resize(shape_.size());
        int stride = 1;
        for (int i = (int)shape_.size() - 1; i >= 0; i--) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

public:
    // ── Constructors ──────────────────────────────
    
    // empty tensor — shape se banao, zeros se fill
    Tensor(std::vector<int> shape);
    
    // existing data se banao — copy karta hai
    Tensor(std::vector<int> shape, const std::vector<float>& data);
    
    // copy constructor
    Tensor(const Tensor& other);
    
    // destructor — memory free karo
    ~Tensor();

    // ── Access ────────────────────────────────────
    
    // element access — indices vector se
    float& at(std::vector<int> indices);
    float  at(std::vector<int> indices) const;
    
    // raw pointer — CUDA ke liye baad mein kaam aayega
    float* data_ptr() { return data_; }
    const float* data_ptr() const { return data_; }
    
    // shape aur size
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& strides() const { return strides_; }
    int total_size() const { return total_size_; }
    int ndim() const { return shape_.size(); }

    // ── Shape operations ──────────────────────────
    
    Tensor transpose() const;           // 2D tensor palat do
    Tensor reshape(std::vector<int> new_shape) const;

    // ── Math operations ───────────────────────────
    
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;  // element-wise
    Tensor operator*(float scalar) const;          // scalar multiply
    
    Tensor matmul(const Tensor& other) const;     // matrix multiply
    
    // ── Utility ───────────────────────────────────
    
    void fill(float value);
    void print() const;
    
    // ── Static factory functions ──────────────────
    
    static Tensor zeros(std::vector<int> shape);
    static Tensor ones(std::vector<int> shape);
    static Tensor random_normal(std::vector<int> shape, 
                                 float mean = 0.0f, 
                                 float std  = 1.0f);
};

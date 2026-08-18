#pragma once

#include "value.h"
#include <vector>
#include <random>

// ── Linear layer (fully connected) ──────────────────────────────
class Linear {
public:
    std::vector<std::vector<Value*>> weights;  // [out_features][in_features]
    std::vector<Value*> bias;                  // [out_features]
    int in_features, out_features;

    Linear(int in_features, int out_features);
    ~Linear();

    // arena: agar diya, har NAYA temporary Value is vector mein track hota
    // hai taaki caller baad mein delete kar sake (params kabhi track nahi hote)
    std::vector<Value*> forward(std::vector<Value*>& input,
                                 std::vector<Value*>* arena = nullptr);
    std::vector<Value*> parameters();
};

// ── ReLU (stateless) ─────────────────────────────────────────────
class ReLU {
public:
    static std::vector<Value*> forward(std::vector<Value*>& input,
                                        std::vector<Value*>* arena = nullptr);
};

// ── Softmax (stateless, numerically stable) ──────────────────────
class Softmax {
public:
    static std::vector<Value*> forward(std::vector<Value*>& input,
                                        std::vector<Value*>* arena = nullptr);
};

// ── CrossEntropy loss (softmax + neg log likelihood) ─────────────
Value* cross_entropy_loss(std::vector<Value*>& logits, int target_class,
                           std::vector<Value*>* arena = nullptr);

// ── Adam optimizer ────────────────────────────────────────────────
class Adam {
public:
    std::vector<Value*> params;
    float lr, beta1, beta2, eps;
    int t;
    std::vector<float> m, v;

    Adam(std::vector<Value*> params, float lr = 0.001f,
         float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

    void step();
    void zero_grad();
};
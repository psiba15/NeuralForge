#pragma once

#include "value.h"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

// ── Neuron ────────────────────────────────────────────────────────
// Ek single neuron — weights + bias + activation
class Neuron {
public:
    std::vector<Value*> weights;
    Value* bias;
    bool use_relu;   // true = relu, false = tanh

    Neuron(int n_inputs, bool use_relu = true);
    ~Neuron();

    Value* forward(std::vector<Value*>& inputs);
    std::vector<Value*> parameters();
};

// ── Layer ─────────────────────────────────────────────────────────
// Multiple neurons — ek saath
class Layer {
public:
    std::vector<Neuron*> neurons;

    Layer(int n_inputs, int n_neurons, bool use_relu = true);
    ~Layer();

    std::vector<Value*> forward(std::vector<Value*>& inputs);
    std::vector<Value*> parameters();
};

// ── MLP ───────────────────────────────────────────────────────────
// Multiple layers stack — complete neural network
class MLP {
public:
    std::vector<Layer*> layers;

    // sizes = {hidden1, hidden2, ..., output}
    MLP(int n_inputs, std::vector<int> layer_sizes);
    ~MLP();

    std::vector<Value*> forward(std::vector<Value*>& inputs);
    std::vector<Value*> parameters();
    void zero_grad();
    void print_summary();
};

// ── Loss functions ────────────────────────────────────────────────
Value* mse_loss(std::vector<Value*>& predicted, 
                std::vector<Value*>& actual);

Value* binary_cross_entropy(Value* predicted, Value* actual);
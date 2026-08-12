#include "neural.h"

// ── Neuron ────────────────────────────────────────────────────────

Neuron::Neuron(int n_inputs, bool use_relu) : use_relu(use_relu) {
    // Xavier initialization
    // std = sqrt(1 / n_inputs)
    std::random_device rd;
    std::mt19937 gen(rd());
    float xavier_std = std::sqrt(1.0f / n_inputs);
    std::normal_distribution<float> dist(0.0f, xavier_std);

    for (int i = 0; i < n_inputs; i++)
        weights.push_back(new Value(dist(gen)));

    bias = new Value(0.0f);  // bias 0 se shuru
}

Neuron::~Neuron() {
    for (auto w : weights) delete w;
    delete bias;
}

Value* Neuron::forward(std::vector<Value*>& inputs) {
    if (inputs.size() != weights.size())
        throw std::invalid_argument("Neuron: input size != weight size");

    // weighted sum: w1*x1 + w2*x2 + ... + b
    Value* act = bias;
    for (int i = 0; i < (int)weights.size(); i++)
        act = act->add(weights[i]->mul(inputs[i]));

    // activation apply karo
    return use_relu ? act->relu() : act->tanh_op();
}

std::vector<Value*> Neuron::parameters() {
    std::vector<Value*> params = weights;
    params.push_back(bias);
    return params;
}

// ── Layer ─────────────────────────────────────────────────────────

Layer::Layer(int n_inputs, int n_neurons, bool use_relu) {
    for (int i = 0; i < n_neurons; i++)
        neurons.push_back(new Neuron(n_inputs, use_relu));
}

Layer::~Layer() {
    for (auto n : neurons) delete n;
}

std::vector<Value*> Layer::forward(std::vector<Value*>& inputs) {
    std::vector<Value*> outputs;
    for (auto neuron : neurons)
        outputs.push_back(neuron->forward(inputs));
    return outputs;
}

std::vector<Value*> Layer::parameters() {
    std::vector<Value*> params;
    for (auto neuron : neurons)
        for (auto p : neuron->parameters())
            params.push_back(p);
    return params;
}

// ── MLP ───────────────────────────────────────────────────────────

MLP::MLP(int n_inputs, std::vector<int> layer_sizes) {
    int in_size = n_inputs;
    for (int i = 0; i < (int)layer_sizes.size(); i++) {
        // last layer mein relu nahi — raw output chahiye
        bool use_relu = (i < (int)layer_sizes.size() - 1);
        layers.push_back(new Layer(in_size, layer_sizes[i], use_relu));
        in_size = layer_sizes[i];
    }
}

MLP::~MLP() {
    for (auto l : layers) delete l;
}

std::vector<Value*> MLP::forward(std::vector<Value*>& inputs) {
    std::vector<Value*> current = inputs;
    for (auto layer : layers)
        current = layer->forward(current);
    return current;
}

std::vector<Value*> MLP::parameters() {
    std::vector<Value*> params;
    for (auto layer : layers)
        for (auto p : layer->parameters())
            params.push_back(p);
    return params;
}

void MLP::zero_grad() {
    for (auto p : parameters())
        p->grad = 0.0f;
}

void MLP::print_summary() {
    std::cout << "MLP Architecture:\n";
    int total = 0;
    for (int i = 0; i < (int)layers.size(); i++) {
        int params = layers[i]->parameters().size();
        total += params;
        std::cout << "  Layer " << i+1 
                  << ": " << layers[i]->neurons.size() 
                  << " neurons, " << params << " params\n";
    }
    std::cout << "  Total parameters: " << total << "\n";
}

// ── Loss functions ────────────────────────────────────────────────

Value* mse_loss(std::vector<Value*>& predicted,
                std::vector<Value*>& actual) {
    if (predicted.size() != actual.size())
        throw std::invalid_argument("MSE: size mismatch");

    // loss = mean((pred - actual)^2)
    Value* total = new Value(0.0f);
    for (int i = 0; i < (int)predicted.size(); i++) {
        Value* diff = predicted[i]->sub(actual[i]);
        Value* sq   = diff->pow_op(2.0f);
        total = total->add(sq);
    }

    // mean
    Value* n = new Value((float)predicted.size());
    return total->div_op(n);
}

Value* binary_cross_entropy(Value* predicted, Value* actual) {
    // loss = -(actual * log(pred) + (1-actual) * log(1-pred))
    Value* one = new Value(1.0f);

    Value* term1 = actual->mul(predicted->log_op());
    Value* term2 = (one->sub(actual))->mul(
                    (one->sub(predicted))->log_op()
                   );

    Value* sum  = term1->add(term2);
    Value* neg  = new Value(-1.0f);
    return neg->mul(sum);
}
#include "../include/layers.h"
#include <cmath>

// ═══════════════════ Linear ═══════════════════

Linear::Linear(int in_features, int out_features)
    : in_features(in_features), out_features(out_features)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    // He initialization — accha kaam karta hai ReLU ke saath
    float std_dev = std::sqrt(2.0f / in_features);
    std::normal_distribution<float> dist(0.0f, std_dev);

    for (int i = 0; i < out_features; ++i) {
        std::vector<Value*> row;
        for (int j = 0; j < in_features; ++j) {
            row.push_back(new Value(dist(gen)));
        }
        weights.push_back(row);
        bias.push_back(new Value(0.0f));
    }
}

Linear::~Linear() {
    for (auto& row : weights)
        for (auto w : row) delete w;
    for (auto b : bias) delete b;
}

std::vector<Value*> Linear::forward(std::vector<Value*>& input) {
    std::vector<Value*> output;
    for (int i = 0; i < out_features; ++i) {
        Value* sum = bias[i];
        for (int j = 0; j < in_features; ++j) {
            Value* prod = weights[i][j]->mul(input[j]);
            sum = sum->add(prod);
        }
        output.push_back(sum);
    }
    return output;
}

std::vector<Value*> Linear::parameters() {
    std::vector<Value*> params;
    for (auto& row : weights)
        for (auto w : row) params.push_back(w);
    for (auto b : bias) params.push_back(b);
    return params;
}

// ═══════════════════ ReLU ═══════════════════

std::vector<Value*> ReLU::forward(std::vector<Value*>& input) {
    std::vector<Value*> output;
    for (auto v : input) output.push_back(v->relu());
    return output;
}

// ═══════════════════ Softmax ═══════════════════
// Numerically stable: max subtract karke phir exp/sum

std::vector<Value*> Softmax::forward(std::vector<Value*>& input) {
    float max_val = input[0]->data;
    for (auto v : input)
        if (v->data > max_val) max_val = v->data;

    Value* max_const = new Value(max_val);

    std::vector<Value*> shifted;
    for (auto v : input) shifted.push_back(v->sub(max_const));

    std::vector<Value*> exps;
    for (auto v : shifted) exps.push_back(v->exp_op());

    Value* sum = exps[0];
    for (size_t i = 1; i < exps.size(); ++i) sum = sum->add(exps[i]);

    std::vector<Value*> probs;
    for (auto v : exps) probs.push_back(v->div_op(sum));

    return probs;
}

// ═══════════════════ CrossEntropy ═══════════════════
// logits -> softmax -> -log(prob[target])

Value* cross_entropy_loss(std::vector<Value*>& logits, int target_class) {
    std::vector<Value*> probs = Softmax::forward(logits);
    Value* p = probs[target_class];
    Value* logp = p->log_op();
    return logp->neg();
}

// ═══════════════════ Adam ═══════════════════

Adam::Adam(std::vector<Value*> params, float lr, float beta1, float beta2, float eps)
    : params(params), lr(lr), beta1(beta1), beta2(beta2), eps(eps), t(0)
{
    m.resize(params.size(), 0.0f);
    v.resize(params.size(), 0.0f);
}

void Adam::step() {
    t++;
    for (size_t i = 0; i < params.size(); ++i) {
        float g = params[i]->grad;

        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

        float m_hat = m[i] / (1.0f - std::pow(beta1, t));
        float v_hat = v[i] / (1.0f - std::pow(beta2, t));

        params[i]->data -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

void Adam::zero_grad() {
    for (auto p : params) p->grad = 0.0f;
}
#include "value.h"

// ── Constructor ───────────────────────────────────────────────────

Value::Value(float data, std::string op)
    : data(data), grad(0.0f), op(op) {
    // backward_fn empty hai by default — leaf nodes ke liye
}

// ── Operations ───────────────────────────────────────────────────

Value* Value::add(Value* other) {
    Value* out = new Value(this->data + other->data, "+");
    out->parents = {this, other};

    // chain rule for addition:
    // d(a+b)/da = 1, d(a+b)/db = 1
    out->backward_fn = [this, other, out]() {
        this->grad  += out->grad * 1.0f;
        other->grad += out->grad * 1.0f;
    };

    return out;
}

Value* Value::mul(Value* other) {
    Value* out = new Value(this->data * other->data, "*");
    out->parents = {this, other};

    // chain rule for multiplication:
    // d(a*b)/da = b, d(a*b)/db = a
    out->backward_fn = [this, other, out]() {
        this->grad  += out->grad * other->data;
        other->grad += out->grad * this->data;
    };

    return out;
}

Value* Value::tanh_op() {
    float t = std::tanh(this->data);
    Value* out = new Value(t, "tanh");
    out->parents = {this};

    // chain rule for tanh:
    // d(tanh(x))/dx = 1 - tanh²(x)
    // out->data already tanh(x) hai — recompute nahi karna
    out->backward_fn = [this, out]() {
        this->grad += out->grad * (1.0f - out->data * out->data);
    };

    return out;
}

Value* Value::pow_op(float exp) {
    float result = std::pow(this->data, exp);
    Value* out = new Value(result, "pow");
    out->parents = {this};

    // chain rule: d(x^n)/dx = n * x^(n-1)
    out->backward_fn = [this, out, exp]() {
        this->grad += out->grad * exp * std::pow(this->data, exp - 1.0f);
    };

    return out;
}

Value* Value::neg() {
    return this->mul(new Value(-1.0f));
}

Value* Value::sub(Value* other) {
    return this->add(other->neg());
}

Value* Value::exp_op() {
    float result = std::exp(this->data);
    Value* out = new Value(result, "exp");
    out->parents = {this};

    // chain rule: d(e^x)/dx = e^x
    // out->data already e^x hai
    out->backward_fn = [this, out]() {
        this->grad += out->grad * out->data;
    };

    return out;
}

Value* Value::div_op(Value* other) {
    // a/b = a * b^(-1)
    Value* inv = other->pow_op(-1.0f);
    return this->mul(inv);
}

Value* Value::log_op() {
    float result = std::log(this->data);
    Value* out = new Value(result, "log");
    out->parents = {this};

    // chain rule: d(ln(x))/dx = 1/x
    out->backward_fn = [this, out]() {
        this->grad += out->grad * (1.0f / this->data);
    };

    return out;
}

Value* Value::relu() {
    float result = this->data > 0.0f ? this->data : 0.0f;
    Value* out = new Value(result, "relu");
    out->parents = {this};

    // chain rule: d(relu(x))/dx = 1 if x > 0, else 0
    out->backward_fn = [this, out]() {
        this->grad += out->grad * (this->data > 0.0f ? 1.0f : 0.0f);
    };

    return out;
}

Value* Value::sigmoid() {
    // sigmoid(x) = 1 / (1 + e^-x)
    float s = 1.0f / (1.0f + std::exp(-this->data));
    Value* out = new Value(s, "sigmoid");
    out->parents = {this};

    // chain rule: d(sigmoid)/dx = sigmoid * (1 - sigmoid)
    out->backward_fn = [this, out]() {
        this->grad += out->grad * out->data * (1.0f - out->data);
    };

    return out;
}

// ── Backward pass ─────────────────────────────────────────────────

void Value::build_topo(Value* v,
                        std::set<Value*>& visited,
                        std::vector<Value*>& topo) {
    // agar already dekha hai toh skip
    if (visited.count(v)) return;
    visited.insert(v);

    // pehle parents ko process karo
    for (Value* parent : v->parents) {
        build_topo(parent, visited, topo);
    }

    // phir current node add karo
    topo.push_back(v);
}

void Value::backward() {
    // Step 1 — topological order nikalo
    std::vector<Value*> topo;
    std::set<Value*> visited;
    build_topo(this, visited, topo);

    // Step 2 — loss ka gradient = 1.0 (hamesha)
    this->grad = 1.0f;

    // Step 3 — reverse order mein backward_fn call karo
    // topo mein last = current node (loss)
    // reverse karne se loss pehle, weights baad mein
    for (int i = (int)topo.size() - 1; i >= 0; i--) {
        if (topo[i]->backward_fn) {
            topo[i]->backward_fn();
        }
    }
}

// ── Utility ───────────────────────────────────────────────────────

void Value::print() const {
    std::cout << "Value(data=" << data
              << ", grad=" << grad
              << ", op=" << op << ")\n";
}
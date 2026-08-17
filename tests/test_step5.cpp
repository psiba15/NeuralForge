#include <cstdio>
#include <cmath>
#include <vector>
#include "../include/value.h"
#include "../include/layers.h"

bool approx(float a, float b, float tol = 1e-2f) {
    return std::fabs(a - b) < tol;
}

int main() {
    printf("=== Step 5 Tests ===\n\n");

    // ---- Test 1: Linear forward shape ----
    Linear fc1(4, 3);
    std::vector<Value*> input;
    for (int i = 0; i < 4; ++i) input.push_back(new Value(0.5f));
    std::vector<Value*> out1 = fc1.forward(input);
    bool test1 = (out1.size() == 3);
    printf("[Test 1] Linear forward output shape: %s\n", test1 ? "PASS" : "FAIL");

    // ---- Test 2: ReLU zeroes negatives ----
    std::vector<Value*> relu_in = { new Value(-2.0f), new Value(3.0f),
                                     new Value(-0.5f), new Value(0.0f) };
    std::vector<Value*> relu_out = ReLU::forward(relu_in);
    bool test2 = approx(relu_out[0]->data, 0.0f) && approx(relu_out[1]->data, 3.0f) &&
                 approx(relu_out[2]->data, 0.0f) && approx(relu_out[3]->data, 0.0f);
    printf("[Test 2] ReLU zeroes negatives: %s\n", test2 ? "PASS" : "FAIL");

    // ---- Test 3: Softmax sums to 1, non-negative ----
    std::vector<Value*> sm_in = { new Value(2.0f), new Value(1.0f), new Value(0.1f) };
    std::vector<Value*> probs = Softmax::forward(sm_in);
    float sum = 0.0f;
    bool all_positive = true;
    for (auto p : probs) {
        sum += p->data;
        if (p->data < 0.0f) all_positive = false;
    }
    bool test3 = approx(sum, 1.0f) && all_positive;
    printf("[Test 3] Softmax sums to 1 & non-negative: %s (sum=%f)\n",
           test3 ? "PASS" : "FAIL", sum);

    // ---- Test 4: CrossEntropy gradient check (numerical vs analytic) ----
    Linear fc2(3, 2);
    std::vector<Value*> params = fc2.parameters();

    auto compute_loss = [&](std::vector<Value*>& in) -> Value* {
        std::vector<Value*> logits = fc2.forward(in);
        return cross_entropy_loss(logits, 0);
    };

    std::vector<Value*> ce_in = { new Value(0.3f), new Value(-0.7f), new Value(1.2f) };
    Value* loss = compute_loss(ce_in);

    for (auto p : params) p->grad = 0.0f;
    loss->backward();

    Value* w = params[0];
    float orig = w->data;
    float eps = 1e-3f;

    w->data = orig + eps;
    std::vector<Value*> ce_in2 = { new Value(0.3f), new Value(-0.7f), new Value(1.2f) };
    Value* loss_plus = compute_loss(ce_in2);

    w->data = orig - eps;
    std::vector<Value*> ce_in3 = { new Value(0.3f), new Value(-0.7f), new Value(1.2f) };
    Value* loss_minus = compute_loss(ce_in3);

    w->data = orig;

    float numerical_grad = (loss_plus->data - loss_minus->data) / (2.0f * eps);
    bool test4 = approx(numerical_grad, w->grad, 1e-1f);
    printf("[Test 4] CrossEntropy gradient check: %s (analytic=%f, numerical=%f)\n",
           test4 ? "PASS" : "FAIL", w->grad, numerical_grad);

    // ---- Test 5: Adam optimizer reduces loss over steps ----
    Linear fc3(2, 2);
    std::vector<Value*> adam_params = fc3.parameters();
    Adam optimizer(adam_params, 0.05f);

    float first_loss = -1.0f, last_loss = -1.0f;
    for (int step = 0; step < 50; ++step) {
        std::vector<Value*> x = { new Value(1.0f), new Value(-1.0f) };
        std::vector<Value*> logits = fc3.forward(x);
        Value* l = cross_entropy_loss(logits, 0);

        optimizer.zero_grad();
        l->backward();
        optimizer.step();

        if (step == 0) first_loss = l->data;
        if (step == 49) last_loss = l->data;
    }

    bool test5 = last_loss < first_loss;
    printf("[Test 5] Adam optimizer reduces loss: %s (first=%f, last=%f)\n",
           test5 ? "PASS" : "FAIL", first_loss, last_loss);

    bool all_pass = test1 && test2 && test3 && test4 && test5;
    printf("\n=== %s ===\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
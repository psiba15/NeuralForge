// // #include <iostream>

// // int main() {
// //     std::cout << "NeuralForge initialized" << std::endl;
// //     return 0;
// // }

// #include <iostream>
// #include <cassert>
// #include <cmath>
// #include "tensor.h"

// // helper — do floats close hain?
// bool close(float a, float b, float eps = 1e-4f) {
//     return std::abs(a - b) < eps;
// }

// void test_basic_creation() {
//     std::cout << "Test 1: Basic creation... ";
    
//     Tensor t({2, 3});
//     assert(t.shape()[0] == 2);
//     assert(t.shape()[1] == 3);
//     assert(t.total_size() == 6);
    
//     std::cout << "PASSED\n";
// }

// void test_element_access() {
//     std::cout << "Test 2: Element access... ";
    
//     Tensor t({2, 3}, {1,2,3,4,5,6});
    
//     assert(close(t.at({0,0}), 1.0f));
//     assert(close(t.at({0,1}), 2.0f));
//     assert(close(t.at({1,0}), 4.0f));
//     assert(close(t.at({1,2}), 6.0f));
    
//     std::cout << "PASSED\n";
// }

// void test_transpose() {
//     std::cout << "Test 3: Transpose... ";
    
//     // [[1,2,3],[4,5,6]] transpose → [[1,4],[2,5],[3,6]]
//     Tensor t({2, 3}, {1,2,3,4,5,6});
//     Tensor tr = t.transpose();
    
//     assert(tr.shape()[0] == 3);
//     assert(tr.shape()[1] == 2);
//     assert(close(tr.at({0,0}), 1.0f));
//     assert(close(tr.at({0,1}), 4.0f));
//     assert(close(tr.at({1,0}), 2.0f));
//     assert(close(tr.at({2,1}), 6.0f));
    
//     std::cout << "PASSED\n";
// }

// void test_matmul() {
//     std::cout << "Test 4: Matrix multiplication... ";
    
//     // A: [[1,2],[3,4]] B: [[5,6],[7,8]]
//     // C[0][0] = 1*5 + 2*7 = 19
//     // C[0][1] = 1*6 + 2*8 = 22
//     // C[1][0] = 3*5 + 4*7 = 43
//     // C[1][1] = 3*6 + 4*8 = 50
//     Tensor A({2, 2}, {1,2,3,4});
//     Tensor B({2, 2}, {5,6,7,8});
//     Tensor C = A.matmul(B);
    
//     assert(close(C.at({0,0}), 19.0f));
//     assert(close(C.at({0,1}), 22.0f));
//     assert(close(C.at({1,0}), 43.0f));
//     assert(close(C.at({1,1}), 50.0f));
    
//     std::cout << "PASSED\n";
// }

// void test_factory_functions() {
//     std::cout << "Test 5: Factory functions... ";
    
//     Tensor z = Tensor::zeros({3, 3});
//     assert(close(z.at({0,0}), 0.0f));
//     assert(close(z.at({2,2}), 0.0f));
    
//     Tensor o = Tensor::ones({2, 4});
//     assert(close(o.at({0,0}), 1.0f));
//     assert(close(o.at({1,3}), 1.0f));
    
//     Tensor r = Tensor::random_normal({100, 100});
//     // mean roughly 0 hona chahiye
//     float sum = 0;
//     for (int i = 0; i < 100; i++)
//         for (int j = 0; j < 100; j++)
//             sum += r.at({i,j});
//     float mean = sum / 10000.0f;
//     assert(std::abs(mean) < 0.1f);  // roughly 0
    
//     std::cout << "PASSED\n";
// }

// void test_operations() {
//     std::cout << "Test 6: Math operations... ";
    
//     Tensor A({2,2}, {1,2,3,4});
//     Tensor B({2,2}, {5,6,7,8});
    
//     Tensor C = A + B;
//     assert(close(C.at({0,0}), 6.0f));
//     assert(close(C.at({1,1}), 12.0f));
    
//     Tensor D = A * 3.0f;
//     assert(close(D.at({0,0}), 3.0f));
//     assert(close(D.at({1,1}), 12.0f));
    
//     std::cout << "PASSED\n";
// }

// int main() {
//     std::cout << "NeuralForge — Step 1: Tensor Tests\n";
//     std::cout << "====================================\n";
    
//     test_basic_creation();
//     test_element_access();
//     test_transpose();
//     test_matmul();
//     test_factory_functions();
//     test_operations();
    
//     std::cout << "====================================\n";
//     std::cout << "Saare tests PASSED — Tensor engine ready!\n";
    
//     return 0;
// }






// test for autograd engine
#include <iostream>
#include <cassert>
#include <cmath>
#include "tensor.h"
#include "value.h"

bool close(float a, float b, float eps = 1e-3f) {
    return std::abs(a - b) < eps;
}

// ── Test 1: Simple addition ───────────────────────────────────────
void test_addition() {
    std::cout << "Test 1: Addition backward... ";

    Value* a = new Value(3.0f);
    Value* b = new Value(4.0f);
    Value* c = a->add(b);   // c = a + b = 7

    c->backward();

    // d(a+b)/da = 1, d(a+b)/db = 1
    assert(close(a->grad, 1.0f));
    assert(close(b->grad, 1.0f));
    assert(close(c->data, 7.0f));

    delete a; delete b; delete c;
    std::cout << "PASSED\n";
}

// ── Test 2: Simple multiplication ────────────────────────────────
void test_multiplication() {
    std::cout << "Test 2: Multiplication backward... ";

    Value* a = new Value(3.0f);
    Value* b = new Value(4.0f);
    Value* c = a->mul(b);   // c = a * b = 12

    c->backward();

    // d(a*b)/da = b = 4, d(a*b)/db = a = 3
    assert(close(a->grad, 4.0f));
    assert(close(b->grad, 3.0f));

    delete a; delete b; delete c;
    std::cout << "PASSED\n";
}

// ── Test 3: Chain rule — x*w + b ─────────────────────────────────
void test_chain_rule() {
    std::cout << "Test 3: Chain rule (x*w + b)... ";

    Value* x = new Value(2.0f);
    Value* w = new Value(3.0f);
    Value* b = new Value(1.0f);

    Value* xw   = x->mul(w);       // xw  = 6
    Value* out  = xw->add(b);      // out = 7
    Value* loss = out->pow_op(2);  // loss = 49

    loss->backward();

    // manually:
    // d(loss)/d(out) = 2 * out = 2 * 7 = 14
    // d(loss)/d(xw)  = 14 * 1 = 14
    // d(loss)/d(w)   = 14 * x = 14 * 2 = 28
    // d(loss)/d(b)   = 14 * 1 = 14
    // d(loss)/d(x)   = 14 * w = 14 * 3 = 42

    assert(close(w->grad, 28.0f));
    assert(close(b->grad, 14.0f));
    assert(close(x->grad, 42.0f));

    std::cout << "PASSED\n";
}

// ── Test 4: Tanh ──────────────────────────────────────────────────
void test_tanh() {
    std::cout << "Test 4: Tanh backward... ";

    Value* x = new Value(0.5f);
    Value* y = x->tanh_op();

    y->backward();

    // d(tanh(x))/dx = 1 - tanh²(x)
    float t = std::tanh(0.5f);
    float expected_grad = 1.0f - t * t;

    assert(close(x->grad, expected_grad));
    assert(close(y->data, t));

    std::cout << "PASSED\n";
}

// ── Test 5: Numerical gradient check — MOST IMPORTANT TEST ───────
// Ye test blindly trust karo — agar ye pass hai, backward 100% correct hai
void test_numerical_gradient() {
    std::cout << "Test 5: Numerical gradient check... ";

    float epsilon = 1e-4f;

    // f(x) = tanh(x*w + b)^2
    // w ka gradient numerically check karenge
    float x_val = 1.5f;
    float w_val = 2.0f;
    float b_val = 0.5f;

    // f(w + eps)
    {
        Value* x = new Value(x_val);
        Value* w = new Value(w_val + epsilon);
        Value* b = new Value(b_val);
        Value* out = x->mul(w)->add(b)->tanh_op()->pow_op(2);
        float loss_plus = out->data;

        // f(w - eps)
        Value* x2 = new Value(x_val);
        Value* w2 = new Value(w_val - epsilon);
        Value* b2 = new Value(b_val);
        Value* out2 = x2->mul(w2)->add(b2)->tanh_op()->pow_op(2);
        float loss_minus = out2->data;

        float numerical_grad = (loss_plus - loss_minus) / (2 * epsilon);

        // autograd se
        Value* xa = new Value(x_val);
        Value* wa = new Value(w_val);
        Value* ba = new Value(b_val);
        Value* loss = xa->mul(wa)->add(ba)->tanh_op()->pow_op(2);
        loss->backward();

        assert(close(wa->grad, numerical_grad, 1e-2f));
    }

    std::cout << "PASSED\n";
}

// ── Test 6: Gradient accumulation ────────────────────────────────
void test_grad_accumulation() {
    std::cout << "Test 6: Gradient accumulation... ";

    // ek value do jagah use ho — dono gradients add hone chahiye
    Value* x = new Value(2.0f);
    Value* a = x->mul(x);   // a = x² — x dono jagah use hua
    // d(x²)/dx = 2x = 4

    a->backward();

    assert(close(x->grad, 4.0f));

    std::cout << "PASSED\n";
}


void test_relu() {
    std::cout << "Test 7: ReLU backward... ";

    // positive input — gradient pass through hona chahiye
    Value* x1 = new Value(3.0f);
    Value* y1 = x1->relu();
    y1->backward();
    assert(close(y1->data, 3.0f));
    assert(close(x1->grad, 1.0f));  // positive tha, grad = 1

    // negative input — gradient zero hona chahiye
    Value* x2 = new Value(-2.0f);
    Value* y2 = x2->relu();
    y2->backward();
    assert(close(y2->data, 0.0f));
    assert(close(x2->grad, 0.0f));  // negative tha, grad = 0

    std::cout << "PASSED\n";
}

void test_sigmoid() {
    std::cout << "Test 8: Sigmoid backward... ";

    Value* x = new Value(0.0f);
    Value* y = x->sigmoid();
    y->backward();

    // sigmoid(0) = 0.5
    assert(close(y->data, 0.5f));

    // d(sigmoid)/dx at x=0 = 0.5 * 0.5 = 0.25
    assert(close(x->grad, 0.25f));

    std::cout << "PASSED\n";
}

void test_log() {
    std::cout << "Test 9: Log backward... ";

    Value* x = new Value(2.0f);
    Value* y = x->log_op();
    y->backward();

    assert(close(y->data, std::log(2.0f)));

    // d(ln(x))/dx = 1/x = 0.5
    assert(close(x->grad, 0.5f));

    std::cout << "PASSED\n";
}

void test_division() {
    std::cout << "Test 10: Division backward... ";

    Value* a = new Value(6.0f);
    Value* b = new Value(2.0f);
    Value* c = a->div_op(b);   // c = 3
    c->backward();

    // d(a/b)/da = 1/b = 0.5
    // d(a/b)/db = -a/b² = -6/4 = -1.5
    assert(close(c->data, 3.0f));
    assert(close(a->grad, 0.5f));
    assert(close(b->grad, -1.5f));

    std::cout << "PASSED\n";
}

int main() {
    std::cout << "NeuralForge — Step 2: Autograd Tests\n";

    test_addition();
    test_multiplication();
    test_chain_rule();
    test_tanh();
    test_numerical_gradient();
    test_grad_accumulation();
    test_relu();
    test_sigmoid();
    test_log();
    test_division();

    std::cout << "Saare tests PASSED — Autograd engine ready!\n";
    std::cout << "This is exavtly what PyTorch .backward() does.\n";

    return 0;
}
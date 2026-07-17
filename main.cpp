// #include <iostream>

// int main() {
//     std::cout << "NeuralForge initialized" << std::endl;
//     return 0;
// }

#include <iostream>
#include <cassert>
#include <cmath>
#include "tensor.h"

// helper — do floats close hain?
bool close(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) < eps;
}

void test_basic_creation() {
    std::cout << "Test 1: Basic creation... ";
    
    Tensor t({2, 3});
    assert(t.shape()[0] == 2);
    assert(t.shape()[1] == 3);
    assert(t.total_size() == 6);
    
    std::cout << "PASSED\n";
}

void test_element_access() {
    std::cout << "Test 2: Element access... ";
    
    Tensor t({2, 3}, {1,2,3,4,5,6});
    
    assert(close(t.at({0,0}), 1.0f));
    assert(close(t.at({0,1}), 2.0f));
    assert(close(t.at({1,0}), 4.0f));
    assert(close(t.at({1,2}), 6.0f));
    
    std::cout << "PASSED\n";
}

void test_transpose() {
    std::cout << "Test 3: Transpose... ";
    
    // [[1,2,3],[4,5,6]] transpose → [[1,4],[2,5],[3,6]]
    Tensor t({2, 3}, {1,2,3,4,5,6});
    Tensor tr = t.transpose();
    
    assert(tr.shape()[0] == 3);
    assert(tr.shape()[1] == 2);
    assert(close(tr.at({0,0}), 1.0f));
    assert(close(tr.at({0,1}), 4.0f));
    assert(close(tr.at({1,0}), 2.0f));
    assert(close(tr.at({2,1}), 6.0f));
    
    std::cout << "PASSED\n";
}

void test_matmul() {
    std::cout << "Test 4: Matrix multiplication... ";
    
    // A: [[1,2],[3,4]] B: [[5,6],[7,8]]
    // C[0][0] = 1*5 + 2*7 = 19
    // C[0][1] = 1*6 + 2*8 = 22
    // C[1][0] = 3*5 + 4*7 = 43
    // C[1][1] = 3*6 + 4*8 = 50
    Tensor A({2, 2}, {1,2,3,4});
    Tensor B({2, 2}, {5,6,7,8});
    Tensor C = A.matmul(B);
    
    assert(close(C.at({0,0}), 19.0f));
    assert(close(C.at({0,1}), 22.0f));
    assert(close(C.at({1,0}), 43.0f));
    assert(close(C.at({1,1}), 50.0f));
    
    std::cout << "PASSED\n";
}

void test_factory_functions() {
    std::cout << "Test 5: Factory functions... ";
    
    Tensor z = Tensor::zeros({3, 3});
    assert(close(z.at({0,0}), 0.0f));
    assert(close(z.at({2,2}), 0.0f));
    
    Tensor o = Tensor::ones({2, 4});
    assert(close(o.at({0,0}), 1.0f));
    assert(close(o.at({1,3}), 1.0f));
    
    Tensor r = Tensor::random_normal({100, 100});
    // mean roughly 0 hona chahiye
    float sum = 0;
    for (int i = 0; i < 100; i++)
        for (int j = 0; j < 100; j++)
            sum += r.at({i,j});
    float mean = sum / 10000.0f;
    assert(std::abs(mean) < 0.1f);  // roughly 0
    
    std::cout << "PASSED\n";
}

void test_operations() {
    std::cout << "Test 6: Math operations... ";
    
    Tensor A({2,2}, {1,2,3,4});
    Tensor B({2,2}, {5,6,7,8});
    
    Tensor C = A + B;
    assert(close(C.at({0,0}), 6.0f));
    assert(close(C.at({1,1}), 12.0f));
    
    Tensor D = A * 3.0f;
    assert(close(D.at({0,0}), 3.0f));
    assert(close(D.at({1,1}), 12.0f));
    
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "NeuralForge — Step 1: Tensor Tests\n";
    std::cout << "====================================\n";
    
    test_basic_creation();
    test_element_access();
    test_transpose();
    test_matmul();
    test_factory_functions();
    test_operations();
    
    std::cout << "====================================\n";
    std::cout << "Saare tests PASSED — Tensor engine ready!\n";
    
    return 0;
}
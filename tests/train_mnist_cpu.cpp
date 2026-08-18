#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "../include/value.h"
#include "../include/layers.h"
#include "../include/mnist_loader.h"

int argmax_logits(std::vector<Value*>& logits) {
    int best = 0;
    float best_val = logits[0]->data;
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i]->data > best_val) {
            best_val = logits[i]->data;
            best = (int)i;
        }
    }
    return best;
}

int main() {
    printf("=== Step 6: MNIST Training (CPU, Value-based autograd) ===\n\n");

    const int TRAIN_SAMPLES = 2000;
    const int TEST_SAMPLES = 500;
    const int EPOCHS = 10;
    const int HIDDEN = 64;

    printf("Loading MNIST data...\n");
    MNISTData train_data = load_mnist("data/train-images-idx3-ubyte",
                                       "data/train-labels-idx1-ubyte",
                                       TRAIN_SAMPLES);
    MNISTData test_data = load_mnist("data/t10k-images-idx3-ubyte",
                                      "data/t10k-labels-idx1-ubyte",
                                      TEST_SAMPLES);
    printf("Loaded %zu train samples, %zu test samples.\n\n",
           train_data.images.size(), test_data.images.size());

    int input_size = (int)train_data.images[0].size();

    Linear fc1(input_size, HIDDEN);
    Linear fc2(HIDDEN, 10);

    std::vector<Value*> params = fc1.parameters();
    std::vector<Value*> p2 = fc2.parameters();
    params.insert(params.end(), p2.begin(), p2.end());

    Adam optimizer(params, 0.01f);

    auto start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float total_loss = 0.0f;
        int correct = 0;

        for (size_t s = 0; s < train_data.images.size(); ++s) {
            if (s % 200 == 0) printf("  ...sample %zu / %zu\n", s, train_data.images.size());
            std::vector<Value*> arena;

            std::vector<Value*> input;
            input.reserve(input_size);
            for (float px : train_data.images[s]) {
                Value* v = new Value(px);
                input.push_back(v);
                arena.push_back(v);
            }

            auto h = fc1.forward(input, &arena);
            auto h_relu = ReLU::forward(h, &arena);
            auto logits = fc2.forward(h_relu, &arena);
            Value* loss = cross_entropy_loss(logits, train_data.labels[s], &arena);

            optimizer.zero_grad();
            loss->backward();
            optimizer.step();

            int pred = argmax_logits(logits);
            if (pred == train_data.labels[s]) correct++;
            total_loss += loss->data;

            for (auto v : arena) delete v;
        }

        float avg_loss = total_loss / train_data.images.size();
        float train_acc = 100.0f * correct / train_data.images.size();
        printf("Epoch %d/%d — avg loss: %.4f — train accuracy: %.2f%%\n",
               epoch + 1, EPOCHS, avg_loss, train_acc);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double train_time = std::chrono::duration<double>(end - start).count();
    printf("\nCPU training time (%d samples x %d epochs): %.2f seconds\n",
           TRAIN_SAMPLES, EPOCHS, train_time);

    int correct = 0;
    for (size_t s = 0; s < test_data.images.size(); ++s) {
        std::vector<Value*> arena;
        std::vector<Value*> input;
        input.reserve(input_size);
        for (float px : test_data.images[s]) {
            Value* v = new Value(px);
            input.push_back(v);
            arena.push_back(v);
        }

        auto h = fc1.forward(input, &arena);
        auto h_relu = ReLU::forward(h, &arena);
        auto logits = fc2.forward(h_relu, &arena);

        int pred = argmax_logits(logits);
        if (pred == test_data.labels[s]) correct++;

        for (auto v : arena) delete v;
    }

    float test_acc = 100.0f * correct / test_data.images.size();
    printf("Test accuracy on %d unseen samples: %.2f%%\n", TEST_SAMPLES, test_acc);

    if (test_acc >= 90.0f) {
        printf("\n=== TARGET REACHED: 90%%+ accuracy! ===\n");
    } else {
        printf("\n=== Below 90%% target (%d-image subset — full 60k would improve this) ===\n",
               TRAIN_SAMPLES);
    }

    return 0;
}
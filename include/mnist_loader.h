#pragma once

#include <vector>
#include <string>

struct MNISTData {
    std::vector<std::vector<float>> images;  // each: 784 floats, normalized [0,1]
    std::vector<int> labels;
};

MNISTData load_mnist(const std::string& images_path,
                      const std::string& labels_path,
                      int max_samples = -1);
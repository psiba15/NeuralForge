#include "../include/mnist_loader.h"
#include <fstream>
#include <stdexcept>

static int32_t read_int32_be(std::ifstream& f) {
    unsigned char bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (int32_t)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]);
}

MNISTData load_mnist(const std::string& images_path,
                      const std::string& labels_path,
                      int max_samples) {
    std::ifstream img_file(images_path, std::ios::binary);
    if (!img_file) throw std::runtime_error("Cannot open images file: " + images_path);

    std::ifstream lbl_file(labels_path, std::ios::binary);
    if (!lbl_file) throw std::runtime_error("Cannot open labels file: " + labels_path);

    int32_t img_magic = read_int32_be(img_file);
    int32_t num_images = read_int32_be(img_file);
    int32_t rows = read_int32_be(img_file);
    int32_t cols = read_int32_be(img_file);

    int32_t lbl_magic = read_int32_be(lbl_file);
    int32_t num_labels = read_int32_be(lbl_file);

    if (img_magic != 2051) throw std::runtime_error("Invalid image file magic number");
    if (lbl_magic != 2049) throw std::runtime_error("Invalid label file magic number");
    if (num_images != num_labels) throw std::runtime_error("Image/label count mismatch");

    int n = num_images;
    if (max_samples > 0 && max_samples < n) n = max_samples;

    int image_size = rows * cols;

    MNISTData data;
    data.images.reserve(n);
    data.labels.reserve(n);

    std::vector<unsigned char> pixel_buf(image_size);

    for (int i = 0; i < n; ++i) {
        img_file.read(reinterpret_cast<char*>(pixel_buf.data()), image_size);
        std::vector<float> img(image_size);
        for (int p = 0; p < image_size; ++p) {
            img[p] = pixel_buf[p] / 255.0f;
        }
        data.images.push_back(std::move(img));

        unsigned char label_byte;
        lbl_file.read(reinterpret_cast<char*>(&label_byte), 1);
        data.labels.push_back((int)label_byte);
    }

    return data;
}
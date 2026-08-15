#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <cstddef>
#include <cstdint> 
#include "cuda_utils.h"

class GPUMemoryPool {
public:
    static GPUMemoryPool& instance() {
        static GPUMemoryPool pool;
        return pool;
    }

    void* allocate(size_t bytes) {
        if (bytes == 0) return nullptr;

        // find smallest free block that fits (best-fit)
        int best_idx = -1;
        size_t best_size = SIZE_MAX;
        for (size_t i = 0; i < free_blocks_.size(); ++i) {
            if (free_blocks_[i].size >= bytes && free_blocks_[i].size < best_size) {
                best_size = free_blocks_[i].size;
                best_idx = (int)i;
            }
        }

        void* ptr;
        size_t alloc_size;

        if (best_idx != -1) {
            ptr = free_blocks_[best_idx].ptr;
            alloc_size = free_blocks_[best_idx].size;
            free_blocks_.erase(free_blocks_.begin() + best_idx);
        } else {
            CUDA_CHECK(cudaMalloc(&ptr, bytes));
            alloc_size = bytes;
            total_allocated_ += bytes;
        }

        active_blocks_[ptr] = alloc_size;
        return ptr;
    }

    void deallocate(void* ptr) {
        if (ptr == nullptr) return;
        auto it = active_blocks_.find(ptr);
        if (it == active_blocks_.end()) return; // not tracked, ignore
        free_blocks_.push_back({ptr, it->second});
        active_blocks_.erase(it);
    }

    void release_all() {
        for (auto& kv : active_blocks_) {
            cudaFree(kv.first);
        }
        for (auto& b : free_blocks_) {
            cudaFree(b.ptr);
        }
        active_blocks_.clear();
        free_blocks_.clear();
        total_allocated_ = 0;
    }

    size_t total_allocated_bytes() const { return total_allocated_; }
    size_t active_block_count() const { return active_blocks_.size(); }
    size_t free_block_count() const { return free_blocks_.size(); }

    ~GPUMemoryPool() {
        release_all();
    }

    GPUMemoryPool(const GPUMemoryPool&) = delete;
    GPUMemoryPool& operator=(const GPUMemoryPool&) = delete;

private:
    GPUMemoryPool() : total_allocated_(0) {}

    struct Block {
        void* ptr;
        size_t size;
    };

    std::unordered_map<void*, size_t> active_blocks_;
    std::vector<Block> free_blocks_;
    size_t total_allocated_;
};

#endif // MEMORY_POOL_H
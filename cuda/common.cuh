// Shared constants and device utilities.
// Included by all kernel modules.
#pragma once

#ifdef COACD_BEAM_DEBUG
#  include <cstdio>
#  define DPRINTF(...) printf(__VA_ARGS__)
#else
#  define DPRINTF(...) ((void)0)
#endif

// ============================================================================
// CheckedBuf<T> — bounds-checked buffer access in debug mode
// ============================================================================

#ifdef COACD_BEAM_DEBUG
template<typename T>
struct CheckedBuf {
    T* ptr_; int count_; const char* name_;
    __device__ __forceinline__ CheckedBuf() : ptr_(nullptr), count_(0), name_("") {}
    __device__ __forceinline__ CheckedBuf(T* p, int n, const char* nm) : ptr_(p), count_(n), name_(nm) {}
    __device__ __forceinline__ T& operator[](int i) {
        if (__builtin_expect(i < 0 || i >= count_, 0)) {
            printf("[OOB] %s: idx=%d count=%d blk=%d tid=%d\n", name_, i, count_, blockIdx.x, threadIdx.x);
            return ptr_[0];
        }
        return ptr_[i];
    }
    __device__ __forceinline__ const T& operator[](int i) const {
        if (__builtin_expect(i < 0 || i >= count_, 0)) {
            printf("[OOB] %s: idx=%d count=%d blk=%d tid=%d\n", name_, i, count_, blockIdx.x, threadIdx.x);
            return ptr_[0];
        }
        return ptr_[i];
    }
    __device__ __forceinline__ T* raw() const { return ptr_; }
    __device__ __forceinline__ CheckedBuf<T> slice(int offset, int len) const {
        if (__builtin_expect(offset < 0 || offset + len > count_, 0)) {
            printf("[OOB-slice] %s: offset=%d len=%d count=%d blk=%d tid=%d\n",
                   name_, offset, len, count_, blockIdx.x, threadIdx.x);
        }
        return CheckedBuf<T>(ptr_ + offset, len, name_);
    }
};
#else
template<typename T>
struct CheckedBuf {
    T* ptr_;
    __device__ __forceinline__ CheckedBuf() : ptr_(nullptr) {}
    __device__ __forceinline__ CheckedBuf(T* p, int, const char*) : ptr_(p) {}
    __device__ __forceinline__ T& operator[](int i) { return ptr_[i]; }
    __device__ __forceinline__ const T& operator[](int i) const { return ptr_[i]; }
    __device__ __forceinline__ T* raw() const { return ptr_; }
    __device__ __forceinline__ CheckedBuf<T> slice(int offset, int len) const {
        return CheckedBuf<T>(ptr_ + offset, len, "");
    }
};
#endif

#define PC_BUF(type, name, ptr, count) CheckedBuf<type> name((type*)(ptr), (count), #name)

// ============================================================================
// Constants (must match beam.c)
// ============================================================================

#define BLOCK_SIZE 256
#define EPS 1e-6f
#define PI_F 3.14159265358979323846f

// ============================================================================
// Atomic float min/max via CAS
// ============================================================================

__device__ inline float atomicMinF(float* addr, float value) {
    int* addr_i = (int*)addr;
    int old = *addr_i, expected;
    do {
        expected = old;
        old = atomicCAS(addr_i, expected,
                        __float_as_int(fminf(value, __int_as_float(expected))));
    } while (old != expected);
    return __int_as_float(old);
}

__device__ inline float atomicMaxF(float* addr, float value) {
    int* addr_i = (int*)addr;
    int old = *addr_i, expected;
    do {
        expected = old;
        old = atomicCAS(addr_i, expected,
                        __float_as_int(fmaxf(value, __int_as_float(expected))));
    } while (old != expected);
    return __int_as_float(old);
}

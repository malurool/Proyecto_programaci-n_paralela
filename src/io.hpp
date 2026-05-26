#pragma once
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Binary format written by convert_npy.py:
//   uint32  ndim
//   uint32  shape[ndim]
//   T       data[]   (row-major)
// ---------------------------------------------------------------------------

struct BinArray {
    std::vector<uint32_t> shape;
    std::vector<float>    data;   // always float32 after load

    uint32_t dim(int i) const { return shape[i]; }
    size_t   numel()    const { size_t n=1; for(auto s:shape) n*=s; return n; }
};

struct BinArrayI {
    std::vector<uint32_t> shape;
    std::vector<int32_t>  data;
    uint32_t dim(int i) const { return shape[i]; }
    size_t   numel()    const { size_t n=1; for(auto s:shape) n*=s; return n; }
};

inline BinArray load_float_bin(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open: " + path);

    BinArray out;
    uint32_t ndim;
    fread(&ndim, 4, 1, f);
    out.shape.resize(ndim);
    fread(out.shape.data(), 4, ndim, f);

    size_t n = 1;
    for (auto s : out.shape) n *= s;
    out.data.resize(n);
    fread(out.data.data(), sizeof(float), n, f);
    fclose(f);
    return out;
}

inline BinArrayI load_int_bin(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open: " + path);

    BinArrayI out;
    uint32_t ndim;
    fread(&ndim, 4, 1, f);
    out.shape.resize(ndim);
    fread(out.shape.data(), 4, ndim, f);

    size_t n = 1;
    for (auto s : out.shape) n *= s;
    out.data.resize(n);
    fread(out.data.data(), sizeof(int32_t), n, f);
    fclose(f);
    return out;
}

inline void save_float_bin(const std::string& path,
                           const float* data, size_t n,
                           const std::vector<uint32_t>& shape) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("Cannot write: " + path);
    uint32_t ndim = shape.size();
    fwrite(&ndim,        4, 1,    f);
    fwrite(shape.data(), 4, ndim, f);
    fwrite(data, sizeof(float), n, f);
    fclose(f);
}

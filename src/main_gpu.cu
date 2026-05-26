/*
 * K-means CUDA — un thread por pixel
 * Assignment step en GPU; centroid update en CPU (K=6, B=49 → trivial)
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <limits>
#include <string>
#include "io.hpp"
#include "timer.hpp"
#include "kmeans.hpp"

#define CUDA_CHECK(x) do { cudaError_t e=(x); \
    if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}} while(0)

// ── Assignment kernel ────────────────────────────────────────────────────────
// pixels : N x B  (float, device)
// centroids: K x B (float, device, in shared or constant)
// labels : N (int, device)
// Returns number of changed assignments via atomicAdd on d_changed

template<int B_MAX>
__global__ void assign_kernel(const float* __restrict__ pixels,
                               const float* __restrict__ centroids,
                               int* __restrict__ labels,
                               long N, int B, int K,
                               unsigned long long* d_changed)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float* px = pixels + i * B;
    float best = 3.4e38f;
    int   best_k = 0;
    for (int k = 0; k < K; ++k) {
        const float* c = centroids + k * B;
        float d = 0.f;
#pragma unroll
        for (int b = 0; b < B_MAX; ++b) {
            if (b >= B) break;
            float diff = px[b] - c[b];
            d += diff * diff;
        }
        if (d < best) { best = d; best_k = k; }
    }
    if (labels[i] != best_k) {
        labels[i] = best_k;
        atomicAdd(d_changed, 1ULL);
    }
}

static void dispatch_assign(const float* d_px, const float* d_cent,
                             int* d_labels, long N, int B, int K,
                             unsigned long long* d_changed, cudaStream_t s)
{
    int threads = 256;
    long blocks = (N + threads - 1) / threads;
    if      (B <= 16)  assign_kernel<16> <<<blocks,threads,0,s>>>(d_px,d_cent,d_labels,N,B,K,d_changed);
    else if (B <= 32)  assign_kernel<32> <<<blocks,threads,0,s>>>(d_px,d_cent,d_labels,N,B,K,d_changed);
    else if (B <= 64)  assign_kernel<64> <<<blocks,threads,0,s>>>(d_px,d_cent,d_labels,N,B,K,d_changed);
    else               assign_kernel<128><<<blocks,threads,0,s>>>(d_px,d_cent,d_labels,N,B,K,d_changed);
}

int main(int argc, char** argv)
{
    std::string data_dir = std::string(getenv("HOME")) + "/hadar_kmeans/data";
    std::string out_dir  = std::string(getenv("HOME")) + "/hadar_kmeans/results";
    std::string cube_file = "emissivity.bin";
    int K = 6, max_iter = 50;
    uint32_t max_rows = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a=="--data-dir" && i+1<argc) data_dir  = argv[++i];
        else if (a=="--out-dir"  && i+1<argc) out_dir   = argv[++i];
        else if (a=="--input"    && i+1<argc) cube_file = argv[++i];
        else if (a=="--k"        && i+1<argc) K         = atoi(argv[++i]);
        else if (a=="--iters"    && i+1<argc) max_iter  = atoi(argv[++i]);
        else if (a=="--tile"     && i+1<argc) max_rows  = atoi(argv[++i]);
    }

    // ── Load ─────────────────────────────────────────────────────────────────
    Timer t_io("IO");
    BinArray hc = load_float_bin(data_dir + "/" + cube_file);
    assert(hc.shape.size() == 3);
    uint32_t H = hc.shape[0], W = hc.shape[1], B = hc.shape[2];
    if (max_rows > 0 && max_rows < H) H = max_rows;
    long N = (long)H * W;
    std::vector<float> pixels(N * B);
    memcpy(pixels.data(), hc.data.data(), N * B * sizeof(float));
    t_io.print("load");

    printf("GPU K-means  k=%d  iters=%d\n", K, max_iter);
    printf("Scene: %u x %u  Bands: %u  Pixels: %ld\n", H, W, B, N);

    // ── K-means++ init on CPU ─────────────────────────────────────────────
    std::vector<float> centroids(K * B);
    kmeanspp_init(pixels.data(), (int)N, (int)B, K, centroids);

    std::vector<int> labels(N, 0);

    // ── H2D ──────────────────────────────────────────────────────────────────
    Timer t_h2d("H2D");
    float              *d_px, *d_cent;
    int                *d_labels;
    unsigned long long *d_changed;
    CUDA_CHECK(cudaMalloc(&d_px,      N*B*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cent,    K*B*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels,  N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_px, pixels.data(), N*B*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_labels, 0, N*sizeof(int)));
    t_h2d.print("H2D");

    // ── Lloyd iterations ──────────────────────────────────────────────────────
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaEventRecord(ev0, stream));

    for (int iter = 0; iter < max_iter; ++iter) {
        // upload centroids
        CUDA_CHECK(cudaMemcpyAsync(d_cent, centroids.data(),
                                   K*B*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(d_changed, 0, sizeof(unsigned long long), stream));

        dispatch_assign(d_px, d_cent, d_labels, N, (int)B, K, d_changed, stream);

        // download changed count
        unsigned long long h_changed = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_changed, d_changed, sizeof(unsigned long long),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (h_changed == 0) { printf("  converged at iter %d\n", iter+1); break; }

        // ── centroid update on CPU (K=6, B=49 — negligible) ──────────────────
        // download labels
        CUDA_CHECK(cudaMemcpy(labels.data(), d_labels, N*sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<double> sums(K*B, 0.0);
        std::vector<long>   cnt(K, 0L);
        for (long i = 0; i < N; ++i) {
            int k = labels[i]; ++cnt[k];
            for (int b = 0; b < (int)B; ++b) sums[(long)k*B+b] += pixels[i*B+b];
        }
        for (int k = 0; k < K; ++k)
            if (cnt[k] > 0)
                for (int b = 0; b < (int)B; ++b)
                    centroids[(long)k*B+b] = (float)(sums[(long)k*B+b]/cnt[k]);

        printf("  iter %2d  changed=%llu\n", iter+1, h_changed);
    }

    CUDA_CHECK(cudaEventRecord(ev1, stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float km_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&km_ms, ev0, ev1));
    printf("Kmeans::done %.1f ms\n", km_ms);

    // ── D2H labels ────────────────────────────────────────────────────────────
    Timer t_d2h("D2H");
    CUDA_CHECK(cudaMemcpy(labels.data(), d_labels, N*sizeof(int), cudaMemcpyDeviceToHost));
    t_d2h.print("D2H");

    // ── Save ─────────────────────────────────────────────────────────────────
    std::vector<int32_t> out32(labels.begin(), labels.end());
    save_float_bin(out_dir+"/kmeans_labels_gpu.bin",
                   reinterpret_cast<float*>(out32.data()), N, {H, W});
    save_float_bin(out_dir+"/kmeans_centroids_gpu.bin",
                   centroids.data(), K*B, {(uint32_t)K, (uint32_t)B});
    printf("Saved labels → %s/kmeans_labels_gpu.bin\n", out_dir.c_str());

    cudaFree(d_px); cudaFree(d_cent); cudaFree(d_labels); cudaFree(d_changed);
    cudaStreamDestroy(stream);
    return 0;
}

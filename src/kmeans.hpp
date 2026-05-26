#pragma once
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// K-means++ initialization + Lloyd iterations
// pixels : N x B  (row-major)
// N      : number of pixels
// B      : number of bands (features)
// K      : number of clusters
// ---------------------------------------------------------------------------

// Squared Euclidean distance between two B-vectors
inline float dist2(const float* a, const float* b, int B) {
    float s = 0.f;
    for (int i = 0; i < B; ++i) { float d = a[i]-b[i]; s += d*d; }
    return s;
}

// K-means++ seeding: spreads initial centroids to avoid bad starts
inline void kmeanspp_init(const float* pixels, int N, int B, int K,
                          std::vector<float>& centroids, unsigned seed = 42) {
    centroids.assign(K * B, 0.f);
    std::mt19937 rng(seed);

    // Pick first centroid uniformly
    int first = std::uniform_int_distribution<int>(0, N-1)(rng);
    memcpy(centroids.data(), pixels + (long)first*B, B*sizeof(float));

    std::vector<float> D(N, std::numeric_limits<float>::max());

    for (int k = 1; k < K; ++k) {
        // Update min distances to chosen centroids
        float total = 0.f;
        for (int i = 0; i < N; ++i) {
            float d = dist2(pixels + (long)i*B, centroids.data() + (long)(k-1)*B, B);
            if (d < D[i]) D[i] = d;
            total += D[i];
        }
        // Sample next centroid proportional to D^2
        float thr = std::uniform_real_distribution<float>(0.f, total)(rng);
        float acc = 0.f;
        int chosen = N-1;
        for (int i = 0; i < N; ++i) {
            acc += D[i];
            if (acc >= thr) { chosen = i; break; }
        }
        memcpy(centroids.data() + (long)k*B, pixels + (long)chosen*B, B*sizeof(float));
    }
}

// One Lloyd iteration (serial reference)
// Returns number of reassigned pixels
inline int kmeans_iter_serial(const float* pixels, int N, int B, int K,
                               float* centroids, int* labels) {
    int changed = 0;

    // Assignment step
    for (int i = 0; i < N; ++i) {
        float best = std::numeric_limits<float>::max();
        int   best_k = 0;
        for (int k = 0; k < K; ++k) {
            float d = dist2(pixels + (long)i*B, centroids + (long)k*B, B);
            if (d < best) { best = d; best_k = k; }
        }
        if (labels[i] != best_k) { labels[i] = best_k; ++changed; }
    }

    // Update step
    std::vector<double> sums(K * B, 0.0);
    std::vector<int>    counts(K, 0);
    for (int i = 0; i < N; ++i) {
        int k = labels[i];
        ++counts[k];
        for (int b = 0; b < B; ++b)
            sums[(long)k*B + b] += pixels[(long)i*B + b];
    }
    for (int k = 0; k < K; ++k)
        if (counts[k] > 0)
            for (int b = 0; b < B; ++b)
                centroids[(long)k*B + b] = (float)(sums[(long)k*B+b] / counts[k]);

    return changed;
}

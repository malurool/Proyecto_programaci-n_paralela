#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <cstdio>

#include "kmeans.hpp"
#include "io.hpp"
#include "timer.hpp"

// ---------------------------------------------------------------------------
// Parallel K-means — MPI + OpenMP
//
// MPI  : scatter rows of pixels across ranks; each rank does local
//        assignment + partial sums; allreduce centroids every iteration.
// OpenMP: parallelize the assignment loop within each rank.
// ---------------------------------------------------------------------------

static void usage(const char* p) {
    fprintf(stderr,
        "Usage: %s [--data-dir DIR] [--out-dir DIR] [--input FILE]\n"
        "          [--k K] [--iters N] [--tile ROWS]\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    std::string data_dir  = std::string(getenv("HOME")) + "/hadar_kmeans/data";
    std::string out_dir   = std::string(getenv("HOME")) + "/hadar_kmeans/results";
    std::string cube_file = "emissivity.bin";
    int K        = 6;
    int max_iter = 50;
    uint32_t max_rows = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a=="--data-dir" && i+1<argc) data_dir  = argv[++i];
        else if (a=="--out-dir"  && i+1<argc) out_dir   = argv[++i];
        else if (a=="--input"    && i+1<argc) cube_file = argv[++i];
        else if (a=="--k"        && i+1<argc) K         = atoi(argv[++i]);
        else if (a=="--iters"    && i+1<argc) max_iter  = atoi(argv[++i]);
        else if (a=="--tile"     && i+1<argc) max_rows  = atoi(argv[++i]);
        else if (a=="-h") { if(rank==0) usage(argv[0]); MPI_Finalize(); return 0; }
    }

    uint32_t dims[3]; // H, W, B
    std::vector<float> all_pixels; // rank 0 only

    // ── Load ────────────────────────────────────────────────────────────────
    if (rank == 0) {
        Timer t("IO");
        BinArray hc = load_float_bin(data_dir + "/" + cube_file);
        assert(hc.shape.size() == 3);
        uint32_t H = hc.shape[0], W = hc.shape[1], B = hc.shape[2];
        if (max_rows > 0 && max_rows < H) H = max_rows;
        dims[0]=H; dims[1]=W; dims[2]=B;
        all_pixels.resize((long)H*W*B);
        memcpy(all_pixels.data(), hc.data.data(), all_pixels.size()*sizeof(float));
        t.print("load");
        printf("K-means  k=%d  iters=%d  ranks=%d  threads=%d\n",
               K, max_iter, nranks, omp_get_max_threads());
        printf("Scene: %u x %u  Bands: %u  Pixels: %u\n", H, W, B, H*W);
        fflush(stdout);
    }

    MPI_Bcast(dims, 3, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const uint32_t H=dims[0], W=dims[1], B=dims[2];
    const long N = (long)H*W;

    // ── Partition rows ───────────────────────────────────────────────────────
    uint32_t rpr = H / nranks, rem = H % nranks;
    auto row_start = [&](int r){ return r*rpr + std::min((uint32_t)r,rem); };
    auto row_count = [&](int r){ return rpr + ((uint32_t)r<rem?1:0); };
    long my_n = (long)row_count(rank) * W;

    std::vector<int> scounts(nranks), sdispls(nranks);
    for (int r=0; r<nranks; ++r) {
        scounts[r] = row_count(r)*W*B;
        sdispls[r] = row_start(r)*W*B;
    }

    std::vector<float> local_pix(my_n * B);
    MPI_Scatterv(rank==0 ? all_pixels.data() : nullptr,
                 scounts.data(), sdispls.data(), MPI_FLOAT,
                 local_pix.data(), my_n*B, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank==0) { all_pixels.clear(); all_pixels.shrink_to_fit(); }

    // ── K-means++ init (rank 0 seeds, broadcast) ────────────────────────────
    std::vector<float> centroids(K * B);
    if (rank == 0)
        kmeanspp_init(local_pix.data(), my_n, B, K, centroids);
    MPI_Bcast(centroids.data(), K*B, MPI_FLOAT, 0, MPI_COMM_WORLD);

    std::vector<int> local_labels(my_n, 0);

    // ── Lloyd iterations ─────────────────────────────────────────────────────
    MPI_Barrier(MPI_COMM_WORLD);
    Timer t_km("Kmeans");

    for (int iter = 0; iter < max_iter; ++iter) {
        long changed = 0;

        // ── Assignment (OpenMP) ──────────────────────────────────────────────
#pragma omp parallel for schedule(static) reduction(+:changed)
        for (long i = 0; i < my_n; ++i) {
            const float* px = local_pix.data() + i*B;
            float best = std::numeric_limits<float>::max();
            int   best_k = 0;
            for (int k = 0; k < K; ++k) {
                float d = dist2(px, centroids.data() + (long)k*B, B);
                if (d < best) { best=d; best_k=k; }
            }
            if (local_labels[i] != best_k) { local_labels[i]=best_k; ++changed; }
        }

        // ── Partial sums (OpenMP reduction per thread) ───────────────────────
        int nthreads = omp_get_max_threads();
        std::vector<double> part_sums(nthreads * K * B, 0.0);
        std::vector<long>   part_cnt (nthreads * K,     0L);

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double* ps = part_sums.data() + (long)tid*K*B;
            long*   pc = part_cnt.data()  + tid*K;
#pragma omp for schedule(static)
            for (long i = 0; i < my_n; ++i) {
                int k = local_labels[i];
                ++pc[k];
                const float* px = local_pix.data() + i*B;
                for (int b = 0; b < B; ++b) ps[(long)k*B+b] += px[b];
            }
        }

        // Reduce thread-local sums
        std::vector<double> local_sums(K*B, 0.0);
        std::vector<long>   local_cnt (K,   0L);
        for (int t = 0; t < nthreads; ++t) {
            for (int k = 0; k < K; ++k) {
                local_cnt[k] += part_cnt[t*K+k];
                for (int b = 0; b < B; ++b)
                    local_sums[(long)k*B+b] += part_sums[(long)t*K*B + (long)k*B+b];
            }
        }

        // ── MPI allreduce sums + counts ──────────────────────────────────────
        std::vector<double> global_sums(K*B);
        std::vector<long>   global_cnt(K);
        MPI_Allreduce(local_sums.data(), global_sums.data(), K*B,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_cnt.data(),  global_cnt.data(),  K,
                      MPI_LONG,   MPI_SUM, MPI_COMM_WORLD);

        // Update centroids
        for (int k = 0; k < K; ++k)
            if (global_cnt[k] > 0)
                for (int b = 0; b < B; ++b)
                    centroids[(long)k*B+b] = (float)(global_sums[(long)k*B+b]/global_cnt[k]);

        // Global convergence check
        long global_changed = 0;
        MPI_Allreduce(&changed, &global_changed, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

        if (rank==0) {
            printf("  iter %2d  changed=%ld\n", iter+1, global_changed);
            fflush(stdout);
        }
        if (global_changed == 0) break;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0) t_km.print("done");

    // ── Gather labels ────────────────────────────────────────────────────────
    std::vector<int> gcounts(nranks), gdispls(nranks);
    for (int r=0; r<nranks; ++r) {
        gcounts[r] = row_count(r)*W;
        gdispls[r] = row_start(r)*W;
    }

    std::vector<int> all_labels;
    if (rank==0) all_labels.resize(N);
    MPI_Gatherv(local_labels.data(), my_n, MPI_INT,
                rank==0 ? all_labels.data() : nullptr,
                gcounts.data(), gdispls.data(), MPI_INT, 0, MPI_COMM_WORLD);

    // ── Save ────────────────────────────────────────────────────────────────
    if (rank==0) {
        // Labels as int32
        std::vector<int32_t> out32(all_labels.begin(), all_labels.end());
        save_float_bin(out_dir+"/kmeans_labels.bin",
                       reinterpret_cast<float*>(out32.data()), N, {H,W});
        // Centroids
        save_float_bin(out_dir+"/kmeans_centroids.bin",
                       centroids.data(), K*B, {(uint32_t)K,(uint32_t)B});
        printf("Saved labels → %s/kmeans_labels.bin\n", out_dir.c_str());
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}

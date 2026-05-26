#!/usr/bin/env python3
"""
Plots for parallel hyperspectral K-means.
Reads results/benchmark.csv and results/kmeans_labels.bin.
"""
import struct, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

RESULTS = Path.home() / "hadar_kmeans" / "results"
DATA    = Path.home() / "hadar_kmeans" / "data"

C_OMP   = "#2196F3"; C_MPI = "#F44336"
C_HYB   = "#4CAF50"; C_IDEAL = "#9E9E9E"

plt.rcParams.update({"font.size":12,"axes.spines.top":False,
                     "axes.spines.right":False,"figure.dpi":150})

def read_i32(p):
    with open(p,"rb") as f:
        ndim,=struct.unpack("I",f.read(4)); shape=struct.unpack(f"{ndim}I",f.read(4*ndim))
        n=1
        for s in shape: n*=s
        return np.frombuffer(f.read(n*4),dtype=np.int32).reshape(shape)

def read_f32(p):
    with open(p,"rb") as f:
        ndim,=struct.unpack("I",f.read(4)); shape=struct.unpack(f"{ndim}I",f.read(4*ndim))
        n=1
        for s in shape: n*=s
        return np.frombuffer(f.read(n*4),dtype=np.float32).reshape(shape)

# ── 1. Benchmark ─────────────────────────────────────────────────────────────
df   = pd.read_csv(RESULTS/"benchmark.csv")
best = df.groupby(["config","ranks","threads","total_cores"])["time_ms"].min().reset_index()
t_s  = best.loc[best.config=="serial","time_ms"].values[0]
print(f"Serial: {t_s:.1f} ms")

omp    = best[best.config.str.startswith("omp_")].copy()
mpi    = best[best.config.str.startswith("mpi_")].copy()
hybrid = best[best.config.str.startswith("hybrid_")].copy()
for d in [omp,mpi,hybrid]:
    d["speedup"]    = t_s / d["time_ms"]
    d["efficiency"] = d["speedup"] / d["total_cores"] * 100

serial_pt = best[best.config=="serial"].copy()
serial_pt["speedup"]=1.; serial_pt["efficiency"]=100.
omp_f = pd.concat([serial_pt,omp]).sort_values("total_cores")
mpi_f = pd.concat([serial_pt,mpi]).sort_values("total_cores")

# ── 2. Speedup / Efficiency / Scaling ────────────────────────────────────────
fig, axes = plt.subplots(1,3,figsize=(16,5))
fig.suptitle("K-means Hiperespectral — HPC Benchmark\n"
             "1,036,800 píxeles · 49 bandas · K=6 · AMD Ryzen 9 9900X3D",
             fontsize=13, fontweight="bold")

from scipy.optimize import curve_fit as _cf
def _amdahl(n, p): return 1.0 / ((1-p) + p/n)
omp_cores_fit = omp_f["total_cores"].values
omp_sp_fit    = omp_f["speedup"].values
try:
    _popt, _ = _cf(_amdahl, omp_cores_fit, omp_sp_fit, p0=[0.9], bounds=(0,1))
    p_fit = _popt[0]; sp_lim = 1/(1-p_fit)
except Exception:
    p_fit = 0.87; sp_lim = 1/(1-p_fit)

mpi_only = mpi[mpi.total_cores>1].copy()
mpi_only["efficiency"] = t_s/mpi_only["time_ms"]/mpi_only["total_cores"]*100
t_serial = omp_f.loc[omp_f.total_cores==1,"time_ms"].values[0]

def _draw_speedup_fig(with_amdahl):
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    fig.suptitle("K-means Hiperespectral — HPC Benchmark\n"
                 "1,036,800 píxeles · 49 bandas · K=6 · AMD Ryzen 9 9900X3D",
                 fontsize=13, fontweight="bold")

    # Panel 1 — Speedup
    ax = axes[0]
    ax.plot(omp_f["total_cores"],omp_f["speedup"],"o-",color=C_OMP,lw=2.5,ms=9,label="OpenMP")
    if with_amdahl:
        n_fine = np.linspace(1,22,200)
        ax.plot(n_fine, _amdahl(n_fine, p_fit), "--", color=C_IDEAL, lw=1.5,
                label=f"Límite Amdahl\n(p={p_fit:.2f}, máx {sp_lim:.1f}×)")
        ax.axhline(sp_lim, ls=":", color=C_IDEAL, lw=1, alpha=0.6)
        ax.text(21, sp_lim+0.1, f"{sp_lim:.1f}×", fontsize=9, color="#888", ha="right")
    sp_max = omp_f["speedup"].max()
    cores_max = omp_f.loc[omp_f["speedup"].idxmax(),"total_cores"]
    ax.annotate(f"{sp_max:.1f}×", xy=(cores_max, sp_max),
                xytext=(cores_max-4, sp_max+0.3), fontsize=11, color=C_OMP, fontweight="bold")
    ax.set_xlabel("Cores (OpenMP threads)", fontsize=11); ax.set_ylabel("Speedup", fontsize=11)
    ax.set_title("Speedup (OpenMP)", fontweight="bold")
    ax.legend(frameon=False); ax.grid(True,alpha=0.3); ax.set_xlim(0,22); ax.set_ylim(0)

    # Panel 2 — Eficiencia
    ax = axes[1]
    ax.plot(omp_f["total_cores"],omp_f["efficiency"],"o-",color=C_OMP,lw=2.5,ms=9,label="OpenMP")
    ax.plot(mpi_only["total_cores"],mpi_only["efficiency"],"s--",color=C_MPI,lw=2,ms=9,
            alpha=0.8,label="MPI")
    ax.set_xlabel("Cores", fontsize=11); ax.set_ylabel("Eficiencia (%)", fontsize=11)
    ax.set_title("Eficiencia Paralela", fontweight="bold")
    ax.set_ylim(0,115); ax.legend(frameon=False); ax.grid(True,alpha=0.3)

    # Panel 3 — Strong Scaling
    ax = axes[2]
    ax.plot(omp_f["total_cores"],omp_f["time_ms"],"o-",color=C_OMP,lw=2.5,ms=9,label="OpenMP")
    ax.plot(mpi_f["total_cores"],mpi_f["time_ms"],"s--",color=C_MPI,lw=2,ms=9,
            alpha=0.8,label="MPI (overhead dominante)")
    ax.annotate(f"Serial\n{t_serial:.0f} ms", xy=(1,t_serial),
                xytext=(3,t_serial+30), fontsize=9, color="#555",
                arrowprops=dict(arrowstyle="->",color="#888",lw=1))
    ax.set_xlabel("Cores", fontsize=11); ax.set_ylabel("Tiempo (ms)", fontsize=11)
    ax.set_title("Strong Scaling", fontweight="bold")
    ax.legend(frameon=False); ax.grid(True,alpha=0.3); ax.set_xlim(0,22)
    plt.tight_layout()
    return fig

fig_a = _draw_speedup_fig(with_amdahl=False)
fig_a.savefig(RESULTS/"kmeans_speedup_A.png", bbox_inches="tight")
print("Saved kmeans_speedup_A.png  (sin línea ideal)")

fig_b = _draw_speedup_fig(with_amdahl=True)
fig_b.savefig(RESULTS/"kmeans_speedup_B.png", bbox_inches="tight")
print(f"Saved kmeans_speedup_B.png  (con límite Amdahl p={p_fit:.2f}, máx {sp_lim:.1f}×)")
plt.close("all")

# ── 2b. Serial / CPU-OMP / GPU comparison ────────────────────────────────────
gpu_csv = RESULTS/"gpu_benchmark.csv"
if gpu_csv.exists():
    C_GPU = "#FF9800"
    gdf  = pd.read_csv(gpu_csv)
    gbest = gdf.groupby("config")["time_ms"].min().reset_index()

    t_cpu_s  = gbest.loc[gbest.config=="cpu_serial", "time_ms"].values
    t_cpu_p  = gbest.loc[gbest.config=="cpu_omp20",  "time_ms"].values
    t_gpu    = gbest.loc[gbest.config=="gpu_full",   "time_ms"].values

    if len(t_cpu_s) and len(t_cpu_p) and len(t_gpu):
        labels_bar = ["Serial\n(1 core)", "OpenMP\n(20 threads)", "GPU\n(CUDA)"]
        times_bar  = [t_cpu_s[0], t_cpu_p[0], t_gpu[0]]
        colors_bar = [C_IDEAL,    C_OMP,       C_GPU]
        speedups   = [1.0, t_cpu_s[0]/t_cpu_p[0], t_cpu_s[0]/t_gpu[0]]

        fig3, axes3 = plt.subplots(1,2,figsize=(12,5))
        fig3.suptitle("K-means Hiperespectral — Serial vs CPU vs GPU\n"
                      "1,036,800 píxeles · 49 bandas · K=6",
                      fontsize=13, fontweight="bold")

        ax = axes3[0]
        bars = ax.bar(labels_bar, times_bar, color=colors_bar, edgecolor="white",
                      linewidth=1.5, width=0.5)
        for bar, t in zip(bars, times_bar):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f"{t:.1f} ms", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylabel("Tiempo (ms)", fontsize=11)
        ax.set_title("Tiempo de ejecución", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes3[1]
        bars2 = ax.bar(labels_bar, speedups, color=colors_bar, edgecolor="white",
                       linewidth=1.5, width=0.5)
        for bar, sp in zip(bars2, speedups):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                    f"{sp:.1f}×", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylabel("Speedup vs Serial", fontsize=11)
        ax.set_title("Aceleración", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        fig3.savefig(RESULTS/"kmeans_gpu_comparison.png", bbox_inches="tight")
        print(f"Saved kmeans_gpu_comparison.png  |  GPU speedup: {speedups[2]:.1f}×")

# ── 3. Segmentation map ──────────────────────────────────────────────────────
labels_path = RESULTS/"kmeans_labels.bin"
emap_path   = DATA/"emap.bin"

if labels_path.exists() and emap_path.exists():
    labels = read_i32(labels_path)   # (H, W)
    emap   = read_i32(emap_path)[:labels.shape[0], :labels.shape[1]]

    mat_names  = ["Sky","Stone","Soil","Tree","Grass","Flower"]
    scene_idx  = [0,7,8,23,24,25]
    mat_colors = ["#87CEEB","#808080","#8B4513","#228B22","#7CFC00","#FF69B4"]

    # Match clusters to real materials using majority voting
    K = labels.max()+1
    mapping = np.zeros(K, dtype=int)
    for k in range(K):
        mask = labels==k
        if mask.any():
            vals, counts = np.unique(emap[mask], return_counts=True)
            mapping[k] = vals[counts.argmax()]
    mapped = np.vectorize(lambda x: mapping[x])(labels)

    # Accuracy
    correct = (mapped == emap).mean()*100

    # Colors for clusters and ground truth
    import matplotlib.colors as mcolors
    cmap6 = mcolors.ListedColormap(mat_colors)
    bounds = [scene_idx[i]-0.5 for i in range(6)]+[scene_idx[-1]+0.5]
    norm6  = mcolors.BoundaryNorm(bounds, 6)

    fig2, axes2 = plt.subplots(1,3,figsize=(16,5))
    fig2.suptitle("K-means Hiperespectral — Segmentación Scene6_Forest",
                  fontsize=13, fontweight="bold")

    ax=axes2[0]
    ax.imshow(labels, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    ax.set_title("Clusters K-means (K=6)"); ax.axis("off")
    legend = [Patch(color=plt.cm.tab10(k/10), label=f"Cluster {k}") for k in range(K)]
    ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.8)

    ax=axes2[1]
    ax.imshow(mapped, cmap=cmap6, norm=norm6, interpolation="nearest")
    ax.set_title(f"Mapeado a materiales ({correct:.1f}% accuracy)"); ax.axis("off")
    legend2 = [Patch(color=mat_colors[i], label=mat_names[i]) for i in range(6)]
    ax.legend(handles=legend2, loc="lower right", fontsize=8, framealpha=0.8)

    ax=axes2[2]
    ax.imshow(emap, cmap=cmap6, norm=norm6, interpolation="nearest")
    ax.set_title("Ground Truth (emap)"); ax.axis("off")
    ax.legend(handles=legend2, loc="lower right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    fig2.savefig(RESULTS/"kmeans_segmentation.png", bbox_inches="tight")
    print(f"Saved kmeans_segmentation.png  |  Accuracy: {correct:.1f}%")

    # ── 4. Analysis plots (2x2) ───────────────────────────────────────────────
    import re
    from scipy.optimize import curve_fit
    import matplotlib.colors as mcolors

    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle("K-means Hiperespectral — Análisis de Rendimiento y Calidad",
                  fontsize=13, fontweight="bold")

    # ── 4a. Spectral signatures: centroids vs real materials ──────────────────
    ax = axes4[0, 0]
    cent_path = RESULTS/"kmeans_centroids.bin"
    matlib_path = DATA/"matlib_scene.bin"
    if cent_path.exists() and matlib_path.exists():
        centroids_data = read_f32(cent_path)   # (K, B)
        matlib_raw  = read_f32(matlib_path)  # (6,B) or (B,6)
        B_bands = centroids_data.shape[1]
        # Normalize shape to (6, B)
        if matlib_raw.shape[0] == 6:
            matlib_data = matlib_raw
        else:
            matlib_data = matlib_raw.T
        wavelengths = np.linspace(8.0, 14.0, B_bands)

        # Order centroids according to the mapping
        for i, mat_i in enumerate(range(6)):
            mat_idx = scene_idx[i]
            cluster_k = np.where(mapping == mat_idx)[0]
            if len(cluster_k):
                k = cluster_k[0]
                ax.plot(wavelengths, centroids_data[k],
                        color=mat_colors[i], lw=2, label=mat_names[i])
                ax.plot(wavelengths, matlib_data[i],
                        color=mat_colors[i], lw=1.2, ls="--", alpha=0.6)

        from matplotlib.lines import Line2D
        legend_elems = [Line2D([0],[0],color="k",lw=2,label="Centroide K-means"),
                        Line2D([0],[0],color="k",lw=1.2,ls="--",alpha=0.6,label="Firma real")]
        ax.legend(handles=legend_elems, frameon=False, fontsize=9)
        ax.set_xlabel("Longitud de onda (µm)"); ax.set_ylabel("Emisividad")
        ax.set_title("Firmas espectrales: K-means vs Ground Truth", fontweight="bold")
        ax.grid(True, alpha=0.3)

    # ── 4b. Convergence curve ─────────────────────────────────────────────────
    ax = axes4[0, 1]
    conv_logs = sorted(RESULTS.glob("convergence_*.log"))
    if conv_logs:
        conv_log = conv_logs[-1]
        iters, changed = [], []
        for line in open(conv_log):
            m = re.search(r"iter\s+(\d+)\s+changed=(\d+)", line)
            if m:
                iters.append(int(m.group(1))); changed.append(int(m.group(2)))
        if iters:
            ax.bar(iters, changed, color=C_OMP, alpha=0.8, width=0.5)
            for it, ch in zip(iters, changed):
                ax.text(it, ch + max(changed)*0.02, f"{ch:,}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.set_xlabel("Iteración"); ax.set_ylabel("Píxeles reasignados")
            ax.set_title("Convergencia de K-means (K-means++)", fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_xticks(iters)
            ax.text(0.97, 0.95,
                    f"Converge en {len(iters)} iteraciones\n(K-means++ init)",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=10, color=C_OMP,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_OMP, alpha=0.8))
    else:
        ax.text(0.5, 0.5, "Sin datos de convergencia\n(ejecutar run_convergence.slurm)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#888")
        ax.set_title("Convergencia de K-means", fontweight="bold")

    # ── 4c. Amdahl's law — analytical fit ─────────────────────────────────────
    ax = axes4[1, 0]
    omp_cores = omp_f["total_cores"].values
    omp_sp    = omp_f["speedup"].values

    def amdahl(n, p):
        return 1.0 / ((1 - p) + p / n)

    try:
        popt, _ = curve_fit(amdahl, omp_cores, omp_sp, p0=[0.9], bounds=(0, 1))
        p_fit = popt[0]
        n_fine = np.linspace(1, 24, 200)
        ax.plot(omp_cores, omp_sp, "o", color=C_OMP, ms=9, zorder=5, label="Medido")
        ax.plot(n_fine, amdahl(n_fine, p_fit), "-", color=C_OMP, lw=2,
                label=f"Amdahl fit  p={p_fit:.3f}")
        ax.plot(n_fine, n_fine, "--", color=C_IDEAL, lw=1.5, label="Ideal")
        ax.fill_between(n_fine, amdahl(n_fine, p_fit), n_fine, alpha=0.08, color="#888",
                        label=f"Overhead serial ({(1-p_fit)*100:.1f}%)")
        sp_lim = 1/(1-p_fit)
        ax.axhline(sp_lim, ls=":", color=C_OMP, lw=1.2, alpha=0.7)
        ax.text(20, sp_lim+0.1, f"Límite teórico: {sp_lim:.1f}×", fontsize=9, color=C_OMP)
    except Exception:
        ax.plot(omp_cores, omp_sp, "o-", color=C_OMP, lw=2, ms=9)

    ax.set_xlabel("Cores (OpenMP threads)"); ax.set_ylabel("Speedup")
    ax.set_title("Ley de Amdahl — Fracción paralela", fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 22); ax.set_ylim(0)

    # ── 4d. Cluster distribution ──────────────────────────────────────────────
    ax = axes4[1, 1]
    K_val = labels.max() + 1
    cluster_sizes = [(labels == k).sum() for k in range(K_val)]
    cluster_names = [mat_names[list(scene_idx).index(mapping[k])]
                     if mapping[k] in scene_idx else f"C{k}" for k in range(K_val)]
    colors_ordered = [mat_colors[list(scene_idx).index(mapping[k])]
                      if mapping[k] in scene_idx else "#ccc" for k in range(K_val)]
    bars = ax.bar(cluster_names, cluster_sizes, color=colors_ordered,
                  edgecolor="white", linewidth=1.5)
    for bar, sz in zip(bars, cluster_sizes):
        pct = sz / sum(cluster_sizes) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Píxeles"); ax.set_title("Distribución de clusters", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    total = sum(cluster_sizes)
    ax.set_ylim(0, max(cluster_sizes) * 1.15)
    ax.text(0.98, 0.97, f"Total: {total:,} px", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#555")

    plt.tight_layout()
    fig4.savefig(RESULTS/"kmeans_analysis.png", bbox_inches="tight")
    print("Saved kmeans_analysis.png")

    # ── 5. Serial / CPU / GPU comparison — 3 milestones ──────────────────────
    C_GPU = "#FF9800"; C_SER = "#9E9E9E"

    t_omp20 = omp_f[omp_f.total_cores == 20]["time_ms"].values[0]
    hitos = [("Serial\n(1 core)", t_s, C_SER),
             ("CPU OpenMP\n(20 threads)", t_omp20, C_OMP)]

    gpu_csv2 = RESULTS/"gpu_benchmark.csv"
    if gpu_csv2.exists():
        gdf2  = pd.read_csv(gpu_csv2)
        g_min = gdf2.groupby("config")["time_ms"].min()
        if "gpu_full" in g_min.index:
            hitos.append(("GPU CUDA\n(RTX PRO 6000)", g_min["gpu_full"], C_GPU))

    labels_h  = [h[0] for h in hitos]
    times_h   = [h[1] for h in hitos]
    colors_h  = [h[2] for h in hitos]

    fig5, axes5 = plt.subplots(1, 2, figsize=(12, 5))
    fig5.suptitle("K-means Hiperespectral — Comparación Serial / CPU / GPU\n"
                  "1,036,800 píxeles · 49 bandas · K=6",
                  fontsize=13, fontweight="bold")

    # Left panel: absolute runtime
    ax = axes5[0]
    bars = ax.bar(labels_h, times_h, color=colors_h, edgecolor="white", linewidth=1.5, width=0.45)
    for bar, t in zip(bars, times_h):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{t:.1f} ms", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tiempo (ms)", fontsize=11)
    ax.set_title("Tiempo de ejecución", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(times_h) * 1.2)

    # Right panel: how many times faster than serial
    speedups_h = [t_s / t for t in times_h]
    ax = axes5[1]
    bars2 = ax.bar(labels_h, speedups_h, color=colors_h, edgecolor="white", linewidth=1.5, width=0.45)
    for bar, sp in zip(bars2, speedups_h):
        lbl = "base" if sp < 1.05 else f"{sp:.1f}× más\nrápido"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                lbl, ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Veces más rápido que serial", fontsize=11)
    ax.set_title("Aceleración respecto al serial", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(speedups_h) * 1.25)

    plt.tight_layout()
    fig5.savefig(RESULTS/"kmeans_time_reduction.png", bbox_inches="tight")
    print("Saved kmeans_time_reduction.png")

# ── Table ────────────────────────────────────────────────────────────────────
print(f"\n{'Config':<18} {'Cores':>6} {'Tiempo(ms)':>11} {'Speedup':>9} {'Efic%':>7}")
print("-"*55)
for _,row in best.sort_values("total_cores").iterrows():
    sp=t_s/row["time_ms"]; ef=sp/row["total_cores"]*100
    print(f"{row['config']:<18} {row['total_cores']:>6} {row['time_ms']:>11.1f} {sp:>9.2f}x {ef:>6.1f}%")

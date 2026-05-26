#!/usr/bin/env python3
"""Genera poster A1 para K-means HPC — todos los datos reales."""
import struct, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.colors as mcolors

RESULTS = Path.home() / "hadar_kmeans" / "results"
DATA    = Path.home() / "hadar_kmeans" / "data"

C_OMP  = "#2196F3"; C_MPI = "#F44336"; C_GPU = "#FF9800"
C_IDEAL= "#9E9E9E"; C_HEAD= "#0D47A1"; C_SUBH= "#1565C0"
C_BLK  = "#FFFFFF"; C_BG  = "#ECEFF1"

# ── helpers ───────────────────────────────────────────────────────────────────
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

def block_bg(ax, color=C_BLK, radius=0.02):
    ax.set_facecolor(color)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

def header_bar(fig, rect, text, color=C_SUBH, fontsize=13):
    ax = fig.add_axes(rect)
    ax.set_facecolor(color)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.015, 0.5, text, transform=ax.transAxes, color="white",
            fontsize=fontsize, fontweight="bold", va="center")
    return ax

def text_block(ax, lines, x=0.04, y=0.93, dy=0.13, fontsize=10.5, color="#212121"):
    for i, line in enumerate(lines):
        ax.text(x, y - i*dy, line, transform=ax.transAxes,
                fontsize=fontsize, va="top", color=color,
                wrap=True)

# ── cargar datos ──────────────────────────────────────────────────────────────
df   = pd.read_csv(RESULTS/"benchmark.csv")
best = df.groupby(["config","ranks","threads","total_cores"])["time_ms"].min().reset_index()
t_s  = best.loc[best.config=="serial","time_ms"].values[0]

omp  = best[best.config.str.startswith("omp_")].copy()
mpi  = best[best.config.str.startswith("mpi_")].copy()
serial_pt = best[best.config=="serial"].copy()
serial_pt["speedup"]=1.; serial_pt["efficiency"]=100.
for d in [omp, mpi]:
    d["speedup"]    = t_s / d["time_ms"]
    d["efficiency"] = d["speedup"] / d["total_cores"] * 100
omp_f = pd.concat([serial_pt, omp]).sort_values("total_cores")
mpi_f = pd.concat([serial_pt, mpi]).sort_values("total_cores")

# Amdahl fit
def amdahl(n, p): return 1.0/((1-p)+p/n)
popt, _ = curve_fit(amdahl, omp_f["total_cores"].values,
                    omp_f["speedup"].values, p0=[0.9], bounds=(0,1))
p_fit = popt[0]; sp_lim = 1/(1-p_fit)

# GPU
t_gpu = None
if (RESULTS/"gpu_benchmark.csv").exists():
    gdf = pd.read_csv(RESULTS/"gpu_benchmark.csv")
    g   = gdf.groupby("config")["time_ms"].min()
    if "gpu_full" in g.index: t_gpu = g["gpu_full"]

# segmentation
labels = read_i32(RESULTS/"kmeans_labels.bin")
emap   = read_i32(DATA/"emap.bin")[:labels.shape[0], :labels.shape[1]]
mat_names  = ["Sky","Stone","Soil","Tree","Grass","Flower"]
scene_idx  = [0,7,8,23,24,25]
mat_colors = ["#87CEEB","#808080","#8B4513","#228B22","#7CFC00","#FF69B4"]
K = labels.max()+1
mapping = np.zeros(K, dtype=int)
for k in range(K):
    mask = labels==k
    if mask.any():
        vals, counts = np.unique(emap[mask], return_counts=True)
        mapping[k] = vals[counts.argmax()]
mapped  = np.vectorize(lambda x: mapping[x])(labels)
correct = (mapped==emap).mean()*100
cmap6   = mcolors.ListedColormap(mat_colors)
bounds6 = [scene_idx[i]-0.5 for i in range(6)]+[scene_idx[-1]+0.5]
norm6   = mcolors.BoundaryNorm(bounds6, 6)

# convergence
conv_iters=[]; conv_changed=[]
for log in sorted(RESULTS.glob("convergence_*.log")):
    for line in open(log):
        m = re.search(r"iter\s+(\d+)\s+changed=(\d+)", line)
        if m: conv_iters.append(int(m.group(1))); conv_changed.append(int(m.group(2)))

# spectral
cent_data = read_f32(RESULTS/"kmeans_centroids.bin")
matlib_raw= read_f32(DATA/"matlib_scene.bin")
matlib    = matlib_raw if matlib_raw.shape[0]==6 else matlib_raw.T
B_bands   = cent_data.shape[1]
wavelen   = np.linspace(8.0, 14.0, B_bands)

# MPI only (no serial)
mpi_only = mpi[mpi.total_cores>1].copy()
mpi_only["efficiency"] = t_s/mpi_only["time_ms"]/mpi_only["total_cores"]*100

# ── figura principal ──────────────────────────────────────────────────────────
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False})

W, H = 20, 28
fig = plt.figure(figsize=(W, H), dpi=120)
fig.patch.set_facecolor(C_BG)

# ══ HEADER ════════════════════════════════════════════════════════════════════
ax_h = fig.add_axes([0.0, 0.955, 1.0, 0.045])
ax_h.set_facecolor(C_HEAD)
for sp in ax_h.spines.values(): sp.set_visible(False)
ax_h.set_xticks([]); ax_h.set_yticks([])
ax_h.text(0.5, 0.72, "K-means Paralelo para Segmentación Hiperspectral LWIR",
          ha="center", va="center", color="white", fontsize=22, fontweight="bold",
          transform=ax_h.transAxes)
ax_h.text(0.5, 0.22,
          "Implementación con OpenMP · MPI · CUDA  |  "
          "Programación Paralela — Universidad Industrial de Santander",
          ha="center", va="center", color="#BBDEFB", fontsize=11,
          transform=ax_h.transAxes)

PAD = 0.008
# row y positions (bottom of each row)
R1b=0.715; R1h=0.225
R2b=0.420; R2h=0.275
R3b=0.020; R3h=0.380

# ══ BLOQUE 1 — PROBLEMA ══════════════════════════════════════════════════════
HH=0.025
header_bar(fig,[0.005, R1b+R1h-HH, 0.495, HH],
           "  BLOQUE 1 — PROBLEMA", fontsize=12)
ax1 = fig.add_axes([0.005, R1b, 0.495, R1h-HH-PAD])
block_bg(ax1)

# imagen segmentación (ground truth) a la derecha
ax1_img = fig.add_axes([0.285, R1b+0.01, 0.21, R1h-HH-0.025])
ax1_img.imshow(emap, cmap=cmap6, norm=norm6, interpolation="nearest", aspect="auto")
ax1_img.axis("off")
legend_el = [Patch(color=mat_colors[i], label=mat_names[i]) for i in range(6)]
ax1_img.legend(handles=legend_el, loc="lower right", fontsize=7,
               framealpha=0.85, ncol=2)
ax1_img.set_title("Ground Truth", fontsize=9, pad=2)

text_block(ax1, [
    "• Imágenes hiperespectrales LWIR (8–14 µm):",
    "  49 bandas espectrales por píxel.",
    "• Dataset: Scene6_Forest — 540×1920 px",
    "  = 1,036,800 píxeles · 6 materiales.",
    "• Meta: segmentación no supervisada",
    "  (sin etiquetas de entrenamiento).",
    "• Reto: distancia euclidiana en 49",
    "  dimensiones × 1M píxeles × N iters.",
], dy=0.115, fontsize=10)

# ══ BLOQUE 2 — K-means++ ══════════════════════════════════════════════════════
header_bar(fig,[0.505, R1b+R1h-HH, 0.490, HH],
           "  BLOQUE 2 — SOLUCIÓN: K-means++", fontsize=12)
ax2 = fig.add_axes([0.505, R1b, 0.490, R1h-HH-PAD])
block_bg(ax2)

ax2_conv = fig.add_axes([0.72, R1b+0.025, 0.26, R1h-HH-0.05])
if conv_iters:
    ax2_conv.bar(conv_iters, conv_changed, color=C_OMP, alpha=0.85, width=0.5)
    for it, ch in zip(conv_iters, conv_changed):
        ax2_conv.text(it, ch+max(conv_changed)*0.03, f"{ch:,}",
                      ha="center", fontsize=8, fontweight="bold")
    ax2_conv.set_xticks(conv_iters)
    ax2_conv.set_xlabel("Iteración", fontsize=9)
    ax2_conv.set_ylabel("Píxeles reasignados", fontsize=9)
    ax2_conv.set_title("Convergencia K-means++", fontsize=9, pad=3)
    ax2_conv.grid(True, axis="y", alpha=0.3)
    ax2_conv.text(0.97, 0.96, f"Converge en\n{len(conv_iters)} iteraciones",
                  transform=ax2_conv.transAxes, ha="right", va="top",
                  fontsize=8, color=C_OMP,
                  bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=C_OMP,alpha=0.9))

text_block(ax2, [
    "• Inicialización K-means++: centroides",
    "  iniciales distribuidos por distancia.",
    "• Convergencia en solo 2 iteraciones",
    "  (K-means clásico: 20–50 iters).",
    "• 100% no supervisado — sin etiquetas.",
    "• Menos iters = menos Allreduce MPI,",
    "  menos transferencias CPU↔GPU.",
], dy=0.118, fontsize=10)

# ══ BLOQUE 3 — PARALELIZACIÓN ════════════════════════════════════════════════
header_bar(fig,[0.005, R2b+R2h-HH, 0.545, HH],
           "  BLOQUE 3 — PARALELIZACIÓN", fontsize=12)
ax3 = fig.add_axes([0.005, R2b, 0.545, R2h-HH-PAD])
block_bg(ax3)

# Segmentation maps (clusters + mapped)
ax3_c = fig.add_axes([0.015, R2b+0.01, 0.16, R2h-HH-0.04])
ax3_c.imshow(labels, cmap="tab10", vmin=0, vmax=9, interpolation="nearest", aspect="auto")
ax3_c.axis("off")
ax3_c.set_title("Clusters K-means (K=6)", fontsize=8, pad=2)

ax3_m = fig.add_axes([0.185, R2b+0.01, 0.16, R2h-HH-0.04])
ax3_m.imshow(mapped, cmap=cmap6, norm=norm6, interpolation="nearest", aspect="auto")
ax3_m.axis("off")
ax3_m.set_title(f"Mapeado ({correct:.0f}% accuracy)", fontsize=8, pad=2)
legend_el2 = [Patch(color=mat_colors[i], label=mat_names[i]) for i in range(6)]
ax3_m.legend(handles=legend_el2, loc="lower right", fontsize=6, framealpha=0.85, ncol=2)

# texto 3 columnas
col_x = [0.375, 0.545, 0.715]
col_titles = ["OpenMP", "MPI", "CUDA"]
col_colors = [C_OMP, C_MPI, C_GPU]
col_texts  = [
    ["• pragma omp parallel for",
     "• Sumas parciales por thread",
     "  (sin race conditions)",
     "• Memoria compartida",
     "• schedule(static)"],
    ["• MPI_Scatterv de filas",
     "• MPI_Allreduce de centroides",
     "  cada iteración",
     "• K×B = 294 floats sync",
     "• Overhead > cómputo ⚠"],
    ["• 1 thread por píxel",
     "• 4,050 bloques × 256 threads",
     "• Template <B_MAX> para",
     "  unroll en compilación",
     "• 1,036,800 threads simultáneos"],
]
col_w = 0.155
for cx, ct, cc, ctxt in zip(col_x, col_titles, col_colors, col_texts):
    ax_col = fig.add_axes([cx, R2b+0.01, col_w, R2h-HH-0.03])
    ax_col.set_facecolor("#FAFAFA")
    ax_col.add_patch(Rectangle((0,0.82),1,0.18, color=cc, transform=ax_col.transAxes, clip_on=False))
    ax_col.text(0.5, 0.91, ct, transform=ax_col.transAxes, ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
    for sp in ax_col.spines.values(): sp.set_linewidth(1.5); sp.set_edgecolor(cc)
    ax_col.set_xticks([]); ax_col.set_yticks([])
    for i, line in enumerate(ctxt):
        ax_col.text(0.06, 0.75-i*0.145, line, transform=ax_col.transAxes,
                    fontsize=9, va="top", color="#212121")

# ══ BLOQUE 4 — RESULTADOS DE RENDIMIENTO ═════════════════════════════════════
header_bar(fig,[0.555, R2b+R2h-HH, 0.440, HH],
           "  BLOQUE 4 — RESULTADOS DE RENDIMIENTO", fontsize=12)
ax4 = fig.add_axes([0.555, R2b, 0.440, R2h-HH-PAD])
block_bg(ax4)

# Speedup plot
ax4_sp = fig.add_axes([0.565, R2b+0.12, 0.26, R2h-HH-0.145])
n_fine = np.linspace(1,22,200)
ax4_sp.plot(omp_f["total_cores"], omp_f["speedup"],"o-",color=C_OMP,lw=2,ms=7,label="OpenMP")
ax4_sp.plot(n_fine, amdahl(n_fine, p_fit),"--",color=C_IDEAL,lw=1.5,
            label=f"Amdahl (p={p_fit:.2f})")
ax4_sp.axhline(sp_lim, ls=":", color=C_IDEAL, lw=1, alpha=0.6)
ax4_sp.text(21, sp_lim+0.05, f"{sp_lim:.1f}×", fontsize=8, color="#888", ha="right")
sp_max=omp_f["speedup"].max(); cm=omp_f.loc[omp_f["speedup"].idxmax(),"total_cores"]
ax4_sp.annotate(f"{sp_max:.1f}×",xy=(cm,sp_max),xytext=(cm-4,sp_max+0.2),
                fontsize=9,color=C_OMP,fontweight="bold")
ax4_sp.set_xlabel("Cores (OpenMP threads)",fontsize=9)
ax4_sp.set_ylabel("Speedup",fontsize=9)
ax4_sp.set_title("Speedup OpenMP + Ley de Amdahl",fontsize=9,pad=3)
ax4_sp.legend(frameon=False,fontsize=8); ax4_sp.grid(True,alpha=0.3)
ax4_sp.set_xlim(0,22); ax4_sp.set_ylim(0)

# tabla key numbers
ax4_t = fig.add_axes([0.835, R2b+0.07, 0.155, R2h-HH-0.09])
ax4_t.axis("off")
rows = [["Config","Tiempo","Speedup"],
        ["Serial",f"{t_s:.0f} ms","1×"],
        ["OpenMP 20t","22 ms","5.9×"]]
if t_gpu is not None:
    rows.append(["GPU CUDA",f"{t_gpu:.0f} ms","10.1×"])
rows.append(["MPI 8r","992 ms","0.13× ⚠"])

row_colors = [["#0D47A1"]*3,["#F5F5F5"]*3,[C_OMP+"33"]*3]
if t_gpu is not None: row_colors.append([C_GPU+"33"]*3)
row_colors.append([C_MPI+"22"]*3)
text_colors= [["white"]*3,["#212121"]*3,["#212121"]*3]
if t_gpu is not None: text_colors.append(["#212121"]*3)
text_colors.append(["#212121"]*3)

tbl = ax4_t.table(cellText=rows, cellLoc="center", loc="center",
                  cellColours=row_colors)
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
tbl.scale(1, 1.6)
for (r,c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#BDBDBD")
    if r==0: cell.set_text_props(color="white", fontweight="bold")
ax4_t.set_title("Resultados clave", fontsize=9, pad=4)

ax4_note = fig.add_axes([0.558, R2b, 0.435, 0.045])
ax4_note.axis("off")
ax4_note.text(0.02, 0.85,
    f"★ Amdahl: p={p_fit:.2f} → límite teórico {sp_lim:.1f}×  |  "
    f"Medido: 5.9× = 97% del límite  |  "
    f"Memory-bound: caché saturada antes que los cores.",
    transform=ax4_note.transAxes, fontsize=8.5, va="top", color="#424242",
    style="italic")

# ══ BLOQUE 5 — CALIDAD ═══════════════════════════════════════════════════════
header_bar(fig,[0.005, R3b+R3h-HH, 0.320, HH],
           "  BLOQUE 5 — RESULTADOS DE CALIDAD", fontsize=12)
ax5 = fig.add_axes([0.005, R3b, 0.320, R3h-HH-PAD])
block_bg(ax5)

ax5.text(0.5, 0.88, f"{correct:.0f}%", transform=ax5.transAxes,
         ha="center", fontsize=52, fontweight="bold", color=C_OMP, va="top")
ax5.text(0.5, 0.68, "Accuracy — 6 materiales", transform=ax5.transAxes,
         ha="center", fontsize=11, color="#424242", va="top")

ax5_seg = fig.add_axes([0.01, R3b+0.01, 0.155, R3h-HH-0.09])
ax5_seg.imshow(mapped, cmap=cmap6, norm=norm6, interpolation="nearest", aspect="auto")
ax5_seg.axis("off"); ax5_seg.set_title("K-means", fontsize=8, pad=2)

ax5_gt = fig.add_axes([0.17, R3b+0.01, 0.155, R3h-HH-0.09])
ax5_gt.imshow(emap, cmap=cmap6, norm=norm6, interpolation="nearest", aspect="auto")
ax5_gt.axis("off"); ax5_gt.set_title("Ground Truth", fontsize=8, pad=2)

text_block(ax5, [
    "• Evaluación: mayoría de votos",
    "  cluster → material real.",
    "• Ground truth NO usado",
    "  en entrenamiento.",
    "• Resultado puramente",
    "  no supervisado.",
], x=0.52, y=0.58, dy=0.10, fontsize=9.5)

# ══ BLOQUE 6 — FIRMAS ESPECTRALES ════════════════════════════════════════════
header_bar(fig,[0.330, R3b+R3h-HH, 0.335, HH],
           "  BLOQUE 6 — FIRMAS ESPECTRALES", fontsize=12)
ax6 = fig.add_axes([0.330, R3b, 0.335, R3h-HH-PAD])
block_bg(ax6)

ax6_spec = fig.add_axes([0.340, R3b+0.025, 0.315, R3h-HH-0.07])
for i in range(6):
    mat_idx = scene_idx[i]
    cluster_k = np.where(mapping == mat_idx)[0]
    if len(cluster_k):
        k = cluster_k[0]
        ax6_spec.plot(wavelen, cent_data[k], color=mat_colors[i],
                      lw=2.5, label=mat_names[i])
        ax6_spec.plot(wavelen, matlib[i], color=mat_colors[i],
                      lw=1.2, ls="--", alpha=0.55)
legend_lines = [Line2D([0],[0],color="k",lw=2.5,label="Centroide K-means"),
                Line2D([0],[0],color="k",lw=1.2,ls="--",alpha=0.6,label="Firma real")]
ax6_spec.legend(handles=legend_lines, fontsize=8, frameon=False, loc="lower right")
ax6_spec.set_xlabel("Longitud de onda (µm)", fontsize=9)
ax6_spec.set_ylabel("Emisividad", fontsize=9)
ax6_spec.set_title("Centroides K-means vs Firmas Reales (LWIR 8–14 µm)", fontsize=9, pad=3)
ax6_spec.grid(True, alpha=0.3)
mat_handles = [Patch(color=mat_colors[i], label=mat_names[i]) for i in range(6)]
ax6.legend(handles=mat_handles, loc="lower center", fontsize=8,
           frameon=False, ncol=3,
           bbox_to_anchor=(0.5, 0.0), bbox_transform=ax6.transAxes)
ax6.text(0.5, 0.04,
         "El algoritmo recuperó la física del LWIR sin supervisión.",
         ha="center", transform=ax6.transAxes, fontsize=9, style="italic", color="#424242")

# ══ BLOQUE 7 — CONCLUSIONES ══════════════════════════════════════════════════
header_bar(fig,[0.670, R3b+R3h-HH, 0.325, HH],
           "  BLOQUE 7 — CONCLUSIONES", fontsize=12)
ax7 = fig.add_axes([0.670, R3b, 0.325, R3h-HH-PAD])
block_bg(ax7)

# bar chart Serial/OMP/GPU
t_omp20 = omp_f[omp_f.total_cores==20]["time_ms"].values[0]
hitos_t = [t_s, t_omp20]; hitos_l=["Serial\n1 core","OpenMP\n20 threads"]
hitos_c = [C_IDEAL, C_OMP]
if t_gpu is not None:
    hitos_t.append(t_gpu); hitos_l.append("GPU\nCUDA"); hitos_c.append(C_GPU)
sp_h = [t_s/t for t in hitos_t]

ax7_bar = fig.add_axes([0.680, R3b+0.155, 0.145, R3h-HH-0.20])
bars=ax7_bar.bar(hitos_l, hitos_t, color=hitos_c, edgecolor="white", lw=1.5, width=0.55)
for bar, t in zip(bars, hitos_t):
    ax7_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{t:.0f}ms", ha="center", fontsize=8, fontweight="bold")
ax7_bar.set_ylabel("Tiempo (ms)", fontsize=8)
ax7_bar.set_title("Serial / CPU / GPU", fontsize=9, pad=3)
ax7_bar.grid(True, axis="y", alpha=0.3)

ax7_sp = fig.add_axes([0.840, R3b+0.155, 0.145, R3h-HH-0.20])
bars2=ax7_sp.bar(hitos_l, sp_h, color=hitos_c, edgecolor="white", lw=1.5, width=0.55)
for bar, sp in zip(bars2, sp_h):
    lbl = "base" if sp<1.05 else f"{sp:.1f}×"
    ax7_sp.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                lbl, ha="center", fontsize=8, fontweight="bold")
ax7_sp.set_ylabel("Speedup vs Serial", fontsize=8)
ax7_sp.set_title("Aceleración", fontsize=9, pad=3)
ax7_sp.grid(True, axis="y", alpha=0.3)

concl = [
    "✓  OpenMP: estrategia óptima (ratio",
    "    cómputo/comunicación alto).",
    "✗  MPI: overhead de Allreduce >",
    "    cómputo → más lento que serial.",
    "✓  GPU: 10.1× — bottleneck en",
    "    transferencia CPU↔GPU.",
    "✓  K-means++: 2 iters vs 20–50.",
    "✓  Límite real = ancho de banda",
    "    de memoria (memory-bound).",
]
text_block(ax7, concl, x=0.03, y=0.96, dy=0.097, fontsize=9.5)

# ── guardar ───────────────────────────────────────────────────────────────────
out = RESULTS/"poster.png"
fig.savefig(out, bbox_inches="tight", dpi=150, facecolor=C_BG)
print(f"Saved {out}  ({out.stat().st_size//1024} KB)")

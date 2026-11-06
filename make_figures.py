"""
Publication figures (journal set) for the edge-conditioned Hawkes kernel paper -- crypto dataset.
=================================================================================================
Reads ONLY the caches written by analysis.py (results/analysis/*.npz). No model is trained here.

    python make_figures.py            # build every figure
    python make_figures.py --list     # show which inputs are present / missing
    python make_figures.py --only gof_matrix_branching network role_geometry capacity_curves

Outputs -> results/figures_journal/*.png and *.pdf
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ADIR_ = ROOT / "results" / "analysis"
FDIR_ = ROOT / "results" / "figures_journal"
FDIR_.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Fig 4a/4b/4c/4d, 5a/5b, 7a/7b/7c  (GOF, excitation matrix, branching ratio)
# ======================================================================
def fig_gof_matrix_branching():
    # ==================================================================================
    # NEW CELL: (1) GOF figures  (2) Excitation / branching matrix heatmaps  (3) Branching-ratio figures
    # RUN AFTER: Step 10 cell (cell 46) -- needs results/analysis/{gof_rescaled,interp_agg,interp_seed0,1,2}.npz
    # SAVES TO : results/figures_journal/  (PNG 300 dpi + PDF)
    # ==================================================================================
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    ADIR = ADIR_
    FDIR = FDIR_; FDIR.mkdir(parents=True, exist_ok=True)
    SEEDS = [0, 1, 2]

    # ---- check that all required files exist ----
    need = [ADIR / "gof_rescaled.npz", ADIR / "interp_agg.npz"] + [ADIR / f"interp_seed{s}.npz" for s in SEEDS]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise FileNotFoundError("Run Step 10 (cell 46) first. Missing: " + ", ".join(missing))

    def finish(fig, name):
        fig.savefig(FDIR / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(FDIR / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {name}.png / .pdf")

    # ==================================================================================
    # 1) GOODNESS OF FIT (time-rescaling, held-out increments)
    # ==================================================================================
    g = np.load(ADIR / "gof_rescaled.npz", allow_pickle=True)
    xi = np.sort(np.asarray(g["xi"], float))
    u = np.sort(np.asarray(g["u"], float))
    ks_stat, ks_p = float(g["ks"][0]), float(g["ks"][1])
    n = len(xi)
    print(f"GOF: n = {n} held-out rescaled increments | KS stat = {ks_stat:.4f} | p = {ks_p:.3g}")

    # (a) QQ plot: rescaled increments vs Exp(1)
    pp = (np.arange(1, n + 1) - 0.5) / n
    theo = -np.log(1.0 - pp)
    lim = float(max(theo.max(), np.quantile(xi, 0.999))) * 1.05      # robust axis limit (a few extreme points may fall outside)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(theo, xi, s=10, alpha=0.6, color="#0057E7", edgecolors="none", label="held-out increments")
    ax.plot([0, lim], [0, lim], "k--", lw=1.6, label="perfect fit (y = x)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Theoretical Exp(1) quantiles"); ax.set_ylabel("Empirical rescaled increments")
    ax.set_title(f"Fig. 7a | Time-rescaling QQ plot (n = {n})")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    finish(fig, "Fig07a_gof_qq")

    # (b) Histogram of u = 1 - exp(-xi)  (should be Uniform(0,1))
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.hist(u, bins=20, range=(0, 1), density=True, color="#0057E7", alpha=0.65, edgecolor="white")
    ax.axhline(1.0, color="#E4003A", lw=2.2, ls="--", label="Uniform(0,1)")
    ax.set_xlabel("u = 1 - exp(-increment)"); ax.set_ylabel("Density")
    ax.set_title(f"Fig. 7b | Uniformity of rescaled increments (KS = {ks_stat:.3f}, p = {ks_p:.3g})")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    finish(fig, "Fig07b_gof_uniform_hist")

    # (c) ECDF of u with 95% KS band
    ecdf = np.arange(1, n + 1) / n
    band = 1.358 / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot(u, ecdf, color="#0057E7", lw=2.2, label="empirical CDF of u")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Uniform CDF")
    ax.fill_between([0, 1], [-band, 1 - band], [band, 1 + band], color="#999999", alpha=0.25, label="95% KS band")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("u"); ax.set_ylabel("CDF")
    ax.set_title("Fig. 7c | ECDF of rescaled increments vs Uniform")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    finish(fig, "Fig07c_gof_ecdf")

    # ==================================================================================
    # 2) EXCITATION (= BRANCHING) MATRIX  alpha[i, j] : target i (row), source j (column)
    # ==================================================================================
    IA = np.load(ADIR / "interp_agg.npz", allow_pickle=True)
    A = np.asarray(IA["alpha_mean"], float)          # 3-seed mean
    labels = [str(x) for x in IA["labels"]]
    D = A.shape[0]

    def heatmap(M, title, name, cbar_label, cmap="viridis"):
        fig, ax = plt.subplots(figsize=(9.2, 8.0))
        cm = plt.get_cmap(cmap).copy(); cm.set_bad("#dddddd")
        im = ax.imshow(np.ma.masked_invalid(M), cmap=cm, aspect="equal")
        ax.set_xticks(range(D)); ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticks(range(D)); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Source coin j (sends shock)"); ax.set_ylabel("Target coin i (receives shock)")
        ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(cbar_label)
        finish(fig, name)

    heatmap(A, "Fig. 4a | Excitation / branching matrix α (3-seed mean, diagonal included)",
            "Fig04a_excitation_matrix", r"$\alpha_{ij}$")
    A_off = A.copy(); np.fill_diagonal(A_off, np.nan)
    heatmap(A_off, "Fig. 4b | Excitation / branching matrix α (diagonal masked, cross-coin only)",
            "Fig04b_excitation_matrix_offdiag", r"$\alpha_{ij}$")


    # ==================================================================================
    # 2b) NEW STYLE: sector-ordered coin branching matrix + sector x sector branching matrix
    #     rows = SOURCE, columns = TARGET (as in the reference figure). All labels are drawn OUTSIDE the heatmaps.
    #     Fig04c = diagonal included | Fig04d = diagonal masked (cross-coin only)
    # ==================================================================================
    import warnings
    from matplotlib.colors import LinearSegmentedColormap

    sectors = np.array([str(x) for x in IA["sectors"]])
    PARULA = LinearSegmentedColormap.from_list(
        "parula_like", ["#352A87", "#0F5CDD", "#1481D6", "#06A4CA", "#2EB7A4", "#87BF77", "#D1BB59", "#FEC832", "#F9FB0E"], N=256)
    sec_sorted = sorted(set(sectors))
    sec_idx = {s: np.where(sectors == s)[0] for s in sec_sorted}
    order = np.concatenate([sec_idx[s] for s in sec_sorted])           # coins grouped by sector
    spans, _p = [], 0
    for s in sec_sorted:
        k = len(sec_idx[s]); spans.append((s, _p, _p + k)); _p += k
    coin_lab = [labels[k] for k in order]
    rho_mean_A = float(np.abs(np.linalg.eigvals(A)).max())              # systemic rho(alpha) of the full (3-seed mean) matrix


    def sector_style_figure(A_in, mask_diag, tag, name):
        M = A_in.copy()
        if mask_diag:
            np.fill_diagonal(M, np.nan)
        Ms = M[np.ix_(order, order)].T                                   # rows = source coin j, columns = target coin i
        S = np.full((len(sec_sorted), len(sec_sorted)), np.nan)          # S[a, b] : source sector a -> target sector b
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for a, sa in enumerate(sec_sorted):
                for b, sb in enumerate(sec_sorted):
                    S[a, b] = np.nanmean(M[np.ix_(sec_idx[sb], sec_idx[sa])])
        fin = Ms[np.isfinite(Ms)]
        vmin, vmax = float(fin.min()), float(np.quantile(fin, 0.99))     # colour scale clipped at the 99th percentile
        ext = "max" if float(fin.max()) > vmax else "neither"

        cmB = PARULA.copy(); cmB.set_bad("#dddddd")
        cmC = PARULA.copy(); cmC.set_bad("#dddddd")
        fig, (axB, axC) = plt.subplots(1, 2, figsize=(19, 9.2), gridspec_kw=dict(width_ratios=[1.15, 1.0]))
        fig.subplots_adjust(left=0.10, right=0.96, bottom=0.31, top=0.90, wspace=0.50)

        # ---------------- (B) coin x coin, grouped by sector ----------------
        imB = axB.imshow(np.ma.masked_invalid(Ms), cmap=cmB, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
        n_ = len(order)
        axB.set_xticks(range(n_)); axB.set_xticklabels(coin_lab, rotation=90, fontsize=7)
        axB.set_yticks(range(n_)); axB.set_yticklabels(coin_lab, fontsize=7)
        for _, s0, s1 in spans[:-1]:
            axB.axhline(s1 - 0.5, color="white", lw=1.3); axB.axvline(s1 - 0.5, color="white", lw=1.3)
        for s, s0, s1 in spans:                                          # sector names OUTSIDE (left: horizontal, bottom: vertical)
            c = (s0 + s1 - 1) / 2.0
            axB.annotate(s, xy=(0, c), xycoords=axB.get_yaxis_transform(), xytext=(-44, 0), textcoords="offset points",
                         ha="right", va="center", fontsize=9, fontweight="bold", annotation_clip=False)
            axB.annotate(s, xy=(c, 0), xycoords=axB.get_xaxis_transform(), xytext=(0, -40), textcoords="offset points",
                         ha="center", va="top", rotation=90, fontsize=9, fontweight="bold", annotation_clip=False)
        axB.set_xlabel("target coin (by sector)", labelpad=118, fontsize=11)
        axB.set_ylabel("source coin (by sector)", labelpad=118, fontsize=11)
        cbB = fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.03, extend=ext)
        cbB.set_label(r"$\alpha_{ij}$  (source $j$ $\rightarrow$ target $i$)", labelpad=12)

        # ---------------- (C) sector x sector ----------------
        imC = axC.imshow(np.ma.masked_invalid(S), cmap=cmC, aspect="equal", interpolation="nearest")
        ns = len(sec_sorted)
        axC.set_xticks(range(ns)); axC.set_xticklabels(sec_sorted, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
        axC.set_yticks(range(ns)); axC.set_yticklabels(sec_sorted, fontsize=9)
        axC.set_xlabel("target sector", labelpad=12, fontsize=11)
        axC.set_ylabel("source sector", labelpad=12, fontsize=11)
        cbC = fig.colorbar(imC, ax=axC, fraction=0.046, pad=0.03)
        cbC.set_label(r"mean $\bar{\alpha}$", labelpad=12)

        fig.suptitle(f"Fig. 4{'d' if mask_diag else 'c'} | Sector-ordered excitation / branching matrix ({tag}; rows = source, columns = target)",
                     fontsize=13, fontweight="bold")
        fig.canvas.draw()
        pB, pC = axB.get_position(), axC.get_position()
        fig.text((pB.x0 + pB.x1) / 2, 0.02, f"(B) {n_}×{n_} coin branching matrix", ha="center", fontsize=11)
        fig.text((pC.x0 + pC.x1) / 2, 0.02, f"(C) {ns}×{ns} sector branching matrix   |   systemic ρ(α) = {rho_mean_A:.2f}",
                 ha="center", fontsize=11)
        finish(fig, name)


    sector_style_figure(A, False, "diagonal included", "Fig04c_excitation_matrix_sector_ordered")
    sector_style_figure(A, True, "diagonal masked, cross-coin only", "Fig04d_excitation_matrix_sector_ordered_offdiag")

    # ==================================================================================
    # 3) BRANCHING RATIO  rho(alpha) = spectral radius
    # ==================================================================================
    seed_alpha, seed_rho_cached = [], []
    for s in SEEDS:
        c = np.load(ADIR / f"interp_seed{s}.npz", allow_pickle=True)
        seed_alpha.append(np.asarray(c["alpha"], float))
        seed_rho_cached.append(float(c["branching"]))
    seed_eig = [np.linalg.eigvals(a) for a in seed_alpha]
    seed_rho = [float(np.abs(e).max()) for e in seed_eig]
    mean_eig = np.linalg.eigvals(A)
    mean_rho = float(np.abs(mean_eig).max())
    for s, r, rc in zip(SEEDS, seed_rho, seed_rho_cached):
        print(f"branching ratio seed {s}: recomputed = {r:.4f} | cached = {rc:.4f}")
    print(f"branching ratio of 3-seed mean alpha = {mean_rho:.4f}  ({'stable' if mean_rho < 1 else 'UNSTABLE'})")

    # (a) eigenvalues in the complex plane with the unit circle
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), "k--", lw=1.6, label="unit circle (ρ = 1)")
    cols = ["#0057E7", "#00A651", "#FF7A00"]
    for s, e, col in zip(SEEDS, seed_eig, cols):
        ax.scatter(e.real, e.imag, s=28, alpha=0.6, color=col, edgecolors="none", label=f"seed {s} eigenvalues")
    ax.scatter(mean_eig.real, mean_eig.imag, s=60, marker="x", color="#E4003A", label=f"3-seed mean (ρ = {mean_rho:.3f})")
    ax.axhline(0, color="#888", lw=0.8); ax.axvline(0, color="#888", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Re(λ)"); ax.set_ylabel("Im(λ)")
    ax.set_title("Fig. 5a | Eigenvalues of the branching matrix α")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    finish(fig, "Fig05a_branching_eigenvalues")

    # (b) branching ratio per seed and for the mean matrix
    names = [f"seed {s}" for s in SEEDS] + ["3-seed mean"]
    vals = seed_rho + [mean_rho]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    bars = ax.bar(names, vals, color=cols + ["#E4003A"], edgecolor="black", width=0.55)
    ax.axhline(1.0, color="black", ls="--", lw=1.6, label="ρ = 1 (stability limit)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(1.1, max(vals) * 1.15))
    ax.set_ylabel("Branching ratio ρ(α)")
    ax.set_title("Fig. 5b | Branching ratio (spectral radius of α)")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    finish(fig, "Fig05b_branching_ratio")

    print("\nDone. Figures saved in:", FDIR)


# ======================================================================
# Network figure, two styles (sector-level + strongest coin links)
# ======================================================================
def fig_network():
    # ==================================================================================
    # NEW CELL (REPLACES the earlier two-panel network cell): two-panel network figure, TWO styles
    #   (a) sector-level contagion network   (b) strongest coin-level excitation links (labels = alpha)
    # RUN AFTER: Step 10 (cell 46). Needs only results/analysis/interp_agg.npz
    # SAVES TO : results/figures_journal/Fig_network_style1_pastel.png/.pdf
    #            results/figures_journal/Fig_network_style2_dark.png/.pdf
    # ==================================================================================
    from pathlib import Path
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Patch

    ADIR = ADIR_
    FDIR = FDIR_; FDIR.mkdir(parents=True, exist_ok=True)

    K_SECTOR_EDGES = 9      # number of strongest sector -> sector links in panel (a)
    K_COIN_EDGES = 14       # number of strongest coin -> coin links in panel (b)

    f = ADIR / "interp_agg.npz"
    if not f.exists():
        raise FileNotFoundError("Run Step 10 (cell 46) first: missing " + str(f))

    IA = np.load(f, allow_pickle=True)
    A = np.asarray(IA["alpha_mean"], float)                 # alpha[i, j] : target i (row), source j (column)
    labels = [str(x) for x in IA["labels"]]
    sectors = np.array([str(x) for x in IA["sectors"]])
    D = A.shape[0]
    secs = sorted(set(sectors))
    sidx = {s: np.where(sectors == s)[0] for s in secs}

    # ----------------------------------------------------------------------------------
    # THEMES  (style 1 = original pastel look, style 2 = new dark / vivid look)
    # ----------------------------------------------------------------------------------
    PASTEL = ["#BBDEFB", "#C8E6C9", "#FFE0B2", "#F8BBD0", "#E1BEE7", "#B2EBF2",
              "#FFF9C4", "#FFCDD2", "#D7CCC8", "#CFD8DC", "#DCEDC8", "#FFCCBC"]
    VIVID = ["#4FC3F7", "#69F0AE", "#FFB74D", "#F06292", "#BA68C8", "#4DD0E1",
             "#FFF176", "#FF8A80", "#A1887F", "#90A4AE", "#AED581", "#FF9E80"]
    THEMES = {
        "style1_pastel": dict(
            fig_bg="white", panel_bg="white", panel_ec="black", title_c="black", node_text="black", node_ec="black",
            sec_arrow="black", palette=PASTEL, coin_edge_mode="fixed", coin_edge_c="#B00020",
            lab_fc="white", lab_text="#B00020", lab_ec="#B00020", var_width=False, legend_fc="white", legend_text="black",
            file="Fig_network_style1_pastel"),
        "style2_dark": dict(
            fig_bg="#0E1621", panel_bg="#16222F", panel_ec="#5C7A99", title_c="#F2F6FA", node_text="#0E1621", node_ec="#F2F6FA",
            sec_arrow="#D5DEE8", palette=VIVID, coin_edge_mode="source", coin_edge_c="#FF5252",
            lab_fc="#0E1621", lab_text="#F2F6FA", lab_ec="#5C7A99", var_width=True, legend_fc="#16222F", legend_text="#F2F6FA",
            file="Fig_network_style2_dark"),
    }

    # ----------------------------------------------------------------------------------
    # DATA PREP (shared by both styles)
    # ----------------------------------------------------------------------------------
    # (a) sector edges: weight = mean alpha over all coin pairs, source sector -> target sector
    sec_edges = []
    for a, sa in enumerate(secs):
        for b, sb in enumerate(secs):
            if a == b:
                continue
            w = float(A[np.ix_(sidx[sb], sidx[sa])].mean())      # rows = targets in sb, columns = sources in sa
            sec_edges.append((w, a, b))
    sec_edges.sort(reverse=True)
    sec_edges = sec_edges[:K_SECTOR_EDGES]
    sec_set = {(a, b) for _, a, b in sec_edges}
    sec_w = {(a, b): w for w, a, b in sec_edges}

    used = sorted({a for _, a, _ in sec_edges} | {b for _, _, b in sec_edges})
    deg = {k: 0 for k in used}
    wsum = {k: 0.0 for k in used}
    for w, a, b in sec_edges:
        deg[a] += 1; deg[b] += 1; wsum[a] += w; wsum[b] += w
    center = max(used, key=lambda k: (deg[k], wsum[k]))
    ring = [k for k in used if k != center]
    R_NODE, R_RING = 0.52, 2.15
    posL = {center: np.array([0.0, 0.0])}
    for n, k in enumerate(ring):
        ang = np.pi / 2 - 2 * np.pi * n / max(len(ring), 1)
        posL[k] = np.array([R_RING * np.cos(ang), R_RING * np.sin(ang)])

    # (b) coin edges: strongest K off-diagonal entries, drawn source j -> target i
    W = A.copy(); np.fill_diagonal(W, 0.0)
    flat = np.argsort(-W, axis=None)[:K_COIN_EDGES]
    coin_edges = []
    for q in flat:
        i, j = np.unravel_index(q, W.shape)                 # target i, source j
        coin_edges.append((float(W[i, j]), int(j), int(i)))
    G = nx.DiGraph()
    for w, j, i in coin_edges:
        G.add_edge(j, i, w=w)
    nodes = list(G.nodes())

    # node box size in data units (width grows with label length)
    size = {n: (0.40 + 0.16 * len(labels[n]), 0.40) for n in nodes}      # ~inches (axes are drawn 1 data unit ~ 1 inch)
    LAB_W, LAB_H = 0.56, 0.28                               # size of an alpha label (data units)


    def comp_layout(sub):
        """Layout of ONE connected component (never mixed with other components -> no collapse)."""
        ns = list(sub.nodes())
        n = len(ns)
        if n == 1:
            return {ns[0]: np.array([0.0, 0.0])}
        if n == 2:
            return {ns[0]: np.array([-1.5, 0.0]), ns[1]: np.array([1.5, 0.0])}
        if n == 3:
            p = nx.circular_layout(sub, scale=1.8)
        else:
            p = nx.kamada_kawai_layout(sub, scale=1.2 + 0.45 * n)
        return {k: np.asarray(v, float) for k, v in p.items()}


    def pack_components(G_):
        """Layout each weakly connected component separately, then shelf-pack them side by side."""
        comps = sorted(nx.weakly_connected_components(G_), key=lambda c: -len(c))
        placed, boxes_c = {}, []
        for c in comps:
            p = comp_layout(G_.subgraph(c).to_undirected())
            xs = np.array([p[k][0] for k in p]); ys = np.array([p[k][1] for k in p])
            boxes_c.append((p, xs.min() - 0.6, xs.max() + 0.6, ys.min() - 0.5, ys.max() + 0.5))
        ROW_W, GAP = 13.0, 0.5
        x_cur, y_cur, row_h = 0.0, 0.0, 0.0
        for p, x0, x1, y0, y1 in boxes_c:
            w_, h_ = x1 - x0, y1 - y0
            if x_cur > 0 and x_cur + w_ > ROW_W:
                x_cur, y_cur, row_h = 0.0, y_cur - row_h - GAP, 0.0
            row_h = max(row_h, h_)
            ox, oy = x_cur - x0, y_cur - y1               # shift so the box top-left sits at (x_cur, y_cur)
            for k in p:
                placed[k] = p[k] + np.array([ox, oy])
            x_cur += w_ + GAP
        return placed


    def resolve(pos, sizes, pad=0.65, iters=800):
        """Push apart any two node boxes that overlap (plus a safety pad)."""
        ids = list(pos)
        for _ in range(iters):
            moved = False
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    u, v = ids[x], ids[y]
                    dx = pos[v][0] - pos[u][0]; dy = pos[v][1] - pos[u][1]
                    ox = (sizes[u][0] + sizes[v][0]) / 2 + pad - abs(dx)
                    oy = (sizes[u][1] + sizes[v][1]) / 2 + pad - abs(dy)
                    if ox > 0 and oy > 0:
                        if ox < oy:
                            s = np.sign(dx) if dx != 0 else 1.0
                            pos[u][0] -= s * ox / 2; pos[v][0] += s * ox / 2
                        else:
                            s = np.sign(dy) if dy != 0 else 1.0
                            pos[u][1] -= s * oy / 2; pos[v][1] += s * oy / 2
                        moved = True
            if not moved:
                break


    posR = pack_components(G)
    posR = {n: np.array(posR[n], float) for n in nodes}
    H_MAX = 5.8                                            # max data height of panel (b)
    _h = max(p[1] for p in posR.values()) - min(p[1] for p in posR.values()) + max(sz[1] for sz in size.values())
    if _h > H_MAX:                                         # compress vertically so that panel (b) is wide and flat
        _c = np.mean([p[1] for p in posR.values()])
        for n in nodes:
            posR[n][1] = _c + (posR[n][1] - _c) * (H_MAX / _h)
    resolve(posR, size)
    # every edge must be long enough to carry its alpha label ON the arrow: push endpoints apart, then de-overlap again
    for _ in range(400):
        _moved = False
        for _, j, i in coin_edges:
            d = posR[i] - posR[j]
            L = float(np.linalg.norm(d)) + 1e-9
            need = (size[i][0] + size[j][0]) / 2 + LAB_W + 0.55
            if L < need:
                u = d / L
                posR[i] = posR[i] + u * (need - L) / 2
                posR[j] = posR[j] - u * (need - L) / 2
                _moved = True
        resolve(posR, size, iters=5)
        if not _moved:
            break
    resolve(posR, size)

    RAD = 0.16


    def bez(p1, p2, t, rad=RAD):
        """Point at parameter t on matplotlib's arc3 curve from p1 to p2."""
        d = p2 - p1
        c = (p1 + p2) / 2 + rad * np.array([d[1], -d[0]])
        return (1 - t) ** 2 * p1 + 2 * t * (1 - t) * c + t ** 2 * p2


    def rect_overlap(c1, s1, c2, s2, pad=0.02):
        return (abs(c1[0] - c2[0]) < (s1[0] + s2[0]) / 2 + pad) and (abs(c1[1] - c2[1]) < (s1[1] + s2[1]) / 2 + pad)


    # label placement: try several positions along every arc; keep the first one that touches no node and no other label
    placed_labels = []
    lab_pos = {}
    order = sorted(range(len(coin_edges)), key=lambda e: -coin_edges[e][0])
    for e in order:
        w, j, i = coin_edges[e]
        best, best_pen = None, None
        _d = posR[i] - posR[j]
        _nrm = np.array([_d[1], -_d[0]]) / (np.linalg.norm(_d) + 1e-9)
        cands = [(t, 0.0) for t in (0.50, 0.42, 0.58, 0.35, 0.65, 0.30, 0.70, 0.26, 0.74)]   # always ON the arrow
        for t, off in cands:
            c = bez(posR[j], posR[i], t) + off * _nrm
            pen = 0
            for n in nodes:
                if rect_overlap(c, (LAB_W, LAB_H), posR[n], size[n], pad=0.15):
                    pen += 1
            for c2 in placed_labels:
                if rect_overlap(c, (LAB_W, LAB_H), c2, (LAB_W, LAB_H)):
                    pen += 1
            if best is None or pen < best_pen:
                best, best_pen = c, pen
            if pen == 0:
                break
        lab_pos[e] = best
        if best_pen:
            print('WARNING: label of edge', labels[j], '->', labels[i], 'still touches', best_pen, 'object(s)')
        placed_labels.append(best)

    xs = [posR[n][0] for n in nodes]; ys = [posR[n][1] for n in nodes]
    mw = max(s[0] for s in size.values())
    XLIM = (min(xs) - mw / 2 - 0.5, max(xs) + mw / 2 + 0.5)
    _yc = (min(ys) + max(ys)) / 2
    _yh = max(max(ys) - min(ys) + 1.8, 6.4)
    YLIM = (_yc - _yh / 2, _yc + _yh / 2)
    w_out = {n: 0.0 for n in range(len(secs))}
    for w, a, b in sec_edges:
        w_out[a] += w
    wmax_sec = max(sec_w.values())
    wmax_coin = max(w for w, _, _ in coin_edges)


    def frame(ax, th):
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", transform=ax.transAxes,
                                    fc=th["panel_bg"], ec=th["panel_ec"], lw=1.6, clip_on=False, zorder=0))


    def draw(th):
        pal = th["palette"]
        scol = {s: pal[k % len(pal)] for k, s in enumerate(secs)}
        _wr = (XLIM[1] - XLIM[0]) / (YLIM[1] - YLIM[0])
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.8 + 6.8 * _wr + 0.6, 8.2), gridspec_kw=dict(width_ratios=[1.0, _wr]))
        fig.patch.set_facecolor(th["fig_bg"])

        # ---------------- (a) sector network ----------------
        frame(axL, th)
        circles = {}
        for k in used:
            c = Circle(posL[k], R_NODE, fc=scol[secs[k]], ec=th["node_ec"], lw=1.5, zorder=3)
            axL.add_patch(c); circles[k] = c
            axL.text(posL[k][0], posL[k][1], secs[k].replace("-", "-\n"), ha="center", va="center",
                     fontsize=9, fontweight="bold", color=th["node_text"], zorder=4)
        drawn = set()
        for w, a, b in sec_edges:
            if (a, b) in drawn:
                continue
            both = (b, a) in sec_set
            lw = (1.6 + 3.4 * w / wmax_sec) if th["var_width"] else 2.2
            axL.add_patch(FancyArrowPatch(posL[a], posL[b], arrowstyle="<|-|>" if both else "-|>", mutation_scale=20,
                                          lw=lw, color=th["sec_arrow"], patchA=circles[a], patchB=circles[b],
                                          shrinkA=0, shrinkB=0, zorder=2))
            drawn.add((a, b)); drawn.add((b, a))
        axL.set_xlim(-3.0, 3.0); axL.set_ylim(-3.0, 3.0)
        axL.set_aspect("equal"); axL.axis("off")
        axL.set_title(f"(a) Sector-level contagion (top {len(sec_edges)} sector links; arrow = source -> target)",
                      fontsize=11, fontweight="bold", color=th["title_c"])

        # ---------------- (b) coin network ----------------
        frame(axR, th)
        boxes = {}
        for n in nodes:
            w_, h_ = size[n]
            b = FancyBboxPatch((posR[n][0] - w_ / 2, posR[n][1] - h_ / 2), w_, h_, boxstyle="round,pad=0,rounding_size=0.06",
                               fc=scol[sectors[n]], ec=th["node_ec"], lw=1.3, zorder=3)
            axR.add_patch(b); boxes[n] = b
            axR.text(posR[n][0], posR[n][1], labels[n], ha="center", va="center", fontsize=10, fontweight="bold",
                     color=th["node_text"], zorder=4)
        for e, (w, j, i) in enumerate(coin_edges):
            col = scol[sectors[j]] if th["coin_edge_mode"] == "source" else th["coin_edge_c"]
            lw = (1.4 + 3.2 * w / wmax_coin) if th["var_width"] else 1.7
            axR.add_patch(FancyArrowPatch(posR[j], posR[i], arrowstyle="-|>", mutation_scale=15, lw=lw, color=col,
                                          connectionstyle=f"arc3,rad={RAD}", patchA=boxes[j], patchB=boxes[i],
                                          shrinkA=0, shrinkB=0, zorder=2))
            lc = col if th["coin_edge_mode"] == "source" else th["lab_ec"]
            tc = th["lab_text"]
            axR.text(lab_pos[e][0], lab_pos[e][1], f"{w:.3f}", ha="center", va="center", fontsize=8.5, color=tc, zorder=5,
                     bbox=dict(boxstyle="round,pad=0.16", fc=th["lab_fc"], ec=lc, lw=1.0))
        axR.set_xlim(*XLIM); axR.set_ylim(*YLIM)
        axR.set_aspect("equal"); axR.axis("off")
        axR.set_title(f"(b) Strongest {len(coin_edges)} coin-level excitation links (label = alpha; arrow = source -> target)",
                      fontsize=11, fontweight="bold", color=th["title_c"])

        # ---------------- legend ----------------
        present = sorted({secs[k] for k in used} | {sectors[n] for n in nodes})
        leg = fig.legend(handles=[Patch(fc=scol[s], ec=th["node_ec"], label=s) for s in present], loc="lower center",
                         ncol=min(len(present), 6), fontsize=9, title="Sector / category", frameon=True,
                         facecolor=th["legend_fc"], edgecolor=th["panel_ec"])
        leg.get_title().set_color(th["legend_text"])
        for t in leg.get_texts():
            t.set_color(th["legend_text"])
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        for ext in ("png", "pdf"):
            fig.savefig(FDIR / f"{th['file']}.{ext}", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print("saved:", FDIR / f"{th['file']}.png", "and .pdf")


    for _name, _th in THEMES.items():
        draw(_th)


# ======================================================================
# Transmitter vs receiver role geometry
# ======================================================================
def fig_role_geometry():
    # ======================================================================
    # NEW CELL: Transmitter vs receiver role geometry (scatter of pairwise cosine similarities)
    # RUN AFTER: Step 10 / cell 46 (needs results/analysis/interp_seed{0,1,2}.npz containing src & tgt)
    # OUTPUT   : results/figures_journal/Fig_role_geometry_transmitter_vs_receiver.png / .pdf
    # ======================================================================

    # ---- Step 1: libraries -------------------------------------------------
    from pathlib import Path
    from itertools import combinations
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    # ---- Step 2: paths and plot style --------------------------------------
    ADIR = ADIR_
    FDIR = FDIR_
    FDIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
        "axes.linewidth": 1.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.major.width": 1.6, "ytick.major.width": 1.6, "pdf.fonttype": 42,
    })

    # ---- Step 3: load cached seeds (src / tgt embeddings) ------------------
    seed_files = sorted(ADIR.glob("interp_seed*.npz"))
    if len(seed_files) == 0:
        raise FileNotFoundError("results/analysis/interp_seed*.npz not found - run Step 10 (cell 46) first.")
    SRC, TGT = [], []
    for f in seed_files:
        d = np.load(f, allow_pickle=True)
        if "src" not in d.files or "tgt" not in d.files:
            raise KeyError(f"{f.name} has no src/tgt - delete interp_seed*.npz and rerun Step 10.")
        SRC.append(d["src"].astype(float)); TGT.append(d["tgt"].astype(float))
    D = SRC[0].shape[0]
    print(f"[role-geometry] seeds found: {[f.stem for f in seed_files]} | D = {D}")

    # ---- Step 4: gauge-fix (same as stage_embed) ---------------------------
    # raw src/tgt are identified only up to a GL(d) transform; the identified object is Z = tgt @ src.T
    # SVD of Z gives rotation-fixed receiver (U) and transmitter (V) coordinates
    def gauge_fix(src, tgt):
        Z = tgt @ src.T
        U, sv, Vt = np.linalg.svd(Z)
        r = max(2, int((sv > 1e-6 * sv[0]).sum()))
        sq = np.sqrt(sv[:r])
        return U[:, :r] * sq, Vt[:r].T * sq

    def cos_mat(X):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return Xn @ Xn.T

    # ---- Step 5: cosine-similarity matrices for every seed -----------------
    rec_sims, tra_sims = [], []
    for s_, t_ in zip(SRC, TGT):
        Ur, Vr = gauge_fix(s_, t_)
        rec_sims.append(cos_mat(Ur))      # receiver-role similarity  (cosine of U)
        tra_sims.append(cos_mat(Vr))      # transmitter-role similarity (cosine of V)

    # ---- Step 6: pairwise points (upper triangle, seed 0) + Pearson r ------
    iu = np.triu_indices(D, 1)
    x = tra_sims[0][iu]                   # x-axis: transmitter-role similarity
    y = rec_sims[0][iu]                   # y-axis: receiver-role similarity
    r_pair, p_pair = stats.pearsonr(x, y)
    print(f"[role-geometry] pairs = {len(x)} | pairwise r = {r_pair:.3f} (p = {p_pair:.3g})")

    # ---- Step 7: seed stability of transmitter geometry --------------------
    if len(tra_sims) >= 2:
        stab = float(np.mean([stats.pearsonr(tra_sims[a][iu], tra_sims[b][iu])[0]
                              for a, b in combinations(range(len(tra_sims)), 2)]))
        stab_rec = float(np.mean([stats.pearsonr(rec_sims[a][iu], rec_sims[b][iu])[0]
                                  for a, b in combinations(range(len(rec_sims)), 2)]))
        print(f"[role-geometry] seed stability: transmitter = {stab:.3f} | receiver = {stab_rec:.3f}")
    else:
        stab = None

    # ---- Step 8: regression line (least squares) ---------------------------
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)

    # ---- Step 9: plot ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 7.7))
    ax.scatter(x, y, s=70, color="#1f6fe5", alpha=0.45, edgecolors="none", zorder=3)
    ax.plot(xx, slope * xx + intercept, color="#e0003c", lw=3.2, ls=(0, (5, 2.4)), zorder=4)
    ax.grid(True, color="#d9d9d9", lw=0.9, zorder=0)
    ax.set_xlabel("Transmitter-role similarity (cosine of V)", fontsize=17)
    ax.set_ylabel("Receiver-role similarity (cosine of U)", fontsize=17)
    ax.set_title("Fig. 13 | Transmitter vs receiver role geometry", fontsize=18, pad=12)
    ax.tick_params(labelsize=14)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    ax.text(0.03, 0.965, f"pairwise r = {r_pair:.2f}", transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="left", zorder=5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=1.4))
    if stab is not None:
        ax.text(0.03, 0.035, f"seed stability (transmitter sim-corr) = {stab:.2f}",
                transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left", zorder=5)
    fig.tight_layout()

    # ---- Step 10: save -----------------------------------------------------
    name = "Fig_role_geometry_transmitter_vs_receiver"
    fig.savefig(FDIR / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FDIR / f"{name}.pdf", bbox_inches="tight")
    print(f"  saved {FDIR / (name + '.png')} / .pdf")


# ======================================================================
# tau x d sensitivity curves (values from tab:sensitivity_tau_dim)
# ======================================================================
def fig_capacity_curves():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # ------------------------------------------------------------------
    # DATA  (mean, std) from tab:sensitivity_tau_dim
    # ------------------------------------------------------------------
    dims = np.array([2, 4, 8, 16])
    taus = [2.5, 3.0, 3.5, 4.0]

    mean = {
        2.5: [-1.8138, -1.7735, -1.7393, -1.7316],
        3.0: [-2.4624, -2.4272, -2.4180, -2.4206],
        3.5: [-3.3145, -3.2517, -3.2401, -3.2537],
        4.0: [-4.4701, -4.3598, -4.3469, -4.3452],
    }
    std = {
        2.5: [0.047, 0.020, 0.014, 0.003],
        3.0: [0.030, 0.023, 0.003, 0.004],
        3.5: [0.018, 0.013, 0.003, 0.003],
        4.0: [0.019, 0.017, 0.011, 0.010],
    }
    CANON = 8
    ci = list(dims).index(CANON)   # index of canonical d

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    def info(tau):
        m, s = np.array(mean[tau]), np.array(std[tau])
        b = int(np.argmax(m))                 # higher (less negative) is better
        gap = m[ci] - m[b]
        return m, s, b, gap

    # ==================================================================
    # FIGURE A : same layout as the PDF, new colours
    # ==================================================================
    LINE_C  = "#0B6E4F"   # deep green
    BAND_C  = "#BFE3D3"
    STAR_C  = "#FF7F0E"   # orange star
    RING_C  = "#7B2CBF"   # purple ring
    TXT_C   = "#2B2B2B"

    figA, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    for ax, tau in zip(axes.ravel(), taus):
        m, s, b, gap = info(tau)
        x = np.arange(len(dims))              # categorical spacing, like the PDF

        ax.fill_between(x, m - s, m + s, color=BAND_C, alpha=0.7, lw=0)
        ax.plot(x, m, "-o", color=LINE_C, lw=2.2, ms=6, zorder=3)

        # value labels
        for i, v in enumerate(m):
            is_b, is_c = (i == b), (i == ci)
            ax.annotate(f"{v:.4f}", (x[i], v), xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=8.5,
                        color=(STAR_C if is_b else RING_C if is_c else TXT_C),
                        fontweight="bold" if is_b else "normal")

        # best d = star ; canonical d=8 = ring
        ax.scatter(x[b], m[b], marker="*", s=380, color=STAR_C, edgecolor="k", lw=1, zorder=5)
        ax.scatter(x[ci], m[ci], s=260, facecolor="none", edgecolor=RING_C, lw=2.2, zorder=4)

        txt = "d=8 is the best" if b == ci else f"d=8 gap to best: {gap:.4f}"
        ax.text(0.03, 0.06, txt, transform=ax.transAxes, color=RING_C, fontsize=9.5)

        pad = 0.35 * (np.max(m + s) - np.min(m - s))
        ax.set_ylim(np.min(m - s) - pad * 0.5, np.max(m + s) + pad)
        ax.set_xticks(x); ax.set_xticklabels(dims)
        ax.set_xlim(-0.4, len(dims) - 0.6)
        ax.set_xlabel(r"Embedding dimension $d$")
        ax.set_ylabel("Held-out LL/event")
        ax.set_title(rf"Threshold $\tau$ = {tau}", fontweight="bold")
        ax.grid(axis="y", color="#DDDDDD", lw=0.7)

    figA.text(0.01, 0.005,
              "Line/number = mean over 3 seeds; shaded band = ±1 std; orange star = best d at that τ; "
              "purple circle = d = 8 (canonical). Higher (less negative) is better.",
              fontsize=8.5, color="#555555")
    figA.tight_layout(rect=(0, 0.03, 1, 1))
    figA.savefig(str(FDIR_ / "fig_capacity_curves_A.pdf"), bbox_inches="tight")
    figA.savefig(str(FDIR_ / "fig_capacity_curves_A.png"), dpi=300, bbox_inches="tight")

    # ==================================================================
    # FIGURE B : changed style (1x4 row, error bars, dashed, diamonds)
    # ==================================================================
    C_MAIN, C_BEST, C_CAN, C_BG = "#1D3557", "#E63946", "#F4A261", "#F1FAEE"

    figB, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True)
    for ax, tau in zip(axes, taus):
        m, s, b, gap = info(tau)
        x = np.arange(len(dims))

        ax.set_facecolor(C_BG)
        ax.axvspan(ci - 0.35, ci + 0.35, color=C_CAN, alpha=0.25, lw=0)       # d=8 highlight
        ax.errorbar(x, m, yerr=s, fmt="D--", color=C_MAIN, ecolor=C_MAIN,
                    elinewidth=1.4, capsize=4, capthick=1.4, ms=6, lw=1.8, zorder=3)
        ax.scatter(x[b], m[b], marker="P", s=200, color=C_BEST,
                   edgecolor="white", lw=1.2, zorder=5)

        # delta-from-best labels under every point
        for i, v in enumerate(m):
            d = v - m[b]
            ax.annotate("best" if i == b else f"{d:+.4f}", (x[i], v - s[i]),
                        xytext=(0, -14), textcoords="offset points", ha="center",
                        fontsize=8.5, color=C_BEST if i == b else "#444444",
                        fontweight="bold" if i == b else "normal")

        ax.set_xticks(x); ax.set_xticklabels(dims)
        ax.set_xlabel(r"Embedding dimension $d$")
        ax.set_title(rf"$\tau = {tau}$", fontsize=13, fontweight="bold", color=C_MAIN)
        ax.grid(True, ls=":", color="#B0B0B0", lw=0.8)
        lo, hi = np.min(m - s), np.max(m + s)
        ax.set_ylim(lo - 0.25 * (hi - lo) - 0.01, hi + 0.15 * (hi - lo))
    axes[0].set_ylabel("Held-out LL/event")

    legend_items = [
        Line2D([], [], marker="D", ls="--", color=C_MAIN, label="Mean ± 1 std (3 seeds)"),
        Line2D([], [], marker="P", ls="", color=C_BEST, ms=10, label="Best $d$ at that $\\tau$"),
        plt.Rectangle((0, 0), 1, 1, color=C_CAN, alpha=0.4, label="$d=8$ (canonical)"),
    ]
    figB.legend(handles=legend_items, loc="lower center", ncol=3, frameon=False, fontsize=10)
    figB.tight_layout(rect=(0, 0.09, 1, 1))
    figB.savefig(str(FDIR_ / "fig_capacity_curves_B.pdf"), bbox_inches="tight")
    figB.savefig(str(FDIR_ / "fig_capacity_curves_B.png"), dpi=300, bbox_inches="tight")

    plt.show()


ALL = [("gof_matrix_branching", fig_gof_matrix_branching), ("network", fig_network),
       ("role_geometry", fig_role_geometry), ("capacity_curves", fig_capacity_curves)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    if a.list:
        for n in ["gof_rescaled.npz", "interp_agg.npz", "interp_seed0.npz", "interp_seed1.npz", "interp_seed2.npz"]:
            print(f"  [{'x' if (ADIR_ / n).exists() else ' '}] {n}")
        return
    print("Building figures -> results/figures_journal/")
    for name, fn in ALL:
        if a.only and name not in a.only:
            continue
        fn()
    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""3D Pareto front in (R, U1, U2): STEER's lambda family vs fixed-representation baselines.

Paste your real coordinates in the block below, then run

    python pareto_ssl/plot_pareto_front_3d.py

which writes pareto_front_3d.pdf and pareto_front_3d.png. The coordinate lists ship EMPTY
so that illustrative numbers can never end up in the paper by accident; `--demo` renders
clearly synthetic values only to preview the layout.

Nothing about the surface is fitted or interpolated. Its vertices are exactly your
non-dominated points; the only construction is WHICH of them share a triangle:

  * "projected_delaunay" (default). Points are projected onto the plane orthogonal to
    (1, 1, 1) -- the direction along which all three objectives improve together -- and
    triangulated there with Delaunay. Neighbouring trade-offs share faces, the triangles
    never cross, and every non-dominated point is a vertex, including those in concave
    regions of the front.
  * "convex_hull". Faces of the 3D convex hull that face the (1, 1, 1) direction (outward
    normal with a positive component along it). Cleaner for convex fronts,
    but points lying in concave regions are not hull vertices and are left off the surface.
"""
import argparse
import warnings

import numpy as np

# PASTE YOUR COORDINATES HERE  —  [R, U1, U2], higher is better on every axis
# Trifeature, 15 shared shapes, corr(U, R) = 0  (data/tri15_corr00).
# Test linear-probe accuracy: R = share (15-way), U1 = unique1 (10-way), U2 = unique2 (10-way).
# Every value is the MEAN over seeds 42-46 of one matched sweep, all launched 2026-09-01:
#   pareto_ssl/results_corr_sweep/_parts/corr00_<method>_s<seed>[...]
#
# STEER = simclr_simplex_enc_decomp_R (STEER-enc), lam_club = 0.5 (the reference CLUB
# weight), `dim` readout (results_readout_dim.json), one row per lambda on the side-5 grid.
steer_points = [
    [0.3762, 0.1057, 0.3714],   # (0.00, 0.00, 1.00)
    [0.4655, 0.1585, 0.3322],   # (0.00, 0.25, 0.75)
    [0.5506, 0.2276, 0.2380],   # (0.00, 0.50, 0.50)
    [0.5619, 0.2724, 0.1496],   # (0.00, 0.75, 0.25)
    [0.5533, 0.2728, 0.1019],   # (0.00, 1.00, 0.00)
    [0.6381, 0.1104, 0.3000],   # (0.25, 0.00, 0.75)
    [0.7724, 0.1695, 0.2165],   # (0.25, 0.25, 0.50)
    [0.8121, 0.2341, 0.1515],   # (0.25, 0.50, 0.25)
    [0.7437, 0.2638, 0.1055],   # (0.25, 0.75, 0.00)
    [0.8931, 0.1187, 0.1922],   # (0.50, 0.00, 0.50)
    [0.9770, 0.1883, 0.1475],   # (0.50, 0.25, 0.25)
    [0.9549, 0.2289, 0.1107],   # (0.50, 0.50, 0.00)
    [0.9884, 0.1256, 0.1582],   # (0.75, 0.00, 0.25)
    [0.9983, 0.1844, 0.1153],   # (0.75, 0.25, 0.00)
    [0.9998, 0.1389, 0.1296],   # (1.00, 0.00, 0.00)
]

# Preference behind each row above, same order. Used to annotate the per-task winners, so a
# reader sees that the shape/deformation/texture winners sit on the simplex VERTICES
# (1,0,0) / (0,1,0) / (0,0,1) rather than having to take the caption's word for it.
# Set to None to switch the annotation off.
steer_lambdas = [
    (0.00, 0.00, 1.00), (0.00, 0.25, 0.75), (0.00, 0.50, 0.50), (0.00, 0.75, 0.25),
    (0.00, 1.00, 0.00), (0.25, 0.00, 0.75), (0.25, 0.25, 0.50), (0.25, 0.50, 0.25),
    (0.25, 0.75, 0.00), (0.50, 0.00, 0.50), (0.50, 0.25, 0.25), (0.50, 0.50, 0.00),
    (0.75, 0.00, 0.25), (0.75, 0.25, 0.00), (1.00, 0.00, 0.00),
]

baseline_points = {
    "CLIP":     [0.9137, 0.1086, 0.2226],   # corr00_clip_s4*/results.json            (5 seeds)
    "GMC":      [0.8134, 0.1866, 0.2078],   # corr00_gmc_s4*/results.json             (5 seeds)
    "CoMM":     [0.6983, 0.1813, 0.3293],   # corr00_comm_s4*/results.json            (5 seeds)
    # factorcl_heads: probes the best-matching head per task (proj_r for R, proj_u for U1,
    # proj_m2_u for U2), i.e. a task-ORACLE, not one fixed representation. benchmark.py marks
    # it display-only for that reason. The single-vector `factorcl` (concat) runs in this
    # sweep never wrote results, so this is the only FactorCL point available.
    "FactorCL": [0.9145, 0.1329, 0.3145],   # corr00_factorcl_heads_s4*/results.json  (5 seeds)
}

# OPTIONS
OUT_STEM = "pareto_front_3d"            # -> <stem>.pdf and <stem>.png
PNG_DPI = 400

# Factor names first, with the PID role in brackets, and the metric spelled out. The values
# plotted are TOP-1 ACCURACY of the linear probe (15-way for shape, 10-way for deformation
# and texture) -- benchmark.py's `_probe` returns clf.score(), i.e. mean accuracy. They are
# NOT AUC; nothing in the Trifeature probe path computes one. Change METRIC only if the
# pasted numbers are actually something else.
METRIC = "ACC"
# Task names only. The R / U_1 / U_2 roles are carried by the magenta vertex labels and by
# the caption, so repeating them on every axis was redundant and made the tick area crowded.
AXIS_LABELS = (rf"Shape" + f"\n({METRIC}, %)",
               rf"Deformation" + f"\n({METRIC}, %)",
               rf"Texture" + f"\n({METRIC}, %)")
# Values are stored as fractions in [0, 1]; ticks are rendered as percentages so the axes
# read 20, 40, 60 rather than 0.2, 0.4, 0.6. Data and limits stay in fraction space.
TICKS_AS_PERCENT = True
# Chance level per axis: 1/n_classes. Trifeature R has 15 shape classes, U1 and U2 have 10
# each, so chance is 6.7% / 10% / 10% -- NOT the same on all three axes, which is exactly
# why it has to be drawn rather than left to the reader. Set to None to omit.
CHANCE = (1 / 15, 1 / 10, 1 / 10)
# "lines"  : dashed reference line at chance on each axis, drawn on the floor/back panes
# "floor"  : additionally shade the at-or-below-chance corner
# None     : draw nothing
SHOW_CHANCE = "lines"
CHANCE_COLOR = "#7a7a7a"
# None     -> [0, 1] on every axis when all values lie in [0, 1], otherwise padded data range.
# "data"   -> padded data range per axis, even for accuracies (use when scores sit far below
#             1, as the Trifeature unique axes do: max 0.39 -- at [0, 1] the front is a smudge)
# "chance" -> lower bound EXACTLY at chance, upper bound padded to the data. The axes then
#             start where a random classifier sits, so distance from the wall reads as
#             "information above chance" with no mental arithmetic. Points below chance
#             would fall outside the box, so the code checks and warns instead of clipping.
# Or set explicitly, e.g. ((0.4, 1.0), (0.3, 1.0), (0.3, 1.0)).
AXIS_LIMITS = "data"

SURFACE_METHOD = "projected_delaunay"   # "projected_delaunay" | "convex_hull"
# Optional: drop surface triangles whose longest edge exceeds this fraction of the front's
# diameter (removes long boundary slivers). None keeps every triangle.
MAX_EDGE_FRAC = None

SHOW_DOMINATED_STEER = True             # dominated STEER points drawn lighter, off-surface
SHOW_BASELINE_FRONT = False             # gray face/segment over non-dominated baselines: off
                                        # (it read as a surface the baselines span, which is
                                        #  not a claim the data supports -- they are points)
FLOOR_PROJECTIONS = True                # faint drop lines to the floor, helps depth reading

VIEW_ELEV, VIEW_AZIM = 22, 38           # camera; tweak if a point hides behind another
BOX_ZOOM = 0.86                         # shrink the 3D box so axis labels are not clipped
LEGEND_NCOL = 3                         # legend sits above the axes in this many columns

STEER_COLOR = "#1f4fa3"
STEER_EDGE = "#0b2350"
STEER_LIGHT = "#9db8e6"
# The lambda that scores best on each objective -- the "every task picks its own preference"
# points. Drawn in magenta on top of the blue family. A point best on two objectives is
# highlighted once and labelled with both.
HIGHLIGHT_TASK_BEST = True
TASK_BEST_COLOR = "#D6006E"
TASK_BEST_EDGE = "#66002F"
TASK_BEST_ANNOTATE = True               # write the winning preference next to those markers
TASK_BEST_TAGS = (r"$R$", r"$U_1$", r"$U_2$")
# Annotate each per-task winner with the PREFERENCE that produced it, e.g. "R (1,0,0)".
# On Trifeature the three winners sit exactly on the simplex vertices, which is the claim
# the figure exists to make, so the coordinates belong on the plot rather than the caption.
TASK_BEST_SHOW_LAMBDA = True
# Label placement, in POINTS on the rendered canvas, per tag. Offsetting in data space does
# not work here: the three winners sit at very different depths, so the same 3D offset lands
# at wildly different on-screen distances, and R/U2 ended up on top of the mesh. These are
# screen-space offsets with a leader line, so what you set is what you see.
# The legend occupies the strip ABOVE the axes, so a label near the top of the box must be
# pushed sideways or downward, never up: U2 is the texture vertex and sits high in the plot,
# which is why its offset points left and slightly down.
TASK_BEST_OFFSETS = {"R": (34, 22), "U_1": (-46, 18), "U_2": (-58, -10)}
TASK_BEST_OFFSET_DEFAULT = (30, 26)

# variant B: label EVERY preference, not just the per-task winners (--label-all)
# Fifteen labels will not fit under the hand-tuned offsets above, so this mode places them
# automatically: each label is pushed radially AWAY from the centroid of the projected
# point cloud (so labels leave the mesh rather than cross it) and the radius is grown
# greedily until the label box clears every label already placed. Placement is done in
# DISPLAY PIXELS after a first draw, because the 3D projection is only known then.
# Which preferences to label. Empty/None = every point. Give a list of lambda tuples to
# label only those -- the figure gets unreadable past a handful of boxes at vertex size,
# and the interesting claim is usually carried by two or three specific preferences.
LABEL_ALL_ONLY = [(0.0, 0.5, 0.5), (0.0, 0.25, 0.75)]
# Styled to match the per-task vertex boxes exactly (same size, weight and pill geometry);
# only the colour differs, so magenta still means "best on this task" and nothing else.
LABEL_ALL_FONTSIZE = 11.0
LABEL_ALL_BOLD = True
LABEL_ALL_COLOR = STEER_EDGE
LABEL_ALL_BOX_EDGE = STEER_COLOR
LABEL_ALL_BASE_OFFSET = 30              # points, first radius tried
LABEL_ALL_STEP = 10                     # points, radius increment per collision retry
LABEL_ALL_MAX_TRIES = 14
LABEL_ALL_BBOX = True                   # white pill behind each label; off = text only
LABEL_ALL_LEADER = True                 # thin line from label back to its marker
# The magenta winners keep their own "R  lambda=(1,0,0)" box in this variant too. Set True
# only if LABEL_ALL_ONLY is empty (every point labelled), where it would print twice.
LABEL_ALL_SUPPRESS_BEST_LAMBDA = False
# Screen-space offsets for the magenta winners IN THIS VARIANT ONLY. The main figure keeps
# TASK_BEST_OFFSETS untouched. Adding the lambda line makes each box taller, and with the
# extra blue boxes present the old offsets pushed R and U_1 INTO the trade-off surface;
# these point away from the centre of the projected cloud instead, so both labels sit
# outside the volume. U_2 still goes left rather than up, because the legend owns the
# strip above the axes.
TASK_BEST_OFFSETS_LABEL_ALL = {"R": (-74, -30), "U_1": (74, 14), "U_2": (-62, -12)}
# Baseline visibility: drop each baseline to the floor with a stem + a hollow ground marker,
# so a point that sits inside the blue surface can still be located in depth.
BASELINE_STEMS = True
BASELINE_MARKER_SIZE = 95
# Okabe–Ito colour-blind-safe palette, one marker shape per baseline
BASELINE_STYLE = {
    "CLIP":     dict(color="#E69F00", marker="s"),
    "GMC":      dict(color="#009E73", marker="^"),
    "CoMM":     dict(color="#D55E00", marker="D"),
    "FactorCL": dict(color="#CC79A7", marker="P"),
}
_FALLBACK_MARKERS = ["v", "X", "h", "<", ">"]


# Pareto logic
def non_dominated_mask(P, tol=1e-12):
    """True for points no other point Pareto-dominates (maximisation on every column).

    q dominates p  iff  q >= p on all objectives and q > p on at least one.
    Exact duplicates do not dominate each other, so both are kept.
    """
    P = np.asarray(P, float)
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        ge = np.all(P >= P[i] - tol, axis=1)
        gt = np.any(P > P[i] + tol, axis=1)
        dom = ge & gt
        dom[i] = False
        if dom.any():
            keep[i] = False
    return keep


def _projected_delaunay(F):
    """Delaunay triangulation of the front in the plane orthogonal to (1, 1, 1)."""
    from scipy.spatial import Delaunay, QhullError
    e1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    e2 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0)
    uv = np.stack([F @ e1, F @ e2], axis=1)
    try:
        return Delaunay(uv).simplices
    except QhullError:
        return None


def _upper_hull_faces(F, tol=1e-9):
    """Convex-hull facets facing the improvement direction: outward normal . (1,1,1) > 0.

    Requiring every normal component >= 0 would be too strict -- a genuine front facet
    between an R-heavy and a U1-heavy point can tilt away from U2 (normal ~ (.6, .6, -.2)).
    """
    from scipy.spatial import ConvexHull, QhullError
    if len(F) < 4:
        return None
    try:
        hull = ConvexHull(F)
    except QhullError:
        return None
    normals = hull.equations[:, :3]
    faces = hull.simplices[normals.sum(axis=1) > tol]
    return faces if len(faces) else None


def front_triangles(F, method=SURFACE_METHOD, max_edge_frac=MAX_EDGE_FRAC):
    """Index triples into F; None when F cannot form a surface (<3 points, or degenerate)."""
    if len(F) < 3:
        return None
    if method == "convex_hull":
        tris = _upper_hull_faces(F)
        if tris is None:
            warnings.warn("convex_hull could not form a surface; using projected_delaunay")
            tris = _projected_delaunay(F)
    elif method == "projected_delaunay":
        tris = _projected_delaunay(F)
    else:
        raise ValueError(f"unknown SURFACE_METHOD {method!r}")
    if tris is None:
        return None
    if max_edge_frac is not None:
        diam = max(np.linalg.norm(F[:, None] - F[None], axis=-1).max(), 1e-12)
        edges = [max(np.linalg.norm(F[a] - F[b]), np.linalg.norm(F[b] - F[c]),
                     np.linalg.norm(F[a] - F[c])) for a, b, c in tris]
        tris = tris[np.asarray(edges) <= max_edge_frac * diam]
        if not len(tris):
            return None
    return tris


# Plot
def _validate(steer, baselines):
    P = np.asarray(steer, float)
    if P.size == 0:
        raise SystemExit("steer_points is empty -- paste your [R, U1, U2] coordinates at the "
                         "top of the file (or run with --demo to preview the layout).")
    if P.ndim != 2 or P.shape[1] != 3:
        raise SystemExit(f"steer_points must be a list of [R, U1, U2]; got shape {P.shape}")
    B = {}
    for name, v in baselines.items():
        v = np.asarray(v, float).reshape(-1)
        if v.shape != (3,):
            raise SystemExit(f"baseline {name!r} must be [R, U1, U2]; got {v.tolist()}")
        B[name] = v
    allv = np.vstack([P] + list(B.values())) if B else P
    if not np.all(np.isfinite(allv)):
        raise SystemExit("non-finite coordinate found (NaN/inf) -- check the pasted values")
    return P, B


def _limits(allv):
    if AXIS_LIMITS == "chance":
        if CHANCE is None:
            raise SystemExit('AXIS_LIMITS="chance" needs CHANCE set')
        hi = allv.max(0)
        below = [(i, float(allv[:, i].min()), CHANCE[i])
                 for i in range(3) if allv[:, i].min() < CHANCE[i] - 1e-9]
        for i, v, c in below:
            warnings.warn(f"axis {i}: a point at {v:.4f} is BELOW chance {c:.4f} and will "
                          f"fall outside the box; use AXIS_LIMITS='data' to keep it visible")
        return tuple((float(CHANCE[i]), float(hi[i] + max(0.06 * (hi[i] - CHANCE[i]), 1e-3)))
                     for i in range(3))
    if AXIS_LIMITS is not None and AXIS_LIMITS != "data":
        return AXIS_LIMITS
    if AXIS_LIMITS is None and allv.min() >= 0.0 and allv.max() <= 1.0:
        return ((0.0, 1.0),) * 3
    lo, hi = allv.min(0), allv.max(0)
    pad = np.maximum(0.06 * (hi - lo), 1e-3)
    return tuple((float(a - p), float(b + p)) for a, b, p in zip(lo, hi, pad))


def _draw_chance(ax, lims):
    """Reference lines at chance on each axis, drawn on the panes that bound the box.

    Chance differs per axis here (1/15 for the 15-way shape task, 1/10 for the two 10-way
    tasks), so a single global line would be wrong; each axis gets its own, placed on the
    floor for x and y and on the left-hand wall for z.
    """
    if not SHOW_CHANCE or CHANCE is None:
        return None
    (x0, x1), (y0, y1), (z0, z1) = lims
    cx, cy, cz = CHANCE
    kw = dict(color=CHANCE_COLOR, lw=1.1, ls=(0, (4, 3)), alpha=0.9, zorder=2)
    if x0 <= cx <= x1:
        ax.plot([cx, cx], [y0, y1], [z0, z0], **kw)          # floor, constant shape
    if y0 <= cy <= y1:
        ax.plot([x0, x1], [cy, cy], [z0, z0], **kw)          # floor, constant deformation
    if z0 <= cz <= z1:
        ax.plot([x0, x0], [y0, y1], [cz, cz], **kw)          # wall, constant texture
    if SHOW_CHANCE == "floor":
        import matplotlib
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        if x0 <= cx and y0 <= cy:
            quad = [[(x0, y0, z0), (cx, y0, z0), (cx, cy, z0), (x0, cy, z0)]]
            ax.add_collection3d(Poly3DCollection(
                quad, facecolor=matplotlib.colors.to_rgba(CHANCE_COLOR, 0.12),
                edgecolor="none", zorder=1))
    from matplotlib.lines import Line2D
    pct = ", ".join(f"{100 * c:.1f}%" for c in CHANCE)
    return Line2D([], [], color=CHANCE_COLOR, lw=1.1, ls=(0, (4, 3)),
                  label=f"chance ({pct})")


def _style_axes(ax, lims):
    import matplotlib.ticker as mticker
    for axis, (lo, hi) in zip((ax.xaxis, ax.yaxis, ax.zaxis), lims):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))                 # transparent panes
        axis.pane.set_edgecolor((0.72, 0.72, 0.72, 0.9))
        axis.set_major_locator(mticker.MaxNLocator(5))
        if TICKS_AS_PERCENT:                                        # 0.42 -> 42
            axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{100 * v:g}"))
        try:                                                        # minimal grid
            axis._axinfo["grid"].update(color=(0.85, 0.85, 0.85, 1.0), linewidth=0.45)
            axis._axinfo["tick"].update(inward_factor=0.0, outward_factor=0.25)
        except (AttributeError, KeyError):
            pass
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_xlabel(AXIS_LABELS[0], labelpad=12)
    ax.set_ylabel(AXIS_LABELS[1], labelpad=12)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(AXIS_LABELS[2], labelpad=10, rotation=90)
    ax.tick_params(axis="both", which="major", pad=2)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    try:
        ax.set_box_aspect((1, 1, 0.9), zoom=BOX_ZOOM)   # zoom < 1 keeps labels inside
    except (AttributeError, TypeError):
        pass


def plot(steer, baselines, out_stem=OUT_STEM, label_all=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 15,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,                 # editable text in the PDF
        "savefig.facecolor": "white",
    })

    P, B = _validate(steer, baselines)
    allv = np.vstack([P] + list(B.values())) if B else P
    lims = _limits(allv)
    floor = lims[2][0]

    nd = non_dominated_mask(P)
    F, D = P[nd], P[~nd]
    B_names = list(B)
    B_arr = np.array([B[n] for n in B_names]) if B else np.empty((0, 3))
    b_nd = non_dominated_mask(B_arr) if len(B_arr) else np.zeros(0, bool)

    print(f"STEER: {len(P)} points, {len(F)} non-dominated")
    for n, keep in zip(B_names, b_nd):
        print(f"  baseline {n:10s} {'non-dominated' if keep else 'dominated'} among baselines")

    fig = plt.figure(figsize=(6.6, 6.0))
    try:
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)  # draw in add order
    except TypeError:
        ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    _style_axes(ax, lims)

    handles = []

    # 0. chance reference, behind everything: one line per axis, since the 15-way shape task
    # and the two 10-way tasks do not share a chance level
    _ch = _draw_chance(ax, lims)
    if _ch is not None:
        handles.append(_ch)

    # 1. baseline frontier (drawn first, so it sits visually behind everything)
    if SHOW_BASELINE_FRONT and b_nd.sum() >= 2:
        BF = B_arr[b_nd]
        tris = front_triangles(BF) if len(BF) >= 3 else None
        if tris is not None:
            ax.add_collection3d(Poly3DCollection(
                [BF[t] for t in tris], facecolor=(0.55, 0.55, 0.55, 0.10),
                edgecolor=(0.45, 0.45, 0.45, 0.55), linewidths=0.8, linestyles="--"))
        elif len(BF) == 2:                          # a single segment is unambiguous
            ax.plot(*BF.T, color=(0.45, 0.45, 0.45, 0.7), lw=1.0, ls="--")
        handles.append(Line2D([], [], color=(0.45, 0.45, 0.45), lw=1.0, ls="--",
                              label="Baseline frontier"))

    # 2. STEER trade-off surface over its non-dominated points
    tris = front_triangles(F)
    if tris is not None:
        ax.add_collection3d(Poly3DCollection(
            [F[t] for t in tris], facecolor=matplotlib.colors.to_rgba(STEER_COLOR, 0.20),
            edgecolor=matplotlib.colors.to_rgba(STEER_EDGE, 0.40), linewidths=0.6))
        handles.append(Patch(facecolor=matplotlib.colors.to_rgba(STEER_COLOR, 0.25),
                             edgecolor=matplotlib.colors.to_rgba(STEER_EDGE, 0.6),
                             label="STEER trade-off surface"))
    elif len(F) == 2:
        ax.plot(*F.T, color=STEER_COLOR, lw=1.4, alpha=0.6)
    else:
        warnings.warn(f"{len(F)} non-dominated STEER point(s) cannot span a surface; "
                      f"drawing points only")

    # 3. floor projections
    if FLOOR_PROJECTIONS:
        for p in np.vstack([F, B_arr]) if len(B_arr) else F:
            ax.plot([p[0], p[0]], [p[1], p[1]], [floor, p[2]],
                    color=(0.5, 0.5, 0.5, 0.30), lw=0.6, ls=":")
        ax.scatter(F[:, 0], F[:, 1], np.full(len(F), floor), s=10,
                   color=matplotlib.colors.to_rgba(STEER_COLOR, 0.18), edgecolors="none",
                   depthshade=False)

    # 4. baselines. A fixed point can land inside the blue surface and become impossible to
    # place in depth, so each gets a stem down to the floor and a hollow ground marker in
    # its own colour -- the floor position is unambiguous even when the point is occluded.
    for i, (name, v) in enumerate(B.items()):
        st = BASELINE_STYLE.get(name, dict(color=f"C{i + 1}",
                                           marker=_FALLBACK_MARKERS[i % len(_FALLBACK_MARKERS)]))
        if BASELINE_STEMS:
            ax.plot([v[0], v[0]], [v[1], v[1]], [floor, v[2]], color=st["color"],
                    lw=1.0, ls="-", alpha=0.45, zorder=4)
            ax.scatter(v[0], v[1], floor, s=42, marker=st["marker"], facecolors="none",
                       edgecolors=st["color"], linewidths=1.1, depthshade=False, zorder=4)
        ax.scatter(*v, s=BASELINE_MARKER_SIZE, marker=st["marker"], color=st["color"],
                   edgecolors="black", linewidths=0.9, depthshade=False, alpha=1.0, zorder=11)
        handles.append(Line2D([], [], ls="", marker=st["marker"], markersize=8,
                              markerfacecolor=st["color"], markeredgecolor="black",
                              markeredgewidth=0.7, label=name))

    # 5. STEER points last, so they read as the dominant layer
    if SHOW_DOMINATED_STEER and len(D):
        ax.scatter(D[:, 0], D[:, 1], D[:, 2], s=55, color=STEER_LIGHT, edgecolors=STEER_EDGE,
                   linewidths=0.6, depthshade=False, alpha=0.9)
    ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=110, color=STEER_COLOR, edgecolors=STEER_EDGE,
               linewidths=1.0, depthshade=False, zorder=10)

    # 6. the per-task winners: for each objective, the single lambda that scores highest
    def _lam_str(i):
        # Only label when the list lines up row-for-row with the plotted points; otherwise a
        # stale or demo point set would be annotated with someone else's preferences.
        lam = globals().get("steer_lambdas")
        if not lam or len(lam) != len(P) or i >= len(lam):
            return ""
        return "(" + ",".join(f"{c:g}" for c in lam[i]) + ")"

    best = {}
    if HIGHLIGHT_TASK_BEST:
        for j, tag in enumerate(TASK_BEST_TAGS):
            best.setdefault(int(np.argmax(P[:, j])), []).append(tag)
        for i, tags in best.items():
            print(f"  best on {' + '.join(tags):12s} -> point {i}  {P[i].round(4).tolist()}"
                  + (f"  lambda={_lam_str(i)}" if _lam_str(i) else ""))
        BP = P[list(best)]
        ax.scatter(BP[:, 0], BP[:, 1], BP[:, 2], s=190, marker="o", color=TASK_BEST_COLOR,
                   edgecolors=TASK_BEST_EDGE, linewidths=1.2, depthshade=False, zorder=12)
        if TASK_BEST_ANNOTATE:
            # Placed in SCREEN space with a leader line back to the marker. Data-space
            # offsets failed because the three winners sit at different depths, so equal 3D
            # offsets project to unequal on-screen distances and the R / U2 labels landed on
            # the mesh. proj_transform gives the marker's 2D position under the current
            # camera; the canvas is drawn first so that projection is the final one.
            from mpl_toolkits.mplot3d import proj3d
            fig.canvas.draw()
            for i, tags in best.items():
                txt = " ".join(tags)
                _show_lam = TASK_BEST_SHOW_LAMBDA and not (label_all and
                                                           LABEL_ALL_SUPPRESS_BEST_LAMBDA)
                if _show_lam and _lam_str(i):
                    txt += f"\n$\\lambda$={_lam_str(i)}"
                x2, y2, _ = proj3d.proj_transform(P[i, 0], P[i, 1], P[i, 2], ax.get_proj())
                key = tags[0].strip("$")
                _offs = TASK_BEST_OFFSETS_LABEL_ALL if label_all else TASK_BEST_OFFSETS
                dx, dy = _offs.get(key, TASK_BEST_OFFSET_DEFAULT)
                ax.annotate(txt, xy=(x2, y2), xycoords="data",
                            xytext=(dx, dy), textcoords="offset points",
                            color=TASK_BEST_EDGE, fontsize=11, fontweight="bold",
                            ha="center", va="center", linespacing=1.15, zorder=30,
                            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                      edgecolor=TASK_BEST_COLOR, linewidth=0.8, alpha=0.95),
                            arrowprops=dict(arrowstyle="-", color=TASK_BEST_COLOR,
                                            lw=0.9, alpha=0.9,
                                            shrinkA=1, shrinkB=6))

    # 6b. --label-all: write the preference next to EVERY STEER point.
    if label_all:
        from mpl_toolkits.mplot3d import proj3d
        fig.canvas.draw()                       # projection is only final after a draw
        proj2d, pts2d = [], []
        for i in range(len(P)):
            x2, y2, _ = proj3d.proj_transform(P[i, 0], P[i, 1], P[i, 2], ax.get_proj())
            proj2d.append((x2, y2))                       # ANCHOR space (projected data)
            pts2d.append(ax.transData.transform((x2, y2)))  # geometry only, in pixels
        proj2d = np.asarray(proj2d, dtype=float)
        pts2d = np.asarray(pts2d, dtype=float)
        centre = pts2d.mean(axis=0)
        px = fig.dpi / 72.0                      # points -> display pixels

        def _box(cx, cy, text):
            lines = text.split("\n")
            w = 0.62 * LABEL_ALL_FONTSIZE * px * max(len(t) for t in lines)
            h = 1.35 * LABEL_ALL_FONTSIZE * px * len(lines)
            return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

        def _hits(a, b):
            return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

        # Outermost points first: they have the most free space, and placing them early
        # stops an interior label from being pushed out into the region they need.
        order = sorted(range(len(P)),
                       key=lambda i: -float(np.hypot(*(pts2d[i] - centre))))
        placed = []
        keep = None
        if LABEL_ALL_ONLY:
            keep = {tuple(round(float(c), 6) for c in t) for t in LABEL_ALL_ONLY}
        _src = globals().get("steer_lambdas") or []
        for i in order:
            lam = _lam_str(i)
            if not lam:
                continue
            if keep is not None:
                if i >= len(_src):
                    continue
                if tuple(round(float(c), 6) for c in _src[i]) not in keep:
                    continue
            txt = " ".join(best[i]) + " " + lam if (i in best) else lam
            v = pts2d[i] - centre
            n = float(np.hypot(*v))
            u = v / n if n > 1e-6 else np.array([1.0, 0.0])
            for k in range(LABEL_ALL_MAX_TRIES):
                r = (LABEL_ALL_BASE_OFFSET + k * LABEL_ALL_STEP) * px
                c = pts2d[i] + u * r
                bb = _box(c[0], c[1], txt)
                if not any(_hits(bb, q) for q in placed):
                    break
            placed.append(bb)
            off = (c - pts2d[i]) / px            # back to offset POINTS for annotate()
            # Anchor in DATA space like the per-task labels: savefig(bbox_inches="tight")
            # re-renders and crops, so a pixel anchor would detach from the axes and every
            # label would pile up in one corner. Only the collision geometry is in pixels.
            ax.annotate(
                txt, xy=(proj2d[i][0], proj2d[i][1]), xycoords="data",
                xytext=tuple(off), textcoords="offset points",
                color=LABEL_ALL_COLOR, fontsize=LABEL_ALL_FONTSIZE,
                fontweight=("bold" if LABEL_ALL_BOLD else "normal"),
                ha="center", va="center", linespacing=1.15, zorder=29,
                bbox=(dict(boxstyle="round,pad=0.25", facecolor="white",
                           edgecolor=LABEL_ALL_BOX_EDGE, linewidth=0.8, alpha=0.95)
                      if LABEL_ALL_BBOX else None),
                arrowprops=(dict(arrowstyle="-", color=LABEL_ALL_BOX_EDGE, lw=0.9, alpha=0.9,
                                 shrinkA=1, shrinkB=6) if LABEL_ALL_LEADER else None))
        print(f"  --label-all: annotated {len(placed)} of {len(P)} preferences"
              + (f"  (restricted to {len(keep)}: "
                 + ", ".join("(" + ",".join(f"{c:g}" for c in t) + ")"
                             for t in LABEL_ALL_ONLY) + ")" if keep is not None else ""))

    steer_handles = [Line2D([], [], ls="", marker="o", markersize=10, markerfacecolor=STEER_COLOR,
                            markeredgecolor=STEER_EDGE, markeredgewidth=1.0,
                            label=r"STEER ($\lambda$ family)")]
    if best:
        steer_handles.append(Line2D([], [], ls="", marker="o", markersize=11,
                                    markerfacecolor=TASK_BEST_COLOR, markeredgecolor=TASK_BEST_EDGE,
                                    markeredgewidth=1.2, label="STEER best per task"))
    if SHOW_DOMINATED_STEER and len(D):
        steer_handles.append(Line2D([], [], ls="", marker="o", markersize=7.5,
                                    markerfacecolor=STEER_LIGHT, markeredgecolor=STEER_EDGE,
                                    markeredgewidth=0.6, label="STEER (dominated)"))
    handles = steer_handles + handles

    # above the axes, outside the 3D box, so it can never cover points or tick labels
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.97)
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
                    ncol=LEGEND_NCOL, columnspacing=1.1,
                    frameon=True, framealpha=1.0, edgecolor=(0.8, 0.8, 0.8),
                    handletextpad=0.5, borderpad=0.5, labelspacing=0.35)
    leg.get_frame().set_linewidth(0.6)

    for ext, kw in (("pdf", {}), ("png", {"dpi": PNG_DPI})):
        fig.savefig(f"{out_stem}.{ext}", bbox_inches="tight", pad_inches=0.12, **kw)
    plt.close(fig)
    print(f"wrote {out_stem}.pdf and {out_stem}.png")


# Demo data -- SYNTHETIC, for previewing the layout only. Never used without --demo.
def _demo():
    rng = np.random.default_rng(7)
    pts = []
    for a in range(5):                                # the 15-point lambda simplex, side 5
        for b in range(5 - a):
            w = np.array([a, b, 4 - a - b]) / 4.0
            pts.append(0.45 + 0.42 * np.sqrt(w) + rng.normal(0, 0.012, 3))
    pts.append([0.62, 0.58, 0.60]); pts.append([0.70, 0.52, 0.55])   # two dominated
    base = {"CLIP": [0.80, 0.42, 0.40], "GMC": [0.74, 0.50, 0.47],
            "CoMM": [0.70, 0.46, 0.55], "FactorCL": [0.66, 0.60, 0.52]}
    return np.clip(np.array(pts), 0, 1).tolist(), base


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--demo", action="store_true",
                    help="render SYNTHETIC demo coordinates to preview the layout")
    ap.add_argument("--out", default=None, help=f"output stem (default {OUT_STEM})")
    ap.add_argument("--label-all", "--label_all", dest="label_all", action="store_true",
                    help="variant B: print the preference next to EVERY STEER point, not "
                         "only the per-task winners. Labels are auto-placed radially "
                         "outward with collision avoidance. Writes to a DIFFERENT default "
                         "stem so the main figure is not overwritten.")
    a = ap.parse_args()
    suffix = "_alllambda" if a.label_all else ""
    if a.demo:
        warnings.warn("--demo: plotting SYNTHETIC coordinates, not results")
        p, b = _demo()
        plot(p, b, a.out or OUT_STEM + "_DEMO" + suffix, label_all=a.label_all)
    else:
        plot(steer_points, baseline_points, a.out or OUT_STEM + suffix,
             label_all=a.label_all)


if __name__ == "__main__":
    main()

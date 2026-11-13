"""
========================================================================
  Reinforced Concrete M-P Interaction Diagram - Tunnel Portal Section
  Structural Engineer Automation Script
  Method : Strain Compatibility + Equivalent Rectangular Stress Block
  Code   : EN 1992-1-1 (Eurocode 2) conventions
  Current fixes:
    1. Analytic normalized D/C ratio from demand ray to envelope boundary
    2. Status based on D/C <= 1.0
    3. Mutually exclusive zone map
    4. Dynamic load-combination discovery from the STAAD ANL file
    5. EC2 design factors retained: gamma_c = 1.5, gamma_s = 1.15
========================================================================
"""
import re
import warnings
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
from matplotlib.path import Path as MPath
from scipy.spatial import ConvexHull
warnings.filterwarnings("ignore")
def build_section(
    fck=25.0,
    fy=500.0,
    h=600.0,
    b=1000.0,
    As_outer=1570.8,
    As_inner=1570.8,
    cover=60.0,
):
    """
    Returns a dict of section properties.
    Units: MPa, mm.
    Cover = nominal cover to bar centre.
    """
    Es = 200_000.0
    fcd = fck / 1.5   # gamma_c = 1.5
    fyd = fy / 1.15   # gamma_s = 1.15
    eta = 1.0 if fck <= 50 else 1.0 - (fck - 50) / 200
    lam = 0.8 if fck <= 50 else 0.8 - (fck - 50) / 400
    return dict(
        fck=fck,
        fy=fy,
        h=h,
        b=b,
        As_outer=As_outer,
        As_inner=As_inner,
        cover=cover,
        Es=Es,
        fcd=fcd,
        fyd=fyd,
        eta=eta,
        lam=lam,
        d_outer=cover,
        d_inner=h - cover,
        eps_cu=0.0035,
    )
def compute_interaction_curve(sec, n_points=500):
    """
    Returns M_arr [kNm/m] and P_arr [kN/m] for a symmetric RC section.
    Sign convention: P > 0 = compression, M >= 0.
    """
    fcd, fyd = sec["fcd"], sec["fyd"]
    eta, lam = sec["eta"], sec["lam"]
    eps_cu = sec["eps_cu"]
    Es = sec["Es"]
    h, b = sec["h"], sec["b"]
    As_o, As_i = sec["As_outer"], sec["As_inner"]
    d_o, d_i = sec["d_outer"], sec["d_inner"]
    d_ref = h / 2.0
    M_list, P_list = [], []
    P0 = eta * fcd * b * h + (As_o + As_i) * (fyd - eta * fcd)
    M_list.append(0.0)
    P_list.append(P0 / 1e3)
    for c in np.linspace(0.04 * h, 1.5 * h, n_points):
        x = min(lam * c, h)
        Fc = eta * fcd * b * x
        eps_o = eps_cu * (c - d_o) / c
        eps_i = eps_cu * (c - d_i) / c
        sig_o = float(np.clip(Es * eps_o, -fyd, fyd))
        sig_i = float(np.clip(Es * eps_i, -fyd, fyd))
        sig_o_net = sig_o - (eta * fcd if eps_o > 0 else 0.0)
        Fs_o = As_o * sig_o_net
        Fs_i = As_i * sig_i
        P_n = Fc + Fs_o + Fs_i
        M_n = (
            Fc * (d_ref - x / 2.0)
            + Fs_o * (d_ref - d_o)
            + Fs_i * (d_ref - d_i)
        )
        M_list.append(abs(M_n) / 1e6)
        P_list.append(P_n / 1e3)
    Pt = -(As_o + As_i) * fyd
    M_list.append(0.0)
    P_list.append(Pt / 1e3)
    return np.array(M_list), np.array(P_list)
def envelope_vertices(M_full, P_full):
    """Return closed convex-hull vertices for the mirrored M-P envelope."""
    hull_pts = np.column_stack([M_full, P_full])
    hull_obj = ConvexHull(hull_pts)
    verts = hull_pts[hull_obj.vertices]
    return np.vstack([verts, verts[0]])
def make_full_envelope(sec):
    """
    Build the complete mirrored (+/-M) convex-hull path for a section.
    Returns M_full, P_full, and a closed MPath.
    """
    M_cap, P_cap = compute_interaction_curve(sec)
    idx = np.argsort(P_cap)[::-1]
    M_s, P_s = M_cap[idx], P_cap[idx]
    M_full = np.concatenate([M_s, -M_s[::-1]])
    P_full = np.concatenate([P_s, P_s[::-1]])
    verts = envelope_vertices(M_full, P_full)
    path = MPath(verts)
    return M_full, P_full, path
def dc_ratio(M_d, P_d, M_full, P_full):
    """
    Intersect the demand ray with the normalized convex envelope and
    return D/C = demand distance / boundary distance along that ray.
    """
    if M_d == 0 and P_d == 0:
        return 0.0
    M_scale = np.max(np.abs(M_full))
    P_scale = np.max(np.abs(P_full))
    if M_scale <= 0 or P_scale <= 0:
        return np.inf
    demand = np.array([M_d / M_scale, P_d / P_scale], dtype=float)
    if np.allclose(demand, 0.0):
        return 0.0
    verts = envelope_vertices(M_full, P_full).astype(float)
    verts[:, 0] /= M_scale
    verts[:, 1] /= P_scale
    def cross_2d(a, b):
        return a[0] * b[1] - a[1] * b[0]
    tol = 1e-10
    t_hits = []
    for a, b in zip(verts[:-1], verts[1:]):
        edge = b - a
        denom = cross_2d(demand, edge)
        if abs(denom) <= tol:
            continue
        t_hit = cross_2d(a, edge) / denom
        u_hit = cross_2d(a, demand) / denom
        if t_hit >= -tol and -tol <= u_hit <= 1.0 + tol:
            t_hits.append(max(t_hit, 0.0))
    if not t_hits:
        return np.inf
    t_boundary = min(t_hits)
    return np.inf if t_boundary <= tol else 1.0 / t_boundary
def discover_combo_labels(lines):
    """Discover load-combination IDs and titles from the STAAD ANL file."""
    combo_labels = {}
    pattern = re.compile(r"\bLOAD\s+COMB\s+(\d+)\b(?:\s+(.*))?$", re.IGNORECASE)
    for line in lines:
        match = pattern.search(line.strip())
        if not match:
            continue
        combo = int(match.group(1))
        title = (match.group(2) or "").strip()
        combo_labels[combo] = f"{combo} {title}" if title else f"Combo {combo}"
    return combo_labels
def build_combo_styles(combo_ids, combo_labels=None):
    """Generate stable labels and marker styles for any combo set."""
    combo_labels = combo_labels or {}
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h"]
    color_cycle = [
        "#00e676",
        "#ffab40",
        "#ff4081",
        "#40c4ff",
        "#ffd740",
        "#7c4dff",
        "#64ffda",
        "#ff6e40",
        "#c6ff00",
        "#18ffff",
        "#ff80ab",
    ]
    combo_styles = {}
    for idx, combo in enumerate(combo_ids):
        combo_styles[combo] = {
            "label": combo_labels.get(combo, f"Combo {combo}"),
            "marker": marker_cycle[idx % len(marker_cycle)],
            "color": color_cycle[idx % len(color_cycle)],
        }
    return combo_styles
def parse_staad_anl(filepath, target_combos=None):
    """
    Parse STAAD.Pro .ANL file and return:
      1. member demand list
      2. combo label mapping
    MEMBER END FORCES rows appear in three token layouts:
      n=9 : MEMB LOAD JT AXIAL SY SZ TOR MY MZ
      n=8 : LOAD JT AXIAL SY SZ TOR MY MZ
      n=7 : JT AXIAL SY SZ TOR MY MZ
    """
    with open(filepath, encoding="latin-1") as f:
        lines = f.readlines()
    combo_labels = discover_combo_labels(lines)
    target_set = set(target_combos) if target_combos is not None else None
    if target_set is None and combo_labels:
        target_set = set(combo_labels)
    demand_dict = {}
    in_section = False
    cur_member = None
    cur_load = None
    for line in lines:
        stripped = line.strip()
        if "MEMBER END FORCES" in stripped.upper():
            in_section = True
            continue
        if in_section and (
            "BEAM DESIGN" in stripped.upper() or
            "COLUMN DESIGN" in stripped.upper()
        ):
            in_section = False
            continue
        if not in_section:
            continue
        parts = stripped.split()
        if not parts:
            continue
        try:
            int(parts[0])
        except ValueError:
            continue
        n = len(parts)
        if n == 9:
            try:
                cur_member = int(parts[0])
                cur_load = int(parts[1])
                axial = float(parts[3])
                mz = float(parts[8])
            except (ValueError, IndexError):
                continue
        elif n == 8:
            try:
                cur_load = int(parts[0])
                axial = float(parts[2])
                mz = float(parts[7])
            except (ValueError, IndexError):
                continue
        elif n == 7:
            try:
                axial = float(parts[1])
                mz = float(parts[6])
            except (ValueError, IndexError):
                continue
        else:
            continue
        if cur_member is None:
            continue
        if target_set is not None and cur_load not in target_set:
            continue
        key = (cur_member, cur_load)
        if key not in demand_dict:
            demand_dict[key] = {"P_vals": [], "M_vals": []}
        demand_dict[key]["P_vals"].append(axial)
        demand_dict[key]["M_vals"].append(abs(mz))
    demand = []
    for (memb, combo), vals in demand_dict.items():
        if not vals["M_vals"]:
            continue
        idx = int(np.argmax(vals["M_vals"]))
        demand.append(
            {
                "member": memb,
                "combo": combo,
                "M": vals["M_vals"][idx],      # kNm/m
                "P": abs(vals["P_vals"][idx]), # kN/m (+ve = compression)
            }
        )
    demand.sort(key=lambda pt: (pt["combo"], pt["member"]))
    present_combos = {pt["combo"] for pt in demand}
    ordered_combos = [combo for combo in combo_labels if combo in present_combos]
    ordered_combos.extend(sorted(present_combos - set(ordered_combos)))
    filtered_labels = {
        combo: combo_labels.get(combo, f"Combo {combo}")
        for combo in ordered_combos
    }
    print(f"Parsed {len(demand)} demand points from: {filepath}")
    return demand, filtered_labels
ZONE_MAP = {
    "Raft": list(range(1, 5)),
    "Walls": list(range(5, 11)),
    "Arch": list(range(11, 17)),
}
def plot_interaction_diagram(sec_raft, sec_wall, demand_data, combo_labels=None):
    """
    Plot the two section envelopes and the governing demand points.
    Demand points are green for OK (D/C <= 1.0) and red for FAIL.
    """
    M_raft, P_raft, path_raft = make_full_envelope(sec_raft)
    M_wall, P_wall, path_wall = make_full_envelope(sec_wall)
    def get_path(member):
        return path_raft if member in ZONE_MAP["Raft"] else path_wall
    def get_envelope(member):
        if member in ZONE_MAP["Raft"]:
            return M_raft, P_raft
        return M_wall, P_wall
    present_combos = {pt["combo"] for pt in demand_data}
    combo_ids = [combo for combo in (combo_labels or {}) if combo in present_combos]
    combo_ids.extend(sorted(present_combos - set(combo_ids)))
    combo_styles = build_combo_styles(combo_ids, combo_labels)
    zone_dc = {zone: [] for zone in ZONE_MAP}
    for pt in demand_data:
        path = get_path(pt["member"])
        M_env, P_env = get_envelope(pt["member"])
        ratio = dc_ratio(pt["M"], pt["P"], M_env, P_env)
        inside_geom = path.contains_point((pt["M"], pt["P"]))
        status_ok = ratio <= 1.0
        pt["inside_geom"] = inside_geom
        pt["status_ok"] = status_ok
        pt["inside"] = status_ok
        pt["dc"] = ratio
        for zone, members in ZONE_MAP.items():
            if pt["member"] in members:
                zone_dc[zone].append(
                    (pt["member"], pt["combo"], pt["M"], pt["P"], ratio, status_ok)
                )
    fig = plt.figure(figsize=(20, 14), facecolor="#0d1117")
    gs = GridSpec(
        3,
        3,
        figure=fig,
        left=0.07,
        right=0.97,
        top=0.93,
        bottom=0.07,
        hspace=0.50,
        wspace=0.35,
    )
    ax_mp = fig.add_subplot(gs[:, :2])
    ax_sec = fig.add_subplot(gs[0, 2])
    ax_tbl = fig.add_subplot(gs[1:, 2])
    ax = ax_mp
    ax.set_facecolor("#0d1117")
    ax.fill(M_raft, P_raft, color="#1a3a5c", alpha=0.40, zorder=1)
    ax.plot(
        M_raft,
        P_raft,
        color="#4fc3f7",
        lw=2.0,
        zorder=2,
        label=f"Capacity - Raft  h={sec_raft['h']:.0f} mm",
    )
    ax.fill(M_wall, P_wall, color="#1a4a2c", alpha=0.35, zorder=1)
    ax.plot(
        M_wall,
        P_wall,
        color="#69f0ae",
        lw=2.0,
        zorder=2,
        ls="--",
        label=f"Capacity - Wall/Arch  h={sec_wall['h']:.0f} mm",
    )
    for P_arr, col, label in [
        (P_raft, "#4fc3f7", f"Squash Raft {max(P_raft):,.0f} kN"),
        (P_wall, "#69f0ae", f"Squash Wall {max(P_wall):,.0f} kN"),
    ]:
        ax.axhline(max(P_arr), color=col, lw=0.7, ls=":", alpha=0.5)
        ax.text(max(M_raft) * 0.52, max(P_arr) * 1.008, label, color=col, fontsize=7.5, va="bottom")
    ax.axhline(0, color="#666", lw=0.5, ls="--", alpha=0.4)
    ax.axvline(0, color="#666", lw=0.5, ls="--", alpha=0.4)
    for M_env, P_env, col, sec in [
        (M_raft, P_raft, "#ffd54f", sec_raft),
        (M_wall, P_wall, "#ffe082", sec_wall),
    ]:
        ib = int(np.argmax(M_env))
        ax.scatter(
            M_env[ib],
            P_env[ib],
            s=110,
            color=col,
            marker="D",
            zorder=6,
            edgecolors="white",
            lw=0.6,
            label=f"Balanced  h={sec['h']:.0f}: ({M_env[ib]:.0f} kNm, {P_env[ib]:.0f} kN)",
        )
    seen = set()
    for pt in demand_data:
        combo = pt["combo"]
        style = combo_styles[combo]
        face = "#00e676" if pt["status_ok"] else "#ff1744"
        size = 80 if pt["status_ok"] else 100
        width = 0.5 if pt["status_ok"] else 0.8
        label = None
        tag = (combo, pt["status_ok"])
        if tag not in seen:
            status = "OK" if pt["status_ok"] else "FAIL"
            label = f"{style['label']} - {status}"
            seen.add(tag)
        ax.scatter(
            pt["M"],
            pt["P"],
            c=face,
            s=size,
            marker=style["marker"],
            edgecolors="white",
            lw=width,
            zorder=5,
            label=label,
        )
    for pt in demand_data:
        ax.annotate(
            str(pt["member"]),
            xy=(pt["M"], pt["P"]),
            xytext=(3, 3),
            textcoords="offset points",
            color="#cccccc",
            fontsize=6,
            alpha=0.75,
        )
    ax.set_xlabel("Moment  M  [kNm/m]", color="white", fontsize=11)
    ax.set_ylabel("Axial Force  P  [kN/m]  (+ve = Compression)", color="white", fontsize=11)
    ax.set_title(
        "M-P Interaction Diagram - Tunnel Portal (EC2, fck=25 MPa, T16@200 EF)",
        color="white",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(color="#2a2a2a", linestyle="--", linewidth=0.5)
    ax.legend(
        fontsize=7.5,
        loc="lower right",
        facecolor="#1a1a2e",
        edgecolor="#555",
        labelcolor="white",
        ncol=1,
        framealpha=0.85,
    )
    ax2 = ax_sec
    ax2.set_facecolor("#0d1117")
    ax2.set_xlim(-0.15, 1.25)
    ax2.set_ylim(-0.15, 1.15)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Typical Cross-Section (Wall / Arch)", color="white", fontsize=8.5, pad=5)
    ax2.add_patch(Rectangle((0, 0), 1, 1, fc="#546e7a", ec="#90a4ae", lw=1.5))
    sec_w = sec_wall
    cn = sec_w["cover"] / sec_w["h"]
    for y0 in [0, 1 - cn]:
        ax2.add_patch(Rectangle((0, y0), 1, cn, fc="none", ec="#b0bec5", ls=":", lw=0.7))
    for xi in np.linspace(0.1, 0.9, 5):
        for yi in [cn, 1 - cn]:
            ax2.add_patch(Circle((xi, yi), 0.032, fc="#f06292", ec="white", lw=0.5, zorder=5))
    ax2.annotate("", xy=(1.1, 0), xytext=(1.1, 1), arrowprops=dict(arrowstyle="<->", color="white", lw=1.0))
    ax2.text(1.14, 0.5, f"h={sec_w['h']:.0f}mm", color="white", fontsize=7, va="center", rotation=90)
    ax2.text(0.5, -0.10, f"b={sec_w['b']:.0f}mm", color="white", fontsize=7, ha="center")
    ax2.text(
        0.5,
        0.50,
        f"fck = {sec_w['fck']} MPa\n"
        f"fy  = {sec_w['fy']} MPa\n"
        f"fcd = {sec_w['fcd']:.2f} MPa\n"
        f"fyd = {sec_w['fyd']:.2f} MPa\n"
        f"As(o) = {sec_w['As_outer']:.0f} mm^2\n"
        f"As(i) = {sec_w['As_inner']:.0f} mm^2\n"
        f"Cover = {sec_w['cover']:.0f} mm",
        color="#e0e0e0",
        fontsize=6.8,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2e", ec="#444", alpha=0.85),
    )
    ax3 = ax_tbl
    ax3.set_facecolor("#0d1117")
    ax3.axis("off")
    ax3.set_title("D/C Ratio Summary (Worst 3 per Zone)", color="white", fontsize=8.5, pad=5)
    headers = ["Zone", "Mem", "Comb", "M\n[kNm]", "P\n[kN]", "D/C", "Status"]
    col_w = [0.15, 0.08, 0.10, 0.14, 0.14, 0.14, 0.13]
    y0, rh = 0.97, 0.062
    for j, (txt, cw) in enumerate(zip(headers, col_w)):
        ax3.text(
            sum(col_w[:j]) + cw / 2,
            y0,
            txt,
            color="#4fc3f7",
            fontsize=7.5,
            ha="center",
            va="top",
            fontweight="bold",
            transform=ax3.transAxes,
        )
    y0 -= rh * 1.3
    ax3.axhline(y0 + rh * 0.3, color="#444", lw=0.8)
    rows = []
    for zone, dcs in zone_dc.items():
        if not dcs:
            rows.append([zone, "-", "-", "-", "-", "-", "-"])
            continue
        dcs_s = sorted(dcs, key=lambda x: x[4], reverse=True)
        for k, (mem, combo, M_val, P_val, ratio, _ok) in enumerate(dcs_s[:3]):
            rows.append(
                [
                    zone if k == 0 else "",
                    str(mem),
                    str(combo),
                    f"{M_val:.1f}",
                    f"{P_val:.1f}",
                    f"{ratio:.3f}",
                    "OK" if ratio <= 1.0 else "FAIL",
                ]
            )
    for i, row in enumerate(rows):
        ax3.axhspan(
            y0 - rh * 0.1,
            y0 + rh * 0.9,
            facecolor="#161b22" if i % 2 == 0 else "#0d1117",
            transform=ax3.transAxes,
            alpha=0.45,
        )
        for j, (cell, cw) in enumerate(zip(row, col_w)):
            col = "#e0e0e0"
            if cell == "OK":
                col = "#00e676"
            elif cell == "FAIL":
                col = "#ff1744"
            else:
                try:
                    v = float(cell)
                    if j == 5:
                        col = "#ff1744" if v > 1.0 else "#ffd54f" if v > 0.85 else "#00e676"
                except (ValueError, TypeError):
                    pass
            ax3.text(
                sum(col_w[:j]) + cw / 2,
                y0,
                cell,
                color=col,
                fontsize=7,
                ha="center",
                va="top",
                transform=ax3.transAxes,
            )
        y0 -= rh
    fig.text(
        0.5,
        0.005,
        f"Raft: fck={sec_raft['fck']} MPa  h={sec_raft['h']:.0f}mm  "
        f"As={sec_raft['As_outer']:.0f}mm^2  fcd={sec_raft['fcd']:.2f}  "
        f"fyd={sec_raft['fyd']:.2f} MPa  |  "
        f"Wall/Arch: h={sec_wall['h']:.0f}mm  "
        f"As={sec_wall['As_outer']:.0f}mm^2  cover={sec_wall['cover']:.0f}mm",
        ha="center",
        color="#777",
        fontsize=7.5,
        bbox=dict(boxstyle="round", fc="#161b22", ec="#333", alpha=0.6),
    )
    plt.suptitle(
        "Tunnel Portal - RC M-P Interaction Diagram  |  False_tunnel_corrected_v2.STD",
        color="#90caf9",
        fontsize=13,
        fontweight="bold",
        y=0.978,
    )
    out = "tunnel_portal_interaction.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="#0d1117")
    print(f"Plot saved to {out}")
    plt.show()
    print("\n" + "=" * 76)
    print("  D/C RATIO SUMMARY  -  TUNNEL PORTAL")
    print("=" * 76)
    print(f"{'Zone':<10} {'Mem':>4} {'Combo':>6} {'M [kNm]':>9} {'P [kN]':>9} {'D/C':>7}  Status")
    print("-" * 76)
    for zone, dcs in zone_dc.items():
        if not dcs:
            print(f"  {zone:<8}  - no data -")
            continue
        dcs_s = sorted(dcs, key=lambda x: x[4], reverse=True)
        for k, (mem, combo, M_val, P_val, ratio, _ok) in enumerate(dcs_s):
            zlbl = zone if k == 0 else ""
            status = "OK" if ratio <= 1.0 else "FAIL"
            print(f"  {zlbl:<8} {mem:>4} {combo:>6} {M_val:>9.2f} {P_val:>9.2f} {ratio:>7.3f}  {status}")
        best = dcs_s[0]
        print(f"  {'':8} -> MAX D/C: Mem {best[0]}  Combo {best[1]}  D/C = {best[4]:.3f}")
        print("-" * 76)
    return zone_dc
if __name__ == "__main__":
    ANL_FILE = r"sample_input.ANL"
    sec_raft = build_section(
        fck=25.0,
        fy=500.0,
        h=800.0,
        b=1000.0,
        As_outer=1005.0,
        As_inner=1005.0,
        cover=60.0,
    )
    sec_wall = build_section(
        fck=25.0,
        fy=500.0,
        h=600.0,
        b=1000.0,
        As_outer=1005.0,
        As_inner=1005.0,
        cover=60.0,
    )
    demand, combo_labels = parse_staad_anl(ANL_FILE)
    if not demand:
        print("No demand data parsed - check ANL file path.")
    else:
        plot_interaction_diagram(sec_raft, sec_wall, demand, combo_labels=combo_labels)

#!/usr/bin/env python3
"""
Qubit State Space Trajectory Control
====================================
Closed loop state feedback control of a single decohering qubit, written as a
classical control problem.

PLANT
    The Bloch equations of a driven qubit with T1 and T2 relaxation:

        r_dot = A0 r + b0 + Omega x r           r = (x, y, z) Bloch vector

    * A0 r + b0   : decoherence (open system part), affine and LINEAR in r
    * Omega x r   : coherent drive (the control input), BILINEAR in (Omega, r)
    * Omega = (Omega_x, Omega_y, 0) are the two resonant microwave quadratures
      (rotating frame, zero detuning), our two control inputs u1, u2.

CONTROLLER (classical state feedback with eigenvalue placement)
        u = u_ss  -  K (r - r_ref)
    * r_ref : reference equilibrium (operating point) closest to the target state
    * u_ss  : feedforward "trim" input that holds r_ref against decoherence
    * K     : gain matrix from pole placement on the linearised model

PHYSICS NOTE (important for interpreting the results)
    A coherent drive only ROTATES the Bloch vector, it cannot change its length.
    Only T1/T2 change the length. Hence:
      - the two ANGULAR directions are fully controllable and can be given any
        eigenvalues (fast, set by K);
      - the RADIAL direction (purity) is only weakly controllable: its eigenvalue
        stays close to the decoherence rate, no matter how large K is. In control
        language it is a stable but practically uncontrollable mode (the system
        is stabilisable, not controllable in practice).
      - a static drive can only hold states on the ellipsoid
        gamma2 (x^2 + y^2) = gamma1 z (1 - z). So a pure target state cannot be
        held forever. The best held state along the target direction has length
        s < 1, which sets a hard decoherence limited ceiling F = (1 + s) / 2 on
        the long time fidelity. The tracking error to the reachable reference
        still goes to zero.

Units: time in microseconds, rates in 1/us, drive amplitudes in rad/us (hbar = 1).
Convention (QuTiP): ground state |0> = basis(2, 0) sits at the NORTH pole (+z).

Usage:  python qubit_state_control.py [--outdir results] [--wn 10] [--zeta 0.6]
Outputs: figures (png and pdf), metrics.json and metrics.tex (macros used by main.tex).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.signal import place_poles

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable

import qutip as qt


# =============================================================================
# 0. PARAMETERS
# =============================================================================
@dataclass
class Params:
    # ---- environment (the "plant disturbance" parameters) --------------------
    T1: float = 50.0        # us, energy relaxation time. Sets gamma1 = 1/T1.
    T2: float = 70.0        # us, total coherence time. Sets gamma2 = 1/T2. Needs T2 <= 2 T1.
    # ---- target state: cos(theta/2)|0> + exp(i phi) sin(theta/2)|1> ----------
    theta_deg: float = 45.0   # polar angle of the target on the Bloch sphere
    phi_deg: float = 60.0     # azimuth of the target (uses BOTH control channels)
    # ---- desired closed loop dynamics (the design specification) -------------
    wn: float = 10.0        # rad/us, natural frequency of the placed complex pole pair
    zeta: float = 0.6       # damping ratio of the placed pair (about 9.5 % overshoot for a 2nd order prototype)
    # ---- simulation window ---------------------------------------------------
    t_end: float = 600.0    # us (about 10 radial time constants, enough to reach steady state)
    t_fast: float = 3.0     # us, window used to zoom on the fast transient


# =============================================================================
# 1. MODEL: BLOCH EQUATIONS AS A STATE SPACE MODEL
# =============================================================================
def skew(w: np.ndarray) -> np.ndarray:
    """Matrix form of the cross product: skew(w) @ r == np.cross(w, r).

    Control meaning: this is how the drive vector Omega enters the state matrix.
    A constant drive Omega adds skew(Omega) to the dynamics matrix (a pure
    rotation, so it contributes imaginary eigenvalues and no damping).
    """
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def decoherence_model(p: Params):
    """Return (A0, b0, gamma1, gamma2, gamma_phi).

    gamma1    = 1/T1   amplitude damping rate. Pulls z toward +1 (ground state).
                Classical analogue: the real pole of a first order lag, time constant T1.
    gamma2    = 1/T2   decay rate of the transverse components x and y.
    gamma_phi = gamma2 - gamma1/2   pure dephasing rate (must be >= 0, i.e. T2 <= 2 T1).
                Dephasing shrinks x, y without moving population.

    A0 = diag(-gamma2, -gamma2, -gamma1) is the drift (open loop, undriven) matrix: a
    stable diagonal system, eigenvalues -1/T2, -1/T2, -1/T1. b0 = (0, 0, gamma1)
    is a constant affine offset (the equilibrium of the undriven qubit is the ground
    state r = (0, 0, 1)). Because of this offset the exact model is affine, not strictly
    linear. It becomes linear in the error variable r - r_ref (see section 3).
    """
    g1 = 1.0 / p.T1
    g2 = 1.0 / p.T2
    gphi = g2 - 0.5 * g1
    if gphi < -1e-15:
        raise ValueError("Unphysical coherence times: need T2 <= 2*T1.")
    A0 = np.diag([-g2, -g2, -g1])
    b0 = np.array([0.0, 0.0, g1])
    return A0, b0, g1, g2, gphi


def augmented_state_space(A0, b0, Omega):
    """Homogeneous 4x4 LTI form for a CONSTANT drive Omega.

    d/dt [r; 1] = [[A0 + skew(Omega), b0], [0, 0]] [r; 1]

    Appending the constant 1 as a fourth state turns the affine Bloch equation into a
    textbook LTI system x_dot = A x. The fourth eigenvalue is 0 (the constant itself).
    The state transition matrix is then simply expm(A_aug t).
    """
    A_aug = np.zeros((4, 4))
    A_aug[:3, :3] = A0 + skew(Omega)
    A_aug[:3, 3] = b0
    return A_aug


# =============================================================================
# 2. TARGET, REFERENCE EQUILIBRIUM AND FEEDFORWARD
# =============================================================================
def reference_equilibrium(p: Params, A0, b0):
    """Find the equilibrium r_ref along the target direction and the input u_ss that holds it.

    Equilibrium condition (r_dot = 0):  A0 r + b0 + Omega x r = 0.
    Dotting with r kills the rotation term:  r . (A0 r + b0) = 0.
    For r = s n (n = unit target direction) this gives the reachable length

        s = -(n . b0) / (n . A0 n)       (scalar, 0 < s <= 1)

    Then Omega x r_ref = d with d = -(A0 r_ref + b0), solved with Omega_z = 0 (resonant drive):

        Omega_x = -d_y / r_z ,   Omega_y = d_x / r_z

    u_ss = (Omega_x, Omega_y) is the steady state input of the regulator, the standard
    "feedforward trim" that cancels the constant disturbance (here: decoherence).
    """
    th, ph = np.deg2rad(p.theta_deg), np.deg2rad(p.phi_deg)
    if not (0.0 < th < np.pi / 2):
        raise ValueError("theta must be in (0, 90) degrees: a static drive cannot hold z <= 0 "
                         "against T1 decay (equilibria of the driven damped qubit have z > 0).")
    n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])  # target Bloch vector
    s = -(n @ b0) / (n @ A0 @ n)
    r_ref = s * n
    d = -(A0 @ r_ref + b0)
    Om_ref = np.array([-d[1] / r_ref[2], d[0] / r_ref[2], 0.0])
    residual = np.linalg.norm(A0 @ r_ref + b0 + np.cross(Om_ref, r_ref))
    assert residual < 1e-12, "reference is not an equilibrium"
    return n, s, r_ref, Om_ref


# =============================================================================
# 3. LINEARISATION, CONTROLLABILITY AND EIGENVALUE PLACEMENT
# =============================================================================
def linearise(A0, r_ref, Om_ref):
    """Small signal model around (r_ref, u_ss) for the error e = r - r_ref, input du = u - u_ss.

        e_dot = A_p e + B du
        A_p = A0 + skew(Om_ref)     (Jacobian of the dynamics w.r.t. the state)
        B   = -skew(r_ref)[:, :2]   (Jacobian w.r.t. the two drive quadratures, since Om x r = -r x Om)

    A_p is the "plant matrix" of the classical state space form. Its open loop
    eigenvalues are lightly damped (rates of order 1/T), i.e. the qubit is nearly marginal.
    """
    A_p = A0 + skew(Om_ref)
    B = -skew(r_ref)[:, :2]
    return A_p, B


def tangent_basis(n):
    """Orthonormal basis E = [e1 e2] of the plane perpendicular to the target direction n.

    These are the ANGULAR error coordinates. The radial coordinate is n itself.
    """
    e1 = np.cross(n, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return np.column_stack([e1, e2])


def design_state_feedback(A_p, B, n, wn, zeta):
    """Eigenvalue placement on the angular (tangent) subsystem; returns K (2x3) and diagnostics.

    Why only the tangent block?  Rotations cannot change the Bloch vector length, so the radial
    direction has almost no control authority (B has no component along n). Forcing it with
    place_poles would need absurdly large gains. So we place the two controllable eigenvalues
    exactly and let the radial mode keep its natural, decoherence limited eigenvalue.

        p_{1,2} = -zeta wn  +/-  j wn sqrt(1 - zeta^2)

    Mapping to classical specs (2nd order prototype):
        zeta  -> percent overshoot  Mp = exp(-pi zeta / sqrt(1 - zeta^2))
        wn    -> speed:  rise time ~ (1 + 1.1 zeta + 1.4 zeta^2)/wn, settling (2 %) ~ 4/(zeta wn)
    The gain acts only on the tangent components of the error: K = Kt E^T.
    """
    E = tangent_basis(n)
    A_tt = E.T @ A_p @ E           # 2x2 angular plant block
    B_t = E.T @ B                  # 2x2 input matrix on the angular block (invertible: both angles are steerable)
    poles = [complex(-zeta * wn, wn * np.sqrt(1 - zeta ** 2)),
             complex(-zeta * wn, -wn * np.sqrt(1 - zeta ** 2))]
    Kt = place_poles(A_tt, B_t, poles).gain_matrix   # Kautsky, Nichols, Van Dooren robust placement
    K = Kt @ E.T                                      # back to Bloch (x, y, z) coordinates
    A_cl = A_p - B @ K                                # closed loop matrix
    ctrb = np.hstack([B, A_p @ B, A_p @ A_p @ B])     # controllability matrix [B, AB, A^2 B]
    diag = dict(
        ctrb_rank=int(np.linalg.matrix_rank(ctrb)),
        ctrb_sv=np.linalg.svd(ctrb, compute_uv=False),
        ol_eig=np.linalg.eigvals(A_p),
        cl_eig=np.linalg.eigvals(A_cl),
        A_tt=A_tt, B_t=B_t, Kt=Kt, E=E, desired=np.array(poles),
    )
    return K, A_cl, diag


def state_transition_matrix(A_cl, t):
    """Phi(t) = exp(A_cl t) via modal decomposition  V diag(exp(lambda t)) V^-1, checked against expm."""
    w, V = np.linalg.eig(A_cl)
    Phi_modal = np.real(V @ np.diag(np.exp(w * t)) @ np.linalg.inv(V))
    Phi_expm = sla.expm(A_cl * t)
    return Phi_expm, float(np.max(np.abs(Phi_modal - Phi_expm)))


# =============================================================================
# 4. SIMULATION
# =============================================================================
def control_law(r, r_ref, Om_ref, K):
    """u = u_ss - K (r - r_ref); returns the full drive vector (Omega_x, Omega_y, 0)."""
    u = Om_ref[:2] - K @ (r - r_ref)
    return np.array([u[0], u[1], 0.0])


def simulate_closed_loop(p, A0, b0, r_ref, Om_ref, K, t_eval):
    """Integrate the NONLINEAR (bilinear) Bloch equations under state feedback."""
    def rhs(t, r):
        return A0 @ r + b0 + np.cross(control_law(r, r_ref, Om_ref, K), r)

    sol = solve_ivp(rhs, [t_eval[0], t_eval[-1]], np.array([0.0, 0.0, 1.0]),
                    t_eval=t_eval, method="DOP853", rtol=1e-10, atol=1e-12)
    R = sol.y.T
    U = np.array([control_law(r, r_ref, Om_ref, K)[:2] for r in R])
    return R, U


def simulate_linear(A_cl, r_ref, t_eval):
    """Linearised closed loop: e(t) = Phi(t) e(0), with e(0) = r(0) - r_ref."""
    e0 = np.array([0.0, 0.0, 1.0]) - r_ref
    w, V = np.linalg.eig(A_cl)
    c = np.linalg.solve(V, e0.astype(complex))
    E_t = np.real((V * c) @ np.exp(np.outer(w, t_eval)))      # shape (3, N)
    return E_t.T


def qutip_lindblad_check(p, gphi, t, U):
    """Independent validation with QuTiP: replay the recorded control through the Lindblad master equation.

        d rho/dt = -i [H, rho] + sum_k ( L_k rho L_k^dag - 1/2 {L_k^dag L_k, rho} )
        H   = (Omega_x sigma_x + Omega_y sigma_y) / 2
        L_1 = sqrt(gamma1) |0><1|          (amplitude damping toward the ground state)
        L_2 = sqrt(gamma_phi / 2) sigma_z  (pure dephasing)

    If the Bloch matrices above are right, <sigma> from the master equation equals r(t).
    """
    g1 = 1.0 / p.T1
    sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
    spx, spy = CubicSpline(t, U[:, 0]), CubicSpline(t, U[:, 1])
    H = [[0.5 * sx, lambda tt: float(spx(tt))],
         [0.5 * sy, lambda tt: float(spy(tt))]]
    c_ops = [np.sqrt(g1) * qt.destroy(2), np.sqrt(gphi / 2.0) * sz]
    res = qt.mesolve(H, qt.basis(2, 0), t, c_ops, e_ops=[sx, sy, sz],
                     options={"nsteps": 500000, "atol": 1e-10, "rtol": 1e-9})
    return np.array(res.expect).T


# =============================================================================
# 5. PERFORMANCE METRICS
# =============================================================================
def first_crossing(t, q, level):
    """Linearly interpolated first time q reaches `level` (q rising)."""
    i = int(np.argmax(q >= level))
    if q[i] < level:
        return float("nan")
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (level - q[i - 1]) * (t[i] - t[i - 1]) / (q[i] - q[i - 1]))


def step_metrics(t, q, band=0.02):
    """Classical step response metrics for a normalised signal q(t): 0 -> 1."""
    t_r = first_crossing(t, q, 0.9) - first_crossing(t, q, 0.1)
    ip = int(np.argmax(q))
    overshoot = max(0.0, (q[ip] - 1.0) * 100.0)
    out = np.where(np.abs(q - 1.0) > band)[0]
    t_s = float(t[min(out[-1] + 1, len(t) - 1)]) if len(out) else 0.0
    return dict(rise_time=t_r, peak_time=float(t[ip]), overshoot_pct=overshoot, settling_time=t_s)


def compute_metrics(p, t, R, U, n, s, r_ref, E, K, A_cl, diag, R_lin):
    # --- angular (fast) loop: progress q(t) along the initial tangent error ---
    delta = (R - r_ref) @ E
    d0 = delta[0]
    q_nl = 1.0 - (delta @ d0) / (d0 @ d0)
    delta_l = (R_lin) @ E
    q_lin = 1.0 - (delta_l @ d0) / (d0 @ d0)
    m_nl = step_metrics(t, q_nl)
    m_lin = step_metrics(t, q_lin)

    lam = diag["cl_eig"]
    lam_c = lam[np.argmax(np.abs(lam.imag))]
    lam_r = lam[np.argmin(np.abs(lam.imag))].real
    wn_ach = abs(lam_c)
    zeta_ach = -lam_c.real / wn_ach
    Mp_th = 100.0 * np.exp(-np.pi * zeta_ach / np.sqrt(1 - zeta_ach ** 2))
    tr_th = (1 + 1.1 * zeta_ach + 1.4 * zeta_ach ** 2) / wn_ach
    ts_th = 4.0 / (zeta_ach * wn_ach)
    tp_th = np.pi / (wn_ach * np.sqrt(1 - zeta_ach ** 2))

    # --- radial (slow) loop: purity relaxation ---
    rad = (R - r_ref) @ n
    out = np.where(np.abs(rad) > 0.02 * abs(rad[0]))[0]
    t_s_rad = float(t[min(out[-1] + 1, len(t) - 1)]) if len(out) else 0.0

    # --- fidelity to the PURE target state: F = <psi_t| rho |psi_t> = (1 + r . n) / 2 ---
    F = 0.5 * (1.0 + R @ n)
    iF = int(np.argmax(F))
    t_F99 = first_crossing(t, F, 0.99)

    return dict(
        wn_achieved=float(wn_ach), zeta_achieved=float(zeta_ach), lam_radial=float(lam_r),
        tau_radial=float(-1.0 / lam_r),
        nl=m_nl, lin=m_lin,
        theory=dict(overshoot_pct=float(Mp_th), rise_time=float(tr_th),
                    settling_time=float(ts_th), peak_time=float(tp_th)),
        radial_settling_time=t_s_rad,
        e_track_final=float(np.linalg.norm(R[-1] - r_ref)),
        e_pure_final=float(np.linalg.norm(R[-1] - n)),
        s=float(s), F_ceiling=float(0.5 * (1 + s)),
        F_peak=float(F[iF]), t_F_peak=float(t[iF]), F_final=float(F[-1]), t_F99=t_F99,
        u_peak=[float(np.abs(U[:, 0]).max()), float(np.abs(U[:, 1]).max())],
        u_final=[float(U[-1, 0]), float(U[-1, 1])],
        lin_vs_nl_max_dev=float(np.max(np.linalg.norm(R - (R_lin + r_ref), axis=1))),
    ), q_nl, q_lin, F


# =============================================================================
# 6. PLOTTING
# =============================================================================
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1 to 3
SEQ_BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]  # sequential blue, steps 250 to 700


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=7.5, length=2.5)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.title.set_size(8.5)
    ax.title.set_color(INK)


def save_fig(fig, outdir, name):
    fig.savefig(outdir / f"{name}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_bloch(outdir, t, R, n, r_ref):
    """3D Bloch sphere with the closed loop trajectory, coloured by time (QuTiP Bloch class)."""
    edges = np.array([0, 0.3, 0.6, 1.0, 1.5, 2.2, 3.5, 20, 100, 300, t[-1]])
    colors = SEQ_BLUE
    b = qt.Bloch(figsize=(4.6, 4.6))
    b.sphere_alpha = 0.06
    b.frame_alpha = 0.08
    b.font_size = 9
    b.font_color = INK2
    b.xlabel = [r"$x$", ""]
    b.ylabel = [r"$y$", ""]
    b.zlabel = [r"$|0\rangle$", r"$|1\rangle$"]
    b.view = [-55, 25]
    for k in range(len(edges) - 1):
        m = (t >= edges[k]) & (t <= edges[k + 1])
        idx = np.where(m)[0]
        i0, i1 = max(idx[0] - 1, 0), idx[-1]
        seg = R[i0:i1 + 1]
        b.add_points([seg[:, 0], seg[:, 1], seg[:, 2]], meth="l", colors=[colors[k]])
    # markers: start (ground state) and end point (steady state)
    b.add_points([[R[0, 0]], [R[0, 1]], [R[0, 2]]], colors=[INK])
    b.add_points([[R[-1, 0]], [R[-1, 1]], [R[-1, 2]]], colors=[C_AQUA])
    b.point_marker = ["o"]
    b.point_size = [14]
    # vectors: pure target state and the reachable steady state
    b.add_vectors(list(n))
    b.add_vectors(list(r_ref))
    b.vector_color = [C_ORANGE, C_AQUA]
    b.vector_width = 2.2
    b.vector_style = "-|>"
    b.render()
    fig = b.fig
    fig.patch.set_facecolor("white")
    # discrete time colour bar for the trajectory segments
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, cmap.N)
    cax = fig.add_axes([0.86, 0.22, 0.022, 0.5])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=np.arange(len(colors) + 1) - 0.5)
    cb.ax.set_yticklabels([f"{e:g}" for e in edges], fontsize=6.5, color=INK2)
    cb.outline.set_edgecolor(AXIS)
    cb.set_label(r"time ($\mu$s)", fontsize=7.5, color=INK2, labelpad=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=INK, markersize=5, label=r"start $|0\rangle$"),
               Line2D([0], [0], color=C_ORANGE, lw=2.2, label=r"pure target $|\psi_t\rangle$"),
               Line2D([0], [0], color=C_AQUA, lw=2.2, label=r"reachable steady state $r_{\rm ref}$")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.99), frameon=False, fontsize=7.5)
    save_fig(fig, outdir, "fig_bloch_trajectory")


def plot_time_response(outdir, p, t, R, U, n, r_ref, F, s):
    fig, axs = plt.subplots(3, 2, figsize=(7.16, 6.3), sharey="row")
    zoom = t <= p.t_fast
    cols = [(0, zoom, (0, p.t_fast), "(a) fast transient: angular loop"),
            (1, np.ones_like(t, bool), (0, p.t_end), "(b) slow relaxation: purity loop")]
    P0, P1 = 0.5 * (1 + R[:, 2]), 0.5 * (1 - R[:, 2])
    for c, m, xl, title in cols:
        a1, a2, a3 = axs[0, c], axs[1, c], axs[2, c]
        a1.plot(t[m], U[m, 0], color=C_BLUE, lw=1.5, label=r"$\Omega_x$")
        a1.plot(t[m], U[m, 1], color=C_ORANGE, lw=1.5, label=r"$\Omega_y$")
        a2.plot(t[m], R[m, 0], color=C_BLUE, lw=1.5, label=r"$x$")
        a2.plot(t[m], R[m, 1], color=C_ORANGE, lw=1.5, label=r"$y$")
        a2.plot(t[m], R[m, 2], color=C_AQUA, lw=1.5, label=r"$z$")
        for k, col in enumerate((C_BLUE, C_ORANGE, C_AQUA)):
            a2.axhline(r_ref[k], color=col, lw=0.8, ls=(0, (4, 3)), alpha=0.8)
        a3.plot(t[m], P0[m], color=C_BLUE, lw=1.5, label=r"$P_0$ (ground)")
        a3.plot(t[m], P1[m], color=C_ORANGE, lw=1.5, label=r"$P_1$ (excited)")
        a3.plot(t[m], F[m], color=C_AQUA, lw=1.5, label=r"$F$ (fidelity to target)")
        a3.axhline(0.5 * (1 + s), color=C_AQUA, lw=0.8, ls=(0, (4, 3)), alpha=0.8)
        for a in (a1, a2, a3):
            a.set_xlim(*xl)
            style_axes(a)
        a1.set_title(title, loc="left")
        a3.set_xlabel(r"time ($\mu$s)")
    axs[0, 0].set_ylabel(r"control $\Omega$ (rad/$\mu$s)")
    axs[1, 0].set_ylabel("Bloch components")
    axs[2, 0].set_ylabel("probability / fidelity")
    axs[0, 0].legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    axs[1, 0].legend(frameon=False, fontsize=7, loc="lower right", ncol=3)
    axs[2, 1].legend(frameon=False, fontsize=7, loc="center right")
    axs[2, 1].annotate(r"ceiling $F=(1+s)/2$", xy=(p.t_end * 0.6, 0.5 * (1 + s)), xytext=(p.t_end * 0.42, 0.975),
                       fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    fig.tight_layout()
    save_fig(fig, outdir, "fig_time_response")


def plot_poles_and_error(outdir, t, R, r_ref, R_lin, diag, p):
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.7))
    ol, cl, des = diag["ol_eig"], diag["cl_eig"], diag["desired"]
    # (a) s plane overview
    a = axs[0]
    a.axvline(0, color=AXIS, lw=0.8)
    a.axhline(0, color=AXIS, lw=0.8)
    a.scatter(ol.real, ol.imag, marker="x", s=36, color=C_ORANGE, lw=1.6, label="open loop", zorder=3)
    a.scatter(cl.real, cl.imag, marker="o", s=30, facecolor="none", edgecolor=C_BLUE, lw=1.6,
              label="closed loop", zorder=4)
    wn = abs(des[0])
    z = -des[0].real / wn
    ang = np.arccos(z)
    a.plot([0, -wn * np.cos(np.pi - ang) * -1], [0, wn * np.sin(np.pi - ang)], color=MUTED, lw=0.7, ls=":")
    a.plot([0, wn * np.cos(np.pi - ang)], [0, -wn * np.sin(np.pi - ang)], color=MUTED, lw=0.7, ls=":")
    a.set_xlabel(r"Re $\lambda$ (1/$\mu$s)")
    a.set_ylabel(r"Im $\lambda$ (1/$\mu$s)")
    a.set_title(r"(a) $s$ plane", loc="left")
    a.set_xlim(-1.35 * wn, 0.15 * wn)
    a.set_ylim(-1.2 * wn, 1.2 * wn)
    a.legend(frameon=False, fontsize=6.8, loc="lower left")
    style_axes(a)
    # (b) zoom near the origin: the radial mode is hardly movable
    b = axs[1]
    b.axvline(0, color=AXIS, lw=0.8)
    b.scatter(ol.real, ol.imag, marker="x", s=36, color=C_ORANGE, lw=1.6, zorder=3)
    b.scatter(cl.real, cl.imag, marker="o", s=30, facecolor="none", edgecolor=C_BLUE, lw=1.6, zorder=4)
    b.set_xlim(-0.03, 0.005)
    b.set_ylim(-0.02, 0.02)
    b.set_xlabel(r"Re $\lambda$ (1/$\mu$s)")
    b.set_title(r"(b) zoom: radial mode", loc="left")
    style_axes(b)
    # (c) error norm, nonlinear vs linearised
    c = axs[2]
    m = t >= 0.01
    en = np.linalg.norm(R - r_ref, axis=1)
    el = np.linalg.norm(R_lin, axis=1)
    c.loglog(t[m], en[m], color=C_BLUE, lw=1.6, label="nonlinear")
    c.loglog(t[m], el[m], color=C_ORANGE, lw=1.2, ls=(0, (4, 2)), label="linearised")
    c.set_xlabel(r"time ($\mu$s)")
    c.set_ylabel(r"$\|r-r_{\rm ref}\|$")
    c.set_title("(c) tracking error", loc="left")
    c.set_ylim(1e-6, 2)
    c.legend(frameon=False, fontsize=7, loc="lower left")
    style_axes(c)
    fig.tight_layout()
    save_fig(fig, outdir, "fig_poles_error")


# =============================================================================
# 7. EXPORT (json + LaTeX macros so the report always matches the simulation)
# =============================================================================
def tex_num(x, sig=4):
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return rf"{m}\times 10^{{{int(e)}}}"
    return s


def tex_mat(M, sig=4):
    M = np.atleast_2d(np.asarray(M, float))
    rows = [" & ".join(tex_num(v, sig) for v in row) for row in M]
    return r"\begin{bmatrix} " + r" \\ ".join(rows) + r" \end{bmatrix}"


def tex_cplx(z, sig=4):
    return rf"{tex_num(z.real, sig)} \pm j\,{tex_num(abs(z.imag), sig)}"


def write_outputs(outdir, p, g, n, s, r_ref, Om_ref, A0, b0, A_p, B, K, A_cl, diag, M, Phi, phi_err, qdev, extra):
    g1, g2, gphi = g
    (outdir / "metrics.json").write_text(json.dumps(dict(params=asdict(p), metrics=M, qutip_max_dev=qdev,
                                                          K=K.tolist(), A_cl=A_cl.tolist()), indent=2, default=float))
    lam = diag["cl_eig"]
    lam_c = lam[np.argmax(np.abs(lam.imag))]
    mac = {
        "TOne": tex_num(p.T1), "TTwo": tex_num(p.T2),
        "gOne": tex_num(g1), "gTwo": tex_num(g2), "gPhi": tex_num(gphi),
        "thT": tex_num(p.theta_deg), "phT": tex_num(p.phi_deg),
        "sRad": tex_num(s), "FCeil": tex_num(M["F_ceiling"], 4),
        "AzeroM": tex_mat(A0), "bzeroV": tex_mat(b0.reshape(3, 1)),
        "nVec": tex_mat(n.reshape(3, 1)), "rRefV": tex_mat(r_ref.reshape(3, 1)),
        "OmRefV": tex_mat(Om_ref[:2].reshape(2, 1)),
        "ApM": tex_mat(A_p), "BM": tex_mat(B), "KM": tex_mat(K), "AclM": tex_mat(A_cl),
        "BtM": tex_mat(diag["B_t"]), "KtM": tex_mat(diag["Kt"]),
        "PhiM": tex_mat(Phi), "PhiErr": tex_num(phi_err, 2),
        "lamClC": tex_cplx(lam_c), "lamClR": tex_num(M["lam_radial"]),
        "lamOlA": tex_cplx(diag["ol_eig"][np.argmax(np.abs(diag["ol_eig"].imag))]),
        "lamOlR": tex_num(diag["ol_eig"][np.argmin(np.abs(diag["ol_eig"].imag))].real),
        "wnDes": tex_num(p.wn), "zetaDes": tex_num(p.zeta),
        "wnAch": tex_num(M["wn_achieved"]), "zetaAch": tex_num(M["zeta_achieved"]),
        "tauRad": tex_num(M["tau_radial"], 3),
        "svOne": tex_num(diag["ctrb_sv"][0], 3), "svTwo": tex_num(diag["ctrb_sv"][1], 3),
        "svThree": tex_num(diag["ctrb_sv"][2], 3), "ctrbRank": str(diag["ctrb_rank"]),
        "MpTh": tex_num(M["theory"]["overshoot_pct"], 3), "MpNl": tex_num(M["nl"]["overshoot_pct"], 3),
        "MpLin": tex_num(M["lin"]["overshoot_pct"], 3),
        "trTh": tex_num(M["theory"]["rise_time"], 3), "trNl": tex_num(M["nl"]["rise_time"], 3),
        "trLin": tex_num(M["lin"]["rise_time"], 3),
        "tsTh": tex_num(M["theory"]["settling_time"], 3), "tsNl": tex_num(M["nl"]["settling_time"], 3),
        "tsLin": tex_num(M["lin"]["settling_time"], 3),
        "tpTh": tex_num(M["theory"]["peak_time"], 3), "tpNl": tex_num(M["nl"]["peak_time"], 3),
        "tpLin": tex_num(M["lin"]["peak_time"], 3),
        "tsRad": tex_num(M["radial_settling_time"], 3),
        "eTrack": tex_num(M["e_track_final"], 2), "ePure": tex_num(M["e_pure_final"], 4),
        "FPeak": tex_num(M["F_peak"], 4), "tFPeak": tex_num(M["t_F_peak"], 3),
        "FFinal": tex_num(M["F_final"], 4), "tFnn": tex_num(M["t_F99"], 3),
        "uxPeak": tex_num(M["u_peak"][0], 3), "uyPeak": tex_num(M["u_peak"][1], 3),
        "uxFinal": tex_num(M["u_final"][0], 3), "uyFinal": tex_num(M["u_final"][1], 3),
        "qutipDev": tex_num(qdev, 2), "linDev": tex_num(M["lin_vs_nl_max_dev"], 2),
        "tEnd": tex_num(p.t_end), "tFastWin": tex_num(p.t_fast),
        "initAngle": tex_num(p.theta_deg),
        "PoneTarget": tex_num(0.5 * (1 - n[2]), 3), "PoneSS": tex_num(0.5 * (1 - r_ref[2]), 3),
        "eInit": tex_num(np.linalg.norm(np.array([0.0, 0.0, 1.0]) - r_ref), 3), "tPhi": "0.5",
    }
    lines = ["% Auto generated by qubit_state_control.py. Do not edit by hand."]
    for k, v in mac.items():
        lines.append(rf"\newcommand{{\{k}}}{{\ensuremath{{{v}}}}}")
    (outdir / "metrics.tex").write_text("\n".join(lines) + "\n")


# =============================================================================
# 8. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--outdir", default="results")
    for f in Params.__dataclass_fields__.values():
        ap.add_argument(f"--{f.name}", type=float, default=f.default)
    args = ap.parse_args()
    p = Params(**{k: v for k, v in vars(args).items() if k in Params.__dataclass_fields__})
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- model, reference, linearisation, design ----
    A0, b0, g1, g2, gphi = decoherence_model(p)
    n, s, r_ref, Om_ref = reference_equilibrium(p, A0, b0)
    A_p, B = linearise(A0, r_ref, Om_ref)
    K, A_cl, diag = design_state_feedback(A_p, B, n, p.wn, p.zeta)
    Phi, phi_err = state_transition_matrix(A_cl, 0.5)

    # ---- simulate (fine grid for the fast transient, coarse grid for the slow relaxation) ----
    t = np.unique(np.concatenate([np.linspace(0, 5, 1501), np.linspace(5, p.t_end, 3000)]))
    R, U = simulate_closed_loop(p, A0, b0, r_ref, Om_ref, K, t)
    R_lin = simulate_linear(A_cl, r_ref, t)                  # error trajectory e(t) of the linear model

    # ---- QuTiP validation of the Bloch model ----
    X = qutip_lindblad_check(p, gphi, t, U)
    qdev = float(np.max(np.abs(X - R)))

    M, q_nl, q_lin, F = compute_metrics(p, t, R, U, n, s, r_ref, diag["E"], K, A_cl, diag, R_lin)

    # ---- figures and exports ----
    plot_bloch(outdir, t, R, n, r_ref)
    plot_time_response(outdir, p, t, R, U, n, r_ref, F, s)
    plot_poles_and_error(outdir, t, R, r_ref, R_lin, diag, p)
    write_outputs(outdir, p, (g1, g2, gphi), n, s, r_ref, Om_ref, A0, b0, A_p, B, K, A_cl, diag, M, Phi, phi_err, qdev, {})

    # ---- console summary ----
    np.set_printoptions(precision=5, suppress=True)
    print(f"gamma1={g1:.5f}  gamma2={g2:.5f}  gamma_phi={gphi:.5f}  [1/us]")
    print(f"target n = {n},  reachable length s = {s:.4f},  fidelity ceiling = {M['F_ceiling']:.4f}")
    print(f"r_ref = {r_ref},  u_ss = {Om_ref[:2]} rad/us")
    print(f"controllability rank = {diag['ctrb_rank']},  singular values = {diag['ctrb_sv']}")
    print(f"open loop eigenvalues   : {diag['ol_eig']}")
    print(f"closed loop eigenvalues : {diag['cl_eig']}")
    print(f"K =\n{K}")
    print(f"QuTiP Lindblad vs Bloch ODE, max |difference| = {qdev:.2e}")
    print(json.dumps({k: v for k, v in M.items()}, indent=2, default=float))
    print(f"Files written to {outdir.resolve()}")


if __name__ == "__main__":
    main()
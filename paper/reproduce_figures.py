"""
Reproduce the simulated curves of the article with the fitted parameters of
parameters.py (model only, the measured curves are not distributed).

    python paper/reproduce_figures.py            # all figures (Figs. 3-11)
    python paper/reproduce_figures.py fig11      # one figure

Figs. 1 and 2 are a flowchart and a sketch redrawn from the literature, not
simulations.

Figures are written to paper/figures/.  Fig. 11 runs 500 000 cycles and takes
a few minutes, as reported in the article.

Author : Iñigo Llavori Osa - Mondragon Unibertsitatea
"""
from pathlib import Path
import sys
import time

import numpy as np

trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x / 1.x

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from tribosim import (CrankSlider, FrictionGenesis, FrictionLoop, GenesisEvent,
                      Sinusoidal, Trapezoidal, WearModel, add_noise)
from tribosim.plotting import apply_style, LIGHT, MID, D1, DARK
import parameters as par

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is needed for the figures: pip install matplotlib")

apply_style()
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("saved", OUT / f"{name}.png")


def loop(c, sr=500, z0_over_x0=None, mu_k=None):
    """One friction loop from a parameter dict of parameters.LOOPS."""
    mk = mu_k if mu_k is not None else c["mu_k"]
    drive = Trapezoidal(c["ramp"]) if c.get("ramp") else Sinusoidal()
    kin = drive.get_stroke(c["x0"], sr, f_test=c["f_test"])
    fl = FrictionLoop(c["P"], c["S"], d_c=c["D_c"], lambda_acc=c["lam_acc"],
                      lambda_dec=c["lam_dec"])
    zx = c["z0_over_x0"] if z0_over_x0 is None else z0_over_x0
    x, q, _ = fl.calculate(zx * c["x0"], c["x0"], mk[0], mk[1],
                           c["mu_s_acc"][0], c["mu_s_acc"][1],
                           c["mu_s_dec"][0], c["mu_s_dec"][1], *kin)
    return x, q


# ---------------------------------------------------------------------------
def fig03():
    """CoF evolution morphologies of the genesis model (Fig. 3)."""
    c = par.GENESIS_SHAPES
    N = c["n_cycles"]
    fig, axs = plt.subplots(2, 5, figsize=(13, 4.6), sharex=True, sharey=True)
    for ax, (title, seed, fw, bw) in zip(axs.ravel(), c["panels"]):
        for kw, col, lab, s in ((fw, DARK, "FW", seed), (bw, MID, "BW", seed + 40)):
            if kw is None:
                continue
            kw = dict(kw, events=[GenesisEvent(n, a, onset=o, recovery=r, shape=sh)
                                  for n, a, o, r, sh in kw["events"]])
            g = FrictionGenesis(N, **kw, sigma=c["sigma"], phi=c["phi"], random_seed=s)
            ax.plot(g.cycles / 1000, g.generate(), lw=0.5, color=col, label=lab)
        if bw is not None:
            ax.legend(fontsize=7, loc="lower right")
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0, 1)
    for ax in axs[1]:
        ax.set_xlabel("cycle n [x1000]")
    for ax in axs[:, 0]:
        ax.set_ylabel("CoF [-]")
    save(fig, "fig03_genesis_shapes")


def _illustration_loop(z0_over_x0, noisy=False):
    c = par.LOOP_ILLUSTRATION
    mu = c["mu_k"]
    kin = CrankSlider().get_stroke(c["x0"], c["n_half"])
    fl = FrictionLoop(c["P"], c["S"], d_c=c["D_c"], lambda_acc=c["lam_acc"],
                      lambda_dec=c["lam_dec"])
    x, q, _ = fl.calculate(z0_over_x0 * c["x0"], c["x0"], mu, mu, mu, mu, mu, mu, *kin)
    if noisy:
        q, x = add_noise(q, x, rng=np.random.default_rng(c["noise_seed"]), **c["noise"])
    return x / c["x0"], q / np.percentile(np.abs(q), 99.0)


def fig04():
    """Clean Coulomb and non-Coulomb loops (Fig. 4)."""
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, zx, title in zip(axs, par.LOOP_ILLUSTRATION["z0_over_x0"],
                             ("(a) Coulomb loop, z0/x0 = 0", "(b) Non-Coulomb loop, z0/x0 = 0.10")):
        ax.plot(*_illustration_loop(zx), color=DARK)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("x/x0 [-]")
    axs[0].set_ylabel("Q/Q_peak [-]")
    save(fig, "fig04_loops")


def fig05():
    """Static-to-kinetic transition over a forward half-stroke (Fig. 5).
    The stick line and the sliding law of Section 2.3.2, written out explicitly
    so that the slip velocity can be plotted."""
    c = par.SLIP_PROFILE
    P, S, mu_k = c["P"], c["S"], c["mu_k"]
    mu_s_a, mu_s_d = c["kappa_acc"] * mu_k, c["kappa_dec"] * mu_k
    n, x0 = c["n"], c["x0"]
    k = np.arange(n + 1)
    x = -x0 * np.cos(np.pi * k / n)
    v = x0 * 2 * np.pi * c["f_test"] * np.sin(np.pi * k / n)
    i_peak = int(np.argmax(v))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    ax1.plot(x / x0, v / v.max(), "k--", lw=0.8, label="machine")
    for D_c, col in zip(c["D_c"], (DARK, MID)):
        Q, vs = np.zeros(n + 1), np.zeros(n + 1)
        Q[0], stick, x_t = -mu_s_d * P, True, None
        for i in range(1, n + 1):
            if stick:
                Q[i] = Q[0] + S * (x[i] - x[0])          # elastic stick line
                if Q[i] < mu_s_a * P:
                    continue
                stick, x_t = False, x[i]
            vs[i] = v[i] * (1 - np.exp(-(x[i] - x_t) / D_c))
            mu_s, lam = (mu_s_a, c["lam_acc"]) if i <= i_peak else (mu_s_d, c["lam_dec"])
            Q[i] = (mu_k + (mu_s - mu_k) * np.exp(-lam * vs[i])) * P
        ax1.plot(x / x0, vs / v.max(), color=col, label=f"slip, D_c = {D_c:g} mm")
        ax2.plot(x / x0, Q / P, color=col)
    for level in (mu_s_a, mu_s_d, mu_k):
        ax2.axhline(level, color="0.6", lw=0.5, ls=":")
    ax1.set_ylabel("v/v_max [-]"); ax1.legend(fontsize=7)
    ax2.set_ylabel("Q/P [-]"); ax2.set_xlabel("x/x0 [-]")
    save(fig, "fig05_slip_profile")


def fig07():
    """Idealised scar cross-sections for p = 2, 4 and 8 (Fig. 7)."""
    c = par.SCAR_PROFILES
    x = np.linspace(-1, 1, 501)
    cols = [LIGHT, MID, D1, DARK, "k"]
    fig, axs = plt.subplots(len(c["p"]), 1, figsize=(5, 5), sharex=True)
    for ax, p in zip(axs, c["p"]):
        for frac, col in zip(c["z0_fractions"], cols):
            ax.plot(x, -frac * (1 - np.abs(x) ** p), color=col)
        ax.set_title(f"p = {p}", fontsize=8)
        ax.set_ylabel("z/z0,max")
    axs[-1].set_xlabel("x/x0 [-]")
    save(fig, "fig07_scar_profiles")


def fig08():
    """Non-Coulomb loop of Fig. 4b with instrumental and tribological noise (Fig. 8)."""
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(*_illustration_loop(par.LOOP_ILLUSTRATION["z0_over_x0"][1], noisy=True),
            color=DARK, lw=0.7)
    ax.set_xlabel("x/x0 [-]"); ax.set_ylabel("Q/Q_peak [-]")
    save(fig, "fig08_noise")


def fig06():
    """Partial to gross slip, contact only and contact + rig (Fig. 6)."""
    P, mu, K_c = 100.0, 0.6, 5.0e4
    dt = 3 * mu * P / (2 * K_c)
    rows = [("contact only", 1e9), ("contact + rig", mu * P / (3 * dt))]
    ratios = (0.5, 0.95, 1.05, 1.5, 3.0)
    fig, axs = plt.subplots(2, 5, figsize=(12, 4.5), sharey=True)
    for i, (lab, S) in enumerate(rows):
        x0t = mu * P / S + dt
        fl = FrictionLoop(P, S, d_c=0.0, contact_stiffness=K_c)
        for j, r in enumerate(ratios):
            x0 = r * x0t
            x, q, info = fl.calculate(0.0, x0, mu, mu, mu, mu, mu, mu,
                                      *Sinusoidal().get_stroke(x0, 2000))
            A = abs(trapz(q, x)) / (4 * np.abs(q).max() * np.abs(x).max())
            ax = axs[i, j]
            ax.plot(x / x0, q / (mu * P), color=MID if info["regime"] == "partial" else DARK)
            ax.set_title(f"x0/x0,t = {r:g}, A = {A:.2f}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{lab}\nQ/(mu P) [-]")
            if i == 1:
                ax.set_xlabel("x/x0 [-]")
    save(fig, "fig06_partial_slip")


def genesis_case(c, seed=0):
    span = c["n_cycles"]
    ev = [GenesisEvent(n, a, onset=max(1, int(0.01 * span)),
                       recovery=max(2, int(0.04 * span)), shape="exponential")
          for n, a in c["events"]]
    return FrictionGenesis(c["n_cycles"], c["mu_i"], mu_inf=c["mu_inf"], tau=c["tau"],
                           betas=c["betas"], breakpoints=c["breakpoints"], events=ev,
                           sigma=c["sigma"], phi=c["phi"], random_seed=seed)


def fig09():
    """Genesis model with the Table 2 parameters (Fig. 9, model only)."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (name, c) in zip(axs.ravel(), par.GENESIS.items()):
        g = genesis_case(c)
        ax.plot(g.cycles, g.generate(), lw=0.5, color=LIGHT, label="model realisation")
        ax.plot(g.cycles, g.backbone(), lw=1.4, color=DARK, label="backbone")
        if name == "LI21":
            ax.set_xscale("log")
        ax.set_title(f"{name}  ({c['ref']})", fontsize=8)
        ax.set_xlabel("cycle n"); ax.set_ylabel("CoF [-]")
    axs[0, 0].legend(fontsize=7)
    save(fig, "fig09_genesis")


def fig10():
    """Friction loops with the Table 3 parameters (Fig. 10, model only)."""
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, (name, c) in zip(axs.ravel(), par.LOOPS.items()):
        x, q = loop(c)
        if "noise" in c:
            q, _ = add_noise(q, x, rng=np.random.default_rng(1), **c["noise"])
        ax.plot(x, q / c["P"], lw=1.0, color=MID)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("CoF [-]")
    save(fig, "fig10_loops")


def fig11():
    """Coupled virtual test of Jin et al. Example 2 (Fig. 11, model only)."""
    v, g = par.VIRTUAL_TEST, par.GENESIS["JIN16"]
    N = v["n_cycles"]
    ev = [GenesisEvent(n, a, onset=v["event_onset"], recovery=v["event_recovery"])
          for n, a in g["events"]]
    gen = FrictionGenesis(N, g["mu_i"], mu_inf=g["mu_inf"], tau=g["tau"], betas=g["betas"],
                          events=ev, sigma=g["sigma"], phi=g["phi"], random_seed=7)
    mu = gen.generate()

    # wear coefficients calibrated on the measured scar volume and half-width
    E_cum = np.cumsum(v["P"] * 2 * v["x0"] * 2 * mu)
    a_flat = v["scar_volume"] / E_cum[-1]
    R, L = v["R_cyl"], v["L_cyl"]
    h_end = (v["scar_half_width"] - v["x0"]) ** 2 / (2 * R)
    V_cyl = L * (R**2 * np.arccos(1 - h_end / R) - (R - h_end) * np.sqrt(2 * R * h_end - h_end**2))
    a_cyl = V_cyl / E_cum[-1]
    w = WearModel(v["x0"], v["P"], a_flat, a_cyl, R, v["E"], v["nu"], v["E"], v["nu"],
                  scar_shape=2, geometry="cylinder", L_cylinder=L).compute(mu, mu)
    zx = w["z0_over_x0"]
    print(f"virtual test: final z0/x0 = {zx[-1]:.3f}, W = {w['W_scar'][-1] * 1e3:.0f} um, "
          f"D = {w['z0'][-1] * 1e3:.1f} um")

    c = dict(P=v["P"], x0=v["x0"], f_test=v["f_test"], S=v["S"], D_c=v["D_c"],
             lam_acc=v["lam_acc"], lam_dec=v["lam_dec"], z0_over_x0=0.0,
             mu_k=None, mu_s_acc=None, mu_s_dec=None)
    t0 = time.perf_counter()
    for n in range(0, N, 1000):                       # timing on a sample of cycles
        m = mu[n]
        c.update(mu_s_acc=(v["kappa_acc"] * m,) * 2, mu_s_dec=(m, m))
        loop(c, sr=100, z0_over_x0=zx[n], mu_k=(m, m))
    per = (time.perf_counter() - t0) / (N // 1000)
    print(f"about {per * N / 60:.1f} min for all {N} loops, "
          f"against {N / v['f_test'] / 3600:.1f} h of physical test")

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs[0, 0].plot(gen.cycles, mu, lw=0.4, color=MID)
    axs[0, 0].set_xlabel("cycle n"); axs[0, 0].set_ylabel("CoF [-]")
    for n, col in zip((5_000, 50_000, N - 1), (LIGHT, MID, DARK)):
        m = mu[n]
        c.update(mu_s_acc=(v["kappa_acc"] * m,) * 2, mu_s_dec=(m, m))
        x, q = loop(c, sr=200, z0_over_x0=zx[n], mu_k=(m, m))
        axs[0, 1].plot(x * 1e3, q, color=col, label=f"n = {n // 1000} k")
        W, D = w["W_scar"][n], w["z0"][n]
        xp = np.linspace(-W, W, 200)
        axs[1, 0].plot(xp * 1e3, -D * (1 - (xp / W) ** 2) * 1e3, color=col, label=f"n = {n // 1000} k")
    axs[0, 1].set_xlabel("x [um]"); axs[0, 1].set_ylabel("Q [N]"); axs[0, 1].legend(fontsize=7)
    axs[1, 0].set_xlabel("x [um]"); axs[1, 0].set_ylabel("depth [um]"); axs[1, 0].legend(fontsize=7)
    axs[1, 1].plot(gen.cycles, zx, color=DARK)
    axs[1, 1].set_xlabel("cycle n"); axs[1, 1].set_ylabel("z0/x0 = D/W [-]")
    save(fig, "fig11_virtual_test")


FIGS = {"fig03": fig03, "fig04": fig04, "fig05": fig05, "fig06": fig06, "fig07": fig07,
        "fig08": fig08, "fig09": fig09, "fig10": fig10, "fig11": fig11}

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()

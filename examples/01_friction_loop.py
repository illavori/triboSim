"""
Example 1 - one friction loop.

Three loops for the same contact: Coulomb, non-Coulomb (worn groove) and with
a static-to-kinetic overshoot at each reversal.

    python examples/01_friction_loop.py

Author : Iñigo Llavori Osa - Mondragon Unibertsitatea
"""
from pathlib import Path
import sys

import numpy as np

trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x / 1.x

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a checkout
from tribosim import FrictionLoop, Sinusoidal

# ---- test conditions -------------------------------------------------------
P = 50.0          # normal load [N]
x0 = 0.5          # stroke half-amplitude [mm]
f_test = 5.0      # test frequency [Hz]
S = 2000.0        # rig stiffness [N/mm]
mu_k = 0.40       # kinetic CoF

# displacement [mm] and velocity [mm/s] of the forward and backward strokes
kin = Sinusoidal().get_stroke(x0, 500, f_test=f_test)


def loop(z0_over_x0=0.0, kappa=1.0, d_c=0.0, lam=0.2):
    """One cycle. kappa = mu_s/mu_k after each reversal, lam in s/mm."""
    fl = FrictionLoop(contact_force=P, stiffness=S, d_c=d_c,
                      lambda_acc=lam, lambda_dec=lam)
    x, q, _ = fl.calculate(
        z0=z0_over_x0 * x0, x0=x0,
        mu_k_fw=mu_k, mu_k_bw=mu_k,
        mu_s_acc_fw=kappa * mu_k, mu_s_acc_bw=kappa * mu_k,   # after the reversal
        mu_s_dec_fw=mu_k, mu_s_dec_bw=mu_k,                   # before the reversal
        x_fw=kin[0], x_bw=kin[1], v_fw=kin[2], v_bw=kin[3])
    return x, q


cases = {
    "Coulomb": loop(),
    "non-Coulomb, z0/x0 = 0.1": loop(z0_over_x0=0.1),
    "static overshoot, mu_s/mu_k = 1.5": loop(kappa=1.5, d_c=0.05),
}
for name, (x, q) in cases.items():
    energy = abs(trapz(q, x))             # dissipated energy per cycle [N mm]
    print(f"{name:36s} peak |Q| = {np.abs(q).max():6.2f} N   E_d = {energy:6.2f} N mm")

try:
    import matplotlib.pyplot as plt
    from tribosim.plotting import apply_style, BLACK, MID, DARK
except ImportError:
    sys.exit(0)
apply_style()
fig, ax = plt.subplots(figsize=(5, 3.5))
styles = [dict(color=BLACK, lw=0.8, ls="--"), dict(color=MID, lw=1.4), dict(color=DARK, lw=1.4)]
labels = ["Coulomb", r"non-Coulomb, $z_0/x_0 = 0.1$", r"overshoot, $\mu_\mathrm{s}/\mu_\mathrm{k} = 1.5$"]
for (x, q), st, lab in zip(cases.values(), styles, labels):
    ax.plot(x, q / P, label=lab, **st)
ax.set_xlabel(r"$x$ [mm]"); ax.set_ylabel(r"$Q/P$ [-]"); ax.legend(loc="center")
out = Path(__file__).parent / "output"; out.mkdir(exist_ok=True)
fig.tight_layout(); fig.savefig(out / "01_friction_loop.png", dpi=150)
print("saved", out / "01_friction_loop.png")

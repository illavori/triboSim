"""
Example 3 - from partial slip to gross slip.

With a contact stiffness K_c the stick branch follows a Cattaneo-Mindlin
(Masing) curve in series with the rig compliance 1/S, and short strokes give
closed partial slip loops.  The energy ratio A = E_d / (4 Q* delta*) of
Fouvry et al. (1995) is printed for each stroke.

    python examples/03_partial_slip.py

Author : Iñigo Llavori Osa - Mondragon Unibertsitatea
"""
from pathlib import Path
import sys

import numpy as np

trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x / 1.x

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tribosim import FrictionLoop, Sinusoidal

P, mu = 100.0, 0.6               # normal load [N], CoF
K_c = 5.0e4                      # initial tangential contact stiffness [N/mm]
S = 1.0e9                        # rig stiffness [N/mm]: rigid, only the contact is compliant

delta_t = 3 * mu * P / (2 * K_c)            # contact slides as a whole at this displacement
x0_t = mu * P / S + delta_t                 # transition stroke (Eq. eq:regime)
print(f"transition stroke x0,t = {x0_t * 1e3:.2f} um")

fl = FrictionLoop(P, S, d_c=0.0, contact_stiffness=K_c)
for ratio in (0.5, 0.95, 1.05, 1.5, 3.0):
    x0 = ratio * x0_t
    kin = Sinusoidal().get_stroke(x0, 2000)
    x, q, info = fl.calculate(0.0, x0, mu, mu, mu, mu, mu, mu, *kin)
    E_d = abs(trapz(q, x))
    A = E_d / (4 * np.abs(q).max() * np.abs(x).max())
    print(f"x0/x0,t = {ratio:4.2f}   regime = {info['regime']:7s}   A = {A:.2f}")

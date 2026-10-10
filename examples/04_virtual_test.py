"""
Example 4 - a complete virtual reciprocating test (ball on flat).

The five modules in sequence:

    genesis -> wear -> kinematics + friction loop (one loop per cycle,
    shaped by the current scar) -> acquisition noise -> CSV export

The ground-truth parameters of every module are known, so the output can be
used to test CoF estimation methods.

    python examples/04_virtual_test.py

Author : Iñigo Llavori Osa - Mondragon Unibertsitatea
"""
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tribosim import (Exporter, FrictionGenesis, FrictionLoop, Sinusoidal,
                      WearModel, add_noise)

# ---- test definition (units: mm, N, MPa, s) ---------------------------------
N = 5000                     # cycles
P = 20.0                     # normal load [N]
x0 = 0.25                    # stroke half-amplitude [mm]
f_test = 10.0                # test frequency [Hz]
f_s = 5000.0                 # sampling frequency [Hz]
S = 1500.0                   # rig stiffness [N/mm]
SR = int(f_s / f_test / 2)   # samples per half-cycle

# 1) CoF genesis, forward and backward strokes
gen = dict(n_cycles=N, tau=300.0, sigma=0.006, phi=0.95)
mu_fw = FrictionGenesis(mu_i=0.20, mu_inf=0.60, random_seed=1, **gen).generate()
mu_bw = FrictionGenesis(mu_i=0.20, mu_inf=0.55, random_seed=2, **gen).generate()

# 2) energy-based wear: 6 mm steel ball on a steel flat
wear = WearModel(x0=x0, contact_force=P, alpha_flat=2e-7, alpha_ball=5e-8,
                 R_ball=3.0, E1=210e3, nu1=0.3, E2=210e3, nu2=0.3).compute(mu_fw, mu_bw)
aspect = wear["z0_over_x0"]           # scar depth / half-length, drives the loop

# 3) one friction loop per cycle
kin = Sinusoidal().get_stroke(x0, SR, f_test=f_test)
loop = FrictionLoop(P, S, d_c=0.01, lambda_acc=0.5, lambda_dec=0.5)
t0 = time.perf_counter()
xs, qs = [], []
for n in range(N):
    x, q, _ = loop.calculate(
        z0=aspect[n] * x0, x0=x0,
        mu_k_fw=mu_fw[n], mu_k_bw=mu_bw[n],
        mu_s_acc_fw=1.3 * mu_fw[n], mu_s_acc_bw=1.3 * mu_bw[n],
        mu_s_dec_fw=mu_fw[n], mu_s_dec_bw=mu_bw[n], x_fw=kin[0], x_bw=kin[1],
        v_fw=kin[2], v_bw=kin[3])
    xs.append(x); qs.append(q)
x, q = np.concatenate(xs), np.concatenate(qs)
print(f"{N} loops in {time.perf_counter() - t0:.1f} s, "
      f"final scar depth {wear['z0'][-1] * 1e3:.2f} um, z0/x0 {aspect[-1]:.3f}")

# 4) acquisition noise: 50 N sensor, 0.5 % of full scale, AR1 at 100 Hz
q_noisy, x_noisy = add_noise(q, x, sampling_rate=f_s, normal_load=P,
                             force_sensor_range=50.0, sigma_rel=0.005,
                             sigma_x_abs=0.001, sigma_ct=0.002, f_c=100.0,
                             rng=np.random.default_rng(0))

# 5) export the last 20 cycles to a self-describing CSV
keep = slice(-20 * 2 * SR, None)
t = np.arange(x.size)[keep] / f_s
out = Path(__file__).parent / "output"
csv = Exporter(out, "virtual_test").export_csv(
    P, 2 * x0, f_test, f_s, t, x_noisy[keep], q_noisy[keep], float(wear["z0"][-1]),
    sensor_info={"Qfs_N": 50.0, "sigma_rel": 0.005})
print("saved", csv)

"""Smoke and invariant tests for the tribosim package.

Runs with pytest, or as a plain script from the repository root:

    python -m pytest            # or
    python tests/test_tribosim.py

Each test checks a physical or numerical invariant (closed loop, analytical
Coulomb area, volume conservation, AR1 statistics, ...) rather than frozen
numbers, so it stays valid when parameters are tuned.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x / 1.x

# Make the package importable when run from a source checkout
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tribosim import (  # noqa: E402
    CrankSlider, Exporter, FrictionGenesis, FrictionLoop, GenesisEvent, Sinusoidal,
    Trapezoidal, WearModel, add_noise, load_session,
)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    return False


def _loop_area(x, q):
    """Closed-path integral of Q dx (positive for a clockwise Q-x loop)."""
    xc, qc = np.append(x, x[0]), np.append(q, q[0])
    return float(np.sum(0.5 * (qc[1:] + qc[:-1]) * np.diff(xc)))


# ----------------------------------------------------------------------
# kinematics
# ----------------------------------------------------------------------

def test_sinusoidal_stroke():
    x0, sr = 1.5, 1001
    x_fw, x_bw, v_fw, v_bw = Sinusoidal().get_stroke(x0, sr)
    assert x_fw.shape == x_bw.shape == v_fw.shape == v_bw.shape == (sr,)
    assert np.isclose(x_fw[0], -x0) and np.isclose(x_fw[-1], x0)
    assert np.isclose(x_bw[0], x0) and np.isclose(x_bw[-1], -x0)
    assert np.all(v_fw >= -1e-12) and np.all(v_bw <= 1e-12)
    th = np.linspace(0, np.pi, sr)
    # interior points only: np.gradient is first order at the ends
    assert np.allclose(np.gradient(x_fw, th)[1:-1], v_fw[1:-1], atol=1e-4 * x0)


def test_trapezoidal_stroke():
    x0, sr = 2.0, 2001
    for rf in (0.05, 0.2, 0.45):
        x_fw, x_bw, v_fw, v_bw = Trapezoidal(rf).get_stroke(x0, sr)
        assert np.isclose(x_fw[0], -x0)
        assert abs(x_fw[-1] - x0) < 1e-3 * x0, (rf, x_fw[-1])
        assert np.all(np.diff(x_fw) >= 0) and np.all(v_fw >= 0)
        assert np.allclose(x_bw, x_fw[::-1]) and np.allclose(v_bw, -v_fw[::-1])
    assert _raises(ValueError, Trapezoidal, 0.0)
    assert _raises(ValueError, Trapezoidal, 0.5)


def test_crank_slider_stroke():
    x0, sr = 1.0, 1000
    x_fw, x_bw, v_fw, v_bw = CrankSlider(75, 150, 2000).get_stroke(x0, sr)
    assert np.isclose(x_fw[0], -x0)
    assert abs(x_fw[-1] - x0) < 1e-4, x_fw[-1]
    assert np.all(np.diff(x_fw) >= -1e-12) and np.all(np.diff(x_bw) <= 1e-12)
    assert np.all(v_fw >= -1e-9) and np.all(v_bw <= 1e-9)
    # R/L -> 0 recovers the sinusoidal profile
    xs, _, vs, _ = Sinusoidal().get_stroke(x0, sr)
    xc, _, vc, _ = CrankSlider(1.0, 1e4, 20000).get_stroke(x0, sr)
    assert np.max(np.abs(xc - xs)) < 1e-2 and np.max(np.abs(vc - vs)) < 1e-2
    assert _raises(ValueError, CrankSlider, 150, 75)
    assert _raises(ValueError, CrankSlider, 75, 150, 1001)


def test_velocity_units_with_f_test():
    """With f_test the velocity is physical (mm/s): peak x0*omega for the sine,
    and lambda in s/mm equal to lambda_rad/omega gives the identical loop."""
    x0, sr, f = 0.05, 1001, 20.0
    w = 2 * np.pi * f
    _, _, v_rad, _ = Sinusoidal().get_stroke(x0, sr)
    _, _, v_si, _ = Sinusoidal().get_stroke(x0, sr, f_test=f)
    assert np.isclose(v_si.max(), x0 * w) and np.allclose(v_si, v_rad * w)
    for kin in (Trapezoidal(0.2), CrankSlider(75, 150, 2000)):
        a, _, va, _ = kin.get_stroke(x0, sr)
        b, _, vb, _ = kin.get_stroke(x0, sr, f_test=f)
        assert np.array_equal(a, b) and np.allclose(vb, va * w)
    kin_rad = Sinusoidal().get_stroke(x0, sr)
    kin_si = Sinusoidal().get_stroke(x0, sr, f_test=f)
    args = (0.0025, x0, 0.8, 0.8, 0.68, 0.68, 0.8, 0.8)
    _, q1, _ = FrictionLoop(450.0, 45e3, d_c=0.006, lambda_acc=60, lambda_dec=5).calculate(*args, *kin_rad)
    _, q2, _ = FrictionLoop(450.0, 45e3, d_c=0.006, lambda_acc=60 / w,
                            lambda_dec=5 / w).calculate(*args, *kin_si)
    assert np.allclose(q1, q2, atol=1e-9)
    assert _raises(ValueError, Sinusoidal().get_stroke, x0, sr, -1.0)


# ----------------------------------------------------------------------
# friction_loop
# ----------------------------------------------------------------------

def _run_loop(z0=0.0, mu_k=0.5, mu_s=None, P=10.0, S=50.0, x0=1.0, sr=2000,
              kin=None, **kw):
    mu_s = mu_k if mu_s is None else mu_s
    x_fw, x_bw, v_fw, v_bw = (kin or Sinusoidal()).get_stroke(x0, sr)
    fl = FrictionLoop(P, S, **kw)       # kw: d_c, lambda_*, contact_stiffness
    return fl, fl.calculate(z0, x0, mu_k, mu_k, mu_s, mu_s, mu_s, mu_s,
                            x_fw, x_bw, v_fw, v_bw)


def test_loop_coulomb_area():
    """Coulomb loop: parallelogram, area 2muP(2x0 - 2muP/S) = 4muPx0(1 - xi)."""
    P, S, x0, mu = 10.0, 50.0, 1.0, 0.5
    _, (x, q, _) = _run_loop(P=P, S=S, x0=x0, mu_k=mu, d_c=0.0)
    assert x.shape == q.shape == (4000,)
    assert np.all(np.isfinite(q))
    assert np.isclose(q.max(), mu * P, rtol=1e-6)
    assert np.isclose(q.min(), -mu * P, rtol=1e-6)
    xi = mu * P / (S * x0)
    expected = 4 * mu * P * x0 * (1 - xi)
    assert abs(_loop_area(x, q) / expected - 1) < 0.01, (_loop_area(x, q), expected)


def test_loop_closed_and_clockwise():
    for kin in (Sinusoidal(), Trapezoidal(0.15), CrankSlider()):
        _, (x, q, st) = _run_loop(z0=0.02, mu_s=0.65, kin=kin)
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(q))
        assert np.isclose(x[0], x[-1], atol=1e-9)
        assert abs(q[0] - q[-1]) < 1e-6 * abs(q).max(), (q[0], q[-1])
        assert _loop_area(x, q) > 0
        assert len(st["disp_fw"]) > 0 and len(st["disp_bw"]) > 0


def test_loop_stick_slope_is_stiffness():
    S = 50.0
    _, (_, _, st) = _run_loop(S=S, z0=0.01)
    slope = np.polyfit(st["disp_fw"], st["Q_fw"], 1)[0]
    assert np.isclose(slope, S, rtol=1e-6)


def test_loop_non_coulomb_rises():
    """A worn scar (z0 > 0) makes |Q| grow towards the stroke ends."""
    P, S, x0, z0 = 10.0, 50.0, 1.0, 0.05
    fl = FrictionLoop(P, S)
    xg = np.linspace(0, 0.9 * x0, 200)
    qf = fl._q_forward(xg, 0.5, z0, x0)
    assert np.all(np.diff(qf) > 0)


def test_q_implicit_residual():
    """Fixed point satisfies Q = P(mu + f s)/(1 - mu f s), s = x - Q/S,
    across the range used in the paper (zeta = z0/x0 <= 0.214)."""
    P, S, x0, mu = 10.0, 50.0, 1.0, 0.5
    fl = FrictionLoop(P, S)
    xg = np.linspace(-x0, x0, 501)
    for zeta in (0.0, 0.05, 0.1, 0.214):
        z0 = zeta * x0
        f = 2 * z0 / x0 ** 2
        for sign, fn in ((1, fl._q_forward), (-1, fl._q_backward)):
            Q = fn(xg, mu, z0, x0)
            s = xg - Q / S
            rhs = P * (sign * mu + f * s) / (1 - sign * mu * f * s)
            res = np.max(np.abs(Q - rhs)) / (mu * P)
            assert res < 1e-4, (zeta, sign, res)


def test_loop_rejects_bad_input():
    assert _raises(ValueError, FrictionLoop, 0.0, 10.0)
    assert _raises(ValueError, FrictionLoop, 1.0, -1.0)
    assert _raises(ValueError, FrictionLoop, 1.0, 1.0, d_c=-0.1)
    assert _raises(ValueError, FrictionLoop, 1.0, 1.0, contact_stiffness=0.0)


def test_masing_fouvry_transition():
    """Mindlin-Masing contact: A = 0.2 at the partial/gross slip transition
    with a rigid rig (Fouvry et al. 1995), lower with a compliant rig."""
    mu, P, Kc = 0.5, 1.0, 10.0
    dt = 1.5 * mu * P / Kc                    # Mindlin slip-onset displacement
    for S in (1e9, 5.0):
        xt = mu * P / S + dt
        At = 0.2 * dt / (dt + mu * P / S)
        for f, regime in ((0.999, "partial"), (1.001, "gross")):
            _, (x, q, st) = _run_loop(x0=f * xt, S=S, mu_k=mu, P=P, d_c=0.0,
                                      sr=4000, contact_stiffness=Kc)
            assert st["regime"] == regime
            A = _loop_area(x, q) / (4 * np.abs(q).max() * np.abs(x).max())
            assert abs(A - At) < 0.01 * At + 2e-3, (S, f, A, At)


def test_masing_partial_slip_analytic():
    """Partial slip loop matches the closed-form Mindlin energy ratio."""
    mu, P, Kc = 0.5, 1.0, 10.0
    dt = 1.5 * mu * P / Kc
    for qr, A_ref in ((0.5, 0.0381), (0.9, 0.1136)):
        x0 = dt * (1 - (1 - qr) ** (2 / 3))
        _, (x, q, st) = _run_loop(x0=x0, S=1e9, mu_k=mu, P=P, sr=4000,
                                  contact_stiffness=Kc)
        assert np.isclose(st["Q_amplitude"], qr * mu * P, rtol=1e-6)
        assert np.isclose(x[0], x[-1]) and abs(q[0] - q[-1]) < 1e-6 * P
        A = _loop_area(x, q) / (4 * st["Q_amplitude"] * x0)
        assert abs(A - A_ref) < 5e-4, (qr, A)


def test_masing_gross_slip_limits():
    """Finite K_c in gross slip: closed, clockwise, no jumps, and K_c -> inf
    recovers the Jenkins loop."""
    kw = dict(z0=0.1, mu_s=0.7, d_c=0.15)     # P = 10, S = 50, x0 = 1
    for kin in (Sinusoidal(), Trapezoidal(0.15), CrankSlider()):
        # K_c = 50 puts the transition at x0_t = 0.35, well inside gross slip
        _, (x, q, st) = _run_loop(kin=kin, contact_stiffness=50.0, **kw)
        assert st["regime"] == "gross" and np.all(np.isfinite(q))
        assert abs(q[0] - q[-1]) < 1e-6 and _loop_area(x, q) > 0
        # the branch lands on the static peak without a jump, then the
        # Oden-Martins drop starts (a property of D_c and lambda, not the splice)
        i = len(st["disp_fw"])
        assert abs(q[i] - q[i - 1]) < 0.01 * np.abs(q).max(), (q[i - 1], q[i])
        assert q[i] >= q[i + 1]
    # K_c -> inf recovers Jenkins when mu_s_dec = mu_k.  With mu_s_dec > mu_k
    # the Jenkins line is re-anchored above the point used for the Oden onset
    # and splices below the static peak, while the Masing branch reaches it.
    xf, xb, vf, vb = Sinusoidal().get_stroke(1.0, 2000)
    args = (0.1, 1.0, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, xf, xb, vf, vb)
    xj, qj, _ = FrictionLoop(10.0, 50.0, d_c=0.15).calculate(*args)
    xm, qm, _ = FrictionLoop(10.0, 50.0, d_c=0.15, contact_stiffness=1e9).calculate(*args)
    assert abs(_loop_area(xm, qm) / _loop_area(xj, qj) - 1) < 1e-3
    assert abs(qm.max() / qj.max() - 1) < 1e-3


# ----------------------------------------------------------------------
# genesis
# ----------------------------------------------------------------------

def test_genesis_run_in_and_drift():
    """Backbone of Eq. eq:genesis: exponential run-in plus continuous piecewise drift."""
    n_cyc, mi, minf, tau = 5000, 0.3, 0.8, 400.0
    g = FrictionGenesis(n_cyc, mi, mu_inf=minf, tau=tau)
    mu = g.generate()
    n = np.arange(1, n_cyc + 1)
    assert np.allclose(mu, mi + (minf - mi) * (1 - np.exp(-n / tau)))
    assert abs(mu[-1] - minf) < 1e-5          # n/tau = 12.5
    # no tau -> no run-in, the backbone starts and stays at mu_i
    assert np.allclose(FrictionGenesis(500, 0.4).generate(), 0.4)
    # one segment is the ordinary linear drift beta*n
    assert np.allclose(FrictionGenesis(500, 0.4, betas=[1e-4]).drift(), 1e-4 * np.arange(1, 501))
    # two segments: rates beta_k on each side of the breakpoint, no step at it
    d = FrictionGenesis(1000, 0.4, betas=[2e-4, -1e-4], breakpoints=[400]).drift()
    assert np.allclose(np.diff(d)[:398], 2e-4) and np.allclose(np.diff(d)[401:], -1e-4)
    assert np.max(np.abs(np.diff(d))) <= 2e-4 + 1e-15


def test_genesis_events_bidirectional():
    """A drop goes down, a peak goes up, and both recover to the backbone."""
    ev = [GenesisEvent(300, -0.1, onset=20, recovery=100, shape="exponential"),
          GenesisEvent(700, +0.08, onset=10, recovery=80, shape="sigmoid")]
    g = FrictionGenesis(1000, 0.5, events=ev)
    e = g.event_term()
    assert np.isclose(e.min(), -0.1) and np.isclose(e.max(), 0.08)
    assert np.argmin(e) == 299 and np.argmax(e) == 699
    assert np.allclose(e[:279], 0) and np.allclose(e[400:689], 0) and np.allclose(e[780:], 0)
    for shape in ("linear", "exponential", "logarithmic", "sigmoid"):
        e = FrictionGenesis(400, 0.5, events=[GenesisEvent(100, -0.1, 5, 50, shape)]).event_term()
        r = e[99:149]
        assert np.all(np.diff(r) >= -1e-12) and abs(r[-1]) < 0.01 * 0.1, shape
    rnd = FrictionGenesis.random_events(10000, 6, sign="both", random_seed=3)
    assert len(rnd) == 6 and {np.sign(x.amplitude) for x in rnd} <= {-1.0, 1.0}
    assert all(x.amplitude < 0 for x in FrictionGenesis.random_events(10000, 4, sign="drop", random_seed=1))


def test_genesis_fluctuation_ar1():
    """eps(n) is AR1 with the requested standard deviation and lag-one correlation."""
    g = FrictionGenesis(200_000, 0.5, sigma=0.01, phi=0.9, random_seed=5)
    eps = g.fluctuation()
    assert abs(eps.std() - 0.01) / 0.01 < 0.05
    assert abs(np.corrcoef(eps[:-1], eps[1:])[0, 1] - 0.9) < 0.01
    a = FrictionGenesis(3000, 0.3, 0.5, 100, sigma=0.01, phi=0.8, random_seed=7).generate()
    b = FrictionGenesis(3000, 0.3, 0.5, 100, sigma=0.01, phi=0.8, random_seed=7).generate()
    assert np.array_equal(a, b)
    assert _raises(ValueError, FrictionGenesis, 100, 0.3, betas=[1e-4, 2e-4])
    assert _raises(ValueError, FrictionGenesis, 100, 0.3, phi=1.0)


# ----------------------------------------------------------------------
# wear
# ----------------------------------------------------------------------

_MAT = dict(E1=210e3, nu1=0.3, E2=210e3, nu2=0.3)


def test_wear_ball_hertz_and_volume():
    F, x0, R, af, ab = 10.0, 1.0, 3.0, 1e-6, 5e-7
    n = 5000
    mu = np.full(n, 0.5)
    wm = WearModel(x0, F, af, ab, R, **_MAT)
    out = wm.compute(mu, mu)
    Es = 1 / (2 * (1 - 0.3 ** 2) / 210e3)
    assert np.isclose(out["E_star"], Es)
    assert np.isclose(out["a0_hertz"], (3 * F * R / (4 * Es)) ** (1 / 3))
    assert np.all(np.diff(out["z0"]) > 0)
    assert np.all(out["a_contact"] >= out["a0_hertz"] - 1e-15)
    # ball cap inversion
    Vb = ab * F * 2 * x0 * 1.0 * np.arange(1, n + 1)
    h = out["h_ball"]
    assert np.allclose(np.pi * h ** 2 * (3 * R - h) / 3, Vb, rtol=1e-8)
    # flat volume conservation: 2a * integral(-z) dx == V_flat
    Vf = af * F * 2 * x0 * 1.0 * n
    V = 2 * out["a_contact"][-1] * trapz(-out["z_total"][-1], out["x"])
    assert abs(V / Vf - 1) < 1e-3, (V, Vf)


def test_wear_cylinder_volume():
    F, x0, R, L, af, ac = 50.0, 1.0, 3.0, 5.0, 1e-6, 5e-7
    n = 5000
    mu = np.full(n, 0.6)
    out = WearModel(x0, F, af, ac, R, **_MAT, geometry="cylinder",
                    L_cylinder=L).compute(mu, mu)
    Es = out["E_star"]
    assert np.isclose(out["b0_hertz"], np.sqrt(4 * F * R / (np.pi * L * Es)))
    Vc = ac * F * 2 * x0 * 1.2 * np.arange(1, n + 1)
    h = out["h_cyl"]
    seg = L * (R ** 2 * np.arccos(1 - h / R) - (R - h) * np.sqrt(2 * R * h - h ** 2))
    assert np.allclose(seg, Vc, rtol=1e-6)
    assert np.allclose(out["z0_over_x0"], out["z0"] / out["W_scar"])
    Vf = af * F * 2 * x0 * 1.2 * n
    V = L * trapz(-out["z_total"][-1], out["x"])
    assert abs(V / Vf - 1) < 1e-3, (V, Vf)


def test_wear_rejects_bad_input():
    assert _raises(ValueError, WearModel, 1, 1, 1e-6, 1e-6, 3, **_MAT, scar_shape=3)
    assert _raises(ValueError, WearModel, 1, 1, 1e-6, 1e-6, 3, **_MAT,
                   geometry="cylinder")
    assert _raises(ValueError, WearModel, 1, 1, 1e-6, 1e-6, 3, **_MAT,
                   geometry="cone")


# ----------------------------------------------------------------------
# acquisition
# ----------------------------------------------------------------------

def test_noise_white_statistics():
    n, P, Qfs = 400_000, 20.0, 50.0
    q, x = np.zeros(n), np.zeros(n)
    qn, xn = add_noise(q, x, 1e4, P, Qfs, "white", sigma_rel=0.01,
                       sigma_x_abs=0.002, sigma_ct=0.005,
                       rng=np.random.default_rng(0))
    expected = np.hypot(0.01 * Qfs, 0.005 * P)
    assert abs(qn.std() / expected - 1) < 0.02
    assert abs(xn.std() / 0.002 - 1) < 0.02
    r1 = np.corrcoef(qn[:-1], qn[1:])[0, 1]
    assert abs(r1) < 0.01


def test_noise_ar1_statistics():
    n, fs, fc = 400_000, 1e4, 100.0
    q, x = np.zeros(n), np.zeros(n)
    qn, _ = add_noise(q, x, fs, 1.0, 10.0, "colored", sigma_rel=0.01,
                      sigma_ct=0.0, f_c=fc, rng=np.random.default_rng(1))
    alpha = np.exp(-2 * np.pi * fc / fs)
    assert abs(qn.std() / 0.1 - 1) < 0.05
    r1 = np.corrcoef(qn[:-1], qn[1:])[0, 1]
    assert abs(r1 - alpha) < 0.01, (r1, alpha)


def test_noise_misc():
    q = np.linspace(-1, 1, 1000).reshape(10, 100)
    x = np.linspace(0, 1, 1000).reshape(10, 100)
    a = add_noise(q, x, 1e3, 1.0, 5.0, rng=np.random.default_rng(3))
    b = add_noise(q, x, 1e3, 1.0, 5.0, rng=np.random.default_rng(3))
    assert a[0].shape == q.shape and np.array_equal(a[0], b[0])
    z = add_noise(q, x, 1e3, 1.0, 5.0, sigma_rel=0.0, sigma_x_abs=0.0, sigma_ct=0.0)
    assert np.array_equal(z[0], q) and np.array_equal(z[1], x)
    assert _raises(ValueError, add_noise, q, x, 1e3, 1.0, 5.0, "pink")


# ----------------------------------------------------------------------
# export and end-to-end
# ----------------------------------------------------------------------

def test_end_to_end_virtual_test():
    """genesis -> wear -> loop (z0 from wear) -> noise -> CSV + session."""
    n_cyc, sr, x0, P, S = 200, 500, 1.0, 10.0, 200.0
    rng = np.random.default_rng(42)
    mu = FrictionGenesis(n_cyc, 0.2, mu_inf=0.5, tau=40.0, sigma=0.005, phi=0.9,
                         random_seed=42).generate()
    wear = WearModel(x0, P, 1e-5, 1e-6, 3.0, **_MAT).compute(mu, mu)
    kin = Sinusoidal().get_stroke(x0, sr)
    fl = FrictionLoop(P, S)
    xs, qs = [], []
    for i in (0, n_cyc // 2, n_cyc - 1):
        m = float(mu[i])
        x, q, _ = fl.calculate(float(wear["z0"][i]), x0, m, m, 1.1 * m, 1.1 * m,
                               m, m, *kin)
        assert np.all(np.isfinite(q)) and _loop_area(x, q) > 0
        xs.append(x)
        qs.append(q)
    x, q = np.concatenate(xs), np.concatenate(qs)
    qn, xn = add_noise(q, x, 1e3, P, 20.0, rng=rng)
    t = np.arange(x.size) / 1e3
    with tempfile.TemporaryDirectory() as td:
        ex = Exporter(td, "smoke")
        p = ex.export_csv(P, 2 * x0, 1.0, 1e3, t, xn, qn, float(wear["z0"][-1]),
                          sensor_info={"Qfs_N": 20.0, "sigma_rel": 0.01})
        lines = p.read_text(encoding="utf-8").splitlines()
        hdr = lines.index("time (ms);position (mm);horizontal force (N)")
        data = np.array([[float(c) for c in ln.split(";")] for ln in lines[hdr + 1:]])
        assert data.shape == (x.size, 3)
        assert np.allclose(data[:, 1], xn, atol=6e-5)
        assert np.allclose(data[:, 2], qn, atol=6e-5)
        sp = ex.save_session({"mu": mu, "z0": wear["z0"], "bad": lambda: 0})
        back = load_session(sp)
        assert np.array_equal(back["mu"], mu) and "bad" not in back


# ----------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc(limit=2)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

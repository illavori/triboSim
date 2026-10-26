"""
Fitted parameters of the article "An open physics-based simulator of reciprocating sliding tests with known friction and wear parameters" (Llavori, Borras and Zabala, 2026).

Only the simulator inputs are given here.  The experimental curves they were
fitted to belong to their authors and are not distributed; see README.md in
this folder for the source of each one.

Units: mm, N, s, MPa.  Rate constants lambda are in s/mm, for the velocity in
mm/s (kinematics called with f_test).

Author : Iñigo Llavori Osa - Mondragon Unibertsitatea
"""
import math

# ---------------------------------------------------------------------------
# Table 2 - CoF genesis fits (Fig. 9)
# ---------------------------------------------------------------------------
# phi_mu is the lag-one autocorrelation per cycle.  The fits measured it on the
# digitised grid (spacing dn cycles), so phi_mu = lag1 ** (1 / dn).
def _phi(lag1, dn):
    return math.exp(math.log(lag1) / dn)


GENESIS = {
    "JIN16": dict(ref="Jin et al. (2016), Tribol. Int. 102, 561-568",
                  n_cycles=500_000, mu_i=0.2915, mu_inf=0.7882, tau=4688.6,
                  betas=[1.477e-8], breakpoints=[],
                  sigma=0.0123, phi=_phi(0.946, 1166.9),
                  events=[(28_199, -0.0509), (57_760, +0.0395),
                          (150_331, -0.0492), (308_635, -0.1032)]),
    "CHO26": dict(ref="Choi et al. (2026), Tribol. Int. 222, 112087",
                  n_cycles=1196, mu_i=0.4439, mu_inf=None, tau=None,
                  betas=[2.716e-4, -9.133e-5], breakpoints=[777],
                  sigma=0.0099, phi=_phi(0.827, 2.39),
                  events=[(3, +0.0563), (683, -0.0306)]),
    "KUM26": dict(ref="Kumar et al. (2026), Wear 584-585, 206384",
                  n_cycles=12016, mu_i=0.7131, mu_inf=None, tau=None,
                  betas=[-1.240e-4, -4.635e-6], breakpoints=[2053],
                  sigma=0.0263, phi=_phi(0.885, 21.0),
                  events=[(1140, +0.0894), (11935, +0.1389)]),
    "LI21": dict(ref="Li et al. (2021), Wear 482-483, 203972",
                 n_cycles=3610, mu_i=0.1000, mu_inf=0.5635, tau=2.9,
                 betas=[-1.966e-3, 2.079e-5], breakpoints=[140],
                 sigma=0.0084, phi=_phi(0.942, 12.4),
                 events=[(1561, -0.0326), (2367, +0.0352)]),
}


# ---------------------------------------------------------------------------
# Table 3 - friction loop fits (Fig. 10)
# ---------------------------------------------------------------------------
def _per_second(lam_rad, f_test):
    """Fitted rate constants were obtained per radian; lambda [s/mm] = lambda_rad / omega."""
    return lam_rad / (2.0 * math.pi * f_test)


LOOPS = {
    "mild non-Coulomb (Jin et al., Ex. 1)": dict(
        P=450.0, x0=0.05, f_test=20.0, S=45_000.0, D_c=0.006, z0_over_x0=0.05,
        mu_k=(0.80, 0.80), mu_s_acc=(0.68, 0.68), mu_s_dec=(0.80, 0.80),
        lam_acc=_per_second(60.0, 20.0), lam_dec=_per_second(5.0, 20.0)),
    "static-to-kinetic (NBR, present study)": dict(
        # 8 mm NBR thin-disc pin on S355 steel, lubricated, 60 C, 2 MPa, 50 mm nominal
        # stroke (measured amplitude 25.5 mm), 0.1 m/s with a trapezoidal profile
        P=99.1, x0=25.5, f_test=1.0, S=300.0, D_c=8.0, z0_over_x0=0.0, ramp=0.10,
        mu_k=(0.130, 0.140), mu_s_acc=(0.310, 0.245), mu_s_dec=(0.153, 0.185),
        lam_acc=0.045, lam_dec=0.11),
    "asymmetric (Kumar et al.)": dict(
        P=20.0, x0=0.99, f_test=5.0, S=150.0, D_c=0.18, z0_over_x0=0.0,
        mu_k=(0.79, 0.275), mu_s_acc=(0.316, 0.11), mu_s_dec=(0.79, 0.275),
        lam_acc=_per_second(4.0, 5.0), lam_dec=_per_second(5.0, 5.0)),
    "noisy (Kafexhiu et al.)": dict(           # force normalised by the load (P = 1)
        P=1.0, x0=1.0, f_test=25.0, S=20.0, D_c=0.02, z0_over_x0=0.0,
        mu_k=(0.1525, 0.1525), mu_s_acc=(0.19, 0.19), mu_s_dec=(0.1525, 0.1525),
        lam_acc=_per_second(5.0, 25.0), lam_dec=_per_second(5.0, 25.0),
        noise=dict(sampling_rate=25_000.0, normal_load=1.0, force_sensor_range=1.0,
                   sigma_rel=0.005, sigma_x_abs=0.0, sigma_ct=0.0, f_c=147.0)),
}


# ---------------------------------------------------------------------------
# Section 3.3 - virtual test of Jin et al. Example 2 (Fig. 11)
# ---------------------------------------------------------------------------
VIRTUAL_TEST = dict(
    n_cycles=500_000, f_test=20.0, P=450.0, x0=0.05,
    S=35_000.0,                       # rig stiffness Jin et al. use for Example 2
    D_c=0.006, kappa_acc=0.85,
    lam_acc=_per_second(60.0, 20.0), lam_dec=_per_second(5.0, 20.0),
    R_cyl=6.0, L_cyl=10.0, E=210_000.0, nu=0.30,      # steel cylinder on steel flat
    scar_volume=0.3162,               # measured Example 2 scar: volume [mm^3]
    scar_half_width=0.3330,           #   parabolic-equivalent half-width W [mm]
    event_onset=400, event_recovery=9000,
)



# ---------------------------------------------------------------------------
# Section 2 - illustrations (Figs. 3, 4, 5, 7 and 8).  Not fitted to any test.
# ---------------------------------------------------------------------------
def _drift(nodes):
    """Piecewise-linear drift through (cycle, increment) nodes from (0, 0),
    flat after the last node: (betas, interior breakpoints)."""
    nodes = [(0.0, 0.0)] + list(nodes)
    betas = [(d1 - d0) / (n1 - n0) for (n0, d0), (n1, d1) in zip(nodes[:-1], nodes[1:])]
    return betas + [0.0], [n for n, _ in nodes[1:]]


def _shape(mu_i, mu_inf=None, tau=None, nodes=None, betas=None, events=()):
    b, p = _drift(nodes) if nodes else (list(betas or []), [])
    return dict(mu_i=mu_i, mu_inf=mu_inf, tau=tau, betas=b, breakpoints=p, events=list(events))


# Fig. 3: (a)-(h) the eight break-in curve shapes of Blau, Tribol. Int. 38
# (2005) 1007-1012, Fig. 2, and (i)-(j) the features the model adds.
# Events as (cycle, amplitude, onset, recovery, recovery shape).
GENESIS_SHAPES = dict(
    n_cycles=10_000, sigma=0.010, phi=0.95,
    panels=[
        ("(a) Rise to a plateau", 10, _shape(0.25, 0.60, 600), None),
        ("(b) Peak and decay", 11, _shape(0.20, 0.80, 120, [(600, 0.0), (3000, -0.25)]), None),
        ("(c) Peak, valley and recovery", 12,
         _shape(0.20, 0.75, 100, [(800, 0.0), (3000, -0.35), (3800, -0.35), (6000, -0.18)]), None),
        ("(d) Hump, then late rise", 13,
         _shape(0.15, 0.40, 120, [(600, 0.0), (1300, -0.06), (6000, -0.06), (7200, 0.30)]), None),
        ("(e) Late sharp rise", 14, _shape(0.20, nodes=[(6800, 0.0), (7300, 0.50)]), None),
        ("(f) Decay to a plateau", 15, _shape(0.75, 0.30, 800), None),
        ("(g) Drop, then rise", 16, _shape(0.40, nodes=[(700, -0.18), (4200, -0.18), (6000, 0.12)]), None),
        ("(h) Steep drop, then rise", 17, _shape(0.80, 0.22, 350, [(4500, 0.0), (6200, 0.30)]), None),
        ("(i) Unexpected events", 18,
         _shape(0.12, 0.45, 300, events=[(3500, -0.10, 100, 600, "exponential"),
                                         (6200, +0.08, 60, 450, "sigmoid"),
                                         (8300, -0.07, 80, 700, "linear")]), None),
        ("(j) Asymmetry and drift", 19, _shape(0.15, 0.45, 300, betas=[+6e-6]),
         _shape(0.12, 0.38, 300, betas=[-4e-6])),
    ],
)

# Figs. 4 and 8: Coulomb and non-Coulomb loops, crank-slider drive (r/l = 0.5)
LOOP_ILLUSTRATION = dict(
    P=200.0, S=1000.0, x0=1.3, mu_k=0.45, D_c=0.30, lam_acc=3.0, lam_dec=8.0,
    n_half=200, z0_over_x0=(0.0, 0.10),
    noise=dict(sampling_rate=10_000.0, normal_load=200.0, force_sensor_range=500.0,
               noise_model="colored", sigma_rel=0.01, sigma_x_abs=0.01,
               sigma_ct=0.01, f_c=100.0),
    noise_seed=42,
)

# Fig. 5: static-to-kinetic transition over a forward half-stroke
SLIP_PROFILE = dict(
    P=500.0, S=1875.0, mu_k=0.30, kappa_acc=1.50, kappa_dec=1.20,
    lam_acc=1.0, lam_dec=0.5, x0=1.0, f_test=1.0, D_c=(0.25, 0.05), n=2000,
)

# Fig. 7: idealised scar cross-sections z = z0 [1 - (x/x0)^p]
SCAR_PROFILES = dict(p=(2, 4, 8), z0_fractions=(0.2, 0.4, 0.6, 0.8, 1.0))

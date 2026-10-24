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
    "static-kinetic (NBR, present study)": dict(
        # 8 mm NBR thin-disc pin on S355 steel, lubricated, 60 C, 2 MPa, 50 mm nominal
        # stroke (measured amplitude 25.5 mm), 0.1 m/s with a trapezoidal profile
        P=99.1, x0=25.5, f_test=1.0, S=300.0, D_c=8.0, z0_over_x0=0.0, ramp=0.10,
        mu_k=(0.130, 0.140), mu_s_acc=(0.310, 0.245), mu_s_dec=(0.153, 0.185),
        lam_acc=0.045, lam_dec=0.11),
    "asymmetric (Kumar et al.)": dict(
        P=20.0, x0=0.99, f_test=5.0, S=150.0, D_c=0.18, z0_over_x0=0.0,
        mu_k=(0.79, 0.275), mu_s_acc=(0.316, 0.11), mu_s_dec=(0.79, 0.275),
        lam_acc=_per_second(4.0, 5.0), lam_dec=_per_second(5.0, 5.0)),
    "noisy, mild (Kafexhiu et al.)": dict(           # force normalised by the load (P = 1)
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
    S=35_000.0,                       # system stiffness reported by Jin et al. for their rig
    D_c=0.006, kappa_acc=0.85,
    lam_acc=_per_second(60.0, 20.0), lam_dec=_per_second(5.0, 20.0),
    R_cyl=6.0, L_cyl=10.0, E=210_000.0, nu=0.30,      # steel cylinder on steel flat
    scar_volume=0.3162,               # measured Example 2 scar: volume [mm^3]
    scar_half_width=0.3330,           #   parabolic-equivalent half-width W [mm]
    event_onset=400, event_recovery=9000,
)

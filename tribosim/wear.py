"""
wear.py
-------
Energy-based wear model for reciprocating ball-on-flat contacts.

Physics
~~~~~~~
Wear volume per cycle is proportional to the dissipated frictional energy,
computed independently for the ball and the flat:

    ΔV_ball_i = α_ball · F_N · (2·x0) · (μ_FW_i + μ_BW_i)
    ΔV_flat_i = α_flat · F_N · (2·x0) · (μ_FW_i + μ_BW_i)

The flat wear scar cross-section is a power-law bowl of maximum depth z0 and
half-length x0 + a_eff (the stroke plus the contact half-width, so the
footprint reaching beyond each reversal is included, as for the cylinder
half-width W):

    z(x) = z0 · (1 - (x / L_h)^p),  L_h = x0 + a_eff,  p = scar_shape (even ≥ 2)

Its area A = 2p/(p+1)·z0·L_h (same convention as the cylinder), so the scar depth
z0(n) is recovered from the accumulated flat wear volume analytically.

The ball wear is modelled as a spherical cap of depth h_ball(n), from
which the contact half-width evolves as:

    a(n) = sqrt(2 · R_ball · h_ball(n) - h_ball(n)²)
         ≈ sqrt(2 · R_ball · h_ball(n))   for h_ball << R_ball

The initial contact half-width is set by Hertzian contact mechanics:

    a0 = (3 · F_N · R_eff / (4 · E_star)) ^ (1/3)

    1/R_eff = 1/R_ball   (flat has infinite radius)
    1/E_star = (1 - ν1²)/E1 + (1 - ν2²)/E2

The effective contact half-width at each cycle is:

    a_eff(n) = max(a0, a_ball(n))

because the Hertzian radius is the lower bound before wear widens the contact.

Author : Iñigo Llavori Osa — Mondragon Unibertsitatea
"""

from __future__ import annotations

import numpy as np


_N_SPATIAL = 1000


class WearModel:
    """
    Compute the evolution of the wear scar geometry over *n_cycles*
    for a ball-on-flat reciprocating contact.

    Parameters
    ----------
    x0 : float
        Half-stroke [mm].
    contact_force : float
        Normal contact force F_N [N].
    alpha_flat : float
        Energy wear coefficient for the flat [mm² N⁻¹].
    alpha_ball : float
        Energy wear coefficient for the ball [mm² N⁻¹].
    R_ball : float
        Ball radius [mm].
    E1, E2 : float
        Young's moduli of flat and ball [MPa].
    nu1, nu2 : float
        Poisson's ratios of flat and ball [-].
    scar_shape : int | None
        Exponent *p* of the power-law scar profile (positive even integer).
        If ``None``, chosen automatically from x0:
            x0 ≤ 1.0 mm  →  p = 2
            1.0 < x0 ≤ 5 mm  →  p = 4
            x0 > 5 mm  →  p = 6
    geometry : str
        Contact geometry, ``"ball"`` (default: point contact + spherical-cap
        counterbody) or ``"cylinder"`` (line contact + cylindrical-segment
        counterbody + prismatic self-similar flat scar).
    L_cylinder : float | None
        Axial length of the cylinder [mm]. Required for ``geometry="cylinder"``.
    z0_over_x0 : float | None
        Retained for API compatibility. Not used by the coupled cylinder-on-flat
        model, where the aspect ratio D/W is computed from the wear rather than
        imposed.

    Notes
    -----
    For ``geometry="cylinder"`` the counterbody radius and wear coefficient are
    taken from ``R_ball`` and ``alpha_ball`` (the curved body). The initial
    footprint is the Hertzian line half-width b0 = sqrt(4·F_N·R / (π·L·E*)). The
    cylinder wears as a circular segment that widens the contact to b_eff, the
    flat scar half-width is W = x0 + b_eff, its depth D = (3/4)·V_flat/(L·W), and
    the running aspect ratio z0/x0 = D/W drives the friction loop.
    """

    def __init__(
        self,
        x0: float,
        contact_force: float,
        alpha_flat: float,
        alpha_ball: float,
        R_ball: float,
        E1: float,
        nu1: float,
        E2: float,
        nu2: float,
        scar_shape: int | None = None,
        geometry: str = "ball",
        L_cylinder: float | None = None,
        z0_over_x0: float | None = None,
    ) -> None:
        if x0 <= 0:
            raise ValueError("x0 must be positive.")
        if contact_force <= 0:
            raise ValueError("contact_force must be positive.")
        if alpha_flat < 0 or alpha_ball < 0:
            raise ValueError("alpha_flat and alpha_ball must be ≥ 0.")
        if R_ball <= 0:
            raise ValueError("R_ball must be positive.")
        if E1 <= 0 or E2 <= 0:
            raise ValueError("Young's moduli must be positive.")
        if not (0 <= nu1 < 0.5) or not (0 <= nu2 < 0.5):
            raise ValueError("Poisson's ratios must be in [0, 0.5).")
        if geometry not in ("ball", "cylinder"):
            raise ValueError("geometry must be 'ball' or 'cylinder'.")
        if geometry == "cylinder":
            if L_cylinder is None or L_cylinder <= 0:
                raise ValueError("L_cylinder must be positive for cylinder geometry.")
            # z0_over_x0 is NOT required: the coupled model computes D/W from the wear.

        self.x0            = x0
        self.contact_force = contact_force
        self.alpha_flat    = alpha_flat
        self.alpha_ball    = alpha_ball
        self.R_ball        = R_ball
        self.E1            = E1
        self.nu1           = nu1
        self.E2            = E2
        self.nu2           = nu2
        self.scar_shape    = scar_shape if scar_shape is not None else self._auto_shape(x0)
        self.geometry      = geometry
        self.L_cylinder    = L_cylinder
        self.z0_over_x0    = z0_over_x0

        if self.scar_shape % 2 != 0 or self.scar_shape < 2:
            raise ValueError("scar_shape must be a positive even integer (2, 4, 6, …).")

        # Hertzian contact radius (initial, before any wear)
        self.E_star = self._compute_E_star()
        self.a0     = self._compute_hertz_radius()


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_shape(x0: float) -> int:
        if x0 <= 1.0:
            return 2
        elif x0 <= 5.0:
            return 4
        return 6

    def _compute_E_star(self) -> float:
        """Reduced modulus E* [MPa]."""
        return 1.0 / ((1.0 - self.nu1**2) / self.E1 +
                      (1.0 - self.nu2**2) / self.E2)

    def _compute_hertz_radius(self) -> float:
        """
        Initial Hertzian contact half-width [mm], before any wear.
        Ball (point):    a0 = (3·F_N·R / (4·E_star)) ^ (1/3)
        Cylinder (line): b0 = sqrt(4·F_N·R / (π·L·E_star))
        """
        if self.geometry == "cylinder":
            return float(np.sqrt(4.0 * self.contact_force * self.R_ball /
                                 (np.pi * self.L_cylinder * self.E_star)))
        return (3.0 * self.contact_force * self.R_ball /
                (4.0 * self.E_star)) ** (1.0 / 3.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        mu_fw: np.ndarray,
        mu_bw: np.ndarray,
    ) -> dict:
        """
        Compute wear scar evolution cycle by cycle.

        Parameters
        ----------
        mu_fw, mu_bw : np.ndarray, shape (n_cycles,)
            Forward and backward CoF arrays.

        Returns
        -------
        dict with keys:
            x          : np.ndarray (N_SPATIAL,)  — spatial coordinate [mm]
            z_total    : np.ndarray (n_cycles, N_SPATIAL) — flat scar profile [mm]
            z0         : np.ndarray (n_cycles,) — flat centreline depth [mm]
            half_length: np.ndarray (n_cycles,) — scar half-length x0 + a_eff [mm]
            z0_over_x0 : np.ndarray (n_cycles,) — aspect ratio z0/(x0 + a_eff),
                         the ball analogue of D/W, which drives the loop
            h_ball     : np.ndarray (n_cycles,) — ball wear depth [mm]
            a_contact  : np.ndarray (n_cycles,) — effective contact half-width [mm]
            a0_hertz   : float — initial Hertzian contact half-width [mm]
            E_star     : float — reduced modulus [MPa]

        For ``geometry="cylinder"`` the returned dict instead carries the
        cylinder-on-flat quantities (see :meth:`_compute_cylinder`).
        """
        if self.geometry == "cylinder":
            return self._compute_cylinder(mu_fw, mu_bw)

        # --- Per-cycle incremental wear volumes ----------------------
        energy_term = self.contact_force * (2.0 * self.x0) * (mu_fw + mu_bw)

        delta_V_flat = self.alpha_flat * energy_term
        delta_V_ball = self.alpha_ball * energy_term

        V_flat_cum = np.cumsum(delta_V_flat)   # [mm³]
        V_ball_cum = np.cumsum(delta_V_ball)   # [mm³]

        # --- Ball wear depth and contact half-width (needed for z0) --
        # V_ball_cap = π * h_ball² * (3*R - h_ball) / 3
        h_ball = self._ball_cap_depth(V_ball_cum)

        # Effective contact half-width: a_eff = sqrt(2*R*h - h²) ≥ a0
        a_ball = np.sqrt(np.maximum(
            2.0 * self.R_ball * h_ball - h_ball**2, 0.0
        ))
        a_contact = np.maximum(a_ball, self.a0)

        # --- Flat scar depth from volume -----------------------------
        # The scar is 2·a_contact wide (transverse).  Along the sliding
        # direction the footprint reaches a_contact beyond each reversal, so
        # the worn half-length is L = x0 + a_contact — the ball analogue of the
        # cylinder half-width W = x0 + b_eff.  The removed material is the
        # power-law bowl z(x) = z0·(1-(x/L)^p) over |x| ≤ L, with the same area
        # convention as the cylinder, A = 2p/(p+1)·z0·L, so the 3-D volume is:
        #   V = 2·a_contact · ∫_{-L}^{L} z0·(1-(x/L)^p) dx = 2·a_contact·z0·2Lp/(p+1)
        # → z0 = (p+1)/(2p)·V / (2·a_contact·L)   [= 3/4 factor for p=2].
        # As the ball wears (a_contact grows) the volume spreads over a longer
        # and wider footprint, so the depth increment decelerates.
        depth_factor  = (self.scar_shape + 1.0) / (2.0 * self.scar_shape)
        half_length   = self.x0 + a_contact
        z0 = depth_factor * V_flat_cum / (2.0 * a_contact * half_length)

        # Flat scar profile on a grid spanning the final half-length ----------
        L_end = float(half_length[-1]) if half_length[-1] > 0 else self.x0
        x = np.linspace(-L_end, L_end, _N_SPATIAL)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(half_length[:, np.newaxis] > 0,
                            np.abs(x[np.newaxis, :]) / half_length[:, np.newaxis], 1.0)
        frac = np.clip(frac, 0.0, 1.0)
        z_total = -z0[:, np.newaxis] * (1.0 - frac ** self.scar_shape)

        return dict(
            x         = x,
            z_total   = z_total,
            z0        = z0,
            half_length = half_length,
            z0_over_x0  = z0 / half_length,
            h_ball    = h_ball,
            a_contact = a_contact,
            a0_hertz  = self.a0,
            E_star    = self.E_star,
        )

    def _compute_cylinder(self, mu_fw: np.ndarray, mu_bw: np.ndarray) -> dict:
        """
        Cylinder-on-flat wear evolution.

        The cylinder (counterbody, radius ``R_ball``, length ``L_cylinder``)
        wears as a circular segment whose growth widens the line contact to
        ``b_eff``. The flat scar half-width is the stroke plus that contact
        half-width, its depth follows from the accumulated volume, and the
        running aspect ratio ``z0_over_x0`` = D/W is returned to drive the
        friction loop (so the loop develops Coulomb -> non-Coulomb as it wears).

        Returns
        -------
        dict with keys:
            x          : np.ndarray (N_SPATIAL,)  — scar coordinate [mm]
            z_total    : np.ndarray (n_cycles, N_SPATIAL) — flat scar profile [mm]
            z0         : np.ndarray (n_cycles,) — physical flat scar depth D [mm]
            W_scar     : np.ndarray (n_cycles,) — flat scar half-width W [mm]
            z0_over_x0 : np.ndarray (n_cycles,) — aspect ratio D/W (evolves)
            h_cyl      : np.ndarray (n_cycles,) — cylinder wear depth [mm]
            b_contact  : np.ndarray (n_cycles,) — contact half-width b_eff [mm]
            pressure   : np.ndarray (n_cycles,) — mean contact pressure [MPa]
            b0_hertz   : float — initial Hertzian line half-width [mm]
            E_star     : float — reduced modulus [MPa]
        """
        R, L = self.R_ball, self.L_cylinder

        energy_term = self.contact_force * (2.0 * self.x0) * (mu_fw + mu_bw)
        V_flat_cum = np.cumsum(self.alpha_flat * energy_term)   # flat scar volume [mm³]
        V_cyl_cum  = np.cumsum(self.alpha_ball * energy_term)   # cylinder volume [mm³]

        # --- Counterbody wear: circular-segment depth → contact half-width ----
        h_cyl = self._cyl_segment_depth(V_cyl_cum)
        b_cyl = np.sqrt(np.maximum(2.0 * R * h_cyl - h_cyl**2, 0.0))
        b_eff = np.maximum(b_cyl, self.a0)                      # self.a0 == b0 here
        pressure = self.contact_force / (2.0 * b_eff * L)       # [MPa]

        # --- Flat scar: prismatic groove, coupled to the loop -----------------
        # Half-width = stroke + contact half-width (worn length 2x0 + 2·b_eff);
        # depth from the volume; aspect ratio z0/x0 = D/W drives the loop.
        # Power-law groove z = D·(1-(x/W)^p) has A_cs = 2p/(p+1)·D·W, so
        #   D = (p+1)/(2p)·V/(L·W)   [= 3/4·V/(L·W) for the parabolic p=2].
        depth_factor = (self.scar_shape + 1.0) / (2.0 * self.scar_shape)
        W_scar = self.x0 + b_eff
        D_scar = depth_factor * V_flat_cum / (L * W_scar)
        z0x0   = D_scar / W_scar

        # --- Scar profile on a fixed grid spanning the final half-width -------
        W_end = float(W_scar[-1]) if W_scar[-1] > 0 else self.x0
        x = np.linspace(-W_end, W_end, _N_SPATIAL)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(W_scar[:, np.newaxis] > 0,
                            np.abs(x[np.newaxis, :]) / W_scar[:, np.newaxis], 1.0)
        frac = np.clip(frac, 0.0, 1.0)
        z_total = -D_scar[:, np.newaxis] * (1.0 - frac ** self.scar_shape)

        return dict(
            x          = x,
            z_total    = z_total,
            z0         = D_scar,
            W_scar     = W_scar,
            z0_over_x0 = z0x0,
            h_cyl      = h_cyl,
            b_contact  = b_eff,
            pressure   = pressure,
            b0_hertz   = self.a0,
            E_star     = self.E_star,
        )

    def _cyl_segment_depth(self, V: np.ndarray) -> np.ndarray:
        """
        Invert the cylindrical-segment volume for the wear depth h(V):

            V_cyl(h) = L·[R²·arccos(1 - h/R) - (R - h)·sqrt(2Rh - h²)]

        via Newton-Raphson, with dV/dh = L·2·sqrt(2Rh - h²) (the chord width).
        Starting guess uses the small-depth segment area (4/3)·sqrt(2R)·h^(3/2).
        """
        R, L = self.R_ball, self.L_cylinder
        h = (3.0 * np.maximum(V, 0.0) /
             (4.0 * L * np.sqrt(2.0 * R))) ** (2.0 / 3.0)       # small-h guess
        h = np.clip(h, 0.0, R)

        for _ in range(12):
            root = np.sqrt(np.maximum(2.0 * R * h - h**2, 0.0))
            arg  = np.clip(1.0 - h / R, -1.0, 1.0)
            f  = L * (R**2 * np.arccos(arg) - (R - h) * root) - V
            df = L * 2.0 * root
            mask = df > 1e-15
            h[mask] -= f[mask] / df[mask]
            h = np.clip(h, 0.0, R)

        return h

    def _ball_cap_depth(self, V: np.ndarray) -> np.ndarray:
        """
        Invert V = π·h²·(3R - h)/3  →  h(V)  via Newton-Raphson.

        Starting guess: h0 = sqrt(V / (π·R))  (small-h approximation).
        Converges in 3–5 iterations for h/R < 0.3.
        """
        R  = self.R_ball
        h  = np.sqrt(np.maximum(V / (np.pi * R), 0.0))   # initial guess

        for _ in range(10):
            f  = np.pi * h**2 * (3.0 * R - h) / 3.0 - V
            df = np.pi * h * (2.0 * R - h)
            # Avoid division by zero at h=0
            mask = np.abs(df) > 1e-15
            h[mask] -= f[mask] / df[mask]
            h = np.maximum(h, 0.0)

        return h

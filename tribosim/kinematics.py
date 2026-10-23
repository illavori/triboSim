"""
kinematics.py
-------------
Displacement and velocity profiles for reciprocating tribometer drives.

Three interchangeable kinematic profiles are implemented, all sharing
the same interface: get_stroke(x0, sr, f_test=None) → x_fw, x_bw, v_fw, v_bw.

Velocity units.  With f_test [Hz] given, the velocities are physical,
in mm/s (the phase-angle derivative times ω = 2π·f_test), which is the
unit the Oden-Martins rate constants λ of the friction loop are expressed
in (Table 1 of the paper).  Without f_test the velocities are returned per
radian of crank angle (mm/rad), kept for backward compatibility.

CrankSlider
    Non-sinusoidal profile governed by crank radius r and rod length l
    (Eqs. eq:crank / eq:crank_vel).  The ratio r/l makes the velocity
    profile asymmetric about the centre of the stroke (the velocity peak
    moves towards one end); the forward and backward strokes stay mirror
    images, with the same speed at a given position.

Sinusoidal
    Pure harmonic profile x(t) = x0·sin(ωt), produced by piezoelectric
    or linear motor drives.  Recovered as the limit r/l → 0 of the
    crank-slider (Eq. eq:sinusoidal).

Trapezoidal
    Constant-velocity profile with finite acceleration/deceleration
    ramps of fractional duration ramp_fraction = t_ramp/(T/2),
    typical of ball-screw and servo-motor drives (Eq. eq:trapezoidal_vel).
    As ramp_fraction → 0 the profile approaches an ideal square-wave
    velocity; as ramp_fraction → 0.5 it approaches the sinusoidal case.

Author : Iñigo Llavori Osa — Mondragon Unibertsitatea
"""

from __future__ import annotations

import numpy as np


def _omega(f_test: float | None) -> float:
    """Angular frequency 2π·f_test [rad/s], or 1 when f_test is not given."""
    if f_test is None:
        return 1.0
    if f_test <= 0:
        raise ValueError("f_test must be positive.")
    return 2.0 * np.pi * f_test


class CrankSlider:
    """
    Crank-slider kinematic model.

    Parameters
    ----------
    crank_radius : float
        Crank radius r [mm].  Default 75 mm.
    rod_length : float
        Connecting rod length l [mm].  Default 150 mm (r/l = 0.5).

    The crank geometry only sets the shape of the profile, which is
    normalised and rescaled to the half-stroke x0 in get_stroke().
    n_points : int
        Number of discretisation points per full revolution (2π).
        The forward and backward strokes each use half of these points.
    """

    def __init__(
        self,
        crank_radius: float = 75.0,
        rod_length: float = 150.0,
        n_points: int = 2000,
    ) -> None:
        if crank_radius <= 0 or rod_length <= 0:
            raise ValueError("Crank radius and rod length must be positive.")
        if rod_length <= crank_radius:
            raise ValueError(
                "Rod length must exceed crank radius (l > r)."
            )
        if n_points < 4 or n_points % 2 != 0:
            raise ValueError("n_points must be an even integer ≥ 4.")

        self.R = crank_radius
        self.L = rod_length
        self.n_points = n_points

        self._position_norm, self._velocity_norm = self._solve()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _solve(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve kinematics for one full crank revolution.

        Returns
        -------
        pos_norm : np.ndarray, shape (n_points,)
            Position normalised to [−1, 1].
        vel_norm : np.ndarray, shape (n_points,)
            Velocity normalised by R·ω (dimensionless).
        """
        theta2 = np.linspace(0.0, 2.0 * np.pi, self.n_points, endpoint=False)
        theta3 = np.arcsin(-(self.R / self.L) * np.sin(theta2))

        x_raw = (self.R + self.L) - self.R * np.cos(theta2) - self.L * np.cos(theta3)

        # Normalise to [−1, 1]
        x_min, x_max = x_raw.min(), x_raw.max()
        pos_norm = 2.0 * (x_raw - x_min) / (x_max - x_min) - 1.0

        # Velocity: dpos/dtheta (chain rule, ω cancels in normalisation)
        vel_norm = np.gradient(pos_norm, theta2)

        return pos_norm, vel_norm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stroke(
        self,
        x0: float,
        sr: int,
        f_test: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the forward and backward displacement/velocity arrays
        sampled at *sr* points per half-stroke.

        The crank-slider profile is resampled to exactly *sr* points
        for both the forward (0 → x0) and backward (x0 → 0) half-strokes.

        Parameters
        ----------
        x0 : float
            Half-stroke amplitude [mm].
        sr : int
            Number of samples per half-cycle (sampling points per stroke).
        f_test : float, optional
            Test frequency [Hz].  When given, velocities are in mm/s.

        Returns
        -------
        x_fw : np.ndarray, shape (sr,)
            Forward stroke displacement [mm].
        x_bw : np.ndarray, shape (sr,)
            Backward stroke displacement [mm].
        v_fw : np.ndarray, shape (sr,)
            Forward stroke velocity [mm/s with f_test, else mm/rad].
        v_bw : np.ndarray, shape (sr,)
            Backward stroke velocity [mm/s with f_test, else mm/rad].
        """
        half = self.n_points // 2

        # Resample to sr points via linear interpolation
        idx_fw = np.linspace(0, half - 1, sr)
        idx_bw = np.linspace(half, self.n_points - 1, sr)

        def resample(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
            return np.interp(idx, np.arange(len(arr)), arr)

        x_fw = resample(self._position_norm, idx_fw) * x0
        x_bw = resample(self._position_norm, idx_bw) * x0

        w = _omega(f_test)
        v_fw = resample(self._velocity_norm, idx_fw) * x0 * w
        v_bw = resample(self._velocity_norm, idx_bw) * x0 * w

        return x_fw, x_bw, v_fw, v_bw


class Sinusoidal:
    """
    Pure sinusoidal kinematic profile (Eq. eq:sinusoidal).

        x(θ) = x0 · (−cos θ),   θ ∈ [0, π]   (forward stroke)
        ẋ(θ) = x0 · sin θ

    This is the limit r/l → 0 of the crank-slider: forward and backward
    strokes are perfectly symmetric and the velocity profile is smooth.
    Produced by piezoelectric actuators and linear motor drives.
    """

    def get_stroke(
        self,
        x0: float,
        sr: int,
        f_test: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return forward / backward displacement and velocity arrays
        (velocity in mm/s when f_test [Hz] is given, else mm/rad)."""
        theta_fw = np.linspace(0.0, np.pi, sr)
        theta_bw = np.linspace(np.pi, 2.0 * np.pi, sr)

        x_fw = x0 * (-np.cos(theta_fw))          # −x0 → +x0
        x_bw = x0 * (-np.cos(theta_bw))          # +x0 → −x0
        w = _omega(f_test)
        v_fw = x0 * w * np.sin(theta_fw)         # ≥ 0
        v_bw = x0 * w * np.sin(theta_bw)         # ≤ 0

        return x_fw, x_bw, v_fw, v_bw


class Trapezoidal:
    """
    Trapezoidal velocity profile (Eq. eq:trapezoidal_vel).

    The velocity ramps linearly from 0 to ±v0 over a fraction
    *ramp_fraction* of the half-cycle, holds constant at ±v0, then
    ramps back to 0.  v0 is set so that the total displacement equals
    2·x0 per half-stroke.

    Parameters
    ----------
    ramp_fraction : float
        t_ramp / (T/2), in the open interval (0, 0.5).
        As ramp_fraction → 0 the profile approaches a square wave.
        As ramp_fraction → 0.5 it approaches a triangular (near-sinusoidal)
        velocity profile.
    """

    def __init__(self, ramp_fraction: float = 0.1) -> None:
        if not 0.0 < ramp_fraction < 0.5:
            raise ValueError("ramp_fraction must be in (0, 0.5).")
        self.ramp_fraction = ramp_fraction

    def get_stroke(
        self,
        x0: float,
        sr: int,
        f_test: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return forward / backward displacement and velocity arrays
        (velocity in mm/s when f_test [Hz] is given, else mm/rad)."""
        # Parametrise the half-stroke on θ ∈ [0, π] to keep units
        # consistent with CrankSlider and Sinusoidal (mm / radian).
        theta = np.linspace(0.0, np.pi, sr)
        theta_ramp = np.pi * self.ramp_fraction

        # Normalised velocity shape (peak = 1)
        v_shape = np.where(
            theta < theta_ramp,
            theta / theta_ramp,
            np.where(
                theta <= np.pi - theta_ramp,
                1.0,
                (np.pi - theta) / theta_ramp,
            ),
        )

        # Scale so that ∫₀^π v dθ = 2·x0  →  v0 = 2·x0 / (π − θ_ramp)
        v0 = 2.0 * x0 / (np.pi - theta_ramp)
        v_fw = v_shape * v0                          # forward velocity ≥ 0

        # Displacement by cumulative integration (trapezoidal rule)
        dtheta = np.pi / (sr - 1) if sr > 1 else np.pi
        x_fw = np.concatenate([[0.0], np.cumsum(0.5 * (v_fw[:-1] + v_fw[1:]) * dtheta)])
        x_fw -= x0                                   # shift so x_fw[0] = −x0

        # Backward stroke: mirror
        x_bw = x_fw[::-1]
        v_bw = -v_fw[::-1]

        # Physical velocity [mm/s] when the test frequency is known
        w = _omega(f_test)
        v_fw = v_fw * w
        v_bw = v_bw * w

        return x_fw, x_bw, v_fw, v_bw

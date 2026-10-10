"""
acquisition.py
--------------
Signal acquisition noise model for the RSTT simulator.

Two physically distinct noise sources corrupt the clean signals:

1. Instrumental noise (AR1, both Q and x):
       eta_inst(t_k) = alpha * eta_inst(t_{k-1}) + sqrt(1-alpha^2) * N(0, sigma)
       alpha         = exp(-2*pi*f_c / f_s)
       sigma_Q_inst  = sigma_rel * F_fs     [N]   — fraction of sensor full-scale
       sigma_x       = sigma_x_abs          [mm]  — absolute

   The sqrt(1-alpha^2) coefficient ensures std(eta_inst) = sigma in steady
   state for any alpha.  Setting alpha = 0 recovers white Gaussian noise.

2. Tribological noise (white Gaussian, Q only):
       eta_tribo(t_k) = N(0, sigma_ct * P)
   Scales with normal load P; a simple stand-in for the scatter of a steadily
   sliding contact (structured noise such as stick-slip is not modelled).

Composite force signal:
       Q_noisy = Q_clean + eta_inst + eta_tribo

Displacement receives only instrumental AR1 noise (no tribological
counterpart).  Setting sigma_ct = 0 recovers the purely instrumental model.

Reference
---------
    Standard AR1 / coloured-noise model; see e.g.
    Kasdin (1995) Proc. IEEE 83(5) for spectral properties.

Author : Iñigo Llavori Osa — Mondragon Unibertsitatea
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# AR1 engine — use scipy.signal.lfilter when available for speed, otherwise
# fall back to a pure-NumPy causal loop (equivalent result).
# ---------------------------------------------------------------------------
try:
    from scipy.signal import lfilter as _lfilter

    def _ar1(n: int, alpha: float, sigma: float,
             rng: np.random.Generator) -> np.ndarray:
        """AR1 noise via scipy IIR filter (fast path).
        Innovation coefficient sqrt(1-alpha^2) ensures std(eta) = sigma."""
        c = float(np.sqrt(1.0 - alpha ** 2))
        w = rng.normal(0.0, sigma, n)
        return _lfilter([c], [1.0, -alpha], w)

except ImportError:

    def _ar1(n: int, alpha: float, sigma: float,  # type: ignore[misc]
             rng: np.random.Generator) -> np.ndarray:
        """AR1 noise via causal loop (fallback, no scipy needed).
        Innovation coefficient sqrt(1-alpha^2) ensures std(eta) = sigma."""
        c = float(np.sqrt(1.0 - alpha ** 2))
        w = rng.normal(0.0, sigma, n)
        eta = np.empty(n)
        eta[0] = c * w[0]
        for k in range(1, n):
            eta[k] = alpha * eta[k - 1] + c * w[k]
        return eta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_noise(
    q_clean: np.ndarray,
    x_clean: np.ndarray,
    sampling_rate: float,
    normal_load: float,
    force_sensor_range: float,
    noise_model: str = "colored",
    sigma_rel: float = 0.01,
    sigma_x_abs: float = 0.002,
    sigma_ct: float = 0.001,
    f_c: float = 100.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add composite noise to Q and x signals.

    Two independent sources are superimposed on the force channel:

    1. Instrumental AR1 noise (force and displacement):
           eta_inst(t_k) = alpha*eta_inst(t_{k-1}) + sqrt(1-alpha^2)*N(0,sigma)
           sigma_Q_inst  = sigma_rel * force_sensor_range
           sigma_x       = sigma_x_abs
       The sqrt(1-alpha^2) coefficient guarantees std(eta_inst) = sigma
       for any alpha.  alpha = 0 (noise_model='white') gives white noise.

    2. Tribological noise — force only, white Gaussian:
           eta_tribo(t_k) = N(0, sigma_ct * normal_load)
       Represents asperity-junction, wear-particle, and contact-instability
       noise.  Scales with normal load; absent from displacement.

    Composite:
           Q_noisy = Q_clean + eta_inst_Q + eta_tribo
           x_noisy = x_clean + eta_inst_x

    Setting sigma_ct = 0 recovers the purely instrumental model.

    Parameters
    ----------
    q_clean : ndarray
        Clean tangential force signal [N].
    x_clean : ndarray
        Clean displacement signal [mm].
    sampling_rate : float
        Acquisition sampling frequency f_s [Hz].
    normal_load : float
        Normal contact force P [N].  Scales tribological noise amplitude.
    force_sensor_range : float
        Full-scale range of the force sensor F_fs [N].
        Sets sigma_Q_inst = sigma_rel * force_sensor_range.
    noise_model : str
        ``'white'``   — uncorrelated instrumental noise (alpha = 0).
        ``'colored'`` — AR1 correlated noise with corner frequency f_c.
    sigma_rel : float
        Instrumental force noise as fraction of sensor full-scale.
        Default 0.01 → 1 % of full scale.
    sigma_x_abs : float
        Displacement noise standard deviation [mm].  Default 0.002 mm.
    sigma_ct : float
        Tribological noise coefficient; sigma_tribo = sigma_ct * P [N].
        Default 0.001.  Set to 0 to disable tribological noise.
    f_c : float
        Instrumental noise corner frequency [Hz].  Only used when
        noise_model='colored'.  Default 100 Hz.
    rng : numpy.random.Generator, optional
        Random generator for reproducibility.  A fresh generator is
        created if None.

    Returns
    -------
    q_noisy : ndarray
        Noisy tangential force, same shape as *q_clean*.
    x_noisy : ndarray
        Noisy displacement, same shape as *x_clean*.
    """
    if noise_model not in ("white", "colored"):
        raise ValueError(
            f"noise_model must be 'white' or 'colored', got {noise_model!r}."
        )
    if rng is None:
        rng = np.random.default_rng()

    n = q_clean.size

    sigma_Q_inst = sigma_rel * force_sensor_range
    sigma_x      = sigma_x_abs

    if noise_model == "white":
        alpha = 0.0
    else:
        alpha = float(np.exp(-2.0 * np.pi * f_c / sampling_rate))
        alpha = np.clip(alpha, 0.0, 1.0 - 1e-9)

    # Instrumental AR1 noise (force + displacement)
    eta_inst_Q = _ar1(n, alpha, sigma_Q_inst, rng)
    eta_inst_x = _ar1(n, alpha, sigma_x, rng)

    # Tribological white noise (force only)
    sigma_tribo = sigma_ct * normal_load
    eta_tribo   = (rng.normal(0.0, sigma_tribo, n).reshape(q_clean.shape)
                   if sigma_tribo > 0 else 0.0)

    q_noisy = q_clean + eta_inst_Q.reshape(q_clean.shape) + eta_tribo
    x_noisy = x_clean + eta_inst_x.reshape(x_clean.shape)

    return q_noisy, x_noisy

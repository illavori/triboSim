"""
genesis.py
----------
Models the evolution of the coefficient of friction (CoF) over the
lifetime of a reciprocating friction test (Section 2.1 of the paper).

Physics / model
~~~~~~~~~~~~~~~
The kinetic CoF at cycle n is the sum of four terms (Eq. eq:genesis):

    mu(n) = mu_i + (mu_inf - mu_i)(1 - exp(-n/tau))      run-in transient
            + d(n)                                        drift
            + sum_j delta_j(n)                            events
            + eps(n)                                      fluctuation

* Run-in transient : exponential approach from the initial CoF mu_i to the
                     asymptotic value mu_inf with time constant tau [cycles].
                     Optional: without tau the run-in is absent and the
                     backbone starts at mu_i.
* Drift            : continuous piecewise-linear trend (Eq. eq:drift), with
                     rate beta_k [1/cycle] on segment k and breakpoints
                     0 = n_0 < n_1 < ... < n_K.  One segment is an ordinary
                     linear drift, beta_k = 0 a plateau.
* Events           : bidirectional excursions.  Each one ramps linearly to a
                     signed amplitude over a short onset window (negative
                     for a drop, positive for a peak) and then recovers to
                     the backbone with a selectable shape (linear,
                     exponential, logarithmic or sigmoid).
* Fluctuation      : zero-mean first-order autoregressive process (AR1) with
                     standard deviation sigma and lag-one autocorrelation phi
                     per cycle, the correlated scatter left by the backbone
                     in measured traces (Table 2 of the paper).

The forward and backward strokes are modelled with two independent
instances.  All methods return ``np.ndarray`` of shape ``(n_cycles,)``.

Author : Iñigo Llavori Osa - Mondragon Unibertsitatea
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

RECOVERY_SHAPES = ("linear", "exponential", "logarithmic", "sigmoid")


@dataclass
class GenesisEvent:
    """One abrupt CoF excursion delta_j(n).

    Parameters
    ----------
    cycle : float
        Cycle at which the excursion peaks (end of the onset ramp).
    amplitude : float
        Signed excursion: negative for a drop, positive for a peak.
    onset : int
        Length of the abrupt linear ramp before *cycle* [cycles].
    recovery : int
        Length of the recovery back to the backbone [cycles].
    shape : str
        Recovery shape, one of ``RECOVERY_SHAPES``.
    """

    cycle: float
    amplitude: float
    onset: int = 1
    recovery: int = 1
    shape: str = "exponential"


class FrictionGenesis:
    """
    Generate a synthetic CoF-vs-cycle trajectory (Eq. eq:genesis).

    Parameters
    ----------
    n_cycles : int
        Total number of test cycles.
    mu_i : float
        Initial CoF, mu(0).
    mu_inf : float, optional
        Asymptotic CoF at the end of the run-in.  Ignored without *tau*.
    tau : float, optional
        Run-in time constant [cycles].  ``None`` -> no run-in transient.
    betas : sequence of float
        Drift rates beta_k [1/cycle], one per segment.  Empty -> no drift.
    breakpoints : sequence of float
        Interior breakpoints n_1 < ... < n_{K-1} [cycles]; ``len(betas) - 1``
        values.
    events : sequence of GenesisEvent
        Abrupt excursions.  See :meth:`random_events` for synthetic ones.
    sigma : float
        Standard deviation of the AR1 fluctuation (CoF units).  0 -> none.
    phi : float
        Lag-one autocorrelation of the fluctuation per cycle, in [0, 1).
    random_seed : int | None
        Seed for reproducibility.  ``None`` -> non-deterministic.
    """

    def __init__(
        self,
        n_cycles: int,
        mu_i: float,
        mu_inf: float | None = None,
        tau: float | None = None,
        betas: Sequence[float] = (),
        breakpoints: Sequence[float] = (),
        events: Sequence[GenesisEvent] = (),
        sigma: float = 0.0,
        phi: float = 0.0,
        random_seed: int | None = None,
    ) -> None:
        if n_cycles < 2:
            raise ValueError("n_cycles must be >= 2.")
        if mu_i <= 0 or (mu_inf is not None and mu_inf <= 0):
            raise ValueError("CoF values must be positive.")
        if tau is not None and tau <= 0:
            raise ValueError("tau must be positive.")
        if len(betas) and len(breakpoints) != len(betas) - 1:
            raise ValueError("breakpoints must have len(betas) - 1 values.")
        if np.any(np.diff(breakpoints) <= 0):
            raise ValueError("breakpoints must be strictly increasing.")
        if sigma < 0:
            raise ValueError("sigma must be >= 0.")
        if not 0.0 <= phi < 1.0:
            raise ValueError("phi must be in [0, 1).")
        for ev in events:
            if ev.shape not in RECOVERY_SHAPES:
                raise ValueError(f"unknown recovery shape {ev.shape!r}.")

        self.n_cycles = n_cycles
        self.mu_i = mu_i
        self.mu_inf = mu_inf if mu_inf is not None else mu_i
        self.tau = tau
        self.betas = list(betas)
        self.breakpoints = list(breakpoints)
        self.events = list(events)
        self.sigma = sigma
        self.phi = phi
        self.rng = np.random.default_rng(random_seed)
        self.cycles = np.arange(1, n_cycles + 1, dtype=float)

    # ------------------------------------------------------------------
    # Terms of Eq. eq:genesis
    # ------------------------------------------------------------------

    def run_in(self) -> np.ndarray:
        """Exponential run-in transient, or the constant mu_i without tau."""
        n = self.cycles
        if self.tau is None:
            return np.full_like(n, self.mu_i)
        return self.mu_i + (self.mu_inf - self.mu_i) * (1.0 - np.exp(-n / self.tau))

    def drift(self) -> np.ndarray:
        """Continuous piecewise-linear drift d(n) (Eq. eq:drift), d(0) = 0."""
        n = self.cycles
        d = np.zeros_like(n)
        pts = [0.0] + self.breakpoints + [np.inf]
        for k, beta in enumerate(self.betas):
            d += beta * (np.clip(n, pts[k], pts[k + 1]) - pts[k])
        return d

    def backbone(self) -> np.ndarray:
        """Deterministic trajectory: run-in plus drift."""
        return self.run_in() + self.drift()

    @staticmethod
    def _recovery_weights(t: np.ndarray, shape: str) -> np.ndarray:
        """Recovery weight w(t), w(0) = 0 -> w(1) = 1, for normalised t in [0, 1].

        ``"exponential"`` recovers fast and then slowly, ``"logarithmic"``
        likewise but more gently, ``"sigmoid"`` slow-fast-slow and
        ``"linear"`` at a constant rate.
        """
        if shape == "exponential":
            k = 5.0
            return (1.0 - np.exp(-k * t)) / (1.0 - np.exp(-k))
        if shape == "logarithmic":
            return np.log1p(t * (np.e - 1.0))
        if shape == "sigmoid":
            k, mid = 10.0, 0.5
            sig = 1.0 / (1.0 + np.exp(-k * (t - mid)))
            return (sig - sig[0]) / (sig[-1] - sig[0])
        return t

    def event_term(self) -> np.ndarray:
        """Sum of the event excursions, sum_j delta_j(n)."""
        out = np.zeros(self.n_cycles)
        for ev in self.events:
            pos = int(round(ev.cycle)) - 1                 # 0-based index of the extreme
            if not 0 <= pos < self.n_cycles:
                continue
            lo = max(0, pos - max(int(ev.onset), 1))
            if pos > lo:                                   # abrupt linear onset
                out[lo:pos] += np.linspace(0.0, ev.amplitude, pos - lo, endpoint=False)
            hi = min(self.n_cycles, pos + max(int(ev.recovery), 1))
            t = np.arange(hi - pos) / max(int(ev.recovery) - 1, 1)
            out[pos:hi] += ev.amplitude * (1.0 - self._recovery_weights(t, ev.shape))
        return out

    def fluctuation(self) -> np.ndarray:
        """Zero-mean AR1 fluctuation eps(n), standard deviation sigma, lag-one phi."""
        if self.sigma == 0.0:
            return np.zeros(self.n_cycles)
        e = self.rng.standard_normal(self.n_cycles) * self.sigma * np.sqrt(1.0 - self.phi**2)
        x = np.empty(self.n_cycles)
        x[0] = self.rng.standard_normal() * self.sigma        # start in the stationary state
        for i in range(1, self.n_cycles):
            x[i] = self.phi * x[i - 1] + e[i]
        return x

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, add_events: bool = True, add_fluctuation: bool = True) -> np.ndarray:
        """Return mu(n) of Eq. eq:genesis for n = 1 ... n_cycles."""
        mu = self.backbone()
        if add_events:
            mu = mu + self.event_term()
        if add_fluctuation:
            mu = mu + self.fluctuation()
        return mu

    @staticmethod
    def random_events(
        n_cycles: int,
        n_events: int,
        amplitude: tuple[float, float] = (0.02, 0.10),
        sign: str = "both",
        onset_frac: float = 0.01,
        recovery_frac: tuple[float, float] = (0.03, 0.06),
        window: tuple[float, float] = (0.1, 0.9),
        shape: str = "exponential",
        random_seed: int | None = None,
    ) -> list[GenesisEvent]:
        """Draw *n_events* events at random positions, for synthetic tests.

        Parameters
        ----------
        amplitude : (min, max)
            Range of the absolute excursion (CoF units).
        sign : ``"both"``, ``"drop"`` or ``"peak"``
            Direction of the excursions.
        onset_frac, recovery_frac
            Onset length and recovery-length range as fractions of n_cycles.
        window : (start, end)
            Fraction of the test in which the events may occur.
        """
        rng = np.random.default_rng(random_seed)
        lo, hi = int(window[0] * n_cycles), int(window[1] * n_cycles)
        if n_events <= 0 or hi - lo < n_events:
            return []
        pos = np.sort(rng.choice(np.arange(lo, hi), size=n_events, replace=False))
        out = []
        for p in pos:
            mag = rng.uniform(*amplitude)
            if sign == "drop" or (sign == "both" and rng.random() < 0.5):
                mag = -mag
            out.append(GenesisEvent(
                cycle=float(p + 1), amplitude=float(mag),
                onset=max(1, int(onset_frac * n_cycles)),
                recovery=max(1, int(rng.uniform(*recovery_frac) * n_cycles)),
                shape=shape))
        return out

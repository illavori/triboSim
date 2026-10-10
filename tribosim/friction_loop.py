"""
friction_loop.py
----------------
Computes the synthetic tangential-force (Q) vs displacement (x) loop
for one cycle of a reciprocating friction test.

Model summary
~~~~~~~~~~~~~
1. **Gross-sliding Q** (Jin et al. 2016, after Mulvihill et al. 2011):

       Q_fwd = P · [μ + f·(x − Q/S)] / [1 − μ·f·(x − Q/S)]
       Q_bwd = P · [−μ + f·(x − Q/S)] / [1 + μ·f·(x − Q/S)]

   with  f = 2·z0 / x0²  (wear scar geometry factor).
   Solved by fixed-point iteration (8 steps), since Q appears on both sides.

2. **Jenkins + D_c slip-velocity recovery**:

       v_slip = |v| · (1 − exp(−s / D_c)),   s = |x − x_{stick_onset}|

   A single slip distance s (since unsticking) is used over the whole
   stroke. There is no braking term: v_slip returns to zero at the next
   reversal because |v| does. D_c [mm] = characteristic slip distance.

   D_c captures the effect of finite contact stiffness: immediately after
   the stick→slide transition the true slip velocity is zero (the contact
   spring is still absorbing displacement), and recovers toward the machine
   velocity over ~3·D_c of displacement.  D_c = 0 → v_slip = |v| everywhere.

3. **Oden-Martins CoF** (asymmetric acc / dec):

       μ^acc = μ_k + (μ_s^acc − μ_k) · exp(−λ^acc · v_slip)
       μ^dec = μ_k + (μ_s^dec − μ_k) · exp(−λ^dec · v_slip)

   μ_s and λ are blended linearly over a few samples around the
   velocity peak, so μ is continuous at the acc/dec switch.

   The acc phase peaks at μ_s^acc (post-reversal, from fully static state);
   the dec phase peaks at the lower μ_s^dec (pre-reversal, sliding history).
   The acc/dec split is detected from the velocity peak — no explicit ẍ needed.

4. **Stick (micro-slip) segments**: Q = S·(x − x_reversal), a line
   through the reversal point with slope S.  The stick segment ends where
   the Oden-Martins Q curve first meets the stiffness line.

The loop is assembled from four segments per stroke:
    [stick | Oden-Martins gross (acc half) | Oden-Martins gross (dec half)]
    × forward + backward

Author : Iñigo Llavori Osa — Mondragon Unibertsitatea
"""

from __future__ import annotations

import numpy as np


# Number of fixed-point iterations for the implicit Q equation
_N_ITER = 8


class FrictionLoop:
    """
    Compute the Q-x friction loop for one cycle.

    Parameters
    ----------
    contact_force : float
        Normal force P [N].
    stiffness : float
        Rig (tribometer) stiffness S [N/mm], the slope of the stick branch.
    d_c : float
        Characteristic slip distance D_c [mm].  Controls how fast the
        slip velocity recovers toward the machine velocity after the
        stick→slide transition:
            v_slip = |v_machine| · (1 − exp(−s / D_c))
        D_c = 0 → instantaneous recovery (v_slip = |v_machine| always).
    lambda_acc : float
        Oden-Martins rate-decay constant λ^acc [s/mm] for the acceleration
        phase (post-reversal).  High λ → sharp CoF drop; low λ → gradual.
    lambda_dec : float
        Oden-Martins rate-decay constant λ^dec [s/mm] for the deceleration
        phase (pre-reversal).
    contact_stiffness : float | None
        Initial tangential stiffness of the contact K_c [N/mm] (Mindlin,
        8·a·G* for a sphere on a plane).  ``None`` (default) keeps the
        Jenkins stick branch, a straight line of slope S, and reproduces the
        validated model exactly.  A finite value puts a Mindlin-Masing contact
        in series with the linear rig compliance 1/S, so the stick branch
        curves into gross sliding and partial slip loops (fretting) appear
        when the stroke is too short to reach gross sliding:

            x0 < μ_s·P/S + 3·μ_s·P/(2·K_c)   →  partial slip
    """

    def __init__(
        self,
        contact_force: float,
        stiffness: float,
        d_c: float = 0.15,
        lambda_acc: float = 5.0,
        lambda_dec: float = 2.0,
        contact_stiffness: float | None = None,
    ) -> None:
        if contact_force <= 0:
            raise ValueError("contact_force must be positive.")
        if stiffness <= 0:
            raise ValueError("stiffness must be positive.")
        if d_c < 0:
            raise ValueError("d_c must be >= 0.")
        if lambda_acc < 0:
            raise ValueError("lambda_acc must be >= 0.")
        if lambda_dec < 0:
            raise ValueError("lambda_dec must be >= 0.")
        if contact_stiffness is not None and contact_stiffness <= 0:
            raise ValueError("contact_stiffness must be positive or None.")

        self.P = contact_force
        self.S = stiffness
        self.d_c = d_c
        self.lambda_acc = lambda_acc
        self.lambda_dec = lambda_dec
        self.K_c = contact_stiffness

    # ------------------------------------------------------------------
    # Private: core Q calculations
    # ------------------------------------------------------------------

    def _q_forward(
        self,
        x: np.ndarray,
        mu: float | np.ndarray,
        z0: float,
        x0: float,
    ) -> np.ndarray:
        """Gross-sliding Q for forward stroke (vectorised, iterative).
        Accepts scalar or array mu (element-wise when array)."""
        f = 2.0 * z0 / x0 ** 2
        Q = self.P * (mu + f * x) / np.where(np.abs(1.0 - mu * f * x) > 1e-12,
                                               1.0 - mu * f * x, 1e-12)
        for _ in range(_N_ITER):
            slip = x - Q / self.S
            denom = 1.0 - mu * f * slip
            denom = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
            Q = self.P * (mu + f * slip) / denom
        return Q

    def _q_backward(
        self,
        x: np.ndarray,
        mu: float | np.ndarray,
        z0: float,
        x0: float,
    ) -> np.ndarray:
        """Gross-sliding Q for backward stroke (vectorised, iterative).
        Accepts scalar or array mu (element-wise when array)."""
        f = 2.0 * z0 / x0 ** 2
        Q = self.P * (-mu + f * x) / np.where(np.abs(1.0 + mu * f * x) > 1e-12,
                                                1.0 + mu * f * x, 1e-12)
        for _ in range(_N_ITER):
            slip = x - Q / self.S
            denom = 1.0 + mu * f * slip
            denom = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
            Q = self.P * (-mu + f * slip) / denom
        return Q

    @staticmethod
    def _find_transition_index(
        q_gross: np.ndarray,
        q_stiffness_line: np.ndarray,
    ) -> int:
        """
        Find the first index where *q_gross* crosses *q_stiffness_line*
        using the minimum-distance criterion (robust to noisy arrays).
        """
        diff = np.abs(q_gross - q_stiffness_line)
        return int(np.argmin(diff))

    @staticmethod
    def _mu_sliding(
        x: np.ndarray,
        v: np.ndarray,
        idx_stick: int,
        idx_peak: int,
        mu_k: float,
        mu_s_acc: float,
        mu_s_dec: float,
        d_c: float,
        lam_acc: float,
        lam_dec: float,
        n_blend: int = 15,
    ) -> np.ndarray:
        """
        Oden-Martins CoF for the full sliding phase (acc + dec).

        A single slip-velocity profile is used for the entire stroke:

            s_acc  = |x − x[idx_stick]|   (monotonically increasing)
            v_slip = |v| · (1 − exp(−s_acc / D_c))

        This ensures v_slip is continuous through the velocity peak (no
        jump in formula at the acc/dec boundary).  The CoF recovers
        naturally toward μ_s near each reversal because |v| → 0 there,
        which drives v_slip → 0 and μ → μ_s without any s_brake term.

        The acc/dec asymmetry comes from different μ_s and λ for each
        phase.  To avoid a discontinuity at the acc/dec switching point,
        both μ_s and λ are linearly blended over a window of n_blend
        samples centred at idx_peak.  The resulting transition is smooth
        and physically represents the gradual change in junction state
        near the velocity maximum.
        """
        n = len(x)
        idx = np.arange(n)

        s_acc = np.abs(x - x[idx_stick])
        if d_c > 0:
            v_slip = np.abs(v) * (1.0 - np.exp(-s_acc / d_c))
        else:
            v_slip = np.abs(v)

        # Linear blend of μ_s and λ across n_blend samples at idx_peak
        i0 = max(idx_stick, idx_peak - n_blend)
        i1 = min(n - 1,      idx_peak + n_blend)
        # alpha = 0 → acc params, alpha = 1 → dec params
        alpha = np.clip((idx - i0) / max(i1 - i0, 1), 0.0, 1.0)

        mu_s_arr = (1.0 - alpha) * mu_s_acc + alpha * mu_s_dec
        lam_arr  = (1.0 - alpha) * lam_acc  + alpha * lam_dec

        mu = mu_k + (mu_s_arr - mu_k) * np.exp(-lam_arr * v_slip)
        mu[:idx_stick] = mu_s_acc   # still stuck → static CoF
        return mu

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        z0: float,
        x0: float,
        mu_k_fw: float,
        mu_k_bw: float,
        mu_s_acc_fw: float,
        mu_s_acc_bw: float,
        mu_s_dec_fw: float,
        mu_s_dec_bw: float,
        x_fw: np.ndarray,
        x_bw: np.ndarray,
        v_fw: np.ndarray,
        v_bw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Assemble the full Q-x loop for one cycle using the Jenkins + D_c
        slip-velocity recovery combined with the asymmetric Oden-Martins CoF.

        Parameters
        ----------
        z0 : float
            Wear scar depth [mm].
        x0 : float
            Half-stroke [mm].
        mu_k_fw, mu_k_bw : float
            Kinetic CoF for forward / backward stroke.
        mu_s_acc_fw, mu_s_acc_bw : float
            Static CoF for the acceleration phase (post-reversal).
        mu_s_dec_fw, mu_s_dec_bw : float
            Static CoF for the deceleration phase (pre-reversal).
        x_fw, x_bw : np.ndarray, shape (sr,)
            Forward / backward displacement profiles [mm].
        v_fw, v_bw : np.ndarray, shape (sr,)
            Forward / backward velocity profiles [mm/s].

        Returns
        -------
        displacement : np.ndarray, shape (2·sr,)
        tangential_force : np.ndarray, shape (2·sr,)
        stick_segments : dict
            Keys: disp_fw, Q_fw, disp_bw, Q_bw — raw stick-zone arrays
            used for stiffness estimation via linear regression.  With
            ``contact_stiffness`` set it also carries ``regime``
            ("partial" or "gross") and ``Q_amplitude`` (Q* for partial slip).
        """
        if self.K_c is not None:
            return self._calculate_masing(
                z0, x0, mu_k_fw, mu_k_bw, mu_s_acc_fw, mu_s_acc_bw,
                mu_s_dec_fw, mu_s_dec_bw, x_fw, x_bw, v_fw, v_bw,
            )

        sr = len(x_fw)

        # ── 1. Kinetic gross-sliding curves (scalar μ_k) ─────────────
        kQ_fw = self._q_forward(x_fw, mu_k_fw, z0, x0)
        kQ_bw = self._q_backward(x_bw, mu_k_bw, z0, x0)

        # ── 2. Initial stiffness lines (anchored at kinetic endpoints) ──
        #    Used only to locate the initial stick→slide index for _mu_sliding.
        #    Will be re-anchored after the Oden-Martins curves are available.
        h_fw_0 = kQ_bw[-1] - self.S * x_fw[0]
        line_fw_0 = self.S * x_fw + h_fw_0

        h_bw_0 = kQ_fw[-1] - self.S * x_bw[0]
        line_bw_0 = self.S * x_bw + h_bw_0

        # ── 3. Gross-sliding onset (using static CoF, initial lines) ─
        sQ_static_fw = self._q_forward(x_fw, mu_s_acc_fw, z0, x0)
        sQ_static_bw = self._q_backward(x_bw, mu_s_acc_bw, z0, x0)
        idx_stick_fw_0 = self._find_transition_index(sQ_static_fw, line_fw_0)
        idx_stick_bw_0 = self._find_transition_index(sQ_static_bw, line_bw_0)

        # ── 4. Acc / dec split at velocity peak ──────────────────────
        idx_peak_fw = int(np.argmax(np.abs(v_fw)))
        idx_peak_bw = int(np.argmax(np.abs(v_bw)))

        # ── 5. Oden-Martins μ arrays (D_c + λ, single v_slip) ───────
        mu_oden_fw = self._mu_sliding(
            x_fw, v_fw, idx_stick_fw_0, idx_peak_fw,
            mu_k_fw, mu_s_acc_fw, mu_s_dec_fw,
            self.d_c, self.lambda_acc, self.lambda_dec,
        )
        mu_oden_bw = self._mu_sliding(
            x_bw, v_bw, idx_stick_bw_0, idx_peak_bw,
            mu_k_bw, mu_s_acc_bw, mu_s_dec_bw,
            self.d_c, self.lambda_acc, self.lambda_dec,
        )

        # ── 6. Q curves with varying μ ───────────────────────────────
        sQ_fw = self._q_forward(x_fw, mu_oden_fw, z0, x0)
        sQ_bw = self._q_backward(x_bw, mu_oden_bw, z0, x0)

        # ── 7. Re-anchor stiffness lines to actual Oden-Martins endpoints
        #    When μ_s_dec > μ_k, the Q at reversal is elevated above the
        #    kinetic value (|v|→0 ⟹ v_slip→0 ⟹ μ→μ_s_dec).  Using the
        #    kinetic endpoint as the stick-line anchor causes a visible gap
        #    between the end of the gross-sliding ramp and the stick segment
        #    of the next stroke.  Anchoring on sQ[-1] fixes this.
        h_fw = sQ_bw[-1] - self.S * x_fw[0]
        line_fw = self.S * x_fw + h_fw

        h_bw = sQ_fw[-1] - self.S * x_bw[0]
        line_bw = self.S * x_bw + h_bw

        # ── 8. Stick-segment end indices (corrected lines) ───────────
        idx_stick_fw = self._find_transition_index(sQ_fw, line_fw)
        idx_stick_bw = self._find_transition_index(sQ_bw, line_bw)

        # ── 9. Assemble four segments per stroke ─────────────────────
        disp_stick_fw = x_fw[:idx_stick_fw]
        Q_stick_fw    = line_fw[:idx_stick_fw]

        disp_stick_bw = x_bw[:idx_stick_bw]
        Q_stick_bw    = line_bw[:idx_stick_bw]

        displacement = np.concatenate([
            disp_stick_fw, x_fw[idx_stick_fw:],
            disp_stick_bw, x_bw[idx_stick_bw:],
        ])
        tangential_force = np.concatenate([
            Q_stick_fw, sQ_fw[idx_stick_fw:],
            Q_stick_bw, sQ_bw[idx_stick_bw:],
        ])

        # ── 10. Ensure exactly 2·sr points via interpolation ────────
        target_len = 2 * sr
        if len(displacement) != target_len:
            old_idx = np.linspace(0, 1, len(displacement))
            new_idx = np.linspace(0, 1, target_len)
            displacement     = np.interp(new_idx, old_idx, displacement)
            tangential_force = np.interp(new_idx, old_idx, tangential_force)

        stick_segments = dict(
            disp_fw=disp_stick_fw,
            Q_fw=Q_stick_fw,
            disp_bw=disp_stick_bw,
            Q_bw=Q_stick_bw,
        )
        return displacement, tangential_force, stick_segments

    # ------------------------------------------------------------------
    # Mindlin-Masing stick branch (only with contact_stiffness)
    # ------------------------------------------------------------------
    #
    # After a reversal the force changes by dq over a displacement run
    #
    #     run(dq) = dq/S + (3·dq_n / (2·K_c)) · [1 − (1 − dq/dq_n)^(2/3)]
    #
    # i.e. the linear rig in series with a Mindlin contact following the
    # Masing rule.  dq_n is the force range at which the whole contact
    # slides: 2·μ_s·P for a partial slip loop, |Q_sliding − Q_reversal| after
    # gross sliding.  The initial slope is 1/S + 1/K_c and the branch meets
    # the sliding force with zero slope.  For a rigid rig and a symmetric
    # loop the energy ratio at the transition is A = 0.2 (Fouvry et al. 1995).

    def _branch_dx(self, dq: np.ndarray | float, dq_n: float) -> np.ndarray | float:
        """Displacement run needed to change the force by dq along a branch."""
        r = np.clip(np.asarray(dq) / dq_n, 0.0, 1.0)
        return dq / self.S + 1.5 * dq_n / self.K_c * (1.0 - (1.0 - r) ** (2.0 / 3.0))

    def _branch_q(self, run: np.ndarray, dq_n: float, n_tab: int = 4001) -> np.ndarray:
        """Force change reached after a displacement run (inverse of
        _branch_dx), saturating at dq_n once the whole contact slides."""
        s = np.linspace(0.0, 1.0, n_tab)
        dq = dq_n * (1.0 - (1.0 - s) ** 3)        # dense where the branch flattens
        return np.interp(run, self._branch_dx(dq, dq_n), dq)

    def _partial_slip_amplitude(self, x0: float, dq_n: float) -> float:
        """Force amplitude Q* of the steady partial slip loop (bisection)."""
        lo, hi = 0.0, 0.5 * dq_n
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self._branch_dx(2.0 * mid, dq_n) < 2.0 * x0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _stick_branch(self, x: np.ndarray, Q_r: float, Q_t: float) -> np.ndarray:
        """Force along a stroke that leaves the reversal (x[0], Q_r) towards
        the sliding force Q_t."""
        sign = 1.0 if Q_t >= Q_r else -1.0
        return Q_r + sign * self._branch_q(np.abs(x - x[0]), abs(Q_t - Q_r))

    def _splice(
        self,
        q_gross: np.ndarray,
        x: np.ndarray,
        Q_r: float,
        sign: float,
        n_iter: int = 4,
    ) -> tuple[int, np.ndarray]:
        """Index where the Masing branch from (x[0], Q_r) meets the sliding
        curve q_gross, and the branch itself.  The sliding force the branch
        aims at is refined a few times, since q_gross varies along the
        stroke when the scar is worn."""
        tol = 1e-9 * self.P
        Q_t = float(q_gross[0])
        for it in range(n_iter):
            branch = self._stick_branch(x, Q_r, Q_t)
            reached = (sign * (branch - q_gross) >= -tol) | (np.abs(branch - Q_t) <= tol)
            hit = np.nonzero(reached[1:])[0] + 1
            idx = int(hit[0]) if hit.size else int(np.argmin(np.abs(branch - q_gross)))
            if it < n_iter - 1:
                Q_t = float(q_gross[idx])
        return idx, branch

    def _calculate_masing(
        self,
        z0: float,
        x0: float,
        mu_k_fw: float,
        mu_k_bw: float,
        mu_s_acc_fw: float,
        mu_s_acc_bw: float,
        mu_s_dec_fw: float,
        mu_s_dec_bw: float,
        x_fw: np.ndarray,
        x_bw: np.ndarray,
        v_fw: np.ndarray,
        v_bw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """:meth:`calculate` with a Mindlin-Masing contact in series with the
        rig.  Same pipeline as the Jenkins model, the stick line replaced by
        the Masing branch.  When the stroke cannot bring the contact to
        sliding the loop is a closed partial slip (fretting) loop, where the
        sliding laws and the scar geometry play no part."""
        # ── Regime ────────────────────────────────────────────────────
        mu_s = 0.5 * (mu_s_acc_fw + mu_s_acc_bw)
        dq_n = 2.0 * mu_s * self.P
        if 2.0 * x0 < self._branch_dx(dq_n, dq_n):
            q_amp = self._partial_slip_amplitude(x0, dq_n)
            Q_fw = -q_amp + self._branch_q(np.abs(x_fw - x_fw[0]), dq_n)
            Q_bw = q_amp - self._branch_q(np.abs(x_bw - x_bw[0]), dq_n)
            stick_segments = dict(disp_fw=x_fw, Q_fw=Q_fw, disp_bw=x_bw, Q_bw=Q_bw,
                                  regime="partial", Q_amplitude=q_amp)
            return (np.concatenate([x_fw, x_bw]), np.concatenate([Q_fw, Q_bw]),
                    stick_segments)

        # ── Gross slip ────────────────────────────────────────────────
        kQ_fw = self._q_forward(x_fw, mu_k_fw, z0, x0)
        kQ_bw = self._q_backward(x_bw, mu_k_bw, z0, x0)

        # The branch climbs to the static threshold (the overshoot peak) and
        # Oden-Martins takes over from that point, so the branch always aims
        # at the static curve and the sliding law starts where it lands.
        # First pass anchored at the kinetic ends, then re-anchored at the
        # actual Oden-Martins reversal forces.
        sQ_static_fw = self._q_forward(x_fw, mu_s_acc_fw, z0, x0)
        sQ_static_bw = self._q_backward(x_bw, mu_s_acc_bw, z0, x0)
        idx_peak_fw = int(np.argmax(np.abs(v_fw)))
        idx_peak_bw = int(np.argmax(np.abs(v_bw)))

        Q_r_fw, Q_r_bw = float(kQ_bw[-1]), float(kQ_fw[-1])
        for _ in range(2):
            idx_stick_fw, br_fw = self._splice(sQ_static_fw, x_fw, Q_r_fw, +1.0)
            idx_stick_bw, br_bw = self._splice(sQ_static_bw, x_bw, Q_r_bw, -1.0)
            mu_oden_fw = self._mu_sliding(
                x_fw, v_fw, idx_stick_fw, idx_peak_fw,
                mu_k_fw, mu_s_acc_fw, mu_s_dec_fw,
                self.d_c, self.lambda_acc, self.lambda_dec,
            )
            mu_oden_bw = self._mu_sliding(
                x_bw, v_bw, idx_stick_bw, idx_peak_bw,
                mu_k_bw, mu_s_acc_bw, mu_s_dec_bw,
                self.d_c, self.lambda_acc, self.lambda_dec,
            )
            sQ_fw = self._q_forward(x_fw, mu_oden_fw, z0, x0)
            sQ_bw = self._q_backward(x_bw, mu_oden_bw, z0, x0)
            Q_r_fw, Q_r_bw = float(sQ_bw[-1]), float(sQ_fw[-1])

        # Final branches from the converged reversal forces
        idx_stick_fw, br_fw = self._splice(sQ_static_fw, x_fw, Q_r_fw, +1.0)
        idx_stick_bw, br_bw = self._splice(sQ_static_bw, x_bw, Q_r_bw, -1.0)

        displacement = np.concatenate([x_fw, x_bw])
        tangential_force = np.concatenate([
            br_fw[:idx_stick_fw], sQ_fw[idx_stick_fw:],
            br_bw[:idx_stick_bw], sQ_bw[idx_stick_bw:],
        ])
        stick_segments = dict(
            disp_fw=x_fw[:idx_stick_fw], Q_fw=br_fw[:idx_stick_fw],
            disp_bw=x_bw[:idx_stick_bw], Q_bw=br_bw[:idx_stick_bw],
            regime="gross", Q_amplitude=None,
        )
        return displacement, tangential_force, stick_segments

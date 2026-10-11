# triboSim user guide

`tribosim` generates virtual reciprocating sliding tribological tests (RSTTs):
the coefficient of friction (CoF) over the cycles, the friction loop of every
cycle, the wear scar, and the raw force and displacement signals a tribometer
would record. Every quantity is produced from parameters you choose, so the
ground truth behind each signal is known.

This guide explains how to use each module. The physics and the comparison
with measurements are described in the article (see [Citing](#10-citing)),
whose section numbers are quoted here.

**Contents**

1. [Installation](#1-installation)
2. [How a virtual test is built](#2-how-a-virtual-test-is-built)
3. [Units and conventions](#3-units-and-conventions)
4. [Quick start](#4-quick-start)
5. [Modules](#5-modules)
   [5.1 Genesis](#51-cof-genesis-frictiongenesis) ·
   [5.2 Kinematics](#52-kinematics-sinusoidal-crankslider-trapezoidal) ·
   [5.3 Friction loop](#53-friction-loop-frictionloop) ·
   [5.4 Wear](#54-wear-wearmodel) ·
   [5.5 Noise](#55-acquisition-noise-add_noise) ·
   [5.6 Export](#56-export-exporter)
6. [A complete virtual test](#6-a-complete-virtual-test)
7. [Reproducing the article](#7-reproducing-the-article)
8. [Scope and limitations](#8-scope-and-limitations)
9. [Tests](#9-tests)
10. [Citing](#10-citing)

---

## 1. Installation

`tribosim` needs Python 3.9 or later and NumPy. SciPy is optional and only
makes the noise generation faster. Matplotlib is only needed for the examples.

```bash
git clone https://github.com/illavori/triboSim.git
cd triboSim
pip install -e .              # the package
pip install -e ".[fast,plot]" # optional: SciPy and Matplotlib
```

You can also run the examples straight from the cloned folder without
installing anything, since they add the repository to the Python path.

---

## 2. How a virtual test is built

A test is a chain of five modules, each one feeding the next:

```
 FrictionGenesis      CoF of every cycle, forward and backward    mu_FW(n), mu_BW(n)
       |
 WearModel            scar depth and width from the dissipated    z0(n), z0/x0(n)
       |              energy
       v
 Kinematics           displacement and velocity of one cycle      x(t), v(t)
 + FrictionLoop       tangential force of cycle n, shaped by      Q(x) for cycle n
       |              mu(n) and by the current scar
       v
 add_noise            instrumental and tribological noise         Q_noisy, x_noisy
       |
 Exporter             CSV with metadata, or a saved session
```

You can use any module on its own. A single friction loop needs only the
kinematics and `FrictionLoop`, and a CoF trajectory only `FrictionGenesis`.

---

## 3. Units and conventions

| Quantity | Unit |
|---|---|
| length, displacement, stroke | mm |
| force | N |
| stiffness | N/mm |
| Young's modulus | MPa |
| time | s, frequency Hz |
| velocity | mm/s (when `f_test` is given, see below) |
| Oden-Martins rate constants λ | s/mm |
| energy wear coefficient α | mm³ per N·mm of dissipated energy |

* **Stroke.** `x0` is the half-amplitude: the counterbody travels from
  `-x0` to `+x0`, so the stroke length is `2*x0`.
* **Forward and backward (FW/BW).** The forward stroke goes from `-x0` to
  `+x0` with positive force, the backward one returns with negative force.
  Every friction parameter can be set separately for each stroke.
* **Acceleration and deceleration (acc/dec).** Each stroke is split at its
  velocity peak. The first half, just after the reversal, uses the `acc`
  parameters and the second half the `dec` ones.
* **Velocity units.** Call `get_stroke(..., f_test=...)` with the test
  frequency so that velocities are physical (mm/s) and the λ values are in
  s/mm, as in the article. Without `f_test` the velocities are per radian of
  crank angle (mm/rad), a legacy unit kept for backward compatibility.
  Switching between the two does not change the loop as long as λ is scaled
  accordingly, λ[s/mm] = λ[rad/mm] / (2π f_test).
* **CoF.** Plotting `Q / P` gives the CoF of the loop.

---

## 4. Quick start

```python
from tribosim import FrictionLoop, Sinusoidal

P, x0, f_test = 50.0, 0.5, 5.0                      # N, mm, Hz
x_fw, x_bw, v_fw, v_bw = Sinusoidal().get_stroke(x0, 500, f_test=f_test)

loop = FrictionLoop(contact_force=P, stiffness=2000.0)   # rig stiffness S [N/mm]
x, Q, _ = loop.calculate(
    z0=0.0, x0=x0,                                       # no wear scar: Coulomb loop
    mu_k_fw=0.4, mu_k_bw=0.4,
    mu_s_acc_fw=0.4, mu_s_acc_bw=0.4,
    mu_s_dec_fw=0.4, mu_s_dec_bw=0.4,
    x_fw=x_fw, x_bw=x_bw, v_fw=v_fw, v_bw=v_bw)
# x [mm] and Q [N] hold one closed loop; Q / P is the CoF
```

The four scripts in [`examples/`](../examples) build up from here to a complete
virtual test.

![Three friction loops](images/friction_loops.png)

For figures in the same style, call `tribosim.plotting.apply_style()` before
plotting. It sets a serif font and a teal colour cycle, and the colours are
available as `LIGHT`, `MID`, `D1` and `DARK` (light to dark) and `BLACK`.

---

## 5. Modules

### 5.1 CoF genesis: `FrictionGenesis`

Generates the CoF of every cycle as the sum of four terms (Section 2.1):

```
mu(n) = mu_i + (mu_inf - mu_i)(1 - exp(-n/tau))    run-in transient
        + d(n)                                     piecewise-linear drift
        + sum_j delta_j(n)                         events (drops and peaks)
        + eps(n)                                   correlated fluctuation (AR1)
```

```python
from tribosim import FrictionGenesis, GenesisEvent

gen = FrictionGenesis(
    n_cycles=20_000,
    mu_i=0.15, mu_inf=0.55, tau=800.0,          # run-in
    betas=[1e-6, -2e-6], breakpoints=[10_000],  # two drift segments
    events=[GenesisEvent(6000, -0.08, onset=150, recovery=1200)],
    sigma=0.008, phi=0.95,                      # fluctuation
    random_seed=1)
mu = gen.generate()          # array of shape (n_cycles,)
```

| Argument | Paper | Meaning |
|---|---|---|
| `n_cycles` | N | number of cycles |
| `mu_i` | μ_i | CoF at the start of sliding |
| `mu_inf` | μ_∞ | CoF at the end of the run-in |
| `tau` | τ | run-in time constant [cycles]. `None` means no run-in, the backbone starts at `mu_i` |
| `betas` | β_k | drift rate of each segment [1/cycle]. Empty means no drift, one value a linear drift, `0` a plateau |
| `breakpoints` | n_k | cycles where one segment ends and the next starts, `len(betas) - 1` values. The drift is continuous at them |
| `events` | δ_j | list of `GenesisEvent` |
| `sigma` | σ_μ | standard deviation of the fluctuation (CoF units) |
| `phi` | φ_μ | lag-one autocorrelation of the fluctuation from cycle to cycle, in [0, 1) |
| `random_seed` | | seed for a reproducible fluctuation |

**Events.** `GenesisEvent(cycle, amplitude, onset, recovery, shape)` describes
an abrupt excursion that peaks at `cycle`. A negative `amplitude` is a drop
and a positive one a peak. The CoF ramps to it linearly over `onset` cycles
and then returns to the backbone over `recovery` cycles, with one of four
shapes: `"linear"`, `"exponential"` (fast then slow, the default),
`"logarithmic"` or `"sigmoid"` (slow, fast, slow).

For synthetic tests you can draw events at random:

```python
events = FrictionGenesis.random_events(20_000, n_events=3, amplitude=(0.02, 0.1),
                                       sign="both", random_seed=7)
```

**Terms separately.** `run_in()`, `drift()`, `backbone()` (run-in plus
drift), `event_term()` and `fluctuation()` return each term, which is useful
for labelling synthetic data. `generate(add_events=True, add_fluctuation=True)`
returns their sum.

**Forward and backward.** Use two instances, for example with different
`mu_inf` for an asymmetric contact or opposite `betas` for diverging strokes.

### 5.2 Kinematics: `Sinusoidal`, `CrankSlider`, `Trapezoidal`

All three return the displacement and velocity of one cycle, split into the
forward and backward strokes:

```python
x_fw, x_bw, v_fw, v_bw = Sinusoidal().get_stroke(x0, sr, f_test=f_test)
```

* `x0` is the half-amplitude [mm] and `sr` the number of samples per stroke.
  Samples are equally spaced in time, as in a tribometer, so they cluster near
  the reversals where the velocity is low. For a sampling frequency `f_s`,
  use `sr = f_s / (2 * f_test)`.
* `Sinusoidal()` is a harmonic profile, x = x0 sin(ωt).
* `CrankSlider(crank_radius=75, rod_length=150, n_points=2000)` is the profile
  of a crank-slider drive. The ratio r/l sets the asymmetry between
  the strokes, and the profile tends to the sinusoidal one as r/l goes to zero.
  The crank sizes only set the shape, the amplitude is always `x0`.
* `Trapezoidal(ramp_fraction=0.1)` is a constant-velocity profile with linear
  ramps, typical of ball-screw and linear-motor drives.
  `ramp_fraction` = t_ramp / (T/2) lies in (0, 0.5).

### 5.3 Friction loop: `FrictionLoop`

Computes the tangential force Q against the displacement x for one cycle
(Section 2.3 of the article).

```python
loop = FrictionLoop(contact_force=P, stiffness=S, d_c=0.05,
                    lambda_acc=0.5, lambda_dec=0.5, contact_stiffness=None)
x, Q, info = loop.calculate(z0, x0,
                            mu_k_fw, mu_k_bw,
                            mu_s_acc_fw, mu_s_acc_bw,
                            mu_s_dec_fw, mu_s_dec_bw,
                            x_fw, x_bw, v_fw, v_bw)
```

Constructor:

| Argument | Paper | Meaning |
|---|---|---|
| `contact_force` | P | normal load [N] |
| `stiffness` | S | rig stiffness [N/mm], the slope of the loop just after each reversal |
| `d_c` | D_c | slip distance over which the slip velocity recovers after sliding starts [mm]. `0` means instantaneous recovery |
| `lambda_acc`, `lambda_dec` | λ^acc, λ^dec | Oden-Martins rate constants [s/mm]. Larger values make the CoF reach μ_k sooner |
| `contact_stiffness` | K_c | optional initial tangential stiffness of the contact [N/mm]. Enables partial slip, see below |

`calculate()` arguments:

| Argument | Paper | Meaning |
|---|---|---|
| `z0` | z_0 | depth of the idealised parabolic groove [mm]. Its ratio to `x0` controls the non-Coulomb curvature of the loop. With the wear module, pass `wear["z0_over_x0"][n] * x0` |
| `x0` | x_0 | half-amplitude [mm] |
| `mu_k_fw`, `mu_k_bw` | μ_FW, μ_BW | kinetic CoF of each stroke, usually from the genesis module |
| `mu_s_acc_*` | μ_s^acc | static CoF just after the reversal |
| `mu_s_dec_*` | μ_s^dec | CoF reached just before the next reversal |
| `x_fw`, ..., `v_bw` | | output of `get_stroke()` |

Choosing the static CoF, with κ = μ_s/μ_k:

* **κ = 1:** no static-to-kinetic transition (Coulomb corners).
* **κ > 1:** a force overshoot after each reversal that decays to μ_k. Its
  width grows with `d_c` and with smaller λ.
* **κ < 1:** the loop enters the plateau through a rounded run-up instead of
  a sharp corner, close to the microslip rounding of a real contact.

`calculate()` returns the displacement and force of the full cycle (forward
then backward, `2*sr` samples each) and a dictionary `info` with the stick
segments. With `contact_stiffness` set it also holds `info["regime"]`
(`"partial"` or `"gross"`) and, for partial slip, `info["Q_amplitude"]`.

**Partial slip.** With `contact_stiffness=K_c` the stick line is replaced by
a Cattaneo-Mindlin (Masing) branch in series with the rig compliance 1/S
(Section 2.3.3). The contact never slides as a whole when the stroke is shorter
than

```
x0_t = mu_s P / S + 3 mu_s P / (2 K_c)
```

and the loop is then a closed partial slip loop. For a sphere on a plane
K_c = 8 a G*, with `a` the contact radius and G* the combined shear modulus.
For a cylinder K_c is best treated as an effective parameter. A very large
`stiffness` (for example `1e9`) leaves only the contact compliance. See
`examples/03_partial_slip.py`.

### 5.4 Wear: `WearModel`

Accumulates wear in proportion to the frictional energy dissipated in each
cycle (Section 2.4), for the flat and for the counterbody, and returns the
evolving scar.

```python
from tribosim import WearModel

wear = WearModel(x0=0.25, contact_force=20.0,
                 alpha_flat=2e-7, alpha_ball=5e-8,      # energy wear coefficients
                 R_ball=3.0,                            # counterbody radius [mm]
                 E1=210e3, nu1=0.3, E2=210e3, nu2=0.3,  # flat, counterbody
                 scar_shape=None,                       # profile exponent p (auto)
                 geometry="ball")                       # or "cylinder"
out = wear.compute(mu_fw, mu_bw)                        # arrays of shape (n_cycles,)
```

* **Ball on flat** (`geometry="ball"`). The ball wears as a spherical cap,
  which widens the contact from its Hertzian radius. Main outputs: `z0` (flat
  scar depth), `a_contact` (contact half-width), `h_ball`, `half_length`
  (x0 + a_eff), `z0_over_x0` (depth over half-length), and `z_total` and `x`
  (the scar profile of every cycle).
* **Cylinder on flat** (`geometry="cylinder"`, with `L_cylinder` the axial
  length; `R_ball` and `alpha_ball` then refer to the cylinder). Main outputs:
  `z0` (scar depth D), `W_scar` (half-width W = x0 + b_eff), `z0_over_x0`
  (D/W), `b_contact`, `h_cyl` and `pressure` (mean contact pressure).
* `scar_shape` is the even exponent p of the profile z = z0 (1 - (x/L)^p).
  By default it is 2 for x0 ≤ 1 mm, 4 up to 5 mm and 6 above.

**Coupling with the loop.** `out["z0_over_x0"]` is the aspect ratio that
shapes the friction loop. Pass `z0 = out["z0_over_x0"][n] * x0` to
`FrictionLoop.calculate()` for cycle `n`, and the loop then develops from
Coulomb to non-Coulomb as the scar grows. The effect is strong in
small-amplitude fretting and weak for long strokes.

### 5.5 Acquisition noise: `add_noise`

Adds the noise of the measuring chain to the clean signals (Section 2.5).

```python
from tribosim import add_noise

Q_noisy, x_noisy = add_noise(Q, x, sampling_rate=5000.0, normal_load=20.0,
                             force_sensor_range=50.0, noise_model="colored",
                             sigma_rel=0.005, sigma_x_abs=0.001,
                             sigma_ct=0.002, f_c=100.0, rng=None)
```

| Argument | Paper | Meaning |
|---|---|---|
| `sampling_rate` | f_s | sampling frequency [Hz] |
| `force_sensor_range` | F_fs | full-scale range of the force sensor [N] |
| `sigma_rel` | σ_Q,rel | instrumental force noise as a fraction of `force_sensor_range` |
| `sigma_x_abs` | σ_x,abs | displacement noise [mm] |
| `noise_model`, `f_c` | f_c | `"colored"`: correlated AR1 noise with correlation frequency f_c. `"white"`: uncorrelated |
| `normal_load`, `sigma_ct` | P, σ_ct | white tribological noise on the force, of standard deviation σ_ct P |
| `rng` | | `numpy.random.Generator` for reproducible noise |

The instrumental noise is set by the sensor range rather than by the load,
so it weighs more on low-friction contacts. The tribological term is a
simple white term for the scatter of a steadily sliding contact.

### 5.6 Export: `Exporter`

```python
from tribosim import Exporter, load_session

ex = Exporter("output", "test01")
ex.export_csv(contact_force=P, stroke=2 * x0, frequency=f_test, sampling_rate=f_s,
              time=t, displacement=x_noisy, tangential_force=Q_noisy,
              final_wear_depth=z0_end, sensor_info={"Qfs_N": 50.0, "sigma_rel": 0.005})
path = ex.save_session({"mu_fw": mu_fw, "z0": out["z0"]})   # any picklable data
data = load_session(path)
```

The CSV is semicolon-delimited. Its header holds the test conditions (load,
stroke, frequencies, final wear depth and, if given, the sensor) and is
followed by three columns: time [ms], position [mm] and horizontal force [N].

---

## 6. A complete virtual test

[`examples/04_virtual_test.py`](../examples/04_virtual_test.py) runs the whole
chain for a ball-on-flat test of 5000 cycles in a few seconds:

1. two `FrictionGenesis` instances give `mu_fw` and `mu_bw`;
2. `WearModel.compute(mu_fw, mu_bw)` gives the scar of every cycle;
3. one `FrictionLoop.calculate()` per cycle, with that cycle's CoF and
   `z0 = z0_over_x0[n] * x0`, gives the force signal;
4. the loops are concatenated into one signal, and `add_noise` corrupts it;
5. `Exporter` writes the last cycles to a CSV.

All the parameters used in the five steps are the ground truth of the
resulting signal.

Some practical points:

* A loop takes about half a millisecond with 100 to 200 samples per stroke,
  so 10⁵ cycles take about a minute. You can also compute a loop only every
  few cycles when the CoF and the scar change slowly.
* The loop is quasi-static, so the result does not depend on `f_test` except
  through the velocity used by the Oden-Martins law.
* Keep `sr` consistent with `f_s` and `f_test` when the noise and the
  sampling density matter.

---

## 7. Reproducing the article

[`paper/`](../paper) holds the fitted parameters of every case in the article
(`parameters.py`) and a script that regenerates the simulated curves:

```bash
python paper/reproduce_figures.py          # Figs. 6, 9, 10 and 11
python paper/reproduce_figures.py fig11    # only the virtual test (a few minutes)
```

The measured curves the parameters were fitted to are not distributed. They
belong to their authors, and [`paper/README.md`](../paper/README.md) lists the
source of each one.

---

## 8. Scope and limitations

The simulator is a reduced-order model and is open to improvement.

* **Quasi-static.** The inertia, stiffness and damping of the
  force-measuring system are not represented, so it suits test frequencies
  well below the natural frequencies of the rig.
* **Contact mechanics not resolved.** The loop follows from an idealised
  scar geometry, not from the pressure and shear fields at the interface.
* **Wear.** The energy wear coefficient is constant, and wear in partial
  slip is not modelled.
* **Noise.** The tribological noise is a simple white term. Structured
  signals such as stick-slip, or fluctuations induced by the third body, are
  not modelled.

Contributions that extend any of these points are welcome.

---

## 9. Tests

```bash
python -m pytest                 # or
python tests/test_tribosim.py
```

The tests check physical and numerical invariants rather than frozen numbers,
for example:

* the analytical area of the Coulomb loop;
* that every loop is closed and runs clockwise;
* the energy ratio of 0.2 at the partial to gross slip transition;
* the conservation of the worn volume;
* the statistics of the AR1 noise and of the CoF fluctuation;
* the equivalence of the velocity units.

---

## 10. Citing

If you use `tribosim`, please cite:

> I. Llavori, X. Borras, A. Zabala, An analytical model for virtual reciprocating sliding
> tribological tests (2026), submitted.

The full reference will be updated on publication. See also `CITATION.cff`.

---

Iñigo Llavori (Mondragon Unibertsitatea) and Xavier Borras (iTribomat - European Tribology Center).

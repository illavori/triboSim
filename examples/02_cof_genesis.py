"""
Example 2 - CoF evolution over the test (friction genesis).

Forward and backward strokes with a run-in, a two-segment drift, a drop and a
peak event, and the correlated cycle-to-cycle fluctuation.

    python examples/02_cof_genesis.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tribosim import FrictionGenesis, GenesisEvent

N = 20_000                                     # cycles

events = [
    GenesisEvent(cycle=6000, amplitude=-0.08, onset=150, recovery=1200, shape="exponential"),
    GenesisEvent(cycle=14000, amplitude=+0.05, onset=80, recovery=800, shape="sigmoid"),
]
common = dict(n_cycles=N, tau=800.0,                     # run-in time constant [cycles]
              betas=[1e-6, -2e-6], breakpoints=[10_000],  # drift rates [1/cycle], breakpoint
              events=events, sigma=0.008, phi=0.95)        # fluctuation std and lag-one corr.

mu_fw = FrictionGenesis(mu_i=0.15, mu_inf=0.55, random_seed=1, **common).generate()
mu_bw = FrictionGenesis(mu_i=0.15, mu_inf=0.50, random_seed=2, **common).generate()
print(f"mean CoF over the last 1000 cycles: FW {mu_fw[-1000:].mean():.3f}, "
      f"BW {mu_bw[-1000:].mean():.3f}")

# Synthetic tests with random events instead of chosen ones:
rnd = FrictionGenesis.random_events(N, n_events=3, sign="both", random_seed=7)
print("random events:", [(int(e.cycle), round(e.amplitude, 3)) for e in rnd])

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(mu_fw, lw=0.6, label="forward")
ax.plot(mu_bw, lw=0.6, label="backward")
ax.set_xlabel("cycle n"); ax.set_ylabel("CoF [-]"); ax.legend()
out = Path(__file__).parent / "output"; out.mkdir(exist_ok=True)
fig.tight_layout(); fig.savefig(out / "02_cof_genesis.png", dpi=150)
print("saved", out / "02_cof_genesis.png")

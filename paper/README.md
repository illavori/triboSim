# Reproducing the article

This folder regenerates the simulated curves of

> I. Llavori, *An analytical model for virtual reciprocating sliding
> tribological tests* (2026), submitted.

* `parameters.py` holds the fitted parameters of every case: Table 2 (CoF
  genesis), Table 3 (friction loops) and the virtual test of Section 3.3.
* `reproduce_figures.py` runs the simulator with them and writes the figures
  to `figures/`.

```bash
python paper/reproduce_figures.py          # Figs. 6, 9, 10 and 11
python paper/reproduce_figures.py fig10    # one figure
```

The virtual test (Fig. 11) runs 500 000 cycles and takes a few minutes. It
prints the final aspect ratio z0/x0 = 0.214, the scar half-width W = 333 um
and the depth D = 71.2 um reported in the article.

## Measured data

Only the model curves are produced. The measured curves the parameters were
fitted to are not distributed, since they belong to their authors. They can
be found in the following sources:

| Case | Source |
|---|---|
| CoF trace and loops, Examples 1 and 2, wear scar of Example 2 | X. Jin, W. Sun, P.H. Shipway, Derivation of a wear scar geometry-independent coefficient of friction from fretting loops exhibiting non-Coulomb frictional behaviour, *Tribology International* 102 (2016) 561-568. [doi:10.1016/j.triboint.2016.06.012](https://doi.org/10.1016/j.triboint.2016.06.012) |
| CoF trace (elastomer) | B.-L. Choi et al., In-situ wear test of ethylene-propylene-diene-monomer composites, *Tribology International* 222 (2026) 112087. [doi:10.1016/j.triboint.2026.112087](https://doi.org/10.1016/j.triboint.2026.112087) |
| CoF trace and asymmetric loop | A. Kumar, R. Mukherjee, M.K. Singh, Synergistic effect of cold rolling and heat treatment on tribological performance of Inconel-718 at room and high temperature, *Wear* 584-585 (2026) 206384. [doi:10.1016/j.wear.2025.206384](https://doi.org/10.1016/j.wear.2025.206384) |
| CoF trace | C. Li, X. Deng, Z. Wang, Friction behaviour and self-lubricating mechanism of low alloy martensitic steel during reciprocating sliding, *Wear* 482-483 (2021) 203972. [doi:10.1016/j.wear.2021.203972](https://doi.org/10.1016/j.wear.2021.203972) |
| Noisy loop | F. Kafexhiu, I. Velkavrh, T. Wright et al., Adaptive algorithm for automated CoF evaluation, *Tribology Letters* 73 (2025) 81. [doi:10.1007/s11249-025-02015-x](https://doi.org/10.1007/s11249-025-02015-x) |
| Static-kinetic loop (NBR) | Present study, not distributed |

## Notes

* The rate constants λ are given in s/mm, for the velocity in mm/s. The test
  frequency of the NBR case is still to be confirmed, and its λ depends on it.
  The loop itself does not.
* The genesis events of Table 2 were detected in the fits. Their onset and
  recovery lengths are not part of the fit, so the values used here are
  illustrative.
* The noisy loop is normalised by the load (P = 1), so its stiffness is given
  per unit load.

---

Iñigo Llavori, Mondragon Unibertsitatea.

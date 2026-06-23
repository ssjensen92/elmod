# Bundled LOC runtime

This directory contains the LOC 1D runtime used by elmod. It was derived from
[LOC](https://github.com/mjuvela/LOC) by Mika Juvela and includes the elmod
integration and performance changes preserved in the `elmod-private` history
through commit `c92e0d1`.

The local changes include persistent-worker execution, OpenCL context/program/
kernel reuse, quieter fitting output, improved diagnostics, and convergence
stopping based on maximum relative level-population change.

LOC is licensed under the GNU General Public License v3.0. The repository-level
[`LICENSE`](../LICENSE) applies, and scientific use should cite Juvela (2020),
*Astronomy & Astrophysics*, 644, A151.

# Figures for "Stationary bistable fronts on a periodic necklace graph"

This repository contains the code that computes Figures 2–5 of

> Y. Mori, T. Ogawa and C. Sintavanuruk,
> *Stationary bistable fronts on a periodic necklace graph: existence, stability and multiplicity*.

Figure 1, the schematic of the necklace graph, is drawn in TikZ in the paper and is not computed here.

The McKean reaction is affine on each side of the threshold, so the stationary equation is solved without discretization.
Between threshold crossings and coefficient interfaces, `necklace/flow.cpp` advances states by the explicit hyperbolic-function solution
and finds the next threshold crossing in closed form.
Fronts with prescribed crossings are evaluated from the closed-form one-crossing profiles in `necklace/green.py`.

## Reproducing the figures

Each route below writes the four PDFs to `fig/` in about a minute.
With the pinned versions, both reproduce the figures of the paper pixel for pixel at 300 dpi;
we checked this on x86_64 Linux.
Other versions of the libraries or of TeX may change the rendering slightly.

### With Nix

With [Nix](https://nixos.org) and flakes enabled, run

```bash
nix develop -c ./make_figures.sh
```

`flake.lock` fixes the nixpkgs revision, and with it the exact versions of Python, the Python packages, the compiler and TeX Live.

### Without Nix

You need:

- Python 3, recent enough for the pinned packages below; the figures in the paper used Python 3.14.6;
- a C++17 compiler, called `g++` unless the environment variable `CXX` names another one (for example `clang++` on macOS);
- a TeX installation with `pdflatex` and the LaTeX packages `amsmath`, `amsfonts`, `pgf`, `type1cm`, `cm-super` and `underscore`.
  TeX Live with the medium or full scheme, and MacTeX, include all of them.

Then run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
./make_figures.sh
```

`requirements.txt` pins NumPy, SciPy, Matplotlib and Shapely to the versions used for the paper.
On first use, `necklace/switching.py` compiles `necklace/flow.cpp` into a shared library next to it.

## Contents

| File | Contents |
| --- | --- |
| `figure2.py` | Figure 2: threshold fluxes, and the pinning interval as a function of the edge lengths |
| `figure3.py` | Figure 3: the eight fronts with crossings in three prescribed period slots, and the collision of a pair |
| `figures4_5.py` | Figures 4 and 5: period-map images of the invariant manifolds, the crossing words of the fronts they carry, and the endpoints of the pinning interval |
| `necklace/flow.cpp` | Exact flow of the switching system between threshold crossings |
| `necklace/switching.py` | The period map, built on `flow.cpp` |
| `necklace/green.py` | Closed-form fronts with prescribed crossings, and checks of their sign on every affine interval |
| `necklace/style.py` | Common typography: LaTeX text in Computer Modern at the printed size |

Each script also checks its numerical results with assertions, for example that every computed front takes the value `a` at its crossings and lies on the correct side of the threshold between them.

## Licence

MIT; see `LICENSE`.

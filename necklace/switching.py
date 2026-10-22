"""Period map of the stationary McKean equation (p u_x)_x + f(u, a) = 0.

States are (u, v) with the flux v = p u_x. The spatial origin of this module is
the midpoint of an interval where p = 1, so one period consists of length l1/2
with p = 1, length l2 with p = 1/alpha, and length l1/2 with p = 1. The paper's
origin is at the left end of an interval with p = 1, that is, x_paper = x + l1/2.

Between threshold crossings the equation is affine with constant coefficients,
and flow.cpp advances states by the explicit hyperbolic-function solution,
computing each threshold crossing in closed form.
"""
from pathlib import Path
import ctypes as ct
import os
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
if (not (HERE/'flow.so').exists()
        or (HERE/'flow.cpp').stat().st_mtime > (HERE/'flow.so').stat().st_mtime):
    subprocess.run([os.environ.get('CXX', 'g++'), '-O3', '-std=c++17', '-shared', '-fPIC',
                    str(HERE/'flow.cpp'), '-o', str(HERE/'flow.so')], check=True)
lib = ct.CDLL(str(HERE/'flow.so'))
_array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
_int_array = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
lib.flow.argtypes = [_array, ct.c_double, ct.c_double, ct.c_double]
lib.period.argtypes = [_array, *([ct.c_double]*4), ct.c_int]
lib.iterate_batch.argtypes = [_array, ct.c_int, ct.c_int, *([ct.c_double]*4), _int_array]


def matrix(t, kappa):
    """Flow over length t of the linear system u' = kappa v, v' = u, where kappa = 1/p."""
    r = np.sqrt(kappa)
    c, s = np.cosh(r*t), np.sinh(r*t)
    return np.array([[c, r*s], [s/r, c]])


class Map:
    """Switching flow and period map for the coefficient (l1, l2, alpha) and threshold a."""

    def __init__(self, l1=.8, l2=1.6, alpha=.25, a=.5):
        self.l1, self.l2, self.alpha, self.a = l1, l2, alpha, a
        self.L = l1+l2
        # Monodromy matrix of the linear system, based at the midpoint phase.
        self.M = matrix(l1/2, 1) @ matrix(l2, alpha) @ matrix(l1/2, 1)
        self.lam = float(np.max(np.linalg.eigvals(self.M).real))
        # Slope of the unstable eigendirection at the midpoint phase.
        self.m = float((self.lam-self.M[0, 0])/self.M[0, 1])

    def flow(self, y, t, kappa):
        """Advance a state over length t in a region where p = 1/kappa."""
        y = np.array(y, dtype=float)
        n = lib.flow(y, t, kappa, self.a)
        if n < 0:
            raise ValueError('degenerate threshold crossing or event cap')
        return y, n

    def step(self, y, direction=1):
        """Apply the period map once (direction=1) or its inverse (direction=-1)."""
        y = np.array(y, dtype=float)
        n = lib.period(y, self.l1, self.l2, self.alpha, self.a, direction)
        if n < 0:
            raise ValueError('degenerate threshold crossing or event cap')
        return y, n

    def seeds(self, t):
        """Points of the straight unstable segment I of (1,0), parametrized by t in [0,1].

        t = 0 gives the preimage of P_0 under the period map, and t = 1 gives P_0.
        """
        c = (1-self.a)*(1/self.lam+(1-1/self.lam)*np.asarray(t))
        return np.column_stack([1-c, -self.m*c])

    def point(self, t, n):
        """Image of the seed with parameter t under the n-th power of the period map."""
        y = self.seeds([t])[0]
        for _ in range(n):
            y, _ = self.step(y)
        return y

    def batch(self, ys, steps=1, counts=None):
        """Apply the period map `steps` times to many states. Escaping states become NaN."""
        ys = np.ascontiguousarray(ys, dtype=float).copy()
        if counts is None:
            counts = np.zeros(len(ys), dtype=np.int32)
        lib.iterate_batch(ys, len(ys), steps, self.l1, self.l2, self.alpha, self.a, counts)
        return ys, counts

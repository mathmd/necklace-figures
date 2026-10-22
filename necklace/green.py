"""Fronts with prescribed threshold crossings, in the paper's coordinate.

The origin is the left end of an interval where p = 1, and the coefficient has
L1 = h, L2 = 2h and alpha = 1/4. The bounded solution V_rho of K V = 1_{x<rho}
is evaluated in closed form from the Floquet solutions y_+ and y_- (eq. (3.21)
of the paper), which are kept in logarithmic form so that long plateaus cause no
loss of accuracy. A front with crossings rho_1 < ... < rho_m is the alternating
sum of these profiles (Lemma 5.1).
"""
import numpy as np
from scipy.optimize import brentq
from .switching import Map, matrix


class GreenFront:
    def __init__(self, h=0.8):
        self.h = float(h)
        self.L = 3 * self.h
        self.alpha = 0.25
        self.mp = Map(l1=self.h, l2=2*self.h, alpha=self.alpha)
        self.loglam = np.log(self.mp.lam)
        self._normalization = {
            sign: self._midpoint_bloch(-self.h/2, sign)[0]
            for sign in (1, -1)
        }

    def _midpoint_bloch(self, x, sign):
        j = int(np.floor(x/self.L))
        z = x-j*self.L
        y = np.array([1.0, sign*self.mp.m])
        for length, kappa in [(self.h/2, 1), (2*self.h, self.alpha), (self.h/2, 1)]:
            dt = min(length, z)
            y = matrix(dt, kappa) @ y
            z -= dt
            if z <= 1e-14:
                break
        if y[0] <= 0:
            raise FloatingPointError('Nonpositive homogeneous mode')
        return sign*j*self.loglam + np.log(y[0]), y[1]/y[0]

    def bloch(self, x, sign):
        logvalue, fluxratio = self._midpoint_bloch(x-self.h/2, sign)
        return logvalue-self._normalization[sign], fluxratio

    def mode(self, x, sign):
        return np.exp(self.bloch(x, sign)[0])

    def b(self, rho):
        _, rp = self.bloch(rho, 1)
        _, rm = self.bloch(rho, -1)
        return rp/(rp-rm)

    def response(self, x, rho):
        value = self.b(rho)
        if x <= rho:
            logx, ratio = self.bloch(x, 1)
            logrho, _ = self.bloch(rho, 1)
            defect = (1-value)*np.exp(logx-logrho)
            return np.array([1-defect, -defect*ratio])
        logx, ratio = self.bloch(x, -1)
        logrho, _ = self.bloch(rho, -1)
        value *= np.exp(logx-logrho)
        return np.array([value, value*ratio])

    def profile(self, xs, rhos):
        # Vectorize in x; retain signed responses in a stable bounded sum.
        xs = np.asarray(xs, dtype=float)
        plus = np.array([self.bloch(x, 1) for x in xs])
        minus = np.array([self.bloch(x, -1) for x in xs])
        result = np.zeros((len(xs), 2))
        for j, rho in enumerate(rhos):
            b = self.b(rho)
            lp, _ = self.bloch(rho, 1)
            lm, _ = self.bloch(rho, -1)
            left = xs <= rho
            result[left, 0] += (-1)**j*(1-(1-b)*np.exp(plus[left, 0]-lp))
            result[left, 1] += (-1)**j*(-(1-b)*np.exp(plus[left, 0]-lp)*plus[left, 1])
            result[~left, 0] += (-1)**j*b*np.exp(minus[~left, 0]-lm)
            result[~left, 1] += (-1)**j*b*np.exp(minus[~left, 0]-lm)*minus[~left, 1]
        return result

    def residual(self, rhos, a):
        return self.profile(rhos, rhos)[:, 0]-a

    def phases(self, a):
        points = [0, self.h, self.L]
        roots = []
        for left, right in zip(points[:-1], points[1:]):
            if (self.b(left)-a)*(self.b(right)-a) < 0:
                roots.append(brentq(lambda x: self.b(x)-a, left, right, xtol=1e-14))
        return roots

    def verify(self, rhos, a):
        """Check all extrema on exact affine subintervals, not a spatial mesh.

        The external tails are exact bounded homogeneous tails. Floating-point
        residuals are diagnostics, not outward-rounded certificates.
        """
        rhos = np.asarray(rhos)
        lo, hi = rhos[0]-12, rhos[-1]+12
        vertices = []
        for j in range(int(np.floor(lo/self.L))-1, int(np.ceil(hi/self.L))+1):
            vertices.extend([j*self.L, j*self.L+self.h])
        breaks = sorted(set([lo, hi, *rhos, *[x for x in vertices if lo < x < hi]]))
        wrong, errors, internal = [], [], []
        umin, umax = 1.0, 0.0
        for x0, x1 in zip(breaks[:-1], breaks[1:]):
            mid = (x0+x1)/2
            kappa = 1.0 if mid % self.L < self.h else self.alpha
            rate = np.sqrt(kappa)
            regime = int(np.searchsorted(rhos, mid) % 2 == 0)
            y0, y1 = self.profile([x0, x1], rhos)
            defect = y0[0]-regime
            length = x1-x0
            times = [0.0, length]
            if abs(defect) > 1e-15:
                argument = -rate*y0[1]/defect
                if 0 < argument < 1:
                    critical = np.arctanh(argument)/rate
                    if 1e-12 < critical < length-1e-12:
                        times.append(critical)
            values = []
            for dt in times:
                y = matrix(dt, kappa) @ np.array([defect, y0[1]])
                value = regime+y[0]
                values.append(value)
                if dt not in [0.0, length]:
                    internal.append({'x': x0+dt, 'u': value, 'regime': regime})
            predicted = matrix(length, kappa) @ np.array([defect, y0[1]]) + [regime, 0]
            errors.append(max(abs(predicted-y1)))
            wrong.append(max(values)-a if regime == 0 else a-min(values))
            umin, umax = min(umin, min(values)), max(umax, max(values))
        return {
            'threshold_residual': float(max(abs(self.residual(rhos, a)))),
            'max_wrong_side_at_all_segment_extrema': float(max(wrong)),
            'max_affine_flow_discrepancy': float(max(errors)),
            'crossing_fluxes': self.profile(rhos, rhos)[:, 1].tolist(),
            'range_on_checked_interval': [float(umin), float(umax)],
            'affine_interval_count': len(breaks)-1,
            'interior_extrema': internal,
            'tail_control': 'Exact bounded homogeneous tails outside first/last crossing.',
        }

# engine_curve.py
# Pure-Python monotone cubic (PCHIP-style) interpolator for ICE torque vs ω.

import math
from typing import Dict, Iterable, List, Union

Number = Union[int, float]

def rpm_to_rad_s(rpm: Number) -> float:
    """Convert rpm to rad/s."""
    return (float(rpm) * 2.0 * math.pi) / 60.0

# Default WOT ICE-only curve as a DICTIONARY: {omega(rad/s): torque(Nm)}
# Sporty 1.6L turbo engine - ~145 kW (195 hp) peak power
DEFAULT_CURVE: Dict[float, float] = {
    rpm_to_rad_s(0):    0.0,
    rpm_to_rad_s(400):  80.0,
    rpm_to_rad_s(800):  130.0,
    rpm_to_rad_s(1000): 170.0,
    rpm_to_rad_s(1500): 220.0,   # plateau starts
    rpm_to_rad_s(2000): 220.0,
    rpm_to_rad_s(2500): 220.0,
    rpm_to_rad_s(3000): 220.0,
    rpm_to_rad_s(3500): 220.0,
    rpm_to_rad_s(4100): 220.0,   # plateau ends
    rpm_to_rad_s(4500): 210.0,
    rpm_to_rad_s(5000): 195.0,
    rpm_to_rad_s(5600): 170.0,   # ~100 kW anchor
    rpm_to_rad_s(6000): 155.0,
    rpm_to_rad_s(6500): 135.0,
    rpm_to_rad_s(6800): 115.0,
}

def _prepare_xy_m(curve_dict: Dict[float, float]):
    """Prepare sorted x,y and PCHIP slopes (Fritsch–Carlson)."""
    x = sorted(curve_dict.keys())
    y = [curve_dict[k] for k in x]
    n = len(x)
    dx = [x[i+1]-x[i] for i in range(n-1)]
    dy = [y[i+1]-y[i] for i in range(n-1)]
    delta = [dy[i]/dx[i] for i in range(n-1)]
    m = [0.0]*n

    # Interior slopes
    for i in range(1, n-1):
        if delta[i-1] * delta[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2*dx[i] + dx[i-1]
            w2 = dx[i] + 2*dx[i-1]
            m[i] = (w1 + w2) / (w1/delta[i-1] + w2/delta[i])

    # Endpoints
    if n == 2:
        m[0] = m[1] = delta[0]
        return x, y, m, dx, delta

    m0 = ((2*dx[0] + dx[1]) * delta[0] - dx[0] * delta[1]) / (dx[0] + dx[1])
    if m0 * delta[0] <= 0: m0 = 0.0
    elif (delta[0]*delta[1] < 0) and (abs(m0) > 2*abs(delta[0])): m0 = 2*delta[0]

    mn = ((2*dx[-1] + dx[-2]) * delta[-1] - dx[-1] * delta[-2]) / (dx[-1] + dx[-2])
    if mn * delta[-1] <= 0: mn = 0.0
    elif (delta[-1]*delta[-2] < 0) and (abs(mn) > 2*abs(delta[-1])): mn = 2*delta[-1]

    m[0], m[-1] = m0, mn
    return x, y, m, dx, delta

# Precompute for default curve
_X_DEF, _Y_DEF, _M_DEF, _DX_DEF, _DELTA_DEF = _prepare_xy_m(DEFAULT_CURVE)

def _evaluate_pchip_scalar(w: float, x, y, m, dx, delta) -> float:
    """Evaluate piecewise-cubic Hermite at scalar w (rad/s)."""
    n = len(x)
    if w <= x[0]:
        return y[0] + delta[0]*(w - x[0])      # left linear extrapolation
    if w >= x[-1]:
        return y[-1] + delta[-1]*(w - x[-1])   # right linear extrapolation

    # Binary search for segment
    lo, hi = 0, n-2
    while lo <= hi:
        mid = (lo + hi) // 2
        if x[mid] <= w <= x[mid+1]:
            i = mid
            break
        elif w < x[mid]:
            hi = mid - 1
        else:
            lo = mid + 1
    else:
        i = max(0, min(n-2, lo))

    h = x[i+1] - x[i]
    t = (w - x[i]) / h
    h00 = (2*t**3 - 3*t**2 + 1)
    h10 = (t**3 - 2*t**2 + t)
    h01 = (-2*t**3 + 3*t**2)
    h11 = (t**3 - t**2)
    return (h00*y[i] + h10*h*m[i] + h01*y[i+1] + h11*h*m[i+1])

def torque_from_omega(omega_rad_s: Union[Number, Iterable[Number]],
                      curve: Dict[float, float] = None):
    """
    Monotone cubic (PCHIP-style) interpolation of torque (Nm) vs angular speed (rad/s).
    - omega_rad_s: float or iterable of floats [rad/s]
    - curve: optional dict {omega(rad/s): torque(Nm)}; defaults to DEFAULT_CURVE
    Returns: float or list[float]
    """
    if curve is None:
        x, y, m, dx, delta = _X_DEF, _Y_DEF, _M_DEF, _DX_DEF, _DELTA_DEF
    else:
        x, y, m, dx, delta = _prepare_xy_m(curve)

    if isinstance(omega_rad_s, (list, tuple)):
        return [_evaluate_pchip_scalar(float(w), x, y, m, dx, delta) for w in omega_rad_s]
    return _evaluate_pchip_scalar(float(omega_rad_s), x, y, m, dx, delta)

def torque_from_rpm(rpm: Union[Number, Iterable[Number]],
                    curve: Dict[float, float] = None):
    """Helper: query by rpm instead of rad/s."""
    if isinstance(rpm, (list, tuple)):
        return torque_from_omega([rpm_to_rad_s(r) for r in rpm], curve=curve)
    return torque_from_omega(rpm_to_rad_s(rpm), curve=curve)

__all__ = [
    "DEFAULT_CURVE",
    "rpm_to_rad_s",
    "torque_from_omega",
    "torque_from_rpm",
]

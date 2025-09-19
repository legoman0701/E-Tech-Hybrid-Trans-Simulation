# carsim_cv2.py
# 2D side-view car sim with two wheels, vertical spring-damper suspensions,
# OpenCV rendering, and front-wheel torque I/O only.
# Units: m, s, N, kg, rad.

from dataclasses import dataclass
import numpy as np
import math
import random
import cv2

@dataclass
class Params:
    # Body
    mass: float = 600.0          # kg
    J: float = 550.0             # kg*m^2 body inertia about COM
    width: float = 2.0           # m (visual)
    height: float = 0.6          # m (visual)
    wheelbase: float = 1.8       # m
    anchor_y: float = -0.25      # m (anchors below COM in body frame)

    # Wheels
    wheel_radius: float = 0.32   # m
    wheel_inertia: float = 1.4   # kg*m^2 (per wheel)
    hub_viscous: float = 0.04    # N*m*s

    # Suspension (per wheel)
    k: float = 25000.0           # N/m
    c: float = 3200.0            # N*s/m
    rest_len: float = 0.30       # m
    max_travel: float = 0.25     # m

    # Contact/friction
    mu: float = 1.3              # Coulomb coefficient
    vs_slip: float = 0.6         # m/s, smooth slip scale (tanh)
    c_rr: float = 0.015          # rolling resistance factor

    # Environment
    g: float = 9.81              # m/s^2
    air_cdA: float = 0.55        # N/(m/s)^2 approx 0.5*rho*Cd*A

class CarSim:
    """
    Public API:
      - CarSim(px_per_m=40.0)
      - reset()                  # re-seeds terrain and resets state
      - set_front_hub_torque(tau_nm: float)
      - step(dt: float) -> dict  # returns front-wheel telemetry dict
      - render(width_px: int, height_px: int) -> np.ndarray (BGR frame)

    Only I/O with the outside world:
      - Input: set_front_hub_torque(...)
      - Output: step(...) returns {'angular_pos_rad','angular_vel_rad_s','reaction_torque_nm'}
                (front wheel only)
    """
    def __init__(self, px_per_m: float = 40.0):
        self.px_per_m = float(px_per_m)
        self.p = Params()
        # Local anchors (rear=0, front=1)
        hb = 0.5 * self.p.wheelbase
        self.anchors_local = [(-hb, self.p.anchor_y), (+hb, self.p.anchor_y)]
        self._seed_terrain()
        self._reset_state()

    # ------------------ Public methods ------------------

    def reset(self):
        self._seed_terrain()
        self._reset_state()

    def set_front_hub_torque(self, tau_nm: float):
        self._tau_in[1] = float(tau_nm)  # front wheel input torque only

    def step(self, dt: float):
        """Advance physics by dt (s). Return front hub telemetry dict."""
        dt = float(dt)
        # Optionally substep for stability if dt is big
        max_sub = 1.0 / 300.0
        nsub = max(1, int(math.ceil(dt / max_sub)))
        h = dt / nsub
        for _ in range(nsub):
            self._substep(h)

        # Return FRONT wheel telemetry only
        i = 1  # front index
        return {
            "angular_pos_rad": self.wheel_ang[i],
            "angular_vel_rad_s": self.wheel_omg[i],
            "reaction_torque_nm": self._tau_out[i],
        }

    def render(self, width_px: int, height_px: int) -> np.ndarray:
        """Return an OpenCV BGR frame (all drawing is done here)."""
        W, H = int(width_px), int(height_px)
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = (18, 18, 22)

        # View transform: keep the car horizontally centered; vertically anchor terrain under COM.
        scale = self.px_per_m
        x_cam = self.x
        y0 = self._terrain(self.x)  # baseline terrain height under car
        base_px = int(H * 0.75)

        def w2s(wx, wy):
            sx = int(W * 0.5 + (wx - x_cam) * scale)
            sy = int(base_px - (wy - y0) * scale)
            return sx, sy

        # Draw terrain polyline across screen
        xs = np.linspace(-W * 0.5 / scale + x_cam, W * 0.5 / scale + x_cam, num=800)
        ys = np.array([self._terrain(x) for x in xs])
        poly = np.array([w2s(x, y) for x, y in zip(xs, ys)], dtype=np.int32)
        cv2.polylines(img, [poly], isClosed=False, color=(70, 70, 70), thickness=2)

        # Draw body as a rotated rectangle
        body_c = (self.x, self.y)
        w_m, h_m = self.p.width, self.p.height
        hw, hh = 0.5 * w_m, 0.5 * h_m
        c, s = math.cos(self.theta), math.sin(self.theta)
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        pts = []
        for lx, ly in corners:
            wx = body_c[0] + (c * lx - s * ly)
            wy = body_c[1] + (s * lx + c * ly)
            pts.append(w2s(wx, wy))
        cv2.polylines(img, [np.array(pts, np.int32)], True, (200, 200, 255), 3)

        # Wheels and suspensions
        wheel_pos = self._wheel_draw_positions()
        for i, (wx, wy) in enumerate(wheel_pos):
            sx, sy = w2s(wx, wy)
            rpx = int(self.p.wheel_radius * scale)
            cv2.circle(img, (sx, sy), rpx, (230, 230, 230), 2)

            # Spoke for rotation
            spoke_x = wx + self.p.wheel_radius * math.cos(self.wheel_ang[i])
            spoke_y = wy + self.p.wheel_radius * math.sin(self.wheel_ang[i])
            cv2.line(img, (sx, sy), w2s(spoke_x, spoke_y), (230, 230, 230), 2)

            # Suspension line (anchor to center)
            ax, ay = self._point_world(self.anchors_local[i])
            cv2.line(img, w2s(ax, ay), (sx, sy), (140, 160, 255), 2)

        # HUD (front wheel telemetry only)
        tele = {
            "vx": self.vx,
            "theta_deg": math.degrees(self.theta),
            "front_ang": self.wheel_ang[1],
            "front_omg": self.wheel_omg[1],
            "front_tau_out": self._tau_out[1],
            "tau_in": self._tau_in[1],
        }
        hud = [
            f"vx={tele['vx']:+6.2f} m/s   pitch={tele['theta_deg']:+6.2f} deg",
            f"Front: ang={tele['front_ang']:+6.3f} rad  omg={tele['front_omg']:+7.3f} rad/s",
            f"tau_in={tele['tau_in']:+7.1f} N·m   tau_out={tele['front_tau_out']:+7.1f} N·m",
        ]
        y = 18
        for line in hud:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
            y += 18

        return img

    # ------------------ Internal: physics & helpers ------------------

    def _reset_state(self):
        p = self.p
        self.x = 0.0
        # Start a bit above ground so suspensions settle
        self.y = self._terrain(0.0) + p.wheel_radius + p.rest_len + 0.12
        self.vx = 0.0
        self.vy = 0.0
        self.theta = 0.0
        self.omega = 0.0

        self.wheel_ang = [0.0, 0.0]
        self.wheel_omg = [0.0, 0.0]
        self._tau_in = [0.0, 0.0]     # only index 1 (front) is used
        self._tau_out = [0.0, 0.0]    # reaction torque (hub feels)

    def _rot_body_to_world(self, lx: float, ly: float):
        c, s = math.cos(self.theta), math.sin(self.theta)
        return (c * lx - s * ly, s * lx + c * ly)

    def _point_world(self, local_pt):
        rx, ry = self._rot_body_to_world(local_pt[0], local_pt[1])
        return (self.x + rx, self.y + ry)

    def _point_world_vel(self, local_pt):
        rx, ry = self._rot_body_to_world(local_pt[0], local_pt[1])
        return (self.vx - self.omega * ry, self.vy + self.omega * rx)

    def _terrain(self, x: float) -> float:
        # Smooth wavy terrain, reproducible by self._terrain_seed
        a1, f1, ph1, a2, f2, ph2, bias = self._terrain_seed
        return (a1 * math.sin(f1 * (x + ph1)) +
                a2 * math.sin(f2 * (x + ph2)) + bias)

    def _seed_terrain(self):
        rng = random.Random()
        seed = random.randint(0, 10_000_000)
        rng.seed(seed)
        a1 = rng.uniform(0.00, 0.10)   # m
        a2 = rng.uniform(0.00, 0.06)   # m
        f1 = rng.uniform(0.15, 0.35)   # rad/m
        f2 = rng.uniform(0.60, 1.10)   # rad/m
        ph1 = rng.uniform(-5.0, 5.0)
        ph2 = rng.uniform(-5.0, 5.0)
        bias = rng.uniform(-0.01, 0.01)
        self._terrain_seed = (a1, f1, ph1, a2, f2, ph2, bias)

    def _wheel_draw_positions(self):
        """Wheel center positions used for drawing (accounts for contact vs air)."""
        pos = []
        for local in self.anchors_local:
            ax, ay = self._point_world(local)
            ground_y = self._terrain(ax)
            raw_gap = ay - ground_y
            compression = (self.p.rest_len + self.p.wheel_radius) - raw_gap
            compression = max(0.0, min(self.p.max_travel, compression))
            in_contact = compression > 1e-7
            if in_contact:
                wy = ground_y + self.p.wheel_radius
            else:
                wy = ay - self.p.rest_len
            pos.append((ax, wy))
        return pos

    def _substep(self, dt: float):
        p = self.p
        Fx, Fy, Tau = 0.0, 0.0, 0.0

        # Aero drag (quadratic)
        speed = math.hypot(self.vx, self.vy)
        if speed > 1e-6:
            drag = p.air_cdA * speed * speed
            Fx -= drag * (self.vx / speed)
            Fy -= drag * (self.vy / speed)

        # Per wheel: suspension + longitudinal friction, wheel dynamics
        for i, local in enumerate(self.anchors_local):
            ax, ay = self._point_world(local)
            avx, avy = self._point_world_vel(local)

            # Terrain contact along vertical ray
            gy = self._terrain(ax)
            raw_gap = ay - gy
            compression = (p.rest_len + p.wheel_radius) - raw_gap
            compression = max(0.0, min(p.max_travel, compression))

            # Compression velocity approximation: anchor moving down increases compression
            comp_vel = -avy

            Fn = p.k * compression + p.c * comp_vel
            if Fn < 0.0:
                Fn = 0.0

            in_contact = compression > 1e-7
            fx_i, fy_i = 0.0, 0.0

            if in_contact and Fn > 0.0:
                # Upward normal
                fy_i += Fn

                # Longitudinal slip/friction (horizontal only; ignore slope for simplicity)
                v_long = avx
                surf = self.wheel_omg[i] * p.wheel_radius
                slip = v_long - surf

                F_c = p.mu * Fn
                F_long = -F_c * math.tanh(slip / max(1e-6, p.vs_slip))

                # Rolling resistance opposes motion
                if abs(v_long) > 1e-3:
                    F_rr = -p.c_rr * Fn * math.copysign(1.0, v_long)
                else:
                    F_rr = 0.0

                fx_i += (F_long + F_rr)

                # Wheel dynamics
                tau_contact = -fx_i * p.wheel_radius
                tau_visc = p.hub_viscous * self.wheel_omg[i]
                tau_net = self._tau_in[i] + tau_contact - tau_visc
                alpha = tau_net / p.wheel_inertia
                self.wheel_omg[i] += alpha * dt
                self.wheel_ang[i] = (self.wheel_ang[i] + self.wheel_omg[i] * dt) % (2 * math.pi)

                # Report reaction torque felt at hub (contact + viscous, opposite sign of applied)
                self._tau_out[i] = -(tau_contact + tau_visc)
            else:
                # Free spin (no ground torque)
                tau_net = self._tau_in[i] - p.hub_viscous * self.wheel_omg[i]
                alpha = tau_net / p.wheel_inertia
                self.wheel_omg[i] += alpha * dt
                self.wheel_ang[i] = (self.wheel_ang[i] + self.wheel_omg[i] * dt) % (2 * math.pi)
                self._tau_out[i] = -( -p.hub_viscous * self.wheel_omg[i] )

            # Apply to body at anchor
            Fx += fx_i
            Fy += fy_i
            rx, ry = self._rot_body_to_world(local[0], local[1])
            Tau += rx * fy_i - ry * fx_i

        # Gravity
        Fy -= p.mass * p.g

        # Integrate body (semi-implicit Euler)
        ax = Fx / p.mass
        ay = Fy / p.mass
        ang_acc = Tau / p.J

        self.vx += ax * dt
        self.vy += ay * dt
        self.omega += ang_acc * dt

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.theta += self.omega * dt

        # Simple floor safety: don't let COM fall far below terrain beneath (rare)
        min_y = self._terrain(self.x) + 0.05
        if self.y < min_y:
            self.y = min_y
            if self.vy < 0.0:
                self.vy = 0.0

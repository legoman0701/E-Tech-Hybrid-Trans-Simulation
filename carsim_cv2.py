#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
carsim_cv2.py
-------------
Reusable 2D car simulator with real-ish physics units and OpenCV rendering.

Public API (minimal I/O):
- set_front_hub_torque(torque_nm): input torque applied at the *front* wheel hub (N·m). Positive drives forward.
- step(dt) -> dict: advances physics by dt seconds and returns a dictionary with front-wheel telemetry:
    {
      "omega_rad_s": float,            # front wheel angular speed
      "angle_rad": float,              # front wheel angle (wrapped)
      "reaction_torque_nm": float,     # torque delivered *back* from wheel to hub (N·m), opposite sign to input when resisting
      "slip_speed_mps": float,         # slip speed at contact (tangential)
      "normal_force_N": float,         # normal force on front wheel
      "tangent_force_N": float,        # friction force magnitude (signed along terrain tangent)
      "speed_mps": float               # chassis x-velocity
    }
- render(width, height) -> np.ndarray (BGR): a frame suitable for cv2.imshow/write.

Notes:
- Units: m, s, kg, N, N·m, rad, rad/s.
- Terrain is a smooth random sum of sines with a tiny base slope.
- Suspension is a *prismatic joint* along chassis local -Y (perpendicular to body),
  with a lateral link to prevent side "flop".
- Ground contact uses a penalty normal model and Coulomb-capped viscous tangential friction.
- Only control/telemetry is the torque to/from the *front* wheel hub.

Author: ChatGPT (GPT-5 Thinking)
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import cv2

# ---------------------------
# Colors (BGR for OpenCV)
# ---------------------------
WHITE = (240, 240, 240)
BLACK = (10, 10, 10)
GREY  = (120, 120, 130)
BLUE  = (220, 140, 70)
GREEN = (90, 180, 100)
BROWN = (60, 80, 120)   # darker ground fill
LINEG = (60, 180, 90)
YELLOW= (60, 220, 240)
BGCLR = (35, 30, 25)

# ---------------------------
# Helpers
# ---------------------------
def clamp(x, a, b):
    return a if x < a else b if x > b else x

def sgn(x: float) -> float:
    if x > 0: return 1.0
    if x < 0: return -1.0
    return 0.0

# ---------------------------
# Terrain
# ---------------------------
class Terrain:
    def __init__(self, seed=None):
        if seed is None:
            seed = random.randrange(1_000_000)
        self.seed = seed
        rnd = random.Random(seed)
        self.components = []
        specs = [
            (80.0,  2.0),
            (40.0,  1.2),
            (20.0,  0.6),
            (250.0, 4.0),
        ]
        for lam, amp in specs:
            phase = rnd.uniform(0, 2*math.pi)
            amp *= rnd.uniform(0.6, 1.4)
            self.components.append((lam, amp, phase))
        self.base_slope = rnd.uniform(-0.02, 0.02)
        self.h0 = 5.0

    def h(self, x):
        y = self.h0 + self.base_slope * x
        for lam, amp, phase in self.components:
            y += amp * math.sin(2 * math.pi * x / lam + phase)
        return y

    def dhdx(self, x):
        dy = self.base_slope
        for lam, amp, phase in self.components:
            dy += amp * (2 * math.pi / lam) * math.cos(2 * math.pi * x / lam + phase)
        return dy

    def normal(self, x):
        slope = self.dhdx(x)
        nx, ny = -slope, 1.0
        inv = 1.0 / math.hypot(nx, ny)
        return nx*inv, ny*inv

    def tangent(self, x):
        slope = self.dhdx(x)
        tx, ty = 1.0, slope
        inv = 1.0 / math.hypot(tx, ty)
        return tx*inv, ty*inv

# ---------------------------
# Rigid bodies
# ---------------------------
@dataclass
class RigidBody:
    m: float
    I: float
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    theta: float = 0.0
    omega: float = 0.0
    fx: float = 0.0
    fy: float = 0.0
    tau: float = 0.0

    def clear_forces(self):
        self.fx = self.fy = self.tau = 0.0

    def add_force(self, fx, fy, rx=0.0, ry=0.0):
        self.fx += fx
        self.fy += fy
        self.tau += rx * fy - ry * fx

    def integrate(self, dt):
        # Semi-implicit Euler
        self.vx += (self.fx / self.m) * dt
        self.vy += (self.fy / self.m) * dt
        self.x  += self.vx * dt
        self.y  += self.vy * dt

        self.omega += (self.tau / self.I) * dt
        self.theta += self.omega * dt

@dataclass
class Wheel(RigidBody):
    r: float = 0.35
    drive: bool = True
    ang: float = 0.0
    last_ground_torque_nm: float = 0.0
    last_tangent_force_N: float = 0.0
    last_normal_force_N: float = 0.0
    last_slip_speed_mps: float = 0.0

    def integrate(self, dt):
        super().integrate(dt)
        self.theta = math.fmod(self.theta + math.pi, 2*math.pi) - math.pi
        self.ang = self.theta

# ---------------------------
# CarSim
# ---------------------------
class CarSim:
    def __init__(self, px_per_m: float = 35.0):
        self.PX_PER_M = px_per_m
        self.WIDTH, self.HEIGHT = 1280, 720

        self.terrain = Terrain()
        self.car = self._make_car()

        # Camera, with no look-ahead (center on chassis x)
        self.cam_x = self.car['chassis'].x
        self.cam_y = self.car['chassis'].y

        # External input
        self.input_torque_front_nm = 0.0

    # -----------------------
    # Construction & helpers
    # -----------------------
    def _make_car(self):
        # Chassis
        chassis = RigidBody(m=420.0, I=100.0, x=5.0, y=self.terrain.h(5.0)+1.5, theta=0.0)

        # Wheels
        wheel_mass = 22.0
        wheel_r = 0.35
        wheel_I = 0.5 * wheel_mass * wheel_r**2
        wb = 1.9
        wheel_front = Wheel(m=wheel_mass, I=wheel_I, x=chassis.x + wb/2, y=chassis.y - 0.5, r=wheel_r, drive=True)
        wheel_rear  = Wheel(m=wheel_mass, I=wheel_I, x=chassis.x - wb/2, y=chassis.y - 0.5, r=wheel_r, drive=False)

        # Place wheels on terrain
        wheel_front.y = self.terrain.h(wheel_front.x) + wheel_front.r + 0.05
        wheel_rear.y  = self.terrain.h(wheel_rear.x)  + wheel_rear.r  + 0.05

        # Suspension model params
        k_s = 18000.0; c_s = 2200.0; rest_len = 0.35
        k_link = 90000.0; c_link = 3000.0
        k_gn = 220000.0; c_gn = 1400.0; mu = 1.0; k_tan = 3600.0; Crr = 0.02

        off_y = -0.1
        mount_front_local = ( +wb/2, -0.2 + off_y )
        mount_rear_local  = ( -wb/2, -0.2 + off_y )

        rho = 1.225; CdA = 0.7

        return {
            'chassis': chassis,
            'wf': wheel_front,
            'wr': wheel_rear,
            'wb': wb,
            'k_s': k_s, 'c_s': c_s, 'rest_len': rest_len,
            'k_link': k_link, 'c_link': c_link,
            'k_gn': k_gn, 'c_gn': c_gn, 'mu': mu, 'k_tan': k_tan, 'Crr': Crr,
            'mount_front_local': mount_front_local,
            'mount_rear_local': mount_rear_local,
            'rho': rho, 'CdA': CdA
        }

    def chassis_local_to_world(self, lx, ly):
        c = self.car['chassis']
        ct = math.cos(c.theta); st = math.sin(c.theta)
        return c.x + ct*lx - st*ly, c.y + st*lx + ct*ly

    # -----------------------
    # External API
    # -----------------------
    def set_front_hub_torque(self, torque_nm: float):
        """Input torque applied at front hub (N·m). Positive drives forward."""
        self.input_torque_front_nm = float(torque_nm)

    
    # -----------
    # Properties / getters for external access (read-anytime)
    # -----------
    @property
    def front_wheel_angle(self) -> float:
        return self.car['wf'].ang

    @property
    def front_wheel_omega(self) -> float:
        return self.car['wf'].omega

    @property
    def front_reaction_torque(self) -> float:
        return -self.car['wf'].last_ground_torque_nm

    @property
    def front_input_torque(self) -> float:
        return self.input_torque_front_nm

    def get_front_wheel_state(self):
        wf = self.car['wf']
        return {
            'angle_rad': wf.ang,
            'omega_rad_s': wf.omega,
            'reaction_torque_nm': -wf.last_ground_torque_nm
        }
    def step(self, dt: float) -> Dict[str, float]:
        """Advance physics by dt seconds and return front wheel telemetry."""
        car = self.car
        ch = car['chassis']; wf = car['wf']; wr = car['wr']

        # Clear
        ch.clear_forces(); wf.clear_forces(); wr.clear_forces()
        wf.last_ground_torque_nm = 0.0
        wf.last_tangent_force_N  = 0.0
        wf.last_normal_force_N   = 0.0
        wf.last_slip_speed_mps   = 0.0
        wr.last_ground_torque_nm = 0.0

        # Gravity
        g = 9.81
        ch.add_force(0.0, -ch.m * g)
        wf.add_force(0.0, -wf.m * g)
        wr.add_force(0.0, -wr.m * g)

        # Suspension forces (prismatic along chassis -Y with lateral link)
        self._suspension_forces(wf, car['mount_front_local'])
        self._suspension_forces(wr, car['mount_rear_local'])

        # Ground contact
        self._ground_contact(wf)
        self._ground_contact(wr)

        # Aero drag on chassis (quadratic)
        vx = ch.vx; v = abs(vx)
        Fd = -0.5 * car['rho'] * car['CdA'] * v * vx
        ch.add_force(Fd, 0.0)

        # Apply external hub torque to FRONT wheel only
        wf.tau += self.input_torque_front_nm

        # Integrate
        ch.integrate(dt)
        wf.integrate(dt)
        wr.integrate(dt)

        # Camera follow (no look-ahead)
        target_x = ch.x
        target_y = max(self.terrain.h(ch.x) + 1.2, ch.y)
        lerp = 0.15
        self.cam_x += (target_x - self.cam_x) * lerp
        self.cam_y += (target_y - self.cam_y) * lerp

        # Reaction torque at hub: equal-and-opposite of ground torque acting on the wheel
        reaction_nm = -wf.last_ground_torque_nm

        return {
            "omega_rad_s": wf.omega,
            "angle_rad": wf.ang,
            "reaction_torque_nm": reaction_nm,
            "slip_speed_mps": wf.last_slip_speed_mps,
            "normal_force_N": wf.last_normal_force_N,
            "tangent_force_N": wf.last_tangent_force_N,
            "speed_mps": ch.vx
        }

    def render(self, width: int = 1280, height: int = 720) -> np.ndarray:
        """Return a BGR frame for OpenCV display/writing."""
        self.WIDTH, self.HEIGHT = width, height
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = BGCLR

        # Terrain
        pts = []
        x0 = self.cam_x - width / (2*self.PX_PER_M) - 2.0
        x1 = self.cam_x + width / (2*self.PX_PER_M) + 2.0
        step = 0.5
        x = x0
        while x <= x1:
            y = self.terrain.h(x)
            px, py = self.world_to_screen(x, y)
            pts.append([px, py])
            x += step
        if len(pts) >= 2:
            poly = np.array([[pts[0][0], height-1]] + pts + [[pts[-1][0], height-1]], dtype=np.int32)
            cv2.fillPoly(img, [poly], BROWN)
            cv2.polylines(img, [np.int32(pts)], False, LINEG, 2, lineType=cv2.LINE_AA)

        # Car
        self._draw_wheel(img, self.car['wr'])
        self._draw_wheel(img, self.car['wf'])
        self._draw_chassis_and_springs(img)

        # HUD
        ch = self.car['chassis']; wf = self.car['wf']
        v = ch.vx; vkmh = v*3.6
        slope_deg = math.degrees(math.atan(self.terrain.dhdx(ch.x)))
        hud = [
            f"Speed: {v:6.2f} m/s  ({vkmh:6.1f} km/h)",
            f"Front:  w={wf.omega:7.2f} rad/s  T_in={self.input_torque_front_nm:6.1f} N·m  T_rxn={-wf.last_ground_torque_nm:6.1f} N·m",
            f"Slip: {wf.last_slip_speed_mps:+.2f} m/s   Fn: {wf.last_normal_force_N:7.1f} N   Ft: {wf.last_tangent_force_N:7.1f} N",
            f"Slope: {slope_deg:+.1f}°"
        ]
        y = 20
        for t in hud:
            cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
            y += 20

        return img

    # -----------------------
    # Internals
    # -----------------------
    def world_to_screen(self, x, y):
        sx = int((x - self.cam_x) * self.PX_PER_M + self.WIDTH // 2)
        sy = int(self.HEIGHT // 2 - (y - self.cam_y) * self.PX_PER_M)
        return sx, sy

    def _draw_wheel(self, img, wheel: Wheel):
        cx, cy = self.world_to_screen(wheel.x, wheel.y)
        rpx = max(2, int(wheel.r * self.PX_PER_M))
        cv2.circle(img, (cx, cy), rpx+2, BLACK, 2, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx, cy), rpx, GREY, 2, lineType=cv2.LINE_AA)
        ex = int(cx + math.cos(-wheel.ang) * rpx)
        ey = int(cy - math.sin(-wheel.ang) * rpx)
        cv2.line(img, (cx, cy), (ex, ey), BLUE, 2, lineType=cv2.LINE_AA)

    def _draw_chassis_and_springs(self, img):
        ch = self.car['chassis']
        W = 2.0; H = 0.4
        corners_local = [(-W/2, -H/2), (W/2, -H/2), (W/2, H/2), (-W/2, H/2)]
        pts = []
        for lx, ly in corners_local:
            x, y = self.chassis_local_to_world(lx, ly)
            pts.append(self.world_to_screen(x, y))
        poly = np.array(pts, dtype=np.int32)
        cv2.fillConvexPoly(img, poly, GREY)
        cv2.polylines(img, [poly], True, BLACK, 2, lineType=cv2.LINE_AA)

        # springs
        mx_f, my_f = self.chassis_local_to_world(*self.car['mount_front_local'])
        mx_r, my_r = self.chassis_local_to_world(*self.car['mount_rear_local'])
        sxf, syf = self.world_to_screen(mx_f, my_f)
        sxr, syr = self.world_to_screen(mx_r, my_r)
        cxf, cyf = self.world_to_screen(self.car['wf'].x, self.car['wf'].y)
        cxr, cyr = self.world_to_screen(self.car['wr'].x, self.car['wr'].y)
        cv2.line(img, (sxf, syf), (cxf, cyf), YELLOW, 2, lineType=cv2.LINE_AA)
        cv2.line(img, (sxr, syr), (cxr, cyr), YELLOW, 2, lineType=cv2.LINE_AA)

    def _suspension_forces(self, wheel: Wheel, mount_local: Tuple[float, float]):
        car = self.car
        ch = car['chassis']

        mx, my = self.chassis_local_to_world(*mount_local)

        ct = math.cos(ch.theta); st = math.sin(ch.theta)
        # local axes
        a_down_x, a_down_y = ( st, -ct)   # body local (0,-1)
        a_lat_x,  a_lat_y  = ( ct,  st)   # body local (+1,0)

        # vector mount->wheel
        dx = wheel.x - mx; dy = wheel.y - my
        L_axial = dx * a_down_x + dy * a_down_y
        L_lat   = dx * a_lat_x  + dy * a_lat_y

        # mount point velocity
        rx = mx - ch.x; ry = my - ch.y
        vcx = ch.vx + (-ch.omega * ry)
        vcy = ch.vy + ( ch.omega * rx)

        vrx = wheel.vx - vcx; vry = wheel.vy - vcy
        v_rel_ax = vrx * a_down_x + vry * a_down_y
        v_rel_lt = vrx * a_lat_x  + vry * a_lat_y

        # spring-damper axial
        x_comp = car['rest_len'] - L_axial
        Fax = car['k_s'] * x_comp - car['c_s'] * v_rel_ax

        # lateral link
        Flat = - car['k_link'] * L_lat - car['c_link'] * v_rel_lt

        fx = Fax * a_down_x + Flat * a_lat_x
        fy = Fax * a_down_y + Flat * a_lat_y

        wheel.add_force(+fx, +fy)
        ch.add_force(-fx, -fy, rx, ry)

    def _ground_contact(self, wheel: Wheel):
        car = self.car
        terr = self.terrain
        hx = terr.h(wheel.x)
        nx, ny = terr.normal(wheel.x)
        tx, ty = terr.tangent(wheel.x)

        depth = (hx + wheel.r) - wheel.y
        if depth > 0.0:
            v_rel_n = wheel.vx * nx + wheel.vy * ny
            Fn = car['k_gn'] * depth - car['c_gn'] * v_rel_n
            if Fn < 0.0: Fn = 0.0
            wheel.add_force(Fn * nx, Fn * ny)
            wheel.last_normal_force_N = Fn

            # tangential slip speed at contact
            v_t = wheel.vx * tx + wheel.vy * ty - wheel.omega * wheel.r
            Ft_visc = - car['k_tan'] * v_t
            Ft_max = car['mu'] * Fn
            Ft = clamp(Ft_visc, -Ft_max, +Ft_max)

            wheel.add_force(Ft * tx, Ft * ty)
            wheel.tau += -Ft * wheel.r

            # record telemetry
            wheel.last_tangent_force_N = Ft
            wheel.last_slip_speed_mps  = v_t

            # rolling resistance as additional force at COM
            v_long = wheel.vx * tx + wheel.vy * ty
            Fr_roll = - car['Crr'] * Fn * sgn(v_long)
            wheel.add_force(Fr_roll * tx, Fr_roll * ty)

            # Track ground torque (from friction only) acting on wheel
            wheel.last_ground_torque_nm += -Ft * wheel.r

    # -----------------------
    # Reset
    # -----------------------
    def reset(self, seed=None):
        self.terrain = Terrain(seed=seed)
        self.car = self._make_car()
        self.cam_x = self.car['chassis'].x
        self.cam_y = self.car['chassis'].y
        self.input_torque_front_nm = 0.0
        return True


# ---------------------------
# Minimal demo (optional)
# ---------------------------
def _demo():
    sim = CarSim(px_per_m=40.0)
    t_in = 0.0
    dt = 1.0/240.0
    clock_hz = 120  # render rate
    frame_interval = int(round(1_000 / clock_hz))

    print("Demo controls (OpenCV window focused):")
    print("  D: +torque,  A: -torque,  S: zero torque,  R: reset terrain,  Q/Esc: quit")
    while True:
        # simple keyboard-based torque
        key = cv2.waitKey(frame_interval) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('d'):
            t_in += 20.0
        elif key == ord('a'):
            t_in -= 20.0
        elif key == ord('s'):
            t_in = 0.0
        elif key == ord('r'):
            sim.reset()

        # physics step(s)
        # integrate multiple substeps per render for stability
        sub = 2
        for _ in range(sub):
            sim.set_front_hub_torque(t_in)
            telem = sim.step(dt)

        frame = sim.render(1280, 720)
        cv2.imshow("CarSim OpenCV (torque control)", frame)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()

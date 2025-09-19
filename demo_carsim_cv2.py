#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_carsim_cv2.py
------------------
Example of using the carsim_cv2 module from another file.
The ONLY communication with the simulator is the torque to/from the *front* wheel.

Controls:
  D: increase torque (+20 N·m)
  A: decrease torque (-20 N·m)
  S: zero torque
  R: reset terrain
  Q or Esc: quit
"""
import cv2
from carsim_cv2 import CarSim

def main():
    sim = CarSim(px_per_m=40.0)
    dt = 1.0/60.0
    torque_in = 0.0

    print("Controls — D:+20Nm  A:-20Nm  S:zero  R:reset  Q/Esc:quit")
    while True:
        key = cv2.waitKey(8) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('d'):
            torque_in += 20.0
        elif key == ord('a'):
            torque_in -= 20.0
        elif key == ord('s'):
            torque_in = 0.0
        elif key == ord('r'):
            sim.reset()

        # Push torque to sim, step, pull reaction torque back
        sim.set_front_hub_torque(torque_in)
        telemetry = sim.step(1.0/120.0)  # single substep, or do more for stability
        # Optional: use telemetry['reaction_torque_nm'] for your engine model
        # print(telemetry)

        frame = sim.render(600, 600)
        cv2.imshow("CarSim OpenCV — torque I/O only", frame)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
q
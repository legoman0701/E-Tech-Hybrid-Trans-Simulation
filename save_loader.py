from math import *
import pygame, time, random

class Gear:
  def __init__(self, number_teeth, inertia, name, poly):
    self.teeth = number_teeth
    self.inertia = inertia
    self.name = name
    self.speed = 0
    self.angle = 0
    self.torque = 0
    self.drag = 0.
    self.poly = poly

def rotate_polygon(points, angle):
  cos_theta = cos(angle)
  sin_theta = sin(angle)
  cx = sum(x for x, y in points) / len(points)
  cy = sum(y for x, y in points) / len(points)
  rotated_points = []
  for x, y in points:
    x -= cx
    y -= cy
    x_new = x * cos_theta - y * sin_theta
    y_new = x * sin_theta + y * cos_theta
    rotated_points.append((x_new + cx, y_new + cy))
  return rotated_points

def create_gear_polygon(radius, teeth, tooth_depth):
  points = []
  angle_step = 2 * pi / (teeth * 2)
  for i in range(teeth * 2):
    angle = i * angle_step
    r = radius + tooth_depth if i % 2 == 0 else radius
    x = r * cos(angle)
    y = r * sin(angle)
    points.append((x, y))
  
  points.append((radius, 0))
  points.append((-radius, 0))
  points.append((0, radius))
  points.append((0, -radius))
  return points








def load_gears_from_file(filename="gears.txt"):
    file  = open(filename, "r")
    lines = file.readlines()
    result = [line.strip().split() for line in lines if line.strip()]

    gears = []
    for i, row in enumerate(result):
        for j, item in enumerate(row):
            if item[-1] == 'C' or item[-1] == 'O':
                if item == 'DC':
                    continue
                teeth = int(item[:-2])
                inertia = teeth*teeth/500
                radius = teeth * 2.5
                poly = create_gear_polygon(radius, teeth, 10)
                gear = Gear(teeth, inertia, f"Gear {len(gears)+1}", poly)
                gear.offset = (j * 150, i * 150)
                gears.append(gear)
    return gears, result

gears, result = load_gears_from_file()

print(f"Loaded {gears} gears from gears.txt")
print(f"Gear arrangement: {result}")
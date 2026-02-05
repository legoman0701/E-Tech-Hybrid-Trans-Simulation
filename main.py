from math import sin, cos, pi, copysign, isfinite
import pygame, time
from engine_curve import torque_from_omega
import multiprocessing
import cv2
import numpy as np
from carsim_cv2 import CarSim

WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
BACKGROUND_COLOR = (30, 30, 30)

class Gear:
  def __init__(self, number_teeth, inertia, name):
    self.teeth = number_teeth
    self.inertia = inertia
    self.name = name
    self.speed = 0
    self.angle = 0
    self.torque = 0
    self.drag = 0.001
    self.offset = (0, 0)

class DogClutch:
  def __init__(self):
    self.engaged = 0 #-1 for left, +1 for right, 0 for none
    self.old_engaged = 0
    self.angle_offset = 0
    self.left_gear_index = 0
    self.right_gear_index = 0
    self.right_most_axle_conected_gear_index = 0
    self.offset = (0, 0)

class Clutch:
  def __init__(self):
    self.engaged = 0.0  # 0.0 to 1.0
    self.angle_offset = 0
    self.left_gear_index = 0
    self.right_gear_index = 0
    self.offset = (0, 0)

class SlippingClutch:
  def __init__(self, clutch_type='P'):
    self.engaged = 0.0  # 0.0 to 1.0, controls clutch pressure
    self.left_gear_index = 0  # Index of the #C gear on the left
    self.right_gear_index = 0  # Index of the #P or #I gear on the right
    self.offset = (0, 0)
    self.clutch_type = clutch_type  # 'P' for pair, 'I' for impair
    self.max_torque = 500.0  # Maximum torque the clutch can transfer [N·m]
    self.stiffness = 50.0  # Clutch spring stiffness

class Engine:
  def __init__(self, inertia, name):
    self.inertia = inertia
    self.name = name
    self.speed = 0
    self.angle = 0
    self.torque = 0
    self.drag = 0.06
    self.tau0 = 10
    self.offset = (0, 0)
    self.throttle = 0
    self.rev_cut = 5800
    self.rev_act = 5600
    self.rev_limit_activated = 0
    self.conected_gear_index = 0

  def apply_physics(self, dt):
    self.torque += max(torque_from_omega(self.speed)*(self.throttle if not self.rev_limit_activated else 0), 0)
    tau_c = self.tau0 * copysign(1, self.speed) if self.speed != 0 else 0
    self.speed += (self.torque - tau_c - self.drag*self.speed) / self.inertia * dt
    if abs(self.speed) < 0.0001: self.speed = 0
    self.torque = 0.0
    self.angle += self.speed * dt

class TransESC:
  def __init__(self):
    self.gears_p = [[0, -1], [0, 1], [1, 0]]
    self.gears_i = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    self.dc_p = [0, 2]
    self.dc_i = [1, 3]
    self.gear = 0
  
  def set_gear(self, dog_clutches):
    if self.gear < 0 or self.gear >= 8:
      return

    if self.gear%2 == 0:
      for i in range(len(self.dc_p)):
        dog_clutches[i].engaged = self.gears_p[(self.gear-1)//2][i]
      for i in range(len(self.dc_i)):
        dog_clutches[i].engaged = 0     

Engines = [None]*3
displayed_numbers = [0, 1, 2, 3, 4, 5, 6]
displayed_numbers.reverse()

R2, R3, R4, MEP_EvA, MEP_EvB, EvA, EvB, FD = 1.125, 0.77688172, 2/3, 1.436206897, 0.761458333, 1.931034483, 1.023809524, 5
Engines_ratios = [R2*FD, R3*FD, R4*FD]

ENG, HSG, MEP = 0, 1, 2

Engines[ENG] = Engine(0.12, "1.6l Engine") # Main Combustion Engine

def solve_gear_joint(g_in, g_out, ratio, s, dt, angle_offset, coef=1, Cdot_only=False):
    #C    = (1.0 * g_in.angle) + s * (ratio * g_out.angle)
    C = 0
    if coef != 1 or Cdot_only:
      C = solve_gear_c(g_in, g_out, ratio, s, dt, 30, angle_offset)
    #Cdot = (1.0 * g_in.speed) + s * (ratio * g_out.speed)
    Cdot = solve_gear_cdot(g_in, g_out, ratio, s, dt)
    lam  =  (C + Cdot)*coef
    if Cdot_only:
      lam = min(max(Cdot, -100), 100)
    #lam  = Cdot
    global error_tot
    error_tot += abs(C)
    
    g_in.torque -= 1.0 * lam
    g_out.torque -= s*ratio * lam

def solve_gear_cdot(g_in, g_out, ratio, s, dt, strength=0.7):
    I1, I2 = g_in.inertia, g_out.inertia
    if dt <= 0 or I1 <= 0 or I2 <= 0: return 0.0
    r = ratio
    cd = g_in.speed + s*r*g_out.speed
    inv = (1.0/I1) + (r*r)*(1.0/I2)
    if not (isfinite(cd) and isfinite(inv) and isfinite(dt)):
      return 0.0
    denom = dt * inv
    if denom == 0:
      return 0.0
    lam = strength*cd/denom
    if not isfinite(lam):
      return 0.0
    return max(-1e6, min(1e6, lam))

def solve_gear_c(g_in, g_out, ratio, s, dt, strength, angle_offset):
    I1, I2 = g_in.inertia, g_out.inertia
    if dt <= 0 or I1 <= 0 or I2 <= 0: return 0.0
    r = ratio
    cd = g_in.angle + (s*r*g_out.angle) - angle_offset
    inv = (1.0/I1) + (r*r)*(1.0/I2)
    if not (isfinite(cd) and isfinite(inv) and isfinite(dt)):
      return 0.0
    denom = dt * inv
    if denom == 0:
      return 0.0
    lam = strength*cd/denom
    if not isfinite(lam):
      return 0.0
    return max(-1e6, min(1e6, lam))

def draw_rpm_gauge(Engines, pos, screen):
  # screen is expected to be a numpy array (cv2 image)
  # pos is (cx, cy) center of the gauge
  cx, cy = pos
  radius = 100
  thickness = 5
  font_scale = 1
  font = cv2.FONT_HERSHEY_SIMPLEX

  # Draw main arc (white)
  cv2.ellipse(
    screen, (int(cx), int(cy)), (radius, radius),
    angle=0, startAngle=0, endAngle=int(-1.25*180),
    color=(255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA
  )
  # Draw red arc (danger zone)
  cv2.ellipse(
    screen, (int(cx), int(cy)), (radius, radius),
    angle=0, startAngle=0, endAngle=int(-0.25*180),
    color=(0, 0, 255), thickness=thickness, lineType=cv2.LINE_AA
  )

  # Draw numbers
  for i, number in enumerate(displayed_numbers):
    alpha = -i/6 * 1.25 * np.pi
    offset_x = int(np.cos(alpha) * 120)
    offset_y = int(np.sin(alpha) * 120)
    text = str(number)
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 2)
    tx = int(cx + offset_x - tw // 2)
    ty = int(cy + offset_y + th // 2)
    cv2.putText(screen, text, (tx, ty), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

  # Draw needle
  rpm = Engines[ENG].speed * 60 / (2 * np.pi)
  alpha = rpm / 1000 / 6
  alpha *= 1.25 * np.pi
  alpha -= 1.25 * np.pi
  needle_x = int(cx + np.cos(alpha) * 90)
  needle_y = int(cy + np.sin(alpha) * 90)
  cv2.line(screen, (int(cx), int(cy)), (needle_x, needle_y), (255, 255, 255), 3, cv2.LINE_AA)
  cv2.imshow("RPM Gauge", screen)
  cv2.waitKey(1)

  return screen

def check_if_gear(item):
    """Check if an item in the grid represents a gear."""
    if item == 'DC' or item in ['CP', 'CI'] or item[0] == 'C':
        return False
    if item[-1] in "OCPI":
        return True
    return False

def load_gears_from_file(filename="gears DCT.txt", Engines=[]):
    """
    Load transmission configuration from a text file.
    
    Returns:
        tuple: (gears, outside_connections, axle_connections, dog_clutches, 
                axle_displacement, grid_data, clutches)
    """
    GRID_SPACING = 150
    GEAR_MODULE = 1.8
    
    # Helper functions
    def get_offset(col, row):
        """Convert grid position to offset coordinates."""
        return (col * GRID_SPACING, row * GRID_SPACING)
    
    def find_gear_index(gears, offset):
        """Find gear index by its offset position."""
        return next((index for index, g in enumerate(gears) if g.offset == offset), None)
    
    def find_adjacent_gear(row, start_col, direction, gear_type='O'):
        """
        Find the nearest gear in a given direction along a row.
        
        Args:
            row: The row to search in
            start_col: Starting column index
            direction: -1 for left, +1 for right
            gear_type: 'O' for output gear, 'C' for connected gear
        """
        col = start_col
        while 0 <= col < len(row):
            col += direction
            if col < 0 or col >= len(row):
                break
            if check_if_gear(row[col]) and row[col][-1] == gear_type:
                return col
        return None
    
    # Read and parse file
    with open(filename, "r") as file:
        lines = file.readlines()
    grid_data = [line.strip().split() for line in lines if line.strip()]
    
    gears = []
    outside_connections = []
    axle_connections = []
    dog_clutches = []
    clutches = []
    slipping_clutches = []
    
    # Step 1: Create all gears from the grid
    for row_idx, row in enumerate(grid_data):
        for col_idx, item in enumerate(row):
            if check_if_gear(item):
                teeth = int(item[:-2])
                inertia = teeth * teeth / 1000000
                gear = Gear(teeth, inertia, item)
                gear.offset = get_offset(col_idx, row_idx)
                gears.append(gear)
    
    # Step 2: Add vertical (external mesh) connections
    for row_idx in range(len(grid_data) - 1):
        for col_idx, item in enumerate(grid_data[row_idx]):
            if check_if_gear(item) and check_if_gear(grid_data[row_idx + 1][col_idx]):
                offset1 = get_offset(col_idx, row_idx)
                offset2 = get_offset(col_idx, row_idx + 1)
                idx1 = find_gear_index(gears, offset1)
                idx2 = find_gear_index(gears, offset2)
                if idx1 is not None and idx2 is not None:
                    outside_connections.append((idx1, idx2))
    
    # Step 3: Add horizontal (axle) connections - adjacent gears
    for row_idx, row in enumerate(grid_data):
        for col_idx in range(len(row) - 1):
            current_item = row[col_idx]
            next_item = row[col_idx + 1]
            
            # Connect adjacent #C gears
            if current_item[-1:] == 'C' and current_item != 'DC':
                if next_item[-1:] == 'C' and next_item != 'DC':
                    offset1 = get_offset(col_idx, row_idx)
                    offset2 = get_offset(col_idx + 1, row_idx)
                    idx1 = find_gear_index(gears, offset1)
                    idx2 = find_gear_index(gears, offset2)
                    if idx1 is not None and idx2 is not None:
                        axle_connections.append((idx1, idx2))
            
            # Connect adjacent #P gears
            if current_item[-1:] == 'P' and check_if_gear(current_item):
                if next_item[-1:] == 'P' and check_if_gear(next_item):
                    offset1 = get_offset(col_idx, row_idx)
                    offset2 = get_offset(col_idx + 1, row_idx)
                    idx1 = find_gear_index(gears, offset1)
                    idx2 = find_gear_index(gears, offset2)
                    if idx1 is not None and idx2 is not None:
                        axle_connections.append((idx1, idx2))
            
            # Connect adjacent #I gears
            if current_item[-1:] == 'I' and check_if_gear(current_item):
                if next_item[-1:] == 'I' and check_if_gear(next_item):
                    offset1 = get_offset(col_idx, row_idx)
                    offset2 = get_offset(col_idx + 1, row_idx)
                    idx1 = find_gear_index(gears, offset1)
                    idx2 = find_gear_index(gears, offset2)
                    if idx1 is not None and idx2 is not None:
                        axle_connections.append((idx1, idx2))
    
    # Step 4: Add horizontal (axle) connections - gears with gaps
    for row_idx, row in enumerate(grid_data):
        col_idx = 0
        while col_idx < len(row):
            item = row[col_idx]
            
            # Check if current item is a connected gear (#C, #P, or #I)
            is_connectable = (
                (item[-1:] == 'C' and item != 'DC' and check_if_gear(item)) or
                (item[-1:] == 'P' and check_if_gear(item)) or
                (item[-1:] == 'I' and check_if_gear(item))
            )
            
            if is_connectable:
                gear_type = item[-1:]
                # Find the next gear with matching type on the same shaft
                search_offset = 1
                found_next = False
                
                while col_idx + search_offset < len(row) and search_offset < 10:  # Limit search distance
                    next_item = row[col_idx + search_offset]
                    
                    # Stop if we hit another type of connected gear (different shaft)
                    if check_if_gear(next_item):
                        next_type = next_item[-1:]
                        if next_type in ['C', 'P', 'I'] and next_type != gear_type:
                            # Different shaft type, stop searching
                            break
                        elif next_type == gear_type:
                            # Found matching gear on same shaft
                            offset1 = get_offset(col_idx, row_idx)
                            offset2 = get_offset(col_idx + search_offset, row_idx)
                            idx1 = find_gear_index(gears, offset1)
                            idx2 = find_gear_index(gears, offset2)
                            if idx1 is not None and idx2 is not None:
                                # Avoid duplicate connections
                                if (idx1, idx2) not in axle_connections and (idx2, idx1) not in axle_connections:
                                    axle_connections.append((idx1, idx2))
                            found_next = True
                            break
                    
                    search_offset += 1
            
            col_idx += 1
    
    # Step 5: Parse dog clutches
    for row_idx, row in enumerate(grid_data):
        for col_idx, item in enumerate(row):
            if item == 'DC':
                clutch = DogClutch()
                clutch.offset = get_offset(col_idx, row_idx)
                
                # Find left output gear
                left_col = find_adjacent_gear(row, col_idx, -1, 'O')
                if left_col is not None:
                    idx = find_gear_index(gears, get_offset(left_col, row_idx))
                    if idx is not None:
                        clutch.left_gear_index = idx
                
                # Find right output gear
                right_col = find_adjacent_gear(row, col_idx, +1, 'O')
                if right_col is not None:
                    idx = find_gear_index(gears, get_offset(right_col, row_idx))
                    if idx is not None:
                        clutch.right_gear_index = idx
                
                # Find rightmost connected gear
                connected_col = find_adjacent_gear(row, col_idx, +1, 'C')
                if connected_col is not None:
                    idx = find_gear_index(gears, get_offset(connected_col, row_idx))
                    if idx is not None:
                        clutch.right_most_axle_conected_gear_index = idx
                
                dog_clutches.append(clutch)
    
    # Step 6: Calculate axle displacements based on gear radii
    axle_displacement = [0]
    for row_idx in range(1, len(grid_data)):
        for col_idx, item in enumerate(grid_data[row_idx]):
            if check_if_gear(item) and check_if_gear(grid_data[row_idx - 1][col_idx]):
                offset1 = get_offset(col_idx, row_idx - 1)
                offset2 = get_offset(col_idx, row_idx)
                idx1 = find_gear_index(gears, offset1)
                idx2 = find_gear_index(gears, offset2)
                
                if idx1 is not None and idx2 is not None:
                    gear1 = gears[idx1]
                    gear2 = gears[idx2]
                    radius1 = gear1.teeth * GEAR_MODULE
                    radius2 = gear2.teeth * GEAR_MODULE
                    axle_displacement.append((radius1 + radius2) + axle_displacement[-1])
                    break
    
    # Step 7: Parse regular clutches
    for row_idx, row in enumerate(grid_data):
        for col_idx, item in enumerate(row):
            if item == 'C':
                clutch = Clutch()
                clutch.offset = get_offset(col_idx, row_idx)
                
                # Find left output gear
                left_col = find_adjacent_gear(row, col_idx, -1, 'O')
                if left_col is not None:
                    idx = find_gear_index(gears, get_offset(left_col, row_idx))
                    if idx is not None:
                        clutch.left_gear_index = idx
                
                # Find right output gear
                right_col = find_adjacent_gear(row, col_idx, +1, 'O')
                if right_col is not None:
                    idx = find_gear_index(gears, get_offset(right_col, row_idx))
                    if idx is not None:
                        clutch.right_gear_index = idx
                
                clutches.append(clutch)
    
    # Step 7b: Parse slipping clutches (CP and CI)
    for row_idx, row in enumerate(grid_data):
        for col_idx, item in enumerate(row):
            if item in ['CP', 'CI']:
                clutch_type = item[1]  # 'P' or 'I'
                clutch = SlippingClutch(clutch_type)
                clutch.offset = get_offset(col_idx, row_idx)
                
                # Find left #C gear
                left_col = find_adjacent_gear(row, col_idx, -1, 'C')
                if left_col is not None:
                    idx = find_gear_index(gears, get_offset(left_col, row_idx))
                    if idx is not None:
                        clutch.left_gear_index = idx
                
                # Find right #P or #I gear (depending on clutch type)
                right_col = find_adjacent_gear(row, col_idx, +1, clutch_type)
                if right_col is not None:
                    idx = find_gear_index(gears, get_offset(right_col, row_idx))
                    if idx is not None:
                        clutch.right_gear_index = idx
                
                slipping_clutches.append(clutch)
    
    # Step 8: Connect engines to their gears
    for row_idx, row in enumerate(grid_data):
        for col_idx, item in enumerate(row):
            if item == "ENG":
                engine_gear_offset = get_offset(col_idx + 1, row_idx)
                idx = find_gear_index(gears, engine_gear_offset)
                if idx is not None:
                    Engines[ENG].conected_gear_index = idx
    
    return gears, outside_connections, axle_connections, dog_clutches, axle_displacement, grid_data, clutches, slipping_clutches

def draw_gear(g, surface, module=1.8, scaling=1.0, pan_offset=(0,0), axle_displacement=[]):
  x, y = g.offset
  y = axle_displacement[y//150]
  #print(y/150, g.name)
  Z = g.teeth
  angle = g.angle
  R = Z * module*scaling
  x = WINDOW_WIDTH/2 + (x + pan_offset[0]) * scaling
  y = WINDOW_HEIGHT/2 + (y + pan_offset[1]) * scaling

  lateral_offset = (pygame.mouse.get_pos()[0] - WINDOW_WIDTH/2) /300
  lateral_offset = max(-20, min(20, lateral_offset)) * R
  tooth_height = int(-(40/R)*abs(lateral_offset/4)+40)*scaling

  # Draw main body
  pygame.draw.rect(surface, (100, 100, 100), (x-tooth_height/2, y - R, tooth_height, 2 * R))
  pygame.draw.ellipse(surface, (100, 100, 100), (x - tooth_height/2 - lateral_offset/2 , y - R, lateral_offset, 2 * R), int(R))
  pygame.draw.ellipse(surface, (100, 100, 100), (x + tooth_height/2 - lateral_offset/2 , y - R, lateral_offset, 2 * R), int(R))
  
  # Draw teeth as alternating rectangles on sides
  tooth_width = (4 * R) / Z
  for i in range(Z):
    alpha = (i / Z) * 2 * pi + angle

    on_downstroke = sin(alpha) > 0
    negative_offset = False
    if lateral_offset < 0: 
      on_downstroke = not on_downstroke
      negative_offset = True
  

    if on_downstroke:
      tooth_x = x - tooth_height/2 + sin(alpha) * abs(lateral_offset/2)
      tooth_y = y + cos(alpha) * R
      #draw tooth
      pygame.draw.rect(surface, (20, 20, 20), (tooth_x, tooth_y-tooth_width*abs(sin(alpha))/2, tooth_height, tooth_width*abs(sin(alpha))))
      #draw a line at the center of the tooth
      #pygame.draw.rect(surface, (255,0,0), (tooth_x, tooth_y, tooth_height, 1))
    else:
      tooth_x = x - tooth_height/2 + sin(alpha) * abs(lateral_offset/2)
      tooth_y = y + cos(alpha) * R
      #draw tooth
      pygame.draw.rect(surface, BACKGROUND_COLOR, (tooth_x + (tooth_height if negative_offset else 0), tooth_y-tooth_width*abs(sin(alpha))/2, 5*scaling, tooth_width*abs(sin(alpha))))

def window_space_trans(points, offset, scaling):
  rotated_points = []
  for x, y in points:
    x*=scaling
    y*=scaling
    x += WINDOW_WIDTH/2 + offset[0]
    y += WINDOW_HEIGHT/2 + offset[1]
    rotated_points.append((x, y))
  return rotated_points

scaling = 1
pan_offset = [0, 0]
is_panning = False
last_mouse_pos = (0, 0)
debug_mode = False

def main():
  global scaling, pan_offset, is_panning, last_mouse_pos, error_tot, debug_mode
  dt = 1/60
  debut = time.time()
  
  gears, outside_conections, axle_conections, dog_clutches, axle_displacement, result, clutches, slipping_clutches = load_gears_from_file(Engines=Engines)

  DCT_Trans = TransESC()
  
  pygame.init()
  screen = pygame.display.set_mode((800, 600))
  pygame.display.set_caption("Transmission Simulation")
  
  
  #sim = CarSim(px_per_m=40.0)

  running = True
  while running:
    debut = time.time()
    dt = min(1/90, dt)
    
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_d:
          debug_mode = not debug_mode
        if event.key == pygame.K_z:DCT_Trans.gear+=1
        if event.key == pygame.K_s:DCT_Trans.gear-=1
        # Slipping clutch controls
        if event.key == pygame.K_1 and len(slipping_clutches) > 0:
          slipping_clutches[0].engaged = 1.0 if slipping_clutches[0].engaged == 0 else 0.0
        if event.key == pygame.K_2 and len(slipping_clutches) > 1:
          slipping_clutches[1].engaged = 1.0 if slipping_clutches[1].engaged == 0 else 0.0
        
      # Zoom with mouse wheel
      if event.type == pygame.MOUSEBUTTONDOWN:
        if event.button == 4:  # Scroll up
          scaling *= 1.1
        if event.button == 5:  # Scroll down
          scaling /= 1.1
        if event.button == 1:  # Left mouse button
          is_panning = True
          last_mouse_pos = pygame.mouse.get_pos()
      if event.type == pygame.MOUSEBUTTONUP:
        if event.button == 1:
          is_panning = False
      if event.type == pygame.MOUSEMOTION:
        if is_panning:
          mx, my = pygame.mouse.get_pos()
          dx = mx - last_mouse_pos[0]
          dy = my - last_mouse_pos[1]
          pan_offset[0] += dx
          pan_offset[1] += dy
          last_mouse_pos = (mx, my)

      if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
        mx, my = pygame.mouse.get_pos()
        for g in gears:
          # For the side view, simply use the rectangle bounds
          center_x = WINDOW_WIDTH/2 + (g.offset[0] + pan_offset[0]) * scaling
          center_y = WINDOW_HEIGHT/2 + (g.offset[1] + pan_offset[1]) * scaling
          gear_width = 40 * scaling
          gear_height = max(60, g.teeth * 8) * scaling
          rect = pygame.Rect(center_x-gear_width/2, center_y-gear_height/2, gear_width, gear_height)
          if rect.collidepoint(mx, my):
            g.torque += 4000
            break

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
      if len(gears) > 0:
        gears[0].torque += 10
    if keys[pygame.K_RIGHT]:
      if len(gears) > 1:
        gears[1].torque += 10
      
    if keys[pygame.K_UP]:
      Engines[ENG].throttle = min(1, Engines[ENG].throttle + 0.1)
    else:
      Engines[ENG].throttle = max(0, Engines[ENG].throttle - 0.1)
    
    if keys[pygame.K_s]:
      Engines[ENG].speed += 1


    if Engines[ENG].speed*60/(2*pi) > Engines[ENG].rev_cut: Engines[ENG].rev_limit_activated = True
    elif Engines[ENG].speed*60/(2*pi) < Engines[ENG].rev_act and Engines[ENG].rev_limit_activated: Engines[ENG].rev_limit_activated = False

    s = +1.0            # +1 external mesh, -1 internal/belt
      
    #dt = min(1/300, dt)
    dt /= 100
    error_tot = 0
    DCT_Trans.set_gear(dog_clutches)
    for i in range(100):
      for index1, index2 in outside_conections:
        g_in = gears[index1]
        g_out = gears[index2]

        ratio = g_out.teeth/g_in.teeth
        solve_gear_joint(g_in, g_out, ratio, s, dt, 0)
      
      for index1, index2 in axle_conections:
        g1 = gears[index1]
        g2 = gears[index2]
        solve_gear_joint(g1, g2, 1.0, -s, dt, 0)
        
      for dog_clutch in dog_clutches:
        if dog_clutch.engaged == 0:
          dog_clutch.old_engaged = 0
          continue
        g1 = gears[dog_clutch.left_gear_index]
        g2 = gears[dog_clutch.right_gear_index]
        g3 = gears[dog_clutch.right_most_axle_conected_gear_index]
        ratio = 1.0
        sync_speed_threshold = 60.0  # rad/s; use soft sync if mismatched

        if dog_clutch.engaged == -1:
          if dog_clutch.old_engaged != -1:
            dog_clutch.angle_offset = g1.angle - g3.angle
            dog_clutch.old_engaged = -1
          if abs(g1.speed - g3.speed) > sync_speed_threshold:
            solve_gear_joint(g1, g3, 1.0, -s, dt, dog_clutch.angle_offset, Cdot_only=True)
          else:
            solve_gear_joint(g1, g3, 1.0, -s, dt, dog_clutch.angle_offset)
        if dog_clutch.engaged == +1:
          if dog_clutch.old_engaged != +1:
            dog_clutch.angle_offset = g2.angle - g3.angle
            dog_clutch.old_engaged = +1
          if abs(g2.speed - g3.speed) > sync_speed_threshold:
            solve_gear_joint(g2, g3, 1.0, -s, dt, dog_clutch.angle_offset, Cdot_only=True)
          else:
            solve_gear_joint(g2, g3, 1.0, -s, dt, dog_clutch.angle_offset)
      
      for engine in Engines:
        if engine == None: continue
        g1 = gears[engine.conected_gear_index]
        solve_gear_joint(engine, g1, 1.0, -s, dt, 0)
      
      # Apply slipping clutch constraints (torque-based, not position-based)
      for slip_clutch in slipping_clutches:
        if slip_clutch.engaged > 0.01:
          g_left = gears[slip_clutch.left_gear_index]
          g_right = gears[slip_clutch.right_gear_index]
          solve_gear_joint(g_left, g_right, 1.0, -s, dt, 0, coef=1, Cdot_only=True)
      
      for g in gears:
        inertia = g.inertia if isfinite(g.inertia) and g.inertia > 1e-8 else 1e-8
        if not (isfinite(g.speed) and isfinite(g.torque) and isfinite(g.drag)):
          g.speed = 0.0
          g.torque = 0.0
        g.speed += ((g.torque) - g.drag * g.speed) / inertia * (dt)
        if not isfinite(g.speed):
          g.speed = 0.0
        g.angle += g.speed * (dt)
        if not isfinite(g.angle):
          g.angle = 0.0
        g.torque = 0.0
      
      for engine in Engines:
        if engine == None: continue
        engine.apply_physics(dt)

      #sim.step(dt)
      #print(g1.speed)
    
    
    #Engines[ENG].torque += torque_from_omega(Engines[ENG].speed)*(Engines[ENG].throttle if not Engines[ENG].rev_limit_activated else 0)
    #tau_c = Engines[ENG].tau0 * copysign(1, Engines[ENG].speed) if Engines[ENG].speed != 0 else 0
    #Engines[ENG].speed += (Engines[ENG].torque - tau_c - Engines[ENG].drag*Engines[ENG].speed) / Engines[ENG].inertia * dt
    #if abs(Engines[ENG].speed) < 0.01: Engines[ENG].speed = 0
    #Engines[ENG].torque = 0.0

    #print(f"Engine: {Engines[ENG].speed*60/(2*pi):.0f} rpm, Engine2 {Engines[MEP].speed*60/(2*pi):.0f} rpm, Engine3 {Engines[HSG].speed*60/(2*pi):.0f} rpm")
    #print("Total constraint error:", tot_error)

    screen.fill(BACKGROUND_COLOR)

    error_tot/=100
    #print("Average constraint error:", error_tot)
    
    for i, row in enumerate(result):
      for j, item in enumerate(row):
        if item == '-':
          y = axle_displacement[i]
          x = WINDOW_WIDTH/2 + (j * 150 + pan_offset[0]) * scaling
          y = WINDOW_HEIGHT/2 + (y + pan_offset[1]) * scaling
          pygame.draw.line(screen, (200,200,200), (x-150*scaling, y), (x+150*scaling, y), max(1, int(3*scaling)))
    
    for i, row in enumerate(result):
      for j in range(len(row)-1):
        item = row[j]
        if check_if_gear(item):
          if check_if_gear(row[j+1]):
            y = axle_displacement[i]
            x = WINDOW_WIDTH/2 + (j * 150 + pan_offset[0]) * scaling
            y = WINDOW_HEIGHT/2 + (y + pan_offset[1]) * scaling
            pygame.draw.line(screen, (200,200,200), (x, y), (x+150*scaling, y), max(1, int(3*scaling)))

    for i, row in enumerate(result):
      for j, item in enumerate(row):
        if item == 'DC':
          y = axle_displacement[i]
          x = WINDOW_WIDTH/2 + (j * 150 + pan_offset[0]) * scaling
          y = WINDOW_HEIGHT/2 + (y + pan_offset[1]) * scaling
          pygame.draw.line(screen, (200,200,200), (x-150*scaling, y), (x+150*scaling, y), max(1, int(3*scaling)))

          c_offset = (j * 150, i * 150)
          idx_c = next(index for index, c in enumerate(dog_clutches) if c.offset == c_offset)
          clutch = dog_clutches[idx_c]
          engagement_offset = clutch.engaged * 30*scaling

          pygame.draw.rect(screen, (150,50,50), (x-15*scaling + engagement_offset, y-15*scaling, 30*scaling, 30*scaling))
        
        # Draw slipping clutches (CP/CI)
        if item in ['CP', 'CI']:
          y = axle_displacement[i]
          x = WINDOW_WIDTH/2 + (j * 150 + pan_offset[0]) * scaling
          y = WINDOW_HEIGHT/2 + (y + pan_offset[1]) * scaling
          pygame.draw.line(screen, (200,200,200), (x-75*scaling, y), (x+75*scaling, y), max(1, int(3*scaling)))
          
          c_offset = (j * 150, i * 150)
          idx_c = next((index for index, c in enumerate(slipping_clutches) if c.offset == c_offset), None)
          if idx_c is not None:
            clutch = slipping_clutches[idx_c]
            # Color based on engagement (0=green, 1=red)
            color_val = int(255 * clutch.engaged)
            color = (color_val, 255 - color_val, 50)
            
            # Draw clutch pack plates
            plate_width = 25 * scaling
            plate_height = 5 * scaling
            spacing = 8 * scaling
            for p in range(3):
              offset_y = (p - 1) * spacing
              pygame.draw.rect(screen, color, (x - plate_width/2, y + offset_y - plate_height/2, plate_width, plate_height))
            
            # Draw label
            font = pygame.font.SysFont("Arial", int(12*scaling))
            text = item
            text_surface = font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (x - 10*scaling, y - 35*scaling))      

    for g in gears:
      draw_gear(g, screen, module=1.8, scaling=scaling, pan_offset=pan_offset, axle_displacement=axle_displacement)
    
    # Debug mode: draw connection lines for hovered gear
    if debug_mode:
      # Find hovered gear
      mx, my = pygame.mouse.get_pos()
      hovered_gear_idx = None
      
      for idx, g in enumerate(gears):
        center_x = WINDOW_WIDTH/2 + (g.offset[0] + pan_offset[0]) * scaling
        center_y = WINDOW_HEIGHT/2 + (axle_displacement[g.offset[1]//150] + pan_offset[1]) * scaling
        gear_radius = g.teeth * 1.8 * scaling
        # Simple circular hit test
        dx = mx - center_x
        dy = my - center_y
        if (dx*dx + dy*dy) < (gear_radius * gear_radius):
          hovered_gear_idx = idx
          break
      
      if hovered_gear_idx is not None:
        # Highlight the hovered gear
        g = gears[hovered_gear_idx]
        center_x = WINDOW_WIDTH/2 + (g.offset[0] + pan_offset[0]) * scaling
        center_y = WINDOW_HEIGHT/2 + (axle_displacement[g.offset[1]//150] + pan_offset[1]) * scaling
        gear_radius = g.teeth * 1.8 * scaling
        pygame.draw.circle(screen, (255, 255, 0), (int(center_x), int(center_y)), int(gear_radius), max(2, int(3*scaling)))
        
        # Draw outside connections (vertical, external mesh) in bright green
        for idx1, idx2 in outside_conections:
          if idx1 == hovered_gear_idx or idx2 == hovered_gear_idx:
            g1 = gears[idx1]
            g2 = gears[idx2]
            x1 = WINDOW_WIDTH/2 + (g1.offset[0] + pan_offset[0]) * scaling
            y1 = WINDOW_HEIGHT/2 + (axle_displacement[g1.offset[1]//150] + pan_offset[1]) * scaling
            x2 = WINDOW_WIDTH/2 + (g2.offset[0] + pan_offset[0]) * scaling
            y2 = WINDOW_HEIGHT/2 + (axle_displacement[g2.offset[1]//150] + pan_offset[1]) * scaling
            pygame.draw.line(screen, (0, 255, 0), (int(x1), int(y1)), (int(x2), int(y2)), max(3, int(4*scaling)))
        
        # Draw axle connections (horizontal) in yellow-green
        for idx1, idx2 in axle_conections:
          if idx1 == hovered_gear_idx or idx2 == hovered_gear_idx:
            g1 = gears[idx1]
            g2 = gears[idx2]
            x1 = WINDOW_WIDTH/2 + (g1.offset[0] + pan_offset[0]) * scaling
            y1 = WINDOW_HEIGHT/2 + (axle_displacement[g1.offset[1]//150] + pan_offset[1]) * scaling
            x2 = WINDOW_WIDTH/2 + (g2.offset[0] + pan_offset[0]) * scaling
            y2 = WINDOW_HEIGHT/2 + (axle_displacement[g2.offset[1]//150] + pan_offset[1]) * scaling
            pygame.draw.line(screen, (150, 255, 0), (int(x1), int(y1)), (int(x2), int(y2)), max(3, int(4*scaling)))
        
        # Draw gear name/info
        font = pygame.font.SysFont("Arial", int(16*scaling) if scaling > 0.5 else 12)
        info_text = f"{g.name} - {g.speed*60/(2*pi):.0f} RPM"
        text_surface = font.render(info_text, True, (255, 255, 0))
        screen.blit(text_surface, (int(center_x - 50*scaling), int(center_y - gear_radius - 25*scaling)))
    
    draw_rpm_gauge(Engines, (260/2, 260/2), np.zeros((260, 260, 3), dtype=np.uint8))
    
    #print(f"{Engines[ENG].angle}%")
      
    fps = 1.0 / dt if dt > 0 else 0.0
    font = pygame.font.SysFont("Arial", 20)
    text_surface = font.render(f"FPS: {fps:.2f}", True, (255, 255, 255))
    screen.blit(text_surface, (10, 10))
    
    text_surface = font.render(f"Gear: {DCT_Trans.gear}", True, (255, 255, 255))
    screen.blit(text_surface, (10, 40))

    pygame.display.flip()
    
    
    #frame = sim.render(600, 600)
    #cv2.imshow("CarSim OpenCV — torque I/O only", frame)
    
    dt = (time.time() - debut)
    while dt <= 1/90:
      dt = (time.time() - debut)
  pygame.quit()

if __name__ == "__main__":
  error_tot = 0
  main()
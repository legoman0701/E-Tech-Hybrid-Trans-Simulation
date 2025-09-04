from math import sin, cos, pi, copysign
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
    self.drag = 0.002
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

  def apply_throttle(self, dt):
    self.torque += torque_from_omega(self.speed)*(self.throttle if not self.rev_limit_activated else 0)
    tau_c = self.tau0 * copysign(1, self.speed) if self.speed != 0 else 0
    self.speed += (self.torque - tau_c - self.drag*self.speed) / self.inertia * dt
    if abs(self.speed) < 0.01: self.speed = 0
    self.torque = 0.0

class E_Engine:
  def __init__(self, inertia, name, max_amp=20):
    self.inertia = inertia
    self.name = name
    self.speed = 0
    self.angle = 0
    self.torque = 0
    self.drag = 0.06
    self.tau0 = 10
    self.offset = (0, 0)
    self.throttle = 0
    self.conected_gear_index = 0
    self.max_amp = max_amp
    self.volt = 400
  
  def get_toque(self):
    power = self.volt * self.max_amp * self.throttle
    omega = self.speed
    if omega == 0:
      return power / 1
    torque = power / omega
    return torque
  
  def apply_throttle(self, dt):
    self.torque += self.get_toque()
    tau_c = self.tau0 * copysign(1, self.speed) if self.speed != 0 else 0
    self.speed += (self.torque - tau_c - self.drag*self.speed) / self.inertia * dt
    if abs(self.speed) < 0.01: self.speed = 0
    self.torque = 0.0


Engines = [None]*3
displayed_numbers = [0, 1, 2, 3, 4, 5, 6]
displayed_numbers.reverse()

MCI, HSG, MEP = 0, 1, 2

Engines[MCI] = Engine(0.12, "1.6l Engine")
Engines[HSG] = E_Engine(10, "HSG", max_amp=20)  # High Voltage Starter Generator
Engines[MEP] = E_Engine(10, "MEP", max_amp=200)  # Main Electric Propulsion

def draw_rpm_gauge(e, pos, screen):
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
  rpm = Engines[MCI].speed * 60 / (2 * np.pi)
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
  if item == 'DC':
    return False
  if item[-1] == 'C' or item[-1] == 'O':
    return True
  return False

def load_gears_from_file(filename="gears.txt", Engines=[]):
    file  = open(filename, "r")
    lines = file.readlines()
    result = [line.strip().split() for line in lines if line.strip()]

    gears = []
    outside_conections = []
    axle_conections = []

    clutches = []

    #add gears
    for i, row in enumerate(result):
      for j, item in enumerate(row):
        if check_if_gear(item):
          teeth = int(item[:-2])
          inertia = teeth*teeth/100000
          gear = Gear(teeth, inertia, item)
          gear.offset = (j * 150, i * 150)
          gears.append(gear)

    #add external conecting gears
    for i in range(len(result)-1):
      row = result[i]
      for j, item in enumerate(row):
        if check_if_gear(item):
          if check_if_gear(result[i+1][j]):
            g1_offset = (j * 150, i * 150)
            g2_offset = (j * 150, (i+1) * 150)
            idx_g1 = next(index for index, g in enumerate(gears) if g.offset == g1_offset)
            idx_g2 = next(index for index, g in enumerate(gears) if g.offset == g2_offset)
            print(gears[idx_g1].name, gears[idx_g2].name)
            outside_conections.append((idx_g1, idx_g2))

    #add axle conecting gears
    for i, row in enumerate(result):
      for j in range(len(row)-1):
        item = row[j]
        if item != 'DC' and item[-1] == 'C':
          if row[j+1] != 'DC' and row[j+1][-1] == 'C':
            g1_offset = (j * 150, i * 150)
            g2_offset = ((j+1) * 150, i * 150)
            idx_g1 = next(index for index, g in enumerate(gears) if g.offset == g1_offset)
            idx_g2 = next(index for index, g in enumerate(gears) if g.offset == g2_offset)
            axle_conections.append((idx_g1, idx_g2))
    
    #add axle conecting gears with gaps or dog_clutchs
    for i, row in enumerate(result):
      for j in range(1, len(row)-1):
        item = row[j]
        if row[j-1] != 'DC' and row[j-1][-1] == 'C':
          if item == '-' or item == "DC" or  item[-1] == 'O':
            g1 = row[j-1]
            offset = 0
            while row[j+offset] == '-' or row[j+offset] == "DC" or row[j+offset][-1] == 'O':
              offset += 1
            g2 = row[j+offset]
            g1_offset = ((j-1) * 150, i * 150)
            g2_offset = ((j+offset) * 150, i * 150)
            idx_g1 = next(index for index, g in enumerate(gears) if g.offset == g1_offset)
            idx_g2 = next(index for index, g in enumerate(gears) if g.offset == g2_offset)
            axle_conections.append((idx_g1, idx_g2))
    
    #add dog clutches
    for i, row in enumerate(result):
      for j, item in enumerate(row):
        if item == 'DC':
          clutch = DogClutch()
          clutch.offset = (j * 150, i * 150)
          #find left gear
          k = j
          while True:
            k -= 1
            if check_if_gear(row[k]) and row[k][-1] == 'O': break
          g1_offset = (k * 150, i * 150)
          idx_g1 = next(index for index, g in enumerate(gears) if g.offset == g1_offset)
          clutch.left_gear_index = idx_g1

          #find right gear
          k = j
          while True:
            k += 1
            if check_if_gear(row[k]) and row[k][-1] == 'O': break
          g2_offset = (k * 150, i * 150)
          idx_g2 = next(index for index, g in enumerate(gears) if g.offset == g2_offset)
          clutch.right_gear_index = idx_g2
          
          #add axle connection to the right most gear
          k = j
          while True:
            k += 1
            if check_if_gear(row[k]) and row[k][-1] == 'C': break
          g3_offset = (k * 150, i * 150)
          idx_g3 = next(index for index, g in enumerate(gears) if g.offset == g3_offset)
          clutch.right_most_axle_conected_gear_index = idx_g3
          
          clutches.append(clutch)
          
    #add axles displacements
    axle_displacement = [0]
    for i in range(1, len(result)):
      row = result[i]
      for j, item in enumerate(row):
        if check_if_gear(item):
          if check_if_gear(result[i-1][j]):
            g1_offset = (j * 150, (i-1) * 150)
            idx_g1 = next(index for index, g in enumerate(gears) if g.offset == g1_offset)
            g2_offset = (j * 150, i * 150)
            idx_g2 = next(index for index, g in enumerate(gears) if g.offset == g2_offset)

            g1 = gears[idx_g1]
            g2 = gears[idx_g2]

            Z1 = g1.teeth
            Z2 = g2.teeth
            R1 = Z1*1.8
            R2 = Z2*1.8

            axle_displacement.append((R1+R2)+axle_displacement[-1])
            
            break
    
    #add motor gear
    for i, row in enumerate(result):
      for j, item in enumerate(row):
        if item == "MCI":
          g1_offset = ((j-1) * 150, i * 150)
          idx_g1 = next(index for index, g in enumerate(gears) if g.offset == g1_offset)
          Engines[MCI].conected_gear_index = idx_g1
        if item == "HSG":
          g1_offset = ((j+1) * 150, i * 150)
          idx_g1 = next(index for index, g in enumerate(gears) if g.offset == g1_offset)
          Engines[HSG].conected_gear_index = idx_g1
        if item == "MEP":
          g1_offset = ((j+1) * 150, i * 150)
          idx_g1 = next(index for index, g in enumerate(gears) if g.offset == g1_offset)
          Engines[MEP].conected_gear_index = idx_g1
    
    return gears, outside_conections, axle_conections, clutches, axle_displacement, result

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

def main():
  global scaling, pan_offset, is_panning, last_mouse_pos
  dt = 1/60
  debut = time.time()
  
  gears, outside_conections, axle_conections, clutches, axle_displacement, result = load_gears_from_file(Engines=Engines)
  
  pygame.init()
  screen = pygame.display.set_mode((800, 600))
  pygame.display.set_caption("Transmission Simulation")
  
  
  #sim = CarSim(px_per_m=40.0)

  running = True
  time.sleep(0.1)
  su = 0
  substep = 20
  while running:
    debut = time.time()
    dt = min(1/300, dt)
    su += 1
    
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_r:clutches[0].engaged = -1
        if event.key == pygame.K_t:clutches[0].engaged = 0
        if event.key == pygame.K_y:clutches[0].engaged = +1
        if event.key == pygame.K_f:clutches[1].engaged = -1
        if event.key == pygame.K_g:clutches[1].engaged = 0
        if event.key == pygame.K_h:clutches[1].engaged = +1
        if event.key == pygame.K_v:clutches[2].engaged = -1
        if event.key == pygame.K_b:clutches[2].engaged = 0
        if event.key == pygame.K_n:clutches[2].engaged = +1
        
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
      Engines[MCI].throttle = min(1, Engines[MCI].throttle + 0.01)
    else:
      Engines[MCI].throttle = max(0, Engines[MCI].throttle - 0.01)

    if keys[pygame.K_m]:
      Engines[MEP].throttle = min(1, Engines[MEP].throttle + 0.01)
    else:
      Engines[MEP].throttle = max(0, Engines[MEP].throttle - 0.01)
    
    if keys[pygame.K_h]:
      Engines[HSG].throttle = min(1, Engines[HSG].throttle + 0.01)
    else:
      Engines[HSG].throttle = max(0, Engines[HSG].throttle - 0.01)


    if Engines[MCI].speed*60/(2*pi) > Engines[MCI].rev_cut: Engines[MCI].rev_limit_activated = True
    elif Engines[MCI].speed*60/(2*pi) < Engines[MCI].rev_act and Engines[MCI].rev_limit_activated: Engines[MCI].rev_limit_activated = False

    s = +1.0            # +1 external mesh, -1 internal/belt
    k = 100           # angular spring (Nm/rad)
    c = 1           # angular damper (Nms/rad)

    tot_error = 0

    for engine in Engines:
      if engine.name == "1.6l Engine":
        g1 = gears[engine.conected_gear_index]

        C    = engine.angle - g1.angle
        Cdot = engine.speed - g1.speed
        lam  = k * C + c * Cdot
        tot_error += abs(C)
        engine.torque -= lam
        g1.torque += lam
      

    for index1, index2 in outside_conections:
      g_in = gears[index1]
      g_out = gears[index2]

      ratio = g_out.teeth/g_in.teeth
      C    = (1.0 * g_in.angle) + s * (ratio * g_out.angle)
      Cdot = (1.0 * g_in.speed) + s * (ratio * g_out.speed)
      lam  = k * C + c * Cdot
      tot_error += abs(C)
      g_in.torque -= 1.0 * lam
      g_out.torque -= s*ratio * lam

    for index1, index2 in axle_conections:
      g1 = gears[index1]
      g2 = gears[index2]

      C    = g1.angle - g2.angle
      Cdot = g1.speed - g2.speed
      lam  = k * C + c * Cdot
      tot_error += abs(C)
      g1.torque -= lam
      g2.torque += lam

    for clutch in clutches:
      if clutch.engaged == 0:
        clutch.old_engaged = 0
        continue
      g1 = gears[clutch.left_gear_index]
      g2 = gears[clutch.right_gear_index]
      g3 = gears[clutch.right_most_axle_conected_gear_index]
      ratio = 1.0

      if clutch.engaged == -1:
        if clutch.old_engaged != -1:
          clutch.angle_offset = g1.angle - g3.angle
          clutch.old_engaged = -1
        ratio = 1.0
        C    = (g1.angle) - (g3.angle) - clutch.angle_offset
        Cdot = (g1.speed) - (g3.speed)
        lam  = k * C + c * Cdot
        tot_error += abs(C)
        g1.torque -= lam
        g3.torque += lam
      if clutch.engaged == +1:
        if clutch.old_engaged != +1:
          clutch.angle_offset = g2.angle - g3.angle
          clutch.old_engaged = +1
        ratio = 1.0
        C    = (g2.angle) - (g3.angle) - clutch.angle_offset
        Cdot = (g2.speed) - (g3.speed)
        lam  = k * C + c * Cdot
        tot_error += abs(C)
        g2.torque -= lam
        g3.torque += lam


    g1 = gears[-1]

    #C    = g1.angle - sim.front_wheel_angle
    #Cdot = g1.speed - sim.front_wheel_omega
    #lam  = c * Cdot * 2
    #tot_error += abs(C)
    #g1.torque -= lam
    #torque_to_wheels = lam
    #print(lam)
    #sim.set_front_hub_torque(torque_to_wheels)

    for g in gears:
      g.speed += ((g.torque) - g.drag * g.speed) / g.inertia * dt
      g.angle += g.speed * dt
      g.torque = 0.0

    #sim.step(dt)
    
    #Engines[MCI].torque += torque_from_omega(Engines[MCI].speed)*(Engines[MCI].throttle if not Engines[MCI].rev_limit_activated else 0)
    #tau_c = Engines[MCI].tau0 * copysign(1, Engines[MCI].speed) if Engines[MCI].speed != 0 else 0
    #Engines[MCI].speed += (Engines[MCI].torque - tau_c - Engines[MCI].drag*Engines[MCI].speed) / Engines[MCI].inertia * dt
    #if abs(Engines[MCI].speed) < 0.01: Engines[MCI].speed = 0
    #Engines[MCI].torque = 0.0

    for engine in Engines:
      engine.apply_throttle(dt)

    if su == substep:
      print(f"Engine: {Engines[MCI].speed*60/(2*pi):.0f} rpm, Engine2 {Engines[MEP].speed*60/(2*pi):.0f} rpm, Engine3 {Engines[HSG].speed*60/(2*pi):.0f} rpm")
      su = 0
      #print("Total constraint error:", tot_error)
      screen.fill(BACKGROUND_COLOR)
      
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
            idx_c = next(index for index, c in enumerate(clutches) if c.offset == c_offset)
            clutch = clutches[idx_c]
            engagement_offset = clutch.engaged * 30*scaling

            pygame.draw.rect(screen, (150,50,50), (x-15*scaling + engagement_offset, y-15*scaling, 30*scaling, 30*scaling))      

      for g in gears:
        draw_gear(g, screen, module=1.8, scaling=scaling, pan_offset=pan_offset, axle_displacement=axle_displacement)

      draw_rpm_gauge(Engines[MCI], (260/2, 260/2), np.zeros((260, 260, 3), dtype=np.uint8))
        
      pygame.display.flip()
      
      #frame = sim.render(1280, 720)
      #cv2.imshow("CarSim", frame)
      
    #while time.time()-debut < 1/sumulation_frequancy:{}
    
    
    dt = (time.time() - debut)
    while dt < 1/2000:
      dt = (time.time() - debut)
  pygame.quit()

if __name__ == "__main__":
  main()
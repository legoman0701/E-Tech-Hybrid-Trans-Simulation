import time, math, pygame
from engine_curve import torque_from_omega

WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600

class Engine:
  def __init__(self, number_teeth, inertia, name):
    self.teeth = number_teeth
    self.inertia = inertia
    self.name = name
    self.speed = 30
    self.angle = 0
    self.torque = 0
    self.drag = 0.06
    self.tau0 = 10
    self.offset = (0, 0)
    self.throttle = 1
    self.rev_cut = 5800
    self.rev_act = 5600
    self.rev_limit_activated = 0

e = Engine(20, 0.12, "Engine")

dt = 0
debut = time.time()

def draw_rpm_gauge(e, pos):
  font = pygame.font.SysFont('Arial', 30)
  #pygame.draw.circle(screen, (255, 0, 0), (WIDTH/2, HEIGHT/2), 100, 5)
  
  pygame.draw.arc(screen, (255, 255, 255), (WIDTH/2-100, HEIGHT/2-100, 200, 200), 0, math.pi*1.25, 5)
  pygame.draw.arc(screen, (255, 0, 0), (WIDTH/2-100, HEIGHT/2-100, 200, 200), 0, math.pi*0.25, 5)

  for i, number in enumerate(displayed_numbers):
    alpha = -i/6*math.pi*1.25
    text_surface = font.render(str(number), True, (255, 255, 255))
    offset = (math.cos(alpha)*120, math.sin(alpha)*120)
    text_rect = text_surface.get_rect(center=(pos[0] + offset[0], pos[1] + offset[1]))
    screen.blit(text_surface, text_rect)
  
  alpha = e.speed*60/(2*math.pi)
  alpha = alpha/1000/6
  alpha *= math.pi*1.25
  alpha -= math.pi*1.25
  
  offset = (math.cos(alpha)*90, math.sin(alpha)*90)
  pygame.draw.line(screen, (255, 255, 255), pos, (pos[0]+offset[0], pos[1]+offset[1]), 3)

pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Transmission Simulation")
WIDTH, HEIGHT = 800, 600

displayed_numbers = [0, 1, 2, 3, 4, 5, 6]
displayed_numbers.reverse()
while True:
  dt = time.time()-debut
  debut = time.time()

  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      break
  
  keys = pygame.key.get_pressed()
  if keys[pygame.K_UP]:
    e.throttle = min(1, e.throttle + 0.01)
  else:
    e.throttle = max(0, e.throttle - 0.01)

  if e.speed*60/(2*math.pi) > e.rev_cut: e.rev_limit_activated = True
  elif e.speed*60/(2*math.pi) < e.rev_act and e.rev_limit_activated: e.rev_limit_activated = False

  e.torque = torque_from_omega(e.speed)*(e.throttle if not e.rev_limit_activated else 0)
  tau_c = e.tau0 * math.copysign(1, e.speed) if e.speed != 0 else 0
  e.speed += (e.torque - tau_c - e.drag*e.speed) / e.inertia * dt
  if abs(e.speed) < 0.01: e.speed = 0

  #print speed in rpm

  screen.fill((30, 30, 30))
  draw_rpm_gauge(e, (WIDTH/2, HEIGHT/2))
  pygame.display.flip()
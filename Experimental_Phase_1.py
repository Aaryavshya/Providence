import pygame
import math
import time

pygame.init()

# Color Gamut
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (100, 149, 237)
PINK = (128,196,169)
RED = (188, 39, 50)
DARK_GREY = (80, 78, 81)

# Font Defination
TextFont = pygame.font.Font('Pixeltype.ttf', 25)

# Screen Information
WIDTH, HEIGHT = 1440, 950
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Planet Simulation")


## Instructions

# Velocity Control
invel = TextFont.render("Planetary Velocity: +/- .1 x initial", False, (128,196,169))
invel_rect = invel.get_rect(center = (1300,50))

# Solar Mass Control
insm = TextFont.render("Solar Movement", False, (128,196,169))
insm_rect = insm.get_rect(center = (1280,750))

# Main Panel
inmain = TextFont.render("Main Panel", False, (128,196,169))
inmain_rect = inmain.get_rect(center = (100,20))

# General Information
gen = TextFont.render("Inner Solar System: Newtonian and Keplerian Planetary Motion Model", False, (128,196,169))
gen_rect = gen.get_rect(center = (350,900))

## Buttons

# button for movement of sun
button_sun_movement_xp = pygame.image.load('functions/right.png')
button_sun_movement_xp = pygame.transform.scale_by(button_sun_movement_xp,.12)
button_sun_movement_rect_xp = button_sun_movement_xp.get_rect(center = (1350,850))

button_sun_movement_yp = pygame.image.load('functions/up.png')
button_sun_movement_yp = pygame.transform.scale_by(button_sun_movement_yp,.12)
button_sun_movement_rect_yp = button_sun_movement_yp.get_rect(center = (1275,810))

button_sun_movement_xn = pygame.image.load('functions/left.png')
button_sun_movement_xn = pygame.transform.scale_by(button_sun_movement_xn,.12)
button_sun_movement_rect_xn = button_sun_movement_xn.get_rect(center = (1200,850))

button_sun_movement_yn = pygame.image.load('functions/down.png')
button_sun_movement_yn = pygame.transform.scale_by(button_sun_movement_yn,.12)
button_sun_movement_rect_yn = button_sun_movement_yn.get_rect(center = (1275,850))

# button for Solar Mass control Module

solar_mass = pygame.image.load('functions/solarmass.png')
solar_mass = pygame.transform.scale_by(solar_mass,.125)
solar_mass_rect = solar_mass.get_rect(center = (50,50))

solar_mass_0 = pygame.image.load('functions/solarmass.png')
solar_mass_0 = pygame.transform.scale_by(solar_mass_0,.125)
solar_mass_rect_0 = solar_mass.get_rect(center = (50,850))

button_sun_mass_in = pygame.image.load('functions/increment.png')
button_sun_mass_in = pygame.transform.scale_by(button_sun_mass_in,.125)
button_sun_mass_rect_in = button_sun_mass_in.get_rect(center = (50,90))

button_sun_mass_de = pygame.image.load('functions/decrement.png')
button_sun_mass_de = pygame.transform.scale_by(button_sun_mass_de,.125)
button_sun_mass_rect_de = button_sun_mass_de.get_rect(center = (50,125))

# Planetary demarcation Module

# Mercury
mer = pygame.image.load('functions/mercury.png')
mer = pygame.transform.scale_by(mer,.125)
mer_rect = mer.get_rect(center = (1250,100))

# Venus
ven = pygame.image.load('functions/venus.png')
ven = pygame.transform.scale_by(ven,.125)
ven_rect = mer.get_rect(center = (1250,150))

# Earth
ert = pygame.image.load('functions/earth.png')
ert = pygame.transform.scale_by(ert,.125)
ert_rect = mer.get_rect(center = (1250,200))

# Mars
mrs = pygame.image.load('functions/mars.png')
mrs = pygame.transform.scale_by(mrs,.125)
mrs_rect = mrs.get_rect(center = (1250,250))

# Vulkan
vul = pygame.image.load('functions/vulkansim.png')
vul = pygame.transform.scale_by(vul,.125)
vul_rect = vul.get_rect(center = (1250,300))

# Planetary Velocity control Module

# mercury
mer_in = pygame.image.load('functions/increment.png')
mer_in = pygame.transform.scale_by(mer_in,.125)
mer_rect_in = mer_in.get_rect(center = (1325,100))

mer_de = pygame.image.load('functions/decrement.png')
mer_de = pygame.transform.scale_by(mer_de,.125)
mer_rect_de = mer_de.get_rect(center = (1375,100))

# venus
ven_in = pygame.image.load('functions/increment.png')
ven_in = pygame.transform.scale_by(ven_in,.125)
ven_rect_in = ven_in.get_rect(center = (1325,150))

ven_de = pygame.image.load('functions/decrement.png')
ven_de = pygame.transform.scale_by(ven_de,.125)
ven_rect_de = ven_de.get_rect(center = (1375,150))

# earth
ert_in = pygame.image.load('functions/increment.png')
ert_in = pygame.transform.scale_by(ert_in,.125)
ert_rect_in = ert_in.get_rect(center = (1325,200))

ert_de = pygame.image.load('functions/decrement.png')
ert_de = pygame.transform.scale_by(ert_de,.125)
ert_rect_de = ert_de.get_rect(center = (1375,200))

# mars
mrs_in = pygame.image.load('functions/increment.png')
mrs_in = pygame.transform.scale_by(mrs_in,.125)
mrs_rect_in = mrs_in.get_rect(center = (1325,250))

mrs_de = pygame.image.load('functions/decrement.png')
mrs_de = pygame.transform.scale_by(mrs_de,.125)
mrs_rect_de = mrs_de.get_rect(center = (1375,250))

# vulkan
vul_in = pygame.image.load('functions/increment.png')
vul_in = pygame.transform.scale_by(vul_in,.125)
vul_rect_in = vul_in.get_rect(center = (1325,300))

vul_de = pygame.image.load('functions/decrement.png')
vul_de = pygame.transform.scale_by(vul_de,.125)
vul_rect_de = vul_de.get_rect(center = (1375,300))

# button for adding Vulkan
button_Vulkan = pygame.image.load('functions/vulkansim.png')
button_Vulkan = pygame.transform.scale_by(button_Vulkan,.125)
button_Vulkan_rect = button_Vulkan.get_rect(center = (50,200))

# button for stopping the animation
button_halt = pygame.image.load('functions/pausesim.png')
button_halt = pygame.transform.scale_by(button_halt,.125)
button_halt_rect = button_halt.get_rect(center = (150,50))

# button for restart 
button_restart = pygame.image.load('functions/startsim.png')
button_restart = pygame.transform.scale_by(button_restart,.125)
button_restart_rect = button_restart.get_rect(center = (150,100))

# button for getting distance
button_get_distance = pygame.image.load('functions/get_distance.png')
button_get_distance = pygame.transform.scale_by(button_get_distance,.125)
button_get_distance_rect = button_restart.get_rect(center = (150,150))


class Planet:

    AU = 149.6e6 * 1000
    G = 6.67428e-11
    SCALE = 250/AU
    TIMESTEP = 3600*24

    def __init__(self, x, y, radius, color, mass):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.mass = mass

        self.orbit = []
        self.sun = False
        self.distance_to_sun = 0

        self.x_vel = 0
        self.y_vel = 0

    def draw(self, win):
        x = self.x * self.SCALE + WIDTH/2
        y = self.y * self.SCALE + HEIGHT/2

        if len(self.orbit) > 2:
            updated_points = []
            for point in self.orbit:
                x,y = point
                x = x*self.SCALE + WIDTH/2
                y = y*self.SCALE + HEIGHT/2
                updated_points.append((x,y))
            pygame.draw.lines(win,self.color, False, updated_points, 2)

        pygame.draw.circle(win, self.color, (x,y), self.radius)

    def get_distance(self):
        x = self.x * self.SCALE + WIDTH/2
        y = self.y * self.SCALE + HEIGHT/2
        if not self.sun:
            Radial_distance = TextFont.render(f'{round(self.distance_to_sun/1000,1)}km',1,WHITE)
            WIN.blit(Radial_distance,(x - Radial_distance.get_width()/2, y - Radial_distance.get_height()/2))
    
    def attraction(self, other):
        other_x, other_y = other.x, other.y
        distance_x = other_x - self.x
        distance_y = other_y - self.y
        distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

        if other.sun:
            self.distance_to_sun = distance

        force = self.G * self.mass * other.mass / distance**2
        theta = math.atan2(distance_y,distance_x)
        force_x = math.cos(theta) * force
        force_y = math.sin(theta) * force
        return force_x, force_y
    
    def update_position(self, planets):
        total_fx = total_fy = 0
        
        for planet in planets:
            if self == planet:
                continue

            fx, fy = self.attraction(planet)
            total_fx += fx
            total_fy += fy

        self.x_vel += total_fx  / self.mass  * self.TIMESTEP
        self.y_vel += total_fy  / self.mass  * self.TIMESTEP

        self.x += self.x_vel * self.TIMESTEP 
        self.y += self.y_vel * self.TIMESTEP 

        self.orbit.append((self.x, self.y))
        if len(self.orbit) > 10000:
            for i in range(1000):
                del self.orbit[i] 


def main():

    # Primary Logic Control
    game_active = True
    run = True

    # Framerate Setting
    clock = pygame.time.Clock()

    # Solar System Information
    sun = Planet(0,0, 30, YELLOW, 1.98892*10**30)
    sun.x_vel = 0
    sun.sun = True

    Vulkan = Planet(.28*Planet.AU, 0, 4, PINK , 2.8*10**23)
    Vulkan.y_vel = -56890
    
    Mercury = Planet(-.387*Planet.AU, 0, 7,DARK_GREY, 3.30*10**23)
    Mercury.y_vel = 47400

    Venus = Planet(-.723 * Planet.AU, 0 ,14, WHITE, 4.8685 * 10**24)
    Venus.y_vel = 35020

    earth = Planet(-1 * Planet.AU, 0, 16, BLUE, 5.9742*10**24)
    earth.y_vel = 29783

    Mars = Planet(-1.524 * Planet.AU, 0, 11, RED, 6.39 * 10**23)
    Mars.y_vel = 24077

    # Additional Functionalities: Extra Planets
    Jupiter = Planet(-5.2*Planet.AU,0,10,(255,165,0),1.898*10**27)
    Jupiter.y_vel = 13024

    Saturn = Planet(-9.5*Planet.AU,0,9,(196,184,84),5.68*10**26)
    Saturn.y_vel = 9678

    Uranus = Planet(-19.8*Planet.AU,0,8,(120,178,250),8.68*10**25)
    Uranus.y_vel = 5600

    # Primary display List
    planets = [sun, earth, Mars, Mercury, Venus]

    # Press counter
    press_counter = 0

    while run:

        clock.tick(60)
        WIN.fill((0,0,0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        ## button Functionality
        mouse_pos = pygame.mouse.get_pos()
        if (button_restart_rect.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN):
            game_active = True
        
        # Solar mass 
        if button_sun_mass_rect_in.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
            press_counter = 1
            if press_counter >= 1:
                press_counter += 1
                sun.mass = press_counter*sun.mass*.01 + sun.mass
        
        if button_sun_mass_rect_de.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
            press_counter = 1
            if press_counter >= 1:
                press_counter += 1
                sun.mass = sun.mass - press_counter*sun.mass*.01

        # Mercury y_vel 
        if mer_rect_in.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
            press_counter = 1
            if press_counter >= 1:
                press_counter += 1
                Mercury.y_vel = Mercury.y_vel + Mercury.y_vel*.1
        
        if mer_rect_de.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
            press_counter = 1
            if press_counter >= 1:
                press_counter += 1
                Mercury.y_vel = Mercury.y_vel - Mercury.y_vel*.1

        # venus y_vel
        if ven_rect_in.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
            press_counter = 1
            if press_counter >= 1:
                press_counter += 1
                Venus.y_vel = Venus.y_vel + Venus.y_vel*.1

        if ven_rect_de.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
            press_counter = 1
            if press_counter >= 1:
                press_counter += 1
                Venus.y_vel = Venus.y_vel - Venus.y_vel*.1

        # earth y_vel
        if ert_rect_in.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
            press_counter = 1
            if press_counter >= 1:
                press_counter += 1
                earth.y_vel = earth.y_vel + earth.y_vel*.1

        if ert_rect_de.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
            press_counter = 1
            if press_counter >= 1:
                press_counter += 1
                earth.y_vel = earth.y_vel - earth.y_vel*.1

        # Mars y_vel
        if mrs_rect_in.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
            press_counter = 1
            if press_counter >= 1:
                press_counter += 1
                Mars.y_vel = Mars.y_vel + Mars.y_vel*.1

        if mrs_rect_de.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
            press_counter = 1
            if press_counter >= 1:
                press_counter += 1
                Mars.y_vel = Mars.y_vel - Mars.y_vel*.1

        # vulkan y_vel
        if vul_rect_in.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
            press_counter = 1
            if press_counter >= 1:
                press_counter += 1
                Vulkan.y_vel = Vulkan.y_vel + Vulkan.y_vel*.1
        
        if vul_rect_de.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
            press_counter = 1
            if press_counter >= 1:
                press_counter += 1
                Vulkan.y_vel = Vulkan.y_vel - Vulkan.y_vel*.1
        
        if solar_mass_rect_0.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
            sun.mass = .000001
            sun.radius = 0

        # Primary Panel Control
        if game_active:

            # Sun movement delay
            if sun.x > 1800 or sun.x < -100:
                sun.x_vel = 0

            # Solar Movement
            if button_sun_movement_rect_xp.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
                sun.x_vel = 10000

            if sun.y > 1200 or sun.y < 0 :
                sun.y_vel = 0
            if button_sun_movement_rect_xn.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
                sun.x_vel = -10000
            
            if button_sun_movement_rect_yp.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
                sun.y_vel = -10000
            
            if button_sun_movement_rect_yn.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
                sun.y_vel = 10000
                
            # Vulkan 
            if button_Vulkan_rect.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
                if press_counter == 0:
                    planets.append(Vulkan)
                    press_counter += 1

            # Pause
            if (button_halt_rect.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN):
                game_active = False

            # Distance
            if (button_get_distance_rect.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN):
                for planet in planets:
                    planet.get_distance()                     
            
            # Primary Display function
            for planet in planets:
                planet.update_position(planets)
                planet.draw(WIN)

            # Button Display 
            WIN.blit(button_get_distance,button_get_distance_rect)
            WIN.blit(button_sun_mass_in,button_sun_mass_rect_in)
            WIN.blit(button_sun_mass_de,button_sun_mass_rect_de)        
            WIN.blit(button_sun_movement_xp,button_sun_movement_rect_xp)
            WIN.blit(button_sun_movement_xn,button_sun_movement_rect_xn)
            WIN.blit(button_sun_movement_yp,button_sun_movement_rect_yp)
            WIN.blit(button_sun_movement_yn,button_sun_movement_rect_yn)
            WIN.blit(button_Vulkan,button_Vulkan_rect)
            WIN.blit(button_halt,button_halt_rect)
            WIN.blit(button_restart,button_restart_rect)

            # Slider Display
            WIN.blit(mer_in,mer_rect_in)
            WIN.blit(mer_de,mer_rect_de)
            WIN.blit(ven_in,ven_rect_in)
            WIN.blit(ven_de,ven_rect_de)
            WIN.blit(ert_in,ert_rect_in)
            WIN.blit(ert_de,ert_rect_de)
            WIN.blit(mrs_in,mrs_rect_in)
            WIN.blit(mrs_de,mrs_rect_de)
            WIN.blit(vul_in,vul_rect_in)
            WIN.blit(vul_de,vul_rect_de)

            WIN.blit(mer,mer_rect)
            WIN.blit(ven,ven_rect)
            WIN.blit(ert,ert_rect)
            WIN.blit(mrs,mrs_rect)
            WIN.blit(vul,vul_rect)
            WIN.blit(solar_mass,solar_mass_rect)
            WIN.blit(solar_mass_0,solar_mass_rect_0)

            # Instructions Display
            WIN.blit(invel,invel_rect)
            WIN.blit(insm,insm_rect)
            WIN.blit(inmain,inmain_rect)
            WIN.blit(gen,gen_rect)

            pygame.display.update()
   
    pygame.quit()

main()
import pygame
from display import Display

class Unit:
    def __init__(self, r, c, image_path, ranged_strength, range_radius, combat_strength,) -> None:

        self.r = r
        self.c = c

        self.image_path = image_path

        image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(image, Display.DEFAULT_IMAGE_SIZE)

        self.ranged_strength = ranged_strength
        self.range_radius = range_radius
    
        self.combat_strength = combat_strength
      
        self.hp = 50


        self.selected = False



    def update():
        ...#.blit(carImg, (x,y))

class Units:
    Tank = lambda r, c: Unit(r, c, image_path='assets/units/tank.png', ranged_strength=0, range_radius=0, combat_strength=50)
    Artillery = lambda r, c: Unit(r, c, image_path='assets/units/artillery.png', ranged_strength=70, range_radius=2, combat_strength=15)

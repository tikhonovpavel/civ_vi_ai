import pygame
import random
from display import DEFAULT_UNIT_IMAGE_SIZE, UI_UNIT_IMAGE_SIZE



class UnitState:
    DEFAULT = 0
    MOVING = 1
    DEFENCE = 2
    
class UnitCategories:
    GREAT_PERSON = 1
    CITIZEN = 2
    MILITARY = 3


class Unit:
    def __init__(self, name, r, c, category, image_path, mp_base, ranged_strength_base, range_radius_base,
                 combat_strength_base, modifiers=None) -> None:

        self.name = name

        self.r = r
        self.c = c

        self.category = category

        self.image_path = image_path

        image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(image, DEFAULT_UNIT_IMAGE_SIZE)
        self.image_ui = pygame.transform.scale(image, UI_UNIT_IMAGE_SIZE)

        self._ranged_strength = ranged_strength_base
        self._range_radius = range_radius_base
    
        self._combat_strength_base = combat_strength_base

        self.hp = random.randint(1, 100)
        self.mp = mp_base

        self.modifiers = modifiers

        self.is_selected = False

        self.state = UnitState.DEFAULT

        # self.moving_to = (None, None)
        self.path = []

        # self.allowed_hexes = None

    def calc_combat_strength(self, ):
        return self._combat_strength_base  # + modifiers

    def update():
        ...#.blit(carImg, (x,y))

class Units:
    Tank = lambda r, c: Unit('tank', r, c, category=UnitCategories.MILITARY, image_path='assets/units/tank.png',
                             mp_base=4,
                             ranged_strength_base=0, range_radius_base=0, combat_strength_base=50)
    
    Artillery = lambda r, c: Unit('artillery', r, c, category=UnitCategories.MILITARY, image_path='assets/units/artillery.png',
                                  mp_base=3,
                                  ranged_strength_base=70, range_radius_base=2, combat_strength_base=15)

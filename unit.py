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


DEFAULT_TILE_COLOR = (0, 255, 0)
SELECTED_TILE_COLOR = (0, 128, 0)
DEFAULT_NATION_IMAGE_SIZE = (15, 10)
UI_NATION_IMAGE_SIZE = (50, 50)


def get_nation_icon(nation):
    return f'assets/nations/{nation}.png'

nation_icons = dict()
for nation in ['rome', 'egypt']:
    image = pygame.image.load(get_nation_icon(nation))
    nation_icons[nation] = {'default': pygame.transform.scale(image, DEFAULT_NATION_IMAGE_SIZE),
                            'ui': pygame.transform.scale(image, UI_NATION_IMAGE_SIZE)}

class Unit:
    def __init__(self, name, player, r, c, tile, category, image_path, mp_base, ranged_strength_base, range_radius_base,
                 combat_strength_base, modifiers=None) -> None:

        self.name = name

        self.player = player

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

        self.hp = random.randint(50, 100)
        self.mp = mp_base

        self.modifiers = modifiers

        self.is_selected = False

        self.state = UnitState.DEFAULT

        self.path = []

    def calc_combat_strength(self, ):
        return self._combat_strength_base  # + modifiers

    def draw(self, screen, game):
        hex = game.map.get(self.r, self.c).hex

        if self.is_selected:
            pygame.draw.polygon(screen, SELECTED_TILE_COLOR, hex.points, 0)
            pygame.draw.polygon(screen, (0, 0, 0), hex.points, 1)

        nation_image = nation_icons[self.player.nation]['default']
        screen.blit(self.image,
                    (hex.x - self.image.get_width() / 2, hex.y - self.image.get_height() / 2))
        screen.blit(nation_image, (hex.x - nation_image.get_width() / 2, hex.y + 7))

        hp_offset = 15
        hp_length = 16
        hp_thickness = 5

        pygame.draw.rect(
            screen,
            (255, 0, 0),
            (hex.x - hp_length / 2, hex.y - hp_offset, hp_length * (self.hp / 100), hp_thickness), )

        pygame.draw.rect(
            screen,
            (0, 0, 0),
            (hex.x - hp_length / 2, hex.y - hp_offset, hp_length, hp_thickness),
            width=1)

class Units:
    Tank = lambda player, r, c, tile: Unit('tank', player, r, c, tile,
                                           category=UnitCategories.MILITARY, image_path='assets/units/tank.png',
                                           mp_base=4, ranged_strength_base=0, range_radius_base=0,
                                           combat_strength_base=50)

    Artillery = lambda player, r, c, tile: Unit('artillery', player, r, c, tile,
                                                category=UnitCategories.MILITARY, 
                                                image_path='assets/units/artillery.png', mp_base=3, 
                                                ranged_strength_base=70, range_radius_base=2, combat_strength_base=15)

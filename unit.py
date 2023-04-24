import math

import pygame
import random
from display import DEFAULT_UNIT_IMAGE_SIZE, UI_UNIT_IMAGE_SIZE

from typing import Tuple, List

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
        self.mp_base = mp_base

        self.modifiers = modifiers

        self.is_selected = False

        self.state = UnitState.DEFAULT

        self.path = []

        self.can_attack = True

    @staticmethod
    def compute_combat_damage(unit1, unit2):
        diff = unit1.calc_combat_strength() - unit2.calc_combat_strength()

        return random.uniform(0.8, 1.2) * 30 * math.exp(diff / 25)
        # return 30 * math.exp(diff / 25 * random.uniform(0.75, 1.25))

    def move(self, game):
        print(f'unit on {(self.r, self.c)}', '\n', self._get_available_path_coords(game), '\n')

        avail_path_coords = self._get_available_path_coords(game)

        if len(avail_path_coords) == 0:
            return

        coord, mp_spent, is_attack = avail_path_coords[-1]
        new_r, new_c = coord

        if is_attack:
            enemy_unit = game.map.get(new_r, new_c).tile['unit']

            enemy_unit_damage = self.compute_combat_damage(self, enemy_unit)
            unit_damage = self.compute_combat_damage(enemy_unit, self)

            print(
                f"{self.player.nation}'s {self.name} hp: {self.hp}, damage: {unit_damage}")
            print(f"{enemy_unit.player.nation}'s {self.name} hp: {enemy_unit.hp}, damage: {enemy_unit_damage}")
            # print()

            unit_r, unit_c = self.r, self.c
            enemy_unit_r, enemy_unit_c = enemy_unit.r, enemy_unit.c

            game.display.show_damage_text(f'-{min(100, int(unit_damage))}', unit_r, unit_c)
            game.display.show_damage_text(f'-{min(100, int(enemy_unit_damage))}', enemy_unit_r, enemy_unit_c)

            if self.hp - unit_damage <= 0:
                self.player.units.remove(self)
                game.map.set_data(self.r, self.c, 'unit', None)

                enemy_unit.hp = max(1, enemy_unit.hp - enemy_unit_damage)

                # return True
            elif enemy_unit.hp - enemy_unit_damage <= 0:
                enemy_unit.player.units.remove(enemy_unit)  # del enemy_unit
                self.hp -= unit_damage

                # result = self.move_unit_one_step(unit)

                # self.r, self.c = enemy_unit.r, enemy_unit.c  # or just r and c
                self.move_unconditionaly(game, enemy_unit.r, enemy_unit.c)

                self.mp -= 1  # -1 to MP because of the attack

            else:
                self.hp -= unit_damage
                enemy_unit.hp -= enemy_unit_damage


                # -1 to MP because of the attack
                self.mp -= 1  # self.selected = False

                if len(avail_path_coords) >= 2:
                    # if avail_path_coords[-1]'s is_attack is True then avail_path_coords[-2][0] will be adjacent,
                    # so it guarantied to be legal move
                    self.move_unconditionaly(game, *avail_path_coords[-2][0])

            self.can_attack = False
            self.path = []
        else:
            self.move_unconditionaly(game, new_r, new_c)
            self.mp = max(0, self.mp - mp_spent)
            # self.path.pop(0)

            self.path = self.path[self.path.index(coord):]


        game.update()

        # self.move_unconditionaly(game, coord)

        # while not done:
        #      = self._move_one_step(game)
        #     game.update()

    def _get_available_path_coords(self, game) -> List[Tuple[Tuple, int, bool]]:
        '''
        is_attack can be true only if it is the last one

        :param game:
        :return: [(coord1, mp_spent1, is_attack1), ...]
        '''
        # result = None
        result = []#[((self.r, self.c), 0, False)]

        if len(self.path) == 0:
            return result

        mp_left = self.mp
        path_coord_prev = self.path[0]

        for i, path_coord in enumerate(self.path[1:]):
            try:
                move_cost = game.map.get_data_edge(path_coord_prev, path_coord)['cost']
            except Exception as err:
                raise err

            if mp_left < move_cost:
                return result

            new_tile_unit = game.map.get(*path_coord).tile['unit']
            try:
                if new_tile_unit is not None:
                    if game.is_enemy(new_tile_unit.player) and self.can_attack:  # it's the last available state

                        result.append((path_coord, self.mp - mp_left + move_cost, True))
                        return result

                    else:  # skip the tile
                        mp_left -= move_cost
                        path_coord_prev = path_coord

                        continue
            except Exception as err:
                raise err

            result.append((path_coord, self.mp - mp_left + move_cost, False))

            mp_left -= move_cost
            path_coord_prev = path_coord

        return result

    # def _move_one_step(self, game) -> Tuple[Tuple, int, bool]:
    #     # self.player = self.get_self.player()
    #
    #     # assert player in self.players
    #     assert game.map.get(self.r, self.c).tile['unit'] == self
    #     # assert self.mp > 0
    #
    #     if len(self.path) < 2:
    #         self.path = []
    #         return True
    #
    #     if self.mp == 0:
    #         return True
    #
    #     new_r, new_c = self.path[1]
    #
    #     new_tile_unit = game.map.get(new_r, new_c).tile['unit']
    #     if new_tile_unit is not None and not game.is_enemy(new_tile_unit.player):
    #         return True
    #
    #     cost = game.map.get_data_edge((self.r, self.c), (new_r, new_c))['cost']
    #
    #     if self.mp < cost:
    #         return True
    #
    #     if new_tile_unit is not None and game.is_enemy(new_tile_unit.player):  # attack
    #         if not self.can_attack:
    #             return True
    #
    #         enemy_unit = new_tile_unit
    #
    #         enemy_unit_damage = self.compute_combat_damage(self, enemy_unit)
    #         unit_damage = self.compute_combat_damage(enemy_unit, self)
    #
    #         print(
    #             f"{self.player.nation}'s {self.name} hp: {self.hp}, damage: {unit_damage}")
    #         print(f"{enemy_unit.player.nation}'s {self.name} hp: {enemy_unit.hp}, damage: {enemy_unit_damage}")
    #         # print()
    #
    #         unit_r, unit_c = self.r, self.c
    #         enemy_unit_r, enemy_unit_c = enemy_unit.r, enemy_unit.c
    #
    #         game.display.show_damage_text(f'-{min(100, int(unit_damage))}', unit_r, unit_c)
    #         game.display.show_damage_text(f'-{min(100, int(enemy_unit_damage))}', enemy_unit_r, enemy_unit_c)
    #
    #         if self.hp - unit_damage <= 0:
    #             self.player.units.remove(self)
    #             game.map.set_data(self.r, self.c, 'unit', None)
    #
    #             enemy_unit.hp = max(1, enemy_unit.hp - enemy_unit_damage)
    #
    #             return True
    #         elif enemy_unit.hp - enemy_unit_damage <= 0:
    #             enemy_unit.player.units.remove(enemy_unit)  # del enemy_unit
    #             self.hp -= unit_damage
    #
    #             # result = self.move_unit_one_step(unit)
    #
    #             # self.r, self.c = enemy_unit.r, enemy_unit.c  # or just r and c
    #             self.move_unconditionaly(game, enemy_unit.r, enemy_unit.c)
    #
    #             self.mp -= 1  # -1 to MP because of the attack
    #
    #         else:
    #             self.hp -= unit_damage
    #             enemy_unit.hp -= enemy_unit_damage
    #
    #             self.move_unconditionaly(game, *self.path[-2])
    #             # self.r, self.c = self.path[-2]
    #
    #             # -1 to MP because of the attack
    #
    #             self.mp -= 1  # self.selected = False
    #
    #         self.can_attack = False
    #         self.path = []
    #
    #         return True
    #     else:  # just move
    #         self.move_unconditionaly(game, new_r, new_c)
    #         self.mp = max(0, self.mp - cost)
    #         self.path.pop(0)
    #
    #         return False

    def move_unconditionaly(self, game, new_r, new_c):
        game.map.set_data(self.r, self.c, 'unit', None)
        game.map.set_data(new_r, new_c, 'unit', self)

        self.r = new_r
        self.c = new_c



    def calc_combat_strength(self, ):
        return self._combat_strength_base  # + modifiers

    def draw(self, screen, game):
        # print(self.r, self.c)
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

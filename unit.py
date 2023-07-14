import math

import pygame
import random

from city import City
from game_object import MilitaryObject

random.seed(1)

from consts import DEFAULT_UNIT_IMAGE_SIZE, UI_UNIT_IMAGE_SIZE

from typing import Tuple, List


class UnitState:
    DEFAULT = 0
    MOVING = 1
    DEFENCE = 2


# class UnitCategories:
#     CITY = 0
#     GREAT_PERSON = 1
#     CITIZEN = 2
#     MILITARY = 3


DEFAULT_TILE_COLOR = (0, 255, 0)
SELECTED_TILE_COLOR = (0, 128, 0)
DEFAULT_NATION_IMAGE_SIZE = (15, 10)
UI_NATION_IMAGE_SIZE = (50, 50)


def get_nation_icon(nation):
    return f'assets/nations/{nation}.png'


nation_icons = dict()
for player_nation in ['Rome', 'Egypt']:
    icon = pygame.image.load(get_nation_icon(player_nation))
    nation_icons[player_nation] = {'default': pygame.transform.scale(icon, DEFAULT_NATION_IMAGE_SIZE),
                                   'ui': pygame.transform.scale(icon, UI_NATION_IMAGE_SIZE)}


class Unit(MilitaryObject):
    def __init__(self, name, player, r, c, category, image_path,
                 mp_base, combat_strength_base, ranged_strength_base, range_radius_base,
                 modifiers=None, sound_attack=None, sound_movement=None) -> None:

        super().__init__(name, player, r, c, category, mp_base, combat_strength_base,
                         ranged_strength_base=ranged_strength_base,
                         range_radius_base=range_radius_base, modifiers=modifiers,
                         sound_attack=sound_attack, sound_movement=sound_movement)
        self.image_path = image_path

        image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(image, DEFAULT_UNIT_IMAGE_SIZE)
        self.image_ui = pygame.transform.scale(image, UI_UNIT_IMAGE_SIZE)

        self.state = UnitState.DEFAULT

        # self.sound_attack = pygame.mixer.Sound(sound_attack) if sound_attack else None
        # self.sound_movement = pygame.mixer.Sound(sound_movement) if sound_movement else None

    def combat_attack(self, game, enemy_r, enemy_c):

        enemy_obj = next(iter(game.map.get(enemy_r, enemy_c).game_objects), None)

        enemy_unit_damage = self.compute_combat_damage(self, enemy_obj)
        unit_damage = self.compute_combat_damage(enemy_obj, self)

        print(f"{self.player.nation}'s {self.name} on {self.r, self.c} => "
              f"{enemy_obj.player.nation}'s {enemy_obj.name} on {enemy_obj.r, enemy_obj.c}. "      
              f"HP: ({self.hp} -> {max(0, self.hp - unit_damage)}) / "
              f"({enemy_obj.hp} -> {max(0, enemy_obj.hp - enemy_unit_damage)})")

        unit_r, unit_c = self.r, self.c
        enemy_unit_r, enemy_unit_c = enemy_obj.r, enemy_obj.c

        if game.sound_marker.state and self.sound_attack:
            self.sound_attack.play()

        game.display.show_damage_text(f'-{min(100, int(unit_damage))}', unit_r, unit_c)
        game.display.show_damage_text(f'-{min(100, int(enemy_unit_damage))}', enemy_unit_r, enemy_unit_c)

        if enemy_obj.hp - enemy_unit_damage <= 0:

            if isinstance(enemy_obj, City):
                for u in game.map.get(enemy_obj.r, enemy_obj.c).game_objects:
                    if isinstance(u, Unit):
                        game.map.get(u.r, u.c).game_objects.remove(u)
                        enemy_obj.player.destroy(u)

                enemy_obj.change_ownership(self.player)

            else:
                enemy_obj.player.destroy(enemy_obj)  # del enemy_unit

            self.hp = max(1, self.hp - unit_damage)
            self.move_unconditionaly(game, enemy_obj.r, enemy_obj.c)

            self.mp -= 1  # -1 to MP because of the attack

        elif self.hp - unit_damage <= 0:

            self.player.destroy(self)
            game.map.remove(self.r, self.c, self)

            enemy_obj.hp = max(1, enemy_obj.hp - enemy_unit_damage)

        else:
            self.hp -= unit_damage
            enemy_obj.hp -= enemy_unit_damage

            # -1 to MP because of the attack
            self.mp -= 1  # self.selected = False

            avail_path_coords = self._get_available_path_coords(game)
            if len(avail_path_coords) >= 2:
                # if avail_path_coords[-1]'s is_attack is True then avail_path_coords[-2][0] will be adjacent,
                # so it guarantied to be legal move
                self.move_unconditionaly(game, *avail_path_coords[-2][0])

        self.can_attack = False
        self.path = []

        enemy_obj.path = []

    def ranged_attack(self, game, enemy_r, enemy_c):
        enemy_unit = next(iter(game.map.get(enemy_r, enemy_c).game_objects), None)
        enemy_unit_damage = MilitaryObject.compute_ranged_damage(self, enemy_unit)

        print(f"{self.player.nation}'s {self.name} on {self.r, self.c} => "
              f"{enemy_unit.player.nation}'s {enemy_unit.name} on {enemy_unit.r, enemy_unit.c}. "      
              f"HP: ({self.hp} -> {max(0, self.hp)}) / "
              f"({enemy_unit.hp} -> {max(0, enemy_unit.hp - enemy_unit_damage)})")

        if game.sound_marker.state and self.sound_attack:
            self.sound_attack.play(maxtime=1500, fade_ms=500)

        game.display.show_damage_text(f'-{min(100, int(enemy_unit_damage))}', enemy_unit.r, enemy_unit.c)

        if enemy_unit.hp - enemy_unit_damage <= 0:
            if not isinstance(enemy_unit, City):
                enemy_unit.player.destroy(enemy_unit)

                game.map.reset(enemy_unit.r, enemy_unit.c)
                # game.map.set(enemy_unit.r, enemy_unit.c, [])
            else:
                enemy_unit.hp = 0
        else:
            enemy_unit.hp -= enemy_unit_damage

        self.mp = 0
        self.path = []
        self.can_attack = False

    def gain_hps(self):
        if self.mp == self.mp_base:
            if any(city.is_cell_inside(self.r, self.c) for city in self.player.cities):
                self.hp = min(100, self.hp + 10)

    def move(self, game):
        # if there is just a transition without an attack - the logic is the same for any type of unit
        if len(self.path) == 0 and self.get_ranged_target(game) is None:
            return

        # check if ranged unit inside the attack radius
        ranged_target = self.get_ranged_target(game)
        if ranged_target is not None:
            self.ranged_attack(game, ranged_target.r, ranged_target.c)
        else:
            avail_path_coords = self._get_available_path_coords(game)
            if len(avail_path_coords) == 0:
                return

            coord, mp_spent, is_attack = avail_path_coords[-1]
            new_r, new_c = coord

            if is_attack and self.category == MilitaryObject.COMBAT:
                self.combat_attack(game, new_r, new_c)
            else:
                self.move_unconditionaly(game, new_r, new_c)
                self.mp = max(0, self.mp - mp_spent)

                self.path = self.path[self.path.index(coord):]
                # self.ranged_target = None

        game.update()

    def _get_available_path_coords(self, game) -> List[Tuple[Tuple[int, int], int, bool]]:
        '''
        is_attack can be true only if it is the last one

        :param game:
        :return: [(coord1, mp_spent1, is_attack1), ...]
        '''
        result = []

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

            new_tile_unit = next(iter(game.map.get(*path_coord).game_objects), None)

            if new_tile_unit is not None:
                if game.is_enemy(new_tile_unit.player) and self.can_attack:  # it's the last available state
                    result.append((path_coord, self.mp - mp_left + move_cost, True))
                    return result
                elif isinstance(new_tile_unit, Unit):  # skip the tile
                    mp_left -= move_cost
                    path_coord_prev = path_coord

                    continue

            result.append((path_coord, self.mp - mp_left + move_cost, False))

            mp_left -= move_cost
            path_coord_prev = path_coord

        return result

    def move_unconditionaly(self, game, new_r, new_c):
        game.map.remove(self.r, self.c, self)
        game.map.get(new_r, new_c).game_objects.add(self)

        self.r = new_r
        self.c = new_c


    def draw(self, screen, game):
        hex = game.map.get(self.r, self.c).geometry

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
    Tank = lambda player, r, c: Unit('tank', player, r, c,
                                     category=MilitaryObject.COMBAT,
                                     image_path='assets/units/tank.png',
                                     mp_base=4, ranged_strength_base=0, range_radius_base=0,
                                     combat_strength_base=85,
                                     sound_attack='assets/sounds/tank_attack.ogg')

    Artillery = lambda player, r, c: Unit('artillery', player, r, c,
                                          category=MilitaryObject.RANGED,
                                          image_path='assets/units/artillery.png', mp_base=3,
                                          ranged_strength_base=100, range_radius_base=2, combat_strength_base=75,
                                          sound_attack='assets/sounds/artillery_attack.ogg')

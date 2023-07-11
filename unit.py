import math

import pygame
import random


random.seed(1)

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

    MILITARY_RANGED = 31
    MILITARY_COMBAT = 32
    MILITARY_NAVY = 33


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


class Unit:
    def __init__(self, name, player, r, c, category, sub_category, image_path,
                 mp_base, combat_strength_base, ranged_strength_base=0, range_radius_base=0,
                 sound_attack=None, sound_movement=None, modifiers=None) -> None:

        self.name = name

        self.player = player

        self.r = r
        self.c = c

        self.category = category
        self.sub_category = sub_category

        self.image_path = image_path

        image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(image, DEFAULT_UNIT_IMAGE_SIZE)
        self.image_ui = pygame.transform.scale(image, UI_UNIT_IMAGE_SIZE)

        self._ranged_strength_base = ranged_strength_base
        self._range_radius_base = range_radius_base
    
        self._combat_strength_base = combat_strength_base

        self.hp = random.randint(50, 100)

        self.mp = mp_base
        self.mp_base = mp_base

        self.modifiers = modifiers

        self.is_selected = False

        self.state = UnitState.DEFAULT

        self.path = []

        self.can_attack = True

        # self.ranged_target = None

        self.sound_attack = pygame.mixer.Sound(sound_attack) if sound_attack else None
        self.sound_movement = pygame.mixer.Sound(sound_movement) if sound_movement else None

    @staticmethod
    def compute_combat_damage(unit1, unit2):
        diff = unit1.calc_combat_strength() - unit2.calc_combat_strength()
        return round(random.uniform(0.8, 1.2) * 30 * math.exp(diff / 25))

    @staticmethod
    def compute_ranged_damage(unit1, unit2):
        diff = unit1.calc_ranged_strength() - unit2.calc_combat_strength()
        return round(random.uniform(0.8, 1.2) * 30 * math.exp(diff / 25))

    def gain_hps(self):
        if self.mp == self.mp_base:
            if any(city.is_cell_inside(self.r, self.c) for city in self.player.cities):
                self.hp = min(100, self.hp + 10)

    def combat_attack(self, game, enemy_r, enemy_c):
        avail_path_coords = self._get_available_path_coords(game)

        enemy_unit = game.map.get(enemy_r, enemy_c).tile['unit']

        enemy_unit_damage = self.compute_combat_damage(self, enemy_unit)
        unit_damage = self.compute_combat_damage(enemy_unit, self)

        print(f"{self.player.nation}'s {self.name} on hex {self.r, self.c} hp: {self.hp}, damage: {unit_damage}")
        print(
            f"{enemy_unit.player.nation}'s {enemy_unit.name} on hex {enemy_unit.r, enemy_unit.c} hp: {enemy_unit.hp}, damage: {enemy_unit_damage}")
        print()

        unit_r, unit_c = self.r, self.c
        enemy_unit_r, enemy_unit_c = enemy_unit.r, enemy_unit.c

        if game.sound_marker.state and self.sound_attack:
            self.sound_attack.play()

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
            # self.ranged_target = None

        else:
            self.hp -= unit_damage
            enemy_unit.hp -= enemy_unit_damage

            # -1 to MP because of the attack
            self.mp -= 1  # self.selected = False
            # self.ranged_target = None

            if len(avail_path_coords) >= 2:
                # if avail_path_coords[-1]'s is_attack is True then avail_path_coords[-2][0] will be adjacent,
                # so it guarantied to be legal move
                self.move_unconditionaly(game, *avail_path_coords[-2][0])

        self.can_attack = False
        self.path = []
        # enemy_unit.can_attack = False
        enemy_unit.path = []

    def get_ranged_target(self, game):
        if len(self.path) == 0:
            return

        potential_target_rc = self.path[-1]
        enemies_on_hex = game.get_units_on_hex(*potential_target_rc, only_enemies=True)

        if len(enemies_on_hex) > 0 \
                and self.sub_category == UnitCategories.MILITARY_RANGED \
                and self.is_within_attack_range(game, *potential_target_rc):

            _, enemy_unit = enemies_on_hex[0]
            return enemy_unit

    def is_within_attack_range(self, game, enemy_r, enemy_c):
        assert self.sub_category == UnitCategories.MILITARY_RANGED

        return game.map.get_distance((self.r, self.c), (enemy_r, enemy_c)) <= self._range_radius_base

    def ranged_attack(self, game, enemy_r, enemy_c):
        enemy_unit = game.map.get(enemy_r, enemy_c).tile['unit']
        enemy_unit_damage = self.compute_combat_damage(self, enemy_unit)

        if game.sound_marker.state and self.sound_attack:
            self.sound_attack.play(maxtime=1500, fade_ms=500)

        game.display.show_damage_text(f'-{min(100, int(enemy_unit_damage))}', enemy_unit.r, enemy_unit.c)

        if enemy_unit.hp - enemy_unit_damage <= 0:
            try:
                enemy_unit.player.units.remove(enemy_unit)  # del enemy_unit
                game.map.set_data(enemy_unit.r, enemy_unit.c, 'unit', None)
            except ValueError as err:
                raise ValueError
        else:
            enemy_unit.hp -= enemy_unit_damage

        self.mp = 0
        self.path = []
        # self.ranged_target = None
        self.can_attack = False

    def move(self, game):
        # if there is just a transition without an attack - the logic is the same for any type of unit
        if len(self.path) == 0 and self.get_ranged_target(game) is None:
            return

        # unit_at_the_end = game.map.get(*self.path[-1]).tile.get('unit')
        # is_enemy_at_the_end = game.is_enemy((unit_at_the_end or self).player)

        # check if ranged unit inside the attack radius
        ranged_target = self.get_ranged_target(game)
        if ranged_target is not None:
            self.ranged_attack(game, ranged_target.r, ranged_target.c)



        # if is_enemy_at_the_end and self.sub_category == UnitCategories.MILITARY_RANGED \
        #         and self.is_within_attack_range(game, *self.path[-1]):
        #     self.ranged_attack(game, *self.path[-1])
        else:
            avail_path_coords = self._get_available_path_coords(game)
            if len(avail_path_coords) == 0:
                return

            coord, mp_spent, is_attack = avail_path_coords[-1]
            new_r, new_c = coord

            if is_attack and self.sub_category == UnitCategories.MILITARY_COMBAT:
                self.combat_attack(game, new_r, new_c)
            else:
                self.move_unconditionaly(game, new_r, new_c)
                self.mp = max(0, self.mp - mp_spent)

                self.path = self.path[self.path.index(coord):]
                # self.ranged_target = None

        game.update()

    def _get_available_path_coords(self, game) -> List[Tuple[Tuple, int, bool]]:
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

    def move_unconditionaly(self, game, new_r, new_c):
        game.map.set_data(self.r, self.c, 'unit', None)
        game.map.set_data(new_r, new_c, 'unit', self)

        self.r = new_r
        self.c = new_c

    def calc_combat_strength(self, ):
        return self._combat_strength_base  # + modifiers

    def calc_ranged_strength(self, ):
        return self._ranged_strength_base  # + modifiers

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
    Tank = lambda player, r, c: Unit('tank', player, r, c,
                                           category=UnitCategories.MILITARY,
                                           sub_category=UnitCategories.MILITARY_COMBAT,
                                           image_path='assets/units/tank.png',
                                           mp_base=4, ranged_strength_base=0, range_radius_base=0,
                                           combat_strength_base=85,
                                           sound_attack='assets/sounds/tank_attack.ogg')

    Artillery = lambda player, r, c: Unit('artillery', player, r, c,
                                                category=UnitCategories.MILITARY,
                                                sub_category=UnitCategories.MILITARY_RANGED,
                                                image_path='assets/units/artillery.png', mp_base=3, 
                                                ranged_strength_base=80, range_radius_base=2, combat_strength_base=60,
                                                sound_attack='assets/sounds/artillery_attack.ogg')

import math
import random

import pygame
from line_profiler_pycharm import profile


class MilitaryObject:
    RANGED = 31
    COMBAT = 32
    NAVY = 33

    def __init__(self, name, player, r, c, category, mp_base,
                 combat_strength_base, ranged_strength_base, range_radius_base,
                 modifiers=None, sound_attack=None, sound_movement=None):
        self.path = []
        self.can_attack = True

        self.name = name
        self._player = player

        self.r = r
        self.c = c

        self.category = category

        self._ranged_strength_base = ranged_strength_base
        self._range_radius_base = range_radius_base

        self._combat_strength_base = combat_strength_base

        self.hp = random.randint(50, 100)

        self.mp = mp_base
        self.mp_base = mp_base

        self.modifiers = modifiers

        self.sound_attack = pygame.mixer.Sound(sound_attack) if sound_attack else None
        self.sound_movement = pygame.mixer.Sound(sound_movement) if sound_movement else None

        self.is_selected = False

    @property
    def player(self):
        return self._player

    def gain_hps(self):
        raise NotImplementedError()

    def combat_attack(self, game, enemy_r, enemy_c):
        raise NotImplementedError()

    def get_ranged_target(self, game):
        if len(self.path) == 0:
            return

        potential_target_rc = self.path[-1]
        enemies_on_hex = game.get_game_objects_on_hex(*potential_target_rc, only_enemies=True)

        if len(enemies_on_hex) > 0 \
                and self.category == MilitaryObject.RANGED \
                and self.is_within_attack_range(game, *potential_target_rc):

            _, enemy_unit = enemies_on_hex[0]
            return enemy_unit

    def is_within_attack_range(self, game, enemy_r, enemy_c):
        assert self.category == MilitaryObject.RANGED

        return game.map.get_distance((self.r, self.c), (enemy_r, enemy_c)) <= self._range_radius_base

    def calc_combat_strength(self, ):
        return self._combat_strength_base  # + modifiers

    def calc_ranged_strength(self, ):
        return self._ranged_strength_base  # + modifiers

    def ranged_attack(self, game, enemy_r, enemy_c):
        raise NotImplementedError()

    @profile
    def move(self):
        raise NotImplementedError()

    @staticmethod
    def compute_combat_damage(unit1, unit2):
        diff = unit1.calc_combat_strength() - unit2.calc_combat_strength()
        return round(random.uniform(0.8, 1.2) * 30 * math.exp(diff / 25))

    @staticmethod
    def compute_ranged_damage(unit1, unit2):
        diff = unit1.calc_ranged_strength() - unit2.calc_combat_strength()
        return round(random.uniform(0.8, 1.2) * 30 * math.exp(diff / 25))

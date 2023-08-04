import pygame

from city import City
from unit import Unit


class Player:

    def __init__(self, nation):

        self.nation = nation
        self.units = []
        self.cities = []

        self.enemy_player = None
        self.enemy_unit = None

        self.ai = None#ai

        self.known_map = None # TODO: туман войны
        # self.ready_to_attack_hex = (None, None)

        self._reward = 0

        self._reward_cum = 0

    @property
    def is_ai(self):
        return self.ai is not None

    @property
    def game_objects(self):
        return self.units + self.cities

    @property
    def reward(self):
        return self._reward

    @property
    def reward_cum(self):
        return self._reward_cum

    def add_reward(self, value):
        self._reward += value
        self._reward_cum += value

    def reset_reward(self):
        self._reward = 0

    #
    # @reward.setter
    # def reward(self, value):
    #     self._reward = value

    def create_paths(self):
        if not self.is_ai:
            raise Exception('create_paths can be called only on AI players')

        self.ai.create_paths()

    # def add_unit(self, unit_type, r, c):
    #     unit = unit_type(self, r, c)
    #     self.units.append(unit)
    #
    #     return unit

    def add_game_obj(self, game_obj):

        if isinstance(game_obj, Unit):
            self.units.append(game_obj)
        elif isinstance(game_obj, City):
            self.cities.append(game_obj)
        else:
            raise NotImplementedError()

        return game_obj

    def destroy(self, game, game_object):
        game_object.hp = 0

        if game_object in self.units:
            game.map.remove(game_object.r, game_object.c, game_object)
            self.units.remove(game_object)
        elif game_object in self.cities:
            self.cities.remove(game_object)
        else:
            raise Exception()

    def set_enemy(self, player, unit):
        self.enemy_player = player
        self.enemy_unit = unit
        # self.ready_to_attack_hex = (hex_r, hex_c)

    def get_enemy(self):
        return self.enemy_player, self.enemy_unit

    def no_attack(self, ):
        self.enemy_player = None
        self.enemy_unit = None
        # self.ready_to_attack_hex = (None, None)

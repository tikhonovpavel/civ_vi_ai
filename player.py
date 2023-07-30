import pygame

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

        self.reward = 0

    @property
    def is_ai(self):
        return self.ai is not None

    @property
    def game_objects(self):
        return self.units + self.cities

    def create_paths(self):
        if not self.is_ai:
            raise Exception('create_paths can be called only on AI players')

        self.ai.create_paths()

    def add_unit(self, unit_type, r, c):
        unit = unit_type(self, r, c)
        self.units.append(unit)

        return unit

    def add_city(self, city):
        self.cities.append(city)

        return city

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

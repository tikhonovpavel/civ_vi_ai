import pygame

class Player:

    def __init__(self, nation, ai=None) -> None:

        self.nation = nation
        self.units = []
        self.cities = []

        self.enemy_player = None
        self.enemy_unit = None

        self.is_ai = ai is not None
        self._ai = ai

        self.known_map = None # TODO: туман войны
        # self.ready_to_attack_hex = (None, None)

    def create_paths(self):
        if not self.is_ai:
            raise Exception('create_paths can be called only on AI players')

        self._ai.create_paths(self)

    def add_unit(self, unit_type, r, c):
        unit = unit_type(self, r, c)
        self.units.append(unit)

        return unit

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

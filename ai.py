import numpy as np
from player import Player


class AI:

    def __init__(self, game):
        # self.player = player
        self.game = game

    def create_paths(self, player: Player):
        pass


class DoNothingAI(AI):
    def __init__(self, game):
        super(DoNothingAI, self).__init__(game)

    def create_paths(self, player: Player):
        pass


class SimpleAI(AI):
    """
    Each unit just walks to the nearest enemy unit and attacks

    """

    def __init__(self, game):
        super(SimpleAI, self).__init__(game)

    def create_paths(self, player: Player):
        enemy_players = self.game.diplomacy.get_enemies(player)
        enemy_objects = sum((enemy.game_objects for enemy in enemy_players), start=[])

        for obj in player.game_objects:
            target_obj, nearest_enemy_dist = self.game.map.get_nearest_obj(obj, enemy_objects)

            if target_obj is None:
                return

            self.game.set_allowed_shortest_path(obj, target_obj.r, target_obj.c)


class SimpleAIHikikomori(AI):
    """
    Each unit just walks to the nearest enemy unit and attacks, only if this enemy is no further than N
    cells away from the homeland borders. If so, goes back to home

    """

    def __init__(self, game, max_distance_from_enemy=3):
        super(SimpleAIHikikomori, self).__init__(game)
        self.max_distance_from_enemy = max_distance_from_enemy

    def create_paths(self, player: Player):
        enemy_players = self.game.diplomacy.get_enemies(player)
        enemy_units = sum((enemy.units for enemy in enemy_players), start=[])

        for obj in player.game_objects:
            target_obj, nearest_enemy_dist = self.game.map.get_nearest_obj(obj, enemy_units)

            if target_obj is None or nearest_enemy_dist > self.max_distance_from_enemy:
                target_obj, _ = self.game.map.get_nearest_obj(obj, player.cities)

            if target_obj is not None:
                self.game.set_allowed_shortest_path(obj, target_obj.r, target_obj.c)

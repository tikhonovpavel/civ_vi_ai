import math
import networkx as nx

# from game import Game
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
        enemy_player = self.game.diplomacy.get_enemies(player)[0]

        for unit in player.units:
            nearest_enemy_unit = None
            nearest_enemy_dist = np.inf

            for enemy_unit in enemy_player.units:
                dist = self.game.map.get_distance((unit.r, unit.c), (enemy_unit.r, enemy_unit.c))
                # dist = nx.shortest_path_length(self.game.map._graph, (unit.r, unit.c), (enemy_unit.r, enemy_unit.c))

                if dist <= nearest_enemy_dist:
                    nearest_enemy_unit = enemy_unit
                    nearest_enemy_dist = dist

            if nearest_enemy_unit is None:
                return

            self.game.set_allowed_shortest_path(unit, nearest_enemy_unit.r, nearest_enemy_unit.c)

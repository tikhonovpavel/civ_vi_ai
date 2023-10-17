import numpy as np
from player import Player


class AI:

    def __init__(self, game, player):
        # self.player = player
        self.game = game
        self.player = player

    def create_paths(self):
        pass


class TrainableAI(AI):
    def __init__(self, game, player):
        super(TrainableAI, self).__init__(game, player)

    def create_paths(self):
        raise NotImplementedError()

    def receive_reward(self, reward):
        raise NotImplementedError()


class DoNothingAI(AI):
    def __init__(self, game, player):
        super(DoNothingAI, self).__init__(game, player)

    def create_paths(self):
        pass


class SimpleAI(AI):
    """
    Each unit just walks to the nearest enemy unit and attacks

    """

    def __init__(self, game, player):
        super(SimpleAI, self).__init__(game, player)

    def create_paths(self):
        player = self.player

        enemy_players = self.game.diplomacy.get_enemies(player)
        enemy_objects = sum((enemy.game_objects for enemy in enemy_players), start=[])

        for obj in player.game_objects:
            target_obj, nearest_enemy_dist = self.game.map.get_nearest_obj(obj, enemy_objects)

            if target_obj is None:
                return

            obj.set_allowed_shortest_path(self.game, target_obj.r, target_obj.c)


class SimpleAIHikikomori(AI):
    """
    Each unit just walks to the nearest enemy unit and attacks, only if this enemy is no further than N
    cells away from the homeland borders. If so, goes back to home

    """

    def __init__(self, game, player, max_distance_from_enemy=3):
        super(SimpleAIHikikomori, self).__init__(game, player)
        self.max_distance_from_enemy = max_distance_from_enemy

    def create_paths(self):
        player = self.player

        enemy_players = self.game.diplomacy.get_enemies(player)
        enemy_units = sum((enemy.units for enemy in enemy_players), start=[])

        for obj in player.game_objects:
            target_obj, nearest_enemy_dist = self.game.map.get_nearest_obj(obj, enemy_units)

            if target_obj is None or nearest_enemy_dist > self.max_distance_from_enemy:
                target_obj, _ = self.game.map.get_nearest_obj(obj, player.cities)

            if target_obj is not None:
                obj.set_allowed_shortest_path(self.game, target_obj.r, target_obj.c)

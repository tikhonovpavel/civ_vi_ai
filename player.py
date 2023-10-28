
from city import City

from unit import Unit


class Player:
    def __init__(self, nation, silent=False):

        self.nation = nation
        self.units = []
        self.cities = []

        self.enemy_player = None
        self.enemy_unit = None

        self.ai = None

        self.known_map = None  # TODO: fog of war

        self._reward = 0
        self._reward_cum = 0

        self.silent = silent

    @property
    def is_ai(self):
        return self.ai is not None

    @property
    def game_objects(self):
        return self.units + self.cities

    def create_paths(self, **kwargs):
        if not self.is_ai:
            raise Exception('create_paths can be called only on AI players')

        self.ai.create_paths(**kwargs)

    def add_game_obj(self, game_obj):
        if isinstance(game_obj, Unit):
            self.units.append(game_obj)
        elif isinstance(game_obj, City):
            self.cities.append(game_obj)
        else:
            raise NotImplementedError()

        return game_obj

    def destroy(self, game, game_object, on_defense):
        game_object.hp = 0

        if game_object in self.units:
            game.map.remove(game_object.r, game_object.c, game_object)
            self.units.remove(game_object)
            
            if not self.silent:
                print(f'unit {game_object} has been destroyed (allegedly)')

        elif game_object in self.cities:
            self.cities.remove(game_object)

        else:
            raise Exception()

    def set_enemy(self, player, unit):
        self.enemy_player = player
        self.enemy_unit = unit

    def get_enemy(self):
        return self.enemy_player, self.enemy_unit

    def no_attack(self, ):
        self.enemy_player = None
        self.enemy_unit = None

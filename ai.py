import numpy as np
import torch


terrain_classes = ['plains', 'hills', 'forest', 'water']
units_classes = [None, 'artillery', 'tank']
cities_classes = [None, 'city']

terrains_label_encoder = {cat: i for i, cat in enumerate(terrain_classes)}



class AI:

    def __init__(self, game, player, silent=False):
        # self.player = player
        self.game = game
        self.player = player
        self.silent = silent

    def create_paths(self):
        pass

class TrainableAI(AI):
    def __init__(self, game, player, silent=False):
        super(TrainableAI, self).__init__(game, player, silent)

    def create_paths(self):
        raise NotImplementedError()

    def receive_reward(self, reward):
        raise NotImplementedError()

    def convert_map_to_tensor(self):
        player = self.player
        game = self.game

        all_units = player.units + sum([p.units for p in game.diplomacy.get_enemies(player)], [])
        all_cities = player.cities + sum([p.cities for p in game.diplomacy.get_enemies(player)], [])

        terrains = game.map.get_terrains_map()
        terrains = [[terrains_label_encoder[name] for name in row] for row in terrains]
        terrains = torch.LongTensor(terrains)
        terrains_onehot = to_onehot_tensor(terrains, num_classes=len(terrain_classes))
        # print(f"Shape of terrains_onehot: {terrains_onehot.shape}")

        my_units = get_attribute_map(player.units, 'category', game.map.n_rows, game.map.n_columns, units_classes)
        my_units_onehot = to_onehot_tensor(my_units, num_classes=len(units_classes))
        # print(f"Shape of my_units_onehot: {my_units_onehot.shape}")

        my_cities = get_attribute_map(player.cities, 'category', game.map.n_rows, game.map.n_columns, cities_classes)
        my_cities_onehot = to_onehot_tensor(my_cities, num_classes=len(cities_classes))
        my_cities_onehot = my_cities_onehot[1:]
        # print(f"Shape of my_cities_onehot: {my_cities_onehot.shape}")

        enemy_units = get_attribute_map(sum((p.units for p in game.diplomacy.get_enemies(player)), []), 'category',
                                        game.map.n_rows, game.map.n_columns, units_classes)
        enemy_units_onehot = to_onehot_tensor(enemy_units, num_classes=len(units_classes))
        # print(f"Shape of enemy_units_onehot: {enemy_units_onehot.shape}")

        enemy_cities = get_attribute_map(sum((p.cities for p in game.diplomacy.get_enemies(player)), []), 'category',
                                         game.map.n_rows, game.map.n_columns, cities_classes)
        enemy_cities_onehot = to_onehot_tensor(enemy_cities, num_classes=len(cities_classes))
        enemy_cities_onehot = enemy_cities_onehot[1:]
        # print(f"Shape of enemy_cities_onehot: {enemy_cities_onehot.shape}")

        units_hp = np.zeros((game.map.n_rows, game.map.n_columns))
        for u in all_units:
            units_hp[u.r][u.c] = u.hp
        units_hp = torch.tensor(units_hp).unsqueeze(0)
        # print(f"Shape of units_hp: {units_hp.shape}")

        cities_hp = np.zeros((game.map.n_rows, game.map.n_columns))
        for city in all_cities:
            cities_hp[city.r][city.c] = city.hp
        cities_hp = torch.tensor(cities_hp).unsqueeze(0)
        # print(f"Shape of cities_hp: {cities_hp.shape}")

        units_mp = np.zeros((game.map.n_rows, game.map.n_columns))
        for u in all_units:
            units_mp[u.r][u.c] = u.mp
        units_mp = torch.tensor(units_mp).unsqueeze(0)
        # print(f"Shape of units_mp: {units_mp.shape}")

        result = torch.vstack([terrains_onehot,
                               my_units_onehot, my_cities_onehot, enemy_units_onehot, enemy_cities_onehot,
                               units_hp, cities_hp, units_mp])
        # print(f"Shape of result: {result.shape}")

        return result

    def create_input_tensor(self, unit):
        """
        :return: (n_units, n_layers, map_n_rows, map_n_columns)
        """
        map_tensor = self.convert_map_to_tensor()

        result = []

        # for unit in player.units:
        current_unit_layer = torch.zeros((self.game.map.n_rows, self.game.map.n_columns))
        current_unit_layer[unit.r][unit.c] = 1
        current_unit_layer = current_unit_layer.unsqueeze(0)

        reachable_cells_layer = torch.zeros((self.game.map.n_rows, self.game.map.n_columns))
        # reachable_cells = unit.get_reachable_cells(self.game)
        # indices = torch.tensor(reachable_cells)
        # reachable_cells_layer[indices[:, 0], indices[:, 1]] = 1
        reachable_cells_layer = reachable_cells_layer.unsqueeze(0)

        result.append(torch.vstack([map_tensor, current_unit_layer, reachable_cells_layer]))
        result = torch.stack(result)

        return result.float()



class DoNothingAI(AI):
    def __init__(self, game, player, silent=False):
        super(DoNothingAI, self).__init__(game, player, silent)

    def create_paths(self):
        pass


class SimpleAI(AI):
    """
    Each unit just walks to the nearest enemy unit and attacks

    """

    def __init__(self, game, player, silent=False):
        super(SimpleAI, self).__init__(game, player, silent)

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

    def __init__(self, game, player, max_distance_from_enemy=3, silent=False):
        super(SimpleAIHikikomori, self).__init__(game, player, silent)
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


def to_onehot_tensor(attribute_map, num_classes):
    attribute_map = torch.LongTensor(attribute_map)

    onehot = torch.nn.functional.one_hot(attribute_map, num_classes=num_classes)
    onehot = onehot.permute(2, 0, 1)

    return onehot


def get_attribute_map(collection, attr_name, n_rows, n_cols, classes):
    label_encoder = {cat: i for i, cat in enumerate(classes)}

    attribute_map = torch.zeros((n_rows, n_cols), dtype=torch.int64)
    for obj in collection:
        attribute_map[obj.r][obj.c] = label_encoder[getattr(obj, attr_name)]

    return attribute_map

import math
from typing import List

import networkx as nx
import random

from game_object import MilitaryObject
from sortedcontainers import SortedList

from city import City

radius = 20


class TerrainType:
    def __init__(self, name, image_path, cost):
        self.name = name
        self.image_path = image_path
        self.cost = cost


class TerrainTypes:
    PLAINS = TerrainType('plains', 'assets/tiles/plains.png', cost=1)
    HILLS = TerrainType('hills', 'assets/tiles/hills.png', cost=2)
    FOREST = TerrainType('forest', 'assets/tiles/forest.png', cost=2)
    # SNOW =
    # TUNDRA =
    # MOUNTAIN =
    # WATER_COAST =
    # WATER_OCEAN =



class Tile:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

        # self.terrain = TerrainTypes.PLAINS
        self.terrain = random.choices([TerrainTypes.PLAINS, TerrainTypes.HILLS, TerrainTypes.FOREST],
                                      [0.5, 0.25, 0.25], k=1)[0]
        
        self.points = []
        
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = angle_deg * (math.pi / 180)
            point_x = x + radius * math.cos(angle_rad)
            point_y = y + radius * math.sin(angle_rad)
            self.points.append((point_x, point_y))


class Map:
    class Cell:
        def __init__(self, tile, game_objects):
            self.game_objects = game_objects
            self.geometry = tile

    def __init__(self, rows, columns, offset_x=50, offset_y=50) -> None:

        self.n_rows = rows
        self.n_columns = columns

        self.tiles = []
        self._graph = nx.DiGraph()

        self._default_value = lambda: SortedList(key=lambda x: 0 if isinstance(x, City) else 1)

        side = math.sqrt(3) / 2 * radius
        
        for r in range(rows):
            row = []
            
            for c in range(columns):
                if r % 2 == 0:
                    row.append(Tile(3 * radius * c + offset_x, side * r + offset_y))
                else:
                    row.append(Tile(1.5 * radius * (c * 2 + 1) + offset_x, side * r + offset_y))

                self._graph.add_node((r, c), game_objects=self._default_value())
                # print(self._graph[(r, c)]['game_objects'])
                # print()

            self.tiles.append(row)

        for r, row in enumerate(self.tiles):
            for c, _ in enumerate(row):
                for neighbour in self.get_neighbours_grid_coords(r, c):
                    if neighbour is None:
                        continue
                    self._graph.add_edge((r, c), neighbour,
                                         # terrain=self.tiles[r][c].terrain,
                                         cost=self.tiles[neighbour[0]][neighbour[1]].terrain.cost,
                                         # cost=self.tiles[r][c].terrain.cost
                                        )

    def get(self, r, c):
        return Map.Cell(self.tiles[r][c],
                        self._graph.nodes[r, c]['game_objects'])

    def get_terrains_map(self):
        return [[self.get(r, c).geometry.terrain.name for c in range(self.n_columns)]
                for r in range(self.n_rows)]

    def set(self, r, c, value: List[MilitaryObject]):
        self._graph.nodes[r, c]['game_objects'] = value

    def reset(self, r, c):
        self._graph.nodes[r, c]['game_objects'] = self._default_value()

    def remove(self, r, c, game_object):
        self.get(r, c).game_objects.remove(game_object)

    # def set(self, r, c, key, value):
    #     self._graph.nodes[r, c][key] = value

    def get_data_edge(self, tile1_coord, tile2_coord):
        return self._graph.edges[tile1_coord, tile2_coord]

    def get_distance(self, from_rc, to_rc, graph=None):
        if graph is None:
            graph = self._graph

        return nx.shortest_path_length(graph, from_rc, to_rc)

    def get_neighbours_grid_coords(self, r, c):
        if r % 2 == 0:
            result = [(r + 1, c), (r + 2, c), (r + 1, c - 1), (r - 1, c - 1), (r - 2, c), (r - 1, c)]
        else:
            result = [(r + 1, c + 1), (r + 2, c), (r + 1, c), (r - 1, c), (r - 2, c), (r - 1, c + 1)]

        result = [(row, col) if 0 <= row < self.n_rows and 0 <= col < self.n_columns else None for row, col in result]

        return result

    def get_nearest_obj(self, obj, collection):
        try:
            result_obj = min(collection, key=lambda o: self.get_distance((obj.r, obj.c), (o.r, o.c)))
            result_dist = self.get_distance((obj.r, obj.c), (result_obj.r, result_obj.c))
        except ValueError:
            return None, math.inf

        return result_obj, result_dist

    def get_grid_coords(self, x, y):
        min_dist, min_r, min_c = math.inf, None, None

        for r, row in enumerate(self.tiles):
            for c, hex in enumerate(row):
                dist = math.dist((hex.x, hex.y), (x, y))
                if dist < min_dist and dist < radius:
                    min_dist = dist
                    min_r, min_c = r, c
        
        return min_r, min_c

    def whom_cell_is_it(self, game, r, c):
        for player in game.players:
            if any(city.is_cell_inside(r, c) for city in player.cities):
                return player.nation

        return None


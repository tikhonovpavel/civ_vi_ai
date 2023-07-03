import math
import networkx as nx
import random

radius = 20


class TerrainType:
    def __init__(self, image_path, cost):
        self.image_path = image_path
        self.cost = cost


class TerrainTypes:
    PLAINS = TerrainType('assets/tiles/plains.png', cost=1)
    HILLS = TerrainType('assets/tiles/hills.png', cost=2)
    FOREST = TerrainType('assets/tiles/forest.png', cost=2)
    # SNOW =
    # TUNDRA =
    # MOUNTAIN =
    # WATER_COAST =
    # WATER_OCEAN =



class Tile:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

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
    class HexTile:
        def __init__(self, hex, tile):
            self.hex = hex
            self.tile = tile

    def __init__(self, rows, columns, offset_x=50, offset_y=50) -> None:

        self.n_rows = rows
        self.n_columns = columns

        self.tiles = []
        self._graph = nx.DiGraph()

        side = math.sqrt(3) / 2 * radius
        
        for r in range(rows):
            row = []
            
            for c in range(columns):
                if r % 2 == 0:
                    row.append(Tile(3 * radius * c + offset_x, side * r + offset_y))
                else:
                    row.append(Tile(1.5 * radius * (c * 2 + 1) + offset_x, side * r + offset_y))
                
                self._graph.add_node((r, c), unit=None)

            self.tiles.append(row)

        for r, row in enumerate(self.tiles):
            for c, _ in enumerate(row):
                for neighbour in self.get_neighbours_grid_coords(r, c):
                    self._graph.add_edge((r, c), neighbour,
                                         # terrain=self.tiles[r][c].terrain,
                                         cost=self.tiles[neighbour[0]][neighbour[1]].terrain.cost,
                                         # cost=self.tiles[r][c].terrain.cost
                                        )

    def get(self, r, c):
        return Map.HexTile(self.tiles[r][c], self._graph.nodes[r, c])

    def set_data(self, r, c, key, value):
        self._graph.nodes[r, c][key] = value

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

        for i in range(len(result)):
            res_r, res_c = result[i]

            result[i] = res_r % self.n_rows, res_c % self.n_columns

        return result

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


import math
import random

import networkx as nx
import pygame

from line_profiler_pycharm import profile

random.seed(42)

class MilitaryObject:
    RANGED = 'ranged'
    COMBAT = 'combat'
    NAVY = 'navy'

    def __init__(self, name, category, player, r, c, role, mp_base,
                 combat_strength_base, ranged_strength_base, range_radius_base, hp=None, mp=None,
                 modifiers=None, path=None, sound_attack=None, sound_movement=None):

        self.path = [] if path is None else path

        self.can_attack = True

        self.name = name
        self.category = category
        self._player = player

        self.r = r
        self.c = c

        self.role = role

        self._ranged_strength_base = ranged_strength_base
        self._range_radius_base = range_radius_base

        self._combat_strength_base = combat_strength_base

        self.hp = random.randint(50, 100) if hp is None else hp

        if mp is None:
            self.mp = mp_base
        else:
            self.mp = mp_base

        self.mp_base = mp_base

        self.modifiers = modifiers

        sound_attack = sound_attack if sound_attack else f'assets/sounds/{category}_attack.ogg'

        try:
            self.sound_attack = pygame.mixer.Sound(sound_attack)
        except pygame.error:
            pass

        # sound_movement = sound_attack if sound_attack else f'assets/sounds/{category}_movement.ogg'
        # self.sound_movement = pygame.mixer.Sound(sound_movement)

        self.is_selected = False

    @property
    def player(self):
        return self._player

    @property
    def coords(self):
        return self.r, self.c

    @property
    def range_radius_base(self):
        return self._range_radius_base

    @property
    def ranged_strength_base(self):
        return self._ranged_strength_base

    @property
    def combat_strength_base(self):
        return self._combat_strength_base

    def gain_hps(self):
        raise NotImplementedError()

    def combat_attack(self, game, enemy_r, enemy_c):
        raise NotImplementedError()

    def get_ranged_target(self, game):
        if len(self.path) == 0:
            return

        potential_target_rc = self.path[-1]
        enemies_on_hex = game.get_game_objects_on_hex(*potential_target_rc, only_enemies=True)

        if len(enemies_on_hex) > 0 \
                and self.role == MilitaryObject.RANGED \
                and self.is_within_attack_range(game, *potential_target_rc):

            _, enemy_unit = enemies_on_hex[0]
            return enemy_unit

    def is_within_attack_range(self, game, enemy_r, enemy_c):
        assert self.role == MilitaryObject.RANGED

        dist = game.map.get_distance((self.r, self.c), (enemy_r, enemy_c), weight='cost')
        enemy_cell_cost = game.map.get(enemy_r, enemy_c).geometry.terrain.cost

        return dist - enemy_cell_cost <= self._range_radius_base

    def calc_combat_strength(self, ):
        return self._combat_strength_base  # + modifiers

    def calc_ranged_strength(self, ):
        return self._ranged_strength_base  # + modifiers

    def ranged_attack(self, game, enemy_r, enemy_c):
        raise NotImplementedError()

    def set_allowed_shortest_path(self, game, to_r, to_c):
        self.path = self.get_allowed_shortest_path(game, to_r, to_c)

    @profile
    def get_allowed_graph(self, game, to_r, to_c):
        current_player = game.get_current_player()

        disallowed_edges = set()
        unit_allowed_hexes = game.map._graph.copy()
        enemies = game.diplomacy.get_enemies(current_player)
        for enemy in enemies:
            for enemy_unit in enemy.game_objects:
                adj_nodes = unit_allowed_hexes.adj[enemy_unit.r, enemy_unit.c].keys()

                if enemy_unit.r == to_r and enemy_unit.c == to_c:
                    disallowed_nodes = [x for x in adj_nodes
                                        if len(unit_allowed_hexes.nodes[x]['game_objects']) > 0
                                        if x != (self.r, self.c)]
                else:
                    disallowed_nodes = adj_nodes

                disallowed_edges.update([(x, (enemy_unit.r, enemy_unit.c)) for x in disallowed_nodes])

        for frm, to in disallowed_edges:
            unit_allowed_hexes.remove_edge(frm, to)

        return unit_allowed_hexes

    @profile
    def get_allowed_shortest_path(self, game, to_r, to_c):
        unit_allowed_hexes = self.get_allowed_graph(game, to_r, to_c)

        frm = (self.r, self.c)
        to = to_r, to_c

        try:
            return nx.shortest_path(unit_allowed_hexes, frm, to, weight='cost')
        except nx.exception.NetworkXNoPath:
            return []

    @profile
    def move(self):
        raise NotImplementedError()

    @staticmethod
    def compute_combat_damage(unit1, unit2):
        diff = unit1.calc_combat_strength() - unit2.calc_combat_strength()
        return round(random.uniform(0.8, 1.2) * 30 * math.exp(diff / 25))

    @staticmethod
    def compute_ranged_damage(unit1, unit2):
        diff = unit1.calc_ranged_strength() - unit2.calc_combat_strength()
        return round(random.uniform(0.8, 1.2) * 30 * math.exp(diff / 25))

import math
import random

import pygame
from line_profiler_pycharm import profile

import networkx as nx

from display import Display

from button import Button, ButtonStates

from map import Map
from player import Player
from unit import Units

DATA_UNIT_ATTR = 'unit'

class Diplomacy:
    WAR = -1
    NEUTRAL = 0
    ALLIES = 1

    def __init__(self, players):
        self._diplomacy_graph = nx.Graph()

        self._players = {p.nation: p for p in players}

        for nation in self._players:
            self._diplomacy_graph.add_node(nation)

    def set_relation(self, player1, player2, relation):
        self._diplomacy_graph.add_edge(player1.nation, player2.nation, weight=relation)

    def is_enemies(self, player1, player2):
        return bool(self._diplomacy_graph.get_edge_data(player1.nation, player2.nation)['weight'] == Diplomacy.WAR)

    def get_enemies(self, player):
        edges = self._diplomacy_graph.edges(player.nation, data=True)
        return [self._players[enemy] for _, enemy, d in edges if d['weight'] == Diplomacy.WAR]


class Game:
    COMBAT_MINIMUM_DAMAGE = 1

    def __init__(self, screen, clock) -> None:
        self.map = Map(30, 15)

        player1 = Player('rome')
        player2 = Player('egypt')

        self.players = [player1, player2]

        self.add_unit(player1, Units.Tank, 15, 3)
        self.add_unit(player1, Units.Tank, 12, 4)
        self.add_unit(player1, Units.Tank, 12, 5)
        self.add_unit(player1, Units.Tank, 8, 3)
        self.add_unit(player1, Units.Artillery, 10, 2)
        self.add_unit(player1, Units.Artillery, 10, 4)

        self.add_unit(player2, Units.Tank, 15, 4)
        self.add_unit(player2, Units.Tank, 5, 11)
        self.add_unit(player2, Units.Tank, 8, 12)
        self.add_unit(player2, Units.Tank, 14, 12)
        self.add_unit(player2, Units.Artillery, 7, 11)
        self.add_unit(player2, Units.Artillery, 17, 12)

        self._current_player_index = 0

        self.next_move_button = Button('Next move', 712, 632, 150, 70, self.next_move)

        # all vs all
        self.diplomacy = Diplomacy(self.players)
        for i in range(len(self.players)):
            for j in range(i + 1, len(self.players)):
                self.diplomacy.set_relation(player1, player2, Diplomacy.WAR)
        for player in self.players:
            self.diplomacy.set_relation(player, player, Diplomacy.ALLIES)

        self.clock = clock

        self.display = Display(screen, self)
        self.update()  # self.players, self.hexagon_grid, self. paths)

    def next_move(self):
        for player in self.players:
            for unit in player.units:
                unit.move(self)

        for player in self.players:
            for unit in player.units:
                unit.mp = unit.mp_base
                unit.can_attack = True
                unit.is_selected = False

        self.update()

    def mouse_motion(self, event):
        x, y = event.pos

        # if self.next_move_button.rect.collidepoint(event.pos):
        #     self.next_move_button.state = ButtonStates.HOVER
        # else:
        #     self.next_move_button.state = ButtonStates.DEFAULT


        self.next_move_button.draw(self.display.screen, self, self.display.text_module)
        pygame.display.update()

        lb, mb, rb = pygame.mouse.get_pressed()
        if rb:
            self.right_button_pressed(*event.pos)


    def add_unit(self, player, unit_type, r, c):
        assert player in self.players

        unit = player.add_unit(unit_type, r, c, self.map.get(r, c))
        self.map.set_data(r, c, DATA_UNIT_ATTR, unit)

        return unit



    def is_enemy(self, other_player):
        return bool(self.diplomacy.is_enemies(self.get_current_player(), other_player))

    def get_current_player(self):
        return self.players[self._current_player_index]

    def get_selected_unit(self):
        return next((u for u in self.get_current_player().units if u.is_selected), None)

    def get_units_on_hex(self, r, c, player=None, only_enemies=False, category=None):
        if player is not None and only_enemies:
            print('Use player and only_enemies simultaneously with caution')

        result = [(p, u) for p in self.players for u in p.units if u.r == r and u.c == c]

        if player is not None:
            result = [(p, u) for p, u in result if p == player]

        if only_enemies:
            result = [(p, u) for p, u in result if self.is_enemy(p)]

        if category is not None:
            result = [(p, u) for p, u in result if u.category == category]

        return result

    def update(self):
        self.display.update_all()  # self.players, self.hexagon_grid, self.paths)

    def left_button_released(self, event):
        mouse_x, mouse_y = event.pos

        if self.next_move_button.rect.collidepoint(event.pos):
            self.next_move_button.click_function()
            # self.next_move_button.state = ButtonStates.PRESSED
            #
            # self.next_move_button.draw(self.display.screen, self, self.display.text_module)
            # pygame.display.update()

        self.get_current_player().no_attack()

        r, c = self.map.get_grid_coords(mouse_x, mouse_y)

        if r is None:
            return

        for unit in self.get_current_player().units:
            unit.is_selected = unit.r == r and unit.c == c

        self.update()

    @profile
    def right_button_pressed(self, mouse_x, mouse_y):
        current_player = self.get_current_player()
        current_player.no_attack()

        unit_selected = self.get_selected_unit()
        if unit_selected is None:
            return

        r, c = self.map.get_grid_coords(mouse_x, mouse_y)
        if r is None:
            return

        # if there is a unit of the same category on the hex - cancel
        if len(self.get_units_on_hex(r, c, player=current_player)) > 0:  # TODO
            unit_selected.path = []
            self.update()
            return

        # if there is an enemy unit on the hex - show attack screen
        enemies_on_hex = self.get_units_on_hex(r, c, only_enemies=True)
        if len(enemies_on_hex) > 0:
            # print('hoba')
            current_player.set_enemy(*enemies_on_hex[0])

            # but if it cannot attack - exit:
            if not unit_selected.can_attack:
                unit_selected.path = []
                return

        disallowed_edges = []
        unit_allowed_hexes = self.map._graph.copy()
        enemies = self.diplomacy.get_enemies(current_player)
        for enemy in enemies:
            for enemy_unit in enemy.units:
                adj_nodes = unit_allowed_hexes.adj[enemy_unit.r, enemy_unit.c].keys()

                if enemy_unit.r == r and enemy_unit.c == c:
                    disallowed_nodes = [x for x in adj_nodes
                                        if unit_allowed_hexes.nodes[x][DATA_UNIT_ATTR] is not None
                                        if x != (unit_selected.r, unit_selected.c)]
                else:
                    disallowed_nodes = adj_nodes


                disallowed_edges.extend([(x, (enemy_unit.r, enemy_unit.c)) for x in disallowed_nodes])


        for frm, to in disallowed_edges:

            # print(f'remove edge from {frm} to {to}')
            unit_allowed_hexes.remove_edge(frm, to)
            # unit_allowed_hexes.remove_edge(to, frm)
        # print()

        # print(disallowed_edges)
        # [rc for rc, d in self.hexagon_grid._graph.nodes.data() if d['unit'] != None and d['unit']]

        frm = (unit_selected.r, unit_selected.c)
        to = r, c
        # unit_selected.path = [frm] + nx.shortest_path(unit_allowed_hexes, frm, to)
        unit_selected.path = nx.shortest_path(unit_allowed_hexes, frm, to, weight='cost')

        self.update()

    # def attack(self, enemy_unit):
    #
    #     return result

    @profile
    def right_button_released(self, mouse_x, mouse_y):
        current_player = self.get_current_player()

        unit_selected = self.get_selected_unit()
        if unit_selected is None:
            return

        r, c = self.map.get_grid_coords(mouse_x, mouse_y)
        if r is None:
            return


        # if there is a unit of the same category on the hex - cancel
        if len(self.get_units_on_hex(r, c, player=current_player)) > 0:  # TODO
            return

        # enemy_player, enemy_unit = current_player.get_enemy()

        if not (len(unit_selected.path) != 0 and (r, c) == unit_selected.path[-1]):
            return

        # confirmation of the move
        unit_selected.move(self)

        return


        # if enemy_player is not None:  # if there is an enemy - attack:
        #
        # # if there is no enemy - just move:
        # else:
        #     self.move_unit(unit_selected, r, c)
        #     # unit_selected.r = r
        #     # unit_selected.c = c
        #
        # unit_selected.selected = False
        # unit_selected.path = []
        #
        # self.update()

    # def set_unit_moving_to(unit, dest_r, dest_c):
    #     unit.state = UnitState.MOVING
    #     unit.set_unit_moving_to(dest_r, dest_c)


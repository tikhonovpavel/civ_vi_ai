import json
from datetime import datetime

import pygame
from line_profiler_pycharm import profile

import networkx as nx

from ai import DoNothingAI
from city import City
from display import Display

from ui import Button, Marker, UI, Text  # , UI

from map import Map
from player import Player
from unit import Units, Unit # , UnitCategories


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

        player1 = Player('Rome')
        player2 = Player('Egypt', ai=DoNothingAI(self))
        # player2 = Player('Egypt', ai=SimpleAI(self))

        self.players = [player1, player2]

        with open('init_state.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

            for p in config:
                for c in p['cities']:
                    coords = set(tuple(x) for x in c['coords'])

                    assert tuple(c['center']) in coords

                    player = self.get_player_by_nation(p['nation'])

                    city = City(c['name'], player, *c['center'], coords, 'assets/city_icon1.png')
                    city.tiles_set = coords

                    self.add_city(player, city)

        self.add_unit(player1, Units.Tank, 15, 3)
        self.add_unit(player1, Units.Tank, 12, 4)
        self.add_unit(player1, Units.Tank, 13, 4)
        self.add_unit(player1, Units.Tank, 12, 5)
        self.add_unit(player1, Units.Tank, 8, 3)
        self.add_unit(player1, Units.Tank, 23, 13)
        self.add_unit(player1, Units.Artillery, 10, 2)
        self.add_unit(player1, Units.Artillery, 14, 4)
        self.add_unit(player1, Units.Artillery, 25, 12)
        self.add_unit(player1, Units.Artillery, 28, 13)
        self.add_unit(player1, Units.Artillery, 28, 14)
        self.add_unit(player1, Units.Artillery, 27, 14)

        self.add_unit(player2, Units.Tank, 15, 4)
        self.add_unit(player2, Units.Tank, 5, 11)
        self.add_unit(player2, Units.Tank, 8, 12)
        self.add_unit(player2, Units.Tank, 14, 12)
        self.add_unit(player2, Units.Tank, 1, 0)
        self.add_unit(player2, Units.Tank, 25, 13)
        self.add_unit(player2, Units.Artillery, 7, 11)
        self.add_unit(player2, Units.Artillery, 7, 13)
        self.add_unit(player2, Units.Artillery, 7, 14)
        self.add_unit(player2, Units.Artillery, 17, 12)

        self._current_player_index = 0

        self.subturn_number = 0
        self.turn_number = 1

        self.next_turn_button = Button('Next turn', 570, 632, 150, 70, self.next_turn)
        self.save_state_button = Button('Save state', 800, 632, 150, 70, self.save_state)
        self.quick_movement_marker = Marker('Quick movement', 770, 720, state=True, click_function=self.update)
        self.sound_marker = Marker('Sound', 770, 720+35, state=True, click_function=self.update)
        self.current_turn_text = Text('Turn: {turn_number}', 550, 690, 1, 1)
        self.current_player_text = Text("({current_player}'s turn)", 550, 730, 1, 1)

        self.ui = UI([self.next_turn_button, self.save_state_button,
                      self.quick_movement_marker, self.sound_marker, self.current_turn_text,
                      self.current_player_text, ])

        self.unit_selected_sound = pygame.mixer.Sound('assets/sounds/select_unit.ogg')
        self.city_selected_sound = pygame.mixer.Sound('assets/sounds/select_unit.ogg')

        # all vs all
        self.diplomacy = Diplomacy(self.players)
        for i in range(len(self.players)):
            for j in range(i + 1, len(self.players)):
                self.diplomacy.set_relation(player1, player2, Diplomacy.WAR)
        for player in self.players:
            self.diplomacy.set_relation(player, player, Diplomacy.ALLIES)

        self.clock = clock
        self.display = Display(screen, self)

        self.current_turn_text.update(turn_number=self.turn_number)
        self.current_player_text.update(current_player=self.players[0].nation)
        self.update()

    def get_player_by_nation(self, nation):
        return next(p for p in self.players if p.nation == nation)

    def save_state(self):
        result = []

        for player in self.players:
            res = {'nation': player.nation, 'cities': []}

            for city in player.cities:
                res['cities'].append({'name': city.name,
                                      'center': [city.r, city.c],
                                      'coords': list(city.tiles_set)})

            res['units'] = []
            for unit in player.units:
                fields = ['name', 'r', 'c', 'category', 'sub_category',
                          'mp', 'hp', 'image_path', 'mb_base', 'ranged_strength_base',
                          'range_radius_base', 'combat_strength_base', 'path']
                unit_json = {k: v for k, v in unit.__dict__.items() if k in fields}

                res['units'].append(unit_json)

            result.append(res)

        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H-%M-%S")
        json.dump(result, open(f'saves/{" vs ".join(p.nation for p in self.players)} {formatted_datetime}.json',
                               'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)

    def next_turn(self):
        self.subturn_number += 1
        if self.subturn_number % len(self.players) == 0:
            self.turn_number += 1

        for obj in self.get_current_player().game_objects:
            obj.move(self)
            obj.gain_hps()

            obj.mp = obj.mp_base
            obj.can_attack = True
            obj.is_selected = False

        self.set_next_current_player()

        self.current_turn_text.update(turn_number=self.turn_number)
        self.current_player_text.update(current_player=self.get_current_player().nation)
        self.update()

        # if now the player is AI, call create_paths method
        # and move units according to the paths
        if self.get_current_player().is_ai:
            self.get_current_player().create_paths()

    def mouse_motion(self, event):
        x, y = event.pos

        self.next_turn_button.draw(self.display.screen, self, self.display.text_module)
        pygame.display.update()

        lb, mb, rb = pygame.mouse.get_pressed()
        if rb:
            self.right_button_pressed(*event.pos)

    def add_unit(self, player, unit_type, r, c):
        assert player in self.players

        unit = player.add_unit(unit_type, r, c)

        self.map.set(r, c, self.map.get(r, c).game_objects + [unit])

        return unit

    def add_city(self, player, city):
        assert player in self.players

        r, c = city.r, city.c

        player.add_city(city)
        self.map.set(r, c, self.map.get(r, c).game_objects + [city])

        return city

    def is_enemy(self, other_player):
        return bool(self.diplomacy.is_enemies(self.get_current_player(), other_player))

    def get_current_player(self):
        return self.players[self._current_player_index]

    def set_next_current_player(self):
        self._current_player_index = (self._current_player_index + 1) % len(self.players)

    def get_selected_unit(self):
        return next((u for u in self.get_current_player().units if u.is_selected), None)

    def get_selected_city(self):
        return next((c for c in self.get_current_player().cities if c.is_selected), None)

    def get_game_objects_on_hex(self, r, c, player=None, only_enemies=False, category=None, game_obj_type=None):
        if player is not None and only_enemies:
            print('Use player and only_enemies simultaneously with caution')

        result = [(p, u) for p in self.players for u in (p.units + p.cities) if u.r == r and u.c == c]

        if player is not None:
            result = [(p, u) for p, u in result if p == player]

        if only_enemies:
            result = [(p, u) for p, u in result if self.is_enemy(p)]

        if category is not None:
            result = [(p, u) for p, u in result if u.category == category]

        if game_obj_type is not None:
            result = [(p, u) for p, u in result if isinstance(u, game_obj_type)]

        return result

    def update(self):
        self.display.update_all()

    def left_button_pressed(self, event):
        mouse_x, mouse_y = event.pos

        self.ui.screen_click(event.pos, self.display)

        self.get_current_player().no_attack()

        r, c = self.map.get_grid_coords(mouse_x, mouse_y)

        if r is None:
            return

        player = self.get_current_player()
        for obj in (player.units + player.cities):
            new_status = (obj.r == r and obj.c == c)

            if self.sound_marker.state and (new_status and not obj.is_selected):
                if isinstance(obj, Unit):
                    self.unit_selected_sound.play()
                elif isinstance(obj, City):
                    self.city_selected_sound.play()

            obj.is_selected = new_status

        self.update()


    @profile
    def right_button_pressed(self, mouse_x, mouse_y):
        current_player = self.get_current_player()
        current_player.no_attack()

        obj_selected = self.get_selected_unit() or self.get_selected_city()

        #  =================== Exit Conditions ===================
        if obj_selected is None:
            return

        r, c = self.map.get_grid_coords(mouse_x, mouse_y)

        if r is None:
            return

        # there is a unit of the same category on the hex
        if len(self.get_game_objects_on_hex(r, c, player=current_player, game_obj_type=Unit)) > 0:  # TODO
            obj_selected.path = []
            self.update()
            return
        # =================== End Of Exit Conditions ===================

        obj_selected.path = []

        # if there is an enemy unit on the hex - show attack screen
        enemies_on_hex = self.get_game_objects_on_hex(r, c, only_enemies=True)
        if len(enemies_on_hex) > 0:
            current_player.set_enemy(*enemies_on_hex[0])

            # but if it cannot attack - exit:
            if not obj_selected.can_attack:
                obj_selected.path = []
                return

        self.set_allowed_shortest_path(obj_selected, r, c)
        self.update()

    def set_allowed_shortest_path(self, unit, to_r, to_c):
        current_player = self.get_current_player()

        disallowed_edges = set()
        unit_allowed_hexes = self.map._graph.copy()
        enemies = self.diplomacy.get_enemies(current_player)
        for enemy in enemies:
            for enemy_unit in enemy.game_objects:
                adj_nodes = unit_allowed_hexes.adj[enemy_unit.r, enemy_unit.c].keys()

                if enemy_unit.r == to_r and enemy_unit.c == to_c:
                    disallowed_nodes = [x for x in adj_nodes
                                        if len(unit_allowed_hexes.nodes[x]['game_objects']) > 0
                                        if x != (unit.r, unit.c)]
                else:
                    disallowed_nodes = adj_nodes

                disallowed_edges.update([(x, (enemy_unit.r, enemy_unit.c)) for x in disallowed_nodes])

        for frm, to in disallowed_edges:
            unit_allowed_hexes.remove_edge(frm, to)

        frm = (unit.r, unit.c)
        to = to_r, to_c

        try:
            unit.path = nx.shortest_path(unit_allowed_hexes, frm, to, weight='cost')
        except nx.exception.NetworkXNoPath:
            unit.path = []

    @profile
    def right_button_released(self, mouse_x, mouse_y):
        current_player = self.get_current_player()

        obj_selected = self.get_selected_unit() or self.get_selected_city()
        if obj_selected is None:
            return

        r, c = self.map.get_grid_coords(mouse_x, mouse_y)

        if r is None:
            return

        # if there is a unit of the same category on the hex - cancel
        if len(self.get_game_objects_on_hex(r, c, player=current_player, game_obj_type=Unit)) > 0:  # TODO
            return

        # confirmation of the move
        if (len(obj_selected.path) != 0 and (r, c) == obj_selected.path[-1]) \
                or (obj_selected.get_ranged_target(self) is not None):
            obj_selected.move(self)

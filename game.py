import json
from datetime import datetime
from itertools import dropwhile

import pygame
import torch
from line_profiler_pycharm import profile

import networkx as nx

from ai import DoNothingAI, SimpleAI, SimpleAIHikikomori, TrainableAI
from city import City
from display import Display
from logger import Logger
# from rl_training import convert_map_to_tensor
from rl_training import PolicyGradientAI

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

    # @log("Start new game")
    @profile
    def __init__(self, config, screen, clock, sound_on=True, autoplay=False, autoplay_max_turns=5,) -> None:


        self._is_started = False

        # player1 = Player('Rome')
        # player1.ai = SimpleAI(self, player1)
        #
        # player2 = Player('Egypt')
        # player2.ai = PolicyGradientAI(self, player2)

        # player2 = Player('Egypt', ai=DoNothingAI(self))
        # player2 = Player('Egypt', ai=SimpleAI(self))
        # player3 = Player('Babylon')
        # player3.ai = SimpleAIHikikomori(self, player3)

        # self.players = [player1, player2, player3]

        self._counter = 0
        self.autoplay_max_turns = autoplay_max_turns
        self._autoplay_turns_counter = 0

        self.players = []

        self.map = Map(16, 5, terrains=config['map'])

        for p in config['players']:
            player = Player(p['nation'])

            if p['ai']:
                player.ai = globals()[p['ai']](self, player)

            self.players.append(player)

            # player = self.get_player_by_nation(p['nation'])

            for c in p['cities']:
                territory = set(tuple(x) for x in c['territory'])
                assert tuple(c['center']) in territory
                city = City(c['name'], player, *c['center'], territory)
                try:
                    self.add_game_obj(player, city)
                except:
                    pass

                for u in p['units']:
                    u['player'] = player
                    unit = Unit(**u)
                    self.add_game_obj(player, unit)

        self._current_player_index = 0

        self.subturn_number = 0
        self.turn_number = 1

        self.next_turn_button = Button('Next turn', 570, 632, 150, 70, self.next_turn)
        self.save_state_button = Button('Save state', 800, 632, 150, 70, self.save_state)
        self.quick_movement_marker = Marker('Quick movement', 770, 720, state=True, click_function=self.update)
        self.sound_marker = Marker('Sound', 770, 720+35, state=sound_on, click_function=self.update)
        self.autoplay_marker = Marker('Autoplay', 770, 720+35*2, state=autoplay, click_function=self.update)
        self.current_turn_text = Text('Turn: {turn_number}', 550, 690, 1, 1)
        self.current_player_text = Text("({current_player}'s turn)", 550, 730, 1, 1)


        self.ui = UI([self.next_turn_button, self.save_state_button,
                      self.quick_movement_marker, self.sound_marker, self.autoplay_marker,
                      self.current_turn_text, self.current_player_text, ])

        try:
            self.unit_selected_sound = pygame.mixer.Sound('assets/sounds/select_unit.ogg')
            self.city_selected_sound = pygame.mixer.Sound('assets/sounds/select_unit.ogg')
        except pygame.error:
            self.unit_selected_sound = None
            self.city_selected_sound = None

        # all vs all
        self.diplomacy = Diplomacy(self.players)
        for i in range(len(self.players)):
            for j in range(i + 1, len(self.players)):
                self.diplomacy.set_relation(self.players[i], self.players[j], Diplomacy.WAR)
        for player in self.players:
            self.diplomacy.set_relation(player, player, Diplomacy.ALLIES)

        self.logger = Logger(map_size=(self.map.n_rows, self.map.n_columns),
                             map=self.map.get_terrains_map(),
                             initial_positions=self._get_initial_positions_string())

        self.clock = clock
        self.display = Display(screen, self)

        self.current_turn_text.update(turn_number=self.turn_number)
        self.current_player_text.update(current_player=self.players[0].nation)
        self.update()

    @property
    def is_started(self):
        return self._is_started

    def start(self):
        self._is_started = True
        self.logger.start_turn(self.get_current_player().nation)

        current_player = self.get_current_player()
        if self.autoplay_marker.state and current_player.is_ai:
            current_player.create_paths()
            self.next_turn()

    def _get_initial_positions_string(self):
        result = []

        for p in self.players:
            result.append({'nation': p.nation,
                           'units': [{'type': u.category,
                                      'coords': (u.r, u.c)} for u in p.units]})
        return result

    def get_player_by_nation(self, nation):
        return next(p for p in self.players if p.nation == nation)

    def save_state(self):
        result = {'map': self.map.get_terrains_map(),
                  'players': []}

        for player in self.players:
            res = {'nation': player.nation,
                   'cities': [],
                   'ai': player.ai.__class__.__name__ if player.is_ai else None}

            for city in player.cities:
                res['cities'].append({'name': city.name,
                                      'center': [city.r, city.c],
                                      'territory': list(city.territory)}, )

            res['units'] = []
            for unit in player.units:
                fields = ['name', 'r', 'c', 'category', 'role', 'sub_role',
                          'mp', 'hp', 'image_path', 'mp_base', 'ranged_strength_base',
                          'range_radius_base', 'combat_strength_base', 'path']
                unit_json = {k: v for k, v in unit.__dict__.items() if k in fields}

                res['units'].append(unit_json)

            result['players'].append(res)

        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H-%M-%S")
        json.dump(result, open(f'saves/{" vs ".join(p.nation for p in self.players)} {formatted_datetime}.json',
                               'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)

    def check_winning_conditions(self, player):
        # return False
        total_cities_n = sum(len(p.cities) for p in self.players)
        return len(player.cities) == total_cities_n

    @profile
    def next_turn(self):
        if not self.is_started:
            raise Exception('Ensure you\'ve called game.start() at the initialization stage')

        current_player = self.get_current_player()
        
        if self.check_winning_conditions(current_player):
            return

        self.subturn_number += 1
        if self.subturn_number % len(self.players) == 0:
            self.turn_number += 1

        for obj in current_player.game_objects:
            obj.move(self)
            obj.gain_hps()

            obj.mp = obj.mp_base
            obj.can_attack = True
            obj.is_selected = False

        if self.check_winning_conditions(current_player):
            print(current_player.nation + ' won!')
            # reward += 1000
            return

        # if isinstance(current_player, TrainableAI):
        #     current_player.receive_reward(reward)

        # ----------------------------------
        # next player
        current_player = self.set_next_current_player()

        self.logger.commit()
        self.logger.start_turn(current_player.nation)

        # get the rewards
        if isinstance(current_player.ai, TrainableAI):
            if self.turn_number != 1:  # then there has been at least 1 call of create_paths
                                       # and the other players also finished their first turn
                current_player.ai.receive_reward(current_player.reward)
                current_player.reset_reward()

        self.current_turn_text.update(turn_number=self.turn_number)
        self.current_player_text.update(current_player=current_player.nation)
        self.update()

        # if now the player is AI, call create_paths method
        # and move units according to the paths
        if current_player.is_ai:
            current_player.create_paths()

        if self.autoplay_marker.state and self._autoplay_turns_counter < self.autoplay_max_turns:
            self._autoplay_turns_counter += 1
            self.next_turn()
        else:
            self._autoplay_turns_counter = 0

    def mouse_motion(self, event):
        x, y = event.pos

        self.next_turn_button.draw(self.display.screen, self, self.display.text_module)
        pygame.display.update()

        lb, mb, rb = pygame.mouse.get_pressed()
        if rb:
            self.right_button_pressed(*event.pos)

    def add_game_obj(self, player, game_obj):
        assert player in self.players

        r, c = game_obj.r, game_obj.c
        self.map.set(r, c, self.map.get(r, c).game_objects + [game_obj])
        player.add_game_obj(game_obj)

        return game_obj


    def is_enemy(self, other_player):
        return bool(self.diplomacy.is_enemies(self.get_current_player(), other_player))

    def get_current_player(self):
        return self.players[self._current_player_index]

    def set_next_current_player(self):
        self._current_player_index = (self._current_player_index + 1) % len(self.players)
        return self.get_current_player()

    def get_selected_unit(self):
        return next((u for u in self.get_current_player().units if u.is_selected), None)

    def get_selected_city(self):
        return next((c for c in self.get_current_player().cities if c.is_selected), None)

    def get_game_objects_on_hex(self, r, c, player=None, only_enemies=False, role=None, game_obj_type=None):
        if player is not None and only_enemies:
            print('Use player and only_enemies simultaneously with caution')

        result = [(p, o) for p in self.players for o in p.game_objects if o.r == r and o.c == c]

        if player is not None:
            result = [(p, u) for p, u in result if p == player]

        if only_enemies:
            result = [(p, u) for p, u in result if self.is_enemy(p)]

        if role is not None:
            result = [(p, u) for p, u in result if u.role == role]

        if game_obj_type is not None:
            result = [(p, u) for p, u in result if isinstance(u, game_obj_type)]

        return result

    def update(self):
        self.display.update_all()

    def left_button_pressed(self, event):
        # if self.check_winning_conditions(self.get_current_player()):
        #     return


        mouse_x, mouse_y = event.pos

        self.ui.screen_click(event.pos, self.display)

        self.get_current_player().no_attack()

        r, c = self.map.get_grid_coords(mouse_x, mouse_y)

        if r is None:
            return

        player = self.get_current_player()

        # если на (r, c) уже был выбран один, то надо снять маркер с него, и поставить на второго
        # но ежели на нём не было выбрано ни одного, то поставить маркер только на первый из них
        # при этом на всех остальных клетках тоже снять маркеры
        rc_objects = []
        for obj in player.game_objects:
            if obj.r == r and obj.c == c:
                rc_objects.append(obj)
            else:
                obj.is_selected = False


        if len(rc_objects) > 0:
            try:
                selected_index = [o.is_selected for o in rc_objects].index(True)
            except ValueError:
                selected_index = -1

            for i, obj in enumerate(rc_objects):
                obj.is_selected = (i == (selected_index + 1) % len(rc_objects))

        # selected_count = 0
        # selected = []
        # for obj in player.game_objects:
        #     selected_count += int(obj.is_selected)
        #     if obj.is_selected:
        #         selected.append((obj.r, obj.c))
        #
        # print(f'how many are selected?: {selected_count}')
        # if selected_count > 1:
        #     print(selected)
        #     print()


        self.update()

    # @profile
    def right_button_pressed(self, mouse_x, mouse_y):
        current_player = self.get_current_player()

        if self.check_winning_conditions(current_player):
            return

        current_player.no_attack()

        obj_selected = self.get_selected_unit() or self.get_selected_city()

        #  =================== Exit Conditions ===================
        if obj_selected is None:
            return

        r, c = self.map.get_grid_coords(mouse_x, mouse_y)

        if r is None:
            return

        # there is a unit of the same role on the hex
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

        obj_selected.set_allowed_shortest_path(self, r, c)
        self.update()

    # @profile
    def right_button_released(self, mouse_x, mouse_y):
        current_player = self.get_current_player()

        if self.check_winning_conditions(current_player):
            return

        obj_selected = self.get_selected_unit() or self.get_selected_city()
        if obj_selected is None:
            return

        r, c = self.map.get_grid_coords(mouse_x, mouse_y)

        if r is None:
            return

        # if there is a unit of the same role on the hex - cancel
        if len(self.get_game_objects_on_hex(r, c, player=current_player, game_obj_type=Unit)) > 0:  # TODO
            return

        # confirmation of the move
        if (len(obj_selected.path) != 0 and (r, c) == obj_selected.path[-1]) \
                or (obj_selected.get_ranged_target(self) is not None):
            obj_selected.move(self)
            self.update()

            # check if the winning conditions holds
            if self.check_winning_conditions(self.get_current_player()):
                print(self.get_current_player().nation + ' won!')
                return

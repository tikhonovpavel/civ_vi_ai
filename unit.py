import math

import networkx as nx
import pygame
import random

from line_profiler_pycharm import profile

import rewards_values
from city import City
from game_object import MilitaryObject
from logger import MoveEvent, CombatAttackEvent, RangedAttackEvent

random.seed(42)

from consts import DEFAULT_UNIT_IMAGE_SIZE, UI_UNIT_IMAGE_SIZE

from typing import Tuple, List


class UnitState:
    DEFAULT = 0
    MOVING = 1
    DEFENCE = 2


# class UnitCategories:
#     CITY = 0
#     GREAT_PERSON = 1
#     CITIZEN = 2
#     MILITARY = 3


DEFAULT_TILE_COLOR = (0, 255, 0)
SELECTED_TILE_COLOR = (0, 128, 0)
DEFAULT_NATION_IMAGE_SIZE = (15, 10)
UI_NATION_IMAGE_SIZE = (50, 50)


def get_nation_icon(nation):
    return f'assets/nations/{nation}.png'


nation_icons = dict()
for player_nation in ['Rome', 'Egypt', 'Babylon']:
    icon = pygame.image.load(get_nation_icon(player_nation))
    nation_icons[player_nation] = {'default': pygame.transform.scale(icon, DEFAULT_NATION_IMAGE_SIZE),
                                   'ui': pygame.transform.scale(icon, UI_NATION_IMAGE_SIZE)}

names = {
    'tank': iter([
        "Iron Behemoth", "Steel Titan", "Armor Avalanche", "Tread Thunder", "Blitz Brute",
        "Metal Maelstrom", "Shell Shock", "Ironclad Intimidator", "Armored Annihilator",
        "Turret Titan", "Vortex Vanguard", "Tungsten Titan", "Bullet Blocker", "Siege Sentinel",
        "Havoc Hammer", "Ground Gripper", "Momentum Monster", "Armor Antagonist",
        "Battlefield Bruiser", "Tectonic Tanker", "Blaze Brute", "Iron Inferno", "Armor Anomaly",
        "Thunder Thrasher", "Shield Shredder", "Turmoil Treader", "Decimator Dreadnought",
        "Steel Saboteur", "Rampage Ranger", "Metal Marauder"]),
    'artillery': iter([
        "Shell Shredder", "Thunder Tosser", "Ballistic Bane", "Shockwave Spitter", "Range Reaper", "Muzzle Menace",
        "Barrage Beast", "Blast Bouncer", "Siege Shooter", "Fire Flinger", "Firestorm Flinger",
        "Shockwave Slinger", "Sonic Slammer", "Havoc Hurler", "Barrage Bouncer", "Rainmaker Renderer",
        "Whiplash Whirler", "Onslaught Orbiter", "Ballistic Blaster", "Cataclysm Catapult", "Quake Queller",
        "Torment Tornado", "Eruption Ejector", "Pinnacle Pulverizer", "Strife Striker", "Inferno Igniter",
        "Blast Bludgeon", "Pandemonium Propeller", "Vortex Vanquisher", "Doom Dispatcher"
    ])}


class Unit(MilitaryObject):
    def __init__(self, category, player, r, c, role, image_path,
                 mp_base, combat_strength_base, ranged_strength_base, range_radius_base, hp=None, mp=None, path=None,
                 modifiers=None, sound_attack=None, sound_movement=None, name=None) -> None:

        if name is None:
            name = next(names[category], None)

        super().__init__(name, category, player, r, c, role, mp_base, combat_strength_base,
                         ranged_strength_base=ranged_strength_base,
                         range_radius_base=range_radius_base, modifiers=modifiers, hp=hp, path=path,
                         sound_attack=sound_attack, sound_movement=sound_movement)
        self.image_path = image_path

        image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(image, DEFAULT_UNIT_IMAGE_SIZE)
        self.image_ui = pygame.transform.scale(image, UI_UNIT_IMAGE_SIZE)

        self.state = UnitState.DEFAULT

        # self.sound_attack = pygame.mixer.Sound(sound_attack) if sound_attack else None
        # self.sound_movement = pygame.mixer.Sound(sound_movement) if sound_movement else None

    def combat_attack(self, game, enemy_r, enemy_c) -> tuple[int, int, int, int]:
        own_reward = 0
        enemy_reward = 0

        enemy_obj = next(iter(game.map.get(enemy_r, enemy_c).game_objects), None)

        enemy_unit_damage = self.compute_combat_damage(self, enemy_obj)
        unit_damage = self.compute_combat_damage(enemy_obj, self)

        game.logger.log_event(CombatAttackEvent(self,
                                                target=enemy_obj,
                                                unit_damage=unit_damage,
                                                enemy_damage=enemy_unit_damage))

        print(f"{self.player.nation}'s {self.category} {self.name} on {self.r, self.c} => "
              f"{enemy_obj.player.nation}'s {enemy_obj.category} {enemy_obj.name} on {enemy_obj.r, enemy_obj.c}. "      
              f"HP: ({self.hp} -> {max(0, self.hp - unit_damage)}) / "
              f"({enemy_obj.hp} -> {max(0, enemy_obj.hp - enemy_unit_damage)})")

        unit_r, unit_c = self.r, self.c
        enemy_unit_r, enemy_unit_c = enemy_obj.r, enemy_obj.c

        if game.sound_marker.state and self.sound_attack:
            self.sound_attack.play()

        game.display.show_damage_text(f'-{min(100, int(unit_damage))}', unit_r, unit_c)
        game.display.show_damage_text(f'-{min(100, int(enemy_unit_damage))}', enemy_unit_r, enemy_unit_c)

        if enemy_obj.hp - enemy_unit_damage <= 0:

            if isinstance(enemy_obj, City):
                # for each unit, which happen to be at the city cell
                for u in game.map.get(enemy_obj.r, enemy_obj.c).game_objects:
                    if isinstance(u, Unit):
                        enemy_obj.player.destroy(game, u, on_defense=True)

                        self.player.add_reward(rewards_values.ENEMY_UNIT_DESTROYED)
                        own_reward += rewards_values.ENEMY_UNIT_DESTROYED

                        enemy_obj.player.add_reward(rewards_values.OWN_UNIT_DESTROYED)
                        enemy_reward += rewards_values.OWN_UNIT_DESTROYED

                enemy_obj.change_ownership(self.player)

                self.player.add_reward(rewards_values.ENEMY_CITY_CAPTURED)
                own_reward += rewards_values.ENEMY_CITY_CAPTURED

                enemy_obj.player.add_reward(rewards_values.OWN_CITY_CAPTURED_BY_ENEMY)
                enemy_reward += rewards_values.OWN_CITY_CAPTURED_BY_ENEMY

            elif isinstance(enemy_obj, Unit):
                enemy_obj.player.destroy(game, enemy_obj, on_defense=True)  # del enemy_unit

                self.player.add_reward(rewards_values.ENEMY_UNIT_DESTROYED)
                own_reward += rewards_values.ENEMY_UNIT_DESTROYED

                enemy_obj.player.add_reward(rewards_values.OWN_UNIT_DESTROYED)
                enemy_reward += rewards_values.OWN_UNIT_DESTROYED
            else:
                raise NotImplementedError()

            self.hp = max(1, self.hp - unit_damage)

            game.logger.log_event(MoveEvent(self,
                                            path=[(self.r, self.c), (enemy_obj.r, enemy_obj.c)]))
            self.move_unconditionally(game, enemy_obj.r, enemy_obj.c)

            self.mp -= 1  # -1 to MP because of the attack

        elif self.hp - unit_damage <= 0:

            self.player.destroy(game, self, on_defense=False)
            enemy_obj.hp = max(1, enemy_obj.hp - enemy_unit_damage)

            self.player.add_reward(rewards_values.OWN_UNIT_DESTROYED)
            own_reward += rewards_values.OWN_UNIT_DESTROYED

        else:
            self.hp -= unit_damage
            enemy_obj.hp -= enemy_unit_damage

            # -1 to MP because of the attack
            self.mp -= 1  # self.selected = False

        self.can_attack = False
        self.path = []

        enemy_obj.path = []

        return unit_damage, enemy_unit_damage, own_reward, enemy_reward

    def ranged_attack(self, game, enemy_r, enemy_c):
        own_reward = 0
        enemy_reward = 0

        enemy_obj = next(iter(game.map.get(enemy_r, enemy_c).game_objects), None)
        enemy_unit_damage = MilitaryObject.compute_ranged_damage(self, enemy_obj)

        print(f"{self.player.nation}'s {self.category} {self.name} on {self.r, self.c} => "
              f"{enemy_obj.player.nation}'s {enemy_obj.category} {enemy_obj.name} on {enemy_obj.r, enemy_obj.c}. "      
              f"HP: ({self.hp} -> {max(0, self.hp)}) / "
              f"({enemy_obj.hp} -> {max(0, enemy_obj.hp - enemy_unit_damage)})")

        if game.sound_marker.state and self.sound_attack:
            self.sound_attack.play(maxtime=1500, fade_ms=500)

        game.display.show_damage_text(f'-{min(100, int(enemy_unit_damage))}', enemy_obj.r, enemy_obj.c)

        if enemy_obj.hp - enemy_unit_damage <= 0:
            if not isinstance(enemy_obj, City):
                enemy_obj.player.destroy(game, enemy_obj, on_defense=True)
                game.map.reset(enemy_obj.r, enemy_obj.c)

                self.player.add_reward(rewards_values.ENEMY_UNIT_DESTROYED)
                own_reward += rewards_values.ENEMY_UNIT_DESTROYED

                enemy_obj.player.add_reward(rewards_values.OWN_UNIT_DESTROYED)
                enemy_reward += rewards_values.OWN_UNIT_DESTROYED

                # game.map.set(enemy_unit.r, enemy_unit.c, [])
            else:
                enemy_obj.hp = 0
        else:
            enemy_obj.hp -= enemy_unit_damage

        self.mp = 0
        self.path = []
        self.can_attack = False

        return enemy_unit_damage, own_reward, enemy_reward

    def gain_hps(self):
        if self.mp == self.mp_base:
            if any(city.is_cell_inside(self.r, self.c) for city in self.player.cities):
                self.hp = min(100, self.hp + 10)

    @profile
    def move_one_cell(self, game, new_r, new_c) -> int:
        if not (new_r, new_c) in game.map.get_neighbours_grid_coords(self.r, self.c):
            if self.role == MilitaryObject.COMBAT or \
                    not (self.is_within_attack_range(game, new_r, new_c)
                         and len(game.get_game_objects_on_hex(new_r, new_c, only_enemies=True)) > 0):
                raise Exception()

        self.path = [(self.r, self.c), (new_r, new_c)]
        return self.move(game)

    @profile
    def move(self, game) -> int:
        own_reward, enemy_reward = 0, 0

        # if there is just a transition without an attack - the logic is the same for any type of unit
        if len(self.path) == 0 and self.get_ranged_target(game) is None:
            return 0

        # check if ranged unit inside the attack radius
        ranged_target = self.get_ranged_target(game)
        if ranged_target is not None:
            enemy_unit_damage, own_rew, enemy_rew = self.ranged_attack(game, ranged_target.r, ranged_target.c)
            own_reward += own_rew
            game.logger.log_event(RangedAttackEvent(self,
                                                    target=ranged_target,
                                                    enemy_damage=enemy_unit_damage))
        else:
            avail_path_coords = self._get_available_path_coords(game)
            if len(avail_path_coords) == 0:
                return 0

            coord, mp_spent, is_attack = avail_path_coords[-1]
            new_r, new_c = coord

            if is_attack and self.role == MilitaryObject.COMBAT:

                if len(avail_path_coords) > 1:
                    path_log = [(self.r, self.c)] + [coords for coords, _, _ in avail_path_coords[:-1]]
                    game.logger.log_event(MoveEvent(self, path=path_log))
                    self.move_unconditionally(game, *avail_path_coords[-2][0])

                unit_damage, enemy_unit_damage, own_rew, enemy_rew = self.combat_attack(game, new_r, new_c)
                own_reward += own_rew
                enemy_reward += enemy_rew

            else:
                path_log = [(self.r, self.c)] + [coords for coords, _, _ in avail_path_coords]
                game.logger.log_event(MoveEvent(self, path=path_log))
                self.move_unconditionally(game, new_r, new_c)
                self.mp = max(0, self.mp - mp_spent)

                self.path = self.path[self.path.index(coord):]

                # self.ranged_target = None

        return own_reward  # , enemy_reward


    def _get_available_path_coords(self, game) -> List[Tuple[Tuple[int, int], int, bool]]:
        '''
        is_attack can be true only if it is the last one, and if it isn't a ranged unit

        :param game:
        :return: [(coord1, mp_spent1, is_attack1), ...]
        '''
        result = []

        if len(self.path) == 0:
            return result

        mp_left = self.mp
        path_coord_prev = self.path[0]

        for i, path_coord in enumerate(self.path[1:]):
            try:
                move_cost = game.map.get_data_edge(path_coord_prev, path_coord)['cost']
            except Exception as err:
                raise err

            if mp_left < move_cost:
                return result

            new_tile_unit = next(iter(game.map.get(*path_coord).game_objects), None)

            if new_tile_unit is not None:
                if game.is_enemy(new_tile_unit.player):
                    if self.can_attack and self.role == MilitaryObject.COMBAT:
                        result.append((path_coord, self.mp - mp_left + move_cost, True))
                        return result
                    else:
                        continue
                elif isinstance(new_tile_unit, Unit):  # skip the tile
                    mp_left -= move_cost
                    path_coord_prev = path_coord

                    continue

            result.append((path_coord, self.mp - mp_left + move_cost, False))

            mp_left -= move_cost
            path_coord_prev = path_coord

        return result

    @profile
    def get_reachable_cells(self, game):
        result = []

        for r in range(max(0, self.r - self.mp * 2), self.r + self.mp * 2 + 1):
            for c in range(max(0, self.c - self.mp // 2), self.c + self.mp // 2 + 1):
                try:
                    if game.map.get_distance((self.r, self.c), (r, c), weight='cost') <= self.mp:
                        result.append((r, c))
                except nx.exception.NetworkXNoPath:
                    pass

        return result

    def move_unconditionally(self, game, new_r, new_c):
        game.map.remove(self.r, self.c, self)
        game.map.get(new_r, new_c).game_objects.add(self)

        self.r = new_r
        self.c = new_c

    def draw(self, screen, game):
        geom = game.map.get(self.r, self.c).geometry

        if self.is_selected:
            pygame.draw.polygon(screen, SELECTED_TILE_COLOR, geom.points, 0)
            pygame.draw.polygon(screen, (0, 0, 0), geom.points, 1)

        nation_image = nation_icons[self.player.nation]['default']
        screen.blit(self.image,
                    (geom.x - self.image.get_width() / 2, geom.y - self.image.get_height() / 2))
        screen.blit(nation_image, (geom.x - nation_image.get_width() / 2, geom.y + 7))

        hp_offset = 15
        hp_length = 16
        hp_thickness = 5

        pygame.draw.rect(
            screen,
            (255, 0, 0),
            (geom.x - hp_length / 2, geom.y - hp_offset, hp_length * (self.hp / 100), hp_thickness), )

        pygame.draw.rect(
            screen,
            (0, 0, 0),
            (geom.x - hp_length / 2, geom.y - hp_offset, hp_length, hp_thickness),
            width=1)

    # def __repr__(self):
    #     return f'{self.name} at {hex(id(self))}'

class Units:
    Tank = lambda player, r, c: Unit('tank', player, r, c,
                                     role=MilitaryObject.COMBAT,
                                     image_path='assets/units/tank.png',
                                     mp_base=4, ranged_strength_base=0, range_radius_base=0,
                                     combat_strength_base=85,
                                     sound_attack='assets/sounds/tank_attack.ogg')

    Artillery = lambda player, r, c: Unit('artillery', player, r, c,
                                          role=MilitaryObject.RANGED,
                                          image_path='assets/units/artillery.png', mp_base=3,
                                          ranged_strength_base=90, range_radius_base=2, combat_strength_base=70,
                                          sound_attack='assets/sounds/artillery_attack.ogg')

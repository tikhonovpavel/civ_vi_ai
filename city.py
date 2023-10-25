import pygame
from line_profiler_pycharm import profile

from consts import UI_CITY_IMAGE_SIZE
from game_object import MilitaryObject
from logger import RangedAttackEvent
from rewards_values import Rewards


class City(MilitaryObject):
    DEFAULT_CITY_IMAGE_SIZE = (42, 42)

    MAX_HP = 200
    MAX_WALLS_HP = 100

    def __init__(self, name, player, center_r, center_c, territory, hp=None, image_path='assets/city_icon1.png',
                 combat_strength_base=95, ranged_strength_base=95, range_radius_base=2, silent=False):
        super().__init__(name, 'city', player, center_r, center_c, role=MilitaryObject.RANGED,
                         mp_base=0, hp=hp, combat_strength_base=combat_strength_base,
                         ranged_strength_base=ranged_strength_base, range_radius_base=range_radius_base,
                         sound_attack='assets/sounds/artillery_attack.ogg', silent=silent)

        assert center_r, center_c in territory

        # self.name = name
        self.r = center_r
        self.c = center_c

        self.image_path = image_path

        image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(image, self.DEFAULT_CITY_IMAGE_SIZE)
        self.image_ui = pygame.transform.scale(image, UI_CITY_IMAGE_SIZE)

        self.hp = City.MAX_HP if hp is None else hp
        self.walls_hp = City.MAX_WALLS_HP

        self._territory = set(tuple(x) for x in territory)

        # self.silent = silent

    @property
    def territory(self):
        return self._territory

    def gain_hps(self):
        self.hp = min(200, self.hp + 15)

    @territory.setter
    def territory(self, value):
        self._territory = value

    def is_cell_inside(self, r, c):
        return next((True for tile in self._territory if tile == (r, c)), False)

    def combat_attack(self, game, enemy_r, enemy_c, calc_rewards_for):
        raise NotImplementedError()

    def ranged_attack(self, game, enemy_r, enemy_c, calc_rewards_for):
        rewards_dict = {player: [] for player in calc_rewards_for}

        enemy_obj = next(iter(game.map.get(enemy_r, enemy_c).game_objects), None)
        enemy_obj_damage = MilitaryObject.compute_ranged_damage(self, enemy_obj)

        if not self.silent:
            print(f"{self.player.nation}'s {self.category} {self.name} on {self.r, self.c} => "
                  f"{enemy_obj.player.nation}'s {enemy_obj.category} {enemy_obj.name} on {enemy_obj.r, enemy_obj.c}. "      
                  f"HP: ({self.hp} -> {max(0, self.hp)}) / "
                  f"({enemy_obj.hp} -> {max(0, enemy_obj.hp - enemy_obj_damage)})")

        if game.sound_marker.state and self.sound_attack:
            self.sound_attack.play(maxtime=1500, fade_ms=500)

        game.display.show_damage_text(f'-{min(100, int(enemy_obj_damage))}', enemy_obj.r, enemy_obj.c)

        if enemy_obj.hp - enemy_obj_damage <= 0:
            if not isinstance(enemy_obj, City):
                enemy_obj.player.destroy(game, enemy_obj, on_defense=True)

                for p in calc_rewards_for:
                    if p == self.player:
                        rewards_dict[p].append(Rewards.get_named_reward(Rewards.ENEMY_UNIT_DAMAGED, enemy_obj_damage, to_unit=self))
                        rewards_dict[p].append(Rewards.get_named_reward(Rewards.ENEMY_UNIT_DESTROYED, to_unit=self))
                    else:
                        rewards_dict[p].append(Rewards.get_named_reward(Rewards.OWN_UNIT_DAMAGED, enemy_obj_damage, to_unit=enemy_obj))
                        rewards_dict[p].append(Rewards.get_named_reward(Rewards.OWN_UNIT_DESTROYED, to_unit=enemy_obj))
            else:
                enemy_obj.hp = 0
        else:
            enemy_obj.hp -= enemy_obj_damage

            for p in calc_rewards_for:
                if p == self.player:
                    rewards_dict[p].append(Rewards.get_named_reward(Rewards.ENEMY_UNIT_DAMAGED, enemy_obj_damage, to_unit=self))
                else:
                    rewards_dict[p].append(Rewards.get_named_reward(Rewards.OWN_UNIT_DAMAGED, enemy_obj_damage, to_unit=enemy_obj))

        self.mp = 0
        self.path = []
        self.can_attack = False

        return enemy_obj_damage, rewards_dict

    @profile
    def move(self, game, calc_rewards_for):
        rewards_dict = {player: [] for player in calc_rewards_for}

        # check if ranged unit inside the attack radius
        ranged_target = self.get_ranged_target(game)
        if ranged_target is not None:
            enemy_obj_damage, rew_dict = self.ranged_attack(game, ranged_target.r, ranged_target.c, calc_rewards_for)
            rewards_dict.append(rew_dict)

            game.logger.log_event(RangedAttackEvent(self,
                                                    target=ranged_target,
                                                    enemy_damage=enemy_obj_damage))

        game.update()

        return rewards_dict

    def change_ownership(self, new_player):
        self.player.cities.remove(self)
        self._player = new_player

        new_player.cities.append(self)
        # new_player.add_city(self)

        self.hp = 100

    def draw(self, screen, game, text_module):
        center_hex = game.map.get(self.r, self.c).geometry

        screen.blit(self.image, (center_hex.x - self.DEFAULT_CITY_IMAGE_SIZE[0] / 2,
                                 center_hex.y - self.DEFAULT_CITY_IMAGE_SIZE[1] / 2))

        hp_offset = 20
        hp_length = 32
        hp_thickness = 8

        pygame.draw.rect(
            screen,
            (255, 0, 0),
            (center_hex.x - hp_length / 2, center_hex.y - hp_offset, hp_length * (self.hp / self.MAX_HP), hp_thickness))

        pygame.draw.rect(
            screen,
            (0, 0, 0),
            (center_hex.x - hp_length / 2, center_hex.y - hp_offset, hp_length, hp_thickness),
            width=1)

        text_module.text_to_screen(screen, self.name, center_hex.x, center_hex.y + 15,
                                   size=15, color=(0, 0, 0), bold=True, align='center')




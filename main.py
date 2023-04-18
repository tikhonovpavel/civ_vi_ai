from line_profiler_pycharm import profile
import math
import pygame
import networkx as nx
import random


from display import Display
from grid import HexagonGrid

from unit import Units, UnitState
from player import Player

# Initialize Pygame
pygame.init()

# Set the dimensions of the screen
screen_width = 1000
screen_height = 800

screen = pygame.display.set_mode((screen_width, screen_height))
screen.fill((255, 255, 255))

# Update the screen
pygame.display.update()

clock = pygame.time.Clock()

COMBAT_MINIMUM_DAMAGE = 1


class Game:
    def __init__(self, ) -> None:
        player1 = Player('rome', [Units.Tank(15, 3), Units.Tank(12, 4), Units.Tank(8, 3), Units.Artillery(10, 2)])
        player2 = Player('egypt', [Units.Tank(5, 11), Units.Tank(8, 12), Units.Tank(14, 12), Units.Artillery(17, 12)])

        self.players = [player1, player2]

        # all vs all
        self.diplomacy = nx.Graph()
        for player in self.players:
            self.diplomacy.add_node(player.nation)
        for i in range(len(self.players)):
            for j in range(i + 1, len(self.players)):
                self.diplomacy.add_edge(self.players[i].nation, self.players[j].nation, weight=-1)

        self._current_player = 0

        self.hexagon_grid = HexagonGrid(30, 15)
        # self.paths = []

        self._display = Display(screen, self)

        self._display.update_all()  # self.players, self.hexagon_grid, self. paths)

    def is_enemy(self, other_player):
        return bool(self.diplomacy.get_edge_data(self.get_current_player().nation, other_player.nation)['weight'])

    def get_current_player(self):
        return self.players[self._current_player]

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

    def update(self, ):
        self._display.update_all()  # self.players, self.hexagon_grid, self.paths)

    def left_button_released(self, mouse_x, mouse_y):
        self.get_current_player().no_attack()

        r, c = self.hexagon_grid.get_grid_coords(mouse_x, mouse_y)

        if r is None:
            return

        for unit in self.get_current_player().units:
            unit.is_selected = unit.r == r and unit.c == c

        self._display.update_all()  # self.players, self.hexagon_grid, self. paths)

    @profile
    def right_button_pressed(self, mouse_x, mouse_y):
        current_player = self.get_current_player()
        current_player.no_attack()

        unit_selected = self.get_selected_unit()
        if unit_selected is None:
            return

        r, c = self.hexagon_grid.get_grid_coords(mouse_x, mouse_y)
        if r is None:
            return

        # if there is a unit of the same category on the hex - cancel
        if len(self.get_units_on_hex(r, c, player=current_player)) > 0:  # TODO
            return

        # if there is an enemy unit on the hex - show attack screen
        enemies = self.get_units_on_hex(r, c, only_enemies=True)
        if len(enemies) > 0:
            # print('hoba')
            current_player.set_enemy(*enemies[0])
            # return

        unit_selected.path = self.hexagon_grid.get_shortest_path((unit_selected.r, unit_selected.c), (r, c))

        self.update()

    @profile
    def right_button_released(self, mouse_x, mouse_y):
        current_player = self.get_current_player()

        unit_selected = self.get_selected_unit()
        if unit_selected is None:
            return

        r, c = self.hexagon_grid.get_grid_coords(mouse_x, mouse_y)
        if r is None:
            return

        enemy_player, enemy_unit = current_player.get_enemy()

        if len(unit_selected.path) != 0 and (r, c) == unit_selected.path[-1]:  # confirmation of the move

            # if there is an enemy - attack:
            if enemy_player is not None:
                enemy_unit_damage = compute_combat_damage(unit_selected, enemy_unit)
                unit_selected_damage = compute_combat_damage(enemy_unit, unit_selected)

                print(f"{current_player.nation}'s {unit_selected.name} hp: {unit_selected.hp}, damage: {unit_selected_damage}")
                print(f"{enemy_player.nation}'s {unit_selected.name} hp: {enemy_unit.hp}, damage: {enemy_unit_damage}")
                print()

                if unit_selected.hp - unit_selected_damage <= 0:
                    current_player.units.remove(unit_selected)
                    enemy_unit.hp = max(1, enemy_unit.hp - enemy_unit_damage)
                elif enemy_unit.hp - enemy_unit_damage <= 0:
                    enemy_player.units.remove(enemy_unit)  # del enemy_unit
                    unit_selected.hp -= unit_selected_damage
                    unit_selected.r, unit_selected.c = enemy_unit.r, enemy_unit.c  # or just r and c

                else:
                    unit_selected.hp -= unit_selected_damage
                    enemy_unit.hp -= enemy_unit_damage
                    unit_selected.r, unit_selected.c = unit_selected.path[-2]

                    # unit_selected.selected = False

            # if there is no enemy - just move:
            else:
                unit_selected.r = r
                unit_selected.c = c

            unit_selected.selected = False
            unit_selected.path = []
        else:  # create a path to the hex
            unit_selected.path = self.hexagon_grid.get_shortest_path((unit_selected.r, unit_selected.c), (r, c))

        self._display.update_all()  # self.players, self.hexagon_grid, self.paths)

    # def set_unit_moving_to(unit, dest_r, dest_c):
    #     unit.state = UnitState.MOVING
    #     unit.set_unit_moving_to(dest_r, dest_c)


def compute_combat_damage(unit1, unit2):
    diff = unit1.calc_combat_strength() - unit2.calc_combat_strength()

    return random.uniform(0.8, 1.2) * 30 * math.exp(diff / 25)
    # return 30 * math.exp(diff / 25 * random.uniform(0.75, 1.25))

@profile
def main():
    game = Game()

    # Run the game loop
    running = True

    while running:

        for event in pygame.event.get():
            lb, mb, rb = pygame.mouse.get_pressed()
            # print(lb, mb, rb)

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEMOTION:
                if rb:
                    game.right_button_pressed(*event.pos)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 1 == left button
                    game.left_button_released(*event.pos)
                #
                if event.button == 3:  # 3 == right button
                    # print('rb pressed')
                    game.right_button_pressed(*event.pos)

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    # print('rb released')
                    # game.right_button_pressed(*event.pos)
                    game.right_button_released(*event.pos)

    pygame.quit()


if __name__ == '__main__':
    main()

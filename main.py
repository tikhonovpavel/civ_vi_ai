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
screen.fill((255,255,255))


# Update the screen
pygame.display.update()


clock = pygame.time.Clock()

class Game:
    def __init__(self, ) -> None:
        player1 = Player('rome', [Units.Tank(15, 3), Units.Tank(12, 4), Units.Tank(8, 3), Units.Artillery(10,2)])
        player2 = Player('egypt', [Units.Tank(5, 11), Units.Tank(8, 12), Units.Tank(14, 12), Units.Artillery(17,12)])

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
        self.paths = []

        self._display = Display(screen, self)

        self._display.update_all()#self.players, self.hexagon_grid, self. paths)


    def is_enemy(self, other_player):
        return bool(self.diplomacy.get_edge_data(self.get_current_player().nation, other_player.nation)['weight'])


    def get_current_player(self):
        return self.players[self._current_player]
    

    def get_selected_unit(self):
        return next((u for u in self.get_current_player().units if u.selected), None)


    def get_units_on_hex(self, r, c, player=None, only_enemies=False, category=None):
        if player is not None and only_enemies:
            print('Use player and only_enemies simultaneously with caution')

        result = [(p, u) for p in self.players for u in p.units if u.r == r and u.c == c ]

        if player is not None:
            result = [(p, u) for p, u in result if p == player]

        if only_enemies:
            result = [(p, u) for p, u in result if self.is_enemy(p)]
        
        if category is not None:
            result = [(p, u) for p, u in result if u.category == category]

        return result


    def update(self,):
        self._display.update_all(self.players, self.hexagon_grid, self.paths)


    def left_click(self, mouse_x, mouse_y):
        self.get_current_player().no_attack()
        
        self.paths = []
        r, c = self.hexagon_grid.get_grid_coords(mouse_x, mouse_y)

        if r is None:
            return 
        
        for unit in self.get_current_player().units:
            unit.selected = unit.r == r and unit.c == c

        self._display.update_all()#self.players, self.hexagon_grid, self. paths)


    def right_click(self, mouse_x, mouse_y):

        self.get_current_player().no_attack()

        r, c = self.hexagon_grid.get_grid_coords(mouse_x, mouse_y)

        if r is None:
            return
        
        unit_selected = self.get_selected_unit()
        if unit_selected is None:
            return
        
        # if there is a unit of the same category on the hex - cancel
        if len(self.get_units_on_hex(r, c, player=self.get_current_player())) > 0: # TODO
            return
        
        # if there is an enemy unit on the hex - show attack screen
        enemy = self.get_units_on_hex(r, c, only_enemies=True)
        if len(enemy) > 0:
            print('hoba')
            self.get_current_player().set_ready_to_attack(*enemy[0])
            # return
            
        if len(self.paths) != 0 and (r, c) == self.paths[0][-1]:  # confirmation of the move

            # if there is an enemy - attack:
            if len(enemy) > 0:
                enemy_player, enemy_unit = enemy[0]
                enemy_unit_demage = compute_combat_demage(unit_selected, enemy_unit)
                unit_selected_demage = compute_combat_demage(enemy_unit, unit_selected)

                print(f'Rome hp: {unit_selected.hp}, demage: {unit_selected_demage}')
                print(f'Egypt hp: {enemy_unit.hp}, demage: {enemy_unit_demage}')
                print()

                if unit_selected.hp - unit_selected_demage <= 0:
                    self.get_current_player().units.remove(unit_selected)
                    enemy_unit.hp = max(1, enemy_unit.hp - enemy_unit_demage)
                elif enemy_unit.hp - enemy_unit_demage <= 0:
                    enemy_player.units.remove(enemy_unit)  # del enemy_unit
                    unit_selected.hp -= unit_selected_demage
                    unit_selected.r, unit_selected.c = enemy_unit.r, enemy_unit.c  # or just r and c
                    
                else:
                    unit_selected.hp -= unit_selected_demage
                    enemy_unit.hp -= enemy_unit_demage
                    unit_selected.r, unit_selected.c = self.paths[0][-2] 
                    # unit_selected.selected = False

            # if there is no enemy - just move:
            else:
                unit_selected.r = r
                unit_selected.c = c
            
            unit_selected.selected = False
            self.paths = []
        else:  # create a path to the hex
            self.paths = [self.hexagon_grid.get_shortest_path((unit_selected.r, unit_selected.c), (r, c))]

        self._display.update_all()#self.players, self.hexagon_grid, self.paths)


    def set_unit_moving_to(unit, dest_r, dest_c):
        unit.state = UnitState.MOVING
        unit.set_unit_moving_to(dest_r, dest_c)


def compute_combat_demage(unit1, unit2):
    diff = unit1.calc_combat_strength() - unit2.calc_combat_strength()
    return 30 * math.exp(diff / 25 * random.uniform(0.75, 1.25))


def main():
    game = Game()
    
    # Run the game loop
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # 1 == left button
                    game.left_click(*event.pos)


                if event.button == 3: # 3 == right button
                    game.right_click(*event.pos)
                        
    pygame.quit()


if __name__ == '__main__':
    main()
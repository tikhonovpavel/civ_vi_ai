import pygame
import networkx as nx

from display import Display
from grid import HexagonGrid

from unit import Units
from player import Player


# Initialize Pygame
pygame.init()

# Set the dimensions of the screen
screen_width = 800
screen_height = 600

screen = pygame.display.set_mode((screen_width, screen_height))
screen.fill((255,255,255))


# Update the screen
pygame.display.update()

# Run the game loop
running = True

clock = pygame.time.Clock()

hexagon_grid = HexagonGrid(30, 12)
paths = []
display = Display(screen)


player1 = Player([Units.Tank(10, 3), Units.Tank(12, 4), Units.Tank(8, 7), Units.Artillery(5,5)])
players = [player1]


display.update_all(players, hexagon_grid, [])


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # 1 == left button

                paths = []
                
                min_r, min_c = hexagon_grid.get_hexagon_grid_coords(*event.pos)

                if min_r is not None:
                    hex = hexagon_grid.get(min_r, min_c)


                    for unit in player1.units:
                        unit.selected = unit.r == min_r and unit.c == min_c

                    display.update_all(players, hexagon_grid, paths)

                    pygame.display.update()


            if event.button == 3: # 3 == right button
                min_r, min_c = hexagon_grid.get_hexagon_grid_coords(*event.pos)

                if min_r is not None:
                    for unit in player1.units:
                        if unit.selected:
                            if len(paths) != 0 and (min_r, min_c) == paths[0][-1]:
                                unit.r = min_r
                                unit.c = min_c
                                unit.selected = False

                                paths = []
                                break
                            
                            paths = [hexagon_grid.get_shortest_path((unit.r, unit.c), (min_r, min_c))]

                    display.update_all(players, hexagon_grid, paths)

                    pygame.display.update()
                    
pygame.quit()

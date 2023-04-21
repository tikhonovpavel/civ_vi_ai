from line_profiler_pycharm import profile
import math
import pygame
import networkx as nx
import random

import map
from button import ButtonStates
from display import Display
from game import Game
from map import Map

from unit import Units, UnitState
from player import Player
import pprint

import datetime# import datetime

# Set the dimensions of the screen
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800


# Update the screen
# pygame.display.update()



@profile
def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.fill((255, 255, 255))
    clock = pygame.time.Clock()
    clock.tick(60)

    game = Game(screen, clock)

    # Run the game loop
    running = True

    while running:
        game.next_move_button.state = ButtonStates.DEFAULT

        for event in pygame.event.get():
            lb, mb, rb = pygame.mouse.get_pressed()
            # print(lb, mb, rb)

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEMOTION:
                game.mouse_motion(event)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 1 == left button
                    game.left_button_released(event)

                if event.button == 2:
                    print(event.pos)
                    pygame.draw.circle(screen, (128, 0, 128), event.pos, radius=3)

                if event.button == 3:  # 3 == right button
                    # print('rb pressed')
                    game.right_button_pressed(*event.pos)

            if event.type == pygame.MOUSEBUTTONUP:

                if event.button == 3:
                    # print('rb released')
                    # game.right_button_pressed(*event.pos)
                    game.right_button_released(*event.pos)

        game.display.update_texts()
        game.next_move_button.draw(screen, game, game.display.text_module)
        pygame.display.update()
    # button.draw(screen)

    pygame.quit()


if __name__ == '__main__':
    main()

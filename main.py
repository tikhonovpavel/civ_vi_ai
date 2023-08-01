# from line_profiler_pycharm import profile

import pygame
from line_profiler_pycharm import profile

from ui import ButtonStates
from game import Game

# Set the dimensions of the screen
from unit import Units

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 900


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
        game.next_turn_button.state = ButtonStates.DEFAULT

        for event in pygame.event.get():
            lb, mb, rb = pygame.mouse.get_pressed()

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEMOTION:
                game.mouse_motion(event)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 1 == left button
                    game.left_button_pressed(event)

                if event.button == 2: # 2 == middle button

                    # res = game.map.get(*tile_coord).game_objects[0].get_reachable_cells(game)
                    # for tile_coord in res:
                    #     if tile_coord not in game.players[0].cities[0].territory:
                    #         game.players[0].cities[0].territory.add(tile_coord)
                    #     else:
                    #         game.players[0].cities[0].territory.remove(tile_coord)


                    try:
                        tile_coord = game.map.get_grid_coords(*event.pos)
                        print(tile_coord)
                    #
                        player = game.players[0]
                        # game.add_game_obj(player, Units.Tank(player, *tile_coord))

                        if tile_coord not in player.cities[0].territory:
                            player.cities[0].territory.add(tile_coord)
                        else:
                            player.cities[0].territory.remove(tile_coord)
                    except:
                        pass

                    game.update()

                if event.button == 3:  # 3 == right button
                    # print('rb pressed')
                    game.right_button_pressed(*event.pos)

            if event.type == pygame.MOUSEBUTTONUP:

                if event.button == 3:
                    ...
                    # print('rb released')
                    # game.right_button_pressed(*event.pos)

                    game.right_button_released(*event.pos)

        game.display.update_texts()
        pygame.display.update()

    pygame.quit()


if __name__ == '__main__':
    main()
